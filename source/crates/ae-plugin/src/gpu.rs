//! Native Premiere GPU entry point. After Effects and non-CUDA Premiere
//! renderers keep using the existing software entry point in lib.rs.
use super::*;
#[cfg(feature = "opencl")]
use ntsc_rs::cuda::opencl::OpenClRenderer;
use ntsc_rs::cuda::{CudaPlan, CudaRenderer, DeviceFrame};
enum Renderer {
    Cuda(CudaRenderer),
    #[cfg(feature = "opencl")]
    OpenCl(OpenClRenderer),
}
use premiere::{self as pr, GpuFilterData, RenderParams, sys};
use std::{
    ffi::c_void,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::Mutex,
};

struct Instance {
    data: GpuFilterData,
    renderer: Renderer,
    workspace: *mut c_void,
    workspace_bytes: usize,
    descriptors: Vec<SettingDescriptor<NtscEffect>>,
}

impl Drop for Instance {
    fn drop(&mut self) {
        if !self.workspace.is_null() {
            let _ = self
                .data
                .gpu_device_suite
                .free_device_memory(self.data.device_index(), self.workspace);
        }
    }
}

fn guarded(f: impl FnOnce() -> Result<(), pr::Error>) -> sys::prSuiteError {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(())) => sys::suiteError_NoError,
        Ok(Err(error)) => error as _,
        Err(_) => sys::suiteError_Fail,
    }
}

unsafe fn pica(instance: *mut sys::PrGPUFilterInstance) -> Result<pr::PicaBasicSuite, pr::Error> {
    if instance.is_null() {
        return Err(pr::Error::InvalidParms);
    }
    let suites = unsafe { (*instance).piSuites };
    if suites.is_null() {
        return Err(pr::Error::InvalidParms);
    }
    let util = unsafe { (*suites).utilFuncs };
    if util.is_null() {
        return Err(pr::Error::InvalidParms);
    }
    let get = unsafe { (*util).getSPBasicSuite }.ok_or(pr::Error::InvalidParms)?;
    Ok(pr::PicaBasicSuite::from_sp_basic_suite_raw(unsafe {
        get()
    }))
}

unsafe extern "C" fn create(instance: *mut sys::PrGPUFilterInstance) -> sys::prSuiteError {
    guarded(|| {
        let _pica = unsafe { pica(instance)? };
        if std::env::var_os("NTSC_RS_DISABLE_GPU").is_some() {
            return Err(pr::Error::NotImplemented);
        }
        let suite = pr::suites::GPUDevice::new()?;
        let info = suite.device_info(unsafe { (*instance).inDeviceIndex })?;
        if info.outMeetsMinimumRequirementsForAcceleration == 0 {
            return Err(pr::Error::NotImplemented);
        }
        let renderer = match info.outDeviceFramework {
            #[cfg(feature = "cuda")]
            sys::PrGPUDeviceFramework_PrGPUDeviceFramework_CUDA => {
                if std::env::var_os("NTSC_RS_DISABLE_CUDA").is_some() {
                    return Err(pr::Error::NotImplemented);
                }
                Renderer::Cuda(
                    unsafe { CudaRenderer::new(info.outContextHandle) }
                        .map_err(|_| pr::Error::NotImplemented)?,
                )
            }
            #[cfg(feature = "opencl")]
            sys::PrGPUDeviceFramework_PrGPUDeviceFramework_OpenCL => Renderer::OpenCl(
                unsafe { OpenClRenderer::new(info.outContextHandle, info.outCommandQueueHandle) }
                    .map_err(|_| pr::Error::NotImplemented)?,
            ),
            _ => return Err(pr::Error::NotImplemented),
        };
        let state = Instance {
            data: GpuFilterData {
                instance_ptr: instance,
                gpu_device_suite: suite,
                gpu_image_processing_suite: pr::suites::GPUImageProcessing::new()?,
                memory_manager_suite: pr::suites::MemoryManager::new()?,
                ppix_suite: pr::suites::PPix::new()?,
                ppix2_suite: pr::suites::PPix2::new()?,
                video_segment_suite: pr::suites::VideoSegment::new()?,
                gpu_info: info,
            },
            renderer,
            workspace: std::ptr::null_mut(),
            workspace_bytes: 0,
            descriptors: NtscEffect::setting_descriptors().into_vec(),
        };
        unsafe {
            (*instance).ioPrivatePluginData = Box::into_raw(Box::new(Mutex::new(state))).cast();
        }
        Ok(())
    })
}

unsafe extern "C" fn dispose(instance: *mut sys::PrGPUFilterInstance) -> sys::prSuiteError {
    guarded(|| {
        let _pica = unsafe { pica(instance)? };
        let ptr = unsafe { (*instance).ioPrivatePluginData };
        if !ptr.is_null() {
            unsafe {
                drop(Box::from_raw(ptr.cast::<Mutex<Instance>>()));
                (*instance).ioPrivatePluginData = std::ptr::null_mut();
            }
        }
        Ok(())
    })
}

fn number(param: pr::Param) -> Result<f64, pr::Error> {
    Ok(match param {
        pr::Param::Int8(v) => v as f64,
        pr::Param::Int16(v) => v as f64,
        pr::Param::Int32(v) => v as f64,
        pr::Param::Int64(v) => v as f64,
        pr::Param::Float32(v) => v as f64,
        pr::Param::Float64(v) => v,
        pr::Param::Bool(v) => u8::from(v) as f64,
        _ => return Err(pr::Error::InvalidParms),
    })
}

fn read_settings(
    data: &GpuFilterData,
    descriptors: &[SettingDescriptor<NtscEffect>],
    time: i64,
) -> Result<NtscEffect, pr::Error> {
    read_settings_from(descriptors, |index| data.param(index, time))
}

fn read_settings_from(
    descriptors: &[SettingDescriptor<NtscEffect>],
    mut read: impl FnMut(usize) -> Result<pr::Param, pr::Error>,
) -> Result<NtscEffect, pr::Error> {
    fn walk(
        read: &mut impl FnMut(usize) -> Result<pr::Param, pr::Error>,
        descriptors: &[SettingDescriptor<NtscEffect>],
        index: &mut usize,
        effect: &mut NtscEffect,
    ) -> Result<(), pr::Error> {
        for descriptor in descriptors {
            let group = matches!(&descriptor.kind, SettingKind::Group { .. });
            if group {
                *index += 1;
            } // AE topic-start marker precedes its Enabled checkbox.
            let mut value = number(read(*index)?)?;
            *index += 1;
            if !value.is_finite() {
                return Err(pr::Error::InvalidParms);
            }
            let field_error = |_| pr::Error::InvalidParms;
            match &descriptor.kind {
                SettingKind::Enumeration { options } => {
                    let option = options
                        .get((value as i64 - 1).max(0) as usize)
                        .ok_or(pr::Error::InvalidParms)?;
                    effect
                        .set_field(&descriptor.id, EnumValue(option.index))
                        .map_err(field_error)?;
                }
                SettingKind::Percentage { logarithmic } => {
                    value *= 0.01;
                    if *logarithmic {
                        value = map_logarithmic(value, 0.0, 1.0, LOG_SLIDER_BASE);
                    }
                    effect
                        .set_field(&descriptor.id, value as f32)
                        .map_err(field_error)?;
                }
                SettingKind::IntRange { .. } => effect
                    .set_field(&descriptor.id, value.round() as i32)
                    .map_err(field_error)?,
                SettingKind::FloatRange { range, logarithmic } => {
                    if *logarithmic {
                        value = map_logarithmic(
                            value,
                            *range.start() as f64,
                            *range.end() as f64,
                            LOG_SLIDER_BASE,
                        );
                    }
                    effect
                        .set_field(&descriptor.id, value as f32)
                        .map_err(field_error)?;
                }
                SettingKind::Boolean => effect
                    .set_field(&descriptor.id, value != 0.0)
                    .map_err(field_error)?,
                SettingKind::Group { children } => {
                    effect
                        .set_field(&descriptor.id, value != 0.0)
                        .map_err(field_error)?;
                    walk(read, children, index, effect)?;
                    *index += 1; // AE topic-end marker.
                }
            }
        }
        Ok(())
    }
    let mut effect = NtscEffect::default();
    // AE index 0 is the input layer, 1/2 are Load Preset / Save Preset.
    // GpuFilterData::param subtracts the input-layer slot for Premiere.
    walk(&mut read, descriptors, &mut 3, &mut effect)?;
    Ok(effect)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    // Encode the values exposed by the software plugin's parameter UI and read
    // them through the GPU adapter. This covers popup ordering, group markers,
    // percentage scaling and both kinds of logarithmic sliders.
    fn encode(
        descriptors: &[SettingDescriptor<NtscEffect>],
        effect: &NtscEffect,
        index: &mut usize,
        values: &mut HashMap<usize, f64>,
    ) {
        for descriptor in descriptors {
            let value = match &descriptor.kind {
                SettingKind::Enumeration { options } => {
                    let value = effect.get_field::<EnumValue>(&descriptor.id).unwrap().0;
                    (options.iter().position(|o| o.index == value).unwrap() + 1) as f64
                }
                SettingKind::Percentage { logarithmic } => {
                    let value = effect.get_field::<f32>(&descriptor.id).unwrap() as f64;
                    (if *logarithmic {
                        map_logarithmic_inverse(value, 0., 1., LOG_SLIDER_BASE)
                    } else {
                        value
                    }) * 100.
                }
                SettingKind::FloatRange { range, logarithmic } => {
                    let value = effect.get_field::<f32>(&descriptor.id).unwrap() as f64;
                    if *logarithmic {
                        map_logarithmic_inverse(
                            value,
                            *range.start() as f64,
                            *range.end() as f64,
                            LOG_SLIDER_BASE,
                        )
                    } else {
                        value
                    }
                }
                SettingKind::IntRange { .. } => {
                    effect.get_field::<i32>(&descriptor.id).unwrap() as f64
                }
                SettingKind::Boolean => {
                    u8::from(effect.get_field::<bool>(&descriptor.id).unwrap()) as f64
                }
                SettingKind::Group { .. } => {
                    *index += 1;
                    u8::from(effect.get_field::<bool>(&descriptor.id).unwrap()) as f64
                }
            };
            values.insert(*index, value);
            *index += 1;
            if let SettingKind::Group { children } = &descriptor.kind {
                encode(children, effect, index, values);
                *index += 1;
            }
        }
    }

    #[test]
    fn gpu_parameters_match_software_parameters() {
        let descriptors = NtscEffect::setting_descriptors();
        for original in [NtscEffect::default(), NtscEffect::legacy_value()] {
            let mut values = HashMap::new();
            encode(&descriptors, &original, &mut 3, &mut values);
            let decoded = read_settings_from(&descriptors, |index| {
                values
                    .get(&index)
                    .copied()
                    .map(pr::Param::Float64)
                    .ok_or(pr::Error::InvalidParms)
            })
            .unwrap();
            assert_eq!(decoded, original);
        }
    }

    #[test]
    fn invalid_gpu_parameters_return_errors() {
        let descriptors = NtscEffect::setting_descriptors();
        assert!(read_settings_from(&descriptors, |_| Ok(pr::Param::Float64(f64::NAN))).is_err());
        assert!(
            read_settings_from(&descriptors, |_| Ok(pr::Param::MemoryPtr(
                std::ptr::null_mut()
            )))
            .is_err()
        );
    }
}

unsafe extern "C" fn render(
    instance: *mut sys::PrGPUFilterInstance,
    params: *const sys::PrGPUFilterRenderParams,
    frames: *const sys::PPixHand,
    count: sys::csSDK_size_t,
    out: *mut sys::PPixHand,
) -> sys::prSuiteError {
    guarded(|| {
        let _pica = unsafe { pica(instance)? };
        if params.is_null() || frames.is_null() || count < 1 || out.is_null() {
            return Err(pr::Error::InvalidParms);
        }
        let ptr = unsafe { (*instance).ioPrivatePluginData };
        if ptr.is_null() {
            return Err(pr::Error::InvalidParms);
        }
        let state = unsafe { &*ptr.cast::<Mutex<Instance>>() };
        let mut state = state.lock().map_err(|_| pr::Error::Fail)?;
        state.data.instance_ptr = instance;
        let render = RenderParams::from_raw(params);
        let effect = read_settings(&state.data, &state.descriptors, render.clip_time())?;
        let source = unsafe { *frames };
        if source.is_null() {
            return Err(pr::Error::InvalidParms);
        }
        let format = state.data.ppix_suite.pixel_format(source)?;
        let half = match format {
            pr::PixelFormat::GpuBgra4444_32f => false,
            pr::PixelFormat::GpuBgra4444_16f => true,
            _ => return Err(pr::Error::RenderInvalidPixelFormat),
        };
        let bounds = state.data.ppix_suite.bounds(source)?;
        let width = (bounds.right - bounds.left) as usize;
        let height = (bounds.bottom - bounds.top) as usize;
        if width != render.render_width() as usize
            || height != render.render_height() as usize
            || render.render_ticks_per_frame() <= 0
        {
            return Err(pr::Error::InvalidParms);
        }
        let frame_num = (render.clip_time() / render.render_ticks_per_frame()).max(0) as usize;
        let scale = if effect
            .scale
            .as_option()
            .is_some_and(|s| s.scale_with_video_size)
        {
            [1.0, 1.0]
        } else {
            let (x, y) = render.downsample_factor();
            [x, y]
        };
        let plan = CudaPlan::new(&effect, (width, height), frame_num, scale)
            .map_err(|_| pr::Error::InvalidParms)?;
        let device = state.data.device_index();
        if matches!(state.renderer, Renderer::Cuda(_))
            && state.workspace_bytes < plan.workspace_bytes()
        {
            // Reserve in MiB increments so random snow counts do not cause
            // allocations on every frame. All memory belongs to Premiere.
            let bytes = plan.workspace_bytes().div_ceil(1024 * 1024) * 1024 * 1024;
            let new = state
                .data
                .gpu_device_suite
                .allocate_device_memory(device, bytes)?;
            if !state.workspace.is_null() {
                if let Err(error) = state
                    .data
                    .gpu_device_suite
                    .free_device_memory(device, state.workspace)
                {
                    let _ = state.data.gpu_device_suite.free_device_memory(device, new);
                    return Err(error);
                }
            }
            state.workspace = new;
            state.workspace_bytes = bytes;
        }
        let (par_num, par_den) = render.render_pixel_aspect_ratio();
        let output = state.data.gpu_device_suite.create_gpu_ppix(
            device,
            format,
            width as i32,
            height as i32,
            par_num as i32,
            par_den as i32,
            render.render_field_type(),
        )?;
        let result = (|| {
            let input = DeviceFrame {
                ptr: state.data.gpu_device_suite.gpu_ppix_data(source)? as u64,
                row_bytes: state.data.ppix_suite.row_bytes(source)?,
                width,
                height,
                half,
            };
            let output_frame = DeviceFrame {
                ptr: state.data.gpu_device_suite.gpu_ppix_data(output)? as u64,
                row_bytes: state.data.ppix_suite.row_bytes(output)?,
                ..input
            };
            let workspace = state.workspace as u64;
            let bytes = state.workspace_bytes;
            let stream = state.data.gpu_info.outCommandQueueHandle;
            unsafe {
                match &mut state.renderer {
                    Renderer::Cuda(renderer) => {
                        renderer.render(&plan, input, output_frame, workspace, bytes, stream)
                    }
                    #[cfg(feature = "opencl")]
                    Renderer::OpenCl(renderer) => renderer.render(&plan, input, output_frame),
                }
            }
            .map_err(|_| pr::Error::Fail)
        })();
        if let Err(error) = result {
            let _ = state.data.ppix_suite.dispose(output);
            return Err(error);
        }
        unsafe {
            *out = output;
        }
        Ok(())
    })
}

#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn xGPUFilterEntry(
    version: u32,
    index: *mut i32,
    startup: sys::prBool,
    suites: sys::piSuitesPtr,
    filter: *mut sys::PrGPUFilter,
    info: *mut sys::PrGPUFilterInfo,
) -> sys::prSuiteError {
    guarded(|| {
        if version < sys::PrSDKGPUFilterInterfaceVersion1
            || index.is_null()
            || suites.is_null()
            || filter.is_null()
            || info.is_null()
        {
            return Err(pr::Error::InvalidParms);
        }
        if startup != 0 {
            if unsafe { *index } != 0 {
                return Err(pr::Error::InvalidParms);
            }
            unsafe {
                (*filter).CreateInstance = Some(create);
                (*filter).DisposeInstance = Some(dispose);
                (*filter).GetFrameDependencies = None;
                (*filter).Precompute = None;
                (*filter).Render = Some(render);
                // Empty match name selects this module's existing "ntsc-rs" PiPL.
                (*info).outMatchName = std::mem::zeroed();
                (*info).outInterfaceVersion = version.min(sys::PrSDKGPUFilterInterfaceVersion);
            }
        }
        Ok(())
    })
}
