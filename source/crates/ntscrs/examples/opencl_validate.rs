//! Run on real CUDA hardware: cargo run -p ntsc-rs --release --features cuda --example cuda_validate
use ntsc_rs::{
    Context, NtscEffect,
    cuda::{CudaPlan, DeviceFrame, opencl::OpenClRenderer},
    settings::standard::*,
    yiq_fielding::Bgrx,
};

fn main() -> Result<(), String> {
    let device = OwnedCl::new()?;
    let mut renderer = unsafe { OpenClRenderer::new(device.context, device.queue)? };
    let ctx = Context::new();
    let mut cases = Vec::new();
    for field in [
        UseField::Both,
        UseField::Upper,
        UseField::Lower,
        UseField::InterleavedUpper,
        UseField::InterleavedLower,
        UseField::Alternating,
    ] {
        for filter in [FilterType::Butterworth, FilterType::ConstantK] {
            for demod in [
                ChromaDemodulationFilter::Box,
                ChromaDemodulationFilter::Notch,
                ChromaDemodulationFilter::OneLineComb,
                ChromaDemodulationFilter::TwoLineComb,
                ChromaDemodulationFilter::TwoD,
                ChromaDemodulationFilter::TwoDAdaptive,
            ] {
                let mut effect = NtscEffect::default();
                effect.use_field = field;
                effect.filter_type = filter;
                effect.chroma_demodulation = demod;
                cases.push((format!("{field:?}/{filter:?}/{demod:?}"), effect));
            }
        }
    }
    let mut custom = NtscEffect::default();
    custom.chroma_delay_horizontal = -3.25;
    custom.chroma_delay_vertical = 3;
    custom.chroma_vert_blend = true;
    custom.chroma_phase_error = 0.17;
    custom.input_luma_filter = LumaLowpass::Box;
    custom.snow_intensity = 0.05;
    custom.head_switching.settings.mid_line.enabled = true;
    custom.tracking_noise.settings.height = 30;
    custom.vhs_settings.settings.chroma_loss = 0.4;
    custom.scale.settings.horizontal_scale = 1.25;
    custom.scale.settings.vertical_scale = 0.75;
    custom.scale.enabled = true;
    cases.push(("custom controls".into(), custom.clone()));
    custom.chroma_delay_vertical = -5;
    custom.chroma_delay_horizontal = 4.5;
    cases.push(("negative vertical delay".into(), custom));
    let mut worst = 0f32;
    let mut checked = 0;
    for (width, height) in [(1, 1), (2, 3), (17, 13), (641, 481)] {
        let input: Vec<f32> = (0..width * height * 4)
            .map(|i| ((i * 31 + i / 13) % 256) as f32 / 255.)
            .collect();
        let src = renderer.allocate(input.len() * 4)?;
        renderer.upload(&src, &input)?;
        let dst = renderer.allocate(input.len() * 4)?;
        for (name, effect) in &cases {
            // The CPU vertical-blend implementation needs two scratch rows.
            if height < 8 && effect.chroma_vert_blend {
                continue;
            }
            let plan = CudaPlan::new(effect, (width, height), 7, [1., 1.])?;
            let frame = DeviceFrame {
                ptr: src.handle as u64,
                row_bytes: (width * 16) as i32,
                width,
                height,
                half: false,
            };
            unsafe {
                renderer.render(
                    &plan,
                    frame,
                    DeviceFrame {
                        ptr: dst.handle as u64,
                        ..frame
                    },
                )?;
            }
            let mut gpu = vec![0.; input.len()];
            renderer.download(&dst, &mut gpu)?;
            let mut cpu = input.clone();
            effect.apply_effect_to_buffer::<Bgrx, f32>(
                &ctx,
                (width, height),
                &mut cpu,
                7,
                [1., 1.],
            );
            let (index, error) = cpu
                .iter()
                .zip(&gpu)
                .enumerate()
                .filter(|(i, _)| i % 4 != 3)
                .map(|(i, (a, b))| (i, (a - b).abs()))
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .unwrap();
            if gpu.iter().any(|v| !v.is_finite()) || error > 0.005 {
                return Err(format!(
                    "{width}x{height} {name}: max error={error} at {index}, CPU={}, OpenCL={}",
                    cpu[index], gpu[index]
                ));
            }
            worst = worst.max(error);
            checked += 1;
        }
    }
    println!("PASS: {checked} CPU/OpenCL comparisons; maximum RGB error {worst:.7}");
    validate_layouts(&mut renderer, &ctx)?;
    Ok(())
}

use std::{ffi::c_void, ptr};
type H = *mut c_void;
struct OwnedCl {
    _lib: libloading::Library,
    context: H,
    queue: H,
    release_context: unsafe extern "system" fn(H) -> i32,
    release_queue: unsafe extern "system" fn(H) -> i32,
}
impl OwnedCl {
    fn new() -> Result<Self, String> {
        unsafe {
            let lib = libloading::Library::new("OpenCL.dll").map_err(|e| e.to_string())?;
            let platforms: libloading::Symbol<
                unsafe extern "system" fn(u32, *mut H, *mut u32) -> i32,
            > = lib.get(b"clGetPlatformIDs\0").unwrap();
            let devices: libloading::Symbol<
                unsafe extern "system" fn(H, u64, u32, *mut H, *mut u32) -> i32,
            > = lib.get(b"clGetDeviceIDs\0").unwrap();
            let create_context: libloading::Symbol<
                unsafe extern "system" fn(*const isize, u32, *const H, H, H, *mut i32) -> H,
            > = lib.get(b"clCreateContext\0").unwrap();
            let create_queue: libloading::Symbol<
                unsafe extern "system" fn(H, H, u64, *mut i32) -> H,
            > = lib.get(b"clCreateCommandQueue\0").unwrap();
            let release_context = *lib
                .get::<unsafe extern "system" fn(H) -> i32>(b"clReleaseContext\0")
                .unwrap();
            let release_queue = *lib
                .get::<unsafe extern "system" fn(H) -> i32>(b"clReleaseCommandQueue\0")
                .unwrap();
            let mut n = 0;
            platforms(0, ptr::null_mut(), &mut n);
            let mut all = vec![ptr::null_mut(); n as usize];
            platforms(n, all.as_mut_ptr(), ptr::null_mut());
            let mut device = ptr::null_mut();
            for platform in all {
                if devices(platform, 4, 1, &mut device, ptr::null_mut()) == 0 {
                    break;
                }
            }
            if device.is_null() {
                return Err("No OpenCL GPU (CPU devices excluded)".into());
            }
            let mut error = 0;
            let context = create_context(
                ptr::null(),
                1,
                &device,
                ptr::null_mut(),
                ptr::null_mut(),
                &mut error,
            );
            if error != 0 {
                return Err(format!("Create context: {error}"));
            }
            let queue = create_queue(context, device, 0, &mut error);
            if error != 0 {
                release_context(context);
                return Err(format!("Create queue: {error}"));
            }
            Ok(Self {
                _lib: lib,
                context,
                queue,
                release_context,
                release_queue,
            })
        }
    }
}
impl Drop for OwnedCl {
    fn drop(&mut self) {
        unsafe {
            (self.release_queue)(self.queue);
            (self.release_context)(self.context);
        }
    }
}

fn validate_layouts(renderer: &mut OpenClRenderer, ctx: &Context) -> Result<(), String> {
    let (w, h) = (19usize, 11usize);
    for half in [false, true] {
        let component_bytes = if half { 2 } else { 4 };
        let pitch = (w * 4 + 12) * component_bytes;
        let mut source = vec![0xAAu8; pitch * h];
        let mut cpu = vec![0f32; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                for c in 0..4 {
                    let value = ((x * 17 + y * 13 + c * 7) % 256) as f32 / 255.;
                    let offset = y * pitch + (x * 4 + c) * component_bytes;
                    if half {
                        let value = half::f16::from_f32(value);
                        source[offset..offset + 2].copy_from_slice(&value.to_bits().to_le_bytes());
                        cpu[(y * w + x) * 4 + c] = value.to_f32();
                    } else {
                        source[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
                        cpu[(y * w + x) * 4 + c] = value;
                    }
                }
            }
        }
        let input_cpu = cpu.clone();
        let src = renderer.allocate(source.len())?;
        let dst = renderer.allocate(source.len())?;
        renderer.upload_bytes(&src, &source)?;
        let mut result = vec![0xCDu8; source.len()];
        renderer.upload_bytes(&dst, &result)?;
        let effect = NtscEffect::default();
        let plan = CudaPlan::new(&effect, (w, h), 7, [1., 1.])?;
        let frame = DeviceFrame {
            ptr: src.handle as u64,
            row_bytes: pitch as i32,
            width: w,
            height: h,
            half,
        };
        unsafe {
            renderer.render(
                &plan,
                frame,
                DeviceFrame {
                    ptr: dst.handle as u64,
                    ..frame
                },
            )?;
        }
        renderer.download_bytes(&dst, &mut result)?;
        effect.apply_effect_to_buffer::<Bgrx, f32>(ctx, (w, h), &mut cpu, 7, [1., 1.]);
        for y in 0..h {
            for x in 0..w {
                for c in 0..4 {
                    let offset = y * pitch + (x * 4 + c) * component_bytes;
                    let actual = if half {
                        half::f16::from_bits(u16::from_le_bytes(
                            result[offset..offset + 2].try_into().unwrap(),
                        ))
                        .to_f32()
                    } else {
                        f32::from_le_bytes(result[offset..offset + 4].try_into().unwrap())
                    };
                    let expected = if c == 3 {
                        input_cpu[(y * w + x) * 4 + c]
                    } else {
                        cpu[(y * w + x) * 4 + c]
                    };
                    if !actual.is_finite() || (actual - expected).abs() > 0.005 {
                        return Err(format!(
                            "Pitched half={half} pixel {x},{y},{c}: {actual} vs {expected}"
                        ));
                    }
                }
            }
            if result[y * pitch + w * 4 * component_bytes..(y + 1) * pitch]
                .iter()
                .any(|v| *v != 0xCD)
            {
                return Err("Row padding was modified".into());
            }
        }
    }
    println!("PASS: pitched BGRA 16f/32f, alpha preservation and untouched row padding");
    Ok(())
}
