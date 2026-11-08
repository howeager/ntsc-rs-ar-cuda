//! CUDA backend for device-resident Premiere BGRA frames.
//! Pixel processing stays on the GPU; small row parameters and sparse snow events
//! use the same CPU RNG as the software renderer.
use super::*;
use alloc::{vec, vec::Vec};
use std::{
    ffi::{CString, c_void},
    sync::Arc,
};
pub mod driver;
#[cfg(feature = "opencl")]
pub mod opencl;
mod plan;
use driver::{DevicePtr, Driver, Handle, Result};
pub use plan::CudaPlan;

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(super) struct Params {
    w: i32,
    h: i32,
    op: i32,
    plane: i32,
    mode: i32,
    delay: i32,
    first: i32,
    frame: i32,
    aux: i32,
    vertical: i32,
    shift: f32,
    frequency: f32,
    intensity: f32,
    detail: i32,
    coefficients: [f32; 7],
    initial: [f32; 4],
    transition: [f32; 9],
}

/// A device-resident BGRA frame. Row pitch is signed and expressed in bytes.
#[derive(Clone, Copy)]
pub struct DeviceFrame {
    pub ptr: DevicePtr,
    pub row_bytes: i32,
    pub width: usize,
    pub height: usize,
    /// Premiere's GPU BGRA 16f format; otherwise BGRA 32f.
    pub half: bool,
}

pub struct CudaRenderer {
    driver: Arc<Driver>,
    context: Handle,
    module: Handle,
    frame_kernel: Handle,
    operation_kernel: Handle,
}

// CUDA modules are context-bound; every operation pushes that context, and
// rendering requires &mut self. No context is left current on the calling thread.
unsafe impl Send for CudaRenderer {}

impl CudaRenderer {
    /// # Safety
    /// `context` must be a live CUDA context and outlive this renderer.
    pub unsafe fn new(context: Handle) -> Result<Self> {
        if context.is_null() {
            return Err("Null CUDA context".into());
        }
        let driver = Driver::load()?;
        let guard = driver.enter(context)?;
        #[cfg(feature = "cuda")]
        let ptx = CString::new(include_str!(concat!(env!("OUT_DIR"), "/ntsc.ptx"))).unwrap();
        #[cfg(not(feature = "cuda"))]
        let ptx = CString::new("").unwrap();
        let mut module = std::ptr::null_mut();
        driver.check(
            unsafe { (driver.module_load)(&mut module, ptx.as_ptr().cast()) },
            "cuModuleLoadData",
        )?;
        let load = |name: &str| -> Result<Handle> {
            let name = CString::new(name).unwrap();
            let mut function = std::ptr::null_mut();
            driver.check(
                unsafe { (driver.function)(&mut function, module, name.as_ptr()) },
                "cuModuleGetFunction",
            )?;
            Ok(function)
        };
        let functions = (|| Ok((load("ntsc_frame")?, load("ntsc_operation")?)))();
        let (frame_kernel, operation_kernel) = match functions {
            Ok(v) => v,
            Err(e) => {
                unsafe {
                    (driver.module_unload)(module);
                }
                return Err(e);
            }
        };
        drop(guard);
        Ok(Self {
            driver,
            context,
            module,
            frame_kernel,
            operation_kernel,
        })
    }

    fn launch(
        &self,
        function: Handle,
        count: usize,
        stream: Handle,
        args: &mut [*mut c_void],
    ) -> Result<()> {
        self.driver.check(
            unsafe {
                (self.driver.launch)(
                    function,
                    count.div_ceil(128) as u32,
                    1,
                    1,
                    128,
                    1,
                    1,
                    0,
                    stream,
                    args.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            },
            "cuLaunchKernel",
        )
    }

    /// Render using memory allocated by the host's GPU device suite.
    ///
    /// # Safety
    /// Frame and workspace pointers must belong to this context, be valid for
    /// their stated dimensions, and remain live until this call returns. Input
    /// and output must not alias. `workspace` must have `plan.workspace_bytes()`
    /// writable bytes. The host must have completed writes to the input first.
    pub unsafe fn render(
        &mut self,
        plan: &CudaPlan,
        input: DeviceFrame,
        output: DeviceFrame,
        workspace: DevicePtr,
        workspace_bytes: usize,
        stream: Handle,
    ) -> Result<()> {
        if input.width != plan.width
            || input.height != plan.height
            || output.width != plan.width
            || output.height != plan.height
            || workspace_bytes < plan.workspace_bytes()
            || input.ptr == 0
            || output.ptr == 0
            || workspace == 0
            || input.ptr == output.ptr
        {
            return Err("Invalid CUDA frame or workspace".into());
        }
        for frame in [input, output] {
            let minimum = frame
                .width
                .checked_mul(if frame.half { 8 } else { 16 })
                .ok_or("Frame too wide")?;
            if (frame.row_bytes as i64).unsigned_abs() < minimum as u64 {
                return Err("Invalid CUDA row pitch".into());
            }
        }
        let _guard = self.driver.enter(self.context)?;
        let n = (plan.width * plan.height) as u64;
        let mut planes = workspace;
        let mut scratch = workspace + n * 3 * 4;
        let mut modulated = scratch + plan.scratch_floats() as u64 * 4;
        let mut auxiliary = modulated + n * 4;
        self.driver.check(
            unsafe {
                (self.driver.upload)(
                    auxiliary,
                    plan.auxiliary.as_ptr().cast(),
                    plan.auxiliary.len() * 4,
                )
            },
            "Upload row parameters",
        )?;
        let result = (|| {
            for field in &plan.fields {
                let mut params = field.params;
                let mut source = input.ptr;
                let mut dest = output.ptr;
                let mut source_pitch = input.row_bytes;
                let mut dest_pitch = output.row_bytes;
                let mut height = plan.height as i32;
                let mut half_in = input.half as i32;
                let mut half_out = output.half as i32;
                params.op = 0;
                self.launch(
                    self.frame_kernel,
                    plan.width * plan.height,
                    stream,
                    &mut [
                        (&mut source as *mut DevicePtr).cast(),
                        (&mut dest as *mut DevicePtr).cast(),
                        (&mut planes as *mut DevicePtr).cast(),
                        (&mut params as *mut Params).cast(),
                        (&mut source_pitch as *mut i32).cast(),
                        (&mut dest_pitch as *mut i32).cast(),
                        (&mut height as *mut i32).cast(),
                        (&mut half_in as *mut i32).cast(),
                        (&mut half_out as *mut i32).cast(),
                    ],
                )?;
                for command in &field.commands {
                    let mut p = *command;
                    if p.op == 1 {
                        // Parallel local filters, prefix of segment boundary
                        // states, then a homogeneous-state correction.
                        let segments = (p.w as usize).div_ceil(128);
                        for (op, count) in [
                            (13, segments * p.h as usize),
                            (14, p.h as usize),
                            (15, segments * p.h as usize),
                        ] {
                            p.op = op;
                            self.launch(
                                self.operation_kernel,
                                count,
                                stream,
                                &mut [
                                    (&mut planes as *mut DevicePtr).cast(),
                                    (&mut scratch as *mut DevicePtr).cast(),
                                    (&mut modulated as *mut DevicePtr).cast(),
                                    (&mut auxiliary as *mut DevicePtr).cast(),
                                    (&mut p as *mut Params).cast(),
                                ],
                            )?;
                        }
                        continue;
                    }
                    // Preserve original samples for spatial passes and demodulation.
                    if p.op == 3 {
                        self.driver.check(
                            unsafe {
                                (self.driver.copy)(
                                    modulated,
                                    planes,
                                    p.w as usize * p.h as usize * 4,
                                    stream,
                                )
                            },
                            "Save modulated signal",
                        )?;
                        continue;
                    }
                    if matches!(p.op, 7 | 8) {
                        self.driver.check(
                            unsafe {
                                (self.driver.copy)(
                                    scratch,
                                    planes,
                                    p.w as usize * p.h as usize * 12,
                                    stream,
                                )
                            },
                            "Save source planes",
                        )?;
                    }
                    let count = if p.op == 1 || p.op == 10 {
                        p.h as usize
                    } else if p.op == 9 {
                        p.detail as usize
                    } else {
                        p.w as usize * p.h as usize
                    };
                    if count == 0 {
                        continue;
                    }
                    self.launch(
                        self.operation_kernel,
                        count,
                        stream,
                        &mut [
                            (&mut planes as *mut DevicePtr).cast(),
                            (&mut scratch as *mut DevicePtr).cast(),
                            (&mut modulated as *mut DevicePtr).cast(),
                            (&mut auxiliary as *mut DevicePtr).cast(),
                            (&mut p as *mut Params).cast(),
                        ],
                    )?;
                }
                params.op = 1;
                self.launch(
                    self.frame_kernel,
                    plan.width * plan.height,
                    stream,
                    &mut [
                        (&mut source as *mut DevicePtr).cast(),
                        (&mut dest as *mut DevicePtr).cast(),
                        (&mut planes as *mut DevicePtr).cast(),
                        (&mut params as *mut Params).cast(),
                        (&mut source_pitch as *mut i32).cast(),
                        (&mut dest_pitch as *mut i32).cast(),
                        (&mut height as *mut i32).cast(),
                        (&mut half_in as *mut i32).cast(),
                        (&mut half_out as *mut i32).cast(),
                    ],
                )?;
            }
            Ok(())
        })();
        // Always drain submitted kernels before the host can free workspace,
        // including the error path. Render completion hands ownership to Premiere.
        let sync = self.driver.check(
            unsafe { (self.driver.sync)(stream) },
            "CUDA render completion",
        );
        result.and(sync)
    }
}

impl Drop for CudaRenderer {
    fn drop(&mut self) {
        if let Ok(_guard) = self.driver.enter(self.context) {
            unsafe {
                (self.driver.module_unload)(self.module);
            }
        }
    }
}
