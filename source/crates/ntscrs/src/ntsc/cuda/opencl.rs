//! OpenCL 1.2 backend: uses the host's context and queue on any supported vendor.
use super::*;
use alloc::{
    format,
    string::{String, ToString},
};
use std::{
    ffi::{c_char, c_void},
    mem::size_of,
    ptr,
    sync::Arc,
};
type H = *mut c_void;
type Result<T> = std::result::Result<T, String>;

macro_rules! api {
    ($($field:ident: $symbol:literal ($($arg:ty),*) -> $ret:ty),* $(,)?) => {
        pub struct Api { _library: libloading::Library, $(pub $field: unsafe extern "system" fn($($arg),*) -> $ret,)* }
        impl Api {
            pub fn load() -> Result<Arc<Self>> {
                #[cfg(windows)] let name = "OpenCL.dll";
                #[cfg(target_os="macos")] let name = "/System/Library/Frameworks/OpenCL.framework/OpenCL";
                #[cfg(not(any(windows,target_os="macos")))] let name = "libOpenCL.so.1";
                unsafe {
                    let library=libloading::Library::new(name).map_err(|e|e.to_string())?;
                    $(let $field=*library.get(concat!($symbol,"\0").as_bytes()).map_err(|e|e.to_string())?;)*
                    Ok(Arc::new(Self{_library:library,$($field,)*}))
                }
            }
        }
    }
}
api! {
    queue_info: "clGetCommandQueueInfo" (H,u32,usize,*mut c_void,*mut usize) -> i32,
    program: "clCreateProgramWithSource" (H,u32,*const *const c_char,*const usize,*mut i32) -> H,
    build: "clBuildProgram" (H,u32,*const H,*const c_char,H,H) -> i32,
    build_info: "clGetProgramBuildInfo" (H,H,u32,usize,*mut c_void,*mut usize) -> i32,
    kernel: "clCreateKernel" (H,*const c_char,*mut i32) -> H,
    buffer: "clCreateBuffer" (H,u64,usize,H,*mut i32) -> H,
    argument: "clSetKernelArg" (H,u32,usize,*const c_void) -> i32,
    launch: "clEnqueueNDRangeKernel" (H,H,u32,*const usize,*const usize,*const usize,u32,*const H,*mut H) -> i32,
    write: "clEnqueueWriteBuffer" (H,H,u32,usize,usize,*const c_void,u32,*const H,*mut H) -> i32,
    read: "clEnqueueReadBuffer" (H,H,u32,usize,usize,H,u32,*const H,*mut H) -> i32,
    copy: "clEnqueueCopyBuffer" (H,H,H,usize,usize,usize,u32,*const H,*mut H) -> i32,
    finish: "clFinish" (H) -> i32,
    release_mem: "clReleaseMemObject" (H) -> i32,
    release_kernel: "clReleaseKernel" (H) -> i32,
    release_program: "clReleaseProgram" (H) -> i32,
}
fn check(code: i32, action: &str) -> Result<()> {
    if code == 0 {
        Ok(())
    } else {
        Err(format!("{action}: OpenCL error {code}"))
    }
}
pub struct Buffer {
    pub handle: H,
    bytes: usize,
    api: Arc<Api>,
}
impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe {
            (self.api.release_mem)(self.handle);
        }
    }
}
pub struct OpenClRenderer {
    api: Arc<Api>,
    context: H,
    queue: H,
    program: H,
    frame: H,
    operation: H,
}
// The host keeps context and queue alive. Every instance serializes rendering.
unsafe impl Send for OpenClRenderer {}
impl OpenClRenderer {
    /// # Safety
    /// Context/queue must be host-owned OpenCL handles valid for this instance.
    pub unsafe fn new(context: H, queue: H) -> Result<Self> {
        if context.is_null() || queue.is_null() {
            return Err("Null OpenCL context/queue".into());
        }
        let api = Api::load()?;
        let mut device: H = ptr::null_mut();
        let mut queue_context: H = ptr::null_mut();
        unsafe {
            check(
                (api.queue_info)(
                    queue,
                    0x1091,
                    size_of::<H>(),
                    (&mut device as *mut H).cast(),
                    ptr::null_mut(),
                ),
                "Query queue device",
            )?;
            check(
                (api.queue_info)(
                    queue,
                    0x1090,
                    size_of::<H>(),
                    (&mut queue_context as *mut H).cast(),
                    ptr::null_mut(),
                ),
                "Query queue context",
            )?;
        }
        if queue_context != context {
            return Err("OpenCL queue/context mismatch".into());
        }
        let mut properties = 0u64;
        unsafe {
            check(
                (api.queue_info)(
                    queue,
                    0x1093,
                    size_of::<u64>(),
                    (&mut properties as *mut u64).cast(),
                    ptr::null_mut(),
                ),
                "Query queue properties",
            )?;
        }
        if properties & 1 != 0 {
            return Err("OpenCL backend requires an in-order host queue".into());
        }
        let source = include_str!("opencl_kernels.cl");
        let data = source.as_ptr().cast();
        let len = source.len();
        let mut error = 0;
        let program = unsafe { (api.program)(context, 1, &data, &len, &mut error) };
        check(error, "Create OpenCL program")?;
        let code = unsafe {
            (api.build)(
                program,
                1,
                &device,
                c"-cl-std=CL1.2".as_ptr(),
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };
        if code != 0 {
            let mut len = 0;
            unsafe {
                (api.build_info)(program, device, 0x1183, 0, ptr::null_mut(), &mut len);
            }
            let mut log = vec![0u8; len];
            unsafe {
                (api.build_info)(
                    program,
                    device,
                    0x1183,
                    len,
                    log.as_mut_ptr().cast(),
                    ptr::null_mut(),
                );
                (api.release_program)(program);
            }
            return Err(format!(
                "OpenCL compilation: {}",
                String::from_utf8_lossy(&log)
            ));
        }
        let frame = unsafe { (api.kernel)(program, c"ntsc_frame".as_ptr(), &mut error) };
        if error != 0 {
            unsafe {
                (api.release_program)(program);
            }
            return Err("Cannot create OpenCL frame kernel".into());
        }
        let operation = unsafe { (api.kernel)(program, c"ntsc_operation".as_ptr(), &mut error) };
        if error != 0 {
            unsafe {
                (api.release_kernel)(frame);
                (api.release_program)(program);
            }
            return Err("Cannot create OpenCL operation kernel".into());
        }
        Ok(Self {
            api,
            context,
            queue,
            program,
            frame,
            operation,
        })
    }
    pub fn allocate(&self, bytes: usize) -> Result<Buffer> {
        let mut error = 0;
        let handle = unsafe {
            (self.api.buffer)(self.context, 1, bytes.max(4), ptr::null_mut(), &mut error)
        };
        check(error, "Allocate OpenCL buffer")?;
        Ok(Buffer {
            handle,
            bytes: bytes.max(4),
            api: self.api.clone(),
        })
    }
    pub fn upload(&self, buffer: &Buffer, data: &[f32]) -> Result<()> {
        if std::mem::size_of_val(data) > buffer.bytes {
            return Err("OpenCL upload exceeds buffer".into());
        }
        check(
            unsafe {
                (self.api.write)(
                    self.queue,
                    buffer.handle,
                    1,
                    0,
                    std::mem::size_of_val(data),
                    data.as_ptr().cast(),
                    0,
                    ptr::null(),
                    ptr::null_mut(),
                )
            },
            "Upload OpenCL data",
        )
    }
    pub fn download(&self, buffer: &Buffer, data: &mut [f32]) -> Result<()> {
        if std::mem::size_of_val(data) > buffer.bytes {
            return Err("OpenCL read exceeds buffer".into());
        }
        check(
            unsafe {
                (self.api.read)(
                    self.queue,
                    buffer.handle,
                    1,
                    0,
                    std::mem::size_of_val(data),
                    data.as_mut_ptr().cast(),
                    0,
                    ptr::null(),
                    ptr::null_mut(),
                )
            },
            "Read OpenCL data",
        )
    }
    pub fn upload_bytes(&self, buffer: &Buffer, data: &[u8]) -> Result<()> {
        if data.len() > buffer.bytes {
            return Err("OpenCL upload exceeds buffer".into());
        }
        check(
            unsafe {
                (self.api.write)(
                    self.queue,
                    buffer.handle,
                    1,
                    0,
                    data.len(),
                    data.as_ptr().cast(),
                    0,
                    ptr::null(),
                    ptr::null_mut(),
                )
            },
            "Upload OpenCL bytes",
        )
    }
    pub fn download_bytes(&self, buffer: &Buffer, data: &mut [u8]) -> Result<()> {
        if data.len() > buffer.bytes {
            return Err("OpenCL read exceeds buffer".into());
        }
        check(
            unsafe {
                (self.api.read)(
                    self.queue,
                    buffer.handle,
                    1,
                    0,
                    data.len(),
                    data.as_mut_ptr().cast(),
                    0,
                    ptr::null(),
                    ptr::null_mut(),
                )
            },
            "Read OpenCL bytes",
        )
    }
    fn arg<T>(&self, kernel: H, index: u32, value: &T) -> Result<()> {
        check(
            unsafe {
                (self.api.argument)(kernel, index, size_of::<T>(), (value as *const T).cast())
            },
            "Set OpenCL argument",
        )
    }
    fn launch(&self, kernel: H, count: usize) -> Result<()> {
        if count == 0 {
            return Ok(());
        }
        // Let the device driver choose a suitable local size for its architecture.
        check(
            unsafe {
                (self.api.launch)(
                    self.queue,
                    kernel,
                    1,
                    ptr::null(),
                    &count,
                    ptr::null(),
                    0,
                    ptr::null(),
                    ptr::null_mut(),
                )
            },
            "Launch OpenCL kernel",
        )
    }
    fn copy(&self, src: H, dst: H, bytes: usize) -> Result<()> {
        check(
            unsafe {
                (self.api.copy)(
                    self.queue,
                    src,
                    dst,
                    0,
                    0,
                    bytes,
                    0,
                    ptr::null(),
                    ptr::null_mut(),
                )
            },
            "Copy OpenCL planes",
        )
    }
    fn frame_pass(
        &self,
        input: DeviceFrame,
        output: DeviceFrame,
        planes: H,
        p: &Params,
    ) -> Result<()> {
        let kernel = self.frame;
        self.arg(kernel, 0, &(input.ptr as H))?;
        self.arg(kernel, 1, &(output.ptr as H))?;
        self.arg(kernel, 2, &planes)?;
        self.arg(kernel, 3, p)?;
        self.arg(kernel, 4, &input.row_bytes)?;
        self.arg(kernel, 5, &output.row_bytes)?;
        self.arg(kernel, 6, &(input.height as i32))?;
        self.arg(kernel, 7, &(input.half as i32))?;
        self.arg(kernel, 8, &(output.half as i32))?;
        self.launch(kernel, input.width * input.height)
    }
    /// # Safety
    /// Frames contain live cl_mem handles in this context, with valid pitches.
    pub unsafe fn render(
        &mut self,
        plan: &CudaPlan,
        input: DeviceFrame,
        output: DeviceFrame,
    ) -> Result<()> {
        if input.ptr == 0
            || output.ptr == 0
            || input.ptr == output.ptr
            || input.width != plan.width
            || input.height != plan.height
            || output.width != plan.width
            || output.height != plan.height
        {
            return Err("Invalid OpenCL frame".into());
        }
        // OpenCL buffers have no pointer-offset semantics; host frames have a
        // zero byte origin and therefore require positive row pitches.
        for frame in [input, output] {
            if frame.row_bytes <= 0
                || (frame.row_bytes as usize) < frame.width * if frame.half { 8 } else { 16 }
            {
                return Err("Unsupported OpenCL row pitch".into());
            }
        }
        let n = plan.width * plan.height;
        let planes = self.allocate(n * 12)?;
        let scratch = self.allocate(plan.scratch_floats() * 4)?;
        let modulated = self.allocate(n * 4)?;
        let auxiliary = self.allocate(plan.auxiliary.len() * 4)?;
        if !plan.auxiliary.is_empty() {
            self.upload(&auxiliary, &plan.auxiliary)?;
        }
        let result = (|| {
            for field in &plan.fields {
                let mut p = field.params;
                p.op = 0;
                self.frame_pass(input, output, planes.handle, &p)?;
                for command in &field.commands {
                    let mut p = *command;
                    if p.op == 3 {
                        self.copy(
                            planes.handle,
                            modulated.handle,
                            p.w as usize * p.h as usize * 4,
                        )?;
                        continue;
                    }
                    if matches!(p.op, 7 | 8) {
                        self.copy(
                            planes.handle,
                            scratch.handle,
                            p.w as usize * p.h as usize * 12,
                        )?;
                    }
                    self.arg(self.operation, 0, &planes.handle)?;
                    self.arg(self.operation, 1, &scratch.handle)?;
                    self.arg(self.operation, 2, &modulated.handle)?;
                    self.arg(self.operation, 3, &auxiliary.handle)?;
                    let passes: Vec<(i32, usize)> = if p.op == 1 {
                        let count = (p.w as usize).div_ceil(128) * p.h as usize;
                        vec![(13, count), (14, p.h as usize), (15, count)]
                    } else {
                        vec![(
                            p.op,
                            if p.op == 10 {
                                p.h as usize
                            } else if p.op == 9 {
                                p.detail as usize
                            } else {
                                p.w as usize * p.h as usize
                            },
                        )]
                    };
                    for (op, count) in passes {
                        p.op = op;
                        self.arg(self.operation, 4, &p)?;
                        self.launch(self.operation, count)?;
                    }
                }
                p.op = 1;
                self.frame_pass(input, output, planes.handle, &p)?;
            }
            Ok(())
        })();
        let sync = check(
            unsafe { (self.api.finish)(self.queue) },
            "Finish OpenCL render",
        );
        result.and(sync)
    }
}
impl Drop for OpenClRenderer {
    fn drop(&mut self) {
        unsafe {
            (self.api.finish)(self.queue);
            (self.api.release_kernel)(self.frame);
            (self.api.release_kernel)(self.operation);
            (self.api.release_program)(self.program);
        }
    }
}
