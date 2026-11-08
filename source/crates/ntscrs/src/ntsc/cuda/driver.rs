//! Small dynamically loaded CUDA Driver API surface. No CUDA runtime DLL is shipped.
use alloc::{
    format,
    string::{String, ToString},
};
use std::{
    ffi::{CStr, c_void},
    sync::Arc,
};

pub type Handle = *mut c_void;
pub type DevicePtr = u64;
pub type Result<T> = std::result::Result<T, String>;

macro_rules! api {
    ($($field:ident: $symbol:literal ($($arg:ty),*) ),* $(,)?) => {
        pub struct Driver {
            _library: libloading::Library,
            $(pub $field: unsafe extern "system" fn($($arg),*) -> i32,)*
        }
        impl Driver {
            pub fn load() -> Result<Arc<Self>> {
                #[cfg(windows)] let name = "nvcuda.dll";
                #[cfg(not(windows))] let name = "libcuda.so.1";
                unsafe {
                    let library = libloading::Library::new(name).map_err(|e| e.to_string())?;
                    $(let $field = *library.get(concat!($symbol, "\0").as_bytes()).map_err(|e| e.to_string())?;)*
                    let driver = Arc::new(Self { _library: library, $($field,)* });
                    driver.check((driver.init)(0), "cuInit")?;
                    Ok(driver)
                }
            }
        }
    }
}
api! {
    init: "cuInit" (u32),
    device_get: "cuDeviceGet" (*mut i32, i32),
    retain: "cuDevicePrimaryCtxRetain" (*mut Handle, i32),
    release: "cuDevicePrimaryCtxRelease_v2" (i32),
    push: "cuCtxPushCurrent_v2" (Handle),
    pop: "cuCtxPopCurrent_v2" (*mut Handle),
    module_load: "cuModuleLoadData" (*mut Handle, *const c_void),
    module_unload: "cuModuleUnload" (Handle),
    function: "cuModuleGetFunction" (*mut Handle, Handle, *const i8),
    alloc: "cuMemAlloc_v2" (*mut DevicePtr, usize),
    free: "cuMemFree_v2" (DevicePtr),
    upload: "cuMemcpyHtoD_v2" (DevicePtr, *const c_void, usize),
    download: "cuMemcpyDtoH_v2" (*mut c_void, DevicePtr, usize),
    copy: "cuMemcpyDtoDAsync_v2" (DevicePtr, DevicePtr, usize, Handle),
    launch: "cuLaunchKernel" (Handle, u32, u32, u32, u32, u32, u32, u32, Handle, *mut *mut c_void, *mut *mut c_void),
    stream_create: "cuStreamCreate" (*mut Handle, u32),
    stream_destroy: "cuStreamDestroy_v2" (Handle),
    sync: "cuStreamSynchronize" (Handle),
    error_name: "cuGetErrorName" (i32, *mut *const i8),
}

impl Driver {
    pub fn check(&self, code: i32, operation: &str) -> Result<()> {
        if code == 0 {
            return Ok(());
        }
        let mut name = std::ptr::null();
        unsafe {
            (self.error_name)(code, &mut name);
        }
        let description = if name.is_null() {
            code.to_string()
        } else {
            unsafe { CStr::from_ptr(name) }
                .to_string_lossy()
                .into_owned()
        };
        Err(format!("{operation}: {description}"))
    }

    pub(super) fn enter(&self, context: Handle) -> Result<ContextGuard<'_>> {
        self.check(unsafe { (self.push)(context) }, "cuCtxPushCurrent")?;
        Ok(ContextGuard(self))
    }
}

pub struct ContextGuard<'a>(&'a Driver);
impl Drop for ContextGuard<'_> {
    fn drop(&mut self) {
        let mut previous = std::ptr::null_mut();
        unsafe {
            (self.0.pop)(&mut previous);
        }
    }
}

/// Used by the hardware validation tool. Premiere supplies its own context and memory.
pub struct OwnedDevice {
    pub driver: Arc<Driver>,
    pub context: Handle,
    device: i32,
}
impl OwnedDevice {
    pub fn new(ordinal: i32) -> Result<Self> {
        let driver = Driver::load()?;
        let mut device = 0;
        driver.check(
            unsafe { (driver.device_get)(&mut device, ordinal) },
            "cuDeviceGet",
        )?;
        let mut context = std::ptr::null_mut();
        driver.check(
            unsafe { (driver.retain)(&mut context, device) },
            "cuDevicePrimaryCtxRetain",
        )?;
        Ok(Self {
            driver,
            context,
            device,
        })
    }
    pub fn allocate(&self, bytes: usize) -> Result<Buffer<'_>> {
        let _guard = self.driver.enter(self.context)?;
        let mut ptr = 0;
        self.driver.check(
            unsafe { (self.driver.alloc)(&mut ptr, bytes.max(1)) },
            "cuMemAlloc",
        )?;
        Ok(Buffer {
            device: self,
            ptr,
            bytes,
        })
    }
}
impl Drop for OwnedDevice {
    fn drop(&mut self) {
        unsafe {
            (self.driver.release)(self.device);
        }
    }
}
pub struct Buffer<'a> {
    device: &'a OwnedDevice,
    pub ptr: DevicePtr,
    bytes: usize,
}
impl Buffer<'_> {
    pub fn upload<T: Copy>(&self, data: &[T]) -> Result<()> {
        let bytes = std::mem::size_of_val(data);
        if bytes > self.bytes {
            return Err("Upload exceeds allocation".into());
        }
        let _guard = self.device.driver.enter(self.device.context)?;
        self.device.driver.check(
            unsafe { (self.device.driver.upload)(self.ptr, data.as_ptr().cast(), bytes) },
            "cuMemcpyHtoD",
        )
    }
    pub fn download<T: Copy>(&self, data: &mut [T]) -> Result<()> {
        let bytes = std::mem::size_of_val(data);
        if bytes > self.bytes {
            return Err("Download exceeds allocation".into());
        }
        let _guard = self.device.driver.enter(self.device.context)?;
        self.device.driver.check(
            unsafe { (self.device.driver.download)(data.as_mut_ptr().cast(), self.ptr, bytes) },
            "cuMemcpyDtoH",
        )
    }
}
impl Drop for Buffer<'_> {
    fn drop(&mut self) {
        if let Ok(_guard) = self.device.driver.enter(self.device.context) {
            unsafe {
                (self.device.driver.free)(self.ptr);
            }
        }
    }
}
