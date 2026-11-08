use core::slice;
use std::{
    marker::PhantomData,
    mem::{ManuallyDrop, MaybeUninit},
    ptr::{self, NonNull},
};

use crate::bindings::*;

pub struct ImageDataHandle<T> {
    handle: OfxImageMemoryHandle,
    _ty: PhantomData<T>,
    len: usize,
    suite: &'static OfxImageEffectSuiteV1,
}

impl<T> ImageDataHandle<T> {
    pub fn new_uninit(
        len: usize,
        suite: &'static OfxImageEffectSuiteV1,
    ) -> OfxResult<ImageDataHandle<MaybeUninit<T>>> {
        let mut handle = ptr::null_mut();
        unsafe {
            (suite.imageMemoryAlloc.unwrap())(
                ptr::null_mut(),
                len * std::mem::size_of::<T>(),
                &mut handle,
            )
            .ofx_ok()?;
        }

        Ok(ImageDataHandle {
            handle,
            _ty: PhantomData,
            len,
            suite,
        })
    }

    pub fn new_zeroed(
        len: usize,
        suite: &'static OfxImageEffectSuiteV1,
    ) -> OfxResult<ImageDataHandle<MaybeUninit<T>>> {
        let mut handle = Self::new_uninit(len, suite)?;

        {
            let mut locked = handle.lock()?;
            let data_slice: &mut [MaybeUninit<T>] = locked.as_mut();
            let data = data_slice.as_mut_ptr();
            unsafe {
                data.write_bytes(0, data_slice.len());
            }
        }

        Ok(handle)
    }

    pub fn lock(&mut self) -> OfxResult<LockedHandle<'_, T>> {
        let mut ptr = ptr::null_mut();
        unsafe {
            (self.suite.imageMemoryLock.unwrap())(self.handle, &mut ptr).ofx_ok()?;
        }
        // Safety: per the API contract of imageMemoryLock, `ptr` must contain a valid pointer if the call succeeded
        let data = unsafe { NonNull::new_unchecked(ptr as *mut T) };

        Ok(LockedHandle { data, parent: self })
    }
}

impl<T> ImageDataHandle<MaybeUninit<T>> {
    pub unsafe fn assume_init(self) -> ImageDataHandle<T> {
        let this = ManuallyDrop::new(self);
        ImageDataHandle {
            handle: this.handle,
            _ty: PhantomData,
            len: this.len,
            suite: this.suite,
        }
    }
}

impl<T> Drop for ImageDataHandle<T> {
    fn drop(&mut self) {
        unsafe {
            (self.suite.imageMemoryFree.unwrap())(self.handle)
                .ofx_ok()
                .unwrap()
        };
    }
}

pub struct LockedHandle<'a, T> {
    data: NonNull<T>,
    parent: &'a mut ImageDataHandle<T>,
}

impl<T> AsRef<[T]> for LockedHandle<'_, T> {
    fn as_ref(&self) -> &[T] {
        unsafe { slice::from_raw_parts(self.data.as_ptr(), self.parent.len) }
    }
}

impl<T> AsMut<[T]> for LockedHandle<'_, T> {
    fn as_mut(&mut self) -> &mut [T] {
        unsafe { slice::from_raw_parts_mut(self.data.as_ptr(), self.parent.len) }
    }
}

impl<T> Drop for LockedHandle<'_, T> {
    fn drop(&mut self) {
        unsafe {
            (self.parent.suite.imageMemoryUnlock.unwrap())(self.parent.handle)
                .ofx_ok()
                .unwrap();
        }
    }
}
