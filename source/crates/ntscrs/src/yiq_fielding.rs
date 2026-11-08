use alloc::{boxed::Box, vec};
use core::mem::{self, MaybeUninit};

use crate::{ctx::Context, settings::UseField, thread_pool::ZipChunks};

use fearless_simd::{Level, dispatch, f32x4, i32x4, prelude::*};

#[inline(always)]
fn rgb_to_yiq<S: Simd>(rgb: f32x4<S>) -> f32x4<S> {
    // This is a matrix multiply with the matrix being:
    // [
    //     [0.299, 0.5959, 0.2115],
    //     [0.587, -0.2746, -0.5227],
    //     [0.114, -0.3213, 0.3112],
    // ]

    let simd = rgb.witness();
    let rr = f32x4::splat(simd, rgb[0]);
    let gg = f32x4::splat(simd, rgb[1]);
    let bb = f32x4::splat(simd, rgb[2]);

    rr.mul_add(
        f32x4::simd_from(simd, [0.299, 0.5959, 0.2115, 0.0]),
        gg.mul_add(
            f32x4::simd_from(simd, [0.587, -0.2746, -0.5227, 0.0]),
            bb * f32x4::simd_from(simd, [0.114, -0.3213, 0.3112, 0.0]),
        ),
    )
}

#[inline(always)]
fn yiq_to_rgb<S: Simd>(yiq: f32x4<S>) -> f32x4<S> {
    // This is a matrix multiply with the matrix being:
    // [
    //    [1.0, 1.0, 1.0],
    //    [0.956, -0.272, -1.106],
    //    [0.619, -0.647, 1.703],
    // ]
    // Since the top row is all ones, we can skip it.

    let simd = yiq.witness();
    let yy = f32x4::splat(simd, yiq[0]);
    let ii = f32x4::splat(simd, yiq[1]);
    let qq = f32x4::splat(simd, yiq[2]);

    qq.mul_add(
        f32x4::simd_from(simd, [0.619, -0.647, 1.703, 0.0]),
        ii.mul_add(f32x4::simd_from(simd, [0.956, -0.272, -1.106, 0.0]), yy),
    )
}

/// How are the fields being stored? This is similar to the `UseField` enum in the settings module, but doesn't include
/// `Alternating`--that is turned into either `Upper` or `Lower` depending on the frame number.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum YiqField {
    /// Use the upper (even-numbered, when indexing from 0) fields from the frame.
    Upper,
    /// Use the lower (odd-numbered, when indexing from 0) fields from the frame.
    Lower,
    /// Use both fields from the frame--somewhat inaccurate due to the lack of interlacing but may look nicer.
    Both,
    /// Use the upper fields and then the lower fields, in effect interlacing then combining them.
    InterleavedUpper,
    /// Use the lower fields and then the upper fields, in effect interlacing then combining them.
    InterleavedLower,
}

impl YiqField {
    /// The number of rows needed in the YIQ buffer to store data for a given field setting.
    #[inline(always)]
    pub fn num_image_rows(&self, image_height: usize) -> usize {
        // On an image with an odd input height, we do ceiling division if we render upper-field-first
        // (take an image 3 pixels tall. it goes render, skip, render--that's 2 renders) but floor division if we
        // render lower-field-first (skip, render, skip--only 1 render).
        match self {
            Self::Upper => image_height.div_ceil(2),
            Self::Lower => (image_height / 2).max(1),
            Self::Both | Self::InterleavedUpper | Self::InterleavedLower => image_height,
        }
    }

    /// The number of rows that correspond to this field for a given vertical resolution. Can be 0 unlike
    /// `num_image_rows`.
    #[inline(always)]
    pub fn num_actual_image_rows(&self, image_height: usize) -> usize {
        // On an image with an odd input height, we do ceiling division if we render upper-field-first
        // (take an image 3 pixels tall. it goes render, skip, render--that's 2 renders) but floor division if we
        // render lower-field-first (skip, render, skip--only 1 render).
        match self {
            Self::Upper => image_height.div_ceil(2),
            Self::Lower => image_height / 2,
            Self::Both | Self::InterleavedUpper | Self::InterleavedLower => image_height,
        }
    }

    /// Flips the field parity--upper becomes lower and vice versa.
    pub fn flip(&self) -> Self {
        match self {
            Self::Upper => Self::Lower,
            Self::Lower => Self::Upper,
            Self::Both => Self::Both,
            Self::InterleavedUpper => Self::InterleavedLower,
            Self::InterleavedLower => Self::InterleavedUpper,
        }
    }
}

mod private {
    pub trait Sealed {}

    // Implement for those same types, but no others.
    impl Sealed for f32 {}
    impl Sealed for u16 {}
    impl Sealed for super::AfterEffectsU16 {}
    impl Sealed for i16 {}
    impl Sealed for u8 {}
}

/// Trait for converting various pixel formats to and from the f32 representation used when processing the image.
pub trait Normalize: Sized + Copy + Send + Sync + private::Sealed {
    const ONE: Self;
    fn from_norm<S: Simd>(value: f32x4<S>) -> [Self; 4];
    fn to_norm<S: Simd>(simd: S, value: [Self; 4]) -> f32x4<S>;
}

impl Normalize for f32 {
    const ONE: Self = 1.0;
    #[inline(always)]
    fn from_norm<S: Simd>(value: f32x4<S>) -> [Self; 4] {
        value.into()
    }

    #[inline(always)]
    fn to_norm<S: Simd>(simd: S, value: [Self; 4]) -> f32x4<S> {
        value.simd_into(simd)
    }
}

impl Normalize for u16 {
    const ONE: Self = Self::MAX;
    #[inline(always)]
    fn from_norm<S: Simd>(value: f32x4<S>) -> [Self; 4] {
        let min = Self::MIN as f32;
        let max = Self::MAX as f32;
        let multiplied: i32x4<S> = (value * max).min(max).max(min).to_int();
        [
            multiplied[0] as u16,
            multiplied[1] as u16,
            multiplied[2] as u16,
            multiplied[3] as u16,
        ]
    }

    #[inline(always)]
    fn to_norm<S: Simd>(simd: S, value: [Self; 4]) -> f32x4<S> {
        let values: f32x4<S> = i32x4::simd_from(
            simd,
            [
                value[0] as i32,
                value[1] as i32,
                value[2] as i32,
                value[3] as i32,
            ],
        )
        .to_float();
        values * (1.0 / Self::MAX as f32)
    }
}

#[repr(transparent)]
#[derive(Clone, Copy)]
/// Special u16 pixel format for After Effects.
/// Ranges from 0 to 32768--anything outside of that will be wrapped.
/// That's right! *Not* 0 to 32767, the maximum for an i16, but one *above* that.
/// As far as I can tell, the values 32769-65535 are entirely unused and wasted. Why, Adobe, why?
pub struct AfterEffectsU16(pub u16);

impl Normalize for AfterEffectsU16 {
    const ONE: Self = Self(32768);
    #[inline(always)]
    fn from_norm<S: Simd>(value: f32x4<S>) -> [Self; 4] {
        let min = 0.0;
        let max = 32768.0;
        let multiplied: i32x4<S> = (value * max).min(max).max(min).to_int();
        [
            Self(multiplied[0] as u16),
            Self(multiplied[1] as u16),
            Self(multiplied[2] as u16),
            Self(multiplied[3] as u16),
        ]
    }

    #[inline(always)]
    fn to_norm<S: Simd>(simd: S, value: [Self; 4]) -> f32x4<S> {
        let values: f32x4<S> = i32x4::simd_from(
            simd,
            [
                value[0].0 as i32,
                value[1].0 as i32,
                value[2].0 as i32,
                value[3].0 as i32,
            ],
        )
        .to_float();
        values * (1.0 / 32768.0)
    }
}

impl Normalize for i16 {
    const ONE: Self = Self::MAX;
    #[inline(always)]
    fn from_norm<S: Simd>(value: f32x4<S>) -> [Self; 4] {
        let min = Self::MIN as f32;
        let max = Self::MAX as f32;
        let multiplied: i32x4<S> = (value * max).min(max).max(min).to_int();
        [
            multiplied[0] as i16,
            multiplied[1] as i16,
            multiplied[2] as i16,
            multiplied[3] as i16,
        ]
    }

    #[inline(always)]
    fn to_norm<S: Simd>(simd: S, value: [Self; 4]) -> f32x4<S> {
        let values: f32x4<S> = i32x4::simd_from(
            simd,
            [
                value[0] as i32,
                value[1] as i32,
                value[2] as i32,
                value[3] as i32,
            ],
        )
        .to_float();
        values * (1.0 / Self::MAX as f32)
    }
}

impl Normalize for u8 {
    const ONE: Self = Self::MAX;
    #[inline(always)]
    fn from_norm<S: Simd>(value: f32x4<S>) -> [Self; 4] {
        let min = Self::MIN as f32;
        let max = Self::MAX as f32;
        let multiplied: i32x4<S> = (value * max).min(max).max(min).to_int();
        [
            multiplied[0] as u8,
            multiplied[1] as u8,
            multiplied[2] as u8,
            multiplied[3] as u8,
        ]
    }

    #[inline(always)]
    fn to_norm<S: Simd>(simd: S, value: [Self; 4]) -> f32x4<S> {
        let values: f32x4<S> = i32x4::simd_from(
            simd,
            [
                value[0] as i32,
                value[1] as i32,
                value[2] as i32,
                value[3] as i32,
            ],
        )
        .to_float();
        values * (1.0 / Self::MAX as f32)
    }
}

/// The data format of a given pixel buffer.
pub trait PixelFormat {
    const NUM_COMPONENTS: usize;
    const RGBA_INDICES: (usize, usize, usize, Option<usize>);
}

macro_rules! impl_pix_fmt {
    ($ty: ident, $num_components: expr, $rgba_indices: expr) => {
        pub struct $ty;
        impl PixelFormat for $ty {
            const NUM_COMPONENTS: usize = $num_components;
            const RGBA_INDICES: (usize, usize, usize, Option<usize>) = $rgba_indices;
        }
    };
}

impl_pix_fmt!(Rgbx, 4, (0, 1, 2, Some(3)));
impl_pix_fmt!(Xrgb, 4, (1, 2, 3, Some(0)));
impl_pix_fmt!(Bgrx, 4, (2, 1, 0, Some(3)));
impl_pix_fmt!(Xbgr, 4, (3, 2, 1, Some(0)));
impl_pix_fmt!(Rgb, 3, (0, 1, 2, None));
impl_pix_fmt!(Bgr, 3, (2, 1, 0, None));

pub const fn pixel_bytes_for<S: PixelFormat, T: Normalize>() -> usize {
    S::NUM_COMPONENTS * core::mem::size_of::<T>()
}

/// How to handle writing back fields that we *didn't* process if we used YiqField::Upper or YiqField::Lower.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeinterlaceMode {
    /// Interpolate between the given fields.
    Bob,
    /// Don't write absent fields at all--just leave whatever was already in the buffer.
    Skip,
}

/// Clip rectangle for copying to/from the YIQ buffer. Cannot be negative, and must be in bounds of both the source and
/// destination--you'll need to do some clamping and coordinate-space transforms yourself. Sorry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct Rect {
    pub top: usize,
    pub left: usize,
    pub bottom: usize,
    pub right: usize,
}

impl Rect {
    pub fn new(top: usize, left: usize, bottom: usize, right: usize) -> Self {
        assert!(
            bottom >= top && right >= left,
            "Invalid rectangle (top: {top}, bottom: {bottom}, left: {left}, right: {right})"
        );
        Self {
            top,
            left,
            bottom,
            right,
        }
    }

    pub fn from_width_height(width: usize, height: usize) -> Self {
        Self {
            top: 0,
            left: 0,
            bottom: height,
            right: width,
        }
    }

    pub fn width(&self) -> usize {
        self.right - self.left
    }

    pub fn height(&self) -> usize {
        self.bottom - self.top
    }
}

/// Settings for how to copy the image to and from the YIQ buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlitInfo {
    /// The rectangular area which will be read out of or written into the other buffer.
    pub rect: Rect,
    /// The coordinates at which to place the image in the buffer being written to, whether the YIQ buffer or output
    /// buffer.
    pub destination: (usize, usize),
    /// Number of bytes per pixel row in the other buffer. May include padding.
    pub row_bytes: usize,
    /// Height of the non-YIQ buffer.
    pub other_buffer_height: usize,
    /// True if the source buffer is y-up instead of y-down.
    pub flip_y: bool,
}

impl BlitInfo {
    /// When you don't need to process a specific rectangle of pixels, you can just use this to process the entire
    /// frame at once.
    pub fn from_full_frame(width: usize, height: usize, row_bytes: usize) -> Self {
        BlitInfo {
            rect: Rect::new(0, 0, height, width),
            destination: (0, 0),
            row_bytes,
            other_buffer_height: height,
            flip_y: false,
        }
    }

    pub fn new(
        rect: Rect,
        destination: (usize, usize),
        row_bytes: usize,
        other_buffer_height: usize,
        flip_y: bool,
    ) -> Self {
        Self {
            rect,
            destination,
            row_bytes,
            other_buffer_height,
            flip_y,
        }
    }
}

/// Borrowed YIQ data in a planar format.
/// Each plane is densely packed with regards to rows--if we skip fields, we just leave them out of these planes, which
/// squashes them vertically.
pub struct YiqView<'a> {
    /// Y (luma) plane.
    pub y: &'a mut [f32],
    /// I (in-phase chroma) plane.
    pub i: &'a mut [f32],
    /// Q (quadrature chroma) plane.
    pub q: &'a mut [f32],
    /// Scratch buffer; used to accelerate some operations which would be much slower in-place.
    pub scratch: &'a mut [f32],
    /// Logical dimensions of the image, counting skipped fields. This does *not* depend on the field setting, and will
    /// *not* tell you how many rows of pixels are being stored in the buffers (use the `num_rows` method instead).
    pub dimensions: (usize, usize),
    /// The source field that this data is for.
    pub field: YiqField,
}

fn slice_to_maybe_uninit<T>(slice: &[T]) -> &[MaybeUninit<T>] {
    // Safety: we know these are all initialized, so it's fine to transmute into a type that makes fewer assumptions
    unsafe { core::slice::from_raw_parts(slice.as_ptr() as _, slice.len()) }
}

/// # Safety:
/// - You must only write initialized values into the slice.
unsafe fn slice_to_maybe_uninit_mut<T>(slice: &mut [T]) -> &mut [MaybeUninit<T>] {
    // Safety: we know these are all initialized, so it's fine to transmute into a type that makes fewer assumptions
    unsafe { core::slice::from_raw_parts_mut(slice.as_mut_ptr() as _, slice.len()) }
}

pub trait PixelTransform: Send + Sync + Copy {
    fn transform_pixel<S: Simd>(&self, pixel: f32x4<S>) -> f32x4<S>;
}
impl<T: Fn([f32; 3]) -> [f32; 3] + Send + Sync + Copy> PixelTransform for T {
    #[inline(always)]
    fn transform_pixel<S: Simd>(&self, pixel: f32x4<S>) -> f32x4<S> {
        let simd = pixel.witness();
        let mut tmp: [f32; 4] = pixel.into();
        let transformed = self([tmp[0], tmp[1], tmp[2]]);
        tmp[0] = transformed[0];
        tmp[1] = transformed[1];
        tmp[2] = transformed[2];
        f32x4::simd_from(simd, tmp)
    }
}
impl PixelTransform for () {
    #[inline(always)]
    fn transform_pixel<S: Simd>(&self, pixel: f32x4<S>) -> f32x4<S> {
        pixel
    }
}

impl<'a> YiqView<'a> {
    /// Split this `YiqView` into two `YiqView`s vertically at a given row.
    pub fn split_at_row(&mut self, idx: usize) -> (Option<YiqView<'_>>, Option<YiqView<'_>>) {
        let (y1, y2) = self.y.split_at_mut(idx * self.dimensions.0);
        let (i1, i2) = self.i.split_at_mut(idx * self.dimensions.0);
        let (q1, q2) = self.q.split_at_mut(idx * self.dimensions.0);
        let (s1, s2) = self.scratch.split_at_mut(idx * self.dimensions.0);
        (
            if !y1.is_empty() {
                Some(YiqView {
                    y: y1,
                    i: i1,
                    q: q1,
                    scratch: s1,
                    dimensions: (self.dimensions.0, self.dimensions.1),
                    field: self.field,
                })
            } else {
                None
            },
            if !y2.is_empty() {
                Some(YiqView {
                    y: y2,
                    i: i2,
                    q: q2,
                    scratch: s2,
                    dimensions: (self.dimensions.0, self.dimensions.1),
                    field: self.field,
                })
            } else {
                None
            },
        )
    }

    /// Number of rows of pixels being stored in this view. This will be smaller than `dimensions.1` if some
    /// fields are being skipped.
    #[inline(always)]
    pub fn num_rows(&self) -> usize {
        self.field.num_image_rows(self.dimensions.1)
    }

    /// Convert (a given part of) the input pixel buffer into YIQ planar format, and optionally apply a color transform
    /// to the pixels beforehand. This method allows padding bytes of the source buffer to be uninitialized, which *may*
    /// be the case for effect plugin APIs (OpenFX and After Effects, which both leave it ambiguous).
    ///
    /// # Safety
    /// - `buf` must be a valid pointer to a buffer of length `len`.
    /// - All data within the portions of `buf` within each row, as specified by `row_bytes` and this view's dimensions,
    ///   must be initialized and valid. Data outside of those portions need not be valid.
    pub unsafe fn set_from_strided_buffer_maybe_uninit<
        S: PixelFormat,
        T: Normalize,
        F: PixelTransform,
    >(
        &mut self,
        ctx: &Context,
        buf: &[MaybeUninit<T>],
        blit_info: BlitInfo,
        pixel_transform: F,
    ) {
        let num_components = S::NUM_COMPONENTS;
        assert_eq!(
            blit_info.row_bytes % core::mem::size_of::<T>(),
            0,
            "Rowbytes not aligned to datatype"
        );
        let row_length = blit_info.row_bytes / core::mem::size_of::<T>();
        assert!(num_components >= 3);
        assert!(
            row_length * S::NUM_COMPONENTS >= blit_info.rect.width(),
            "Blit rectangle width exceeds rowbytes"
        );

        assert!(blit_info.rect.width() + blit_info.destination.0 <= self.dimensions.0);
        assert!(blit_info.rect.height() + blit_info.destination.1 <= self.dimensions.1);

        #[inline(always)]
        unsafe fn blit_row_simd_inner<S: Simd, P: PixelFormat, T: Normalize, F: PixelTransform>(
            simd: S,
            y: &mut [f32],
            i: &mut [f32],
            q: &mut [f32],
            buf: &[MaybeUninit<T>],
            blit_info: &BlitInfo,
            src_row_idx: usize,
            pixel_transform: F,
        ) {
            let row_length = blit_info.row_bytes / core::mem::size_of::<T>();
            let src_offset = src_row_idx * row_length;
            let (r_idx, g_idx, b_idx, ..) = P::RGBA_INDICES;
            for idx in 0..blit_info.rect.width() {
                let src_pixel_idx = idx + blit_info.rect.left;
                let rgba = unsafe {
                    T::to_norm(
                        simd,
                        [
                            buf[((src_pixel_idx * P::NUM_COMPONENTS) + src_offset) + r_idx]
                                .assume_init(),
                            buf[((src_pixel_idx * P::NUM_COMPONENTS) + src_offset) + g_idx]
                                .assume_init(),
                            buf[((src_pixel_idx * P::NUM_COMPONENTS) + src_offset) + b_idx]
                                .assume_init(),
                            T::ONE,
                        ],
                    )
                };
                let transformed = pixel_transform.transform_pixel(rgba);
                let yiq_pixel = rgb_to_yiq(transformed);
                let dst_pixel_idx = idx + blit_info.destination.0;
                let yiq_channels: [f32; 4] = yiq_pixel.into();
                y[dst_pixel_idx] = yiq_channels[0];
                i[dst_pixel_idx] = yiq_channels[1];
                q[dst_pixel_idx] = yiq_channels[2];
            }
        }

        unsafe fn blit_row_simd<P: PixelFormat, T: Normalize, F: PixelTransform>(
            level: Level,
            y: &mut [f32],
            i: &mut [f32],
            q: &mut [f32],
            buf: &[MaybeUninit<T>],
            blit_info: &BlitInfo,
            src_row_idx: usize,
            pixel_transform: F,
        ) {
            dispatch!(level, simd => unsafe { blit_row_simd_inner::<_, P, T, F>(simd, y, i, q, buf, blit_info, src_row_idx, pixel_transform) })
        }

        match self.field {
            YiqField::Upper | YiqField::Lower | YiqField::Both => {
                let Self { y, i, q, .. } = self;
                ctx.thread_pool.install(|| {
                    let (width, _) = self.dimensions;
                    let mut field = self.field;

                    let num_skipped_rows = match field {
                        YiqField::Upper => blit_info.destination.1.div_ceil(2),
                        YiqField::Lower => blit_info.destination.1 / 2,
                        _ => blit_info.destination.1,
                    };

                    if blit_info.destination.1 & 1 == 1 {
                        field = field.flip();
                    }

                    let num_rect_rows = field.num_image_rows(blit_info.rect.height());

                    let lines = [y, i, q].map(|plane| {
                        &mut plane
                            [num_skipped_rows * width..(num_skipped_rows + num_rect_rows) * width]
                    });
                    ZipChunks::new(lines, width).par_for_each(|row_idx, [y, i, q]| {
                        // For interleaved fields, we write the first field into the first half of the buffer,
                        // and the second field into the second half.
                        let mut src_row_idx = match field {
                            YiqField::Upper => row_idx * 2,
                            YiqField::Lower => (row_idx * 2) + 1,
                            YiqField::Both => row_idx,
                            _ => unreachable!(),
                        };
                        src_row_idx += blit_info.rect.top;
                        if blit_info.flip_y {
                            src_row_idx = blit_info.other_buffer_height - src_row_idx - 1;
                        }
                        if blit_info.rect.height() == 1 {
                            src_row_idx = 0;
                        }
                        unsafe {
                            blit_row_simd::<S, T, F>(
                                ctx.level,
                                y,
                                i,
                                q,
                                buf,
                                &blit_info,
                                src_row_idx,
                                pixel_transform,
                            );
                        }
                    });
                })
            }
            YiqField::InterleavedUpper => {
                let num_upper_rows = YiqField::Upper.num_actual_image_rows(self.dimensions.1);
                let (mut upper, mut lower) = self.split_at_row(num_upper_rows);
                if let Some(upper) = upper.as_mut() {
                    upper.field = YiqField::Upper;
                    unsafe {
                        upper.set_from_strided_buffer_maybe_uninit::<S, T, F>(
                            ctx,
                            buf,
                            blit_info,
                            pixel_transform,
                        )
                    };
                };

                if let Some(lower) = lower.as_mut() {
                    lower.field = YiqField::Lower;
                    unsafe {
                        lower.set_from_strided_buffer_maybe_uninit::<S, T, F>(
                            ctx,
                            buf,
                            blit_info,
                            pixel_transform,
                        )
                    };
                };
            }
            YiqField::InterleavedLower => {
                let num_lower_rows = YiqField::Lower.num_actual_image_rows(self.dimensions.1);
                let (mut lower, mut upper) = self.split_at_row(num_lower_rows);
                if let Some(upper) = upper.as_mut() {
                    upper.field = YiqField::Upper;
                    unsafe {
                        upper.set_from_strided_buffer_maybe_uninit::<S, T, F>(
                            ctx,
                            buf,
                            blit_info,
                            pixel_transform,
                        )
                    };
                };

                if let Some(lower) = lower.as_mut() {
                    lower.field = YiqField::Lower;
                    unsafe {
                        lower.set_from_strided_buffer_maybe_uninit::<S, T, F>(
                            ctx,
                            buf,
                            blit_info,
                            pixel_transform,
                        )
                    };
                };
            }
        }
    }

    /// Convert (a given part of) the input pixel buffer into YIQ planar format, and optionally apply a color transform
    /// to the pixels beforehand.
    pub fn set_from_strided_buffer<S: PixelFormat, T: Normalize, F: PixelTransform>(
        &mut self,
        ctx: &Context,
        buf: &[T],
        blit_info: BlitInfo,
        pixel_transform: F,
    ) {
        // Safety: We know this data is valid because it's a slice.
        unsafe {
            self.set_from_strided_buffer_maybe_uninit::<S, T, F>(
                ctx,
                slice_to_maybe_uninit(buf),
                blit_info,
                pixel_transform,
            )
        }
    }

    /// Convert (a given part of) the YIQ planar data back into the given pixel fornat, and optionally apply a color
    /// transform to the pixels, before writing it into the destination buffer. This method allows you to write into a
    /// buffer which may not be initialized beforehand.
    pub fn write_to_strided_buffer_maybe_uninit<S: PixelFormat, T: Normalize, F: PixelTransform>(
        &self,
        ctx: &Context,
        dst: &mut [MaybeUninit<T>],
        mut blit_info: BlitInfo,
        deinterlace_mode: DeinterlaceMode,
        pixel_transform: F,
    ) {
        // If we flip the Y coordinate, we need to flip the blit rectangle and destination coords as well. If we were
        // doing "for each source pixel, write to the destination", we could just flip the coordinate of the pixel we
        // write to, but we want to do this in parallel which requires "for each destination pixel, *read* from the
        // source".
        if blit_info.flip_y {
            blit_info.rect.top = blit_info.other_buffer_height - blit_info.rect.top;
            blit_info.rect.bottom = blit_info.other_buffer_height - blit_info.rect.bottom;
            mem::swap(&mut blit_info.rect.bottom, &mut blit_info.rect.top);

            let distance_to_bottom =
                blit_info.other_buffer_height - (blit_info.rect.height() + blit_info.destination.1);
            blit_info.destination.1 = distance_to_bottom;
        }

        assert!(S::NUM_COMPONENTS >= 3);
        assert!(
            blit_info.row_bytes / core::mem::size_of::<T>() * S::NUM_COMPONENTS
                >= blit_info.rect.width(),
            "Blit rectangle width exceeds rowbytes"
        );
        assert_eq!(
            blit_info.row_bytes % core::mem::size_of::<T>(),
            0,
            "Rowbytes not aligned to datatype"
        );
        assert!(blit_info.rect.width() + blit_info.destination.0 <= self.dimensions.0);
        assert!(blit_info.rect.height() + blit_info.destination.1 <= self.dimensions.1);

        #[inline(always)]
        fn write_single_row_simd_inner<S: Simd, P: PixelFormat, T: Normalize, F: PixelTransform>(
            simd: S,
            view: &YiqView,
            blit_info: &BlitInfo,
            deinterlace_mode: DeinterlaceMode,
            mut dst_row_idx: usize,
            dst_row: &mut [MaybeUninit<T>],
            pixel_transform: F,
        ) {
            let (r_idx, g_idx, b_idx, a_idx) = P::RGBA_INDICES;
            let width = view.dimensions.0;
            let output_height = blit_info.other_buffer_height;
            let num_rows = view.num_rows();
            // If the row index modulo 2 equals this number, that row was not rendered in the source data and we need to
            // interpolate between the rows above and beneath it.
            let skip_field: usize = match view.field {
                YiqField::Upper => 1,
                YiqField::Lower => 0,
                // The row index modulo 2 never reaches 2, meaning we don't skip any rows
                YiqField::Both | YiqField::InterleavedUpper | YiqField::InterleavedLower => 2,
            };

            match (deinterlace_mode, view.field) {
                (DeinterlaceMode::Bob, YiqField::Upper | YiqField::Lower) => {
                    // Limit to the actual width of the output (rowbytes may include trailing padding)
                    let dst_row = &mut dst_row[blit_info.destination.0 * P::NUM_COMPONENTS
                        ..(blit_info.destination.0 + blit_info.rect.width()) * P::NUM_COMPONENTS];
                    dst_row_idx += blit_info.rect.top;
                    if blit_info.flip_y {
                        dst_row_idx = output_height - dst_row_idx - 1;
                    }
                    // Inner fields with lines above and below them. Interpolate between those fields
                    if (dst_row_idx & 1) == skip_field
                        && dst_row_idx != 0
                        && dst_row_idx != output_height - 1
                    {
                        for (pix_idx, pixel) in
                            dst_row.chunks_exact_mut(P::NUM_COMPONENTS).enumerate()
                        {
                            let src_idx_lower =
                                ((dst_row_idx - 1) >> 1) * width + pix_idx + blit_info.rect.left;
                            let src_idx_upper =
                                ((dst_row_idx + 1) >> 1) * width + pix_idx + blit_info.rect.left;

                            let upper_pixel = f32x4::simd_from(
                                simd,
                                [
                                    view.y[src_idx_upper],
                                    view.i[src_idx_upper],
                                    view.q[src_idx_upper],
                                    0.0,
                                ],
                            );
                            let lower_pixel = f32x4::simd_from(
                                simd,
                                [
                                    view.y[src_idx_lower],
                                    view.i[src_idx_lower],
                                    view.q[src_idx_lower],
                                    0.0,
                                ],
                            );

                            let interp_pixel = (upper_pixel + lower_pixel) * 0.5;

                            let rgba = T::from_norm(
                                pixel_transform.transform_pixel(yiq_to_rgb(interp_pixel)),
                            );
                            pixel[r_idx] = MaybeUninit::new(rgba[0]);
                            pixel[g_idx] = MaybeUninit::new(rgba[1]);
                            pixel[b_idx] = MaybeUninit::new(rgba[2]);
                            if let Some(a_idx) = a_idx {
                                pixel[a_idx] = MaybeUninit::new(T::ONE);
                            }
                        }
                    } else {
                        // Copy the field directly
                        for (pix_idx, pixel) in
                            dst_row.chunks_exact_mut(P::NUM_COMPONENTS).enumerate()
                        {
                            let src_idx = (dst_row_idx >> 1).min(num_rows - 1) * width
                                + pix_idx
                                + blit_info.rect.left;
                            let rgba = T::from_norm(pixel_transform.transform_pixel(yiq_to_rgb(
                                f32x4::simd_from(
                                    simd,
                                    [view.y[src_idx], view.i[src_idx], view.q[src_idx], 0.0],
                                ),
                            )));
                            pixel[r_idx] = MaybeUninit::new(rgba[0]);
                            pixel[g_idx] = MaybeUninit::new(rgba[1]);
                            pixel[b_idx] = MaybeUninit::new(rgba[2]);
                            if let Some(a_idx) = a_idx {
                                pixel[a_idx] = MaybeUninit::new(T::ONE);
                            }
                        }
                    }
                }
                (DeinterlaceMode::Skip, YiqField::Upper | YiqField::Lower) => {
                    // Limit to the actual width of the output (rowbytes may include trailing padding)
                    let dst_row = &mut dst_row[blit_info.destination.0 * P::NUM_COMPONENTS
                        ..(blit_info.destination.0 + blit_info.rect.width()) * P::NUM_COMPONENTS];
                    dst_row_idx += blit_info.rect.top;
                    if blit_info.flip_y {
                        dst_row_idx = output_height - dst_row_idx - 1;
                    }
                    if (dst_row_idx & 1) == skip_field {
                        return;
                    }
                    for (pix_idx, pixel) in dst_row.chunks_exact_mut(P::NUM_COMPONENTS).enumerate()
                    {
                        let src_idx = (dst_row_idx >> 1).min(num_rows - 1) * width
                            + pix_idx
                            + blit_info.rect.left;
                        let rgba = T::from_norm(pixel_transform.transform_pixel(yiq_to_rgb(
                            f32x4::simd_from(
                                simd,
                                [view.y[src_idx], view.i[src_idx], view.q[src_idx], 0.0],
                            ),
                        )));
                        pixel[r_idx] = MaybeUninit::new(rgba[0]);
                        pixel[g_idx] = MaybeUninit::new(rgba[1]);
                        pixel[b_idx] = MaybeUninit::new(rgba[2]);
                        if let Some(a_idx) = a_idx {
                            pixel[a_idx] = MaybeUninit::new(T::ONE);
                        }
                    }
                }
                (_, YiqField::InterleavedUpper | YiqField::InterleavedLower) => {
                    // Limit to the actual width of the output (rowbytes may include trailing padding)
                    let dst_row = &mut dst_row[blit_info.destination.0 * P::NUM_COMPONENTS
                        ..(blit_info.destination.0 + blit_info.rect.width()) * P::NUM_COMPONENTS];
                    dst_row_idx += blit_info.rect.top;
                    if blit_info.flip_y {
                        dst_row_idx = output_height - dst_row_idx - 1;
                    }
                    let row_offset = match view.field {
                        YiqField::InterleavedUpper => {
                            YiqField::Upper.num_image_rows(view.dimensions.1) * (dst_row_idx & 1)
                        }
                        YiqField::InterleavedLower => {
                            YiqField::Lower.num_image_rows(view.dimensions.1)
                                * (1 - (dst_row_idx & 1))
                        }
                        _ => unreachable!(),
                    };
                    // handle edge case where there's only one row and the mode is InterleavedLower
                    let interleaved_row_idx =
                        ((dst_row_idx >> 1) + row_offset).min(view.dimensions.1 - 1);
                    let src_idx = interleaved_row_idx * width;
                    for (pix_idx, pixel) in dst_row.chunks_exact_mut(P::NUM_COMPONENTS).enumerate()
                    {
                        let rgba = T::from_norm(pixel_transform.transform_pixel(yiq_to_rgb(
                            f32x4::simd_from(
                                simd,
                                [
                                    view.y[src_idx + pix_idx + blit_info.rect.left],
                                    view.i[src_idx + pix_idx + blit_info.rect.left],
                                    view.q[src_idx + pix_idx + blit_info.rect.left],
                                    0.0,
                                ],
                            ),
                        )));
                        pixel[r_idx] = MaybeUninit::new(rgba[0]);
                        pixel[g_idx] = MaybeUninit::new(rgba[1]);
                        pixel[b_idx] = MaybeUninit::new(rgba[2]);
                        if let Some(a_idx) = a_idx {
                            pixel[a_idx] = MaybeUninit::new(T::ONE);
                        }
                    }
                }
                _ => {
                    // Limit to the actual width of the output (rowbytes may include trailing padding)
                    let dst_row = &mut dst_row[blit_info.destination.0 * P::NUM_COMPONENTS
                        ..(blit_info.destination.0 + blit_info.rect.width()) * P::NUM_COMPONENTS];
                    dst_row_idx += blit_info.rect.top;
                    if blit_info.flip_y {
                        dst_row_idx = output_height - dst_row_idx - 1;
                    }
                    for (pix_idx, pixel) in dst_row.chunks_exact_mut(P::NUM_COMPONENTS).enumerate()
                    {
                        let src_idx =
                            dst_row_idx.min(num_rows - 1) * width + pix_idx + blit_info.rect.left;
                        let rgba = T::from_norm(pixel_transform.transform_pixel(yiq_to_rgb(
                            f32x4::simd_from(
                                simd,
                                [view.y[src_idx], view.i[src_idx], view.q[src_idx], 0.0],
                            ),
                        )));
                        pixel[r_idx] = MaybeUninit::new(rgba[0]);
                        pixel[g_idx] = MaybeUninit::new(rgba[1]);
                        pixel[b_idx] = MaybeUninit::new(rgba[2]);
                        if let Some(a_idx) = a_idx {
                            pixel[a_idx] = MaybeUninit::new(T::ONE);
                        }
                    }
                }
            }
        }

        fn write_single_row_simd<P: PixelFormat, T: Normalize, F: PixelTransform>(
            level: Level,
            view: &YiqView,
            blit_info: &BlitInfo,
            deinterlace_mode: DeinterlaceMode,
            dst_row_idx: usize,
            dst_row: &mut [MaybeUninit<T>],
            pixel_transform: F,
        ) {
            dispatch!(level, simd => write_single_row_simd_inner::<_, P, T, F>(simd, view, blit_info, deinterlace_mode, dst_row_idx, dst_row, pixel_transform))
        }

        ctx.thread_pool.install(|| {
            let row_length = blit_info.row_bytes / core::mem::size_of::<T>();

            let skip_rows = blit_info.destination.1;
            let take_rows = blit_info.rect.height();
            let chunks = ZipChunks::new(
                [&mut dst[skip_rows * row_length..(skip_rows + take_rows) * row_length]],
                row_length,
            );

            chunks.par_for_each(|dst_row_idx, [dst_row]| {
                write_single_row_simd::<S, T, F>(
                    ctx.level,
                    self,
                    &blit_info,
                    deinterlace_mode,
                    dst_row_idx,
                    dst_row,
                    pixel_transform,
                );
            });
        });
    }

    pub fn write_to_strided_buffer<S: PixelFormat, T: Normalize, F: PixelTransform>(
        &self,
        ctx: &Context,
        dst: &mut [T],
        blit_info: BlitInfo,
        deinterlace_mode: DeinterlaceMode,
        pixel_transform: F,
    ) {
        self.write_to_strided_buffer_maybe_uninit::<S, T, F>(
            ctx,
            unsafe { slice_to_maybe_uninit_mut(dst) },
            blit_info,
            deinterlace_mode,
            pixel_transform,
        )
    }

    pub fn from_parts(buf: &'a mut [f32], dimensions: (usize, usize), field: YiqField) -> Self {
        let num_pixels = dimensions.0 * field.num_image_rows(dimensions.1);
        assert_eq!(
            buf.len(),
            num_pixels * 4,
            "buffer length: {}, expected buffer length: {}",
            buf.len(),
            num_pixels * 4
        );
        let (y, iqs) = buf.split_at_mut(num_pixels);
        let (i, qs) = iqs.split_at_mut(num_pixels);
        let (q, s) = qs.split_at_mut(num_pixels);
        YiqView {
            y,
            i,
            q,
            scratch: s,
            dimensions,
            field,
        }
    }

    /// Calculate the length (in elements, not bytes) of a buffer needed to hold a YiqView with the given dimensions and
    /// field.
    pub fn buf_length_for(dimensions: (usize, usize), field: YiqField) -> usize {
        dimensions.0 * field.num_image_rows(dimensions.1) * 4
    }

    /// Calculate the maximum length (in elements, not bytes) of a buffer needed to hold a YiqView with the given
    /// dimensions and `use_field` effect setting. The actual length may vary depending on the frame number if
    /// `use_field` is set to `UseField::Alternating`, and this returns an upper bound.
    pub fn max_buf_length_for(dimensions: (usize, usize), field: UseField) -> usize {
        let num_rows = match field {
            UseField::Alternating => YiqField::Upper
                .num_image_rows(dimensions.1)
                .max(YiqField::Lower.num_image_rows(dimensions.1)),
            _ => field.to_yiq_field(0).num_image_rows(dimensions.1),
        };
        dimensions.0 * num_rows * 4
    }
}

/// Owned YIQ data. If you bring your own buffer, you probably don't need this.
pub struct YiqOwned {
    /// Densely-packed planar YUV data. The Y plane comes first in memory, then I, then Q.
    data: Box<[f32]>,
    /// This refers to the "logical" dimensions, meaning that the number of scanlines is the same no matter whether any
    /// fields are being skipped.
    dimensions: (usize, usize),
    /// The source field that this data is for.
    field: YiqField,
}

impl YiqOwned {
    pub fn from_strided_buffer<S: PixelFormat, T: Normalize>(
        ctx: &Context,
        buf: &[T],
        row_bytes: usize,
        width: usize,
        height: usize,
        field: YiqField,
    ) -> Self {
        let mut data = vec![0f32; YiqView::buf_length_for((width, height), field)];
        let mut view = YiqView::from_parts(&mut data, (width, height), field);

        view.set_from_strided_buffer::<S, T, _>(
            ctx,
            buf,
            BlitInfo::from_full_frame(width, height, row_bytes),
            (),
        );

        YiqOwned {
            data: data.into_boxed_slice(),
            dimensions: (width, height),
            field,
        }
    }
}

impl<'a> From<&'a mut YiqOwned> for YiqView<'a> {
    fn from(value: &'a mut YiqOwned) -> Self {
        YiqView::from_parts(&mut value.data, value.dimensions, value.field)
    }
}

/// These tests exercise the public API surface only, with one exception: `AfterEffectsU16` has no
/// public constructor or accessor, so the tests for it construct values via its private field.
///
/// Most tests use grayscale pixels because gray round-trips through the YIQ conversion (almost)
/// exactly: for r = g = b = v, Y = v and I = Q = 0, since the matrix coefficients in each of the
/// I/Q rows sum to zero. This makes the row-indexing behavior directly observable in the Y plane
/// without worrying about color conversion error.
#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    fn ctx() -> &'static Context {
        crate::ctx::global()
    }

    /// Marker for "this value was never written".
    const SENT: f32 = -999.0;
    /// Width used for the row-mapping tests. Column handling is covered by the sub-rect tests.
    const W: usize = 4;
    const RGBX_ROW_BYTES: usize = W * 4 * size_of::<f32>();

    #[track_caller]
    fn assert_rows_eq(actual: &[f32], expected: &[f32], context: &str) {
        assert_eq!(
            actual.len(),
            expected.len(),
            "{context}: row count mismatch: got {actual:?}, expected {expected:?}"
        );
        for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (a - e).abs() <= 0.01,
                "{context}: row {i}: got {a}, expected {e} (full: got {actual:?}, expected {expected:?})"
            );
        }
    }

    /// Rgbx f32 image where every pixel in row `r` is gray with value `values[r]`. The alpha
    /// channel is junk, to verify that input alpha is ignored.
    fn gray_image(width: usize, values: &[f32]) -> Vec<f32> {
        let mut img = Vec::with_capacity(width * values.len() * 4);
        for &v in values {
            for _ in 0..width {
                img.extend_from_slice(&[v, v, v, 7.0]);
            }
        }
        img
    }

    /// Gray value of each row of the frame, 10, 20, 30, ...
    fn row_values(height: usize) -> Vec<f32> {
        (0..height).map(|r| 10.0 * (r + 1) as f32).collect()
    }

    /// Collapse a plane into one value per row, asserting that each row is uniform.
    #[track_caller]
    fn plane_rows(plane: &[f32], width: usize) -> Vec<f32> {
        plane
            .chunks_exact(width)
            .map(|row| {
                for v in row {
                    assert!((v - row[0]).abs() <= 0.01, "plane row not uniform: {row:?}");
                }
                row[0]
            })
            .collect()
    }

    /// Collapse an Rgbx f32 output buffer into one gray value per row. Rows that were never
    /// written collapse to `SENT`; written rows must be uniform gray with alpha set to 1.
    #[track_caller]
    fn out_rows(buf: &[f32], width: usize) -> Vec<f32> {
        buf.chunks_exact(width * 4)
            .map(|row| {
                if row[0] == SENT {
                    assert!(
                        row.iter().all(|&v| v == SENT),
                        "partially-written row: {row:?}"
                    );
                    return SENT;
                }
                for px in row.chunks_exact(4) {
                    for c in 0..3 {
                        assert!(
                            (px[c] - row[0]).abs() <= 0.01,
                            "output row not uniform gray: {row:?}"
                        );
                    }
                    assert_eq!(px[3], 1.0, "alpha not set to ONE: {row:?}");
                }
                row[0]
            })
            .collect()
    }

    /// Convert a full-frame gray-rows image (row r = 10 * (r + 1)) into a YIQ buffer with the
    /// given field, returning the per-row Y values of the buffer.
    fn set_rows(height: usize, field: YiqField) -> Vec<f32> {
        let img = gray_image(W, &row_values(height));
        let mut buf = vec![SENT; YiqView::buf_length_for((W, height), field)];
        let mut view = YiqView::from_parts(&mut buf, (W, height), field);
        view.set_from_strided_buffer::<Rgbx, f32, _>(
            ctx(),
            &img,
            BlitInfo::from_full_frame(W, height, RGBX_ROW_BYTES),
            (),
        );
        plane_rows(view.y, W)
    }

    /// Fill a YIQ buffer's rows with the given gray values, write it back out to an Rgbx f32
    /// frame prefilled with sentinels, and return the per-row output values.
    fn write_rows(
        height: usize,
        field: YiqField,
        mode: DeinterlaceMode,
        buf_rows: &[f32],
    ) -> Vec<f32> {
        let mut buf = vec![0.0; YiqView::buf_length_for((W, height), field)];
        let view = YiqView::from_parts(&mut buf, (W, height), field);
        assert_eq!(
            view.num_rows(),
            buf_rows.len(),
            "test bug: wrong number of buffer rows"
        );
        for (r, &v) in buf_rows.iter().enumerate() {
            view.y[r * W..(r + 1) * W].fill(v);
        }
        let mut out = vec![SENT; W * height * 4];
        view.write_to_strided_buffer::<Rgbx, f32, _>(
            ctx(),
            &mut out,
            BlitInfo::from_full_frame(W, height, RGBX_ROW_BYTES),
            mode,
            (),
        );
        out_rows(&out, W)
    }

    /// Full set -> write round trip on a gray-rows image; returns per-row output values.
    fn set_then_write(height: usize, field: YiqField, mode: DeinterlaceMode) -> Vec<f32> {
        let img = gray_image(W, &row_values(height));
        let mut buf = vec![0.0; YiqView::buf_length_for((W, height), field)];
        let mut view = YiqView::from_parts(&mut buf, (W, height), field);
        let blit = BlitInfo::from_full_frame(W, height, RGBX_ROW_BYTES);
        view.set_from_strided_buffer::<Rgbx, f32, _>(ctx(), &img, blit, ());
        let mut out = vec![SENT; W * height * 4];
        view.write_to_strided_buffer::<Rgbx, f32, _>(ctx(), &mut out, blit, mode, ());
        out_rows(&out, W)
    }

    /// Deterministic pseudo-random color for pixel index `i`.
    fn test_color(i: usize) -> [f32; 3] {
        let h = (i as u32).wrapping_mul(2654435761);
        [
            (h >> 8 & 0xFF) as f32 / 255.0,
            (h >> 16 & 0xFF) as f32 / 255.0,
            (h >> 24 & 0xFF) as f32 / 255.0,
        ]
    }

    // ---- YiqField ----

    #[test]
    fn num_image_rows() {
        use YiqField::*;
        // (field, expected rows for heights 1..=5)
        let cases = [
            (Upper, [1, 1, 2, 2, 3]),
            (Lower, [1, 1, 1, 2, 2]),
            (Both, [1, 2, 3, 4, 5]),
            (InterleavedUpper, [1, 2, 3, 4, 5]),
            (InterleavedLower, [1, 2, 3, 4, 5]),
        ];
        for (field, expected) in cases {
            for (h, e) in expected.iter().enumerate() {
                assert_eq!(
                    field.num_image_rows(h + 1),
                    *e,
                    "{field:?}, height {}",
                    h + 1
                );
            }
        }
    }

    #[test]
    fn num_actual_image_rows() {
        use YiqField::*;
        // Differs from num_image_rows only in that Lower can return 0.
        assert_eq!(Lower.num_actual_image_rows(1), 0);
        assert_eq!(Lower.num_actual_image_rows(2), 1);
        assert_eq!(Lower.num_actual_image_rows(3), 1);
        assert_eq!(Upper.num_actual_image_rows(1), 1);
        assert_eq!(Upper.num_actual_image_rows(5), 3);
        assert_eq!(Both.num_actual_image_rows(5), 5);
        assert_eq!(InterleavedUpper.num_actual_image_rows(5), 5);
        assert_eq!(InterleavedLower.num_actual_image_rows(5), 5);
    }

    #[test]
    fn field_flip() {
        use YiqField::*;
        assert_eq!(Upper.flip(), Lower);
        assert_eq!(Lower.flip(), Upper);
        assert_eq!(Both.flip(), Both);
        assert_eq!(InterleavedUpper.flip(), InterleavedLower);
        assert_eq!(InterleavedLower.flip(), InterleavedUpper);
        for field in [Upper, Lower, Both, InterleavedUpper, InterleavedLower] {
            assert_eq!(field.flip().flip(), field);
        }
    }

    // ---- Rect / buffer sizing ----

    #[test]
    fn rect_dimensions() {
        let rect = Rect::new(1, 2, 5, 7);
        assert_eq!(rect.width(), 5);
        assert_eq!(rect.height(), 4);
        let full = Rect::from_width_height(6, 3);
        assert_eq!((full.left, full.top, full.right, full.bottom), (0, 0, 6, 3));
    }

    #[test]
    #[should_panic]
    fn rect_invalid_vertical() {
        let _ = Rect::new(3, 0, 2, 5);
    }

    #[test]
    #[should_panic]
    fn rect_invalid_horizontal() {
        let _ = Rect::new(0, 5, 3, 2);
    }

    #[test]
    fn buf_length() {
        use YiqField::*;
        for (field, h, rows) in [(Upper, 5, 3), (Lower, 5, 2), (Both, 5, 5), (Lower, 1, 1)] {
            assert_eq!(YiqView::buf_length_for((4, h), field), 4 * rows * 4);
        }
        // Alternating must be sized for whichever per-frame field is larger.
        assert_eq!(
            YiqView::max_buf_length_for((4, 5), UseField::Alternating),
            4 * 3 * 4
        );
        assert_eq!(
            YiqView::max_buf_length_for((4, 5), UseField::Upper),
            4 * 3 * 4
        );
        assert_eq!(
            YiqView::max_buf_length_for((4, 5), UseField::Lower),
            4 * 2 * 4
        );
        assert_eq!(
            YiqView::max_buf_length_for((4, 5), UseField::Both),
            4 * 5 * 4
        );
    }

    #[test]
    fn from_parts_planes() {
        let mut buf = vec![0.0; YiqView::buf_length_for((4, 5), YiqField::Upper)];
        let view = YiqView::from_parts(&mut buf, (4, 5), YiqField::Upper);
        assert_eq!(view.num_rows(), 3);
        assert_eq!(view.y.len(), 4 * 3);
        assert_eq!(view.i.len(), 4 * 3);
        assert_eq!(view.q.len(), 4 * 3);
        assert_eq!(view.scratch.len(), 4 * 3);
        assert_eq!(view.dimensions, (4, 5));
    }

    #[test]
    #[should_panic]
    fn from_parts_wrong_length() {
        let mut buf = vec![0.0; 123];
        let _ = YiqView::from_parts(&mut buf, (4, 5), YiqField::Both);
    }

    #[test]
    fn split_at_row_parts() {
        let mut buf = vec![0.0; YiqView::buf_length_for((4, 6), YiqField::Both)];
        let mut view = YiqView::from_parts(&mut buf, (4, 6), YiqField::Both);

        let (first, second) = view.split_at_row(2);
        let (first, second) = (first.unwrap(), second.unwrap());
        assert_eq!(first.y.len(), 4 * 2);
        assert_eq!(second.y.len(), 4 * 4);
        assert_eq!(first.dimensions, (4, 6));
        assert_eq!(second.dimensions, (4, 6));
        assert_eq!(first.field, YiqField::Both);

        let (first, second) = view.split_at_row(0);
        assert!(first.is_none());
        assert!(second.is_some());

        let (first, second) = view.split_at_row(6);
        assert!(first.is_some());
        assert!(second.is_none());
    }

    // ---- set: which source rows end up in which buffer rows ----

    #[test]
    fn set_field_row_selection() {
        use YiqField::*;
        // Source row r has value 10 * (r + 1).
        let cases: &[(YiqField, usize, &[f32])] = &[
            (Upper, 1, &[10.]),
            (Upper, 2, &[10.]),
            (Upper, 3, &[10., 30.]),
            (Upper, 4, &[10., 30.]),
            (Upper, 5, &[10., 30., 50.]),
            // A 1-row image stores row 0 even in Lower mode.
            (Lower, 1, &[10.]),
            (Lower, 2, &[20.]),
            (Lower, 3, &[20.]),
            (Lower, 4, &[20., 40.]),
            (Lower, 5, &[20., 40.]),
            (Both, 1, &[10.]),
            (Both, 3, &[10., 20., 30.]),
            (Both, 5, &[10., 20., 30., 40., 50.]),
            // Interleaved: one field's rows in the first half of the buffer, the other field's
            // rows in the second half.
            (InterleavedUpper, 1, &[10.]),
            (InterleavedUpper, 2, &[10., 20.]),
            (InterleavedUpper, 3, &[10., 30., 20.]),
            (InterleavedUpper, 4, &[10., 30., 20., 40.]),
            (InterleavedUpper, 5, &[10., 30., 50., 20., 40.]),
            (InterleavedLower, 1, &[10.]),
            (InterleavedLower, 2, &[20., 10.]),
            (InterleavedLower, 3, &[20., 10., 30.]),
            (InterleavedLower, 4, &[20., 40., 10., 30.]),
            (InterleavedLower, 5, &[20., 40., 10., 30., 50.]),
        ];
        for (field, height, expected) in cases {
            assert_rows_eq(
                &set_rows(*height, *field),
                expected,
                &alloc::format!("{field:?}, height {height}"),
            );
        }
    }

    // ---- write: deinterlacing ----

    #[test]
    fn write_bob() {
        use YiqField::*;
        // Buffer rows have values 10, 20, 30, ...; expected output is per *frame* row.
        let cases: &[(YiqField, usize, &[f32], &[f32])] = &[
            (Upper, 1, &[10.], &[10.]),
            (Upper, 2, &[10.], &[10., 10.]),
            (Upper, 3, &[10., 20.], &[10., 15., 20.]),
            (Upper, 4, &[10., 20.], &[10., 15., 20., 20.]),
            (Upper, 5, &[10., 20., 30.], &[10., 15., 20., 25., 30.]),
            (Lower, 1, &[10.], &[10.]),
            (Lower, 2, &[10.], &[10., 10.]),
            (Lower, 3, &[10.], &[10., 10., 10.]),
            (Lower, 4, &[10., 20.], &[10., 10., 15., 20.]),
            (Lower, 5, &[10., 20.], &[10., 10., 15., 20., 20.]),
        ];
        for (field, height, buf_rows, expected) in cases {
            assert_rows_eq(
                &write_rows(*height, *field, DeinterlaceMode::Bob, buf_rows),
                expected,
                &alloc::format!("{field:?}, height {height}"),
            );
        }
    }

    #[test]
    fn write_skip() {
        use YiqField::*;
        let cases: &[(YiqField, usize, &[f32], &[f32])] = &[
            (Upper, 1, &[10.], &[10.]),
            (Upper, 2, &[10.], &[10., SENT]),
            (Upper, 3, &[10., 20.], &[10., SENT, 20.]),
            (Upper, 4, &[10., 20.], &[10., SENT, 20., SENT]),
            (Upper, 5, &[10., 20., 30.], &[10., SENT, 20., SENT, 30.]),
            // For a 1-row image in Lower mode, the buffer's one row (which *was* filled by set)
            // is never written back out in Skip mode.
            (Lower, 1, &[10.], &[SENT]),
            (Lower, 2, &[10.], &[SENT, 10.]),
            (Lower, 3, &[10.], &[SENT, 10., SENT]),
            (Lower, 4, &[10., 20.], &[SENT, 10., SENT, 20.]),
            (Lower, 5, &[10., 20.], &[SENT, 10., SENT, 20., SENT]),
        ];
        for (field, height, buf_rows, expected) in cases {
            assert_rows_eq(
                &write_rows(*height, *field, DeinterlaceMode::Skip, buf_rows),
                expected,
                &alloc::format!("{field:?}, height {height}"),
            );
        }
    }

    #[test]
    fn write_interleaved() {
        use YiqField::*;
        // The deinterlace mode is irrelevant for interleaved fields; both modes must produce
        // identical output.
        let cases: &[(YiqField, usize, &[f32], &[f32])] = &[
            (InterleavedUpper, 1, &[10.], &[10.]),
            (InterleavedUpper, 2, &[10., 20.], &[10., 20.]),
            (InterleavedUpper, 3, &[10., 20., 30.], &[10., 30., 20.]),
            (
                InterleavedUpper,
                4,
                &[10., 20., 30., 40.],
                &[10., 30., 20., 40.],
            ),
            (
                InterleavedUpper,
                5,
                &[10., 20., 30., 40., 50.],
                &[10., 40., 20., 50., 30.],
            ),
            (InterleavedLower, 1, &[10.], &[10.]),
            (InterleavedLower, 2, &[10., 20.], &[20., 10.]),
            (InterleavedLower, 3, &[10., 20., 30.], &[20., 10., 30.]),
            (
                InterleavedLower,
                4,
                &[10., 20., 30., 40.],
                &[30., 10., 40., 20.],
            ),
            (
                InterleavedLower,
                5,
                &[10., 20., 30., 40., 50.],
                &[30., 10., 40., 20., 50.],
            ),
        ];
        for mode in [DeinterlaceMode::Bob, DeinterlaceMode::Skip] {
            for (field, height, buf_rows, expected) in cases {
                assert_rows_eq(
                    &write_rows(*height, *field, mode, buf_rows),
                    expected,
                    &alloc::format!("{field:?}, height {height}, {mode:?}"),
                );
            }
        }
    }

    #[test]
    fn write_both_identity() {
        for mode in [DeinterlaceMode::Bob, DeinterlaceMode::Skip] {
            for height in 1..=5 {
                let buf_rows = row_values(height);
                assert_rows_eq(
                    &write_rows(height, YiqField::Both, mode, &buf_rows),
                    &buf_rows,
                    &alloc::format!("height {height}, {mode:?}"),
                );
            }
        }
    }

    #[test]
    fn round_trip_identity_fields() {
        use YiqField::*;
        // set followed by write is the identity for any field mode that stores every row.
        for field in [Both, InterleavedUpper, InterleavedLower] {
            for height in 1..=5 {
                assert_rows_eq(
                    &set_then_write(height, field, DeinterlaceMode::Bob),
                    &row_values(height),
                    &alloc::format!("{field:?}, height {height}"),
                );
            }
        }
    }

    // ---- sub-rect blits ----

    #[test]
    fn set_sub_rect() {
        // Per-pixel values so both row and column mapping are observable:
        // source pixel (r, c) is gray with value 100 * (r + 1) + c.
        let (src_w, src_h) = (6usize, 6usize);
        let mut img = vec![0.0f32; src_w * src_h * 4];
        for r in 0..src_h {
            for c in 0..src_w {
                let v = 100.0 * (r + 1) as f32 + c as f32;
                img[(r * src_w + c) * 4..][..3].fill(v);
                img[(r * src_w + c) * 4 + 3] = 7.0;
            }
        }

        // Copy source rows 1..5, columns 2..5 to destination (1, 2) in a 5x7 buffer.
        let mut buf = vec![SENT; YiqView::buf_length_for((5, 7), YiqField::Both)];
        let mut view = YiqView::from_parts(&mut buf, (5, 7), YiqField::Both);
        let blit = BlitInfo::new(
            Rect::new(1, 2, 5, 5),
            (1, 2),
            src_w * 4 * size_of::<f32>(),
            src_h,
            false,
        );
        view.set_from_strided_buffer::<Rgbx, f32, _>(ctx(), &img, blit, ());

        for row in 0..7 {
            for col in 0..5 {
                let got = view.y[row * 5 + col];
                if (2..6).contains(&row) && (1..4).contains(&col) {
                    // Buffer (row, col) <- source (row - dest.1 + rect.top, col - dest.0 + rect.left),
                    // i.e. source (row - 1, col + 1), which holds 100 * (row - 1 + 1) + (col + 1).
                    let expected = 100.0 * row as f32 + (col + 1) as f32;
                    assert!(
                        (got - expected).abs() <= 0.01,
                        "buffer ({row}, {col}): got {got}, expected {expected}"
                    );
                } else {
                    assert_eq!(
                        got, SENT,
                        "buffer ({row}, {col}) written outside blit region"
                    );
                }
            }
        }
    }

    #[test]
    fn set_sub_rect_field_parity() {
        // An Upper-field view of a 6-row frame stores frame rows 0, 2, 4 in its 3 buffer rows.
        // Blit 2 source rows (2..4) to destination row 2: frame row 2 is stored (buffer row 1)
        // and receives the rect's row 0, i.e. source row 2 (value 30).
        let img = gray_image(W, &row_values(6));
        let blit_at = |dest_y: usize| {
            BlitInfo::new(Rect::new(2, 0, 4, W), (0, dest_y), RGBX_ROW_BYTES, 6, false)
        };

        let mut buf = vec![SENT; YiqView::buf_length_for((W, 6), YiqField::Upper)];
        let mut view = YiqView::from_parts(&mut buf, (W, 6), YiqField::Upper);
        view.set_from_strided_buffer::<Rgbx, f32, _>(ctx(), &img, blit_at(2), ());
        assert_rows_eq(
            &plane_rows(view.y, W),
            &[SENT, 30., SENT],
            "even destination",
        );

        // Same blit at destination row 1 (odd): frame row 2 is now the rect's row 1, i.e.
        // source row 3 (value 40). This exercises the field-parity flip for odd destinations.
        let mut buf = vec![SENT; YiqView::buf_length_for((W, 6), YiqField::Upper)];
        let mut view = YiqView::from_parts(&mut buf, (W, 6), YiqField::Upper);
        view.set_from_strided_buffer::<Rgbx, f32, _>(ctx(), &img, blit_at(1), ());
        assert_rows_eq(
            &plane_rows(view.y, W),
            &[SENT, 40., SENT],
            "odd destination",
        );
    }

    #[test]
    fn write_sub_rect() {
        // YIQ pixel (r, c) is gray with value 100 * (r + 1) + c.
        let mut buf = vec![0.0; YiqView::buf_length_for((5, 6), YiqField::Both)];
        let view = YiqView::from_parts(&mut buf, (5, 6), YiqField::Both);
        for r in 0..6 {
            for c in 0..5 {
                view.y[r * 5 + c] = 100.0 * (r + 1) as f32 + c as f32;
            }
        }

        // Write YIQ rows 1..4, columns 1..4 to destination (1, 2) in a 5x6 output frame.
        let mut out = vec![SENT; 5 * 6 * 4];
        let blit = BlitInfo::new(
            Rect::new(1, 1, 4, 4),
            (1, 2),
            5 * 4 * size_of::<f32>(),
            6,
            false,
        );
        view.write_to_strided_buffer::<Rgbx, f32, _>(
            ctx(),
            &mut out,
            blit,
            DeinterlaceMode::Bob,
            (),
        );

        for row in 0..6 {
            for col in 0..5 {
                let px = &out[(row * 5 + col) * 4..][..4];
                if (2..5).contains(&row) && (1..4).contains(&col) {
                    // Output (row, col) <- YIQ (row - dest.1 + rect.top, col - dest.0 + rect.left)
                    let expected = 100.0 * row as f32 + col as f32;
                    for c in 0..3 {
                        assert!(
                            (px[c] - expected).abs() <= 0.01,
                            "output ({row}, {col}): got {px:?}, expected {expected}"
                        );
                    }
                    assert_eq!(px[3], 1.0);
                } else {
                    assert!(
                        px.iter().all(|&v| v == SENT),
                        "output ({row}, {col}) written outside blit region: {px:?}"
                    );
                }
            }
        }
    }

    // ---- flip_y ----

    #[test]
    fn set_flip_y() {
        use YiqField::*;
        // A y-up source image: buffer rows should sample the source bottom-up.
        let cases: &[(YiqField, &[f32])] = &[
            (Both, &[50., 40., 30., 20., 10.]),
            (Upper, &[50., 30., 10.]),
            (Lower, &[40., 20.]),
        ];
        for (field, expected) in cases {
            let img = gray_image(W, &row_values(5));
            let mut buf = vec![SENT; YiqView::buf_length_for((W, 5), *field)];
            let mut view = YiqView::from_parts(&mut buf, (W, 5), *field);
            let blit = BlitInfo::new(
                Rect::from_width_height(W, 5),
                (0, 0),
                RGBX_ROW_BYTES,
                5,
                true,
            );
            view.set_from_strided_buffer::<Rgbx, f32, _>(ctx(), &img, blit, ());
            assert_rows_eq(
                &plane_rows(view.y, W),
                expected,
                &alloc::format!("{field:?}"),
            );
        }
    }

    #[test]
    fn write_flip_y_full_frame() {
        let mut buf = vec![0.0; YiqView::buf_length_for((W, 5), YiqField::Both)];
        let view = YiqView::from_parts(&mut buf, (W, 5), YiqField::Both);
        for (r, &v) in row_values(5).iter().enumerate() {
            view.y[r * W..(r + 1) * W].fill(v);
        }
        let mut out = vec![SENT; W * 5 * 4];
        let blit = BlitInfo::new(
            Rect::from_width_height(W, 5),
            (0, 0),
            RGBX_ROW_BYTES,
            5,
            true,
        );
        view.write_to_strided_buffer::<Rgbx, f32, _>(
            ctx(),
            &mut out,
            blit,
            DeinterlaceMode::Bob,
            (),
        );
        assert_rows_eq(
            &out_rows(&out, W),
            &[50., 40., 30., 20., 10.],
            "write flip_y",
        );
    }

    #[test]
    fn write_flip_y_sub_rect() {
        // YIQ rows 1..4 (values 20, 30, 40) written to destination row 2 of a 6-row y-up output:
        // in y-down terms the data lands at rows 1..4, vertically mirrored.
        let mut buf = vec![0.0; YiqView::buf_length_for((W, 6), YiqField::Both)];
        let view = YiqView::from_parts(&mut buf, (W, 6), YiqField::Both);
        for (r, &v) in row_values(6).iter().enumerate() {
            view.y[r * W..(r + 1) * W].fill(v);
        }
        let mut out = vec![SENT; W * 6 * 4];
        let blit = BlitInfo::new(Rect::new(1, 0, 4, W), (0, 2), RGBX_ROW_BYTES, 6, true);
        view.write_to_strided_buffer::<Rgbx, f32, _>(
            ctx(),
            &mut out,
            blit,
            DeinterlaceMode::Bob,
            (),
        );
        assert_rows_eq(
            &out_rows(&out, W),
            &[SENT, 40., 30., 20., SENT, SENT],
            "write flip_y sub-rect",
        );
    }

    #[test]
    fn flip_y_round_trip() {
        // Setting and writing with flip_y on both sides is the identity.
        let img = gray_image(W, &row_values(4));
        let mut buf = vec![0.0; YiqView::buf_length_for((W, 4), YiqField::Both)];
        let mut view = YiqView::from_parts(&mut buf, (W, 4), YiqField::Both);
        let blit = BlitInfo::new(
            Rect::from_width_height(W, 4),
            (0, 0),
            RGBX_ROW_BYTES,
            4,
            true,
        );
        view.set_from_strided_buffer::<Rgbx, f32, _>(ctx(), &img, blit, ());
        let mut out = vec![SENT; W * 4 * 4];
        view.write_to_strided_buffer::<Rgbx, f32, _>(
            ctx(),
            &mut out,
            blit,
            DeinterlaceMode::Bob,
            (),
        );
        assert_rows_eq(&out_rows(&out, W), &row_values(4), "flip_y round trip");
    }

    // ---- row_bytes padding ----

    #[test]
    fn rowbytes_padding() {
        // Rows padded with 3 extra f32 elements: padding must be ignored on read and
        // preserved on write.
        const PAD: usize = 3;
        let row_len = W * 4 + PAD;
        let row_bytes = row_len * size_of::<f32>();
        let height = 4;

        let mut img = vec![SENT; row_len * height];
        for (r, &v) in row_values(height).iter().enumerate() {
            for c in 0..W {
                img[r * row_len + c * 4..][..3].fill(v);
                img[r * row_len + c * 4 + 3] = 7.0;
            }
        }

        let mut buf = vec![0.0; YiqView::buf_length_for((W, height), YiqField::Both)];
        let mut view = YiqView::from_parts(&mut buf, (W, height), YiqField::Both);
        let blit = BlitInfo::from_full_frame(W, height, row_bytes);
        view.set_from_strided_buffer::<Rgbx, f32, _>(ctx(), &img, blit, ());
        assert_rows_eq(
            &plane_rows(view.y, W),
            &row_values(height),
            "set with padding",
        );

        let mut out = vec![SENT; row_len * height];
        view.write_to_strided_buffer::<Rgbx, f32, _>(
            ctx(),
            &mut out,
            blit,
            DeinterlaceMode::Bob,
            (),
        );
        for (r, &v) in row_values(height).iter().enumerate() {
            let row = &out[r * row_len..(r + 1) * row_len];
            for px in row[..W * 4].chunks_exact(4) {
                for c in 0..3 {
                    assert!(
                        (px[c] - v).abs() <= 0.01,
                        "row {r}: got {px:?}, expected {v}"
                    );
                }
                assert_eq!(px[3], 1.0);
            }
            assert!(
                row[W * 4..].iter().all(|&v| v == SENT),
                "row {r}: padding was overwritten: {:?}",
                &row[W * 4..]
            );
        }
    }

    // ---- MaybeUninit variants ----

    #[test]
    fn set_from_uninit_padding() {
        // Genuinely uninitialized padding, as the OpenFX/AE APIs may provide. (Run under Miri to
        // additionally verify the padding is never read.)
        const PAD: usize = 3;
        let row_len = W * 4 + PAD;
        let height = 4;

        let mut img: Vec<MaybeUninit<f32>> = Vec::new();
        img.resize_with(row_len * height, MaybeUninit::uninit);
        for (r, &v) in row_values(height).iter().enumerate() {
            for c in 0..W {
                for ch in 0..4 {
                    img[r * row_len + c * 4 + ch] = MaybeUninit::new(if ch == 3 { 7.0 } else { v });
                }
            }
        }

        let mut buf = vec![0.0; YiqView::buf_length_for((W, height), YiqField::Both)];
        let mut view = YiqView::from_parts(&mut buf, (W, height), YiqField::Both);
        unsafe {
            view.set_from_strided_buffer_maybe_uninit::<Rgbx, f32, _>(
                ctx(),
                &img,
                BlitInfo::from_full_frame(W, height, row_len * size_of::<f32>()),
                (),
            );
        }
        assert_rows_eq(
            &plane_rows(view.y, W),
            &row_values(height),
            "set from uninit",
        );
    }

    #[test]
    fn write_to_uninit() {
        // Writing every row (Both) must initialize the entire buffer; writing with Skip must
        // initialize exactly the rows belonging to the stored field.
        let height = 4;

        let mut buf = vec![0.0; YiqView::buf_length_for((W, height), YiqField::Both)];
        let view = YiqView::from_parts(&mut buf, (W, height), YiqField::Both);
        for (r, &v) in row_values(height).iter().enumerate() {
            view.y[r * W..(r + 1) * W].fill(v);
        }
        let mut out: Vec<MaybeUninit<f32>> = Vec::new();
        out.resize_with(W * height * 4, MaybeUninit::uninit);
        view.write_to_strided_buffer_maybe_uninit::<Rgbx, f32, _>(
            ctx(),
            &mut out,
            BlitInfo::from_full_frame(W, height, RGBX_ROW_BYTES),
            DeinterlaceMode::Bob,
            (),
        );
        let out: Vec<f32> = out.iter().map(|v| unsafe { v.assume_init() }).collect();
        assert_rows_eq(
            &out_rows(&out, W),
            &row_values(height),
            "write all to uninit",
        );

        let mut buf = vec![0.0; YiqView::buf_length_for((W, height), YiqField::Upper)];
        let view = YiqView::from_parts(&mut buf, (W, height), YiqField::Upper);
        view.y[..W].fill(10.0);
        view.y[W..2 * W].fill(20.0);
        let mut out: Vec<MaybeUninit<f32>> = Vec::new();
        out.resize_with(W * height * 4, MaybeUninit::uninit);
        view.write_to_strided_buffer_maybe_uninit::<Rgbx, f32, _>(
            ctx(),
            &mut out,
            BlitInfo::from_full_frame(W, height, RGBX_ROW_BYTES),
            DeinterlaceMode::Skip,
            (),
        );
        // Only the upper-field rows (0 and 2) are initialized; rows 1 and 3 must not be read.
        for (row, expected) in [(0, 10.0), (2, 20.0)] {
            let row_data: Vec<f32> = out[row * W * 4..(row + 1) * W * 4]
                .iter()
                .map(|v| unsafe { v.assume_init() })
                .collect();
            for px in row_data.chunks_exact(4) {
                for c in 0..3 {
                    assert!(
                        (px[c] - expected).abs() <= 0.01,
                        "row {row}: got {px:?}, expected {expected}"
                    );
                }
                assert_eq!(px[3], 1.0);
            }
        }
    }

    // ---- pixel transforms ----

    #[test]
    fn pixel_transform_on_set() {
        let img = gray_image(W, &row_values(3));
        let mut buf = vec![0.0; YiqView::buf_length_for((W, 3), YiqField::Both)];
        let mut view = YiqView::from_parts(&mut buf, (W, 3), YiqField::Both);
        view.set_from_strided_buffer::<Rgbx, f32, _>(
            ctx(),
            &img,
            BlitInfo::from_full_frame(W, 3, RGBX_ROW_BYTES),
            |rgb: [f32; 3]| rgb.map(|v| v * 2.0),
        );
        assert_rows_eq(&plane_rows(view.y, W), &[20., 40., 60.], "transform on set");
    }

    #[test]
    fn pixel_transform_on_write() {
        let mut buf = vec![0.0; YiqView::buf_length_for((W, 3), YiqField::Both)];
        let view = YiqView::from_parts(&mut buf, (W, 3), YiqField::Both);
        for (r, &v) in row_values(3).iter().enumerate() {
            view.y[r * W..(r + 1) * W].fill(v);
        }
        let mut out = vec![SENT; W * 3 * 4];
        view.write_to_strided_buffer::<Rgbx, f32, _>(
            ctx(),
            &mut out,
            BlitInfo::from_full_frame(W, 3, RGBX_ROW_BYTES),
            DeinterlaceMode::Bob,
            |rgb: [f32; 3]| rgb.map(|v| v + 5.0),
        );
        assert_rows_eq(&out_rows(&out, W), &[15., 25., 35.], "transform on write");
    }

    // ---- YiqOwned ----

    #[test]
    fn yiq_owned_matches_view() {
        let img = gray_image(W, &row_values(5));
        let mut owned = YiqOwned::from_strided_buffer::<Rgbx, f32>(
            ctx(),
            &img,
            RGBX_ROW_BYTES,
            W,
            5,
            YiqField::InterleavedUpper,
        );
        let view: YiqView = (&mut owned).into();
        assert_eq!(view.dimensions, (W, 5));
        assert_eq!(view.field, YiqField::InterleavedUpper);
        assert_eq!(view.num_rows(), 5);
        assert_rows_eq(
            &plane_rows(view.y, W),
            &[10., 30., 50., 20., 40.],
            "YiqOwned",
        );
    }

    // ---- pixel formats and data types ----

    fn round_trip_format<P: PixelFormat>(name: &str) {
        let (w, h) = (5usize, 4usize);
        let nc = P::NUM_COMPONENTS;
        let (ri, gi, bi, ai) = P::RGBA_INDICES;
        let mut img = vec![0.0f32; w * h * nc];
        for p in 0..w * h {
            let c = test_color(p);
            img[p * nc + ri] = c[0];
            img[p * nc + gi] = c[1];
            img[p * nc + bi] = c[2];
            if let Some(ai) = ai {
                img[p * nc + ai] = 0.25;
            }
        }

        let mut buf = vec![0.0; YiqView::buf_length_for((w, h), YiqField::Both)];
        let mut view = YiqView::from_parts(&mut buf, (w, h), YiqField::Both);
        let blit = BlitInfo::from_full_frame(w, h, w * nc * size_of::<f32>());
        view.set_from_strided_buffer::<P, f32, _>(ctx(), &img, blit, ());
        let mut out = vec![0.0f32; w * h * nc];
        view.write_to_strided_buffer::<P, f32, _>(ctx(), &mut out, blit, DeinterlaceMode::Bob, ());

        for p in 0..w * h {
            let c = test_color(p);
            for (expected, idx) in [(c[0], ri), (c[1], gi), (c[2], bi)] {
                let got = out[p * nc + idx];
                assert!(
                    (got - expected).abs() <= 0.004,
                    "{name}: pixel {p}: got {got}, expected {expected}"
                );
            }
            if let Some(ai) = ai {
                assert_eq!(out[p * nc + ai], 1.0, "{name}: alpha not set to ONE");
            }
        }
    }

    #[test]
    fn round_trip_pixel_formats() {
        round_trip_format::<Rgbx>("Rgbx");
        round_trip_format::<Xrgb>("Xrgb");
        round_trip_format::<Bgrx>("Bgrx");
        round_trip_format::<Xbgr>("Xbgr");
        round_trip_format::<Rgb>("Rgb");
        round_trip_format::<Bgr>("Bgr");
    }

    #[test]
    fn cross_format_swizzle() {
        // A pure-red pixel read in as Rgbx must land in the right channel of each output format.
        let img = [1.0f32, 0.0, 0.0, 0.5];
        let mut buf = vec![0.0; YiqView::buf_length_for((1, 1), YiqField::Both)];
        let mut view = YiqView::from_parts(&mut buf, (1, 1), YiqField::Both);
        view.set_from_strided_buffer::<Rgbx, f32, _>(
            ctx(),
            &img,
            BlitInfo::from_full_frame(1, 1, 4 * size_of::<f32>()),
            (),
        );

        let mut bgrx = [SENT; 4];
        view.write_to_strided_buffer::<Bgrx, f32, _>(
            ctx(),
            &mut bgrx,
            BlitInfo::from_full_frame(1, 1, 4 * size_of::<f32>()),
            DeinterlaceMode::Bob,
            (),
        );
        assert_rows_eq(&bgrx, &[0., 0., 1., 1.], "Bgrx layout");

        let mut rgb = [SENT; 3];
        view.write_to_strided_buffer::<Rgb, f32, _>(
            ctx(),
            &mut rgb,
            BlitInfo::from_full_frame(1, 1, 3 * size_of::<f32>()),
            DeinterlaceMode::Bob,
            (),
        );
        assert_rows_eq(&rgb, &[1., 0., 0.], "Rgb layout");
    }

    /// Test-only conversions for building input buffers and reading outputs in each data type.
    trait TestPixel: Normalize {
        fn from_f32(v: f32) -> Self;
        fn to_f32(self) -> f32;
    }
    impl TestPixel for f32 {
        fn from_f32(v: f32) -> Self {
            v
        }
        fn to_f32(self) -> f32 {
            self
        }
    }
    impl TestPixel for u8 {
        fn from_f32(v: f32) -> Self {
            (v * 255.0).round() as u8
        }
        fn to_f32(self) -> f32 {
            self as f32 / 255.0
        }
    }
    impl TestPixel for u16 {
        fn from_f32(v: f32) -> Self {
            (v * 65535.0).round() as u16
        }
        fn to_f32(self) -> f32 {
            self as f32 / 65535.0
        }
    }
    impl TestPixel for i16 {
        fn from_f32(v: f32) -> Self {
            (v * 32767.0).round() as i16
        }
        fn to_f32(self) -> f32 {
            self as f32 / 32767.0
        }
    }
    impl TestPixel for AfterEffectsU16 {
        // The only place these tests reach past the public API: AfterEffectsU16 has no public
        // constructor or accessor.
        fn from_f32(v: f32) -> Self {
            AfterEffectsU16((v * 32768.0).round() as u16)
        }
        fn to_f32(self) -> f32 {
            self.0 as f32 / 32768.0
        }
    }

    fn round_trip_data_type<T: TestPixel>(name: &str, tol: f32) {
        let (w, h) = (5usize, 4usize);
        let mut img = Vec::with_capacity(w * h * 4);
        for p in 0..w * h {
            let c = test_color(p);
            img.push(T::from_f32(c[0]));
            img.push(T::from_f32(c[1]));
            img.push(T::from_f32(c[2]));
            img.push(T::from_f32(0.25)); // junk alpha
        }

        let mut buf = vec![0.0; YiqView::buf_length_for((w, h), YiqField::Both)];
        let mut view = YiqView::from_parts(&mut buf, (w, h), YiqField::Both);
        let blit = BlitInfo::from_full_frame(w, h, w * 4 * size_of::<T>());
        view.set_from_strided_buffer::<Rgbx, T, _>(ctx(), &img, blit, ());
        let mut out = vec![T::from_f32(0.0); w * h * 4];
        view.write_to_strided_buffer::<Rgbx, T, _>(ctx(), &mut out, blit, DeinterlaceMode::Bob, ());

        for p in 0..w * h {
            let c = test_color(p);
            for ch in 0..3 {
                let got = out[p * 4 + ch].to_f32();
                assert!(
                    (got - c[ch]).abs() <= tol,
                    "{name}: pixel {p} channel {ch}: got {got}, expected {}",
                    c[ch]
                );
            }
            let alpha = out[p * 4 + 3].to_f32();
            assert!(
                (alpha - 1.0).abs() <= tol,
                "{name}: pixel {p}: alpha is {alpha}, expected 1.0"
            );
        }
    }

    #[test]
    fn round_trip_data_types() {
        round_trip_data_type::<f32>("f32", 0.004);
        round_trip_data_type::<u8>("u8", 0.012);
        round_trip_data_type::<u16>("u16", 0.004);
        round_trip_data_type::<i16>("i16", 0.004);
        round_trip_data_type::<AfterEffectsU16>("AfterEffectsU16", 0.004);
    }

    #[test]
    fn one_pixel_wide() {
        let img = gray_image(1, &row_values(3));
        let mut buf = vec![0.0; YiqView::buf_length_for((1, 3), YiqField::Both)];
        let mut view = YiqView::from_parts(&mut buf, (1, 3), YiqField::Both);
        let blit = BlitInfo::from_full_frame(1, 3, 4 * size_of::<f32>());
        view.set_from_strided_buffer::<Rgbx, f32, _>(ctx(), &img, blit, ());
        let mut out = vec![SENT; 1 * 3 * 4];
        view.write_to_strided_buffer::<Rgbx, f32, _>(
            ctx(),
            &mut out,
            blit,
            DeinterlaceMode::Bob,
            (),
        );
        assert_rows_eq(&out_rows(&out, 1), &row_values(3), "1px wide");
    }

    // ---- assertion checks ----

    #[test]
    #[should_panic(expected = "Rowbytes not aligned")]
    fn set_misaligned_row_bytes() {
        let img = gray_image(W, &row_values(2));
        let mut buf = vec![0.0; YiqView::buf_length_for((W, 2), YiqField::Both)];
        let mut view = YiqView::from_parts(&mut buf, (W, 2), YiqField::Both);
        view.set_from_strided_buffer::<Rgbx, f32, _>(
            ctx(),
            &img,
            BlitInfo::from_full_frame(W, 2, RGBX_ROW_BYTES + 1),
            (),
        );
    }

    #[test]
    #[should_panic]
    fn set_rect_taller_than_view() {
        let img = gray_image(W, &row_values(5));
        let mut buf = vec![0.0; YiqView::buf_length_for((W, 4), YiqField::Both)];
        let mut view = YiqView::from_parts(&mut buf, (W, 4), YiqField::Both);
        view.set_from_strided_buffer::<Rgbx, f32, _>(
            ctx(),
            &img,
            BlitInfo::from_full_frame(W, 5, RGBX_ROW_BYTES),
            (),
        );
    }
}
