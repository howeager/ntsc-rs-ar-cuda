extern crate criterion;

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use ntsc_rs::{
    NtscEffect,
    ctx::Context,
    yiq_fielding::{BlitInfo, DeinterlaceMode, Rgb, YiqView, pixel_bytes_for},
};

const BENCH_IMAGE: &'static [u8] = include_bytes!("./balloons.png");

fn criterion_benchmark(c: &mut Criterion) {
    let img = image::load_from_memory_with_format(BENCH_IMAGE, image::ImageFormat::Png).unwrap();
    let img = img.to_rgb8();
    // Reuse the pool as the application does; creating threads for every sample
    // makes short frame-processing benchmarks unnecessarily expensive and noisy.
    let ctx = Context::new();
    for (name, img) in [
        ("full effect", img.clone()),
        (
            "full effect 1080p",
            image::imageops::resize(&img, 1920, 1080, image::imageops::FilterType::Triangle),
        ),
    ] {
        c.bench_function(name, |b| {
            b.iter_batched_ref(
                || {
                    let img = img.clone();
                    let width = img.width() as usize;
                    let height = img.height() as usize;
                    let effect = NtscEffect::default();
                    let data = img.into_raw();
                    let scratch = vec![
                        0f32;
                        YiqView::buf_length_for(
                            (width, height),
                            effect.use_field.to_yiq_field(0)
                        )
                    ];
                    let dest = vec![0u8; data.len()];
                    (width, height, effect, data, scratch, dest)
                },
                |(width, height, effect, buf, scratch, dest)| {
                    let mut yiq = YiqView::from_parts(
                        scratch,
                        (*width, *height),
                        effect.use_field.to_yiq_field(0),
                    );
                    let row_bytes = *width * pixel_bytes_for::<Rgb, u8>();
                    let blit_info = BlitInfo::from_full_frame(*width, *height, row_bytes);
                    yiq.set_from_strided_buffer::<Rgb, u8, _>(&ctx, buf, blit_info, ());
                    effect.apply_effect_to_yiq(&ctx, &mut yiq, 0, [1.0, 1.0]);
                    yiq.write_to_strided_buffer::<Rgb, u8, _>(
                        &ctx,
                        dest,
                        blit_info,
                        DeinterlaceMode::Bob,
                        (),
                    );
                    black_box(dest);
                },
                criterion::BatchSize::LargeInput,
            );
        });
    }
}

criterion_group! {
    name = benches;
    config = {
        let config = Criterion::default();

        config
    };
    targets = criterion_benchmark
}
criterion_main!(benches);
