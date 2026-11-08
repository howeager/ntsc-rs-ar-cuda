//! Run on real CUDA hardware: cargo run -p ntsc-rs --release --features cuda --example cuda_validate
use ntsc_rs::{
    Context, NtscEffect,
    cuda::{CudaPlan, CudaRenderer, DeviceFrame, driver::OwnedDevice},
    settings::standard::*,
    yiq_fielding::Bgrx,
};
use std::time::Instant;

fn main() -> Result<(), String> {
    let device = OwnedDevice::new(0)?;
    let mut renderer = unsafe { CudaRenderer::new(device.context)? };
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
        let src = device.allocate(input.len() * 4)?;
        src.upload(&input)?;
        let dst = device.allocate(input.len() * 4)?;
        for (name, effect) in &cases {
            // The CPU vertical-blend implementation needs two scratch rows.
            if height < 8 && effect.chroma_vert_blend {
                continue;
            }
            let plan = CudaPlan::new(effect, (width, height), 7, [1., 1.])?;
            let workspace = device.allocate(plan.workspace_bytes())?;
            let frame = DeviceFrame {
                ptr: src.ptr,
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
                        ptr: dst.ptr,
                        ..frame
                    },
                    workspace.ptr,
                    plan.workspace_bytes(),
                    std::ptr::null_mut(),
                )?;
            }
            let mut gpu = vec![0.; input.len()];
            dst.download(&mut gpu)?;
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
                    "{width}x{height} {name}: max error={error} at {index}, CPU={}, CUDA={}",
                    cpu[index], gpu[index]
                ));
            }
            worst = worst.max(error);
            checked += 1;
        }
    }
    println!("PASS: {checked} CPU/CUDA comparisons; maximum RGB error {worst:.7}");
    validate_layouts(&device, &mut renderer, &ctx)?;
    if std::env::args().any(|arg| arg == "--check-only") {
        return Ok(());
    }
    for (width, height) in [(640, 480), (1920, 1080)] {
        let effect = NtscEffect::default();
        let input: Vec<f32> = (0..width * height * 4)
            .map(|i| (i % 256) as f32 / 255.)
            .collect();
        let src = device.allocate(input.len() * 4)?;
        src.upload(&input)?;
        let dst = device.allocate(input.len() * 4)?;
        let plan = CudaPlan::new(&effect, (width, height), 0, [1., 1.])?;
        let workspace = device.allocate(plan.workspace_bytes() + 4 * 1024 * 1024)?;
        let frame = DeviceFrame {
            ptr: src.ptr,
            row_bytes: (width * 16) as i32,
            width,
            height,
            half: false,
        };
        for _ in 0..5 {
            unsafe {
                renderer.render(
                    &plan,
                    frame,
                    DeviceFrame {
                        ptr: dst.ptr,
                        ..frame
                    },
                    workspace.ptr,
                    plan.workspace_bytes(),
                    std::ptr::null_mut(),
                )?;
            }
        }
        let start = Instant::now();
        for f in 0..30 {
            let plan = CudaPlan::new(&effect, (width, height), f, [1., 1.])?;
            unsafe {
                renderer.render(
                    &plan,
                    frame,
                    DeviceFrame {
                        ptr: dst.ptr,
                        ..frame
                    },
                    workspace.ptr,
                    plan.workspace_bytes(),
                    std::ptr::null_mut(),
                )?;
            }
        }
        let gpu_ms = start.elapsed().as_secs_f64() * 1000. / 30.;
        let mut cpu = input.clone();
        let start = Instant::now();
        for f in 0..30 {
            cpu.copy_from_slice(&input);
            effect.apply_effect_to_buffer::<Bgrx, f32>(
                &ctx,
                (width, height),
                &mut cpu,
                f,
                [1., 1.],
            );
        }
        let cpu_ms = start.elapsed().as_secs_f64() * 1000. / 30.;
        println!(
            "{width}x{height}: CPU {cpu_ms:.3} ms, device-resident CUDA (including plan preparation) {gpu_ms:.3} ms, {:.2}x",
            cpu_ms / gpu_ms
        );
    }
    Ok(())
}

fn validate_layouts(
    device: &OwnedDevice,
    renderer: &mut CudaRenderer,
    ctx: &Context,
) -> Result<(), String> {
    use half::f16;
    let (width, height) = (13usize, 9usize);
    let stride = (width + 3) * 4;
    let mut checked = 0;
    for field in [
        UseField::Both,
        UseField::Upper,
        UseField::Lower,
        UseField::InterleavedUpper,
        UseField::InterleavedLower,
        UseField::Alternating,
    ] {
        let mut effect = NtscEffect::default();
        effect.use_field = field;
        for half in [false, true] {
            for reverse in [false, true] {
                let logical: Vec<f32> = (0..width * height * 4)
                    .map(|i| {
                        let value = ((i * 31 + i / 13) % 256) as f32 / 255.;
                        if half {
                            f16::from_f32(value).to_f32()
                        } else {
                            value
                        }
                    })
                    .collect();
                let mut physical = vec![-999f32; stride * height];
                for y in 0..height {
                    let row = if reverse { height - 1 - y } else { y };
                    physical[row * stride..row * stride + width * 4]
                        .copy_from_slice(&logical[y * width * 4..(y + 1) * width * 4]);
                }
                let bytes_per_component = if half { 2 } else { 4 };
                let src = device.allocate(physical.len() * bytes_per_component)?;
                if half {
                    src.upload(
                        &physical
                            .iter()
                            .map(|&v| f16::from_f32(v).to_bits())
                            .collect::<Vec<_>>(),
                    )?;
                } else {
                    src.upload(&physical)?;
                }
                let dst = device.allocate(physical.len() * bytes_per_component)?;
                let plan = CudaPlan::new(&effect, (width, height), 7, [1., 1.])?;
                let workspace = device.allocate(plan.workspace_bytes())?;
                let offset = if reverse {
                    (height - 1) * stride * bytes_per_component
                } else {
                    0
                };
                let frame = DeviceFrame {
                    ptr: src.ptr + offset as u64,
                    row_bytes: (stride * bytes_per_component) as i32 * if reverse { -1 } else { 1 },
                    width,
                    height,
                    half,
                };
                unsafe {
                    renderer.render(
                        &plan,
                        frame,
                        DeviceFrame {
                            ptr: dst.ptr + offset as u64,
                            ..frame
                        },
                        workspace.ptr,
                        plan.workspace_bytes(),
                        std::ptr::null_mut(),
                    )?;
                }
                if half {
                    let mut data = vec![0u16; physical.len()];
                    dst.download(&mut data)?;
                    physical = data
                        .into_iter()
                        .map(|v| f16::from_bits(v).to_f32())
                        .collect();
                } else {
                    dst.download(&mut physical)?;
                }
                let mut expected = logical.clone();
                effect.apply_effect_to_buffer::<Bgrx, f32>(
                    ctx,
                    (width, height),
                    &mut expected,
                    7,
                    [1., 1.],
                );
                for y in 0..height {
                    for x in 0..width * 4 {
                        let row = if reverse { height - 1 - y } else { y };
                        let gpu = physical[row * stride + x];
                        let cpu = if x % 4 == 3 {
                            logical[y * width * 4 + x]
                        } else {
                            expected[y * width * 4 + x]
                        };
                        let cpu = if half {
                            f16::from_f32(cpu).to_f32()
                        } else {
                            cpu
                        };
                        if !gpu.is_finite() || (gpu - cpu).abs() > 0.003 {
                            return Err(format!(
                                "Padded/reversed/half layout mismatch: field={field:?} half={half} reverse={reverse} at {x},{y}: {cpu} vs {gpu}"
                            ));
                        }
                    }
                }
                checked += 1;
            }
        }
    }
    println!("PASS: {checked} padded/signed-pitch/half-float/alpha comparisons");
    Ok(())
}
