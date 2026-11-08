use super::*;

pub(super) struct FieldPlan {
    pub params: Params,
    pub commands: Vec<Params>,
}

/// A validated frame's GPU commands and small CPU-generated control data.
pub struct CudaPlan {
    pub(super) width: usize,
    pub(super) height: usize,
    pub(super) fields: Vec<FieldPlan>,
    pub(super) auxiliary: Vec<f32>,
}

struct Builder<'a> {
    plan: &'a mut CudaPlan,
    commands: Vec<Params>,
    p: Params,
    ctx: EffectCtx,
}

impl CudaPlan {
    pub fn new(
        effect: &NtscEffect,
        dimensions: (usize, usize),
        frame_num: usize,
        scale: [f32; 2],
    ) -> Result<Self> {
        let (width, height) = dimensions;
        if width == 0
            || height == 0
            || width > 65_536
            || height > 65_536
            || width
                .checked_mul(height)
                .is_none_or(|n| n > i32::MAX as usize / 8)
        {
            return Err("Unsupported GPU frame dimensions".into());
        }
        if scale.iter().any(|v| !v.is_finite() || *v <= 0.0) {
            return Err("Invalid GPU render scale".into());
        }
        let mut plan = Self {
            width,
            height,
            fields: Vec::new(),
            auxiliary: Vec::new(),
        };
        let interleaved = matches!(
            effect.use_field,
            UseField::InterleavedUpper | UseField::InterleavedLower
        );
        let fields: Vec<(usize, usize, usize)> = match effect.use_field {
            UseField::Both => vec![(0, 1, frame_num)],
            UseField::InterleavedUpper => vec![(0, 2, frame_num * 2), (1, 2, frame_num * 2 + 1)],
            UseField::InterleavedLower => vec![(1, 2, frame_num * 2), (0, 2, frame_num * 2 + 1)],
            UseField::Upper => vec![(0, 2, frame_num)],
            UseField::Lower => vec![(1, 2, frame_num)],
            UseField::Alternating => vec![(1 - (frame_num & 1), 2, frame_num)],
        };
        for (first, step, frame) in fields {
            if interleaved && height == 1 && first == 1 {
                continue;
            }
            // A one-line image has no lower field; the CPU uses its only row.
            let first = if height == 1 { 0 } else { first };
            let rows = (height - first).div_ceil(step);
            let p = Params {
                w: width as i32,
                h: rows as i32,
                first: first as i32,
                mode: step as i32,
                detail: (!interleaved && step == 2) as i32,
                frame: frame as i32,
                ..Default::default()
            };
            let ctx = EffectCtx::new(crate::ctx::global(), effect, height, frame, scale);
            if !ctx.horizontal_scale.is_finite()
                || !ctx.vertical_scale.is_finite()
                || ctx.horizontal_scale <= 0.0
                || ctx.vertical_scale <= 0.0
            {
                return Err("Unsupported GPU bandwidth scale".into());
            }
            let mut builder = Builder {
                plan: &mut plan,
                commands: Vec::new(),
                p,
                ctx,
            };
            builder.build(effect)?;
            let commands = builder.commands;
            plan.fields.push(FieldPlan {
                params: p,
                commands,
            });
        }
        Ok(plan)
    }
    pub fn workspace_bytes(&self) -> usize {
        (self.width * self.height * 4 + self.scratch_floats() + self.auxiliary.len()) * 4
    }
    pub(super) fn scratch_floats(&self) -> usize {
        let n = self.width * self.height;
        (n * 3).max(n + 6 * self.width.div_ceil(128) * self.height)
    }
}

impl Builder<'_> {
    fn push(&mut self, op: i32, plane: i32, mode: i32) {
        self.commands.push(Params {
            op,
            plane,
            mode,
            ..self.p
        });
    }
    fn filter(
        &mut self,
        plane: i32,
        filter: TransferFunction,
        first: bool,
        delay: usize,
    ) -> Result<()> {
        let (coefficients, initial, len) = filter.cuda_coefficients();
        let mut transition = [0.0; 9];
        for column in 0..len - 1 {
            let mut z = [0.0; 4];
            z[column] = 1.0;
            for _ in 0..128 {
                let out = z[0];
                for i in 0..len - 1 {
                    z[i] = (-coefficients[4 + i]).mul_add(out, z[i + 1]);
                }
            }
            for row in 0..len - 1 {
                transition[row * 3 + column] = z[row];
            }
        }
        if coefficients
            .iter()
            .chain(initial.iter())
            .any(|v| !v.is_finite())
            || delay > 1_000_000
        {
            return Err("Unstable GPU filter settings".into());
        }
        self.commands.push(Params {
            op: 1,
            plane,
            coefficients,
            initial,
            transition,
            mode: len as i32,
            first: first as i32,
            delay: delay as i32,
            ..self.p
        });
        Ok(())
    }
    fn chroma_lowpass(&mut self, mode: ChromaLowpass, kind: FilterType) -> Result<()> {
        let rate = NTSC_RATE * self.ctx.horizontal_scale;
        match mode {
            ChromaLowpass::None => {}
            ChromaLowpass::Full => {
                self.filter(1, make_lowpass_for_type(1_300_000.0, rate, kind), false, 2)?;
                self.filter(2, make_lowpass_for_type(600_000.0, rate, kind), false, 4)?;
            }
            ChromaLowpass::Light => {
                for plane in [1, 2] {
                    self.filter(
                        plane,
                        make_lowpass_for_type(2_600_000.0, rate, kind),
                        false,
                        1,
                    )?;
                }
            }
        }
        Ok(())
    }
    fn noise(&mut self, plane: i32, settings: &FbmNoiseSettings, seed: u64) {
        let rng = SplitMix64::new(self.ctx.seed)
            .mix(seed)
            .mix(self.ctx.frame_num as u64);
        self.row_noise(
            plane,
            &rng,
            0,
            self.p.h as usize,
            0,
            settings.frequency / self.ctx.horizontal_scale,
            settings.intensity,
            settings.detail.try_into().unwrap_or_default(),
            false,
        );
    }
    fn row_noise(
        &mut self,
        plane: i32,
        rng: &SplitMix64,
        start: usize,
        rows: usize,
        cutoff: usize,
        frequency: f32,
        intensity: f32,
        detail: u32,
        tracking: bool,
    ) {
        let aux = self.plan.auxiliary.len() as i32;
        for y in 0..self.p.h as usize {
            let index = y.saturating_sub(start) + cutoff;
            let mut row_rng = rng.clone().mix(index as u64);
            let seed = row_rng.random::<i32>();
            let offset = row_rng.random::<f32>() * self.p.w as f32;
            let scale = if tracking {
                (index as f32 / rows as f32).powi(2) * 4.0
            } else {
                1.0
            };
            let amount = if y < start {
                0.0
            } else {
                intensity * scale * 0.25
            };
            self.plan
                .auxiliary
                .extend_from_slice(&[f32::from_bits(seed as u32), offset, amount]);
        }
        self.commands.push(Params {
            op: 6,
            plane,
            aux,
            frequency,
            detail: detail.clamp(1, 5) as i32,
            ..self.p
        });
    }
    fn sparse(&mut self, row: &[f32], y: usize) {
        for (x, &value) in row.iter().enumerate().filter(|(_, v)| **v != 0.0) {
            self.plan
                .auxiliary
                .extend_from_slice(&[f32::from_bits((y * self.p.w as usize + x) as u32), value]);
        }
    }
    fn snow(&mut self, intensity: f32, anisotropy: f32) {
        let aux = self.plan.auxiliary.len();
        let rng = SplitMix64::new(self.ctx.seed)
            .mix(noise_seeds::SNOW)
            .mix(self.ctx.frame_num as u64);
        let mut row = vec![0.0; self.p.w as usize];
        for y in 0..self.p.h as usize {
            row.fill(0.0);
            self.ctx.row_speckles(
                &mut row,
                &mut rng.clone().mix(y as u64),
                intensity,
                anisotropy,
            );
            self.sparse(&row, y);
        }
        self.commands.push(Params {
            op: 9,
            aux: aux as i32,
            detail: ((self.plan.auxiliary.len() - aux) / 2) as i32,
            ..self.p
        });
    }
    fn shift_rows(
        &mut self,
        mask: i32,
        extend: bool,
        vertical: i32,
        shifts: &[f32],
        starts: &[usize],
    ) {
        let aux = self.plan.auxiliary.len() as i32;
        for y in 0..self.p.h as usize {
            self.plan
                .auxiliary
                .extend_from_slice(&[shifts[y], starts[y] as f32]);
        }
        self.commands.push(Params {
            op: 7,
            plane: mask,
            mode: extend as i32,
            vertical,
            aux,
            ..self.p
        });
    }
    fn head_switching(&mut self, settings: &HeadSwitchingSettings) {
        let num_rows = (settings.height.max(0) as f32 * self.ctx.vertical_scale).round() as usize;
        let offset = (settings.offset.max(0) as f32 * self.ctx.vertical_scale).round() as usize;
        if offset >= num_rows {
            return;
        }
        let affected = num_rows - offset;
        let h = self.p.h as usize;
        let w = self.p.w as usize;
        let start = h.saturating_sub(affected);
        let cutoff = affected.saturating_sub(h);
        let rng = SplitMix64::new(self.ctx.seed)
            .mix(noise_seeds::HEAD_SWITCHING)
            .mix(self.ctx.frame_num as u64);
        let mut shifts = vec![0.0; h];
        let mut starts = vec![w; h];
        let mut transient = None;
        for y in start..h {
            let index = affected - (y - start + cutoff);
            let row_shift =
                settings.horiz_shift * ((index + offset) as f32 / num_rows as f32).powf(1.5);
            shifts[y] = (row_shift + rng.clone().mix(index as u64).random::<f32>() - 0.5)
                * self.ctx.horizontal_scale;
            starts[y] = 0;
            if index == affected
                && let Some(mid) = settings.mid_line.as_option()
            {
                let mut rng = SplitMix64::new(self.ctx.seed)
                    .mix(noise_seeds::HEAD_SWITCHING_MID_LINE_JITTER)
                    .mix(self.ctx.frame_num as u64);
                let jitter = ((rng.random::<f32>() + rng.random::<f32>()) * 0.5 - 0.5) * mid.jitter;
                starts[y] = (w as f32 * (mid.position + jitter)) as usize;
                transient = Some((y, starts[y], (rng.random::<f32>() + 0.5) * 0.5));
            }
        }
        self.shift_rows(1, false, 0, &shifts, &starts);
        if let Some((y, start, intensity)) = transient {
            let aux = self.plan.auxiliary.len();
            let length = 16.0 * self.ctx.horizontal_scale;
            for x in start..start.saturating_add(length.ceil() as usize).min(w) {
                self.plan.auxiliary.extend_from_slice(&[
                    f32::from_bits((y * w + x) as u32),
                    (1.0 - (x - start) as f32 / length).powi(3) * intensity,
                ]);
            }
            self.commands.push(Params {
                op: 9,
                aux: aux as i32,
                detail: ((self.plan.auxiliary.len() - aux) / 2) as i32,
                ..self.p
            });
        }
    }
    fn tracking(&mut self, settings: &TrackingNoiseSettings) {
        let rows = (settings.height.max(0) as f32 * self.ctx.vertical_scale).round() as usize;
        if rows == 0 {
            return;
        }
        let h = self.p.h as usize;
        let start = h.saturating_sub(rows);
        let cutoff = rows.saturating_sub(h);
        let mut rng = SplitMix64::new(self.ctx.seed)
            .mix(noise_seeds::TRACKING_NOISE)
            .mix(self.ctx.frame_num as u64);
        let seed = rng.random::<i32>();
        let offset = rng.random::<f32>() * h as f32;
        let mut noise = vec![0.0; rows.min(h)];
        sample_noise_1d::<Simplex1d, _>(
            self.ctx.level,
            &Simplex {
                seed,
                frequency: 0.5,
            },
            [offset],
            [noise.len()],
            &mut noise,
        );
        let mut shifts = vec![0.0; h];
        let mut starts = vec![self.p.w as usize; h];
        for y in start..h {
            shifts[y] = noise[y - start]
                * ((y - start + cutoff) as f32 / rows as f32)
                * settings.wave_intensity
                * 0.25
                * self.ctx.horizontal_scale;
            starts[y] = 0;
        }
        self.shift_rows(1, false, 0, &shifts, &starts);
        self.row_noise(
            0,
            &rng,
            start,
            rows,
            cutoff,
            0.25 / self.ctx.horizontal_scale,
            settings.noise_intensity,
            1,
            true,
        );
        let aux = self.plan.auxiliary.len();
        let mut row = vec![0.0; self.p.w as usize];
        for y in start..h {
            let index = y - start + cutoff;
            row.fill(0.0);
            self.ctx.row_speckles(
                &mut row,
                &mut rng.clone().mix(index as u64),
                settings.snow_intensity * (index as f32 / rows as f32).powi(2),
                settings.snow_anisotropy,
            );
            self.sparse(&row, y);
        }
        self.commands.push(Params {
            op: 9,
            aux: aux as i32,
            detail: ((self.plan.auxiliary.len() - aux) / 2) as i32,
            ..self.p
        });
    }
    fn rotate(&mut self, intensity: f32, noisy: bool) {
        let aux = self.plan.auxiliary.len() as i32;
        let rng = SplitMix64::new(self.ctx.seed)
            .mix(noise_seeds::VIDEO_CHROMA_PHASE)
            .mix(self.ctx.frame_num as u64);
        for y in 0..self.p.h {
            let phase = if noisy {
                (rng.clone().mix(y as u64).random::<f32>() - 0.5) * 2.0 * intensity
            } else {
                intensity
            };
            let (sin, cos) = (phase * PI * 2.0).sin_cos();
            self.plan.auxiliary.extend_from_slice(&[sin, cos]);
        }
        self.commands.push(Params {
            op: 11,
            aux,
            ..self.p
        });
    }
    fn build(&mut self, s: &NtscEffect) -> Result<()> {
        let scale = self.ctx.horizontal_scale;
        let rate = NTSC_RATE * scale;
        match s.input_luma_filter {
            LumaLowpass::None => {}
            LumaLowpass::Box => {
                if self.p.w >= 2 {
                    self.push(10, 0, 0);
                }
            }
            LumaLowpass::Notch => self.filter(0, make_notch_filter(0.5, 2.0), true, 0)?,
        }
        self.chroma_lowpass(s.chroma_lowpass_in, s.filter_type)?;
        self.push(2, 0, 0);
        if s.composite_sharpening != 0.0 {
            self.filter(
                0,
                make_lowpass((315000000.0 / 88.0 / 2.0) * scale, rate)
                    .with_scale(-s.composite_sharpening),
                false,
                0,
            )?;
        }
        if s.composite_noise.enabled {
            self.noise(0, &s.composite_noise.settings, noise_seeds::VIDEO_COMPOSITE);
        }
        if s.snow_intensity > 0.0 {
            self.snow(s.snow_intensity * 0.01, s.snow_anisotropy);
        }
        if s.head_switching.enabled {
            self.head_switching(&s.head_switching.settings);
        }
        if s.tracking_noise.enabled {
            self.tracking(&s.tracking_noise.settings);
        }
        let mut demod = s.chroma_demodulation;
        if (self.p.h == 1
            && !matches!(
                demod,
                ChromaDemodulationFilter::Box | ChromaDemodulationFilter::Notch
            ))
            || (self.p.w < 3
                && matches!(
                    demod,
                    ChromaDemodulationFilter::TwoD | ChromaDemodulationFilter::TwoDAdaptive
                ))
        {
            demod = ChromaDemodulationFilter::Notch;
        }
        self.push(3, 0, 0);
        match demod {
            ChromaDemodulationFilter::Notch => {
                self.filter(0, make_notch_filter(0.5, 2.0), false, 0)?
            }
            _ => self.push(
                4,
                0,
                match demod {
                    ChromaDemodulationFilter::Box => 0,
                    ChromaDemodulationFilter::OneLineComb => 1,
                    ChromaDemodulationFilter::TwoLineComb => 2,
                    ChromaDemodulationFilter::TwoD => 3,
                    ChromaDemodulationFilter::TwoDAdaptive => 4,
                    _ => unreachable!(),
                },
            ),
        }
        self.push(5, 0, 0);
        if s.luma_smear > 0.0 {
            self.filter(
                0,
                make_lowpass(f32::exp2(-4.0 * s.luma_smear) * 0.25, scale),
                false,
                0,
            )?;
        }
        if s.ringing.enabled {
            let r = &s.ringing.settings;
            self.filter(
                0,
                make_notch_filter((r.frequency / scale).clamp(0.0, 1.0), r.power)
                    .with_scale(r.intensity),
                true,
                1,
            )?;
        }
        if s.luma_noise.enabled {
            self.noise(0, &s.luma_noise.settings, noise_seeds::VIDEO_LUMA);
        }
        if s.chroma_noise.enabled {
            self.noise(1, &s.chroma_noise.settings, noise_seeds::VIDEO_CHROMA_I);
            self.noise(2, &s.chroma_noise.settings, noise_seeds::VIDEO_CHROMA_Q);
        }
        if s.chroma_phase_error > 0.0 {
            self.rotate(s.chroma_phase_error, false);
        }
        if s.chroma_phase_noise_intensity > 0.0 {
            self.rotate(s.chroma_phase_noise_intensity, true);
        }
        if s.chroma_delay_horizontal != 0.0 || s.chroma_delay_vertical != 0 {
            let shifts = vec![s.chroma_delay_horizontal * scale; self.p.h as usize];
            self.shift_rows(
                6,
                false,
                (s.chroma_delay_vertical as f32 * self.ctx.vertical_scale).round() as i32,
                &shifts,
                &vec![0; self.p.h as usize],
            );
        }
        if s.vhs_settings.enabled {
            let v = &s.vhs_settings.settings;
            if v.edge_wave.enabled && v.edge_wave.settings.intensity > 0.0 {
                let settings = &v.edge_wave.settings;
                let mut rng = SplitMix64::new(self.ctx.seed).mix(noise_seeds::EDGE_WAVE);
                let seed = rng.random::<i32>();
                let offset = rng.random::<f32>() * self.p.h as f32;
                let mut shifts = vec![0.0; self.p.h as usize];
                let noise = Fbm {
                    seed,
                    octaves: settings.detail.clamp(1, 5) as usize,
                    gain: FRAC_1_SQRT_2,
                    lacunarity: 2.0,
                    frequency: settings.frequency / self.ctx.vertical_scale,
                };
                sample_noise_2d::<Simplex2d, _>(
                    self.ctx.level,
                    &noise,
                    [offset, self.ctx.frame_num as f32 * settings.speed],
                    [shifts.len(), 1],
                    &mut shifts,
                );
                for shift in &mut shifts {
                    *shift = (*shift / 0.022) * settings.intensity * 0.5 * scale;
                }
                self.shift_rows(7, true, 0, &shifts, &vec![0; self.p.h as usize]);
            }
            if let Some(VHSTapeParams {
                luma_cut,
                chroma_cut,
                chroma_delay,
            }) = v.tape_speed.filter_params()
            {
                self.filter(
                    0,
                    make_lowpass_for_type(luma_cut, rate, s.filter_type),
                    false,
                    0,
                )?;
                for plane in [1, 2] {
                    self.filter(
                        plane,
                        make_lowpass_for_type(chroma_cut, rate, s.filter_type),
                        false,
                        (chroma_delay as f32 * scale).round() as usize,
                    )?;
                }
                self.filter(0, make_lowpass(luma_cut, rate).with_scale(-1.6), false, 0)?;
            }
            if v.chroma_loss > 0.0 {
                let aux = self.plan.auxiliary.len() as i32;
                let mut flags = vec![0.0; self.p.h as usize];
                let mut rng = SplitMix64::new(self.ctx.seed)
                    .mix(noise_seeds::CHROMA_LOSS)
                    .mix(self.ctx.frame_num as u64);
                let dist = geometric_lambda(v.chroma_loss as f64);
                let mut row = 0usize;
                loop {
                    row = row.saturating_add(rng.random_geometric(dist));
                    if row >= flags.len() {
                        break;
                    }
                    flags[row] = 1.0;
                    row += 1;
                }
                self.plan.auxiliary.extend_from_slice(&flags);
                self.commands.push(Params {
                    op: 12,
                    aux,
                    ..self.p
                });
            }
            if v.sharpen.enabled
                && let Some(VHSTapeParams { luma_cut, .. }) = v.tape_speed.filter_params()
            {
                let sharpen = &v.sharpen.settings;
                let extra = if s.filter_type == FilterType::ConstantK {
                    4.0
                } else {
                    1.0
                };
                self.filter(
                    0,
                    make_lowpass_for_type(
                        luma_cut * extra * sharpen.frequency,
                        rate,
                        s.filter_type,
                    )
                    .with_scale(-sharpen.intensity * 2.0 * sharpen.frequency),
                    false,
                    0,
                )?;
            }
        }
        if s.chroma_vert_blend && self.plan.height >= 2 {
            self.push(8, 0, 0);
        }
        self.chroma_lowpass(s.chroma_lowpass_out, s.filter_type)
    }
}
