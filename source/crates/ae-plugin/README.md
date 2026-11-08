# ntsc-rs After Effects plugin

ntsc-rs can be used from After Effects and Premiere! When built, the plugin can be found under the "Stylize" category.

## Building the plugin

See [the documentation on the ntsc-rs website](https://ntsc.rs/docs/building-from-source/) for up-to-date information.

## Windows CUDA build for Premiere

```powershell
cargo build -p ntsc-rs-ae-plugin --release --features cuda
```

This adds Premiere's native `xGPUFilterEntry` alongside the existing software
effect. Copy `target/release/ntsc_rs_ae_plugin.dll` to
`C:\Program Files\Adobe\Common\Plug-ins\7.0\MediaCore\ntsc-rs-ae.aex`, then restart
Premiere. Find **NTSC-rs** under **Video Effects > Stylize** and use the Mercury
Playback Engine GPU Acceleration (CUDA) project renderer.

Building requires the MSVC build tools and a CUDA toolkit providing `nvcc`.
Set `CUDA_PATH` to its installation directory, or set `NVCC` to the compiler.
The build embeds PTX targeting compute capability 7.5 (NVIDIA Turing or newer).
Deployment needs a driver compatible with the toolkit used to build the PTX;
the toolkit and CUDA runtime DLLs do not need to be distributed with the effect.
GStreamer is not required for this plugin.

The CUDA path supports Premiere's BGRA 16f and 32f frames, signed row pitches,
all field/demodulation modes, and the existing effect controls. Pixel processing
stays on the GPU. Seeded per-row parameters and sparse snow transients are
prepared on the CPU using the existing RNG. Recursive filters use parallel
segments with a boundary-state correction. Floating-point ordering differs from
the software path, so output is close rather than bit-identical.

After Effects continues using the CPU renderer. Premiere also uses the CPU path
when its device is not CUDA, the kernels cannot load during instance creation,
or `NTSC_RS_DISABLE_CUDA` is set before launching Premiere. The CUDA feature is
optional; a normal build retains the existing CPU-only behavior.

### Validation

```powershell
cargo test -p ntsc-rs --features cuda --lib
cargo test -p ntsc-rs-ae-plugin --features cuda --lib
cargo run -p ntsc-rs --release --features cuda --example cuda_validate
# Skip timing when running under NVIDIA Compute Sanitizer:
cargo run -p ntsc-rs --release --features cuda --example cuda_validate -- --check-only
```

The hardware tool compares CUDA output with CPU output across field/filter
combinations, small and odd frame sizes, custom settings, padded/reversed rows,
half floats, and alpha. It then benchmarks device-resident CUDA frames including
CPU plan preparation. At 1080p on an RTX 3070 Ti/i9-12900K the development build
measured approximately 5.9 ms on CUDA versus 13.8 ms on the CPU. At 640x480, CUDA
was slightly slower (about 2.6 ms versus 2.3 ms); small previews can be dominated
by kernel-launch overhead. These are processing benchmarks, not Premiere export
timings, and exclude host-to-device frame transfers.
