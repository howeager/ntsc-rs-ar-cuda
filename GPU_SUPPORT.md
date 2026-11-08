# NTSC GPU support

The Windows x64 plugin now includes native Premiere CUDA and OpenCL renderers in one binary. The existing filename `ntsc-rs-ae-cuda.aex` is retained so the installer and existing installations keep working. The source was recovered from the supplied `ntsc-rs.rar` archive and is included in `source/` with its original licenses.

Premiere provides the active renderer's device/context/queue. CUDA uses NVIDIA's driver API; OpenCL compiles the same processing stages for the selected OpenCL device, including supported AMD and Intel devices. The plugin does not force a particular GPU model or change Premiere's project renderer. The host must expose the corresponding CUDA/OpenCL framework. Unsupported frameworks use the original CPU effect path; Apple Metal and DirectX GPU-surface backends are not implemented.

The OpenCL backend shares the CUDA plan builder and preserves NTSC controls, field modes, CPU-generated noise seeds, recursive filters, and BGRA 16f/32f frame handling. Rendering uses GPU buffers throughout; there is no full-frame CPU readback. Small control arrays are uploaded from the CPU. OpenCL requires a host-provided in-order queue and positive frame row pitches.

`NTSC_RS_DISABLE_GPU` disables native GPU initialization. `NTSC_RS_DISABLE_CUDA` disables CUDA only, as before. If backend initialization fails, the effect declines native GPU support so Premiere can use its CPU path.

## Build

Install Rust/MSVC Build Tools, Windows SDK and libclang as required by the upstream bindings. Run `Build_GPU.ps1`. With the CUDA Toolkit installed, it builds CUDA+OpenCL. Without nvcc it builds OpenCL only; that build cannot accelerate Premiere's CUDA renderer. End users of the prebuilt combined plugin do not need the CUDA Toolkit.

## Validation

The OpenCL renderer passed 292 comparisons against the CPU reference, across six field modes, two filter types, six chroma-demodulation modes, tiny/odd image sizes, and custom distortion controls. Maximum RGB difference was 0.0001372 on an RTX 3070 Ti. Alpha, pitched 16f/32f frames, CUDA regressions and parameter mapping are checked separately. Premiere host integration and AMD/Intel hardware are not yet validated; these builds should be tried on a copy of a project before replacing a production installation.
