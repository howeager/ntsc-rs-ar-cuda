use std::{env, path::PathBuf, process::Command};

fn main() {
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-changed=src/ntsc/cuda/kernels.cu");
    if env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }
    let nvcc = env::var_os("NVCC").map(PathBuf::from).unwrap_or_else(|| {
        env::var_os("CUDA_PATH")
            .map(PathBuf::from)
            .map(|p| {
                p.join("bin")
                    .join(if cfg!(windows) { "nvcc.exe" } else { "nvcc" })
            })
            .unwrap_or_else(|| "nvcc".into())
    });
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("ntsc.ptx");
    let mut cmd = Command::new(nvcc);
    cmd.args([
        "--ptx",
        "--std=c++14",
        "--gpu-architecture=compute_75",
        "--fmad=false",
    ]);
    if cfg!(windows) {
        let compiler = cc::Build::new().get_compiler();
        cmd.arg("--compiler-bindir")
            .arg(compiler.path().parent().unwrap());
        cmd.envs(compiler.env().iter().cloned());
    }
    let result = cmd
        .arg("src/ntsc/cuda/kernels.cu")
        .arg("-o")
        .arg(out)
        .output()
        .expect("CUDA feature requires nvcc (set CUDA_PATH or NVCC)");
    assert!(
        result.status.success(),
        "CUDA compilation failed:\n{}\n{}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
}
