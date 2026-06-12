use std::env;
use std::process::Command;

// Detect a nightly toolchain so the portable SIMD kernels can be gated on
// it; stable builds fall back to the scalar paths.
fn main() {
    println!("cargo:rustc-check-cfg=cfg(nightly_portable_simd)");
    let rustc = env::var("RUSTC").unwrap_or_else(|_| "rustc".into());
    let is_nightly = Command::new(rustc)
        .arg("--version")
        .output()
        .is_ok_and(|out| String::from_utf8_lossy(&out.stdout).contains("nightly"));
    if is_nightly {
        println!("cargo:rustc-cfg=nightly_portable_simd");
    }
}
