//! CPU instruction-set and topology probing.

/// A snapshot of CPU capabilities relevant to engine dispatch (SIMD width,
/// logical core count). M0 covers the common x86-64 and AArch64 flags via
/// compile-time `target_feature` plus a runtime core count.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CpuInfo {
    /// Number of logical cores available to the process.
    pub logical_cores: usize,
    /// Architecture family name (`"x86_64"`, `"aarch64"`, ...).
    pub arch: &'static str,
    /// x86-64 SSE2 is available (always true on `x86_64`).
    pub sse2: bool,
    /// x86-64 AVX2 is available.
    pub avx2: bool,
    /// x86-64 AVX-512F is available.
    pub avx512f: bool,
    /// AArch64 NEON is available (always true on `aarch64`).
    pub neon: bool,
}

impl CpuInfo {
    /// Probe the current CPU.
    pub fn detect() -> Self {
        let logical_cores = logical_core_count();
        Self {
            logical_cores,
            arch: arch_name(),
            sse2: detect_sse2(),
            avx2: detect_avx2(),
            avx512f: detect_avx512f(),
            neon: detect_neon(),
        }
    }

    /// Widest useful SIMD lane count in `f32` lanes, inferred from flags.
    pub fn preferred_f32_lanes(&self) -> usize {
        if self.avx512f {
            16
        } else if self.avx2 {
            8
        } else if self.sse2 || self.neon {
            4
        } else {
            1
        }
    }
}

const fn arch_name() -> &'static str {
    if cfg!(target_arch = "x86_64") {
        "x86_64"
    } else if cfg!(target_arch = "aarch64") {
        "aarch64"
    } else if cfg!(target_arch = "x86") {
        "x86"
    } else if cfg!(target_arch = "wasm32") {
        "wasm32"
    } else {
        "unknown"
    }
}

#[cfg(feature = "std")]
fn logical_core_count() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

#[cfg(not(feature = "std"))]
fn logical_core_count() -> usize {
    1
}

#[cfg(all(feature = "std", target_arch = "x86_64"))]
fn detect_sse2() -> bool {
    true // guaranteed by the x86_64 baseline
}
#[cfg(all(feature = "std", target_arch = "x86_64"))]
fn detect_avx2() -> bool {
    std::is_x86_feature_detected!("avx2")
}
#[cfg(all(feature = "std", target_arch = "x86_64"))]
fn detect_avx512f() -> bool {
    std::is_x86_feature_detected!("avx512f")
}
#[cfg(all(feature = "std", target_arch = "x86_64"))]
fn detect_neon() -> bool {
    false
}

#[cfg(all(feature = "std", target_arch = "aarch64"))]
fn detect_sse2() -> bool {
    false
}
#[cfg(all(feature = "std", target_arch = "aarch64"))]
fn detect_avx2() -> bool {
    false
}
#[cfg(all(feature = "std", target_arch = "aarch64"))]
fn detect_avx512f() -> bool {
    false
}
#[cfg(all(feature = "std", target_arch = "aarch64"))]
fn detect_neon() -> bool {
    true // mandatory on AArch64
}

#[cfg(not(any(
    all(feature = "std", target_arch = "x86_64"),
    all(feature = "std", target_arch = "aarch64")
)))]
fn detect_sse2() -> bool {
    cfg!(target_feature = "sse2")
}
#[cfg(not(any(
    all(feature = "std", target_arch = "x86_64"),
    all(feature = "std", target_arch = "aarch64")
)))]
fn detect_avx2() -> bool {
    cfg!(target_feature = "avx2")
}
#[cfg(not(any(
    all(feature = "std", target_arch = "x86_64"),
    all(feature = "std", target_arch = "aarch64")
)))]
fn detect_avx512f() -> bool {
    cfg!(target_feature = "avx512f")
}
#[cfg(not(any(
    all(feature = "std", target_arch = "x86_64"),
    all(feature = "std", target_arch = "aarch64")
)))]
fn detect_neon() -> bool {
    cfg!(target_feature = "neon")
}
