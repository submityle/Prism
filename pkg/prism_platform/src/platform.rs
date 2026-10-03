//! Platform description and capability flags.

use crate::cpu::CpuInfo;

/// Coarse operating-system family.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Os {
    /// Microsoft Windows.
    Windows,
    /// Apple macOS / iOS.
    Apple,
    /// Linux and other POSIX desktops.
    Linux,
    /// Android.
    Android,
    /// WebAssembly / browser.
    Web,
    /// Unknown or bare-metal target.
    Unknown,
}

impl Os {
    /// The OS family this binary was compiled for.
    pub const fn current() -> Os {
        if cfg!(target_os = "windows") {
            Os::Windows
        } else if cfg!(any(target_os = "macos", target_os = "ios")) {
            Os::Apple
        } else if cfg!(target_os = "android") {
            Os::Android
        } else if cfg!(target_os = "linux") {
            Os::Linux
        } else if cfg!(target_arch = "wasm32") {
            Os::Web
        } else {
            Os::Unknown
        }
    }
}

/// Static capability flags describing what platform services are compiled in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlatformCaps {
    /// `std` is available (OS services like threads/files can be built).
    pub has_std: bool,
    /// A monotonic clock is available.
    pub has_monotonic_clock: bool,
}

impl PlatformCaps {
    /// Capabilities for the current build.
    pub const fn current() -> Self {
        Self {
            has_std: cfg!(feature = "std"),
            has_monotonic_clock: cfg!(feature = "std"),
        }
    }
}

/// A resolved description of the running platform.
#[derive(Clone, Debug)]
pub struct Platform {
    /// Operating-system family.
    pub os: Os,
    /// CPU capability snapshot.
    pub cpu: CpuInfo,
    /// Compiled-in capability flags.
    pub caps: PlatformCaps,
}

impl Platform {
    /// Probe and describe the current platform.
    pub fn current() -> Self {
        Self {
            os: Os::current(),
            cpu: CpuInfo::detect(),
            caps: PlatformCaps::current(),
        }
    }
}
