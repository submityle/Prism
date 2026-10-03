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
    /// Zero-copy memory-mapped files are available (the real `mmap` /
    /// `MapViewOfFile` primitive, not the read-into-buffer fallback).
    ///
    /// Requires `std` and a Unix or Windows target. When `false`, [`crate::fs::mmap`]
    /// still works via an honest fallback that copies the range into a heap
    /// buffer — only the zero-copy property is lost.
    pub has_mmap: bool,
    /// A native OS file-watcher backend is compiled in (`kqueue` on macOS/BSD).
    ///
    /// Requires the `watch` feature. When `false` but the `watch` feature is
    /// on, [`crate::fs::watch`] still works through the portable `stat`-based
    /// polling backend; this bit specifically reflects a *native* backend.
    pub has_native_file_watch: bool,
    /// Dynamic libraries can be loaded and **unloaded** at runtime
    /// (`dlopen`/`dlclose` or `LoadLibrary`/`FreeLibrary`).
    ///
    /// Requires the `dynlib` feature and a Unix or Windows target. When
    /// `false`, [`crate::dynlib`] (if compiled) reports
    /// [`crate::dynlib::DynlibError::Unsupported`]; some platforms (notably the
    /// Web target, design doc §10) have no runtime loader at all.
    pub has_dynlib_unload: bool,
}

impl PlatformCaps {
    /// Capabilities for the current build.
    pub const fn current() -> Self {
        Self {
            has_std: cfg!(feature = "std"),
            has_monotonic_clock: cfg!(feature = "std"),
            has_mmap: cfg!(feature = "std") && cfg!(any(unix, windows)),
            has_native_file_watch: cfg!(feature = "watch")
                && cfg!(any(
                    target_vendor = "apple",
                    target_os = "freebsd",
                    target_os = "netbsd",
                    target_os = "openbsd",
                    target_os = "dragonfly"
                )),
            has_dynlib_unload: cfg!(feature = "dynlib") && cfg!(any(unix, windows)),
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
