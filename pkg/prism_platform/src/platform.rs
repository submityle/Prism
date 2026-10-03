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
    /// Child processes can be spawned (design doc §11 子进程).
    ///
    /// Requires `std` and a target with a process model. `true` on the desktop
    /// OSes; `false` on `wasm` targets, which have no `fork`/`exec` facility.
    /// When `false`, [`crate::process::Command`] (if compiled) still exists but
    /// every spawn attempt fails at the OS layer.
    pub has_subprocess: bool,
    /// A real system wall clock is available (design doc §7 墙钟, §11).
    ///
    /// Requires `std`; backs [`crate::wallclock`]. Distinct from
    /// [`PlatformCaps::has_monotonic_clock`]: the wall clock is UTC/calendar
    /// time (can step), the monotonic clock is for measuring elapsed time.
    pub has_wall_clock: bool,
    /// Real crash capture is available (design doc §16, §22 M6): the
    /// async-signal-safe `POSIX` signal backend is compiled in.
    ///
    /// `true` on desktop `POSIX` hosts (Linux and macOS); `false` on Web,
    /// Android, iOS, and other targets, where [`crate::crash::install`]
    /// returns [`crate::crash::CrashError::Unsupported`] and callers should
    /// degrade gracefully. The in-process [`crate::crash::mock`] backend is
    /// independent of this bit.
    pub has_crash_capture: bool,
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
            has_subprocess: cfg!(feature = "std") && cfg!(not(target_family = "wasm")),
            has_wall_clock: cfg!(feature = "std"),
            has_crash_capture: crate::crash::SUPPORTED,
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
