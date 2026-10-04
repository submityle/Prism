//! Platform sandbox model and the capability bits derived from it (design
//! §24.5).
//!
//! Mobile, console, and Web hosts run the engine inside a sandbox that
//! restricts file-system and network reach; the upper layers should query the
//! sandbox's *capabilities* and degrade gracefully (e.g. route all writes
//! through a container directory, disable a direct-socket transport) rather
//! than discover the limits by hitting `EPERM` at runtime.
//!
//! This module models the sandbox as portable data: a [`SandboxModel`] enum
//! and the [`SandboxCapabilities`] it implies. The mapping is deterministic and
//! testable. The *conservative* defaults ([`SandboxModel::default_for`]) encode
//! only what is guaranteed by the platform (Android and Web always sandbox the
//! process; desktop defaults to [`SandboxModel::None`] unless a store/hardened
//! configuration says otherwise). A live probe of the actual sandbox profile
//! (entitlements, seccomp filter, `AppContainer` SID) requires per-OS syscalls
//! that are not wired yet.

use crate::platform::Os;

/// The platform sandbox the process runs inside.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum SandboxModel {
    /// No OS sandbox (typical desktop executable).
    None,
    /// Apple App Sandbox (Mac App Store / iOS container).
    AppleAppSandbox,
    /// Apple Hardened Runtime (notarized desktop app, no App Sandbox).
    AppleHardenedRuntime,
    /// Windows `AppContainer` (Store / `UWP` packaged app).
    WindowsAppContainer,
    /// A Linux seccomp-bpf syscall filter.
    LinuxSeccomp,
    /// A Linux user/mount namespace (Flatpak/Snap/container).
    LinuxNamespace,
    /// The Android application sandbox (per-UID, `SELinux`-confined).
    AndroidAppSandbox,
    /// A Web browser's origin/process sandbox (`WebAssembly` host).
    WebBrowser,
    /// Sandbox model not yet determined.
    #[default]
    Unknown,
}

impl SandboxModel {
    /// The *guaranteed* sandbox model for `os` with no further probing.
    ///
    /// Conservative: it only commits to a sandbox the platform always imposes
    /// (Android, Web). Desktop OSes return [`SandboxModel::None`] because a bare
    /// executable is unsandboxed by default; a Store/hardened configuration is
    /// reported by a real probe, not assumed here.
    #[must_use]
    pub const fn default_for(os: Os) -> Self {
        match os {
            Os::Android => SandboxModel::AndroidAppSandbox,
            Os::Web => SandboxModel::WebBrowser,
            Os::Windows | Os::Apple | Os::Linux => SandboxModel::None,
            Os::Unknown => SandboxModel::Unknown,
        }
    }

    /// Whether this model confines the process at all.
    #[must_use]
    pub const fn is_sandboxed(self) -> bool {
        !matches!(self, SandboxModel::None | SandboxModel::Unknown)
    }

    /// The capabilities this sandbox model grants the engine.
    #[must_use]
    pub const fn capabilities(self) -> SandboxCapabilities {
        use Access::{Container, Full, None as NoAccess, Unknown as UnknownAccess};
        match self {
            // Unsandboxed desktop; and Apple Hardened Runtime, which does not
            // confine the file system or network (its restriction is on
            // unsigned dynamic code, modeled by the signing policy, not here).
            SandboxModel::None | SandboxModel::AppleHardenedRuntime => SandboxCapabilities {
                filesystem: Full,
                network: Full,
                can_spawn_process: true,
            },
            // Apple App Sandbox / iOS: file access confined to the app
            // container (plus user-granted picks); network allowed; no child
            // processes.
            SandboxModel::AppleAppSandbox => SandboxCapabilities {
                filesystem: Container,
                network: Full,
                can_spawn_process: false,
            },
            // Flatpak/Snap/container namespaces: FS confined to the sandbox
            // mount, network typically allowed, process spawning allowed inside
            // the namespace.
            SandboxModel::LinuxNamespace => SandboxCapabilities {
                filesystem: Container,
                network: Full,
                can_spawn_process: true,
            },
            // Container-confined storage with permission-gated network and no
            // arbitrary child processes: Windows AppContainer (package
            // container, capability-gated network) and the Android app sandbox
            // (per-app private storage, permission-gated network).
            SandboxModel::WindowsAppContainer | SandboxModel::AndroidAppSandbox => {
                SandboxCapabilities {
                    filesystem: Container,
                    network: UnknownAccess,
                    can_spawn_process: false,
                }
            }
            // Web: no direct file system, network only via fetch/XHR to allowed
            // origins, no processes.
            SandboxModel::WebBrowser => SandboxCapabilities {
                filesystem: NoAccess,
                network: UnknownAccess,
                can_spawn_process: false,
            },
            // seccomp filters syscalls but not paths, so FS/network reach is
            // filter-dependent and treated as unknown with no new processes;
            // an undetermined model is equally conservative.
            SandboxModel::LinuxSeccomp | SandboxModel::Unknown => SandboxCapabilities {
                filesystem: UnknownAccess,
                network: UnknownAccess,
                can_spawn_process: false,
            },
        }
    }
}

/// The reach the engine has to a class of resource under a sandbox.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum Access {
    /// Full, unrestricted access.
    Full,
    /// Access confined to a sandbox container / allow-list.
    Container,
    /// No access at all.
    None,
    /// Access level not yet determined.
    #[default]
    Unknown,
}

impl Access {
    /// Whether any access at all is available (conservatively `false` for
    /// [`Access::Unknown`]).
    #[must_use]
    pub const fn is_available(self) -> bool {
        matches!(self, Access::Full | Access::Container)
    }
}

/// The capability bits the upper layers consult before touching the file
/// system, network, or spawning processes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SandboxCapabilities {
    /// File-system reach.
    pub filesystem: Access,
    /// Network reach.
    pub network: Access,
    /// Whether the process may spawn child processes.
    pub can_spawn_process: bool,
}
