//! Exploit-mitigation knobs and the per-platform *default-policy* baseline
//! (design §24.5).
//!
//! AAA engines ship a hardened release and want a single, portable description
//! of which exploit mitigations are expected to be in force (ASLR, DEP/NX,
//! stack canaries, CFG, position-independent code, pointer authentication, …).
//! This module models each mitigation as a stable enum plus a tri-state
//! [`MitigationStatus`], and exposes the OS *release-build default policy* as a
//! deterministic table.
//!
//! Honesty: [`Mitigation::platform_default`] reports what a correctly
//! configured hardened release build on that OS is *expected* to enable by
//! default. It is a documented policy reference, **not** a live read of the
//! running process. A real per-process probe (reading the PE/ELF/Mach-O load
//! configuration, `/proc/self`, `mach` task flags, …) requires per-OS syscalls
//! that are not wired yet; [`super::SecurityPosture::is_probed`] stays `false`
//! until it is, so callers never mistake the baseline for a verified read.

use crate::platform::Os;

/// A single exploit-mitigation technique the platform may enforce.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Mitigation {
    /// Address-space layout randomization: randomized base of image, stack,
    /// heap, and shared libraries.
    Aslr,
    /// Data execution prevention / no-execute pages (W^X): writable memory is
    /// not executable.
    DepNx,
    /// Position-independent executable: the main image itself is relocatable,
    /// so ASLR covers it too.
    Pie,
    /// Stack smashing protection ("stack canary" / `/GS`): a guard value
    /// detects linear stack-buffer overflows before return.
    StackCanary,
    /// Control-flow integrity for indirect calls (Windows Control Flow Guard,
    /// Intel CET / ARM BTI forward-edge).
    ControlFlowGuard,
    /// Backward-edge control-flow integrity: a shadow stack (Intel CET) or ARM
    /// pointer authentication of return addresses.
    ShadowStack,
    /// ARM pointer authentication (`PAC`) of code/data pointers.
    PointerAuth,
    /// Full RELRO: the GOT is mapped read-only after relocation (ELF).
    Relro,
    /// `_FORTIFY_SOURCE`: compile-time bounds-checked libc wrappers.
    FortifySource,
    /// The OS enforces that only validly code-signed executables run.
    CodeSigningEnforced,
}

impl Mitigation {
    /// Every modeled mitigation, in a stable order.
    pub const ALL: [Mitigation; 10] = [
        Mitigation::Aslr,
        Mitigation::DepNx,
        Mitigation::Pie,
        Mitigation::StackCanary,
        Mitigation::ControlFlowGuard,
        Mitigation::ShadowStack,
        Mitigation::PointerAuth,
        Mitigation::Relro,
        Mitigation::FortifySource,
        Mitigation::CodeSigningEnforced,
    ];

    /// A short, stable identifier (useful for logs and telemetry keys).
    #[must_use]
    pub const fn id(self) -> &'static str {
        match self {
            Mitigation::Aslr => "aslr",
            Mitigation::DepNx => "dep_nx",
            Mitigation::Pie => "pie",
            Mitigation::StackCanary => "stack_canary",
            Mitigation::ControlFlowGuard => "control_flow_guard",
            Mitigation::ShadowStack => "shadow_stack",
            Mitigation::PointerAuth => "pointer_auth",
            Mitigation::Relro => "relro",
            Mitigation::FortifySource => "fortify_source",
            Mitigation::CodeSigningEnforced => "code_signing_enforced",
        }
    }

    /// The *expected* status of this mitigation for a hardened **release**
    /// build on `os`, per that platform's documented defaults.
    ///
    /// This is a policy reference, not a live probe (see the module docs). It
    /// returns [`MitigationStatus::Enforced`] only where the platform's own
    /// defaults reliably enable the mitigation for shipping native builds,
    /// [`MitigationStatus::NotApplicable`] where the concept does not apply to
    /// the target, and [`MitigationStatus::Unknown`] where it depends on
    /// opt-in build flags or hardware that cannot be assumed.
    #[must_use]
    pub const fn platform_default(self, os: Os) -> MitigationStatus {
        use MitigationStatus::{Enforced, NotApplicable, Unknown};
        match self {
            // ASLR, DEP/NX (W^X), and the stack canary are all on by default on
            // every modern native OS (kernel for ASLR/NX, the toolchain default
            // — MSVC `/GS`, clang/gcc `-fstack-protector-strong` — for the
            // canary) and are not applicable to the Web/unknown targets.
            Mitigation::Aslr | Mitigation::DepNx | Mitigation::StackCanary => match os {
                Os::Windows | Os::Apple | Os::Linux | Os::Android => Enforced,
                Os::Web | Os::Unknown => NotApplicable,
            },

            // PIE: mandatory on Apple and Android, default on modern Linux
            // distros. Windows uses image ASLR rather than ELF-style PIE.
            Mitigation::Pie => match os {
                Os::Apple | Os::Android | Os::Linux => Enforced,
                Os::Windows | Os::Web | Os::Unknown => NotApplicable,
            },

            // Control-flow integrity (forward-edge CFG/CET/BTI and backward-edge
            // shadow stack / return-address PAC) plus pointer authentication all
            // depend on opt-in build flags (`/guard:cf`, `-mbranch-protection`)
            // and/or hardware, so none can be assumed on a native target.
            Mitigation::ControlFlowGuard
            | Mitigation::ShadowStack
            | Mitigation::PointerAuth => match os {
                Os::Web | Os::Unknown => NotApplicable,
                Os::Windows | Os::Apple | Os::Linux | Os::Android => Unknown,
            },

            // Full RELRO and _FORTIFY_SOURCE are ELF/libc concepts and distro
            // defaults on Linux/Android; they do not apply elsewhere.
            Mitigation::Relro | Mitigation::FortifySource => match os {
                Os::Linux | Os::Android => Enforced,
                Os::Windows | Os::Apple | Os::Web | Os::Unknown => NotApplicable,
            },

            // Mandatory code-signing enforcement: Apple (Gatekeeper / mandatory
            // on Apple silicon) and Android (package signing). Opt-in on
            // Windows/Linux, not applicable to the Web.
            Mitigation::CodeSigningEnforced => match os {
                Os::Apple | Os::Android => Enforced,
                Os::Web => NotApplicable,
                Os::Windows | Os::Linux | Os::Unknown => Unknown,
            },
        }
    }
}

/// The tri-state (plus not-applicable) status of a mitigation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum MitigationStatus {
    /// The mitigation is in force.
    Enforced,
    /// The mitigation is definitively off.
    NotEnforced,
    /// The mitigation does not apply to this platform/target.
    NotApplicable,
    /// Status is unknown — not yet probed, or build-flag/hardware dependent.
    #[default]
    Unknown,
}

impl MitigationStatus {
    /// Whether the mitigation is known to be active.
    #[must_use]
    pub const fn is_enforced(self) -> bool {
        matches!(self, MitigationStatus::Enforced)
    }

    /// Whether this status represents a *weakness* the engine should warn
    /// about: a mitigation that is definitively off on a platform where it
    /// otherwise applies. `Unknown` and `NotApplicable` are not weaknesses.
    #[must_use]
    pub const fn is_weakness(self) -> bool {
        matches!(self, MitigationStatus::NotEnforced)
    }
}

impl core::fmt::Display for MitigationStatus {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let s = match self {
            MitigationStatus::Enforced => "enforced",
            MitigationStatus::NotEnforced => "not-enforced",
            MitigationStatus::NotApplicable => "n/a",
            MitigationStatus::Unknown => "unknown",
        };
        f.write_str(s)
    }
}
