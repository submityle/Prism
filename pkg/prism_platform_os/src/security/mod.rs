//! Real-OS security-posture probe filling
//! [`prism_platform::security::SecurityPosture`] (design §24.5).
//!
//! [`prism_platform`]'s [`SecurityPosture::detect`] is a deliberately
//! conservative baseline: it reports each platform's *expected* mitigation
//! policy and a conservative sandbox model but leaves
//! [`SecurityPosture::is_probed`] `false`, because reading the mitigations that
//! are actually in force for *this* running process needs per-OS syscalls that
//! do not belong in a portable, unsafe-free kernel. This module is the real
//! probe that earns `is_probed() == true` by reading the running image.
//!
//! ## Honest boundary
//!
//! The genuine read is implemented and verified only for **Apple Silicon
//! macOS** (`aarch64`). It reads two real sources for *this* process:
//!
//! * the main executable's Mach-O header flags via `_dyld_get_image_header`,
//!   from which [`Mitigation::Pie`] (the `MH_PIE` bit) and
//!   [`Mitigation::PointerAuth`] (the `arm64e` CPU sub-type) are read; and
//! * the kernel code-signing status word via `csops(CS_OPS_STATUS)`, from which
//!   [`Mitigation::CodeSigningEnforced`] (the `CS_VALID` bit) and the
//!   [`SandboxModel::AppleHardenedRuntime`] sandbox model (the `CS_RUNTIME`
//!   bit) are read.
//!
//! What it does **not** claim: `csops` reports the kernel's code-signing status
//! bits for the process, not a trust-chain verdict, so this probe never upgrades
//! the posture to a "trusted signer" claim. The architecturally-guaranteed
//! mitigations that are not individually read back here (`ASLR`, `DEP/NX`, the
//! stack canary) keep their [`SecurityPosture::baseline`] values, which on
//! Apple Silicon are correct by construction. On every other target
//! [`probe_security`] returns [`ProbeError::Unsupported`] rather than guessing.
//!
//! [`SecurityPosture`]: prism_platform::security::SecurityPosture
//! [`SecurityPosture::detect`]: prism_platform::security::SecurityPosture::detect
//! [`SecurityPosture::baseline`]: prism_platform::security::SecurityPosture::baseline
//! [`SecurityPosture::is_probed`]: prism_platform::security::SecurityPosture::is_probed
//! [`Mitigation::Pie`]: prism_platform::security::Mitigation::Pie
//! [`Mitigation::PointerAuth`]: prism_platform::security::Mitigation::PointerAuth
//! [`Mitigation::CodeSigningEnforced`]: prism_platform::security::Mitigation::CodeSigningEnforced
//! [`SandboxModel::AppleHardenedRuntime`]: prism_platform::security::SandboxModel::AppleHardenedRuntime

use crate::error::ProbeError;
use prism_platform::security::SecurityPosture;

/// Probe the host OS for a genuine [`SecurityPosture`] whose
/// [`SecurityPosture::is_probed`] is `true`.
///
/// Verified on Apple Silicon macOS; see the module docs for the honest
/// boundary and what is (and is not) read back.
///
/// # Errors
///
/// Returns [`ProbeError::Unsupported`] on targets without a verified backend,
/// [`ProbeError::Os`] if an OS query fails (for example `csops` denied by a
/// sandbox), or [`ProbeError::Invalid`] if the running image cannot be located.
///
/// [`SecurityPosture`]: prism_platform::security::SecurityPosture
/// [`SecurityPosture::is_probed`]: prism_platform::security::SecurityPosture::is_probed
pub fn probe_security() -> Result<SecurityPosture, ProbeError> {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        macos_arm::probe()
    }
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    {
        Err(ProbeError::Unsupported)
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod macos_arm {
    use super::ProbeError;
    use core::ffi::c_void;
    use prism_platform::platform::Os;
    use prism_platform::security::{Mitigation, MitigationStatus, SandboxModel, SecurityPosture};

    // --- Mach-O header ------------------------------------------------------

    /// `MH_PIE`: the main image is a position-independent executable, so ASLR
    /// randomizes its base too.
    const MH_PIE: u32 = 0x0020_0000;
    /// Mask that strips the capability bits from a Mach-O `cpusubtype`, leaving
    /// the feature sub-type in the low 24 bits.
    const CPU_SUBTYPE_MASK: i32 = 0x00ff_ffff;
    /// `CPU_SUBTYPE_ARM64E`: the pointer-authentication-carrying arm64 variant.
    const CPU_SUBTYPE_ARM64E: i32 = 2;

    /// The 64-bit Mach-O header (`mach_header_64`). Only the `flags` and
    /// `cpusubtype` fields are read; the rest are present so the layout matches
    /// what `_dyld_get_image_header` points at.
    #[repr(C)]
    struct MachHeader64 {
        magic: u32,
        cputype: i32,
        cpusubtype: i32,
        filetype: u32,
        ncmds: u32,
        sizeofcmds: u32,
        flags: u32,
        reserved: u32,
    }

    // --- Kernel code-signing status (csops) ---------------------------------

    /// `CS_OPS_STATUS`: ask `csops` for the process's code-signing status word.
    const CS_OPS_STATUS: u32 = 0;
    /// `CS_VALID`: the process has a currently-valid code signature.
    const CS_VALID: u32 = 0x0000_0001;
    /// `CS_RUNTIME`: the process runs under the Apple Hardened Runtime.
    const CS_RUNTIME: u32 = 0x0001_0000;

    #[expect(
        unsafe_code,
        reason = "declaring the macOS csops and _dyld_get_image_header C entry points"
    )]
    unsafe extern "C" {
        /// `_dyld_get_image_header(0)` returns the Mach-O header of the main
        /// executable for the running process.
        fn _dyld_get_image_header(image_index: u32) -> *const MachHeader64;
        /// `csops` queries code-signing operations for a process; with
        /// `CS_OPS_STATUS` it writes the status word into `useraddr`.
        fn csops(
            pid: libc::pid_t,
            ops: u32,
            useraddr: *mut c_void,
            usersize: libc::size_t,
        ) -> libc::c_int;
    }

    /// Read `(flags, cpusubtype)` from this process's main Mach-O image header.
    fn main_image_header() -> Result<(u32, i32), ProbeError> {
        #[expect(
            unsafe_code,
            reason = "_dyld_get_image_header is the documented way to reach the running main image"
        )]
        // SAFETY: image index 0 is defined to be the main executable's Mach-O
        // header for the current process; `_dyld_get_image_header` takes no
        // pointers and returns a pointer the dynamic linker keeps valid for the
        // process lifetime.
        let hdr = unsafe { _dyld_get_image_header(0) };
        if hdr.is_null() {
            return Err(ProbeError::Invalid(
                "dyld returned no main image header".to_string(),
            ));
        }
        #[expect(
            unsafe_code,
            reason = "the index-0 image is a valid, process-lifetime mach_header_64 on 64-bit macOS"
        )]
        // SAFETY: `hdr` is non-null and, on 64-bit macOS, points at a fully
        // initialized `mach_header_64`; we only read the `flags` and
        // `cpusubtype` scalar fields and never retain the reference.
        let (flags, cpusubtype) = unsafe { ((*hdr).flags, (*hdr).cpusubtype) };
        Ok((flags, cpusubtype))
    }

    /// Read this process's kernel code-signing status word via `csops`.
    fn code_signing_status() -> Result<u32, ProbeError> {
        let mut status: u32 = 0;
        #[expect(
            unsafe_code,
            reason = "csops(CS_OPS_STATUS) is the kernel interface for this process's signing status"
        )]
        // SAFETY: `getpid()` is the current process, which may always query its
        // own status; `&mut status` points at a live `u32` and `usersize` is
        // exactly its size, so the kernel writes at most four initialized bytes
        // into it.
        let rc = unsafe {
            csops(
                libc::getpid(),
                CS_OPS_STATUS,
                core::ptr::from_mut(&mut status).cast::<c_void>(),
                size_of::<u32>() as libc::size_t,
            )
        };
        if rc != 0 {
            return Err(ProbeError::Os(std::io::Error::last_os_error()));
        }
        Ok(status)
    }

    /// Build a verified Apple-Silicon [`SecurityPosture`] from the running
    /// image's Mach-O flags and kernel code-signing status.
    pub fn probe() -> Result<SecurityPosture, ProbeError> {
        let (mh_flags, cpusubtype) = main_image_header()?;
        let cs_status = code_signing_status()?;

        // PIE is read directly from the Mach-O header; Rust executables on
        // macOS are always position-independent, so this reads back Enforced.
        let pie = if mh_flags & MH_PIE != 0 {
            MitigationStatus::Enforced
        } else {
            MitigationStatus::NotEnforced
        };

        // CS_VALID means the kernel accepted this process's code signature and
        // is enforcing it; Apple Silicon refuses to run unsigned code at all.
        let code_signing = if cs_status & CS_VALID != 0 {
            MitigationStatus::Enforced
        } else {
            MitigationStatus::NotEnforced
        };

        // The Hardened Runtime shows up as a code-signing status bit, not an
        // App Sandbox; absence of the bit means an ordinary unsandboxed desktop
        // process.
        let sandbox = if cs_status & CS_RUNTIME != 0 {
            SandboxModel::AppleHardenedRuntime
        } else {
            SandboxModel::None
        };

        let mut builder = SecurityPosture::builder(Os::Apple)
            .mitigation(Mitigation::Pie, pie)
            .mitigation(Mitigation::CodeSigningEnforced, code_signing)
            .sandbox(sandbox);

        // Pointer authentication is observable only on the arm64e sub-type.
        // Plain arm64 does not expose a PAC-in-use bit here, so we leave the
        // baseline Unknown rather than falsely asserting it is off.
        if cpusubtype & CPU_SUBTYPE_MASK == CPU_SUBTYPE_ARM64E {
            builder = builder.mitigation(Mitigation::PointerAuth, MitigationStatus::Enforced);
        }

        Ok(builder.mark_probed().build())
    }
}
