//! Fallback crash backend for targets without a real signal mechanism
//! (Web / Android / iOS and other non-desktop hosts).
//!
//! Every install/uninstall honestly returns [`CrashError::Unsupported`] and the
//! [`crate::PlatformCaps`] crash bit is cleared, so upper layers degrade
//! gracefully instead of crashing (design doc §23 risk #6). The in-process
//! [`crate::crash::mock`] backend (feature `mock`) still works here, since it
//! never touches the `OS`.

use super::CrashError;

/// Installing real handlers is unsupported on this target.
pub(super) fn install() -> core::result::Result<(), CrashError> {
    Err(CrashError::Unsupported)
}

/// Uninstalling real handlers is unsupported on this target.
pub(super) fn uninstall() -> core::result::Result<(), CrashError> {
    Err(CrashError::Unsupported)
}

/// Real handlers are never installed on this target.
pub(super) fn is_installed() -> bool {
    false
}
