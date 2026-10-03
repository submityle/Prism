//! The development-only authentication and command-whitelist boundary.
//!
//! The remote surface is a liability in a shipping game, so it is gated twice.
//! First, [`remote_authoring_enabled`] reports whether the build even permits
//! remote authoring; it follows `debug_assertions`, so a release build reports
//! `false` and every command is refused. Second, within an enabled build an
//! [`Authenticator`] checks a presented [`AccessToken`] against the expected
//! one and then runs the command through the shared
//! [`CapabilitySet`](crate::command::CapabilitySet) allow-list. Token
//! comparison folds over the whole token so it does not short-circuit on the
//! first differing byte.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the security boundary of design section 38: authentication plus
//! command whitelist, enabled only in development builds. It reuses the
//! [`CapabilitySet`](crate::command::CapabilitySet) and
//! [`CommandError`](crate::command::CommandError) contracts so one allow-list
//! governs both direct validation and authenticated authorization.

use crate::command::{AuthoringCommand, CapabilitySet, CommandError};

/// Length in bytes of an [`AccessToken`].
pub const TOKEN_LEN: usize = 32;

/// Returns `true` when the current build permits remote authoring.
///
/// This tracks `debug_assertions`: development builds enable the surface while
/// release builds compile with it reporting `false`, which causes every
/// authorization to fail closed.
#[must_use]
pub const fn remote_authoring_enabled() -> bool {
    cfg!(debug_assertions)
}

/// A fixed-length shared secret presented by a remote tool.
#[derive(Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct AccessToken {
    bytes: [u8; TOKEN_LEN],
}

impl core::fmt::Debug for AccessToken {
    // Redacts the secret bytes so a token never leaks through logging.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("AccessToken").finish_non_exhaustive()
    }
}

impl AccessToken {
    /// Wraps raw `bytes` as a token.
    #[must_use]
    pub const fn new(bytes: [u8; TOKEN_LEN]) -> Self {
        Self { bytes }
    }

    /// Borrows the raw token bytes.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; TOKEN_LEN] {
        &self.bytes
    }

    /// Compares two tokens without short-circuiting on the first difference.
    #[must_use]
    pub fn matches(&self, other: &AccessToken) -> bool {
        let mut diff: u8 = 0;
        for (a, b) in self.bytes.iter().zip(other.bytes.iter()) {
            diff |= a ^ b;
        }
        diff == 0
    }
}

/// Why an authorization attempt failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum AuthError {
    /// Remote authoring is disabled for this build.
    RemoteDisabled,
    /// The presented token did not match the expected one.
    InvalidToken,
    /// The token matched but the command failed the allow-list or range check.
    Rejected(CommandError),
}

/// Guards the remote surface with a build gate, a token, and an allow-list.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Authenticator {
    enabled: bool,
    expected: AccessToken,
    capabilities: CapabilitySet,
}

impl Authenticator {
    /// Creates an authenticator whose enabled state follows the build gate.
    ///
    /// In a release build this is disabled and refuses every command; in a
    /// development build it enforces the token and allow-list.
    #[must_use]
    pub fn new(expected: AccessToken, capabilities: CapabilitySet) -> Self {
        Self {
            enabled: remote_authoring_enabled(),
            expected,
            capabilities,
        }
    }

    /// Creates an authenticator with an explicit `enabled` gate.
    ///
    /// This exists so the disabled (release-like) path can be exercised
    /// independently of the build profile.
    #[must_use]
    pub fn with_gate(
        enabled: bool,
        expected: AccessToken,
        capabilities: CapabilitySet,
    ) -> Self {
        Self {
            enabled,
            expected,
            capabilities,
        }
    }

    /// Returns whether this authenticator is currently enabled.
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Borrows the allow-list this authenticator enforces.
    #[must_use]
    pub fn capabilities(&self) -> &CapabilitySet {
        &self.capabilities
    }

    /// Verifies that remote authoring is enabled and `presented` is valid.
    ///
    /// # Errors
    ///
    /// Returns [`AuthError::RemoteDisabled`] when the build gate is off, or
    /// [`AuthError::InvalidToken`] when the token does not match.
    pub fn authenticate(&self, presented: &AccessToken) -> Result<(), AuthError> {
        if !self.enabled {
            return Err(AuthError::RemoteDisabled);
        }
        if !self.expected.matches(presented) {
            return Err(AuthError::InvalidToken);
        }
        Ok(())
    }

    /// Authenticates `presented` and then checks `command` against the
    /// allow-list and value ranges.
    ///
    /// # Errors
    ///
    /// Returns [`AuthError::RemoteDisabled`] or [`AuthError::InvalidToken`]
    /// from authentication, or [`AuthError::Rejected`] wrapping the
    /// [`CommandError`] from validation.
    pub fn authorize(
        &self,
        presented: &AccessToken,
        command: &AuthoringCommand,
    ) -> Result<(), AuthError> {
        self.authenticate(presented)?;
        command
            .validate(&self.capabilities)
            .map_err(AuthError::Rejected)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::command::{CommandKind, EventId};

    fn token(seed: u8) -> AccessToken {
        let mut bytes = [0u8; TOKEN_LEN];
        for (i, b) in bytes.iter_mut().enumerate() {
            *b = seed ^ (i as u8);
        }
        AccessToken::new(bytes)
    }

    #[test]
    fn disabled_build_refuses_everything() {
        let auth = Authenticator::with_gate(false, token(1), CapabilitySet::full());
        let cmd = AuthoringCommand::TriggerEvent {
            event: EventId(1),
        };
        assert_eq!(
            auth.authorize(&token(1), &cmd),
            Err(AuthError::RemoteDisabled)
        );
    }

    #[test]
    fn wrong_token_is_rejected() {
        let auth = Authenticator::with_gate(true, token(1), CapabilitySet::full());
        assert_eq!(auth.authenticate(&token(2)), Err(AuthError::InvalidToken));
    }

    #[test]
    fn correct_token_authenticates() {
        let auth = Authenticator::with_gate(true, token(9), CapabilitySet::full());
        assert_eq!(auth.authenticate(&token(9)), Ok(()));
    }

    #[test]
    fn authorized_command_passes() {
        let auth = Authenticator::with_gate(true, token(3), CapabilitySet::full());
        let cmd = AuthoringCommand::TriggerEvent {
            event: EventId(7),
        };
        assert_eq!(auth.authorize(&token(3), &cmd), Ok(()));
    }

    #[test]
    fn command_outside_allow_list_is_rejected() {
        let caps = CapabilitySet {
            writes_enabled: true,
            trigger_event: true,
            ..CapabilitySet::denied()
        };
        let auth = Authenticator::with_gate(true, token(4), caps);
        let blocked = AuthoringCommand::SwapSnapshot {
            snapshot: crate::command::SnapshotId(1),
        };
        assert_eq!(
            auth.authorize(&token(4), &blocked),
            Err(AuthError::Rejected(CommandError::NotPermitted(
                CommandKind::SwapSnapshot
            )))
        );
    }

    #[test]
    fn token_matches_is_order_independent_result() {
        let a = token(5);
        let b = token(5);
        let c = token(6);
        assert!(a.matches(&b));
        assert!(!a.matches(&c));
    }

    #[test]
    fn default_authenticator_follows_build_gate() {
        let auth = Authenticator::new(token(1), CapabilitySet::full());
        assert_eq!(auth.is_enabled(), remote_authoring_enabled());
    }
}
