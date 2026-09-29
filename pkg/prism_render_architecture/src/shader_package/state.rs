//! Lifecycle state-machine transitions for a shader package.
//!
//! The build system moves a package through a small, checked lifecycle:
//!
//! ```text
//! Missing --> Validating --> Ready
//!                  \-------> Rejected
//! ```
//!
//! Terminal states (`Ready`, `Rejected`) may re-enter `Validating` when the
//! source is rebuilt or a rejected package is re-submitted after a fix. Every
//! other transition — including no-op self transitions — is rejected so callers
//! never silently skip validation.

use super::PackageState;

/// Why a requested [`PackageState`] transition was refused.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct TransitionError {
    /// State the package was in.
    pub from: PackageState,
    /// State the caller attempted to move to.
    pub to: PackageState,
}

impl PackageState {
    /// Returns `true` when the package is in a terminal (validated) state.
    #[must_use]
    pub const fn is_terminal(self) -> bool {
        matches!(self, Self::Ready | Self::Rejected)
    }

    /// Reports whether moving from `self` to `next` is a legal transition.
    #[must_use]
    pub const fn can_transition_to(self, next: Self) -> bool {
        matches!(
            (self, next),
            (Self::Missing, Self::Validating)
                | (Self::Validating, Self::Ready)
                | (Self::Validating, Self::Rejected)
                | (Self::Ready, Self::Validating)
                | (Self::Rejected, Self::Validating)
        )
    }

    /// Attempts a transition, returning the new state or a [`TransitionError`].
    ///
    /// # Errors
    ///
    /// Returns [`TransitionError`] when `next` is not reachable from `self`.
    pub const fn try_transition(self, next: Self) -> Result<Self, TransitionError> {
        if self.can_transition_to(next) {
            Ok(next)
        } else {
            Err(TransitionError {
                from: self,
                to: next,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn happy_path_reaches_ready() {
        let s = PackageState::Missing;
        let s = s.try_transition(PackageState::Validating).unwrap();
        let s = s.try_transition(PackageState::Ready).unwrap();
        assert_eq!(s, PackageState::Ready);
        assert!(s.is_terminal());
    }

    #[test]
    fn validating_can_reject() {
        let s = PackageState::Validating
            .try_transition(PackageState::Rejected)
            .unwrap();
        assert_eq!(s, PackageState::Rejected);
        assert!(s.is_terminal());
    }

    #[test]
    fn terminal_states_can_revalidate() {
        assert!(PackageState::Ready.can_transition_to(PackageState::Validating));
        assert!(PackageState::Rejected.can_transition_to(PackageState::Validating));
    }

    #[test]
    fn skipping_validation_is_illegal() {
        let err = PackageState::Missing
            .try_transition(PackageState::Ready)
            .unwrap_err();
        assert_eq!(
            err,
            TransitionError {
                from: PackageState::Missing,
                to: PackageState::Ready,
            }
        );
    }

    #[test]
    fn self_transitions_are_illegal() {
        for state in [
            PackageState::Missing,
            PackageState::Validating,
            PackageState::Ready,
            PackageState::Rejected,
        ] {
            assert!(!state.can_transition_to(state));
            assert!(state.try_transition(state).is_err());
        }
    }

    #[test]
    fn terminal_states_cannot_jump_between_each_other() {
        assert!(!PackageState::Ready.can_transition_to(PackageState::Rejected));
        assert!(!PackageState::Rejected.can_transition_to(PackageState::Ready));
    }
}
