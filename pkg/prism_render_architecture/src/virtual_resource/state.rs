//! Residency lifecycle state machine and transition validation.
//!
//! Every virtual resource climbs a fixed lifecycle:
//! `Missing -> Requested -> Uploading -> Resident -> Retiring`. This module
//! encodes exactly which transitions are legal and rejects the rest, so callers
//! that drive resource state cannot skip an upload, resurrect a dropped page out
//! of order, or otherwise corrupt the machine. Validation is pure integer / enum
//! matching, so it is exact and deterministic and holds no `GPU` handle: the
//! actual upload that carries a resource from `Requested` to `Resident` is
//! issued by the backend, pending the `GPU` backend.

use super::ResidencyState;

/// A rejected [`ResidencyState`] transition.
///
/// Returned by [`ResidencyState::try_transition`] when the requested move is not
/// part of the legal lifecycle. Carries both endpoints so the caller can log or
/// assert on the exact illegal edge.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct IllegalTransition {
    /// The state the resource was in.
    pub from: ResidencyState,
    /// The state the caller tried to move to.
    pub to: ResidencyState,
}

impl ResidencyState {
    /// Whether the resource currently occupies physical storage.
    #[must_use]
    pub const fn is_resident(self) -> bool {
        matches!(self, Self::Resident)
    }

    /// Whether an upload is outstanding (`Requested` or `Uploading`).
    ///
    /// Pending resources are the scheduler's in-flight work: something asked for
    /// them and the backend has not yet confirmed residency.
    #[must_use]
    pub const fn is_pending(self) -> bool {
        matches!(self, Self::Requested | Self::Uploading)
    }

    /// Whether the resource is being wound down (`Retiring`).
    #[must_use]
    pub const fn is_retiring(self) -> bool {
        matches!(self, Self::Retiring)
    }

    /// Whether the resource is fully dropped and untracked physically.
    #[must_use]
    pub const fn is_missing(self) -> bool {
        matches!(self, Self::Missing)
    }

    /// Whether moving from `self` to `next` is a legal lifecycle transition.
    ///
    /// The legal edges are the forward chain plus the cancel / revive / retire
    /// edges that keep the machine closed:
    ///
    /// * `Missing -> Requested`
    /// * `Requested -> Uploading` and `Requested -> Missing` (cancel)
    /// * `Uploading -> Resident` and `Uploading -> Missing` (upload aborted)
    /// * `Resident -> Retiring`
    /// * `Retiring -> Missing` (retirement complete) and `Retiring -> Resident`
    ///   (revived before it finished retiring)
    ///
    /// Every other pair, including a same-state self-loop, is illegal.
    #[must_use]
    pub const fn can_transition_to(self, next: Self) -> bool {
        matches!(
            (self, next),
            (Self::Missing, Self::Requested)
                | (Self::Requested, Self::Uploading)
                | (Self::Requested, Self::Missing)
                | (Self::Uploading, Self::Resident)
                | (Self::Uploading, Self::Missing)
                | (Self::Resident, Self::Retiring)
                | (Self::Retiring, Self::Missing)
                | (Self::Retiring, Self::Resident)
        )
    }

    /// Returns `next` if the transition is legal, else an [`IllegalTransition`].
    ///
    /// # Errors
    ///
    /// Returns [`IllegalTransition`] when the `self -> next` edge is not part of
    /// the lifecycle defined by [`ResidencyState::can_transition_to`].
    pub const fn try_transition(self, next: Self) -> Result<Self, IllegalTransition> {
        if self.can_transition_to(next) {
            Ok(next)
        } else {
            Err(IllegalTransition {
                from: self,
                to: next,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: [ResidencyState; 5] = [
        ResidencyState::Missing,
        ResidencyState::Requested,
        ResidencyState::Uploading,
        ResidencyState::Resident,
        ResidencyState::Retiring,
    ];

    #[test]
    fn forward_chain_is_legal() {
        assert!(ResidencyState::Missing.can_transition_to(ResidencyState::Requested));
        assert!(ResidencyState::Requested.can_transition_to(ResidencyState::Uploading));
        assert!(ResidencyState::Uploading.can_transition_to(ResidencyState::Resident));
        assert!(ResidencyState::Resident.can_transition_to(ResidencyState::Retiring));
        assert!(ResidencyState::Retiring.can_transition_to(ResidencyState::Missing));
    }

    #[test]
    fn cancel_and_revive_edges_are_legal() {
        assert!(ResidencyState::Requested.can_transition_to(ResidencyState::Missing));
        assert!(ResidencyState::Uploading.can_transition_to(ResidencyState::Missing));
        assert!(ResidencyState::Retiring.can_transition_to(ResidencyState::Resident));
    }

    #[test]
    fn skipping_a_stage_is_illegal() {
        // Cannot jump straight from Missing to Resident without uploading.
        let err = ResidencyState::Missing
            .try_transition(ResidencyState::Resident)
            .unwrap_err();
        assert_eq!(err.from, ResidencyState::Missing);
        assert_eq!(err.to, ResidencyState::Resident);
        // Cannot go from Requested straight to Resident.
        assert!(!ResidencyState::Requested.can_transition_to(ResidencyState::Resident));
        // Cannot re-upload a resident resource.
        assert!(!ResidencyState::Resident.can_transition_to(ResidencyState::Uploading));
    }

    #[test]
    fn self_loops_are_illegal() {
        for state in ALL {
            assert!(
                !state.can_transition_to(state),
                "self-loop should be rejected for {state:?}"
            );
            assert!(state.try_transition(state).is_err());
        }
    }

    #[test]
    fn try_transition_reports_endpoints_on_failure() {
        let err = ResidencyState::Resident
            .try_transition(ResidencyState::Requested)
            .unwrap_err();
        assert_eq!(
            err,
            IllegalTransition {
                from: ResidencyState::Resident,
                to: ResidencyState::Requested,
            }
        );
    }

    #[test]
    fn predicates_classify_states() {
        assert!(ResidencyState::Resident.is_resident());
        assert!(ResidencyState::Requested.is_pending());
        assert!(ResidencyState::Uploading.is_pending());
        assert!(!ResidencyState::Resident.is_pending());
        assert!(ResidencyState::Retiring.is_retiring());
        assert!(ResidencyState::Missing.is_missing());
    }

    #[test]
    fn legal_edge_count_is_exactly_eight() {
        let mut legal = 0usize;
        for from in ALL {
            for to in ALL {
                if from.can_transition_to(to) {
                    legal += 1;
                }
            }
        }
        assert_eq!(legal, 8);
    }
}
