//! Deterministic decision quality, cadence, and utility-selection primitives.
//!
//! This module intentionally knows nothing about distance, entities, gameplay
//! actions, or a particular clock. Callers map their domain signal to the
//! integer `relevance` value and choose when to sample it with
//! [`FixedFrameCadence`]. Integer scores and explicit tie-break rules keep the
//! result reproducible across machines and execution orders.

use core::fmt;

/// The four quality levels understood by a decision scheduler.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum DecisionLod {
    /// Full-rate, highest-detail decision evaluation.
    Near,
    /// Reduced-rate decision evaluation for relevant but less important work.
    Far,
    /// Visibility-independent low-rate evaluation for work outside the view.
    Offscreen,
    /// Inactive evaluation level for explicitly dormant work.
    Dormant,
}

/// Domain-independent visibility supplied to a LOD controller.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum DecisionVisibility {
    /// The subject is eligible for relevance-based Near/Far selection.
    Visible,
    /// The subject is not visible and should use the Offscreen level.
    Offscreen,
    /// The subject is explicitly inactive and should use the Dormant level.
    Dormant,
}

/// A sampled input to [`DecisionLodController`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecisionObservation {
    relevance: u32,
    visibility: DecisionVisibility,
}

impl DecisionObservation {
    /// Creates an observation from a caller-defined relevance value.
    ///
    /// Larger values mean more relevant work. The value has no prescribed
    /// unit, so a caller may use a quantized importance, screen-space error,
    /// or any other deterministic domain metric.
    pub const fn new(relevance: u32, visibility: DecisionVisibility) -> Self {
        Self {
            relevance,
            visibility,
        }
    }

    /// Returns the caller-provided relevance value.
    pub const fn relevance(self) -> u32 {
        self.relevance
    }

    /// Returns the sampled visibility classification.
    pub const fn visibility(self) -> DecisionVisibility {
        self.visibility
    }
}

/// Thresholds for relevance-based LOD transitions.
///
/// Relevance is ordered from high to low. The required ordering is
/// `near_enter >= near_exit >= far_enter >= far_exit`. Enter thresholds make
/// promotion harder than staying in a level, which supplies hysteresis while
/// preserving a single deterministic classification for every input.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LodThresholds {
    near_enter: u32,
    near_exit: u32,
    far_enter: u32,
    far_exit: u32,
}

impl LodThresholds {
    /// Creates validated Near/Far thresholds.
    pub const fn new(
        near_enter: u32,
        near_exit: u32,
        far_enter: u32,
        far_exit: u32,
    ) -> Result<Self, LodThresholdError> {
        if near_enter < near_exit {
            return Err(LodThresholdError::NearEnterBelowExit {
                near_enter,
                near_exit,
            });
        }
        if near_exit < far_enter {
            return Err(LodThresholdError::NearExitBelowFarEnter {
                near_exit,
                far_enter,
            });
        }
        if far_enter < far_exit {
            return Err(LodThresholdError::FarEnterBelowExit {
                far_enter,
                far_exit,
            });
        }
        Ok(Self {
            near_enter,
            near_exit,
            far_enter,
            far_exit,
        })
    }

    /// Relevance required to promote into Near from another relevance level.
    pub const fn near_enter(self) -> u32 {
        self.near_enter
    }

    /// Relevance below which a stable Near level demotes to Far.
    pub const fn near_exit(self) -> u32 {
        self.near_exit
    }

    /// Relevance required to promote into Far from Offscreen or Dormant.
    pub const fn far_enter(self) -> u32 {
        self.far_enter
    }

    /// Relevance below which a stable Far level demotes to Offscreen.
    pub const fn far_exit(self) -> u32 {
        self.far_exit
    }
}

/// Invalid ordering for [`LodThresholds`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LodThresholdError {
    /// `near_enter` was lower than `near_exit`.
    NearEnterBelowExit {
        /// Relevance required to enter Near.
        near_enter: u32,
        /// Relevance required to remain in Near.
        near_exit: u32,
    },
    /// `near_exit` was lower than `far_enter`.
    NearExitBelowFarEnter {
        /// Relevance required to remain in Near.
        near_exit: u32,
        /// Relevance required to enter Far.
        far_enter: u32,
    },
    /// `far_enter` was lower than `far_exit`.
    FarEnterBelowExit {
        /// Relevance required to enter Far.
        far_enter: u32,
        /// Relevance required to remain in Far.
        far_exit: u32,
    },
}

impl fmt::Display for LodThresholdError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NearEnterBelowExit {
                near_enter,
                near_exit,
            } => write!(
                formatter,
                "near_enter ({near_enter}) must be at least near_exit ({near_exit})"
            ),
            Self::NearExitBelowFarEnter {
                near_exit,
                far_enter,
            } => write!(
                formatter,
                "near_exit ({near_exit}) must be at least far_enter ({far_enter})"
            ),
            Self::FarEnterBelowExit {
                far_enter,
                far_exit,
            } => write!(
                formatter,
                "far_enter ({far_enter}) must be at least far_exit ({far_exit})"
            ),
        }
    }
}

/// A deterministic, consecutive-observation LOD controller.
///
/// An observation that disagrees with the current level must remain the
/// desired level for `stability_frames` consecutive calls before the level
/// changes. The caller can combine this with [`DecisionLodCadence`] to define
/// exactly which fixed frames produce observations.
pub struct DecisionLodController {
    thresholds: LodThresholds,
    stability_frames: u32,
    current: DecisionLod,
    pending: Option<DecisionLod>,
    pending_frames: u32,
}

impl DecisionLodController {
    /// Creates a controller with an explicit initial level and stability.
    ///
    /// `stability_frames` counts consecutive observations, not wall-clock
    /// time. A value of one applies a transition at the first disagreeing
    /// observation.
    pub const fn new(
        thresholds: LodThresholds,
        stability_frames: u32,
        initial: DecisionLod,
    ) -> Result<Self, LodStabilityError> {
        if stability_frames == 0 {
            return Err(LodStabilityError::ZeroFrames);
        }
        Ok(Self {
            thresholds,
            stability_frames,
            current: initial,
            pending: None,
            pending_frames: 0,
        })
    }

    /// Returns the thresholds used by this controller.
    pub const fn thresholds(&self) -> LodThresholds {
        self.thresholds
    }

    /// Returns the number of consecutive observations required for a change.
    pub const fn stability_frames(&self) -> u32 {
        self.stability_frames
    }

    /// Returns the currently committed LOD.
    pub const fn current(&self) -> DecisionLod {
        self.current
    }

    /// Returns the transition target currently accumulating stability.
    pub const fn pending(&self) -> Option<DecisionLod> {
        self.pending
    }

    /// Returns the number of consecutive observations accumulated for the
    /// pending target, or zero when no target is pending.
    pub const fn pending_frames(&self) -> u32 {
        self.pending_frames
    }

    /// Computes the hysteresis-aware target for an observation.
    pub const fn target_for(&self, observation: DecisionObservation) -> DecisionLod {
        match observation.visibility {
            DecisionVisibility::Dormant => DecisionLod::Dormant,
            DecisionVisibility::Offscreen => DecisionLod::Offscreen,
            DecisionVisibility::Visible => match self.current {
                DecisionLod::Near => {
                    if observation.relevance >= self.thresholds.near_exit {
                        DecisionLod::Near
                    } else if observation.relevance >= self.thresholds.far_exit {
                        DecisionLod::Far
                    } else {
                        DecisionLod::Offscreen
                    }
                }
                DecisionLod::Far => {
                    if observation.relevance >= self.thresholds.near_enter {
                        DecisionLod::Near
                    } else if observation.relevance >= self.thresholds.far_exit {
                        DecisionLod::Far
                    } else {
                        DecisionLod::Offscreen
                    }
                }
                DecisionLod::Offscreen | DecisionLod::Dormant => {
                    if observation.relevance >= self.thresholds.near_enter {
                        DecisionLod::Near
                    } else if observation.relevance >= self.thresholds.far_enter {
                        DecisionLod::Far
                    } else {
                        DecisionLod::Offscreen
                    }
                }
            },
        }
    }

    /// Applies one observation and returns the resulting state summary.
    #[must_use]
    pub fn observe(&mut self, observation: DecisionObservation) -> LodUpdate {
        let previous = self.current;
        let target = self.target_for(observation);
        if target == self.current {
            self.pending = None;
            self.pending_frames = 0;
            return LodUpdate {
                previous,
                current: self.current,
                changed: false,
                pending: None,
                pending_frames: 0,
            };
        }

        if self.pending == Some(target) {
            self.pending_frames = self.pending_frames.saturating_add(1);
        } else {
            self.pending = Some(target);
            self.pending_frames = 1;
        }

        if self.pending_frames >= self.stability_frames {
            self.current = target;
            self.pending = None;
            self.pending_frames = 0;
            LodUpdate {
                previous,
                current: self.current,
                changed: true,
                pending: None,
                pending_frames: 0,
            }
        } else {
            LodUpdate {
                previous,
                current: self.current,
                changed: false,
                pending: self.pending,
                pending_frames: self.pending_frames,
            }
        }
    }

    /// Discards any in-progress transition while retaining the committed LOD.
    pub fn clear_pending(&mut self) {
        self.pending = None;
        self.pending_frames = 0;
    }
}

/// Error returned when LOD stability is configured with zero frames.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LodStabilityError {
    /// At least one observation is required for a transition.
    ZeroFrames,
}

impl fmt::Display for LodStabilityError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroFrames => formatter.write_str("stability_frames must be greater than zero"),
        }
    }
}

/// Result of one [`DecisionLodController::observe`] call.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LodUpdate {
    /// LOD before this observation.
    pub previous: DecisionLod,
    /// Committed LOD after this observation.
    pub current: DecisionLod,
    /// Whether this observation committed a transition.
    pub changed: bool,
    /// Pending target while stability is accumulating.
    pub pending: Option<DecisionLod>,
    /// Consecutive observation count for `pending`.
    pub pending_frames: u32,
}

/// Error returned when a fixed-frame cadence has no positive period.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CadenceError {
    /// A cadence period of zero cannot produce a fixed schedule.
    ZeroPeriod,
}

impl fmt::Display for CadenceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroPeriod => formatter.write_str("cadence period must be greater than zero"),
        }
    }
}

/// A fixed-frame schedule with a deterministic modulo phase.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FixedFrameCadence {
    period: u32,
    phase: u32,
}

impl FixedFrameCadence {
    /// Creates a cadence that is due when `frame % period == phase % period`.
    pub const fn new(period: u32, phase: u32) -> Result<Self, CadenceError> {
        if period == 0 {
            return Err(CadenceError::ZeroPeriod);
        }
        Ok(Self {
            period,
            phase: phase % period,
        })
    }

    /// Creates a cadence with phase zero.
    pub const fn every(period: u32) -> Result<Self, CadenceError> {
        Self::new(period, 0)
    }

    /// Returns the number of frames between due frames.
    pub const fn period(self) -> u32 {
        self.period
    }

    /// Returns the normalized modulo phase.
    pub const fn phase(self) -> u32 {
        self.phase
    }

    /// Returns whether evaluation is due at `frame`.
    pub const fn is_due(self, frame: u64) -> bool {
        frame % (self.period as u64) == self.phase as u64
    }

    /// Returns the first due frame at or after `frame`.
    ///
    /// `None` is returned only if the next frame number would overflow `u64`.
    pub const fn next_due_at_or_after(self, frame: u64) -> Option<u64> {
        let period = self.period as u64;
        let current_phase = frame % period;
        let delta = if current_phase <= self.phase as u64 {
            self.phase as u64 - current_phase
        } else {
            period - current_phase + self.phase as u64
        };
        frame.checked_add(delta)
    }
}

/// Fixed periods for each decision LOD.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecisionLodPeriods {
    near: u32,
    far: u32,
    offscreen: u32,
    dormant: u32,
}

impl DecisionLodPeriods {
    /// Creates validated periods, in frames, for all four LODs.
    pub const fn new(
        near: u32,
        far: u32,
        offscreen: u32,
        dormant: u32,
    ) -> Result<Self, CadenceError> {
        if near == 0 || far == 0 || offscreen == 0 || dormant == 0 {
            return Err(CadenceError::ZeroPeriod);
        }
        Ok(Self {
            near,
            far,
            offscreen,
            dormant,
        })
    }

    /// Returns the configured period for `lod`.
    pub const fn for_lod(self, lod: DecisionLod) -> u32 {
        match lod {
            DecisionLod::Near => self.near,
            DecisionLod::Far => self.far,
            DecisionLod::Offscreen => self.offscreen,
            DecisionLod::Dormant => self.dormant,
        }
    }
}

/// A cadence table selected by the current [`DecisionLod`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecisionLodCadence {
    near: FixedFrameCadence,
    far: FixedFrameCadence,
    offscreen: FixedFrameCadence,
    dormant: FixedFrameCadence,
}

impl DecisionLodCadence {
    /// Creates a cadence table from individual fixed schedules.
    pub const fn new(
        near: FixedFrameCadence,
        far: FixedFrameCadence,
        offscreen: FixedFrameCadence,
        dormant: FixedFrameCadence,
    ) -> Self {
        Self {
            near,
            far,
            offscreen,
            dormant,
        }
    }

    /// Creates one fixed schedule per LOD using a common phase value.
    pub fn from_periods(periods: DecisionLodPeriods, phase: u32) -> Result<Self, CadenceError> {
        Ok(Self {
            near: FixedFrameCadence::new(periods.near, phase)?,
            far: FixedFrameCadence::new(periods.far, phase)?,
            offscreen: FixedFrameCadence::new(periods.offscreen, phase)?,
            dormant: FixedFrameCadence::new(periods.dormant, phase)?,
        })
    }

    /// Returns the fixed schedule associated with `lod`.
    pub const fn for_lod(self, lod: DecisionLod) -> FixedFrameCadence {
        match lod {
            DecisionLod::Near => self.near,
            DecisionLod::Far => self.far,
            DecisionLod::Offscreen => self.offscreen,
            DecisionLod::Dormant => self.dormant,
        }
    }

    /// Returns whether the current `lod` is due at `frame`.
    pub const fn is_due(self, lod: DecisionLod, frame: u64) -> bool {
        self.for_lod(lod).is_due(frame)
    }
}

/// A scored candidate for deterministic utility selection.
///
/// `key` is a stable caller-owned identity. It is used as the first tie-break
/// after utility, so selection does not depend on storage or query order.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UtilityCandidate<T> {
    /// Stable identity used to break equal-utility ties.
    pub key: u64,
    /// Quantized utility score; larger values win.
    pub utility: i64,
    /// Project-defined candidate payload.
    pub value: T,
}

impl<T> UtilityCandidate<T> {
    /// Creates a scored candidate.
    pub const fn new(key: u64, utility: i64, value: T) -> Self {
        Self {
            key,
            utility,
            value,
        }
    }
}

/// The winning candidate's stable selection metadata.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct UtilitySelection {
    index: usize,
    key: u64,
    utility: i64,
}

impl UtilitySelection {
    /// Returns the winning candidate's index in the input slice.
    pub const fn index(self) -> usize {
        self.index
    }

    /// Returns the winning candidate's stable key.
    pub const fn key(self) -> u64 {
        self.key
    }

    /// Returns the winning candidate's utility score.
    pub const fn utility(self) -> i64 {
        self.utility
    }
}

/// Selects the highest-utility candidate without allocating or reordering.
///
/// Tie-break order is: higher utility, lower stable key, then lower source
/// index. The final rule makes duplicate keys deterministic as well, while the
/// stable key makes equal-utility candidates independent of query order when
/// keys are unique.
#[must_use]
pub fn select_utility<T>(candidates: &[UtilityCandidate<T>]) -> Option<UtilitySelection> {
    let mut best: Option<UtilitySelection> = None;
    for (index, candidate) in candidates.iter().enumerate() {
        let selection = UtilitySelection {
            index,
            key: candidate.key,
            utility: candidate.utility,
        };
        let replace = match best {
            None => true,
            Some(current) => {
                candidate.utility > current.utility
                    || (candidate.utility == current.utility
                        && (candidate.key < current.key
                            || (candidate.key == current.key && index < current.index)))
            }
        };
        if replace {
            best = Some(selection);
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    fn thresholds() -> LodThresholds {
        LodThresholds::new(80, 60, 40, 20).expect("valid thresholds")
    }

    #[test]
    fn thresholds_reject_invalid_ordering() {
        assert_eq!(
            LodThresholds::new(10, 20, 5, 1),
            Err(LodThresholdError::NearEnterBelowExit {
                near_enter: 10,
                near_exit: 20,
            })
        );
        assert_eq!(
            LodThresholds::new(20, 10, 15, 1),
            Err(LodThresholdError::NearExitBelowFarEnter {
                near_exit: 10,
                far_enter: 15,
            })
        );
        assert_eq!(
            LodThresholds::new(20, 15, 10, 11),
            Err(LodThresholdError::FarEnterBelowExit {
                far_enter: 10,
                far_exit: 11,
            })
        );
    }

    #[test]
    fn stability_blocks_flapping_and_hysteresis_keeps_near() {
        let mut controller = DecisionLodController::new(thresholds(), 2, DecisionLod::Far)
            .expect("non-zero stability");

        let first = controller.observe(DecisionObservation::new(85, DecisionVisibility::Visible));
        assert_eq!(first.current, DecisionLod::Far);
        assert_eq!(first.pending, Some(DecisionLod::Near));
        assert_eq!(first.pending_frames, 1);

        let second = controller.observe(DecisionObservation::new(10, DecisionVisibility::Visible));
        assert!(!second.changed);
        assert_eq!(second.pending, Some(DecisionLod::Offscreen));
        assert_eq!(controller.current(), DecisionLod::Far);

        let near = controller.observe(DecisionObservation::new(85, DecisionVisibility::Visible));
        assert_eq!(near.pending, Some(DecisionLod::Near));
        let near = controller.observe(DecisionObservation::new(85, DecisionVisibility::Visible));
        assert!(near.changed);
        assert_eq!(near.current, DecisionLod::Near);

        let hysteretic =
            controller.target_for(DecisionObservation::new(65, DecisionVisibility::Visible));
        assert_eq!(hysteretic, DecisionLod::Near);
    }

    #[test]
    fn visibility_selects_offscreen_and_dormant_deterministically() {
        let mut controller = DecisionLodController::new(thresholds(), 1, DecisionLod::Near)
            .expect("non-zero stability");
        assert_eq!(
            controller
                .observe(DecisionObservation::new(
                    u32::MAX,
                    DecisionVisibility::Offscreen,
                ))
                .current,
            DecisionLod::Offscreen
        );
        assert_eq!(
            controller
                .observe(DecisionObservation::new(0, DecisionVisibility::Dormant))
                .current,
            DecisionLod::Dormant
        );
        assert_eq!(
            controller
                .observe(DecisionObservation::new(
                    u32::MAX,
                    DecisionVisibility::Visible,
                ))
                .current,
            DecisionLod::Near
        );
    }

    #[test]
    fn cadence_is_frame_based_and_phase_is_normalized() {
        let cadence = FixedFrameCadence::new(4, 9).expect("non-zero period");
        assert_eq!(cadence.phase(), 1);
        assert!(cadence.is_due(1));
        assert!(cadence.is_due(5));
        assert!(!cadence.is_due(4));
        assert_eq!(cadence.next_due_at_or_after(6), Some(9));
        assert_eq!(cadence.next_due_at_or_after(9), Some(9));
        assert_eq!(FixedFrameCadence::every(0), Err(CadenceError::ZeroPeriod));
    }

    #[test]
    fn lod_cadence_maps_each_level_to_its_fixed_period() {
        let periods = DecisionLodPeriods::new(1, 2, 4, 8).expect("valid periods");
        let cadence = DecisionLodCadence::from_periods(periods, 0).expect("valid cadences");
        assert!(cadence.is_due(DecisionLod::Near, 3));
        assert!(!cadence.is_due(DecisionLod::Far, 3));
        assert!(cadence.is_due(DecisionLod::Far, 4));
        assert!(cadence.is_due(DecisionLod::Offscreen, 8));
        assert!(cadence.is_due(DecisionLod::Dormant, 16));
        assert_eq!(periods.for_lod(DecisionLod::Offscreen), 4);
    }

    #[test]
    fn utility_selection_uses_score_key_and_index_tie_breaks() {
        let candidates = [
            UtilityCandidate::new(40, 100, "late"),
            UtilityCandidate::new(10, 100, "preferred"),
            UtilityCandidate::new(1, 99, "lower_score"),
        ];
        let selection = select_utility(&candidates).expect("non-empty candidates");
        assert_eq!(selection.index(), 1);
        assert_eq!(selection.key(), 10);
        assert_eq!(selection.utility(), 100);
        assert_eq!(candidates[selection.index()].value, "preferred");

        let duplicate_keys = [
            UtilityCandidate::new(7, 12, 0),
            UtilityCandidate::new(7, 12, 1),
        ];
        assert_eq!(select_utility(&duplicate_keys).unwrap().index(), 0);
        assert_eq!(select_utility::<i32>(&[]), None);
    }
}
