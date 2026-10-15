use core::fmt;

/// The representation weight of a world item.
///
/// `A` is a live interactive hierarchy, `B` is a streamed block, and `C` is a
/// lightweight proxy suitable for large static populations.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ResidencyTier {
    /// Interactive, materialized representation.
    A,
    /// Streamed block representation.
    B,
    /// Proxy representation.
    C,
}

impl ResidencyTier {
    /// Returns the detail rank, where a larger value is more detailed.
    pub const fn detail_rank(self) -> u8 {
        match self {
            Self::A => 2,
            Self::B => 1,
            Self::C => 0,
        }
    }

    /// Returns whether moving to `other` increases residency detail.
    pub const fn is_upgrade_to(self, other: Self) -> bool {
        other.detail_rank() > self.detail_rank()
    }

    /// Returns whether moving to `other` decreases residency detail.
    pub const fn is_downgrade_to(self, other: Self) -> bool {
        other.detail_rank() < self.detail_rank()
    }
}

/// The currently committed representation state of an item.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ResidencyState {
    /// The item is materialized for interaction.
    TierA,
    /// The item is represented by a streamed block.
    TierB,
    /// The item is represented by a proxy.
    TierC,
}

impl ResidencyState {
    /// Converts the state to its policy tier.
    pub const fn tier(self) -> ResidencyTier {
        match self {
            Self::TierA => ResidencyTier::A,
            Self::TierB => ResidencyTier::B,
            Self::TierC => ResidencyTier::C,
        }
    }

    /// Returns whether the state transition is allowed by the world design.
    ///
    /// All three pairs are allowed in both directions, including direct `A`
    /// and `C` transitions for materialization penetration.
    pub const fn can_transition_to(self, other: Self) -> bool {
        let _ = (self, other);
        true
    }

    /// Applies a legal state transition.
    pub const fn transition_to(self, other: Self) -> Option<Self> {
        if self.can_transition_to(other) {
            Some(other)
        } else {
            None
        }
    }

    /// Returns whether the transition increases detail.
    pub const fn is_upgrade_to(self, other: Self) -> bool {
        self.tier().is_upgrade_to(other.tier())
    }

    /// Returns whether the transition decreases detail.
    pub const fn is_downgrade_to(self, other: Self) -> bool {
        self.tier().is_downgrade_to(other.tier())
    }
}

impl From<ResidencyTier> for ResidencyState {
    fn from(value: ResidencyTier) -> Self {
        match value {
            ResidencyTier::A => Self::TierA,
            ResidencyTier::B => Self::TierB,
            ResidencyTier::C => Self::TierC,
        }
    }
}

impl From<ResidencyState> for ResidencyTier {
    fn from(value: ResidencyState) -> Self {
        value.tier()
    }
}

/// Errors produced when constructing an invalid distance policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PolicyError {
    /// The four distance thresholds are not monotonically ordered.
    ThresholdOrder,
}

impl fmt::Display for PolicyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ThresholdOrder => f.write_str("residency thresholds are not ordered"),
        }
    }
}

/// Distance hysteresis used to turn an observation into a target tier.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResidencyPolicy {
    /// Distance at or below which a C item may promote directly to A.
    pub tier_a_enter_distance: u32,
    /// Distance above which an A item may leave A.
    pub tier_a_exit_distance: u32,
    /// Distance at or below which a C item may promote to B.
    pub tier_b_enter_distance: u32,
    /// Distance above which a B item may demote to C.
    pub tier_b_exit_distance: u32,
    /// Minimum number of frames an item remains in a committed tier.
    pub min_resident_frames: u32,
}

impl ResidencyPolicy {
    /// Creates a policy with ordered thresholds and explicit minimum hold time.
    pub const fn new(
        tier_a_enter_distance: u32,
        tier_a_exit_distance: u32,
        tier_b_enter_distance: u32,
        tier_b_exit_distance: u32,
        min_resident_frames: u32,
    ) -> Result<Self, PolicyError> {
        if tier_a_enter_distance > tier_a_exit_distance
            || tier_a_exit_distance > tier_b_enter_distance
            || tier_b_enter_distance > tier_b_exit_distance
        {
            return Err(PolicyError::ThresholdOrder);
        }
        Ok(Self {
            tier_a_enter_distance,
            tier_a_exit_distance,
            tier_b_enter_distance,
            tier_b_exit_distance,
            min_resident_frames,
        })
    }

    /// Computes a hysteretic target state for the current state and distance.
    pub const fn target_state(self, state: ResidencyState, distance: u32) -> ResidencyState {
        match state {
            ResidencyState::TierA => {
                if distance <= self.tier_a_exit_distance {
                    ResidencyState::TierA
                } else if distance <= self.tier_b_exit_distance {
                    ResidencyState::TierB
                } else {
                    ResidencyState::TierC
                }
            }
            ResidencyState::TierB => {
                if distance <= self.tier_a_enter_distance {
                    ResidencyState::TierA
                } else if distance <= self.tier_b_exit_distance {
                    ResidencyState::TierB
                } else {
                    ResidencyState::TierC
                }
            }
            ResidencyState::TierC => {
                if distance <= self.tier_a_enter_distance {
                    ResidencyState::TierA
                } else if distance <= self.tier_b_enter_distance {
                    ResidencyState::TierB
                } else {
                    ResidencyState::TierC
                }
            }
        }
    }
}

/// Per-item state and the deterministic cost used by the scheduler.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResidencyRecord {
    /// Stable content or persistent identity.
    pub id: u64,
    /// Committed representation state.
    pub state: ResidencyState,
    /// Distance from the current interest origin in integer world units.
    pub distance: u32,
    /// Larger values win ties during promotion; smaller values yield first on demotion.
    pub priority: u32,
    /// Entity work required to promote this item.
    pub materialization_entities: u32,
    /// Byte work required to promote this item.
    pub materialization_bytes: u64,
    /// Frame at which the committed state last changed.
    pub last_transition_frame: Option<u64>,
}

impl ResidencyRecord {
    /// Creates a record with no prior transition and therefore no artificial hold.
    pub const fn new(
        id: u64,
        state: ResidencyState,
        distance: u32,
        priority: u32,
        materialization_entities: u32,
        materialization_bytes: u64,
    ) -> Self {
        Self {
            id,
            state,
            distance,
            priority,
            materialization_entities,
            materialization_bytes,
            last_transition_frame: None,
        }
    }

    /// Sets the last committed transition frame.
    pub const fn with_last_transition_frame(mut self, frame: u64) -> Self {
        self.last_transition_frame = Some(frame);
        self
    }
}

/// Per-frame promotion capacity.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaterializationBudget {
    /// Maximum entity writes allowed in one planning pass.
    pub max_entities: u32,
    /// Maximum bytes allowed in one planning pass.
    pub max_bytes: u64,
    used_entities: u32,
    used_bytes: u64,
}

impl MaterializationBudget {
    /// Creates an unused budget.
    pub const fn new(max_entities: u32, max_bytes: u64) -> Self {
        Self {
            max_entities,
            max_bytes,
            used_entities: 0,
            used_bytes: 0,
        }
    }

    /// Returns the amount already reserved by the plan.
    pub const fn used(self) -> (u32, u64) {
        (self.used_entities, self.used_bytes)
    }

    /// Returns the unspent entity and byte capacity.
    pub const fn remaining(self) -> (u32, u64) {
        (
            self.max_entities.saturating_sub(self.used_entities),
            self.max_bytes.saturating_sub(self.used_bytes),
        )
    }

    /// Tries to reserve a promotion cost without exceeding either limit.
    pub const fn try_reserve(&mut self, entities: u32, bytes: u64) -> bool {
        if entities > self.max_entities.saturating_sub(self.used_entities)
            || bytes > self.max_bytes.saturating_sub(self.used_bytes)
        {
            return false;
        }
        self.used_entities += entities;
        self.used_bytes += bytes;
        true
    }
}
