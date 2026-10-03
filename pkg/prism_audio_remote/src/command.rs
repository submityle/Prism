//! Whitelisted write-command contract for the remote authoring surface.
//!
//! A remote tool never pokes real-time memory. Instead it submits one of a
//! small, closed set of [`AuthoringCommand`] values, each of which maps onto an
//! existing engine command-ring entry (change a parameter, trigger an event,
//! switch a state, swap a snapshot, set a bus gain). Every command is checked
//! against a [`CapabilitySet`] allow-list and a value-range gate before it is
//! allowed to proceed, so an under-privileged session can observe telemetry yet
//! be forbidden from issuing writes at all. Only the contract and its
//! validation live here; actually enqueuing onto the ring is the engine's job.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the whitelisted write path of design section 38. Each variant is
//! a serializable mirror of a command-ring entry from design section 21, so a
//! remote edit and an in-game edit are indistinguishable once validated and
//! therefore share one deterministic, replayable semantics. The
//! [`CapabilitySet`] here is also consumed by [`crate::auth`] to form the
//! development-only security boundary.

/// Stable identifier for an authored event (one-shot or sustained).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct EventId(pub u64);

/// Stable identifier for a mix bus.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct BusId(pub u64);

/// Stable identifier for a state group (a set of mutually exclusive states).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct StateGroupId(pub u64);

/// Stable identifier for a single state within a [`StateGroupId`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct StateId(pub u64);

/// Stable identifier for a mixer snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct SnapshotId(pub u64);

/// Stable identifier for an automatable parameter (an RTPC or node control).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ParameterTarget(pub u64);

/// Smallest permitted value for a [`AuthoringCommand::SetParameter`] write.
pub const PARAMETER_MIN: f32 = -1.0e6;
/// Largest permitted value for a [`AuthoringCommand::SetParameter`] write.
pub const PARAMETER_MAX: f32 = 1.0e6;
/// Smallest permitted bus gain in decibels.
pub const BUS_GAIN_DB_MIN: f32 = -120.0;
/// Largest permitted bus gain in decibels.
pub const BUS_GAIN_DB_MAX: f32 = 24.0;

/// The closed set of write commands a remote tool may submit.
///
/// The set is deliberately small: anything not expressible here cannot be done
/// remotely at all, which is what keeps the attack surface bounded.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum AuthoringCommand {
    /// Set an automatable parameter (an RTPC or node control) to `value`.
    SetParameter {
        /// Which parameter to drive.
        target: ParameterTarget,
        /// The new parameter value.
        value: f32,
    },
    /// Post an event into the running engine.
    TriggerEvent {
        /// Which event to post.
        event: EventId,
    },
    /// Switch the active state of a state group.
    SetState {
        /// Which group to update.
        group: StateGroupId,
        /// The state to make active within that group.
        state: StateId,
    },
    /// Recall a mixer snapshot.
    SwapSnapshot {
        /// Which snapshot to recall.
        snapshot: SnapshotId,
    },
    /// Set the output gain of a mix bus, in decibels.
    SetBusGain {
        /// Which bus to adjust.
        bus: BusId,
        /// The new gain in decibels.
        gain_db: f32,
    },
}

/// A coarse classification of an [`AuthoringCommand`] used for allow-list
/// checks and audit logging.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum CommandKind {
    /// A [`AuthoringCommand::SetParameter`].
    SetParameter,
    /// A [`AuthoringCommand::TriggerEvent`].
    TriggerEvent,
    /// A [`AuthoringCommand::SetState`].
    SetState,
    /// A [`AuthoringCommand::SwapSnapshot`].
    SwapSnapshot,
    /// A [`AuthoringCommand::SetBusGain`].
    SetBusGain,
}

/// The reason a command was refused by [`AuthoringCommand::validate`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum CommandError {
    /// The session is not permitted to issue any write command.
    WritesDisabled,
    /// Writes are enabled, but this specific command kind is not allowed.
    NotPermitted(CommandKind),
    /// A numeric field was not finite (it was `NaN` or infinite).
    NonFinite,
    /// A numeric field was finite but outside the permitted range.
    OutOfRange,
}

/// A per-session allow-list describing which command kinds are permitted.
///
/// A fresh set denies everything; callers opt in explicitly. This is the
/// single source of truth shared with [`crate::auth::Authenticator`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct CapabilitySet {
    /// Master gate: when `false`, every write is refused regardless of the
    /// per-kind flags below.
    pub writes_enabled: bool,
    /// Allow [`AuthoringCommand::SetParameter`].
    pub set_parameter: bool,
    /// Allow [`AuthoringCommand::TriggerEvent`].
    pub trigger_event: bool,
    /// Allow [`AuthoringCommand::SetState`].
    pub set_state: bool,
    /// Allow [`AuthoringCommand::SwapSnapshot`].
    pub swap_snapshot: bool,
    /// Allow [`AuthoringCommand::SetBusGain`].
    pub set_bus_gain: bool,
}

impl CapabilitySet {
    /// A capability set that forbids every write.
    #[must_use]
    pub const fn denied() -> Self {
        Self {
            writes_enabled: false,
            set_parameter: false,
            trigger_event: false,
            set_state: false,
            swap_snapshot: false,
            set_bus_gain: false,
        }
    }

    /// A capability set that permits every write command.
    #[must_use]
    pub const fn full() -> Self {
        Self {
            writes_enabled: true,
            set_parameter: true,
            trigger_event: true,
            set_state: true,
            swap_snapshot: true,
            set_bus_gain: true,
        }
    }

    /// Returns `true` when a command of `kind` is permitted by this set.
    #[must_use]
    pub const fn permits(&self, kind: CommandKind) -> bool {
        if !self.writes_enabled {
            return false;
        }
        match kind {
            CommandKind::SetParameter => self.set_parameter,
            CommandKind::TriggerEvent => self.trigger_event,
            CommandKind::SetState => self.set_state,
            CommandKind::SwapSnapshot => self.swap_snapshot,
            CommandKind::SetBusGain => self.set_bus_gain,
        }
    }
}

impl Default for CapabilitySet {
    #[inline]
    fn default() -> Self {
        Self::denied()
    }
}

impl AuthoringCommand {
    /// Returns the coarse [`CommandKind`] of this command.
    #[must_use]
    pub const fn kind(&self) -> CommandKind {
        match self {
            AuthoringCommand::SetParameter { .. } => CommandKind::SetParameter,
            AuthoringCommand::TriggerEvent { .. } => CommandKind::TriggerEvent,
            AuthoringCommand::SetState { .. } => CommandKind::SetState,
            AuthoringCommand::SwapSnapshot { .. } => CommandKind::SwapSnapshot,
            AuthoringCommand::SetBusGain { .. } => CommandKind::SetBusGain,
        }
    }

    /// Checks this command against a capability allow-list and value ranges.
    ///
    /// Validation happens in three stages: the master write gate, the per-kind
    /// allow-list, and finally numeric range checks for commands that carry a
    /// float. A command that passes is safe to translate into a command-ring
    /// entry.
    ///
    /// # Errors
    ///
    /// Returns [`CommandError`] describing the first failed stage.
    pub fn validate(&self, caps: &CapabilitySet) -> Result<(), CommandError> {
        if !caps.writes_enabled {
            return Err(CommandError::WritesDisabled);
        }
        let kind = self.kind();
        if !caps.permits(kind) {
            return Err(CommandError::NotPermitted(kind));
        }
        match self {
            AuthoringCommand::SetParameter { value, .. } => {
                check_range(*value, PARAMETER_MIN, PARAMETER_MAX)
            }
            AuthoringCommand::SetBusGain { gain_db, .. } => {
                check_range(*gain_db, BUS_GAIN_DB_MIN, BUS_GAIN_DB_MAX)
            }
            AuthoringCommand::TriggerEvent { .. }
            | AuthoringCommand::SetState { .. }
            | AuthoringCommand::SwapSnapshot { .. } => Ok(()),
        }
    }
}

/// Verifies `value` is finite and within the inclusive range `[min, max]`.
fn check_range(value: f32, min: f32, max: f32) -> Result<(), CommandError> {
    if !value.is_finite() {
        return Err(CommandError::NonFinite);
    }
    if value < min || value > max {
        return Err(CommandError::OutOfRange);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn denied_set_refuses_everything() {
        let caps = CapabilitySet::denied();
        let cmd = AuthoringCommand::TriggerEvent {
            event: EventId(7),
        };
        assert_eq!(cmd.validate(&caps), Err(CommandError::WritesDisabled));
    }

    #[test]
    fn per_kind_allow_list_is_enforced() {
        let caps = CapabilitySet {
            writes_enabled: true,
            trigger_event: true,
            ..CapabilitySet::denied()
        };
        let allowed = AuthoringCommand::TriggerEvent {
            event: EventId(1),
        };
        assert_eq!(allowed.validate(&caps), Ok(()));

        let blocked = AuthoringCommand::SwapSnapshot {
            snapshot: SnapshotId(2),
        };
        assert_eq!(
            blocked.validate(&caps),
            Err(CommandError::NotPermitted(CommandKind::SwapSnapshot))
        );
    }

    #[test]
    fn parameter_range_is_checked() {
        let caps = CapabilitySet::full();
        let ok = AuthoringCommand::SetParameter {
            target: ParameterTarget(1),
            value: 0.5,
        };
        assert_eq!(ok.validate(&caps), Ok(()));

        let too_big = AuthoringCommand::SetParameter {
            target: ParameterTarget(1),
            value: PARAMETER_MAX * 2.0,
        };
        assert_eq!(too_big.validate(&caps), Err(CommandError::OutOfRange));
    }

    #[test]
    fn non_finite_values_are_rejected() {
        let caps = CapabilitySet::full();
        let nan = AuthoringCommand::SetBusGain {
            bus: BusId(3),
            gain_db: f32::NAN,
        };
        assert_eq!(nan.validate(&caps), Err(CommandError::NonFinite));
    }

    #[test]
    fn bus_gain_range_bounds() {
        let caps = CapabilitySet::full();
        let hot = AuthoringCommand::SetBusGain {
            bus: BusId(4),
            gain_db: BUS_GAIN_DB_MAX + 0.001,
        };
        assert_eq!(hot.validate(&caps), Err(CommandError::OutOfRange));

        let edge = AuthoringCommand::SetBusGain {
            bus: BusId(4),
            gain_db: BUS_GAIN_DB_MAX,
        };
        assert_eq!(edge.validate(&caps), Ok(()));
    }

    #[test]
    fn kind_round_trips() {
        let cmd = AuthoringCommand::SetState {
            group: StateGroupId(1),
            state: StateId(2),
        };
        assert_eq!(cmd.kind(), CommandKind::SetState);
    }
}
