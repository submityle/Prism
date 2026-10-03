//! MPE zone configuration: lower and upper zones.
//!
//! MIDI Polyphonic Expression splits the sixteen channels of a port into one or
//! two zones. A zone owns one manager (master) channel for zone-wide control
//! and a contiguous block of member channels, one note per channel so that each
//! note can bend, press, and change timbre independently. The Lower Zone takes
//! channel 1 as its manager and counts its members upward from channel 2; the
//! Upper Zone takes channel 16 as its manager and counts its members downward
//! from channel 15. [`MpeZone`] captures that layout and answers the membership
//! questions the allocator needs.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML. The zone layout is
//! from the publicly published MPE specification.
//!
//! # Relationship
//! Implements the zone part of design section 52 (MPE per-note channel
//! splitting). Consumed by [`crate::mpe::allocator`].

/// Which MPE zone a configuration describes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum ZoneKind {
    /// The Lower Zone: manager on channel 0, members ascending from channel 1.
    Lower,
    /// The Upper Zone: manager on channel 15, members descending from 14.
    Upper,
}

/// The number of channels on a MIDI port.
pub const CHANNELS_PER_PORT: u8 = 16;

/// An MPE zone: a manager channel plus a block of member channels.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct MpeZone {
    kind: ZoneKind,
    member_count: u8,
}

impl MpeZone {
    /// Creates a Lower Zone with `member_count` member channels (clamped to the
    /// `1..=15` range the single zone can hold).
    #[must_use]
    pub const fn lower(member_count: u8) -> Self {
        Self {
            kind: ZoneKind::Lower,
            member_count: clamp_members(member_count),
        }
    }

    /// Creates an Upper Zone with `member_count` member channels (clamped to the
    /// `1..=15` range the single zone can hold).
    #[must_use]
    pub const fn upper(member_count: u8) -> Self {
        Self {
            kind: ZoneKind::Upper,
            member_count: clamp_members(member_count),
        }
    }

    /// Returns which zone this is.
    #[must_use]
    pub const fn kind(self) -> ZoneKind {
        self.kind
    }

    /// Returns the number of member channels.
    #[must_use]
    pub const fn member_count(self) -> u8 {
        self.member_count
    }

    /// Returns the manager (master) channel index.
    #[must_use]
    pub const fn manager_channel(self) -> u8 {
        match self.kind {
            ZoneKind::Lower => 0,
            ZoneKind::Upper => CHANNELS_PER_PORT - 1,
        }
    }

    /// Returns the member channel for a zero-based ordinal, or `None` when the
    /// ordinal is beyond the member count.
    ///
    /// Lower-zone ordinals count upward from channel 1; upper-zone ordinals
    /// count downward from channel 14.
    #[must_use]
    pub const fn member_channel(self, ordinal: u8) -> Option<u8> {
        if ordinal >= self.member_count {
            return None;
        }
        let channel = match self.kind {
            ZoneKind::Lower => 1 + ordinal,
            ZoneKind::Upper => CHANNELS_PER_PORT - 2 - ordinal,
        };
        Some(channel)
    }

    /// Returns `true` when `channel` is this zone's manager channel.
    #[must_use]
    pub const fn is_manager(self, channel: u8) -> bool {
        channel == self.manager_channel()
    }

    /// Returns `true` when `channel` is one of this zone's member channels.
    #[must_use]
    pub const fn is_member(self, channel: u8) -> bool {
        match self.kind {
            ZoneKind::Lower => channel >= 1 && channel <= self.member_count,
            ZoneKind::Upper => {
                channel <= CHANNELS_PER_PORT - 2
                    && channel >= CHANNELS_PER_PORT - 1 - self.member_count
            }
        }
    }
}

const fn clamp_members(member_count: u8) -> u8 {
    if member_count < 1 {
        1
    } else if member_count > CHANNELS_PER_PORT - 1 {
        CHANNELS_PER_PORT - 1
    } else {
        member_count
    }
}

#[cfg(test)]
mod tests {
    use super::{MpeZone, ZoneKind};

    #[test]
    fn lower_zone_layout() {
        let zone = MpeZone::lower(7);
        assert_eq!(zone.kind(), ZoneKind::Lower);
        assert_eq!(zone.manager_channel(), 0);
        assert_eq!(zone.member_channel(0), Some(1));
        assert_eq!(zone.member_channel(6), Some(7));
        assert_eq!(zone.member_channel(7), None);
        assert!(zone.is_manager(0));
        assert!(zone.is_member(1));
        assert!(zone.is_member(7));
        assert!(!zone.is_member(8));
    }

    #[test]
    fn upper_zone_layout() {
        let zone = MpeZone::upper(7);
        assert_eq!(zone.manager_channel(), 15);
        assert_eq!(zone.member_channel(0), Some(14));
        assert_eq!(zone.member_channel(6), Some(8));
        assert_eq!(zone.member_channel(7), None);
        assert!(zone.is_manager(15));
        assert!(zone.is_member(14));
        assert!(zone.is_member(8));
        assert!(!zone.is_member(7));
    }

    #[test]
    fn member_count_is_clamped() {
        assert_eq!(MpeZone::lower(0).member_count(), 1);
        assert_eq!(MpeZone::lower(200).member_count(), 15);
    }
}
