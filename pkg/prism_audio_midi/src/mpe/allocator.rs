//! Deterministic MPE member-channel allocation.
//!
//! When an MPE controller plays a note it is assigned to one of its zone's
//! member channels so that the note owns a private pitch bend, pressure, and
//! timbre. [`MpeAllocator`] performs that assignment with a fixed policy and no
//! randomness: it prefers a free member channel in round-robin order (spreading
//! consecutive notes across the zone the way an MPE source expects), and when
//! every channel already holds a note it stacks the new note onto the
//! least-loaded channel, breaking ties by the channel that has been idle the
//! longest and then by the lowest member ordinal. Releasing a note decrements
//! that channel's load so the slot can be reused. Given the same sequence of
//! allocate and release calls the allocator always returns the same channels.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML. The round-robin
//! member-channel convention follows the publicly published MPE specification.
//!
//! # Relationship
//! Implements the member-channel allocation of design section 52 (MPE). Built
//! on [`crate::mpe::zone::MpeZone`]; the channel it returns is the one a note's
//! [`crate::expression`] state and [`crate::mapping`] writes are attributed to.

use crate::mpe::zone::{CHANNELS_PER_PORT, MpeZone};

/// A deterministic allocator of MPE member channels for one zone.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct MpeAllocator {
    zone: MpeZone,
    load: [u16; CHANNELS_PER_PORT as usize],
    last_order: [u64; CHANNELS_PER_PORT as usize],
    cursor: u8,
    next_order: u64,
}

impl MpeAllocator {
    /// Creates an allocator for `zone` with every member channel idle.
    #[must_use]
    pub fn new(zone: MpeZone) -> Self {
        Self {
            zone,
            load: [0; CHANNELS_PER_PORT as usize],
            last_order: [0; CHANNELS_PER_PORT as usize],
            cursor: 0,
            next_order: 0,
        }
    }

    /// Returns the zone this allocator serves.
    #[must_use]
    pub const fn zone(&self) -> MpeZone {
        self.zone
    }

    /// Reconfigures the allocator for a new zone, clearing all load. Any notes
    /// that were allocated under the previous zone are forgotten, so callers
    /// should release or re-voice outstanding notes around a reconfigure.
    pub fn reconfigure(&mut self, zone: MpeZone) {
        self.zone = zone;
        self.load = [0; CHANNELS_PER_PORT as usize];
        self.last_order = [0; CHANNELS_PER_PORT as usize];
        self.cursor = 0;
        self.next_order = 0;
    }

    /// Allocates a member channel for a new note and returns it.
    ///
    /// A free channel is preferred, searched in round-robin order from the
    /// current cursor; otherwise the least-loaded channel is chosen, with ties
    /// broken by longest-idle and then lowest ordinal. The returned value is
    /// always a valid member channel because every zone has at least one
    /// member, so this never returns `None` for a configured zone; the
    /// `Option` is kept for symmetry and future zone-less states.
    pub fn allocate(&mut self) -> Option<u8> {
        let count = self.zone.member_count();
        if count == 0 {
            return None;
        }
        let ordinal = self
            .find_free_ordinal(count)
            .unwrap_or_else(|| self.least_loaded_ordinal(count));
        self.load[ordinal as usize] += 1;
        self.last_order[ordinal as usize] = self.next_order;
        self.next_order += 1;
        self.cursor = (ordinal + 1) % count;
        self.zone.member_channel(ordinal)
    }

    /// Releases one note from `channel`, decrementing its load. Returns `true`
    /// when `channel` was a member channel carrying at least one note.
    pub fn release(&mut self, channel: u8) -> bool {
        match self.ordinal_of(channel) {
            Some(ordinal) if self.load[ordinal as usize] > 0 => {
                self.load[ordinal as usize] -= 1;
                true
            }
            _ => false,
        }
    }

    /// Returns the number of notes currently assigned to `channel`, or `0` when
    /// `channel` is not a member channel of this zone.
    #[must_use]
    pub fn active_notes(&self, channel: u8) -> u16 {
        match self.ordinal_of(channel) {
            Some(ordinal) => self.load[ordinal as usize],
            None => 0,
        }
    }

    /// Returns the total number of outstanding note allocations across the zone.
    #[must_use]
    pub fn total_active(&self) -> u32 {
        let count = self.zone.member_count() as usize;
        self.load[..count].iter().map(|&n| u32::from(n)).sum()
    }

    fn find_free_ordinal(&self, count: u8) -> Option<u8> {
        for step in 0..count {
            let ordinal = (self.cursor + step) % count;
            if self.load[ordinal as usize] == 0 {
                return Some(ordinal);
            }
        }
        None
    }

    fn least_loaded_ordinal(&self, count: u8) -> u8 {
        let mut best = 0u8;
        let mut best_load = u16::MAX;
        let mut best_order = u64::MAX;
        for ordinal in 0..count {
            let load = self.load[ordinal as usize];
            let order = self.last_order[ordinal as usize];
            if load < best_load || (load == best_load && order < best_order) {
                best = ordinal;
                best_load = load;
                best_order = order;
            }
        }
        best
    }

    fn ordinal_of(&self, channel: u8) -> Option<u8> {
        let count = self.zone.member_count();
        (0..count).find(|&ordinal| self.zone.member_channel(ordinal) == Some(channel))
    }
}

#[cfg(test)]
mod tests {
    use super::MpeAllocator;
    use crate::mpe::zone::MpeZone;

    #[test]
    fn round_robin_spreads_notes() {
        let mut alloc = MpeAllocator::new(MpeZone::lower(4));
        // Members are channels 1..=4; consecutive notes land on each in turn.
        assert_eq!(alloc.allocate(), Some(1));
        assert_eq!(alloc.allocate(), Some(2));
        assert_eq!(alloc.allocate(), Some(3));
        assert_eq!(alloc.allocate(), Some(4));
        // Each member now holds exactly one note.
        for channel in 1..=4 {
            assert_eq!(alloc.active_notes(channel), 1);
        }
        assert_eq!(alloc.total_active(), 4);
    }

    #[test]
    fn release_recycles_channel() {
        let mut alloc = MpeAllocator::new(MpeZone::lower(3));
        assert_eq!(alloc.allocate(), Some(1));
        assert_eq!(alloc.allocate(), Some(2));
        assert_eq!(alloc.allocate(), Some(3));
        // Free channel 2; the next allocation reuses it because it is the only
        // idle member.
        assert!(alloc.release(2));
        assert_eq!(alloc.active_notes(2), 0);
        assert_eq!(alloc.allocate(), Some(2));
        assert_eq!(alloc.active_notes(2), 1);
    }

    #[test]
    fn stacks_on_least_loaded_when_full() {
        let mut alloc = MpeAllocator::new(MpeZone::lower(2));
        // Fill both members once (channels 1 and 2).
        assert_eq!(alloc.allocate(), Some(1));
        assert_eq!(alloc.allocate(), Some(2));
        // Both loaded equally; tie broken by longest-idle, which is channel 1
        // (allocated first), so the stack lands there.
        assert_eq!(alloc.allocate(), Some(1));
        assert_eq!(alloc.active_notes(1), 2);
        assert_eq!(alloc.active_notes(2), 1);
        // Next stack goes to the now-least-loaded channel 2.
        assert_eq!(alloc.allocate(), Some(2));
        assert_eq!(alloc.active_notes(2), 2);
    }

    #[test]
    fn upper_zone_members_descend() {
        let mut alloc = MpeAllocator::new(MpeZone::upper(3));
        assert_eq!(alloc.allocate(), Some(14));
        assert_eq!(alloc.allocate(), Some(13));
        assert_eq!(alloc.allocate(), Some(12));
    }

    #[test]
    fn release_rejects_non_member() {
        let mut alloc = MpeAllocator::new(MpeZone::lower(2));
        // Channel 0 is the manager, not a member.
        assert!(!alloc.release(0));
        // A member with no outstanding note cannot be released.
        assert!(!alloc.release(1));
    }

    #[test]
    fn reconfigure_clears_state() {
        let mut alloc = MpeAllocator::new(MpeZone::lower(4));
        alloc.allocate();
        alloc.allocate();
        alloc.reconfigure(MpeZone::upper(2));
        assert_eq!(alloc.total_active(), 0);
        assert_eq!(alloc.allocate(), Some(14));
    }
}
