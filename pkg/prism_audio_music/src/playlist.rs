//! **Playlists**: how a run of segments is ordered, repeated, and randomised.
//!
//! A [`Playlist`] names an ordered list of [`PlaylistItem`]s (each a segment
//! plus a repeat count) and a [`PlaylistMode`] describing how the planner walks
//! them: straight through once, looping, ping-ponging back and forth, or in a
//! deterministic random / shuffle order. The authoring data is pure and
//! immutable; the *walk* lives in a [`PlaylistCursor`], a small resumable state
//! machine the live [`crate::system::MusicSystem`] advances one segment at a
//! time.
//!
//! Randomisation is driven by the system's seeded [`crate::rng::Rng`], so a
//! given seed and playlist always yield the same sequence (replays, lockstep,
//! golden tests). `Shuffle` draws without replacement from a bag that refills
//! and reshuffles once exhausted, so every segment plays once per cycle.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam
//! Audio, or Google Resonance Audio source or derived code**. The ordering
//! modes and the bag-shuffle are reconstructed from first principles (a classic
//! Fisher-Yates shuffle) over plain data. No AI/ML.
//!
//! # Relationship
//!
//! A [`Playlist`] is stored in [`crate::model::MusicModel`]. A
//! [`PlaylistCursor`] is held by [`crate::system::MusicSystem`], advanced by
//! [`crate::system::MusicSystem::advance_to`], and seeded by the system's
//! [`crate::rng::Rng`]. Each [`PlaylistCursor::next`] yields the next
//! [`crate::id::SegmentId`] to schedule at the previous segment's hand-off
//! point.

use alloc::vec::Vec;

use crate::id::{PlaylistId, SegmentId};
use crate::rng::Rng;

/// How a [`Playlist`] walks its items.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum PlaylistMode {
    /// Play each item once in order, then stop.
    Sequence,
    /// Play in order, wrapping back to the first item forever.
    Loop,
    /// Play forward to the last item, then backward to the first, bouncing
    /// forever (endpoints are not repeated).
    PingPong,
    /// Pick each next item uniformly at random (with replacement).
    Random,
    /// Pick each next item from a without-replacement bag that reshuffles once
    /// every item has played.
    Shuffle,
}

/// One entry in a [`Playlist`]: a segment and how many times it plays before
/// the cursor advances.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct PlaylistItem {
    /// The segment this item plays.
    pub segment: SegmentId,
    /// How many consecutive times the segment plays (treated as at least `1`).
    pub repeat: u32,
}

impl PlaylistItem {
    /// Builds an item that plays `segment` `repeat` times (minimum one).
    #[must_use]
    pub fn new(segment: SegmentId, repeat: u32) -> Self {
        Self { segment, repeat }
    }

    /// Builds an item that plays `segment` exactly once.
    #[must_use]
    pub fn once(segment: SegmentId) -> Self {
        Self {
            segment,
            repeat: 1,
        }
    }

    /// The effective play count, never less than one.
    #[inline]
    #[must_use]
    fn plays(&self) -> u32 {
        self.repeat.max(1)
    }
}

/// An ordered list of segments plus a walk [`PlaylistMode`].
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct Playlist {
    /// Stable id this playlist is referenced by.
    pub id: PlaylistId,
    /// Items in authored order.
    pub items: Vec<PlaylistItem>,
    /// How the items are walked.
    pub mode: PlaylistMode,
}

impl Playlist {
    /// Builds an empty playlist with the given walk mode.
    #[must_use]
    pub fn new(id: PlaylistId, mode: PlaylistMode) -> Self {
        Self {
            id,
            items: Vec::new(),
            mode,
        }
    }

    /// Appends an item, returning `self` for builder-style chaining.
    #[must_use]
    pub fn with_item(mut self, item: PlaylistItem) -> Self {
        self.items.push(item);
        self
    }

    /// Returns the number of items.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Returns whether the playlist has no items.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

/// A resumable walk over a [`Playlist`]'s items.
///
/// The cursor lazily initialises on the first [`PlaylistCursor::next`] call so
/// a `Random` or `Shuffle` playlist draws its first item from the system RNG at
/// the moment playback starts. It stores only indices and small bookkeeping, so
/// it is cheap to clone and serialise.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct PlaylistCursor {
    /// The playlist this cursor walks.
    playlist: PlaylistId,
    /// Index of the current item (meaningful once `started`).
    index: usize,
    /// Ping-pong direction: `true` while walking forward.
    ascending: bool,
    /// Remaining plays of the current item before the cursor advances.
    plays_left: u32,
    /// Remaining shuffled indices for `Shuffle` (drawn from the back).
    bag: Vec<usize>,
    /// Whether the first item has been chosen yet.
    started: bool,
    /// Whether a `Sequence` walk has run off the end.
    finished: bool,
}

impl PlaylistCursor {
    /// Builds a fresh cursor for `playlist`, positioned before the first item.
    #[must_use]
    pub fn new(playlist: PlaylistId) -> Self {
        Self {
            playlist,
            index: 0,
            ascending: true,
            plays_left: 0,
            bag: Vec::new(),
            started: false,
            finished: false,
        }
    }

    /// Returns the id of the playlist this cursor walks.
    #[inline]
    #[must_use]
    pub fn playlist(&self) -> PlaylistId {
        self.playlist
    }

    /// Returns whether the walk has run off the end (only possible for
    /// [`PlaylistMode::Sequence`]).
    #[inline]
    #[must_use]
    pub fn is_finished(&self) -> bool {
        self.finished
    }

    /// Refills `bag` with every index in a deterministic shuffled order.
    ///
    /// Uses an in-place Fisher-Yates shuffle driven by `rng`; items are drawn
    /// from the back of the bag.
    fn refill_bag(&mut self, len: usize, rng: &mut Rng) {
        self.bag.clear();
        self.bag.extend(0..len);
        // Fisher-Yates: for i from len-1 down to 1, swap with a random j <= i.
        let mut i = len;
        while i > 1 {
            i -= 1;
            let j = rng.next_index(i + 1);
            self.bag.swap(i, j);
        }
    }

    /// Advances to the next item, returning its segment, or `None` once a
    /// [`PlaylistMode::Sequence`] walk is exhausted or the playlist is empty.
    ///
    /// `rng` supplies the draws for [`PlaylistMode::Random`] and
    /// [`PlaylistMode::Shuffle`]; it is untouched for the deterministic modes.
    pub fn next(&mut self, playlist: &Playlist, rng: &mut Rng) -> Option<SegmentId> {
        let len = playlist.items.len();
        if self.finished || len == 0 {
            return None;
        }

        if !self.started {
            self.started = true;
            self.index = match playlist.mode {
                PlaylistMode::Sequence | PlaylistMode::Loop | PlaylistMode::PingPong => 0,
                PlaylistMode::Random => rng.next_index(len),
                PlaylistMode::Shuffle => {
                    self.refill_bag(len, rng);
                    self.bag.pop().unwrap_or(0)
                }
            };
            self.ascending = true;
            self.plays_left = playlist.items[self.index].plays();
        }

        let segment = playlist.items[self.index].segment;
        self.plays_left -= 1;
        if self.plays_left == 0 {
            self.advance(playlist, rng, len);
        }
        Some(segment)
    }

    /// Steps `index` to the next item per the playlist mode once the current
    /// item's repeats are spent.
    fn advance(&mut self, playlist: &Playlist, rng: &mut Rng, len: usize) {
        match playlist.mode {
            PlaylistMode::Sequence => {
                if self.index + 1 < len {
                    self.index += 1;
                } else {
                    self.finished = true;
                }
            }
            PlaylistMode::Loop => {
                self.index = (self.index + 1) % len;
            }
            PlaylistMode::PingPong => self.step_ping_pong(len),
            PlaylistMode::Random => {
                self.index = rng.next_index(len);
            }
            PlaylistMode::Shuffle => {
                if self.bag.is_empty() {
                    self.refill_bag(len, rng);
                }
                self.index = self.bag.pop().unwrap_or(0);
            }
        }
        if !self.finished {
            self.plays_left = playlist.items[self.index].plays();
        }
    }

    /// Bounces `index` between the endpoints without repeating them.
    fn step_ping_pong(&mut self, len: usize) {
        if len <= 1 {
            return;
        }
        if self.ascending {
            if self.index + 1 >= len {
                self.ascending = false;
                self.index -= 1;
            } else {
                self.index += 1;
            }
        } else if self.index == 0 {
            self.ascending = true;
            self.index += 1;
        } else {
            self.index -= 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn playlist(mode: PlaylistMode) -> Playlist {
        Playlist::new(PlaylistId::new(1), mode)
            .with_item(PlaylistItem::once(SegmentId::new(0)))
            .with_item(PlaylistItem::once(SegmentId::new(1)))
            .with_item(PlaylistItem::once(SegmentId::new(2)))
    }

    fn take(pl: &Playlist, seed: u64, n: usize) -> Vec<u32> {
        let mut cur = PlaylistCursor::new(pl.id);
        let mut rng = Rng::new(seed);
        let mut out = Vec::new();
        for _ in 0..n {
            match cur.next(pl, &mut rng) {
                Some(s) => out.push(s.get()),
                None => break,
            }
        }
        out
    }

    #[test]
    fn sequence_plays_once_then_stops() {
        let pl = playlist(PlaylistMode::Sequence);
        assert_eq!(take(&pl, 1, 10), alloc::vec![0, 1, 2]);
    }

    #[test]
    fn loop_wraps_forever() {
        let pl = playlist(PlaylistMode::Loop);
        assert_eq!(take(&pl, 1, 7), alloc::vec![0, 1, 2, 0, 1, 2, 0]);
    }

    #[test]
    fn ping_pong_bounces_without_repeating_endpoints() {
        let pl = playlist(PlaylistMode::PingPong);
        assert_eq!(take(&pl, 1, 8), alloc::vec![0, 1, 2, 1, 0, 1, 2, 1]);
    }

    #[test]
    fn repeat_count_holds_each_item() {
        let pl = Playlist::new(PlaylistId::new(2), PlaylistMode::Sequence)
            .with_item(PlaylistItem::new(SegmentId::new(0), 2))
            .with_item(PlaylistItem::new(SegmentId::new(1), 3));
        assert_eq!(take(&pl, 1, 10), alloc::vec![0, 0, 1, 1, 1]);
    }

    #[test]
    fn zero_repeat_treated_as_one() {
        let pl = Playlist::new(PlaylistId::new(3), PlaylistMode::Sequence)
            .with_item(PlaylistItem::new(SegmentId::new(5), 0));
        assert_eq!(take(&pl, 1, 4), alloc::vec![5]);
    }

    #[test]
    fn random_is_deterministic_for_a_seed() {
        let pl = playlist(PlaylistMode::Random);
        let a = take(&pl, 0xABCD, 20);
        let b = take(&pl, 0xABCD, 20);
        assert_eq!(a, b);
        assert_eq!(a.len(), 20);
        assert!(a.iter().all(|&s| s < 3));
    }

    #[test]
    fn different_seeds_give_different_random_walks() {
        let pl = playlist(PlaylistMode::Random);
        assert_ne!(take(&pl, 1, 30), take(&pl, 2, 30));
    }

    #[test]
    fn shuffle_covers_every_item_once_per_cycle() {
        let pl = playlist(PlaylistMode::Shuffle);
        let seq = take(&pl, 0x1234, 6);
        let mut first = seq[0..3].to_vec();
        let mut second = seq[3..6].to_vec();
        first.sort_unstable();
        second.sort_unstable();
        assert_eq!(first, alloc::vec![0, 1, 2]);
        assert_eq!(second, alloc::vec![0, 1, 2]);
    }

    #[test]
    fn shuffle_is_deterministic_for_a_seed() {
        let pl = playlist(PlaylistMode::Shuffle);
        assert_eq!(take(&pl, 7, 12), take(&pl, 7, 12));
    }

    #[test]
    fn empty_playlist_yields_nothing() {
        let pl = Playlist::new(PlaylistId::new(9), PlaylistMode::Loop);
        assert!(pl.is_empty());
        assert_eq!(take(&pl, 1, 5), Vec::<u32>::new());
    }
}
