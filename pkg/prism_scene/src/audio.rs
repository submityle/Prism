//! Scene-owned audio source-node descriptors.
//!
//! This module deliberately contains only stable scene data. The runtime audio
//! crate consumes the descriptor and decides whether a resident or streaming
//! backend can materialize it; scene instantiation never claims that a device
//! or decoder exists.

use prism_ecs::prelude::Component;

/// A materialized scene source-node request.
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub struct AudioSourceNode {
    /// Bank id of the media entry.
    pub bank_id: u32,
    /// Bank-local media entry id.
    pub entry_id: u32,
    /// Whether the source should advance after materialization.
    pub playing: bool,
    /// Whether the source is authored as a looping route.
    pub looping: bool,
    /// Linear route gain encoded as bits for deterministic scene data.
    pub gain_bits: u32,
}

impl AudioSourceNode {
    /// Builds a source descriptor with unity gain.
    #[must_use]
    pub const fn new(bank_id: u32, entry_id: u32) -> Self {
        Self {
            bank_id,
            entry_id,
            playing: true,
            looping: false,
            gain_bits: 1.0f32.to_bits(),
        }
    }

    /// Returns the authored linear gain, sanitizing non-finite values at the
    /// runtime boundary rather than silently changing serialized scene data.
    #[must_use]
    pub fn gain(&self) -> f32 {
        let gain = f32::from_bits(self.gain_bits);
        if gain.is_finite() && gain >= 0.0 {
            gain
        } else {
            0.0
        }
    }

    /// Sets whether the source advances.
    #[must_use]
    pub const fn with_playing(mut self, playing: bool) -> Self {
        self.playing = playing;
        self
    }

    /// Sets the scene route's looping intent.
    #[must_use]
    pub const fn with_looping(mut self, looping: bool) -> Self {
        self.looping = looping;
        self
    }

    /// Sets the authored linear gain.
    #[must_use]
    pub fn with_gain(mut self, gain: f32) -> Self {
        self.gain_bits = gain.to_bits();
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptor_keeps_explicit_route_data() {
        let source = AudioSourceNode::new(7, 11)
            .with_playing(false)
            .with_looping(true)
            .with_gain(0.25);
        assert_eq!(source.bank_id, 7);
        assert_eq!(source.entry_id, 11);
        assert!(!source.playing);
        assert!(source.looping);
        assert_eq!(source.gain(), 0.25);
    }

    #[test]
    fn gain_sanitizes_negative_and_non_finite_authored_values() {
        assert_eq!(AudioSourceNode::new(1, 2).with_gain(-1.0).gain(), 0.0);
        assert_eq!(AudioSourceNode::new(1, 2).with_gain(f32::NAN).gain(), 0.0);
        assert_eq!(
            AudioSourceNode::new(1, 2).with_gain(f32::INFINITY).gain(),
            0.0
        );
        assert_eq!(
            AudioSourceNode::new(1, 2)
                .with_gain(f32::NEG_INFINITY)
                .gain(),
            0.0
        );
    }
}
