//! Parallel haptic send bus that mixes per-source contributions.
//!
//! Sources feed the haptic path exactly as they feed an aux reverb bus: each
//! registers a send with a gain, and every audio block its signal is summed
//! into a shared mix buffer scaled by that gain. Once all sources for a block
//! have contributed, the accumulated mix is handed to a
//! [`crate::transcode::HapticTranscoder`] to produce the felt waveform. This
//! keeps the haptic signal sample-synchronous with the audio it is derived
//! from while letting each source dial in how strongly it rumbles.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the haptic bus of design section 36 and mirrors the aux-send
//! contract of design section 17. The mix buffer reuses
//! `prism_audio_core::buffer::AudioBuffer` and its `add_scaled` mixing
//! primitive rather than re-implementing summation.

use alloc::vec::Vec;

use prism_audio_core::buffer::{AudioBuffer, ChannelLayout};
use prism_audio_core::math::Sample;

use crate::transcode::HapticTranscoder;
use crate::waveform::HapticWaveform;

/// A single registered send: a source identifier and its haptic gain.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct HapticSend {
    /// Opaque identifier of the contributing source.
    pub source_id: u64,
    /// Linear gain applied to this source before mixing.
    pub gain: Sample,
}

/// Mixes per-source haptic sends into one buffer for the transcoder.
///
/// The bus owns a fixed-capacity mix buffer sized at construction so the hot
/// path never reallocates. Clear it at the start of each block, accumulate
/// every active source, then transcode.
#[derive(Debug, Clone)]
pub struct HapticBus {
    layout: ChannelLayout,
    sends: Vec<HapticSend>,
    mix: AudioBuffer,
}

impl HapticBus {
    /// Builds a bus mixing `layout` buffers of up to `capacity_frames` frames.
    #[must_use]
    pub fn new(layout: ChannelLayout, capacity_frames: usize) -> Self {
        Self {
            layout,
            sends: Vec::new(),
            mix: AudioBuffer::new(layout, capacity_frames),
        }
    }

    /// Returns the channel layout of the mix buffer.
    #[inline]
    #[must_use]
    pub fn layout(&self) -> ChannelLayout {
        self.layout
    }

    /// Returns the number of registered sends.
    #[inline]
    #[must_use]
    pub fn send_count(&self) -> usize {
        self.sends.len()
    }

    /// Registers (or updates) a send for `source_id` with the given `gain`.
    ///
    /// Registering the same source again overwrites its gain rather than
    /// adding a duplicate.
    pub fn add_send(&mut self, source_id: u64, gain: Sample) {
        if let Some(send) = self.sends.iter_mut().find(|s| s.source_id == source_id) {
            send.gain = gain;
        } else {
            self.sends.push(HapticSend { source_id, gain });
        }
    }

    /// Removes the send for `source_id`, returning `true` if one existed.
    pub fn remove_send(&mut self, source_id: u64) -> bool {
        let before = self.sends.len();
        self.sends.retain(|s| s.source_id != source_id);
        self.sends.len() != before
    }

    /// Returns the gain registered for `source_id`, if any.
    #[must_use]
    pub fn send_gain(&self, source_id: u64) -> Option<Sample> {
        self.sends
            .iter()
            .find(|s| s.source_id == source_id)
            .map(|s| s.gain)
    }

    /// Zeroes the mix buffer in preparation for a new block.
    pub fn clear(&mut self) {
        self.mix.clear();
    }

    /// Adds `source`'s signal into the mix, scaled by its registered gain.
    ///
    /// Returns `true` when the source had a send and was mixed in; sources
    /// without a registered send are ignored.
    ///
    /// # Panics
    ///
    /// Panics if `source`'s layout differs from the bus layout.
    pub fn accumulate(&mut self, source_id: u64, source: &AudioBuffer) -> bool {
        if let Some(gain) = self.send_gain(source_id) {
            self.mix.add_scaled(source, gain);
            true
        } else {
            false
        }
    }

    /// Returns the current accumulated mix buffer.
    #[inline]
    #[must_use]
    pub fn mix(&self) -> &AudioBuffer {
        &self.mix
    }

    /// Transcodes the accumulated mix into a haptic waveform.
    #[must_use]
    pub fn transcode(&self, transcoder: &mut HapticTranscoder) -> HapticWaveform {
        transcoder.transcode(&self.mix)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source(value: Sample, frames: usize) -> AudioBuffer {
        let mut buf = AudioBuffer::new(ChannelLayout::Mono, frames);
        for s in buf.channel_mut(0).iter_mut() {
            *s = value;
        }
        buf
    }

    #[test]
    fn add_send_is_idempotent_on_id() {
        let mut bus = HapticBus::new(ChannelLayout::Mono, 8);
        bus.add_send(1, 0.5);
        bus.add_send(1, 0.9);
        assert_eq!(bus.send_count(), 1);
        assert!((bus.send_gain(1).unwrap() - 0.9).abs() < 1e-6);
    }

    #[test]
    fn remove_send_reports_presence() {
        let mut bus = HapticBus::new(ChannelLayout::Mono, 8);
        bus.add_send(7, 1.0);
        assert!(bus.remove_send(7));
        assert!(!bus.remove_send(7));
        assert_eq!(bus.send_count(), 0);
    }

    #[test]
    fn accumulate_applies_gain() {
        let mut bus = HapticBus::new(ChannelLayout::Mono, 4);
        bus.add_send(1, 0.5);
        bus.clear();
        assert!(bus.accumulate(1, &source(1.0, 4)));
        for &s in bus.mix().channel(0) {
            assert!((s - 0.5).abs() < 1e-6);
        }
    }

    #[test]
    fn accumulate_sums_sources() {
        let mut bus = HapticBus::new(ChannelLayout::Mono, 4);
        bus.add_send(1, 0.5);
        bus.add_send(2, 0.25);
        bus.clear();
        bus.accumulate(1, &source(1.0, 4));
        bus.accumulate(2, &source(1.0, 4));
        for &s in bus.mix().channel(0) {
            assert!((s - 0.75).abs() < 1e-6);
        }
    }

    #[test]
    fn accumulate_ignores_unregistered() {
        let mut bus = HapticBus::new(ChannelLayout::Mono, 4);
        bus.clear();
        assert!(!bus.accumulate(42, &source(1.0, 4)));
        for &s in bus.mix().channel(0) {
            assert!(s.abs() < 1e-6);
        }
    }

    #[test]
    fn clear_resets_mix() {
        let mut bus = HapticBus::new(ChannelLayout::Mono, 4);
        bus.add_send(1, 1.0);
        bus.accumulate(1, &source(1.0, 4));
        bus.clear();
        for &s in bus.mix().channel(0) {
            assert!(s.abs() < 1e-6);
        }
    }

    #[test]
    fn transcode_runs_on_mix() {
        use crate::transcode::{HapticTranscoder, TranscodeConfig};
        let mut bus = HapticBus::new(ChannelLayout::Mono, 4_800);
        bus.add_send(1, 1.0);
        bus.clear();
        bus.accumulate(1, &source(0.5, 4_800));
        let mut tc = HapticTranscoder::new(48_000, TranscodeConfig::default());
        let wf = bus.transcode(&mut tc);
        assert_eq!(wf.channel_count(), 2);
        assert_eq!(wf.len(), 100);
    }
}
