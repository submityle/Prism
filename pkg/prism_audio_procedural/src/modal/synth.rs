//! The modal synthesiser: a block renderer over a modal bank.
//!
//! [`ModalSynth`] is the real-time face of the modal stage. It owns a
//! [`ModalBank`], a shared [`TransientBurst`] contact-noise generator, and a
//! preallocated queue of sample-accurate impacts. Impacts scheduled for the
//! current block are injected at their exact sample offset; a per-sample
//! continuous drive (friction/rolling noise) can be mixed in to colour the same
//! resonators, so discrete strikes and continuous contact share one bank
//! exactly as the design requires. The render loop performs no allocation, no
//! locking, and no panic: the impact queue and all buffers are sized once at
//! construction.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the real-time contract of design section 47.2; aggregates
//! [`crate::modal::bank`] and [`crate::modal::excitation`] and is driven by the
//! continuous stage ([`crate::continuous`]).

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::contact::event::ContactPoint;
use crate::modal::bank::{Mode, ModalBank};
use crate::modal::excitation::TransientBurst;
use prism_audio_core::math::Sample;

/// A sample-accurate impact queued for the current block.
#[derive(Clone, Copy, Debug)]
struct ScheduledImpact {
    sample_offset: u32,
    amplitude: Sample,
    point: ContactPoint,
    brightness: Sample,
    transient_len: u32,
}

/// Dry mix of the contact transient added alongside the resonated output.
const DRY_TRANSIENT_MIX: Sample = 0.3;

/// A block-rate modal synthesiser.
#[derive(Clone, Debug)]
pub struct ModalSynth {
    bank: ModalBank,
    transient: TransientBurst,
    queue: Vec<ScheduledImpact>,
    impact_budget: usize,
}

impl ModalSynth {
    /// Creates a synth with a bank of `mode_capacity` modes, an impact queue of
    /// `impact_budget` entries, and a transient generator seeded with `seed`.
    #[must_use]
    pub fn new(sample_rate: u32, mode_capacity: usize, impact_budget: usize, seed: u64) -> Self {
        let impact_budget = impact_budget.max(1);
        Self {
            bank: ModalBank::new(sample_rate, mode_capacity),
            transient: TransientBurst::new(seed),
            queue: Vec::with_capacity(impact_budget),
            impact_budget,
        }
    }

    /// Loads the mode table (off the audio thread).
    #[inline]
    pub fn configure(&mut self, modes: &[Mode]) {
        self.bank.configure(modes);
    }

    /// Caps the number of audible modes (quality governor hook).
    #[inline]
    pub fn set_quality_cap(&mut self, n: usize) {
        self.bank.set_quality_cap(n);
    }

    /// Returns a shared reference to the underlying bank.
    #[inline]
    #[must_use]
    pub fn bank(&self) -> &ModalBank {
        &self.bank
    }

    /// Queues an impact at `sample_offset` within the next rendered block.
    ///
    /// Over-budget schedules are dropped (the budget is enforced upstream by the
    /// contact bus), keeping the queue allocation-free.
    pub fn schedule_impact(
        &mut self,
        sample_offset: u32,
        amplitude: Sample,
        point: ContactPoint,
        brightness: Sample,
        transient_len: u32,
    ) {
        if self.queue.len() >= self.impact_budget {
            return;
        }
        if !amplitude.is_finite() || amplitude <= 0.0 {
            return;
        }
        self.queue.push(ScheduledImpact {
            sample_offset,
            amplitude,
            point,
            brightness,
            transient_len,
        });
    }

    /// Renders `frames` samples into `out`, injecting any queued impacts and
    /// mixing in the optional per-sample `drive` (friction/rolling excitation).
    ///
    /// The queue is consumed and cleared. `out` is written for `frames` samples
    /// (clamped to its length); `drive`, when present, is read for the same
    /// span.
    pub fn render(&mut self, frames: usize, drive: Option<&[Sample]>, out: &mut [Sample]) {
        let frames = frames.min(out.len());
        if frames == 0 {
            self.queue.clear();
            return;
        }
        // Order impacts by offset and clamp into the block, then walk a cursor.
        let last = (frames - 1) as u32;
        for imp in &mut self.queue {
            imp.sample_offset = imp.sample_offset.min(last);
        }
        self.queue.sort_unstable_by_key(|i| i.sample_offset);
        let mut cursor = 0usize;
        for (n, slot) in out.iter_mut().take(frames).enumerate() {
            let n = n as u32;
            while cursor < self.queue.len() && self.queue[cursor].sample_offset <= n {
                let imp = self.queue[cursor];
                self.bank.excite_impact(imp.amplitude, imp.point, imp.brightness);
                self.transient.trigger(imp.amplitude, imp.transient_len);
                cursor += 1;
            }
            let drive_n = drive.and_then(|d| d.get(n as usize)).copied().unwrap_or(0.0);
            let transient_n = self.transient.tick();
            let bank_out = self.bank.tick(drive_n + transient_n);
            *slot = bank_out + DRY_TRANSIENT_MIX * transient_n;
        }
        self.queue.clear();
    }

    /// Returns `true` while the bank still rings above `threshold`.
    #[inline]
    #[must_use]
    pub fn is_ringing(&self, threshold: Sample) -> bool {
        self.bank.is_ringing(threshold) || self.transient.is_active()
    }

    /// Silences the synth.
    pub fn reset(&mut self) {
        self.bank.reset();
        self.queue.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn modes() -> [Mode; 3] {
        [
            Mode::new(220.0, 0.5, 1.0),
            Mode::new(440.0, 0.4, 0.7),
            Mode::new(880.0, 0.3, 0.5),
        ]
    }

    #[test]
    fn impact_produces_output() {
        let mut synth = ModalSynth::new(48_000, 16, 8, 1);
        synth.configure(&modes());
        synth.schedule_impact(0, 1.0, ContactPoint::new(0.5), 0.6, 32);
        let mut out = alloc::vec![0.0; 1024];
        synth.render(1024, None, &mut out);
        let peak = out.iter().fold(0.0f32, |m, &s| m.max(s.abs()));
        assert!(peak > 0.01, "peak={peak}");
    }

    #[test]
    fn same_schedule_same_output() {
        let render = || {
            let mut synth = ModalSynth::new(48_000, 16, 8, 7);
            synth.configure(&modes());
            synth.schedule_impact(10, 1.0, ContactPoint::new(0.3), 0.5, 24);
            synth.schedule_impact(200, 0.6, ContactPoint::new(0.8), 0.9, 24);
            let mut out = alloc::vec![0.0; 512];
            synth.render(512, None, &mut out);
            out
        };
        assert_eq!(render(), render());
    }

    #[test]
    fn sample_offset_delays_onset() {
        let mut synth = ModalSynth::new(48_000, 16, 8, 1);
        synth.configure(&modes());
        synth.schedule_impact(256, 1.0, ContactPoint::new(0.5), 0.6, 16);
        let mut out = alloc::vec![0.0; 512];
        synth.render(512, None, &mut out);
        let before = out[..200].iter().fold(0.0f32, |m, &s| m.max(s.abs()));
        let after = out[256..].iter().fold(0.0f32, |m, &s| m.max(s.abs()));
        assert!(before < 1e-6, "before={before}");
        assert!(after > 0.01, "after={after}");
    }

    #[test]
    fn drive_colours_output() {
        let mut synth = ModalSynth::new(48_000, 16, 8, 1);
        synth.configure(&modes());
        let drive = alloc::vec![0.5; 512];
        let mut out = alloc::vec![0.0; 512];
        synth.render(512, Some(&drive), &mut out);
        let peak = out.iter().fold(0.0f32, |m, &s| m.max(s.abs()));
        assert!(peak > 0.0);
    }

    #[test]
    fn budget_limits_queue() {
        let mut synth = ModalSynth::new(48_000, 16, 2, 1);
        synth.configure(&modes());
        for _ in 0..10 {
            synth.schedule_impact(0, 1.0, ContactPoint::new(0.5), 0.5, 8);
        }
        // Should not panic or grow unbounded.
        let mut out = alloc::vec![0.0; 64];
        synth.render(64, None, &mut out);
    }
}
