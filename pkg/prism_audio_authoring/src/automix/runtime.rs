//! The compiled, tickable auto-mix state driven on the modulation matrix.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! CRIWARE, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Realizes the runtime half of design section 46.8. A [`CompiledAutoMix`] owns
//! the design section 12 [`ModMatrix`] produced by [`super::compiler`] plus one
//! [`super::activity::ActivityEnvelope`] per ducking rule. Each block the host
//! reports per-category activity, the envelopes apply attack/release
//! ballistics, the matrix folds every rule's shaped reduction with
//! [`crate::modulation::ModMix::Max`], and the resolved reduction becomes a
//! per-category gain multiplier. All smoothing is block-rate and deterministic
//! so output is golden-reproducible (design section 24).

use alloc::vec;
use alloc::vec::Vec;

use prism_audio_core::Sample;

use crate::modulation::source::ModContext;
use crate::modulation::{BusId, ModMatrix};

use super::activity::ActivityEnvelope;
use super::category::CategoryId;

/// The runtime wiring for one compiled [`super::rule::DuckingRule`].
///
/// Binds the trigger category whose activity drives the duck to the modulation
/// input bus the smoothed activity is written to, together with the per-rule
/// attack/release envelope.
#[derive(Debug, Clone, Copy)]
pub(super) struct RuleWiring {
    /// Category whose raw activity feeds this rule's envelope.
    trigger: CategoryId,
    /// Modulation input bus that carries the smoothed activity into the matrix.
    input_bus: BusId,
    /// Per-rule attack/release ballistics.
    envelope: ActivityEnvelope,
}

impl RuleWiring {
    /// Builds a wiring from its parts; called by [`super::compiler`].
    pub(super) fn new(trigger: CategoryId, input_bus: BusId, envelope: ActivityEnvelope) -> Self {
        Self {
            trigger,
            input_bus,
            envelope,
        }
    }
}

/// A compiled, tickable declarative auto-mix.
///
/// Produced by [`super::ruleset::AutoMixRuleset::compile`]. The host sets
/// per-category activity each block via [`CompiledAutoMix::set_activity`] (or
/// [`CompiledAutoMix::set_active`]), advances with
/// [`CompiledAutoMix::advance`], and reads the resulting per-category gain with
/// [`CompiledAutoMix::gain`].
#[derive(Debug)]
pub struct CompiledAutoMix {
    /// The modulation matrix carrying input buses, reduction buses, and routes.
    matrix: ModMatrix,
    /// Reduction bus per category, indexed by [`CategoryId`].
    reduction_bus: Vec<BusId>,
    /// Per-rule wirings (trigger, input bus, envelope).
    rules: Vec<RuleWiring>,
    /// Latest host-reported raw activity per category, indexed by id.
    raw_activity: Vec<Sample>,
    /// Output sample rate in Hz.
    sample_rate: u32,
}

impl CompiledAutoMix {
    /// Assembles a compiled auto-mix from parts; called by [`super::compiler`].
    pub(super) fn assemble(
        matrix: ModMatrix,
        reduction_bus: Vec<BusId>,
        rules: Vec<RuleWiring>,
        category_count: usize,
        sample_rate: u32,
    ) -> Self {
        Self {
            matrix,
            reduction_bus,
            rules,
            raw_activity: vec![0.0; category_count],
            sample_rate,
        }
    }

    /// Returns the number of categories this auto-mix governs.
    #[must_use]
    pub fn category_count(&self) -> usize {
        self.reduction_bus.len()
    }

    /// Sets the raw activity level of `category` for the next block.
    ///
    /// `raw` is clamped to be non-negative; `0` means idle and `1` means fully
    /// active. Out-of-range ids are ignored.
    pub fn set_activity(&mut self, category: CategoryId, raw: Sample) {
        if let Some(slot) = self.raw_activity.get_mut(category.0) {
            *slot = raw.max(0.0);
        }
    }

    /// Sets `category` fully active (`1.0`) or idle (`0.0`).
    pub fn set_active(&mut self, category: CategoryId, active: bool) {
        self.set_activity(category, if active { 1.0 } else { 0.0 });
    }

    /// Advances the auto-mix by `frames`, updating every category's gain.
    ///
    /// Each rule's envelope tracks its trigger's raw activity and the smoothed
    /// result is written onto the rule's modulation input bus; the matrix then
    /// resolves every reduction bus as the strongest shaped reduction targeting
    /// it.
    ///
    /// # Panics
    ///
    /// Panics if `frames` is zero.
    pub fn advance(&mut self, frames: u32) {
        assert!(frames > 0, "frames must be non-zero");
        for wiring in &mut self.rules {
            let raw = self
                .raw_activity
                .get(wiring.trigger.0)
                .copied()
                .unwrap_or(0.0);
            let smoothed = wiring.envelope.process_block(raw, frames);
            self.matrix.set_bus_base(wiring.input_bus, smoothed);
        }
        let ctx = ModContext::new(self.sample_rate, frames);
        self.matrix.tick(&ctx);
    }

    /// Returns the current reduction fraction applied to `category` in `[0, 1)`.
    ///
    /// `0` means no reduction; larger values mean the category is more ducked.
    #[must_use]
    pub fn reduction(&self, category: CategoryId) -> Sample {
        self.reduction_bus
            .get(category.0)
            .map_or(0.0, |&bus| self.matrix.bus_value(bus).clamp(0.0, 1.0))
    }

    /// Returns the current linear gain multiplier for `category` in `(0, 1]`.
    ///
    /// This is `1 - reduction`, the factor to apply to the category's bus gain.
    #[must_use]
    pub fn gain(&self, category: CategoryId) -> Sample {
        (1.0 - self.reduction(category)).clamp(0.0, 1.0)
    }

    /// Returns the current gain for `category` in decibels (`<= 0`).
    ///
    /// A fully recovered category returns `0` dB; silence returns
    /// [`Sample::NEG_INFINITY`].
    #[must_use]
    pub fn gain_db(&self, category: CategoryId) -> Sample {
        let gain = self.gain(category);
        if gain <= 0.0 {
            return Sample::NEG_INFINITY;
        }
        /// `20 / ln(10)`, converting a natural-log amplitude ratio to decibels.
        const DB_PER_LN: Sample = 8.685_889;
        DB_PER_LN * bevy_math::ops::ln(gain)
    }

    /// Resets every envelope, activity reading, and bus to its initial state.
    pub fn reset(&mut self) {
        for wiring in &mut self.rules {
            wiring.envelope.reset();
            self.matrix.set_bus_base(wiring.input_bus, 0.0);
        }
        for slot in &mut self.raw_activity {
            *slot = 0.0;
        }
        self.matrix.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::automix::rule::{DuckingRule, decibels_to_linear};
    use crate::automix::ruleset::AutoMixRuleset;

    const SR: u32 = 48_000;
    const BLOCK: u32 = 256;

    fn close(a: Sample, b: Sample) -> bool {
        (a - b).abs() < 1e-3
    }

    fn dialogue_ducks_music() -> (CompiledAutoMix, CategoryId, CategoryId) {
        let mut set = AutoMixRuleset::new();
        let dialogue = set.add_category("dialogue", 100);
        let music = set.add_category("music", 10);
        set.add_rule(
            DuckingRule::new(dialogue, music, 12.0)
                .with_attack(0.01)
                .with_release(0.1),
        );
        let mix = set.compile(SR).expect("ruleset compiles");
        (mix, dialogue, music)
    }

    #[test]
    fn idle_leaves_unity_gain() {
        let (mut mix, _dialogue, music) = dialogue_ducks_music();
        for _ in 0..50 {
            mix.advance(BLOCK);
        }
        assert!(close(mix.gain(music), 1.0));
        assert!(close(mix.reduction(music), 0.0));
    }

    #[test]
    fn active_trigger_ducks_target_to_floor() {
        let (mut mix, dialogue, music) = dialogue_ducks_music();
        mix.set_active(dialogue, true);
        for _ in 0..200 {
            mix.advance(BLOCK);
        }
        let floor = decibels_to_linear(-12.0);
        assert!(close(mix.gain(music), floor));
        // The trigger itself is never ducked by this ruleset.
        assert!(close(mix.gain(dialogue), 1.0));
    }

    #[test]
    fn release_recovers_toward_unity() {
        let (mut mix, dialogue, music) = dialogue_ducks_music();
        mix.set_active(dialogue, true);
        for _ in 0..200 {
            mix.advance(BLOCK);
        }
        let ducked = mix.gain(music);
        mix.set_active(dialogue, false);
        for _ in 0..5 {
            mix.advance(BLOCK);
        }
        let recovering = mix.gain(music);
        assert!(recovering > ducked);
        for _ in 0..200 {
            mix.advance(BLOCK);
        }
        assert!(close(mix.gain(music), 1.0));
    }

    #[test]
    fn strongest_duck_wins_when_two_triggers_target_one() {
        let mut set = AutoMixRuleset::new();
        let dialogue = set.add_category("dialogue", 100);
        let ui = set.add_category("ui", 80);
        let music = set.add_category("music", 10);
        set.add_rule(DuckingRule::new(dialogue, music, 12.0).with_attack(0.005));
        set.add_rule(DuckingRule::new(ui, music, 6.0).with_attack(0.005));
        let mut mix = set.compile(SR).expect("compiles");
        mix.set_active(dialogue, true);
        mix.set_active(ui, true);
        for _ in 0..200 {
            mix.advance(BLOCK);
        }
        // The 12 dB duck dominates the 6 dB duck (max reduction / min gain).
        assert!(close(mix.gain(music), decibels_to_linear(-12.0)));
    }

    #[test]
    fn gain_db_matches_attenuation() {
        let (mut mix, dialogue, music) = dialogue_ducks_music();
        mix.set_active(dialogue, true);
        for _ in 0..300 {
            mix.advance(BLOCK);
        }
        assert!(close(mix.gain_db(music), -12.0));
    }

    #[test]
    fn reset_restores_unity() {
        let (mut mix, dialogue, music) = dialogue_ducks_music();
        mix.set_active(dialogue, true);
        for _ in 0..200 {
            mix.advance(BLOCK);
        }
        assert!(mix.gain(music) < 0.5);
        mix.reset();
        // After reset, one settled tick returns to unity with no activity.
        mix.advance(BLOCK);
        assert!(close(mix.gain(music), 1.0));
    }
}
