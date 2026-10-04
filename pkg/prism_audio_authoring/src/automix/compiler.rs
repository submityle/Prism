//! Lowering a declarative [`AutoMixRuleset`] onto the design section 12
//! modulation matrix.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! CRIWARE, or Google Resonance Audio source or derived code; no AI/ML. The
//! category-ducking idea is a classical mixing technique (side-chain ducking by
//! content class); this compiler is an independent implementation that lowers
//! it, as data, onto a generic modulation matrix.
//!
//! # Relationship
//! Implements the compile step of design section 46.8.
//! [`AutoMixRuleset::compile`] validates the ruleset, then builds a
//! [`crate::modulation::ModMatrix`] with one reduction bus per
//! [`super::category::Category`] and, per [`super::rule::DuckingRule`], one
//! activity input bus plus a [`crate::modulation::ModMix::Max`]-folded, unipolar
//! modulation route carrying the rule's shaped reduction onto its target's
//! reduction bus. The resulting [`super::runtime::CompiledAutoMix`] realizes the
//! runtime half (see [`super::runtime`]) and stays golden-reproducible (design
//! section 24).

use alloc::format;
use alloc::vec::Vec;

use crate::modulation::{ModMatrix, ModMix, ModRoute, Polarity, RouteInput};

use super::activity::ActivityEnvelope;
use super::ruleset::{AutoMixError, AutoMixRuleset};
use super::runtime::{CompiledAutoMix, RuleWiring};

impl AutoMixRuleset {
    /// Compiles this ruleset into a tickable [`CompiledAutoMix`] for
    /// `sample_rate`.
    ///
    /// Validates the ruleset (see [`AutoMixRuleset::validate`]), then lowers it
    /// onto a fresh [`ModMatrix`]:
    ///
    /// - one **reduction bus** per category (base `0`, pushed in
    ///   [`super::category::CategoryId`] order so the runtime can address
    ///   `reduction_bus[category.0]` directly) that resolves to the strongest
    ///   reduction targeting that category;
    /// - one **activity input bus** per rule (base `0`) that the runtime fills
    ///   each block with the rule trigger's smoothed activity;
    /// - one **route** per rule carrying its input bus through the rule's curve
    ///   at depth [`super::rule::DuckingRule::reduction_depth`], folded with
    ///   [`ModMix::Max`] onto the target's reduction bus so the loudest declared
    ///   duck wins when several triggers target one category.
    ///
    /// # Errors
    ///
    /// Returns the first [`AutoMixError`] produced by validation when a rule
    /// references an unknown category or ducks a category against itself.
    pub fn compile(&self, sample_rate: u32) -> Result<CompiledAutoMix, AutoMixError> {
        self.validate()?;

        let categories = self.categories();
        let category_count = categories.len();

        let mut matrix = ModMatrix::new();

        // One reduction bus per category, in CategoryId order.
        let mut reduction_bus = Vec::with_capacity(category_count);
        for (id, _category) in categories.iter() {
            let bus = matrix.add_bus(&format!("automix.reduction.{}", id.0), 0.0);
            reduction_bus.push(bus);
        }

        // One activity input bus, route, and envelope per rule.
        let rules = self.rules();
        let mut wirings = Vec::with_capacity(rules.len());
        for (rule_index, rule) in rules.iter().enumerate() {
            let input_bus = matrix.add_bus(&format!("automix.input.{rule_index}"), 0.0);
            let route = ModRoute::new(
                RouteInput::Bus(input_bus),
                reduction_bus[rule.target.0],
                rule.reduction_depth(),
            )
            .with_polarity(Polarity::Unipolar)
            .with_curve(rule.curve.clone())
            .with_mix(ModMix::Max);
            matrix.add_route(route);

            let envelope =
                ActivityEnvelope::new(sample_rate, rule.attack_seconds, rule.release_seconds);
            wirings.push(RuleWiring::new(rule.trigger, input_bus, envelope));
        }

        Ok(CompiledAutoMix::assemble(
            matrix,
            reduction_bus,
            wirings,
            category_count,
            sample_rate,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::automix::rule::DuckingRule;
    use prism_audio_core::Sample;

    const SR: u32 = 48_000;
    const BLOCK: u32 = 256;

    fn close(a: Sample, b: Sample) -> bool {
        (a - b).abs() < 1e-6
    }

    #[test]
    fn compile_reports_category_count() {
        let mut set = AutoMixRuleset::new();
        let dialogue = set.add_category("dialogue", 100);
        let music = set.add_category("music", 10);
        set.add_rule(DuckingRule::new(dialogue, music, 9.0));
        let mix = set.compile(SR).expect("ruleset compiles");
        assert_eq!(mix.category_count(), 2);
    }

    #[test]
    fn empty_ruleset_compiles_to_no_categories() {
        let set = AutoMixRuleset::new();
        let mix = set.compile(SR).expect("empty ruleset compiles");
        assert_eq!(mix.category_count(), 0);
    }

    #[test]
    fn category_without_rules_stays_at_unity() {
        let mut set = AutoMixRuleset::new();
        let music = set.add_category("music", 10);
        let mut mix = set.compile(SR).expect("ruleset compiles");
        mix.advance(BLOCK);
        assert!(close(mix.gain(music), 1.0));
    }

    #[test]
    fn self_duck_fails_to_compile() {
        let mut set = AutoMixRuleset::new();
        let a = set.add_category("a", 1);
        set.add_rule(DuckingRule::new(a, a, 6.0));
        assert_eq!(
            set.compile(SR).expect_err("self-duck is rejected"),
            AutoMixError::SelfDuck {
                rule_index: 0,
                category: a,
            }
        );
    }
}
