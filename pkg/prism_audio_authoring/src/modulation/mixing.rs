//! Combination rules for overlapping modulation contributions.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the parameter mixing rules of design section 12 (`Mix` as sum,
//! `Multiply`, `Max`, and `Min`). Used by control buses (see
//! `super::control_bus`) and the modulation matrix (see `super::matrix`) when
//! several sources write the same target.

use prism_audio_core::Sample;

/// How a new modulation contribution folds into a running accumulator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum ModMix {
    /// Additive mixing: `acc + value`. This is the default stacking rule.
    Add,
    /// Multiplicative mixing: `acc * value`, useful for gating or scaling.
    Multiply,
    /// Takes the louder of the two: `max(acc, value)`.
    Max,
    /// Takes the quieter of the two: `min(acc, value)`.
    Min,
}

impl ModMix {
    /// Returns the accumulator seed that leaves the first contribution
    /// unchanged for this rule.
    ///
    /// `Add` seeds with `0`, `Multiply` with `1`, `Max` with negative infinity,
    /// and `Min` with positive infinity. Folding the first contribution onto
    /// the identity therefore reproduces that contribution exactly.
    #[inline]
    #[must_use]
    pub fn identity(self) -> Sample {
        match self {
            ModMix::Add => 0.0,
            ModMix::Multiply => 1.0,
            ModMix::Max => Sample::NEG_INFINITY,
            ModMix::Min => Sample::INFINITY,
        }
    }

    /// Folds `value` into the running `acc` according to the rule.
    #[inline]
    #[must_use]
    pub fn combine(self, acc: Sample, value: Sample) -> Sample {
        match self {
            ModMix::Add => acc + value,
            ModMix::Multiply => acc * value,
            ModMix::Max => acc.max(value),
            ModMix::Min => acc.min(value),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: Sample = 1.0e-6;

    #[test]
    fn identity_leaves_first_contribution() {
        for rule in [ModMix::Add, ModMix::Multiply, ModMix::Max, ModMix::Min] {
            let v = 0.37;
            let out = rule.combine(rule.identity(), v);
            assert!((out - v).abs() < EPS, "rule {rule:?} -> {out}");
        }
    }

    #[test]
    fn add_sums() {
        assert!((ModMix::Add.combine(0.25, 0.5) - 0.75).abs() < EPS);
    }

    #[test]
    fn multiply_scales() {
        assert!((ModMix::Multiply.combine(0.5, 0.5) - 0.25).abs() < EPS);
    }

    #[test]
    fn max_and_min_select() {
        assert!((ModMix::Max.combine(0.2, 0.8) - 0.8).abs() < EPS);
        assert!((ModMix::Min.combine(0.2, 0.8) - 0.2).abs() < EPS);
    }
}
