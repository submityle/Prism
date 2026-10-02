//! **RTPC** (real-time parameter control): continuous game quantities (speed,
//! health, tension, RPM) that drive engine parameters through authored mapping
//! curves.
//!
//! An RTPC is defined once (id, valid range, default), its live value is held
//! per scope (globally or per game object), and one or more [`RtpcBinding`]s
//! translate that value — through a [`ParameterCurve`] — into concrete
//! [`ParameterSetting`]s on engine targets. The same RTPC can fan out to many
//! targets with different curves (speed → volume *and* low-pass cutoff).
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam
//! Audio, or Google Resonance Audio source or derived code**. The game-value →
//! curve → parameter mapping is reconstructed from first principles as plain
//! data over [`crate::curve`]. No AI/ML.
//!
//! # Relationship
//!
//! [`RtpcRegistry`] holds definitions and bindings; live values live in
//! [`crate::system::EventSystem`]. Evaluating a binding yields a
//! [`ParameterSetting`] (from [`crate::parameter`]) that the resolver forwards
//! as a [`crate::action::ResolvedAction::SetParameter`].

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use prism_audio_core::math::Sample;

use crate::curve::ParameterCurve;
use crate::id::RtpcId;
use crate::parameter::{ParameterSetting, ParameterTarget};

/// Declares an RTPC game parameter: its valid range and default value.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct RtpcDefinition {
    /// Stable id of this parameter.
    pub id: RtpcId,
    /// Lowest meaningful game value.
    pub min: Sample,
    /// Highest meaningful game value.
    pub max: Sample,
    /// Value before the game sets one explicitly.
    pub default: Sample,
}

impl RtpcDefinition {
    /// Builds a definition, ordering the range and clamping the default into it.
    #[must_use]
    pub fn new(id: RtpcId, min: Sample, max: Sample, default: Sample) -> Self {
        let (lo, hi) = if max < min { (max, min) } else { (min, max) };
        Self { id, min: lo, max: hi, default: default.clamp(lo, hi) }
    }

    /// Clamps `value` into this parameter's declared range, neutralising
    /// non-finite inputs to the default.
    #[must_use]
    pub fn clamp(&self, value: Sample) -> Sample {
        if value.is_finite() {
            value.clamp(self.min, self.max)
        } else {
            self.default
        }
    }
}

/// Maps one RTPC's value, through a curve, onto one engine target.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct RtpcBinding {
    /// Which RTPC drives this binding.
    pub rtpc: RtpcId,
    /// The engine parameter it steers.
    pub target: ParameterTarget,
    /// The mapping from game value to target value.
    pub curve: ParameterCurve,
}

impl RtpcBinding {
    /// Builds a binding.
    #[must_use]
    pub fn new(rtpc: RtpcId, target: ParameterTarget, curve: ParameterCurve) -> Self {
        Self { rtpc, target, curve }
    }

    /// Evaluates the binding at a game `value`, returning the resolved setting.
    #[must_use]
    pub fn evaluate(&self, value: Sample) -> ParameterSetting {
        ParameterSetting::new(self.target, self.curve.sample(value))
    }
}

/// Registry of RTPC definitions and their bindings.
///
/// Bindings are indexed by driving RTPC so the runtime can, on a value change,
/// cheaply fetch every target that RTPC affects and emit the resolved settings.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RtpcRegistry {
    definitions: BTreeMap<RtpcId, RtpcDefinition>,
    bindings: BTreeMap<RtpcId, Vec<RtpcBinding>>,
}

impl RtpcRegistry {
    /// Creates an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self { definitions: BTreeMap::new(), bindings: BTreeMap::new() }
    }

    /// Registers (or replaces) an RTPC definition.
    pub fn define(&mut self, def: RtpcDefinition) {
        self.definitions.insert(def.id, def);
    }

    /// Adds a binding from an RTPC to a target.
    pub fn bind(&mut self, binding: RtpcBinding) {
        self.bindings.entry(binding.rtpc).or_default().push(binding);
    }

    /// Returns the definition of `rtpc`, if registered.
    #[must_use]
    pub fn definition(&self, rtpc: RtpcId) -> Option<&RtpcDefinition> {
        self.definitions.get(&rtpc)
    }

    /// Returns the bindings driven by `rtpc` (empty slice if none).
    #[must_use]
    pub fn bindings_for(&self, rtpc: RtpcId) -> &[RtpcBinding] {
        self.bindings.get(&rtpc).map_or(&[], Vec::as_slice)
    }

    /// Evaluates every binding driven by `rtpc` at `value`, pushing the
    /// resolved settings into `out`. Values are clamped to the RTPC's declared
    /// range first when a definition exists.
    pub fn evaluate_into(&self, rtpc: RtpcId, value: Sample, out: &mut Vec<ParameterSetting>) {
        let clamped = self
            .definitions
            .get(&rtpc)
            .map_or(value, |def| def.clamp(value));
        for binding in self.bindings_for(rtpc) {
            out.push(binding.evaluate(clamped));
        }
    }

    /// Number of registered definitions.
    #[must_use]
    pub fn len(&self) -> usize {
        self.definitions.len()
    }

    /// Returns `true` when no RTPC is defined.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.definitions.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::curve::ParameterCurve;
    use alloc::vec::Vec;

    const EPS: Sample = 1e-3;

    fn close(a: Sample, b: Sample) -> bool {
        (a - b).abs() <= EPS
    }

    #[test]
    fn definition_orders_range_and_clamps_default() {
        let d = RtpcDefinition::new(RtpcId::new(1), 10.0, 0.0, 999.0);
        assert!(close(d.min, 0.0));
        assert!(close(d.max, 10.0));
        assert!(close(d.default, 10.0));
    }

    #[test]
    fn clamp_respects_bounds_and_neutralises_non_finite() {
        let d = RtpcDefinition::new(RtpcId::new(1), 0.0, 100.0, 50.0);
        assert!(close(d.clamp(-5.0), 0.0));
        assert!(close(d.clamp(150.0), 100.0));
        assert!(close(d.clamp(Sample::NAN), 50.0));
        assert!(close(d.clamp(Sample::INFINITY), 50.0));
    }

    #[test]
    fn binding_evaluates_through_curve() {
        let curve = ParameterCurve::line(0.0, -60.0, 100.0, 0.0);
        let b = RtpcBinding::new(RtpcId::new(1), ParameterTarget::VolumeDb, curve);
        let s = b.evaluate(50.0);
        assert_eq!(s.target, ParameterTarget::VolumeDb);
        assert!(close(s.value, -30.0));
    }

    #[test]
    fn evaluate_into_fans_out_and_clamps_domain() {
        let mut reg = RtpcRegistry::new();
        reg.define(RtpcDefinition::new(RtpcId::new(1), 0.0, 100.0, 0.0));
        reg.bind(RtpcBinding::new(
            RtpcId::new(1),
            ParameterTarget::VolumeDb,
            ParameterCurve::line(0.0, -60.0, 100.0, 0.0),
        ));
        reg.bind(RtpcBinding::new(
            RtpcId::new(1),
            ParameterTarget::LowpassCutoffHz,
            ParameterCurve::line(0.0, 200.0, 100.0, 20_000.0),
        ));
        let mut out = Vec::new();
        // 150 is above the declared max and must be clamped to 100 first.
        reg.evaluate_into(RtpcId::new(1), 150.0, &mut out);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].target, ParameterTarget::VolumeDb);
        assert!(close(out[0].value, 0.0));
        assert_eq!(out[1].target, ParameterTarget::LowpassCutoffHz);
        assert!((out[1].value - 20_000.0).abs() <= 1.0);
    }

    #[test]
    fn unknown_rtpc_has_no_bindings_and_registry_tracks_len() {
        let mut reg = RtpcRegistry::new();
        assert!(reg.is_empty());
        assert!(reg.bindings_for(RtpcId::new(9)).is_empty());
        reg.define(RtpcDefinition::new(RtpcId::new(1), 0.0, 1.0, 0.0));
        assert!(!reg.is_empty());
        assert_eq!(reg.len(), 1);
        assert!(reg.definition(RtpcId::new(1)).is_some());
        assert!(reg.definition(RtpcId::new(2)).is_none());
    }

    #[test]
    fn evaluate_into_without_definition_uses_raw_value() {
        let mut reg = RtpcRegistry::new();
        // Bind without a definition: value passes through unclamped.
        reg.bind(RtpcBinding::new(
            RtpcId::new(5),
            ParameterTarget::Pan,
            ParameterCurve::line(0.0, -1.0, 1.0, 1.0),
        ));
        let mut out = Vec::new();
        reg.evaluate_into(RtpcId::new(5), 0.5, &mut out);
        assert_eq!(out.len(), 1);
        assert!(close(out[0].value, 0.0));
    }
}
