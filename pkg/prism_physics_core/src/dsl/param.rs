//! Named, range-bounded parameter storage for editor-driven tuning.
//!
//! A [`ParameterStore`] holds named parameters, each with a value and an
//! inclusive `[min, max]` slider range. Writes are clamped to the range, so an
//! editor slider (or a hot-reload update) can never push a parameter outside
//! its declared bounds. Vector parameters clamp component-wise.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. A
//! name-keyed parameter table with clamped setters is a generic, publicly
//! documented data-management pattern.

use glam::Vec3;

use crate::math::scalar::Real;

/// A typed parameter value.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum ParamValue {
    /// A scalar parameter.
    Real(Real),
    /// A three-component vector parameter.
    Vector(Vec3),
    /// A boolean parameter.
    Bool(bool),
}

/// A parameter's current value and its editable range.
///
/// The range applies to scalar and vector parameters; boolean parameters
/// ignore it.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ParamSpec {
    /// The current (already clamped) value.
    pub value: ParamValue,
    /// Inclusive lower bound for scalar/vector components.
    pub min: Real,
    /// Inclusive upper bound for scalar/vector components.
    pub max: Real,
}

fn clamp_value(value: ParamValue, min: Real, max: Real) -> ParamValue {
    let (lo, hi) = if min <= max { (min, max) } else { (max, min) };
    match value {
        ParamValue::Real(x) => ParamValue::Real(x.clamp(lo, hi)),
        ParamValue::Vector(v) => ParamValue::Vector(Vec3::new(
            v.x.clamp(lo, hi),
            v.y.clamp(lo, hi),
            v.z.clamp(lo, hi),
        )),
        ParamValue::Bool(b) => ParamValue::Bool(b),
    }
}

/// A collection of named, range-bounded parameters.
///
/// Backed by an insertion-ordered vector; parameter counts are small (editor
/// sliders), so linear lookup is more than adequate and keeps the type
/// trivially serializable.
#[derive(Clone, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ParameterStore {
    names: Vec<String>,
    specs: Vec<ParamSpec>,
}

impl ParameterStore {
    /// Creates an empty parameter store.
    #[must_use]
    pub fn new() -> Self {
        ParameterStore::default()
    }

    fn index_of(&self, name: &str) -> Option<usize> {
        self.names.iter().position(|n| n == name)
    }

    fn insert(&mut self, name: &str, spec: ParamSpec) {
        if let Some(i) = self.index_of(name) {
            self.specs[i] = spec;
        } else {
            self.names.push(name.to_string());
            self.specs.push(spec);
        }
    }

    /// Defines (or replaces) a scalar parameter, clamping `value` into
    /// `[min, max]`.
    pub fn define_real(&mut self, name: &str, value: Real, min: Real, max: Real) {
        let value = clamp_value(ParamValue::Real(value), min, max);
        self.insert(name, ParamSpec { value, min, max });
    }

    /// Defines (or replaces) a vector parameter, clamping each component into
    /// `[min, max]`.
    pub fn define_vector(&mut self, name: &str, value: Vec3, min: Real, max: Real) {
        let value = clamp_value(ParamValue::Vector(value), min, max);
        self.insert(name, ParamSpec { value, min, max });
    }

    /// Defines (or replaces) a boolean parameter.
    pub fn define_bool(&mut self, name: &str, value: bool) {
        self.insert(
            name,
            ParamSpec {
                value: ParamValue::Bool(value),
                min: 0.0,
                max: 1.0,
            },
        );
    }

    /// Sets a scalar parameter, clamping to its stored range.
    ///
    /// Returns `true` if the parameter exists and is scalar.
    pub fn set_real(&mut self, name: &str, value: Real) -> bool {
        if let Some(i) = self.index_of(name) {
            let spec = &mut self.specs[i];
            if matches!(spec.value, ParamValue::Real(_)) {
                spec.value = clamp_value(ParamValue::Real(value), spec.min, spec.max);
                return true;
            }
        }
        false
    }

    /// Sets a vector parameter, clamping each component to its stored range.
    ///
    /// Returns `true` if the parameter exists and is a vector.
    pub fn set_vector(&mut self, name: &str, value: Vec3) -> bool {
        if let Some(i) = self.index_of(name) {
            let spec = &mut self.specs[i];
            if matches!(spec.value, ParamValue::Vector(_)) {
                spec.value = clamp_value(ParamValue::Vector(value), spec.min, spec.max);
                return true;
            }
        }
        false
    }

    /// Sets a boolean parameter.
    ///
    /// Returns `true` if the parameter exists and is boolean.
    pub fn set_bool(&mut self, name: &str, value: bool) -> bool {
        if let Some(i) = self.index_of(name) {
            let spec = &mut self.specs[i];
            if matches!(spec.value, ParamValue::Bool(_)) {
                spec.value = ParamValue::Bool(value);
                return true;
            }
        }
        false
    }

    /// Returns the scalar value of `name`, if it is a defined scalar.
    #[must_use]
    pub fn get_real(&self, name: &str) -> Option<Real> {
        match self.specs[self.index_of(name)?].value {
            ParamValue::Real(x) => Some(x),
            _ => None,
        }
    }

    /// Returns the vector value of `name`, if it is a defined vector.
    #[must_use]
    pub fn get_vector(&self, name: &str) -> Option<Vec3> {
        match self.specs[self.index_of(name)?].value {
            ParamValue::Vector(v) => Some(v),
            _ => None,
        }
    }

    /// Returns the boolean value of `name`, if it is a defined boolean.
    #[must_use]
    pub fn get_bool(&self, name: &str) -> Option<bool> {
        match self.specs[self.index_of(name)?].value {
            ParamValue::Bool(b) => Some(b),
            _ => None,
        }
    }

    /// Returns the full [`ParamSpec`] for `name`, if defined.
    #[must_use]
    pub fn spec(&self, name: &str) -> Option<ParamSpec> {
        self.index_of(name).map(|i| self.specs[i])
    }

    /// Returns `true` if a parameter named `name` is defined.
    #[must_use]
    pub fn contains(&self, name: &str) -> bool {
        self.index_of(name).is_some()
    }

    /// Returns the parameter names in definition order.
    #[must_use]
    pub fn names(&self) -> &[String] {
        &self.names
    }

    /// Returns the number of defined parameters.
    #[must_use]
    pub fn len(&self) -> usize {
        self.names.len()
    }

    /// Returns `true` when no parameters are defined.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_real_clamps_into_range() {
        let mut store = ParameterStore::new();
        store.define_real("stiffness", 0.5, 0.0, 1.0);
        assert!(store.set_real("stiffness", 5.0));
        assert!((store.get_real("stiffness").unwrap() - 1.0).abs() < 1e-6);
        assert!(store.set_real("stiffness", -3.0));
        assert!((store.get_real("stiffness").unwrap() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn vector_clamps_componentwise() {
        let mut store = ParameterStore::new();
        store.define_vector("gravity", Vec3::new(0.0, -50.0, 0.0), -10.0, 10.0);
        let g = store.get_vector("gravity").unwrap();
        assert!((g.y + 10.0).abs() < 1e-6);
    }

    #[test]
    fn type_mismatched_setters_fail() {
        let mut store = ParameterStore::new();
        store.define_real("x", 1.0, 0.0, 2.0);
        assert!(!store.set_bool("x", true));
        assert!(store.get_bool("x").is_none());
    }

    #[test]
    fn bookkeeping() {
        let mut store = ParameterStore::new();
        assert!(store.is_empty());
        store.define_bool("enabled", true);
        assert!(store.contains("enabled"));
        assert_eq!(store.len(), 1);
        assert_eq!(store.get_bool("enabled"), Some(true));
    }
}
