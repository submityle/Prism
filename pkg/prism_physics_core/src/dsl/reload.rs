//! Hot-reload registry for live constraint editing.
//!
//! A [`HotReloadRegistry`] owns the DSL source string, the currently compiled
//! [`CompiledConstraint`], and a [`ParameterStore`]. Editors call
//! [`reload_source`](HotReloadRegistry::reload_source) after a source edit and
//! [`set_param`](HotReloadRegistry::set_param) after moving a slider; both bump
//! a monotonically increasing version so consumers can cheaply detect when
//! they must refresh. Crucially, a failed recompile keeps the previously good
//! program intact and leaves the version unchanged, so a typo in the editor
//! never crashes the running simulation.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. Versioned
//! hot reloading with atomic swap-on-success is a generic, publicly documented
//! live-editing pattern.

use crate::dsl::compiler::{compile_constraint, CompiledConstraint};
use crate::dsl::error::DslError;
use crate::dsl::param::{ParamValue, ParameterStore};
use crate::dsl::parser::parse_constraint;

/// Compiles `source` into a [`CompiledConstraint`], surfacing lex/parse/compile
/// errors as a single [`DslError`].
fn compile_source(source: &str) -> Result<CompiledConstraint, DslError> {
    let decl = parse_constraint(source)?;
    compile_constraint(&decl)
}

/// Owns the live source, its compiled program, and tunable parameters.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct HotReloadRegistry {
    source: String,
    compiled: CompiledConstraint,
    params: ParameterStore,
    version: u64,
    dirty: bool,
}

impl HotReloadRegistry {
    /// Creates a registry from an initial `source` and parameter store.
    ///
    /// # Errors
    ///
    /// Returns a [`DslError`] if the initial source fails to compile.
    pub fn new(source: &str, params: ParameterStore) -> Result<Self, DslError> {
        let compiled = compile_source(source)?;
        Ok(HotReloadRegistry {
            source: source.to_string(),
            compiled,
            params,
            version: 1,
            dirty: false,
        })
    }

    /// Recompiles from new `source`.
    ///
    /// On success the source and program are swapped in, the version is bumped,
    /// and the dirty flag is set. On failure the previous program and version
    /// are preserved and the error is returned.
    ///
    /// # Errors
    ///
    /// Returns a [`DslError`] if the new source fails to compile; the existing
    /// state is untouched in that case.
    pub fn reload_source(&mut self, source: &str) -> Result<(), DslError> {
        let compiled = compile_source(source)?;
        self.compiled = compiled;
        self.source = source.to_string();
        self.version += 1;
        self.dirty = true;
        Ok(())
    }

    /// Updates a parameter, bumping the version and setting the dirty flag.
    ///
    /// # Errors
    ///
    /// Returns [`DslError::Compile`] if `name` is unknown or its type does not
    /// match `value`.
    pub fn set_param(&mut self, name: &str, value: ParamValue) -> Result<(), DslError> {
        let ok = match value {
            ParamValue::Real(x) => self.params.set_real(name, x),
            ParamValue::Vector(v) => self.params.set_vector(name, v),
            ParamValue::Bool(b) => self.params.set_bool(name, b),
        };
        if !ok {
            return Err(DslError::compile(format!(
                "unknown or type-mismatched parameter '{name}'"
            )));
        }
        self.version += 1;
        self.dirty = true;
        Ok(())
    }

    /// Returns the current version (bumped on every successful reload or param
    /// change).
    #[must_use]
    pub fn version(&self) -> u64 {
        self.version
    }

    /// Returns the current source text.
    #[must_use]
    pub fn source(&self) -> &str {
        &self.source
    }

    /// Returns the currently compiled constraint.
    #[must_use]
    pub fn compiled(&self) -> &CompiledConstraint {
        &self.compiled
    }

    /// Returns the parameter store.
    #[must_use]
    pub fn params(&self) -> &ParameterStore {
        &self.params
    }

    /// Returns a mutable reference to the parameter store.
    ///
    /// Prefer [`set_param`](Self::set_param) for changes that should bump the
    /// version; this accessor is for bulk configuration before publishing.
    pub fn params_mut(&mut self) -> &mut ParameterStore {
        &mut self.params
    }

    /// Returns `true` if state changed since the last [`clear_dirty`](Self::clear_dirty).
    #[must_use]
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Clears the dirty flag after a consumer has refreshed.
    pub fn clear_dirty(&mut self) {
        self.dirty = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> ParameterStore {
        let mut s = ParameterStore::new();
        s.define_real("rest", 1.0, 0.0, 10.0);
        s
    }

    #[test]
    fn new_starts_at_version_one() {
        let r = HotReloadRegistry::new(
            "constraint d(rest) { residual = length(b - a) - rest; }",
            store(),
        )
        .unwrap();
        assert_eq!(r.version(), 1);
        assert!(!r.is_dirty());
    }

    #[test]
    fn set_param_bumps_version_and_updates_value() {
        let mut r = HotReloadRegistry::new(
            "constraint d(rest) { residual = length(b - a) - rest; }",
            store(),
        )
        .unwrap();
        let v0 = r.version();
        r.set_param("rest", ParamValue::Real(3.0)).unwrap();
        assert_eq!(r.version(), v0 + 1);
        assert!((r.params().get_real("rest").unwrap() - 3.0).abs() < 1e-6);
        assert!(r.is_dirty());
    }

    #[test]
    fn reload_source_swaps_and_bumps() {
        let mut r = HotReloadRegistry::new(
            "constraint d(rest) { residual = length(b - a) - rest; }",
            store(),
        )
        .unwrap();
        let v0 = r.version();
        r.reload_source("constraint d(rest) { residual = distance(a, b) - rest; }")
            .unwrap();
        assert_eq!(r.version(), v0 + 1);
        assert!(r.source().contains("distance"));
    }

    #[test]
    fn failed_reload_preserves_previous_program() {
        let good = "constraint d(rest) { residual = length(b - a) - rest; }";
        let mut r = HotReloadRegistry::new(good, store()).unwrap();
        let v0 = r.version();
        let before = r.compiled().clone();
        let err = r.reload_source("constraint d( { residual = ").unwrap_err();
        assert!(matches!(
            err,
            DslError::Lex { .. } | DslError::Parse { .. } | DslError::Compile { .. }
        ));
        assert_eq!(r.version(), v0);
        assert_eq!(r.compiled(), &before);
    }

    #[test]
    fn unknown_param_is_error() {
        let mut r = HotReloadRegistry::new(
            "constraint d(rest) { residual = length(b - a) - rest; }",
            store(),
        )
        .unwrap();
        assert!(r.set_param("missing", ParamValue::Real(1.0)).is_err());
    }
}
