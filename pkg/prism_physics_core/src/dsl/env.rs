//! Variable-slot environment for the constraint DSL.
//!
//! The compiler maps every named variable to a slot in a flat scalar array.
//! Scalar parameters take a single slot; points take three consecutive slots
//! for their x/y/z components. Both the compiler (to emit
//! [`PushScalar`](crate::dsl::bytecode::OpCode::PushScalar) /
//! [`PushPoint`](crate::dsl::bytecode::OpCode::PushPoint)) and the runtime (to
//! fill the value array) consult this mapping.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. A
//! name-to-slot symbol table is standard, publicly documented
//! compiler-construction knowledge.

/// Whether a bound variable is a scalar or a three-component point.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum VarKind {
    /// A single scalar occupying one slot.
    Scalar,
    /// A point occupying three consecutive slots (x, y, z).
    Point,
}

impl VarKind {
    /// Returns how many slots this kind occupies.
    #[must_use]
    pub fn slot_count(self) -> usize {
        match self {
            VarKind::Scalar => 1,
            VarKind::Point => 3,
        }
    }
}

/// A resolved variable: its kind and its base slot index.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct VarBinding {
    /// Whether the variable is a scalar or a point.
    pub kind: VarKind,
    /// Index of the first slot the variable occupies.
    pub base: usize,
}

/// A symbol table mapping variable names to slot bindings.
#[derive(Clone, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct Environment {
    names: Vec<String>,
    bindings: Vec<VarBinding>,
    total: usize,
}

impl Environment {
    /// Creates an empty environment.
    #[must_use]
    pub fn new() -> Self {
        Environment::default()
    }

    /// Defines `name` with the given `kind`, returning its binding.
    ///
    /// Re-defining an existing name returns the existing binding unchanged (so
    /// its kind is not altered), which keeps slot assignment stable.
    pub fn define(&mut self, name: &str, kind: VarKind) -> VarBinding {
        if let Some(existing) = self.lookup(name) {
            return existing;
        }
        let binding = VarBinding {
            kind,
            base: self.total,
        };
        self.total += kind.slot_count();
        self.names.push(name.to_string());
        self.bindings.push(binding);
        binding
    }

    /// Looks up `name`, returning its binding if defined.
    #[must_use]
    pub fn lookup(&self, name: &str) -> Option<VarBinding> {
        self.names
            .iter()
            .position(|n| n == name)
            .map(|i| self.bindings[i])
    }

    /// Returns the total number of scalar slots required.
    #[must_use]
    pub fn total_slots(&self) -> usize {
        self.total
    }

    /// Returns the number of defined variables.
    #[must_use]
    pub fn len(&self) -> usize {
        self.names.len()
    }

    /// Returns `true` when no variables are defined.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }

    /// Returns the defined variable names in definition order.
    #[must_use]
    pub fn names(&self) -> &[String] {
        &self.names
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalars_and_points_allocate_expected_slots() {
        let mut env = Environment::new();
        let rest = env.define("rest", VarKind::Scalar);
        let a = env.define("a", VarKind::Point);
        let b = env.define("b", VarKind::Point);
        assert_eq!(rest.base, 0);
        assert_eq!(a.base, 1);
        assert_eq!(b.base, 4);
        assert_eq!(env.total_slots(), 7);
    }

    #[test]
    fn redefining_returns_existing() {
        let mut env = Environment::new();
        let first = env.define("x", VarKind::Scalar);
        let second = env.define("x", VarKind::Scalar);
        assert_eq!(first, second);
        assert_eq!(env.len(), 1);
    }

    #[test]
    fn lookup_missing_is_none() {
        let env = Environment::new();
        assert!(env.lookup("nope").is_none());
        assert!(env.is_empty());
    }
}
