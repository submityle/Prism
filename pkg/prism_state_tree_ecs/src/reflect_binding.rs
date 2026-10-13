//! `prism_reflect` adapter for compiled `StateTree` binding slots.

use alloc::vec::Vec;
use core::fmt;
use prism_reflect::{AccessError, AccessPlan, Reflect};
use prism_state_tree_core::{BindingPath, BindingPathError, BindingSlot};

/// A compiled reflected path assigned to one stable `StateTree` slot.
#[derive(Debug, Clone)]
pub struct ReflectBindingPlan {
    entries: Vec<(BindingSlot, BindingPath, AccessPlan)>,
}

/// Failure while compiling a reflected binding plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReflectBindingError {
    /// The binding path is not a valid `StateTree` path.
    InvalidPath(BindingPathError),
    /// Two entries claim the same slot.
    DuplicateSlot(BindingSlot),
    /// `prism_reflect` could not resolve the path against the sample root.
    Resolve {
        /// The slot whose path could not be resolved.
        slot: BindingSlot,
        /// The underlying reflected access failure.
        error: AccessError,
    },
    /// The resolved field has a different reflected type than the contract.
    TypeMismatch {
        /// The slot whose type contract failed.
        slot: BindingSlot,
        /// The expected reflected type name.
        expected: &'static str,
        /// The actual reflected type name.
        actual: &'static str,
    },
}

impl fmt::Display for ReflectBindingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPath(error) => write!(f, "invalid StateTree binding path: {error}"),
            Self::DuplicateSlot(slot) => write!(f, "duplicate StateTree binding slot {}", slot.0),
            Self::Resolve { slot, error } => {
                write!(f, "slot {} path resolution failed: {error}", slot.0)
            }
            Self::TypeMismatch {
                slot,
                expected,
                actual,
            } => write!(
                f,
                "slot {} expected reflected type `{expected}`, got `{actual}`",
                slot.0
            ),
        }
    }
}

impl ReflectBindingPlan {
    /// Compiles reflected paths once, validating them against `root`.
    pub fn compile(
        root: &dyn Reflect,
        bindings: Vec<(BindingSlot, &str)>,
    ) -> Result<Self, ReflectBindingError> {
        Self::compile_typed(
            root,
            bindings.into_iter().map(|(slot, path)| (slot, path, None)),
        )
    }

    /// Compiles reflected paths and validates optional target type names.
    pub fn compile_typed<'a, I>(
        root: &dyn Reflect,
        bindings: I,
    ) -> Result<Self, ReflectBindingError>
    where
        I: IntoIterator<Item = (BindingSlot, &'a str, Option<&'static str>)>,
    {
        let bindings: Vec<_> = bindings.into_iter().collect();
        let mut entries = Vec::with_capacity(bindings.len());
        for (slot, raw_path, expected_type) in bindings {
            if entries.iter().any(|(existing, _, _)| *existing == slot) {
                return Err(ReflectBindingError::DuplicateSlot(slot));
            }
            let path = BindingPath::parse(raw_path).map_err(ReflectBindingError::InvalidPath)?;
            let plan = AccessPlan::compile(root, path.as_str())
                .map_err(|error| ReflectBindingError::Resolve { slot, error })?;
            if let Some(expected) = expected_type {
                let actual = plan.get(root).map_or("<unresolved>", Reflect::type_name);
                if actual != expected {
                    return Err(ReflectBindingError::TypeMismatch {
                        slot,
                        expected,
                        actual,
                    });
                }
            }
            entries.push((slot, path, plan));
        }
        entries.sort_by_key(|(slot, _, _)| *slot);
        Ok(Self { entries })
    }

    /// Returns the stable slot/path pairs in slot order.
    pub fn entries(&self) -> impl Iterator<Item = (BindingSlot, &str)> {
        self.entries
            .iter()
            .map(|(slot, path, _)| (*slot, path.as_str()))
    }

    /// Reads one compiled slot without reparsing or string lookup.
    #[must_use]
    pub fn get<'a>(&self, root: &'a dyn Reflect, slot: BindingSlot) -> Option<&'a dyn Reflect> {
        let index = self
            .entries
            .binary_search_by_key(&slot, |(candidate, _, _)| *candidate)
            .ok()?;
        self.entries[index].2.get(root)
    }
}

#[cfg(test)]
mod tests {
    use alloc::{vec, vec::Vec};
    use prism_reflect::{Reflect, ReflectRef};
    use prism_state_tree_core::BindingSlot;

    use super::{ReflectBindingError, ReflectBindingPlan};

    #[derive(Reflect)]
    struct Target {
        health: Health,
    }

    #[derive(Reflect)]
    struct Health {
        current: i32,
    }

    #[test]
    fn compiles_paths_into_stable_slots_and_reads_without_reparsing() {
        let target = Target {
            health: Health { current: 42 },
        };
        let plan =
            ReflectBindingPlan::compile(&target, vec![(BindingSlot(4), "health.current")]).unwrap();
        assert_eq!(
            plan.entries().collect::<Vec<_>>(),
            vec![(BindingSlot(4), "health.current")]
        );
        let value = plan.get(&target, BindingSlot(4)).unwrap();
        assert_eq!(value.downcast_ref::<i32>(), Some(&42));
    }

    #[test]
    fn rejects_duplicate_slots_and_unresolvable_paths() {
        let target = Target {
            health: Health { current: 42 },
        };
        assert!(matches!(
            ReflectBindingPlan::compile(
                &target,
                vec![
                    (BindingSlot(1), "health"),
                    (BindingSlot(1), "health.current")
                ]
            ),
            Err(ReflectBindingError::DuplicateSlot(BindingSlot(1)))
        ));
        assert!(matches!(
            ReflectBindingPlan::compile(&target, vec![(BindingSlot(1), "health.missing")]),
            Err(ReflectBindingError::Resolve { .. })
        ));
        assert!(matches!(
            ReflectBindingPlan::compile_typed(
                &target,
                vec![(BindingSlot(1), "health.current", Some("f32"))]
            ),
            Err(ReflectBindingError::TypeMismatch {
                slot: BindingSlot(1),
                expected: "f32",
                actual: "i32"
            })
        ));
        assert!(matches!(
            ReflectBindingPlan::compile(&target, vec![(BindingSlot(1), "health.current")])
                .unwrap()
                .get(&target, BindingSlot(1))
                .unwrap()
                .reflect_ref(),
            ReflectRef::Value(_)
        ));
    }
}
