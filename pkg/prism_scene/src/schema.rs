//! Deterministic content-schema migration for scene templates.
//!
//! The registry deliberately operates on a validated [`SceneTemplate`] rather
//! than guessing the meaning of opaque component bytes. Applications register
//! the semantic conversion for each adjacent schema pair; the runtime then
//! follows that chain with bounded, deterministic steps.

use alloc::vec::Vec;

use crate::{SceneTemplate, TemplateError, SCENE_TEMPLATE_SCHEMA};

/// Failure returned by one registered schema conversion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SchemaMigrationError {
    /// The migration callback rejected the template for a domain-specific reason.
    Rejected(&'static str),
    /// The callback returned a template that violates a scene invariant.
    InvalidTemplate(TemplateError),
    /// No registered adjacent step could advance the source schema.
    MissingStep {
        /// Schema at which the migration chain stopped.
        from: u32,
        /// Requested destination schema.
        target: u32,
    },
    /// The configured migration budget was exhausted.
    StepLimitExceeded {
        /// Maximum number of steps allowed by the registry.
        limit: usize,
    },
    /// A registry entry was malformed or duplicated.
    InvalidRegistration(&'static str),
}

/// One adjacent, forward-only content-schema conversion.
#[derive(Clone, Copy)]
pub struct SchemaMigration {
    /// Source content schema.
    pub from: u32,
    /// Destination content schema.
    pub to: u32,
    /// Semantic conversion supplied by the owning application/schema crate.
    pub apply: fn(SceneTemplate) -> Result<SceneTemplate, SchemaMigrationError>,
}

impl core::fmt::Debug for SchemaMigration {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SchemaMigration")
            .field("from", &self.from)
            .field("to", &self.to)
            .finish_non_exhaustive()
    }
}

/// A bounded registry of forward schema steps.
#[derive(Clone, Debug)]
pub struct SchemaMigrationRegistry {
    steps: Vec<SchemaMigration>,
    max_steps: usize,
}

impl SchemaMigrationRegistry {
    /// Creates an empty registry with a conservative default step budget.
    #[must_use]
    pub fn new() -> Self {
        Self {
            steps: Vec::new(),
            max_steps: 64,
        }
    }

    /// Changes the maximum number of conversions allowed in one migration.
    #[must_use]
    pub fn with_max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = max_steps;
        self
    }

    /// Registers one unique forward step.
    pub fn register(&mut self, migration: SchemaMigration) -> Result<(), SchemaMigrationError> {
        if migration.from == 0 || migration.to == 0 || migration.from >= migration.to {
            return Err(SchemaMigrationError::InvalidRegistration(
                "schema steps must be forward and non-zero",
            ));
        }
        if self.steps.iter().any(|step| step.from == migration.from) {
            return Err(SchemaMigrationError::InvalidRegistration(
                "duplicate source schema",
            ));
        }
        self.steps.push(migration);
        self.steps.sort_unstable_by_key(|step| step.from);
        Ok(())
    }

    /// Migrates a template to `target`, validating every callback result.
    pub fn migrate(
        &self,
        mut template: SceneTemplate,
        target: u32,
    ) -> Result<SceneTemplate, SchemaMigrationError> {
        template
            .validate()
            .map_err(SchemaMigrationError::InvalidTemplate)?;
        if target == 0 || template.schema > target {
            return Err(SchemaMigrationError::MissingStep {
                from: template.schema,
                target,
            });
        }
        let mut steps = 0usize;
        while template.schema < target {
            if steps >= self.max_steps {
                return Err(SchemaMigrationError::StepLimitExceeded {
                    limit: self.max_steps,
                });
            }
            let migration = self
                .steps
                .iter()
                .find(|step| step.from == template.schema)
                .ok_or(SchemaMigrationError::MissingStep {
                    from: template.schema,
                    target,
                })?;
            if migration.to > target {
                return Err(SchemaMigrationError::MissingStep {
                    from: template.schema,
                    target,
                });
            }
            template = (migration.apply)(template)?;
            if template.schema != migration.to {
                return Err(SchemaMigrationError::Rejected(
                    "migration callback returned an unexpected schema",
                ));
            }
            template
                .validate()
                .map_err(SchemaMigrationError::InvalidTemplate)?;
            steps += 1;
        }
        Ok(template)
    }
}

impl SchemaMigrationRegistry {
    /// Registry containing the wire-v1/content-v1 to current-schema upgrade.
    #[must_use]
    pub fn with_builtin_migrations() -> Self {
        fn identity_v1_to_v2(
            mut template: SceneTemplate,
        ) -> Result<SceneTemplate, SchemaMigrationError> {
            template.schema = SCENE_TEMPLATE_SCHEMA;
            Ok(template)
        }

        let mut registry = Self::new();
        registry
            .register(SchemaMigration {
                from: 1,
                to: SCENE_TEMPLATE_SCHEMA,
                apply: identity_v1_to_v2,
            })
            .expect("built-in scene schema migration is valid");
        registry
    }
}

impl Default for SchemaMigrationRegistry {
    fn default() -> Self {
        Self::with_builtin_migrations()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SceneTemplate;
    use prism_asset::StableGuid;

    #[test]
    fn rejects_missing_steps_and_bounded_chain_overflow() {
        let mut template = SceneTemplate::empty(StableGuid::from_u128(77));
        template.schema = 1;
        let empty = SchemaMigrationRegistry::new();
        assert_eq!(
            empty.migrate(template.clone(), SCENE_TEMPLATE_SCHEMA),
            Err(SchemaMigrationError::MissingStep {
                from: 1,
                target: SCENE_TEMPLATE_SCHEMA
            })
        );
        let registry = SchemaMigrationRegistry::default().with_max_steps(0);
        assert_eq!(
            registry.migrate(template, SCENE_TEMPLATE_SCHEMA),
            Err(SchemaMigrationError::StepLimitExceeded { limit: 0 })
        );
    }
}
