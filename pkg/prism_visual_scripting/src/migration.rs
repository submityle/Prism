//! Explicit, digest-bound state migrations for durable checkpoints.

use crate::{Value, VsError, VsResult};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::collections::BTreeMap;
use std::sync::Arc;

const MAX_MIGRATION_STEPS: usize = 1024;

pub type MigrationFn = Arc<dyn Fn(Value) -> VsResult<Value> + Send + Sync>;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MigrationDescriptor {
    pub from: u32,
    pub to: u32,
    pub digest: [u8; 32],
}

pub struct MigrationStep {
    pub descriptor: MigrationDescriptor,
    transform: MigrationFn,
}

#[derive(Default)]
pub struct MigrationRegistry {
    steps: BTreeMap<u32, MigrationStep>,
}

impl MigrationRegistry {
    pub fn register(
        &mut self,
        from: u32,
        to: u32,
        digest: [u8; 32],
        transform: MigrationFn,
    ) -> VsResult<()> {
        if to <= from {
            return Err(VsError::Validate("migration target must be newer".into()));
        }
        if self.steps.contains_key(&from) {
            return Err(VsError::Runtime(format!(
                "migration from schema {from} is already registered"
            )));
        }
        self.steps.insert(
            from,
            MigrationStep {
                descriptor: MigrationDescriptor { from, to, digest },
                transform,
            },
        );
        Ok(())
    }

    pub fn plan(&self, from: u32, to: u32) -> VsResult<Vec<MigrationDescriptor>> {
        if from > to {
            return Err(VsError::Validate(
                "migration source is newer than target".into(),
            ));
        }
        let mut current = from;
        let mut descriptors = Vec::new();
        while current < to {
            let step = self.steps.get(&current).ok_or_else(|| {
                VsError::Runtime(format!("no migration registered from schema {current}"))
            })?;
            if step.descriptor.to > to {
                return Err(VsError::Runtime(format!(
                    "migration from {} skips requested target {}",
                    step.descriptor.from, to
                )));
            }
            descriptors.push(step.descriptor.clone());
            if descriptors.len() > MAX_MIGRATION_STEPS {
                return Err(VsError::Validate(format!(
                    "migration chain exceeds {MAX_MIGRATION_STEPS} steps"
                )));
            }
            current = step.descriptor.to;
        }
        Ok(descriptors)
    }

    pub fn migrate(&self, mut value: Value, from: u32, to: u32) -> VsResult<Value> {
        let plan = self.plan(from, to)?;
        for descriptor in plan {
            let step = self.steps.get(&descriptor.from).ok_or_else(|| {
                VsError::Runtime(format!(
                    "migration plan became invalid at schema {}",
                    descriptor.from
                ))
            })?;
            value = (step.transform)(value)?;
        }
        Ok(value)
    }

    pub fn chain_digest(&self, from: u32, to: u32) -> VsResult<[u8; 32]> {
        let mut hasher = Sha256::new();
        for descriptor in self.plan(from, to)? {
            hasher.update(descriptor.digest);
        }
        Ok(hasher.finalize().into())
    }
}

pub fn migration_digest(name: &str, from: u32, to: u32) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(name.as_bytes());
    digest.update(from.to_le_bytes());
    digest.update(to.to_le_bytes());
    digest.finalize().into()
}

#[cfg(test)]
mod tests {
    use super::MigrationRegistry;
    use crate::Value;
    use std::sync::Arc;

    #[test]
    fn migration_plan_rejects_excessive_chain() {
        let mut registry = MigrationRegistry::default();
        for schema in 0..1025u32 {
            registry
                .register(
                    schema,
                    schema + 1,
                    [0; 32],
                    Arc::new(|value: Value| Ok::<Value, crate::VsError>(value)),
                )
                .unwrap();
        }
        let error = registry.plan(0, 1025).unwrap_err();
        assert!(error.to_string().contains("exceeds 1024"));
    }
}
