//! Capability and taint policy enforcement at the effect boundary.

use crate::{EffectClass, EffectSpec, Value, VsError, VsResult};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaintSet(pub BTreeSet<String>);

impl TaintSet {
    pub fn insert(&mut self, label: impl Into<String>) -> VsResult<()> {
        let label = label.into();
        if label.trim().is_empty() {
            return Err(VsError::Validate("taint label must not be empty".into()));
        }
        self.0.insert(label);
        Ok(())
    }

    pub fn contains(&self, label: &str) -> bool {
        self.0.contains(label)
    }
}

pub trait TaintProvider: Send + Sync {
    fn taint(&self, inputs: &[Value]) -> TaintSet;
}

pub trait EffectPolicy: Send + Sync {
    fn check_effect(&self, spec: &EffectSpec, taint: &TaintSet) -> VsResult<()>;
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaintPolicy {
    pub forbidden: BTreeMap<String, BTreeSet<EffectClass>>,
    pub capability_allowlist: Option<BTreeSet<String>>,
}

impl TaintPolicy {
    pub fn deny(&mut self, label: impl Into<String>, class: EffectClass) -> VsResult<()> {
        let label = label.into();
        if label.trim().is_empty() {
            return Err(VsError::Validate("taint label must not be empty".into()));
        }
        self.forbidden.entry(label).or_default().insert(class);
        Ok(())
    }

    pub fn allow_capability(&mut self, capability: impl Into<String>) -> VsResult<()> {
        let capability = capability.into();
        if capability.trim().is_empty() {
            return Err(VsError::Validate(
                "allowed capability must not be empty".into(),
            ));
        }
        self.capability_allowlist
            .get_or_insert_with(BTreeSet::new)
            .insert(capability);
        Ok(())
    }
}

impl EffectPolicy for TaintPolicy {
    fn check_effect(&self, spec: &EffectSpec, taint: &TaintSet) -> VsResult<()> {
        if let Some(allowlist) = &self.capability_allowlist
            && !allowlist.contains(&spec.capability)
        {
            return Err(VsError::Runtime(format!(
                "effect capability {} is not allowed by policy",
                spec.capability
            )));
        }
        for label in &taint.0 {
            if self
                .forbidden
                .get(label)
                .is_some_and(|classes| classes.contains(&spec.class))
            {
                return Err(VsError::Runtime(format!(
                    "taint label {label} cannot flow into {:?} effect",
                    spec.class
                )));
            }
        }
        Ok(())
    }
}
