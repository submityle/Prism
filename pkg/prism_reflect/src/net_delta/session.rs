//! Baseline-tracked replication state (design §24.5).
//!
//! [`ReplicationState`] keeps the last acknowledged snapshot of a reflected
//! value (its *baseline*) and, each tick, produces a compact field delta
//! against the current value. This is the stateful front door the engine's
//! replication system (`prism_replication`) drives: snapshot once, then emit
//! only the fields that moved since the baseline, optionally filtered by a
//! [`ReplicationPlan`] so `#[reflect(no_replicate)]` fields never leave the
//! host.

use crate::integration::ReplicationPlan;
use crate::net_delta::codec::encode_delta;
use crate::net_delta::{DeltaError, DirtyMask};
use crate::reflect::Reflect;
use alloc::boxed::Box;
use alloc::vec::Vec;

/// Tracks a baseline snapshot and emits field-level deltas against it.
///
/// The baseline is stored as an owned reflected clone, so the state is
/// self-contained and does not borrow the live value between ticks.
pub struct ReplicationState {
    baseline: Box<dyn Reflect>,
    plan: Option<ReplicationPlan>,
}

impl ReplicationState {
    /// Capture `initial` as the baseline, replicating every field.
    #[must_use]
    pub fn new(initial: &dyn Reflect) -> Self {
        Self {
            baseline: initial.reflect_clone(),
            plan: None,
        }
    }

    /// Capture `initial` as the baseline, replicating only the fields selected
    /// by `plan`.
    #[must_use]
    pub fn with_plan(initial: &dyn Reflect, plan: ReplicationPlan) -> Self {
        Self {
            baseline: initial.reflect_clone(),
            plan: Some(plan),
        }
    }

    /// The current baseline snapshot.
    #[must_use]
    pub fn baseline(&self) -> &dyn Reflect {
        self.baseline.as_ref()
    }

    /// Compute the dirty-field mask of `current` relative to the baseline,
    /// honouring the replication plan if one is set.
    ///
    /// # Errors
    /// Returns [`DeltaError::NotAStruct`] or [`DeltaError::FieldCountMismatch`]
    /// if `current` is not the same named-field struct shape as the baseline.
    pub fn dirty(&self, current: &dyn Reflect) -> Result<DirtyMask, DeltaError> {
        match &self.plan {
            Some(plan) => DirtyMask::changed_in_plan(self.baseline.as_ref(), current, plan),
            None => DirtyMask::changed(self.baseline.as_ref(), current),
        }
    }

    /// Encode a compact delta carrying the fields of `current` that changed
    /// since the baseline (filtered by the plan).
    ///
    /// The byte string is ready to apply on a remote replica with
    /// [`decode_and_apply`](crate::net_delta::decode_and_apply). It does not
    /// advance the baseline; call [`commit`](ReplicationState::commit) once the
    /// delta is acknowledged.
    ///
    /// # Errors
    /// Propagates any [`DeltaError`] from dirty computation or encoding.
    pub fn encode(&self, current: &dyn Reflect) -> Result<Vec<u8>, DeltaError> {
        let dirty = self.dirty(current)?;
        encode_delta(current, &dirty)
    }

    /// Advance the baseline to `current`, so subsequent deltas are measured
    /// from here.
    pub fn commit(&mut self, current: &dyn Reflect) {
        self.baseline = current.reflect_clone();
    }
}
