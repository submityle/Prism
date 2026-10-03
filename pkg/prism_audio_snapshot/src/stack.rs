//! Weighted resolution of several snapshots into one blended target map.
//!
//! Activating more than one snapshot at a time (for example a base ambience
//! plus a partial combat layer) produces a single target by normalizing the
//! weights and combining each parameter in its own domain. A parameter is only
//! blended across the snapshots that actually define it; snapshots that omit a
//! parameter do not pull its value. The blending domain of each parameter is
//! taken from the lowest-id snapshot that defines it, which keeps the result
//! deterministic even if authoring data disagrees on a parameter's kind.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Reads [`crate::snapshot::Snapshot`] targets from
//! [`crate::registry::SnapshotRegistry`] and combines them with
//! [`crate::blend::blend_weighted`] into [`crate::resolved::ResolvedParameters`].

use alloc::collections::BTreeMap;
use alloc::collections::BTreeSet;
use alloc::vec::Vec;

use prism_audio_core::math::Sample;

use crate::blend;
use crate::parameter::{ParameterId, ParameterKind};
use crate::registry::SnapshotRegistry;
use crate::resolved::ResolvedParameters;
use crate::snapshot::SnapshotId;

/// Collects, in deterministic order, the blending domain of every parameter
/// defined by any of `ids` present in `registry`.
///
/// When several snapshots define the same parameter with different kinds, the
/// lowest-id snapshot wins.
#[must_use]
pub fn blend_kinds(
    registry: &SnapshotRegistry,
    ids: &[SnapshotId],
) -> BTreeMap<ParameterId, ParameterKind> {
    let sorted: BTreeSet<SnapshotId> = ids.iter().copied().collect();
    let mut kinds = BTreeMap::new();
    for id in sorted {
        if let Some(snapshot) = registry.get(id) {
            for target in snapshot.targets() {
                kinds.entry(target.id).or_insert(target.kind);
            }
        }
    }
    kinds
}

/// Resolves a weighted set of snapshots into a single blended target map.
///
/// Weights are normalized per parameter across the snapshots that define it.
/// Snapshots missing from `registry` are skipped. A single entry with any
/// positive weight resolves to that snapshot's targets unchanged.
#[must_use]
pub fn resolve_blend(
    registry: &SnapshotRegistry,
    weighted: &[(SnapshotId, Sample)],
) -> ResolvedParameters {
    let ids: Vec<SnapshotId> = weighted.iter().map(|&(id, _)| id).collect();
    let kinds = blend_kinds(registry, &ids);
    let mut out = ResolvedParameters::new();
    for (&param, &kind) in &kinds {
        let mut values: Vec<(Sample, Sample)> = Vec::new();
        for &(id, weight) in weighted {
            let Some(snapshot) = registry.get(id) else {
                continue;
            };
            let Some(target) = snapshot.get(param) else {
                continue;
            };
            values.push((target.value, weight));
        }
        if let Some(value) = blend::blend_weighted(kind, &values) {
            out.set(param, value);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::snapshot::Snapshot;

    const EPS: Sample = 1e-6;

    fn registry_with(snapshots: &[Snapshot]) -> SnapshotRegistry {
        let mut reg = SnapshotRegistry::new();
        for s in snapshots {
            reg.register(s.clone());
        }
        reg
    }

    fn snap(id: u32, targets: &[(u32, ParameterKind, Sample)]) -> Snapshot {
        let mut s = Snapshot::new(SnapshotId::new(id));
        for &(pid, kind, value) in targets {
            s.set(ParameterId::new(pid), kind, value);
        }
        s
    }

    #[test]
    fn single_snapshot_resolves_to_its_targets() {
        let reg = registry_with(&[snap(1, &[(1, ParameterKind::Linear, 0.4)])]);
        let r = resolve_blend(&reg, &[(SnapshotId::new(1), 0.3)]);
        assert!((r.get(ParameterId::new(1)).expect("present") - 0.4).abs() < EPS);
    }

    #[test]
    fn two_snapshots_blend_with_normalized_weights() {
        let reg = registry_with(&[
            snap(1, &[(1, ParameterKind::Linear, 0.0)]),
            snap(2, &[(1, ParameterKind::Linear, 8.0)]),
        ]);
        // Weights 1 and 3 -> 0.75 toward 8 = 6.0.
        let r = resolve_blend(&reg, &[(SnapshotId::new(1), 1.0), (SnapshotId::new(2), 3.0)]);
        assert!((r.get(ParameterId::new(1)).expect("present") - 6.0).abs() < EPS);
    }

    #[test]
    fn parameter_defined_in_one_snapshot_uses_only_that_snapshot() {
        let reg = registry_with(&[
            snap(1, &[(1, ParameterKind::Linear, 2.0)]),
            snap(2, &[(2, ParameterKind::Linear, 5.0)]),
        ]);
        let r = resolve_blend(&reg, &[(SnapshotId::new(1), 1.0), (SnapshotId::new(2), 1.0)]);
        assert!((r.get(ParameterId::new(1)).expect("present") - 2.0).abs() < EPS);
        assert!((r.get(ParameterId::new(2)).expect("present") - 5.0).abs() < EPS);
    }

    #[test]
    fn missing_snapshot_is_skipped() {
        let reg = registry_with(&[snap(1, &[(1, ParameterKind::Linear, 3.0)])]);
        let r = resolve_blend(&reg, &[(SnapshotId::new(1), 1.0), (SnapshotId::new(99), 1.0)]);
        assert!((r.get(ParameterId::new(1)).expect("present") - 3.0).abs() < EPS);
    }

    #[test]
    fn kind_taken_from_lowest_id_snapshot() {
        let reg = registry_with(&[
            snap(1, &[(1, ParameterKind::Hertz, 100.0)]),
            snap(2, &[(1, ParameterKind::Linear, 400.0)]),
        ]);
        let kinds = blend_kinds(&reg, &[SnapshotId::new(2), SnapshotId::new(1)]);
        assert_eq!(kinds.get(&ParameterId::new(1)).copied(), Some(ParameterKind::Hertz));
        // Hertz domain -> geometric mean of 100 and 400 is 200.
        let r = resolve_blend(&reg, &[(SnapshotId::new(1), 1.0), (SnapshotId::new(2), 1.0)]);
        assert!((r.get(ParameterId::new(1)).expect("present") - 200.0).abs() < 1e-2);
    }

    #[test]
    fn empty_weighted_resolves_to_empty() {
        let reg = registry_with(&[snap(1, &[(1, ParameterKind::Linear, 1.0)])]);
        let r = resolve_blend(&reg, &[]);
        assert!(r.is_empty());
    }
}
