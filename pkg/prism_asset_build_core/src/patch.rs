//! Deterministic build-manifest patch planning.
//!
//! A patch is computed from two committed build manifests, never from the
//! current source directory.  The planner only describes logical artifact
//! changes and byte accounting; compression, encryption, signing, upload, and
//! installation remain separate publication stages.

use std::collections::{BTreeMap, BTreeSet};

use crate::types::{ArtifactDescriptor, AssetId, BuildRevision, ContentHash, Manifest};

/// The kind of change observed for one logical asset.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PatchChange {
    /// The target revision introduces an asset absent from the base.
    Added,
    /// The target revision removes an asset present in the base.
    Removed,
    /// Both revisions contain the asset, but its descriptor changed.
    Changed,
}

/// One deterministic logical change in a patch.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PatchEntry {
    /// Stable logical asset identity.
    pub asset: AssetId,
    /// Change classification.
    pub change: PatchChange,
    /// Descriptor in the base revision, if present.
    pub base: Option<ArtifactDescriptor>,
    /// Descriptor in the target revision, if present.
    pub target: Option<ArtifactDescriptor>,
    /// Whether the target content bytes can be reused from the base closure.
    /// This is true for unchanged content hashes, even when the artifact key or
    /// other descriptor metadata changed.
    pub content_reused: bool,
}

/// Deterministic result of comparing two committed build manifests.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PatchPlan {
    /// Revision the patch applies to.
    pub base_revision: BuildRevision,
    /// Revision produced by the patch.
    pub target_revision: BuildRevision,
    /// Added, changed, and removed assets in ascending [`AssetId`] order.
    pub entries: Vec<PatchEntry>,
    /// Unique target content hashes that are not present in the base closure.
    pub new_content: Vec<ContentHash>,
    /// Sum of bytes for `new_content`, with each content hash counted once.
    pub download_bytes: u64,
    /// Signed target-minus-base artifact bytes. A negative value means the
    /// installed artifact set shrinks after applying the patch.
    pub install_growth_bytes: i128,
}

/// Errors raised while planning a patch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PatchPlanError {
    /// The target revision is not newer than the base revision.
    TargetNotNewer {
        /// Base manifest revision.
        base: BuildRevision,
        /// Candidate target revision.
        target: BuildRevision,
    },
    /// Artifact byte accounting overflowed `u64`.
    ByteOverflow,
}

impl std::fmt::Display for PatchPlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TargetNotNewer { base, target } => write!(
                f,
                "patch target revision {} is not newer than base revision {}",
                target.0, base.0
            ),
            Self::ByteOverflow => f.write_str("patch byte accounting overflowed u64"),
        }
    }
}

impl std::error::Error for PatchPlanError {}

/// Computes a deterministic patch from `base` to `target`.
pub fn plan_patch(base: &Manifest, target: &Manifest) -> Result<PatchPlan, PatchPlanError> {
    if target.revision <= base.revision {
        return Err(PatchPlanError::TargetNotNewer {
            base: base.revision,
            target: target.revision,
        });
    }

    let base_hashes: BTreeSet<ContentHash> = base
        .entries
        .values()
        .map(|descriptor| descriptor.content_hash)
        .collect();
    let mut entries = Vec::new();
    let mut target_hashes = BTreeMap::<ContentHash, u64>::new();
    let mut asset_ids = BTreeSet::new();
    asset_ids.extend(base.entries.keys().copied());
    asset_ids.extend(target.entries.keys().copied());

    let mut base_total = 0_u64;
    for descriptor in base.entries.values() {
        base_total = base_total
            .checked_add(descriptor.byte_len)
            .ok_or(PatchPlanError::ByteOverflow)?;
    }
    let mut target_total = 0_u64;
    for descriptor in target.entries.values() {
        target_total = target_total
            .checked_add(descriptor.byte_len)
            .ok_or(PatchPlanError::ByteOverflow)?;
    }

    for asset in asset_ids {
        match (base.entries.get(&asset), target.entries.get(&asset)) {
            (None, Some(target_descriptor)) => {
                target_hashes
                    .entry(target_descriptor.content_hash)
                    .or_insert(target_descriptor.byte_len);
                entries.push(PatchEntry {
                    asset,
                    change: PatchChange::Added,
                    base: None,
                    target: Some(target_descriptor.clone()),
                    content_reused: base_hashes.contains(&target_descriptor.content_hash),
                });
            }
            (Some(base_descriptor), None) => entries.push(PatchEntry {
                asset,
                change: PatchChange::Removed,
                base: Some(base_descriptor.clone()),
                target: None,
                content_reused: false,
            }),
            (Some(base_descriptor), Some(target_descriptor))
                if base_descriptor != target_descriptor =>
            {
                target_hashes
                    .entry(target_descriptor.content_hash)
                    .or_insert(target_descriptor.byte_len);
                entries.push(PatchEntry {
                    asset,
                    change: PatchChange::Changed,
                    base: Some(base_descriptor.clone()),
                    target: Some(target_descriptor.clone()),
                    content_reused: base_descriptor.content_hash == target_descriptor.content_hash
                        || base_hashes.contains(&target_descriptor.content_hash),
                });
            }
            (Some(_), Some(_)) | (None, None) => {}
        }
    }

    let new_content = target_hashes
        .into_iter()
        .filter_map(|(hash, byte_len)| (!base_hashes.contains(&hash)).then_some((hash, byte_len)))
        .collect::<Vec<_>>();
    let download_bytes = new_content.iter().try_fold(0_u64, |total, (_, len)| {
        total.checked_add(*len).ok_or(PatchPlanError::ByteOverflow)
    })?;
    let new_content = new_content.into_iter().map(|(hash, _)| hash).collect();

    Ok(PatchPlan {
        base_revision: base.revision,
        target_revision: target.revision,
        entries,
        new_content,
        download_bytes,
        install_growth_bytes: i128::from(target_total) - i128::from(base_total),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ArtifactKey, ManifestEntry};

    fn descriptor(key: u8, bytes: &[u8]) -> ArtifactDescriptor {
        ArtifactDescriptor::for_bytes(ArtifactKey([key; 32]), 1, 1, 1, Vec::new(), bytes)
    }

    fn manifest(
        revision: u64,
        entries: impl IntoIterator<Item = (u128, ArtifactDescriptor)>,
    ) -> Manifest {
        Manifest::new(
            BuildRevision(revision),
            entries
                .into_iter()
                .map(|(asset, descriptor)| ManifestEntry {
                    asset: AssetId(asset),
                    descriptor,
                }),
        )
        .expect("valid manifest")
    }

    #[test]
    fn patch_is_deterministic_and_reuses_content_by_hash() {
        let shared = descriptor(1, b"shared");
        let base = manifest(3, [(1, shared.clone()), (2, descriptor(2, b"removed"))]);
        let target = manifest(4, [(1, descriptor(9, b"shared")), (3, shared.clone())]);

        let plan = plan_patch(&base, &target).expect("patch plan");
        assert_eq!(plan.base_revision, BuildRevision(3));
        assert_eq!(plan.target_revision, BuildRevision(4));
        assert_eq!(plan.entries.len(), 3);
        assert_eq!(plan.entries[0].change, PatchChange::Changed);
        assert!(plan.entries[0].content_reused);
        assert_eq!(plan.entries[1].change, PatchChange::Removed);
        assert_eq!(plan.entries[2].change, PatchChange::Added);
        assert!(plan.entries[2].content_reused);
        assert_eq!(plan.download_bytes, 0);
        assert_eq!(plan.install_growth_bytes, -1);
    }

    #[test]
    fn patch_counts_each_new_content_hash_once() {
        let base = manifest(1, []);
        let target = manifest(
            2,
            [(1, descriptor(1, b"same")), (2, descriptor(2, b"same"))],
        );
        let plan = plan_patch(&base, &target).expect("patch plan");
        assert_eq!(plan.new_content.len(), 1);
        assert_eq!(plan.download_bytes, 4);
        assert_eq!(plan.install_growth_bytes, 8);
    }

    #[test]
    fn patch_rejects_non_newer_target() {
        let base = manifest(4, []);
        let target = manifest(4, []);
        assert_eq!(
            plan_patch(&base, &target),
            Err(PatchPlanError::TargetNotNewer {
                base: BuildRevision(4),
                target: BuildRevision(4),
            })
        );
    }
}
