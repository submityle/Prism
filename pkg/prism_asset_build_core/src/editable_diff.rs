//! Validated semantic differences between two editable asset snapshots.
//!
//! Changes borrow their inputs, including opaque payloads. Subassets match only
//! by stable ID; names never imply identity or redirects. Collection changes
//! follow canonical input order. Cook impact describes changes to the fields
//! visible through `EditableCookInput`, not a prediction of output bytes or
//! changes in external dependency artifacts.

use crate::editable::{
    EditableAsset, EditableAssetError, EditableDependency, EditableDependencyKind,
    EditableSubAsset, EditableSubAssetReference, SourceRecord,
};
use crate::types::AssetId;
use crate::EditablePayload;
use std::fmt;

/// Before and after values of one changed field. Large values remain borrowed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EditableValueChange<T> {
    pub before: T,
    pub after: T,
}

/// One changed subasset, ordered by stable ID in the enclosing diff.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EditableSubAssetDiff<'a> {
    Added(&'a EditableSubAsset),
    Removed(&'a EditableSubAsset),
    /// Both records have the same ID; name and/or kind changed. This reports
    /// facts and does not enforce reimport redirect or kind migration policy.
    Modified {
        before: &'a EditableSubAsset,
        after: &'a EditableSubAsset,
    },
}

/// Canonically ordered set differences. Kind transitions are remove/add pairs.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EditableSetDiff<'a, T> {
    pub added: Vec<&'a T>,
    pub removed: Vec<&'a T>,
}

impl<T> EditableSetDiff<'_, T> {
    pub fn is_empty(&self) -> bool {
        self.added.is_empty() && self.removed.is_empty()
    }
}

/// Independent impact flags; a diff can affect both domains.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct EditableDiffImpact {
    /// At least one cooker-visible input field changed.
    pub cook: bool,
    /// Source, preview, payload storage, or EditorOnly edge metadata changed.
    pub editor_import: bool,
}

/// Structured differences, without copying payload, settings, source or preview.
///
/// A changed asset identity is reported as a scalar change and affects cook.
/// This API compares two valid snapshots; it does not publish, reimport, resolve
/// external references or require both snapshots to have the same identity.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EditableAssetDiff<'a> {
    pub asset: Option<EditableValueChange<AssetId>>,
    pub kind: Option<EditableValueChange<u128>>,
    pub flags: Option<EditableValueChange<u32>>,
    pub canonical_settings: Option<EditableValueChange<&'a [u8]>>,
    pub source: Option<EditableValueChange<&'a SourceRecord>>,
    pub subassets: Vec<EditableSubAssetDiff<'a>>,
    pub dependencies: EditableSetDiff<'a, EditableDependency>,
    pub subasset_references: EditableSetDiff<'a, EditableSubAssetReference>,
    pub authoring_payload: Option<EditableValueChange<&'a EditablePayload>>,
    /// Preserves the distinction between no preview and an empty preview.
    pub preview: Option<EditableValueChange<Option<&'a [u8]>>>,
}

impl EditableAssetDiff<'_> {
    /// Content identity affects Cook; changing Inline/External storage affects
    /// editor/import metadata. External content is compared by hash and length
    /// without reading CAS. Materialization must still verify those bytes.
    pub fn authoring_payload_impact(&self) -> EditableDiffImpact {
        self.authoring_payload
            .map_or(EditableDiffImpact::default(), |change| EditableDiffImpact {
                cook: !change.before.semantic_eq(change.after),
                editor_import: std::mem::discriminant(change.before)
                    != std::mem::discriminant(change.after),
            })
    }

    /// Mirrors the projection into `EditableCookInput`: source, preview and
    /// EditorOnly edges are excluded; subasset names and all flags are included.
    pub fn impact(&self) -> EditableDiffImpact {
        let dependency_kinds = self
            .dependencies
            .added
            .iter()
            .chain(&self.dependencies.removed)
            .map(|edge| edge.kind);
        let reference_kinds = self
            .subasset_references
            .added
            .iter()
            .chain(&self.subasset_references.removed)
            .map(|edge| edge.kind);
        let payload = self.authoring_payload_impact();
        let mut impact = EditableDiffImpact {
            cook: self.asset.is_some()
                || self.kind.is_some()
                || self.flags.is_some()
                || self.canonical_settings.is_some()
                || !self.subassets.is_empty()
                || payload.cook,
            editor_import: self.source.is_some() || self.preview.is_some() || payload.editor_import,
        };
        for kind in dependency_kinds.chain(reference_kinds) {
            if kind == EditableDependencyKind::EditorOnly {
                impact.editor_import = true;
            } else {
                impact.cook = true;
            }
        }
        impact
    }

    /// Whether this diff changes the saved input visible to a cooker.
    pub fn affects_cook(&self) -> bool {
        self.impact().cook
    }

    /// Whether all compared fields are unchanged, including editor metadata.
    pub fn is_empty(&self) -> bool {
        self.impact() == EditableDiffImpact::default()
    }
}

/// Validation failure with the offending snapshot identified. Before is checked
/// first, so failures are deterministic even when both inputs are invalid.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EditableDiffError {
    InvalidBefore(EditableAssetError),
    InvalidAfter(EditableAssetError),
}

impl fmt::Display for EditableDiffError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidBefore(error) => write!(f, "invalid before editable asset: {error}"),
            Self::InvalidAfter(error) => write!(f, "invalid after editable asset: {error}"),
        }
    }
}

impl std::error::Error for EditableDiffError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidBefore(error) | Self::InvalidAfter(error) => Some(error),
        }
    }
}

/// Validates both inputs and compares exact semantic fields without encoding or
/// cloning them. Inline bytes compare exactly; external payload content identity
/// uses both hash and length, while storage changes remain visible metadata.
/// Allocation is proportional to changed records;
/// sorted collections are merged in linear time.
pub fn diff_editable_assets<'a>(
    before: &'a EditableAsset,
    after: &'a EditableAsset,
) -> Result<EditableAssetDiff<'a>, EditableDiffError> {
    before
        .validate()
        .map_err(EditableDiffError::InvalidBefore)?;
    after.validate().map_err(EditableDiffError::InvalidAfter)?;

    let mut subassets = Vec::new();
    merge_by(
        &before.subassets,
        &after.subassets,
        |left, right| left.id.cmp(&right.id),
        |left, right| match (left, right) {
            (Some(before), Some(after)) if before != after => {
                subassets.push(EditableSubAssetDiff::Modified { before, after });
            }
            (Some(value), None) => subassets.push(EditableSubAssetDiff::Removed(value)),
            (None, Some(value)) => subassets.push(EditableSubAssetDiff::Added(value)),
            _ => {}
        },
    );
    Ok(EditableAssetDiff {
        asset: changed(before.asset, after.asset),
        kind: changed(before.kind, after.kind),
        flags: changed(before.flags, after.flags),
        canonical_settings: changed(
            before.canonical_settings.as_slice(),
            after.canonical_settings.as_slice(),
        ),
        source: changed(&before.source, &after.source),
        subassets,
        dependencies: set_diff(&before.dependencies, &after.dependencies),
        subasset_references: set_diff(&before.subasset_references, &after.subasset_references),
        authoring_payload: changed(&before.authoring_payload, &after.authoring_payload),
        preview: changed(before.preview.as_deref(), after.preview.as_deref()),
    })
}

fn changed<T: PartialEq>(before: T, after: T) -> Option<EditableValueChange<T>> {
    if before == after {
        None
    } else {
        Some(EditableValueChange { before, after })
    }
}

fn set_diff<'a, T: Ord>(before: &'a [T], after: &'a [T]) -> EditableSetDiff<'a, T> {
    let mut diff = EditableSetDiff {
        added: Vec::new(),
        removed: Vec::new(),
    };
    merge_by(before, after, T::cmp, |left, right| match (left, right) {
        (Some(value), None) => diff.removed.push(value),
        (None, Some(value)) => diff.added.push(value),
        _ => {}
    });
    diff
}

fn merge_by<'a, T>(
    before: &'a [T],
    after: &'a [T],
    compare: impl Fn(&T, &T) -> std::cmp::Ordering,
    mut visit: impl FnMut(Option<&'a T>, Option<&'a T>),
) {
    let (mut left, mut right) = (0, 0);
    while left < before.len() || right < after.len() {
        let ordering = match (before.get(left), after.get(right)) {
            (Some(a), Some(b)) => compare(a, b),
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (None, None) => break,
        };
        match ordering {
            std::cmp::Ordering::Less => {
                visit(Some(&before[left]), None);
                left += 1;
            }
            std::cmp::Ordering::Greater => {
                visit(None, Some(&after[right]));
                right += 1;
            }
            std::cmp::Ordering::Equal => {
                visit(Some(&before[left]), Some(&after[right]));
                left += 1;
                right += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::editable::EditableDependencyKind::{EditorOnly, Hard, Soft};

    fn asset() -> EditableAsset {
        EditableAsset::new(
            AssetId(1),
            42,
            0,
            b"settings".to_vec(),
            SourceRecord::embedded("source.bin", "test/raw", b"source".to_vec()).unwrap(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            b"payload".to_vec(),
            None,
        )
        .unwrap()
    }

    fn subasset(id: u64, name: &str, kind: u128) -> EditableSubAsset {
        EditableSubAsset {
            id,
            name: name.into(),
            kind,
        }
    }

    fn dependency(id: u128, kind: EditableDependencyKind) -> EditableDependency {
        EditableDependency {
            asset: AssetId(id),
            kind,
        }
    }

    fn reference(
        id: u128,
        subasset: u64,
        kind: EditableDependencyKind,
    ) -> EditableSubAssetReference {
        EditableSubAssetReference {
            asset: AssetId(id),
            subasset,
            kind,
        }
    }

    #[test]
    fn identical_snapshots_have_no_changes() {
        let before = asset();
        let after = before.clone();
        let diff = diff_editable_assets(&before, &after).unwrap();
        assert!(diff.is_empty());
        assert!(!diff.affects_cook());
        assert!(diff.dependencies.is_empty());
        assert!(diff.subasset_references.is_empty());
    }

    #[test]
    fn subassets_match_ids_and_changes_have_global_id_order() {
        let mut before = asset();
        before.subassets = vec![
            subasset(2, "old", 42),
            subasset(4, "same-name", 42),
            subasset(6, "kind", 42),
            subasset(8, "unchanged", 42),
        ];
        let mut after = before.clone();
        after.subassets = vec![
            subasset(1, "first", 42),
            subasset(2, "renamed", 42),
            subasset(5, "same-name", 42),
            subasset(6, "kind", 43),
            subasset(8, "unchanged", 42),
        ];
        let diff = diff_editable_assets(&before, &after).unwrap();
        assert_eq!(
            diff.subassets,
            vec![
                EditableSubAssetDiff::Added(&after.subassets[0]),
                EditableSubAssetDiff::Modified {
                    before: &before.subassets[0],
                    after: &after.subassets[1]
                },
                EditableSubAssetDiff::Removed(&before.subassets[1]),
                EditableSubAssetDiff::Added(&after.subassets[2]),
                EditableSubAssetDiff::Modified {
                    before: &before.subassets[2],
                    after: &after.subassets[3]
                },
            ]
        );
        assert_eq!(diff, diff_editable_assets(&before, &after).unwrap());
        assert!(diff.affects_cook());
    }

    #[test]
    fn edge_changes_are_canonical_and_kind_transitions_are_remove_add() {
        let mut before = asset();
        before.dependencies = vec![
            dependency(2, Hard),
            dependency(4, EditorOnly),
            dependency(7, Soft),
        ];
        before.subasset_references = vec![reference(2, 1, Hard), reference(4, 3, EditorOnly)];
        let mut after = before.clone();
        after.dependencies = vec![
            dependency(2, Soft),
            dependency(3, Hard),
            dependency(7, Soft),
        ];
        after.subasset_references = vec![reference(2, 1, Soft), reference(3, 2, Hard)];
        let diff = diff_editable_assets(&before, &after).unwrap();
        assert_eq!(
            diff.dependencies.removed,
            vec![&before.dependencies[0], &before.dependencies[1]]
        );
        assert_eq!(
            diff.dependencies.added,
            vec![&after.dependencies[0], &after.dependencies[1]]
        );
        assert_eq!(
            diff.subasset_references.removed,
            before.subasset_references.iter().collect::<Vec<_>>()
        );
        assert_eq!(
            diff.subasset_references.added,
            after.subasset_references.iter().collect::<Vec<_>>()
        );
        assert_eq!(
            diff.impact(),
            EditableDiffImpact {
                cook: true,
                editor_import: true
            }
        );
    }

    #[test]
    fn source_preview_and_editor_only_edges_do_not_affect_cook() {
        let before = asset();
        let mut after = before.clone();
        after.source =
            SourceRecord::embedded("other.bin", "new/format", b"new source".to_vec()).unwrap();
        after.preview = Some(Vec::new());
        after.dependencies = vec![dependency(2, EditorOnly)];
        after.subasset_references = vec![reference(2, 3, EditorOnly)];
        let diff = diff_editable_assets(&before, &after).unwrap();
        assert!(diff.source.is_some());
        assert_eq!(
            diff.preview,
            Some(EditableValueChange {
                before: None,
                after: Some(&[][..])
            })
        );
        assert_eq!(
            diff.impact(),
            EditableDiffImpact {
                cook: false,
                editor_import: true
            }
        );
        assert!(!diff.is_empty());
        let reversed = diff_editable_assets(&after, &before).unwrap();
        assert!(!reversed.affects_cook());
        assert_eq!(reversed.dependencies.removed, vec![&after.dependencies[0]]);
    }

    #[test]
    fn every_cooker_visible_field_affects_cook_independently() {
        let before = asset();
        let changes: &[fn(&mut EditableAsset)] = &[
            |a| a.asset = AssetId(9),
            |a| a.kind += 1,
            |a| a.flags = 1,
            |a| a.canonical_settings.push(1),
            |a| a.authoring_payload = b"changed payload".to_vec().into(),
            |a| a.subassets.push(subasset(1, "part", 42)),
            |a| a.dependencies.push(dependency(2, Hard)),
            |a| a.dependencies.push(dependency(2, Soft)),
            |a| a.subasset_references.push(reference(2, 1, Hard)),
            |a| a.subasset_references.push(reference(2, 1, Soft)),
        ];
        for mutate in changes {
            let mut after = before.clone();
            mutate(&mut after);
            let diff = diff_editable_assets(&before, &after).unwrap();
            assert_eq!(
                diff.impact(),
                EditableDiffImpact {
                    cook: true,
                    editor_import: false
                }
            );
        }
    }

    #[test]
    fn editor_only_to_soft_transition_affects_both_domains() {
        let mut before = asset();
        before.dependencies = vec![dependency(2, EditorOnly)];
        let mut after = before.clone();
        after.dependencies[0].kind = Soft;
        assert_eq!(
            diff_editable_assets(&before, &after).unwrap().impact(),
            EditableDiffImpact {
                cook: true,
                editor_import: true
            }
        );
    }

    #[test]
    fn editor_only_edges_can_change_alongside_retained_cook_edges() {
        for kind in [Hard, Soft] {
            let mut before = asset();
            before.dependencies = vec![dependency(2, kind)];
            before.subasset_references = vec![reference(2, 3, kind)];
            let mut after = before.clone();
            after.dependencies.push(dependency(2, EditorOnly));
            after.subasset_references.push(reference(2, 3, EditorOnly));
            let diff = diff_editable_assets(&before, &after).unwrap();
            assert_eq!(diff.dependencies.added, vec![&after.dependencies[1]]);
            assert!(diff.dependencies.removed.is_empty());
            assert_eq!(
                diff.subasset_references.added,
                vec![&after.subasset_references[1]]
            );
            assert!(diff.subasset_references.removed.is_empty());
            assert_eq!(
                diff.impact(),
                EditableDiffImpact {
                    cook: false,
                    editor_import: true
                }
            );
        }
    }

    #[test]
    fn source_storage_and_preview_bytes_are_metadata_only() {
        let mut before = asset();
        before.preview = Some(vec![1, 2, 3]);
        let mut after = before.clone();
        // Same source digest, different storage: the editable metadata changed.
        after.source.embedded_source = None;
        after.preview.as_mut().unwrap()[2] = 4;
        let diff = diff_editable_assets(&before, &after).unwrap();
        let source = diff.source.unwrap();
        assert_eq!(source.before.content_hash, source.after.content_hash);
        assert!(std::ptr::eq(source.before, &before.source));
        assert!(std::ptr::eq(source.after, &after.source));
        assert_eq!(
            diff.preview,
            Some(EditableValueChange {
                before: Some(&[1, 2, 3][..]),
                after: Some(&[1, 2, 4][..]),
            })
        );
        assert_eq!(
            diff.impact(),
            EditableDiffImpact {
                cook: false,
                editor_import: true
            }
        );
    }

    #[test]
    fn same_length_payload_changes_are_exact_and_borrow_original_allocations() {
        let mut before = asset();
        let mut after = asset();
        before.authoring_payload = vec![0; 1024 * 1024].into();
        let mut bytes = vec![0; 1024 * 1024];
        *bytes.last_mut().unwrap() = 1;
        after.authoring_payload = bytes.into();
        after.canonical_settings[0] = b'S';
        after.preview = Some(vec![1, 2, 3]);
        let diff = diff_editable_assets(&before, &after).unwrap();
        let payload = diff.authoring_payload.unwrap();
        assert!(std::ptr::eq(payload.before, &before.authoring_payload));
        assert!(std::ptr::eq(payload.after, &after.authoring_payload));
        assert_eq!(
            diff.canonical_settings.unwrap().after.as_ptr(),
            after.canonical_settings.as_ptr()
        );
        assert_eq!(
            diff.preview.unwrap().after.unwrap().as_ptr(),
            after.preview.as_ref().unwrap().as_ptr()
        );
        assert!(diff.affects_cook());
    }

    #[test]
    fn payload_storage_changes_are_nonempty_editor_only_in_both_directions() {
        let before = asset();
        let mut after = before.clone();
        after.authoring_payload = EditablePayload::External(crate::EditablePayloadRef {
            content_hash: before.authoring_payload.content_hash(),
            byte_len: before.authoring_payload.byte_len(),
        });
        for (before, after) in [(&before, &after), (&after, &before)] {
            let diff = diff_editable_assets(before, after).unwrap();
            assert!(!diff.is_empty());
            assert_eq!(
                diff.impact(),
                EditableDiffImpact {
                    cook: false,
                    editor_import: true
                }
            );
            assert_eq!(diff.authoring_payload_impact(), diff.impact());
            let change = diff.authoring_payload.unwrap();
            assert!(std::ptr::eq(change.before, &before.authoring_payload));
            assert!(std::ptr::eq(change.after, &after.authoring_payload));
        }
        assert!(diff_editable_assets(&after, &after).unwrap().is_empty());
    }

    #[test]
    fn external_payload_content_requires_both_hash_and_length() {
        let mut before = asset();
        before.authoring_payload = EditablePayload::External(crate::EditablePayloadRef {
            content_hash: before.authoring_payload.content_hash(),
            byte_len: before.authoring_payload.byte_len(),
        });
        for reference in [
            crate::EditablePayloadRef {
                content_hash: crate::ContentHash::digest(b"other"),
                byte_len: before.authoring_payload.byte_len(),
            },
            crate::EditablePayloadRef {
                content_hash: before.authoring_payload.content_hash(),
                byte_len: before.authoring_payload.byte_len() + 1,
            },
        ] {
            let mut after = before.clone();
            after.authoring_payload = EditablePayload::External(reference);
            let diff = diff_editable_assets(&before, &after).unwrap();
            assert_eq!(
                diff.impact(),
                EditableDiffImpact {
                    cook: true,
                    editor_import: false
                }
            );
            assert!(!diff.is_empty());
        }
        let after = asset();
        before.authoring_payload = EditablePayload::External(crate::EditablePayloadRef {
            content_hash: crate::ContentHash::digest(b"other"),
            byte_len: after.authoring_payload.byte_len(),
        });
        assert_eq!(
            diff_editable_assets(&before, &after).unwrap().impact(),
            EditableDiffImpact {
                cook: true,
                editor_import: true
            }
        );
    }

    #[test]
    fn validation_rejects_bad_inputs_even_if_both_are_identical() {
        let valid = asset();
        let mut invalid = valid.clone();
        invalid.dependencies = vec![dependency(3, Soft), dependency(2, Hard)];
        assert_eq!(
            diff_editable_assets(&invalid, &invalid),
            Err(EditableDiffError::InvalidBefore(
                EditableAssetError::NonCanonicalDependencies
            ))
        );
        assert_eq!(
            diff_editable_assets(&valid, &invalid),
            Err(EditableDiffError::InvalidAfter(
                EditableAssetError::NonCanonicalDependencies
            ))
        );
        invalid = valid.clone();
        invalid.subassets = vec![subasset(1, "a", 42), subasset(1, "b", 42)];
        assert_eq!(
            diff_editable_assets(&valid, &invalid),
            Err(EditableDiffError::InvalidAfter(
                EditableAssetError::NonCanonicalSubassets
            ))
        );
        invalid = valid.clone();
        invalid.subasset_references = vec![reference(2, 1, Soft), reference(2, 1, Soft)];
        assert_eq!(
            diff_editable_assets(&valid, &invalid),
            Err(EditableDiffError::InvalidAfter(
                EditableAssetError::NonCanonicalSubAssetReferences
            ))
        );
        invalid = valid.clone();
        invalid.dependencies = vec![dependency(1, Hard)];
        assert_eq!(
            diff_editable_assets(&valid, &invalid),
            Err(EditableDiffError::InvalidAfter(
                EditableAssetError::SelfDependency
            ))
        );
        invalid = valid.clone();
        invalid.source.embedded_source.as_mut().unwrap().push(1);
        assert_eq!(
            diff_editable_assets(&valid, &invalid),
            Err(EditableDiffError::InvalidAfter(
                EditableAssetError::EmbeddedSourceHashMismatch
            ))
        );
    }
}
