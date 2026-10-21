//! Deterministic three-way merging of validated editable snapshots.
//!
//! Stable IDs identify records. Source, settings, preview and payload are atomic
//! values: no format is guessed from opaque bytes. Conflicts borrow all three
//! inputs; a caller must resolve every conflict before an owned result exists.
//! Payload storage is part of that atomic value: concurrent storage and content
//! edits require an explicit choice, without resolving external CAS objects.
//! Publication remains a workspace compare-and-save operation, so a resolved
//! merge cannot silently overwrite a newer saved revision.

use crate::editable::{
    EditableAsset, EditableAssetError, EditableDependency, EditableSubAsset,
    EditableSubAssetReference, SourceRecord, EDITABLE_FIXED_ENCODED_BYTES, MAX_DEPENDENCIES,
    MAX_EDITABLE_ASSET_BYTES, MAX_SUBASSETS, MAX_SUBASSET_REFERENCES,
};
use crate::{AssetId, EditablePayload};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

/// A stable conflict address. Dependency strengths for one target are one
/// value, so concurrent strength changes cannot silently create mixed edges.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum EditableMergePath {
    Kind,
    Flags,
    Settings,
    Source,
    AuthoringPayload,
    Preview,
    SubAsset(u64),
    SubAssetName(u64),
    SubAssetKind(u64),
    Dependencies(AssetId),
    SubAssetReferences { asset: AssetId, subasset: u64 },
}

/// Explicit selection for one conflicted value. Base also permits abandoning
/// both competing edits. A caller can edit the resolved asset before saving it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EditableMergeSide {
    Base,
    Local,
    Incoming,
}

/// Borrowed typed conflict evidence. Missing subassets and absent previews are
/// distinct from present empty values. No payload bytes are copied for display.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EditableMergeValue<'a> {
    U128(u128),
    U32(u32),
    Bytes(&'a [u8]),
    Payload(&'a EditablePayload),
    Source(&'a SourceRecord),
    Preview(Option<&'a [u8]>),
    SubAsset(Option<&'a EditableSubAsset>),
    Text(&'a str),
    Dependencies(&'a [EditableDependency]),
    SubAssetReferences(&'a [EditableSubAssetReference]),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EditableMergeConflict<'a> {
    pub path: EditableMergePath,
    pub base: EditableMergeValue<'a>,
    pub local: EditableMergeValue<'a>,
    pub incoming: EditableMergeValue<'a>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EditableMergeError {
    InvalidInput {
        side: EditableMergeSide,
        error: EditableAssetError,
    },
    IdentityMismatch {
        base: AssetId,
        local: AssetId,
        incoming: AssetId,
    },
    UnknownResolution(EditableMergePath),
    UnresolvedConflict(EditableMergePath),
    InvalidBudget {
        requested: u64,
        limit: u64,
    },
    OutputBudgetExceeded {
        actual: u64,
        limit: u64,
    },
    InvalidMerged(EditableAssetError),
}

impl fmt::Display for EditableMergeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput { side, error } => write!(f, "invalid {side:?} merge input: {error}"),
            Self::IdentityMismatch { base, local, incoming } => write!(
                f, "merge asset identities differ: base {base:?}, local {local:?}, incoming {incoming:?}"
            ),
            Self::UnknownResolution(path) => write!(f, "resolution has no matching conflict: {path:?}"),
            Self::UnresolvedConflict(path) => write!(f, "unresolved editable conflict: {path:?}"),
            Self::InvalidBudget { requested, limit } => write!(f, "invalid merge budget {requested}, expected 1..={limit}"),
            Self::OutputBudgetExceeded { actual, limit } => write!(f, "merged editable package needs {actual} bytes, limit is {limit}"),
            Self::InvalidMerged(error) => write!(f, "invalid merged editable asset: {error}"),
        }
    }
}

impl std::error::Error for EditableMergeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidInput { error, .. } | Self::InvalidMerged(error) => Some(error),
            _ => None,
        }
    }
}

/// Immutable merge evidence. Input validation is performed once, before any
/// comparison. Holding this plan borrows the snapshots and prevents their
/// mutation. Work is linear in collection sizes plus byte comparisons and
/// conflict sorting; only record metadata is allocated until resolution.
#[derive(Debug)]
pub struct EditableMergePlan<'a> {
    base: &'a EditableAsset,
    local: &'a EditableAsset,
    incoming: &'a EditableAsset,
    conflicts: Vec<EditableMergeConflict<'a>>,
}

/// Plan without reading, publishing, importing or touching a cache. All three
/// snapshots must have the same immutable asset identity. A source record is
/// atomic, keeping captured content, format and content hash together.
pub fn plan_editable_merge<'a>(
    base: &'a EditableAsset,
    local: &'a EditableAsset,
    incoming: &'a EditableAsset,
) -> Result<EditableMergePlan<'a>, EditableMergeError> {
    for (side, asset) in [
        (EditableMergeSide::Base, base),
        (EditableMergeSide::Local, local),
        (EditableMergeSide::Incoming, incoming),
    ] {
        asset
            .validate()
            .map_err(|error| EditableMergeError::InvalidInput { side, error })?;
    }
    if base.asset != local.asset || base.asset != incoming.asset {
        return Err(EditableMergeError::IdentityMismatch {
            base: base.asset,
            local: local.asset,
            incoming: incoming.asset,
        });
    }
    let mut selector = Selector {
        conflicts: Vec::new(),
        resolutions: None,
    };
    // This is only a borrowed projection. A conflicted plan never exposes an
    // implicit local choice as a merge result.
    let _ = project(base, local, incoming, &mut selector);
    selector
        .conflicts
        .sort_unstable_by_key(|conflict| conflict.path);
    Ok(EditableMergePlan {
        base,
        local,
        incoming,
        conflicts: selector.conflicts,
    })
}

impl<'a> EditableMergePlan<'a> {
    pub fn conflicts(&self) -> &[EditableMergeConflict<'a>] {
        &self.conflicts
    }

    /// Resolve all conflicts and materialize exactly one owned candidate.
    /// Unknown/stale addresses are rejected. The caller can tighten the hard
    /// package budget; zero or a larger budget is invalid. Encoded size and
    /// record counts are checked before cloning large selected byte buffers.
    /// This budget is not a process allocator sandbox. Workspace publication
    /// additionally validates cross-asset references and expected saved hashes.
    pub fn resolve(
        &self,
        resolutions: &BTreeMap<EditableMergePath, EditableMergeSide>,
        max_encoded_bytes: u64,
    ) -> Result<EditableAsset, EditableMergeError> {
        let hard_limit = MAX_EDITABLE_ASSET_BYTES as u64;
        if max_encoded_bytes == 0 || max_encoded_bytes > hard_limit {
            return Err(EditableMergeError::InvalidBudget {
                requested: max_encoded_bytes,
                limit: hard_limit,
            });
        }
        for path in resolutions.keys() {
            if self
                .conflicts
                .binary_search_by_key(path, |conflict| conflict.path)
                .is_err()
            {
                return Err(EditableMergeError::UnknownResolution(*path));
            }
        }
        for conflict in &self.conflicts {
            if !resolutions.contains_key(&conflict.path) {
                return Err(EditableMergeError::UnresolvedConflict(conflict.path));
            }
        }
        let projection = project(
            self.base,
            self.local,
            self.incoming,
            &mut Selector {
                conflicts: Vec::new(),
                resolutions: Some(resolutions),
            },
        );
        for (field, actual, limit) in [
            ("subassets", projection.subassets.len(), MAX_SUBASSETS),
            (
                "dependencies",
                projection.dependencies.len(),
                MAX_DEPENDENCIES,
            ),
            (
                "subasset references",
                projection.references.len(),
                MAX_SUBASSET_REFERENCES,
            ),
        ] {
            if actual > limit {
                return Err(EditableMergeError::InvalidMerged(
                    EditableAssetError::FieldTooLarge {
                        field,
                        actual,
                        limit,
                    },
                ));
            }
        }
        let actual = EDITABLE_FIXED_ENCODED_BYTES as u64 + projection.variable_bytes();
        if actual > max_encoded_bytes {
            return Err(EditableMergeError::OutputBudgetExceeded {
                actual,
                limit: max_encoded_bytes,
            });
        }
        let mut names = BTreeSet::new();
        for subasset in &projection.subassets {
            if !names.insert(subasset.name) {
                return Err(EditableMergeError::InvalidMerged(
                    EditableAssetError::DuplicateSubAssetName,
                ));
            }
        }
        let output = EditableAsset::new(
            self.base.asset,
            projection.kind,
            projection.flags,
            projection.settings.to_vec(),
            projection.source.clone(),
            projection
                .subassets
                .into_iter()
                .map(|entry| EditableSubAsset {
                    id: entry.id,
                    name: entry.name.to_owned(),
                    kind: entry.kind,
                })
                .collect(),
            projection.dependencies,
            projection.references,
            projection.payload.clone(),
            projection.preview.map(<[u8]>::to_vec),
        )
        .map_err(EditableMergeError::InvalidMerged)?;
        debug_assert_eq!(output.validated_encoded_len().ok(), Some(actual as usize));
        Ok(output)
    }
}

fn automatic<T: PartialEq + Copy>(base: T, local: T, incoming: T) -> Option<T> {
    if local == incoming || incoming == base {
        Some(local)
    } else if local == base {
        Some(incoming)
    } else {
        None
    }
}

struct Selector<'a, 'r> {
    conflicts: Vec<EditableMergeConflict<'a>>,
    resolutions: Option<&'r BTreeMap<EditableMergePath, EditableMergeSide>>,
}

impl<'a> Selector<'a, '_> {
    fn select<T: PartialEq + Copy>(
        &mut self,
        path: EditableMergePath,
        base: T,
        local: T,
        incoming: T,
        value: impl Fn(T) -> EditableMergeValue<'a>,
    ) -> T {
        if let Some(selected) = automatic(base, local, incoming) {
            return selected;
        }
        if let Some(resolutions) = self.resolutions {
            // resolve preflights every conflict against these immutable inputs.
            match resolutions[&path] {
                EditableMergeSide::Base => base,
                EditableMergeSide::Local => local,
                EditableMergeSide::Incoming => incoming,
            }
        } else {
            self.conflicts.push(EditableMergeConflict {
                path,
                base: value(base),
                local: value(local),
                incoming: value(incoming),
            });
            local
        }
    }
}

struct SubAsset<'a> {
    id: u64,
    name: &'a str,
    kind: u128,
}

struct Projection<'a> {
    kind: u128,
    flags: u32,
    settings: &'a [u8],
    source: &'a SourceRecord,
    payload: &'a EditablePayload,
    preview: Option<&'a [u8]>,
    subassets: Vec<SubAsset<'a>>,
    dependencies: Vec<EditableDependency>,
    references: Vec<EditableSubAssetReference>,
}

impl Projection<'_> {
    fn variable_bytes(&self) -> u64 {
        // Per-field limits on all three validated inputs bound these additions,
        // including the union's at-most-threefold record count.
        self.settings.len() as u64
            + self.source.uri.len() as u64
            + self.source.format.len() as u64
            + self.payload.encoded_len() as u64
            + self
                .source
                .embedded_source
                .as_ref()
                .map_or(0, |b| 4 + b.len() as u64)
            + self.preview.map_or(0, |b| 4 + b.len() as u64)
            + self
                .subassets
                .iter()
                .map(|e| 8 + 4 + 16 + e.name.len() as u64)
                .sum::<u64>()
            + self.dependencies.len() as u64 * (16 + 1)
            + self.references.len() as u64 * (16 + 8 + 1)
    }
}

fn project<'a>(
    base: &'a EditableAsset,
    local: &'a EditableAsset,
    incoming: &'a EditableAsset,
    selector: &mut Selector<'a, '_>,
) -> Projection<'a> {
    use EditableMergePath as P;
    use EditableMergeValue as V;
    let mut projection = Projection {
        kind: selector.select(P::Kind, base.kind, local.kind, incoming.kind, V::U128),
        flags: selector.select(P::Flags, base.flags, local.flags, incoming.flags, V::U32),
        settings: selector.select(
            P::Settings,
            base.canonical_settings.as_slice(),
            local.canonical_settings.as_slice(),
            incoming.canonical_settings.as_slice(),
            V::Bytes,
        ),
        source: selector.select(
            P::Source,
            &base.source,
            &local.source,
            &incoming.source,
            V::Source,
        ),
        payload: selector.select(
            P::AuthoringPayload,
            &base.authoring_payload,
            &local.authoring_payload,
            &incoming.authoring_payload,
            V::Payload,
        ),
        preview: selector.select(
            P::Preview,
            base.preview.as_deref(),
            local.preview.as_deref(),
            incoming.preview.as_deref(),
            V::Preview,
        ),
        subassets: Vec::new(),
        dependencies: Vec::new(),
        references: Vec::new(),
    };
    walk_groups(
        &base.subassets,
        &local.subassets,
        &incoming.subassets,
        |v| v.id,
        |id, b, l, i| {
            if let ([b], [l], [i]) = (b, l, i) {
                projection.subassets.push(SubAsset {
                    id,
                    name: selector.select(
                        P::SubAssetName(id),
                        b.name.as_str(),
                        l.name.as_str(),
                        i.name.as_str(),
                        V::Text,
                    ),
                    kind: selector.select(P::SubAssetKind(id), b.kind, l.kind, i.kind, V::U128),
                });
            } else if let Some(entry) = selector.select(
                P::SubAsset(id),
                b.first(),
                l.first(),
                i.first(),
                V::SubAsset,
            ) {
                projection.subassets.push(SubAsset {
                    id,
                    name: &entry.name,
                    kind: entry.kind,
                });
            }
        },
    );
    walk_groups(
        &base.dependencies,
        &local.dependencies,
        &incoming.dependencies,
        |v| v.asset,
        |asset, b, l, i| {
            projection.dependencies.extend_from_slice(selector.select(
                P::Dependencies(asset),
                b,
                l,
                i,
                V::Dependencies,
            ));
        },
    );
    walk_groups(
        &base.subasset_references,
        &local.subasset_references,
        &incoming.subasset_references,
        |v| (v.asset, v.subasset),
        |(asset, subasset), b, l, i| {
            projection.references.extend_from_slice(selector.select(
                P::SubAssetReferences { asset, subasset },
                b,
                l,
                i,
                V::SubAssetReferences,
            ));
        },
    );
    projection
}

/// Three-way linear walk of canonical record groups. No hash table or sort of
/// the input is needed, and strength groups retain their canonical order.
fn walk_groups<'a, T, K: Copy + Ord>(
    base: &'a [T],
    local: &'a [T],
    incoming: &'a [T],
    key: impl Fn(&T) -> K,
    mut visit: impl FnMut(K, &'a [T], &'a [T], &'a [T]),
) {
    let mut remaining = [base, local, incoming];
    while let Some(next) = remaining.iter().filter_map(|s| s.first().map(&key)).min() {
        let [b, l, i] = remaining.each_mut().map(|slice| {
            let count = slice.iter().take_while(|value| key(value) == next).count();
            let (group, rest) = slice.split_at(count);
            *slice = rest;
            group
        });
        visit(next, b, l, i);
    }
}
