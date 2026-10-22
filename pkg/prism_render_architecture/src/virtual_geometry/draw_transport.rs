//! Versioned, device-independent CLST draw transport shared by geometry
//! planning and render backends.
//!
//! The transport contains logical identities and descriptor ranges only. It
//! deliberately has no ECS or device-resource concepts, so batches can cross
//! world/backend boundaries without leaking entities, buffers, pipelines, or
//! bind groups. Construction canonicalizes record order and rejects ambiguous
//! or stale identities before a batch can be published.

use alloc::vec::Vec;
use core::fmt;

use crate::{abi::GenerationalHandle, virtual_geometry::GeometryPageKey};

/// Current wire-contract version for [`GeometryDrawBatch`].
pub const GEOMETRY_DRAW_TRANSPORT_VERSION: u32 = 1;

/// Stable CLST asset identity and the generations that qualify its content and
/// current runtime publication.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ClstAssetIdentity {
    /// Stable asset registry index; `u32::MAX` is reserved.
    pub asset_id: u32,
    /// Digest supplied by the content publisher.
    pub content_revision: [u8; 32],
    /// Runtime publication generation containing resident pages. This is the
    /// generation allocated by `VirtualGeometryRuntimeRegistry`.
    pub publication_generation: u64,
}

/// Logical cluster descriptor carried between planning and rendering.
///
/// `payload_offset` and `payload_size` address bytes in the page named by the
/// enclosing record. The indices are asset-local, not GPU addresses.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ClstClusterDescriptor {
    /// Stable CLST descriptor-table index.
    pub cluster_index: u32,
    /// Hierarchy node represented by this descriptor.
    pub node_index: u32,
    /// Byte offset within the logical page payload.
    pub payload_offset: u32,
    /// Byte length of the cluster payload.
    pub payload_size: u32,
    /// Number of vertices decoded from this cluster.
    pub vertex_count: u32,
    /// Number of triangles decoded from this cluster.
    pub triangle_count: u32,
}

/// Common revisions identifying one immutable planning snapshot.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GeometryDrawBatchHeader {
    /// Transport schema version. Unsupported versions are rejected.
    pub version: u32,
    /// Monotonic frame revision.
    pub frame_revision: u64,
    /// Monotonic source snapshot revision used to plan this frame.
    pub snapshot_revision: u64,
}

/// One logical cluster selected for a view in a scene snapshot.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GeometryDrawRecord {
    /// Generational view identity.
    pub view: GenerationalHandle,
    /// Generational scene identity.
    pub scene: GenerationalHandle,
    /// Generational geometry identity.
    pub geometry: GenerationalHandle,
    /// Revision of the geometry content/configuration bound to the identity.
    pub geometry_revision: u64,
    /// Revision of the cut/selection that produced this draw.
    pub selection_revision: u64,
    /// CLST content/publication identity.
    pub asset: ClstAssetIdentity,
    /// Logical page containing the cluster payload.
    pub page: GeometryPageKey,
    /// Asset-local CLST cluster descriptor.
    pub cluster: ClstClusterDescriptor,
    /// Generational material identity used to shade the cluster.
    pub material: GenerationalHandle,
}

/// Canonically ordered, validated records for one planning snapshot.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GeometryDrawBatch {
    header: GeometryDrawBatchHeader,
    records: Vec<GeometryDrawRecord>,
}

impl GeometryDrawBatch {
    /// Validates and constructs a batch, sorting records by their complete
    /// logical draw key for deterministic cross-backend processing.
    pub fn new(
        header: GeometryDrawBatchHeader,
        mut records: Vec<GeometryDrawRecord>,
    ) -> Result<Self, GeometryDrawBatchError> {
        if header.version != GEOMETRY_DRAW_TRANSPORT_VERSION {
            return Err(GeometryDrawBatchError::UnknownVersion(header.version));
        }
        if header.frame_revision == 0 {
            return Err(GeometryDrawBatchError::InvalidFrameRevision);
        }
        if header.snapshot_revision == 0 {
            return Err(GeometryDrawBatchError::InvalidSnapshotRevision);
        }
        u32::try_from(records.len()).map_err(|_| GeometryDrawBatchError::RecordCountOverflow)?;

        for (index, record) in records.iter().enumerate() {
            validate_handle(record.view)?;
            validate_handle(record.scene)?;
            validate_handle(record.geometry)?;
            validate_handle(record.material)?;
            validate_asset(record.asset)?;
            if record.geometry_revision == 0 {
                return Err(GeometryDrawBatchError::InvalidGeometryRevision { index });
            }
            if record.selection_revision == 0 {
                return Err(GeometryDrawBatchError::InvalidSelectionRevision { index });
            }
            if record.page.asset != record.asset.asset_id {
                return Err(GeometryDrawBatchError::PageAssetMismatch { index });
            }
            validate_cluster(record.cluster, index)?;
        }

        records.sort_unstable_by_key(|record| {
            (
                record.view,
                record.scene,
                record.geometry,
                record.geometry_revision,
                record.selection_revision,
                record.asset,
                record.page,
                record.cluster,
                record.material,
            )
        });
        for pair in records.windows(2) {
            if same_draw_key(pair[0], pair[1]) {
                return Err(GeometryDrawBatchError::DuplicateDraw {
                    view: pair[0].view,
                    scene: pair[0].scene,
                    geometry: pair[0].geometry,
                    page: pair[0].page,
                    cluster_index: pair[0].cluster.cluster_index,
                });
            }
        }

        Ok(Self { header, records })
    }

    /// Validates an already materialized batch, including its canonical order.
    /// This is useful at trust boundaries where the value came from decoding.
    pub fn validate(&self) -> Result<(), GeometryDrawBatchError> {
        let rebuilt = Self::new(self.header, self.records.clone())?;
        if rebuilt.records != self.records {
            return Err(GeometryDrawBatchError::NonCanonicalOrder);
        }
        Ok(())
    }

    /// Shared frame and source-snapshot revisions.
    #[must_use]
    pub const fn header(&self) -> GeometryDrawBatchHeader {
        self.header
    }

    /// Canonically sorted logical draw records.
    #[must_use]
    pub fn records(&self) -> &[GeometryDrawRecord] {
        &self.records
    }

    /// Number of records, representable in the transport's `u32` count field.
    #[must_use]
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Whether this batch contains no cluster draws.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }
}

fn validate_handle(handle: GenerationalHandle) -> Result<(), GeometryDrawBatchError> {
    if !handle.is_valid() {
        return Err(GeometryDrawBatchError::InvalidIdentity);
    }
    if handle.generation == 0 {
        return Err(GeometryDrawBatchError::InvalidGeneration);
    }
    Ok(())
}

fn validate_asset(asset: ClstAssetIdentity) -> Result<(), GeometryDrawBatchError> {
    if asset.asset_id == u32::MAX {
        return Err(GeometryDrawBatchError::InvalidIdentity);
    }
    if asset.publication_generation == 0 {
        return Err(GeometryDrawBatchError::InvalidGeneration);
    }
    Ok(())
}

fn validate_cluster(
    cluster: ClstClusterDescriptor,
    index: usize,
) -> Result<(), GeometryDrawBatchError> {
    if cluster.cluster_index == u32::MAX || cluster.node_index == u32::MAX {
        return Err(GeometryDrawBatchError::InvalidClusterIdentity { index });
    }
    if cluster.payload_size == 0 || cluster.vertex_count == 0 || cluster.triangle_count == 0 {
        return Err(GeometryDrawBatchError::EmptyClusterDescriptor { index });
    }
    cluster
        .payload_offset
        .checked_add(cluster.payload_size)
        .ok_or(GeometryDrawBatchError::ClusterRangeOverflow { index })?;
    cluster
        .triangle_count
        .checked_mul(3)
        .ok_or(GeometryDrawBatchError::ClusterIndexCountOverflow { index })?;
    Ok(())
}

fn same_draw_key(left: GeometryDrawRecord, right: GeometryDrawRecord) -> bool {
    left.view == right.view
        && left.scene == right.scene
        && left.geometry == right.geometry
        && left.geometry_revision == right.geometry_revision
        && left.selection_revision == right.selection_revision
        && left.asset == right.asset
        && left.page == right.page
        && left.cluster.cluster_index == right.cluster.cluster_index
}

/// Why a logical draw batch was rejected.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GeometryDrawBatchError {
    /// The batch schema is not supported by this implementation.
    UnknownVersion(u32),
    /// Frame revision zero is reserved as invalid.
    InvalidFrameRevision,
    /// Snapshot revision zero is reserved as invalid.
    InvalidSnapshotRevision,
    /// Record count cannot be represented by the transport count field.
    RecordCountOverflow,
    /// An identity uses its reserved invalid index.
    InvalidIdentity,
    /// A generational identity or publication uses generation zero.
    InvalidGeneration,
    /// A geometry revision is zero.
    InvalidGeometryRevision { index: usize },
    /// A cut/selection revision is zero.
    InvalidSelectionRevision { index: usize },
    /// The logical page belongs to a different asset than the descriptor.
    PageAssetMismatch { index: usize },
    /// A cluster index or hierarchy node uses its reserved invalid value.
    InvalidClusterIdentity { index: usize },
    /// A required cluster range/count is empty.
    EmptyClusterDescriptor { index: usize },
    /// Payload offset plus length exceeds the descriptor address space.
    ClusterRangeOverflow { index: usize },
    /// Triangle-to-index expansion overflows `u32`.
    ClusterIndexCountOverflow { index: usize },
    /// Two records name the same logical draw in the same snapshot.
    DuplicateDraw {
        view: GenerationalHandle,
        scene: GenerationalHandle,
        geometry: GenerationalHandle,
        page: GeometryPageKey,
        cluster_index: u32,
    },
    /// Decoded records are valid but not in canonical order.
    NonCanonicalOrder,
}

impl fmt::Display for GeometryDrawBatchError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid geometry draw batch: {self:?}")
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;

    fn handle(index: u32) -> GenerationalHandle {
        GenerationalHandle::new(index, 1)
    }

    fn record(cluster_index: u32) -> GeometryDrawRecord {
        GeometryDrawRecord {
            view: handle(1),
            scene: handle(2),
            geometry: handle(3),
            geometry_revision: 4,
            selection_revision: 5,
            asset: ClstAssetIdentity {
                asset_id: 6,
                content_revision: [8; 32],
                publication_generation: 9,
            },
            page: GeometryPageKey::new(6, 10),
            cluster: ClstClusterDescriptor {
                cluster_index,
                node_index: cluster_index + 20,
                payload_offset: 0,
                payload_size: 64,
                vertex_count: 3,
                triangle_count: 1,
            },
            material: handle(11),
        }
    }

    fn header() -> GeometryDrawBatchHeader {
        GeometryDrawBatchHeader {
            version: GEOMETRY_DRAW_TRANSPORT_VERSION,
            frame_revision: 12,
            snapshot_revision: 13,
        }
    }

    #[test]
    fn sorts_records_deterministically_and_validates() {
        let batch = GeometryDrawBatch::new(header(), vec![record(3), record(1), record(2)])
            .expect("valid draw records");
        assert_eq!(
            batch
                .records()
                .iter()
                .map(|record| record.cluster.cluster_index)
                .collect::<Vec<_>>(),
            [1, 2, 3]
        );
        assert_eq!(batch.header(), header());
        assert_eq!(batch.len(), 3);
        batch.validate().expect("canonical validated batch");
    }

    #[test]
    fn rejects_unknown_versions_and_zero_snapshot_revisions() {
        let mut unsupported = header();
        unsupported.version += 1;
        assert_eq!(
            GeometryDrawBatch::new(unsupported, vec![]),
            Err(GeometryDrawBatchError::UnknownVersion(unsupported.version))
        );
        let mut invalid = header();
        invalid.snapshot_revision = 0;
        assert_eq!(
            GeometryDrawBatch::new(invalid, vec![]),
            Err(GeometryDrawBatchError::InvalidSnapshotRevision)
        );
    }

    #[test]
    fn rejects_duplicate_cluster_draws_even_when_material_differs() {
        let mut second = record(4);
        second.material = handle(12);
        let duplicate = record(4);
        assert!(matches!(
            GeometryDrawBatch::new(header(), vec![duplicate, second]),
            Err(GeometryDrawBatchError::DuplicateDraw {
                cluster_index: 4,
                ..
            })
        ));
    }

    #[test]
    fn rejects_invalid_generations_and_reserved_identity_indices() {
        let mut invalid = record(1);
        invalid.geometry.generation = 0;
        assert_eq!(
            GeometryDrawBatch::new(header(), vec![invalid]),
            Err(GeometryDrawBatchError::InvalidGeneration)
        );
        let mut invalid = record(1);
        invalid.material.index = u32::MAX;
        assert_eq!(
            GeometryDrawBatch::new(header(), vec![invalid]),
            Err(GeometryDrawBatchError::InvalidIdentity)
        );
        let mut invalid = record(1);
        invalid.asset.publication_generation = 0;
        assert_eq!(
            GeometryDrawBatch::new(header(), vec![invalid]),
            Err(GeometryDrawBatchError::InvalidGeneration)
        );
    }

    #[test]
    fn rejects_mismatched_page_and_asset_and_overflowing_ranges() {
        let mut mismatch = record(1);
        mismatch.page.asset = 99;
        assert_eq!(
            GeometryDrawBatch::new(header(), vec![mismatch]),
            Err(GeometryDrawBatchError::PageAssetMismatch { index: 0 })
        );
        let mut overflow = record(1);
        overflow.cluster.payload_offset = u32::MAX;
        assert_eq!(
            GeometryDrawBatch::new(header(), vec![overflow]),
            Err(GeometryDrawBatchError::ClusterRangeOverflow { index: 0 })
        );
    }

    #[test]
    fn rejects_empty_descriptors() {
        let mut empty = record(1);
        empty.cluster.triangle_count = 0;
        assert_eq!(
            GeometryDrawBatch::new(header(), vec![empty]),
            Err(GeometryDrawBatchError::EmptyClusterDescriptor { index: 0 })
        );
    }
}
