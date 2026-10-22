//! Deterministic per-view production draw records for geometry selections.
//!
//! This is the scene-side boundary between visibility decisions and backend
//! draw preparation. Every selection is resolved against current snapshots;
//! one stale item rejects the complete phase rather than leaking partial work.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use prism_render_architecture::{
    gpu_scene::{CpuRenderScene, SceneHandle},
    virtual_geometry::{
        CutCluster, GeometryPageKey, VirtualGeometryGpuDrawItem, VirtualGeometryGpuPublication,
    },
};

use super::{
    representation::{
        GeometryRepresentation, GeometryRepresentationRejection, GeometryRepresentationSelections,
        VirtualGeometryRepresentationIdentity,
    },
    runtime::RenderGeometryRegistry,
};

/// One backend-neutral draw record emitted for an exact per-view selection.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GeometryDrawRecord {
    /// Resident prepared mesh LOD with exact physical range and buffer classes.
    Lod {
        scene: SceneHandle,
        geometry: prism_render_architecture::gpu_scene::GeometryHandle,
        geometry_revision: u32,
        selection_revision: u64,
        level: u32,
        primitive_kind: prism_render_architecture::geometry::GeometryPrimitiveKind,
        element_count: u32,
        first_element: u32,
        base_vertex: i32,
        vertex_count: u32,
        vertex_buffer_class: u32,
        index_buffer_class: u32,
    },
    /// Resident CLST cluster draw item with publication identity proof.
    Cluster {
        scene: SceneHandle,
        geometry: prism_render_architecture::gpu_scene::GeometryHandle,
        geometry_revision: u32,
        selection_revision: u64,
        publication: VirtualGeometryRepresentationIdentity,
        page: GeometryPageKey,
        item: VirtualGeometryGpuDrawItem,
    },
}

impl GeometryDrawRecord {
    /// Scene instance whose draw this record represents.
    #[must_use]
    pub const fn scene(self) -> SceneHandle {
        match self {
            Self::Lod { scene, .. } | Self::Cluster { scene, .. } => scene,
        }
    }
}

/// Immutable draw-phase input for one exact view and visibility revision.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GeometryDrawPhase {
    view: SceneHandle,
    selection_revision: u64,
    records: Vec<GeometryDrawRecord>,
}

impl GeometryDrawPhase {
    /// Exact generational view identity.
    #[must_use]
    pub const fn view(&self) -> SceneHandle {
        self.view
    }

    /// Visibility snapshot revision consumed to build this phase.
    #[must_use]
    pub const fn selection_revision(&self) -> u64 {
        self.selection_revision
    }

    /// Stable records ordered by generational scene identity.
    #[must_use]
    pub fn records(&self) -> &[GeometryDrawRecord] {
        &self.records
    }

    /// Whether this view has no visible geometry work.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }
}

/// Why a complete draw phase was rejected. No records are returned on error.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GeometryDrawPhaseError {
    /// The selection snapshot is not the requested visibility revision.
    SnapshotRevisionMismatch { expected: u64, available: u64 },
    /// The view has no entries in this snapshot.
    ViewUnavailable,
    /// Scene identity is stale or the scene row no longer exists.
    SceneUnavailable(SceneHandle),
    /// Scene row and selected geometry identity disagree.
    SceneGeometryMismatch(SceneHandle),
    /// Geometry record is absent or its generation is stale.
    GeometryUnavailable(SceneHandle),
    /// The selected resident LOD or its draw mapping is unavailable.
    LodUnavailable { scene: SceneHandle, level: u32 },
    /// Selection contract rejected the selected representation.
    SelectionRejected {
        scene: SceneHandle,
        reason: GeometryRepresentationRejection,
    },
    /// Selection contract version is not understood by this consumer.
    UnsupportedSelectionVersion { scene: SceneHandle, supplied: u32 },
    /// CLST publication for the exact asset identity is unavailable.
    VirtualPublicationUnavailable { scene: SceneHandle, asset_id: u32 },
    /// Selection and live CLST publication identities differ.
    VirtualPublicationMismatch { scene: SceneHandle },
    /// The selected CLST node is absent, malformed, duplicated, or not resident.
    ClusterUnavailable { scene: SceneHandle, node: u32 },
}

/// Builds a complete deterministic draw phase from one visibility snapshot.
///
/// CLST selections require exact artifact revision and runtime generation. LOD
/// selections require an exact resident level; no implicit mesh fallback is
/// allowed at this boundary.
pub fn build_geometry_draw_phase(
    view: SceneHandle,
    selection_revision: u64,
    selections: &GeometryRepresentationSelections,
    scene: &CpuRenderScene,
    geometries: &RenderGeometryRegistry,
    virtual_publications: &BTreeMap<u32, &VirtualGeometryGpuPublication>,
) -> Result<GeometryDrawPhase, GeometryDrawPhaseError> {
    if selections.revision() != selection_revision {
        return Err(GeometryDrawPhaseError::SnapshotRevisionMismatch {
            expected: selection_revision,
            available: selections.revision(),
        });
    }
    let view_selections = selections
        .for_view(view)
        .ok_or(GeometryDrawPhaseError::ViewUnavailable)?;
    let mut records = Vec::with_capacity(view_selections.len());

    // Snapshot maps are ordered, so command generation is independent of ECS
    // query order and insertion history.
    for (&scene_handle, selection) in view_selections {
        if selection.version != super::representation::GEOMETRY_REPRESENTATION_CONTRACT_VERSION {
            return Err(GeometryDrawPhaseError::UnsupportedSelectionVersion {
                scene: scene_handle,
                supplied: selection.version,
            });
        }
        if selection.identity.view != Some(view)
            || selection.identity.scene != scene_handle
            || selection.identity.selection_revision != selection_revision
        {
            return Err(GeometryDrawPhaseError::SelectionRejected {
                scene: scene_handle,
                reason: GeometryRepresentationRejection::ViewMismatch,
            });
        }
        let instance = scene
            .get(scene_handle)
            .ok_or(GeometryDrawPhaseError::SceneUnavailable(scene_handle))?;
        if instance.geometry != selection.identity.geometry {
            return Err(GeometryDrawPhaseError::SceneGeometryMismatch(scene_handle));
        }
        let record = geometries
            .record(instance.geometry)
            .ok_or(GeometryDrawPhaseError::GeometryUnavailable(scene_handle))?;
        if record.revision != selection.identity.geometry_revision {
            return Err(GeometryDrawPhaseError::SelectionRejected {
                scene: scene_handle,
                reason: GeometryRepresentationRejection::GeometryRevisionMismatch {
                    selected: selection.identity.geometry_revision,
                    available: record.revision,
                },
            });
        }

        match selection.selected {
            GeometryRepresentation::Lod { level } => {
                if selection.virtual_geometry.is_some() {
                    return Err(GeometryDrawPhaseError::SelectionRejected {
                        scene: scene_handle,
                        reason: GeometryRepresentationRejection::UnexpectedVirtualGeometryIdentity,
                    });
                }
                let lod = record
                    .lods
                    .iter()
                    .find(|lod| lod.level == level && lod.resident)
                    .copied()
                    .ok_or(GeometryDrawPhaseError::LodUnavailable {
                        scene: scene_handle,
                        level,
                    })?;
                let (vertex_buffer_class, index_buffer_class) = geometries
                    .buffer_classes_for_lod(instance.geometry, level)
                    .ok_or(GeometryDrawPhaseError::LodUnavailable {
                        scene: scene_handle,
                        level,
                    })?;
                records.push(GeometryDrawRecord::Lod {
                    scene: scene_handle,
                    geometry: instance.geometry,
                    geometry_revision: record.revision,
                    selection_revision,
                    level,
                    primitive_kind: lod.primitive_kind,
                    element_count: lod.element_count,
                    first_element: lod.first_element,
                    base_vertex: lod.base_vertex,
                    vertex_count: lod.vertex_count,
                    vertex_buffer_class,
                    index_buffer_class,
                });
            }
            GeometryRepresentation::Cluster { cluster: node } => {
                let Some(identity) = selection.virtual_geometry else {
                    return Err(GeometryDrawPhaseError::SelectionRejected {
                        scene: scene_handle,
                        reason: GeometryRepresentationRejection::VirtualGeometryIdentityMissing,
                    });
                };
                let publication = virtual_publications
                    .get(&identity.asset_id)
                    .copied()
                    .ok_or(GeometryDrawPhaseError::VirtualPublicationUnavailable {
                        scene: scene_handle,
                        asset_id: identity.asset_id,
                    })?;
                if publication.generation() != identity.generation
                    || publication.content_revision() != identity.content_revision
                {
                    return Err(GeometryDrawPhaseError::VirtualPublicationMismatch {
                        scene: scene_handle,
                    });
                }
                let descriptor = publication
                    .clusters()
                    .iter()
                    .find(|cluster| cluster.node_index == node)
                    .ok_or(GeometryDrawPhaseError::ClusterUnavailable {
                        scene: scene_handle,
                        node,
                    })?;
                let page = publication.page(descriptor.page_index).ok_or(
                    GeometryDrawPhaseError::ClusterUnavailable {
                        scene: scene_handle,
                        node,
                    },
                )?;
                let page_key = publication
                    .page_keys()
                    .get(descriptor.page_index as usize)
                    .copied()
                    .ok_or(GeometryDrawPhaseError::ClusterUnavailable {
                        scene: scene_handle,
                        node,
                    })?;
                if !page.is_resident() {
                    return Err(GeometryDrawPhaseError::ClusterUnavailable {
                        scene: scene_handle,
                        node,
                    });
                }
                let cut = [CutCluster {
                    node,
                    page: page_key,
                }];
                let list = publication.draw_list(scene_handle, &cut).map_err(|_| {
                    GeometryDrawPhaseError::ClusterUnavailable {
                        scene: scene_handle,
                        node,
                    }
                })?;
                let item = list.items().first().copied().ok_or(
                    GeometryDrawPhaseError::ClusterUnavailable {
                        scene: scene_handle,
                        node,
                    },
                )?;
                records.push(GeometryDrawRecord::Cluster {
                    scene: scene_handle,
                    geometry: instance.geometry,
                    geometry_revision: record.revision,
                    selection_revision,
                    publication: identity,
                    page: page_key,
                    item,
                });
            }
            GeometryRepresentation::ConservativeRayTracing => {
                return Err(GeometryDrawPhaseError::SelectionRejected {
                    scene: scene_handle,
                    reason: GeometryRepresentationRejection::RepresentationUnavailable,
                });
            }
        }
    }

    Ok(GeometryDrawPhase {
        view,
        selection_revision,
        records,
    })
}

#[cfg(test)]
mod tests {
    use alloc::{collections::BTreeMap, vec};

    use bevy_asset::{uuid::Uuid, AssetId};
    use bevy_mesh::Mesh;
    use prism_render_architecture::{
        abi::GenerationalHandle,
        geometry::{GeometryLodRecord, GeometryRecord},
        gpu_scene::{
            CpuRenderScene, InstanceRecord, SceneBounds, SceneOperation, SceneTransform,
            SceneTransactionBuilder,
        },
        virtual_geometry::{
            ClusterNode, GeometryPageKey, VirtualGeometryAsset, VirtualGeometryAssetHeader,
            VirtualGeometryClusterRecord, VirtualGeometryCompression, VirtualGeometryGpuPublication,
            VirtualGeometryMaterialBinding, VirtualGeometryPage, VirtualGeometryPageRecord,
            VirtualGeometryPhysicalPage, VIRTUAL_GEOMETRY_ASSET_SCHEMA,
        },
    };

    use super::*;
    use crate::geometry::representation::{
        GeometryRepresentationIdentity, GeometryRepresentationSelection,
    };

    const VIEW: SceneHandle = SceneHandle::new(9, 2);
    const SCENE_A: SceneHandle = SceneHandle::new(3, 4);
    const SCENE_B: SceneHandle = SceneHandle::new(1, 8);
    const GEOMETRY: prism_render_architecture::gpu_scene::GeometryHandle =
        GenerationalHandle::new(7, 5);
    const MATERIAL: prism_render_architecture::gpu_scene::SceneMaterialHandle =
        GenerationalHandle::new(4, 6);

    fn geometry_registry() -> RenderGeometryRegistry {
        let mut registry = RenderGeometryRegistry::default();
        registry.upsert(
            AssetId::Uuid {
                uuid: Uuid::from_u128(101),
            },
            GeometryRecord {
                handle: GEOMETRY,
                revision: 12,
                lods: vec![
                    GeometryLodRecord {
                        level: 0,
                        element_count: 3,
                        vertex_count: 3,
                        resident: true,
                        fallback: true,
                        ..Default::default()
                    },
                    GeometryLodRecord {
                        level: 2,
                        element_count: 6,
                        vertex_count: 6,
                        resident: true,
                        ..Default::default()
                    },
                ],
                ..Default::default()
            },
        );
        registry
    }

    fn scene() -> CpuRenderScene {
        let mut scene = CpuRenderScene::default();
        let mut transaction = SceneTransactionBuilder::new(1, 1);
        transaction.push(SceneOperation::Create {
            handle: SCENE_A,
            record: InstanceRecord {
                current_transform: SceneTransform::IDENTITY,
                previous_transform: SceneTransform::IDENTITY,
                bounds: SceneBounds::default(),
                geometry: GEOMETRY,
                material: MATERIAL,
                ..Default::default()
            },
        });
        transaction.push(SceneOperation::Create {
            handle: SCENE_B,
            record: InstanceRecord {
                current_transform: SceneTransform::IDENTITY,
                previous_transform: SceneTransform::IDENTITY,
                bounds: SceneBounds::default(),
                geometry: GEOMETRY,
                material: MATERIAL,
                ..Default::default()
            },
        });
        assert!(scene.apply(&transaction.finish()).errors.is_empty());
        scene
    }

    fn lod_selection(scene: SceneHandle, level: u32) -> GeometryRepresentationSelection {
        GeometryRepresentationSelection::new(
            GeometryRepresentationIdentity {
                scene,
                geometry: GEOMETRY,
                geometry_revision: 12,
                selection_revision: 31,
                view: Some(VIEW),
            },
            GeometryRepresentation::Lod { level },
            false,
        )
    }

    fn virtual_asset() -> VirtualGeometryAsset {
        let page = GeometryPageKey::new(55, 0);
        let bounds = SceneBounds {
            center: [0.0; 3],
            radius: 1.0,
            half_extents: [1.0; 3],
            _padding: 0.0,
        };
        VirtualGeometryAsset {
            header: VirtualGeometryAssetHeader {
                asset_id: 55,
                schema_version: VIRTUAL_GEOMETRY_ASSET_SCHEMA,
                page_size: 1024,
                build_settings_hash: [1; 32],
                content_hash: [2; 32],
                bounds,
                material_slot_count: 1,
                fallback_mesh_hash: [3; 32],
            },
            nodes: vec![ClusterNode::leaf(bounds, 0.0, page)],
            roots: vec![0],
            pages: vec![VirtualGeometryPage {
                record: VirtualGeometryPageRecord {
                    key: page,
                    compression: VirtualGeometryCompression::None,
                    uncompressed_size: 4,
                    priority: 1,
                    cluster_start: 0,
                    cluster_count: 1,
                },
                payload: vec![1, 2, 3, 4],
            }],
            clusters: vec![VirtualGeometryClusterRecord {
                node_index: 0,
                page,
                material_slot: 0,
                vertex_count: 3,
                triangle_count: 1,
                payload_offset: 0,
                payload_size: 4,
                bounds,
                normal_cone: [0.0, 0.0, 1.0, 1.0],
            }],
        }
    }

    fn selections(
        entries: impl IntoIterator<Item = (SceneHandle, GeometryRepresentationSelection)>,
    ) -> GeometryRepresentationSelections {
        let mut selections = GeometryRepresentationSelections::default();
        selections.begin_frame(31);
        for (scene, selection) in entries {
            selections.insert(VIEW, scene, selection);
        }
        selections
    }

    #[test]
    fn lod_phase_is_stable_and_keeps_exact_selected_level() {
        let selections = selections([
            (SCENE_A, lod_selection(SCENE_A, 2)),
            (SCENE_B, lod_selection(SCENE_B, 0)),
        ]);
        let phase = build_geometry_draw_phase(
            VIEW,
            31,
            &selections,
            &scene(),
            &geometry_registry(),
            &BTreeMap::new(),
        )
        .expect("resident selections produce a phase");
        assert_eq!(phase.records().len(), 2);
        assert_eq!(phase.records()[0].scene(), SCENE_B);
        assert_eq!(phase.records()[1].scene(), SCENE_A);
        assert!(matches!(
            phase.records()[1],
            GeometryDrawRecord::Lod { level: 2, .. }
        ));
    }

    #[test]
    fn nonresident_lod_fails_closed_without_substituting_fallback() {
        let selections = selections([(SCENE_A, lod_selection(SCENE_A, 1))]);
        let error = build_geometry_draw_phase(
            VIEW,
            31,
            &selections,
            &scene(),
            &geometry_registry(),
            &BTreeMap::new(),
        )
        .expect_err("an unavailable exact level must reject the phase");
        assert_eq!(
            error,
            GeometryDrawPhaseError::LodUnavailable {
                scene: SCENE_A,
                level: 1,
            }
        );
    }

    #[test]
    fn cluster_phase_requires_matching_publication_generation_and_residency() {
        let publication = VirtualGeometryGpuPublication::build(
            &virtual_asset(),
            [2; 32],
            19,
            &[VirtualGeometryPhysicalPage {
                key: GeometryPageKey::new(55, 0),
                physical_page: 3,
                byte_offset: 3072,
                byte_len: 4,
                generation: 19,
            }],
            &[VirtualGeometryMaterialBinding {
                material_slot: 0,
                material: MATERIAL,
            }],
        )
        .expect("test publication is valid");
        let identity = VirtualGeometryRepresentationIdentity {
            asset_id: 55,
            generation: 19,
            content_revision: [2; 32],
        };
        let selection = GeometryRepresentationSelection::virtual_cluster(
            GeometryRepresentationIdentity {
                scene: SCENE_A,
                geometry: GEOMETRY,
                geometry_revision: 12,
                selection_revision: 31,
                view: Some(VIEW),
            },
            0,
            identity,
        );
        let mut publications = BTreeMap::new();
        publications.insert(55, &publication);
        let phase = build_geometry_draw_phase(
            VIEW,
            31,
            &selections([(SCENE_A, selection)]),
            &scene(),
            &geometry_registry(),
            &publications,
        )
        .expect("resident cluster produces a phase");
        assert!(matches!(
            phase.records()[0],
            GeometryDrawRecord::Cluster {
                publication: VirtualGeometryRepresentationIdentity {
                    generation: 19,
                    ..
                },
                ..
            }
        ));

        let stale = GeometryRepresentationSelection::virtual_cluster(
            selection.identity,
            0,
            VirtualGeometryRepresentationIdentity {
                generation: 20,
                ..identity
            },
        );
        let error = build_geometry_draw_phase(
            VIEW,
            31,
            &selections([(SCENE_A, stale)]),
            &scene(),
            &geometry_registry(),
            &publications,
        )
        .expect_err("a stale publication must reject the whole phase");
        assert_eq!(
            error,
            GeometryDrawPhaseError::VirtualPublicationMismatch { scene: SCENE_A }
        );
    }

    #[test]
    fn stale_snapshot_revision_fails_before_reading_draw_inputs() {
        let selections = selections([(SCENE_A, lod_selection(SCENE_A, 0))]);
        assert_eq!(
            build_geometry_draw_phase(
                VIEW,
                32,
                &selections,
                &scene(),
                &geometry_registry(),
                &BTreeMap::new(),
            ),
            Err(GeometryDrawPhaseError::SnapshotRevisionMismatch {
                expected: 32,
                available: 31,
            })
        );
    }
}
