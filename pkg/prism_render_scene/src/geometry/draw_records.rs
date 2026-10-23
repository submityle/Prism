//! Deterministic per-view production draw records for geometry selections.
//!
//! This is the scene-side boundary between visibility decisions and backend
//! draw preparation. Every selection is resolved against current snapshots;
//! one stale item rejects the complete phase rather than leaking partial work.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use bevy_ecs::prelude::{Res, ResMut, Resource};

use prism_render_architecture::{
    abi::GenerationalHandle,
    gpu_scene::{CpuRenderScene, SceneHandle},
    virtual_geometry::{
        ClstAssetIdentity, ClstClusterDescriptor, CutCluster, GeometryDrawBatch,
        GeometryDrawBatchError, GeometryDrawBatchHeader, GeometryPageKey,
        VirtualGeometryGpuDrawItem, VirtualGeometryGpuPublication, GEOMETRY_DRAW_TRANSPORT_VERSION,
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

/// CPU mirror of generation-bound virtual-geometry publications available to
/// the scene draw planner. The render backend owns the eventual GPU buffers;
/// this resource keeps the scene-side validation input explicit and immutable
/// for one planning pass.
#[derive(Resource, Clone, Debug, Default)]
pub struct VirtualGeometryDrawPublications {
    publications: BTreeMap<u32, VirtualGeometryGpuPublication>,
    residency: BTreeMap<u32, VirtualGeometryResidencyState>,
}

/// Incrementally mirrored page residency for one exact publication lifetime.
#[derive(Clone, Debug, Default)]
struct VirtualGeometryResidencyState {
    content_revision: [u8; 32],
    generation: u64,
    active: bool,
    pages: BTreeMap<GeometryPageKey, bool>,
}

impl VirtualGeometryDrawPublications {
    /// Applies a generation-bound page residency delta exported by RenderApp.
    /// A new generation replaces old rows; stale generations are ignored.
    pub fn apply_residency_delta(
        &mut self,
        asset_id: u32,
        content_revision: [u8; 32],
        generation: u64,
        active: bool,
        changed_pages: &[(GeometryPageKey, bool)],
    ) {
        if !active {
            let state = self.residency.entry(asset_id).or_default();
            if state.generation > generation {
                return;
            }
            if state.generation < generation {
                *state = VirtualGeometryResidencyState {
                    content_revision,
                    generation,
                    active: false,
                    pages: BTreeMap::new(),
                };
            } else if state.generation == generation && state.content_revision == content_revision {
                state.active = false;
                state.pages.clear();
            } else {
                return;
            }
            return;
        }
        let state = self.residency.entry(asset_id).or_default();
        if state.generation > generation
            || (state.generation == generation
                && (state.content_revision != content_revision || !state.active))
        {
            return;
        }
        if state.generation < generation {
            *state = VirtualGeometryResidencyState {
                content_revision,
                generation,
                active: true,
                pages: BTreeMap::new(),
            };
        }
        state.active = true;
        for &(key, resident) in changed_pages {
            if key.asset == asset_id {
                state.pages.insert(key, resident);
            }
        }
    }

    /// Returns whether the current exact asset generation has the requested
    /// logical page resident. Unknown/stale page state is always nonresident.
    #[must_use]
    pub fn is_page_resident(
        &self,
        asset_id: u32,
        content_revision: [u8; 32],
        generation: u64,
        key: GeometryPageKey,
    ) -> bool {
        self.residency.get(&asset_id).is_some_and(|state| {
            state.active
                && state.content_revision == content_revision
                && state.generation == generation
                && state.pages.get(&key).copied().unwrap_or(false)
        })
    }

    /// Replaces one asset publication atomically.
    pub fn replace(&mut self, publication: VirtualGeometryGpuPublication) {
        self.publications
            .insert(publication.asset_id(), publication);
    }

    /// Removes a retired asset publication.
    pub fn retire(&mut self, asset_id: u32) -> Option<VirtualGeometryGpuPublication> {
        if let Some(state) = self.residency.get_mut(&asset_id) {
            // Keep the generation/revision tombstone so delayed deltas cannot
            // resurrect this lifetime, but release the potentially large page
            // map as soon as its publication is retired.
            state.active = false;
            state.pages.clear();
        }
        self.publications.remove(&asset_id)
    }

    /// Returns the current publication for one asset.
    #[must_use]
    pub fn get(&self, asset_id: u32) -> Option<&VirtualGeometryGpuPublication> {
        self.publications.get(&asset_id)
    }
}

/// Complete per-view geometry planning output produced after visibility.
#[derive(Resource, Clone, Debug, Default)]
pub struct GeometryDrawPhases {
    revision: u64,
    phases: BTreeMap<SceneHandle, GeometryDrawPhase>,
    rejections: Vec<(SceneHandle, GeometryDrawPhaseError)>,
}

impl GeometryDrawPhases {
    /// Visibility revision consumed by this snapshot.
    #[must_use]
    pub const fn revision(&self) -> u64 {
        self.revision
    }

    /// Returns the planned phase for one exact view.
    #[must_use]
    pub fn get(&self, view: SceneHandle) -> Option<&GeometryDrawPhase> {
        self.phases.get(&view)
    }

    /// Returns all rejected views from the last planning pass.
    #[must_use]
    pub fn rejections(&self) -> &[(SceneHandle, GeometryDrawPhaseError)] {
        &self.rejections
    }

    /// Builds a versioned, device-independent CLST batch for one exact view.
    /// The caller supplies the owning frame revision; no ECS entity or GPU
    /// resource identity is synthesized at this boundary.
    pub fn build_clst_batch(
        &self,
        view: SceneHandle,
        frame_revision: u64,
        publications: &VirtualGeometryDrawPublications,
    ) -> Result<GeometryDrawBatch, GeometryDrawBatchBuildError> {
        if frame_revision == 0 {
            return Err(GeometryDrawBatchBuildError::InvalidFrameRevision);
        }
        let phase = self
            .get(view)
            .ok_or(GeometryDrawBatchBuildError::ViewUnavailable(view))?;
        if phase.selection_revision == 0 || phase.selection_revision != self.revision {
            return Err(GeometryDrawBatchBuildError::StalePhase {
                phase_revision: phase.selection_revision,
                available_revision: self.revision,
            });
        }
        let mut records = Vec::new();
        for record in &phase.records {
            let GeometryDrawRecord::Cluster {
                scene,
                geometry,
                geometry_revision,
                selection_revision,
                publication,
                page,
                item,
            } = *record
            else {
                continue;
            };
            let live = publications.get(publication.asset_id).ok_or(
                GeometryDrawBatchBuildError::PublicationUnavailable(publication.asset_id),
            )?;
            if live.generation() != publication.generation
                || live.content_revision() != publication.content_revision
                || item.asset_id != publication.asset_id
                || item.scene_index != scene.index
                || item.scene_generation != scene.generation
                || (u64::from(item.publication_generation[0])
                    | (u64::from(item.publication_generation[1]) << 32))
                    != publication.generation
            {
                return Err(GeometryDrawBatchBuildError::PublicationMismatch(scene));
            }
            let cluster = *live.clusters().get(item.cluster_index as usize).ok_or(
                GeometryDrawBatchBuildError::ClusterUnavailable {
                    scene,
                    cluster_index: item.cluster_index,
                },
            )?;
            let actual_page = *live.page_keys().get(cluster.page_index as usize).ok_or(
                GeometryDrawBatchBuildError::ClusterUnavailable {
                    scene,
                    cluster_index: item.cluster_index,
                },
            )?;
            if actual_page != page
                || actual_page.asset != publication.asset_id
                || !live
                    .page(cluster.page_index)
                    .is_some_and(|row| row.is_resident())
                || cluster.material_index != item.material_index
                || cluster.material_generation != item.material_generation
            {
                return Err(GeometryDrawBatchBuildError::ClusterUnavailable {
                    scene,
                    cluster_index: item.cluster_index,
                });
            }
            records.push(
                prism_render_architecture::virtual_geometry::GeometryDrawRecord {
                    view,
                    scene,
                    geometry,
                    geometry_revision: u64::from(geometry_revision),
                    selection_revision,
                    asset: ClstAssetIdentity {
                        asset_id: publication.asset_id,
                        content_revision: publication.content_revision,
                        publication_generation: publication.generation,
                    },
                    page,
                    cluster: ClstClusterDescriptor {
                        cluster_index: item.cluster_index,
                        node_index: cluster.node_index,
                        payload_offset: cluster.payload_offset,
                        payload_size: cluster.payload_size,
                        vertex_count: cluster.vertex_count,
                        triangle_count: cluster.triangle_count,
                    },
                    material: GenerationalHandle::new(
                        item.material_index,
                        item.material_generation,
                    ),
                },
            );
        }
        GeometryDrawBatch::new(
            GeometryDrawBatchHeader {
                version: GEOMETRY_DRAW_TRANSPORT_VERSION,
                frame_revision,
                snapshot_revision: phase.selection_revision,
            },
            records,
        )
        .map_err(GeometryDrawBatchBuildError::InvalidBatch)
    }
}

/// Failure while explicitly translating scene draw phases into CLST transport.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GeometryDrawBatchBuildError {
    InvalidFrameRevision,
    ViewUnavailable(SceneHandle),
    StalePhase {
        phase_revision: u64,
        available_revision: u64,
    },
    PublicationUnavailable(u32),
    PublicationMismatch(SceneHandle),
    ClusterUnavailable {
        scene: SceneHandle,
        cluster_index: u32,
    },
    InvalidBatch(GeometryDrawBatchError),
}

/// Builds and publishes the scene-side geometry phase for every visibility
/// view. The complete phase is discarded on one stale item, so queue consumers
/// can never draw a mixture of revisions.
pub(crate) fn build_geometry_draw_phases(
    selections: Res<GeometryRepresentationSelections>,
    scene: Res<super::super::scene::RenderGpuScene>,
    geometries: Res<RenderGeometryRegistry>,
    publications: Res<VirtualGeometryDrawPublications>,
    mut output: ResMut<GeometryDrawPhases>,
) {
    output.revision = selections.revision();
    output.phases.clear();
    output.rejections.clear();
    for (view, _) in selections.iter() {
        match build_geometry_draw_phase(
            view,
            selections.revision(),
            &selections,
            scene.mirror(),
            &geometries,
            &publications,
        ) {
            Ok(phase) => {
                output.phases.insert(view, phase);
            }
            Err(error) => output.rejections.push((view, error)),
        }
    }
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
    virtual_publications: &VirtualGeometryDrawPublications,
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
                let publication = virtual_publications.get(identity.asset_id).ok_or(
                    GeometryDrawPhaseError::VirtualPublicationUnavailable {
                        scene: scene_handle,
                        asset_id: identity.asset_id,
                    },
                )?;
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
                let page_key = publication
                    .page_keys()
                    .get(descriptor.page_index as usize)
                    .copied()
                    .ok_or(GeometryDrawPhaseError::ClusterUnavailable {
                        scene: scene_handle,
                        node,
                    })?;
                if !publication
                    .page(descriptor.page_index)
                    .is_some_and(|page| page.is_resident())
                {
                    return Err(GeometryDrawPhaseError::ClusterUnavailable {
                        scene: scene_handle,
                        node,
                    });
                }
                // Direct consumers that construct a selection without the
                // visibility producer still get an exact single-node cut.
                let fallback_cut = [CutCluster {
                    node,
                    page: page_key,
                }];
                let cut = selections
                    .cluster_cut(view, scene_handle)
                    .unwrap_or(&fallback_cut);
                if cut.is_empty() || !cut.iter().any(|cluster| cluster.node == node) {
                    return Err(GeometryDrawPhaseError::ClusterUnavailable {
                        scene: scene_handle,
                        node,
                    });
                }
                let list = publication.draw_list(scene_handle, cut).map_err(|_| {
                    GeometryDrawPhaseError::ClusterUnavailable {
                        scene: scene_handle,
                        node,
                    }
                })?;
                if list.items().len() != cut.len() {
                    return Err(GeometryDrawPhaseError::ClusterUnavailable {
                        scene: scene_handle,
                        node,
                    });
                }
                for (cluster, item) in cut.iter().zip(list.items().iter().copied()) {
                    records.push(GeometryDrawRecord::Cluster {
                        scene: scene_handle,
                        geometry: instance.geometry,
                        geometry_revision: record.revision,
                        selection_revision,
                        publication: identity,
                        page: cluster.page,
                        item,
                    });
                }
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
    use alloc::vec;

    use bevy_asset::{uuid::Uuid, AssetId};
    use bevy_mesh::Mesh;
    use prism_render_architecture::{
        abi::GenerationalHandle,
        geometry::{GeometryLodRecord, GeometryRecord},
        gpu_scene::{
            CpuRenderScene, InstanceRecord, SceneBounds, SceneOperation, SceneTransactionBuilder,
            SceneTransform,
        },
        virtual_geometry::{
            ClusterNode, GeometryPageKey, VirtualGeometryAsset, VirtualGeometryAssetHeader,
            VirtualGeometryClusterRecord, VirtualGeometryCompression,
            VirtualGeometryGpuPublication, VirtualGeometryMaterialBinding, VirtualGeometryPage,
            VirtualGeometryPageRecord, VirtualGeometryPhysicalPage, VIRTUAL_GEOMETRY_ASSET_SCHEMA,
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
            &VirtualGeometryDrawPublications::default(),
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
            &VirtualGeometryDrawPublications::default(),
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
        let mut publications = VirtualGeometryDrawPublications::default();
        publications.replace(publication.clone());
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
                publication: VirtualGeometryRepresentationIdentity { generation: 19, .. },
                ..
            }
        ));

        let mut phases = GeometryDrawPhases::default();
        phases.revision = 31;
        phases.phases.insert(VIEW, phase.clone());
        let batch = phases
            .build_clst_batch(VIEW, 44, &publications)
            .expect("exact scene data translates to a shared CLST batch");
        assert_eq!(batch.header().frame_revision, 44);
        assert_eq!(batch.header().snapshot_revision, 31);
        assert_eq!(batch.len(), 1);
        assert_eq!(batch.records()[0].material, MATERIAL);
        assert_eq!(batch.records()[0].cluster.node_index, 0);

        let replacement = VirtualGeometryGpuPublication::build(
            &virtual_asset(),
            [2; 32],
            20,
            &[],
            &[VirtualGeometryMaterialBinding {
                material_slot: 0,
                material: MATERIAL,
            }],
        )
        .expect("retired generation replacement is valid");
        publications.replace(replacement);
        assert_eq!(
            phases.build_clst_batch(VIEW, 44, &publications),
            Err(GeometryDrawBatchBuildError::PublicationMismatch(SCENE_A))
        );

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
    fn stale_retirement_cannot_remove_newer_residency_generation() {
        let key = GeometryPageKey::new(55, 0);
        let mut publications = VirtualGeometryDrawPublications::default();
        publications.apply_residency_delta(55, [2; 32], 20, true, &[(key, true)]);
        publications.apply_residency_delta(55, [1; 32], 19, false, &[]);
        assert!(publications.is_page_resident(55, [2; 32], 20, key));
        publications.apply_residency_delta(55, [2; 32], 20, false, &[]);
        assert!(!publications.is_page_resident(55, [2; 32], 20, key));
        publications.apply_residency_delta(55, [2; 32], 20, true, &[(key, true)]);
        assert!(!publications.is_page_resident(55, [2; 32], 20, key));
        publications.apply_residency_delta(55, [3; 32], 21, true, &[(key, true)]);
        assert!(publications.is_page_resident(55, [3; 32], 21, key));
    }

    #[test]
    fn retiring_publication_releases_residency_pages_and_keeps_tombstone() {
        let key = GeometryPageKey::new(55, 0);
        let mut publications = VirtualGeometryDrawPublications::default();
        publications.apply_residency_delta(55, [2; 32], 20, true, &[(key, true)]);
        let publication = VirtualGeometryGpuPublication::build(
            &virtual_asset(),
            [2; 32],
            20,
            &[],
            &[VirtualGeometryMaterialBinding {
                material_slot: 0,
                material: MATERIAL,
            }],
        )
        .expect("publication fixture is valid");
        publications.replace(publication);

        assert!(publications.retire(55).is_some());
        assert!(!publications.is_page_resident(55, [2; 32], 20, key));

        // The same retired lifetime cannot be reactivated by a delayed delta.
        publications.apply_residency_delta(55, [2; 32], 20, true, &[(key, true)]);
        assert!(!publications.is_page_resident(55, [2; 32], 20, key));
        // A genuinely newer publication lifetime can reuse the asset identity.
        publications.apply_residency_delta(55, [3; 32], 21, true, &[(key, true)]);
        assert!(publications.is_page_resident(55, [3; 32], 21, key));
    }

    #[test]
    fn cluster_phase_emits_every_cluster_in_the_selected_cut() {
        let mut asset = virtual_asset();
        let second_node = ClusterNode::leaf(asset.nodes[0].bounds, 0.0, asset.pages[0].record.key);
        asset.nodes.push(second_node);
        asset.roots.push(1);
        asset.pages[0].record.cluster_count = 2;
        let mut second_cluster = asset.clusters[0];
        second_cluster.node_index = 1;
        asset.clusters.push(second_cluster);
        let page = asset.pages[0].record.key;
        let publication = VirtualGeometryGpuPublication::build(
            &asset,
            [2; 32],
            19,
            &[VirtualGeometryPhysicalPage {
                key: page,
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
        .expect("two-node publication is valid");
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
        let mut selections = selections([(SCENE_A, selection)]);
        selections.insert_cluster_selection(
            VIEW,
            SCENE_A,
            selection,
            vec![CutCluster { node: 0, page }, CutCluster { node: 1, page }],
            vec![(page, 1.0)],
        );
        let mut publications = VirtualGeometryDrawPublications::default();
        publications.replace(publication);
        let phase = build_geometry_draw_phase(
            VIEW,
            31,
            &selections,
            &scene(),
            &geometry_registry(),
            &publications,
        )
        .expect("complete resident cut produces all records");
        assert_eq!(phase.records().len(), 2);
        assert!(phase.records().iter().all(|record| matches!(
            record,
            GeometryDrawRecord::Cluster { page: selected_page, .. } if *selected_page == page
        )));
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
                &VirtualGeometryDrawPublications::default(),
            ),
            Err(GeometryDrawPhaseError::SnapshotRevisionMismatch {
                expected: 32,
                available: 31,
            })
        );
    }
}
