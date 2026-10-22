//! Versioned per-view/per-draw diagnostics shared by direct and indirect opaque draws.
//!
//! `slot` is a view-local scene-capacity slot. It is deliberately not a geometry
//! identifier: the row carries generational scene identity, selected LOD and an
//! independently valid cluster identity. Standard Mesh rows deliberately mark
//! virtual cluster/cut/page fields unavailable because those sources do not
//! exist on the standard Mesh path.

use std::collections::HashMap;

use bevy_ecs::{prelude::*, world::FromWorld};
use bevy_material::{
    bind_group_layout_entries::{binding_types::storage_buffer_read_only, BindGroupLayoutEntries},
    descriptor::BindGroupLayoutDescriptor,
};
use bevy_render::{
    render_resource::{
        BindGroup, BindGroupEntries, BindGroupLayout, Buffer, BufferUsages, ShaderStages,
        ShaderType,
    },
    renderer::{RenderDevice, RenderQueue},
};
use bytemuck::{Pod, Zeroable};
use prism_render_architecture::abi::GenerationalHandle;

pub(crate) const OPAQUE_DEBUG_METADATA_ABI_VERSION: u32 = 1;
pub(crate) const OPAQUE_DEBUG_CLUSTER_UNAVAILABLE: u32 = u32::MAX;
pub(crate) const OPAQUE_DEBUG_FLAG_LOD_VALID: u32 = 1 << 0;
pub(crate) const OPAQUE_DEBUG_FLAG_PAGE_RESIDENCY_VALID: u32 = 1 << 1;
pub(crate) const OPAQUE_DEBUG_FLAG_CUT_VALID: u32 = 1 << 3;
pub(crate) const OPAQUE_DEBUG_MODE_STANDARD_MESH: u32 = 1;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Pod, ShaderType, Zeroable)]
pub(crate) struct OpaqueDebugMetadata {
    pub abi_version: u32,
    pub view_index: u32,
    pub view_generation: u32,
    pub scene_index: u32,
    pub scene_generation: u32,
    pub lod_level: u32,
    pub cluster_index: u32,
    pub cluster_generation: u32,
    pub cluster_valid: u32,
    pub flags: u32,
    pub mode: u32,
    pub cut_level: u32,
}

struct ViewDebugBuffer {
    buffer: Buffer,
    bind_group: BindGroup,
    capacity: u32,
}

#[derive(Resource)]
pub(crate) struct OpaqueDebugMetadataBuffers {
    pub(crate) layout: BindGroupLayout,
    pub(crate) layout_descriptor: BindGroupLayoutDescriptor,
    views: HashMap<GenerationalHandle, ViewDebugBuffer>,
    scratch: Vec<OpaqueDebugMetadata>,
}

impl FromWorld for OpaqueDebugMetadataBuffers {
    fn from_world(world: &mut World) -> Self {
        let device = world.resource::<RenderDevice>();
        let entries = BindGroupLayoutEntries::sequential(
            ShaderStages::VERTEX,
            (storage_buffer_read_only::<OpaqueDebugMetadata>(false),),
        );
        Self {
            layout: device.create_bind_group_layout("prism opaque debug metadata", &entries),
            layout_descriptor: BindGroupLayoutDescriptor::new(
                "prism opaque debug metadata",
                &entries,
            ),
            views: HashMap::new(),
            scratch: Vec::new(),
        }
    }
}

impl OpaqueDebugMetadataBuffers {
    pub(crate) fn prepare(
        &mut self,
        device: &RenderDevice,
        queue: &RenderQueue,
        visibility: &crate::visibility::runtime::UnifiedVisibilityState,
        scene: &crate::RenderGpuScene,
        geometry: &crate::RenderGeometryRegistry,
        selections: &crate::geometry::GeometryRepresentationSelections,
    ) {
        let scene_capacity = scene.mirror().capacity().min(u32::MAX as usize) as u32;
        let requested_capacity = visibility
            .frame
            .work_items
            .iter()
            .map(|item| item.scene.index.saturating_add(1))
            .max()
            .unwrap_or(0)
            .min(scene_capacity);
        let capacity = metadata_capacity(
            requested_capacity,
            scene_capacity,
            device.limits().max_storage_buffer_binding_size,
        );
        if capacity == 0 {
            self.views.clear();
            return;
        }
        self.views
            .retain(|view, _| visibility.draw_bins.iter().any(|bins| bins.view == *view));

        for bins in &visibility.draw_bins {
            let Some(output) = visibility.frame.views.get(&bins.view) else {
                continue;
            };
            let start = output.visible_instances.start as usize;
            let end = output
                .visible_instances
                .start
                .saturating_add(output.visible_instances.count) as usize;
            let Some(rows) = visibility.frame.work_items.get(start..end) else {
                continue;
            };
            self.scratch
                .resize(capacity as usize, OpaqueDebugMetadata::unavailable());
            self.scratch.fill(OpaqueDebugMetadata::unavailable());
            for work in rows.iter().filter(|work| {
                work.pass_mask.0 & prism_render_visibility::RenderPassMask::OPAQUE.0 != 0
            }) {
                let Some(slot) = self.scratch.get_mut(work.scene.index as usize) else {
                    continue;
                };
                let selected = selections
                    .for_view(bins.view)
                    .and_then(|view| view.get(&work.scene))
                    .and_then(|selection| {
                        let record = geometry.record(work.geometry)?;
                        selected_standard_lod(
                            *selection,
                            bins.view,
                            work.scene,
                            work.geometry,
                            selections.revision(),
                            record,
                        )
                    });
                if let Some(lod_level) = selected {
                    *slot = OpaqueDebugMetadata::standard_mesh(
                        bins.view.index,
                        bins.view.generation,
                        work.scene.index,
                        work.scene.generation,
                        lod_level,
                    );
                }
            }
            let byte_len = (self.scratch.len() * std::mem::size_of::<OpaqueDebugMetadata>()) as u64;
            let needs_reallocation = self
                .views
                .get(&bins.view)
                .is_none_or(|entry| entry.capacity < capacity);
            if needs_reallocation {
                let buffer =
                    device.create_buffer(&bevy_render::render_resource::BufferDescriptor {
                        label: Some("prism opaque per-view debug metadata"),
                        size: byte_len,
                        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
                let bind_group = device.create_bind_group(
                    "prism opaque per-view debug metadata",
                    &self.layout,
                    &BindGroupEntries::sequential((buffer.as_entire_binding(),)),
                );
                self.views.insert(
                    bins.view,
                    ViewDebugBuffer {
                        buffer,
                        bind_group,
                        capacity,
                    },
                );
            }
            let entry = self.views.get(&bins.view).expect("view buffer prepared");
            queue.write_buffer(&entry.buffer, 0, bytemuck::cast_slice(&self.scratch));
        }
    }

    pub(crate) fn bind_group(&self, view: GenerationalHandle) -> Option<&BindGroup> {
        self.views.get(&view).map(|entry| &entry.bind_group)
    }
}

fn selected_standard_lod(
    selection: crate::geometry::GeometryRepresentationSelection,
    view: GenerationalHandle,
    scene: GenerationalHandle,
    geometry: prism_render_architecture::gpu_scene::GeometryHandle,
    snapshot_revision: u64,
    record: &prism_render_architecture::geometry::GeometryRecord,
) -> Option<u32> {
    let identity = selection.identity;
    if selection.version != crate::geometry::GEOMETRY_REPRESENTATION_CONTRACT_VERSION
        || identity.scene != scene
        || identity.geometry != geometry
        || identity.geometry_revision != record.revision
        || identity.selection_revision != snapshot_revision
        || identity.view != Some(view)
    {
        return None;
    }
    let crate::geometry::GeometryRepresentation::Lod { level } = selection.selected else {
        return None;
    };
    record
        .resolve_lod(level)
        .filter(|lod| lod.level == level && lod.resident)
        .map(|_| level)
}

impl OpaqueDebugMetadata {
    pub(crate) const fn unavailable() -> Self {
        Self {
            abi_version: OPAQUE_DEBUG_METADATA_ABI_VERSION,
            view_index: 0,
            view_generation: 0,
            scene_index: 0,
            scene_generation: 0,
            lod_level: u32::MAX,
            cluster_index: OPAQUE_DEBUG_CLUSTER_UNAVAILABLE,
            cluster_generation: 0,
            cluster_valid: 0,
            flags: 0,
            mode: 0,
            cut_level: u32::MAX,
        }
    }

    pub(crate) fn lod(
        view_index: u32,
        view_generation: u32,
        scene_index: u32,
        scene_generation: u32,
        lod_level: u32,
    ) -> Self {
        Self {
            abi_version: OPAQUE_DEBUG_METADATA_ABI_VERSION,
            view_index,
            view_generation,
            scene_index,
            scene_generation,
            lod_level,
            cluster_index: OPAQUE_DEBUG_CLUSTER_UNAVAILABLE,
            cluster_generation: 0,
            cluster_valid: 0,
            flags: 0,
            mode: OPAQUE_DEBUG_MODE_STANDARD_MESH,
            cut_level: u32::MAX,
        }
    }

    pub(crate) fn standard_mesh(
        view_index: u32,
        view_generation: u32,
        scene_index: u32,
        scene_generation: u32,
        lod_level: u32,
    ) -> Self {
        Self {
            cluster_index: OPAQUE_DEBUG_CLUSTER_UNAVAILABLE,
            cluster_generation: 0,
            cluster_valid: 0,
            flags: OPAQUE_DEBUG_FLAG_LOD_VALID,
            mode: OPAQUE_DEBUG_MODE_STANDARD_MESH,
            cut_level: u32::MAX,
            ..Self::lod(
                view_index,
                view_generation,
                scene_index,
                scene_generation,
                lod_level,
            )
        }
    }

    pub(crate) fn valid_for(
        self,
        view_index: u32,
        view_generation: u32,
        scene_index: u32,
        scene_generation: u32,
    ) -> bool {
        self.abi_version == OPAQUE_DEBUG_METADATA_ABI_VERSION
            && self.view_index == view_index
            && self.view_generation == view_generation
            && self.scene_index == scene_index
            && self.scene_generation == scene_generation
            && self.flags & OPAQUE_DEBUG_FLAG_LOD_VALID != 0
    }
}

/// Computes the per-view metadata address used by direct and indirect draws.
pub(crate) fn metadata_slot(scene_index: u32, scene_capacity: u32) -> Option<u32> {
    (scene_index < scene_capacity).then_some(scene_index)
}

fn metadata_capacity(requested: u32, scene_capacity: u32, max_binding_size: u64) -> u32 {
    let max_rows = (max_binding_size / std::mem::size_of::<OpaqueDebugMetadata>() as u64)
        .min(u64::from(u32::MAX)) as u32;
    requested.min(scene_capacity).min(max_rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_and_indirect_address_same_view_local_metadata_row() {
        let scene_capacity = 4096;
        let direct = metadata_slot(37, scene_capacity);
        let indirect_first_instance = metadata_slot(37, scene_capacity);
        assert_eq!(direct, indirect_first_instance);
        assert_eq!(direct, Some(37));
    }

    #[test]
    fn one_scene_instance_can_have_distinct_lod_metadata_per_view() {
        let a = OpaqueDebugMetadata::lod(0, 3, 12, 8, 0);
        let b = OpaqueDebugMetadata::lod(1, 6, 12, 8, 2);
        assert_eq!(
            metadata_slot(a.scene_index, 64),
            metadata_slot(b.scene_index, 64)
        );
        assert_ne!(a.view_generation, b.view_generation);
        assert_eq!((a.lod_level, b.lod_level), (0, 2));
        assert_eq!((a.view_generation, b.view_generation), (3, 6));
    }

    #[test]
    fn lod_metadata_never_claims_geometry_as_cluster_identity() {
        let metadata = OpaqueDebugMetadata::lod(0, 1, 4, 9, 2);
        assert_eq!(metadata.cluster_index, OPAQUE_DEBUG_CLUSTER_UNAVAILABLE);
        assert_eq!(metadata.cluster_valid, 0);
        assert_eq!(metadata.mode, OPAQUE_DEBUG_MODE_STANDARD_MESH);
    }

    #[test]
    fn unavailable_cluster_and_virtual_diagnostics_are_rendered_as_unavailable() {
        let source = include_str!("../shaders/opaque.wesl");
        assert!(source.contains("debug_metadata.cluster_valid == 1u"));
        assert!(source.contains("debug_metadata.cut_level != 0xffffffffu"));
        assert!(source.contains("debug_lod_valid && (debug_metadata.flags & 2u) != 0u"));
        assert_eq!(OPAQUE_DEBUG_CLUSTER_UNAVAILABLE, u32::MAX);
        assert_eq!(OPAQUE_DEBUG_FLAG_PAGE_RESIDENCY_VALID & (1 << 2), 0);
    }

    #[test]
    fn standard_mesh_metadata_does_not_claim_virtual_cluster_cut_or_page_data() {
        let metadata = OpaqueDebugMetadata::standard_mesh(2, 7, 4, 11, 3);
        assert_eq!(
            (metadata.cluster_index, metadata.cluster_generation),
            (OPAQUE_DEBUG_CLUSTER_UNAVAILABLE, 0)
        );
        assert_eq!(metadata.cluster_valid, 0);
        assert_eq!(metadata.mode, OPAQUE_DEBUG_MODE_STANDARD_MESH);
        assert_eq!(metadata.cut_level, u32::MAX);
        assert_eq!(metadata.flags & OPAQUE_DEBUG_FLAG_CUT_VALID, 0);
        assert_eq!(metadata.flags & OPAQUE_DEBUG_FLAG_PAGE_RESIDENCY_VALID, 0);
        assert_ne!(metadata.flags & OPAQUE_DEBUG_FLAG_LOD_VALID, 0);
        assert!(metadata.valid_for(2, 7, 4, 11));
        assert!(!metadata.valid_for(2, 8, 4, 11));
        assert!(!metadata.valid_for(2, 7, 4, 12));
    }

    #[test]
    fn shader_debug_modes_have_stable_unique_ids_and_share_metadata_gate() {
        use crate::opaque::pipeline::{uses_debug_metadata, GpuSceneDebugView};
        assert_eq!(GpuSceneDebugView::ClusterIdentity as u32, 8);
        assert_eq!(GpuSceneDebugView::Cut as u32, 9);
        assert_eq!(GpuSceneDebugView::PageResidency as u32, 10);
        assert!(uses_debug_metadata(GpuSceneDebugView::ClusterIdentity));
        assert!(uses_debug_metadata(GpuSceneDebugView::Cut));
        assert!(uses_debug_metadata(GpuSceneDebugView::PageResidency));
        assert!(!uses_debug_metadata(GpuSceneDebugView::Shaded));
    }

    #[test]
    fn unavailable_metadata_is_explicit_and_generation_safe() {
        let metadata = OpaqueDebugMetadata::unavailable();
        assert_eq!(metadata.abi_version, OPAQUE_DEBUG_METADATA_ABI_VERSION);
        assert_eq!(metadata.view_generation, 0);
        assert_eq!(metadata.scene_generation, 0);
        assert_eq!(metadata.lod_level, u32::MAX);
        assert_eq!(metadata.cluster_index, OPAQUE_DEBUG_CLUSTER_UNAVAILABLE);
        assert_eq!(metadata.cluster_valid, 0);
    }

    #[test]
    fn metadata_capacity_respects_scene_and_device_binding_limits() {
        let stride = std::mem::size_of::<OpaqueDebugMetadata>() as u32;
        assert_eq!(metadata_capacity(100, 80, u64::from(stride * 3)), 3);
        assert_eq!(metadata_capacity(100, 2, u64::from(stride * 3)), 2);
        assert_eq!(metadata_capacity(100, 80, u64::from(stride - 1)), 0);
    }

    #[test]
    fn selected_lod_requires_matching_view_scene_geometry_and_revisions() {
        use prism_render_architecture::{
            abi::GenerationalHandle,
            geometry::{GeometryLodRecord, GeometryRecord},
            gpu_scene::GeometryHandle,
        };

        let view = GenerationalHandle {
            index: 2,
            generation: 7,
        };
        let scene = GenerationalHandle {
            index: 4,
            generation: 11,
        };
        let geometry = GeometryHandle {
            index: 9,
            generation: 3,
        };
        let record = GeometryRecord {
            handle: geometry,
            revision: 5,
            lods: vec![GeometryLodRecord {
                level: 2,
                resident: true,
                ..Default::default()
            }],
            ..Default::default()
        };
        let selection = crate::geometry::visibility_lod_selection(geometry, 5, scene, view, 13, 2);
        assert_eq!(
            selected_standard_lod(selection, view, scene, geometry, 13, &record),
            Some(2)
        );
        assert_eq!(
            selected_standard_lod(selection, view, scene, geometry, 14, &record),
            None
        );
        assert_eq!(
            selected_standard_lod(
                selection,
                GenerationalHandle {
                    generation: 8,
                    ..view
                },
                scene,
                geometry,
                13,
                &record,
            ),
            None
        );
    }
}
