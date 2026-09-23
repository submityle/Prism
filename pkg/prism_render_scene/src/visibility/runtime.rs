use bevy_ecs::prelude::*;
use bevy_platform::collections::HashMap;
use bevy_render::view::RetainedViewEntity;
use prism_render_architecture::abi::GenerationalHandle;
use prism_render_visibility::{GpuViewRecord, VisibilityFrame};

#[derive(Resource, Clone, Copy, Debug, Eq, PartialEq)]
pub struct UnifiedVisibilityEnabled(pub bool);

impl Default for UnifiedVisibilityEnabled {
    fn default() -> Self {
        Self(true)
    }
}

#[derive(Resource, Clone, Copy, Debug)]
pub struct UnifiedVisibilitySettings {
    pub max_work_items: u32,
    pub camera_cut_distance: f32,
}

impl Default for UnifiedVisibilitySettings {
    fn default() -> Self {
        Self {
            max_work_items: 1 << 20,
            camera_cut_distance: 100.0,
        }
    }
}

#[derive(Resource, Default)]
pub(crate) struct UnifiedVisibilityState {
    pub views: Vec<GpuViewRecord>,
    pub frame: VisibilityFrame,
    handles: HashMap<RetainedViewEntity, GenerationalHandle>,
    previous_clip: HashMap<RetainedViewEntity, [[f32; 4]; 4]>,
    previous_positions: HashMap<RetainedViewEntity, [f32; 3]>,
    history_epochs: HashMap<RetainedViewEntity, u64>,
    previous_lods: alloc::collections::BTreeMap<
        (
            GenerationalHandle,
            prism_render_architecture::gpu_scene::SceneHandle,
        ),
        u16,
    >,
    occluded: alloc::collections::BTreeSet<(
        GenerationalHandle,
        prism_render_architecture::gpu_scene::SceneHandle,
    )>,
    next_view_index: u32,
}

#[derive(Resource)]
pub(crate) struct VisibilityFrameGraph {
    pub compiled: prism_render_architecture::frame_graph::CompiledGpuFrameGraph,
}

impl UnifiedVisibilityState {
    pub fn handle(&mut self, retained: RetainedViewEntity) -> GenerationalHandle {
        *self.handles.entry(retained).or_insert_with(|| {
            self.next_view_index = self.next_view_index.saturating_add(1).max(1);
            GenerationalHandle {
                index: self.next_view_index,
                generation: 1,
            }
        })
    }

    pub fn previous(
        &self,
        retained: RetainedViewEntity,
    ) -> (Option<[[f32; 4]; 4]>, Option<[f32; 3]>) {
        (
            self.previous_clip.get(&retained).copied(),
            self.previous_positions.get(&retained).copied(),
        )
    }

    pub fn previous_history_epoch(&self, retained: RetainedViewEntity) -> Option<u64> {
        self.history_epochs.get(&retained).copied()
    }

    pub fn remember(
        &mut self,
        retained: RetainedViewEntity,
        clip: [[f32; 4]; 4],
        position: [f32; 3],
        history_epoch: u64,
    ) {
        self.previous_clip.insert(retained, clip);
        self.previous_positions.insert(retained, position);
        self.history_epochs.insert(retained, history_epoch);
    }

    pub fn previous_lods(
        &self,
    ) -> &alloc::collections::BTreeMap<
        (
            GenerationalHandle,
            prism_render_architecture::gpu_scene::SceneHandle,
        ),
        u16,
    > {
        &self.previous_lods
    }

    pub fn occluded(
        &self,
    ) -> &alloc::collections::BTreeSet<(
        GenerationalHandle,
        prism_render_architecture::gpu_scene::SceneHandle,
    )> {
        &self.occluded
    }

    pub fn commit_lods(&mut self) {
        self.previous_lods.clear();
        for (&view, output) in &self.frame.views {
            let start = output.visible_instances.start as usize;
            let end = start + output.visible_instances.count as usize;
            for item in &self.frame.work_items[start..end] {
                self.previous_lods
                    .insert((view, item.scene), item.lod_or_cluster as u16);
            }
        }
    }
}

#[derive(Resource, Clone, Copy, Debug, Default)]
pub struct PrismVisibilityDiagnostics {
    pub views: u32,
    pub input_instances: u32,
    pub visible_instances: u32,
    pub work_items: u32,
    pub material_bins: u32,
    pub frustum_rejected: u32,
    pub occlusion_rejected: u32,
    pub stale_handles: u32,
    pub missing_geometry: u32,
    pub missing_material: u32,
    pub overflows: u32,
    pub camera_cuts: u32,
    pub buffer_version: u32,
    pub gpu_compute_dispatches: u32,
    pub cpu_reference_frames: u32,
}
