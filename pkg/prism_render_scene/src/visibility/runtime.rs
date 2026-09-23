use bevy_ecs::prelude::*;
use bevy_platform::collections::HashMap;
use bevy_render::view::RetainedViewEntity;
use prism_render_architecture::abi::GenerationalHandle;
use prism_render_visibility::{GpuViewRecord, ViewDrawBins, VisibilityFrame};

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
    pub gpu_parity_max_items_per_view: u32,
    pub gpu_parity_readback: bool,
    pub indirect_first_instance: bool,
}

impl Default for UnifiedVisibilitySettings {
    fn default() -> Self {
        Self {
            max_work_items: 1 << 20,
            camera_cut_distance: 100.0,
            gpu_parity_max_items_per_view: 1 << 16,
            gpu_parity_readback: true,
            indirect_first_instance: false,
        }
    }
}

#[derive(Resource, Default)]
pub(crate) struct UnifiedVisibilityState {
    pub views: Vec<GpuViewRecord>,
    pub frame: VisibilityFrame,
    pub draw_bins: Vec<ViewDrawBins>,
    handles: HashMap<RetainedViewEntity, GenerationalHandle>,
    retained_by_handle: HashMap<GenerationalHandle, RetainedViewEntity>,
    previous_clip: HashMap<RetainedViewEntity, [[f32; 4]; 4]>,
    previous_positions: HashMap<RetainedViewEntity, [f32; 3]>,
    history_epochs: HashMap<RetainedViewEntity, u64>,
    last_seen: HashMap<RetainedViewEntity, u64>,
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
    frame_index: u64,
}

#[derive(Resource)]
pub(crate) struct VisibilityFrameGraph {
    pub compiled: prism_render_architecture::frame_graph::CompiledGpuFrameGraph,
}

impl UnifiedVisibilityState {
    pub fn begin_frame(&mut self) {
        self.frame_index = self.frame_index.saturating_add(1);
    }

    pub fn handle(&mut self, retained: RetainedViewEntity) -> GenerationalHandle {
        self.last_seen.insert(retained, self.frame_index);
        let handle = *self.handles.entry(retained).or_insert_with(|| {
            self.next_view_index = self.next_view_index.saturating_add(1).max(1);
            GenerationalHandle {
                index: self.next_view_index,
                generation: 1,
            }
        });
        self.retained_by_handle.insert(handle, retained);
        handle
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

    pub fn retire_missing_views(&mut self) {
        let current = self.frame_index;
        let stale: Vec<_> = self
            .last_seen
            .iter()
            .filter_map(|(view, seen)| (*seen != current).then_some(*view))
            .collect();
        for view in stale {
            if let Some(handle) = self.handles.remove(&view) {
                self.retained_by_handle.remove(&handle);
                self.previous_lods
                    .retain(|(candidate, _), _| *candidate != handle);
                self.occluded.retain(|(candidate, _)| *candidate != handle);
            }
            self.previous_clip.remove(&view);
            self.previous_positions.remove(&view);
            self.history_epochs.remove(&view);
            self.last_seen.remove(&view);
        }
    }

    pub fn retained_view(&self, handle: GenerationalHandle) -> Option<RetainedViewEntity> {
        self.retained_by_handle.get(&handle).copied()
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
    pub gpu_candidates: u32,
    pub pipeline_not_ready: u32,
    pub parity_frames: u64,
    pub parity_matching_views: u64,
    pub parity_mismatched_views: u64,
    pub parity_dropped_frames: u64,
    pub parity_readback_failures: u64,
    pub parity_overflowed_views: u64,
    pub indirect_identity_fallbacks: u32,
    pub gpu_indexed_commands: u64,
    pub gpu_non_indexed_commands: u64,
    pub parity_mismatched_command_counts: u64,
    pub parity_mismatched_bin_counts: u64,
    pub draw_bins: u32,
    pub draw_bin_capacity: u32,
}
