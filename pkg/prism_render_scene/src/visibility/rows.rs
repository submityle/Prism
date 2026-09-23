use bevy_math::{Mat4, Vec4};
use bevy_render::{
    impl_atomic_pod,
    render_resource::{AtomicPod, ShaderType},
};
use bytemuck::{Pod, Zeroable};
use prism_render_architecture::abi::GenerationalHandle;
use prism_render_visibility::{GpuRenderWorkItem, GpuViewRecord};

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, ShaderType, Zeroable)]
pub struct RenderVisibilityView {
    pub clip_from_world: Mat4,
    pub previous_clip_from_world: Mat4,
    pub frustum_planes: [Vec4; 6],
    pub world_position_lod_scale: Vec4,
    pub viewport: [u32; 4],
    pub handle_index: u32,
    pub handle_generation: u32,
    pub layer_mask: u32,
    pub flags: u32,
    pub history_epoch_low: u32,
    pub history_epoch_high: u32,
    pub _padding: [u32; 2],
}
impl_atomic_pod!(RenderVisibilityView, RenderVisibilityViewBlob);

impl From<&GpuViewRecord> for RenderVisibilityView {
    fn from(view: &GpuViewRecord) -> Self {
        Self {
            clip_from_world: Mat4::from_cols_array_2d(&view.clip_from_world),
            previous_clip_from_world: Mat4::from_cols_array_2d(&view.previous_clip_from_world),
            frustum_planes: view.frustum_planes.map(Vec4::from_array),
            world_position_lod_scale: Vec4::new(
                view.world_position[0],
                view.world_position[1],
                view.world_position[2],
                view.lod_scale,
            ),
            viewport: view.viewport,
            handle_index: view.handle.index,
            handle_generation: view.handle.generation,
            layer_mask: view.layer_mask,
            flags: view.flags.0,
            history_epoch_low: view.history_epoch as u32,
            history_epoch_high: (view.history_epoch >> 32) as u32,
            _padding: [0; 2],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Pod, ShaderType, Zeroable)]
pub struct RenderVisibilityWorkItem {
    pub scene_index: u32,
    pub scene_generation: u32,
    pub geometry_index: u32,
    pub geometry_generation: u32,
    pub material_index: u32,
    pub material_generation: u32,
    pub lod_or_cluster: u32,
    pub pass_mask: u32,
    pub visibility_stages: u32,
    pub sort_key_low: u32,
    pub sort_key_high: u32,
    pub _padding: u32,
}
impl_atomic_pod!(RenderVisibilityWorkItem, RenderVisibilityWorkItemBlob);

impl From<GpuRenderWorkItem> for RenderVisibilityWorkItem {
    fn from(item: GpuRenderWorkItem) -> Self {
        Self {
            scene_index: item.scene.index,
            scene_generation: item.scene.generation,
            geometry_index: item.geometry.index,
            geometry_generation: item.geometry.generation,
            material_index: item.material.index,
            material_generation: item.material.generation,
            lod_or_cluster: item.lod_or_cluster,
            pass_mask: item.pass_mask.0,
            visibility_stages: item.visibility_stages.0,
            sort_key_low: item.sort_key.0 as u32,
            sort_key_high: (item.sort_key.0 >> 32) as u32,
            _padding: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Pod, ShaderType, Zeroable)]
pub struct RenderVisibilityRange {
    pub view_index: u32,
    pub view_generation: u32,
    pub start: u32,
    pub count: u32,
}
impl_atomic_pod!(RenderVisibilityRange, RenderVisibilityRangeBlob);

impl RenderVisibilityRange {
    pub fn new(view: GenerationalHandle, start: u32, count: u32) -> Self {
        Self {
            view_index: view.index,
            view_generation: view.generation,
            start,
            count,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Pod, ShaderType, Zeroable)]
pub struct RenderVisibilityCounter {
    pub visible_count: u32,
    pub rejected_count: u32,
    pub overflow_count: u32,
    pub indexed_count: u32,
    pub non_indexed_count: u32,
    pub _padding: [u32; 3],
}
impl_atomic_pod!(RenderVisibilityCounter, RenderVisibilityCounterBlob);

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Pod, ShaderType, Zeroable)]
pub struct RenderVisibilityDispatch {
    pub view_index: u32,
    pub candidate_count: u32,
    pub output_start: u32,
    pub output_end: u32,
    pub indirect_first_instance: u32,
    pub bin_start: u32,
    pub candidate_bin_start: u32,
    pub _padding: u32,
}
impl_atomic_pod!(RenderVisibilityDispatch, RenderVisibilityDispatchBlob);

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Pod, ShaderType, Zeroable)]
pub struct RenderVisibilityIndirect {
    pub index_count: u32,
    pub instance_count: u32,
    pub first_index: u32,
    pub base_vertex: i32,
    pub first_instance: u32,
}
impl_atomic_pod!(RenderVisibilityIndirect, RenderVisibilityIndirectBlob);

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Pod, ShaderType, Zeroable)]
pub struct RenderVisibilityNonIndexedIndirect {
    pub vertex_count: u32,
    pub instance_count: u32,
    pub first_vertex: u32,
    pub first_instance: u32,
}
impl_atomic_pod!(RenderVisibilityNonIndexedIndirect, RenderVisibilityNonIndexedIndirectBlob);

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Pod, ShaderType, Zeroable)]
pub struct RenderDrawBinHeader {
    pub geometry_index: u32,
    pub geometry_generation: u32,
    pub pipeline_class: u32,
    pub vertex_buffer_class: u32,
    pub index_buffer_class: u32,
    pub indexed: u32,
    pub command_start: u32,
    pub command_capacity: u32,
    pub command_count: u32,
    pub view_index: u32,
    pub view_generation: u32,
    pub lod_or_cluster: u32,
    pub primitive_kind: u32,
    pub _padding_tail: [u32; 3],
}
impl_atomic_pod!(RenderDrawBinHeader, RenderDrawBinHeaderBlob);

impl From<prism_render_visibility::GpuDrawBinHeader> for RenderDrawBinHeader {
    fn from(value: prism_render_visibility::GpuDrawBinHeader) -> Self {
        bytemuck::cast(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layouts_match_visibility_shader_contract() {
        assert_eq!(size_of::<RenderVisibilityView>(), 288);
        assert_eq!(size_of::<RenderVisibilityWorkItem>(), 48);
        assert_eq!(size_of::<RenderVisibilityRange>(), 16);
        assert_eq!(size_of::<RenderVisibilityCounter>(), 32);
        assert_eq!(size_of::<RenderVisibilityDispatch>(), 32);
        assert_eq!(size_of::<RenderVisibilityIndirect>(), 20);
        assert_eq!(size_of::<RenderVisibilityNonIndexedIndirect>(), 16);
        assert_eq!(size_of::<RenderDrawBinHeader>(), 64);
    }
}
