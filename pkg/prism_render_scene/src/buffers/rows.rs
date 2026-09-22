use bevy_math::Vec4;
use bevy_render::{impl_atomic_pod, render_resource::AtomicPod};
use bytemuck::{Pod, Zeroable};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub struct RenderGpuSceneInstance {
    pub generation: u32,
    pub flags: u32,
    pub geometry_index: u32,
    pub geometry_generation: u32,
    pub material_index: u32,
    pub material_generation: u32,
    pub render_layers: u32,
    pub active: u32,
}

impl_atomic_pod!(RenderGpuSceneInstance, RenderGpuSceneInstanceBlob);

#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub struct RenderGpuSceneTransform {
    pub row_0: Vec4,
    pub row_1: Vec4,
    pub row_2: Vec4,
}

impl_atomic_pod!(RenderGpuSceneTransform, RenderGpuSceneTransformBlob);

#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub struct RenderGpuSceneBounds {
    pub center_radius: Vec4,
    pub half_extents: Vec4,
}

impl_atomic_pod!(RenderGpuSceneBounds, RenderGpuSceneBoundsBlob);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layouts_match_shader_contract() {
        assert_eq!(size_of::<RenderGpuSceneInstance>(), 32);
        assert_eq!(size_of::<RenderGpuSceneTransform>(), 48);
        assert_eq!(size_of::<RenderGpuSceneBounds>(), 32);
    }
}
