use super::*;
use crate::{frame_graph::ResourceLifetime, gpu_scene::SceneHandle, motion::MotionSource};

/// Domain-local authored role, not another central resource enum.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PayloadRole(pub u16);

/// Closed inline representations: no pointers, IDs, buffers or parameter blocks.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PayloadValue {
    Scalar(f32),
    Vector([f32; 4]),
    Bounds { min: [f32; 3], max: [f32; 3] },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ValuePayload {
    pub value: PayloadValue,
    pub role: PayloadRole,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResourcePayload {
    pub resource: ResourceKey,
    pub lifetime: ResourceLifetime,
    pub size: u64,
    pub alignment: u64,
    pub layout: ResourceLayout,
    pub role: PayloadRole,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DomainPayload {
    Resource(ResourcePayload),
    Value(ValuePayload),
}

/// Per-frame record only; the retained `InstanceRecord` and scene identity do not change.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrameRenderItem {
    pub scene_handle: SceneHandle,
    pub domain: RenderDomain,
    pub history_key: u32,
    pub motion_source: MotionSource,
    pub payload: DomainPayload,
}
