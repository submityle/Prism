use super::*;
use crate::{
    abi::GenerationalHandle, frame_graph::ResourceLifetime, gpu_scene::SceneHandle,
    motion::MotionSource,
};

/// Generational identity for an object owned by a non-surface render domain.
/// This is deliberately distinct from `SceneHandle`: domain objects are not
/// inserted into the GPU Scene merely to acquire a Fabric identity.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct DomainObjectHandle(GenerationalHandle);

impl DomainObjectHandle {
    #[must_use]
    pub const fn new(index: u32, generation: u32) -> Self {
        Self(GenerationalHandle::new(index, generation))
    }

    #[must_use]
    pub const fn index(self) -> u32 {
        self.0.index
    }

    #[must_use]
    pub const fn generation(self) -> u32 {
        self.0.generation
    }
}

/// Stable identity of one frame-rendered object. Surface objects retain their
/// actual GPU Scene handle; every other domain carries its own generational
/// handle and is never represented by a fabricated `SceneHandle`.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum RenderObjectId {
    Surface(SceneHandle),
    Particle(DomainObjectHandle),
    Volume(DomainObjectHandle),
    Hair(DomainObjectHandle),
    Decal(DomainObjectHandle),
}

impl RenderObjectId {
    #[must_use]
    pub const fn domain(self) -> RenderDomain {
        match self {
            Self::Surface(_) => RenderDomain::Surface,
            Self::Particle(_) => RenderDomain::Particle,
            Self::Volume(_) => RenderDomain::Volume,
            Self::Hair(_) => RenderDomain::Hair,
            Self::Decal(_) => RenderDomain::Decal,
        }
    }

    /// Returns the retained scene identity only for real Surface objects.
    #[must_use]
    pub const fn scene_handle(self) -> Option<SceneHandle> {
        match self {
            Self::Surface(handle) => Some(handle),
            Self::Particle(_) | Self::Volume(_) | Self::Hair(_) | Self::Decal(_) => None,
        }
    }
}

/// Domain-local authored role, not another central resource enum.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
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

impl DomainPayload {
    pub const fn role(self) -> PayloadRole {
        match self {
            Self::Resource(payload) => payload.role,
            Self::Value(payload) => payload.role,
        }
    }
}

/// Per-frame record only; the retained `InstanceRecord` and scene identity do not change.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrameRenderItem {
    pub object_id: RenderObjectId,
    pub domain: RenderDomain,
    pub history_key: u32,
    pub motion_source: MotionSource,
    pub payload: DomainPayload,
}

impl FrameRenderItem {
    /// Roles are local to one extracted object/domain/history stream. Keeping
    /// this identity explicit lets compilation reject accidental overwrites
    /// while still allowing different domains to reuse their authored roles.
    pub const fn payload_role(self) -> PayloadRole {
        self.payload.role()
    }
}
