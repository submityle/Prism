//! The [`PrismRenderScene`] resource: Prism's stable scene plus the registries
//! and bookkeeping the Bevy plugin needs to mirror entities and run culling.

use alloc::collections::{BTreeMap, BTreeSet};

use bevy_ecs::entity::{Entity, EntityHashMap};
use bevy_ecs::prelude::Resource;
use bevy_math::Mat4;

use prism_render_architecture::gpu_scene::{
    CpuRenderScene, GeometryHandle, SceneHandle, SceneMaterialHandle,
};
use prism_render_material::MaterialRecord;
use prism_render_visibility::{GeometryLodChain, ViewHandle};


/// Per-camera culling state tracked across frames.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ViewState {
    /// Stable Prism view handle allocated for this camera entity.
    pub(crate) handle: ViewHandle,
    /// Last frame's `clip_from_world`, forwarded as the view's previous matrix.
    pub(crate) previous_clip: Mat4,
}

/// Resource holding Prism's incremental GPU scene and the data `cull_view`
/// consumes, kept in sync with the Bevy world by [`crate::PrismVisibilityPlugin`].
///
/// Register backend geometry and materials up front with [`Self::insert_geometry`]
/// and [`Self::insert_material`]; the plugin mirrors tagged entities into the
/// scene and runs culling every frame.
#[derive(Resource)]
pub struct PrismRenderScene {
    /// Stable, incrementally updated CPU mirror of the GPU scene.
    pub(crate) scene: CpuRenderScene,
    /// Backend geometry LOD chains, keyed by handle.
    pub(crate) geometry: BTreeMap<GeometryHandle, GeometryLodChain>,
    /// Backend material records, keyed by handle.
    pub(crate) materials: BTreeMap<SceneMaterialHandle, MaterialRecord>,
    /// Live mapping from Bevy entity to its allocated scene handle.
    pub(crate) entity_to_handle: EntityHashMap<SceneHandle>,
    /// Reverse mapping used to resolve cull results back to entities.
    pub(crate) handle_to_entity: BTreeMap<SceneHandle, Entity>,
    /// Dense, sorted handle list passed to `cull_view`; rebuilt on membership change.
    pub(crate) handles: Vec<SceneHandle>,
    /// Monotonic index allocator for new scene handles.
    pub(crate) next_index: u32,
    /// Frame epoch, bumped once per sync.
    pub(crate) frame: u64,
    /// Monotonic transaction sequence number.
    pub(crate) seq: u64,
    /// Per-camera culling state.
    pub(crate) views: EntityHashMap<ViewState>,
    /// Monotonic index allocator for new view handles.
    pub(crate) next_view_index: u32,
    /// Previous-frame LOD selection, consumed by `cull_view` for hysteresis.
    pub(crate) previous_lods: BTreeMap<(ViewHandle, SceneHandle), u16>,
    /// Instances known to be occluded (unused until the HZB slice lands).
    pub(crate) occluded: BTreeSet<(ViewHandle, SceneHandle)>,
    /// Maximum work items a single view may emit.
    pub(crate) capacity: u32,
}

impl Default for PrismRenderScene {
    fn default() -> Self {
        Self {
            scene: CpuRenderScene::default(),
            geometry: BTreeMap::new(),
            materials: BTreeMap::new(),
            entity_to_handle: EntityHashMap::default(),
            handle_to_entity: BTreeMap::new(),
            handles: Vec::new(),
            next_index: 1,
            frame: 0,
            seq: 0,
            views: EntityHashMap::default(),
            next_view_index: 0,
            previous_lods: BTreeMap::new(),
            occluded: BTreeSet::new(),
            capacity: 1 << 20,
        }
    }
}

impl PrismRenderScene {
    /// Registers (or replaces) a backend geometry LOD chain under its handle.
    pub fn insert_geometry(&mut self, chain: GeometryLodChain) {
        self.geometry.insert(chain.geometry, chain);
    }

    /// Registers (or replaces) a backend material record under its handle.
    pub fn insert_material(&mut self, record: MaterialRecord) {
        self.materials.insert(record.handle, record);
    }

    /// Sets the maximum number of work items a single view may emit.
    pub fn set_view_capacity(&mut self, capacity: u32) {
        self.capacity = capacity;
    }

    /// Number of live instances currently mirrored into the scene.
    #[must_use]
    pub fn instance_count(&self) -> usize {
        self.entity_to_handle.len()
    }

    /// Resolves the scene handle allocated for `entity`, if any.
    #[must_use]
    pub fn handle_for(&self, entity: Entity) -> Option<SceneHandle> {
        self.entity_to_handle.get(&entity).copied()
    }

    /// Allocates the next monotonic scene handle.
    ///
    /// Index `0` is reserved by [`CpuRenderScene`] as the null handle, so the
    /// allocator starts at `1`. The generation is `1` because a freshly created
    /// slot defaults to generation `0` and `Create` requires the incoming
    /// generation to strictly exceed the retired one.
    pub(crate) fn allocate_handle(&mut self) -> SceneHandle {
        let handle = SceneHandle::new(self.next_index, 1);
        self.next_index += 1;
        handle
    }

    /// Allocates the next monotonic view handle (generation 0).
    pub(crate) fn allocate_view_handle(&mut self) -> ViewHandle {
        let handle = ViewHandle::new(self.next_view_index, 0);
        self.next_view_index += 1;
        handle
    }

    /// Rebuilds the dense, sorted handle list from the reverse map.
    pub(crate) fn rebuild_handles(&mut self) {
        self.handles = self.handle_to_entity.keys().copied().collect();
    }
}
