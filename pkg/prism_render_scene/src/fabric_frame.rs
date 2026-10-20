//! Lower the retained render-world scene into the Render Fabric frame contract.
//!
//! `FrameRenderItem` is intentionally smaller than a scene instance. Geometry
//! and material handles remain in the persistent [`RenderGpuScene`] mirror;
//! this module only emits the per-frame identity, temporal classification and
//! the real value payload needed by the first surface slice.

use bevy_ecs::prelude::{Query, Res, ResMut, Resource};
use prism_render_architecture::{
    gpu_scene::{SceneHandle, SceneMaterialHandle},
    motion::MotionSource,
    render_fabric::{
        DomainPayload, FrameRenderItem, PayloadRole, PayloadValue, RenderDomain, RenderObjectId,
        ValuePayload,
    },
};
use thiserror::Error;

use crate::{
    extract::ExtractedSceneInstance, geometry::RenderGeometryRegistry, scene::RenderGpuScene,
};

/// The first real payload role emitted by the scene adapter.
///
/// Bounds are an inline value because they are not a graph resource. The
/// geometry and material rows referenced by the scene handle stay in the
/// persistent GPU Scene tables.
pub const SURFACE_BOUNDS_PAYLOAD_ROLE: PayloadRole = PayloadRole(0);

/// A frame-local, non-empty set of Render Fabric items.
#[derive(Resource, Clone, Debug, Default, PartialEq)]
pub struct RenderFabricFrame {
    frame: u64,
    items: Vec<FrameRenderItem>,
}

/// Monotonic render-world frame identity used by the Fabric extraction seam.
#[derive(Resource, Clone, Copy, Debug, Default)]
pub(crate) struct RenderFabricFrameClock(u64);

/// Builds the frame-local Fabric items after scene changes and geometry
/// registrations have been applied for this render frame.
pub(crate) fn build_render_fabric_frame_system(
    instances: Query<&ExtractedSceneInstance>,
    scene: Res<RenderGpuScene>,
    geometry: Res<RenderGeometryRegistry>,
    mut clock: ResMut<RenderFabricFrameClock>,
    mut frame: ResMut<RenderFabricFrame>,
) {
    clock.0 = clock
        .0
        .checked_add(1)
        .expect("render Fabric frame overflow");
    frame.frame = clock.0;
    rebuild_render_fabric_items(instances.iter(), &scene, &geometry, &mut frame.items)
        .expect("committed render-world instances must lower to valid Fabric items");
}

impl RenderFabricFrame {
    /// Builds the first surface frame from the actual render-world scene.
    pub fn from_instances<'a, I>(
        frame: u64,
        instances: I,
        scene: &RenderGpuScene,
        geometry: &RenderGeometryRegistry,
    ) -> Result<Self, RenderFabricFrameError>
    where
        I: IntoIterator<Item = &'a ExtractedSceneInstance>,
    {
        let mut items = Vec::new();
        rebuild_render_fabric_items(instances, scene, geometry, &mut items)?;

        Ok(Self { frame, items })
    }

    /// Frame number used to produce this transient item list.
    pub const fn frame(&self) -> u64 {
        self.frame
    }

    /// The actual Render Fabric items. Every entry has a real scene handle and
    /// a non-empty value payload.
    pub fn items(&self) -> &[FrameRenderItem] {
        &self.items
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }
}

fn rebuild_render_fabric_items<'a, I>(
    instances: I,
    scene: &RenderGpuScene,
    geometry: &RenderGeometryRegistry,
    items: &mut Vec<FrameRenderItem>,
) -> Result<(), RenderFabricFrameError>
where
    I: IntoIterator<Item = &'a ExtractedSceneInstance>,
{
    items.clear();
    for instance in instances {
        items.push(frame_render_item(instance, scene, geometry)?);
    }
    items.sort_unstable_by_key(|item| {
        (
            item.object_id,
            item.domain as u32,
            item.history_key,
            item.payload_role(),
        )
    });

    for pair in items.windows(2) {
        let first = pair[0];
        let second = pair[1];
        if (
            first.object_id,
            first.domain as u32,
            first.history_key,
            first.payload_role(),
        ) == (
            second.object_id,
            second.domain as u32,
            second.history_key,
            second.payload_role(),
        ) {
            return Err(RenderFabricFrameError::DuplicateItem {
                object: first.object_id,
                domain: first.domain,
                history_key: first.history_key,
                role: first.payload_role(),
            });
        }
    }
    Ok(())
}

/// Build the frame item list without requiring a temporary wrapper at the
/// caller. This is the render-world entry point for the first slice.
pub fn build_render_fabric_frame<'a, I>(
    frame: u64,
    instances: I,
    scene: &RenderGpuScene,
    geometry: &RenderGeometryRegistry,
) -> Result<RenderFabricFrame, RenderFabricFrameError>
where
    I: IntoIterator<Item = &'a ExtractedSceneInstance>,
{
    RenderFabricFrame::from_instances(frame, instances, scene, geometry)
}

/// Convert one committed render-world instance into a real Fabric item.
pub fn frame_render_item(
    instance: &ExtractedSceneInstance,
    scene: &RenderGpuScene,
    geometry: &RenderGeometryRegistry,
) -> Result<FrameRenderItem, RenderFabricFrameError> {
    let handle = instance
        .handle
        .ok_or(RenderFabricFrameError::UncommittedInstance)?;
    let record = scene
        .mirror()
        .get(handle)
        .ok_or(RenderFabricFrameError::StaleSceneHandle { handle })?;

    if let Some(expected) = instance.geometry
        && expected != record.geometry
    {
        return Err(RenderFabricFrameError::GeometryHandleMismatch {
            handle,
            extracted: expected,
            scene: record.geometry,
        });
    }
    if instance.material != SceneMaterialHandle::default() && instance.material != record.material {
        return Err(RenderFabricFrameError::MaterialHandleMismatch {
            handle,
            extracted: instance.material,
            scene: record.material,
        });
    }
    if geometry.record(record.geometry).is_none() {
        return Err(RenderFabricFrameError::MissingGeometry {
            handle,
            geometry: record.geometry,
        });
    }
    if !record.material.is_valid() {
        return Err(RenderFabricFrameError::InvalidMaterial {
            handle,
            material: record.material,
        });
    }

    let bounds = record.bounds;
    let min = [
        bounds.center[0] - bounds.half_extents[0],
        bounds.center[1] - bounds.half_extents[1],
        bounds.center[2] - bounds.half_extents[2],
    ];
    let max = [
        bounds.center[0] + bounds.half_extents[0],
        bounds.center[1] + bounds.half_extents[1],
        bounds.center[2] + bounds.half_extents[2],
    ];
    if min.iter().chain(max.iter()).any(|value| !value.is_finite()) {
        return Err(RenderFabricFrameError::NonFiniteBounds { handle });
    }

    Ok(FrameRenderItem {
        object_id: RenderObjectId::Surface(handle),
        domain: RenderDomain::Surface,
        // The first slice has no independent history allocator yet. The scene
        // slot is the stable stream key; the generational scene handle still
        // guards stale identity when a slot is recycled.
        history_key: handle.index,
        // The retained scene currently exposes one current/previous rigid
        // transform pair, so Rigid is the truthful classification for this
        // slice. Skinned/particle sources need their own extracted contracts.
        motion_source: MotionSource::Rigid,
        payload: DomainPayload::Value(ValuePayload {
            value: PayloadValue::Bounds { min, max },
            role: SURFACE_BOUNDS_PAYLOAD_ROLE,
        }),
    })
}

#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
pub enum RenderFabricFrameError {
    #[error("render-world instance has no committed GPU Scene handle")]
    UncommittedInstance,
    #[error("GPU Scene handle {handle:?} is stale or absent from the mirror")]
    StaleSceneHandle { handle: SceneHandle },
    #[error("scene handle {handle:?} has no resident geometry record for {geometry:?}")]
    MissingGeometry {
        handle: SceneHandle,
        geometry: prism_render_architecture::gpu_scene::GeometryHandle,
    },
    #[error(
        "scene handle {handle:?} extracted geometry {extracted:?}, but the mirror owns {scene:?}"
    )]
    GeometryHandleMismatch {
        handle: SceneHandle,
        extracted: prism_render_architecture::gpu_scene::GeometryHandle,
        scene: prism_render_architecture::gpu_scene::GeometryHandle,
    },
    #[error(
        "scene handle {handle:?} extracted material {extracted:?}, but the mirror owns {scene:?}"
    )]
    MaterialHandleMismatch {
        handle: SceneHandle,
        extracted: SceneMaterialHandle,
        scene: SceneMaterialHandle,
    },
    #[error("scene handle {handle:?} has an invalid material handle {material:?}")]
    InvalidMaterial {
        handle: SceneHandle,
        material: SceneMaterialHandle,
    },
    #[error("scene handle {handle:?} has non-finite bounds")]
    NonFiniteBounds { handle: SceneHandle },
    #[error(
        "render item {object:?}/{domain:?}/history {history_key}/role {role:?} appears more than once in one Fabric frame"
    )]
    DuplicateItem {
        object: RenderObjectId,
        domain: RenderDomain,
        history_key: u32,
        role: PayloadRole,
    },
}

#[cfg(test)]
mod tests {
    use bevy_asset::{Assets, RenderAssetUsages};
    use bevy_ecs::{entity::Entity, world::FromWorld, world::World};
    use bevy_mesh::{Mesh, Mesh3d, PrimitiveTopology};
    use bevy_render::sync_world::MainEntity;
    use bevy_transform::components::GlobalTransform;
    use prism_render_architecture::{
        geometry::{GeometryLodRecord, GeometryRecord},
        gpu_scene::{
            GeometryHandle, InstanceRecord, SceneMaterialHandle, SceneOperation,
            SceneTransactionBuilder, SceneTransform,
        },
    };

    use super::*;
    use crate::buffers::GpuSceneBuffers;

    fn committed_scene() -> (
        RenderGpuScene,
        RenderGeometryRegistry,
        ExtractedSceneInstance,
    ) {
        let mut assets = Assets::<Mesh>::default();
        let mesh = assets.add(Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        ));
        let geometry_handle = GeometryHandle::new(7, 1);
        let material_handle = SceneMaterialHandle::new(11, 3);
        let mut geometry = RenderGeometryRegistry::default();
        assert!(geometry.upsert(
            mesh.id(),
            GeometryRecord {
                handle: geometry_handle,
                lods: vec![GeometryLodRecord {
                    level: 0,
                    resident: true,
                    fallback: true,
                    ..Default::default()
                }],
                ..Default::default()
            },
        ));

        let mut scene = RenderGpuScene::new(16);
        let scene_handle = scene.allocate().expect("scene slot");
        assert!(scene.bind_geometry_mesh(geometry_handle, mesh.id()));
        let record = InstanceRecord {
            current_transform: SceneTransform::IDENTITY,
            previous_transform: SceneTransform::IDENTITY,
            bounds: prism_render_architecture::gpu_scene::SceneBounds {
                center: [2.0, 3.0, 4.0],
                radius: 5.0,
                half_extents: [1.0, 2.0, 3.0],
                _padding: 0.0,
            },
            geometry: geometry_handle,
            material: material_handle,
            render_layers: 1,
            flags: 0,
        };
        let mut transaction_builder = SceneTransactionBuilder::for_producer(1, 1, 1);
        transaction_builder.push(SceneOperation::Create {
            handle: scene_handle,
            record,
        });
        let transaction = transaction_builder.finish();
        let mut world = World::new();
        let mut buffers = GpuSceneBuffers::from_world(&mut world);
        let report = scene.apply_transaction(&mut buffers, &transaction);
        assert!(report.errors.is_empty());

        let extracted = ExtractedSceneInstance {
            main_entity: MainEntity::from(Entity::from_raw_u32(42).unwrap()),
            handle: Some(scene_handle),
            transform: GlobalTransform::IDENTITY,
            bounds: None,
            mesh: Mesh3d(mesh.clone()),
            lod_meshes: Vec::new(),
            geometry: Some(geometry_handle),
            material: material_handle,
            material_asset: None,
            flags: 0,
            render_layers: 1,
        };
        (scene, geometry, extracted)
    }

    #[test]
    fn converts_committed_scene_to_a_real_non_empty_item() {
        let (scene, geometry, extracted) = committed_scene();
        let frame = build_render_fabric_frame(9, [&extracted], &scene, &geometry).unwrap();

        assert_eq!(frame.frame(), 9);
        assert_eq!(frame.len(), 1);
        let item = frame.items()[0];
        assert_eq!(
            item.object_id,
            RenderObjectId::Surface(extracted.handle.unwrap())
        );
        assert_eq!(item.domain, RenderDomain::Surface);
        assert_eq!(item.motion_source, MotionSource::Rigid);
        assert_eq!(
            item.history_key,
            item.object_id.scene_handle().unwrap().index
        );
        assert_eq!(item.payload_role(), SURFACE_BOUNDS_PAYLOAD_ROLE);
        assert!(matches!(
            item.payload,
            DomainPayload::Value(ValuePayload {
                value: PayloadValue::Bounds {
                    min: [1.0, 1.0, 1.0],
                    max: [3.0, 5.0, 7.0],
                },
                ..
            })
        ));
    }

    #[test]
    fn rebuilding_items_reuses_the_frame_allocation() {
        let (scene, geometry, extracted) = committed_scene();
        let mut items = Vec::with_capacity(4);
        let capacity = items.capacity();

        rebuild_render_fabric_items([&extracted], &scene, &geometry, &mut items).unwrap();

        assert_eq!(items.len(), 1);
        assert_eq!(items.capacity(), capacity);
    }

    #[test]
    fn rejects_uncommitted_or_missing_geometry_instead_of_emitting_empty_items() {
        let (scene, geometry, mut extracted) = committed_scene();
        extracted.handle = None;
        assert_eq!(
            frame_render_item(&extracted, &scene, &geometry),
            Err(RenderFabricFrameError::UncommittedInstance)
        );

        let (scene, _, extracted) = committed_scene();
        let empty_geometry = RenderGeometryRegistry::default();
        assert!(matches!(
            frame_render_item(&extracted, &scene, &empty_geometry),
            Err(RenderFabricFrameError::MissingGeometry { .. })
        ));
    }

    #[test]
    fn frame_order_is_deterministic_for_reversed_instance_input() {
        let (mut scene, geometry, extracted) = committed_scene();
        let first = extracted.handle.expect("committed handle");
        let second = scene.allocate().expect("second scene slot");
        let record = *scene.mirror().get(first).expect("committed record");
        let mut transaction_builder = SceneTransactionBuilder::for_producer(2, 2, 2);
        transaction_builder.push(SceneOperation::Create {
            handle: second,
            record,
        });
        let transaction = transaction_builder.finish();
        let mut world = World::new();
        let mut buffers = GpuSceneBuffers::from_world(&mut world);
        let report = scene.apply_transaction(&mut buffers, &transaction);
        assert!(report.errors.is_empty());

        let mut second_instance = extracted.clone();
        second_instance.handle = Some(second);
        let frame = build_render_fabric_frame(2, [&second_instance, &extracted], &scene, &geometry)
            .unwrap();

        assert_eq!(
            frame
                .items()
                .iter()
                .map(|item| item.object_id.scene_handle().unwrap())
                .collect::<Vec<_>>(),
            vec![first, second]
        );
    }

    #[test]
    fn duplicate_frame_handles_are_rejected() {
        let (scene, geometry, extracted) = committed_scene();
        let error =
            build_render_fabric_frame(1, [&extracted, &extracted], &scene, &geometry).unwrap_err();
        assert!(matches!(
            error,
            RenderFabricFrameError::DuplicateItem { .. }
        ));
    }
}
