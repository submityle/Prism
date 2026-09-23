//! Extract-time construction and render-world upload of the compute-friendly
//! surface tables.
//!
//! [`build_shading_geometry`] needs the CPU mesh attributes that only live in
//! the main world's `Assets<Mesh>`, so the payload is built during extraction
//! and staged by asset id. The render world then binds each staged payload to
//! the stable geometry handle assigned by [`RenderGpuScene`] and repacks the
//! GPU storage buffers whenever anything changes. This mirrors the split used
//! by [`super::systems::sync_geometry_registry`] for the rasterizer tables,
//! but retains the vertex/primitive rows the compute resolve pass needs.

use std::collections::{HashMap, HashSet};

use bevy_asset::{AssetEvent, AssetId, Assets};
use bevy_ecs::{message::MessageReader, prelude::*};
use bevy_mesh::{Mesh, Mesh3d};
use bevy_render::{
    renderer::{RenderDevice, RenderQueue},
    Extract,
};

use super::{
    shading::{build_shading_geometry, RenderShadingGeometry},
    shading_buffers::RenderShadingGeometryBuffers,
    shading_registry::RenderShadingGeometryRegistry,
};
use crate::{extract::PrismGpuSceneEntity, scene::RenderGpuScene};

/// One asset's staged surface table plus a monotonic revision. The revision
/// lets the render world detect a rebuilt table even when the handle index is
/// later reused for a different asset.
struct StagedShadingTable {
    geometry: RenderShadingGeometry,
    revision: u32,
}

/// Durable render-world cache of surface tables keyed by mesh asset.
///
/// Rebuilding a table walks every vertex and index, so a table is only
/// regenerated when its asset is added or modified, never every frame.
#[derive(Resource, Default)]
pub(crate) struct ShadingGeometryStaging {
    tables: HashMap<AssetId<Mesh>, StagedShadingTable>,
}

impl ShadingGeometryStaging {
    fn stage(&mut self, asset: AssetId<Mesh>, geometry: RenderShadingGeometry) {
        let revision = self
            .tables
            .get(&asset)
            .map_or(1, |table| table.revision.wrapping_add(1).max(1));
        self.tables
            .insert(asset, StagedShadingTable { geometry, revision });
    }

    fn remove(&mut self, asset: AssetId<Mesh>) {
        self.tables.remove(&asset);
    }

    fn get(&self, asset: AssetId<Mesh>) -> Option<&StagedShadingTable> {
        self.tables.get(&asset)
    }
}

/// Builds the surface table for every Prism mesh that was added or modified
/// this frame while its CPU attributes are still reachable in the main world.
pub(crate) fn extract_shading_geometry(
    meshes: Extract<Res<Assets<Mesh>>>,
    instances: Extract<Query<&Mesh3d, With<PrismGpuSceneEntity>>>,
    mut events: Extract<MessageReader<AssetEvent<Mesh>>>,
    mut staging: ResMut<ShadingGeometryStaging>,
    mut built: Local<HashSet<AssetId<Mesh>>>,
) {
    let referenced: HashSet<AssetId<Mesh>> = instances.iter().map(|mesh| mesh.0.id()).collect();

    // Meshes whose contents changed this frame must be rebuilt even when the
    // asset id is unchanged; unloaded meshes are dropped from the cache.
    let mut dirty: HashSet<AssetId<Mesh>> = HashSet::new();
    for event in events.read() {
        match event {
            AssetEvent::Added { id }
            | AssetEvent::Modified { id }
            | AssetEvent::LoadedWithDependencies { id } => {
                dirty.insert(*id);
            }
            AssetEvent::Removed { id } | AssetEvent::Unused { id } => {
                staging.remove(*id);
                built.remove(id);
            }
        }
    }

    for asset in referenced {
        let needs_build = dirty.contains(&asset) || !built.contains(&asset);
        if !needs_build {
            continue;
        }
        let Some(mesh) = meshes.get(asset) else {
            continue;
        };
        // Unsupported topology or attribute layouts keep any prior table
        // intact and are retried the next time the asset changes.
        if let Ok(geometry) = build_shading_geometry(mesh) {
            staging.stage(asset, geometry);
            built.insert(asset);
        }
    }
}

/// Binds staged surface tables to their stable geometry handles, drops slots
/// whose geometry the scene retired, and repacks the GPU buffers on change.
pub(crate) fn sync_shading_geometry_registry(
    scene: Res<RenderGpuScene>,
    staging: Res<ShadingGeometryStaging>,
    mut registry: ResMut<RenderShadingGeometryRegistry>,
    mut buffers: ResMut<RenderShadingGeometryBuffers>,
) {
    let mut live: HashSet<u32> = HashSet::new();
    for (asset, handle) in scene.geometry_assets() {
        live.insert(handle.index);
        if let Some(table) = staging.get(asset) {
            registry.upsert(handle, table.revision, table.geometry.clone());
        }
    }
    registry.retain_live(|index| live.contains(&index));

    if registry.take_dirty() {
        buffers.rebuild(&registry);
    }
}

/// Streams the packed surface tables to the GPU once per frame.
pub(crate) fn upload_shading_geometry_buffers(
    mut buffers: ResMut<RenderShadingGeometryBuffers>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
) {
    buffers.upload(&device, &queue);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::{RenderShadingPrimitive, RenderShadingVertex};
    use prism_render_architecture::abi::GenerationalHandle;

    fn geometry(marker: f32) -> RenderShadingGeometry {
        RenderShadingGeometry {
            vertices: vec![RenderShadingVertex {
                position: [marker; 3],
                ..Default::default()
            }],
            primitives: vec![RenderShadingPrimitive {
                indices: [0, 0, 0],
                flags: 0,
            }],
            flags: 0,
        }
    }

    #[test]
    fn staging_bumps_revision_only_on_restage() {
        let mut staging = ShadingGeometryStaging::default();
        let asset = AssetId::<Mesh>::default();
        staging.stage(asset, geometry(1.0));
        assert_eq!(staging.get(asset).unwrap().revision, 1);
        staging.stage(asset, geometry(2.0));
        assert_eq!(staging.get(asset).unwrap().revision, 2);
        staging.remove(asset);
        assert!(staging.get(asset).is_none());
    }

    #[test]
    fn registry_retains_only_live_geometry() {
        let mut registry = RenderShadingGeometryRegistry::default();
        registry.upsert(GenerationalHandle { index: 1, generation: 1 }, 1, geometry(1.0));
        registry.upsert(GenerationalHandle { index: 4, generation: 1 }, 1, geometry(4.0));
        let _ = registry.take_dirty();

        let live: HashSet<u32> = [1u32].into_iter().collect();
        registry.retain_live(|index| live.contains(&index));

        assert!(registry.take_dirty());
        assert!(registry.get(1).is_some());
        assert!(registry.get(4).is_none());
    }
}
