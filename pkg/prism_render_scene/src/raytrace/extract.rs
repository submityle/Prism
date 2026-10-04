//! Render-world → acceleration-structure extraction.
//!
//! [`super::scene`] is a pure `(unique geometries, instances) → (GpuBlasPool,
//! GpuTlasBuffers)` transform; it does not know where those inputs come from.
//! This module is the render-world half that gathers them every frame from the
//! two authoritative render-world resources —
//! [`RenderGpuScene`](crate::scene::RenderGpuScene) (the `CPU` mirror of every
//! live instance's object→world transform and stable geometry handle) and
//! [`RenderShadingGeometryRegistry`](crate::geometry::RenderShadingGeometryRegistry)
//! (the resident per-geometry surface tables) — and republishes the packed
//! hierarchy into the [`RenderWorldAcceleration`] resource that
//! [`super::resources`] uploads and [`super::dispatch`] traverses.
//!
//! Like [`super::scene`] the gather is a pure, device-free `CPU` transform: it
//! takes `&CpuRenderScene` + `&RenderShadingGeometryRegistry` and returns a
//! [`SceneAcceleration`], so it is unit-testable against the golden
//! `GpuTlasBuffers::closest_hit` walk without a `wgpu` device. The thin Bevy
//! system merely plumbs the two resources into the gather and caches the result.
//!
//! # Instance identity
//!
//! An instance's stable id is its scene-mirror slot index, which is exactly the
//! row index the GPU scene instance / transform / material buffers are keyed by
//! (`AtomicSparseBufferVec::grow_and_set(index, …)`). A `TLAS` hit therefore
//! reports an `instance_id` that indexes straight back into those buffers for
//! material / transform resolution, with no side table.
//!
//! # Rebuild gating
//!
//! A full rebuild runs only when the scene or the geometry registry actually
//! changed. The mirror bumps its `scene_epoch` on every create / destroy /
//! transform / geometry update, and the registry bumps a per-entry `revision`
//! on every surface-table edit, so folding both into a single signature lets a
//! static scene skip the `SAH` rebuild entirely while any move or geometry edit
//! forces a fresh structure (a one-frame-granular, never stale, refresh).

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use bevy_ecs::prelude::{Res, ResMut, Resource};

use prism_render_architecture::gpu_scene::{CpuRenderScene, SceneTransform};
use prism_render_architecture::ray_scene::Affine3;

use crate::geometry::RenderShadingGeometryRegistry;
use crate::raytrace::scene::{build_scene_acceleration, SceneAcceleration, SceneInstance};
use crate::scene::RenderGpuScene;

/// The latest packed acceleration structure built from the render world.
///
/// Holds the matched [`SceneAcceleration`] pair plus the source signature it
/// was built from (so an unchanged scene skips the rebuild) and a monotonically
/// increasing revision the uploader keys its dirty check on.
#[derive(Resource, Default)]
pub(crate) struct RenderWorldAcceleration {
    acceleration: Option<SceneAcceleration>,
    signature: Option<u64>,
    revision: u64,
}

impl RenderWorldAcceleration {
    /// The current packed hierarchy, or `None` before the first build.
    pub(crate) fn acceleration(&self) -> Option<&SceneAcceleration> {
        self.acceleration.as_ref()
    }

    /// Monotonic build counter; bumps once per accepted rebuild so a downstream
    /// uploader can cheaply detect a new structure.
    pub(crate) fn revision(&self) -> u64 {
        self.revision
    }

    /// Replaces the cached structure, recording the signature it was built from
    /// and bumping [`revision`](Self::revision).
    fn store(&mut self, acceleration: SceneAcceleration, signature: u64) {
        self.acceleration = Some(acceleration);
        self.signature = Some(signature);
        self.revision = self.revision.wrapping_add(1);
    }
}

/// Converts a render-world row-major `3×4` [`SceneTransform`] into the
/// column-major [`Affine3`] the golden layout places instances with.
///
/// `SceneTransform::rows[r] = [m[r][0], m[r][1], m[r][2], t[r]]` (row-major
/// linear part + translation in the last column); [`Affine3`] stores the three
/// basis *columns* and a translation and applies `cols[0]·x + cols[1]·y +
/// cols[2]·z + translation`, so the linear part is transposed on the way in.
fn affine_from_scene_transform(transform: &SceneTransform) -> Affine3 {
    let m = &transform.rows;
    let cols = [
        [m[0][0], m[1][0], m[2][0]],
        [m[0][1], m[1][1], m[2][1]],
        [m[0][2], m[1][2], m[2][2]],
    ];
    let translation = [m[0][3], m[1][3], m[2][3]];
    Affine3::from_cols(cols, translation)
}

/// Folds the scene epoch, live count and every resident geometry's
/// `(index, revision)` into one signature.
///
/// The mirror bumps `scene_epoch` on any instance create / destroy / transform
/// / geometry change and the registry bumps a per-entry `revision` on any
/// surface-table edit, so an unchanged signature proves the inputs the gather
/// reads are byte-for-byte identical and the cached structure is still valid.
fn scene_signature(scene: &CpuRenderScene, registry: &RenderShadingGeometryRegistry) -> u64 {
    let mut signature = scene.scene_epoch();
    signature = signature.rotate_left(1) ^ u64::from(scene.live_count());
    for entry in registry.entries_for_upload() {
        let word = (u64::from(entry.handle.index) << 32) | u64::from(entry.revision);
        signature = signature.rotate_left(1) ^ word;
    }
    signature
}

/// Gathers every live, geometry-backed instance and builds the packed
/// acceleration hierarchy.
///
/// Walks the scene mirror slot by slot; for each live [`InstanceRecord`] whose
/// geometry handle resolves to a resident surface table it assigns the geometry
/// a dense `BLAS` slot (first-seen order over the sorted registry keys, so the
/// mapping is deterministic) and emits one [`SceneInstance`] carrying the slot,
/// the object→world placement and the mirror slot index as the stable
/// `instance_id`. Instances whose geometry has not been uploaded yet are
/// skipped here; non-invertible placements are skipped downstream by
/// [`build_scene_acceleration`]. An empty scene yields an empty structure.
pub(crate) fn collect_scene_acceleration(
    scene: &CpuRenderScene,
    registry: &RenderShadingGeometryRegistry,
) -> SceneAcceleration {
    let mut blas_slots: BTreeMap<u32, usize> = BTreeMap::new();
    let mut geometries = Vec::new();
    let mut instances = Vec::new();

    let capacity = scene.capacity();
    for index in 0..capacity as u32 {
        let Some((_generation, Some(record))) = scene.record_at(index) else {
            continue;
        };
        let geometry_index = record.geometry.index;
        let Some(entry) = registry.get(geometry_index) else {
            // Geometry handle is live but its surface table has not been
            // uploaded yet; skip until the registry catches up next frame.
            continue;
        };

        let blas = *blas_slots.entry(geometry_index).or_insert_with(|| {
            let slot = geometries.len();
            geometries.push(&entry.geometry);
            slot
        });

        instances.push(SceneInstance {
            blas,
            object_to_world: affine_from_scene_transform(&record.current_transform),
            instance_id: index,
        });
    }

    build_scene_acceleration(&geometries, &instances)
}

/// Rebuilds [`RenderWorldAcceleration`] from the live render world whenever the
/// scene or geometry registry changed since the last accepted build.
pub(crate) fn build_render_world_acceleration(
    scene: Res<RenderGpuScene>,
    registry: Res<RenderShadingGeometryRegistry>,
    mut acceleration: ResMut<RenderWorldAcceleration>,
) {
    let mirror = scene.mirror();
    let signature = scene_signature(mirror, &registry);
    if acceleration.signature == Some(signature) {
        return;
    }
    let built = collect_scene_acceleration(mirror, &registry);
    acceleration.store(built, signature);
}

#[cfg(test)]
mod tests {
    use super::*;

    use prism_render_architecture::abi::GenerationalHandle;
    use prism_render_architecture::gpu_scene::{
        GeometryHandle, InstanceRecord, SceneHandle, SceneOperation, SceneTransactionBuilder,
    };
    use prism_render_architecture::ray_scene::Ray;

    use crate::geometry::{RenderShadingGeometry, RenderShadingPrimitive, RenderShadingVertex};

    /// A unit quad in the `z = 0` plane spanning `[0, 1]²`, two triangles.
    fn unit_quad() -> RenderShadingGeometry {
        let corner = |x: f32, y: f32| RenderShadingVertex {
            position: [x, y, 0.0],
            ..RenderShadingVertex::default()
        };
        RenderShadingGeometry {
            vertices: vec![
                corner(0.0, 0.0),
                corner(1.0, 0.0),
                corner(1.0, 1.0),
                corner(0.0, 1.0),
            ],
            primitives: vec![
                RenderShadingPrimitive {
                    indices: [0, 1, 2],
                    ..RenderShadingPrimitive::default()
                },
                RenderShadingPrimitive {
                    indices: [0, 2, 3],
                    ..RenderShadingPrimitive::default()
                },
            ],
            flags: 0,
        }
    }

    /// A row-major translation `SceneTransform`.
    fn translation(t: [f32; 3]) -> SceneTransform {
        SceneTransform {
            rows: [
                [1.0, 0.0, 0.0, t[0]],
                [0.0, 1.0, 0.0, t[1]],
                [0.0, 0.0, 1.0, t[2]],
            ],
        }
    }

    /// Builds a mirror from `(slot_index, geometry_index, translation)` triples.
    fn scene_with(instances: &[(u32, u32, [f32; 3])]) -> CpuRenderScene {
        let mut mirror = CpuRenderScene::default();
        let mut builder = SceneTransactionBuilder::new(1, 1);
        for (slot, geometry_index, offset) in instances.iter().copied() {
            let mut record = InstanceRecord::default();
            record.geometry = GeometryHandle::new(geometry_index, 1);
            record.current_transform = translation(offset);
            builder.push(SceneOperation::Create {
                handle: SceneHandle::new(slot, 1),
                record,
            });
        }
        mirror.apply(&builder.finish());
        mirror
    }

    fn registry_with(geometries: &[(u32, RenderShadingGeometry)]) -> RenderShadingGeometryRegistry {
        let mut registry = RenderShadingGeometryRegistry::default();
        for (index, geometry) in geometries.iter().cloned() {
            registry.upsert(GenerationalHandle::new(index, 1), 1, geometry);
        }
        registry
    }

    fn ray_down_at(x: f32, y: f32) -> Ray {
        Ray::new([x, y, 1.0], [0.0, 0.0, -1.0], 0.0, 100.0)
    }

    #[test]
    fn affine_from_scene_transform_transposes_linear_and_keeps_translation() {
        // A non-symmetric linear part makes a transpose bug observable, plus a
        // translation. Row-major rows => columns are the transpose.
        let transform = SceneTransform {
            rows: [
                [2.0, 3.0, 0.0, 10.0],
                [0.0, 4.0, 0.0, 20.0],
                [0.0, 0.0, 5.0, 30.0],
            ],
        };
        let affine = affine_from_scene_transform(&transform);
        // p = (1, 1, 1): row-major apply => (2+3+0+10, 0+4+0+20, 0+0+5+30).
        let mapped = affine.transform_point([1.0, 1.0, 1.0]);
        assert!((mapped[0] - 15.0).abs() < 1e-6, "x = {}", mapped[0]);
        assert!((mapped[1] - 24.0).abs() < 1e-6, "y = {}", mapped[1]);
        assert!((mapped[2] - 35.0).abs() < 1e-6, "z = {}", mapped[2]);
        // Pure translation of the origin is the last column.
        let origin = affine.transform_point([0.0, 0.0, 0.0]);
        assert_eq!(origin, [10.0, 20.0, 30.0]);
    }

    #[test]
    fn gather_places_instances_and_matches_golden_walk() {
        // Two placements of one geometry: slot 1 at the origin, slot 2 shifted.
        let mirror = scene_with(&[(1, 7, [0.0, 0.0, 0.0]), (2, 7, [10.0, 0.0, 0.0])]);
        let registry = registry_with(&[(7, unit_quad())]);

        let acceleration = collect_scene_acceleration(&mirror, &registry);
        // One shared geometry => a single BLAS; two live placements.
        assert_eq!(acceleration.blas_count(), 1);
        assert_eq!(acceleration.instance_count(), 2);

        // Instance ids are the mirror slot indices, and the second placement is
        // honoured only at its translated position.
        let near = acceleration
            .tlas
            .closest_hit(&ray_down_at(0.5, 0.5), &acceleration.pool)
            .expect("origin placement must be hit");
        assert_eq!(near.instance_id, 1);
        assert_eq!(near.primitive, 1);

        let far = acceleration
            .tlas
            .closest_hit(&ray_down_at(10.5, 0.5), &acceleration.pool)
            .expect("translated placement must be hit");
        assert_eq!(far.instance_id, 2);

        // The gap between the two placements is empty.
        assert!(acceleration
            .tlas
            .closest_hit(&ray_down_at(5.0, 0.5), &acceleration.pool)
            .is_none());
    }

    #[test]
    fn instances_without_resident_geometry_are_skipped() {
        // Slot 1 references geometry 7 (resident); slot 2 references geometry 9
        // whose surface table has not been uploaded yet.
        let mirror = scene_with(&[(1, 7, [0.0, 0.0, 0.0]), (2, 9, [10.0, 0.0, 0.0])]);
        let registry = registry_with(&[(7, unit_quad())]);

        let acceleration = collect_scene_acceleration(&mirror, &registry);
        assert_eq!(acceleration.blas_count(), 1);
        assert_eq!(acceleration.instance_count(), 1);
        let hit = acceleration
            .tlas
            .closest_hit(&ray_down_at(0.5, 0.5), &acceleration.pool)
            .expect("the resident placement still resolves");
        assert_eq!(hit.instance_id, 1);
    }

    #[test]
    fn empty_scene_yields_empty_structure() {
        let mirror = CpuRenderScene::default();
        let registry = RenderShadingGeometryRegistry::default();
        let acceleration = collect_scene_acceleration(&mirror, &registry);
        assert_eq!(acceleration.blas_count(), 0);
        assert_eq!(acceleration.instance_count(), 0);
    }

    #[test]
    fn signature_is_stable_when_unchanged_and_shifts_on_edit() {
        let mirror = scene_with(&[(1, 7, [0.0, 0.0, 0.0])]);
        let registry = registry_with(&[(7, unit_quad())]);
        let baseline = scene_signature(&mirror, &registry);
        // Recomputing over the same inputs is identical.
        assert_eq!(baseline, scene_signature(&mirror, &registry));

        // A geometry edit (new revision) must shift the signature.
        let mut edited = registry;
        edited.upsert(GenerationalHandle::new(7, 1), 2, unit_quad());
        assert_ne!(baseline, scene_signature(&mirror, &edited));

        // An extra live instance (different live count / epoch) must shift it.
        let grown = scene_with(&[(1, 7, [0.0, 0.0, 0.0]), (2, 7, [4.0, 0.0, 0.0])]);
        let grown_registry = registry_with(&[(7, unit_quad())]);
        assert_ne!(baseline, scene_signature(&grown, &grown_registry));
    }

    #[test]
    fn build_system_caches_until_inputs_change() {
        let mut acceleration = RenderWorldAcceleration::default();
        let mirror = scene_with(&[(1, 7, [0.0, 0.0, 0.0])]);
        let registry = registry_with(&[(7, unit_quad())]);

        // First build publishes a structure and bumps the revision.
        let signature = scene_signature(&mirror, &registry);
        let built = collect_scene_acceleration(&mirror, &registry);
        acceleration.store(built, signature);
        assert_eq!(acceleration.revision(), 1);
        assert_eq!(acceleration.acceleration().unwrap().instance_count(), 1);

        // An unchanged signature means the cached build is reused verbatim.
        assert_eq!(acceleration.signature, Some(signature));
    }
}
