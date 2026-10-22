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
use alloc::sync::Arc;
use alloc::vec::Vec;

use bevy_ecs::prelude::{Res, ResMut, Resource};

use prism_render_architecture::ray_scene::Affine3;
use prism_render_architecture::{
    abi::GenerationalHandle,
    gpu_scene::{CpuRenderScene, SceneTransform},
};

use crate::geometry::{
    resolve_geometry_representation, GeometryRepresentation, GeometryRepresentationRejection,
    GeometryRepresentationSelection, GeometryRepresentationSelections,
    RenderShadingGeometryRegistry,
};
use crate::raytrace::scene::{build_scene_acceleration, SceneAcceleration, SceneInstance};
use crate::scene::RenderGpuScene;

/// Packed ray acceleration structures built from exact per-view visibility snapshots.
#[derive(Resource, Default)]
pub(crate) struct RenderWorldAcceleration {
    view_accelerations: BTreeMap<GenerationalHandle, CachedViewAcceleration>,
    signature: Option<SceneAccelerationSourceRevision>,
    selection_revision: u64,
    revision: u64,
}

/// Per-view hierarchy with its semantic key and currently accepted snapshot.
#[derive(Clone, Debug)]
struct CachedViewAcceleration {
    acceleration: Arc<SceneAcceleration>,
    semantic_signature: Vec<SemanticSelection>,
    selection_revision: u64,
}

/// Stable selection meaning; the per-frame snapshot revision is not semantic.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct SemanticSelection {
    scene: GenerationalHandle,
    geometry: prism_render_architecture::gpu_scene::GeometryHandle,
    geometry_revision: u32,
    representation: GeometryRepresentation,
    allow_conservative_rt_fallback: bool,
}

/// Collision-free source identity for the cached BLAS/TLAS pair.
#[derive(Clone, Debug, Eq, PartialEq)]
struct SceneAccelerationSourceRevision {
    scene_epoch: u64,
    live_count: usize,
    geometry_revisions: Vec<(u32, u32, u32)>,
}

/// Per-build accounting for representation decisions made by RT extraction.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct RayGeometrySelectionDiagnostics {
    /// Instances whose exact view-selected representation was available.
    pub(crate) exact: u32,
    /// Instances explicitly resolved to the conservative RT representation.
    pub(crate) conservative_fallback: u32,
    /// Instances rejected because the selection was stale, unsupported, or unavailable.
    pub(crate) rejected: u32,
    /// Rejection reason counts, stable in enum order through the typed key.
    pub(crate) rejection_reasons: BTreeMap<GeometryRepresentationRejection, u32>,
}

/// Output of extraction when view representation selections are supplied.
#[derive(Clone, Debug)]
pub(crate) struct RayGeometryExtraction {
    pub(crate) acceleration: SceneAcceleration,
    pub(crate) diagnostics: RayGeometrySelectionDiagnostics,
}

/// Resident RT payloads for one geometry generation and geometry-table revision.
#[derive(Clone, Debug, Default)]
pub(crate) struct RayGeometryRepresentations {
    geometry: Option<prism_render_architecture::gpu_scene::GeometryHandle>,
    geometry_revision: u32,
    payloads: BTreeMap<GeometryRepresentation, crate::geometry::RenderShadingGeometry>,
}

impl RayGeometryRepresentations {
    #[must_use]
    pub(crate) fn new(
        geometry: prism_render_architecture::gpu_scene::GeometryHandle,
        geometry_revision: u32,
    ) -> Self {
        Self {
            geometry: Some(geometry),
            geometry_revision,
            payloads: BTreeMap::new(),
        }
    }

    pub(crate) fn insert(
        &mut self,
        representation: GeometryRepresentation,
        geometry: crate::geometry::RenderShadingGeometry,
    ) {
        self.payloads.insert(representation, geometry);
    }
}

impl RenderWorldAcceleration {
    /// Monotonic build counter; bumps once per accepted rebuild so a downstream
    /// uploader can cheaply detect a new structure.
    pub(crate) fn revision(&self) -> u64 {
        self.revision
    }

    /// Replaces the cached structure, recording the signature it was built from
    /// and bumping [`revision`](Self::revision).
    fn store(
        &mut self,
        signature: SceneAccelerationSourceRevision,
        view_accelerations: BTreeMap<GenerationalHandle, CachedViewAcceleration>,
        selection_revision: u64,
        rebuilt: bool,
    ) {
        self.view_accelerations = view_accelerations;
        self.signature = Some(signature);
        self.selection_revision = selection_revision;
        if rebuilt {
            self.revision = self.revision.wrapping_add(1);
        }
    }

    /// Returns the acceleration structure built from one exact visibility
    /// snapshot, if that view was present in the latest frame.
    pub(crate) fn acceleration_for_view(
        &self,
        view: GenerationalHandle,
        selection_revision: u64,
    ) -> Option<&SceneAcceleration> {
        let cached = self.view_accelerations.get(&view)?;
        (cached.selection_revision == selection_revision).then_some(cached.acceleration.as_ref())
    }

    pub(crate) fn acceleration_revision(&self) -> u64 {
        self.revision
    }
}

fn semantic_selection_signature(
    selections: &BTreeMap<GenerationalHandle, GeometryRepresentationSelection>,
) -> Vec<SemanticSelection> {
    selections
        .iter()
        .map(|(scene, selection)| SemanticSelection {
            scene: *scene,
            geometry: selection.identity.geometry,
            geometry_revision: selection.identity.geometry_revision,
            representation: selection.selected,
            allow_conservative_rt_fallback: selection.allow_conservative_rt_fallback,
        })
        .collect()
}

fn selection_snapshot_is_current(
    scene: &CpuRenderScene,
    registry: &RenderShadingGeometryRegistry,
    view: GenerationalHandle,
    revision: u64,
    selections: &BTreeMap<GenerationalHandle, GeometryRepresentationSelection>,
) -> bool {
    selections.iter().all(|(scene_handle, selection)| {
        let Some((generation, Some(record))) = scene.record_at(scene_handle.index) else {
            return false;
        };
        selection.version == crate::geometry::GEOMETRY_REPRESENTATION_CONTRACT_VERSION
            && *scene_handle
                == (GenerationalHandle {
                    index: scene_handle.index,
                    generation,
                })
            && selection.identity.scene == *scene_handle
            && selection.identity.view == Some(view)
            && selection.identity.selection_revision == revision
            && selection.identity.geometry == record.geometry
            && registry.get(record.geometry.index).is_some_and(|entry| {
                entry.handle == record.geometry
                    && entry.revision == selection.identity.geometry_revision
            })
    })
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
/// `(index, generation, revision)` into one signature.
///
/// The mirror bumps `scene_epoch` on any instance create / destroy / transform
/// / geometry change and the registry bumps a per-entry `revision` on any
/// surface-table edit, so an unchanged signature proves the inputs the gather
/// reads are byte-for-byte identical and the cached structure is still valid.
fn scene_signature(
    scene: &CpuRenderScene,
    registry: &RenderShadingGeometryRegistry,
) -> SceneAccelerationSourceRevision {
    SceneAccelerationSourceRevision {
        scene_epoch: scene.scene_epoch(),
        live_count: scene.live_count() as usize,
        geometry_revisions: registry
            .entries_for_upload()
            .map(|entry| (entry.handle.index, entry.handle.generation, entry.revision))
            .collect(),
    }
}

/// Collects the resident CPU surface payloads needed by one exact view
/// selection. The base table is always retained as the explicit conservative
/// RT fallback; a selected LOD is added only when its own table has the same
/// geometry generation and base-table revision.
fn representation_sets_for_view(
    selections: &BTreeMap<GenerationalHandle, GeometryRepresentationSelection>,
    registry: &RenderShadingGeometryRegistry,
) -> BTreeMap<GenerationalHandle, RayGeometryRepresentations> {
    let mut sets = BTreeMap::new();
    for (&scene_handle, selection) in selections {
        let geometry = selection.identity.geometry;
        let Some(base) = registry.get(geometry.index) else {
            continue;
        };
        if base.handle != geometry || base.revision != selection.identity.geometry_revision {
            continue;
        }
        let mut set = RayGeometryRepresentations::new(geometry, base.revision);
        set.insert(
            GeometryRepresentation::Lod { level: 0 },
            base.geometry.clone(),
        );
        if let GeometryRepresentation::Lod { level } = selection.selected
            && level != 0
            && let Some(lod) = registry.lod(geometry, level)
            && lod.handle == geometry
            && lod.revision == base.revision
        {
            set.insert(GeometryRepresentation::Lod { level }, lod.geometry.clone());
        }
        sets.insert(scene_handle, set);
    }
    sets
}

/// Gathers only instances present in this view's validated visibility snapshot.
/// Builds RT geometry against the exact per-instance selections published for
/// one view. The currently resident shading surface is explicitly treated as
/// the conservative RT representation; other selections can only use it when
/// their contract permits that fallback and all identities/revisions match.
pub(crate) fn collect_scene_acceleration_for_view(
    scene: &CpuRenderScene,
    registry: &RenderShadingGeometryRegistry,
    selections: &BTreeMap<GenerationalHandle, GeometryRepresentationSelection>,
    view: GenerationalHandle,
    selection_revision: u64,
) -> RayGeometryExtraction {
    collect_scene_acceleration_with_representations(
        scene,
        registry,
        selections,
        &BTreeMap::new(),
        view,
        selection_revision,
    )
}

/// Builds one view's RT hierarchy from its exact selected resident payloads.
/// The shading registry's geometry is the only implicit payload and is always
/// identified as `ConservativeRayTracing`, never as a raster LOD.
pub(crate) fn collect_scene_acceleration_with_representations(
    scene: &CpuRenderScene,
    registry: &RenderShadingGeometryRegistry,
    selections: &BTreeMap<GenerationalHandle, GeometryRepresentationSelection>,
    representation_sets: &BTreeMap<GenerationalHandle, RayGeometryRepresentations>,
    view: GenerationalHandle,
    selection_revision: u64,
) -> RayGeometryExtraction {
    let mut blas_slots: BTreeMap<
        (
            prism_render_architecture::gpu_scene::GeometryHandle,
            GeometryRepresentation,
        ),
        usize,
    > = BTreeMap::new();
    let mut geometries = Vec::new();
    let mut instances = Vec::new();
    let mut diagnostics = RayGeometrySelectionDiagnostics::default();

    let capacity = scene.capacity();
    for index in 0..capacity as u32 {
        let Some((generation, Some(record))) = scene.record_at(index) else {
            continue;
        };
        let scene_handle = GenerationalHandle { index, generation };
        let geometry_index = record.geometry.index;
        let Some(entry) = registry.get(geometry_index) else {
            // Geometry handle is live but its surface table has not been
            // uploaded yet; skip until the registry catches up next frame.
            continue;
        };
        // Scene slots are reusable. An index hit alone may resolve the
        // replacement geometry after the original handle has retired; reject
        // that stale reference instead of building a BLAS from unrelated data.
        if entry.handle != record.geometry || entry.revision == 0 {
            continue;
        }

        let Some(selection) = selections.get(&scene_handle).copied() else {
            continue;
        };
        let representation_set = representation_sets.get(&scene_handle);
        if let Some(set) = representation_set {
            if set.geometry != Some(record.geometry) || set.geometry_revision != entry.revision {
                diagnostics.rejected += 1;
                let reason = if set.geometry != Some(record.geometry) {
                    GeometryRepresentationRejection::GeometryGenerationMismatch
                } else {
                    GeometryRepresentationRejection::GeometryRevisionMismatch {
                        selected: set.geometry_revision,
                        available: entry.revision,
                    }
                };
                *diagnostics.rejection_reasons.entry(reason).or_default() += 1;
                continue;
            }
        }
        let mut available: Vec<_> = representation_set
            .into_iter()
            .flat_map(|set| set.payloads.keys().copied())
            .collect();
        available.push(GeometryRepresentation::ConservativeRayTracing);
        let resolved = match resolve_geometry_representation(
            selection,
            scene_handle,
            record.geometry,
            entry.revision,
            Some(view),
            selection_revision,
            &available,
        ) {
            Ok(resolved) => {
                if resolved.used_fallback {
                    diagnostics.conservative_fallback += 1;
                } else {
                    diagnostics.exact += 1;
                }
                resolved
            }
            Err(reason) => {
                diagnostics.rejected += 1;
                *diagnostics.rejection_reasons.entry(reason).or_default() += 1;
                continue;
            }
        };
        let blas_key = (record.geometry, resolved.representation);
        let blas = *blas_slots.entry(blas_key).or_insert_with(|| {
            let slot = geometries.len();
            let selected_geometry = if resolved.used_fallback {
                &entry.geometry
            } else {
                representation_set
                    .and_then(|set| set.payloads.get(&resolved.representation))
                    .unwrap_or(&entry.geometry)
            };
            geometries.push(selected_geometry);
            slot
        });

        instances.push(SceneInstance {
            blas,
            object_to_world: affine_from_scene_transform(&record.current_transform),
            instance_id: index,
        });
    }

    RayGeometryExtraction {
        acceleration: build_scene_acceleration(&geometries, &instances),
        diagnostics,
    }
}

/// Rebuilds [`RenderWorldAcceleration`] from the live render world whenever the
/// scene or geometry registry changed since the last accepted build.
pub(crate) fn build_render_world_acceleration(
    scene: Res<RenderGpuScene>,
    registry: Res<RenderShadingGeometryRegistry>,
    selections: Res<GeometryRepresentationSelections>,
    mut acceleration: ResMut<RenderWorldAcceleration>,
) {
    update_render_world_acceleration(&mut acceleration, scene.mirror(), &registry, &selections);
}

fn update_render_world_acceleration(
    acceleration: &mut RenderWorldAcceleration,
    mirror: &CpuRenderScene,
    registry: &RenderShadingGeometryRegistry,
    selections: &GeometryRepresentationSelections,
) {
    let signature = scene_signature(mirror, registry);
    if acceleration.signature.as_ref() == Some(&signature)
        && acceleration.selection_revision == selections.revision()
    {
        return;
    }
    let mut view_accelerations = BTreeMap::new();
    let mut rebuilt = acceleration.signature.as_ref() != Some(&signature);
    for (view, view_selections) in selections.iter() {
        let semantic_signature = semantic_selection_signature(view_selections);
        if acceleration.signature.as_ref() == Some(&signature)
            && selection_snapshot_is_current(
                mirror,
                registry,
                view,
                selections.revision(),
                view_selections,
            )
            && let Some(cached) = acceleration.view_accelerations.get(&view)
            && cached.semantic_signature == semantic_signature
        {
            let mut refreshed = cached.clone();
            refreshed.selection_revision = selections.revision();
            view_accelerations.insert(view, refreshed);
            continue;
        }
        rebuilt = true;
        let representation_sets = representation_sets_for_view(view_selections, registry);
        let extracted = collect_scene_acceleration_with_representations(
            mirror,
            registry,
            view_selections,
            &representation_sets,
            view,
            selections.revision(),
        );
        view_accelerations.insert(
            view,
            CachedViewAcceleration {
                acceleration: Arc::new(extracted.acceleration),
                semantic_signature,
                selection_revision: selections.revision(),
            },
        );
    }
    rebuilt |= acceleration.view_accelerations.len() != view_accelerations.len();
    acceleration.store(
        signature,
        view_accelerations,
        selections.revision(),
        rebuilt,
    );
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

    fn scene_with_geometry_handle(geometry: GeometryHandle) -> CpuRenderScene {
        let mut mirror = CpuRenderScene::default();
        let mut record = InstanceRecord::default();
        record.geometry = geometry;
        let mut builder = SceneTransactionBuilder::new(1, 1);
        builder.push(SceneOperation::Create {
            handle: SceneHandle::new(1, 1),
            record,
        });
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

    fn test_view() -> GenerationalHandle {
        GenerationalHandle::new(99, 1)
    }

    fn collect_scene_acceleration(
        scene: &CpuRenderScene,
        registry: &RenderShadingGeometryRegistry,
    ) -> SceneAcceleration {
        let selections = (0..scene.capacity() as u32)
            .filter_map(|index| {
                let (generation, record) = scene.record_at(index)?;
                let record = record.as_ref()?;
                let handle = GenerationalHandle::new(index, generation);
                Some((
                    handle,
                    GeometryRepresentationSelection::new(
                        crate::geometry::GeometryRepresentationIdentity {
                            scene: handle,
                            geometry: record.geometry,
                            geometry_revision: registry.get(record.geometry.index)?.revision,
                            selection_revision: 1,
                            view: Some(test_view()),
                        },
                        GeometryRepresentation::Lod { level: 0 },
                        true,
                    ),
                ))
            })
            .collect();
        collect_scene_acceleration_for_view(scene, registry, &selections, test_view(), 1)
            .acceleration
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
    fn stale_geometry_generation_never_binds_reused_registry_slot() {
        let stale_handle = GeometryHandle::new(7, 1);
        let replacement_handle = GeometryHandle::new(7, 2);
        let mirror = scene_with_geometry_handle(stale_handle);
        let mut registry = RenderShadingGeometryRegistry::default();
        registry.upsert(replacement_handle, 1, unit_quad());

        let acceleration = collect_scene_acceleration(&mirror, &registry);
        assert_eq!(acceleration.blas_count(), 0);
        assert_eq!(acceleration.instance_count(), 0);
    }

    #[test]
    fn view_lod_selection_uses_declared_conservative_rt_representation() {
        use crate::geometry::{
            GeometryRepresentation, GeometryRepresentationIdentity, GeometryRepresentationSelection,
        };

        let geometry = GeometryHandle::new(7, 1);
        let mirror = scene_with_geometry_handle(geometry);
        let registry = registry_with(&[(7, unit_quad())]);
        let selection = GeometryRepresentationSelection::new(
            GeometryRepresentationIdentity {
                scene: GenerationalHandle::new(1, 1),
                geometry,
                geometry_revision: 1,
                selection_revision: 12,
                view: Some(test_view()),
            },
            GeometryRepresentation::Lod { level: 2 },
            true,
        );
        let extraction = collect_scene_acceleration_for_view(
            &mirror,
            &registry,
            &BTreeMap::from([(GenerationalHandle::new(1, 1), selection)]),
            test_view(),
            12,
        );

        assert_eq!(extraction.acceleration.instance_count(), 1);
        assert_eq!(extraction.diagnostics.exact, 0);
        assert_eq!(extraction.diagnostics.conservative_fallback, 1);
        assert_eq!(extraction.diagnostics.rejected, 0);
    }

    #[test]
    fn matching_view_lod_payload_is_the_geometry_used_for_rt_blas() {
        use crate::geometry::{
            GeometryRepresentation, GeometryRepresentationIdentity, GeometryRepresentationSelection,
        };

        let geometry = GeometryHandle::new(7, 1);
        let mirror = scene_with_geometry_handle(geometry);
        let registry = registry_with(&[(7, unit_quad())]);
        let selection = GeometryRepresentationSelection::new(
            GeometryRepresentationIdentity {
                scene: GenerationalHandle::new(1, 1),
                geometry,
                geometry_revision: 1,
                selection_revision: 12,
                view: Some(test_view()),
            },
            GeometryRepresentation::Lod { level: 2 },
            true,
        );
        let mut representations = RayGeometryRepresentations::new(geometry, 1);
        let tiny_lod = RenderShadingGeometry {
            vertices: vec![
                RenderShadingVertex {
                    position: [0.0, 0.0, 0.0],
                    ..Default::default()
                },
                RenderShadingVertex {
                    position: [0.25, 0.0, 0.0],
                    ..Default::default()
                },
                RenderShadingVertex {
                    position: [0.0, 0.25, 0.0],
                    ..Default::default()
                },
            ],
            primitives: vec![RenderShadingPrimitive {
                indices: [0, 1, 2],
                ..Default::default()
            }],
            flags: 0,
        };
        representations.insert(GeometryRepresentation::Lod { level: 2 }, tiny_lod);
        let extraction = collect_scene_acceleration_with_representations(
            &mirror,
            &registry,
            &BTreeMap::from([(GenerationalHandle::new(1, 1), selection)]),
            &BTreeMap::from([(GenerationalHandle::new(1, 1), representations)]),
            test_view(),
            12,
        );

        assert_eq!(extraction.acceleration.instance_count(), 1);
        assert_eq!(extraction.diagnostics.exact, 1);
        assert_eq!(extraction.diagnostics.conservative_fallback, 0);
        assert!(extraction
            .acceleration
            .tlas
            .closest_hit(&ray_down_at(0.75, 0.25), &extraction.acceleration.pool)
            .is_none());
    }

    #[test]
    fn view_selection_revision_mismatch_is_rejected_without_fallback() {
        use crate::geometry::{
            GeometryRepresentation, GeometryRepresentationIdentity, GeometryRepresentationSelection,
        };

        let geometry = GeometryHandle::new(7, 1);
        let mirror = scene_with_geometry_handle(geometry);
        let registry = registry_with(&[(7, unit_quad())]);
        let mut selection = GeometryRepresentationSelection::new(
            GeometryRepresentationIdentity {
                scene: GenerationalHandle::new(1, 1),
                geometry,
                geometry_revision: 1,
                selection_revision: 12,
                view: Some(test_view()),
            },
            GeometryRepresentation::Lod { level: 2 },
            true,
        );
        // An unknown schema is rejected before even its declared
        // fallback can be considered.
        selection.version += 1;
        let extraction = collect_scene_acceleration_for_view(
            &mirror,
            &registry,
            &BTreeMap::from([(GenerationalHandle::new(1, 1), selection)]),
            test_view(),
            12,
        );
        assert_eq!(extraction.acceleration.instance_count(), 0);
        assert_eq!(extraction.diagnostics.rejected, 1);

        let selection = GeometryRepresentationSelection::new(
            GeometryRepresentationIdentity {
                scene: GenerationalHandle::new(1, 1),
                geometry,
                geometry_revision: 1,
                selection_revision: 12,
                view: Some(test_view()),
            },
            GeometryRepresentation::Lod { level: 2 },
            true,
        );
        let extraction = collect_scene_acceleration_for_view(
            &mirror,
            &registry,
            &BTreeMap::from([(GenerationalHandle::new(1, 1), selection)]),
            test_view(),
            13,
        );
        assert_eq!(extraction.acceleration.instance_count(), 0);
        assert_eq!(extraction.diagnostics.rejected, 1);
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
    fn signature_changes_when_a_geometry_slot_is_reused() {
        let mirror = scene_with(&[(1, 7, [0.0, 0.0, 0.0])]);
        let original = registry_with(&[(7, unit_quad())]);
        let baseline = scene_signature(&mirror, &original);

        let mut replacement = RenderShadingGeometryRegistry::default();
        replacement.upsert(GenerationalHandle::new(7, 2), 1, unit_quad());
        assert_ne!(baseline, scene_signature(&mirror, &replacement));
    }

    #[test]
    fn two_views_build_and_retrieve_only_their_selected_instances() {
        let mirror = scene_with(&[(1, 7, [0.0, 0.0, 0.0]), (2, 7, [10.0, 0.0, 0.0])]);
        let registry = registry_with(&[(7, unit_quad())]);
        let view_a = GenerationalHandle::new(20, 1);
        let view_b = GenerationalHandle::new(21, 3);
        let scene_a = GenerationalHandle::new(1, 1);
        let scene_b = GenerationalHandle::new(2, 1);
        let make_selection = |scene, view| {
            GeometryRepresentationSelection::new(
                crate::geometry::GeometryRepresentationIdentity {
                    scene,
                    geometry: GeometryHandle::new(7, 1),
                    geometry_revision: 1,
                    selection_revision: 44,
                    view: Some(view),
                },
                GeometryRepresentation::Lod { level: 0 },
                false,
            )
        };
        let mut selections = GeometryRepresentationSelections::default();
        selections.begin_frame(44);
        selections.insert(view_a, scene_a, make_selection(scene_a, view_a));
        selections.insert(view_b, scene_b, make_selection(scene_b, view_b));
        let mut cache = RenderWorldAcceleration::default();
        update_render_world_acceleration(&mut cache, &mirror, &registry, &selections);

        let a = cache.acceleration_for_view(view_a, 44).unwrap();
        let b = cache.acceleration_for_view(view_b, 44).unwrap();
        assert_eq!(a.instance_count(), 1);
        assert_eq!(b.instance_count(), 1);
        assert_eq!(
            a.tlas
                .closest_hit(&ray_down_at(0.5, 0.5), &a.pool)
                .unwrap()
                .instance_id,
            1,
        );
        assert_eq!(
            b.tlas
                .closest_hit(&ray_down_at(10.5, 0.5), &b.pool)
                .unwrap()
                .instance_id,
            2,
        );
        assert!(a
            .tlas
            .closest_hit(&ray_down_at(10.5, 0.5), &a.pool)
            .is_none());
        assert!(b
            .tlas
            .closest_hit(&ray_down_at(0.5, 0.5), &b.pool)
            .is_none());
        assert!(cache.acceleration_for_view(view_a, 45).is_none());
        assert!(cache
            .acceleration_for_view(GenerationalHandle::new(view_a.index, 2), 44)
            .is_none());
    }

    #[test]
    fn semantic_selection_cache_reuses_sah_across_visibility_revisions() {
        let mirror = scene_with_geometry_handle(GeometryHandle::new(7, 1));
        let registry = registry_with(&[(7, unit_quad())]);
        let view = test_view();
        let scene = GenerationalHandle::new(1, 1);
        let make_snapshot = |revision| {
            let selection = GeometryRepresentationSelection::new(
                crate::geometry::GeometryRepresentationIdentity {
                    scene,
                    geometry: GeometryHandle::new(7, 1),
                    geometry_revision: 1,
                    selection_revision: revision,
                    view: Some(view),
                },
                GeometryRepresentation::Lod { level: 0 },
                false,
            );
            let mut snapshot = GeometryRepresentationSelections::default();
            snapshot.begin_frame(revision);
            snapshot.insert(view, scene, selection);
            snapshot
        };

        let mut cache = RenderWorldAcceleration::default();
        let first_snapshot = make_snapshot(100);
        update_render_world_acceleration(&mut cache, &mirror, &registry, &first_snapshot);
        let first = cache.acceleration_for_view(view, 100).unwrap() as *const SceneAcceleration;
        let first_build_revision = cache.revision();

        let next_snapshot = make_snapshot(101);
        update_render_world_acceleration(&mut cache, &mirror, &registry, &next_snapshot);
        let next = cache.acceleration_for_view(view, 101).unwrap() as *const SceneAcceleration;

        assert_eq!(
            first, next,
            "semantic stability must reuse the same SAH allocation"
        );
        assert_eq!(cache.revision(), first_build_revision);
        assert!(cache.acceleration_for_view(view, 100).is_none());
    }

    #[test]
    fn build_system_caches_until_inputs_change() {
        let mut acceleration = RenderWorldAcceleration::default();
        let mirror = scene_with(&[(1, 7, [0.0, 0.0, 0.0])]);
        let registry = registry_with(&[(7, unit_quad())]);

        // First build publishes a structure and bumps the revision.
        let signature = scene_signature(&mirror, &registry);
        let view = test_view();
        let built = collect_scene_acceleration(&mirror, &registry);
        acceleration.store(
            signature.clone(),
            BTreeMap::from([(
                view,
                CachedViewAcceleration {
                    acceleration: Arc::new(built),
                    semantic_signature: Vec::new(),
                    selection_revision: 1,
                },
            )]),
            1,
            true,
        );
        assert_eq!(acceleration.revision(), 1);
        assert_eq!(
            acceleration
                .acceleration_for_view(view, 1)
                .unwrap()
                .instance_count(),
            1
        );
        assert!(acceleration.acceleration_for_view(view, 2).is_none());
        assert!(acceleration
            .acceleration_for_view(GenerationalHandle::new(view.index, view.generation + 1), 1)
            .is_none());

        // An unchanged signature means the cached build is reused verbatim.
        assert_eq!(acceleration.signature.as_ref(), Some(&signature));
    }
}
