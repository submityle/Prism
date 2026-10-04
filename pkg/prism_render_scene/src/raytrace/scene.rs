//! Scene → acceleration-structure build.
//!
//! The traversal service ([`super::resources`] / [`super::dispatch`]) already
//! knows how to upload a packed [`GpuBlasPool`] / [`GpuTlasBuffers`] to the
//! `GPU` and walk the `tlas_traverse` kernel over it, and the golden
//! `prism_render_architecture::ray_scene` layout knows how to serialize a
//! built [`Bvh`] / [`Tlas`] into those packed word buffers. The one piece that
//! did not exist is the bridge *into* that layout from the live render world:
//! turning the stable per-geometry surface table
//! ([`RenderShadingGeometry`](crate::geometry::RenderShadingGeometry)) and the
//! per-instance object→world transforms into the acceleration hierarchy the
//! kernel traverses.
//!
//! This module is that bridge, and nothing more. It performs no device work and
//! holds no `GPU` handles: it is a pure, `CPU`-verifiable transform from
//! `(unique geometries, instances)` to `(GpuBlasPool, GpuTlasBuffers)`. The
//! render-world extraction system gathers the inputs and feeds the output to
//! [`super::resources`]; the kernel-facing word layout, the `BLAS`→pool
//! rebasing and the world→object instance transforms are all owned by the
//! golden layout, so this file never reinterprets the `ABI`.
//!
//! # Primitive identity
//!
//! Each triangle keeps its index into the geometry's
//! [`primitives`](crate::geometry::RenderShadingGeometry::primitives) array as
//! its stable [`Triangle::primitive`] id. That is exactly the primitive index a
//! visibility-buffer hit reports and the surface table is addressed by, so a
//! `BLAS` hit's `primitive` can be fed straight back into surface
//! reconstruction without a remap. The `SAH` builder reorders triangles
//! internally but always reports hits by this id.

use alloc::vec::Vec;

use prism_render_architecture::ray_scene::{
    Affine3, Bvh, GpuBlasPool, GpuTlasBuffers, Instance, Tlas, Triangle,
};

use crate::geometry::RenderShadingGeometry;

/// One instance to place into the top-level structure.
///
/// `blas` indexes the unique-geometry slice passed alongside this instance to
/// [`build_scene_acceleration`]; `object_to_world` is the placement transform
/// and `instance_id` the stable id carried through to a hit so downstream
/// shading can resolve the instance's material/transform.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct SceneInstance {
    /// Index into the unique-geometry (`BLAS`) slice.
    pub(crate) blas: usize,
    /// Object→world placement of this instance.
    pub(crate) object_to_world: Affine3,
    /// Stable user-facing instance id reported on a hit.
    pub(crate) instance_id: u32,
}

/// The packed acceleration hierarchy ready for upload by [`super::resources`].
///
/// `pool` holds every unique geometry's bottom-level structure concatenated
/// into shared node/triangle buffers; `tlas` holds the top-level nodes and the
/// reordered instance records that reference the pool. The two are a matched
/// pair — a [`GpuTlasBuffers`] only traverses correctly against the
/// [`GpuBlasPool`] it was built with — so they travel together.
#[derive(Clone, Debug)]
pub(crate) struct SceneAcceleration {
    /// Shared bottom-level pool over all unique geometries.
    pub(crate) pool: GpuBlasPool,
    /// Top-level structure over the placed instances.
    pub(crate) tlas: GpuTlasBuffers,
}

impl SceneAcceleration {
    /// Number of `BLAS` entries in the pool.
    #[must_use]
    pub(crate) fn blas_count(&self) -> usize {
        self.pool.blas_count()
    }

    /// Number of packed instances in the top-level structure.
    #[must_use]
    pub(crate) fn instance_count(&self) -> usize {
        self.tlas.instances.len()
            / prism_render_architecture::ray_scene::INSTANCE_WORDS
    }
}

/// Converts one geometry's surface table into traversal triangles.
///
/// A triangle whose indices fall outside the vertex table is dropped rather
/// than panicking: the registry builder validates indices up front, so this is
/// a defensive guard against a corrupted/partial row, not an expected path. The
/// surviving triangles keep their primitive-array index as their stable id.
fn geometry_triangles(geometry: &RenderShadingGeometry) -> Vec<Triangle> {
    let mut triangles = Vec::with_capacity(geometry.primitives.len());
    for (primitive, prim) in geometry.primitives.iter().enumerate() {
        let [a, b, c] = prim.indices;
        let (Some(v0), Some(v1), Some(v2)) = (
            geometry.vertices.get(a as usize),
            geometry.vertices.get(b as usize),
            geometry.vertices.get(c as usize),
        ) else {
            continue;
        };
        triangles.push(Triangle::new(
            v0.position,
            v1.position,
            v2.position,
            primitive as u32,
        ));
    }
    triangles
}

/// Builds the full scene acceleration hierarchy from unique geometries and the
/// instances that place them.
///
/// `geometries` is the de-duplicated set of surface tables — one bottom-level
/// structure is built per entry — and `instances` references them by index. An
/// instance whose `blas` is out of range, or whose `object_to_world` is
/// non-invertible (degenerate/zero scale, which has no world→object inverse to
/// transform rays with), is skipped: it cannot be traversed, so admitting it
/// would only corrupt the top-level bounds. Every surviving instance is placed
/// into a [`Tlas`] over the shared [`GpuBlasPool`] and the pair is returned
/// ready for upload.
#[must_use]
pub(crate) fn build_scene_acceleration(
    geometries: &[&RenderShadingGeometry],
    instances: &[SceneInstance],
) -> SceneAcceleration {
    let blases: Vec<Bvh> = geometries
        .iter()
        .map(|geometry| Bvh::build(&geometry_triangles(geometry)))
        .collect();
    let pool = GpuBlasPool::from_blases(&blases);

    let placed: Vec<Instance> = instances
        .iter()
        .filter(|instance| instance.blas < blases.len())
        .filter_map(|instance| {
            Instance::new(instance.object_to_world, instance.blas, instance.instance_id)
        })
        .collect();
    let tlas = Tlas::build(&placed, &blases);
    let tlas = GpuTlasBuffers::from_tlas(&tlas);

    SceneAcceleration { pool, tlas }
}

#[cfg(test)]
mod tests {
    use super::*;

    use prism_render_architecture::ray_scene::Ray;

    use crate::geometry::{RenderShadingPrimitive, RenderShadingVertex};

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

    /// A `-z`-looking ray that pierces the quad at world `(x, y, 0)`.
    fn ray_down_at(x: f32, y: f32) -> Ray {
        Ray::new([x, y, 1.0], [0.0, 0.0, -1.0], 0.0, 100.0)
    }

    #[test]
    fn single_instance_hits_match_golden_walk() {
        let quad = unit_quad();
        let scene = build_scene_acceleration(
            &[&quad],
            &[SceneInstance {
                blas: 0,
                object_to_world: Affine3::identity(),
                instance_id: 7,
            }],
        );
        assert_eq!(scene.blas_count(), 1);
        assert_eq!(scene.instance_count(), 1);

        // A ray through the lower triangle (primitive 0).
        let hit = scene
            .tlas
            .closest_hit(&ray_down_at(0.75, 0.25), &scene.pool)
            .expect("ray through the quad must hit");
        assert_eq!(hit.instance_id, 7);
        assert_eq!(hit.primitive, 0);
        assert!((hit.t - 1.0).abs() < 1e-4, "t = {}", hit.t);

        // A ray through the upper triangle (primitive 1).
        let hit = scene
            .tlas
            .closest_hit(&ray_down_at(0.25, 0.75), &scene.pool)
            .expect("ray through the quad must hit");
        assert_eq!(hit.primitive, 1);

        // A ray clear of the quad misses.
        assert!(scene
            .tlas
            .closest_hit(&ray_down_at(5.0, 5.0), &scene.pool)
            .is_none());
    }

    #[test]
    fn shared_blas_instanced_twice_resolves_distinct_instances() {
        let quad = unit_quad();
        let scene = build_scene_acceleration(
            &[&quad],
            &[
                SceneInstance {
                    blas: 0,
                    object_to_world: Affine3::identity(),
                    instance_id: 100,
                },
                SceneInstance {
                    blas: 0,
                    object_to_world: Affine3::from_translation([10.0, 0.0, 0.0]),
                    instance_id: 200,
                },
            ],
        );
        // One geometry, two placements: a single BLAS, two instances.
        assert_eq!(scene.blas_count(), 1);
        assert_eq!(scene.instance_count(), 2);

        let near = scene
            .tlas
            .closest_hit(&ray_down_at(0.5, 0.5), &scene.pool)
            .expect("hit on the first placement");
        assert_eq!(near.instance_id, 100);

        let far = scene
            .tlas
            .closest_hit(&ray_down_at(10.5, 0.5), &scene.pool)
            .expect("hit on the translated placement");
        assert_eq!(far.instance_id, 200);
    }

    #[test]
    fn degenerate_and_dangling_instances_are_skipped() {
        let quad = unit_quad();
        let scene = build_scene_acceleration(
            &[&quad],
            &[
                // Valid.
                SceneInstance {
                    blas: 0,
                    object_to_world: Affine3::identity(),
                    instance_id: 1,
                },
                // Non-invertible (zero-scale) transform: no world→object inverse.
                SceneInstance {
                    blas: 0,
                    object_to_world: Affine3::from_cols(
                        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                        [0.0, 0.0, 0.0],
                    ),
                    instance_id: 2,
                },
                // Dangling BLAS reference.
                SceneInstance {
                    blas: 99,
                    object_to_world: Affine3::identity(),
                    instance_id: 3,
                },
            ],
        );
        assert_eq!(scene.instance_count(), 1);
        let hit = scene
            .tlas
            .closest_hit(&ray_down_at(0.5, 0.5), &scene.pool)
            .expect("the one valid instance still resolves");
        assert_eq!(hit.instance_id, 1);
    }

    #[test]
    fn triangles_with_dangling_indices_are_dropped_not_panicked() {
        let mut quad = unit_quad();
        // Corrupt the second primitive to reference a missing vertex.
        quad.primitives[1].indices = [0, 2, 99];
        let triangles = geometry_triangles(&quad);
        // Only the valid first triangle survives, keeping its primitive id 0.
        assert_eq!(triangles.len(), 1);
        assert_eq!(triangles[0].primitive, 0);
    }

    #[test]
    fn empty_scene_builds_empty_hierarchy() {
        let scene = build_scene_acceleration(&[], &[]);
        assert_eq!(scene.blas_count(), 0);
        assert_eq!(scene.instance_count(), 0);
    }
}
