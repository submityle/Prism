//! `GPU`-ready flat buffer layout for the matrix-motion [`MotionTlas`], plus a
//! packed traversal that reproduces the in-memory motion walk bit-for-bit.
//!
//! This is the temporal counterpart of [`super::gpu_layout::GpuTlasBuffers`].
//! A static `TLAS` can precompute one `world_to_object` inverse per instance and
//! pack it directly, but a matrix-motion instance is blended per ray at a
//! normalized shutter `time`, so the inverse cannot be baked at build time.
//! Instead each packed record stores *both* key object→world poses (`from` and
//! `to`); the traversal blends them component-wise at `time`, inverts the blend,
//! and transforms the ray into object space — exactly the arithmetic a
//! `WESL`/`GPU` matrix-motion kernel performs, and exactly what
//! [`MotionTlas`](super::motion::MotionTlas) does in memory. Because the packed
//! poses round-trip through `to_bits`/`from_bits` unchanged and the blend +
//! inverse reuse the same [`MotionInstance`] path, the packed walk returns the
//! bit-for-bit identical [`TlasPackedHit`].
//!
//! Encoding is dependency-free: every buffer is a `Vec<u32>` with `f32` fields
//! stored as their `to_bits` pattern, and the [`MOTION_INSTANCE_WORDS`] stride
//! keeps each record 16-byte aligned, matching the shared `BLAS` pool and node
//! layout from [`super::gpu_layout`].

use super::bvh::Aabb;
use super::gpu_layout::{GpuBlasPool, TlasPackedHit, NODE_WORDS};
use super::motion::{MotionInstance, MotionTlas};
use super::tlas::Affine3;
use super::traversal::Ray;

/// `u32` words per packed motion instance (112 bytes, 16-byte aligned).
///
/// Layout: `from` object→world linear columns `c0.xyz` (0..3), `c1.xyz` (3..6),
/// `c2.xyz` (6..9), translation `t.xyz` (9..12); `to` object→world `c0.xyz`
/// (12..15), `c1.xyz` (15..18), `c2.xyz` (18..21), translation `t.xyz` (21..24);
/// `blas_index` (24), `instance_id` (25), DXR-style 8-bit visibility `mask` in
/// the low byte of word 26, padding (27..28). Both poses are stored so the ray
/// blend and its inverse can be recomputed per shutter `time`.
pub const MOTION_INSTANCE_WORDS: usize = 28;

/// Writes an [`Affine3`] object→world pose into `out[base..base + 12]` as three
/// linear columns followed by the translation (each component as `to_bits`).
fn write_affine(out: &mut [u32], base: usize, m: &Affine3) {
    let c = m.columns();
    write_vec3(out, base, c[0]);
    write_vec3(out, base + 3, c[1]);
    write_vec3(out, base + 6, c[2]);
    write_vec3(out, base + 9, m.translation());
}

/// Reads an [`Affine3`] back from `words[base..base + 12]`.
fn read_affine(words: &[u32], base: usize) -> Affine3 {
    Affine3::from_cols(
        [
            read_vec3(words, base),
            read_vec3(words, base + 3),
            read_vec3(words, base + 6),
        ],
        read_vec3(words, base + 9),
    )
}

/// Writes three `to_bits` words at `out[base..base + 3]`.
fn write_vec3(out: &mut [u32], base: usize, v: [f32; 3]) {
    out[base] = v[0].to_bits();
    out[base + 1] = v[1].to_bits();
    out[base + 2] = v[2].to_bits();
}

/// Reads three `from_bits` words at `words[base..base + 3]` back into `[f32; 3]`.
fn read_vec3(words: &[u32], base: usize) -> [f32; 3] {
    [
        f32::from_bits(words[base]),
        f32::from_bits(words[base + 1]),
        f32::from_bits(words[base + 2]),
    ]
}

/// Pops the top of the short traversal stack, or `None` when empty.
#[inline]
fn pop(stack: &mut [u32; 64], sp: &mut usize) -> Option<u32> {
    if *sp == 0 {
        return None;
    }
    *sp -= 1;
    Some(stack[*sp])
}

/// Flattened, `GPU`-uploadable buffers for a built [`MotionTlas`].
///
/// `nodes` reuses the shared [`NODE_WORDS`] node layout (built once over the
/// conservative swept bounds); `instances` packs each motion instance's two key
/// poses with the [`MOTION_INSTANCE_WORDS`] stride, in the `MotionTLAS`
/// reordered instance order. The `BLAS` geometry lives in a shared
/// [`GpuBlasPool`], so one `BLAS` is stored once and referenced by many
/// instances.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct GpuMotionTlasBuffers {
    /// Packed top-level node records, [`NODE_WORDS`] words each.
    pub nodes: Vec<u32>,
    /// Packed motion-instance records, [`MOTION_INSTANCE_WORDS`] words each.
    pub instances: Vec<u32>,
}

impl GpuMotionTlasBuffers {
    /// Serializes a built [`MotionTlas`] into a packed node and instance buffer.
    ///
    /// Node order and instance order are preserved exactly, so
    /// [`TlasPackedHit::instance_index`] matches
    /// [`MotionTlas::instances`](super::motion::MotionTlas::instances).
    #[must_use]
    pub fn from_motion_tlas(tlas: &MotionTlas) -> Self {
        let mut nodes = vec![0u32; tlas.nodes().len() * NODE_WORDS];
        for (i, node) in tlas.nodes().iter().enumerate() {
            let b = i * NODE_WORDS;
            write_vec3(&mut nodes, b, node.bounds.min);
            write_vec3(&mut nodes, b + 3, node.bounds.max);
            nodes[b + 6] = node.first_primitive;
            nodes[b + 7] = node.second_child;
            nodes[b + 8] = u32::from(node.primitive_count);
            nodes[b + 9] = u32::from(node.axis);
        }
        let mut instances = vec![0u32; tlas.instances().len() * MOTION_INSTANCE_WORDS];
        for (i, inst) in tlas.instances().iter().enumerate() {
            let b = i * MOTION_INSTANCE_WORDS;
            write_affine(&mut instances, b, &inst.from_pose());
            write_affine(&mut instances, b + 12, &inst.to_pose());
            instances[b + 24] = inst.blas() as u32;
            instances[b + 25] = inst.instance_id();
            instances[b + 26] = u32::from(inst.mask());
        }
        Self { nodes, instances }
    }

    /// Number of packed top-level nodes.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len() / NODE_WORDS
    }

    /// Number of packed motion instances.
    #[must_use]
    pub fn instance_count(&self) -> usize {
        self.instances.len() / MOTION_INSTANCE_WORDS
    }

    /// The DXR-style 8-bit visibility mask packed for instance `i`.
    ///
    /// Reads the low byte of word 26 of the record (see
    /// [`MOTION_INSTANCE_WORDS`]). Panics if `i` is out of range, matching slice
    /// indexing.
    #[must_use]
    pub fn instance_mask(&self, i: usize) -> u8 {
        (self.instances[i * MOTION_INSTANCE_WORDS + 26] & 0xFF) as u8
    }

    /// True when the `MotionTLAS` holds no instances.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Decoded bounds of packed top-level node `i`.
    fn node_bounds(&self, i: usize) -> Aabb {
        let b = i * NODE_WORDS;
        Aabb::new(read_vec3(&self.nodes, b), read_vec3(&self.nodes, b + 3))
    }

    /// Reconstructs the [`MotionInstance`] packed at record `i` (poses, `BLAS`
    /// index, id, mask), or `None` if either decoded key pose is degenerate.
    ///
    /// The reconstruction reuses [`MotionInstance::with_mask`] so the subsequent
    /// pose blend and inverse follow the identical arithmetic the in-memory
    /// [`MotionTlas`] walk uses, guaranteeing bit-for-bit parity.
    fn read_instance(&self, i: usize) -> Option<(MotionInstance, u32)> {
        let b = i * MOTION_INSTANCE_WORDS;
        let from = read_affine(&self.instances, b);
        let to = read_affine(&self.instances, b + 12);
        let blas = self.instances[b + 24] as usize;
        let instance_id = self.instances[b + 25];
        let mask = (self.instances[b + 26] & 0xFF) as u8;
        let inst = MotionInstance::with_mask(from, to, blas, instance_id, mask)?;
        Some((inst, instance_id))
    }

    /// Nearest intersection along the world-space `ray` at shutter `time`.
    #[must_use]
    pub fn closest_hit(
        &self,
        ray: &Ray,
        time: f32,
        pool: &GpuBlasPool,
    ) -> Option<TlasPackedHit> {
        self.closest_hit_masked(ray, time, pool, 0xFF)
    }

    /// Nearest intersection at `time` restricted to `ray_mask`-included
    /// instances, the packed GPU-ABI twin of
    /// [`MotionTlas::closest_hit_masked`](super::motion::MotionTlas::closest_hit_masked).
    #[must_use]
    pub fn closest_hit_masked(
        &self,
        ray: &Ray,
        time: f32,
        pool: &GpuBlasPool,
        ray_mask: u8,
    ) -> Option<TlasPackedHit> {
        self.walk_closest(ray, time, pool, ray_mask, false)
    }

    /// Watertight nearest intersection at shutter `time`.
    #[must_use]
    pub fn closest_hit_watertight(
        &self,
        ray: &Ray,
        time: f32,
        pool: &GpuBlasPool,
    ) -> Option<TlasPackedHit> {
        self.walk_closest(ray, time, pool, 0xFF, true)
    }

    /// Watertight nearest intersection at `time` restricted to `ray_mask`.
    #[must_use]
    pub fn closest_hit_watertight_masked(
        &self,
        ray: &Ray,
        time: f32,
        pool: &GpuBlasPool,
        ray_mask: u8,
    ) -> Option<TlasPackedHit> {
        self.walk_closest(ray, time, pool, ray_mask, true)
    }

    /// True when *any* instance intersects `ray` at shutter `time`.
    #[must_use]
    pub fn any_hit(&self, ray: &Ray, time: f32, pool: &GpuBlasPool) -> bool {
        self.any_hit_masked(ray, time, pool, 0xFF)
    }

    /// Occlusion query at `time` restricted to `ray_mask`-included instances.
    #[must_use]
    pub fn any_hit_masked(
        &self,
        ray: &Ray,
        time: f32,
        pool: &GpuBlasPool,
        ray_mask: u8,
    ) -> bool {
        self.walk_any(ray, time, pool, ray_mask, false)
    }

    /// Watertight occlusion query at shutter `time`.
    #[must_use]
    pub fn any_hit_watertight(&self, ray: &Ray, time: f32, pool: &GpuBlasPool) -> bool {
        self.any_hit_masked_watertight(ray, time, pool, 0xFF)
    }

    /// Watertight occlusion query at `time` restricted to `ray_mask`.
    #[must_use]
    pub fn any_hit_masked_watertight(
        &self,
        ray: &Ray,
        time: f32,
        pool: &GpuBlasPool,
        ray_mask: u8,
    ) -> bool {
        self.walk_any(ray, time, pool, ray_mask, true)
    }

    /// Shared explicit-stack closest-hit walk over the packed buffers.
    ///
    /// For each leaf instance the packed key poses are blended at `time` and the
    /// blend inverted (degenerate blends are skipped), the ray is transformed
    /// into object space, and the shared `pool` `BLAS` is queried with the
    /// `watertight`-selected walk; the running `t_max` shrinks across instances
    /// exactly like the in-memory motion walk.
    fn walk_closest(
        &self,
        ray: &Ray,
        time: f32,
        pool: &GpuBlasPool,
        ray_mask: u8,
        watertight: bool,
    ) -> Option<TlasPackedHit> {
        if self.nodes.is_empty() {
            return None;
        }
        let mut best: Option<TlasPackedHit> = None;
        let mut best_t = ray.t_max();
        let t_min = ray.t_min();

        let mut stack = [0u32; 64];
        let mut sp = 0usize;
        let mut node_index = 0u32;
        loop {
            let ni = node_index as usize;
            let bounds = self.node_bounds(ni);
            if ray.aabb_interval(&bounds, t_min, best_t).is_some() {
                let base = ni * NODE_WORDS;
                let primitive_count = self.nodes[base + 8];
                if primitive_count > 0 {
                    let start = self.nodes[base + 6] as usize;
                    let end = start + primitive_count as usize;
                    for inst_idx in start..end {
                        let Some((inst, instance_id)) = self.read_instance(inst_idx) else {
                            continue;
                        };
                        if inst.mask() & ray_mask == 0 {
                            continue;
                        }
                        let Some(world_to_object) = inst.world_to_object_at(time) else {
                            continue;
                        };
                        let obj_origin = world_to_object.transform_point(ray.origin());
                        let obj_dir = world_to_object.transform_vector(ray.direction());
                        let obj_ray = Ray::new(obj_origin, obj_dir, t_min, best_t);
                        let hit = if watertight {
                            pool.closest_hit_watertight(inst.blas(), &obj_ray)
                        } else {
                            pool.closest_hit(inst.blas(), &obj_ray)
                        };
                        if let Some(hit) = hit
                            && hit.t < best_t
                        {
                            best_t = hit.t;
                            best = Some(TlasPackedHit {
                                t: hit.t,
                                u: hit.u,
                                v: hit.v,
                                primitive: hit.primitive,
                                instance_id,
                                instance_index: inst_idx as u32,
                            });
                        }
                    }
                    match pop(&mut stack, &mut sp) {
                        Some(n) => node_index = n,
                        None => break,
                    }
                } else {
                    let first_child = node_index + 1;
                    let second_child = self.nodes[base + 7];
                    let axis = self.nodes[base + 9] as usize;
                    let neg = ray.direction()[axis] < 0.0;
                    let (near, far) = if neg {
                        (second_child, first_child)
                    } else {
                        (first_child, second_child)
                    };
                    if sp < stack.len() {
                        stack[sp] = far;
                        sp += 1;
                    }
                    node_index = near;
                }
            } else {
                match pop(&mut stack, &mut sp) {
                    Some(n) => node_index = n,
                    None => break,
                }
            }
        }
        best
    }

    /// Shared explicit-stack any-hit walk over the packed buffers.
    ///
    /// Returns on the first instance for which the mask predicate holds, the
    /// pose blend at `time` is non-degenerate, and the object-space `BLAS` query
    /// reports occlusion; never shrinks `t_max`.
    fn walk_any(
        &self,
        ray: &Ray,
        time: f32,
        pool: &GpuBlasPool,
        ray_mask: u8,
        watertight: bool,
    ) -> bool {
        if self.nodes.is_empty() {
            return false;
        }
        let t_min = ray.t_min();
        let t_max = ray.t_max();

        let mut stack = [0u32; 64];
        let mut sp = 0usize;
        let mut node_index = 0u32;
        loop {
            let ni = node_index as usize;
            let bounds = self.node_bounds(ni);
            if ray.aabb_interval(&bounds, t_min, t_max).is_some() {
                let base = ni * NODE_WORDS;
                let primitive_count = self.nodes[base + 8];
                if primitive_count > 0 {
                    let start = self.nodes[base + 6] as usize;
                    let end = start + primitive_count as usize;
                    for inst_idx in start..end {
                        let Some((inst, _)) = self.read_instance(inst_idx) else {
                            continue;
                        };
                        if inst.mask() & ray_mask == 0 {
                            continue;
                        }
                        let Some(world_to_object) = inst.world_to_object_at(time) else {
                            continue;
                        };
                        let obj_origin = world_to_object.transform_point(ray.origin());
                        let obj_dir = world_to_object.transform_vector(ray.direction());
                        let obj_ray = Ray::new(obj_origin, obj_dir, t_min, t_max);
                        let occluded = if watertight {
                            pool.any_hit_watertight(inst.blas(), &obj_ray)
                        } else {
                            pool.any_hit(inst.blas(), &obj_ray)
                        };
                        if occluded {
                            return true;
                        }
                    }
                    match pop(&mut stack, &mut sp) {
                        Some(n) => node_index = n,
                        None => break,
                    }
                } else {
                    let first_child = node_index + 1;
                    if sp < stack.len() {
                        stack[sp] = self.nodes[base + 7];
                        sp += 1;
                    }
                    node_index = first_child;
                }
            } else {
                match pop(&mut stack, &mut sp) {
                    Some(n) => node_index = n,
                    None => break,
                }
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ray_scene::bvh::{Bvh, Triangle};

    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed | 1)
        }
        fn next_u32(&mut self) -> u32 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            (x >> 32) as u32
        }
        fn unit(&mut self) -> f32 {
            self.next_u32() as f32 / u32::MAX as f32
        }
        fn range(&mut self, lo: f32, hi: f32) -> f32 {
            lo + (hi - lo) * self.unit()
        }
    }

    fn sample_blas() -> Bvh {
        let tris = vec![
            Triangle::new([-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], 0),
            Triangle::new([-1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0], 1),
            Triangle::new([0.5, -1.0, -1.0], [0.5, 1.0, -1.0], [0.5, 1.0, 1.0], 2),
            Triangle::new([0.5, -1.0, -1.0], [0.5, 1.0, 1.0], [0.5, -1.0, 1.0], 3),
        ];
        Bvh::build(&tris)
    }

    fn random_affine(rng: &mut Rng) -> Affine3 {
        let quat = [
            rng.range(-1.0, 1.0),
            rng.range(-1.0, 1.0),
            rng.range(-1.0, 1.0),
            rng.range(-1.0, 1.0),
        ];
        let rot = Affine3::from_quaternion(quat);
        let scale = Affine3::from_scale([
            rng.range(0.4, 2.5),
            rng.range(0.4, 2.5),
            rng.range(0.4, 2.5),
        ]);
        let trans = Affine3::from_translation([
            rng.range(-5.0, 5.0),
            rng.range(-5.0, 5.0),
            rng.range(-5.0, 5.0),
        ]);
        trans.compose(&rot.compose(&scale))
    }

    fn random_scene(rng: &mut Rng, count: u32) -> Vec<MotionInstance> {
        (0..count)
            .map(|id| {
                let from = random_affine(rng);
                let to = random_affine(rng);
                let mask = (rng.next_u32() as u8) | 0x01; // never zero
                MotionInstance::with_mask(from, to, 0, id, mask).unwrap()
            })
            .collect()
    }

    #[test]
    fn motion_instance_words_is_16_byte_aligned() {
        assert_eq!(MOTION_INSTANCE_WORDS, 28);
        assert_eq!((MOTION_INSTANCE_WORDS * 4) % 16, 0);
    }

    #[test]
    fn packed_buffer_shapes_and_mask_match_in_memory() {
        let blases = vec![sample_blas()];
        let mut rng = Rng::new(0x2468_ACE0);
        let scene = random_scene(&mut rng, 9);
        let tlas = MotionTlas::build(&scene, &blases);
        let gpu = GpuMotionTlasBuffers::from_motion_tlas(&tlas);

        assert_eq!(gpu.node_count(), tlas.node_count());
        assert_eq!(gpu.instance_count(), tlas.instances().len());
        assert_eq!(gpu.is_empty(), tlas.is_empty());
        // Packed instance order preserves MotionTlas reorder, so masks line up.
        for (i, inst) in tlas.instances().iter().enumerate() {
            assert_eq!(gpu.instance_mask(i), inst.mask(), "mask[{i}]");
        }
    }

    #[test]
    fn packed_matches_in_memory_bit_for_bit_all_variants() {
        let blases = vec![sample_blas()];
        let pool = GpuBlasPool::from_blases(&blases);
        let mut rng = Rng::new(0xC0FF_EE42);
        let scene = random_scene(&mut rng, 8);
        let tlas = MotionTlas::build(&scene, &blases);
        let gpu = GpuMotionTlasBuffers::from_motion_tlas(&tlas);

        for _ in 0..6000 {
            let time = rng.range(-0.2, 1.2); // exercise the shutter clamp too
            let ray_mask = rng.next_u32() as u8;
            let origin =
                [rng.range(-7.0, 7.0), rng.range(-7.0, 7.0), rng.range(-7.0, 7.0)];
            let target =
                [rng.range(-2.0, 2.0), rng.range(-2.0, 2.0), rng.range(-2.0, 2.0)];
            let dir = [
                target[0] - origin[0],
                target[1] - origin[1],
                target[2] - origin[2],
            ];
            let ray = Ray::infinite(origin, dir);

            for &wt in &[false, true] {
                let want = if wt {
                    tlas.closest_hit_watertight_masked(&ray, time, &blases, ray_mask)
                } else {
                    tlas.closest_hit_masked(&ray, time, &blases, ray_mask)
                };
                let got = if wt {
                    gpu.closest_hit_watertight_masked(&ray, time, &pool, ray_mask)
                } else {
                    gpu.closest_hit_masked(&ray, time, &pool, ray_mask)
                };
                match (want, got) {
                    (None, None) => {}
                    (Some(w), Some(g)) => {
                        assert_eq!(w.t.to_bits(), g.t.to_bits(), "t wt={wt}");
                        assert_eq!(w.u.to_bits(), g.u.to_bits(), "u wt={wt}");
                        assert_eq!(w.v.to_bits(), g.v.to_bits(), "v wt={wt}");
                        assert_eq!(w.primitive, g.primitive, "prim wt={wt}");
                        assert_eq!(w.instance_id, g.instance_id, "id wt={wt}");
                        assert_eq!(w.instance_index, g.instance_index, "idx wt={wt}");
                    }
                    (a, b) => panic!("closest packed mismatch wt={wt}: {a:?} vs {b:?}"),
                }

                let want_occ = if wt {
                    tlas.any_hit_watertight_masked(&ray, time, &blases, ray_mask)
                } else {
                    tlas.any_hit_masked(&ray, time, &blases, ray_mask)
                };
                let got_occ = if wt {
                    gpu.any_hit_masked_watertight(&ray, time, &pool, ray_mask)
                } else {
                    gpu.any_hit_masked(&ray, time, &pool, ray_mask)
                };
                assert_eq!(want_occ, got_occ, "any packed mismatch wt={wt}");
            }
        }
    }

    #[test]
    fn packed_mask_all_matches_unmasked_entry_points() {
        let blases = vec![sample_blas()];
        let pool = GpuBlasPool::from_blases(&blases);
        let mut rng = Rng::new(0x7777_1111);
        let scene = random_scene(&mut rng, 6);
        let tlas = MotionTlas::build(&scene, &blases);
        let gpu = GpuMotionTlasBuffers::from_motion_tlas(&tlas);

        for _ in 0..2000 {
            let time = rng.range(0.0, 1.0);
            let origin =
                [rng.range(-6.0, 6.0), rng.range(-6.0, 6.0), rng.range(-6.0, 6.0)];
            let dir = [rng.range(-1.0, 1.0), rng.range(-1.0, 1.0), rng.range(-1.0, 1.0)];
            let ray = Ray::infinite(origin, dir);

            let plain = gpu.closest_hit(&ray, time, &pool);
            let masked = gpu.closest_hit_masked(&ray, time, &pool, 0xFF);
            assert_eq!(plain, masked);
            assert_eq!(
                gpu.any_hit(&ray, time, &pool),
                gpu.any_hit_masked(&ray, time, &pool, 0xFF)
            );
            // ray_mask 0 hits nothing.
            assert!(gpu.closest_hit_masked(&ray, time, &pool, 0x00).is_none());
            assert!(!gpu.any_hit_masked(&ray, time, &pool, 0x00));
        }
    }

    #[test]
    fn packed_empty_motion_tlas_never_hits() {
        let blases = vec![sample_blas()];
        let pool = GpuBlasPool::from_blases(&blases);
        let tlas = MotionTlas::build(&[], &blases);
        let gpu = GpuMotionTlasBuffers::from_motion_tlas(&tlas);
        assert!(gpu.is_empty());
        assert_eq!(gpu.node_count(), 0);
        assert_eq!(gpu.instance_count(), 0);
        let ray = Ray::infinite([0.0, 0.0, 5.0], [0.0, 0.0, -1.0]);
        assert!(gpu.closest_hit(&ray, 0.5, &pool).is_none());
        assert!(!gpu.any_hit(&ray, 0.5, &pool));
    }
}
