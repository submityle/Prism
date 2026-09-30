//! `GPU`-ready flat buffer layout for the [`DiskBvh`], plus a packed traversal
//! that reads those buffers and reproduces the in-memory walk bit-for-bit.
//!
//! This is the disk counterpart of [`super::sphere_gpu_layout`] and
//! [`super::cylinder_gpu_layout`]: the `CPU` [`DiskBvh`] is convenient for
//! building and testing, but a compute kernel binds plain storage buffers. This
//! module pins the exact word layout the procedural-primitive intersection
//! kernel consumes and proves — with the same [`Disk::intersect`] and
//! [`Ray::aabb_interval`] arithmetic the in-memory walk uses — that a walk over
//! the flattened buffers returns the identical hit.
//!
//! Node records reuse the shared [`NODE_WORDS`] layout so a kernel can share one
//! `BVH`-node decoder across triangle, sphere, box, curve, cylinder, and disk
//! hierarchies; only the leaf payload differs. Encoding is dependency-free:
//! every buffer is a `Vec<u32>` with `f32` fields stored as their `to_bits`
//! pattern, and the disk stride is a multiple of four words (16-byte aligned).

use super::bvh::Aabb;
use super::disk::{Disk, DiskBvh, DiskHit};
use super::gpu_layout::NODE_WORDS;
use super::traversal::Ray;

/// `u32` words per packed disk (32 bytes, 16-byte aligned).
///
/// Layout: `center.xyz` (0..3), `normal.xyz` (3..6), `radius` (6),
/// `primitive` (7).
pub const DISK_WORDS: usize = 8;

/// Writes an `[f32; 3]` as three `to_bits` words at `out[base..base + 3]`.
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

/// Pops the top node index off the traversal stack, or `None` when empty.
fn pop(stack: &mut [u32; 64], sp: &mut usize) -> Option<u32> {
    if *sp == 0 {
        None
    } else {
        *sp -= 1;
        Some(stack[*sp])
    }
}

/// Flat, `GPU`-uploadable form of a [`DiskBvh`].
///
/// `nodes` holds the shared [`NODE_WORDS`] `BVH` records and `disks` holds the
/// leaf payloads packed with the [`DISK_WORDS`] stride in reordered
/// (leaf-contiguous) primitive order, so a leaf's
/// `[first_primitive, first_primitive + primitive_count)` range indexes `disks`
/// exactly as it indexes [`DiskBvh::disks`] in memory.
#[derive(Clone, Debug, PartialEq)]
pub struct GpuDiskBvhBuffers {
    /// Packed `BVH` nodes, [`NODE_WORDS`] words each.
    pub nodes: Vec<u32>,
    /// Packed disk records, [`DISK_WORDS`] words each, leaf-contiguous.
    pub disks: Vec<u32>,
}

impl GpuDiskBvhBuffers {
    /// Serializes a built [`DiskBvh`] into flat node and disk buffers.
    ///
    /// Node and disk order are preserved exactly, so the packed
    /// `first_primitive`/`primitive_count` ranges index the packed disk array
    /// the same way the in-memory leaf ranges index [`DiskBvh::disks`].
    #[must_use]
    pub fn from_bvh(bvh: &DiskBvh) -> Self {
        let mut nodes = vec![0u32; bvh.nodes().len() * NODE_WORDS];
        for (i, node) in bvh.nodes().iter().enumerate() {
            let b = i * NODE_WORDS;
            write_vec3(&mut nodes, b, node.bounds.min);
            write_vec3(&mut nodes, b + 3, node.bounds.max);
            nodes[b + 6] = node.first_primitive;
            nodes[b + 7] = node.second_child;
            nodes[b + 8] = u32::from(node.primitive_count);
            nodes[b + 9] = u32::from(node.axis);
        }
        let mut disks = vec![0u32; bvh.disks().len() * DISK_WORDS];
        for (i, disk) in bvh.disks().iter().enumerate() {
            let b = i * DISK_WORDS;
            write_vec3(&mut disks, b, disk.center());
            write_vec3(&mut disks, b + 3, disk.normal());
            disks[b + 6] = disk.radius().to_bits();
            disks[b + 7] = disk.primitive();
        }
        Self { nodes, disks }
    }

    /// Number of packed nodes.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len() / NODE_WORDS
    }

    /// Number of packed disks.
    #[must_use]
    pub fn disk_count(&self) -> usize {
        self.disks.len() / DISK_WORDS
    }

    /// True when there are no nodes to traverse.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Decoded bounds of packed node `i`.
    fn node_bounds(&self, i: usize) -> Aabb {
        let b = i * NODE_WORDS;
        Aabb::new(read_vec3(&self.nodes, b), read_vec3(&self.nodes, b + 3))
    }

    /// Decoded disk `i`.
    fn disk(&self, i: usize) -> Disk {
        let b = i * DISK_WORDS;
        // Decode verbatim: the packed normal is already unit length, so routing
        // it back through `Disk::new` would re-normalize and perturb the low
        // bits, breaking the bit-for-bit parity contract.
        Disk::from_parts(
            read_vec3(&self.disks, b),
            read_vec3(&self.disks, b + 3),
            f32::from_bits(self.disks[b + 6]),
            self.disks[b + 7],
        )
    }

    /// Nearest intersection along `ray` by walking the packed buffers.
    ///
    /// Mirrors [`DiskBvh::closest_hit`] exactly — same slab rejection, same
    /// near/far child ordering by split-axis sign, same running `t_max` shrink
    /// and the same [`Disk::intersect`] test — so the result is bit-for-bit
    /// identical and a `GPU` kernel binding these buffers can be diffed against
    /// it.
    #[must_use]
    pub fn closest_hit(&self, ray: &Ray) -> Option<DiskHit> {
        if self.nodes.is_empty() {
            return None;
        }
        let mut ray = *ray;
        let mut best: Option<DiskHit> = None;

        let mut stack = [0u32; 64];
        let mut sp = 0usize;
        let mut node_index = 0u32;
        loop {
            let ni = node_index as usize;
            let bounds = self.node_bounds(ni);
            if ray.aabb_interval(&bounds, ray.t_min(), ray.t_max()).is_some() {
                let base = ni * NODE_WORDS;
                let primitive_count = self.nodes[base + 8];
                if primitive_count > 0 {
                    let start = self.nodes[base + 6] as usize;
                    let end = start + primitive_count as usize;
                    for pi in start..end {
                        let disk = self.disk(pi);
                        if let Some(hit) = disk.intersect(&ray) {
                            ray = Ray::new(ray.origin(), ray.direction(), ray.t_min(), hit.t);
                            best = Some(hit);
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

    /// True when *any* disk intersects `ray`; mirrors [`DiskBvh::any_hit`].
    #[must_use]
    pub fn any_hit(&self, ray: &Ray) -> bool {
        if self.nodes.is_empty() {
            return false;
        }
        let mut stack = [0u32; 64];
        let mut sp = 0usize;
        let mut node_index = 0u32;
        loop {
            let ni = node_index as usize;
            let bounds = self.node_bounds(ni);
            if ray.aabb_interval(&bounds, ray.t_min(), ray.t_max()).is_some() {
                let base = ni * NODE_WORDS;
                let primitive_count = self.nodes[base + 8];
                if primitive_count > 0 {
                    let start = self.nodes[base + 6] as usize;
                    let end = start + primitive_count as usize;
                    for pi in start..end {
                        if self.disk(pi).intersect(ray).is_some() {
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

    /// Small deterministic xorshift `RNG`, matching the other `ray_scene`
    /// suites so tests never depend on an external crate.
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

    fn random_disk(rng: &mut Rng, primitive: u32) -> Disk {
        let center = [
            rng.range(-5.0, 5.0),
            rng.range(-5.0, 5.0),
            rng.range(-5.0, 5.0),
        ];
        let mut normal = [
            rng.range(-1.0, 1.0),
            rng.range(-1.0, 1.0),
            rng.range(-1.0, 1.0),
        ];
        let len2 = normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2];
        if len2 < 1e-3 {
            normal = [0.0, 0.0, 1.0];
        }
        Disk::new(center, normal, rng.range(0.2, 1.2), primitive)
    }

    fn random_scene(rng: &mut Rng, count: u32) -> Vec<Disk> {
        (0..count).map(|i| random_disk(rng, i)).collect()
    }

    #[test]
    fn disk_stride_is_sixteen_byte_aligned() {
        assert_eq!(DISK_WORDS % 4, 0);
    }

    #[test]
    fn from_bvh_preserves_counts_and_buffer_shape() {
        let mut rng = Rng::new(0xABCD_1234);
        let disks = random_scene(&mut rng, 40);
        let bvh = DiskBvh::build(&disks);
        let gpu = GpuDiskBvhBuffers::from_bvh(&bvh);
        assert_eq!(gpu.node_count(), bvh.node_count());
        assert_eq!(gpu.disk_count(), bvh.primitive_count());
        assert_eq!(gpu.nodes.len(), gpu.node_count() * NODE_WORDS);
        assert_eq!(gpu.disks.len(), gpu.disk_count() * DISK_WORDS);
    }

    #[test]
    fn empty_buffers_never_hit() {
        let bvh = DiskBvh::build(&[]);
        let gpu = GpuDiskBvhBuffers::from_bvh(&bvh);
        assert!(gpu.is_empty());
        assert_eq!(gpu.node_count(), 0);
        assert_eq!(gpu.disk_count(), 0);
        let ray = Ray::infinite([0.0, 0.0, 0.0], [0.0, 0.0, -1.0]);
        assert!(gpu.closest_hit(&ray).is_none());
        assert!(!gpu.any_hit(&ray));
    }

    #[test]
    fn decoded_disk_round_trips_fields() {
        let disk = Disk::new([0.1, 0.2, 0.3], [0.4, -0.5, 0.6], 0.45, 9);
        let bvh = DiskBvh::build(&[disk]);
        let gpu = GpuDiskBvhBuffers::from_bvh(&bvh);
        let decoded = gpu.disk(0);
        // Compare against the normalized/folded disk the constructor produced.
        assert_eq!(decoded.center(), disk.center());
        for k in 0..3 {
            assert_eq!(decoded.normal()[k].to_bits(), disk.normal()[k].to_bits());
        }
        assert_eq!(decoded.radius().to_bits(), disk.radius().to_bits());
        assert_eq!(decoded.primitive(), disk.primitive());
    }

    #[test]
    fn packed_walk_matches_in_memory_bvh_bit_for_bit() {
        let mut rng = Rng::new(0x5EED_F00D);
        let disks = random_scene(&mut rng, 64);
        let bvh = DiskBvh::build(&disks);
        let gpu = GpuDiskBvhBuffers::from_bvh(&bvh);

        for _ in 0..3_000 {
            let origin = [
                rng.range(-10.0, 10.0),
                rng.range(-10.0, 10.0),
                rng.range(-10.0, 10.0),
            ];
            let dir = [
                rng.range(-1.0, 1.0),
                rng.range(-1.0, 1.0),
                rng.range(-1.0, 1.0),
            ];
            if dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2] < 1e-6 {
                continue;
            }
            let ray = Ray::infinite(origin, dir);

            let cpu = bvh.closest_hit(&ray);
            let packed = gpu.closest_hit(&ray);
            match (cpu, packed) {
                (None, None) => {}
                (Some(c), Some(p)) => {
                    assert_eq!(c.primitive, p.primitive);
                    assert_eq!(c.t.to_bits(), p.t.to_bits(), "t bits differ");
                    assert_eq!(c.front_face, p.front_face);
                    for k in 0..3 {
                        assert_eq!(
                            c.normal[k].to_bits(),
                            p.normal[k].to_bits(),
                            "normal[{k}] bits differ"
                        );
                    }
                }
                (c, p) => panic!("hit disagreement: {c:?} vs {p:?}"),
            }
            assert_eq!(bvh.any_hit(&ray), gpu.any_hit(&ray));
        }
    }
}
