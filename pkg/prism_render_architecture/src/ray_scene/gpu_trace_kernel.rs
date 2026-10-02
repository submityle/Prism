//! Software stackless `BVH`-on-compute traversal kernel: `WESL` source plus the
//! `CPU` dispatch twin that proves it against the golden reference.
//!
//! This is the first real `GPU` traversal kernel to land in `ray_scene`. It is
//! the authoritative *software* fallback of the ray-tracing backend (the
//! `SoftwareBvh` tier), the compute companion to hardware ray tracing
//! (`ray_query` / `acceleration_structure`, `Lumen`-style `HWRT`): when the
//! device cannot offer hardware ray tracing, the renderer walks the same packed
//! `BVH` on compute that the `CPU` reference walks.
//!
//! [`SOFTWARE_BVH_TRACE_WESL`] is the shader; [`dispatch_closest_hit`] is its
//! bit-exact `CPU` twin. The twin consumes the identical ray-record `ABI`
//! ([`super::gpu_trace_io`]), walks the identical packed buffers
//! ([`super::traversal_stackless_gpu_layout::GpuStacklessBvh`], already proven
//! equal to [`super::bvh::Bvh::closest_hit`] bit-for-bit), and writes the
//! identical hit-record `ABI`. Because the sandbox has no `GPU`, this twin is
//! the correctness proof: the parity test diffs decoded dispatch hits against
//! the `CPU` golden, and the structural test keeps the `WESL` entry point and
//! `ABI` word counts in sync with the `Rust` constants.

use super::gpu_trace_io::{decode_ray, encode_hit, TRACE_HIT_WORDS, TRACE_RAY_WORDS};
use super::traversal_stackless_gpu_layout::GpuStacklessBvh;

/// `WESL` source of the software stackless `BVH` traversal compute kernel.
///
/// Bound into the crate so the shader ships and is covered by the structural
/// test; the standalone `naga`/`wesl` compile check runs out of tree.
pub const SOFTWARE_BVH_TRACE_WESL: &str = include_str!("software_bvh_trace.wesl");

/// Runs one kernel invocation on the `CPU`: decodes one packed ray record,
/// walks `bvh` stacklessly, and encodes the committed hit record.
///
/// `ray_words` must hold at least [`TRACE_RAY_WORDS`] words; the returned record
/// holds [`TRACE_HIT_WORDS`] words.
#[must_use]
pub fn trace_closest_hit(bvh: &GpuStacklessBvh, ray_words: &[u32]) -> [u32; TRACE_HIT_WORDS] {
    let ray = decode_ray(ray_words);
    encode_hit(bvh.closest_hit(&ray))
}

/// Runs the whole dispatch on the `CPU`: one invocation per ray record, exactly
/// as the `WESL` kernel maps one `global_invocation_id` to one record.
///
/// `rays` is a flat `array<u32>` of [`TRACE_RAY_WORDS`]-word records; the result
/// is the parallel flat `array<u32>` of [`TRACE_HIT_WORDS`]-word hit records.
/// A trailing partial record (fewer than [`TRACE_RAY_WORDS`] words) is ignored,
/// matching the kernel's `ray_index >= ray_total` early-out.
#[must_use]
pub fn dispatch_closest_hit(bvh: &GpuStacklessBvh, rays: &[u32]) -> Vec<u32> {
    let ray_total = rays.len() / TRACE_RAY_WORDS;
    let mut hits = vec![0u32; ray_total * TRACE_HIT_WORDS];
    for i in 0..ray_total {
        let rbase = i * TRACE_RAY_WORDS;
        let record = trace_closest_hit(bvh, &rays[rbase..rbase + TRACE_RAY_WORDS]);
        let hbase = i * TRACE_HIT_WORDS;
        hits[hbase..hbase + TRACE_HIT_WORDS].copy_from_slice(&record);
    }
    hits
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ray_scene::bvh::{Bvh, Triangle};
    use crate::ray_scene::gpu_trace_io::{decode_hit, encode_ray, WORKGROUP_SIZE};
    use crate::ray_scene::traversal::Ray;

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

    fn random_scene(rng: &mut Rng, count: u32) -> Vec<Triangle> {
        (0..count)
            .map(|i| {
                let base = [
                    rng.range(-6.0, 6.0),
                    rng.range(-6.0, 6.0),
                    rng.range(-6.0, 6.0),
                ];
                let edge = |rng: &mut Rng| {
                    [
                        base[0] + rng.range(-1.5, 1.5),
                        base[1] + rng.range(-1.5, 1.5),
                        base[2] + rng.range(-1.5, 1.5),
                    ]
                };
                Triangle::new(base, edge(rng), edge(rng), i)
            })
            .collect()
    }

    #[test]
    fn empty_scene_dispatch_is_all_misses() {
        let bvh = Bvh::build(&[]);
        let gpu = GpuStacklessBvh::from_bvh(&bvh);
        let ray = Ray::infinite([0.0, 0.0, 0.0], [0.0, 0.0, -1.0]);
        let rays = encode_ray(&ray).to_vec();
        let hits = dispatch_closest_hit(&gpu, &rays);
        assert_eq!(hits.len(), TRACE_HIT_WORDS);
        assert!(decode_hit(&hits).is_none());
    }

    #[test]
    fn trailing_partial_ray_record_is_ignored() {
        let mut rng = Rng::new(0x00C0_FFEE);
        let tris = random_scene(&mut rng, 24);
        let bvh = Bvh::build(&tris);
        let gpu = GpuStacklessBvh::from_bvh(&bvh);

        let ray = Ray::infinite([0.0, 0.0, 10.0], [0.0, 0.0, -1.0]);
        let mut rays = encode_ray(&ray).to_vec();
        // Append a short, incomplete record that must be dropped.
        rays.extend_from_slice(&[1, 2, 3]);
        let hits = dispatch_closest_hit(&gpu, &rays);
        assert_eq!(hits.len(), TRACE_HIT_WORDS);
    }

    #[test]
    fn dispatch_matches_cpu_golden_bit_for_bit() {
        let mut rng = Rng::new(0x51A6_7E55);
        let tris = random_scene(&mut rng, 96);
        let bvh = Bvh::build(&tris);
        let gpu = GpuStacklessBvh::from_bvh(&bvh);

        // Pack a batch of random rays into one dispatch buffer.
        let ray_count = 2_000usize;
        let mut rays = Vec::with_capacity(ray_count * TRACE_RAY_WORDS);
        let mut refs = Vec::with_capacity(ray_count);
        while refs.len() < ray_count {
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
            rays.extend_from_slice(&encode_ray(&ray));
            refs.push(ray);
        }

        let hits = dispatch_closest_hit(&gpu, &rays);
        assert_eq!(hits.len(), ray_count * TRACE_HIT_WORDS);

        for (i, ray) in refs.iter().enumerate() {
            let hbase = i * TRACE_HIT_WORDS;
            let decoded = decode_hit(&hits[hbase..hbase + TRACE_HIT_WORDS]);
            // Dispatch twin == packed stackless walk == CPU golden, bit-for-bit.
            assert_eq!(decoded, gpu.closest_hit(ray));
            assert_eq!(decoded, bvh.closest_hit(ray));
        }
    }

    #[test]
    fn wesl_kernel_declares_expected_abi() {
        let s = SOFTWARE_BVH_TRACE_WESL;
        assert!(s.contains("@compute"));
        assert!(s.contains("fn trace_closest_hit"));
        assert!(s.contains("bvh_nodes"));
        assert!(s.contains("bvh_triangles"));
        assert!(s.contains("bvh_escape"));
        assert!(s.contains("trace_rays"));
        assert!(s.contains("trace_hits"));

        // Workgroup size and ABI word counts stay in sync with the Rust side.
        assert!(s.contains(&format!("@workgroup_size({WORKGROUP_SIZE}, 1, 1)")));
        assert!(s.contains(&format!("RAY_WORDS: u32 = {TRACE_RAY_WORDS}u")));
        assert!(s.contains(&format!("HIT_WORDS: u32 = {TRACE_HIT_WORDS}u")));
    }
}
