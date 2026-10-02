//! Hardware `ray_query` traversal kernel: the first real `rayQuery` `WESL`
//! shader in `ray_scene` plus the host-side `ABI` constants that bind it.
//!
//! This is the hardware-ray-tracing (`HWRT`) tier of the ray-tracing backend,
//! aligned with UE5 `Lumen`'s `HWRT` path (`DXR` / Vulkan ray tracing). Where
//! [`super::gpu_trace_kernel`] walks a packed `BVH` on compute (the software
//! `SWRT` fallback), [`HARDWARE_RAY_QUERY_WESL`] delegates traversal to the
//! device acceleration structure through the `ray_query` intrinsics
//! (`rayQueryInitialize` / `rayQueryProceed` /
//! `rayQueryGetCommittedIntersection`).
//!
//! Both kernels share the [`super::gpu_trace_io`] record `ABI` — one
//! [`super::gpu_trace_io::TRACE_RAY_WORDS`]-word ray record in, one
//! [`super::gpu_trace_io::TRACE_HIT_WORDS`]-word hit record out per invocation —
//! so a host swaps the `SWRT` and `HWRT` kernels behind one buffer contract and
//! decodes either result with [`super::gpu_trace_io::decode_hit`].
//!
//! Because the sandbox has no `GPU` and `acceleration_structure` has no `CPU`
//! twin, this kernel cannot be proven by a bit-exact parity test the way the
//! software kernel is; its correctness guard here is twofold: the shader is
//! validated through the same `wesl` + `naga` path the renderer uses (with the
//! `RAY_QUERY` capability) out of tree, and the structural tests below keep its
//! entry point, intrinsics, and record `ABI` word counts in lockstep with the
//! shared [`super::gpu_trace_io`] constants.
//!
//! Barycentric note: a committed hardware intersection reports barycentrics in
//! the `DXR` / Vulkan ray-tracing convention, so the host reads hit words `1..3`
//! as those committed barycentrics rather than the software Möller–Trumbore
//! `u`/`v`; the two kernels share the record *layout*, not a bit-exact result.

/// `WESL` source of the hardware `ray_query` traversal compute kernel.
///
/// Bound into the crate so the shader ships and is covered by the structural
/// tests; the standalone `wesl` + `naga` compile check (which needs the
/// `RAY_QUERY` capability) runs out of tree.
pub const HARDWARE_RAY_QUERY_WESL: &str = include_str!("hardware_ray_query.wesl");

/// `RayDesc` flag word requesting default traversal (no face culling, no opaque
/// forcing); mirrors the `RAY_FLAG_NONE` constant in the kernel.
pub const RAY_FLAG_NONE: u32 = 0;

/// `RayDesc` instance cull-mask selecting every instance (all eight mask bits
/// set); mirrors the `RAY_CULL_MASK_ALL` constant in the kernel.
pub const RAY_CULL_MASK_ALL: u32 = 0xFF;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ray_scene::gpu_trace_io::{TRACE_HIT_WORDS, TRACE_RAY_WORDS, WORKGROUP_SIZE};

    #[test]
    fn wesl_enables_hardware_ray_query() {
        let s = HARDWARE_RAY_QUERY_WESL;
        assert!(s.contains("enable wgpu_ray_query;"));
        assert!(s.contains("acceleration_structure"));
        assert!(s.contains("rayQueryInitialize"));
        assert!(s.contains("rayQueryProceed"));
        assert!(s.contains("rayQueryGetCommittedIntersection"));
        assert!(s.contains("RAY_QUERY_INTERSECTION_NONE"));
    }

    #[test]
    fn wesl_shares_trace_io_abi() {
        let s = HARDWARE_RAY_QUERY_WESL;
        assert!(s.contains("fn trace_closest_hit"));
        assert!(s.contains("trace_rays"));
        assert!(s.contains("trace_hits"));
        assert!(s.contains(&format!("@workgroup_size({WORKGROUP_SIZE}, 1, 1)")));
        assert!(s.contains(&format!("RAY_WORDS: u32 = {TRACE_RAY_WORDS}u")));
        assert!(s.contains(&format!("HIT_WORDS: u32 = {TRACE_HIT_WORDS}u")));
    }

    #[test]
    fn ray_desc_constants_match_wesl() {
        let s = HARDWARE_RAY_QUERY_WESL;
        assert!(s.contains(&format!("RAY_FLAG_NONE: u32 = {RAY_FLAG_NONE}u")));
        assert!(s.contains(&format!("RAY_CULL_MASK_ALL: u32 = {RAY_CULL_MASK_ALL}u")));
    }
}
