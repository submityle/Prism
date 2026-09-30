//! Real-device parity for the vis-buffer payload codec twin:
//! [`GpuVisPayloadCodec`] must reproduce the CPU golden
//! [`pack_cluster_triangle`](prism_render_architecture::virtual_geometry::pack_cluster_triangle),
//! [`cluster_of`](prism_render_architecture::virtual_geometry::cluster_of) and
//! [`triangle_of`](prism_render_architecture::virtual_geometry::triangle_of)
//! for every `(cluster_id, triangle_id)` pair.
//!
//! The test skips (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable
//! core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The payload is packed and unpacked with integer shift and mask, which are
//! defined identically on every backend, so the packed word and both unpacked
//! fields are bit-exact and asserted index-for-index with no tolerance. The
//! scene exercises the triangle-id mask boundary (`0`, `127`, and `128`
//! overflowing back to `0`), large cluster ids that occupy the high bits, the
//! round-trip identity (`cluster_of(pack(c, t)) == c` and
//! `triangle_of(pack(c, t)) == t & 127`), a `>64`-invocation dispatch that
//! spans multiple workgroups, and the empty input.
//!
//! Provenance: Prism's own vis-buffer payload bit layout; no Unreal Engine
//! source or derived code.

use prism_render_architecture::virtual_geometry::{
    cluster_of, pack_cluster_triangle, triangle_of, CLUSTER_TRIANGLE_BITS,
};
use prism_virtual_geometry_gpu::{GpuContext, GpuVisPayloadCodec, PayloadCodec, PayloadInput};

fn input(cluster_id: u32, triangle_id: u32) -> PayloadInput {
    PayloadInput {
        cluster_id,
        triangle_id,
    }
}

/// The reference result for one pair: the packed payload plus both unpacked
/// fields, straight from the CPU golden const fns.
fn expected(pair: &PayloadInput) -> PayloadCodec {
    let packed = pack_cluster_triangle(pair.cluster_id, pair.triangle_id);
    PayloadCodec {
        packed,
        cluster: cluster_of(packed),
        triangle: triangle_of(packed),
    }
}

/// A scene spanning the mask boundary and the high-bit cluster range.
fn scene() -> Vec<PayloadInput> {
    vec![
        // Triangle-id mask boundary: 0, the max in-range value, and the first
        // overflow that wraps back to 0 after masking.
        input(1, 0),
        input(1, 127),
        input(1, 128),
        // Large cluster ids occupying the high 25 bits.
        input(0x0012_3456, 0),
        input(0x0012_3456, 127),
        // A cluster that fills every high bit, with a mid-range triangle.
        input(0x01FF_FFFF, 63),
        // Both fields zero: the all-zero payload.
        input(0, 0),
    ]
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_vis_payload_codec_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping vis-payload-codec parity: no wgpu adapter on this host");
        return;
    };
    let codec = GpuVisPayloadCodec::new(&ctx);
    let inputs = scene();

    let gpu = codec.encode(&ctx, &inputs);
    assert_eq!(gpu.len(), inputs.len(), "one result per input");

    for (i, pair) in inputs.iter().enumerate() {
        let want = expected(pair);
        assert_eq!(
            gpu[i], want,
            "codec mismatch for pair {pair:?}: gpu {:?}, cpu {want:?}",
            gpu[i]
        );
        // Round-trip: the low CLUSTER_TRIANGLE_BITS carry the masked triangle
        // id, the rest carry the cluster id.
        assert_eq!(
            gpu[i].triangle,
            pair.triangle_id & ((1 << CLUSTER_TRIANGLE_BITS) - 1),
            "triangle field must be the masked triangle id"
        );
        assert_eq!(
            gpu[i].cluster, pair.cluster_id,
            "cluster field must round-trip the cluster id"
        );
    }
}

#[test]
fn empty_scene_encodes_to_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let codec = GpuVisPayloadCodec::new(&ctx);
    let out = codec.encode(&ctx, &[]);
    assert!(out.is_empty(), "no inputs encode to no results");
}

#[test]
fn triangle_id_mask_boundary_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let codec = GpuVisPayloadCodec::new(&ctx);
    // Sweep the triangle id across and past the 7-bit boundary; every value at
    // or above 128 must wrap to `value & 127`, identical to the golden.
    let inputs: Vec<PayloadInput> = [0u32, 1, 63, 126, 127, 128, 129, 255, 256]
        .into_iter()
        .map(|t| input(42, t))
        .collect();
    let gpu = codec.encode(&ctx, &inputs);
    let want: Vec<PayloadCodec> = inputs.iter().map(expected).collect();
    assert_eq!(gpu, want, "mask boundary must match the golden");
}

#[test]
fn multi_workgroup_dispatch_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let codec = GpuVisPayloadCodec::new(&ctx);
    // 130 pairs span three workgroups (workgroup_size 64); vary both fields so
    // tiling and per-thread indexing are both exercised.
    let inputs: Vec<PayloadInput> = (0..130u32).map(|i| input(i * 7 + 1, i)).collect();
    let gpu = codec.encode(&ctx, &inputs);
    let want: Vec<PayloadCodec> = inputs.iter().map(expected).collect();
    assert_eq!(gpu.len(), 130, "one result per input across workgroups");
    assert_eq!(gpu, want, "tiled dispatch must match the golden");
}
