//! Real-device parity for the 64-bit vis-buffer word codec twin:
//! [`GpuVisWordCodec`] must reproduce the CPU golden
//! [`pack_vis`](prism_render_architecture::virtual_geometry::pack_vis),
//! [`vis_depth`](prism_render_architecture::virtual_geometry::vis_depth) and
//! [`vis_payload`](prism_render_architecture::virtual_geometry::vis_payload)
//! for every `(depth_key, payload)` pair.
//!
//! The test skips (with a printed notice) when the host has no `wgpu` adapter
//! or when the adapter lacks the 64-bit integer feature the `u64` kernel
//! needs, so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any device that supports it.
//!
//! # Parity criterion
//!
//! The word is packed and unpacked with 64-bit shift, OR and mask, which are
//! defined identically on every backend that supports them, so the packed
//! word and both unpacked fields are bit-exact and asserted index-for-index
//! with no tolerance. The scene exercises the field boundaries (`0`,
//! `0xFFFF_FFFF`, the `0x8000_0000` sign bit, and mixed high/low patterns),
//! the round-trip identity (`vis_depth(pack(d, p)) == d` and
//! `vis_payload(pack(d, p)) == p`), a `>64`-invocation dispatch that spans
//! multiple workgroups, and the empty input.
//!
//! Provenance: Prism's own vis-buffer word bit layout; no Unreal Engine source
//! or derived code.

use prism_render_architecture::virtual_geometry::{pack_vis, vis_depth, vis_payload};
use prism_virtual_geometry_gpu::{GpuContext, GpuVisWordCodec, VisWordCodec, VisWordInput};

fn input(depth_key: u32, payload: u32) -> VisWordInput {
    VisWordInput { depth_key, payload }
}

/// The reference result for one pair: the packed word plus both unpacked
/// fields, straight from the CPU golden const fns.
fn expected(pair: &VisWordInput) -> VisWordCodec {
    let packed = pack_vis(pair.depth_key, pair.payload);
    VisWordCodec {
        packed,
        depth: vis_depth(packed),
        payload: vis_payload(packed),
    }
}

/// A scene spanning both field boundaries and the sign-bit patterns.
fn scene() -> Vec<VisWordInput> {
    vec![
        // Both fields zero: the all-zero word.
        input(0, 0),
        // Low field saturated, high field zero: isolates the low 32 bits.
        input(0, 0xFFFF_FFFF),
        // High field saturated, low field zero: isolates the high 32 bits.
        input(0xFFFF_FFFF, 0),
        // Both saturated: every bit set.
        input(0xFFFF_FFFF, 0xFFFF_FFFF),
        // Sign bit set in each field independently.
        input(0x8000_0000, 1),
        input(1, 0x8000_0000),
        // Mixed high/low bit patterns that must not bleed across the split.
        input(0x1234_5678, 0x9ABC_DEF0),
        input(0xDEAD_BEEF, 0x0BAD_F00D),
    ]
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without the 64-bit integer feature"
)]
fn gpu_vis_word_codec_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping vis-word-codec parity: no wgpu adapter on this host");
        return;
    };
    let Some(codec) = GpuVisWordCodec::new(&ctx) else {
        eprintln!("skipping vis-word-codec parity: adapter lacks the 64-bit integer feature");
        return;
    };
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
        // Round-trip: the high 32 bits carry the depth key, the low 32 bits
        // carry the payload, with no cross-contamination.
        assert_eq!(
            gpu[i].depth, pair.depth_key,
            "depth field must round-trip the depth key"
        );
        assert_eq!(
            gpu[i].payload, pair.payload,
            "payload field must round-trip the payload"
        );
        assert_eq!(
            gpu[i].packed,
            (u64::from(pair.depth_key) << 32) | u64::from(pair.payload),
            "packed word must place the depth key in the high 32 bits"
        );
    }
}

#[test]
fn empty_scene_encodes_to_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let Some(codec) = GpuVisWordCodec::new(&ctx) else {
        return;
    };
    let out = codec.encode(&ctx, &[]);
    assert!(out.is_empty(), "no inputs encode to no results");
}

#[test]
fn field_boundaries_match_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let Some(codec) = GpuVisWordCodec::new(&ctx) else {
        return;
    };
    // Sweep each field across its extremes and the sign bit while holding the
    // other at a fixed non-trivial value, so neither field bleeds into the
    // other.
    let edges = [0u32, 1, 0x7FFF_FFFF, 0x8000_0000, 0x8000_0001, 0xFFFF_FFFF];
    let mut inputs = Vec::new();
    for &d in &edges {
        inputs.push(input(d, 0x0BAD_F00D));
    }
    for &p in &edges {
        inputs.push(input(0xDEAD_BEEF, p));
    }
    let gpu = codec.encode(&ctx, &inputs);
    let want: Vec<VisWordCodec> = inputs.iter().map(expected).collect();
    assert_eq!(gpu, want, "field boundaries must match the golden");
}

#[test]
fn multi_workgroup_dispatch_matches_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let Some(codec) = GpuVisWordCodec::new(&ctx) else {
        return;
    };
    // 130 pairs span three workgroups (workgroup_size 64); vary both fields so
    // tiling and per-thread indexing are both exercised.
    let inputs: Vec<VisWordInput> = (0..130u32)
        .map(|i| input(i.wrapping_mul(2_654_435_761), i.wrapping_mul(40_503).wrapping_add(7)))
        .collect();
    let gpu = codec.encode(&ctx, &inputs);
    let want: Vec<VisWordCodec> = inputs.iter().map(expected).collect();
    assert_eq!(gpu.len(), 130, "one result per input across workgroups");
    assert_eq!(gpu, want, "tiled dispatch must match the golden");
}
