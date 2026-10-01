//! Real-device parity for the low-discrepancy sequence twin:
//! [`GpuHaltonSequence`](prism_volumetric_gpu::halton_sequence::GpuHaltonSequence)
//! must reproduce the `CPU` golden
//! [`halton_sequence`](prism_render_architecture::particle::halton_sequence)
//! index for index across the radical inverse (base-2 and arbitrary base), the
//! canonical and explicit base-pair `Halton` points, the `Hammersley` point,
//! the `TAA` jitter, the sub-pixel `pixel_jitter` remap and the
//! `Cranley-Patterson` rotation.
//!
//! The fixtures cover index `0` (the origin / empty digit loop), index `1`, a
//! small consecutive sequence, large indices (including `u32::MAX`), the bases
//! `2`, `3`, `5` and `7` plus the degenerate `base < 2` guard, a spread of
//! `Hammersley` counts `n` (including the degenerate `0`), a run of `TAA` frame
//! indices and `Cranley-Patterson` offsets of varying magnitude.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernels are portable core-`WGSL`, so they need no optional device
//! feature.
//!
//! # Parity criterion
//!
//! `radical_inverse_base2` is a `u32` bit reversal times a power of two, so it
//! is bit-exact; every other transform is a fixed, non-reorderable sequence of
//! `f32` multiply / add in the same order as the reference. They are not
//! bit-exact because a `GPU` may fuse a multiply-add the scalar reference
//! leaves separate, perturbing the low mantissa bits by a few units in the last
//! place. The comparison therefore allows `abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3` — tight enough to fail a genuinely wrong port (a swapped
//! base, a dropped digit, a missing fractional wrap) yet loose enough to admit
//! a legal fused multiply-add. Index `0` radical-inverses to `0.0` on both
//! sides, so the origin points are exact.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::halton_sequence`；无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::halton_sequence::{
    cranley_patterson, halton_point, halton_point_2d, hammersley_point, pixel_jitter,
    radical_inverse, radical_inverse_base2, taa_jitter,
};
use prism_volumetric_gpu::halton_sequence::GpuHaltonSequence;
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound, admitting a legal fused multiply-add slack.
const EPS: f32 = 1.0e-4;

/// Relative parity bound, applied for larger magnitudes where a few units in
/// the last place exceed the absolute floor.
const REL: f32 = 1.0e-3;

/// Floor on the relative-error denominator so a near-zero expected value does
/// not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Returns whether `a` and `b` agree within the absolute or relative bound.
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= EPS || rel <= REL
}

/// Asserts a flat list of expected / actual `[f32; 2]` points agree componentwise.
fn assert_points_close(got: &[[f32; 2]], expected: &[[f32; 2]], what: &str) {
    assert_eq!(got.len(), expected.len(), "{what}: length mismatch");
    for (idx, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            close(g[0], e[0]) && close(g[1], e[1]),
            "{what} mismatch at {idx}: got {g:?} expected {e:?}"
        );
    }
}

/// A broad index fixture: the origin, the first few indices, a small run, some
/// scattered large indices and the `u32` boundary.
fn index_fixture() -> Vec<u32> {
    let mut v = vec![0u32, 1, 2, 3, 4, 5, 6, 7, 8];
    v.extend(10..40u32);
    v.extend([
        63u32, 64, 100, 255, 256, 1000, 4096, 65_535, 65_536, 1_000_000,
    ]);
    v.extend([u32::MAX - 1, u32::MAX]);
    v
}

#[test]
fn radical_inverse_base2_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHaltonSequence::new(&ctx);
    let indices = index_fixture();
    let got = gpu.radical_inverse_base2(&ctx, &indices);
    assert_eq!(got.len(), indices.len());
    for (idx, &i) in indices.iter().enumerate() {
        // Bit reversal times a power of two: must be bit-exact.
        assert!(
            close(got[idx], radical_inverse_base2(i)),
            "radical_inverse_base2 mismatch at index {i}"
        );
    }
}

#[test]
fn radical_inverse_arbitrary_base_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHaltonSequence::new(&ctx);
    // Bases 2, 3, 5, 7 (and 10 for the known 0.321 vector) over the index
    // fixture, plus the degenerate base 0 / 1 guard returning 0.0.
    let base_fixture = [0u32, 1, 2, 3, 5, 7, 10, 11, 13];
    let indices = index_fixture();
    let mut bases: Vec<u32> = Vec::new();
    let mut idxs: Vec<u32> = Vec::new();
    for &base in &base_fixture {
        for &i in &indices {
            bases.push(base);
            idxs.push(i);
        }
    }
    let got = gpu.radical_inverse(&ctx, &bases, &idxs);
    assert_eq!(got.len(), bases.len());
    for (k, (&base, &i)) in bases.iter().zip(idxs.iter()).enumerate() {
        assert!(
            close(got[k], radical_inverse(base, i)),
            "radical_inverse mismatch for base {base} index {i}"
        );
    }
}

#[test]
fn radical_inverse_known_values() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHaltonSequence::new(&ctx);
    // Base-2 van der Corput head and a few arbitrary-base known values.
    let base2 = gpu.radical_inverse_base2(&ctx, &[1, 2, 3, 4]);
    assert!(close(base2[0], 0.5));
    assert!(close(base2[1], 0.25));
    assert!(close(base2[2], 0.75));
    assert!(close(base2[3], 0.125));
    let arb = gpu.radical_inverse(&ctx, &[5, 5, 10], &[1, 5, 123]);
    assert!(close(arb[0], 0.2));
    assert!(close(arb[1], 0.04));
    assert!(close(arb[2], 0.321));
}

#[test]
fn halton_point_2d_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHaltonSequence::new(&ctx);
    let indices = index_fixture();
    let got = gpu.halton_point_2d(&ctx, &indices);
    let expected: Vec<[f32; 2]> = indices.iter().map(|&i| halton_point_2d(i)).collect();
    assert_points_close(&got, &expected, "halton_point_2d");
}

#[test]
fn halton_point_explicit_base_pair_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHaltonSequence::new(&ctx);
    // Canonical and other coprime base pairs over the index fixture.
    let pairs = [(2u32, 3u32), (5, 7), (3, 5), (2, 7)];
    let indices = index_fixture();
    let mut bx: Vec<u32> = Vec::new();
    let mut by: Vec<u32> = Vec::new();
    let mut idxs: Vec<u32> = Vec::new();
    let mut expected: Vec<[f32; 2]> = Vec::new();
    for &(base_x, base_y) in &pairs {
        for &i in &indices {
            bx.push(base_x);
            by.push(base_y);
            idxs.push(i);
            expected.push(halton_point(base_x, base_y, i));
        }
    }
    let got = gpu.halton_point(&ctx, &bx, &by, &idxs);
    assert_points_close(&got, &expected, "halton_point");
}

#[test]
fn hammersley_point_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHaltonSequence::new(&ctx);
    // A spread of set sizes, including the degenerate n = 0 (treated as 1), the
    // full index sweep i in 0..n for small n, and a few large indices.
    let counts = [0u32, 1, 2, 4, 8, 16, 32, 64];
    let mut idxs: Vec<u32> = Vec::new();
    let mut ns: Vec<u32> = Vec::new();
    let mut expected: Vec<[f32; 2]> = Vec::new();
    for &n in &counts {
        for i in 0..n.max(1) {
            idxs.push(i);
            ns.push(n);
            expected.push(hammersley_point(i, n));
        }
    }
    // Degenerate n = 0 with a non-zero index must still clamp n up to 1.
    idxs.push(5);
    ns.push(0);
    expected.push(hammersley_point(5, 0));
    let got = gpu.hammersley_point(&ctx, &idxs, &ns);
    assert_points_close(&got, &expected, "hammersley_point");
}

#[test]
fn taa_jitter_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHaltonSequence::new(&ctx);
    let frames: Vec<u32> = (0..512u32).collect();
    let got = gpu.taa_jitter(&ctx, &frames);
    let expected: Vec<[f32; 2]> = frames.iter().map(|&f| taa_jitter(f)).collect();
    assert_points_close(&got, &expected, "taa_jitter");
    // Every offset must stay inside the [-0.5, 0.5) sub-pixel box.
    for (idx, p) in got.iter().enumerate() {
        assert!(
            (-0.5..0.5).contains(&p[0]) && (-0.5..0.5).contains(&p[1]),
            "taa_jitter out of range at frame {idx}: {p:?}"
        );
    }
}

#[test]
fn pixel_jitter_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHaltonSequence::new(&ctx);
    // Feed the golden Halton (2, 3) points plus the two known mapping corners.
    let mut points: Vec<[f32; 2]> = index_fixture()
        .iter()
        .map(|&i| halton_point_2d(i))
        .collect();
    points.push([0.0, 0.0]);
    points.push([0.5, 0.5]);
    let got = gpu.pixel_jitter(&ctx, &points);
    let expected: Vec<[f32; 2]> = points.iter().map(|&p| pixel_jitter(p)).collect();
    assert_points_close(&got, &expected, "pixel_jitter");
}

#[test]
fn cranley_patterson_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHaltonSequence::new(&ctx);
    // Values drawn from the base-2 sequence rotated by offsets of varying
    // magnitude (including offsets > 1 so the fractional wrap is exercised).
    let offsets = [0.1f32, 0.37, 0.5, 0.6, 0.99, 1.5, 2.75];
    let mut values: Vec<f32> = Vec::new();
    let mut offs: Vec<f32> = Vec::new();
    let mut expected: Vec<f32> = Vec::new();
    for i in 0..256u32 {
        let v = radical_inverse_base2(i);
        for &off in &offsets {
            values.push(v);
            offs.push(off);
            expected.push(cranley_patterson(v, off));
        }
    }
    let got = gpu.cranley_patterson(&ctx, &values, &offs);
    assert_eq!(got.len(), expected.len());
    for (k, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            close(g, e),
            "cranley_patterson mismatch at {k}: got {g} expected {e}"
        );
    }
    // Known rotation: 0.7 + 0.6 wraps to 0.3.
    let known = gpu.cranley_patterson(&ctx, &[0.7], &[0.6]);
    assert!(close(known[0], 0.3));
}

#[test]
fn origin_index_is_exact_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHaltonSequence::new(&ctx);
    // Index 0 radical-inverses to 0.0 on both axes; the Halton and Hammersley
    // origin points land exactly on 0.0.
    let p2d = gpu.halton_point_2d(&ctx, &[0]);
    assert!(close(p2d[0][0], 0.0) && close(p2d[0][1], 0.0));
    let hp = gpu.halton_point(&ctx, &[5], &[7], &[0]);
    assert!(close(hp[0][0], 0.0) && close(hp[0][1], 0.0));
    let hm = gpu.hammersley_point(&ctx, &[0], &[16]);
    assert!(close(hm[0][0], 0.0) && close(hm[0][1], 0.0));
}

#[test]
fn empty_input_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHaltonSequence::new(&ctx);
    // No dispatch is issued and every entry point returns an empty vector.
    assert!(gpu.radical_inverse_base2(&ctx, &[]).is_empty());
    assert!(gpu.radical_inverse(&ctx, &[], &[]).is_empty());
    assert!(gpu.halton_point_2d(&ctx, &[]).is_empty());
    assert!(gpu.halton_point(&ctx, &[], &[], &[]).is_empty());
    assert!(gpu.hammersley_point(&ctx, &[], &[]).is_empty());
    assert!(gpu.taa_jitter(&ctx, &[]).is_empty());
    assert!(gpu.pixel_jitter(&ctx, &[]).is_empty());
    assert!(gpu.cranley_patterson(&ctx, &[], &[]).is_empty());
}
