//! Real-device parity for the per-strand curly-hair rest-state twin:
//! [`GpuRestHelix`] must reproduce the `CPU` golden
//! [`build_rest_helix`](prism_render_architecture::hair::rest_helix::build_rest_helix)
//! for a batch of strands sharing one coil-segment count, covering the straight
//! (zero-radius / zero-`darboux`) strand, curly coils with isotropic and
//! anisotropic bending compliance, the single-segment (no interior joint) case,
//! the segment clamp, and the empty-batch no-op.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The construction is trig-free closed-form geometry (both sides restrict
//! themselves to `sqrt`, `dot`, `cross` and a unit-complex multiply) evaluated in
//! the same order, so `CPU` and `GPU` diverge only through legal
//! fused-multiply-add contraction. Parity is asserted per component to within
//! `abs_diff < 1e-4` or `rel_diff < 1e-3` — tight enough to fail a genuinely
//! wrong port (a dropped segment, a wrong basis, a missing curvature guard),
//! loose enough to admit the fma contraction. The curly strands also assert a
//! non-zero `darboux` magnitude and the expected anisotropic compliance so a
//! no-op (all-zero) kernel cannot pass. All inputs use pre-normalized complex
//! rotation steps built from exact rational triples (no `sin`/`cos`).
//!
//! Provenance: standard discrete-elastic-rod helical rest construction; no
//! Unreal Engine source or derived code.

use prism_hair_gpu::rest_helix::{GpuRestHelix, GpuRestHelixOut, GpuRestHelixStrand};
use prism_hair_gpu::GpuContext;
use prism_render_architecture::hair::rest_helix::{build_rest_helix, CurlParams, RestHelix, Vec3};

/// Asserts a single scalar matches within the documented fma tolerance.
fn assert_close(got: f32, expected: f32, label: &str) {
    let abs_diff = (got - expected).abs();
    let rel_diff = abs_diff / expected.abs().max(1e-6);
    assert!(
        abs_diff < 1e-4 || rel_diff < 1e-3,
        "{label}: gpu {got}, cpu {expected} (abs {abs_diff}, rel {rel_diff})"
    );
}

/// Asserts a `[f32; 3]` matches its golden `Vec3` component-wise.
fn assert_vec_close(got: [f32; 3], expected: Vec3, label: &str) {
    assert_close(got[0], expected.x, &format!("{label}.x"));
    assert_close(got[1], expected.y, &format!("{label}.y"));
    assert_close(got[2], expected.z, &format!("{label}.z"));
}

/// Asserts one `GPU` strand output equals the `CPU` golden rest helix.
fn assert_strand_parity(got: &GpuRestHelixOut, golden: &RestHelix, label: &str) {
    assert_eq!(
        got.positions.len(),
        golden.positions.len(),
        "{label}: position count"
    );
    for (i, (g, c)) in got.positions.iter().zip(&golden.positions).enumerate() {
        assert_vec_close(*g, *c, &format!("{label} position[{i}]"));
    }
    assert_eq!(
        got.rest_lengths.len(),
        golden.rest_lengths.len(),
        "{label}: rest-length count"
    );
    for (i, (g, c)) in got
        .rest_lengths
        .iter()
        .zip(&golden.rest_lengths)
        .enumerate()
    {
        assert_close(*g, *c, &format!("{label} rest_length[{i}]"));
    }
    assert_eq!(
        got.rest_darboux.len(),
        golden.rest_darboux.len(),
        "{label}: darboux count"
    );
    for (i, (g, c)) in got
        .rest_darboux
        .iter()
        .zip(&golden.rest_darboux)
        .enumerate()
    {
        assert_vec_close(*g, *c, &format!("{label} darboux[{i}]"));
    }
    // Both compliance arrays are constant per strand in the golden; compare the
    // single `GPU` scalar against the first entry.
    assert_close(
        got.tangential_compliance,
        golden.tangential_compliance[0],
        &format!("{label} tangential_compliance"),
    );
    assert_close(
        got.normal_compliance,
        golden.normal_compliance[0],
        &format!("{label} normal_compliance"),
    );
}

/// The strand roots used by the mixed-batch parity test, paired with their
/// golden `CurlParams`. A `(3,4,5)` right triangle gives the exact unit complex
/// step `(0.6, 0.8)`; a `(1, sqrt3)/2` triple gives `(0.5, 0.8660254)`.
fn mixed_batch() -> Vec<(GpuRestHelixStrand, Vec3, CurlParams)> {
    vec![
        // Straight hair: zero radius => a line along the tangent, zero darboux.
        (
            GpuRestHelixStrand {
                root: [0.0, 0.0, 0.0],
                root_tangent: [0.0, 1.0, 0.0],
                radius: 0.0,
                pitch_per_segment: 0.5,
                rot_cos_step: 1.0,
                rot_sin_step: 0.0,
                bend_stiffness_ratio: 1.0,
            },
            Vec3::new(0.0, 1.0, 0.0),
            CurlParams::new(0.0, 0.5, 8, 1.0, 0.0, 1.0),
        ),
        // Anisotropic curl: radius 2, pitch 1, 60-degree exact step, ratio 3.
        (
            GpuRestHelixStrand {
                root: [1.0, -2.0, 0.5],
                root_tangent: [0.0, 0.0, 1.0],
                radius: 2.0,
                pitch_per_segment: 1.0,
                rot_cos_step: 0.5,
                rot_sin_step: 0.866_025_4,
                bend_stiffness_ratio: 3.0,
            },
            Vec3::new(0.0, 0.0, 1.0),
            CurlParams::new(2.0, 1.0, 8, 0.5, 0.866_025_4, 3.0),
        ),
        // Isotropic curl along a tilted axis: 3-4-5 step, ratio 1.
        (
            GpuRestHelixStrand {
                root: [-3.0, 4.0, 2.0],
                root_tangent: [1.0, 2.0, -2.0],
                radius: 1.5,
                pitch_per_segment: 0.75,
                rot_cos_step: 0.6,
                rot_sin_step: 0.8,
                bend_stiffness_ratio: 1.0,
            },
            Vec3::new(1.0, 2.0, -2.0),
            CurlParams::new(1.5, 0.75, 8, 0.6, 0.8, 1.0),
        ),
    ]
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_mixed_strands_match_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping rest-helix parity: no wgpu adapter on this host");
        return;
    };

    let segments = 8usize;
    let batch = mixed_batch();
    let strands: Vec<GpuRestHelixStrand> = batch.iter().map(|(s, _, _)| *s).collect();

    let helix = GpuRestHelix::new(&ctx);
    let gpu = helix.eval(&ctx, &strands, segments);
    assert_eq!(gpu.len(), batch.len(), "one output per strand");

    for (i, (_, tangent, params)) in batch.iter().enumerate() {
        let root = Vec3::new(strands[i].root[0], strands[i].root[1], strands[i].root[2]);
        let golden = build_rest_helix(root, *tangent, *params);
        assert_strand_parity(&gpu[i], &golden, &format!("strand {i}"));
    }

    // Straight strand: every darboux vanishes.
    for (j, d) in gpu[0].rest_darboux.iter().enumerate() {
        let mag = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        assert!(mag < 1e-4, "straight strand darboux[{j}] = {mag}");
    }
    // Curly strands bend: at least one darboux has real magnitude, defeating a
    // no-op kernel.
    for (s, out) in gpu.iter().enumerate().take(3).skip(1) {
        let max_mag = out
            .rest_darboux
            .iter()
            .map(|d| (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt())
            .fold(0.0f32, f32::max);
        assert!(
            max_mag > 1e-3,
            "curly strand {s} darboux magnitude {max_mag}"
        );
    }
    // Anisotropic strand (ratio 3): tangential compliance 1/3, normal 1.
    assert_close(
        gpu[1].tangential_compliance,
        1.0 / 3.0,
        "strand 1 tangential",
    );
    assert_close(gpu[1].normal_compliance, 1.0, "strand 1 normal");
    // Isotropic strand (ratio 1): both compliances equal 1.
    assert_close(gpu[2].tangential_compliance, 1.0, "strand 2 tangential");
    assert_close(gpu[2].normal_compliance, 1.0, "strand 2 normal");
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_single_segment_has_no_interior_joint() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping rest-helix parity: no wgpu adapter on this host");
        return;
    };

    let strands = [GpuRestHelixStrand {
        root: [0.0, 0.0, 0.0],
        root_tangent: [0.0, 1.0, 0.0],
        radius: 2.0,
        pitch_per_segment: 1.0,
        rot_cos_step: 0.6,
        rot_sin_step: 0.8,
        bend_stiffness_ratio: 2.0,
    }];
    let helix = GpuRestHelix::new(&ctx);
    let gpu = helix.eval(&ctx, &strands, 1);
    assert_eq!(gpu.len(), 1);

    let golden = build_rest_helix(
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        CurlParams::new(2.0, 1.0, 1, 0.6, 0.8, 2.0),
    );
    assert_strand_parity(&gpu[0], &golden, "single-segment strand");
    assert_eq!(gpu[0].positions.len(), 2, "two vertices");
    assert_eq!(gpu[0].rest_lengths.len(), 1, "one segment");
    assert!(gpu[0].rest_darboux.is_empty(), "no interior joint");
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_segment_count_is_clamped() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping rest-helix parity: no wgpu adapter on this host");
        return;
    };

    // A zero segment count clamps up to one segment, matching the golden.
    let strands = [GpuRestHelixStrand {
        root: [0.0, 0.0, 0.0],
        root_tangent: [0.0, 1.0, 0.0],
        radius: 1.0,
        pitch_per_segment: 1.0,
        rot_cos_step: 0.6,
        rot_sin_step: 0.8,
        bend_stiffness_ratio: 1.0,
    }];
    let helix = GpuRestHelix::new(&ctx);
    let gpu = helix.eval(&ctx, &strands, 0);
    assert_eq!(gpu.len(), 1);
    assert_eq!(gpu[0].positions.len(), 2, "zero segments clamp to one");
    assert_eq!(gpu[0].rest_lengths.len(), 1);
    assert!(gpu[0].rest_darboux.is_empty());

    let golden = build_rest_helix(
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        CurlParams::new(1.0, 1.0, 0, 0.6, 0.8, 1.0),
    );
    assert_strand_parity(&gpu[0], &golden, "clamped strand");
}

#[test]
fn empty_batch_is_a_no_op() {
    // The empty-batch guard returns before touching the GPU, so it runs and
    // asserts even on hosts without a wgpu adapter.
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let helix = GpuRestHelix::new(&ctx);
    assert!(
        helix.eval(&ctx, &[], 8).is_empty(),
        "empty strand batch => empty"
    );
}
