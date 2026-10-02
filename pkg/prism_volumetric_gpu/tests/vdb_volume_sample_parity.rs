//! Real-device parity for the sparse-`VDB` *gradient / surface-normal* twin:
//! [`GpuVdbVolumeSample`] must reproduce the `CPU` golden gradient
//! ([`sample_vdb_gradient`](prism_render_architecture::particle::vdb_volume_sample::sample_vdb_gradient)
//! / [`VdbTree::sample_gradient`](prism_render_architecture::particle::vdb_volume_sample::VdbTree::sample_gradient),
//! design `docs/prism_particle_engine_design_zh.md` §8.3) for the raw
//! central-difference gradient and the guard-normalized unit gradient across a
//! linear ramp, a random sparse field, a flat active region and an empty tree.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! upload-dispatch-and-readback on any real device. The kernel is portable
//! core-`WGSL` (integer shifts, masks and compares plus `floor`, multiply/add
//! and the one allowed `sqrt`), so it needs no optional device feature and runs
//! unmodified on Metal, Vulkan and DX12.
//!
//! # Parity criterion
//!
//! Each corner lookup is an exact integer tree descent and the six-tap central
//! difference is pure multiply/add, so `CPU` and `GPU` evaluate the identical
//! closed-form algebra. Non-degenerate gradient components are asserted to
//! within `abs_diff <= 1e-4` or `rel_diff <= 1e-3` (`REL_FLOOR = 1e-6`) — far
//! tighter than any meaningful gradient difference, yet enough to fail a wrong
//! port (a swapped step, a dropped axis, a misordered corner weight). The flat
//! and empty scenes assert an exact all-zero gradient on both backends,
//! bit-compared, so the degenerate guard is checked for an identical outcome,
//! and several scenes assert a clearly non-zero component so a degenerate
//! constant kernel could not pass.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::vdb_volume_sample`；无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::vdb_volume_sample::VdbTree;
use prism_volumetric_gpu::vdb_volume_sample::cpu_reference;
use prism_volumetric_gpu::{
    GpuContext, GpuVdbVolumeSample, VdbVolumeSampleQuery, VdbVolumeSampleResult,
};

/// Absolute parity bound on a gradient component.
const ABS_EPS: f32 = 1.0e-4;

/// Relative parity bound on a gradient component.
const REL_EPS: f32 = 1.0e-3;

/// Floor on the relative-error denominator so a near-zero expected value does
/// not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Per-component magnitude proving a gradient is clearly non-degenerate (far
/// above the `GRAD_EPS_SQ = 1e-12` guard band), so a constant kernel can not
/// pass.
const DISTINCT: f32 = 1.0e-2;

/// Returns whether `a` and `b` agree within the absolute or relative bound.
fn approx(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= ABS_EPS || rel <= REL_EPS
}

/// Returns the three gradient components of a result.
fn components(result: &VdbVolumeSampleResult) -> [f32; 3] {
    match result {
        VdbVolumeSampleResult::Gradient(g) => *g,
    }
}

/// Returns whether a `GPU` gradient matches the `CPU` reference component-wise
/// within the continuous-field bound.
fn results_match(got: &VdbVolumeSampleResult, want: &VdbVolumeSampleResult) -> bool {
    let a = components(got);
    let b = components(want);
    a.iter().zip(b.iter()).all(|(x, y)| approx(*x, *y))
}

/// Returns whether a gradient is bit-exactly the positive-zero vector on this
/// backend, so a degenerate region can be compared for an identical outcome
/// rather than a merely approximate one.
fn is_exact_zero(result: &VdbVolumeSampleResult) -> bool {
    components(result).iter().all(|c| c.to_bits() == 0)
}

/// Returns the largest absolute gradient component, a non-degeneracy witness.
fn max_abs(result: &VdbVolumeSampleResult) -> f32 {
    components(result)
        .iter()
        .fold(0.0_f32, |acc, c| acc.max(c.abs()))
}

/// Builds a tree, failing loudly if the root dimensions are degenerate (every
/// test picks a valid non-zero extent).
fn tree_with(root_dims: [u32; 3], background: f32) -> VdbTree {
    VdbTree::new(root_dims, background).expect("non-zero root dimensions build a tree")
}

/// Evaluates a single query on device and asserts it matches the `CPU` golden.
fn check(gpu: &GpuVdbVolumeSample, ctx: &GpuContext, tree: &VdbTree, query: VdbVolumeSampleQuery) {
    let want = cpu_reference(tree, &query);
    let got = gpu.evaluate(ctx, tree, std::slice::from_ref(&query));
    assert_eq!(got.len(), 1, "one query yields one result");
    assert!(
        results_match(&got[0], &want),
        "GPU gradient {:?} must match CPU {:?} for {:?}",
        got[0],
        want,
        query
    );
}

/// A tiny integer linear-congruential generator; only integer and shift work,
/// so no transcendental appears. Returns a value in `[0, 1)`.
fn lcg(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let bits = (*state >> 40) as u32;
    (bits & 0x00ff_ffff) as f32 / 16_777_216.0
}

/// Draws a value in `[lo, hi)` from the generator.
fn rand_range(state: &mut u64, lo: f32, hi: f32) -> f32 {
    lo + lcg(state) * (hi - lo)
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_batch_is_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping vdb-volume-sample parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuVdbVolumeSample::new(&ctx);

    // A sparse tree with no queries issues no dispatch (a storage buffer can
    // not be zero-sized) and returns an empty vector.
    let tree = tree_with([1, 1, 1], 0.0);
    let got = gpu.evaluate(&ctx, &tree, &[]);
    assert!(got.is_empty(), "an empty batch yields an empty result");
}

#[test]
fn raw_gradient_matches_ramp() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuVdbVolumeSample::new(&ctx);

    // A linear ramp along +X on the line y=5, z=5: voxel (x,5,5) holds `x`.
    // The trilinear density rises one unit per voxel along X, so the raw
    // central-difference gradient is clearly non-zero along X and zero along
    // the flat Y/Z axes — a clean, well-conditioned check.
    let mut tree = tree_with([1, 1, 1], 0.0);
    for x in 0..12 {
        assert!(
            tree.set_voxel([x, 5, 5], x as f32),
            "ramp voxel {x} is in domain"
        );
    }

    let positions = [[5.0, 5.0, 5.0], [6.0, 5.0, 5.0], [8.0, 5.0, 5.0]];
    for pos in positions {
        check(&gpu, &ctx, &tree, VdbVolumeSampleQuery::RawGradient { pos });
    }

    // Non-degeneracy: the ramp midpoint has a clearly non-zero gradient.
    let mid = cpu_reference(
        &tree,
        &VdbVolumeSampleQuery::RawGradient { pos: positions[0] },
    );
    assert!(
        max_abs(&mid) > DISTINCT,
        "the ramp gradient must be clearly non-zero, got {mid:?}"
    );
}

#[test]
fn unit_gradient_matches_field() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuVdbVolumeSample::new(&ctx);

    // A diagonal ramp over a filled block: voxel (x,y,z) holds `x + y + z`, so
    // the raw gradient points along (1,1,1) and the unit gradient is its
    // normalization — a non-axis-aligned surface normal that exercises the
    // guarded `1/sqrt(len^2)` on every component at once.
    let mut tree = tree_with([1, 1, 1], 0.0);
    for z in 0..16 {
        for y in 0..16 {
            for x in 0..16 {
                let value = (x + y + z) as f32;
                assert!(
                    tree.set_voxel([x, y, z], value),
                    "field voxel ({x},{y},{z}) is in domain"
                );
            }
        }
    }

    let positions = [[6.0, 6.0, 6.0], [8.0, 7.0, 9.0], [10.0, 5.0, 11.0]];
    for pos in positions {
        check(
            &gpu,
            &ctx,
            &tree,
            VdbVolumeSampleQuery::UnitGradient { pos },
        );
    }

    // Non-degeneracy: the normalized gradient has a clearly non-zero component.
    let unit = cpu_reference(
        &tree,
        &VdbVolumeSampleQuery::UnitGradient { pos: positions[0] },
    );
    assert!(
        max_abs(&unit) > DISTINCT,
        "the unit gradient must be clearly non-zero, got {unit:?}"
    );
}

#[test]
fn degenerate_region_is_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuVdbVolumeSample::new(&ctx);

    // Two flat scenes: an empty tree (uniform background everywhere) and a
    // block filled with a single constant value. Both have an identically zero
    // central difference, so the raw gradient is zero and the unit gradient
    // falls into the `len^2 <= GRAD_EPS_SQ` guard, yielding the exact zero
    // vector on both backends — compared bit-for-bit.
    let empty = tree_with([1, 1, 1], 0.4);

    let mut flat = tree_with([1, 1, 1], 0.0);
    for z in 0..16 {
        for y in 0..16 {
            for x in 0..16 {
                assert!(
                    flat.set_voxel([x, y, z], 0.7),
                    "flat voxel ({x},{y},{z}) is in domain"
                );
            }
        }
    }

    let cases = [
        (&empty, [10.0_f32, 10.0, 10.0]),
        (&empty, [5.5, 7.5, 3.5]),
        (&flat, [8.0, 8.0, 8.0]),
        (&flat, [6.5, 9.5, 7.5]),
    ];
    for (tree, pos) in cases {
        for query in [
            VdbVolumeSampleQuery::RawGradient { pos },
            VdbVolumeSampleQuery::UnitGradient { pos },
        ] {
            let want = cpu_reference(tree, &query);
            assert!(
                is_exact_zero(&want),
                "the CPU gradient must be exactly zero in a flat region for {query:?}, got {want:?}"
            );
            let got = gpu.evaluate(&ctx, tree, std::slice::from_ref(&query));
            assert_eq!(got.len(), 1, "one query yields one result");
            assert!(
                is_exact_zero(&got[0]),
                "the GPU gradient must be exactly zero in a flat region for {query:?}, got {:?}",
                got[0]
            );
        }
    }
}

#[test]
fn random_batch_matches_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuVdbVolumeSample::new(&ctx);

    // A random sparse field: a 16^3 block of varied values, so interior
    // gradients are well-conditioned (squared length of order one, far above
    // the 1e-12 guard). Query points stay inside [5, 11]^3, keeping every
    // +/-GRAD_STEP(0.5) and +/-1 trilinear neighbour inside the filled block.
    let mut state: u64 = 0x1234_5678_9abc_def0;
    let mut tree = tree_with([1, 1, 1], 0.0);
    for z in 0..16 {
        for y in 0..16 {
            for x in 0..16 {
                let value = rand_range(&mut state, 0.1, 1.0);
                assert!(
                    tree.set_voxel([x, y, z], value),
                    "random voxel ({x},{y},{z}) is in domain"
                );
            }
        }
    }

    // Build a mixed batch of raw and unit queries at random interior points,
    // rejecting any point whose reference gradient is too small to stay clear
    // of the degenerate classification band.
    let mut queries: Vec<VdbVolumeSampleQuery> = Vec::new();
    let mut rounds = 0;
    while queries.len() < 48 && rounds < 4096 {
        rounds += 1;
        let pos = [
            rand_range(&mut state, 5.0, 11.0),
            rand_range(&mut state, 5.0, 11.0),
            rand_range(&mut state, 5.0, 11.0),
        ];
        let raw = cpu_reference(&tree, &VdbVolumeSampleQuery::RawGradient { pos });
        if max_abs(&raw) <= DISTINCT {
            continue;
        }
        queries.push(VdbVolumeSampleQuery::RawGradient { pos });
        queries.push(VdbVolumeSampleQuery::UnitGradient { pos });
    }
    assert!(
        queries.len() >= 2,
        "the random field must yield well-conditioned sample points"
    );

    let got = gpu.evaluate(&ctx, &tree, &queries);
    assert_eq!(got.len(), queries.len(), "one result per query");

    let mut saw_nonzero = false;
    for (query, result) in queries.iter().zip(got.iter()) {
        let want = cpu_reference(&tree, query);
        assert!(
            results_match(result, &want),
            "GPU gradient {result:?} must match CPU {want:?} for {query:?}"
        );
        if max_abs(result) > DISTINCT {
            saw_nonzero = true;
        }
    }
    assert!(
        saw_nonzero,
        "at least one random query must produce a clearly non-zero gradient"
    );
}
