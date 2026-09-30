//! Real-device parity for the froxel injection-weight twin:
//! [`GpuFroxelInjection`] must reproduce the `CPU` golden
//! [`froxel_injection_weight`](prism_render_architecture::volumetric::fog::froxel_injection_weight)
//! across a spread of depth-slice indices and slice counts.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The kernel contains no transcendental call — an unsigned `min`, a division
//! and a saturating clamp — so `CPU` and `GPU` evaluate the same closed-form
//! algebra on the same unsigned-integer inputs. Values are asserted to within
//! `abs_diff < 1e-6` or `rel_diff < 1e-5` — tight enough to fail a wrong port
//! (a dropped clamp, an off-by-one slice count). The scenes also assert the
//! documented `[0, 1]` range, that the nearest slice gets full weight, that a
//! zero slice count yields `0`, that an out-of-range index clamps to `0`, and
//! that the weight decreases monotonically with depth, so a degenerate kernel
//! could not pass.
//!
//! Provenance: standard additive froxel-injection weighting; no Unreal Engine
//! source or derived code.

use prism_render_architecture::volumetric::fog::froxel_injection_weight;
use prism_volumetric_gpu::{FroxelInjectionQuery, GpuContext, GpuFroxelInjection};

/// Asserts every `gpu` value matches the `CPU` golden to within the documented
/// tolerance and stays inside the unit range.
fn assert_parity(queries: &[FroxelInjectionQuery], gpu: &[f32]) {
    assert_eq!(gpu.len(), queries.len(), "one weight per query");
    for (i, q) in queries.iter().enumerate() {
        let exp = froxel_injection_weight(q.depth_slice, q.slice_count);
        let got = gpu[i];
        let abs_diff = (got - exp).abs();
        let rel_diff = abs_diff / exp.abs().max(1e-6);
        assert!(
            abs_diff < 1e-6 || rel_diff < 1e-5,
            "froxel injection weight mismatch for query {i} ({q:?}): gpu {got}, cpu {exp} \
             (abs {abs_diff}, rel {rel_diff})"
        );
        assert!(
            (0.0..=1.0).contains(&got),
            "gpu froxel injection weight must stay in the unit range: {got}"
        );
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_froxel_injection_matches_cpu_golden_across_queries() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping froxel injection parity: no wgpu adapter on this host");
        return;
    };
    let gpu_inj = GpuFroxelInjection::new(&ctx);

    // A deterministic spread: the nearest slice, the midpoint, the farthest
    // slice, an out-of-range index that must clamp to zero weight, and a zero
    // slice count that must yield zero.
    let mut queries: Vec<FroxelInjectionQuery> = vec![
        FroxelInjectionQuery {
            depth_slice: 0,
            slice_count: 16,
        },
        FroxelInjectionQuery {
            depth_slice: 8,
            slice_count: 16,
        },
        FroxelInjectionQuery {
            depth_slice: 16,
            slice_count: 16,
        },
        FroxelInjectionQuery {
            depth_slice: 40,
            slice_count: 16,
        },
        FroxelInjectionQuery {
            depth_slice: 5,
            slice_count: 0,
        },
    ];
    // A deterministic depth sweep across a fixed slice count, plus the same
    // sweep at a different slice count, to exercise the integer division.
    for k in 0..=64 {
        queries.push(FroxelInjectionQuery {
            depth_slice: k,
            slice_count: 64,
        });
    }
    for k in 0..=32 {
        queries.push(FroxelInjectionQuery {
            depth_slice: k,
            slice_count: 32,
        });
    }

    let gpu = gpu_inj.eval(&ctx, &queries);
    assert_parity(&queries, &gpu);

    assert!(
        (gpu[0] - 1.0).abs() < 1e-6,
        "the nearest slice gets full weight: {}",
        gpu[0]
    );
    assert!(
        (gpu[1] - 0.5).abs() < 1e-6,
        "the midpoint slice gets half weight: {}",
        gpu[1]
    );
    assert!(
        gpu[2].abs() < 1e-6,
        "the farthest slice gets zero weight: {}",
        gpu[2]
    );
    assert!(
        gpu[3].abs() < 1e-6,
        "an out-of-range index clamps to zero weight: {}",
        gpu[3]
    );
    assert!(
        gpu[4].abs() < 1e-6,
        "a zero slice count yields zero weight: {}",
        gpu[4]
    );
}

#[test]
fn gpu_froxel_injection_decreases_monotonically_with_depth() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu_inj = GpuFroxelInjection::new(&ctx);

    // At a fixed slice count the injection weight falls monotonically from full
    // weight at the nearest slice to zero at (and beyond) the farthest slice.
    let slice_count = 48u32;
    let queries: Vec<FroxelInjectionQuery> = (0..=slice_count)
        .map(|depth_slice| FroxelInjectionQuery {
            depth_slice,
            slice_count,
        })
        .collect();

    let gpu = gpu_inj.eval(&ctx, &queries);
    assert_parity(&queries, &gpu);

    for w in gpu.windows(2) {
        assert!(
            w[1] <= w[0] + 1e-6,
            "weight must be monotone non-increasing with depth: {} then {}",
            w[0],
            w[1]
        );
    }
    assert!(
        (gpu[0] - 1.0).abs() < 1e-6,
        "the nearest slice starts at full weight"
    );
    assert!(
        gpu[gpu.len() - 1].abs() < 1e-6,
        "the farthest slice ends at zero weight"
    );
}

#[test]
fn empty_queries_yields_no_results() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu_inj = GpuFroxelInjection::new(&ctx);
    let out = gpu_inj.eval(&ctx, &[]);
    assert!(out.is_empty(), "an empty query slice yields no values");
}
