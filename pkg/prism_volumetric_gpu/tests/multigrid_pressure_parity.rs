//! Real-device parity for the geometric-multigrid pressure-projection per-cell
//! primitive twin:
//! [`GpuMultigridPressure`](prism_volumetric_gpu::multigrid_pressure::GpuMultigridPressure)
//! must reproduce the `CPU` golden
//! [`multigrid_pressure`](prism_render_architecture::particle::multigrid_pressure)
//! lane for lane across every operation code the single `solve` kernel
//! dispatches — the forward-difference `divergence_forward`, the wall-aware
//! backward-difference `gradient_backward`, one weighted-`Jacobi` sweep
//! `jacobi_smooth`, the per-cell `level_residual_l2` term, the stencil
//! `diagonal`, the cell-centred `axis_contributors` prolongation stencil, the
//! trilinear `prolong` fold and the `restrict` scale `1 / 2^k` from `pow2_f32`.
//!
//! The named fixtures cover an empty batch; the forward divergence with both
//! in-grid and out-of-grid forward faces; the backward gradient with dropped
//! low faces; one `Jacobi` relaxation and one residual term under both wall
//! models; the standalone diagonal for a `Dirichlet` wall and several live
//! `Neumann` counts clear of the lone-cell edge; the cell-centred prolongation
//! stencil for an interior index, both grid edges and the un-coarsened
//! degenerate; the trilinear fold across axis-length combinations; the
//! restriction scale over `0..=3` coarsened axes; a mixed-variant batch that
//! pins input ordering; and a large pseudo-random batch over the tolerance-safe
//! operations compared lane for lane.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Every twinned routine is fixed closed-form algebra (multiply-add plus one
//! guarded reciprocal, and an integer-doubling `pow2_f32`), so `CPU` and `GPU`
//! evaluate the same expression. They are not bit-exact: a `GPU` may fuse a
//! multiply-add the scalar reference leaves separate, perturbing the low
//! mantissa bits by a few units in the last place. The comparison therefore
//! allows `abs_diff <= 1e-4` or `rel_diff <= 1e-3` on continuous values and
//! asserts an *exact* match on the discrete stencil-index and stencil-length
//! codes. Every fixture is placed clear of the branch-critical thresholds — the
//! lone-cell `Neumann` diagonal (`live == 0`), the `axis_contributors` grid
//! edges and a positive `inv_h2` — so no legal `ULP` perturbation can flip a
//! discrete verdict.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::multigrid_pressure`；
//! standard geometric-multigrid pressure-projection primitives (forward
//! divergence, wall-aware backward gradient, weighted-`Jacobi` relaxation,
//! 7-point `Laplacian` residual, full-weighting restriction and trilinear
//! prolongation stencils); no third-party engine source or derived code.

use prism_render_architecture::particle::multigrid_pressure::PressureBoundary;
use prism_volumetric_gpu::multigrid_pressure::{
    cpu_reference, GpuMultigridPressure, MultigridPressureQuery, MultigridPressureResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound on continuous values. A `GPU` may fuse a multiply-add
/// the scalar reference leaves separate, perturbing the low mantissa bits by a
/// few units in the last place; `1e-4` admits that legal slack while still
/// failing a genuinely wrong port.
const EPS: f32 = 1.0e-4;

/// Relative parity bound, applied for larger magnitudes where a few units in the
/// last place exceed the absolute floor.
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

/// Asserts strict lane-for-lane parity of one `GPU` result against the `CPU`
/// golden: the stencil index and length codes match exactly, and every
/// continuous value matches within tolerance.
fn assert_lane(lane: usize, got: &MultigridPressureResult, want: &MultigridPressureResult) {
    match (got, want) {
        (
            MultigridPressureResult::Scalar { value: sg },
            MultigridPressureResult::Scalar { value: sc },
        ) => {
            assert!(close(*sg, *sc), "lane {lane}: scalar gpu {sg} vs cpu {sc}");
        }
        (MultigridPressureResult::Vector { v: vg }, MultigridPressureResult::Vector { v: vc }) => {
            for ch in 0..3 {
                assert!(
                    close(vg[ch], vc[ch]),
                    "lane {lane}: vector[{ch}] gpu {vg:?} vs cpu {vc:?}"
                );
            }
        }
        (
            MultigridPressureResult::Axis {
                idx: ig,
                weight: wg,
                len: lg,
            },
            MultigridPressureResult::Axis {
                idx: ic,
                weight: wc,
                len: lc,
            },
        ) => {
            assert_eq!(lg, lc, "lane {lane}: axis len gpu {lg} vs cpu {lc}");
            // Only the first `len` indices/weights are meaningful; compare those.
            let live = *lc as usize;
            for k in 0..live {
                assert_eq!(
                    ig[k], ic[k],
                    "lane {lane}: axis idx[{k}] gpu {ig:?} vs cpu {ic:?}"
                );
                assert!(
                    close(wg[k], wc[k]),
                    "lane {lane}: axis weight[{k}] gpu {wg:?} vs cpu {wc:?}"
                );
            }
        }
        _ => panic!("lane {lane}: result variant mismatch gpu {got:?} vs cpu {want:?}"),
    }
}

/// Runs the `GPU` dispatch and asserts strict parity against the `CPU` golden
/// for every lane; returns the `GPU` verdicts for extra per-test assertions.
fn check(
    ctx: &GpuContext,
    gpu: &GpuMultigridPressure,
    queries: &[MultigridPressureQuery],
) -> Vec<MultigridPressureResult> {
    let got = gpu.eval(ctx, queries);
    assert_eq!(got.len(), queries.len(), "one result per query");
    for (lane, (g, q)) in got.iter().zip(queries.iter()).enumerate() {
        assert_lane(lane, g, &cpu_reference(q));
    }
    got
}

/// A tiny integer linear-congruential generator; only integer and divide work,
/// so no transcendental appears. Returns a value in `[0, 1)`.
fn lcg(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let bits = (*state >> 40) as u32;
    (bits & 0x00ff_ffff) as f32 / 16_777_216.0
}

/// A pseudo-random `f32` in `[-1, 1)`, derived from the integer `lcg`.
fn signed(state: &mut u64) -> f32 {
    lcg(state) * 2.0 - 1.0
}

/// A pseudo-random boolean, derived from the integer `lcg`.
fn flag(state: &mut u64) -> bool {
    lcg(state) < 0.5
}

#[test]
fn empty_batch_is_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMultigridPressure::new(&ctx);
    let got = gpu.eval(&ctx, &[]);
    assert!(got.is_empty(), "an empty batch yields an empty result");
}

#[test]
fn divergence_forward_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMultigridPressure::new(&ctx);
    let queries = [
        // All forward faces inside the grid.
        MultigridPressureQuery::DivergenceForward {
            here: [0.3, -0.4, 0.2],
            forward: [0.7, 0.1, -0.5],
            forward_in_grid: [true, true, true],
        },
        // High-x and high-z faces outside the grid read zero.
        MultigridPressureQuery::DivergenceForward {
            here: [0.25, 0.6, -0.1],
            forward: [0.9, -0.3, 0.8],
            forward_in_grid: [false, true, false],
        },
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn gradient_backward_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMultigridPressure::new(&ctx);
    let queries = [
        // All backward faces inside the grid.
        MultigridPressureQuery::GradientBackward {
            here: 0.75,
            backward: [0.2, -0.1, 0.5],
            backward_in_grid: [true, true, true],
        },
        // Low-y face dropped (zero gradient wall).
        MultigridPressureQuery::GradientBackward {
            here: -0.4,
            backward: [0.6, 0.3, -0.2],
            backward_in_grid: [true, false, true],
        },
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn jacobi_cell_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMultigridPressure::new(&ctx);
    let queries = [
        // Dirichlet: full six-face diagonal.
        MultigridPressureQuery::JacobiCell {
            center: 0.5,
            neighbors: [0.1, 0.2, -0.3, 0.4, -0.2, 0.6],
            neighbor_in_grid: [true, true, true, true, true, true],
            rhs: 0.25,
            inv_h2: 4.0,
            omega: 0.8,
            boundary: PressureBoundary::Dirichlet,
        },
        // Neumann: four live neighbours (clear of the lone-cell edge).
        MultigridPressureQuery::JacobiCell {
            center: -0.2,
            neighbors: [0.3, -0.5, 0.7, 0.0, 0.2, -0.4],
            neighbor_in_grid: [true, false, true, false, true, true],
            rhs: -0.15,
            inv_h2: 2.5,
            omega: 0.6,
            boundary: PressureBoundary::Neumann,
        },
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn residual_cell_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMultigridPressure::new(&ctx);
    let queries = [
        MultigridPressureQuery::ResidualCell {
            center: 0.4,
            neighbors: [0.2, -0.1, 0.5, 0.3, -0.6, 0.1],
            neighbor_in_grid: [true, true, true, true, true, true],
            rhs: 0.35,
            inv_h2: 3.0,
            boundary: PressureBoundary::Dirichlet,
        },
        MultigridPressureQuery::ResidualCell {
            center: -0.3,
            neighbors: [0.1, 0.0, -0.2, 0.0, 0.4, 0.5],
            neighbor_in_grid: [true, false, true, false, true, true],
            rhs: 0.1,
            inv_h2: 1.5,
            boundary: PressureBoundary::Neumann,
        },
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn diagonal_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMultigridPressure::new(&ctx);
    let queries = [
        MultigridPressureQuery::Diagonal {
            boundary: PressureBoundary::Dirichlet,
            live: 6,
        },
        // Several live Neumann counts, each clear of the lone-cell `live == 0`
        // edge where the diagonal clamps to one.
        MultigridPressureQuery::Diagonal {
            boundary: PressureBoundary::Neumann,
            live: 1,
        },
        MultigridPressureQuery::Diagonal {
            boundary: PressureBoundary::Neumann,
            live: 3,
        },
        MultigridPressureQuery::Diagonal {
            boundary: PressureBoundary::Neumann,
            live: 6,
        },
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn axis_contributors_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMultigridPressure::new(&ctx);
    let queries = [
        // Interior even index: parent 2, far 1, two contributors.
        MultigridPressureQuery::AxisContributors {
            fine_index: 4,
            coarse_n: 8,
        },
        // Interior odd index: parent 2, far 3, two contributors.
        MultigridPressureQuery::AxisContributors {
            fine_index: 5,
            coarse_n: 8,
        },
        // Low edge even index: parent 0, far quarter folds back (one
        // contributor).
        MultigridPressureQuery::AxisContributors {
            fine_index: 0,
            coarse_n: 8,
        },
        // High edge odd index: parent 7 = coarse_n - 1, far folds back.
        MultigridPressureQuery::AxisContributors {
            fine_index: 15,
            coarse_n: 8,
        },
        // Un-coarsened degenerate: a one-cell coarse axis yields a single
        // clamped parent.
        MultigridPressureQuery::AxisContributors {
            fine_index: 3,
            coarse_n: 1,
        },
        // Empty coarse axis: zero contributors.
        MultigridPressureQuery::AxisContributors {
            fine_index: 0,
            coarse_n: 0,
        },
    ];
    let got = check(&ctx, &gpu, &queries);
    // Pin the discrete stencil shapes explicitly.
    assert!(matches!(
        got[0],
        MultigridPressureResult::Axis {
            idx: [2, 1],
            len: 2,
            ..
        }
    ));
    assert!(matches!(
        got[2],
        MultigridPressureResult::Axis {
            idx: [0, _],
            len: 1,
            ..
        }
    ));
    assert!(matches!(
        got[5],
        MultigridPressureResult::Axis { len: 0, .. }
    ));
}

#[test]
fn prolong_cell_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMultigridPressure::new(&ctx);
    let coarse = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
    let queries = [
        // Full 2x2x2 tensor fold.
        MultigridPressureQuery::ProlongCell {
            coarse_values: coarse,
            weight_x: [0.75, 0.25],
            weight_y: [0.75, 0.25],
            weight_z: [0.75, 0.25],
            len_x: 2,
            len_y: 2,
            len_z: 2,
        },
        // Degenerate x-axis (one contributor), full y and z.
        MultigridPressureQuery::ProlongCell {
            coarse_values: coarse,
            weight_x: [1.0, 0.0],
            weight_y: [0.75, 0.25],
            weight_z: [0.75, 0.25],
            len_x: 1,
            len_y: 2,
            len_z: 2,
        },
        // All axes degenerate: a single coarse sample.
        MultigridPressureQuery::ProlongCell {
            coarse_values: coarse,
            weight_x: [1.0, 0.0],
            weight_y: [1.0, 0.0],
            weight_z: [1.0, 0.0],
            len_x: 1,
            len_y: 1,
            len_z: 1,
        },
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn restrict_scale_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMultigridPressure::new(&ctx);
    let queries = [
        MultigridPressureQuery::RestrictScale { coarsened_axes: 0 },
        MultigridPressureQuery::RestrictScale { coarsened_axes: 1 },
        MultigridPressureQuery::RestrictScale { coarsened_axes: 2 },
        MultigridPressureQuery::RestrictScale { coarsened_axes: 3 },
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn mixed_variant_batch_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMultigridPressure::new(&ctx);
    let queries = [
        MultigridPressureQuery::DivergenceForward {
            here: [0.1, 0.2, 0.3],
            forward: [0.4, 0.5, 0.6],
            forward_in_grid: [true, false, true],
        },
        MultigridPressureQuery::RestrictScale { coarsened_axes: 2 },
        MultigridPressureQuery::GradientBackward {
            here: 0.9,
            backward: [0.1, 0.2, 0.3],
            backward_in_grid: [false, true, true],
        },
        MultigridPressureQuery::AxisContributors {
            fine_index: 7,
            coarse_n: 6,
        },
        MultigridPressureQuery::Diagonal {
            boundary: PressureBoundary::Neumann,
            live: 5,
        },
        MultigridPressureQuery::JacobiCell {
            center: 0.33,
            neighbors: [0.2, -0.2, 0.4, -0.4, 0.1, -0.1],
            neighbor_in_grid: [true, true, true, true, true, true],
            rhs: 0.05,
            inv_h2: 2.0,
            omega: 0.7,
            boundary: PressureBoundary::Dirichlet,
        },
        MultigridPressureQuery::ProlongCell {
            coarse_values: [0.5, 0.4, 0.3, 0.2, 0.1, 0.0, -0.1, -0.2],
            weight_x: [0.75, 0.25],
            weight_y: [0.75, 0.25],
            weight_z: [1.0, 0.0],
            len_x: 2,
            len_y: 2,
            len_z: 1,
        },
    ];
    check(&ctx, &gpu, &queries);
}

#[test]
fn large_random_batch_matches() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuMultigridPressure::new(&ctx);
    let mut state = 0x1234_5678_9abc_def0u64;
    let mut queries = Vec::new();
    for _ in 0..256 {
        let kind = (lcg(&mut state) * 8.0) as u32;
        let query = match kind {
            0 => MultigridPressureQuery::DivergenceForward {
                here: [signed(&mut state), signed(&mut state), signed(&mut state)],
                forward: [signed(&mut state), signed(&mut state), signed(&mut state)],
                forward_in_grid: [flag(&mut state), flag(&mut state), flag(&mut state)],
            },
            1 => MultigridPressureQuery::GradientBackward {
                here: signed(&mut state),
                backward: [signed(&mut state), signed(&mut state), signed(&mut state)],
                backward_in_grid: [flag(&mut state), flag(&mut state), flag(&mut state)],
            },
            2 => {
                // Dirichlet keeps the diagonal a constant six, clear of any
                // live-count edge; inv_h2 stays comfortably positive.
                let inv_h2 = 0.5 + lcg(&mut state) * 4.0;
                MultigridPressureQuery::JacobiCell {
                    center: signed(&mut state),
                    neighbors: [
                        signed(&mut state),
                        signed(&mut state),
                        signed(&mut state),
                        signed(&mut state),
                        signed(&mut state),
                        signed(&mut state),
                    ],
                    neighbor_in_grid: [true, true, true, true, true, true],
                    rhs: signed(&mut state),
                    inv_h2,
                    omega: 0.3 + lcg(&mut state) * 0.6,
                    boundary: PressureBoundary::Dirichlet,
                }
            }
            3 => {
                let inv_h2 = 0.5 + lcg(&mut state) * 4.0;
                MultigridPressureQuery::ResidualCell {
                    center: signed(&mut state),
                    neighbors: [
                        signed(&mut state),
                        signed(&mut state),
                        signed(&mut state),
                        signed(&mut state),
                        signed(&mut state),
                        signed(&mut state),
                    ],
                    neighbor_in_grid: [true, true, true, true, true, true],
                    rhs: signed(&mut state),
                    inv_h2,
                    boundary: PressureBoundary::Dirichlet,
                }
            }
            4 => MultigridPressureQuery::Diagonal {
                boundary: PressureBoundary::Dirichlet,
                live: 1 + (lcg(&mut state) * 6.0) as u32,
            },
            5 => {
                // Keep the fine index strictly interior so the integer edge
                // tests never sit on a tie: coarse_n >= 4 and 2 <= parent <= cn-2.
                let coarse_n = 4 + (lcg(&mut state) * 8.0) as u32;
                let parent = 2 + (lcg(&mut state) * (coarse_n as f32 - 3.0)) as u32;
                let fine_index = parent * 2 + (lcg(&mut state) * 2.0) as u32;
                MultigridPressureQuery::AxisContributors {
                    fine_index,
                    coarse_n,
                }
            }
            6 => MultigridPressureQuery::ProlongCell {
                coarse_values: [
                    signed(&mut state),
                    signed(&mut state),
                    signed(&mut state),
                    signed(&mut state),
                    signed(&mut state),
                    signed(&mut state),
                    signed(&mut state),
                    signed(&mut state),
                ],
                weight_x: [0.75, 0.25],
                weight_y: [0.75, 0.25],
                weight_z: [0.75, 0.25],
                len_x: 2,
                len_y: 2,
                len_z: 2,
            },
            _ => MultigridPressureQuery::RestrictScale {
                coarsened_axes: (lcg(&mut state) * 4.0) as u32,
            },
        };
        queries.push(query);
    }
    check(&ctx, &gpu, &queries);
}
