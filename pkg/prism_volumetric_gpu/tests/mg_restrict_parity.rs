//! Real-device parity for the multigrid full-weighting restriction twin:
//! [`GpuMgRestrict`] must reproduce the `CPU` golden `restrict` from
//! [`multigrid_pressure`](prism_render_architecture::particle::multigrid_pressure)
//! across random fields and a spread of even, odd, non-cubic and single-axis
//! coarsened resolutions, plus the structural constant-preservation property and
//! the degenerate grids the reference guards.
//!
//! The golden `restrict` (and the private helpers it leans on) are not exported,
//! so this test carries a line-for-line mirror of each — tagged with a
//! `MIRROR of ...` provenance note — and compares the on-device gather against
//! that mirrored scatter. The scatter and the gather express the identical
//! bilinear form, so they agree up to floating-point reassociation.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device.
//!
//! # Parity criterion
//!
//! Each term is multiply-add with no transcendental call, so `CPU` and `GPU`
//! evaluate the same algebra. Values are asserted to within `abs_diff <= 1e-4`
//! or `rel_diff <= 1e-3` — loose enough to admit a `GPU` fused multiply-add, yet
//! tight enough to fail a wrong port (a swapped weight, a dropped mirror fold, a
//! missing normalization, a transposed axis).
//!
//! Provenance: standard `Briggs` multigrid full-weighting restriction; no Unreal
//! Engine source or derived code.

use prism_render_architecture::particle::fluid::GridResolution;
use prism_volumetric_gpu::mg_restrict::{GpuMgRestrict, GpuMgRestrictQuery};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity tolerance, a decade above single-term multiply-add rounding.
const ABS_EPS: f32 = 1.0e-4;

/// Relative parity tolerance for samples large enough that the absolute floor is
/// pessimistic.
const REL_EPS: f32 = 1.0e-3;

/// Relative-tolerance floor so a near-zero reference never divides by zero.
const REL_FLOOR: f32 = 1.0e-6;

// ----------------------------------------------------------------------------
// Mirror of the golden `multigrid_pressure` restriction and its private helpers.
// Each block is an exact transcription of the private reference so this test can
// compare against it without the symbols being exported.
// ----------------------------------------------------------------------------

/// Up to two coarse indices and their interpolation weights for one fine index
/// on one axis.
///
/// MIRROR of `multigrid_pressure.rs::AxisStencil` (private, exact transcription)
#[derive(Clone, Copy, Debug)]
struct AxisStencil {
    idx: [u32; 2],
    weight: [f32; 2],
    len: usize,
}

/// MIRROR of `multigrid_pressure.rs::coarse_axis` (private, exact transcription)
fn coarse_axis(n: u32) -> u32 {
    if n <= 1 {
        n
    } else {
        n.div_ceil(2)
    }
}

/// MIRROR of `multigrid_pressure.rs::coarsen` (private, exact transcription)
fn coarsen(res: GridResolution) -> GridResolution {
    GridResolution::new(
        coarse_axis(res.nx),
        coarse_axis(res.ny),
        coarse_axis(res.nz),
    )
}

/// MIRROR of `multigrid_pressure.rs::axis_contributors` (private, exact transcription)
fn axis_contributors(i: u32, coarse_n: u32) -> AxisStencil {
    if coarse_n == 0 {
        return AxisStencil {
            idx: [0, 0],
            weight: [0.0, 0.0],
            len: 0,
        };
    }
    let parent = i / 2;
    let parent = if parent >= coarse_n {
        coarse_n - 1
    } else {
        parent
    };
    let own_weight = 0.75;
    let far_weight = 0.25;
    let far_is_lower = i.is_multiple_of(2);
    let far_in_range = if far_is_lower {
        parent > 0
    } else {
        parent + 1 < coarse_n
    };
    if far_in_range {
        let far = if far_is_lower { parent - 1 } else { parent + 1 };
        AxisStencil {
            idx: [parent, far],
            weight: [own_weight, far_weight],
            len: 2,
        }
    } else {
        AxisStencil {
            idx: [parent, 0],
            weight: [own_weight + far_weight, 0.0],
            len: 1,
        }
    }
}

/// MIRROR of `multigrid_pressure.rs::coarsened_axis_count` (private, exact transcription)
fn coarsened_axis_count(fine: GridResolution, coarse: GridResolution) -> u32 {
    let mut k = 0u32;
    if coarse.nx < fine.nx {
        k += 1;
    }
    if coarse.ny < fine.ny {
        k += 1;
    }
    if coarse.nz < fine.nz {
        k += 1;
    }
    k
}

/// MIRROR of `multigrid_pressure.rs::pow2_f32` (private, exact transcription)
fn pow2_f32(k: u32) -> f32 {
    let mut value = 1.0f32;
    let mut remaining = k;
    while remaining > 0 {
        value *= 2.0;
        remaining -= 1;
    }
    value
}

/// MIRROR of `multigrid_pressure.rs::restrict` (private, exact transcription)
fn golden_restrict(fine: &[f32], fine_res: GridResolution, coarse_res: GridResolution) -> Vec<f32> {
    let coarse_count = coarse_res.voxel_count() as usize;
    let fine_count = fine_res.voxel_count() as usize;
    let mut coarse = vec![0.0f32; coarse_count];
    if fine.len() < fine_count {
        return coarse;
    }
    let k = coarsened_axis_count(fine_res, coarse_res);
    let scale = 1.0 / pow2_f32(k);
    for z in 0..fine_res.nz {
        let sz = axis_contributors(z, coarse_res.nz);
        for y in 0..fine_res.ny {
            let sy = axis_contributors(y, coarse_res.ny);
            for x in 0..fine_res.nx {
                let sx = axis_contributors(x, coarse_res.nx);
                let value = fine[fine_res.linear_index(x, y, z) as usize];
                for iz in 0..sz.len {
                    let wz = sz.weight[iz];
                    let cz = sz.idx[iz];
                    for iy in 0..sy.len {
                        let wy = sy.weight[iy];
                        let cy = sy.idx[iy];
                        for ix in 0..sx.len {
                            let cidx = coarse_res.linear_index(sx.idx[ix], cy, cz) as usize;
                            coarse[cidx] += scale * sx.weight[ix] * wy * wz * value;
                        }
                    }
                }
            }
        }
    }
    coarse
}

// ----------------------------------------------------------------------------
// Test helpers.
// ----------------------------------------------------------------------------

/// A tiny deterministic linear-congruential generator so the "random" fields are
/// reproducible run to run without pulling in an external crate. The constants
/// are the Numerical Recipes `LCG` multiplier and increment.
struct Lcg {
    state: u32,
}

impl Lcg {
    fn new(seed: u32) -> Self {
        Lcg { state: seed }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(1_664_525)
            .wrapping_add(1_013_904_223);
        self.state
    }

    /// A reproducible `f32` in `[-range, range]` using pure integer scaling (no
    /// transcendental method).
    fn next_signed(&mut self, range: f32) -> f32 {
        let unit = (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32;
        (unit * 2.0 - 1.0) * range
    }

    fn field(&mut self, count: usize, range: f32) -> Vec<f32> {
        (0..count).map(|_| self.next_signed(range)).collect()
    }
}

/// Asserts the `GPU` coarse field matches the mirrored golden element for
/// element to within the documented tolerance.
fn assert_parity(label: &str, cpu: &[f32], gpu: &[f32]) {
    assert_eq!(cpu.len(), gpu.len(), "{label}: field length mismatch");
    for (i, (c, g)) in cpu.iter().zip(gpu.iter()).enumerate() {
        let abs_diff = (c - g).abs();
        let rel_diff = abs_diff / c.abs().max(REL_FLOOR);
        assert!(
            abs_diff <= ABS_EPS || rel_diff <= REL_EPS,
            "{label}: cell {i} mismatch: cpu {c}, gpu {g} (abs {abs_diff}, rel {rel_diff})"
        );
    }
}

/// Runs one mirrored-`CPU`-vs-`GPU` scenario on the coarsened resolution and
/// asserts parity.
fn check_scenario(
    label: &str,
    engine: &GpuMgRestrict,
    ctx: &GpuContext,
    fine: &[f32],
    fine_res: GridResolution,
) {
    let coarse_res = coarsen(fine_res);
    let cpu = golden_restrict(fine, fine_res, coarse_res);
    let gpu = engine.restrict(
        ctx,
        &GpuMgRestrictQuery {
            fine: fine.to_vec(),
            fine_res,
            coarse_res,
        },
    );
    assert_parity(label, &cpu, &gpu.coarse);
}

// ----------------------------------------------------------------------------
// Tests.
// ----------------------------------------------------------------------------

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_matches_cpu_across_resolutions() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping mg-restrict parity: no wgpu adapter on this host");
        return;
    };
    let engine = GpuMgRestrict::new(&ctx);

    // Even, odd, non-cubic and taller resolutions, each on its own reproducible
    // random fine field.
    let resolutions = [
        GridResolution::uniform(4),
        GridResolution::uniform(5),
        GridResolution::uniform(8),
        GridResolution::new(6, 3, 2),
        GridResolution::new(3, 7, 5),
        GridResolution::new(7, 7, 7),
    ];
    for (seed, res) in resolutions.into_iter().enumerate() {
        let mut rng = Lcg::new(0x5E57_u32.wrapping_add(seed as u32));
        let fine = rng.field(res.voxel_count() as usize, 3.0);
        let coarse = coarsen(res);
        let label = format!(
            "random case {seed} (fine {}x{}x{} -> coarse {}x{}x{})",
            res.nx, res.ny, res.nz, coarse.nx, coarse.ny, coarse.nz
        );
        check_scenario(&label, &engine, &ctx, &fine, res);
    }
}

#[test]
fn gpu_matches_cpu_single_axis_coarsened() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let engine = GpuMgRestrict::new(&ctx);

    // Only one axis is larger than one, so only one axis is halved: k = 1 and
    // scale = 1/2. This exercises the single-axis-coarsening path explicitly.
    let cases = [
        GridResolution::new(8, 1, 1),
        GridResolution::new(1, 6, 1),
        GridResolution::new(1, 1, 5),
    ];
    for (seed, res) in cases.into_iter().enumerate() {
        // Confirm the fixture really coarsens exactly one axis.
        assert_eq!(
            coarsened_axis_count(res, coarsen(res)),
            1,
            "fixture {seed} should coarsen exactly one axis"
        );
        let mut rng = Lcg::new(0xA1E5_u32.wrapping_add(seed as u32));
        let fine = rng.field(res.voxel_count() as usize, 2.5);
        check_scenario(
            &format!("single-axis case {seed}"),
            &engine,
            &ctx,
            &fine,
            res,
        );
    }
}

#[test]
fn gpu_preserves_constant_field() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let engine = GpuMgRestrict::new(&ctx);

    // The `GPU` twin must reproduce the golden full-weighting restriction exactly
    // (the mirror-parity contract) on every resolution. The additional structural
    // guarantee -- that a constant fine field restricts to the *same* constant on
    // every coarse cell -- only holds when each axis has an even extent, because
    // the folded `(3/4, 1/4)` cell-centered weights sum to the full `2^k`
    // normalization on every coarse cell exactly then. On an odd axis the final
    // coarse cell has a single fine child, so its weight sum is halved and the
    // golden restriction itself does not preserve the constant there (the golden
    // module's own `restrict_preserves_constant_field` test uses an even grid for
    // this reason). We therefore assert mirror parity everywhere and the stronger
    // constant-preservation property only on fully even grids.
    let value = 1.75f32;
    let resolutions = [
        GridResolution::uniform(4),
        GridResolution::uniform(5),
        GridResolution::new(6, 4, 2),
        GridResolution::new(8, 6, 4),
    ];
    for res in resolutions {
        let coarse_res = coarsen(res);
        let fine = vec![value; res.voxel_count() as usize];
        let gpu = engine.restrict(
            &ctx,
            &GpuMgRestrictQuery {
                fine: fine.clone(),
                fine_res: res,
                coarse_res,
            },
        );
        let cpu = golden_restrict(&fine, res, coarse_res);
        let label = format!("constant {}x{}x{}", res.nx, res.ny, res.nz);
        assert_parity(&label, &cpu, &gpu.coarse);
        let all_even = res.nx % 2 == 0 && res.ny % 2 == 0 && res.nz % 2 == 0;
        if all_even {
            for (i, &c) in gpu.coarse.iter().enumerate() {
                let abs_diff = (c - value).abs();
                let rel_diff = abs_diff / value.abs().max(REL_FLOOR);
                assert!(
                    abs_diff <= ABS_EPS || rel_diff <= REL_EPS,
                    "{label}: coarse cell {i} should preserve the constant {value}, got {c}"
                );
            }
        }
    }
}

#[test]
fn gpu_handles_short_fine_slice() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let engine = GpuMgRestrict::new(&ctx);

    // A fine slice shorter than the grid demands an all-zero coarse field, with
    // no dispatch — the exact reference guard.
    let fine_res = GridResolution::uniform(4);
    let coarse_res = coarsen(fine_res);
    let short = vec![1.0f32; (fine_res.voxel_count() as usize) - 1];
    let gpu = engine.restrict(
        &ctx,
        &GpuMgRestrictQuery {
            fine: short.clone(),
            fine_res,
            coarse_res,
        },
    );
    let cpu = golden_restrict(&short, fine_res, coarse_res);
    assert_parity("short fine slice", &cpu, &gpu.coarse);
    assert!(
        gpu.coarse.iter().all(|&c| c.abs() <= ABS_EPS),
        "short fine slice yields an all-zero coarse field"
    );
}

#[test]
fn gpu_handles_empty_grid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let engine = GpuMgRestrict::new(&ctx);

    // A zero-extent coarse grid yields an empty field with no dispatch.
    let fine_res = GridResolution::new(0, 4, 4);
    let coarse_res = coarsen(fine_res);
    let gpu = engine.restrict(
        &ctx,
        &GpuMgRestrictQuery {
            fine: Vec::new(),
            fine_res,
            coarse_res,
        },
    );
    assert!(
        gpu.coarse.is_empty(),
        "zero-extent grid yields an empty coarse field"
    );
}
