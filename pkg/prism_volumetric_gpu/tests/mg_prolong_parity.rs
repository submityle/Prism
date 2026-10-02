//! Real-device parity for the geometric-multigrid prolongation twin:
//! [`GpuMgProlong`] must reproduce the `CPU` golden `prolong` (the coarse-to-fine
//! inter-grid transfer extracted from
//! [`multigrid_pressure`](prism_render_architecture::particle::multigrid_pressure))
//! across random coarse fields on several resolution pairs — cubic, non-cubic,
//! even and odd extents, single-axis coarsening — plus the constant-field
//! invariant and the degenerate grids the reference guards.
//!
//! The golden `prolong` and its `axis_contributors` / `coarsen` helpers are
//! private to the architecture crate, so this test carries a faithful,
//! line-for-line re-implementation of their arithmetic (the same `(3/4, 1/4)`
//! cell-centered split, the same mirror fold, the same `z`-`y`-`x` accumulation)
//! and asserts the twin matches it. The re-implementation is the reference; it
//! is not imported from the golden module.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable
//! core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each fine cell is a multiply-add tensor product of three per-axis stencils,
//! accumulated in the identical order the reference uses, so `CPU` and `GPU`
//! evaluate the same algebra. Values are asserted to within `abs_diff <= 1e-4`
//! or `rel_diff <= 1e-3` — loose enough to admit a `GPU` fused multiply-add, yet
//! tight enough to fail a wrong port (a swapped stencil weight, a dropped mirror
//! fold, a transposed axis).
//!
//! Provenance: standard `Briggs` multigrid trilinear prolongation; no Unreal
//! Engine source or derived code.

use prism_render_architecture::particle::fluid::GridResolution;
use prism_volumetric_gpu::mg_prolong::{GpuMgProlong, GpuMgProlongQuery, GpuMgProlongResult};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity tolerance, a decade above the single multiply-add rounding
/// so the `GPU` fused multiply-add stays inside it while a genuinely wrong port
/// falls outside.
const ABS_EPS: f32 = 1.0e-4;

/// Relative parity tolerance, applied for samples large enough that the
/// absolute floor is pessimistic.
const REL_EPS: f32 = 1.0e-3;

/// Relative-difference floor so a near-zero reference value does not blow up the
/// relative test.
const REL_FLOOR: f32 = 1.0e-6;

/// A tiny deterministic linear-congruential generator so the "random" fields
/// are reproducible run to run without pulling in any external math or random
/// crate. The constants are the Numerical Recipes `LCG` multiplier and
/// increment.
struct Lcg {
    state: u32,
}

impl Lcg {
    fn new(seed: u32) -> Self {
        Lcg { state: seed }
    }

    /// Advances the generator and returns the next raw word.
    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(1_664_525)
            .wrapping_add(1_013_904_223);
        self.state
    }

    /// A reproducible `f32` in `[-range, range]`, built only from integer
    /// arithmetic (no transcendental or inexact helper).
    fn next_signed(&mut self, range: f32) -> f32 {
        let unit = (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32;
        (unit * 2.0 - 1.0) * range
    }

    /// A reproducible field of `count` scalars in `[-range, range]`.
    fn field(&mut self, count: usize, range: f32) -> Vec<f32> {
        (0..count).map(|_| self.next_signed(range)).collect()
    }
}

// MIRROR of multigrid_pressure.rs::prolong + helpers (private, exact transcription)

/// Up to two coarse indices and their interpolation weights for one fine index
/// on one axis.
#[derive(Clone, Copy, Debug)]
struct AxisStencil {
    idx: [u32; 2],
    weight: [f32; 2],
    len: usize,
}

/// Faithful re-implementation of the golden private `coarse_axis`.
fn golden_coarse_axis(n: u32) -> u32 {
    if n <= 1 {
        n
    } else {
        n.div_ceil(2)
    }
}

/// Faithful re-implementation of the golden private `coarsen`.
fn golden_coarsen(res: GridResolution) -> GridResolution {
    GridResolution::new(
        golden_coarse_axis(res.nx),
        golden_coarse_axis(res.ny),
        golden_coarse_axis(res.nz),
    )
}

/// Faithful re-implementation of the golden private `coarsened_axis_count`.
///
/// Prolongation itself does not use it (it is the restriction normalization),
/// but it is mirrored here alongside its siblings for a complete transcription
/// of the inter-grid transfer helpers.
#[expect(
    dead_code,
    reason = "mirrored for a complete transcription; prolongation does not use it"
)]
fn golden_coarsened_axis_count(fine: GridResolution, coarse: GridResolution) -> u32 {
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

/// Faithful re-implementation of the golden private `axis_contributors`.
fn golden_axis_contributors(i: u32, coarse_n: u32) -> AxisStencil {
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

/// Faithful re-implementation of the golden private `prolong`.
fn golden_prolong(
    coarse: &[f32],
    coarse_res: GridResolution,
    fine_res: GridResolution,
) -> Vec<f32> {
    let fine_count = fine_res.voxel_count() as usize;
    let coarse_count = coarse_res.voxel_count() as usize;
    let mut fine = vec![0.0f32; fine_count];
    if coarse.len() < coarse_count {
        return fine;
    }
    for z in 0..fine_res.nz {
        let sz = golden_axis_contributors(z, coarse_res.nz);
        for y in 0..fine_res.ny {
            let sy = golden_axis_contributors(y, coarse_res.ny);
            for x in 0..fine_res.nx {
                let sx = golden_axis_contributors(x, coarse_res.nx);
                let mut acc = 0.0f32;
                for iz in 0..sz.len {
                    let wz = sz.weight[iz];
                    let cz = sz.idx[iz];
                    for iy in 0..sy.len {
                        let wy = sy.weight[iy];
                        let cy = sy.idx[iy];
                        for ix in 0..sx.len {
                            let cidx = coarse_res.linear_index(sx.idx[ix], cy, cz) as usize;
                            acc += sx.weight[ix] * wy * wz * coarse[cidx];
                        }
                    }
                }
                fine[fine_res.linear_index(x, y, z) as usize] = acc;
            }
        }
    }
    fine
}

/// Asserts two scalars match to within the documented tolerance.
fn close(label: &str, cpu: f32, gpu: f32) {
    let abs_diff = (cpu - gpu).abs();
    let rel_diff = abs_diff / cpu.abs().max(REL_FLOOR);
    assert!(
        abs_diff <= ABS_EPS || rel_diff <= REL_EPS,
        "{label}: cpu {cpu}, gpu {gpu} (abs {abs_diff}, rel {rel_diff})"
    );
}

/// Asserts the `GPU` fine field matches the `CPU` reference element-wise.
fn assert_parity(label: &str, cpu: &[f32], gpu: &GpuMgProlongResult) {
    assert_eq!(cpu.len(), gpu.fine.len(), "{label}: fine length mismatch");
    for (i, (c, g)) in cpu.iter().zip(gpu.fine.iter()).enumerate() {
        close(&format!("{label}: cell {i}"), *c, *g);
    }
}

/// Builds a query from a coarse field and its two resolutions.
fn query(
    coarse: Vec<f32>,
    coarse_res: GridResolution,
    fine_res: GridResolution,
) -> GpuMgProlongQuery {
    GpuMgProlongQuery {
        coarse,
        coarse_res,
        fine_res,
    }
}

/// Runs one `CPU`-vs-`GPU` scenario and asserts parity.
fn check_scenario(label: &str, engine: &GpuMgProlong, ctx: &GpuContext, q: &GpuMgProlongQuery) {
    let cpu = golden_prolong(&q.coarse, q.coarse_res, q.fine_res);
    let gpu = engine.prolong(ctx, q);
    assert_parity(label, &cpu, &gpu);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_matches_cpu_across_resolution_pairs() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping mg-prolong parity: no wgpu adapter on this host");
        return;
    };
    let engine = GpuMgProlong::new(&ctx);

    // Each fine resolution is coarsened by the golden rule to form a matching
    // valid coarse grid: cubic even, cubic odd, non-cubic, and a tall column —
    // exercising even/odd extents and anisotropic shapes together.
    let fines = [
        GridResolution::uniform(4),
        GridResolution::uniform(5),
        GridResolution::new(6, 4, 2),
        GridResolution::new(3, 5, 7),
        GridResolution::uniform(8),
        GridResolution::new(2, 6, 3),
    ];
    for (seed, fine_res) in fines.into_iter().enumerate() {
        let coarse_res = golden_coarsen(fine_res);
        let coarse_count = coarse_res.voxel_count() as usize;
        let mut rng = Lcg::new(0x51ED_u32.wrapping_add(seed as u32 * 7 + 1));
        let coarse = rng.field(coarse_count, 2.0);
        let label = format!(
            "resolution pair {seed} (coarse {}x{}x{} -> fine {}x{}x{})",
            coarse_res.nx, coarse_res.ny, coarse_res.nz, fine_res.nx, fine_res.ny, fine_res.nz
        );
        check_scenario(&label, &engine, &ctx, &query(coarse, coarse_res, fine_res));
    }
}

#[test]
fn gpu_matches_cpu_on_single_axis_coarsening() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let engine = GpuMgProlong::new(&ctx);

    // Only one axis is genuinely halved; the other two keep their extent (the
    // "axis not coarsened" degenerate branch of axis_contributors on those
    // axes). One case per axis.
    let cases = [
        (GridResolution::new(2, 5, 5), GridResolution::new(4, 5, 5)),
        (GridResolution::new(5, 2, 5), GridResolution::new(5, 4, 5)),
        (GridResolution::new(5, 5, 3), GridResolution::new(5, 5, 6)),
    ];
    for (seed, (coarse_res, fine_res)) in cases.into_iter().enumerate() {
        let coarse_count = coarse_res.voxel_count() as usize;
        let mut rng = Lcg::new(0x3C7A_u32.wrapping_add(seed as u32 * 13 + 1));
        let coarse = rng.field(coarse_count, 1.5);
        let label = format!("single-axis coarsening {seed}");
        check_scenario(&label, &engine, &ctx, &query(coarse, coarse_res, fine_res));
    }
}

#[test]
fn gpu_preserves_constant_field() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let engine = GpuMgProlong::new(&ctx);

    // A constant coarse field must prolong to the same constant on every fine
    // cell, since the per-axis weights sum to one — the property the coarse-grid
    // correction relies on. Assert the structural invariant directly, not only
    // GPU-vs-CPU parity.
    let fine_res = GridResolution::new(6, 5, 4);
    let coarse_res = golden_coarsen(fine_res);
    let coarse_count = coarse_res.voxel_count() as usize;
    let constant = 1.375f32;
    let coarse = vec![constant; coarse_count];
    let q = query(coarse, coarse_res, fine_res);

    let gpu = engine.prolong(&ctx, &q);
    assert_eq!(gpu.fine.len(), fine_res.voxel_count() as usize);
    for (i, g) in gpu.fine.iter().enumerate() {
        close(&format!("constant fine cell {i}"), constant, *g);
    }

    // And it still agrees with the golden reference on the same input.
    check_scenario("constant field parity", &engine, &ctx, &q);
}

#[test]
fn gpu_matches_cpu_on_empty_fine_grid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let engine = GpuMgProlong::new(&ctx);

    // An empty fine grid (a zero extent) short-circuits on the host to an empty
    // field, matching the reference.
    let fine_res = GridResolution::new(0, 4, 4);
    let coarse_res = GridResolution::new(0, 2, 2);
    let q = query(Vec::new(), coarse_res, fine_res);
    let cpu = golden_prolong(&q.coarse, q.coarse_res, q.fine_res);
    let gpu = engine.prolong(&ctx, &q);
    assert_eq!(cpu.len(), 0, "golden empty-fine field should be empty");
    assert_eq!(
        gpu.fine, cpu,
        "empty fine grid should return an empty field"
    );
}

#[test]
fn gpu_matches_cpu_on_insufficient_coarse() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let engine = GpuMgProlong::new(&ctx);

    // A coarse slice shorter than coarse_res.voxel_count() returns an all-zero
    // fine field, the exact guard the reference takes.
    let fine_res = GridResolution::uniform(4);
    let coarse_res = GridResolution::uniform(2);
    let short = vec![1.0f32; (coarse_res.voxel_count() as usize) - 1];
    let q = query(short, coarse_res, fine_res);
    let cpu = golden_prolong(&q.coarse, q.coarse_res, q.fine_res);
    let gpu = engine.prolong(&ctx, &q);
    assert_eq!(
        gpu.fine.len(),
        fine_res.voxel_count() as usize,
        "insufficient coarse should still size the fine field"
    );
    for (i, g) in gpu.fine.iter().enumerate() {
        close(&format!("insufficient coarse fine cell {i}"), 0.0, *g);
    }
    assert_parity("insufficient coarse", &cpu, &gpu);
}
