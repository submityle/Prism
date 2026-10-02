//! Real-device parity for the vorticity-confinement twin: [`GpuFluidVorticity`]
//! must reproduce the `CPU` golden curl and confinement force across random
//! velocity fields, several grid resolutions, the clamp-to-edge boundary the
//! reference samplers use, and the degenerate grids the reference guards.
//!
//! The twin owns two pure functions of the golden
//! [`fluid`](prism_render_architecture::particle::fluid) module: the
//! central-difference curl
//! ([`NeighborVelocities::curl`](prism_render_architecture::particle::fluid::NeighborVelocities::curl))
//! and the confinement force
//! ([`vorticity_confinement_force`](prism_render_architecture::particle::fluid::vorticity_confinement_force)).
//! For every voxel the host reference rebuilds the same six clamp-to-edge
//! neighbors, folds the identical [`NeighborVelocities`], takes the golden curl,
//! and evaluates the golden force with the per-voxel gradient of `‖ω‖`; the
//! `GPU` output is then asserted against that reference element for element.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable
//! core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The curl, cross and scale are closed-form multiply-add algebra evaluated in
//! the golden's term order, plus one normalize `sqrt`, so `CPU` and `GPU`
//! compute the same expression. Values are asserted to within
//! `abs_diff <= 1e-4` or `rel_diff <= 1e-3` — loose enough to admit a `GPU`
//! fused multiply-add or a `sqrt` reciprocal that differs by a few `ULP`, yet
//! tight enough to fail a wrong port (a swapped neighbor, a transposed cross
//! product, a dropped boundary clamp).
//!
//! Every non-flat gradient fixture keeps its squared length far above the
//! golden `EPS_LEN_SQ` of `1e-12`, so the device and the reference always take
//! the same `normalize_or_zero` branch; flat fixtures use the exact zero vector
//! so both deterministically yield the zero force.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::fluid`；无第三方引擎源码或衍生代码。
#![forbid(unsafe_code)]

use prism_render_architecture::particle::fluid::{
    vorticity_confinement_force, GridResolution, NeighborVelocities, VorticityParams,
};
use prism_render_architecture::particle::Vec3;
use prism_volumetric_gpu::fluid_vorticity::{GpuFluidVorticity, GpuVorticityQuery};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity tolerance. Chosen a decade above the single multiply-add
/// rounding so a legal `GPU` fused multiply-add stays inside it while a
/// genuinely wrong port falls outside.
const ABS_EPS: f32 = 1.0e-4;

/// Relative parity tolerance, applied for samples large enough that the
/// absolute floor is pessimistic.
const REL_EPS: f32 = 1.0e-3;

/// Relative-difference floor, so a near-zero reference value does not blow the
/// relative term up.
const REL_FLOOR: f32 = 1.0e-6;

/// A tiny deterministic linear-congruential generator so the "random" fields
/// are reproducible run to run without pulling in an external math crate. The
/// constants are the Numerical Recipes `LCG` multiplier and increment.
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

    /// A reproducible `f32` in `[-range, range]`.
    fn next_signed(&mut self, range: f32) -> f32 {
        let unit = (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32;
        (unit * 2.0 - 1.0) * range
    }

    /// A reproducible velocity field of `count` samples with components in
    /// `[-range, range]`.
    fn velocity_field(&mut self, count: usize, range: f32) -> Vec<Vec3> {
        (0..count)
            .map(|_| {
                Vec3::new(
                    self.next_signed(range),
                    self.next_signed(range),
                    self.next_signed(range),
                )
            })
            .collect()
    }

    /// A reproducible gradient field whose every entry has squared length far
    /// above the golden `EPS_LEN_SQ`, keeping the `GPU` and the reference on the
    /// same `normalize_or_zero` branch (rejection sampling: a vector that lands
    /// too close to zero is replaced by a fixed non-degenerate direction).
    fn gradient_field(&mut self, count: usize, range: f32) -> Vec<Vec3> {
        (0..count)
            .map(|_| {
                let mut g = Vec3::new(
                    self.next_signed(range),
                    self.next_signed(range),
                    self.next_signed(range),
                );
                // 1e-2 sits nine orders of magnitude above the 1e-12 floor, so
                // both sides always normalize rather than return zero.
                if g.length_squared() < 1.0e-2 {
                    g = Vec3::new(range, 0.0, 0.0);
                }
                g
            })
            .collect()
    }
}

/// Clamp-to-edge neighbor index one step down an axis (replicate boundary).
fn clamp_minus(coord: u32) -> u32 {
    if coord > 0 {
        coord - 1
    } else {
        0
    }
}

/// Clamp-to-edge neighbor index one step up an axis (replicate boundary).
fn clamp_plus(coord: u32, extent: u32) -> u32 {
    let c = coord + 1;
    if c > extent - 1 {
        extent - 1
    } else {
        c
    }
}

/// Host reference: for each voxel, sample the six clamp-to-edge velocity
/// neighbors, fold the golden [`NeighborVelocities`], take the golden curl, and
/// evaluate the golden confinement force with the per-voxel gradient. Returns
/// the per-voxel curl and force in row-major input order.
fn cpu_reference(query: &GpuVorticityQuery) -> (Vec<Vec3>, Vec<Vec3>) {
    let res = query.resolution;
    let (nx, ny, nz) = (res.nx, res.ny, res.nz);
    let count = res.voxel_count() as usize;
    let mut curl = Vec::with_capacity(count);
    let mut force = Vec::with_capacity(count);
    for idx in 0..count as u32 {
        let x = idx % nx;
        let plane = idx / nx;
        let y = plane % ny;
        let z = plane / ny;
        let xp = clamp_plus(x, nx);
        let xm = clamp_minus(x);
        let yp = clamp_plus(y, ny);
        let ym = clamp_minus(y);
        let zp = clamp_plus(z, nz);
        let zm = clamp_minus(z);
        let sample = |cx, cy, cz| query.velocity[res.linear_index(cx, cy, cz) as usize];
        let neighbors = NeighborVelocities {
            x_plus: sample(xp, y, z),
            x_minus: sample(xm, y, z),
            y_plus: sample(x, yp, z),
            y_minus: sample(x, ym, z),
            z_plus: sample(x, y, zp),
            z_minus: sample(x, y, zm),
        };
        let c = neighbors.curl(query.inv_2h);
        let f = vorticity_confinement_force(
            c,
            query.magnitude_gradient[idx as usize],
            query.params,
            query.cell_size,
        );
        curl.push(c);
        force.push(f);
    }
    (curl, force)
}

/// Asserts the `GPU` field matches the `CPU` golden element for element to
/// within the documented tolerance.
fn assert_parity(label: &str, cpu: &[Vec3], gpu: &[Vec3]) {
    assert_eq!(cpu.len(), gpu.len(), "{label}: field length mismatch");
    for (i, (c, g)) in cpu.iter().zip(gpu.iter()).enumerate() {
        for (cv, gv, axis) in [(c.x, g.x, "x"), (c.y, g.y, "y"), (c.z, g.z, "z")] {
            let abs_diff = (cv - gv).abs();
            let rel_diff = abs_diff / cv.abs().max(REL_FLOOR);
            assert!(
                abs_diff <= ABS_EPS || rel_diff <= REL_EPS,
                "{label}: cell {i} axis {axis} mismatch: cpu {cv}, gpu {gv} \
                 (abs {abs_diff}, rel {rel_diff})"
            );
        }
    }
}

/// Builds a query with a random velocity field and a non-degenerate gradient
/// field, so both sides normalize rather than fall into the zero branch.
fn random_query(
    res: GridResolution,
    seed: u32,
    epsilon: f32,
    inv_2h: f32,
    cell_size: f32,
) -> GpuVorticityQuery {
    let count = res.voxel_count() as usize;
    let mut rng = Lcg::new(seed);
    let velocity = rng.velocity_field(count, 3.0);
    let magnitude_gradient = rng.gradient_field(count, 2.0);
    GpuVorticityQuery {
        resolution: res,
        velocity,
        magnitude_gradient,
        params: VorticityParams { epsilon },
        inv_2h,
        cell_size,
    }
}

/// Runs one `CPU`-vs-`GPU` scenario and asserts parity on both outputs.
fn check_scenario(
    label: &str,
    engine: &GpuFluidVorticity,
    ctx: &GpuContext,
    query: &GpuVorticityQuery,
) {
    let (cpu_curl, cpu_force) = cpu_reference(query);
    let gpu = engine.evaluate(ctx, query);
    assert_parity(&format!("{label} curl"), &cpu_curl, &gpu.curl);
    assert_parity(&format!("{label} force"), &cpu_force, &gpu.force);
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_matches_cpu_across_random_fields() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping fluid-vorticity parity: no wgpu adapter on this host");
        return;
    };
    let engine = GpuFluidVorticity::new(&ctx);

    // A spread of resolutions (cubic, anisotropic and a flat 1-D chain) each on
    // its own reproducible random field.
    let cases = [
        (
            GridResolution::uniform(2),
            0x0001_u32,
            0.5_f32,
            0.5_f32,
            1.0_f32,
        ),
        (GridResolution::uniform(3), 0x1234, 1.5, 0.25, 0.5),
        (GridResolution::uniform(5), 0xBEEF, 2.0, 0.75, 1.25),
        (GridResolution::new(6, 3, 2), 0xC0DE, 0.8, 0.5, 2.0),
        (GridResolution::new(4, 1, 1), 0xFADE, 1.0, 1.0, 1.0),
        (GridResolution::new(1, 5, 1), 0xA11C, 1.2, 0.5, 0.75),
    ];
    for (res, seed, epsilon, inv_2h, cell_size) in cases {
        let query = random_query(res, seed, epsilon, inv_2h, cell_size);
        check_scenario(&format!("random {res:?}"), &engine, &ctx, &query);
    }
}

#[test]
fn gpu_matches_cpu_rigid_rotation() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let engine = GpuFluidVorticity::new(&ctx);

    // A rigid-rotation velocity field v = (-y, x, 0) over a 5x5x5 grid: the
    // curl is a constant along +Z in the interior, a structured non-trivial
    // field that catches a transposed cross or a swapped neighbor.
    let res = GridResolution::uniform(5);
    let count = res.voxel_count() as usize;
    let mut velocity = Vec::with_capacity(count);
    for idx in 0..count as u32 {
        let x = idx % res.nx;
        let plane = idx / res.nx;
        let y = plane % res.ny;
        velocity.push(Vec3::new(-(y as f32), x as f32, 0.0));
    }
    // A uniform non-degenerate gradient along +X so every voxel normalizes.
    let magnitude_gradient = vec![Vec3::new(1.0, 0.0, 0.0); count];
    let inv_2h = 0.5;
    let query = GpuVorticityQuery {
        resolution: res,
        velocity,
        magnitude_gradient,
        params: VorticityParams { epsilon: 1.5 },
        inv_2h,
        cell_size: 1.0,
    };

    // Structural sanity: an interior voxel must see the analytic curl.z = 2
    // (central difference 4 * inv_2h), so the reference itself is non-trivial.
    let (cpu_curl, _) = cpu_reference(&query);
    let interior = res.linear_index(2, 2, 2) as usize;
    let expected_z = 4.0 * inv_2h;
    assert!(
        (cpu_curl[interior].z - expected_z).abs() <= ABS_EPS,
        "interior curl.z should be {expected_z}, got {}",
        cpu_curl[interior].z
    );

    check_scenario("rigid rotation", &engine, &ctx, &query);
}

#[test]
fn gpu_matches_cpu_flat_gradient_zero_force() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let engine = GpuFluidVorticity::new(&ctx);

    // A non-trivial velocity field (so curl is non-zero) but an exactly zero
    // gradient everywhere: both sides take the normalize zero branch, so the
    // force must be exactly zero while the curl still matches.
    let res = GridResolution::uniform(4);
    let count = res.voxel_count() as usize;
    let mut rng = Lcg::new(0x5EED);
    let velocity = rng.velocity_field(count, 2.0);
    let magnitude_gradient = vec![Vec3::ZERO; count];
    let query = GpuVorticityQuery {
        resolution: res,
        velocity,
        magnitude_gradient,
        params: VorticityParams { epsilon: 2.0 },
        inv_2h: 0.5,
        cell_size: 1.5,
    };

    let gpu = engine.evaluate(&ctx, &query);
    let (cpu_curl, _) = cpu_reference(&query);
    assert_parity("flat curl", &cpu_curl, &gpu.curl);
    for (i, f) in gpu.force.iter().enumerate() {
        for (axis, v) in [("x", f.x), ("y", f.y), ("z", f.z)] {
            assert!(
                v.abs() <= ABS_EPS,
                "flat gradient force cell {i} axis {axis} should be zero, got {v}"
            );
        }
    }
}

#[test]
fn gpu_matches_cpu_single_voxel() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let engine = GpuFluidVorticity::new(&ctx);

    // A lone voxel clamps every neighbor onto itself, so every central
    // difference cancels and the curl (hence the force) is zero. Both sides
    // must agree.
    let res = GridResolution::uniform(1);
    let query = GpuVorticityQuery {
        resolution: res,
        velocity: vec![Vec3::new(2.0, -3.0, 1.5)],
        magnitude_gradient: vec![Vec3::new(1.0, 1.0, 1.0)],
        params: VorticityParams { epsilon: 1.0 },
        inv_2h: 0.5,
        cell_size: 1.0,
    };
    check_scenario("single voxel", &engine, &ctx, &query);

    let gpu = engine.evaluate(&ctx, &query);
    for (axis, v) in [
        ("x", gpu.curl[0].x),
        ("y", gpu.curl[0].y),
        ("z", gpu.curl[0].z),
    ] {
        assert!(
            v.abs() <= ABS_EPS,
            "single-voxel curl axis {axis} should be zero, got {v}"
        );
    }
}

#[test]
fn gpu_matches_cpu_clamp_boundary() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let engine = GpuFluidVorticity::new(&ctx);

    // A 3x3x3 grid exercises every clamp face: the boundary voxels replicate
    // the edge cell, the center voxel sees true two-sided differences. Parity
    // across all 27 voxels proves the clamp convention matches the reference sampler.
    let res = GridResolution::uniform(3);
    let query = random_query(res, 0x7777, 1.3, 0.5, 1.0);
    check_scenario("clamp boundary", &engine, &ctx, &query);
}

#[test]
fn gpu_handles_degenerate_grids() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let engine = GpuFluidVorticity::new(&ctx);

    // A zero-extent grid: empty result, no dispatch.
    let empty_res = GridResolution::new(0, 4, 4);
    let empty = engine.evaluate(
        &ctx,
        &GpuVorticityQuery {
            resolution: empty_res,
            velocity: Vec::new(),
            magnitude_gradient: Vec::new(),
            params: VorticityParams { epsilon: 1.0 },
            inv_2h: 0.5,
            cell_size: 1.0,
        },
    );
    assert!(empty.curl.is_empty(), "zero-extent grid yields no curl");
    assert!(empty.force.is_empty(), "zero-extent grid yields no force");

    // A too-short velocity array: the guard returns empty rather than reading
    // past it.
    let res = GridResolution::uniform(3);
    let short = engine.evaluate(
        &ctx,
        &GpuVorticityQuery {
            resolution: res,
            velocity: vec![Vec3::ZERO; 3],
            magnitude_gradient: vec![Vec3::new(1.0, 0.0, 0.0); res.voxel_count() as usize],
            params: VorticityParams { epsilon: 1.0 },
            inv_2h: 0.5,
            cell_size: 1.0,
        },
    );
    assert!(short.curl.is_empty(), "too-short velocity yields no curl");
    assert!(short.force.is_empty(), "too-short velocity yields no force");

    // A too-short gradient array is guarded the same way.
    let short_grad = engine.evaluate(
        &ctx,
        &GpuVorticityQuery {
            resolution: res,
            velocity: vec![Vec3::ZERO; res.voxel_count() as usize],
            magnitude_gradient: vec![Vec3::new(1.0, 0.0, 0.0); 2],
            params: VorticityParams { epsilon: 1.0 },
            inv_2h: 0.5,
            cell_size: 1.0,
        },
    );
    assert!(
        short_grad.curl.is_empty() && short_grad.force.is_empty(),
        "too-short gradient yields no field"
    );
}
