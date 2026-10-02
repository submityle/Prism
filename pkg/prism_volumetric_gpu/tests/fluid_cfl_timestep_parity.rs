//! Real-device parity for the fluid `CFL`/timestep/advection twin:
//! [`GpuFluidCflTimestep`](prism_volumetric_gpu::fluid_cfl_timestep::GpuFluidCflTimestep)
//! must reproduce the `CPU` golden
//! [`cfl_number`](prism_render_architecture::particle::fluid::cfl_number),
//! [`stable_timestep`](prism_render_architecture::particle::fluid::stable_timestep)
//! and
//! [`advect_particle`](prism_render_architecture::particle::fluid::advect_particle)
//! across an empty batch, a single query, a large pseudo-random batch, negative
//! positions, velocities and steps, and the two degenerate denominators the
//! reference guards (a cell size at zero, a still field, and both at once).
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-`WGSL`,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each query is a fixed, non-reorderable pair of guarded divisions plus a
//! multiply-add, so `CPU` and `GPU` evaluate the same closed form. They are not
//! bit-exact: a `GPU` may fuse a multiply-add the scalar reference leaves
//! separate, perturbing the low mantissa bits by a few units in the last place.
//! The comparison therefore allows `abs_diff <= 1e-4` or `rel_diff <= 1e-3` on
//! the continuous values. The degenerate branch is pinned by fixtures that keep
//! every denominator either exactly zero or far above the golden `EPS_LEN_SQ`
//! floor, so the device and the reference always take the same branch and the
//! zero-return cases match within tolerance of exactly zero.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::fluid`；无第三方
//! 引擎源码或衍生代码。

use prism_render_architecture::particle::fluid::{advect_particle, cfl_number, stable_timestep};
use prism_render_architecture::particle::Vec3;
use prism_volumetric_gpu::fluid_cfl_timestep::{GpuCflTimestepQuery, GpuFluidCflTimestep};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity bound on the continuous values. A `GPU` may fuse a
/// multiply-add the scalar reference leaves separate, perturbing the low
/// mantissa bits by a few units in the last place; `1e-4` admits that legal
/// slack while still failing a genuinely wrong port.
const ABS_EPS: f32 = 1.0e-4;

/// Relative parity bound, applied for larger magnitudes where a few units in the
/// last place exceed the absolute floor.
const REL_EPS: f32 = 1.0e-3;

/// Floor on the relative-error denominator so a near-zero expected value does
/// not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// A tiny deterministic linear-congruential generator so the "random" fixtures
/// are reproducible run to run without pulling in an external crate. The
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
        // Numerical Recipes constants; wrapping arithmetic keeps it in range.
        self.state = self
            .state
            .wrapping_mul(1_664_525)
            .wrapping_add(1_013_904_223);
        self.state
    }

    /// A reproducible `f32` in `[-range, range]`.
    fn next_signed(&mut self, range: f32) -> f32 {
        // Map the top 24 bits into [0, 1), then into the symmetric range.
        let unit = (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32;
        (unit * 2.0 - 1.0) * range
    }

    /// A reproducible non-zero magnitude in `[min_mag, min_mag + range]` with a
    /// pseudo-random sign, used for denominators kept far from the golden
    /// `EPS_LEN_SQ` floor so the device and the reference take the same branch.
    fn next_denominator(&mut self, min_mag: f32, range: f32) -> f32 {
        let unit = (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32;
        let mag = min_mag + unit * range;
        if self.next_u32() & 1 == 0 {
            mag
        } else {
            -mag
        }
    }

    /// A reproducible position with components in `[-range, range]`.
    fn next_vec3(&mut self, range: f32) -> Vec3 {
        Vec3::new(
            self.next_signed(range),
            self.next_signed(range),
            self.next_signed(range),
        )
    }
}

/// Asserts two scalars match to within the documented tolerance.
fn assert_scalar(label: &str, cpu: f32, gpu: f32) {
    let abs_diff = (cpu - gpu).abs();
    let rel_diff = abs_diff / cpu.abs().max(REL_FLOOR);
    assert!(
        abs_diff <= ABS_EPS || rel_diff <= REL_EPS,
        "{label}: cpu {cpu}, gpu {gpu} (abs {abs_diff}, rel {rel_diff})"
    );
}

/// Asserts two positions match component for component to within tolerance.
fn assert_vec3(label: &str, cpu: Vec3, gpu: Vec3) {
    for (cv, gv, axis) in [
        (cpu.x, gpu.x, "x"),
        (cpu.y, gpu.y, "y"),
        (cpu.z, gpu.z, "z"),
    ] {
        assert_scalar(&format!("{label} axis {axis}"), cv, gv);
    }
}

/// Runs the whole batch on the device and asserts every lane matches the three
/// golden functions evaluated on that same query, in input order.
fn check_batch(
    label: &str,
    engine: &GpuFluidCflTimestep,
    ctx: &GpuContext,
    queries: &[GpuCflTimestepQuery],
) {
    let gpu = engine.evaluate(ctx, queries);
    assert_eq!(gpu.len(), queries.len(), "{label}: result length mismatch");
    for (i, (q, g)) in queries.iter().zip(gpu.iter()).enumerate() {
        let cpu_cfl = cfl_number(q.max_velocity, q.dt, q.cell_size);
        let cpu_stable = stable_timestep(q.max_velocity, q.cell_size, q.cfl_target);
        let cpu_adv = advect_particle(q.pos, q.velocity, q.dt);
        assert_scalar(&format!("{label} query {i} cfl"), cpu_cfl, g.cfl);
        assert_scalar(
            &format!("{label} query {i} stable_dt"),
            cpu_stable,
            g.stable_dt,
        );
        assert_vec3(
            &format!("{label} query {i} advected_pos"),
            cpu_adv,
            g.advected_pos,
        );
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "a skip notice on headless hosts keeps the suite green and visibly skipped"
)]
fn gpu_matches_cpu_across_random_batch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping fluid-cfl-timestep parity: no wgpu adapter on this host");
        return;
    };
    let engine = GpuFluidCflTimestep::new(&ctx);
    let mut rng = Lcg::new(0x0CF1_7E57);

    // A large pseudo-random batch with both denominators kept far above the
    // golden EPS_LEN_SQ floor (|cell_size|, |max_velocity| >= 0.1), so every
    // lane takes the division branch and the multiply-add advection.
    let queries: Vec<GpuCflTimestepQuery> = (0..4096)
        .map(|_| GpuCflTimestepQuery {
            max_velocity: rng.next_denominator(0.1, 20.0),
            dt: rng.next_signed(0.5),
            cell_size: rng.next_denominator(0.1, 5.0),
            cfl_target: rng.next_signed(1.5),
            pos: rng.next_vec3(50.0),
            velocity: rng.next_vec3(30.0),
        })
        .collect();

    check_batch("random batch", &engine, &ctx, &queries);
}

#[test]
fn gpu_matches_cpu_single_query() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let engine = GpuFluidCflTimestep::new(&ctx);

    // A single hand-chosen query exercises the one-thread dispatch path.
    let queries = vec![GpuCflTimestepQuery {
        max_velocity: 4.0,
        dt: 0.25,
        cell_size: 2.0,
        cfl_target: 0.8,
        pos: Vec3::new(1.5, -2.0, 3.0),
        velocity: Vec3::new(-0.5, 1.0, 2.0),
    }];

    check_batch("single query", &engine, &ctx, &queries);
}

#[test]
fn gpu_matches_cpu_negative_inputs() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let engine = GpuFluidCflTimestep::new(&ctx);

    // Negative positions, velocities and step keep the signs flowing through
    // the guarded divisions and the multiply-add; both denominators stay far
    // from the floor.
    let queries = vec![
        GpuCflTimestepQuery {
            max_velocity: -6.0,
            dt: -0.3,
            cell_size: -1.5,
            cfl_target: -0.9,
            pos: Vec3::new(-4.0, -5.0, -6.0),
            velocity: Vec3::new(-1.0, -2.0, -3.0),
        },
        GpuCflTimestepQuery {
            max_velocity: 3.0,
            dt: -0.4,
            cell_size: -2.5,
            cfl_target: 1.1,
            pos: Vec3::new(7.0, -8.0, 9.0),
            velocity: Vec3::new(2.0, -1.0, 0.5),
        },
    ];

    check_batch("negative inputs", &engine, &ctx, &queries);
}

#[test]
fn gpu_matches_cpu_degenerate_cell_size() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let engine = GpuFluidCflTimestep::new(&ctx);

    // A cell size of exactly zero is at or below the golden EPS_LEN_SQ floor, so
    // the CFL number must be exactly zero (the reference returns 0.0 rather than
    // dividing by zero). The max velocity stays far above the floor so the
    // stable step still takes its division branch.
    let queries = vec![GpuCflTimestepQuery {
        max_velocity: 5.0,
        dt: 0.2,
        cell_size: 0.0,
        cfl_target: 0.7,
        pos: Vec3::new(2.0, 3.0, -1.0),
        velocity: Vec3::new(1.0, -1.0, 2.0),
    }];

    let gpu = engine.evaluate(&ctx, &queries);
    assert_eq!(gpu.len(), 1, "degenerate cell size: result length");
    // Pin the zero branch: the golden returns exactly 0.0, the device matches.
    assert_scalar("degenerate cell size cfl", 0.0, gpu[0].cfl);
    assert_eq!(gpu[0].cfl, 0.0, "degenerate cell size: cfl must be zero");
    check_batch("degenerate cell size", &engine, &ctx, &queries);
}

#[test]
fn gpu_matches_cpu_degenerate_max_velocity() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let engine = GpuFluidCflTimestep::new(&ctx);

    // A max velocity of exactly zero is at or below the floor, so the stable
    // step must be exactly zero (a still field yields a zero step). The cell
    // size stays far above the floor so the CFL number still divides.
    let queries = vec![GpuCflTimestepQuery {
        max_velocity: 0.0,
        dt: 0.3,
        cell_size: 1.5,
        cfl_target: 0.9,
        pos: Vec3::new(-2.0, 1.0, 4.0),
        velocity: Vec3::new(0.5, 0.5, -0.5),
    }];

    let gpu = engine.evaluate(&ctx, &queries);
    assert_eq!(gpu.len(), 1, "degenerate max velocity: result length");
    // Pin the zero branch: the golden returns exactly 0.0, the device matches.
    assert_scalar("degenerate max velocity stable_dt", 0.0, gpu[0].stable_dt);
    assert_eq!(
        gpu[0].stable_dt, 0.0,
        "degenerate max velocity: stable_dt must be zero"
    );
    check_batch("degenerate max velocity", &engine, &ctx, &queries);
}

#[test]
fn gpu_matches_cpu_both_denominators_degenerate() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let engine = GpuFluidCflTimestep::new(&ctx);

    // Both denominators at exactly zero: the CFL number and the stable step are
    // both zero, while the advection (a pure multiply-add with dt) still runs.
    let queries = vec![GpuCflTimestepQuery {
        max_velocity: 0.0,
        dt: 0.5,
        cell_size: 0.0,
        cfl_target: 1.0,
        pos: Vec3::new(3.0, -3.0, 6.0),
        velocity: Vec3::new(2.0, 4.0, -2.0),
    }];

    let gpu = engine.evaluate(&ctx, &queries);
    assert_eq!(gpu.len(), 1, "both degenerate: result length");
    assert_eq!(gpu[0].cfl, 0.0, "both degenerate: cfl must be zero");
    assert_eq!(
        gpu[0].stable_dt, 0.0,
        "both degenerate: stable_dt must be zero"
    );
    check_batch("both degenerate", &engine, &ctx, &queries);
}

#[test]
fn gpu_matches_cpu_mixed_degenerate_and_normal_batch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let engine = GpuFluidCflTimestep::new(&ctx);
    let mut rng = Lcg::new(0xDEAD_BEEF);

    // A batch that interleaves degenerate denominators (exact zero) with normal
    // ones, proving each lane picks its own branch independently.
    let mut queries = Vec::new();
    for i in 0..256 {
        let degenerate_cell = i % 3 == 0;
        let degenerate_vel = i % 4 == 0;
        queries.push(GpuCflTimestepQuery {
            max_velocity: if degenerate_vel {
                0.0
            } else {
                rng.next_denominator(0.1, 10.0)
            },
            dt: rng.next_signed(0.5),
            cell_size: if degenerate_cell {
                0.0
            } else {
                rng.next_denominator(0.1, 3.0)
            },
            cfl_target: rng.next_signed(1.0),
            pos: rng.next_vec3(20.0),
            velocity: rng.next_vec3(15.0),
        });
    }

    check_batch("mixed batch", &engine, &ctx, &queries);
}

#[test]
fn gpu_empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let engine = GpuFluidCflTimestep::new(&ctx);

    // An empty batch is a host short-circuit: no dispatch, an empty result.
    let gpu = engine.evaluate(&ctx, &[]);
    assert!(gpu.is_empty(), "empty batch must yield an empty result");
}
