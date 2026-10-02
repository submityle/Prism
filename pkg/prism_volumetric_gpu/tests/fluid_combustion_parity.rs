//! Real-device parity for the combustion-coupling twin: [`GpuFluidCombustion`]
//! must reproduce the `CPU` golden
//! [`step_combustion`](prism_render_architecture::particle::fluid::step_combustion),
//! [`buoyancy_force`](prism_render_architecture::particle::fluid::buoyancy_force),
//! and
//! [`heat_haze_distortion`](prism_render_architecture::particle::fluid::heat_haze_distortion)
//! across random voxel batches, ignited and quiescent states, the saturating
//! burn clamp, and the degenerate empty batch.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable
//! core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Every update is multiply-add only, with each expression accumulated in the
//! identical order the reference uses, so `CPU` and `GPU` evaluate the same
//! algebra. Values are asserted to within `abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3` — loose enough to admit a `GPU` fused multiply-add, yet
//! tight enough to fail a wrong port (a swapped term, a dropped clamp, a missing
//! cooling step).
//!
//! # Fixtures
//!
//! The deterministic `LCG` fixtures reject-sample away from the ignition and
//! burn-clamp branch thresholds so a tiny rounding difference can never flip a
//! branch; the dedicated ignited / quiescent / clamped scenarios then pin each
//! branch explicitly. The fixtures use no external math library and no
//! transcendental method.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::fluid` 的燃烧耦合
//! 纯函数真机 parity；无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::fluid::{
    buoyancy_force, heat_haze_distortion, step_combustion, CombustionParams, CombustionState,
};
use prism_render_architecture::particle::Vec3;
use prism_volumetric_gpu::fluid_combustion::{GpuFluidCombustion, GpuFluidCombustionQuery};
use prism_volumetric_gpu::GpuContext;

/// Absolute parity tolerance. Chosen a decade above the single multiply-add
/// rounding so a legal `GPU` fused multiply-add stays inside it while a
/// genuinely wrong port falls outside.
const ABS_EPS: f32 = 1.0e-4;

/// Relative parity tolerance, applied for samples large enough that the
/// absolute floor is pessimistic.
const REL_EPS: f32 = 1.0e-3;

/// Relative-tolerance floor so near-zero references do not divide by a tiny
/// magnitude.
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
        self.state = self
            .state
            .wrapping_mul(1_664_525)
            .wrapping_add(1_013_904_223);
        self.state
    }

    /// A reproducible `f32` in `[0, 1)`.
    fn next_unit(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }

    /// A reproducible `f32` in `[lo, hi)`.
    fn next_range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + self.next_unit() * (hi - lo)
    }
}

/// Asserts two scalars match within the documented tolerance.
fn assert_close(label: &str, cpu: f32, gpu: f32) {
    let abs_diff = (cpu - gpu).abs();
    let rel_diff = abs_diff / cpu.abs().max(REL_FLOOR);
    assert!(
        abs_diff <= ABS_EPS || rel_diff <= REL_EPS,
        "{label}: mismatch cpu {cpu}, gpu {gpu} (abs {abs_diff}, rel {rel_diff})"
    );
}

/// Asserts two [`Vec3`] match within the documented tolerance.
fn assert_vec3(label: &str, cpu: Vec3, gpu: Vec3) {
    assert_close(&format!("{label}.x"), cpu.x, gpu.x);
    assert_close(&format!("{label}.y"), cpu.y, gpu.y);
    assert_close(&format!("{label}.z"), cpu.z, gpu.z);
}

/// Runs one batch on the device and asserts every voxel matches the three
/// golden pure functions.
fn check_batch(
    label: &str,
    engine: &GpuFluidCombustion,
    ctx: &GpuContext,
    q: &GpuFluidCombustionQuery,
) {
    let gpu = engine.step(ctx, q);
    assert_eq!(gpu.states.len(), q.states.len(), "{label}: state count");
    assert_eq!(
        gpu.buoyancy.len(),
        q.states.len(),
        "{label}: buoyancy count"
    );
    assert_eq!(gpu.haze.len(), q.states.len(), "{label}: haze count");
    for i in 0..q.states.len() {
        let cpu_state = step_combustion(q.states[i], q.params, q.dt);
        let cpu_buoy = buoyancy_force(q.states[i], q.params);
        let cpu_haze = heat_haze_distortion(q.gradients[i], q.strengths[i]);
        assert_close(
            &format!("{label}[{i}].temperature"),
            cpu_state.temperature,
            gpu.states[i].temperature,
        );
        assert_close(
            &format!("{label}[{i}].fuel"),
            cpu_state.fuel,
            gpu.states[i].fuel,
        );
        assert_close(
            &format!("{label}[{i}].smoke"),
            cpu_state.smoke,
            gpu.states[i].smoke,
        );
        assert_vec3(&format!("{label}[{i}].buoyancy"), cpu_buoy, gpu.buoyancy[i]);
        assert_vec3(&format!("{label}[{i}].haze"), cpu_haze, gpu.haze[i]);
    }
}

/// A fixed, reasonable combustion parameter set used across the fixtures.
fn base_params() -> CombustionParams {
    CombustionParams {
        ignition_temperature: 2.0,
        burn_rate: 1.5,
        smoke_yield: 0.6,
        heat_yield: 1.2,
        cooling_rate: 0.3,
        ambient_temperature: 0.5,
        buoyancy_alpha: 0.8,
        buoyancy_beta: 0.4,
    }
}

/// Builds a reproducible batch of `count` voxels, reject-sampling the state so
/// no voxel sits within a margin of the ignition temperature or the burn clamp
/// threshold `burn_rate*dt`. Each voxel is independently either comfortably
/// ignited or comfortably quiescent.
fn random_batch(
    rng: &mut Lcg,
    count: usize,
    params: CombustionParams,
    dt: f32,
) -> GpuFluidCombustionQuery {
    // Margin keeping the state clear of both branch thresholds.
    let margin = 0.25f32;
    let clamp_threshold = params.burn_rate * dt;
    let mut states = Vec::with_capacity(count);
    let mut gradients = Vec::with_capacity(count);
    let mut strengths = Vec::with_capacity(count);
    for i in 0..count {
        // Alternate ignited / quiescent so both branches are exercised.
        let ignited = i % 2 == 0;
        let temperature = if ignited {
            rng.next_range(
                params.ignition_temperature + margin,
                params.ignition_temperature + 4.0,
            )
        } else {
            rng.next_range(
                params.ignition_temperature - 4.0,
                params.ignition_temperature - margin,
            )
        };
        // Keep fuel clear of 0 and of the clamp threshold so the clamp branch is
        // unambiguous; this scenario stays on the unclamped side.
        let fuel = rng.next_range(clamp_threshold + margin, clamp_threshold + 4.0);
        let smoke = rng.next_range(0.0, 3.0);
        states.push(CombustionState::new(temperature, fuel, smoke));
        gradients.push(Vec3::new(
            rng.next_range(-2.0, 2.0),
            rng.next_range(-2.0, 2.0),
            rng.next_range(-2.0, 2.0),
        ));
        strengths.push(rng.next_range(0.0, 2.0));
    }
    GpuFluidCombustionQuery {
        states,
        gradients,
        strengths,
        params,
        dt,
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_matches_cpu_across_random_batches() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping fluid-combustion parity: no wgpu adapter on this host");
        return;
    };
    let engine = GpuFluidCombustion::new(&ctx);
    let params = base_params();
    for (seed, (count, dt)) in [(37usize, 0.4f32), (128, 0.7), (257, 1.0), (64, 0.2)]
        .into_iter()
        .enumerate()
    {
        let mut rng = Lcg::new(0x51ED_u32.wrapping_add(seed as u32));
        let q = random_batch(&mut rng, count, params, dt);
        check_batch(
            &format!("random batch {seed} (count={count}, dt={dt})"),
            &engine,
            &ctx,
            &q,
        );
    }
}

#[test]
fn gpu_matches_cpu_on_ignited_voxels() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let engine = GpuFluidCombustion::new(&ctx);
    let params = base_params();
    let dt = 0.5;
    // All comfortably above ignition with plenty of fuel: the burn + cooling
    // path runs for every voxel.
    let states = vec![
        CombustionState::new(3.0, 2.0, 0.0),
        CombustionState::new(4.5, 3.0, 1.0),
        CombustionState::new(2.5, 5.0, 2.5),
    ];
    let gradients = vec![
        Vec3::new(1.0, -0.5, 0.25),
        Vec3::new(-2.0, 1.5, 0.0),
        Vec3::new(0.3, 0.3, -0.3),
    ];
    let strengths = vec![0.5, 1.25, 0.0];
    let q = GpuFluidCombustionQuery {
        states,
        gradients,
        strengths,
        params,
        dt,
    };
    check_batch("ignited", &engine, &ctx, &q);
}

#[test]
fn gpu_matches_cpu_on_quiescent_voxels() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let engine = GpuFluidCombustion::new(&ctx);
    let params = base_params();
    let dt = 0.75;
    // All comfortably below ignition: only the linear cooling term runs, fuel
    // and smoke are untouched.
    let states = vec![
        CombustionState::new(1.0, 1.0, 0.5),
        CombustionState::new(0.0, 0.0, 0.0),
        CombustionState::new(1.5, 4.0, 3.0),
    ];
    let gradients = vec![
        Vec3::new(0.0, 2.0, 0.0),
        Vec3::new(-1.0, -1.0, -1.0),
        Vec3::new(0.75, 0.0, 1.0),
    ];
    let strengths = vec![0.9, 0.1, 1.5];
    let q = GpuFluidCombustionQuery {
        states,
        gradients,
        strengths,
        params,
        dt,
    };
    check_batch("quiescent", &engine, &ctx, &q);
}

#[test]
fn gpu_matches_cpu_on_burn_clamp() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let engine = GpuFluidCombustion::new(&ctx);
    let params = base_params();
    // dt large enough that burn_rate*dt (= 1.5*3 = 4.5) exceeds the small fuel
    // left, so the saturating `burned > fuel` clamp fires. Fuel is kept clear of
    // the threshold by a wide margin so the branch is unambiguous.
    let dt = 3.0;
    let states = vec![
        CombustionState::new(3.0, 0.5, 0.0),
        CombustionState::new(5.0, 1.0, 2.0),
    ];
    let gradients = vec![Vec3::new(1.0, 1.0, 1.0), Vec3::new(-0.5, 0.5, -0.5)];
    let strengths = vec![0.5, 2.0];
    let q = GpuFluidCombustionQuery {
        states,
        gradients,
        strengths,
        params,
        dt,
    };
    check_batch("burn clamp", &engine, &ctx, &q);
}

#[test]
fn gpu_matches_cpu_on_single_voxel() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let engine = GpuFluidCombustion::new(&ctx);
    let params = base_params();
    let q = GpuFluidCombustionQuery {
        states: vec![CombustionState::new(3.5, 2.0, 1.0)],
        gradients: vec![Vec3::new(0.4, -0.8, 1.2)],
        strengths: vec![1.1],
        params,
        dt: 0.6,
    };
    check_batch("single voxel", &engine, &ctx, &q);
}

#[test]
fn gpu_handles_empty_batch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let engine = GpuFluidCombustion::new(&ctx);
    let q = GpuFluidCombustionQuery {
        states: Vec::new(),
        gradients: Vec::new(),
        strengths: Vec::new(),
        params: base_params(),
        dt: 0.5,
    };
    let out = engine.step(&ctx, &q);
    assert!(out.states.is_empty(), "empty batch must return no states");
    assert!(
        out.buoyancy.is_empty(),
        "empty batch must return no buoyancy"
    );
    assert!(out.haze.is_empty(), "empty batch must return no haze");
}
