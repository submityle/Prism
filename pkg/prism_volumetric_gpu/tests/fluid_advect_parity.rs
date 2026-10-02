//! Real-device parity for the semi-Lagrangian advection twin:
//! [`GpuFluidAdvect`](prism_volumetric_gpu::fluid_advect::GpuFluidAdvect) must
//! reproduce the `CPU` golden
//! [`fluid`](prism_render_architecture::particle::fluid) advection cluster
//! value for value — the back-trace, the trilinear field sample, the in-field
//! advect, the `MacCormack` correction and the eight trilinear corner weights.
//!
//! The fixtures cover the degenerate and interior shapes the golden unit tests
//! exercise: an empty query batch (no dispatch), a degenerate field (zero-voxel
//! grid and a short field, both sampled as the zero vector), per-function
//! parity on small grids with a deterministic ramp field, clamp-to-edge
//! addressing for out-of-range positions, and a larger deterministic field swept
//! at many fractional positions. Interior sample positions stay at clearly
//! fractional coordinates, away from exact integer voxel boundaries, so the
//! `floor`-based base-voxel split is never evaluated at a branch-critical point.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each output is a fixed sequence of multiply-adds plus the sampler's `floor`
//! and integer `clamp`, so `CPU` and `GPU` evaluate the same closed form but need
//! not be bit-exact (a `GPU` may contract a multiply-add). The comparison is per
//! component with `abs_diff <= 1e-4 || rel_diff <= 1e-3` (`REL_FLOOR = 1e-6`),
//! tight enough to catch a genuinely wrong port yet loose enough to admit legal
//! fused multiply-add contraction.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::fluid` 的半拉格朗日
//! 平流纯函数簇；无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::fluid::{
    advect_particle_in_field, maccormack_corrected, sample_velocity_field,
    semi_lagrangian_backtrace, trilinear_weights, GridResolution,
};
use prism_render_architecture::particle::Vec3;
use prism_volumetric_gpu::fluid_advect::{GpuAdvectQuery, GpuAdvectResult, GpuFluidAdvect};
use prism_volumetric_gpu::GpuContext;

/// Absolute tolerance for the continuous-quantity parity comparison.
const ABS_TOL: f32 = 1.0e-4;
/// Relative tolerance for the continuous-quantity parity comparison.
const REL_TOL: f32 = 1.0e-3;
/// Floor on the relative-tolerance denominator so values near zero still use a
/// meaningful scale.
const REL_FLOOR: f32 = 1.0e-6;

/// Returns `true` when two continuous values agree within the module tolerance.
fn close(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    if diff <= ABS_TOL {
        return true;
    }
    let scale = a.abs().max(b.abs()).max(REL_FLOOR);
    diff / scale <= REL_TOL
}

/// Asserts two vectors agree component by component within tolerance.
fn assert_vec3_close(gpu: Vec3, cpu: Vec3, label: &str) {
    assert!(
        close(gpu.x, cpu.x) && close(gpu.y, cpu.y) && close(gpu.z, cpu.z),
        "{label}: GPU ({}, {}, {}) vs CPU ({}, {}, {})",
        gpu.x,
        gpu.y,
        gpu.z,
        cpu.x,
        cpu.y,
        cpu.z
    );
}

/// A small integer `LCG` used to synthesize deterministic field and query data
/// without any external math library or transcendental call.
struct Lcg {
    /// The current `64`-bit state.
    state: u64,
}

impl Lcg {
    /// Seeds the generator, forcing an odd state so the stream never degenerates.
    fn new(seed: u64) -> Self {
        Lcg { state: seed | 1 }
    }

    /// Advances the state and returns the high `32` bits.
    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.state >> 32) as u32
    }

    /// A deterministic signed fraction in roughly `[-5, 5)`, pure integer math
    /// scaled to `f32` with no transcendental call.
    fn next_signed(&mut self) -> f32 {
        let n = (self.next_u32() % 1000) as f32;
        n * 0.01 - 5.0
    }
}

/// Builds a deterministic row-major velocity field of `res.voxel_count()`
/// samples from an integer stream.
fn make_field(res: GridResolution, seed: u64) -> Vec<Vec3> {
    let mut rng = Lcg::new(seed);
    let count = res.voxel_count() as usize;
    let mut field = Vec::with_capacity(count);
    for _ in 0..count {
        field.push(Vec3::new(
            rng.next_signed(),
            rng.next_signed(),
            rng.next_signed(),
        ));
    }
    field
}

/// Replicates the golden clamp-and-floor to recover the in-cell fraction, so the
/// returned corner weights can be validated independently of the sampler.
fn expected_frac(pos: Vec3, res: GridResolution) -> Vec3 {
    let max_x = (res.nx - 1) as f32;
    let max_y = (res.ny - 1) as f32;
    let max_z = (res.nz - 1) as f32;
    let cx = pos.x.clamp(0.0, max_x);
    let cy = pos.y.clamp(0.0, max_y);
    let cz = pos.z.clamp(0.0, max_z);
    Vec3::new(cx - cx.floor(), cy - cy.floor(), cz - cz.floor())
}

/// Builds a query whose five input vectors are distinct so a swapped field is
/// caught; `MacCormack` inputs are offset from the position so the correction is
/// non-trivial.
fn make_query(pos: Vec3, velocity: Vec3, rng: &mut Lcg) -> GpuAdvectQuery {
    GpuAdvectQuery {
        pos,
        velocity,
        mac_forward: Vec3::new(rng.next_signed(), rng.next_signed(), rng.next_signed()),
        mac_original: Vec3::new(rng.next_signed(), rng.next_signed(), rng.next_signed()),
        mac_back: Vec3::new(rng.next_signed(), rng.next_signed(), rng.next_signed()),
    }
}

/// Asserts every result in a batch matches the golden functions, optionally
/// including the trilinear-weight check (skipped for a degenerate field).
fn assert_batch_parity(
    results: &[GpuAdvectResult],
    queries: &[GpuAdvectQuery],
    field: &[Vec3],
    res: GridResolution,
    dt: f32,
    check_weights: bool,
    label: &str,
) {
    assert_eq!(
        results.len(),
        queries.len(),
        "{label}: result count mismatch"
    );
    for (i, (r, q)) in results.iter().zip(queries.iter()).enumerate() {
        let cpu_backtrace = semi_lagrangian_backtrace(q.pos, q.velocity, dt);
        let cpu_sampled = sample_velocity_field(field, res, q.pos);
        let cpu_advected = advect_particle_in_field(q.pos, field, res, dt);
        let cpu_mac = maccormack_corrected(q.mac_forward, q.mac_original, q.mac_back);
        assert_vec3_close(
            r.backtrace,
            cpu_backtrace,
            &format!("{label}[{i}] backtrace"),
        );
        assert_vec3_close(r.sampled, cpu_sampled, &format!("{label}[{i}] sampled"));
        assert_vec3_close(r.advected, cpu_advected, &format!("{label}[{i}] advected"));
        assert_vec3_close(r.maccormack, cpu_mac, &format!("{label}[{i}] maccormack"));
        if check_weights {
            let cpu_weights = trilinear_weights(expected_frac(q.pos, res));
            for (k, (&gpu_w, &cpu_w)) in r.weights.iter().zip(cpu_weights.iter()).enumerate() {
                assert!(
                    close(gpu_w, cpu_w),
                    "{label}[{i}] weight {k}: GPU {} vs CPU {}",
                    gpu_w,
                    cpu_w
                );
            }
        }
    }
}

#[test]
fn empty_query_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let advect = GpuFluidAdvect::new(&ctx);
    let res = GridResolution::new(4, 4, 4);
    let field = make_field(res, 0x1234);
    let out = advect.advect(&ctx, &field, res, 0.1, &[]);
    assert!(out.is_empty(), "empty batch should produce no results");
}

#[test]
fn interior_sample_parity_small_grid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let advect = GpuFluidAdvect::new(&ctx);
    let res = GridResolution::new(4, 4, 4);
    let field = make_field(res, 0xA5A5);
    let dt = 0.1;
    let mut rng = Lcg::new(0x9001);
    // Interior fractional positions well away from integer voxel boundaries.
    let positions = [
        Vec3::new(1.3, 0.7, 2.4),
        Vec3::new(2.6, 1.1, 0.8),
        Vec3::new(0.4, 2.9, 1.6),
        Vec3::new(2.2, 2.3, 2.7),
    ];
    let queries: Vec<GpuAdvectQuery> = positions
        .iter()
        .map(|&p| {
            let v = Vec3::new(rng.next_signed(), rng.next_signed(), rng.next_signed());
            make_query(p, v, &mut rng)
        })
        .collect();
    let out = advect.advect(&ctx, &field, res, dt, &queries);
    assert_batch_parity(&out, &queries, &field, res, dt, true, "interior");
}

#[test]
fn clamp_to_edge_parity() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let advect = GpuFluidAdvect::new(&ctx);
    let res = GridResolution::new(3, 2, 5);
    let field = make_field(res, 0x7777);
    let dt = 0.05;
    let mut rng = Lcg::new(0x2024);
    // Positions clearly outside every face so clamp-to-edge is exercised on both
    // the low and high side of each axis.
    let positions = [
        Vec3::new(-3.5, -2.0, -1.5),
        Vec3::new(10.0, 9.0, 20.0),
        Vec3::new(-1.0, 8.0, 1.4),
        Vec3::new(5.0, -4.0, 3.6),
    ];
    let queries: Vec<GpuAdvectQuery> = positions
        .iter()
        .map(|&p| {
            let v = Vec3::new(rng.next_signed(), rng.next_signed(), rng.next_signed());
            make_query(p, v, &mut rng)
        })
        .collect();
    let out = advect.advect(&ctx, &field, res, dt, &queries);
    assert_batch_parity(&out, &queries, &field, res, dt, true, "clamp");
}

#[test]
fn degenerate_zero_voxel_grid_samples_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let advect = GpuFluidAdvect::new(&ctx);
    // A zero-voxel grid: the field is invalid, so sampled is the zero vector and
    // advected is the untouched position.
    let res = GridResolution::new(0, 4, 4);
    let field: Vec<Vec3> = Vec::new();
    let dt = 0.2;
    let mut rng = Lcg::new(0x55AA);
    let queries = [
        make_query(
            Vec3::new(1.3, 0.7, 2.4),
            Vec3::new(0.5, -0.5, 0.25),
            &mut rng,
        ),
        make_query(
            Vec3::new(-2.0, 3.1, 0.9),
            Vec3::new(-1.0, 2.0, -3.0),
            &mut rng,
        ),
    ];
    let out = advect.advect(&ctx, &field, res, dt, &queries);
    // No weights check: the golden sampler never computes weights on this path.
    assert_batch_parity(&out, &queries, &field, res, dt, false, "zero_voxel");
    for (i, r) in out.iter().enumerate() {
        assert_vec3_close(r.sampled, Vec3::ZERO, &format!("zero_voxel[{i}] sampled"));
        assert_vec3_close(
            r.advected,
            queries[i].pos,
            &format!("zero_voxel[{i}] advected"),
        );
    }
}

#[test]
fn degenerate_short_field_samples_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let advect = GpuFluidAdvect::new(&ctx);
    let res = GridResolution::new(4, 4, 4);
    // Fewer samples than voxel_count marks the field invalid.
    let field = vec![Vec3::new(1.0, 2.0, 3.0); 10];
    let dt = 0.15;
    let mut rng = Lcg::new(0xBEEF);
    let queries = [
        make_query(
            Vec3::new(1.6, 2.4, 0.8),
            Vec3::new(1.0, 0.0, -1.0),
            &mut rng,
        ),
        make_query(
            Vec3::new(2.2, 1.3, 3.1),
            Vec3::new(-2.0, 1.5, 0.5),
            &mut rng,
        ),
    ];
    let out = advect.advect(&ctx, &field, res, dt, &queries);
    assert_batch_parity(&out, &queries, &field, res, dt, false, "short_field");
    for (i, r) in out.iter().enumerate() {
        assert_vec3_close(r.sampled, Vec3::ZERO, &format!("short_field[{i}] sampled"));
        assert_vec3_close(
            r.advected,
            queries[i].pos,
            &format!("short_field[{i}] advected"),
        );
    }
}

#[test]
fn large_deterministic_sweep_parity() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let advect = GpuFluidAdvect::new(&ctx);
    let res = GridResolution::new(8, 6, 7);
    let field = make_field(res, 0xC0FFEE);
    let dt = 0.0375;
    let mut rng = Lcg::new(0x1357);
    let max_x = (res.nx - 1) as f32;
    let max_y = (res.ny - 1) as f32;
    let max_z = (res.nz - 1) as f32;
    let mut queries = Vec::with_capacity(200);
    for _ in 0..200 {
        // Interior fractional positions nudged by 0.5 so they never land exactly
        // on an integer voxel boundary.
        let fx = (rng.next_u32() % ((res.nx - 1) * 10)) as f32 * 0.1;
        let fy = (rng.next_u32() % ((res.ny - 1) * 10)) as f32 * 0.1;
        let fz = (rng.next_u32() % ((res.nz - 1) * 10)) as f32 * 0.1;
        let pos = Vec3::new(
            (fx + 0.05).clamp(0.0, max_x),
            (fy + 0.05).clamp(0.0, max_y),
            (fz + 0.05).clamp(0.0, max_z),
        );
        let v = Vec3::new(rng.next_signed(), rng.next_signed(), rng.next_signed());
        queries.push(make_query(pos, v, &mut rng));
    }
    let out = advect.advect(&ctx, &field, res, dt, &queries);
    assert_batch_parity(&out, &queries, &field, res, dt, true, "sweep");
}
