//! Real-device parity for the `GPU` closed-mesh cloth-pressure kernel against
//! its `CPU` golden twin.
//!
//! Every test acquires a headless device with `GpuContext::try_headless()` and
//! skips cleanly when no adapter is available (for example inside a sandbox),
//! so the suite is a no-op rather than a failure off a real `GPU`.
//!
//! Pressure is a single global compliant-`XPBD` constraint over the whole
//! shell: the signed enclosed volume (divergence theorem) is driven toward a
//! target with one accumulated Lagrange multiplier, carried across all
//! iterations of the substep. The `CPU` golden ([`cpu_cloth_pressure`]) and the
//! device both delegate the arithmetic to the same `prism_physics_core`
//! projection, so the only divergence is the per-pass fused-multiply-add, tree
//! reduction, and division rounding; parity is checked within a tight relative
//! tolerance rather than bit-for-bit, matching the rest of the solver.
//!
//! Provenance: signed-volume-via-divergence-theorem pressure, its compliant
//! `XPBD` projection, and the `CSR` scatter->gather reformulation are standard
//! techniques. No Unreal Engine source or derived code.

use glam::Vec3;
use prism_physics_gpu::context::GpuContext;
use prism_physics_gpu::{cpu_cloth_pressure, GpuClothPressure};

/// A tiny deterministic `xorshift64*` generator so the tests need no external
/// crate for reproducible jitter.
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Rng {
        Rng { state: seed | 1 }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    /// A float in `[lo, hi)`.
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        let unit = (self.next_u64() >> 40) as f32 / (1_u64 << 24) as f32;
        lo + (hi - lo) * unit
    }
}

/// Relative per-component tolerance: the device sums the volume and denominator
/// through a workgroup tree reduction (a different summation order than the
/// sequential golden) and applies the correction with fused multiply-add, so
/// only the low bits can diverge.
const TOL: f32 = 1.0e-4;

fn close(a: Vec3, b: Vec3) -> bool {
    let scale = a.abs().max(b.abs()).max(Vec3::ONE);
    (a.x - b.x).abs() <= TOL * scale.x
        && (a.y - b.y).abs() <= TOL * scale.y
        && (a.z - b.z).abs() <= TOL * scale.z
}

/// The eight corners of the axis-aligned unit cube `[0,1]^3`.
fn unit_cube_positions() -> Vec<Vec3> {
    vec![
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(1.0, 1.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(1.0, 0.0, 1.0),
        Vec3::new(1.0, 1.0, 1.0),
        Vec3::new(0.0, 1.0, 1.0),
    ]
}

/// The twelve outward-wound triangles of the unit cube.
fn unit_cube_triangles() -> Vec<[u32; 3]> {
    vec![
        [0, 2, 1],
        [0, 3, 2],
        [4, 5, 6],
        [4, 6, 7],
        [0, 1, 5],
        [0, 5, 4],
        [3, 6, 2],
        [3, 7, 6],
        [0, 4, 7],
        [0, 7, 3],
        [1, 2, 6],
        [1, 6, 5],
    ]
}

/// Independent brute-force signed volume (divergence theorem) used to anchor the
/// inflation/deflation assertions against first-principles math.
fn brute_force_volume(positions: &[Vec3], triangles: &[[u32; 3]]) -> f32 {
    let mut sum = 0.0;
    for tri in triangles {
        let p0 = positions[tri[0] as usize];
        let p1 = positions[tri[1] as usize];
        let p2 = positions[tri[2] as usize];
        sum += p0.dot(p1.cross(p2));
    }
    sum / 6.0
}

fn assert_parity(label: &str, gpu: &[Vec3], cpu: &[Vec3]) {
    assert_eq!(gpu.len(), cpu.len(), "{label}: length mismatch");
    for (i, (g, c)) in gpu.iter().zip(cpu).enumerate() {
        assert!(
            close(*g, *c),
            "{label}: vertex {i} diverged: gpu={g:?} cpu={c:?}"
        );
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "skip notice is the standard pattern for GPU tests off a real device"
)]
fn gpu_matches_cpu_inflating_across_iterations() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping cloth_pressure_parity: no GPU adapter available");
        return;
    };
    let kernel = GpuClothPressure::new(&ctx);
    let positions = unit_cube_positions();
    let triangles = unit_cube_triangles();
    let rest = brute_force_volume(&positions, &triangles);
    let inverse_masses = vec![1.0_f32; positions.len()];
    let dt = 1.0 / 60.0;
    let target = 2.0 * rest;
    for iterations in [1_u32, 4, 16, 48] {
        let gpu = kernel.solve(
            &ctx,
            &positions,
            &inverse_masses,
            &triangles,
            target,
            0.0,
            dt,
            iterations,
        );
        let cpu = cpu_cloth_pressure(
            &positions,
            &inverse_masses,
            &triangles,
            target,
            0.0,
            dt,
            iterations,
        );
        assert_parity(&format!("inflate/iterations={iterations}"), &gpu, &cpu);
    }
    // Sanity: the device actually inflated the shell toward the target.
    let gpu = kernel.solve(
        &ctx,
        &positions,
        &inverse_masses,
        &triangles,
        target,
        0.0,
        dt,
        64,
    );
    let inflated = brute_force_volume(&gpu, &triangles);
    assert!(inflated > rest + 0.1, "device did not inflate: {inflated}");
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "skip notice is the standard pattern for GPU tests off a real device"
)]
fn gpu_matches_cpu_deflating_and_compliant() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping cloth_pressure_parity: no GPU adapter available");
        return;
    };
    let kernel = GpuClothPressure::new(&ctx);
    let positions = unit_cube_positions();
    let triangles = unit_cube_triangles();
    let rest = brute_force_volume(&positions, &triangles);
    let inverse_masses = vec![1.0_f32; positions.len()];
    let dt = 1.0 / 60.0;
    // Deflate toward half volume, with a non-zero compliance to exercise the
    // alpha_tilde term in the denominator and the lambda feedback.
    let target = 0.5 * rest;
    for compliance in [0.0_f32, 1.0e-4, 1.0e-2] {
        for iterations in [1_u32, 8, 32] {
            let gpu = kernel.solve(
                &ctx,
                &positions,
                &inverse_masses,
                &triangles,
                target,
                compliance,
                dt,
                iterations,
            );
            let cpu = cpu_cloth_pressure(
                &positions,
                &inverse_masses,
                &triangles,
                target,
                compliance,
                dt,
                iterations,
            );
            assert_parity(
                &format!("deflate/compliance={compliance}/iterations={iterations}"),
                &gpu,
                &cpu,
            );
        }
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "skip notice is the standard pattern for GPU tests off a real device"
)]
fn gpu_matches_cpu_with_pinned_vertices() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping cloth_pressure_parity: no GPU adapter available");
        return;
    };
    let kernel = GpuClothPressure::new(&ctx);
    let positions = unit_cube_positions();
    let triangles = unit_cube_triangles();
    let rest = brute_force_volume(&positions, &triangles);
    // Pin the bottom face (vertices 0..=3); only the top four may move.
    let inverse_masses = vec![0.0_f32, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
    let dt = 1.0 / 60.0;
    let target = 2.5 * rest;
    let gpu = kernel.solve(
        &ctx,
        &positions,
        &inverse_masses,
        &triangles,
        target,
        0.0,
        dt,
        32,
    );
    let cpu = cpu_cloth_pressure(
        &positions,
        &inverse_masses,
        &triangles,
        target,
        0.0,
        dt,
        32,
    );
    assert_parity("pinned", &gpu, &cpu);
    for p in 0..4 {
        assert!(close(gpu[p], positions[p]), "pinned vertex {p} moved");
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "skip notice is the standard pattern for GPU tests off a real device"
)]
fn gpu_matches_cpu_on_jittered_shell() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping cloth_pressure_parity: no GPU adapter available");
        return;
    };
    let kernel = GpuClothPressure::new(&ctx);
    let triangles = unit_cube_triangles();
    let mut rng = Rng::new(0x9e37_1234);
    // Randomly perturb the cube corners so the gradients are irregular and the
    // reduction order sensitivity is exercised.
    let positions: Vec<Vec3> = unit_cube_positions()
        .into_iter()
        .map(|p| p + Vec3::new(rng.range(-0.3, 0.3), rng.range(-0.3, 0.3), rng.range(-0.3, 0.3)))
        .collect();
    let rest = brute_force_volume(&positions, &triangles);
    let inverse_masses: Vec<f32> = (0..positions.len())
        .map(|_| rng.range(0.5, 2.0))
        .collect();
    let dt = 1.0 / 90.0;
    let target = 1.6 * rest;
    for iterations in [1_u32, 6, 24] {
        let gpu = kernel.solve(
            &ctx,
            &positions,
            &inverse_masses,
            &triangles,
            target,
            5.0e-4,
            dt,
            iterations,
        );
        let cpu = cpu_cloth_pressure(
            &positions,
            &inverse_masses,
            &triangles,
            target,
            5.0e-4,
            dt,
            iterations,
        );
        assert_parity(&format!("jittered/iterations={iterations}"), &gpu, &cpu);
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "skip notice is the standard pattern for GPU tests off a real device"
)]
fn gpu_empty_or_zero_is_noop() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping cloth_pressure_parity: no GPU adapter available");
        return;
    };
    let kernel = GpuClothPressure::new(&ctx);
    let positions = unit_cube_positions();
    let triangles = unit_cube_triangles();
    let inverse_masses = vec![1.0_f32; positions.len()];
    let dt = 1.0 / 60.0;

    let no_tris = kernel.solve(&ctx, &positions, &inverse_masses, &[], 2.0, 0.0, dt, 8);
    assert_eq!(no_tris, positions, "empty triangle set must be a no-op");

    let zero_iter = kernel.solve(
        &ctx,
        &positions,
        &inverse_masses,
        &triangles,
        2.0,
        0.0,
        dt,
        0,
    );
    assert_eq!(zero_iter, positions, "zero iterations must be a no-op");

    let bad_dt = kernel.solve(
        &ctx,
        &positions,
        &inverse_masses,
        &triangles,
        2.0,
        0.0,
        0.0,
        8,
    );
    assert_eq!(bad_dt, positions, "non-positive dt must be a no-op");
}
