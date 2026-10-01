//! Real-device parity for the two-pass `GPU` cloth aerodynamics kernel against
//! its `CPU` golden twin.
//!
//! Every test acquires a headless device with `GpuContext::try_headless()` and
//! skips cleanly when no adapter is available (for example inside a sandbox),
//! so the suite is a no-op rather than a failure off a real `GPU`.
//!
//! The pass is the Jacobi reformulation of `prism_physics_core`'s sequential
//! `apply_aero_forces`: triangle filtering, the per-triangle wind (steady field
//! plus integer-hashed turbulence), and the per-vertex incidence `CSR` are
//! integer-exact and identical on both sides, so the only divergence between
//! device and twin is the per-face drag/lift fused-multiply-add and
//! division/square-root rounding; parity is therefore checked within a tight
//! tolerance rather than bit-for-bit, the same model the rest of the solver
//! uses.
//!
//! Provenance: the per-triangle drag/lift decomposition and the optional
//! quadratic dynamic-pressure term are the standard, publicly documented cloth
//! aerodynamics model. No Unreal Engine source or derived code.

use glam::Vec3;
use prism_physics_gpu::context::GpuContext;
use prism_physics_gpu::{cpu_cloth_aero, ClothAeroParams, ClothAeroTriangle, GpuClothAero};

/// A tiny deterministic `xorshift64*` generator so the tests need no external
/// crate for reproducible fold jitter.
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

/// Per-component tolerance: each face force is a normalise (division and square
/// root, with `GPU` fused multiply-add in the dot products) scaled by area and
/// summed across incident faces, so the device and twin agree only to a few
/// `ULP` of the applied velocity increment.
const TOL: f32 = 1.0e-4;

fn close(a: Vec3, b: Vec3) -> bool {
    let scale = a.abs().max(b.abs()).max(Vec3::ONE);
    (a.x - b.x).abs() <= TOL * scale.x
        && (a.y - b.y).abs() <= TOL * scale.y
        && (a.z - b.z).abs() <= TOL * scale.z
}

/// Builds a `rows x cols` cloth grid lying in the `xy` plane with a seeded `z`
/// fold jitter and a seeded initial velocity, plus the two triangles per quad
/// that catch the wind. Returns `(positions, velocities, inverse_masses,
/// triangles)`.
fn cloth_grid(
    rows: usize,
    cols: usize,
    seed: u64,
) -> (Vec<Vec3>, Vec<Vec3>, Vec<f32>, Vec<ClothAeroTriangle>) {
    let mut rng = Rng::new(seed);
    let mut positions = Vec::with_capacity(rows * cols);
    let mut velocities = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        for c in 0..cols {
            positions.push(Vec3::new(
                c as f32,
                r as f32,
                // A real out-of-plane fold so faces present varied normals.
                rng.range(-0.5, 0.5),
            ));
            velocities.push(Vec3::new(
                rng.range(-0.3, 0.3),
                rng.range(-0.3, 0.3),
                rng.range(-0.3, 0.3),
            ));
        }
    }
    let inverse_masses = vec![1.0_f32; positions.len()];
    let idx = |r: usize, c: usize| (r * cols + c) as u32;

    let mut triangles = Vec::new();
    for r in 0..rows.saturating_sub(1) {
        for c in 0..cols.saturating_sub(1) {
            // Two triangles per quad, both wound counter-clockwise.
            triangles.push(ClothAeroTriangle::new(
                idx(r, c),
                idx(r, c + 1),
                idx(r + 1, c),
            ));
            triangles.push(ClothAeroTriangle::new(
                idx(r, c + 1),
                idx(r + 1, c + 1),
                idx(r + 1, c),
            ));
        }
    }
    (positions, velocities, inverse_masses, triangles)
}

fn assert_parity(label: &str, gpu: &[Vec3], cpu: &[Vec3]) {
    assert_eq!(gpu.len(), cpu.len(), "{label}: length mismatch");
    for (i, (g, c)) in gpu.iter().zip(cpu).enumerate() {
        assert!(
            close(*g, *c),
            "{label}: particle {i} diverged: gpu={g:?} cpu={c:?}"
        );
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "skip notice is the standard pattern for GPU tests off a real device"
)]
fn gpu_matches_cpu_across_parameters() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping cloth_aero_parity: no GPU adapter available");
        return;
    };
    let kernel = GpuClothAero::new(&ctx);
    let (positions, velocities, inverse_masses, triangles) = cloth_grid(6, 7, 0x5eed_a300);
    let dt = 1.0 / 60.0;

    // Steady drag-only, mixed drag/lift, turbulent, and quadratic (air density)
    // regimes all exercise distinct arithmetic paths in the force kernel.
    let cases = [
        (
            "drag_only",
            ClothAeroParams::new([2.0, 0.0, 1.5], 0.0, 1.0, 0.0),
        ),
        (
            "drag_lift",
            ClothAeroParams::new([1.0, -0.5, 2.0], 0.0, 0.8, 0.4),
        ),
        (
            "turbulent",
            ClothAeroParams::new([0.5, 0.2, 1.0], 0.75, 1.2, 0.3),
        ),
        (
            "quadratic",
            ClothAeroParams::new([2.0, 0.0, 1.0], 0.4, 1.0, 0.5).with_air_density(1.3),
        ),
    ];
    for (label, params) in cases {
        let gpu = kernel.solve(
            &ctx,
            &positions,
            &velocities,
            &inverse_masses,
            &triangles,
            params,
            dt,
        );
        let cpu = cpu_cloth_aero(
            &positions,
            &velocities,
            &inverse_masses,
            &triangles,
            params,
            dt,
        );
        assert_parity(label, &gpu, &cpu);
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "skip notice is the standard pattern for GPU tests off a real device"
)]
fn gpu_matches_cpu_with_pinned_particles() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping cloth_aero_parity: no GPU adapter available");
        return;
    };
    let kernel = GpuClothAero::new(&ctx);
    let (positions, velocities, mut inverse_masses, triangles) = cloth_grid(5, 5, 0xabcd_0007);
    // Pin the four corners.
    let cols = 5;
    for &p in &[
        0_usize,
        cols - 1,
        inverse_masses.len() - cols,
        inverse_masses.len() - 1,
    ] {
        inverse_masses[p] = 0.0;
    }
    let dt = 1.0 / 60.0;
    let params = ClothAeroParams::new([1.5, 0.5, 2.0], 0.3, 1.0, 0.2);
    let gpu = kernel.solve(
        &ctx,
        &positions,
        &velocities,
        &inverse_masses,
        &triangles,
        params,
        dt,
    );
    let cpu = cpu_cloth_aero(
        &positions,
        &velocities,
        &inverse_masses,
        &triangles,
        params,
        dt,
    );
    assert_parity("pinned", &gpu, &cpu);
    // The pinned corners must keep their original velocity on the device.
    for &p in &[
        0_usize,
        cols - 1,
        inverse_masses.len() - cols,
        inverse_masses.len() - 1,
    ] {
        assert!(close(gpu[p], velocities[p]), "pinned corner {p} moved");
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "skip notice is the standard pattern for GPU tests off a real device"
)]
fn gpu_empty_triangles_is_noop() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping cloth_aero_parity: no GPU adapter available");
        return;
    };
    let kernel = GpuClothAero::new(&ctx);
    let (positions, velocities, inverse_masses, _) = cloth_grid(4, 4, 0x1);
    let params = ClothAeroParams::new([1.0, 0.0, 1.0], 0.0, 1.0, 0.0);
    let out = kernel.solve(
        &ctx,
        &positions,
        &velocities,
        &inverse_masses,
        &[],
        params,
        1.0 / 60.0,
    );
    assert_eq!(out, velocities, "empty triangle set must be a no-op");
    // A non-positive dt is also a no-op.
    let zero = kernel.solve(
        &ctx,
        &positions,
        &velocities,
        &inverse_masses,
        &[ClothAeroTriangle::new(0, 1, 4)],
        params,
        0.0,
    );
    assert_eq!(zero, velocities, "non-positive dt must be a no-op");
}
