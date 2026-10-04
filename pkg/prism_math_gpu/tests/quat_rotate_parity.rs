//! Real-device parity for the §24.1 quaternion-rotate shader mirror.
//!
//! Each test builds a batch of vectors and a unit quaternion, runs the CPU
//! reference [`quat_rotate_vec3`], runs the `GPU` kernel on a real adapter, and
//! asserts the two agree within a small absolute+relative tolerance. The kernel
//! does floating-point arithmetic and Metal compiles WGSL under fast-math (FMA
//! contraction / reassociation), so the §24.1 contract is a *tolerance*
//! round-trip, not a bit-exact one. The suite skips gracefully when no adapter
//! is available so it still passes on a device-less CI image while running the
//! full dispatch on a real `GPU`.

use prism_math::Quat;
use prism_math::Vec3;
use prism_math::shader_mirror::quat_rotate_vec3;
use prism_math_gpu::GpuContext;
use prism_math_gpu::GpuQuatRotate;

/// Acquires a device, or prints a skip note and returns `None` on hosts without
/// a usable adapter.
#[expect(
    clippy::print_stderr,
    reason = "test-only skip note when no GPU adapter is present"
)]
fn with_gpu() -> Option<GpuContext> {
    match GpuContext::try_headless() {
        Some(ctx) => Some(ctx),
        None => {
            eprintln!("skipping: no usable GPU adapter on this host");
            None
        }
    }
}

/// Deterministic LCG mapped to `f32`, no transcendentals in the stream itself.
struct Lcg {
    state: u32,
}

impl Lcg {
    fn new(seed: u32) -> Lcg {
        Lcg { state: seed }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(1_664_525)
            .wrapping_add(1_013_904_223);
        self.state
    }

    /// `f32` in `[-range, range)`.
    fn signed(&mut self, range: f32) -> f32 {
        let v = self.next_u32() >> 8; // 24 bits of entropy
        let unit = (v as f32) / 16_777_216.0_f32;
        (unit * 2.0 - 1.0) * range
    }
}

/// Absolute+relative tolerance comparison for a rotated-vector component.
///
/// Rotation preserves length, so an absolute epsilon scaled by the operand
/// magnitude is the right yardstick; `1e-5` comfortably admits last-ULP FMA
/// rounding on a unit-ish vector while rejecting any real algorithmic drift.
fn close(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    let scale = a.abs().max(b.abs()).max(1.0);
    diff <= 1.0e-5 * scale
}

fn assert_vecs_close(cpu: &[Vec3], gpu: &[Vec3]) {
    assert_eq!(cpu.len(), gpu.len(), "length mismatch");
    for (i, (c, g)) in cpu.iter().zip(gpu.iter()).enumerate() {
        assert!(
            close(c.x, g.x) && close(c.y, g.y) && close(c.z, g.z),
            "component {i} drifted: cpu=({}, {}, {}) gpu=({}, {}, {})",
            c.x,
            c.y,
            c.z,
            g.x,
            g.y,
            g.z,
        );
    }
}

#[test]
fn identity_quat_is_passthrough() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuQuatRotate::new(&ctx);
    let input = [
        Vec3::new(1.0, 2.0, 3.0),
        Vec3::new(-4.0, 5.0, -6.0),
        Vec3::new(0.0, 0.0, 0.0),
    ];
    let gpu = kernel.rotate(&ctx, Quat::IDENTITY, &input);
    assert_vecs_close(&input, &gpu);
}

#[test]
fn axis_rotations_match_cpu_reference() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuQuatRotate::new(&ctx);
    let quats = [
        Quat::from_rotation_x(0.7),
        Quat::from_rotation_y(-1.3),
        Quat::from_rotation_z(2.1),
        Quat::from_axis_angle(Vec3::new(1.0, 2.0, 3.0).normalize(), 0.9),
    ];
    let input = [
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(3.0, -2.0, 1.5),
    ];
    for q in quats {
        let cpu: Vec<Vec3> = input.iter().map(|&v| quat_rotate_vec3(q, v)).collect();
        let gpu = kernel.rotate(&ctx, q, &input);
        assert_vecs_close(&cpu, &gpu);
    }
}

#[test]
fn large_multi_workgroup_batch_matches_cpu() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuQuatRotate::new(&ctx);
    // Well past one 64-lane workgroup to exercise the dispatch grid.
    let count = 4096 + 37;
    let mut rng = Lcg::new(0xC0FF_EE01);
    let input: Vec<Vec3> = (0..count)
        .map(|_| Vec3::new(rng.signed(10.0), rng.signed(10.0), rng.signed(10.0)))
        .collect();
    let q = Quat::from_axis_angle(Vec3::new(0.3, -0.7, 0.5).normalize(), 1.21);
    let cpu: Vec<Vec3> = input.iter().map(|&v| quat_rotate_vec3(q, v)).collect();
    let gpu = kernel.rotate(&ctx, q, &input);
    assert_vecs_close(&cpu, &gpu);
}

#[test]
fn empty_input_returns_empty() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuQuatRotate::new(&ctx);
    let gpu = kernel.rotate(&ctx, Quat::IDENTITY, &[]);
    assert!(gpu.is_empty());
}
