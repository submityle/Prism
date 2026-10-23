//! Real-device parity for the §24.7 hierarchy-propagation twin.
//!
//! Each test builds a transform forest, packs the device-free upload payload
//! with [`ComputeHierarchyInput::pack`], runs the CPU reference
//! [`propagate_by_levels`], runs the `GPU` kernel on a real adapter, and asserts
//! the two sets of world matrices agree within a small absolute+relative
//! tolerance. The kernel does floating-point affine composition and Metal
//! compiles WGSL under fast-math (FMA contraction / reassociation), so the twin
//! is a *tolerance* round-trip, not a bit-exact one. The suite skips gracefully
//! when no adapter is available so it still passes on a device-less CI image
//! while running the full dispatch on a real `GPU`. Set `PRISM_REQUIRE_GPU=1`
//! to make adapter/device absence a hard failure during physical-GPU acceptance.

use prism_math::{Quat, Vec3};
use prism_transform::compute_hierarchy::{
    propagate_by_levels, ComputeHierarchyInput, LevelSchedule,
};
use prism_transform::gpu_upload::MatrixLayout;
use prism_transform::hierarchy::Hierarchy;
use prism_transform::{GlobalTransform, Transform};
use prism_transform_gpu::buffer;
use prism_transform_gpu::{GpuContext, GpuHierarchyOutput, GpuHierarchyPropagate};
use wgpu::CommandEncoderDescriptor;

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
            assert_ne!(
                std::env::var("PRISM_REQUIRE_GPU").as_deref(),
                Ok("1"),
                "PRISM_REQUIRE_GPU=1 but no usable GPU adapter/device was acquired"
            );
            eprintln!("skipping: no usable GPU adapter on this host");
            None
        }
    }
}

/// Deterministic LCG mapped to `f32` so the test inputs are reproducible.
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
        let v = self.next_u32() >> 8;
        let unit = (v as f32) / 16_777_216.0_f32;
        (unit * 2.0 - 1.0) * range
    }

    /// A non-trivial local transform: random translation, a rotation about a
    /// random axis, and a mildly non-uniform positive scale.
    fn transform(&mut self) -> Transform {
        let axis =
            Vec3::new(self.signed(1.0), self.signed(1.0), self.signed(1.0) + 0.001).normalize();
        let angle = self.signed(core::f32::consts::PI);
        let scale = Vec3::new(
            0.5 + (self.next_u32() >> 8) as f32 / 16_777_216.0,
            0.5 + (self.next_u32() >> 8) as f32 / 16_777_216.0,
            0.5 + (self.next_u32() >> 8) as f32 / 16_777_216.0,
        );
        Transform::from_xyz(self.signed(10.0), self.signed(10.0), self.signed(10.0))
            .with_rotation(Quat::from_axis_angle(axis, angle))
            .with_scale(scale)
    }
}

/// Absolute+relative tolerance comparison for a world-matrix scalar.
fn close(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    let scale = a.abs().max(b.abs()).max(1.0);
    diff <= 1e-4 * scale
}

/// Assert two world transforms agree within tolerance on every scalar.
fn assert_world_close(cpu: GlobalTransform, gpu: GlobalTransform, node: usize) {
    let a = cpu.affine();
    let b = gpu.affine();
    let cpu_cols = [
        a.matrix3.x_axis,
        a.matrix3.y_axis,
        a.matrix3.z_axis,
        a.translation,
    ];
    let gpu_cols = [
        b.matrix3.x_axis,
        b.matrix3.y_axis,
        b.matrix3.z_axis,
        b.translation,
    ];
    for (c, (ca, cb)) in cpu_cols.iter().zip(gpu_cols.iter()).enumerate() {
        assert!(
            close(ca.x, cb.x) && close(ca.y, cb.y) && close(ca.z, cb.z),
            "node {node} column {c} mismatch: cpu={ca:?} gpu={cb:?}"
        );
    }
}

/// Run the CPU reference and GPU kernel on `hierarchy`/`locals` and assert
/// per-node parity.
fn check(ctx: &GpuContext, hierarchy: &Hierarchy, locals: &[Transform]) {
    let schedule = LevelSchedule::build(hierarchy).expect("forest schedule builds");
    let n = hierarchy.len();

    let mut cpu = vec![GlobalTransform::IDENTITY; n];
    propagate_by_levels(hierarchy, &schedule, locals, &mut cpu).expect("cpu reference runs");

    let input =
        ComputeHierarchyInput::pack(hierarchy, &schedule, locals, MatrixLayout::RowMajor3x4);
    let kernel = GpuHierarchyPropagate::new(ctx);
    let gpu = kernel.propagate(ctx, &input);

    assert_eq!(gpu.len(), n, "one world matrix per node");
    for i in 0..n {
        assert_world_close(cpu[i], gpu[i], i);
    }
}

#[test]
fn empty_hierarchy_returns_empty() {
    let Some(ctx) = with_gpu() else { return };
    let hierarchy = Hierarchy::new();
    let kernel = GpuHierarchyPropagate::new(&ctx);
    let schedule = LevelSchedule::build(&hierarchy).unwrap();
    let input = ComputeHierarchyInput::pack(&hierarchy, &schedule, &[], MatrixLayout::RowMajor3x4);
    assert!(kernel.propagate(&ctx, &input).is_empty());
}

#[test]
fn single_chain_parity() {
    let Some(ctx) = with_gpu() else { return };
    // A deep single chain: root -> c1 -> c2 -> ... maximises level count so the
    // per-level dispatch ordering is exercised end to end.
    let mut hierarchy = Hierarchy::new();
    let mut rng = Lcg::new(0x1234_5678);
    let mut locals = Vec::new();

    let root = hierarchy.spawn_root();
    locals.push(rng.transform());
    let mut parent = root;
    for _ in 0..31 {
        let child = hierarchy.spawn_child(parent);
        locals.push(rng.transform());
        parent = child;
    }

    check(&ctx, &hierarchy, &locals);
}

#[test]
fn branching_forest_parity() {
    let Some(ctx) = with_gpu() else { return };
    // Several roots, each with a few generations of fan-out children.
    let mut hierarchy = Hierarchy::new();
    let mut rng = Lcg::new(0x9e37_79b9);
    let mut locals = Vec::new();

    let mut frontier = Vec::new();
    for _ in 0..4 {
        let root = hierarchy.spawn_root();
        locals.push(rng.transform());
        frontier.push(root);
    }
    for _ in 0..4 {
        let mut next = Vec::new();
        for &p in &frontier {
            let fanout = 1 + (rng.next_u32() % 3) as usize;
            for _ in 0..fanout {
                let child = hierarchy.spawn_child(p);
                locals.push(rng.transform());
                next.push(child);
            }
        }
        frontier = next;
    }

    check(&ctx, &hierarchy, &locals);
}

#[test]
fn wide_level_multi_workgroup_parity() {
    let Some(ctx) = with_gpu() else { return };
    // A single root with a very wide second level (> one workgroup of 64) to
    // exercise multi-workgroup dispatch within a level.
    let mut hierarchy = Hierarchy::new();
    let mut rng = Lcg::new(0x0bad_f00d);
    let mut locals = Vec::new();

    let root = hierarchy.spawn_root();
    locals.push(rng.transform());
    for _ in 0..4000 {
        let _child = hierarchy.spawn_child(root);
        locals.push(rng.transform());
    }

    check(&ctx, &hierarchy, &locals);
}

#[test]
fn identity_locals_passthrough() {
    let Some(ctx) = with_gpu() else { return };
    // All-identity locals: every world matrix must stay identity, isolating the
    // decode/encode path from the compose arithmetic.
    let mut hierarchy = Hierarchy::new();
    let root = hierarchy.spawn_root();
    let a = hierarchy.spawn_child(root);
    let _b = hierarchy.spawn_child(a);
    let _c = hierarchy.spawn_child(root);
    let locals = vec![Transform::IDENTITY; hierarchy.len()];

    check(&ctx, &hierarchy, &locals);
}

#[test]
fn recorded_output_is_resident_and_reused() {
    let Some(ctx) = with_gpu() else { return };
    let kernel = GpuHierarchyPropagate::new(&ctx);
    let mut output = GpuHierarchyOutput::new();
    let mut hierarchy = Hierarchy::new();
    let root = hierarchy.spawn_root();
    hierarchy.spawn_child(root);
    let locals = [
        Transform::from_xyz(5.0, -2.0, 1.0),
        Transform::from_xyz(1.0, 2.0, 3.0),
    ];
    let schedule = LevelSchedule::build(&hierarchy).unwrap();
    let input =
        ComputeHierarchyInput::pack(&hierarchy, &schedule, &locals, MatrixLayout::RowMajor3x4);

    let mut encoder = ctx
        .device()
        .create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_transform_resident_output_test"),
        });
    kernel.record(&ctx, &input, &mut output, &mut encoder);
    assert_eq!(output.len(), 2);
    assert!(!output.is_empty());
    let first_capacity = output.capacity_nodes();
    let resident = output
        .storage_buffer(&ctx)
        .expect("non-empty output buffer");
    let stage = buffer::staging(ctx.device(), "resident_output_test_stage", 2 * 48);
    buffer::copy(&mut encoder, resident, &stage, 2 * 48);
    ctx.queue().submit([encoder.finish()]);
    let worlds: Vec<[f32; 12]> = buffer::read_back(&ctx, &stage);
    assert_eq!(worlds.len(), 2);
    assert_eq!(
        &worlds[0],
        &[1.0, 0.0, 0.0, 5.0, 0.0, 1.0, 0.0, -2.0, 0.0, 0.0, 1.0, 1.0]
    );
    assert_eq!(
        &worlds[1],
        &[1.0, 0.0, 0.0, 6.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 4.0]
    );

    let mut smaller = Hierarchy::new();
    smaller.spawn_root();
    let smaller_schedule = LevelSchedule::build(&smaller).unwrap();
    let smaller_input = ComputeHierarchyInput::pack(
        &smaller,
        &smaller_schedule,
        &[Transform::IDENTITY],
        MatrixLayout::RowMajor3x4,
    );
    let mut encoder = ctx
        .device()
        .create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_transform_resident_output_reuse_test"),
        });
    kernel.record(&ctx, &smaller_input, &mut output, &mut encoder);
    assert_eq!(output.len(), 1);
    assert_eq!(output.capacity_nodes(), first_capacity);
    assert!(output.storage_buffer(&ctx).is_some());
    ctx.queue().submit([encoder.finish()]);
    ctx.wait();
}
