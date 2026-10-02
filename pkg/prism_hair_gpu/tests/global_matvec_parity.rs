//! Real-device parity for the projective-dynamics global dense matvec twin:
//! [`GpuHairGlobalMatvec`] must reproduce the `CPU` golden
//! [`GlobalSystem::mul`](prism_render_architecture::hair::projective_global::GlobalSystem::mul)
//! for the dense product `A x` of the assembled `SPD` system, including the
//! empty-constraint (pure inertial diagonal) case, the zero-input case, the
//! short-`x` zero-padding contract, a single-particle system, and a system
//! larger than one `64`-thread workgroup.
//!
//! # Golden source
//!
//! The matrix is the real architecture solver's assembled system: a
//! [`ConstraintSet`](prism_render_architecture::hair::projective_global::ConstraintSet)
//! is factored by
//! [`GlobalSystem::factor`](prism_render_architecture::hair::projective_global::GlobalSystem::factor),
//! then [`GlobalSystem::matrix`](prism_render_architecture::hair::projective_global::GlobalSystem::matrix)
//! exposes the exact row-major `n * n` buffer the device uploads and
//! [`GlobalSystem::mul`](prism_render_architecture::hair::projective_global::GlobalSystem::mul)
//! supplies the golden product. The device result is also cross-checked against
//! the crate's own [`reference_global_matvec`], an independent same-order
//! reimplementation.
//!
//! # Parity criterion
//!
//! The product is a fixed-order sum of exact products, so the only `CPU` vs
//! `GPU` divergence is legal fused-multiply-add contraction in the row
//! accumulation. Each coordinate component is asserted to within
//! `abs_diff < 1e-4` or `rel_diff < 1e-3` - tight enough to fail a genuinely
//! wrong port (a transposed index, a dropped column, a mismatched stride),
//! loose enough to admit fma contraction. The output length is asserted exactly.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-`WGSL`,
//! so it needs no optional device feature.
//!
//! Provenance: standard dense matrix-vector product plus a `wgpu` compute
//! dispatch; no Unreal Engine source or derived code.

use prism_hair_gpu::global_matvec::{reference_global_matvec, GpuHairGlobalMatvec};
use prism_hair_gpu::GpuContext;
use prism_render_architecture::hair::projective_global::{
    BendConstraint, ConstraintSet, EdgeConstraint, GlobalSystem, SolverConfig, Vec3,
};

/// Acquires a headless context, or `None` (with a skip notice) when the host has
/// no `wgpu` adapter so the suite stays green off-device.
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn context_or_skip(label: &str) -> Option<GpuContext> {
    match GpuContext::try_headless() {
        Some(ctx) => Some(ctx),
        None => {
            eprintln!("skipping {label}: no wgpu adapter on this host");
            None
        }
    }
}

/// True when `got` matches `want` within the matvec tolerance.
fn close(got: f32, want: f32) -> bool {
    let diff = (got - want).abs();
    diff < 1.0e-4 || diff <= 1.0e-3 * want.abs()
}

/// Factors the system for `constraints` at dimension `n` with a default config.
fn factor(constraints: &ConstraintSet, n: usize) -> GlobalSystem {
    GlobalSystem::factor(constraints, n, SolverConfig::default())
}

/// Asserts the device matvec of `sys` applied to `x` matches both the
/// architecture golden `sys.mul` and the crate reference, componentwise.
fn assert_matvec_parity(ctx: &GpuContext, sys: &GlobalSystem, x: &[Vec3]) {
    let n = sys.dim();
    let x_arr: Vec<[f32; 3]> = x.iter().map(|v| [v.x, v.y, v.z]).collect();
    let got = GpuHairGlobalMatvec::new(ctx).eval(ctx, sys.matrix(), n, &x_arr);
    let golden = sys.mul(x);
    let reference = reference_global_matvec(sys.matrix(), n, &x_arr);

    assert_eq!(got.len(), n, "output length mismatch");
    assert_eq!(golden.len(), n, "golden length mismatch");
    assert_eq!(reference.len(), n, "reference length mismatch");

    for i in 0..n {
        let g = got[i];
        let w = golden[i];
        let r = reference[i];
        assert!(close(g[0], w.x), "row {i}.x gpu {} vs golden {}", g[0], w.x);
        assert!(close(g[1], w.y), "row {i}.y gpu {} vs golden {}", g[1], w.y);
        assert!(close(g[2], w.z), "row {i}.z gpu {} vs golden {}", g[2], w.z);
        assert!(close(g[0], r[0]), "row {i}.x gpu {} vs ref {}", g[0], r[0]);
        assert!(close(g[1], r[1]), "row {i}.y gpu {} vs ref {}", g[1], r[1]);
        assert!(close(g[2], r[2]), "row {i}.z gpu {} vs ref {}", g[2], r[2]);
    }
}

#[test]
fn gpu_single_edge_two_particles() {
    let Some(ctx) = context_or_skip("gpu_single_edge_two_particles") else {
        return;
    };
    let cs = ConstraintSet::single_edge(EdgeConstraint::new(0, 1, 1.5, 4.0));
    let sys = factor(&cs, 2);
    let x = [Vec3::new(1.0, -2.0, 0.5), Vec3::new(-0.25, 3.0, 2.0)];
    assert_matvec_parity(&ctx, &sys, &x);
}

#[test]
fn gpu_multi_particle_chain() {
    let Some(ctx) = context_or_skip("gpu_multi_particle_chain") else {
        return;
    };
    // A five-particle chain of edge-length constraints.
    let mut cs = ConstraintSet::new();
    for i in 0..4 {
        cs.edges
            .push(EdgeConstraint::new(i, i + 1, 1.0 + i as f32 * 0.1, 3.0));
    }
    let sys = factor(&cs, 5);
    let x: Vec<Vec3> = (0..5)
        .map(|i| Vec3::new(i as f32, (2 * i) as f32 - 3.0, 0.5 * i as f32))
        .collect();
    assert_matvec_parity(&ctx, &sys, &x);
}

#[test]
fn gpu_chain_with_bend() {
    let Some(ctx) = context_or_skip("gpu_chain_with_bend") else {
        return;
    };
    // Edges plus a bend constraint around the center particle.
    let mut cs = ConstraintSet::new();
    cs.edges.push(EdgeConstraint::new(0, 1, 1.0, 2.0));
    cs.edges.push(EdgeConstraint::new(1, 2, 1.0, 2.0));
    cs.bends
        .push(BendConstraint::new(0, 1, 2, Vec3::new(0.0, 0.1, 0.0), 1.5));
    let sys = factor(&cs, 3);
    let x = [
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 0.5, -0.5),
        Vec3::new(2.0, 0.0, 1.0),
    ];
    assert_matvec_parity(&ctx, &sys, &x);
}

#[test]
fn gpu_empty_constraints_pure_inertia() {
    let Some(ctx) = context_or_skip("gpu_empty_constraints_pure_inertia") else {
        return;
    };
    // With no constraints the system is a pure diagonal mass/dt^2 * I, so
    // A x = (mass/dt^2) x; the matvec must still match exactly.
    let cs = ConstraintSet::new();
    let sys = factor(&cs, 4);
    let x: Vec<Vec3> = (0..4)
        .map(|i| Vec3::new(i as f32 + 1.0, -(i as f32), 2.0 * i as f32))
        .collect();
    assert_matvec_parity(&ctx, &sys, &x);
}

#[test]
fn gpu_zero_input_yields_zero() {
    let Some(ctx) = context_or_skip("gpu_zero_input_yields_zero") else {
        return;
    };
    let mut cs = ConstraintSet::new();
    cs.edges.push(EdgeConstraint::new(0, 1, 2.0, 5.0));
    cs.edges.push(EdgeConstraint::new(1, 2, 2.0, 5.0));
    let sys = factor(&cs, 3);
    let x = [Vec3::ZERO; 3];
    let got = GpuHairGlobalMatvec::new(&ctx).eval(&ctx, sys.matrix(), 3, &[[0.0; 3]; 3]);
    assert_eq!(got.len(), 3);
    for (i, row) in got.iter().enumerate() {
        assert!(close(row[0], 0.0), "row {i}.x = {}", row[0]);
        assert!(close(row[1], 0.0), "row {i}.y = {}", row[1]);
        assert!(close(row[2], 0.0), "row {i}.z = {}", row[2]);
    }
    assert_matvec_parity(&ctx, &sys, &x);
}

#[test]
fn gpu_single_particle_system() {
    let Some(ctx) = context_or_skip("gpu_single_particle_system") else {
        return;
    };
    // n = 1: a 1x1 diagonal system, the smallest non-empty dispatch.
    let cs = ConstraintSet::new();
    let sys = factor(&cs, 1);
    let x = [Vec3::new(3.0, -4.0, 5.0)];
    assert_matvec_parity(&ctx, &sys, &x);
}

#[test]
fn gpu_short_x_is_zero_padded() {
    let Some(ctx) = context_or_skip("gpu_short_x_is_zero_padded") else {
        return;
    };
    // The golden pads a short x with zeros; the twin must agree. Build a system
    // of dimension 4 but hand only two input particles.
    let mut cs = ConstraintSet::new();
    for i in 0..3 {
        cs.edges.push(EdgeConstraint::new(i, i + 1, 1.0, 2.0));
    }
    let sys = factor(&cs, 4);
    let short = [Vec3::new(1.0, 2.0, 3.0), Vec3::new(-1.0, 0.5, 2.0)];
    // Golden over the short slice (its own zero padding).
    let golden = sys.mul(&short);
    // Device over the same short slice (eval zero-pads to n internally).
    let short_arr: Vec<[f32; 3]> = short.iter().map(|v| [v.x, v.y, v.z]).collect();
    let got = GpuHairGlobalMatvec::new(&ctx).eval(&ctx, sys.matrix(), 4, &short_arr);
    assert_eq!(got.len(), 4);
    for i in 0..4 {
        assert!(close(got[i][0], golden[i].x), "row {i}.x");
        assert!(close(got[i][1], golden[i].y), "row {i}.y");
        assert!(close(got[i][2], golden[i].z), "row {i}.z");
    }
}

#[test]
fn gpu_large_system_crosses_workgroup_boundary() {
    let Some(ctx) = context_or_skip("gpu_large_system_crosses_workgroup_boundary") else {
        return;
    };
    // A 90-particle chain => a 90x90 system, so the one-thread-per-row dispatch
    // spans more than one 64-thread workgroup.
    let n = 90;
    let mut cs = ConstraintSet::new();
    for i in 0..(n - 1) {
        cs.edges.push(EdgeConstraint::new(
            i,
            i + 1,
            1.0 + (i % 3) as f32 * 0.05,
            2.5,
        ));
    }
    // A couple of bends to make the band wider than tridiagonal.
    for i in 0..(n - 2) {
        cs.bends
            .push(BendConstraint::new(i, i + 1, i + 2, Vec3::ZERO, 0.75));
    }
    let sys = factor(&cs, n);
    let x: Vec<Vec3> = (0..n)
        .map(|i| {
            let f = i as f32;
            Vec3::new(f * 0.5 - 10.0, (i % 7) as f32 - 3.0, (i % 5) as f32 * 0.25)
        })
        .collect();
    assert_matvec_parity(&ctx, &sys, &x);
}
