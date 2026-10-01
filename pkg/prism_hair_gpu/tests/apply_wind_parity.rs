//! Real-device parity for the per-particle wind pre-pass twin:
//! [`GpuHairApplyWind`] must reproduce the `CPU` golden
//! [`apply_wind`](prism_render_architecture::hair::wind::apply_wind) for a batch
//! of guide particles advanced under one shared [`WindField`] over `(time,
//! dt)`, including the pinned skip, the `dt^2` integration scaling and the
//! non-positive-`dt` no-op the reference guards.
//!
//! # Parity criterion
//!
//! The wind field is a closed-form polynomial (the reference avoids `f32::sin`)
//! and the `pos + accel * dt^2` update is a plain multiply-add, so the only
//! `CPU` vs `GPU` divergence is legal fused-multiply-add contraction. Each free
//! particle's position component is asserted to within `abs_diff < 1e-4` or
//! `rel_diff < 1e-3`. Pinned particles and every particle under a calm field
//! are copied through with no arithmetic, so those are asserted *bit-for-bit*
//! (comparing raw `f32` bit patterns) — a kernel that moved a pinned root or
//! dropped the `dt^2` could not pass. Free particles under a live field are
//! also asserted to have actually moved, so a no-op kernel could not pass
//! either.
//!
//! The suite drives a steady field (free moved along +x, pinned fixed), a calm
//! field (all unchanged), a gusty + turbulent field with a mixed pinned/free
//! batch, a non-positive `dt` no-op, an all-pinned batch, the empty batch, and
//! a 200-particle batch that crosses the 64-wide dispatch boundary.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-`WGSL`,
//! so it needs no optional device feature. Test data uses only integer/affine
//! arithmetic — never `f32::sin`/`cos` — so it stays deterministic without
//! introducing transcendental divergence.
//!
//! Provenance: standard steady+gust+turbulence wind pre-pass plus a `wgpu`
//! compute dispatch; no Unreal Engine source or derived code.

use prism_hair_gpu::apply_wind::{reference_apply_wind, GpuHairApplyWind};
use prism_hair_gpu::GpuContext;
use prism_render_architecture::hair::dynamics::{StrandParticle, Vec3};
use prism_render_architecture::hair::wind::WindField;

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

/// True when `got` matches `want` within the fused-multiply-add tolerance.
fn close(got: f32, want: f32) -> bool {
    let diff = (got - want).abs();
    diff < 1.0e-4 || diff <= 1.0e-3 * want.abs()
}

/// True when `got` equals `want` bit-for-bit, avoiding a float `==` comparison
/// (clippy `float_cmp`) by comparing raw `f32` bit patterns.
fn bits_eq(got: [f32; 3], want: [f32; 3]) -> bool {
    got[0].to_bits() == want[0].to_bits()
        && got[1].to_bits() == want[1].to_bits()
        && got[2].to_bits() == want[2].to_bits()
}

/// A free particle at `(x, y, z)`.
fn free(x: f32, y: f32, z: f32) -> StrandParticle {
    StrandParticle::free(Vec3::new(x, y, z))
}

/// A pinned (kinematic) particle at `(x, y, z)`.
fn pinned(x: f32, y: f32, z: f32) -> StrandParticle {
    StrandParticle::pinned(Vec3::new(x, y, z))
}

/// Dispatches one batch through the device twin.
fn run(
    ctx: &GpuContext,
    particles: &[StrandParticle],
    field: WindField,
    time: f32,
    dt: f32,
) -> Vec<[f32; 3]> {
    GpuHairApplyWind::new(ctx).eval(ctx, particles, field, time, dt)
}

/// Asserts the device result matches the golden component-wise within tolerance.
fn assert_batch_parity(
    ctx: &GpuContext,
    particles: &[StrandParticle],
    field: WindField,
    time: f32,
    dt: f32,
) -> Vec<[f32; 3]> {
    let got = run(ctx, particles, field, time, dt);
    let want = reference_apply_wind(particles, field, time, dt);
    assert_eq!(got.len(), want.len(), "particle count mismatch");
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        for axis in 0..3 {
            assert!(
                close(g[axis], w[axis]),
                "particle {i} axis {axis}: gpu {} vs cpu {}",
                g[axis],
                w[axis]
            );
        }
    }
    got
}

/// A steady wind heading along `+x` with no gust or turbulence.
fn steady_x(speed: f32) -> WindField {
    WindField {
        direction: Vec3::new(1.0, 0.0, 0.0),
        speed,
        gust_amplitude: 0.0,
        gust_frequency: 0.0,
        turbulence: 0.0,
    }
}

#[test]
fn gpu_steady_wind_pushes_free_leaves_pinned() {
    let Some(ctx) = context_or_skip("gpu_steady_wind_pushes_free_leaves_pinned") else {
        return;
    };
    let particles = [
        free(0.0, 0.0, 0.0),
        pinned(1.0, 2.0, 3.0),
        free(2.0, 0.0, 0.0),
    ];
    let field = steady_x(4.0);
    let dt = 0.5;
    let got = assert_batch_parity(&ctx, &particles, field, 0.0, dt);

    // Pinned particle copied through bit-for-bit.
    assert!(
        bits_eq(got[1], [1.0, 2.0, 3.0]),
        "pinned moved: {:?}",
        got[1]
    );
    // Free particles pushed strictly along +x (dt^2 * speed = 0.25 * 4 = 1).
    assert!(got[0][0] > 0.0, "free particle 0 should move +x: {got:?}");
    assert!(
        got[2][0] > particles[2].position.x,
        "free particle 2 should move +x"
    );
    // No off-axis motion under a pure +x steady field (exact zero).
    assert!(
        got[0][1].to_bits() == 0.0f32.to_bits(),
        "off-axis y: {got:?}"
    );
    assert!(
        got[0][2].to_bits() == 0.0f32.to_bits(),
        "off-axis z: {got:?}"
    );
}

#[test]
fn gpu_calm_field_moves_nothing() {
    let Some(ctx) = context_or_skip("gpu_calm_field_moves_nothing") else {
        return;
    };
    let particles = [
        free(1.0, 2.0, 3.0),
        free(-4.0, 5.0, -6.0),
        pinned(7.0, 8.0, 9.0),
    ];
    let got = assert_batch_parity(&ctx, &particles, WindField::CALM, 2.5, 0.25);
    // A calm field adds zero acceleration, so every position is copied through
    // exactly, free and pinned alike.
    assert!(bits_eq(got[0], [1.0, 2.0, 3.0]), "{got:?}");
    assert!(bits_eq(got[1], [-4.0, 5.0, -6.0]), "{got:?}");
    assert!(bits_eq(got[2], [7.0, 8.0, 9.0]), "{got:?}");
}

#[test]
fn gpu_gusty_turbulent_mixed_batch() {
    let Some(ctx) = context_or_skip("gpu_gusty_turbulent_mixed_batch") else {
        return;
    };
    let field = WindField {
        direction: Vec3::new(2.0, 1.0, -1.0),
        speed: 3.0,
        gust_amplitude: 1.5,
        gust_frequency: 0.75,
        turbulence: 0.5,
    };
    let particles = [
        free(0.0, 0.0, 0.0),
        pinned(1.0, 1.0, 1.0),
        free(2.0, -1.0, 3.0),
        free(-3.0, 2.0, -2.0),
        pinned(4.0, 0.0, 0.0),
    ];
    let got = assert_batch_parity(&ctx, &particles, field, 1.25, 0.3);
    // Pinned particles are never moved.
    assert!(bits_eq(got[1], [1.0, 1.0, 1.0]), "{got:?}");
    assert!(bits_eq(got[4], [4.0, 0.0, 0.0]), "{got:?}");
    // At least one free particle moved off its start under the live field.
    let moved = !bits_eq(got[0], [0.0, 0.0, 0.0])
        || !bits_eq(got[2], [2.0, -1.0, 3.0])
        || !bits_eq(got[3], [-3.0, 2.0, -2.0]);
    assert!(
        moved,
        "a live gusty field must move at least one free particle"
    );
}

#[test]
fn gpu_nonpositive_dt_is_noop() {
    let Some(ctx) = context_or_skip("gpu_nonpositive_dt_is_noop") else {
        return;
    };
    let particles = [free(1.0, 2.0, 3.0), free(4.0, 5.0, 6.0)];
    let field = steady_x(10.0);
    // dt == 0 and dt < 0 both no-op: positions returned unchanged.
    let got0 = assert_batch_parity(&ctx, &particles, field, 1.0, 0.0);
    assert!(bits_eq(got0[0], [1.0, 2.0, 3.0]), "{got0:?}");
    assert!(bits_eq(got0[1], [4.0, 5.0, 6.0]), "{got0:?}");
    let got_neg = assert_batch_parity(&ctx, &particles, field, 1.0, -0.5);
    assert!(bits_eq(got_neg[0], [1.0, 2.0, 3.0]), "{got_neg:?}");
    assert!(bits_eq(got_neg[1], [4.0, 5.0, 6.0]), "{got_neg:?}");
}

#[test]
fn gpu_all_pinned_batch_unchanged() {
    let Some(ctx) = context_or_skip("gpu_all_pinned_batch_unchanged") else {
        return;
    };
    let particles = [pinned(1.0, 2.0, 3.0), pinned(-4.0, 5.0, -6.0)];
    let field = WindField {
        direction: Vec3::new(1.0, 1.0, 1.0),
        speed: 5.0,
        gust_amplitude: 2.0,
        gust_frequency: 1.0,
        turbulence: 1.0,
    };
    // Every particle pinned: all copied through exactly even under a strong
    // field (and a real dispatch still runs).
    let got = assert_batch_parity(&ctx, &particles, field, 0.5, 0.4);
    assert!(bits_eq(got[0], [1.0, 2.0, 3.0]), "{got:?}");
    assert!(bits_eq(got[1], [-4.0, 5.0, -6.0]), "{got:?}");
}

#[test]
fn gpu_empty_batch_yields_empty() {
    let Some(ctx) = context_or_skip("gpu_empty_batch_yields_empty") else {
        return;
    };
    let got = run(&ctx, &[], steady_x(1.0), 0.0, 0.5);
    assert!(got.is_empty());
}

#[test]
fn gpu_large_batch_crosses_workgroup_boundary() {
    let Some(ctx) = context_or_skip("gpu_large_batch_crosses_workgroup_boundary") else {
        return;
    };
    // 200 particles (> 3 * 64) so the one-thread-per-particle dispatch spans
    // several workgroups. Positions are generated by integer affine arithmetic
    // (no transcendental calls), and every third particle is pinned so the
    // skip/move branch is exercised across workgroup boundaries.
    let particles: Vec<StrandParticle> = (0..200)
        .map(|k| {
            let i = k;
            let x = (i - 100) as f32;
            let y = (2 * i) as f32;
            let z = (i * i % 13) as f32;
            if k % 3 == 0 {
                pinned(x, y, z)
            } else {
                free(x, y, z)
            }
        })
        .collect();
    let field = WindField {
        direction: Vec3::new(0.0, 1.0, 0.0),
        speed: 2.0,
        gust_amplitude: 0.5,
        gust_frequency: 0.5,
        turbulence: 0.25,
    };
    let got = assert_batch_parity(&ctx, &particles, field, 0.75, 0.2);
    assert_eq!(got.len(), 200);
    // Every pinned particle (index % 3 == 0) is unchanged.
    for k in (0..200).step_by(3) {
        let p = particles[k].position;
        assert!(
            bits_eq(got[k], [p.x, p.y, p.z]),
            "pinned particle {k} moved"
        );
    }
}
