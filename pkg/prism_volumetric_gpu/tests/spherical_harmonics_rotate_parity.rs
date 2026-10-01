//! Real-device parity for the spherical-harmonic rotation twin:
//! [`GpuSphericalHarmonicsRotate`](prism_volumetric_gpu::spherical_harmonics_rotate::GpuSphericalHarmonicsRotate)
//! must reproduce the `CPU` golden
//! [`particle::spherical_harmonics_rotate`](prism_render_architecture::particle::spherical_harmonics_rotate)
//! across identity, per-axis right-angle, random-quaternion and degenerate
//! rotations, verifying band `L1` and band `L2` across all three `RGB`
//! channels.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each coefficient is rational-plus-`sqrt` algebra with no transcendental call
//! and no reorderable reduction, so `CPU` and `GPU` evaluate the same closed
//! form in the same order. The comparison allows `abs_diff <= 1e-5` or
//! `rel_diff <= 1e-5` — loose enough to admit a legal fused multiply-add
//! contraction, yet tight enough to fail a wrong port (a swapped band-`L1`
//! shuffle, a transposed band-`L2` matrix, a wrong `u`/`v` coefficient, a
//! dropped `RGB` channel). Several scenarios additionally assert the rotated
//! coefficients are non-trivial (differ from the input) so a degenerate
//! passthrough kernel could not pass.
//!
//! Provenance: standard `Ivanic`-`Ruedenberg` real-`SH` rotation; no Unreal
//! Engine source or derived code.

use prism_render_architecture::particle::spherical_harmonics_rotate::{
    rotate_l2, rotation_matrix_from_quat, ShL1Rgb,
};
use prism_volumetric_gpu::spherical_harmonics_rotate::{
    GpuSphericalHarmonicsRotate, ShRotationProbe, ShRotationResult, L2_COEFF_COUNT,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute/relative parity bound. A `GPU` may fuse a multiply-add the scalar
/// reference leaves separate, perturbing the low mantissa bits by a few units
/// in the last place; `1e-5` admits that legal slack while still failing a
/// genuinely wrong port.
const EPS: f32 = 1.0e-5;

/// Relative-error denominator floor, keeping the ratio away from zero.
const REL_FLOOR: f32 = 1.0e-6;

/// Returns whether `a` and `b` agree within the absolute or relative bound.
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= EPS || rel <= EPS
}

/// Asserts two `RGB` triples agree channel-wise within [`close`].
fn close_triple(gpu: [f32; 3], cpu: [f32; 3], what: &str, idx: usize) {
    assert!(
        close(gpu[0], cpu[0]) && close(gpu[1], cpu[1]) && close(gpu[2], cpu[2]),
        "{what} mismatch at probe {idx}: gpu {gpu:?}, cpu {cpu:?}"
    );
}

/// The `CPU` golden result for one probe: rotate band `L1` for every channel and
/// band `L2` per channel, keeping the band-`L0` term unchanged.
fn cpu_rotate(probe: &ShRotationProbe) -> ShRotationResult {
    let rot = rotation_matrix_from_quat(probe.quat[0], probe.quat[1], probe.quat[2], probe.quat[3]);
    let l1 = probe.l1.rotate(rot);
    let mut l2 = [[0.0_f32; 3]; L2_COEFF_COUNT];
    for ch in 0..3 {
        let coeffs = [
            probe.l2[0][ch],
            probe.l2[1][ch],
            probe.l2[2][ch],
            probe.l2[3][ch],
            probe.l2[4][ch],
        ];
        let rotated = rotate_l2(coeffs, rot);
        for (k, slot) in l2.iter_mut().enumerate() {
            slot[ch] = rotated[k];
        }
    }
    ShRotationResult { l1, l2 }
}

/// Dispatches `probes` on the `GPU` and asserts every probe matches the `CPU`
/// golden, band `L1` (including the invariant band-`L0` term) and band `L2`
/// across all three `RGB` channels. Returns the `GPU` results for extra checks.
fn check_batch(
    ctx: &GpuContext,
    gpu: &GpuSphericalHarmonicsRotate,
    probes: &[ShRotationProbe],
) -> Vec<ShRotationResult> {
    let results = gpu.eval(ctx, probes);
    assert_eq!(results.len(), probes.len(), "one result per probe");
    for (idx, (probe, got)) in probes.iter().zip(results.iter()).enumerate() {
        let want = cpu_rotate(probe);
        close_triple(got.l1.l0, want.l1.l0, "band-L0", idx);
        for k in 0..3 {
            close_triple(got.l1.l1[k], want.l1.l1[k], "band-L1", idx);
        }
        for k in 0..L2_COEFF_COUNT {
            close_triple(got.l2[k], want.l2[k], "band-L2", idx);
        }
    }
    results
}

/// A tiny integer linear-congruential generator; only integer and divide work,
/// so no transcendental appears. Returns a value in `[-1, 1)`.
fn lcg(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let bits = (*state >> 40) as u32;
    let unit = (bits & 0x00ff_ffff) as f32 / 16_777_216.0;
    unit * 2.0 - 1.0
}

/// A fixed, non-trivial band-`L0`/`L1` payload with distinct `RGB` channels.
fn sample_l1() -> ShL1Rgb {
    ShL1Rgb {
        l0: [0.2, 0.4, 0.6],
        l1: [
            [0.1, -0.2, 0.3],
            [0.4, 0.5, -0.6],
            [-0.7, 0.8, 0.9],
        ],
    }
}

/// A fixed, non-trivial band-`L2` payload with distinct `RGB` channels.
fn sample_l2() -> [[f32; 3]; L2_COEFF_COUNT] {
    [
        [0.11, -0.22, 0.33],
        [-0.44, 0.55, 0.66],
        [0.77, -0.88, 0.19],
        [0.29, 0.37, -0.41],
        [-0.53, 0.61, 0.71],
    ]
}

/// Returns whether any band-`L1` or band-`L2` coefficient moved relative to the
/// input, confirming the rotation is non-trivial.
fn is_non_trivial(before: &ShRotationProbe, after: &ShRotationResult) -> bool {
    let mut moved = false;
    for k in 0..3 {
        for ch in 0..3 {
            if (before.l1.l1[k][ch] - after.l1.l1[k][ch]).abs() > 1.0e-3 {
                moved = true;
            }
        }
    }
    for k in 0..L2_COEFF_COUNT {
        for ch in 0..3 {
            if (before.l2[k][ch] - after.l2[k][ch]).abs() > 1.0e-3 {
                moved = true;
            }
        }
    }
    moved
}

/// Quaternion for a right-angle rotation about the given axis `(ax, ay, az)`.
///
/// Uses the half-angle sine/cosine of 45 degrees (`1/sqrt(2)`) supplied as a
/// literal, so the test itself calls no trigonometric function.
fn quat_axis_90(ax: f32, ay: f32, az: f32) -> [f32; 4] {
    // cos(45 deg) = sin(45 deg) = 1 / sqrt(2).
    let h = core::f32::consts::FRAC_1_SQRT_2;
    [ax * h, ay * h, az * h, h]
}

#[test]
fn identity_rotation_is_passthrough() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSphericalHarmonicsRotate::new(&ctx);
    let probe = ShRotationProbe {
        quat: [0.0, 0.0, 0.0, 1.0],
        l1: sample_l1(),
        l2: sample_l2(),
    };
    let results = check_batch(&ctx, &gpu, &[probe]);
    // Identity must leave every coefficient (and the band-L0 term) untouched.
    close_triple(results[0].l1.l0, probe.l1.l0, "identity band-L0", 0);
    for k in 0..3 {
        close_triple(results[0].l1.l1[k], probe.l1.l1[k], "identity band-L1", 0);
    }
    for k in 0..L2_COEFF_COUNT {
        close_triple(results[0].l2[k], probe.l2[k], "identity band-L2", 0);
    }
}

#[test]
fn rotation_90_about_z_matches_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSphericalHarmonicsRotate::new(&ctx);
    let probe = ShRotationProbe {
        quat: quat_axis_90(0.0, 0.0, 1.0),
        l1: sample_l1(),
        l2: sample_l2(),
    };
    let results = check_batch(&ctx, &gpu, &[probe]);
    assert!(
        is_non_trivial(&probe, &results[0]),
        "a 90-degree z rotation must move some coefficient"
    );
}

#[test]
fn rotation_90_about_x_matches_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSphericalHarmonicsRotate::new(&ctx);
    let probe = ShRotationProbe {
        quat: quat_axis_90(1.0, 0.0, 0.0),
        l1: sample_l1(),
        l2: sample_l2(),
    };
    let results = check_batch(&ctx, &gpu, &[probe]);
    assert!(
        is_non_trivial(&probe, &results[0]),
        "a 90-degree x rotation must move some coefficient"
    );
}

#[test]
fn rotation_90_about_y_matches_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSphericalHarmonicsRotate::new(&ctx);
    let probe = ShRotationProbe {
        quat: quat_axis_90(0.0, 1.0, 0.0),
        l1: sample_l1(),
        l2: sample_l2(),
    };
    let results = check_batch(&ctx, &gpu, &[probe]);
    assert!(
        is_non_trivial(&probe, &results[0]),
        "a 90-degree y rotation must move some coefficient"
    );
}

#[test]
fn random_quaternions_match_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSphericalHarmonicsRotate::new(&ctx);
    let mut state = 0x1234_5678_9abc_def0_u64;
    let mut probes = Vec::new();
    for _ in 0..24 {
        // Non-unit quaternions on purpose; the robust conversion normalizes.
        let quat = [lcg(&mut state), lcg(&mut state), lcg(&mut state), lcg(&mut state)];
        let mut l1 = sample_l1();
        for k in 0..3 {
            for ch in 0..3 {
                l1.l1[k][ch] = lcg(&mut state);
            }
            l1.l0[k] = lcg(&mut state);
        }
        let mut l2 = sample_l2();
        for coeff in &mut l2 {
            for ch in coeff.iter_mut() {
                *ch = lcg(&mut state);
            }
        }
        probes.push(ShRotationProbe { quat, l1, l2 });
    }
    let results = check_batch(&ctx, &gpu, &probes);
    // At least one random rotation must be non-trivial, proving the batch path
    // actually rotates rather than copying inputs.
    assert!(
        probes
            .iter()
            .zip(results.iter())
            .any(|(p, r)| is_non_trivial(p, r)),
        "a batch of random rotations must move some coefficient"
    );
}

#[test]
fn zero_quaternion_acts_as_identity() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSphericalHarmonicsRotate::new(&ctx);
    // Zero quaternion: robust conversion returns identity, so this is a
    // passthrough that still exercises the full dispatch.
    let probe = ShRotationProbe {
        quat: [0.0, 0.0, 0.0, 0.0],
        l1: sample_l1(),
        l2: sample_l2(),
    };
    let results = check_batch(&ctx, &gpu, &[probe]);
    close_triple(results[0].l1.l1[0], probe.l1.l1[0], "zero-quat band-L1", 0);
    close_triple(results[0].l2[0], probe.l2[0], "zero-quat band-L2", 0);
}

#[test]
fn zero_coefficients_stay_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSphericalHarmonicsRotate::new(&ctx);
    // Degenerate all-zero payload under a genuine rotation: output must remain
    // zero (a linear map sends zero to zero) and still match the CPU golden.
    let probe = ShRotationProbe {
        quat: quat_axis_90(0.3, -0.4, 0.5),
        l1: ShL1Rgb {
            l0: [0.0, 0.0, 0.0],
            l1: [[0.0; 3]; 3],
        },
        l2: [[0.0; 3]; L2_COEFF_COUNT],
    };
    let results = check_batch(&ctx, &gpu, &[probe]);
    for k in 0..3 {
        for ch in 0..3 {
            assert!(
                results[0].l1.l1[k][ch].abs() <= EPS,
                "zero band-L1 coefficient must stay zero"
            );
        }
    }
    for k in 0..L2_COEFF_COUNT {
        for ch in 0..3 {
            assert!(
                results[0].l2[k][ch].abs() <= EPS,
                "zero band-L2 coefficient must stay zero"
            );
        }
    }
}

#[test]
fn empty_batch_yields_no_results() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuSphericalHarmonicsRotate::new(&ctx);
    let out = gpu.eval(&ctx, &[]);
    assert!(out.is_empty(), "an empty batch yields no results");
}
