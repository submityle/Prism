//! M6 tests: dual quaternions, `f16`, octahedral normals, `SoA` batches, and
//! swizzles.
//!
//! Tests run with `std` available (the crate is only `no_std` for non-test
//! builds), so `f64`/`std` facilities are used purely as test scaffolding and
//! independent oracles; they never participate in the code paths under test.
//!
//! Coverage:
//! - **`f16`:** exact known bit patterns, infinities/`NaN`/subnormals, a
//!   brute-force nearest-representable oracle (proving round-to-nearest), tie
//!   to even, overflow saturation, and vector round-trips.
//! - **Octahedral:** unit-direction round-trips (both hemispheres and axes)
//!   within tolerance and the packed snorm path within its coarser tolerance.
//! - **Dual quat:** parity with rotation-plus-translation, composition
//!   associativity against point transforms, inverse, normalize invariants,
//!   `nlerp`/`sclerp` endpoints, constant-speed half-angle, and weighted blend.
//! - **`SoA`:** array-of-structures round-trip and batch-vs-per-element parity.
//! - **Swizzle:** representative permutations and broadcasts.

use crate::float::f32 as mf;
use crate::octahedral;
use crate::prelude::*;

const PI: f32 = core::f32::consts::PI;

fn approx(a: f32, b: f32, eps: f32) -> bool {
    (a - b).abs() <= eps
}

// --------------------------------------------------------------------------
// f16
// --------------------------------------------------------------------------

#[test]
fn f16_known_bit_patterns() {
    assert_eq!(F16::from_f32(0.0).to_bits(), 0x0000);
    assert_eq!(F16::from_f32(-0.0).to_bits(), 0x8000);
    assert_eq!(F16::from_f32(1.0).to_bits(), 0x3C00);
    assert_eq!(F16::from_f32(-1.0).to_bits(), 0xBC00);
    assert_eq!(F16::from_f32(2.0).to_bits(), 0x4000);
    assert_eq!(F16::from_f32(0.5).to_bits(), 0x3800);
    assert_eq!(F16::from_f32(65504.0).to_bits(), 0x7BFF); // max finite
                                                          // smallest positive subnormal 2^-24 and normal 2^-14
    assert_eq!(F16::from_f32(f32::from_bits(0x3380_0000)).to_bits(), 0x0001);
    assert_eq!(F16::from_f32(f32::from_bits(0x3880_0000)).to_bits(), 0x0400);
}

#[test]
fn f16_specials() {
    assert!(F16::from_f32(f32::INFINITY).is_infinite());
    assert_eq!(F16::from_f32(f32::INFINITY).to_bits(), 0x7C00);
    assert_eq!(F16::from_f32(f32::NEG_INFINITY).to_bits(), 0xFC00);
    assert!(F16::from_f32(f32::NAN).is_nan());
    assert!(F16::from_f32(f32::NAN).to_f32().is_nan());
    // overflow saturates to signed infinity
    assert!(F16::from_f32(70000.0).is_infinite());
    assert!(F16::from_f32(70000.0).to_f32() > 0.0);
    assert!(F16::from_f32(-70000.0).to_f32() < 0.0);
    // underflow to zero
    assert_eq!(F16::from_f32(1.0e-30).to_bits(), 0x0000);
    // round-trip of specials in f16->f32
    assert_eq!(F16::INFINITY.to_f32(), f32::INFINITY);
    assert!(F16::NAN.to_f32().is_nan());
    assert_eq!(
        F16::from_f32(f32::from_bits(0x3380_0000)).to_f32(),
        f32::from_bits(0x3380_0000)
    );
}

/// Brute-force the nearest representable finite `f16` to `x` by scanning all
/// bit patterns. Independent oracle for the production rounding path.
fn nearest_f16_error(x: f32) -> f32 {
    let mut best = f32::INFINITY;
    for bits in 0u16..=0xFFFF {
        // Skip inf/nan encodings.
        if (bits & 0x7C00) == 0x7C00 {
            continue;
        }
        let v = F16::from_bits(bits).to_f32();
        let e = (v - x).abs();
        if e < best {
            best = e;
        }
    }
    best
}

#[test]
fn f16_round_to_nearest_matches_brute_force() {
    // A spread of magnitudes inside the finite f16 range.
    let samples = [
        0.3333_f32,
        1.0 / 7.0,
        123.456,
        -98.765,
        0.001_234,
        1024.5,
        0.0625,
        -0.1,
        42.0,
        core::f32::consts::PI,
        1000.0,
        0.5003,
        -2.5,
        60000.0,
        0.00006,
        7.7777,
    ];
    for &x in &samples {
        let got = (F16::from_f32(x).to_f32() - x).abs();
        let best = nearest_f16_error(x);
        assert!(
            got <= best + 0.0,
            "x={x}: chosen err {got} > nearest {best}",
        );
    }
}

#[test]
fn f16_round_ties_to_even() {
    // Halfway between 1.0 (0x3C00, even) and 1+2^-10 (0x3C01, odd) rounds down.
    let tie_down = 1.0 + f32::from_bits(0x3A00_0000);
    assert_eq!(F16::from_f32(tie_down).to_bits(), 0x3C00);
    // Halfway between 0x3C02 (even) and 0x3C01 (odd) rounds up to even 0x3C02.
    let a = F16::from_bits(0x3C01).to_f32();
    let b = F16::from_bits(0x3C02).to_f32();
    assert_eq!(F16::from_f32(0.5 * (a + b)).to_bits(), 0x3C02);
}

#[test]
fn f16_vec_round_trips() {
    let v3 = vec3(1.0, -2.0, 0.5);
    assert_eq!(F16Vec3::from_vec3(v3).to_vec3(), v3);
    let v4 = vec4(0.25, -0.5, 8.0, -16.0);
    assert_eq!(F16Vec4::from_vec4(v4).to_vec4(), v4);
    let v2 = vec2(3.0, -4.0);
    assert_eq!(F16Vec2::from_vec2(v2).to_vec2(), v2);
}

#[test]
fn f16_arithmetic_matches_f32_rounding() {
    let a = F16::from_f32(1.5);
    let b = F16::from_f32(2.25);
    assert_eq!((a + b), F16::from_f32(3.75));
    assert_eq!((a * b), F16::from_f32(1.5 * 2.25));
    assert_eq!((-a).to_f32(), -1.5);
}

// --------------------------------------------------------------------------
// octahedral
// --------------------------------------------------------------------------

fn sample_dirs() -> impl Iterator<Item = Vec3> {
    let mut v = alloc::vec![Vec3::X, -Vec3::X, Vec3::Y, -Vec3::Y, Vec3::Z, -Vec3::Z];
    // A deterministic spread over the sphere.
    let n = 400;
    for i in 0..n {
        let u = (i as f32 + 0.5) / n as f32;
        let z = 1.0 - 2.0 * u;
        let r = (1.0 - z * z).max(0.0).sqrt();
        let phi = (i as f32) * 2.399_963_2; // golden angle
        let (sp, cp) = mf::sin_cos(phi);
        v.push(Vec3::new(r * cp, r * sp, z).normalize());
    }
    v.into_iter()
}

#[test]
fn octahedral_full_precision_round_trip() {
    for n in sample_dirs() {
        let d = octahedral::decode(octahedral::encode(n));
        let dot = n.dot(d);
        assert!(dot > 1.0 - 1.0e-5, "dot={dot} for {n:?}");
    }
}

#[test]
fn octahedral_packed_round_trip() {
    for n in sample_dirs() {
        let d = octahedral::unpack_snorm(octahedral::pack_snorm(n));
        let dot = n.dot(d);
        assert!(dot > 1.0 - 1.0e-3, "packed dot={dot} for {n:?}");
    }
}

// --------------------------------------------------------------------------
// dual quaternion
// --------------------------------------------------------------------------

fn rot_y(a: f32) -> Quat {
    Quat::from_rotation_y(a)
}

#[test]
fn dq_matches_rotation_translation() {
    let q = rot_y(0.7);
    let t = vec3(3.0, -1.0, 2.0);
    let dq = DualQuat::from_rotation_translation(q, t);
    // decomposition round-trips
    let (q2, t2) = dq.to_rotation_translation();
    assert!(q.abs_diff_eq(q2, 1.0e-5));
    assert!(approx(t.x, t2.x, 1.0e-5) && approx(t.y, t2.y, 1.0e-5) && approx(t.z, t2.z, 1.0e-5));
    // transform parity
    for p in [vec3(1.0, 2.0, 3.0), vec3(-5.0, 0.0, 1.0), Vec3::ZERO] {
        let expect = q.mul_vec3(p) + t;
        let got = dq.transform_point3(p);
        assert!((expect - got).length() < 1.0e-4, "{expect:?} vs {got:?}");
    }
}

#[test]
fn dq_composition_is_associative_on_points() {
    let a = DualQuat::from_rotation_translation(rot_y(0.5), vec3(1.0, 0.0, 0.0));
    let b = DualQuat::from_rotation_translation(Quat::from_rotation_x(0.9), vec3(0.0, 2.0, -1.0));
    let p = vec3(2.0, -3.0, 4.0);
    let composed = (a * b).transform_point3(p);
    let stepwise = a.transform_point3(b.transform_point3(p));
    assert!((composed - stepwise).length() < 1.0e-4);
}

#[test]
fn dq_inverse_round_trips() {
    let dq = DualQuat::from_rotation_translation(Quat::from_rotation_z(1.2), vec3(4.0, 5.0, 6.0));
    let id = dq.inverse() * dq;
    let p = vec3(7.0, -8.0, 9.0);
    assert!((id.transform_point3(p) - p).length() < 1.0e-4);
}

#[test]
fn dq_normalize_keeps_unit_invariants() {
    let dq = DualQuat::from_rotation_translation(rot_y(0.3), vec3(10.0, -2.0, 3.0));
    // Scale both parts then renormalize.
    let scaled = DualQuat::from_real_dual(
        Quat::from_xyzw(
            dq.real.x * 3.0,
            dq.real.y * 3.0,
            dq.real.z * 3.0,
            dq.real.w * 3.0,
        ),
        Quat::from_xyzw(
            dq.dual.x * 3.0,
            dq.dual.y * 3.0,
            dq.dual.z * 3.0,
            dq.dual.w * 3.0,
        ),
    );
    let n = scaled.normalize();
    assert!(approx(n.real.length(), 1.0, 1.0e-5));
    assert!(approx(n.real.dot(n.dual), 0.0, 1.0e-5));
    // Still the same rigid transform.
    let p = vec3(1.0, 1.0, 1.0);
    assert!((n.transform_point3(p) - dq.transform_point3(p)).length() < 1.0e-4);
}

#[test]
fn dq_nlerp_endpoints() {
    let a = DualQuat::from_rotation_translation(rot_y(0.2), vec3(0.0, 0.0, 0.0));
    let b = DualQuat::from_rotation_translation(rot_y(1.1), vec3(2.0, 0.0, 0.0));
    let p = vec3(1.0, 0.0, 0.0);
    assert!((a.nlerp(b, 0.0).transform_point3(p) - a.transform_point3(p)).length() < 1.0e-4);
    assert!((a.nlerp(b, 1.0).transform_point3(p) - b.transform_point3(p)).length() < 1.0e-4);
}

#[test]
fn dq_sclerp_endpoints_and_half_angle() {
    let a = DualQuat::from_rotation(Quat::IDENTITY);
    let b = DualQuat::from_rotation(rot_y(PI * 0.5)); // 90 degrees
                                                      // endpoints
    let p = vec3(1.0, 0.0, 0.0);
    assert!((a.sclerp(b, 0.0).transform_point3(p) - a.transform_point3(p)).length() < 1.0e-4);
    assert!((a.sclerp(b, 1.0).transform_point3(p) - b.transform_point3(p)).length() < 1.0e-4);
    // half-way should be a 45-degree rotation about Y
    let mid = a.sclerp(b, 0.5);
    let expect = DualQuat::from_rotation(rot_y(PI * 0.25));
    assert!((mid.transform_point3(p) - expect.transform_point3(p)).length() < 1.0e-4);
}

#[test]
fn dq_sclerp_pure_translation() {
    let a = DualQuat::from_translation(vec3(0.0, 0.0, 0.0));
    let b = DualQuat::from_translation(vec3(10.0, 0.0, 0.0));
    let mid = a.sclerp(b, 0.5);
    let p = vec3(0.0, 0.0, 0.0);
    assert!((mid.transform_point3(p) - vec3(5.0, 0.0, 0.0)).length() < 1.0e-4);
}

#[test]
fn dq_blend_weighted_single_is_identity_blend() {
    let a = DualQuat::from_rotation_translation(rot_y(0.6), vec3(1.0, 2.0, 3.0));
    let blended = DualQuat::blend_weighted(&[(a, 1.0)]);
    let p = vec3(3.0, 2.0, 1.0);
    assert!((blended.transform_point3(p) - a.transform_point3(p)).length() < 1.0e-4);
    // empty -> identity
    let empty = DualQuat::blend_weighted(&[]);
    assert!((empty.transform_point3(p) - p).length() < 1.0e-6);
}

// --------------------------------------------------------------------------
// SoA
// --------------------------------------------------------------------------

#[test]
fn soa_aos_round_trip() {
    let src = [
        vec3(1.0, 2.0, 3.0),
        vec3(-4.0, 5.0, -6.0),
        vec3(0.5, 0.0, 9.0),
    ];
    let soa = SoaVec3::from_aos(&src);
    assert_eq!(soa.len(), 3);
    assert_eq!(soa.to_aos(), src.to_vec());
    assert_eq!(soa.xs(), &[1.0, -4.0, 0.5]);
}

#[test]
fn soa_transform_points_parity() {
    let src = [
        vec3(1.0, 2.0, 3.0),
        vec3(-1.0, 0.5, 2.0),
        vec3(4.0, -4.0, 1.0),
    ];
    let a = Affine3::from_scale_rotation_translation(
        vec3(2.0, 1.0, 0.5),
        rot_y(0.8),
        vec3(1.0, -2.0, 3.0),
    );
    let mut soa = SoaVec3::from_aos(&src);
    soa.transform_points(&a);
    for (i, &p) in src.iter().enumerate() {
        let expect = a.transform_point3(p);
        assert!((soa.get(i) - expect).length() < 1.0e-4);
    }
}

#[test]
fn soa_batch_ops_parity() {
    let src = [
        vec3(3.0, 4.0, 0.0),
        vec3(1.0, 2.0, 2.0),
        vec3(0.0, 0.0, 5.0),
    ];
    let soa = SoaVec3::from_aos(&src);
    // dot against self == length_squared
    let mut dots = [0.0_f32; 3];
    soa.dot_batch(&soa, &mut dots);
    for (i, &p) in src.iter().enumerate() {
        assert!(approx(dots[i], p.length_squared(), 1.0e-4));
    }
    // length
    let mut lens = [0.0_f32; 3];
    soa.length_batch(&mut lens);
    for (i, &p) in src.iter().enumerate() {
        assert!(approx(lens[i], p.length(), 1.0e-4));
    }
    // normalize
    let mut nsoa = SoaVec3::from_aos(&src);
    nsoa.normalize();
    for (i, &p) in src.iter().enumerate() {
        assert!((nsoa.get(i) - p.normalize()).length() < 1.0e-4);
    }
}

// --------------------------------------------------------------------------
// swizzle
// --------------------------------------------------------------------------

#[test]
fn swizzle_basic() {
    let v = vec3(1.0, 2.0, 3.0);
    assert_eq!(v.zyx(), vec3(3.0, 2.0, 1.0));
    assert_eq!(v.xy(), vec2(1.0, 2.0));
    assert_eq!(v.xxxx(), vec4(1.0, 1.0, 1.0, 1.0));

    let w = vec4(1.0, 2.0, 3.0, 4.0);
    assert_eq!(w.wzyx(), vec4(4.0, 3.0, 2.0, 1.0));
    assert_eq!(w.xy(), vec2(1.0, 2.0));
    assert_eq!(w.xyz(), vec3(1.0, 2.0, 3.0));

    let a = vec3a(5.0, 6.0, 7.0);
    assert_eq!(a.zyx(), vec3a(7.0, 6.0, 5.0));
    assert_eq!(a.xy(), vec2(5.0, 6.0));
}
