//! §24.2 tests: deterministic pose quantization with hand-computed oracles.
//!
//! Each numeric oracle is derived by hand from the round-half-up rule so the
//! codec is pinned to an exact code, not just "close enough".

use crate::quantize::{
    BoundedQuantizer, LayeredQuantizer, PoseQuantizer, QuatQuantized, QuatQuantizer,
    ScaleQuantizer,
};
use crate::Transform;
use prism_math::{vec3, Quat, Vec3};

fn approx(a: f32, b: f32, eps: f32) {
    assert!((a - b).abs() <= eps, "expected {a} ≈ {b} (eps {eps})");
}

fn quat_dot(a: Quat, b: Quat) -> f32 {
    a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w
}

// ---- bounded translation: exact grid + hand-computed off-grid code ---------

#[test]
fn bounded_grid_points_decode_exactly() {
    // min 0, max 255, 8 bits -> levels 255, grid points are the integers 0..=255.
    let q = BoundedQuantizer::new(Vec3::ZERO, vec3(255.0, 255.0, 255.0), 8);
    for &v in &[0.0f32, 1.0, 100.0, 200.0, 255.0] {
        let enc = q.encode(vec3(v, v, v));
        let dec = q.decode(enc);
        approx(dec.x, v, 1.0e-4);
        approx(dec.y, v, 1.0e-4);
        approx(dec.z, v, 1.0e-4);
    }
}

#[test]
fn bounded_off_grid_matches_hand_computed_code() {
    // range [0, 10], 8 bits (levels 255), value 5.0.
    //   normalized = 5/10 = 0.5
    //   scaled     = 0.5 * 255 + 0.5 = 128.0  -> code 128
    //   decode     = 128/255 * 10 = 5.019607...
    let q = BoundedQuantizer::new(Vec3::ZERO, vec3(10.0, 10.0, 10.0), 8);
    let enc = q.encode(vec3(5.0, 5.0, 5.0));
    assert_eq!(enc.codes, [128, 128, 128]);
    let dec = q.decode(enc);
    approx(dec.x, 5.019_608, 1.0e-5);

    // The round-trip error must respect the advertised half-step bound.
    let bound = q.max_abs_error();
    approx(bound.x, 0.5 * 10.0 / 255.0, 1.0e-6);
    assert!((dec.x - 5.0).abs() <= bound.x + 1.0e-6);
}

#[test]
fn bounded_saturates_out_of_range() {
    let q = BoundedQuantizer::new(Vec3::ZERO, vec3(10.0, 10.0, 10.0), 8);
    // Below min and above max both saturate to the extreme codes.
    assert_eq!(q.encode(vec3(-5.0, -5.0, -5.0)).codes, [0, 0, 0]);
    assert_eq!(q.encode(vec3(50.0, 50.0, 50.0)).codes, [255, 255, 255]);
}

// ---- smallest-three quaternion codec ---------------------------------------

#[test]
fn quat_identity_round_trips_exactly() {
    let q = QuatQuantizer::DEFAULT;
    let enc = q.encode(Quat::IDENTITY);
    // w is the largest component (index 3); kept x,y,z are all zero.
    assert_eq!(enc.largest, 3);
    // Zero maps to the midpoint code: round(0.5 * 1023 + 0.5) = 512.
    assert_eq!((enc.a, enc.b, enc.c), (512, 512, 512));
    let dec = q.decode(enc);
    approx(quat_dot(dec, Quat::IDENTITY).abs(), 1.0, 1.0e-6);
}

#[test]
fn quat_rotation_round_trips_within_component_bound() {
    let q = QuatQuantizer::DEFAULT;
    let originals = [
        Quat::from_rotation_z(core::f32::consts::FRAC_PI_2),
        Quat::from_rotation_x(0.7),
        Quat::from_axis_angle(vec3(1.0, 2.0, 3.0).normalize(), 1.23),
    ];
    for &orig in &originals {
        let dec = q.decode(q.encode(orig));
        // Same rotation => |dot| ≈ 1 (allow the small quantization gap).
        let d = quat_dot(dec, orig.normalize()).abs();
        assert!(d > 0.9999, "dot {d} too small for {orig:?}");
    }
    // 10 bits over [-1/sqrt2, 1/sqrt2]: half-step bound.
    approx(
        q.max_component_error(),
        core::f32::consts::FRAC_1_SQRT_2 / 1023.0,
        1.0e-7,
    );
}

#[test]
fn quat_to_bits_round_trips_bit_pattern() {
    let packed = QuatQuantized {
        largest: 2,
        a: 1023,
        b: 512,
        c: 7,
    };
    let word = packed.to_bits();
    // a in bits 0..10, b in 10..20, c in 20..30, largest in 30..32.
    assert_eq!(word & 0x3ff, 1023);
    assert_eq!((word >> 10) & 0x3ff, 512);
    assert_eq!((word >> 20) & 0x3ff, 7);
    assert_eq!((word >> 30) & 0x3, 2);
    assert_eq!(QuatQuantized::from_bits(word), packed);
}

// ---- scale: 1-bit unit bypass ----------------------------------------------

#[test]
fn scale_unit_fast_path_is_exact() {
    let q = ScaleQuantizer::new(0.1, 10.0, 10, 1.0e-3);
    let enc = q.encode(Vec3::ONE);
    assert!(enc.is_unit);
    assert_eq!(q.decode(enc), Vec3::ONE);
}

#[test]
fn scale_non_unit_respects_bound() {
    let q = ScaleQuantizer::new(0.0, 4.0, 8, 1.0e-3);
    let enc = q.encode(vec3(2.0, 3.0, 0.5));
    assert!(!enc.is_unit);
    let dec = q.decode(enc);
    let bound = q.max_abs_error();
    approx(bound, 0.5 * 4.0 / 255.0, 1.0e-6);
    assert!((dec.x - 2.0).abs() <= bound + 1.0e-6);
    assert!((dec.y - 3.0).abs() <= bound + 1.0e-6);
    assert!((dec.z - 0.5).abs() <= bound + 1.0e-6);
}

// ---- layered big-world translation -----------------------------------------

#[test]
fn layered_split_matches_hand_computed_cell_and_offset() {
    // cell_size 100, 10 bits (levels 1023), value 250.5.
    //   cell      = floor(250.5 / 100) = 2
    //   remainder = 250.5 - 2*100 = 50.5
    //   norm      = 50.5 / 100 = 0.505
    //   code      = round(0.505 * 1023 + 0.5) = floor(517.115) = 517
    //   decode    = 2*100 + 517/1023 * 100 = 250.5376...
    let q = LayeredQuantizer::new(100.0, 10);
    let enc = q.encode(vec3(250.5, 0.0, 0.0));
    assert_eq!(enc.cell[0], 2);
    assert_eq!(enc.offset[0], 517);
    let dec = q.decode(enc);
    approx(dec.x, 250.537_6, 1.0e-3);
    assert!((dec.x - 250.5).abs() <= q.max_abs_error() + 1.0e-5);
}

#[test]
fn layered_handles_negative_coordinates() {
    // value -50 with cell_size 100: floor(-0.5) = -1, remainder = 50.
    let q = LayeredQuantizer::new(100.0, 10);
    let enc = q.encode(vec3(-50.0, 0.0, 0.0));
    assert_eq!(enc.cell[0], -1);
    let dec = q.decode(enc);
    assert!((dec.x - (-50.0)).abs() <= q.max_abs_error() + 1.0e-5);
}

#[test]
fn layered_precision_is_uniform_far_from_origin() {
    // The absolute error is the same near and far from the origin, unlike a
    // single global quantizer whose step grows with the extent.
    let q = LayeredQuantizer::new(100.0, 10);
    let bound = q.max_abs_error();
    for &base in &[0.0f32, 1_000.0, 1_000_000.0] {
        let v = vec3(base + 33.3, 0.0, 0.0);
        let dec = q.decode(q.encode(v));
        assert!((dec.x - v.x).abs() <= bound + 1.0e-2);
    }
}

// ---- whole-pose codec -------------------------------------------------------

#[test]
fn pose_codec_round_trips_whole_transform() {
    let codec = PoseQuantizer::new(
        BoundedQuantizer::new(vec3(-100.0, -100.0, -100.0), vec3(100.0, 100.0, 100.0), 16),
        QuatQuantizer::DEFAULT,
        ScaleQuantizer::new(0.1, 10.0, 12, 1.0e-3),
    );
    let t = Transform {
        translation: vec3(12.5, -7.25, 33.0),
        rotation: Quat::from_rotation_y(0.9),
        scale: vec3(2.0, 2.0, 2.0),
    };
    let dec = codec.decode(&codec.encode(&t));

    let terr = codec.max_translation_error();
    assert!((dec.translation.x - t.translation.x).abs() <= terr.x + 1.0e-3);
    assert!((dec.translation.y - t.translation.y).abs() <= terr.y + 1.0e-3);
    assert!((dec.translation.z - t.translation.z).abs() <= terr.z + 1.0e-3);

    let serr = codec.max_scale_error();
    assert!((dec.scale.x - t.scale.x).abs() <= serr + 1.0e-3);

    let d = quat_dot(dec.rotation, t.rotation.normalize()).abs();
    assert!(d > 0.9999, "rotation dot {d} too small");
}

#[test]
fn pose_codec_unit_scale_decodes_exactly() {
    let codec = PoseQuantizer::new(
        BoundedQuantizer::new(Vec3::ZERO, vec3(10.0, 10.0, 10.0), 10),
        QuatQuantizer::DEFAULT,
        ScaleQuantizer::new(0.1, 10.0, 10, 1.0e-3),
    );
    let t = Transform {
        translation: Vec3::ZERO,
        rotation: Quat::IDENTITY,
        scale: Vec3::ONE,
    };
    let dec = codec.decode(&codec.encode(&t));
    assert_eq!(dec.scale, Vec3::ONE);
}
