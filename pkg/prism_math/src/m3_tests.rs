//! M3 big-world correctness tests: `f64` type algebra and round-trips,
//! `f32`<->`f64` conversion parity, and the grid-cell origin-rebasing
//! precision contract (including the ±100 km jitter-free guarantee).

use crate::prelude::*;

fn approx(a: f64, b: f64, eps: f64) -> bool {
    (a - b).abs() <= eps
}
fn dvec3_approx(a: DVec3, b: DVec3, eps: f64) -> bool {
    approx(a.x, b.x, eps) && approx(a.y, b.y, eps) && approx(a.z, b.z, eps)
}
fn dmat3_approx(a: DMat3, b: DMat3, eps: f64) -> bool {
    dvec3_approx(a.x_axis, b.x_axis, eps)
        && dvec3_approx(a.y_axis, b.y_axis, eps)
        && dvec3_approx(a.z_axis, b.z_axis, eps)
}

// ---- DVec algebra ---------------------------------------------------------

#[test]
fn dvec3_ops() {
    let a = dvec3(1.0, 2.0, 3.0);
    let b = dvec3(4.0, 5.0, 6.0);
    assert_eq!(a + b, dvec3(5.0, 7.0, 9.0));
    assert_eq!(b - a, dvec3(3.0, 3.0, 3.0));
    assert_eq!(a * 2.0, dvec3(2.0, 4.0, 6.0));
    assert_eq!(2.0 * a, dvec3(2.0, 4.0, 6.0));
    assert_eq!(a.dot(b), 32.0);
    assert_eq!(DVec3::X.cross(DVec3::Y), DVec3::Z);
    assert!(approx(a.length_squared(), 14.0, 1e-12));
}

#[test]
fn dvec3_normalize_is_unit() {
    let v = dvec3(3.0, 4.0, 12.0);
    assert!(approx(v.normalize().length(), 1.0, 1e-15));
    assert_eq!(DVec3::ZERO.normalize_or_zero(), DVec3::ZERO);
    // f64 keeps precision on a magnitude that would lose it badly in f32.
    let big = dvec3(1.0e12, 0.0, 0.0);
    assert!(approx(big.normalize().x, 1.0, 1e-15));
}

#[test]
fn dvec4_truncate_extend_round_trip() {
    let v = dvec4(1.0, 2.0, 3.0, 4.0);
    assert_eq!(v.truncate().extend(4.0), v);
    assert_eq!(dvec2(1.0, 2.0).extend(3.0), dvec3(1.0, 2.0, 3.0));
}

#[test]
fn dvec3_index() {
    let mut v = dvec3(1.0, 2.0, 3.0);
    assert_eq!(v[0], 1.0);
    assert_eq!(v[2], 3.0);
    v[1] = 9.0;
    assert_eq!(v.y, 9.0);
}

// ---- DMat round-trips -----------------------------------------------------

#[test]
fn dmat3_inverse_round_trip() {
    let m = DMat3::from_quat(DQuat::from_axis_angle(dvec3(1.0, 2.0, 3.0).normalize(), 0.9))
        * DMat3::from_scale(dvec3(2.0, 0.5, 1.5));
    let id = m * m.inverse();
    assert!(dvec3_approx(id * DVec3::X, DVec3::X, 1e-12));
    assert!(dvec3_approx(id * DVec3::Y, DVec3::Y, 1e-12));
    assert!(dvec3_approx(id * DVec3::Z, DVec3::Z, 1e-12));
}

#[test]
fn dmat3_determinant_matches_scale() {
    let m = DMat3::from_scale(dvec3(2.0, 3.0, 4.0));
    assert!(approx(m.determinant(), 24.0, 1e-12));
    // Pure rotation has unit determinant.
    let r = DMat3::from_quat(DQuat::from_rotation_z(0.7));
    assert!(approx(r.determinant(), 1.0, 1e-12));
}

#[test]
fn dmat4_inverse_round_trip() {
    let m = DMat4::from_scale_rotation_translation(
        dvec3(2.0, 0.5, 1.5),
        DQuat::from_axis_angle(dvec3(0.3, 1.0, -0.5).normalize(), 1.2),
        dvec3(10.0, -20.0, 30.0),
    );
    let id = m * m.inverse();
    let p = dvec3(1.0, 2.0, 3.0);
    assert!(dvec3_approx(id.transform_point3(p), p, 1e-9));
}

#[test]
fn dmat4_mul_associates() {
    let a = DMat4::from_quat(DQuat::from_rotation_z(0.3));
    let b = DMat4::from_scale(dvec3(2.0, 3.0, 0.5));
    let c = DMat4::from_translation(dvec3(1.0, 2.0, 3.0));
    let lhs = (a * b) * c;
    let rhs = a * (b * c);
    let p = dvec3(1.0, -2.0, 3.0);
    assert!(dvec3_approx(lhs.transform_point3(p), rhs.transform_point3(p), 1e-12));
}

// ---- DQuat ----------------------------------------------------------------

#[test]
fn dquat_rotation_known_value() {
    // 90 deg about Z sends +X to +Y.
    let q = DQuat::from_rotation_z(core::f64::consts::FRAC_PI_2);
    assert!(dvec3_approx(q.mul_vec3(DVec3::X), DVec3::Y, 1e-12));
    assert!(dvec3_approx(q * DVec3::Y, DVec3::NEG_X, 1e-12));
}

#[test]
fn dquat_mat3_round_trip() {
    let q = DQuat::from_axis_angle(dvec3(1.0, -2.0, 0.5).normalize(), 1.3);
    let back = DQuat::from_mat3(DMat3::from_quat(q));
    assert!(q.abs_diff_eq(back, 1e-12));
}

#[test]
fn dquat_slerp_endpoints_and_midpoint() {
    let a = DQuat::IDENTITY;
    let b = DQuat::from_rotation_y(core::f64::consts::FRAC_PI_2);
    assert!(a.slerp(b, 0.0).abs_diff_eq(a, 1e-12));
    assert!(a.slerp(b, 1.0).abs_diff_eq(b, 1e-12));
    let mid = a.slerp(b, 0.5);
    let expected = DQuat::from_rotation_y(core::f64::consts::FRAC_PI_4);
    assert!(mid.abs_diff_eq(expected, 1e-12));
}

#[test]
fn dquat_compose_applies_right_first() {
    let rz = DQuat::from_rotation_z(core::f64::consts::FRAC_PI_2);
    let rx = DQuat::from_rotation_x(core::f64::consts::FRAC_PI_2);
    // (rz * rx) applies rx first, then rz. Compare against matrix composition.
    let v = dvec3(0.3, 0.4, 0.5);
    let q = rz * rx;
    let m = DMat3::from_quat(rz) * DMat3::from_quat(rx);
    assert!(dvec3_approx(q.mul_vec3(v), m * v, 1e-12));
}

// ---- DAffine3 -------------------------------------------------------------

#[test]
fn daffine3_inverse_round_trip() {
    let a = DAffine3::from_scale_rotation_translation(
        dvec3(2.0, 0.5, 1.5),
        DQuat::from_axis_angle(dvec3(0.2, 0.9, -0.3).normalize(), 0.8),
        dvec3(100.0, -200.0, 300.0),
    );
    let id = a * a.inverse();
    let p = dvec3(3.0, -4.0, 5.0);
    assert!(dvec3_approx(id.transform_point3(p), p, 1e-9));
}

#[test]
fn daffine3_srt_recovers_components() {
    let s = dvec3(2.0, 3.0, 0.5);
    let r = DQuat::from_axis_angle(dvec3(0.1, 1.0, 0.2).normalize(), 0.6);
    let t = dvec3(5.0, 6.0, 7.0);
    let a = DAffine3::from_scale_rotation_translation(s, r, t);
    let (s2, r2, t2) = a.to_scale_rotation_translation();
    assert!(dvec3_approx(s, s2, 1e-12));
    assert!(r.abs_diff_eq(r2, 1e-12));
    assert!(dvec3_approx(t, t2, 1e-12));
}

#[test]
fn daffine3_mat4_round_trip() {
    let a = DAffine3::from_scale_rotation_translation(
        dvec3(1.0, 2.0, 3.0),
        DQuat::from_rotation_x(0.5),
        dvec3(1.0, 2.0, 3.0),
    );
    let back = DAffine3::from_mat4(a.to_mat4());
    assert!(dmat3_approx(a.matrix3, back.matrix3, 1e-12));
    assert!(dvec3_approx(a.translation, back.translation, 1e-12));
}

// ---- f32 <-> f64 conversion parity ----------------------------------------

#[test]
fn conversion_round_trips_f32() {
    let v = vec3(1.5, -2.25, 3.75); // exactly representable in both precisions
    assert_eq!(v.as_dvec3().as_vec3(), v);
    let q = Quat::from_rotation_z(0.5);
    assert!(q.as_dquat().as_quat().abs_diff_eq(q, 1e-6));
}

#[test]
fn conversion_parity_rotation() {
    // The f64 path reproduces the f32 facade within f32 tolerance.
    let angle = 0.73_f32;
    let f32_rot = Mat3::from_quat(Quat::from_rotation_y(angle));
    let f64_rot = DMat3::from_quat(DQuat::from_rotation_y(angle as f64));
    let v = vec3(1.0, 2.0, 3.0);
    let a = f32_rot * v;
    let b = f64_rot.as_mat3() * v;
    assert!((a - b).length() <= 1e-5);
}

// ---- Grid-cell origin rebasing: the precision contract --------------------

const CELL: f64 = GridCell::CELL_SIZE;

#[test]
fn gridcell_from_dvec3_floor() {
    assert_eq!(GridCell::from_dvec3(dvec3(0.0, 0.0, 0.0)), GridCell::ZERO);
    assert_eq!(GridCell::from_dvec3(dvec3(CELL, 2.0 * CELL, -1.0)), GridCell::new(1, 2, -1));
    // Just below a boundary stays in the lower cell; negatives floor downward.
    assert_eq!(GridCell::from_dvec3(dvec3(-0.001, 0.0, 0.0)).x, -1);
}

#[test]
fn gridcell_relative_translation_is_exact() {
    let a = GridCell::new(100_000, -50_000, 7);
    let b = GridCell::new(1, 2, 3);
    let t = a.relative_translation(b);
    // Difference of indices scaled by the exact power-of-two cell size: no error.
    assert_eq!(t.x, (100_000i64 - 1) as f64 * CELL);
    assert_eq!(t.y, (-50_000i64 - 2) as f64 * CELL);
    assert_eq!(t.z, (7i64 - 3) as f64 * CELL);
    // A cell relative to itself is the origin.
    assert_eq!(a.relative_translation(a), DVec3::ZERO);
}

#[test]
fn gridposition_round_trip_preserves_mm_at_100km() {
    // 100 km from the origin, with a sub-millimetre feature.
    let world = dvec3(100_000.0, 23_456.789, -100_000.0 + 0.012_3);
    let gp = GridPosition::from_dvec3(world);
    let back = gp.to_dvec3();
    // The exact f64 position reconstructs to within one local-offset ULP:
    // CELL * 2^-23 ~= 0.122 mm.
    let tol = CELL * 2f64.powi(-23);
    assert!(dvec3_approx(back, world, tol), "round trip error exceeded {tol} m");
    // Offset is canonicalised into [0, CELL).
    assert!(gp.offset.x >= 0.0 && gp.offset.x < CELL as f32);
}

#[test]
fn rebasing_resolves_mm_jitter_that_naive_f32_loses() {
    // Camera parked ~123 km from the world origin. Two scene points 1 mm apart.
    let base = dvec3(123_000.0, 45_000.0, -98_000.0);
    let p0 = GridPosition::from_dvec3(base);
    let p1 = GridPosition::from_dvec3(base + dvec3(0.001, 0.0, 0.0));

    // Naive narrowing to f32 world coordinates collapses the 1 mm difference:
    // at ~123 km the f32 ULP (~0.0156 m) swallows it entirely.
    let n0 = base.as_vec3();
    let n1 = (base + dvec3(0.001, 0.0, 0.0)).as_vec3();
    assert_eq!(n0, n1, "naive f32 world coords should collapse a 1 mm offset");

    // Rebasing both points to the camera's cell keeps the 1 mm resolvable.
    let cam = GridCell::from_dvec3(base);
    let r0 = p0.rebased_offset(cam);
    let r1 = p1.rebased_offset(cam);
    assert!((r1.x - r0.x - 0.001).abs() <= 1e-4, "rebased 1 mm offset lost: {} vs {}", r0.x, r1.x);
}

#[test]
fn rebasing_small_offsets_stay_submillimetre_near_camera() {
    // A point 2 km from the camera cell, with the scene 100 km from origin.
    let cam = GridCell::from_dvec3(dvec3(100_000.0, 0.0, 0.0));
    let world = cam.origin() + dvec3(2_000.0, 10.5, -3.25);
    let gp = GridPosition::from_dvec3(world);
    let rebased = gp.rebased_offset(cam);
    // Reconstruct the true camera-relative position in f64 and compare.
    let truth = (world - cam.origin()).as_vec3();
    assert!((rebased - truth).length() <= 5e-4, "near-camera rebasing exceeded 0.5 mm");
}

#[test]
fn gridposition_recenter_folds_drift_into_cell() {
    // Integrate motion that pushes the local offset past a cell boundary.
    let mut gp = GridPosition::new(GridCell::new(5, 0, 0), vec3(CELL as f32 - 1.0, 0.0, 0.0));
    gp.offset.x += 3.0; // now outside [0, CELL)
    let before = gp.to_dvec3();
    let fixed = gp.recenter();
    assert_eq!(fixed.cell.x, 6);
    assert!(fixed.offset.x >= 0.0 && fixed.offset.x < CELL as f32);
    // Recentering preserves the world position (within local-offset ULP).
    assert!(dvec3_approx(fixed.to_dvec3(), before, CELL * 2f64.powi(-23)));
}

#[test]
fn relative_transform_matches_translation() {
    let a = GridCell::new(50, -3, 12);
    let b = GridCell::new(1, 1, 1);
    let xf = a.relative_transform(b);
    assert_eq!(xf.translation, a.relative_translation(b));
    assert_eq!(xf.matrix3, DMat3::IDENTITY);
    // Transforming a local point adds the exact inter-cell translation.
    let local = dvec3(1.0, 2.0, 3.0);
    assert_eq!(xf.transform_point3(local), local + a.relative_translation(b));
}
