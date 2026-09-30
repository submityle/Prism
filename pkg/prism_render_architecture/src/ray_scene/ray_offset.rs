//! Watertight secondary-ray origin offset (self-intersection avoidance).
//!
//! When a closest-hit walk spawns a shadow, reflection, or `GI` ray from a
//! surface point, floating-point rounding in the reconstructed hit position lets
//! the new ray immediately re-intersect the surface it left, producing shadow
//! acne and speckle. Nudging the origin by a fixed epsilon fails at both ends of
//! the scene scale: too small near the far field, too large near the origin.
//!
//! This module implements the adaptive integer-`ULP` offset of Wächter and
//! Binder ("A Fast and Robust Method for Avoiding Self-Intersection", *Ray
//! Tracing Gems*, chapter 6) — the same technique production tracers use. It
//! scales the push by the floating-point magnitude of each coordinate: far from
//! the origin, where `ULP`s are large, it steps a fixed number of representable
//! values along the surface normal; near the origin, where `ULP`s underflow the
//! useful range, it falls back to a small absolute step. The result stays just
//! off the surface at every scale without a hand-tuned scene-dependent epsilon.
//!
//! Only bit reinterpretation ([`f32::to_bits`] / [`f32::from_bits`]), integer
//! and floating add/multiply, and comparisons are used — no transcendental
//! functions — so a `GPU` twin (`int_as_float` / `float_as_int` with the same
//! constants) reproduces it bit-for-bit.

/// Coordinate magnitude below which the additive (near-origin) branch is taken.
///
/// Inside `[-ORIGIN, ORIGIN]` the exponent is small enough that integer-`ULP`
/// stepping barely moves the value, so a fixed fractional step is used instead.
const ORIGIN: f32 = 1.0 / 32.0;

/// Absolute step (scaled by the normal) used in the near-origin branch.
const FLOAT_SCALE: f32 = 1.0 / 65_536.0;

/// Number of representable `f32` values stepped per unit of normal in the
/// integer-`ULP` branch.
const INT_SCALE: f32 = 256.0;

/// Offsets a surface hit point off the geometry so a secondary ray spawned from
/// it will not self-intersect the originating primitive.
///
/// `point` is the surface hit position (for example
/// [`super::bvh::Triangle::point_at`]) and `normal` is the unit geometric normal
/// oriented toward the side the new ray leaves on (for a reflection or a shadow
/// ray toward a light above the surface, the front-facing
/// [`super::bvh::Triangle::geometric_normal`]; flip it for a transmitted ray).
///
/// The returned origin is displaced along `normal` by an amount that adapts to
/// each coordinate's floating-point scale, so it is robust from millimetre props
/// to kilometre terrain without a tuned epsilon. Passing a zero normal (e.g. a
/// degenerate triangle) leaves the coordinate on the additive branch with a zero
/// step, so the point is returned effectively unchanged rather than corrupted.
#[must_use]
pub fn offset_ray_origin(point: [f32; 3], normal: [f32; 3]) -> [f32; 3] {
    [
        offset_component(point[0], normal[0]),
        offset_component(point[1], normal[1]),
        offset_component(point[2], normal[2]),
    ]
}

/// Offsets a single coordinate per the Wächter-Binder rule.
///
/// Far from the origin (`|p| >= ORIGIN`) the coordinate is stepped by
/// `INT_SCALE * n` representable values, added in the integer domain and away
/// from zero so the push always moves along the normal's sign. Near the origin
/// the fixed `FLOAT_SCALE * n` step is added directly, where floating-point
/// resolution is fine enough for a plain additive nudge.
fn offset_component(p: f32, n: f32) -> f32 {
    // Truncate toward zero, matching the reference `int3(int_scale * n)` cast.
    let of_i = (INT_SCALE * n) as i32;
    let signed = if p < 0.0 { -of_i } else { of_i };
    let p_i = f32::from_bits(((p.to_bits() as i32).wrapping_add(signed)) as u32);
    if p.abs() < ORIGIN {
        p + FLOAT_SCALE * n
    } else {
        p_i
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }

    fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
        [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
    }

    #[test]
    fn offset_moves_along_the_normal() {
        // A far-field point on a +z-facing surface must be pushed toward +z.
        let p = [1000.0, -250.0, 37.5];
        let n = [0.0, 0.0, 1.0];
        let o = offset_ray_origin(p, n);
        let delta = sub(o, p);
        assert!(dot(delta, n) > 0.0, "offset must move along the normal");
        // The offset is tiny relative to the coordinate magnitude.
        assert!(delta[2] > 0.0 && delta[2] < 1.0);
    }

    #[test]
    fn offset_direction_follows_normal_sign() {
        // Negative normal pushes the point the other way.
        let p = [1000.0, 250.0, 37.5];
        let up = offset_ray_origin(p, [0.0, 0.0, 1.0]);
        let down = offset_ray_origin(p, [0.0, 0.0, -1.0]);
        assert!(up[2] > p[2]);
        assert!(down[2] < p[2]);
    }

    #[test]
    fn near_origin_uses_additive_branch() {
        // |p| < ORIGIN on every axis -> the fixed FLOAT_SCALE step is used.
        let p = [0.01, -0.02, 0.0];
        let n = [1.0, 1.0, 1.0];
        let o = offset_ray_origin(p, n);
        assert!((o[0] - (p[0] + FLOAT_SCALE)).abs() <= 1.0e-9);
        assert!((o[1] - (p[1] + FLOAT_SCALE)).abs() <= 1.0e-9);
        assert!((o[2] - (p[2] + FLOAT_SCALE)).abs() <= 1.0e-9);
    }

    #[test]
    fn offset_scales_with_coordinate_magnitude() {
        // The integer-ULP push grows with the exponent: a far point moves more
        // in absolute terms than a nearer one for the same normal.
        let n = [0.0, 0.0, 1.0];
        let near = offset_ray_origin([0.0, 0.0, 1.0], n)[2] - 1.0;
        let far = offset_ray_origin([0.0, 0.0, 65_536.0], n)[2] - 65_536.0;
        assert!(near > 0.0 && far > 0.0);
        assert!(far > near, "far-field ULP step must exceed the near-field one");
    }

    #[test]
    fn zero_normal_leaves_point_unchanged() {
        // A degenerate normal must not corrupt the origin (no NaN, no jump).
        let p = [12.0, -3.0, 0.5];
        let o = offset_ray_origin(p, [0.0, 0.0, 0.0]);
        assert_eq!(o, p);
    }

    #[test]
    fn offset_is_deterministic() {
        let p = [5.0, 5.0, 5.0];
        let n = [0.577, 0.577, 0.577];
        assert_eq!(offset_ray_origin(p, n), offset_ray_origin(p, n));
    }
}
