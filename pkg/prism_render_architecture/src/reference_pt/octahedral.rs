//! Trigonometry-free octahedral mapping between unit directions and the square.
//!
//! The environment light stores its radiance on the unit square `[-1, 1]^2` and
//! must convert between a world-space direction and its texel coordinate without
//! any `sin`/`cos` (the workspace determinism policy forbids transcendentals
//! other than `sqrt`). The octahedral parameterization of Cigolle, Donow,
//! Evangelakos, Mara, `McGuire` and Meyer ("A Survey of Efficient Representations
//! for Independent Unit Vectors", 2014) does exactly that: it folds the sphere
//! onto the surface of the unit `L1` octahedron `|x| + |y| + |z| = 1` and then
//! unfolds that surface onto the square using only additions, multiplications,
//! absolute values and sign copies.
//!
//! # Solid-angle Jacobian
//!
//! Importance sampling the environment needs the change of measure between the
//! square's area element and the sphere's solid-angle element. A square point
//! maps to the octahedron point `m` (with `|m|_1 = 1`) and then radially to the
//! sphere, so the solid angle subtended by an area element is
//! `d_omega = |m . (m_u x m_v)| / |m|^3 dA`. Working the cross product out on
//! both octahedron facets gives a numerator of exactly `1`, so the Jacobian is
//! `d_omega / dA = 1 / |m|^3`. For a unit direction `d` the matching octahedron
//! point is `m = d / |d|_1`, hence `|m| = 1 / |d|_1` and the Jacobian equals
//! `|d|_1^3`. That closed form (only an `L1` norm and one cube) is all the
//! environment sampler needs to turn a square-space density into a solid-angle
//! density and back; see [`solid_angle_jacobian`].

use super::Vec3;

/// Returns `+1.0` for a non-negative component and `-1.0` otherwise.
///
/// Uses `copysign` so a negative zero is treated as negative, matching the
/// branch taken by the forward and inverse maps and keeping them exact inverses
/// on the fold seams.
fn sign_unit(x: f32) -> f32 {
    1.0_f32.copysign(x)
}

/// Projects a unit direction onto the octahedral square `[-1, 1]^2`.
///
/// The upper hemisphere (`dir.z >= 0`) maps directly through the `L1`
/// projection; the lower hemisphere is reflected across the diagonals of the
/// square so the whole sphere tiles it bijectively. A zero vector maps to the
/// square origin.
#[must_use]
pub fn direction_to_square(dir: Vec3) -> (f32, f32) {
    let l1 = dir.x.abs() + dir.y.abs() + dir.z.abs();
    let inv = if l1 > 0.0 { 1.0 / l1 } else { 0.0 };
    let px = dir.x * inv;
    let py = dir.y * inv;
    if dir.z >= 0.0 {
        (px, py)
    } else {
        (
            (1.0 - py.abs()) * sign_unit(px),
            (1.0 - px.abs()) * sign_unit(py),
        )
    }
}

/// Reconstructs the unit direction for a point `(u, v)` on the octahedral square
/// `[-1, 1]^2`.
///
/// Inverts [`direction_to_square`]: the folded octahedron height is
/// `z = 1 - |u| - |v|`, negative on the lower-hemisphere wedge where the point
/// is reflected back across the diagonals. The result is normalized, so a point
/// exactly on a seam still yields a unit vector.
#[must_use]
pub fn square_to_direction(u: f32, v: f32) -> Vec3 {
    let z = 1.0 - u.abs() - v.abs();
    let (x, y) = if z >= 0.0 {
        (u, v)
    } else {
        (
            (1.0 - v.abs()) * sign_unit(u),
            (1.0 - u.abs()) * sign_unit(v),
        )
    };
    Vec3::new(x, y, z).normalize_or_zero()
}

/// The solid-angle-to-area Jacobian `d_omega / dA = |d|_1^3` of the octahedral
/// map at unit direction `dir`.
///
/// A density expressed per unit area on the octahedral square maps to a density
/// per unit solid angle by *dividing* by this value (and the inverse direction
/// multiplies by it), since `p_omega d_omega = p_area dA` with `d_omega / dA`
/// equal to this Jacobian; this is the only measure correction the environment
/// importance sampler applies.
#[must_use]
pub fn solid_angle_jacobian(dir: Vec3) -> f32 {
    let l1 = dir.x.abs() + dir.y.abs() + dir.z.abs();
    l1 * l1 * l1
}

#[cfg(test)]
mod tests {
    use super::super::sampler::Rng;
    use super::super::PI;
    use super::*;

    /// Draws a uniformly distributed unit vector by rejection sampling the cube,
    /// avoiding the trigonometry a direct spherical draw would need.
    fn random_direction(rng: &mut Rng) -> Vec3 {
        loop {
            let x = 2.0 * rng.next_f32() - 1.0;
            let y = 2.0 * rng.next_f32() - 1.0;
            let z = 2.0 * rng.next_f32() - 1.0;
            let r2 = x * x + y * y + z * z;
            if r2 > 1.0e-6 && r2 <= 1.0 {
                let inv = 1.0 / r2.sqrt();
                return Vec3::new(x * inv, y * inv, z * inv);
            }
        }
    }

    #[test]
    fn axis_directions_map_to_expected_corners() {
        // +Z sits at the square centre, the equator axes at the edge midpoints,
        // and -Z spreads to the four corners.
        let (u, v) = direction_to_square(Vec3::new(0.0, 0.0, 1.0));
        assert!(u.abs() < 1e-6 && v.abs() < 1e-6);
        let plus_x = direction_to_square(Vec3::new(1.0, 0.0, 0.0));
        assert!((plus_x.0 - 1.0).abs() < 1e-6 && plus_x.1.abs() < 1e-6);
        let minus_z = direction_to_square(Vec3::new(0.0, 0.0, -1.0));
        assert!((minus_z.0.abs() - 1.0).abs() < 1e-6 && (minus_z.1.abs() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn round_trip_preserves_direction() {
        let mut rng = Rng::with_stream(17, 1);
        for _ in 0..5_000 {
            let dir = random_direction(&mut rng);
            let (u, v) = direction_to_square(dir);
            assert!(u >= -1.0 - 1e-5 && u <= 1.0 + 1e-5);
            assert!(v >= -1.0 - 1e-5 && v <= 1.0 + 1e-5);
            let back = square_to_direction(u, v);
            let drift = back.sub(dir).length();
            assert!(drift < 1e-4, "round trip drifted by {drift}");
        }
    }

    #[test]
    fn decode_always_returns_a_unit_vector() {
        let mut rng = Rng::with_stream(29, 2);
        for _ in 0..5_000 {
            let u = 2.0 * rng.next_f32() - 1.0;
            let v = 2.0 * rng.next_f32() - 1.0;
            let dir = square_to_direction(u, v);
            assert!((dir.length() - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn jacobian_integrates_to_the_whole_sphere() {
        // Uniformly sampling the square (area 4) and averaging the Jacobian must
        // recover the sphere's total solid angle, 4 pi, confirming the closed
        // form matches the geometric map.
        let mut rng = Rng::with_stream(41, 3);
        let count = 400_000u32;
        let mut sum = 0.0f64;
        for _ in 0..count {
            let u = 2.0 * rng.next_f32() - 1.0;
            let v = 2.0 * rng.next_f32() - 1.0;
            let dir = square_to_direction(u, v);
            sum += f64::from(solid_angle_jacobian(dir));
        }
        let square_area = 4.0f64;
        let estimate = sum / f64::from(count) * square_area;
        let expected = 4.0 * f64::from(PI);
        assert!(
            (estimate / expected - 1.0).abs() < 5e-3,
            "integral {estimate} should approach 4 pi = {expected}"
        );
    }
}
