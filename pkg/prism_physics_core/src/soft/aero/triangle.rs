//! Per-triangle aerodynamic force and deterministic turbulence jitter.
//!
//! Wind acts on a cloth mesh *per triangle*, not per particle, because lift and
//! drag depend on how each face is oriented relative to the airflow: a face
//! broadside to the wind catches far more force than one edge-on to it, and
//! only a triangle carries the surface normal that distinction needs. Each
//! triangle decomposes the relative wind (ambient wind minus the face's own
//! velocity) into a component along the face normal (drag) and the remaining
//! in-plane component (lift); the sum is scaled by the triangle area (and, in
//! the quadratic model, by the dynamic pressure).
//!
//! The optional turbulence is a hand-rolled integer avalanche hash of the
//! vertex indices, never a transcendental, so the same mesh in the same wind
//! always deforms identically and can be golden-tested on any platform.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. The
//! drag/lift decomposition and the optional quadratic dynamic-pressure term are
//! the standard, publicly documented cloth aerodynamics model.

use glam::Vec3;

use crate::math::scalar::Real;

use super::field::AeroParams;

/// Squared length below which a triangle's edge-cross is treated as degenerate.
const EPS_LEN_SQ: Real = 1e-12;

/// Computes the aerodynamic force on one triangle from the relative wind.
///
/// `p0`/`p1`/`p2` are the vertex positions and `v0`/`v1`/`v2` their velocities.
/// The face normal comes from the cross product of two edges; its length is
/// twice the triangle area, so the area is `0.5 * |cross|`. The relative wind
/// `wind - face_velocity` (where the face velocity is the average of the three
/// vertex velocities) is split into a component along the unit normal (scaled
/// by `drag`) and the remaining in-plane component (scaled by `lift`), and the
/// sum is multiplied by the pressure scale. A degenerate triangle (near-zero
/// normal) contributes no force and returns [`Vec3::ZERO`]. `aero` is sanitized
/// internally, so the result is finite for finite inputs. Only `sqrt` is used;
/// no transcendental functions are called.
#[must_use]
#[expect(
    clippy::too_many_arguments,
    reason = "a triangle force is intrinsically defined by its three positions, \
              three velocities, the wind, and the coefficients; grouping them \
              into structs here would obscure the plain math"
)]
pub fn triangle_aero_force(
    p0: Vec3,
    p1: Vec3,
    p2: Vec3,
    v0: Vec3,
    v1: Vec3,
    v2: Vec3,
    wind: Vec3,
    aero: AeroParams,
) -> Vec3 {
    let aero = aero.sanitized();
    let cross = (p1 - p0).cross(p2 - p0);
    let cross_len_sq = cross.length_squared();
    if cross_len_sq <= EPS_LEN_SQ {
        return Vec3::ZERO;
    }
    let area = 0.5 * cross_len_sq.sqrt();
    let normal = cross.normalize_or_zero();

    let face_velocity = (v0 + v1 + v2) * (1.0 / 3.0);
    let relative = wind - face_velocity;

    let normal_component = normal * relative.dot(normal);
    let tangent_component = relative - normal_component;

    // Directional force per unit pressure: drag along the normal, lift in-plane.
    let directional = normal_component * aero.drag + tangent_component * aero.lift;

    // Pressure scale. The linear model (density <= 0) uses the triangle area
    // directly, matching the historical `area * relative_wind` force. The
    // quadratic model (density > 0) additionally scales by the dynamic pressure
    // factor `0.5 * air_density * relative_wind_magnitude`, so the force grows
    // with the square of the airspeed. Only `sqrt` is used.
    let pressure = if aero.air_density > 0.0 {
        area * (0.5 * aero.air_density * relative.length_squared().sqrt())
    } else {
        area
    };

    directional * pressure
}

/// Scrambles an integer seed into a reproducible value in `[-1, 1]`.
///
/// This is an integer avalanche (an `xorshift`-multiply mix) followed by a map
/// from the high bits to a fraction, using only integer arithmetic so the
/// result is bit-identical on every platform. No transcendental function is
/// called.
fn hash_to_unit(seed: u32) -> Real {
    let mut h = seed.wrapping_mul(0x9E37_79B1);
    h ^= h >> 15;
    h = h.wrapping_mul(0x85EB_CA77);
    h ^= h >> 13;
    h = h.wrapping_mul(0xC2B2_AE3D);
    h ^= h >> 16;
    let unit = (h >> 8) as Real * (1.0 / 16_777_216.0);
    unit * 2.0 - 1.0
}

/// Builds a deterministic turbulence offset for one triangle from its vertex
/// indices, scaled by `turbulence`.
///
/// A zero (or negative) turbulence yields [`Vec3::ZERO`]; otherwise the three
/// indices are combined with the classic spatial-hash primes and hashed once
/// per axis, so each triangle gets a stable, index-derived jitter direction.
#[must_use]
pub fn turbulence_offset(indices: [u32; 3], turbulence: Real) -> Vec3 {
    if turbulence <= 0.0 {
        return Vec3::ZERO;
    }
    let base = indices[0].wrapping_mul(73_856_093)
        ^ indices[1].wrapping_mul(19_349_663)
        ^ indices[2].wrapping_mul(83_492_791);
    Vec3::new(
        hash_to_unit(base ^ 0x00A5_5A00),
        hash_to_unit(base ^ 0x5A00_00A5),
        hash_to_unit(base ^ 0x00FF_00FF),
    ) * turbulence
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A unit right triangle in the XY plane whose normal points along `+Z`.
    fn xy_triangle() -> (Vec3, Vec3, Vec3) {
        (
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        )
    }

    #[test]
    fn degenerate_triangle_has_no_force() {
        let p = Vec3::new(1.0, 1.0, 1.0);
        let force = triangle_aero_force(
            p,
            p,
            p,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::new(3.0, 0.0, 0.0),
            AeroParams::new(1.0, 1.0),
        );
        assert_eq!(force, Vec3::ZERO);
    }

    #[test]
    fn wind_along_normal_is_pure_drag() {
        let (p0, p1, p2) = xy_triangle();
        // Wind straight into the face (+Z); drag only. Area is 0.5.
        let force = triangle_aero_force(
            p0,
            p1,
            p2,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::new(0.0, 0.0, 2.0),
            AeroParams::new(1.0, 0.0),
        );
        assert!(force.x.abs() < 1.0e-6);
        assert!(force.y.abs() < 1.0e-6);
        assert!((force.z - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn wind_in_plane_is_pure_lift() {
        let (p0, p1, p2) = xy_triangle();
        // Wind in the face plane (+X); lift only. Area 0.5, lift 2 => 3.0.
        let force = triangle_aero_force(
            p0,
            p1,
            p2,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::new(3.0, 0.0, 0.0),
            AeroParams::new(0.0, 2.0),
        );
        assert!((force.x - 3.0).abs() < 1.0e-6);
        assert!(force.y.abs() < 1.0e-6);
        assert!(force.z.abs() < 1.0e-6);
    }

    #[test]
    fn quadratic_model_scales_with_airspeed_squared() {
        let (p0, p1, p2) = xy_triangle();
        // Linear drag force along the normal at airspeed 2 => area*2 = 1.0.
        let linear = triangle_aero_force(
            p0,
            p1,
            p2,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::new(0.0, 0.0, 2.0),
            AeroParams::new(1.0, 0.0),
        );
        // Quadratic adds a 0.5 * density * |wind| factor. With density 1 and
        // |wind| 2 that factor is 1.0, so the force equals the linear one.
        let quad = triangle_aero_force(
            p0,
            p1,
            p2,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::new(0.0, 0.0, 2.0),
            AeroParams::new(1.0, 0.0).with_air_density(1.0),
        );
        assert!((quad.z - linear.z).abs() < 1.0e-6, "quad {quad:?} linear {linear:?}");
        // Doubling the airspeed quadruples the quadratic force.
        let quad_fast = triangle_aero_force(
            p0,
            p1,
            p2,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::new(0.0, 0.0, 4.0),
            AeroParams::new(1.0, 0.0).with_air_density(1.0),
        );
        assert!((quad_fast.z - 4.0 * quad.z).abs() < 1.0e-5);
    }

    #[test]
    fn turbulence_zero_is_no_offset() {
        assert_eq!(turbulence_offset([0, 1, 2], 0.0), Vec3::ZERO);
        assert_eq!(turbulence_offset([0, 1, 2], -1.0), Vec3::ZERO);
    }

    #[test]
    fn turbulence_is_deterministic_and_bounded() {
        let a = turbulence_offset([3, 7, 11], 1.0);
        let b = turbulence_offset([3, 7, 11], 1.0);
        assert_eq!(a, b);
        assert!(a.x.abs() <= 1.0 && a.y.abs() <= 1.0 && a.z.abs() <= 1.0);
        // A different triangle hashes to a different jitter.
        let c = turbulence_offset([4, 7, 11], 1.0);
        assert_ne!(a, c);
    }
}
