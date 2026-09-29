//! Wind-field coupling and per-triangle aerodynamics for cloth.
//!
//! A garment only reads as *cloth* when the surrounding air pushes on it: a
//! steady breeze billows a skirt, a gust snaps a flag taut, and the drag of the
//! air is what lets a cape settle instead of swinging like a rigid sheet. This
//! module models that coupling the way production cloth engines do (UE5
//! `Chaos` Cloth aerodynamics and NVIDIA `NvCloth`): the wind acts on the mesh
//! *per triangle*, not per particle, because lift and drag depend on how each
//! face is oriented relative to the airflow. A face broadside to the wind
//! catches far more force than one edge-on to it, and only a triangle carries
//! the surface normal needed to express that.
//!
//! Each triangle decomposes the relative wind (ambient wind minus the face's
//! own velocity) into a component along the face normal (drag, the push that
//! resists the flow) and a component in the face plane (lift, the sideways
//! push that makes fabric flutter). The resulting force is scaled by the
//! triangle area and split evenly across its three vertices as a velocity
//! increment, mirroring the external pre-pass style of the sibling hair wind
//! module. Everything is deterministic and finite: pinned vertices never move,
//! out-of-range triangles are skipped rather than panicking, and the optional
//! turbulence is a hand-rolled integer hash of the vertex indices (never a
//! transcendental) so the same mesh in the same wind always deforms
//! identically and can be golden-tested.

use super::{ClothParticle, Vec3, EPS_LEN_SQ};

/// Ambient wind sampled as a single world-space velocity plus a turbulence
/// strength.
///
/// `velocity` is the steady airflow every face feels; `turbulence` in `0..=1`
/// scales a small, deterministic per-triangle jitter added to that airflow so
/// the garment does not move as one rigid clump. A default (zero velocity, zero
/// turbulence) field exerts no force, so callers may always run the pass
/// unconditionally.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct WindField {
    /// Steady wind velocity in world units per second.
    pub velocity: Vec3,
    /// Turbulence strength, clamped to `0..=1`; scales the per-triangle jitter.
    pub turbulence: f32,
}

impl WindField {
    /// Builds a wind field from a steady velocity and turbulence strength.
    ///
    /// The inputs are stored verbatim; call [`WindField::sanitized`] to obtain a
    /// finite, range-clamped copy before simulating.
    #[must_use]
    pub fn new(velocity: Vec3, turbulence: f32) -> Self {
        Self {
            velocity,
            turbulence,
        }
    }

    /// Returns a copy with any non-finite velocity component replaced by zero
    /// and `turbulence` clamped to `0..=1` (a `NaN` becomes `0`).
    ///
    /// This guarantees the field can never inject `NaN` or a negative
    /// turbulence into the simulation.
    #[must_use]
    pub fn sanitized(self) -> Self {
        Self {
            velocity: sanitize_vec(self.velocity),
            turbulence: sanitize_unit(self.turbulence),
        }
    }
}

/// Per-triangle aerodynamic coefficients.
///
/// `drag` scales the force along the face normal (resisting the airflow) and
/// `lift` scales the in-plane force (the sideways push that makes fabric
/// flutter). Both are dimensionless multipliers on `area * relative_wind`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AeroParams {
    /// Normal-direction (drag) coefficient; clamped non-negative.
    pub drag: f32,
    /// In-plane (lift) coefficient; clamped non-negative.
    pub lift: f32,
}

impl AeroParams {
    /// Builds coefficients from a drag and lift value.
    ///
    /// The inputs are stored verbatim; call [`AeroParams::sanitized`] for a
    /// finite, non-negative copy before simulating.
    #[must_use]
    pub fn new(drag: f32, lift: f32) -> Self {
        Self { drag, lift }
    }

    /// Returns a copy with `drag` and `lift` clamped non-negative and any
    /// `NaN` replaced by `0`.
    #[must_use]
    pub fn sanitized(self) -> Self {
        Self {
            drag: sanitize_non_negative(self.drag),
            lift: sanitize_non_negative(self.lift),
        }
    }
}

/// Computes the aerodynamic force on one triangle from the relative wind.
///
/// `p0`/`p1`/`p2` are the vertex positions and `v0`/`v1`/`v2` their velocities.
/// The face normal comes from the cross product of two edges; its length is
/// twice the triangle area, so the area is `0.5 * |cross|`. The relative wind
/// `wind - face_velocity` (where the face velocity is the average of the three
/// vertex velocities) is split into a component along the unit normal (scaled
/// by `drag`) and the remaining in-plane component (scaled by `lift`), and the
/// sum is multiplied by the area. A degenerate triangle (near-zero normal)
/// contributes no force and returns [`Vec3::ZERO`]. `aero` is sanitized
/// internally, so the result is finite for finite inputs. Only `sqrt` is used;
/// no transcendental functions are called.
#[must_use]
pub fn triangle_wind_force(
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
    let cross = p1.sub(p0).cross(p2.sub(p0));
    let cross_len_sq = cross.length_squared();
    if cross_len_sq <= EPS_LEN_SQ {
        return Vec3::ZERO;
    }
    let area = 0.5 * cross_len_sq.sqrt();
    let normal = cross.normalize_or_zero();

    let face_velocity = v0.add(v1).add(v2).scale(1.0 / 3.0);
    let relative = wind.sub(face_velocity);

    let normal_component = normal.scale(relative.dot(normal));
    let tangent_component = relative.sub(normal_component);

    normal_component
        .scale(aero.drag)
        .add(tangent_component.scale(aero.lift))
        .scale(area)
}

/// Applies wind-driven aerodynamic forces to `particles` over `dt` seconds.
///
/// For each triangle in `triangles` the per-face force from
/// [`triangle_wind_force`] is spread evenly across its three vertices: each
/// free vertex gains `(force / 3) * inverse_mass * dt` as a velocity increment.
/// Pinned vertices are skipped, and triangles that index outside `particles`
/// are ignored rather than panicking. A non-positive or non-finite `dt`, an
/// empty particle set, and an empty triangle set are all no-ops. `wind` and
/// `aero` are sanitized once up front, and the optional turbulence adds a
/// deterministic per-triangle jitter derived from the vertex indices, so the
/// whole pass is reproducible.
pub fn apply_aero_forces(
    particles: &mut [ClothParticle],
    triangles: &[[u32; 3]],
    wind: &WindField,
    aero: AeroParams,
    dt: f32,
) {
    if dt <= 0.0 || !dt.is_finite() || particles.is_empty() || triangles.is_empty() {
        return;
    }
    let field = wind.sanitized();
    let aero = aero.sanitized();
    let count = particles.len();

    for indices in triangles {
        let i0 = indices[0] as usize;
        let i1 = indices[1] as usize;
        let i2 = indices[2] as usize;
        if i0 >= count || i1 >= count || i2 >= count {
            continue;
        }

        let p0 = particles[i0].position;
        let p1 = particles[i1].position;
        let p2 = particles[i2].position;
        let v0 = particles[i0].velocity;
        let v1 = particles[i1].velocity;
        let v2 = particles[i2].velocity;

        let wind_vec = field
            .velocity
            .add(turbulence_offset(*indices, field.turbulence));
        let force = triangle_wind_force(p0, p1, p2, v0, v1, v2, wind_vec, aero);
        let per_vertex = force.scale(1.0 / 3.0);

        let targets = [i0, i1, i2];
        for &index in &targets {
            let particle = &mut particles[index];
            if particle.is_pinned() {
                continue;
            }
            let delta = per_vertex.scale(particle.inverse_mass * dt);
            particle.velocity = particle.velocity.add(delta);
        }
    }
}

/// Replaces a non-finite scalar with `0`, leaving finite values unchanged.
fn sanitize_finite(x: f32) -> f32 {
    if x.is_finite() {
        x
    } else {
        0.0
    }
}

/// Replaces every non-finite component of a vector with `0`.
fn sanitize_vec(v: Vec3) -> Vec3 {
    Vec3::new(
        sanitize_finite(v.x),
        sanitize_finite(v.y),
        sanitize_finite(v.z),
    )
}

/// Clamps a scalar to `0..=1`, mapping any non-finite input to `0`.
fn sanitize_unit(x: f32) -> f32 {
    if x.is_finite() {
        x.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Clamps a scalar to be non-negative, mapping any non-finite input to `0`.
fn sanitize_non_negative(x: f32) -> f32 {
    if x.is_finite() {
        x.max(0.0)
    } else {
        0.0
    }
}

/// Scrambles an integer seed into a reproducible value in `[-1, 1]`.
///
/// This is an integer avalanche (an `xorshift`-multiply mix) followed by a map
/// from the high bits to a fraction, using only integer arithmetic and one
/// multiply so the result is bit-identical on every platform. No transcendental
/// function is called.
fn hash_to_unit(seed: u32) -> f32 {
    let mut h = seed.wrapping_mul(0x9E37_79B1);
    h ^= h >> 15;
    h = h.wrapping_mul(0x85EB_CA77);
    h ^= h >> 13;
    h = h.wrapping_mul(0xC2B2_AE3D);
    h ^= h >> 16;
    let unit = (h >> 8) as f32 * (1.0 / 16_777_216.0);
    unit * 2.0 - 1.0
}

/// Builds a deterministic turbulence offset for one triangle from its vertex
/// indices, scaled by `turbulence`.
///
/// A zero (or negative) turbulence yields [`Vec3::ZERO`]; otherwise the three
/// indices are combined with the classic spatial-hash primes and hashed once
/// per axis, so each triangle gets a stable, index-derived jitter direction.
fn turbulence_offset(indices: [u32; 3], turbulence: f32) -> Vec3 {
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
    )
    .scale(turbulence)
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
    fn windfield_default_is_calm() {
        let field = WindField::default();
        assert_eq!(field.velocity, Vec3::ZERO);
        assert!(field.turbulence.abs() < 1.0e-6);
    }

    #[test]
    fn windfield_sanitize_clamps_turbulence_and_nan() {
        let dirty = WindField {
            velocity: Vec3::new(f32::NAN, 1.0, f32::INFINITY),
            turbulence: f32::NAN,
        };
        let clean = dirty.sanitized();
        assert_eq!(clean.velocity, Vec3::new(0.0, 1.0, 0.0));
        assert!(clean.turbulence.abs() < 1.0e-6);

        let over = WindField::new(Vec3::ZERO, 5.0).sanitized();
        assert!((over.turbulence - 1.0).abs() < 1.0e-6);
        let under = WindField::new(Vec3::ZERO, -2.0).sanitized();
        assert!(under.turbulence.abs() < 1.0e-6);
    }

    #[test]
    fn aeroparams_sanitize_clamps_negative_and_nan() {
        let clean = AeroParams::new(-3.0, f32::NAN).sanitized();
        assert!(clean.drag.abs() < 1.0e-6);
        assert!(clean.lift.abs() < 1.0e-6);
        let kept = AeroParams::new(0.5, 2.0).sanitized();
        assert!((kept.drag - 0.5).abs() < 1.0e-6);
        assert!((kept.lift - 2.0).abs() < 1.0e-6);
    }

    #[test]
    fn degenerate_triangle_has_no_force() {
        let p = Vec3::new(1.0, 1.0, 1.0);
        let force = triangle_wind_force(
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
        let force = triangle_wind_force(
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
        // Wind in the face plane (+X); lift only. Area is 0.5, lift is 2.
        let force = triangle_wind_force(
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
    fn face_velocity_reduces_relative_wind() {
        let (p0, p1, p2) = xy_triangle();
        // Face moving with the wind (+Z) feels no relative flow, hence no force.
        let along = Vec3::new(0.0, 0.0, 2.0);
        let force = triangle_wind_force(
            p0,
            p1,
            p2,
            along,
            along,
            along,
            along,
            AeroParams::new(1.0, 1.0),
        );
        assert!(force.length() < 1.0e-6);
    }

    #[test]
    fn apply_spreads_force_over_free_vertices() {
        let (p0, p1, p2) = xy_triangle();
        let mut particles = [
            ClothParticle::new(p0, 1.0),
            ClothParticle::new(p1, 1.0),
            ClothParticle::new(p2, 1.0),
        ];
        let triangles = [[0u32, 1, 2]];
        let wind = WindField::new(Vec3::new(0.0, 0.0, 2.0), 0.0);
        apply_aero_forces(
            &mut particles,
            &triangles,
            &wind,
            AeroParams::new(1.0, 0.0),
            1.0,
        );
        // Force (0,0,1) split three ways: each vertex gains z = 1/3.
        for particle in &particles {
            assert!(particle.velocity.x.abs() < 1.0e-6);
            assert!(particle.velocity.y.abs() < 1.0e-6);
            assert!((particle.velocity.z - 1.0 / 3.0).abs() < 1.0e-6);
        }
    }

    #[test]
    fn apply_skips_pinned_vertices() {
        let (p0, p1, p2) = xy_triangle();
        let mut particles = [
            ClothParticle::pinned(p0),
            ClothParticle::new(p1, 1.0),
            ClothParticle::new(p2, 1.0),
        ];
        let triangles = [[0u32, 1, 2]];
        let wind = WindField::new(Vec3::new(0.0, 0.0, 2.0), 0.0);
        apply_aero_forces(
            &mut particles,
            &triangles,
            &wind,
            AeroParams::new(1.0, 0.0),
            1.0,
        );
        assert_eq!(particles[0].velocity, Vec3::ZERO);
        assert!((particles[1].velocity.z - 1.0 / 3.0).abs() < 1.0e-6);
        assert!((particles[2].velocity.z - 1.0 / 3.0).abs() < 1.0e-6);
    }

    #[test]
    fn apply_ignores_out_of_range_triangles() {
        let (p0, p1, _) = xy_triangle();
        let mut particles = [ClothParticle::new(p0, 1.0), ClothParticle::new(p1, 1.0)];
        let triangles = [[0u32, 1, 5]];
        let wind = WindField::new(Vec3::new(0.0, 0.0, 2.0), 0.0);
        apply_aero_forces(
            &mut particles,
            &triangles,
            &wind,
            AeroParams::new(1.0, 1.0),
            1.0,
        );
        for particle in &particles {
            assert_eq!(particle.velocity, Vec3::ZERO);
        }
    }

    #[test]
    fn apply_is_noop_for_bad_dt_and_empty_inputs() {
        let (p0, p1, p2) = xy_triangle();
        let mut particles = [
            ClothParticle::new(p0, 1.0),
            ClothParticle::new(p1, 1.0),
            ClothParticle::new(p2, 1.0),
        ];
        let triangles = [[0u32, 1, 2]];
        let wind = WindField::new(Vec3::new(0.0, 0.0, 2.0), 0.0);
        let aero = AeroParams::new(1.0, 1.0);

        apply_aero_forces(&mut particles, &triangles, &wind, aero, 0.0);
        apply_aero_forces(&mut particles, &triangles, &wind, aero, -1.0);
        apply_aero_forces(&mut particles, &triangles, &wind, aero, f32::NAN);
        apply_aero_forces(&mut particles, &[], &wind, aero, 1.0);
        for particle in &particles {
            assert_eq!(particle.velocity, Vec3::ZERO);
        }

        let mut empty: [ClothParticle; 0] = [];
        apply_aero_forces(&mut empty, &triangles, &wind, aero, 1.0);
        assert!(empty.is_empty());
    }

    #[test]
    fn apply_is_deterministic_with_turbulence() {
        let (p0, p1, p2) = xy_triangle();
        let make = || {
            [
                ClothParticle::new(p0, 1.0),
                ClothParticle::new(p1, 1.0),
                ClothParticle::new(p2, 1.0),
            ]
        };
        let triangles = [[0u32, 1, 2]];
        let wind = WindField::new(Vec3::new(0.0, 0.0, 2.0), 0.7);
        let aero = AeroParams::new(1.0, 0.5);

        let mut a = make();
        let mut b = make();
        apply_aero_forces(&mut a, &triangles, &wind, aero, 0.5);
        apply_aero_forces(&mut b, &triangles, &wind, aero, 0.5);
        for (pa, pb) in a.iter().zip(b.iter()) {
            assert_eq!(pa.velocity, pb.velocity);
        }
    }

    #[test]
    fn turbulence_changes_the_result() {
        let (p0, p1, p2) = xy_triangle();
        let triangles = [[0u32, 1, 2]];
        let aero = AeroParams::new(1.0, 1.0);

        let mut calm = [
            ClothParticle::new(p0, 1.0),
            ClothParticle::new(p1, 1.0),
            ClothParticle::new(p2, 1.0),
        ];
        let mut gusty = [
            ClothParticle::new(p0, 1.0),
            ClothParticle::new(p1, 1.0),
            ClothParticle::new(p2, 1.0),
        ];
        apply_aero_forces(
            &mut calm,
            &triangles,
            &WindField::new(Vec3::new(0.0, 0.0, 2.0), 0.0),
            aero,
            0.5,
        );
        apply_aero_forces(
            &mut gusty,
            &triangles,
            &WindField::new(Vec3::new(0.0, 0.0, 2.0), 1.0),
            aero,
            0.5,
        );
        let mut differs = false;
        for (c, g) in calm.iter().zip(gusty.iter()) {
            if (c.velocity.sub(g.velocity)).length() > 1.0e-6 {
                differs = true;
            }
        }
        assert!(differs);
    }

    #[test]
    fn hash_stays_in_unit_range() {
        for seed in 0..256u32 {
            let value = hash_to_unit(seed.wrapping_mul(2_654_435_761));
            assert!((-1.0..=1.0).contains(&value));
        }
    }

    #[test]
    fn turbulence_offset_is_zero_when_disabled() {
        assert_eq!(turbulence_offset([0, 1, 2], 0.0), Vec3::ZERO);
        assert_eq!(turbulence_offset([3, 4, 5], -1.0), Vec3::ZERO);
        assert!(turbulence_offset([0, 1, 2], 1.0).length_squared() > 0.0);
    }
}
