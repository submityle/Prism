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

use super::{physics_bridge, ClothParticle, Vec3};

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
    /// Fluid (air) density scaling the quadratic drag/lift term, clamped
    /// non-negative. A non-positive density selects the linear
    /// `area * relative_wind` model (the historical default); a positive
    /// density selects the UE5 `Chaos`-style quadratic model, where the
    /// per-face force additionally scales by `0.5 * air_density *
    /// relative_wind_magnitude`, so it grows with the square of the airspeed
    /// the way real aerodynamic drag does.
    pub air_density: f32,
}

impl AeroParams {
    /// Builds coefficients from a drag and lift value, leaving the fluid
    /// density at zero so the linear (historical) aerodynamic model is used.
    ///
    /// The inputs are stored verbatim; call [`AeroParams::sanitized`] for a
    /// finite, non-negative copy before simulating. Use
    /// [`AeroParams::with_air_density`] to opt into the quadratic model.
    #[must_use]
    pub fn new(drag: f32, lift: f32) -> Self {
        Self {
            drag,
            lift,
            air_density: 0.0,
        }
    }

    /// Returns a copy with the fluid density set to `air_density`, opting the
    /// coefficients into the quadratic (airspeed-squared) aerodynamic model.
    ///
    /// A non-positive `air_density` keeps the linear model; the value is
    /// stored verbatim and clamped non-negative by [`AeroParams::sanitized`].
    #[must_use]
    pub fn with_air_density(mut self, air_density: f32) -> Self {
        self.air_density = air_density;
        self
    }

    /// Returns a copy with `drag`, `lift` and `air_density` clamped
    /// non-negative and any `NaN` replaced by `0`.
    #[must_use]
    pub fn sanitized(self) -> Self {
        Self {
            drag: sanitize_non_negative(self.drag),
            lift: sanitize_non_negative(self.lift),
            air_density: sanitize_non_negative(self.air_density),
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
    // Delegate to the single-source physics-engine aerodynamics so the render
    // and solver paths share one force model. The hand-rolled render `Vec3`
    // columns are adapted to `glam` and the render coefficients are forwarded
    // verbatim; `prism_physics_core` sanitizes them internally with the same
    // clamp and runs the identical drag/lift/area math, so the result is
    // bit-for-bit identical to the former inline implementation.
    let force = prism_physics_core::soft::aero::triangle_aero_force(
        physics_bridge::to_glam(p0),
        physics_bridge::to_glam(p1),
        physics_bridge::to_glam(p2),
        physics_bridge::to_glam(v0),
        physics_bridge::to_glam(v1),
        physics_bridge::to_glam(v2),
        physics_bridge::to_glam(wind),
        prism_physics_core::soft::aero::AeroParams::new(aero.drag, aero.lift)
            .with_air_density(aero.air_density),
    );
    physics_bridge::from_glam(force)
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
    // Delegate to the single-source physics-engine aero pre-pass. The render
    // particle slice is projected into the `(positions, velocities,
    // inverse_masses)` columns the solver consumes (pinned particles map to a
    // zero inverse mass), the wind and coefficients are forwarded verbatim, and
    // the solved velocities are written back. `prism_physics_core` performs the
    // same up-front sanitize, deterministic per-triangle turbulence, and
    // per-vertex velocity increment, so the pass is bit-for-bit identical to
    // the former inline loop (including the empty/degenerate no-ops).
    let (positions, mut velocities, inverse_masses) = physics_bridge::to_soa_full(particles);
    prism_physics_core::soft::aero::apply_aero_forces(
        &positions,
        &mut velocities,
        &inverse_masses,
        triangles,
        &prism_physics_core::soft::aero::WindField::new(
            physics_bridge::to_glam(wind.velocity),
            wind.turbulence,
        ),
        prism_physics_core::soft::aero::AeroParams::new(aero.drag, aero.lift)
            .with_air_density(aero.air_density),
        dt,
    );
    physics_bridge::write_velocities_back(particles, &velocities);
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

/// Builds a deterministic turbulence offset for one triangle from its vertex
/// indices, scaled by `turbulence`.
///
/// A zero (or negative) turbulence yields [`Vec3::ZERO`]; otherwise the three
/// indices are combined with the classic spatial-hash primes and hashed once
/// per axis, so each triangle gets a stable, index-derived jitter direction.
pub(super) fn turbulence_offset(indices: [u32; 3], turbulence: f32) -> Vec3 {
    // Delegate to the single-source physics-engine turbulence hash so the
    // deterministic per-triangle jitter has exactly one definition. The physics
    // function performs the same integer spatial hash and `[-1, 1]` avalanche,
    // so the offset is bit-for-bit identical to the former inline hash.
    physics_bridge::from_glam(prism_physics_core::soft::aero::turbulence_offset(
        indices, turbulence,
    ))
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
    fn turbulence_offset_is_zero_when_disabled() {
        assert_eq!(turbulence_offset([0, 1, 2], 0.0), Vec3::ZERO);
        assert_eq!(turbulence_offset([3, 4, 5], -1.0), Vec3::ZERO);
        assert!(turbulence_offset([0, 1, 2], 1.0).length_squared() > 0.0);
    }

    #[test]
    fn quadratic_model_scales_by_dynamic_pressure() {
        const EPS: f32 = 1.0e-5;
        let (p0, p1, p2) = xy_triangle();
        let wind = Vec3::new(3.0, 0.0, 2.0);
        let drag = 1.25;
        let lift = 0.5;
        let density = 1.225;
        let linear = triangle_wind_force(
            p0,
            p1,
            p2,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::ZERO,
            wind,
            AeroParams::new(drag, lift),
        );
        let quadratic = triangle_wind_force(
            p0,
            p1,
            p2,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::ZERO,
            wind,
            AeroParams::new(drag, lift).with_air_density(density),
        );
        // Faces at rest make the relative wind equal `wind`; the quadratic
        // model multiplies the linear force by `0.5 * air_density * |wind|`.
        let speed = wind.length_squared().sqrt();
        let factor = 0.5 * density * speed;
        assert!((quadratic.x - linear.x * factor).abs() < EPS);
        assert!((quadratic.y - linear.y * factor).abs() < EPS);
        assert!((quadratic.z - linear.z * factor).abs() < EPS);
    }

    #[test]
    fn non_positive_density_keeps_linear_model() {
        let (p0, p1, p2) = xy_triangle();
        let wind = Vec3::new(3.0, 0.0, 2.0);
        let linear = triangle_wind_force(
            p0,
            p1,
            p2,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::ZERO,
            wind,
            AeroParams::new(1.0, 0.5),
        );
        let zero_density = triangle_wind_force(
            p0,
            p1,
            p2,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::ZERO,
            wind,
            AeroParams::new(1.0, 0.5).with_air_density(0.0),
        );
        let negative_density = triangle_wind_force(
            p0,
            p1,
            p2,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::ZERO,
            wind,
            AeroParams::new(1.0, 0.5).with_air_density(-4.0),
        );
        assert_eq!(zero_density, linear);
        assert_eq!(negative_density, linear);
    }

    #[test]
    fn aeroparams_sanitize_clamps_air_density() {
        let cleaned_negative = AeroParams::new(1.0, 1.0).with_air_density(-2.0).sanitized();
        assert!(cleaned_negative.air_density.abs() < 1.0e-6);
        let cleaned_nan = AeroParams::new(1.0, 1.0)
            .with_air_density(f32::NAN)
            .sanitized();
        assert!(cleaned_nan.air_density.abs() < 1.0e-6);
        let kept = AeroParams::new(1.0, 1.0)
            .with_air_density(1.225)
            .sanitized();
        assert!((kept.air_density - 1.225).abs() < 1.0e-6);
    }
}
