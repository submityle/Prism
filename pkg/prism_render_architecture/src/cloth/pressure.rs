//! Pressure constraint: closed-mesh volume preservation for inflatable cloth.
//!
//! This is design §6, the pressure (gas / volume) constraint that keeps a
//! *closed* triangle mesh at a target enclosed volume by pushing its particles
//! along the outward triangle normals. It is the XPBD volume analogue of the
//! per-edge distance constraints in [`super::dynamics`], and it is what turns a
//! garment shell into a balloon, an inflated down jacket, an air bladder, or a
//! billowing membrane — the same pressure model shipped by Houdini `Vellum` and
//! NVIDIA `NvCloth`, and conceptually by the UE5 `Chaos` cloth aerodynamics
//! slot, expressed here from first principles without reusing their code.
//!
//! # Math
//!
//! The signed enclosed volume of a closed, outward-wound triangle mesh follows
//! from the divergence theorem as a sum of tetrahedra spanned from the origin:
//!
//! ```text
//! V = (1/6) * Σ_tri  p0 · (p1 × p2)
//! ```
//!
//! The pressure constraint drives that volume toward a target `k * V_rest`,
//! where `k` (the overpressure) inflates (`k > 1`) or deflates (`k < 1`) the
//! shell:
//!
//! ```text
//! C = V - k * V_rest
//! ```
//!
//! Each triangle contributes a gradient to each of its three vertices, the
//! partial derivative of that triangle's tetra volume:
//!
//! ```text
//! ∂V/∂p0 = (1/6)(p1 × p2)   ∂V/∂p1 = (1/6)(p2 × p0)   ∂V/∂p2 = (1/6)(p0 × p1)
//! ```
//!
//! Vertices shared by many triangles accumulate the sum of those contributions.
//! The single XPBD projection then mirrors the distance solver: with
//! `α̃ = compliance / dt²`,
//!
//! ```text
//! Δλ  = -C / ( Σ_i w_i |∇_i|² + α̃ )
//! x_i += w_i · Δλ · ∇_i
//! ```
//!
//! Pinned particles (`inverse_mass <= 0`, weight `w = 0`) never move, out-of-range
//! triangle indices are skipped, a (near) zero denominator is a no-op, and the
//! whole routine is deterministic array-in / array-out math using only `sqrt`
//! (indirectly, via [`super::Vec3`]); no transcendental functions are called and
//! no `NaN` is produced from well-formed input.

use alloc::vec;
use alloc::vec::Vec;

use super::{ClothParticle, Compliance, Vec3, EPS_LEN_SQ};

/// One sixth, the constant factor in the tetra-volume and gradient formulas.
const INV_SIX: f32 = 1.0 / 6.0;

/// Computes the signed enclosed volume of a closed triangle mesh.
///
/// `positions` is the vertex array and `triangles` indexes it with outward
/// winding (counter-clockwise seen from outside), so a well-formed shell yields
/// a positive volume. Triangles whose indices fall outside `positions` are
/// skipped rather than panicking, which keeps the function total for partial or
/// malformed meshes. The result is `0.0` for an empty mesh.
#[must_use]
pub fn mesh_volume(positions: &[Vec3], triangles: &[[u32; 3]]) -> f32 {
    let mut sum = 0.0;
    for tri in triangles {
        let Some((p0, p1, p2)) = fetch_triangle_positions(positions, *tri) else {
            continue;
        };
        sum += p0.dot(p1.cross(p2));
    }
    sum * INV_SIX
}

/// Tuning for one pressure (volume) projection.
///
/// The target enclosed volume is `overpressure * rest_volume`; an overpressure
/// above one inflates the shell (balloon / down jacket), below one deflates it.
/// `compliance` softens the constraint exactly like the distance solver: zero is
/// perfectly rigid, larger values yield a springier, slower volume response.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PressureParams {
    /// Reference (rest) enclosed volume the shell relaxes toward at unit
    /// overpressure.
    pub rest_volume: f32,
    /// Multiplier applied to `rest_volume` to obtain the target volume; `> 1`
    /// inflates, `< 1` deflates.
    pub overpressure: f32,
    /// XPBD compliance `α` (inverse stiffness) of the volume constraint.
    pub compliance: Compliance,
}

impl Default for PressureParams {
    /// Neutral parameters: zero rest volume, unit overpressure, rigid.
    fn default() -> Self {
        Self {
            rest_volume: 0.0,
            overpressure: 1.0,
            compliance: Compliance::RIGID,
        }
    }
}

impl PressureParams {
    /// Builds pressure parameters from a rest volume, overpressure, and
    /// compliance.
    #[must_use]
    pub fn new(rest_volume: f32, overpressure: f32, compliance: Compliance) -> Self {
        Self {
            rest_volume,
            overpressure,
            compliance,
        }
    }

    /// Returns a defensively cleaned copy safe to feed the solver.
    ///
    /// The rest volume is made non-negative (its magnitude is used, so callers
    /// may pass a signed volume straight from [`mesh_volume`]); the overpressure
    /// is clamped strictly positive so the target never collapses to or below
    /// zero; `NaN` inputs fall back to the neutral defaults (zero rest volume,
    /// unit overpressure). Compliance is already clamped non-negative by
    /// [`Compliance::value`], so it is passed through unchanged.
    #[must_use]
    pub fn sanitized(self) -> Self {
        let rest_volume = if self.rest_volume.is_nan() {
            0.0
        } else {
            self.rest_volume.abs()
        };
        let overpressure = if self.overpressure.is_nan() {
            1.0
        } else {
            self.overpressure.max(f32::MIN_POSITIVE)
        };
        Self {
            rest_volume,
            overpressure,
            compliance: self.compliance,
        }
    }

    /// The target enclosed volume `overpressure * rest_volume`.
    #[must_use]
    pub fn target_volume(self) -> f32 {
        self.overpressure * self.rest_volume
    }
}

/// Applies one XPBD pressure projection to a closed mesh in place.
///
/// The routine computes the current signed volume and the per-vertex volume
/// gradient (accumulating every incident triangle's contribution), forms the
/// volume error `C = V - target`, and performs a single compliant XPBD update
/// that nudges each free particle along its accumulated gradient toward the
/// target volume. Call it once per substep from the pipeline (see
/// [`apply_pressure`]).
///
/// It is a no-op when `dt <= 0`, when there are no particles, or when there are
/// no triangles. Triangles indexing outside `particles` are skipped, pinned
/// particles never move, and a denominator at or below [`EPS_LEN_SQ`] (a
/// degenerate / collapsed mesh) is skipped so the update never divides by zero
/// or emits `NaN`. `params` is sanitized internally, so raw authored values are
/// safe to pass.
pub fn project_pressure(
    particles: &mut [ClothParticle],
    triangles: &[[u32; 3]],
    params: PressureParams,
    dt: f32,
) {
    if dt <= 0.0 || particles.is_empty() || triangles.is_empty() {
        return;
    }
    let params = params.sanitized();
    let count = particles.len();

    // Accumulate the signed volume and the per-vertex gradient in one pass.
    let mut gradients: Vec<Vec3> = vec![Vec3::ZERO; count];
    let mut volume = 0.0;
    for tri in triangles {
        let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        if i0 >= count || i1 >= count || i2 >= count {
            continue;
        }
        let p0 = particles[i0].position;
        let p1 = particles[i1].position;
        let p2 = particles[i2].position;
        volume += p0.dot(p1.cross(p2));
        gradients[i0] = gradients[i0].add(p1.cross(p2).scale(INV_SIX));
        gradients[i1] = gradients[i1].add(p2.cross(p0).scale(INV_SIX));
        gradients[i2] = gradients[i2].add(p0.cross(p1).scale(INV_SIX));
    }
    volume *= INV_SIX;

    let error = volume - params.target_volume();

    // Denominator: Σ w_i |∇_i|² + α̃.
    let mut denom = 0.0;
    for (i, grad) in gradients.iter().enumerate() {
        let w = particle_weight(particles[i]);
        if w <= 0.0 {
            continue;
        }
        denom += w * grad.length_squared();
    }
    let alpha_tilde = params.compliance.value() / (dt * dt);
    denom += alpha_tilde;
    if denom < EPS_LEN_SQ {
        return;
    }

    let d_lambda = -error / denom;
    for (i, grad) in gradients.iter().enumerate() {
        let w = particle_weight(particles[i]);
        if w <= 0.0 {
            continue;
        }
        particles[i].position = particles[i].position.add(grad.scale(w * d_lambda));
    }
}

/// Pipeline hook: applies the pressure constraint for one solver substep.
///
/// This is the entry point the dynamics pipeline calls each substep after its
/// distance-constraint projection; it simply forwards to [`project_pressure`]
/// with the substep `dt`, keeping the pressure model a self-contained,
/// order-independent Jacobi projection that composes with the other colored
/// constraint passes.
pub fn apply_pressure(
    particles: &mut [ClothParticle],
    triangles: &[[u32; 3]],
    params: PressureParams,
    dt_sub: f32,
) {
    project_pressure(particles, triangles, params, dt_sub);
}

/// Fetches the three vertex positions of a triangle, or `None` when any index
/// is out of range.
fn fetch_triangle_positions(positions: &[Vec3], tri: [u32; 3]) -> Option<(Vec3, Vec3, Vec3)> {
    let p0 = positions.get(tri[0] as usize)?;
    let p1 = positions.get(tri[1] as usize)?;
    let p2 = positions.get(tri[2] as usize)?;
    Some((*p0, *p1, *p2))
}

/// Effective inverse mass of a particle: zero when pinned, otherwise its stored
/// inverse mass.
fn particle_weight(particle: ClothParticle) -> f32 {
    if particle.is_pinned() {
        0.0
    } else {
        particle.inverse_mass
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Absolute tolerance for volume comparisons.
    const EPS: f32 = 1.0e-4;

    /// The eight corners of the axis-aligned unit cube `[0,1]³`.
    fn unit_cube_positions() -> Vec<Vec3> {
        vec![
            Vec3::new(0.0, 0.0, 0.0), // 0
            Vec3::new(1.0, 0.0, 0.0), // 1
            Vec3::new(1.0, 1.0, 0.0), // 2
            Vec3::new(0.0, 1.0, 0.0), // 3
            Vec3::new(0.0, 0.0, 1.0), // 4
            Vec3::new(1.0, 0.0, 1.0), // 5
            Vec3::new(1.0, 1.0, 1.0), // 6
            Vec3::new(0.0, 1.0, 1.0), // 7
        ]
    }

    /// The twelve outward-wound triangles of the unit cube.
    fn unit_cube_triangles() -> Vec<[u32; 3]> {
        vec![
            [0, 2, 1],
            [0, 3, 2], // -Z
            [4, 5, 6],
            [4, 6, 7], // +Z
            [0, 1, 5],
            [0, 5, 4], // -Y
            [3, 6, 2],
            [3, 7, 6], // +Y
            [0, 4, 7],
            [0, 7, 3], // -X
            [1, 2, 6],
            [1, 6, 5], // +X
        ]
    }

    /// Builds a movable-particle array from positions with unit inverse mass.
    fn free_particles(positions: &[Vec3]) -> Vec<ClothParticle> {
        positions
            .iter()
            .map(|p| ClothParticle::new(*p, 1.0))
            .collect()
    }

    /// Recomputes the mesh volume from a particle array.
    fn particle_volume(particles: &[ClothParticle], triangles: &[[u32; 3]]) -> f32 {
        let positions: Vec<Vec3> = particles.iter().map(|p| p.position).collect();
        mesh_volume(&positions, triangles)
    }

    #[test]
    fn unit_cube_volume_is_one() {
        let volume = mesh_volume(&unit_cube_positions(), &unit_cube_triangles());
        assert!((volume - 1.0).abs() < EPS, "unit cube volume: {volume}");
    }

    #[test]
    fn empty_mesh_volume_is_zero() {
        assert!(mesh_volume(&[], &[]).abs() < EPS);
        let positions = unit_cube_positions();
        assert!(mesh_volume(&positions, &[]).abs() < EPS);
    }

    #[test]
    fn out_of_range_triangle_is_skipped_in_volume() {
        let positions = unit_cube_positions();
        let mut triangles = unit_cube_triangles();
        triangles.push([99, 100, 101]);
        let volume = mesh_volume(&positions, &triangles);
        assert!((volume - 1.0).abs() < EPS, "volume with junk tri: {volume}");
    }

    #[test]
    fn sanitized_clamps_and_scrubs_nan() {
        let raw = PressureParams::new(-3.0, -2.0, Compliance(-1.0));
        let clean = raw.sanitized();
        assert!((clean.rest_volume - 3.0).abs() < EPS);
        assert!(clean.overpressure > 0.0);
        assert!(clean.compliance.value() < EPS);

        let nan = PressureParams::new(f32::NAN, f32::NAN, Compliance::RIGID);
        let clean = nan.sanitized();
        assert!(clean.rest_volume.abs() < EPS);
        assert!((clean.overpressure - 1.0).abs() < EPS);
    }

    #[test]
    fn inflation_drives_volume_toward_target() {
        let triangles = unit_cube_triangles();
        let mut particles = free_particles(&unit_cube_positions());
        let params = PressureParams::new(1.0, 2.0, Compliance::RIGID);
        for _ in 0..200 {
            project_pressure(&mut particles, &triangles, params, 1.0 / 60.0);
        }
        let volume = particle_volume(&particles, &triangles);
        assert!((volume - 2.0).abs() < 1.0e-2, "inflated volume: {volume}");
    }

    #[test]
    fn deflation_shrinks_volume() {
        let triangles = unit_cube_triangles();
        let mut particles = free_particles(&unit_cube_positions());
        let params = PressureParams::new(1.0, 0.5, Compliance::RIGID);
        for _ in 0..200 {
            project_pressure(&mut particles, &triangles, params, 1.0 / 60.0);
        }
        let volume = particle_volume(&particles, &triangles);
        assert!((volume - 0.5).abs() < 1.0e-2, "deflated volume: {volume}");
    }

    #[test]
    fn zero_dt_is_a_no_op() {
        let triangles = unit_cube_triangles();
        let before = free_particles(&unit_cube_positions());
        let mut particles = before.clone();
        let params = PressureParams::new(1.0, 2.0, Compliance::RIGID);
        project_pressure(&mut particles, &triangles, params, 0.0);
        assert_eq!(particles, before);
    }

    #[test]
    fn empty_mesh_projection_is_a_no_op() {
        let mut particles = free_particles(&unit_cube_positions());
        let before = particles.clone();
        let params = PressureParams::new(1.0, 2.0, Compliance::RIGID);
        project_pressure(&mut particles, &[], params, 1.0 / 60.0);
        assert_eq!(particles, before);
    }

    #[test]
    fn out_of_range_triangle_does_not_panic_in_projection() {
        let mut particles = free_particles(&unit_cube_positions());
        let triangles = [[0u32, 1, 2], [50, 60, 70]];
        let params = PressureParams::new(1.0, 2.0, Compliance::RIGID);
        // Must not panic.
        project_pressure(&mut particles, &triangles, params, 1.0 / 60.0);
    }

    #[test]
    fn pinned_particles_never_move() {
        let triangles = unit_cube_triangles();
        let positions = unit_cube_positions();
        let mut particles = free_particles(&positions);
        // Pin the first two corners in place.
        particles[0] = ClothParticle::pinned(positions[0]);
        particles[1] = ClothParticle::pinned(positions[1]);
        let params = PressureParams::new(1.0, 3.0, Compliance::RIGID);
        for _ in 0..100 {
            project_pressure(&mut particles, &triangles, params, 1.0 / 60.0);
        }
        assert_eq!(particles[0].position, positions[0]);
        assert_eq!(particles[1].position, positions[1]);
    }

    #[test]
    fn projection_is_deterministic() {
        let triangles = unit_cube_triangles();
        let params = PressureParams::new(1.0, 2.0, Compliance(0.01));
        let mut a = free_particles(&unit_cube_positions());
        let mut b = free_particles(&unit_cube_positions());
        for _ in 0..50 {
            project_pressure(&mut a, &triangles, params, 1.0 / 60.0);
            project_pressure(&mut b, &triangles, params, 1.0 / 60.0);
        }
        assert_eq!(a, b);
    }

    #[test]
    fn projection_produces_no_nan() {
        let triangles = unit_cube_triangles();
        let mut particles = free_particles(&unit_cube_positions());
        let params = PressureParams::new(1.0, 2.5, Compliance::RIGID);
        for _ in 0..100 {
            project_pressure(&mut particles, &triangles, params, 1.0 / 60.0);
        }
        for p in &particles {
            assert!(!p.position.x.is_nan());
            assert!(!p.position.y.is_nan());
            assert!(!p.position.z.is_nan());
        }
    }

    #[test]
    fn apply_pressure_matches_project_pressure() {
        let triangles = unit_cube_triangles();
        let params = PressureParams::new(1.0, 2.0, Compliance::RIGID);
        let mut a = free_particles(&unit_cube_positions());
        let mut b = free_particles(&unit_cube_positions());
        for _ in 0..25 {
            project_pressure(&mut a, &triangles, params, 1.0 / 60.0);
            apply_pressure(&mut b, &triangles, params, 1.0 / 60.0);
        }
        assert_eq!(a, b);
    }
}
