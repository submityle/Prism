//! The `CPU` golden twin for the cloth aero kernel.
//!
//! The authoritative per-face force lives in [`prism_physics_core`] as
//! `triangle_aero_force` (with `turbulence_offset` for the jitter); rather than
//! copy that arithmetic and risk drift, [`cpu_cloth_aero`] *delegates* every
//! face force to it and only owns the two-phase Jacobi schedule the
//! [`GpuClothAero`](super::gpu::GpuClothAero) kernel runs: compute every face
//! force from one frozen snapshot, then have each vertex gather the forces of
//! its incident faces (ascending by triangle) and write only its own velocity
//! slot.
//!
//! This is the Jacobi reformulation of `prism_physics_core`'s sequential
//! `apply_aero_forces`; the two agree exactly when no vertex is shared between
//! faces (verified in this module's tests against the engine's own pass), and
//! the Jacobi form is what a race-free kernel can execute.
//!
//! The twin is anchored, in this module's tests, against an independent
//! brute-force reference that re-derives the per-face force formula inline (and
//! re-hashes the turbulence from first principles), closing the loop without
//! reusing the engine's arithmetic (no fake parity).
//!
//! # Provenance
//!
//! The per-triangle drag/lift decomposition and the integer-hash turbulence are
//! standard, publicly documented cloth-aerodynamics techniques. No Unreal
//! Engine source or derived code.

use alloc::vec::Vec;

use glam::Vec3;
use prism_physics_core::{triangle_aero_force, turbulence_offset, AeroParams, WindField};

use super::{ClothAeroParams, ClothAeroTriangle, Real};

/// Runs one Jacobi aero velocity pre-pass on the `CPU`, returning the updated
/// velocities.
///
/// This is the golden twin of
/// [`GpuClothAero::solve`](super::gpu::GpuClothAero::solve): every in-range
/// face force is computed from the frozen `positions`/`velocities` snapshot and
/// spread evenly across its three corners (`force / 3`), then each free vertex
/// gathers the per-corner forces of its incident faces (ascending by triangle)
/// and gains `gathered * (inverse_mass * dt)`. Pinned vertices (inverse mass
/// `0`) keep their velocity, out-of-range triangles are skipped, and a
/// non-positive/non-finite `dt`, an empty particle set, an empty triangle set,
/// or mismatched column lengths all leave `velocities` unchanged.
#[must_use]
pub fn cpu_cloth_aero(
    positions: &[Vec3],
    velocities: &[Vec3],
    inverse_masses: &[Real],
    triangles: &[ClothAeroTriangle],
    params: ClothAeroParams,
    dt: Real,
) -> Vec<Vec3> {
    let count = positions.len();
    if dt <= 0.0
        || !dt.is_finite()
        || count == 0
        || triangles.is_empty()
        || velocities.len() != count
        || inverse_masses.len() != count
    {
        return velocities.to_vec();
    }

    let field = WindField::new(Vec3::from_array(params.velocity), params.turbulence).sanitized();
    let aero = AeroParams::new(params.drag, params.lift)
        .with_air_density(params.air_density)
        .sanitized();

    let count_u32 = u32::try_from(count).unwrap_or(u32::MAX);

    // Phase 1: each in-range face's evenly-shared force, from the frozen
    // snapshot. The per-vertex incidence is accumulated in the same walk so each
    // vertex lists its faces in ascending triangle order.
    let mut per_vertex_force: Vec<Vec3> = Vec::with_capacity(triangles.len());
    let mut incidence: Vec<Vec<usize>> = Vec::with_capacity(count);
    incidence.resize_with(count, Vec::new);
    for tri in triangles {
        let [i0, i1, i2] = tri.indices();
        if i0 >= count_u32 || i1 >= count_u32 || i2 >= count_u32 {
            continue;
        }
        let wind = field.velocity + turbulence_offset([i0, i1, i2], field.turbulence);
        let force = triangle_aero_force(
            positions[i0 as usize],
            positions[i1 as usize],
            positions[i2 as usize],
            velocities[i0 as usize],
            velocities[i1 as usize],
            velocities[i2 as usize],
            wind,
            aero,
        );
        let ti = per_vertex_force.len();
        per_vertex_force.push(force * (1.0 / 3.0));
        for &v in &[i0, i1, i2] {
            incidence[v as usize].push(ti);
        }
    }

    // Phase 2: each vertex gathers its incident faces' shared force, then scales
    // the sum by `inverse_mass * dt` once and adds it to the frozen velocity.
    let mut out = velocities.to_vec();
    for v in 0..count {
        let w = inverse_masses[v];
        if w <= 0.0 {
            continue;
        }
        let faces = &incidence[v];
        if faces.is_empty() {
            continue;
        }
        let mut acc = Vec3::ZERO;
        for &ti in faces {
            acc += per_vertex_force[ti];
        }
        out[v] += acc * (w * dt);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_physics_core::apply_aero_forces;

    fn tri(i0: u32, i1: u32, i2: u32) -> ClothAeroTriangle {
        ClothAeroTriangle::new(i0, i1, i2)
    }

    /// An independent re-derivation of one face's evenly-shared force, re-hashed
    /// from the vertex indices without touching the engine's helpers.
    fn brute_per_vertex_force(
        p0: Vec3,
        p1: Vec3,
        p2: Vec3,
        v0: Vec3,
        v1: Vec3,
        v2: Vec3,
        wind: Vec3,
        drag: f32,
        lift: f32,
        air_density: f32,
    ) -> Vec3 {
        let cross = (p1 - p0).cross(p2 - p0);
        let cross_len_sq = cross.length_squared();
        if cross_len_sq <= 1.0e-12 {
            return Vec3::ZERO;
        }
        let area = 0.5 * cross_len_sq.sqrt();
        let normal = cross * (1.0 / cross_len_sq.sqrt());
        let face_velocity = (v0 + v1 + v2) * (1.0 / 3.0);
        let relative = wind - face_velocity;
        let normal_component = normal * relative.dot(normal);
        let tangent_component = relative - normal_component;
        let directional = normal_component * drag + tangent_component * lift;
        let pressure = if air_density > 0.0 {
            area * (0.5 * air_density * relative.length_squared().sqrt())
        } else {
            area
        };
        directional * pressure * (1.0 / 3.0)
    }

    fn brute_hash_to_unit(seed: u32) -> f32 {
        let mut h = seed.wrapping_mul(0x9E37_79B1);
        h ^= h >> 15;
        h = h.wrapping_mul(0x85EB_CA77);
        h ^= h >> 13;
        h = h.wrapping_mul(0xC2B2_AE3D);
        h ^= h >> 16;
        let unit = (h >> 8) as f32 * (1.0 / 16_777_216.0);
        unit * 2.0 - 1.0
    }

    fn brute_turbulence(indices: [u32; 3], turbulence: f32) -> Vec3 {
        if turbulence <= 0.0 {
            return Vec3::ZERO;
        }
        let base = indices[0].wrapping_mul(73_856_093)
            ^ indices[1].wrapping_mul(19_349_663)
            ^ indices[2].wrapping_mul(83_492_791);
        Vec3::new(
            brute_hash_to_unit(base ^ 0x00A5_5A00),
            brute_hash_to_unit(base ^ 0x5A00_00A5),
            brute_hash_to_unit(base ^ 0x00FF_00FF),
        ) * turbulence
    }

    fn brute_force(
        positions: &[Vec3],
        velocities: &[Vec3],
        inverse_masses: &[Real],
        triangles: &[ClothAeroTriangle],
        params: ClothAeroParams,
        dt: Real,
    ) -> Vec<Vec3> {
        let count = positions.len();
        let count_u32 = count as u32;
        // Sanitize independently (clamp wind finite, turbulence 0..=1, coeffs
        // non-negative) to match the golden's up-front sanitize.
        let wind_steady = Vec3::new(
            if params.velocity[0].is_finite() {
                params.velocity[0]
            } else {
                0.0
            },
            if params.velocity[1].is_finite() {
                params.velocity[1]
            } else {
                0.0
            },
            if params.velocity[2].is_finite() {
                params.velocity[2]
            } else {
                0.0
            },
        );
        let turb = params.turbulence.clamp(0.0, 1.0);
        let drag = params.drag.max(0.0);
        let lift = params.lift.max(0.0);
        let air = params.air_density.max(0.0);

        let mut per_vertex_force: Vec<Vec3> = Vec::new();
        let mut incidence: Vec<Vec<usize>> = Vec::new();
        incidence.resize_with(count, Vec::new);
        for tri in triangles {
            let [i0, i1, i2] = tri.indices();
            if i0 >= count_u32 || i1 >= count_u32 || i2 >= count_u32 {
                continue;
            }
            let wind = wind_steady + brute_turbulence([i0, i1, i2], turb);
            let pvf = brute_per_vertex_force(
                positions[i0 as usize],
                positions[i1 as usize],
                positions[i2 as usize],
                velocities[i0 as usize],
                velocities[i1 as usize],
                velocities[i2 as usize],
                wind,
                drag,
                lift,
                air,
            );
            let ti = per_vertex_force.len();
            per_vertex_force.push(pvf);
            for &v in &[i0, i1, i2] {
                incidence[v as usize].push(ti);
            }
        }
        let mut out = velocities.to_vec();
        for v in 0..count {
            let w = inverse_masses[v];
            if w <= 0.0 {
                continue;
            }
            let mut acc = Vec3::ZERO;
            for &ti in &incidence[v] {
                acc += per_vertex_force[ti];
            }
            out[v] += acc * (w * dt);
        }
        out
    }

    fn max_abs_diff(a: &[Vec3], b: &[Vec3]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (*x - *y).abs().max_element())
            .fold(0.0_f32, f32::max)
    }

    /// A 3x3 grid of particles triangulated into a two-triangle-per-quad sheet,
    /// so interior vertices are shared by several faces.
    fn grid_sheet() -> (Vec<Vec3>, Vec<Vec3>, Vec<Real>, Vec<ClothAeroTriangle>) {
        let mut positions = Vec::new();
        for y in 0..3 {
            for x in 0..3 {
                positions.push(Vec3::new(x as f32, y as f32, 0.0));
            }
        }
        let velocities = alloc::vec![Vec3::ZERO; positions.len()];
        let inverse_masses = alloc::vec![1.0_f32; positions.len()];
        let idx = |x: u32, y: u32| y * 3 + x;
        let mut triangles = Vec::new();
        for y in 0..2u32 {
            for x in 0..2u32 {
                triangles.push(tri(idx(x, y), idx(x + 1, y), idx(x, y + 1)));
                triangles.push(tri(idx(x + 1, y), idx(x + 1, y + 1), idx(x, y + 1)));
            }
        }
        (positions, velocities, inverse_masses, triangles)
    }

    #[test]
    fn empty_or_noop_inputs_return_velocities_unchanged() {
        let (positions, velocities, inverse_masses, triangles) = grid_sheet();
        let p = ClothAeroParams::new([0.0, 0.0, 2.0], 0.0, 1.0, 0.3);
        let none = cpu_cloth_aero(&positions, &velocities, &inverse_masses, &[], p, 1.0 / 60.0);
        assert_eq!(none, velocities);
        let zero = cpu_cloth_aero(&positions, &velocities, &inverse_masses, &triangles, p, 0.0);
        assert_eq!(zero, velocities);
    }

    #[test]
    fn matches_brute_force_reference() {
        let (positions, velocities, inverse_masses, triangles) = grid_sheet();
        for p in [
            ClothAeroParams::new([0.0, 0.0, 2.0], 0.0, 1.0, 0.0),
            ClothAeroParams::new([1.0, -0.5, 2.0], 0.0, 0.8, 0.4),
            ClothAeroParams::new([0.5, 0.0, 1.5], 0.6, 1.0, 0.0).with_air_density(1.225),
        ] {
            let golden = cpu_cloth_aero(
                &positions,
                &velocities,
                &inverse_masses,
                &triangles,
                p,
                1.0 / 60.0,
            );
            let brute = brute_force(
                &positions,
                &velocities,
                &inverse_masses,
                &triangles,
                p,
                1.0 / 60.0,
            );
            assert!(
                max_abs_diff(&golden, &brute) <= 1.0e-6,
                "golden drifted from the brute-force anchor for {p:?}"
            );
        }
    }

    #[test]
    fn single_triangle_matches_engine_sequential_pass() {
        // With no shared vertices the Jacobi gather equals the engine's
        // sequential scatter exactly.
        let positions = alloc::vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let velocities = alloc::vec![Vec3::ZERO; 3];
        let inverse_masses = alloc::vec![1.0_f32; 3];
        let triangles = alloc::vec![tri(0, 1, 2)];
        let p = ClothAeroParams::new([0.0, 0.0, 2.0], 0.0, 1.0, 0.0);
        let golden = cpu_cloth_aero(
            &positions,
            &velocities,
            &inverse_masses,
            &triangles,
            p,
            1.0 / 60.0,
        );

        let mut engine_vel = velocities.clone();
        apply_aero_forces(
            &positions,
            &mut engine_vel,
            &inverse_masses,
            &[[0, 1, 2]],
            &WindField::new(Vec3::new(0.0, 0.0, 2.0), 0.0),
            AeroParams::new(1.0, 0.0),
            1.0 / 60.0,
        );
        assert!(
            max_abs_diff(&golden, &engine_vel) <= 1.0e-6,
            "single-triangle Jacobi diverged from the engine's sequential pass"
        );
    }

    #[test]
    fn pinned_vertices_keep_their_velocity() {
        let (positions, velocities, mut inverse_masses, triangles) = grid_sheet();
        inverse_masses[0] = 0.0;
        inverse_masses[4] = 0.0; // the shared interior vertex
        let p = ClothAeroParams::new([0.0, 0.0, 2.0], 0.0, 1.0, 0.0);
        let out = cpu_cloth_aero(
            &positions,
            &velocities,
            &inverse_masses,
            &triangles,
            p,
            1.0 / 60.0,
        );
        assert_eq!(out[0], velocities[0]);
        assert_eq!(out[4], velocities[4]);
    }

    #[test]
    fn out_of_range_triangle_is_skipped() {
        let positions = alloc::vec![Vec3::ZERO, Vec3::X, Vec3::Y];
        let velocities = alloc::vec![Vec3::ZERO; 3];
        let inverse_masses = alloc::vec![1.0_f32; 3];
        let triangles = alloc::vec![tri(0, 1, 9)];
        let p = ClothAeroParams::new([0.0, 0.0, 2.0], 0.0, 1.0, 0.0);
        let out = cpu_cloth_aero(
            &positions,
            &velocities,
            &inverse_masses,
            &triangles,
            p,
            1.0 / 60.0,
        );
        assert_eq!(out, velocities);
    }
}
