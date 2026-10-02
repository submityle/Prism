//! Jacobi (parallel-safe) inter-layer garment coupling — the `GPU`-faithful golden.
//!
//! [`super::layers::resolve_layer_coupling`] keeps stacked garments from
//! interpenetrating in **Gauss-Seidel** order: it walks the cross-layer sample
//! pairs in a fixed cell order and each contact scatters its correction onto the
//! two particles *in place*, so a later pair already sees the moved positions of
//! an earlier one. That is the sequential `CPU` reference, but it does not map to
//! a `GPU` compute kernel: on the `GPU` every particle is updated in parallel from
//! the *same* read-only snapshot, which is a **Jacobi** iteration, not
//! Gauss-Seidel. The two converge to the same separated, correctly stacked state
//! but are never bit-for-bit identical on a multi-contact pass, so a faithful
//! `WGSL`/`WESL` twin needs its own golden rather than borrowing the Gauss-Seidel
//! one (design section 9: no fake parity). This mirrors the sibling
//! [`super::virtual_particles_jacobi`] contract that owns the virtual-particle
//! tier's `GPU` golden.
//!
//! The own-slot Jacobi kernel itself (the frozen-snapshot accumulation and the
//! per-pair half-correction) lives in the physics engine
//! ([`prism_physics_core::soft::collision::resolve_layer_coupling_jacobi`]) as
//! the single source of truth, next to its Gauss-Seidel sibling. This module
//! keeps the render-side public golden entry point and the parameter guards,
//! bridges the compact particle layout to the engine's structure-of-arrays
//! columns through [`physics_bridge`](super::physics_bridge), and projects
//! through that one implementation — there is no second copy of the spatial-hash
//! accumulation here.

use alloc::vec::Vec;

use glam::Vec3 as GlamVec3;

use super::layers::{to_physics_params, LayerParams};
use super::{physics_bridge, ClothParticle, Vec3};
use prism_physics_core::soft::collision as physics_collision;

/// Runs one Jacobi inter-layer coupling pass, the parallel-safe twin of
/// [`super::layers::resolve_layer_coupling`] and the `pub` golden the GPU twin
/// mirrors value-for-value.
///
/// One call is a single Jacobi iteration (accumulate every participating
/// particle's own half of its cross-layer contacts from the frozen snapshot,
/// then apply); repeated calls converge to the same separated, correctly stacked
/// state the Gauss-Seidel core reaches, without ever depending on evaluation
/// order. `layer_of[i]` is particle `i`'s layer number (lower = inner) and
/// `normals[i]` its outward surface normal; both are parallel to `particles`.
/// Only cross-layer pairs interact and a particle missing a layer or normal
/// entry is skipped. The pass is a no-op when `thickness`/`cell_size` are
/// non-positive or there are fewer than two particles.
///
/// The accumulation and projection are delegated to
/// [`prism_physics_core::soft::collision::resolve_layer_coupling_jacobi`]; this
/// wrapper only guards the cheap disabled cases (to skip the structure-of-arrays
/// allocation), converts to the engine's columns, and writes the solved
/// positions back. Pinned particles map to a zero inverse mass through
/// [`physics_bridge::to_soa`], so the engine leaves them fixed.
pub fn resolve_layer_coupling_jacobi(
    particles: &mut [ClothParticle],
    layer_of: &[u32],
    normals: &[Vec3],
    params: LayerParams,
) {
    let params = params.sanitized();
    if params.thickness <= 0.0 || params.cell_size <= 0.0 || particles.len() < 2 {
        return;
    }

    let (mut positions, inverse_masses) = physics_bridge::to_soa(particles);
    let glam_normals: Vec<GlamVec3> = normals.iter().map(|n| physics_bridge::to_glam(*n)).collect();
    physics_collision::resolve_layer_coupling_jacobi(
        &mut positions,
        &inverse_masses,
        layer_of,
        &glam_normals,
        to_physics_params(params),
    );
    physics_bridge::write_positions_back(particles, &positions);
}

#[cfg(test)]
mod tests {
    use super::super::layers::resolve_layer_coupling;
    use super::*;

    fn free(x: f32, y: f32, z: f32) -> ClothParticle {
        ClothParticle::new(Vec3::new(x, y, z), 1.0)
    }

    fn pinned(x: f32, y: f32, z: f32) -> ClothParticle {
        ClothParticle::new(Vec3::new(x, y, z), 0.0)
    }

    fn params() -> LayerParams {
        LayerParams {
            thickness: 0.1,
            cell_size: 0.2,
        }
    }

    fn assert_positions_eq(a: &[ClothParticle], b: &[ClothParticle]) {
        assert_eq!(a.len(), b.len());
        for (i, (pa, pb)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (pa.position.x - pb.position.x).abs() < 1e-6,
                "x mismatch at {i}"
            );
            assert!(
                (pa.position.y - pb.position.y).abs() < 1e-6,
                "y mismatch at {i}"
            );
            assert!(
                (pa.position.z - pb.position.z).abs() < 1e-6,
                "z mismatch at {i}"
            );
        }
    }

    /// A single oriented cross-layer contact must land on exactly the same
    /// positions the Gauss-Seidel core reaches: one contact has no ordering.
    #[test]
    fn single_oriented_contact_matches_gauss_seidel() {
        let base = [free(0.0, 0.0, 0.0), free(0.0, -0.05, 0.0)];
        let layer_of = [0u32, 1u32];
        let normals = [Vec3::new(0.0, 1.0, 0.0), Vec3::ZERO];

        let mut gs = base;
        resolve_layer_coupling(&mut gs, &layer_of, &normals, params());
        let mut jac = base;
        resolve_layer_coupling_jacobi(&mut jac, &layer_of, &normals, params());
        assert_positions_eq(&gs, &jac);

        // And the outer particle is actually lifted to the +normal side.
        let signed = jac[1].position.sub(jac[0].position).y;
        assert!(signed >= 0.1 - 1e-6, "signed separation {signed}");
    }

    /// The radial fallback (no usable inner normal) must also match the core.
    #[test]
    fn single_radial_contact_matches_gauss_seidel() {
        let base = [free(0.0, 0.0, 0.0), free(0.02, 0.0, 0.0)];
        let layer_of = [0u32, 1u32];
        let normals = [Vec3::ZERO, Vec3::ZERO];
        let mut gs = base;
        resolve_layer_coupling(&mut gs, &layer_of, &normals, params());
        let mut jac = base;
        resolve_layer_coupling_jacobi(&mut jac, &layer_of, &normals, params());
        assert_positions_eq(&gs, &jac);
        let dist = jac[1].position.sub(jac[0].position).length_squared().sqrt();
        assert!(dist >= 0.1 - 1e-6, "radial separation {dist}");
    }

    /// A pinned inner particle takes the whole correction onto the outer, and
    /// the Jacobi result still matches the Gauss-Seidel core.
    #[test]
    fn pinned_inner_matches_gauss_seidel() {
        let base = [pinned(0.0, 0.0, 0.0), free(0.0, -0.05, 0.0)];
        let layer_of = [0u32, 1u32];
        let normals = [Vec3::new(0.0, 1.0, 0.0), Vec3::ZERO];
        let mut gs = base;
        resolve_layer_coupling(&mut gs, &layer_of, &normals, params());
        let mut jac = base;
        resolve_layer_coupling_jacobi(&mut jac, &layer_of, &normals, params());
        assert_positions_eq(&gs, &jac);
        assert!((jac[0].position.y - 0.0).abs() < 1e-9, "inner pinned moved");
    }

    /// Same-layer particles never couple: the pass leaves them untouched.
    #[test]
    fn same_layer_pair_is_untouched() {
        let base = [free(0.0, 0.0, 0.0), free(0.0, 0.01, 0.0)];
        let layer_of = [2u32, 2u32];
        let normals = [Vec3::new(0.0, 1.0, 0.0), Vec3::ZERO];
        let mut jac = base;
        resolve_layer_coupling_jacobi(&mut jac, &layer_of, &normals, params());
        assert_positions_eq(&base, &jac);
    }

    /// A separated cross-layer pair (already beyond thickness) is untouched.
    #[test]
    fn separated_pair_is_untouched() {
        let base = [free(0.0, 0.0, 0.0), free(0.0, 0.5, 0.0)];
        let layer_of = [0u32, 1u32];
        let normals = [Vec3::new(0.0, 1.0, 0.0), Vec3::ZERO];
        let mut jac = base;
        resolve_layer_coupling_jacobi(&mut jac, &layer_of, &normals, params());
        assert_positions_eq(&base, &jac);
    }

    /// Non-positive params or fewer than two particles leave every particle
    /// untouched (the disabled guard runs before any projection).
    #[test]
    fn disabled_params_yield_zero_corrections() {
        let base = [free(0.0, 0.0, 0.0), free(0.0, -0.05, 0.0)];
        let layer_of = [0u32, 1u32];
        let normals = [Vec3::new(0.0, 1.0, 0.0), Vec3::ZERO];
        let mut jac = base;
        resolve_layer_coupling_jacobi(
            &mut jac,
            &layer_of,
            &normals,
            LayerParams {
                thickness: 0.0,
                cell_size: 0.2,
            },
        );
        assert_positions_eq(&base, &jac);
    }

    /// Applying one Jacobi pass displaces both endpoints by their own half, so an
    /// equal-mass contact moves the inner down and the outer up symmetrically —
    /// the two-sided Gauss-Seidel move reproduced from the frozen snapshot.
    #[test]
    fn both_endpoints_receive_their_half() {
        let base = [free(0.0, 0.0, 0.0), free(0.0, -0.05, 0.0)];
        let layer_of = [0u32, 1u32];
        let normals = [Vec3::new(0.0, 1.0, 0.0), Vec3::ZERO];
        let mut jac = base;
        resolve_layer_coupling_jacobi(&mut jac, &layer_of, &normals, params());
        // Equal mass: inner pushed down by half the penetration, outer up by half.
        let d_inner = jac[0].position.y - base[0].position.y;
        let d_outer = jac[1].position.y - base[1].position.y;
        assert!(d_inner < 0.0, "inner half should be negative, got {d_inner}");
        assert!(d_outer > 0.0, "outer half should be positive, got {d_outer}");
        assert!((d_inner + d_outer).abs() < 1e-6, "equal-mass halves cancel");
    }
}
