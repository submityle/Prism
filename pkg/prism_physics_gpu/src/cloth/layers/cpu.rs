//! The `CPU` golden twin for the multi-layer garment coupling pass.
//!
//! The authoritative parallel-safe pass lives in [`prism_physics_core`] as
//! [`resolve_layer_coupling_jacobi`]: a frozen-snapshot Jacobi reformulation of
//! the Gauss-Seidel inter-layer coupling core that gathers every particle's own
//! half of the separating push from a read-only snapshot and applies the summed
//! per-particle correction once. Rather than copy that arithmetic (and risk it
//! drifting from the engine), this twin *delegates* to it over a cloned
//! position column and returns the applied particle positions, which is exactly
//! what the [`GpuClothLayerCoupling`](super::gpu::GpuClothLayerCoupling) kernel
//! reproduces. The parity suite then compares the two within a tight tolerance.
//!
//! # Provenance
//!
//! The layer-number stacking constraint and inverse-mass-weighted separation
//! are standard position-based dynamics; the Jacobi own-slot accumulate/apply
//! split is standard parallel position-based dynamics. No Unreal Engine source
//! or derived code.

use alloc::vec::Vec;

use glam::Vec3;
use prism_physics_core::{resolve_layer_coupling_jacobi, LayerParams};

/// Scalar type shared with [`prism_physics_core`] (`f32`).
type Real = f32;

/// Resolves every penetrating cross-layer particle pair with one parallel-safe
/// (Jacobi) inter-layer coupling iteration and returns the applied particle
/// positions.
///
/// This is the golden twin of
/// [`GpuClothLayerCoupling::solve`](super::gpu::GpuClothLayerCoupling::solve):
/// both reproduce the engine's own parallel-safe inter-layer coupling pass.
///
/// A non-positive sanitized `thickness`/`cell_size`, fewer than two particles,
/// a mismatched `inverse_masses` length, or a particle missing a `layer_of`
/// entry degrades exactly as the delegate does. The input slices are not
/// mutated; the returned `Vec` carries the post-pass positions.
#[must_use]
pub fn cpu_cloth_layer_coupling(
    positions: &[Vec3],
    inverse_masses: &[Real],
    layer_of: &[u32],
    normals: &[Vec3],
    params: LayerParams,
) -> Vec<Vec3> {
    let mut out = positions.to_vec();
    resolve_layer_coupling_jacobi(&mut out, inverse_masses, layer_of, normals, params);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> LayerParams {
        LayerParams {
            thickness: 0.1,
            cell_size: 0.2,
        }
    }

    #[test]
    fn penetrating_cross_layer_pair_separates_to_thickness() {
        let positions = [Vec3::ZERO, Vec3::new(0.0, -0.05, 0.0)];
        let im = [1.0, 1.0];
        let layer_of = [0u32, 1u32];
        let normals = [Vec3::new(0.0, 1.0, 0.0), Vec3::ZERO];
        let out = cpu_cloth_layer_coupling(&positions, &im, &layer_of, &normals, params());
        let signed = (out[1] - out[0]).y;
        assert!(signed >= 0.1 - 1e-5, "outer driven to +normal side: {signed}");
    }

    #[test]
    fn pinned_inner_never_moves() {
        let positions = [Vec3::ZERO, Vec3::new(0.0, -0.05, 0.0)];
        let im = [0.0, 1.0];
        let layer_of = [0u32, 1u32];
        let normals = [Vec3::new(0.0, 1.0, 0.0), Vec3::ZERO];
        let out = cpu_cloth_layer_coupling(&positions, &im, &layer_of, &normals, params());
        assert_eq!(out[0], positions[0], "pinned inner particle must not move");
    }

    #[test]
    fn disabled_thickness_leaves_inputs_untouched() {
        let positions = [Vec3::ZERO, Vec3::new(0.0, -0.05, 0.0)];
        let im = [1.0, 1.0];
        let layer_of = [0u32, 1u32];
        let normals = [Vec3::new(0.0, 1.0, 0.0), Vec3::ZERO];
        let out = cpu_cloth_layer_coupling(
            &positions,
            &im,
            &layer_of,
            &normals,
            LayerParams {
                thickness: 0.0,
                cell_size: 0.2,
            },
        );
        assert_eq!(out, positions.to_vec());
    }
}
