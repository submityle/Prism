//! Multi-layer garment coupling (design §6.7).
//!
//! A dressed character stacks garments — shirt under jacket, lining under
//! skirt. Each garment simulates its own cloth, but nothing stops an outer
//! layer from sinking through the layer beneath it. Production engines (Houdini
//! `Vellum`, UE5 `Chaos`) give every cloth a *layer number* and add inter-layer
//! collision constraints that both keep the layers apart and preserve their
//! stacking order: the higher-numbered (outer) layer always ends up on the
//! outward side of the lower-numbered (inner) one.
//!
//! The contact kernel itself lives in the physics engine
//! ([`prism_physics_core::soft::collision::resolve_layer_coupling`]) as the
//! single source of truth. This module keeps the render-side public API and
//! parameter guards, converts the compact particle layout to the engine's
//! structure-of-arrays columns through [`physics_bridge`](super::physics_bridge),
//! and projects through that one implementation — there is no second copy of the
//! spatial-hash sweep or the per-pair projection here.
//!
//! Like every collision pass it is stateless array-in / array-out, only uses
//! `sqrt`, skips out-of-range indices instead of panicking, and iterates in a
//! fixed cell / index order for determinism.

use alloc::vec::Vec;

use glam::Vec3 as GlamVec3;

use super::{physics_bridge, ClothParticle, Vec3};
use prism_physics_core::soft::collision as physics_collision;

/// Tuning for the inter-layer coupling pass.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LayerParams {
    /// Minimum separation enforced between particles of different layers (the
    /// combined cloth thickness). Non-positive disables the pass.
    pub thickness: f32,
    /// Spatial-hash cell size. The 27-cell neighborhood search is only correct
    /// when this is at least `thickness`; the pass raises it internally if an
    /// author sets it smaller. Non-positive disables the pass.
    pub cell_size: f32,
}

impl Default for LayerParams {
    /// A thin default separation with a cell sized to match it.
    fn default() -> Self {
        Self {
            thickness: 0.01,
            cell_size: 0.01,
        }
    }
}

impl LayerParams {
    /// Returns a copy with `NaN` scrubbed to zero and `cell_size` raised to at
    /// least `thickness`, so the neighborhood search always covers the
    /// separation radius. A non-positive `thickness` or `cell_size` still
    /// disables the pass (checked by [`resolve_layer_coupling`]).
    #[must_use]
    pub fn sanitized(self) -> Self {
        let thickness = if self.thickness.is_nan() {
            0.0
        } else {
            self.thickness
        };
        let mut cell_size = if self.cell_size.is_nan() {
            0.0
        } else {
            self.cell_size
        };
        if thickness > 0.0 && cell_size < thickness {
            cell_size = thickness;
        }
        Self {
            thickness,
            cell_size,
        }
    }
}

/// Converts the render-side [`LayerParams`] into the physics engine's
/// field-identical parameter struct. The engine re-sanitizes internally, so a
/// raw or already-sanitized value projects identically.
#[inline]
#[must_use]
fn to_physics_params(params: LayerParams) -> physics_collision::LayerParams {
    physics_collision::LayerParams {
        thickness: params.thickness,
        cell_size: params.cell_size,
    }
}

/// Maps a world-space position to its integer spatial-hash cell.
///
/// `cell_size` is assumed positive (the caller guards this). The cast saturates
/// rather than wrapping, so an extreme coordinate still buckets deterministically
/// and never panics. Shared with [`super::layers_jacobi`] so both solvers bucket
/// identically.
pub(crate) fn cell_of(pos: Vec3, cell_size: f32) -> (i32, i32, i32) {
    let inv = 1.0 / cell_size;
    let cx = (pos.x * inv).floor() as i32;
    let cy = (pos.y * inv).floor() as i32;
    let cz = (pos.z * inv).floor() as i32;
    (cx, cy, cz)
}

/// Keeps stacked garment layers from interpenetrating while preserving their
/// stacking order.
///
/// `layer_of[i]` is particle `i`'s layer number (lower = inner) and
/// `normals[i]` is its outward surface normal; both are parallel to
/// `particles`. Only pairs whose layer numbers differ interact. A particle
/// missing a layer number or normal entry is skipped. The pass is a no-op when
/// `thickness`/`cell_size` are non-positive or there are fewer than two
/// particles.
///
/// The projection itself is delegated to
/// [`prism_physics_core::soft::collision::resolve_layer_coupling`]; this wrapper
/// only guards the cheap disabled cases (to skip the structure-of-arrays
/// allocation), converts to the engine's columns, and writes the solved
/// positions back. Pinned particles map to a zero inverse mass through
/// [`physics_bridge::to_soa`], so the engine leaves them fixed.
pub fn resolve_layer_coupling(
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
    physics_collision::resolve_layer_coupling(
        &mut positions,
        &inverse_masses,
        layer_of,
        &glam_normals,
        to_physics_params(params),
    );
    physics_bridge::write_positions_back(particles, &positions);
}

/// Accumulates an area-weighted outward vertex normal for every particle from a
/// triangle mesh, writing one [`Vec3`] per particle into `out` (resized and
/// cleared first so index `i` is particle `i`'s normal).
///
/// Each face contributes its unnormalized cross product `(p1 - p0) x (p2 - p0)`
/// — whose magnitude is twice the triangle area — to each of its three
/// vertices, so larger faces weigh more and the per-vertex sum is the standard
/// area-weighted normal. Every vertex normal is normalized at the end; a vertex
/// touched by no face (or by only degenerate faces) is left at zero, which the
/// coupling pass reads as "no preferred side" and resolves radially. Winding is
/// assumed consistent (counter-clockwise seen from outside) so the normals face
/// outward, matching the pressure pass. Out-of-range indices are skipped and
/// never panic, so a truncated triangle set is safe.
///
/// The geometry is computed by
/// [`prism_physics_core::soft::collision::accumulate_vertex_normals`]; this
/// wrapper only bridges the compact and `glam` vector layouts.
pub fn accumulate_vertex_normals(positions: &[Vec3], triangles: &[[u32; 3]], out: &mut Vec<Vec3>) {
    let glam_positions: Vec<GlamVec3> =
        positions.iter().map(|p| physics_bridge::to_glam(*p)).collect();
    let mut glam_out: Vec<GlamVec3> = Vec::new();
    physics_collision::accumulate_vertex_normals(&glam_positions, triangles, &mut glam_out);
    out.clear();
    out.reserve(glam_out.len());
    out.extend(glam_out.iter().map(|n| physics_bridge::from_glam(*n)));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cloth::ClothParticle;

    /// A free particle at `pos`.
    fn free(pos: Vec3) -> ClothParticle {
        ClothParticle {
            position: pos,
            velocity: Vec3::ZERO,
            inverse_mass: 1.0,
        }
    }

    #[test]
    fn sanitized_raises_cell_size_and_scrubs_nan() {
        let p = LayerParams {
            thickness: 0.2,
            cell_size: 0.05,
        }
        .sanitized();
        assert!((p.cell_size - 0.2).abs() < 1e-9);
        let n = LayerParams {
            thickness: f32::NAN,
            cell_size: f32::NAN,
        }
        .sanitized();
        assert!((n.thickness - 0.0).abs() < 1e-9);
        assert!((n.cell_size - 0.0).abs() < 1e-9);
    }

    #[test]
    fn oriented_contact_pushes_outer_to_the_outward_side() {
        // Inner at origin, outward normal +y; outer sits just below it (wrong
        // side). The pass must lift the outer above the inner by `thickness`.
        let mut particles = [free(Vec3::ZERO), free(Vec3::new(0.0, -0.05, 0.0))];
        let layer_of = [0u32, 1u32];
        let normals = [Vec3::new(0.0, 1.0, 0.0), Vec3::ZERO];
        resolve_layer_coupling(
            &mut particles,
            &layer_of,
            &normals,
            LayerParams {
                thickness: 0.1,
                cell_size: 0.2,
            },
        );
        // Signed separation along +y must be at least thickness.
        let signed = particles[1].position.sub(particles[0].position).y;
        assert!(signed >= 0.1 - 1e-6, "signed separation {signed}");
    }

    #[test]
    fn equal_mass_split_moves_both_symmetrically() {
        let mut particles = [free(Vec3::ZERO), free(Vec3::new(0.0, 0.02, 0.0))];
        let layer_of = [0u32, 1u32];
        let normals = [Vec3::new(0.0, 1.0, 0.0), Vec3::ZERO];
        resolve_layer_coupling(
            &mut particles,
            &layer_of,
            &normals,
            LayerParams {
                thickness: 0.1,
                cell_size: 0.2,
            },
        );
        // Both moved half the penetration (0.08): inner down, outer up.
        assert!((particles[0].position.y - (-0.04)).abs() < 1e-5);
        assert!((particles[1].position.y - 0.06).abs() < 1e-5);
    }

    #[test]
    fn same_layer_pairs_are_ignored() {
        let mut particles = [free(Vec3::ZERO), free(Vec3::new(0.0, 0.01, 0.0))];
        let layer_of = [2u32, 2u32];
        let normals = [Vec3::new(0.0, 1.0, 0.0), Vec3::ZERO];
        resolve_layer_coupling(
            &mut particles,
            &layer_of,
            &normals,
            LayerParams {
                thickness: 0.1,
                cell_size: 0.2,
            },
        );
        // Untouched: intra-layer contacts are self-collision's job.
        assert!((particles[0].position.y - 0.0).abs() < 1e-9);
        assert!((particles[1].position.y - 0.01).abs() < 1e-9);
    }

    #[test]
    fn pinned_inner_moves_only_the_outer() {
        let mut particles = [
            ClothParticle::pinned(Vec3::ZERO),
            free(Vec3::new(0.0, -0.05, 0.0)),
        ];
        let layer_of = [0u32, 1u32];
        let normals = [Vec3::new(0.0, 1.0, 0.0), Vec3::ZERO];
        resolve_layer_coupling(
            &mut particles,
            &layer_of,
            &normals,
            LayerParams {
                thickness: 0.1,
                cell_size: 0.2,
            },
        );
        // Inner is pinned; only the outer moves, to +thickness along the normal.
        assert!((particles[0].position.y - 0.0).abs() < 1e-9);
        assert!((particles[1].position.y - 0.1).abs() < 1e-5);
    }

    #[test]
    fn no_normal_falls_back_to_radial_separation() {
        // Both normals zero -> symmetric radial push apart to `thickness`.
        let mut particles = [free(Vec3::ZERO), free(Vec3::new(0.03, 0.0, 0.0))];
        let layer_of = [0u32, 1u32];
        let normals = [Vec3::ZERO, Vec3::ZERO];
        resolve_layer_coupling(
            &mut particles,
            &layer_of,
            &normals,
            LayerParams {
                thickness: 0.1,
                cell_size: 0.2,
            },
        );
        let dist = particles[0].position.distance(particles[1].position);
        assert!((dist - 0.1).abs() < 1e-5, "separated distance {dist}");
    }

    #[test]
    fn far_apart_layers_are_untouched() {
        let mut particles = [free(Vec3::ZERO), free(Vec3::new(0.0, 5.0, 0.0))];
        let layer_of = [0u32, 1u32];
        let normals = [Vec3::new(0.0, 1.0, 0.0), Vec3::ZERO];
        resolve_layer_coupling(
            &mut particles,
            &layer_of,
            &normals,
            LayerParams {
                thickness: 0.1,
                cell_size: 0.2,
            },
        );
        assert!((particles[1].position.y - 5.0).abs() < 1e-9);
    }

    #[test]
    fn missing_layer_entries_are_skipped() {
        // Only one layer entry: the second particle never participates.
        let mut particles = [free(Vec3::ZERO), free(Vec3::new(0.0, 0.01, 0.0))];
        let layer_of = [0u32];
        let normals = [Vec3::new(0.0, 1.0, 0.0)];
        resolve_layer_coupling(
            &mut particles,
            &layer_of,
            &normals,
            LayerParams {
                thickness: 0.1,
                cell_size: 0.2,
            },
        );
        assert!((particles[1].position.y - 0.01).abs() < 1e-9);
    }

    #[test]
    fn coupling_is_deterministic() {
        let build = || {
            [
                free(Vec3::new(0.0, 0.0, 0.0)),
                free(Vec3::new(0.0, -0.03, 0.0)),
                free(Vec3::new(0.01, -0.02, 0.0)),
            ]
        };
        let layer_of = [0u32, 1u32, 1u32];
        let normals = [
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let params = LayerParams {
            thickness: 0.1,
            cell_size: 0.2,
        };
        let mut a = build();
        let mut b = build();
        resolve_layer_coupling(&mut a, &layer_of, &normals, params);
        resolve_layer_coupling(&mut b, &layer_of, &normals, params);
        for (pa, pb) in a.iter().zip(b.iter()) {
            assert!(pa.position.distance(pb.position) < 1e-9);
        }
    }

    #[test]
    fn vertex_normals_of_a_flat_sheet_point_up() {
        // Two triangles tiling a unit quad in the y = 0 plane, wound CCW seen
        // from +y, must give every touched vertex a +y unit normal.
        let positions = alloc::vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 1.0),
            Vec3::new(0.0, 0.0, 1.0),
        ];
        // Wound counter-clockwise seen from +y so the right-hand normal is +y.
        let triangles = alloc::vec![[0u32, 2, 1], [0, 3, 2]];
        let mut normals: Vec<Vec3> = Vec::new();
        accumulate_vertex_normals(&positions, &triangles, &mut normals);
        assert_eq!(normals.len(), 4);
        for n in &normals {
            assert!((n.x).abs() < 1e-6, "normal not vertical: {n:?}");
            assert!((n.z).abs() < 1e-6, "normal not vertical: {n:?}");
            assert!((n.y - 1.0).abs() < 1e-6, "normal not +y unit: {n:?}");
        }
    }

    #[test]
    fn vertex_normals_skip_out_of_range_and_leave_untouched_zero() {
        // An out-of-range face is skipped (no panic) and a vertex no face
        // touches stays zero, which the coupling pass reads as "no side".
        let positions = alloc::vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(5.0, 0.0, 5.0),
        ];
        // First face wound for a +y normal; second indexes vertex 9 (absent).
        let triangles = alloc::vec![[0u32, 2, 1], [0, 2, 9]];
        let mut normals: Vec<Vec3> = Vec::new();
        accumulate_vertex_normals(&positions, &triangles, &mut normals);
        assert_eq!(normals.len(), 4);
        assert!((normals[0].y - 1.0).abs() < 1e-6);
        // Vertex 3 is touched by no valid face -> left at zero.
        assert!(normals[3].length_squared() < 1e-12);
    }
}
