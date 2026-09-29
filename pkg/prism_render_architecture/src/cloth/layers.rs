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
//! [`resolve_layer_coupling`] implements that as a deterministic uniform
//! spatial-hash pass over a *combined* particle buffer, filtered so only
//! cross-layer pairs interact (intra-layer contacts are the job of
//! [`super::collision::resolve_self_collision`]). Each cross-layer contact is
//! resolved along the inner particle's outward normal, so the constraint is a
//! one-sided plane that forces the outer layer to the `+normal` side at least
//! `thickness` away — separation and ordering in one projection. When the
//! inner normal is unavailable (zero length), the pass falls back to a
//! symmetric radial minimum-distance push so it still prevents interpenetration
//! without a preferred side.
//!
//! Like every collision pass in this module it is stateless array-in /
//! array-out, only uses `sqrt`, skips out-of-range indices instead of
//! panicking, and iterates in a fixed cell / index order for determinism.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use super::{ClothParticle, Vec3, EPS_LEN_SQ};

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

/// Maps a world-space position to its integer spatial-hash cell.
///
/// `cell_size` is assumed positive (the caller guards this). The cast saturates
/// rather than wrapping, so an extreme coordinate still buckets deterministically
/// and never panics.
fn cell_of(pos: Vec3, cell_size: f32) -> (i32, i32, i32) {
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

    let mut grid: BTreeMap<(i32, i32, i32), Vec<u32>> = BTreeMap::new();
    for (index, particle) in particles.iter().enumerate() {
        // Only particles that carry a layer number participate.
        if index < layer_of.len() {
            let cell = cell_of(particle.position, params.cell_size);
            grid.entry(cell).or_default().push(index as u32);
        }
    }

    let thickness_sq = params.thickness * params.thickness;
    for (&cell, bucket) in &grid {
        for &a in bucket {
            let ai = a as usize;
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        let neighbor = (cell.0 + dx, cell.1 + dy, cell.2 + dz);
                        let Some(nbucket) = grid.get(&neighbor) else {
                            continue;
                        };
                        for &b in nbucket {
                            if b <= a {
                                continue;
                            }
                            let bi = b as usize;
                            // Same-layer contacts belong to self-collision.
                            if layer_of[ai] == layer_of[bi] {
                                continue;
                            }
                            resolve_layer_pair(
                                particles,
                                layer_of,
                                normals,
                                ai,
                                bi,
                                params.thickness,
                                thickness_sq,
                            );
                        }
                    }
                }
            }
        }
    }
}

/// Resolves one cross-layer contact, orienting the push by the inner particle's
/// outward normal so the outer layer is driven to the outward side.
///
/// Falls back to a symmetric radial minimum-distance push when the inner normal
/// is (near) zero, which still separates the pair but without a preferred side.
/// Two pinned particles cannot move, so the contact is left as-is.
fn resolve_layer_pair(
    particles: &mut [ClothParticle],
    layer_of: &[u32],
    normals: &[Vec3],
    ai: usize,
    bi: usize,
    thickness: f32,
    thickness_sq: f32,
) {
    // Lower layer number is the inner surface whose normal orients the contact.
    let (inner, outer) = if layer_of[ai] < layer_of[bi] {
        (ai, bi)
    } else {
        (bi, ai)
    };

    let w_inner = particles[inner].inverse_mass.max(0.0);
    let w_outer = particles[outer].inverse_mass.max(0.0);
    let w_sum = w_inner + w_outer;
    if w_sum <= 0.0 {
        return;
    }

    let p_inner = particles[inner].position;
    let p_outer = particles[outer].position;
    let normal = normals.get(inner).copied().unwrap_or(Vec3::ZERO);
    let unit = normal.normalize_or_zero();

    if unit.length_squared() > EPS_LEN_SQ {
        // Oriented plane contact: force the outer particle to at least
        // `thickness` along the inner's outward normal.
        let signed = p_outer.sub(p_inner).dot(unit);
        if signed >= thickness {
            return;
        }
        let penetration = thickness - signed;
        particles[inner].position = p_inner.add(unit.scale(-penetration * (w_inner / w_sum)));
        particles[outer].position = p_outer.add(unit.scale(penetration * (w_outer / w_sum)));
        return;
    }

    // No usable normal: symmetric radial separation.
    let delta = p_outer.sub(p_inner);
    let dist_sq = delta.length_squared();
    if dist_sq >= thickness_sq {
        return;
    }
    let (dir, penetration) = if dist_sq <= EPS_LEN_SQ {
        (Vec3::new(1.0, 0.0, 0.0), thickness)
    } else {
        let dist = dist_sq.sqrt();
        (delta.scale(1.0 / dist), thickness - dist)
    };
    particles[inner].position = p_inner.add(dir.scale(-penetration * (w_inner / w_sum)));
    particles[outer].position = p_outer.add(dir.scale(penetration * (w_outer / w_sum)));
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
pub fn accumulate_vertex_normals(positions: &[Vec3], triangles: &[[u32; 3]], out: &mut Vec<Vec3>) {
    out.clear();
    out.resize(positions.len(), Vec3::ZERO);
    for tri in triangles {
        let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        if i0 >= positions.len() || i1 >= positions.len() || i2 >= positions.len() {
            continue;
        }
        let face = positions[i1]
            .sub(positions[i0])
            .cross(positions[i2].sub(positions[i0]));
        out[i0] = out[i0].add(face);
        out[i1] = out[i1].add(face);
        out[i2] = out[i2].add(face);
    }
    for normal in out.iter_mut() {
        *normal = normal.normalize_or_zero();
    }
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
