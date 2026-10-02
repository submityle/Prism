//! Jacobi (parallel-safe) inter-layer garment coupling — the `GPU`-faithful golden.
//!
//! [`super::resolve_layer_coupling`] keeps stacked garments from
//! interpenetrating in **Gauss-Seidel** order: it walks the cross-layer sample
//! pairs in a fixed cell order and each contact scatters its correction onto the
//! two particles *in place*, so a later pair already sees the moved positions of
//! an earlier one. That is the sequential `CPU` reference, but it does not map to
//! a `GPU` compute kernel: on the `GPU` every particle is updated in parallel from
//! the *same* read-only snapshot, which is a **Jacobi** iteration, not
//! Gauss-Seidel. The two converge to the same separated, correctly stacked state
//! but are never bit-for-bit identical on a multi-contact pass, so a faithful
//! `WGSL` twin needs its own golden rather than borrowing the Gauss-Seidel one
//! (no fake parity). This mirrors the sibling
//! [`super::resolve_self_collision_jacobi`] contract that owns the point
//! self-collision tier's `GPU` golden.
//!
//! [`accumulate_layer_jacobi_corrections`] is the own-slot core: one invocation
//! owns `out[a]` for particle `a`, gathers its 27-cell neighborhood from a
//! prebuilt uniform spatial hash, and sums *only `a`'s own half* of the
//! separating push against every penetrating cross-layer neighbor `b != a`,
//! reading positions only from the frozen input snapshot. It never reads another
//! slot, so it is exactly the body of a per-particle `GPU` kernel with no
//! atomics. The half for particle `a` is oriented by the *inner* (lower-numbered)
//! layer particle's outward normal so the outer layer is driven to the `+normal`
//! side at least `thickness` away; when that normal is unavailable (zero length)
//! it falls back to the symmetric radial minimum-distance push. The neighborhood
//! reduces in strictly ascending index order (ascending [`BTreeMap`] cells,
//! ascending bucket indices), so the float reduction is deterministic and the
//! `GPU` twin can mirror the result value-for-value.
//!
//! Guards match [`super::resolve_layer_coupling`]: a non-positive
//! `thickness`/`cell_size`, a mismatched `inverse_masses` length, or fewer than
//! two particles yields all-zero corrections; same-layer pairs are skipped (they
//! belong to self-collision); a particle missing a `layer_of` entry never
//! participates; jointly immovable pairs (`w_sum <= 0`) receive nothing;
//! coincident particles separate along `+X`.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use glam::Vec3;

use super::layers::LayerParams;
use super::{cell_of, EPS_LEN_SQ};
use crate::math::scalar::Real;

/// Accumulates each participating particle's Jacobi inter-layer coupling
/// correction into its own slot.
///
/// `out` is resized to `positions.len()` and fully overwritten (cleared to
/// [`Vec3::ZERO`] first); entry `out[a]` is the total position correction
/// particle `a` should receive this pass. `layer_of[i]` is particle `i`'s layer
/// number (lower = inner) and `normals[i]` its outward surface normal; both are
/// parallel to `positions`. All reads are from the *input* `positions` snapshot
/// (never from `out`), so the result is independent of evaluation order and maps
/// directly to a per-invocation `GPU` kernel.
pub(crate) fn accumulate_layer_jacobi_corrections(
    positions: &[Vec3],
    inverse_masses: &[Real],
    layer_of: &[u32],
    normals: &[Vec3],
    params: LayerParams,
    out: &mut Vec<Vec3>,
) {
    out.clear();
    out.resize(positions.len(), Vec3::ZERO);

    let params = params.sanitized();
    if params.thickness <= 0.0
        || params.cell_size <= 0.0
        || positions.len() < 2
        || inverse_masses.len() != positions.len()
    {
        return;
    }

    // Bucket every particle that carries a layer number by its frozen position.
    // Indices are pushed in ascending order, so both the cell traversal and the
    // per-bucket traversal are stable.
    let mut grid: BTreeMap<(i32, i32, i32), Vec<u32>> = BTreeMap::new();
    for (index, &pos) in positions.iter().enumerate() {
        if index < layer_of.len() {
            let cell = cell_of(pos, params.cell_size);
            grid.entry(cell).or_default().push(index as u32);
        }
    }

    // Each participating particle owns `out[a]` and sums only its own half of
    // every cross-layer penetrating pair, reading the frozen snapshot. This is
    // the body of a per-particle GPU kernel: it writes one slot and reads none.
    let thickness_sq = params.thickness * params.thickness;
    for (&cell, bucket) in &grid {
        for &a in bucket {
            let ai = a as usize;
            let mut acc = Vec3::ZERO;
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        let neighbor = (cell.0 + dx, cell.1 + dy, cell.2 + dz);
                        let Some(nbucket) = grid.get(&neighbor) else {
                            continue;
                        };
                        for &b in nbucket {
                            if b == a {
                                continue;
                            }
                            let bi = b as usize;
                            // Same-layer contacts belong to self-collision.
                            if layer_of[ai] == layer_of[bi] {
                                continue;
                            }
                            acc += half_layer_correction(
                                positions,
                                inverse_masses,
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
            out[ai] = acc;
        }
    }
}

/// Particle `ai`'s own half of the cross-layer separating push against neighbor
/// `bi`.
///
/// The lower layer number is the inner surface whose outward normal orients the
/// contact, so the outer particle is driven to at least `thickness` along that
/// normal. Falls back to a symmetric radial minimum-distance push when the inner
/// normal is (near) zero. Returns [`Vec3::ZERO`] when the pair is already
/// separated or jointly immovable. Positions come from the frozen `positions`
/// snapshot, so the value is order-independent and matches what a `GPU`
/// invocation for `ai` would compute.
#[expect(
    clippy::too_many_arguments,
    reason = "raw-column contact kernel takes its state as explicit index-aligned slices"
)]
fn half_layer_correction(
    positions: &[Vec3],
    inverse_masses: &[Real],
    layer_of: &[u32],
    normals: &[Vec3],
    ai: usize,
    bi: usize,
    thickness: Real,
    thickness_sq: Real,
) -> Vec3 {
    // Lower layer number is the inner surface whose normal orients the contact.
    let (inner, outer) = if layer_of[ai] < layer_of[bi] {
        (ai, bi)
    } else {
        (bi, ai)
    };

    let w_inner = inverse_masses[inner].max(0.0);
    let w_outer = inverse_masses[outer].max(0.0);
    let w_sum = w_inner + w_outer;
    if w_sum <= 0.0 {
        return Vec3::ZERO;
    }

    let p_inner = positions[inner];
    let p_outer = positions[outer];
    let normal = normals.get(inner).copied().unwrap_or(Vec3::ZERO);
    let unit = normal.normalize_or_zero();

    if unit.length_squared() > EPS_LEN_SQ {
        // Oriented plane contact: force the outer particle to at least
        // `thickness` along the inner's outward normal.
        let signed = (p_outer - p_inner).dot(unit);
        if signed >= thickness {
            return Vec3::ZERO;
        }
        let penetration = thickness - signed;
        return if ai == inner {
            unit * (-penetration * (w_inner / w_sum))
        } else {
            unit * (penetration * (w_outer / w_sum))
        };
    }

    // No usable normal: symmetric radial separation.
    let delta = p_outer - p_inner;
    let dist_sq = delta.length_squared();
    if dist_sq >= thickness_sq {
        return Vec3::ZERO;
    }
    let (dir, penetration) = if dist_sq <= EPS_LEN_SQ {
        (Vec3::new(1.0, 0.0, 0.0), thickness)
    } else {
        let dist = dist_sq.sqrt();
        (delta * (1.0 / dist), thickness - dist)
    };
    if ai == inner {
        dir * (-penetration * (w_inner / w_sum))
    } else {
        dir * (penetration * (w_outer / w_sum))
    }
}

/// Runs one Jacobi inter-layer coupling pass in place — the parallel-safe twin
/// of [`super::resolve_layer_coupling`].
///
/// One call is a single Jacobi iteration (accumulate from the frozen snapshot,
/// then apply); repeated calls converge to the same separated, correctly stacked
/// state the Gauss-Seidel core reaches, without ever depending on evaluation
/// order. A single isolated contact resolves bit-for-bit identically to the
/// Gauss-Seidel core in one pass.
pub fn resolve_layer_coupling_jacobi(
    positions: &mut [Vec3],
    inverse_masses: &[Real],
    layer_of: &[u32],
    normals: &[Vec3],
    params: LayerParams,
) {
    let mut corrections = Vec::new();
    accumulate_layer_jacobi_corrections(
        positions,
        inverse_masses,
        layer_of,
        normals,
        params,
        &mut corrections,
    );
    for (pos, corr) in positions.iter_mut().zip(corrections.iter()) {
        *pos += *corr;
    }
}

#[cfg(test)]
mod tests {
    use super::super::resolve_layer_coupling;
    use super::*;

    fn params() -> LayerParams {
        LayerParams {
            thickness: 0.1,
            cell_size: 0.2,
        }
    }

    fn assert_positions_eq(a: &[Vec3], b: &[Vec3]) {
        assert_eq!(a.len(), b.len());
        for (i, (pa, pb)) in a.iter().zip(b.iter()).enumerate() {
            assert!((pa.x - pb.x).abs() < 1e-6, "x mismatch at {i}");
            assert!((pa.y - pb.y).abs() < 1e-6, "y mismatch at {i}");
            assert!((pa.z - pb.z).abs() < 1e-6, "z mismatch at {i}");
        }
    }

    #[test]
    fn single_oriented_contact_matches_gauss_seidel() {
        let base = [Vec3::ZERO, Vec3::new(0.0, -0.05, 0.0)];
        let inv = [1.0, 1.0];
        let layer_of = [0u32, 1u32];
        let normals = [Vec3::new(0.0, 1.0, 0.0), Vec3::ZERO];

        let mut gs = base;
        resolve_layer_coupling(&mut gs, &inv, &layer_of, &normals, params());
        let mut jac = base;
        resolve_layer_coupling_jacobi(&mut jac, &inv, &layer_of, &normals, params());
        assert_positions_eq(&gs, &jac);

        let signed = (jac[1] - jac[0]).y;
        assert!(signed >= 0.1 - 1e-6, "signed separation {signed}");
    }

    #[test]
    fn single_radial_contact_matches_gauss_seidel() {
        let base = [Vec3::ZERO, Vec3::new(0.02, 0.0, 0.0)];
        let inv = [1.0, 1.0];
        let layer_of = [0u32, 1u32];
        let normals = [Vec3::ZERO, Vec3::ZERO];
        let mut gs = base;
        resolve_layer_coupling(&mut gs, &inv, &layer_of, &normals, params());
        let mut jac = base;
        resolve_layer_coupling_jacobi(&mut jac, &inv, &layer_of, &normals, params());
        assert_positions_eq(&gs, &jac);
        let dist = (jac[1] - jac[0]).length();
        assert!(dist >= 0.1 - 1e-6, "radial separation {dist}");
    }

    #[test]
    fn pinned_inner_matches_gauss_seidel() {
        let base = [Vec3::ZERO, Vec3::new(0.0, -0.05, 0.0)];
        let inv = [0.0, 1.0];
        let layer_of = [0u32, 1u32];
        let normals = [Vec3::new(0.0, 1.0, 0.0), Vec3::ZERO];
        let mut gs = base;
        resolve_layer_coupling(&mut gs, &inv, &layer_of, &normals, params());
        let mut jac = base;
        resolve_layer_coupling_jacobi(&mut jac, &inv, &layer_of, &normals, params());
        assert_positions_eq(&gs, &jac);
        assert!((jac[0].y - 0.0).abs() < 1e-9, "inner pinned moved");
    }

    #[test]
    fn same_layer_pair_is_untouched() {
        let base = [Vec3::ZERO, Vec3::new(0.0, 0.01, 0.0)];
        let inv = [1.0, 1.0];
        let layer_of = [2u32, 2u32];
        let normals = [Vec3::new(0.0, 1.0, 0.0), Vec3::ZERO];
        let mut jac = base;
        resolve_layer_coupling_jacobi(&mut jac, &inv, &layer_of, &normals, params());
        assert_positions_eq(&base, &jac);
    }

    #[test]
    fn separated_pair_is_untouched() {
        let base = [Vec3::ZERO, Vec3::new(0.0, 0.5, 0.0)];
        let inv = [1.0, 1.0];
        let layer_of = [0u32, 1u32];
        let normals = [Vec3::new(0.0, 1.0, 0.0), Vec3::ZERO];
        let mut jac = base;
        resolve_layer_coupling_jacobi(&mut jac, &inv, &layer_of, &normals, params());
        assert_positions_eq(&base, &jac);
    }

    #[test]
    fn disabled_params_yield_zero_corrections() {
        let positions = [Vec3::ZERO, Vec3::new(0.0, -0.05, 0.0)];
        let inv = [1.0, 1.0];
        let layer_of = [0u32, 1u32];
        let normals = [Vec3::new(0.0, 1.0, 0.0), Vec3::ZERO];
        let mut out = Vec::new();
        accumulate_layer_jacobi_corrections(
            &positions,
            &inv,
            &layer_of,
            &normals,
            LayerParams {
                thickness: 0.0,
                cell_size: 0.2,
            },
            &mut out,
        );
        assert_eq!(out.len(), positions.len());
        assert!(out.iter().all(|c| c.length_squared() == 0.0));
    }

    #[test]
    fn both_endpoints_receive_their_half() {
        let positions = [Vec3::ZERO, Vec3::new(0.0, -0.05, 0.0)];
        let inv = [1.0, 1.0];
        let layer_of = [0u32, 1u32];
        let normals = [Vec3::new(0.0, 1.0, 0.0), Vec3::ZERO];
        let mut out = Vec::new();
        accumulate_layer_jacobi_corrections(&positions, &inv, &layer_of, &normals, params(), &mut out);
        assert!(out[0].y < 0.0, "inner half should be negative, got {}", out[0].y);
        assert!(out[1].y > 0.0, "outer half should be positive, got {}", out[1].y);
        assert!((out[0].y + out[1].y).abs() < 1e-6, "equal-mass halves cancel");
    }

    #[test]
    fn iterated_jacobi_converges_towards_gauss_seidel_cluster() {
        let build = || {
            [
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(0.0, -0.03, 0.0),
                Vec3::new(0.01, -0.02, 0.0),
            ]
        };
        let inv = [1.0, 1.0, 1.0];
        let layer_of = [0u32, 1u32, 1u32];
        let normals = [
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let mut gs = build();
        resolve_layer_coupling(&mut gs, &inv, &layer_of, &normals, params());
        let mut jac = build();
        for _ in 0..64 {
            resolve_layer_coupling_jacobi(&mut jac, &inv, &layer_of, &normals, params());
        }
        // Every cross-layer pair ends at least `thickness` apart in both.
        for p in [&gs, &jac] {
            assert!((p[1] - p[0]).length() >= 0.1 - 1e-3);
            assert!((p[2] - p[0]).length() >= 0.1 - 1e-3);
        }
    }
}
