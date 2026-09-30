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
//! Guards match [`super::layers::resolve_layer_coupling`]: a non-positive
//! `thickness`/`cell_size` or fewer than two particles yields all-zero
//! corrections; same-layer pairs are skipped (they belong to self-collision);
//! a particle missing a `layer_of` entry never participates; jointly immovable
//! pairs (`w_sum <= 0`) receive nothing; coincident particles separate along
//! `+X`. The call writes only corrections; see
//! [`resolve_layer_coupling_jacobi`] to fold them back into positions.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use super::layers::{cell_of, LayerParams};
use super::{ClothParticle, Vec3, EPS_LEN_SQ};

/// Accumulates each participating particle's Jacobi inter-layer coupling
/// correction into its own slot.
///
/// `out` is resized to `particles.len()` and fully overwritten (cleared to
/// [`Vec3::ZERO`] first); entry `out[a]` is the total position correction
/// particle `a` should receive this pass. `layer_of[i]` is particle `i`'s layer
/// number (lower = inner) and `normals[i]` its outward surface normal; both are
/// parallel to `particles`. All reads are from the *input* `particles` snapshot
/// (never from `out`), so the result is independent of evaluation order and maps
/// directly to a per-invocation `GPU` kernel. See
/// [`resolve_layer_coupling_jacobi`] to fold the result back into positions.
pub(crate) fn accumulate_layer_jacobi_corrections(
    particles: &[ClothParticle],
    layer_of: &[u32],
    normals: &[Vec3],
    params: LayerParams,
    out: &mut Vec<Vec3>,
) {
    out.clear();
    out.resize(particles.len(), Vec3::ZERO);

    let params = params.sanitized();
    if params.thickness <= 0.0 || params.cell_size <= 0.0 || particles.len() < 2 {
        return;
    }

    // Bucket every particle that carries a layer number by its frozen position.
    // Indices are pushed in ascending order, so both the cell traversal and the
    // per-bucket traversal are stable.
    let mut grid: BTreeMap<(i32, i32, i32), Vec<u32>> = BTreeMap::new();
    for (index, particle) in particles.iter().enumerate() {
        if index < layer_of.len() {
            let cell = cell_of(particle.position, params.cell_size);
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
                            acc = acc.add(half_layer_correction(
                                particles,
                                layer_of,
                                normals,
                                ai,
                                bi,
                                params.thickness,
                                thickness_sq,
                            ));
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
/// separated, jointly immovable, or when `ai` is not the endpoint being
/// displaced. Positions come from the frozen `particles` snapshot, so the value
/// is order-independent and matches what a `GPU` invocation for `ai` would
/// compute.
fn half_layer_correction(
    particles: &[ClothParticle],
    layer_of: &[u32],
    normals: &[Vec3],
    ai: usize,
    bi: usize,
    thickness: f32,
    thickness_sq: f32,
) -> Vec3 {
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
        return Vec3::ZERO;
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
            return Vec3::ZERO;
        }
        let penetration = thickness - signed;
        return if ai == inner {
            unit.scale(-penetration * (w_inner / w_sum))
        } else {
            unit.scale(penetration * (w_outer / w_sum))
        };
    }

    // No usable normal: symmetric radial separation.
    let delta = p_outer.sub(p_inner);
    let dist_sq = delta.length_squared();
    if dist_sq >= thickness_sq {
        return Vec3::ZERO;
    }
    let (dir, penetration) = if dist_sq <= EPS_LEN_SQ {
        (Vec3::new(1.0, 0.0, 0.0), thickness)
    } else {
        let dist = dist_sq.sqrt();
        (delta.scale(1.0 / dist), thickness - dist)
    };
    if ai == inner {
        dir.scale(-penetration * (w_inner / w_sum))
    } else {
        dir.scale(penetration * (w_outer / w_sum))
    }
}

/// Runs one Jacobi inter-layer coupling pass, the parallel-safe twin of
/// [`super::layers::resolve_layer_coupling`].
///
/// One call is a single Jacobi iteration (accumulate from the frozen snapshot,
/// then apply); repeated calls converge to the same separated, correctly stacked
/// state the Gauss-Seidel core reaches, without ever depending on evaluation
/// order. `corrections` shorter than `particles` leaves the tail untouched.
pub fn resolve_layer_coupling_jacobi(
    particles: &mut [ClothParticle],
    layer_of: &[u32],
    normals: &[Vec3],
    params: LayerParams,
) {
    let mut corrections = Vec::new();
    accumulate_layer_jacobi_corrections(particles, layer_of, normals, params, &mut corrections);
    let n = particles.len().min(corrections.len());
    for i in 0..n {
        particles[i].position = particles[i].position.add(corrections[i]);
    }
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
            assert!((pa.position.x - pb.position.x).abs() < 1e-6, "x mismatch at {i}");
            assert!((pa.position.y - pb.position.y).abs() < 1e-6, "y mismatch at {i}");
            assert!((pa.position.z - pb.position.z).abs() < 1e-6, "z mismatch at {i}");
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

    /// Non-positive params or fewer than two particles produce zero corrections.
    #[test]
    fn disabled_params_yield_zero_corrections() {
        let particles = [free(0.0, 0.0, 0.0), free(0.0, -0.05, 0.0)];
        let layer_of = [0u32, 1u32];
        let normals = [Vec3::new(0.0, 1.0, 0.0), Vec3::ZERO];
        let mut out = Vec::new();
        accumulate_layer_jacobi_corrections(
            &particles,
            &layer_of,
            &normals,
            LayerParams {
                thickness: 0.0,
                cell_size: 0.2,
            },
            &mut out,
        );
        assert_eq!(out.len(), particles.len());
        assert!(out.iter().all(|c| c.length_squared() == 0.0));
    }

    /// The own-slot accumulation writes both endpoints' halves so that applying
    /// them reproduces the two-sided Gauss-Seidel move for a single contact.
    #[test]
    fn both_endpoints_receive_their_half() {
        let particles = [free(0.0, 0.0, 0.0), free(0.0, -0.05, 0.0)];
        let layer_of = [0u32, 1u32];
        let normals = [Vec3::new(0.0, 1.0, 0.0), Vec3::ZERO];
        let mut out = Vec::new();
        accumulate_layer_jacobi_corrections(&particles, &layer_of, &normals, params(), &mut out);
        // Equal mass: inner pushed down by half the penetration, outer up by half.
        assert!(out[0].y < 0.0, "inner half should be negative, got {}", out[0].y);
        assert!(out[1].y > 0.0, "outer half should be positive, got {}", out[1].y);
        assert!((out[0].y + out[1].y).abs() < 1e-6, "equal-mass halves cancel");
    }
}
