//! Spatial vertex welding and degenerate-triangle cleanup for the `CPU`
//! golden path.
//!
//! Meshes assembled from independent triangles, exported from `DCC` tools, or
//! produced by tessellation often carry duplicate vertices: the same world
//! position stored many times with its own index. Welding collapses coincident
//! (or near-coincident) vertices onto a single representative and rewrites the
//! index buffer to match, shrinking the vertex pool and — crucially — making
//! the surface topologically watertight so downstream smooth-normal and
//! adjacency passes see shared edges rather than cracks.
//!
//! [`weld_vertices`] merges every vertex within a Euclidean `tolerance` of an
//! earlier representative:
//!
//! * `tolerance == 0.0` welds only **bit-identical** positions (after
//!   canonicalizing `-0.0` to `+0.0`), using an exact hash of the packed
//!   coordinate bits — the fast, lossless path for re-indexing split meshes.
//! * `tolerance > 0.0` snaps positions into a uniform spatial-hash grid of
//!   cell size `tolerance` and, for each incoming vertex, scans the surrounding
//!   `3 × 3 × 3` cells for a representative within the true distance. Checking
//!   the neighbourhood (rather than a single cell) avoids the classic grid-snap
//!   artefact where two points closer than `tolerance` straddle a cell boundary
//!   and fail to merge.
//!
//! The first vertex to occupy a representative keeps its attributes
//! (positions, normals, `UV`s), matching the de-facto "weld keeps first"
//! behaviour and avoiding the seam-averaging surprises that blending
//! attributes across a merged `UV` seam would cause. After remapping, any
//! triangle whose three corners no longer reference three distinct vertices has
//! collapsed to zero area and is dropped, and vertices left unreferenced are
//! compacted out so the result carries no dead data.
//!
//! All math is linear plus a single comparison against the squared tolerance,
//! honouring the golden-path ban on `f32` transcendental functions.

use std::collections::HashMap;

use super::triangle_mesh::{TriangleMesh, TriangleMeshError};

/// Errors returned by [`weld_vertices`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WeldError {
    /// The supplied tolerance was negative or not finite.
    InvalidTolerance,
    /// Rebuilding the welded [`TriangleMesh`] failed. Because the welded pools
    /// are internally consistent by construction this does not occur in
    /// practice, but the underlying error is surfaced rather than panicked on.
    Rebuild(TriangleMeshError),
}

impl core::fmt::Display for WeldError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidTolerance => {
                write!(f, "weld tolerance must be a finite, non-negative value")
            }
            Self::Rebuild(err) => write!(f, "failed to rebuild welded mesh: {err}"),
        }
    }
}

impl std::error::Error for WeldError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidTolerance => None,
            Self::Rebuild(err) => Some(err),
        }
    }
}

/// Welds vertices of `mesh` that lie within `tolerance` of each other, rewrites
/// the index buffer, drops triangles that collapse to zero area, and compacts
/// away unreferenced vertices.
///
/// Pass `tolerance == 0.0` for exact (bit-identical) welding. The first vertex
/// mapped to a representative donates its normal and `UV` (if present); mixed
/// normal/`UV` presence across the pool is preserved exactly as the original
/// mesh declared it (an empty pool stays empty).
///
/// # Errors
///
/// Returns [`WeldError::InvalidTolerance`] when `tolerance` is negative or not
/// finite, and [`WeldError::Rebuild`] if the welded pools somehow fail
/// [`TriangleMesh::new`] validation.
pub fn weld_vertices(mesh: &TriangleMesh, tolerance: f32) -> Result<TriangleMesh, WeldError> {
    if !tolerance.is_finite() || tolerance < 0.0 {
        return Err(WeldError::InvalidTolerance);
    }

    let positions = mesh.positions();
    let has_normals = mesh.has_normals();
    let has_uvs = mesh.has_uvs();

    // `remap[v]` is the representative index each original vertex maps to.
    let mut remap = vec![u32::MAX; positions.len()];
    let mut rep_positions: Vec<[f32; 3]> = Vec::new();
    let mut rep_normals: Vec<[f32; 3]> = Vec::new();
    let mut rep_uvs: Vec<[f32; 2]> = Vec::new();

    if tolerance == 0.0 {
        weld_exact(mesh, &mut remap, &mut rep_positions, &mut rep_normals, &mut rep_uvs);
    } else {
        weld_tolerant(
            mesh,
            tolerance,
            &mut remap,
            &mut rep_positions,
            &mut rep_normals,
            &mut rep_uvs,
        );
    }

    // Rewrite the index buffer through the remap, dropping any triangle whose
    // corners no longer reference three distinct representatives.
    let mut welded_indices: Vec<[u32; 3]> = Vec::with_capacity(mesh.indices().len());
    for tri in mesh.indices() {
        let a = remap[tri[0] as usize];
        let b = remap[tri[1] as usize];
        let c = remap[tri[2] as usize];
        if a != b && b != c && a != c {
            welded_indices.push([a, b, c]);
        }
    }

    // Compact out representatives that no triangle references so the result
    // carries no dead vertices.
    compact_unreferenced(
        &mut welded_indices,
        &mut rep_positions,
        &mut rep_normals,
        &mut rep_uvs,
    );

    TriangleMesh::new(
        rep_positions,
        if has_normals { rep_normals } else { Vec::new() },
        if has_uvs { rep_uvs } else { Vec::new() },
        welded_indices,
    )
    .map_err(WeldError::Rebuild)
}

/// Welds only bit-identical positions using an exact hash of the packed
/// coordinate bits (with `-0.0` canonicalized to `+0.0`).
fn weld_exact(
    mesh: &TriangleMesh,
    remap: &mut [u32],
    rep_positions: &mut Vec<[f32; 3]>,
    rep_normals: &mut Vec<[f32; 3]>,
    rep_uvs: &mut Vec<[f32; 2]>,
) {
    let positions = mesh.positions();
    let mut seen: HashMap<[u32; 3], u32> = HashMap::with_capacity(positions.len());
    for (v, &p) in positions.iter().enumerate() {
        let key = [canonical_bits(p[0]), canonical_bits(p[1]), canonical_bits(p[2])];
        let rep = *seen.entry(key).or_insert_with(|| {
            let idx = rep_positions.len() as u32;
            push_representative(mesh, v, rep_positions, rep_normals, rep_uvs);
            idx
        });
        remap[v] = rep;
    }
}

/// Welds vertices within `tolerance` using a `tolerance`-sized spatial-hash
/// grid, scanning the `3 × 3 × 3` neighbourhood so boundary-straddling pairs
/// still merge.
fn weld_tolerant(
    mesh: &TriangleMesh,
    tolerance: f32,
    remap: &mut [u32],
    rep_positions: &mut Vec<[f32; 3]>,
    rep_normals: &mut Vec<[f32; 3]>,
    rep_uvs: &mut Vec<[f32; 2]>,
) {
    let positions = mesh.positions();
    let inv_cell = 1.0 / tolerance;
    let tol2 = tolerance * tolerance;
    // Each grid cell holds the representative indices whose position landed in
    // it, so neighbourhood scans only touch nearby candidates.
    let mut grid: HashMap<[i64; 3], Vec<u32>> = HashMap::new();

    for (v, &p) in positions.iter().enumerate() {
        let cell = cell_of(p, inv_cell);
        let mut found: Option<u32> = None;
        'search: for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let neighbour = [cell[0] + dx, cell[1] + dy, cell[2] + dz];
                    if let Some(bucket) = grid.get(&neighbour) {
                        for &rep in bucket {
                            if dist2(p, rep_positions[rep as usize]) <= tol2 {
                                found = Some(rep);
                                break 'search;
                            }
                        }
                    }
                }
            }
        }

        let rep = match found {
            Some(rep) => rep,
            None => {
                let idx = rep_positions.len() as u32;
                push_representative(mesh, v, rep_positions, rep_normals, rep_uvs);
                grid.entry(cell).or_default().push(idx);
                idx
            }
        };
        remap[v] = rep;
    }
}

/// Appends vertex `v`'s position (and attributes, when present) as a new
/// representative.
fn push_representative(
    mesh: &TriangleMesh,
    v: usize,
    rep_positions: &mut Vec<[f32; 3]>,
    rep_normals: &mut Vec<[f32; 3]>,
    rep_uvs: &mut Vec<[f32; 2]>,
) {
    rep_positions.push(mesh.positions()[v]);
    if mesh.has_normals() {
        rep_normals.push(mesh.normals()[v]);
    }
    if mesh.has_uvs() {
        rep_uvs.push(mesh.uvs()[v]);
    }
}

/// Rewrites `indices` and the attribute pools so only representatives that at
/// least one triangle references survive, preserving their first-use order.
fn compact_unreferenced(
    indices: &mut [[u32; 3]],
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    uvs: &mut Vec<[f32; 2]>,
) {
    let old_len = positions.len();
    let mut old_to_new = vec![u32::MAX; old_len];
    let mut next = 0u32;
    for tri in indices.iter() {
        for &idx in tri {
            if old_to_new[idx as usize] == u32::MAX {
                old_to_new[idx as usize] = next;
                next += 1;
            }
        }
    }

    let new_len = next as usize;
    if new_len == old_len {
        return; // every representative is referenced; nothing to compact.
    }

    let has_normals = !normals.is_empty();
    let has_uvs = !uvs.is_empty();
    let mut new_positions = vec![[0.0f32; 3]; new_len];
    let mut new_normals = if has_normals { vec![[0.0f32; 3]; new_len] } else { Vec::new() };
    let mut new_uvs = if has_uvs { vec![[0.0f32; 2]; new_len] } else { Vec::new() };
    for (old, &new) in old_to_new.iter().enumerate() {
        if new != u32::MAX {
            let slot = new as usize;
            new_positions[slot] = positions[old];
            if has_normals {
                new_normals[slot] = normals[old];
            }
            if has_uvs {
                new_uvs[slot] = uvs[old];
            }
        }
    }

    for tri in indices.iter_mut() {
        tri[0] = old_to_new[tri[0] as usize];
        tri[1] = old_to_new[tri[1] as usize];
        tri[2] = old_to_new[tri[2] as usize];
    }

    *positions = new_positions;
    *normals = new_normals;
    *uvs = new_uvs;
}

/// Packs a coordinate's bits for exact hashing, mapping `-0.0` to `+0.0` so the
/// two zero encodings weld together.
fn canonical_bits(x: f32) -> u32 {
    if x == 0.0 { 0.0f32.to_bits() } else { x.to_bits() }
}

/// The integer grid cell a position occupies for a given inverse cell size.
fn cell_of(p: [f32; 3], inv_cell: f32) -> [i64; 3] {
    [
        (p[0] * inv_cell).floor() as i64,
        (p[1] * inv_cell).floor() as i64,
        (p[2] * inv_cell).floor() as i64,
    ]
}

/// Squared Euclidean distance between two points.
fn dist2(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    dx * dx + dy * dy + dz * dz
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two triangles forming a quad but authored as six independent vertices
    /// (the shared diagonal is duplicated), with no attributes.
    fn split_quad() -> TriangleMesh {
        let positions = vec![
            [0.0, 0.0, 0.0], // tri 0
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0], // tri 1 (dupes 0)
            [1.0, 1.0, 0.0], // (dupes 2)
            [0.0, 1.0, 0.0],
        ];
        let indices = vec![[0, 1, 2], [3, 4, 5]];
        TriangleMesh::new(positions, vec![], vec![], indices).expect("valid split quad")
    }

    #[test]
    fn exact_weld_merges_duplicate_vertices() {
        let welded = weld_vertices(&split_quad(), 0.0).expect("weld");
        // Six authored vertices collapse to the four distinct corners.
        assert_eq!(welded.vertex_count(), 4);
        // Both triangles survive (still three distinct corners each).
        assert_eq!(welded.triangle_count(), 2);
    }

    #[test]
    fn exact_weld_preserves_distinct_positions() {
        let welded = weld_vertices(&split_quad(), 0.0).expect("weld");
        let mut seen = welded.positions().to_vec();
        seen.sort_by(|a, b| a.partial_cmp(b).unwrap());
        seen.dedup();
        assert_eq!(seen.len(), 4, "all kept positions must be distinct");
    }

    #[test]
    fn exact_weld_leaves_unique_mesh_untouched() {
        let positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let mesh = TriangleMesh::new(positions, vec![], vec![], vec![[0, 1, 2]]).expect("mesh");
        let welded = weld_vertices(&mesh, 0.0).expect("weld");
        assert_eq!(welded.vertex_count(), 3);
        assert_eq!(welded.triangle_count(), 1);
    }

    #[test]
    fn negative_zero_welds_with_positive_zero() {
        let positions = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-0.0, -0.0, -0.0], // bitwise different from vertex 0 but same point
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let indices = vec![[0, 1, 2], [3, 4, 5]];
        let mesh = TriangleMesh::new(positions, vec![], vec![], indices).expect("mesh");
        let welded = weld_vertices(&mesh, 0.0).expect("weld");
        assert_eq!(welded.vertex_count(), 3, "±0 must weld together");
    }

    #[test]
    fn tolerant_weld_merges_near_coincident() {
        // Vertex 3 sits 1e-4 from vertex 0 — within a 1e-3 tolerance.
        let positions = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1e-4, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let indices = vec![[0, 1, 2], [3, 4, 5]];
        let mesh = TriangleMesh::new(positions, vec![], vec![], indices).expect("mesh");
        let welded = weld_vertices(&mesh, 1e-3).expect("weld");
        assert_eq!(welded.vertex_count(), 3, "near-coincident vertices merge");
    }

    #[test]
    fn tolerant_weld_keeps_vertices_outside_tolerance() {
        let positions = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.0, 0.0], // well outside a 1e-3 tolerance of any corner
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let indices = vec![[0, 1, 2], [3, 4, 5]];
        let mesh = TriangleMesh::new(positions, vec![], vec![], indices).expect("mesh");
        let welded = weld_vertices(&mesh, 1e-3).expect("weld");
        assert_eq!(welded.vertex_count(), 4, "distant vertex is not merged");
    }

    #[test]
    fn tolerant_weld_merges_across_cell_boundary() {
        // Two points 1e-5 apart straddling the x=0 cell boundary of a 1e-3 grid
        // (cells floor(x*1000)): -5e-6 → cell -1, +5e-6 → cell 0. A single-cell
        // snap would miss them; the 3×3×3 scan merges them.
        let positions = vec![
            [-5e-6, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [5e-6, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let indices = vec![[0, 1, 2], [3, 4, 5]];
        let mesh = TriangleMesh::new(positions, vec![], vec![], indices).expect("mesh");
        let welded = weld_vertices(&mesh, 1e-3).expect("weld");
        assert_eq!(welded.vertex_count(), 3, "boundary-straddling pair must weld");
    }

    #[test]
    fn degenerate_triangle_is_dropped() {
        // Welding collapses two corners of the second triangle onto one point,
        // so it becomes zero-area and must be removed.
        let positions = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [2.0, 2.0, 2.0],
            [2.0, 2.0, 2.0], // identical to vertex 3 → tri 1 collapses
            [3.0, 3.0, 3.0],
        ];
        let indices = vec![[0, 1, 2], [3, 4, 5]];
        let mesh = TriangleMesh::new(positions, vec![], vec![], indices).expect("mesh");
        let welded = weld_vertices(&mesh, 0.0).expect("weld");
        assert_eq!(welded.triangle_count(), 1, "collapsed triangle dropped");
    }

    #[test]
    fn unreferenced_vertices_are_compacted() {
        // Dropping the degenerate triangle leaves vertex 5 unreferenced; it must
        // not survive in the compacted pool.
        let positions = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [2.0, 2.0, 2.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0],
        ];
        let indices = vec![[0, 1, 2], [3, 4, 5]];
        let mesh = TriangleMesh::new(positions, vec![], vec![], indices).expect("mesh");
        let welded = weld_vertices(&mesh, 0.0).expect("weld");
        assert_eq!(welded.vertex_count(), 3, "only the surviving triangle's corners remain");
    }

    #[test]
    fn attributes_follow_the_first_representative() {
        let positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]];
        let normals = vec![
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0], // dupe of vertex 0 with a different normal
        ];
        let uvs = vec![[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.9, 0.9]];
        let indices = vec![[0, 1, 2], [3, 1, 2]];
        let mesh = TriangleMesh::new(positions, normals, uvs, indices).expect("mesh");
        let welded = weld_vertices(&mesh, 0.0).expect("weld");
        assert_eq!(welded.vertex_count(), 3);
        // Vertex 3 welds onto vertex 0, whose normal/UV (first seen) are kept.
        assert_eq!(welded.normals()[0], [0.0, 0.0, 1.0]);
        assert_eq!(welded.uvs()[0], [0.1, 0.2]);
    }

    #[test]
    fn empty_attribute_pools_stay_empty() {
        let welded = weld_vertices(&split_quad(), 0.0).expect("weld");
        assert!(!welded.has_normals());
        assert!(!welded.has_uvs());
    }

    #[test]
    fn invalid_tolerance_is_rejected() {
        assert_eq!(weld_vertices(&split_quad(), -1.0), Err(WeldError::InvalidTolerance));
        assert_eq!(weld_vertices(&split_quad(), f32::NAN), Err(WeldError::InvalidTolerance));
        assert_eq!(
            weld_vertices(&split_quad(), f32::INFINITY),
            Err(WeldError::InvalidTolerance)
        );
    }
}
