//! Static topology tables for the marching-tetrahedra iso-surface extractor.
//!
//! Instead of the 256-case Marching Cubes table (whose ambiguous face
//! configurations can produce cracks and non-manifold edges between adjacent
//! cubes), Prism subdivides every grid cell into six tetrahedra using the
//! Freudenthal–Kuhn decomposition and marches each tetrahedron. Because the
//! Kuhn decomposition tiles space as a single consistent simplicial complex —
//! every cell splits its shared faces along the same diagonal — the extracted
//! surface is guaranteed watertight and edge-manifold. The tetrahedron case
//! table has only 16 entries and is small enough to be obviously correct.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. The
//! marching-tetrahedra decomposition and case handling follow Doi & Koide 1991
//! and Bourke, "Polygonising a scalar field using tetrahedrons"; the
//! space-filling cube subdivision is the classical Freudenthal–Kuhn
//! triangulation.

/// The eight cube-corner offsets, indexed `0..8` by `(dx, dy, dz)` bits.
///
/// Corner `c` has offset `(c & 1, (c >> 1) & 1, (c >> 2) & 1)` — this matches
/// the ordering used by [`CUBE_TETRAHEDRA`].
pub const CUBE_CORNERS: [[u32; 3]; 8] = [
    [0, 0, 0], // 0
    [1, 0, 0], // 1
    [0, 1, 0], // 2
    [1, 1, 0], // 3
    [0, 0, 1], // 4
    [1, 0, 1], // 5
    [0, 1, 1], // 6
    [1, 1, 1], // 7
];

/// The six tetrahedra of the Freudenthal–Kuhn decomposition, as indices into
/// [`CUBE_CORNERS`]. All six share the main diagonal `0 → 7`, and every one is
/// a monotone path that adds the axes in a fixed order, so neighbouring cells
/// agree on how each shared face is split.
pub const CUBE_TETRAHEDRA: [[usize; 4]; 6] = [
    [0, 1, 3, 7], // +x, +y, +z
    [0, 1, 5, 7], // +x, +z, +y
    [0, 2, 3, 7], // +y, +x, +z
    [0, 2, 6, 7], // +y, +z, +x
    [0, 4, 5, 7], // +z, +x, +y
    [0, 4, 6, 7], // +z, +y, +x
];

/// The six edges of a tetrahedron as pairs of local vertex indices `0..4`.
/// Edge `e` connects `TET_EDGES[e][0]` and `TET_EDGES[e][1]`.
pub const TET_EDGES: [[usize; 2]; 6] = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]];

/// Triangulation of a tetrahedron indexed by the 4-bit inside-mask (bit `v` set
/// when vertex `v` is inside the iso-surface). Each entry lists triangles as
/// triples of *edge* indices into [`TET_EDGES`]; `-1` terminates the list.
///
/// Winding is chosen so the triangle normal points toward the outside
/// (decreasing field). Cases with two vertices inside emit a quad as two
/// triangles sharing a diagonal edge, keeping the surface edge-manifold.
pub const TET_TRIANGLES: [[i8; 7]; 16] = [
    [-1, -1, -1, -1, -1, -1, -1], // 0000 : empty
    [0, 1, 2, -1, -1, -1, -1],    // 0001 : v0 inside
    [0, 3, 4, -1, -1, -1, -1],    // 0010 : v1 inside
    [1, 2, 4, 1, 4, 3, -1],       // 0011 : v0,v1 inside
    [1, 5, 3, -1, -1, -1, -1],    // 0100 : v2 inside
    [0, 5, 3, 0, 2, 5, -1],       // 0101 : v0,v2 inside
    [0, 1, 5, 0, 5, 4, -1],       // 0110 : v1,v2 inside
    [2, 5, 4, -1, -1, -1, -1],    // 0111 : v0,v1,v2 inside (v3 out)
    [2, 4, 5, -1, -1, -1, -1],    // 1000 : v3 inside
    [0, 4, 5, 0, 5, 1, -1],       // 1001 : v0,v3 inside
    [0, 3, 5, 0, 5, 2, -1],       // 1010 : v1,v3 inside
    [1, 3, 5, -1, -1, -1, -1],    // 1011 : v0,v1,v3 inside (v2 out)
    [1, 3, 4, 1, 4, 2, -1],       // 1100 : v2,v3 inside
    [0, 3, 4, -1, -1, -1, -1],    // 1101 : v0,v2,v3 inside (v1 out)
    [0, 2, 1, -1, -1, -1, -1],    // 1110 : v1,v2,v3 inside (v0 out)
    [-1, -1, -1, -1, -1, -1, -1], // 1111 : full
];
