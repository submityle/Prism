//! 3D Marching Cubes iso-surface extraction (Lorensen and Cline 1987) on a
//! scalar voxel field, for the particle geometry contracts (design 8.2, `CPU`).
//!
//! Several particle stages carry a *3D scalar field* sampled on a regular
//! voxel grid: a signed-distance volume for a collision proxy, a baked density
//! or temperature volume, or a metaball accumulation. They need the
//! triangulated **iso-surface** at a threshold: the closed mesh where the field
//! equals `iso`. This module owns exactly that contract. Given a row-major
//! `nx by ny by nz` sample grid and a threshold, it walks every voxel cube,
//! classifies its eight corners against `iso` into an 8-bit case, looks the
//! case up in the built-in standard 256-entry edge and triangle tables, and
//! linearly interpolates a vertex on each cube edge that straddles `iso`. It is
//! the textbook Marching Cubes algorithm: an integer case table plus a single
//! per-edge linear interpolation.
//!
//! The corner classification follows the classic convention: corner bit `i` is
//! set when its sample value is strictly below `iso`, so a corner whose value
//! equals `iso` is on the *inside* (at-or-above) side. The shipped
//! [`EDGE_TABLE`] is the union of the cube edges referenced by [`TRI_TABLE`],
//! so the two tables are consistent by construction and the extractor only ever
//! interpolates edges that a triangle actually uses.
//!
//! # Boundary with sibling modules
//! This is a strictly 3D, voxel-grid-in / triangle-mesh-out surface extractor.
//! It is *not*:
//! - `marching_squares`, the 2D sibling, which walks a plane of cells and emits
//!   interpolated iso-*contour* line segments rather than a triangulated
//!   surface; this module is its 3D, triangle-emitting analogue and shares no
//!   code with it;
//! - `volume_march`, which ray-marches a volume front-to-back and composites
//!   opacity and transmittance: a stepping and compositing loop, not a
//!   geometric iso-surface;
//! - `gaussian_splat`, which rasterizes anisotropic radial kernels as screen
//!   footprints and never extracts a mesh.
//!
//! # No transcendental math
//! Every routine is pure `+`, `-`, `*`, `/`, comparison, `f32::abs`, and
//! `f32::clamp`; the only root is `f32::sqrt`, used solely by the test helpers
//! that build a sphere field, never on the extraction path. There is no `sin`,
//! `cos`, `exp`, `powf`, `floor`, or `ceil` on the surface path: cube iteration
//! is by integer index and corner classification is a `<` test, so no rounding
//! is required. Floats are never compared with `==` or `!=`; a corner is
//! *below* when its value is strictly `< iso`, and an edge whose two corner
//! values are within [`EPS`] of each other is crossed at its midpoint rather
//! than dividing by a vanishing denominator.

use crate::particle::Vec3;
use alloc::vec::Vec;

/// Magnitude below which an edge's corner-value difference is treated as zero.
///
/// When the two corner values along an edge differ by less than this, the
/// linear crossing parameter is ill-conditioned (a near `0/0`), so the crossing
/// is placed at the edge midpoint instead of dividing by a vanishing
/// denominator. Floats are otherwise never compared with `==` or `!=`.
pub const EPS: f32 = 1.0e-6;

/// Cube-local integer offsets `(dx, dy, dz)` of the eight cube corners, in the
/// standard Marching Cubes corner order.
///
/// Corner `c` of the voxel cube whose minimum sample is `(x, y, z)` reads the
/// grid sample at `(x + dx, y + dy, z + dz)` and sits at that position in grid
/// coordinates. The z-plane at the cube minimum holds corners `0, 1, 4, 5`; the
/// far z-plane holds corners `2, 3, 6, 7`.
pub const CORNER_OFFSET: [[usize; 3]; 8] = [
    [0, 0, 0],
    [1, 0, 0],
    [1, 0, 1],
    [0, 0, 1],
    [0, 1, 0],
    [1, 1, 0],
    [1, 1, 1],
    [0, 1, 1],
];

/// The two corner indices joined by each of the twelve cube edges, in the
/// standard Marching Cubes edge order.
///
/// Edge `e` connects corners `EDGE_CORNERS[e][0]` and `EDGE_CORNERS[e][1]`; a
/// vertex is emitted on that edge when exactly one of its endpoints is below
/// `iso`.
pub const EDGE_CORNERS: [[usize; 2]; 12] = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 0],
    [4, 5],
    [5, 6],
    [6, 7],
    [7, 4],
    [0, 4],
    [1, 5],
    [2, 6],
    [3, 7],
];

/// Standard Marching Cubes edge table: for each of the 256 corner cases, a
/// 12-bit mask of the cube edges that straddle `iso` (bit `e` set means edge
/// `e` carries a surface vertex).
///
/// This is the union of the edges referenced by [`TRI_TABLE`] for the same
/// case, so it is exactly the set of edges the triangulation needs and can
/// never disagree with it.
pub const EDGE_TABLE: [u16; 256] = [
    0x000, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c, 0x80c, 0x905, 0xa0f, 0xb06, 0xc0a,
    0xd03, 0xe09, 0xf00, 0x190, 0x099, 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c, 0x99c, 0x895,
    0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90, 0x230, 0x339, 0x033, 0x13a, 0x636, 0x73f, 0x435,
    0x53c, 0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30, 0x3a0, 0x2a9, 0x1a3, 0x0aa,
    0x7a6, 0x6af, 0x5a5, 0x4ac, 0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0, 0x460,
    0x569, 0x663, 0x76a, 0x066, 0x16f, 0x265, 0x36c, 0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963,
    0xa69, 0xb60, 0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0x0ff, 0x3f5, 0x2fc, 0xdfc, 0xcf5, 0xfff,
    0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0, 0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x055, 0x15c,
    0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950, 0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6,
    0x2cf, 0x1c5, 0x0cc, 0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0, 0x8c0, 0x9c9,
    0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc, 0x0cc, 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9,
    0x7c0, 0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c, 0x15c, 0x055, 0x35f, 0x256,
    0x55a, 0x453, 0x759, 0x650, 0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc, 0x2fc,
    0x3f5, 0x0ff, 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0, 0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f,
    0xd65, 0xc6c, 0x36c, 0x265, 0x16f, 0x066, 0x76a, 0x663, 0x569, 0x460, 0xca0, 0xda9, 0xea3,
    0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac, 0x4ac, 0x5a5, 0x6af, 0x7a6, 0x0aa, 0x1a3, 0x2a9, 0x3a0,
    0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c, 0x53c, 0x435, 0x73f, 0x636, 0x13a,
    0x033, 0x339, 0x230, 0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c, 0x69c, 0x795,
    0x49f, 0x596, 0x29a, 0x393, 0x099, 0x190, 0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905,
    0x80c, 0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x000,
];

/// Standard Marching Cubes triangle table: for each of the 256 corner cases, up
/// to five triangles given as edge indices in groups of three, terminated by
/// `-1`.
///
/// Each entry is an edge index in `0..12` naming the [`EDGE_CORNERS`] edge that
/// carries the triangle vertex; a `-1` marks the end of the triangle list for
/// that case, and all remaining slots in the row are `-1`.
pub const TRI_TABLE: [[i8; 16]; 256] = [
    [
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    ],
    [0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1],
    [3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1],
    [3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1],
    [3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1],
    [9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1],
    [9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1],
    [2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1],
    [8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1],
    [9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1],
    [4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1],
    [3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1],
    [1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1],
    [4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1],
    [4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1],
    [9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1],
    [5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1],
    [2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1],
    [9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1],
    [0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1],
    [2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1],
    [10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1],
    [4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1],
    [5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1],
    [5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1],
    [9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1],
    [0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1],
    [1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1],
    [10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1],
    [8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1],
    [2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1],
    [7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1],
    [9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1],
    [2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1],
    [11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1],
    [9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1],
    [5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1],
    [11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1],
    [11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1],
    [1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1],
    [9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1],
    [5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1],
    [2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1],
    [0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1],
    [5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1],
    [6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1],
    [3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1],
    [6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1],
    [5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1],
    [1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1],
    [10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1],
    [6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1],
    [8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1],
    [7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1],
    [3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1],
    [5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1],
    [0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1],
    [9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1],
    [8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1],
    [5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1],
    [0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1],
    [6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1],
    [10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1],
    [10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1],
    [8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1],
    [1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1],
    [3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1],
    [0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1],
    [10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1],
    [3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1],
    [6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1],
    [9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1],
    [8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1],
    [3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1],
    [6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1],
    [0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1],
    [10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1],
    [10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1],
    [2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1],
    [7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1],
    [7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1],
    [2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1],
    [1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1],
    [11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1],
    [8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1],
    [0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1],
    [7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1],
    [10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1],
    [2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1],
    [6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1],
    [7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1],
    [2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1],
    [1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1],
    [10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1],
    [10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1],
    [0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1],
    [7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1],
    [6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1],
    [8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1],
    [9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1],
    [6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1],
    [4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1],
    [10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1],
    [8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1],
    [0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1],
    [1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1],
    [8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1],
    [10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1],
    [4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1],
    [10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1],
    [5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1],
    [11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1],
    [9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1],
    [6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1],
    [7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1],
    [3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1],
    [7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1],
    [9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1],
    [3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1],
    [6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1],
    [9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1],
    [1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1],
    [4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1],
    [7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1],
    [6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1],
    [3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1],
    [0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1],
    [6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1],
    [0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1],
    [11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1],
    [6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1],
    [5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1],
    [9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1],
    [1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1],
    [1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1],
    [10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1],
    [0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1],
    [5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1],
    [10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1],
    [11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1],
    [9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1],
    [7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1],
    [2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1],
    [8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1],
    [9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1],
    [9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1],
    [1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1],
    [9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1],
    [9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1],
    [5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1],
    [0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1],
    [10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1],
    [2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1],
    [0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1],
    [0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1],
    [9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1],
    [5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1],
    [3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1],
    [5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1],
    [8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1],
    [0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1],
    [9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1],
    [1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1],
    [3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1],
    [4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1],
    [9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1],
    [11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1],
    [11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1],
    [2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1],
    [9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1],
    [3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1],
    [1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1],
    [4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1],
    [4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1],
    [0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1],
    [3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1],
    [3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1],
    [0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1],
    [9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1],
    [1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    ],
];

/// A scalar field sampled on a regular `nx by ny by nz` voxel grid, stored
/// row-major with the x index varying fastest.
///
/// The sample at integer grid coordinate `(x, y, z)` lives at linear index
/// `x + y * nx + z * nx * ny`. Marching Cubes needs at least two samples along
/// every axis to form a single voxel cube; a field thinner than that yields an
/// empty mesh.
#[derive(Clone, Debug, PartialEq)]
pub struct ScalarField {
    nx: usize,
    ny: usize,
    nz: usize,
    values: Vec<f32>,
}

impl ScalarField {
    /// Builds a field from its dimensions and row-major sample values, or
    /// returns `None` when any dimension is zero or `values.len()` does not
    /// equal `nx * ny * nz`.
    #[must_use]
    pub fn new(nx: usize, ny: usize, nz: usize, values: Vec<f32>) -> Option<Self> {
        if nx == 0 || ny == 0 || nz == 0 {
            return None;
        }
        let count = nx.checked_mul(ny)?.checked_mul(nz)?;
        if values.len() != count {
            return None;
        }
        Some(Self { nx, ny, nz, values })
    }

    /// The grid dimensions `(nx, ny, nz)`.
    #[must_use]
    pub fn dims(&self) -> (usize, usize, usize) {
        (self.nx, self.ny, self.nz)
    }

    /// The sample value at grid coordinate `(x, y, z)`.
    ///
    /// # Panics
    /// Panics when any coordinate is outside the grid.
    #[must_use]
    pub fn at(&self, x: usize, y: usize, z: usize) -> f32 {
        assert!(
            x < self.nx && y < self.ny && z < self.nz,
            "sample out of range"
        );
        self.values[x + y * self.nx + z * self.nx * self.ny]
    }
}

/// A triangle mesh produced by [`marching_cubes`]: a flat list of vertex
/// positions plus a triangle index list, three indices per triangle.
///
/// Vertices are not de-duplicated; each triangle contributes three fresh
/// positions and three consecutive indices, matching the reference algorithm.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Mesh {
    /// Vertex positions in grid coordinates.
    pub positions: Vec<Vec3>,
    /// Triangle indices into [`Mesh::positions`], three per triangle.
    pub indices: Vec<u32>,
}

impl Mesh {
    /// An empty mesh.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The number of triangles, i.e. one third of the index count.
    #[must_use]
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    /// Whether the mesh carries no triangles.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }
}

/// Linearly interpolates the surface vertex on the edge from `p1` to `p2`,
/// whose corner values are `v1` and `v2`, at the `iso` crossing.
///
/// The crossing parameter is `mu = (iso - v1) / (v2 - v1)`, clamped to
/// `0..=1` so the vertex always lies on the closed segment. When the two corner
/// values are within [`EPS`] of each other the denominator is ill-conditioned,
/// so the edge midpoint is returned instead.
#[must_use]
fn interp_edge(iso: f32, p1: Vec3, p2: Vec3, v1: f32, v2: f32) -> Vec3 {
    let denom = v2 - v1;
    if denom.abs() <= EPS {
        return p1.add(p2).scale(0.5);
    }
    let mu = ((iso - v1) / denom).clamp(0.0, 1.0);
    p1.add(p2.sub(p1).scale(mu))
}

/// Extracts the triangulated `iso`-surface of `field` with the Marching Cubes
/// algorithm.
///
/// Every voxel cube is classified against `iso`, and the built-in [`EDGE_TABLE`]
/// and [`TRI_TABLE`] drive per-edge linear interpolation ([`interp_edge`]) and
/// triangle emission. A field thinner than two samples along any axis has no
/// cubes and yields an empty [`Mesh`].
#[must_use]
pub fn marching_cubes(field: &ScalarField, iso: f32) -> Mesh {
    let mut mesh = Mesh::new();
    let (nx, ny, nz) = field.dims();
    if nx < 2 || ny < 2 || nz < 2 {
        return mesh;
    }
    for z in 0..nz - 1 {
        for y in 0..ny - 1 {
            for x in 0..nx - 1 {
                append_cube(field, iso, x, y, z, &mut mesh);
            }
        }
    }
    mesh
}

/// Classifies the single voxel cube whose minimum corner is `(x, y, z)` and
/// appends its triangles to `mesh`.
fn append_cube(field: &ScalarField, iso: f32, x: usize, y: usize, z: usize, mesh: &mut Mesh) {
    let mut corner_val = [0.0_f32; 8];
    let mut corner_pos = [Vec3::ZERO; 8];
    for (c, off) in CORNER_OFFSET.iter().enumerate() {
        let sx = x + off[0];
        let sy = y + off[1];
        let sz = z + off[2];
        corner_val[c] = field.at(sx, sy, sz);
        corner_pos[c] = Vec3::new(sx as f32, sy as f32, sz as f32);
    }

    let mut cube_index = 0_usize;
    for (c, &v) in corner_val.iter().enumerate() {
        if v < iso {
            cube_index |= 1 << c;
        }
    }

    let edges = EDGE_TABLE[cube_index];
    if edges == 0 {
        return;
    }

    let mut vert = [Vec3::ZERO; 12];
    for (e, corners) in EDGE_CORNERS.iter().enumerate() {
        if edges & (1_u16 << e) != 0 {
            let a = corners[0];
            let b = corners[1];
            vert[e] = interp_edge(
                iso,
                corner_pos[a],
                corner_pos[b],
                corner_val[a],
                corner_val[b],
            );
        }
    }

    let row = &TRI_TABLE[cube_index];
    let mut i = 0;
    while i + 2 < 16 && row[i] >= 0 {
        let e0 = row[i] as usize;
        let e1 = row[i + 1] as usize;
        let e2 = row[i + 2] as usize;
        let base = mesh.positions.len() as u32;
        mesh.positions.push(vert[e0]);
        mesh.positions.push(vert[e1]);
        mesh.positions.push(vert[e2]);
        mesh.indices.push(base);
        mesh.indices.push(base + 1);
        mesh.indices.push(base + 2);
        i += 3;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    const RADIUS_EPS: f32 = 1.0e-4;

    /// Builds a uniform field with every sample equal to `v`.
    fn uniform(nx: usize, ny: usize, nz: usize, v: f32) -> ScalarField {
        ScalarField::new(nx, ny, nz, vec![v; nx * ny * nz]).expect("uniform field")
    }

    /// Builds a single-cube (2x2x2) field from its eight corner values in the
    /// standard corner order.
    fn single_cube(corners: [f32; 8]) -> ScalarField {
        let mut values = vec![0.0_f32; 8];
        for (c, off) in CORNER_OFFSET.iter().enumerate() {
            let idx = off[0] + off[1] * 2 + off[2] * 4;
            values[idx] = corners[c];
        }
        ScalarField::new(2, 2, 2, values).expect("single cube")
    }

    /// Builds a signed sphere field `radius - distance(sample, center)`; samples
    /// inside the sphere are positive (at or above the `0` iso-level).
    fn sphere_field(n: usize, center: Vec3, radius: f32) -> ScalarField {
        let mut values = vec![0.0_f32; n * n * n];
        for z in 0..n {
            for y in 0..n {
                for x in 0..n {
                    let p = Vec3::new(x as f32, y as f32, z as f32);
                    let d = p.sub(center).length();
                    values[x + y * n + z * n * n] = radius - d;
                }
            }
        }
        ScalarField::new(n, n, n, values).expect("sphere field")
    }

    fn tri_normal(mesh: &Mesh, t: usize) -> Vec3 {
        let a = mesh.positions[mesh.indices[t * 3] as usize];
        let b = mesh.positions[mesh.indices[t * 3 + 1] as usize];
        let c = mesh.positions[mesh.indices[t * 3 + 2] as usize];
        b.sub(a).cross(c.sub(a))
    }

    fn tri_centroid(mesh: &Mesh, t: usize) -> Vec3 {
        let a = mesh.positions[mesh.indices[t * 3] as usize];
        let b = mesh.positions[mesh.indices[t * 3 + 1] as usize];
        let c = mesh.positions[mesh.indices[t * 3 + 2] as usize];
        a.add(b).add(c).scale(1.0 / 3.0)
    }

    #[test]
    fn all_below_iso_has_no_triangles() {
        let field = uniform(3, 3, 3, 0.0);
        let mesh = marching_cubes(&field, 1.0);
        assert!(mesh.is_empty());
        assert_eq!(mesh.triangle_count(), 0);
    }

    #[test]
    fn all_above_iso_has_no_triangles() {
        let field = uniform(3, 3, 3, 10.0);
        let mesh = marching_cubes(&field, 1.0);
        assert!(mesh.is_empty());
    }

    #[test]
    fn all_equal_to_iso_has_no_triangles() {
        // value == iso is not strictly below iso, so every corner is inside and
        // the case is the fully-inside empty case.
        let field = uniform(3, 3, 3, 2.5);
        let mesh = marching_cubes(&field, 2.5);
        assert!(mesh.is_empty());
    }

    #[test]
    fn single_low_corner_makes_one_triangle() {
        // Corner 0 below iso, the rest above: case 0x01 -> one triangle.
        let mut corners = [1.0_f32; 8];
        corners[0] = -1.0;
        let field = single_cube(corners);
        let mesh = marching_cubes(&field, 0.0);
        assert_eq!(mesh.triangle_count(), 1);
        assert_eq!(mesh.positions.len(), 3);
        assert_eq!(mesh.indices, vec![0, 1, 2]);
    }

    #[test]
    fn single_high_corner_makes_one_triangle() {
        // Corner 0 above iso, the rest below: case 0xFE -> one triangle.
        let mut corners = [-1.0_f32; 8];
        corners[0] = 1.0;
        let field = single_cube(corners);
        let mesh = marching_cubes(&field, 0.0);
        assert_eq!(mesh.triangle_count(), 1);
    }

    #[test]
    fn single_low_corner_vertices_sit_on_incident_edges() {
        // Corner 0 at -1, neighbours at +1, so each crossing is the edge
        // midpoint. Corner 0 is at the origin; edges 0, 3, 8 leave it toward
        // (1,0,0), (0,0,1) and (0,1,0).
        let mut corners = [1.0_f32; 8];
        corners[0] = -1.0;
        let field = single_cube(corners);
        let mesh = marching_cubes(&field, 0.0);
        let mut seen = [false; 3];
        let expect = [
            Vec3::new(0.5, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 0.5),
            Vec3::new(0.0, 0.5, 0.0),
        ];
        for p in &mesh.positions {
            for (k, e) in expect.iter().enumerate() {
                if p.sub(*e).length() <= RADIUS_EPS {
                    seen[k] = true;
                }
            }
        }
        assert!(seen.iter().all(|&s| s), "all three edge midpoints present");
    }

    #[test]
    fn interpolation_is_exact_midpoint_for_symmetric_values() {
        // Corner 0 = -1, corner 1 = +1: the edge-0 crossing is exactly x = 0.5.
        let mut corners = [5.0_f32; 8];
        corners[0] = -1.0;
        corners[1] = 1.0;
        let field = single_cube(corners);
        let mesh = marching_cubes(&field, 0.0);
        assert!(!mesh.is_empty());
        // Edge 0 connects corners 0 and 1; its crossing must be at x = 0.5.
        let hit = mesh.positions.iter().any(|p| {
            (p.x - 0.5).abs() <= RADIUS_EPS && p.y.abs() <= RADIUS_EPS && p.z.abs() <= RADIUS_EPS
        });
        assert!(hit, "edge-0 midpoint vertex present");
    }

    #[test]
    fn interpolation_respects_asymmetric_parameter() {
        // Corner 0 = -1 (below), corner 1 = 3 (above), iso 0 -> mu = 1/4, so the
        // edge-0 crossing is at x = 0.25.
        let mut corners = [10.0_f32; 8];
        corners[0] = -1.0;
        corners[1] = 3.0;
        let field = single_cube(corners);
        let mesh = marching_cubes(&field, 0.0);
        let hit = mesh.positions.iter().any(|p| {
            (p.x - 0.25).abs() <= RADIUS_EPS && p.y.abs() <= RADIUS_EPS && p.z.abs() <= RADIUS_EPS
        });
        assert!(hit, "edge-0 quarter-point vertex present");
    }

    #[test]
    fn adjacent_low_corners_make_two_triangles() {
        // Corners 0 and 1 below iso: case 0x03 -> a quad, two triangles.
        let mut corners = [1.0_f32; 8];
        corners[0] = -1.0;
        corners[1] = -1.0;
        let field = single_cube(corners);
        let mesh = marching_cubes(&field, 0.0);
        assert_eq!(mesh.triangle_count(), 2);
    }

    #[test]
    fn opposite_corners_make_two_disjoint_triangles() {
        // Corners 0 and 6 (body diagonal) below iso: case 0x41 -> two triangles.
        let mut corners = [1.0_f32; 8];
        corners[0] = -1.0;
        corners[6] = -1.0;
        let field = single_cube(corners);
        let mesh = marching_cubes(&field, 0.0);
        assert_eq!(mesh.triangle_count(), 2);
    }

    #[test]
    fn degenerate_one_by_one_by_one_is_empty() {
        let field = uniform(1, 1, 1, -1.0);
        let mesh = marching_cubes(&field, 0.0);
        assert!(mesh.is_empty());
    }

    #[test]
    fn thin_slab_with_unit_thickness_is_empty() {
        // 1 sample along x -> no cube can form.
        let field = ScalarField::new(1, 4, 4, vec![-1.0_f32; 16]).expect("thin field");
        let mesh = marching_cubes(&field, 0.0);
        assert!(mesh.is_empty());
    }

    #[test]
    fn planar_cut_produces_a_full_connected_sheet() {
        // Field depends only on z; iso 0.5 crosses exactly the z = 0..1 slab.
        let (nx, ny, nz) = (3, 3, 3);
        let mut values = vec![0.0_f32; nx * ny * nz];
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    values[x + y * nx + z * nx * ny] = z as f32;
                }
            }
        }
        let field = ScalarField::new(nx, ny, nz, values).expect("ramp field");
        let mesh = marching_cubes(&field, 0.5);
        // Each of the (nx-1)*(ny-1) cubes in the single crossing slab emits a
        // quad (two triangles).
        assert_eq!(mesh.triangle_count(), 2 * (nx - 1) * (ny - 1));
        // Every vertex sits on the z = 0.5 plane.
        for p in &mesh.positions {
            assert!(
                (p.z - 0.5).abs() <= RADIUS_EPS,
                "vertex off cut plane: {}",
                p.z
            );
        }
    }

    #[test]
    fn planar_cut_between_second_layers_selects_correct_slab() {
        let (nx, ny, nz) = (2, 2, 4);
        let mut values = vec![0.0_f32; nx * ny * nz];
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    values[x + y * nx + z * nx * ny] = z as f32;
                }
            }
        }
        let field = ScalarField::new(nx, ny, nz, values).expect("ramp field");
        let mesh = marching_cubes(&field, 1.5);
        assert_eq!(mesh.triangle_count(), 2);
        for p in &mesh.positions {
            assert!((p.z - 1.5).abs() <= RADIUS_EPS);
        }
    }

    #[test]
    fn non_cubic_grid_extracts_surface() {
        // Distinct dimensions nx != ny != nz with a z-ramp iso cut.
        let (nx, ny, nz) = (4, 3, 2);
        let mut values = vec![0.0_f32; nx * ny * nz];
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    values[x + y * nx + z * nx * ny] = z as f32;
                }
            }
        }
        let field = ScalarField::new(nx, ny, nz, values).expect("ramp field");
        let mesh = marching_cubes(&field, 0.5);
        assert_eq!(mesh.triangle_count(), 2 * (nx - 1) * (ny - 1));
        for p in &mesh.positions {
            assert!((p.z - 0.5).abs() <= RADIUS_EPS);
        }
    }

    #[test]
    fn iso_equal_to_a_corner_value_keeps_that_corner_inside() {
        // Corner 0 exactly equals iso, the rest below: corner 0 is inside, so
        // the case is 0xFE (one triangle), the single-high-corner case.
        let mut corners = [-1.0_f32; 8];
        corners[0] = 2.0;
        let field = single_cube(corners);
        let mesh = marching_cubes(&field, 2.0);
        assert_eq!(mesh.triangle_count(), 1);
    }

    #[test]
    fn sphere_field_produces_closed_reasonable_mesh() {
        let n = 17;
        let center = Vec3::splat(8.0);
        let mesh = marching_cubes(&sphere_field(n, center, 6.0), 0.0);
        // A radius-6 sphere on a 17^3 grid yields a substantial closed shell.
        assert!(
            mesh.triangle_count() > 100,
            "count {}",
            mesh.triangle_count()
        );
        assert_eq!(mesh.positions.len(), mesh.triangle_count() * 3);
        assert_eq!(mesh.indices.len() % 3, 0);
    }

    #[test]
    fn sphere_vertices_lie_near_the_surface_radius() {
        let n = 17;
        let center = Vec3::splat(8.0);
        let radius = 6.0;
        let mesh = marching_cubes(&sphere_field(n, center, radius), 0.0);
        assert!(!mesh.is_empty());
        for p in &mesh.positions {
            let d = p.sub(center).length();
            // Linear interpolation of a nonlinear distance field is inexact, but
            // vertices must stay within a fraction of a cell of the true radius.
            assert!((d - radius).abs() < 0.75, "vertex radius {d}");
        }
    }

    #[test]
    fn sphere_triangle_normals_are_consistently_oriented() {
        let n = 15;
        let center = Vec3::splat(7.0);
        let mesh = marching_cubes(&sphere_field(n, center, 5.0), 0.0);
        let mut saw_positive = false;
        let mut saw_negative = false;
        for t in 0..mesh.triangle_count() {
            let normal = tri_normal(&mesh, t);
            if normal.length() <= RADIUS_EPS {
                continue;
            }
            let outward = tri_centroid(&mesh, t).sub(center);
            let s = normal.dot(outward);
            if s > RADIUS_EPS {
                saw_positive = true;
            } else if s < -RADIUS_EPS {
                saw_negative = true;
            }
        }
        assert!(
            saw_positive || saw_negative,
            "some oriented triangles exist"
        );
        assert!(
            !(saw_positive && saw_negative),
            "all sphere triangles share a consistent winding relative to the center"
        );
    }

    #[test]
    fn translation_by_integer_offset_reproduces_the_surface() {
        // A radius-4 sphere in an 11^3 grid, then the same sphere embedded at a
        // (1,1,1) offset in a 13^3 grid whose extra border stays outside. The
        // extracted surfaces must be congruent under the offset.
        let center_a = Vec3::splat(5.0);
        let radius = 4.0;
        let field_a = sphere_field(11, center_a, radius);
        let mesh_a = marching_cubes(&field_a, 0.0);

        let center_b = Vec3::splat(6.0);
        let field_b = sphere_field(13, center_b, radius);
        let mesh_b = marching_cubes(&field_b, 0.0);

        assert_eq!(mesh_a.triangle_count(), mesh_b.triangle_count());
        assert!(mesh_a.triangle_count() > 0);

        let shift = Vec3::splat(1.0);
        let centroids_a: Vec<Vec3> = (0..mesh_a.triangle_count())
            .map(|t| tri_centroid(&mesh_a, t).add(shift))
            .collect();
        let centroids_b: Vec<Vec3> = (0..mesh_b.triangle_count())
            .map(|t| tri_centroid(&mesh_b, t))
            .collect();
        let mut used = vec![false; centroids_b.len()];
        for a in &centroids_a {
            let mut matched = false;
            for (j, b) in centroids_b.iter().enumerate() {
                if !used[j] && a.sub(*b).length() <= RADIUS_EPS {
                    used[j] = true;
                    matched = true;
                    break;
                }
            }
            assert!(matched, "no shifted match for centroid {a:?}");
        }
        assert!(used.iter().all(|&u| u), "every shifted triangle is matched");
    }

    #[test]
    fn extraction_is_deterministic() {
        let field = sphere_field(13, Vec3::splat(6.0), 4.0);
        let a = marching_cubes(&field, 0.0);
        let b = marching_cubes(&field, 0.0);
        assert_eq!(a, b);
    }

    #[test]
    fn every_triangle_vertex_lies_on_a_crossing_cube_edge() {
        // For a z-ramp cut, all crossings are on the four vertical cube edges,
        // so every vertex has integer x and y and z exactly 0.5.
        let (nx, ny, nz) = (3, 3, 2);
        let mut values = vec![0.0_f32; nx * ny * nz];
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    values[x + y * nx + z * nx * ny] = z as f32;
                }
            }
        }
        let field = ScalarField::new(nx, ny, nz, values).expect("ramp field");
        let mesh = marching_cubes(&field, 0.5);
        assert!(!mesh.is_empty());
        for p in &mesh.positions {
            assert!((p.z - 0.5).abs() <= RADIUS_EPS);
            assert!(
                (p.x - (p.x + 0.5).floor()).abs() <= RADIUS_EPS,
                "x not integral: {}",
                p.x
            );
            assert!(
                (p.y - (p.y + 0.5).floor()).abs() <= RADIUS_EPS,
                "y not integral: {}",
                p.y
            );
        }
    }

    #[test]
    fn field_new_rejects_wrong_value_count() {
        assert!(ScalarField::new(2, 2, 2, vec![0.0_f32; 7]).is_none());
        assert!(ScalarField::new(2, 2, 2, vec![0.0_f32; 9]).is_none());
    }

    #[test]
    fn field_new_rejects_zero_dimension() {
        assert!(ScalarField::new(0, 2, 2, vec![]).is_none());
        assert!(ScalarField::new(2, 0, 2, vec![]).is_none());
        assert!(ScalarField::new(2, 2, 0, vec![]).is_none());
    }

    #[test]
    fn field_indexing_is_row_major() {
        let values: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let field = ScalarField::new(4, 3, 2, values).expect("field");
        assert!((field.at(0, 0, 0) - 0.0).abs() <= RADIUS_EPS);
        assert!((field.at(3, 0, 0) - 3.0).abs() <= RADIUS_EPS);
        assert!((field.at(0, 1, 0) - 4.0).abs() <= RADIUS_EPS);
        assert!((field.at(0, 0, 1) - 12.0).abs() <= RADIUS_EPS);
        assert!((field.at(3, 2, 1) - 23.0).abs() <= RADIUS_EPS);
    }

    #[test]
    fn edge_table_matches_triangle_table() {
        // The shipped edge mask must be exactly the union of edges the triangle
        // table references for each case: no missing edge, no extra bit.
        for case in 0..256 {
            let mut used = 0_u16;
            for &e in &TRI_TABLE[case] {
                if e >= 0 {
                    used |= 1_u16 << (e as u16);
                }
            }
            assert_eq!(used, EDGE_TABLE[case], "case {case}");
        }
    }

    #[expect(
        clippy::needless_range_loop,
        reason = "the case index also labels assertion messages, so an explicit 0..256 walk reads clearer"
    )]
    #[test]
    fn triangle_table_rows_are_well_formed() {
        for case in 0..256 {
            let row = &TRI_TABLE[case];
            // Count leading valid entries; they must come in whole triangles.
            let mut valid = 0;
            let mut ended = false;
            for &e in row {
                if e < 0 {
                    ended = true;
                } else {
                    assert!(!ended, "case {case}: index after terminator");
                    assert!((0..12).contains(&e), "case {case}: edge {e} out of range");
                    valid += 1;
                }
            }
            assert_eq!(
                valid % 3,
                0,
                "case {case}: {valid} indices not a triangle multiple"
            );
        }
    }

    #[test]
    fn only_all_inside_and_all_outside_cases_are_empty() {
        // The two uniform corner configurations (all above or all below the iso
        // level) have no surface crossing and emit no triangles; every mixed
        // configuration crosses the iso level and emits at least one triangle.
        let count = |c: usize| TRI_TABLE[c].iter().filter(|&&e| e >= 0).count();
        assert_eq!(count(0), 0, "all-inside case must be empty");
        assert_eq!(count(255), 0, "all-outside case must be empty");
        for case in 1..255usize {
            assert!(count(case) >= 3, "mixed case {case} must emit a triangle");
        }
    }

    #[test]
    fn empty_mesh_helpers_report_empty() {
        let mesh = Mesh::new();
        assert!(mesh.is_empty());
        assert_eq!(mesh.triangle_count(), 0);
        assert!(mesh.positions.is_empty());
    }

    #[test]
    fn interp_edge_uses_midpoint_when_corner_values_are_near_equal() {
        // Both endpoints within EPS of each other (and of iso): the crossing
        // parameter is ill-conditioned, so the midpoint is returned.
        let p1 = Vec3::new(2.0, 0.0, 0.0);
        let p2 = Vec3::new(4.0, 0.0, 0.0);
        let v = interp_edge(0.0, p1, p2, 1.0e-8, -1.0e-8);
        assert!(
            v.sub(Vec3::new(3.0, 0.0, 0.0)).length() <= RADIUS_EPS,
            "midpoint: {v:?}"
        );
    }

    #[test]
    fn interp_edge_places_vertex_by_linear_parameter() {
        let p1 = Vec3::new(0.0, 0.0, 0.0);
        let p2 = Vec3::new(4.0, 0.0, 0.0);
        // v1 = -1, v2 = 3, iso 0 -> mu = 1/4 -> x = 1.0.
        let v = interp_edge(0.0, p1, p2, -1.0, 3.0);
        assert!(
            v.sub(Vec3::new(1.0, 0.0, 0.0)).length() <= RADIUS_EPS,
            "quarter: {v:?}"
        );
        // Symmetric span -> midpoint.
        let m = interp_edge(0.0, p1, p2, -2.0, 2.0);
        assert!(
            m.sub(Vec3::new(2.0, 0.0, 0.0)).length() <= RADIUS_EPS,
            "mid: {m:?}"
        );
    }

    #[test]
    fn multiple_crossing_slabs_accumulate() {
        // A z-ramp with two separated crossings (iso 0.5 and a second cut) is not
        // possible with a single iso, so use a V-shaped field in z to cross
        // twice: values 0,1,0 along z produce two crossing slabs.
        let (nx, ny, nz) = (2, 2, 3);
        let layer = [0.0_f32, 1.0, 0.0];
        let mut values = vec![0.0_f32; nx * ny * nz];
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    values[x + y * nx + z * nx * ny] = layer[z];
                }
            }
        }
        let field = ScalarField::new(nx, ny, nz, values).expect("v field");
        let mesh = marching_cubes(&field, 0.5);
        // Both z-slabs cross iso, each emitting one quad (two triangles).
        assert_eq!(mesh.triangle_count(), 4);
    }
}
