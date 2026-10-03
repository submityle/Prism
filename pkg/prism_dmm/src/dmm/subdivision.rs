//! Micro-vertex subdivision: level, counts, and the canonical lattice order.
//!
//! A `DMM` tessellates a base triangle into a regular micro-mesh. At
//! subdivision `level L` each base edge is cut into `n == 2^L` equal segments,
//! yielding `4^L` micro-triangles but — crucially for a *displaced* micro-map —
//! `(n + 1) * (n + 2) / 2` micro-**vertices**. Every micro-vertex lands on an
//! integer barycentric lattice point `(w0, w1, w2)` with `w0 + w1 + w2 == n`.
//! The displacement scalar baked by this crate is stored *per micro-vertex*,
//! not per micro-triangle.
//!
//! # Canonical ordering
//!
//! Prism enumerates micro-vertices in a deterministic row-major order so the
//! `GPU` twin can reproduce the exact index of every micro-vertex. Writing the
//! lattice coordinate as `(a, b)` with the implied weight `n - a - b` on
//! vertex `0`:
//!
//! * the outer loop walks `b` from `0` to `n` (the row index, i.e. the step
//!   count toward base vertex `2`);
//! * the inner loop walks `a` from `0` to `n - b` (the step count toward base
//!   vertex `1`);
//! * the emitted micro-vertex is `(n - a - b, a, b)`.
//!
//! Row `b` therefore contributes `n - b + 1` micro-vertices, and summing over
//! all rows gives exactly `(n + 1) * (n + 2) / 2`.

use alloc::vec::Vec;

/// Maximum supported subdivision level.
///
/// Kept deliberately modest: level `5` already expands one base triangle into
/// `1024` micro-triangles and `561` micro-vertices, which is plenty for a
/// `CPU` golden used to validate a `GPU` twin.
pub const MAX_SUBDIVISION_LEVEL: u8 = 5;

/// A validated `DMM` subdivision level in `0..=5`.
///
/// The level `L` fixes the segment count `n == 2^L`, the micro-vertex count
/// `(n + 1) * (n + 2) / 2`, and the micro-triangle count `4^L`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DmmSubdivisionLevel(u8);

impl DmmSubdivisionLevel {
    /// Creates a level, returning [`None`] when `level > MAX_SUBDIVISION_LEVEL`.
    #[must_use]
    pub const fn new(level: u8) -> Option<Self> {
        if level <= MAX_SUBDIVISION_LEVEL {
            Some(Self(level))
        } else {
            None
        }
    }

    /// Returns the raw level value.
    #[must_use]
    pub const fn get(self) -> u8 {
        self.0
    }

    /// Number of edge segments `n == 2^level`.
    #[must_use]
    pub const fn segments(self) -> u32 {
        1u32 << self.0
    }

    /// Number of micro-vertices `(n + 1) * (n + 2) / 2`.
    #[must_use]
    pub const fn micro_vertex_count(self) -> u32 {
        let n = self.segments();
        (n + 1) * (n + 2) / 2
    }

    /// Number of micro-triangles `4^level`.
    #[must_use]
    pub const fn micro_triangle_count(self) -> u32 {
        1u32 << (2 * self.0 as u32)
    }
}

/// Returns the integer barycentric lattice coordinate of the micro-vertex at
/// canonical `index`, or [`None`] when `index` is out of range.
///
/// The returned `[w0, w1, w2]` always satisfies `w0 + w1 + w2 == n`.
#[must_use]
pub fn micro_vertex_at(level: DmmSubdivisionLevel, index: u32) -> Option<[u32; 3]> {
    let n = level.segments();
    let mut remaining = index;
    let mut b = 0u32;
    while b <= n {
        let row_len = n - b + 1;
        if remaining < row_len {
            let a = remaining;
            let w0 = n - a - b;
            return Some([w0, a, b]);
        }
        remaining -= row_len;
        b += 1;
    }
    None
}

/// Returns every micro-vertex lattice coordinate in canonical order.
///
/// The length equals [`DmmSubdivisionLevel::micro_vertex_count`], and the
/// `i`-th entry equals `micro_vertex_at(level, i)`.
#[must_use]
pub fn micro_vertices(level: DmmSubdivisionLevel) -> Vec<[u32; 3]> {
    let n = level.segments();
    let mut out = Vec::with_capacity(level.micro_vertex_count() as usize);
    for b in 0..=n {
        for a in 0..=(n - b) {
            let w0 = n - a - b;
            out.push([w0, a, b]);
        }
    }
    out
}

/// Converts an integer barycentric lattice coordinate into normalized `f32`
/// barycentric weights that sum to `1.0`.
///
/// Division is by the segment count `n`, which is always at least `1` (level
/// `0` has `n == 1`), so there is no division by zero and the result never
/// contains `NaN`.
#[must_use]
pub fn barycentric_f32(level: DmmSubdivisionLevel, vertex: [u32; 3]) -> [f32; 3] {
    let inv_n = 1.0 / level.segments() as f32;
    [
        vertex[0] as f32 * inv_n,
        vertex[1] as f32 * inv_n,
        vertex[2] as f32 * inv_n,
    ]
}
