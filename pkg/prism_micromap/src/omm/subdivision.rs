//! Micro-triangle subdivision: level, counts, and the canonical ordering.
//!
//! A base triangle at subdivision `level` is split into `4^level`
//! micro-triangles by cutting each edge into `2^level` equal segments. Every
//! micro-triangle vertex therefore lands on an integer barycentric lattice
//! point `(w0, w1, w2)` with `w0 + w1 + w2 == 2^level`.
//!
//! # Canonical ordering
//!
//! Prism enumerates micro-triangles in a deterministic **row-major triangle
//! strip** order so the `GPU` twin can reproduce the exact index of every
//! micro-triangle. Using lattice coordinates `(a, b)` where `a` is the step
//! count toward vertex `1`, `b` the step count toward vertex `2`, and the
//! implied weight on vertex `0` is `n - a - b` (with `n = 2^level`):
//!
//! * an **upright** micro-triangle `up(a, b)` (needs `a + b <= n - 1`) has
//!   vertices `L(a, b)`, `L(a + 1, b)`, `L(a, b + 1)`;
//! * an **inverted** micro-triangle `dn(a, b)` (needs `a + b <= n - 2`) has
//!   vertices `L(a + 1, b)`, `L(a, b + 1)`, `L(a + 1, b + 1)`.
//!
//! Row `b` runs from `0` to `n - 1` and emits, left to right,
//! `up(0, b), dn(0, b), up(1, b), dn(1, b), ..., up(n - 1 - b, b)` for a total
//! of `2 * (n - b) - 1` micro-triangles. Summed over all rows this is exactly
//! `n^2 == 4^level`.
//!
//! This ordering is Prism's own canonical layout, not the vendor "bird curve"
//! permutation; a later encode step can remap indices when feeding a concrete
//! `DXR` / Vulkan build if bit-identical vendor ordering is required.

use alloc::vec::Vec;

/// Maximum supported subdivision level (`DXR` caps micromaps at level `12`,
/// i.e. `4^12 == 16_777_216` micro-triangles).
pub const MAX_SUBDIVISION_LEVEL: u8 = 12;

/// A validated micromap subdivision level in `0..=12`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SubdivisionLevel(u8);

impl SubdivisionLevel {
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

    /// Number of micro-triangles `4^level`.
    #[must_use]
    pub const fn micro_triangle_count(self) -> u32 {
        1u32 << (2 * self.0 as u32)
    }
}

/// A single micro-triangle expressed by its three integer barycentric lattice
/// vertices, each summing to [`SubdivisionLevel::segments`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MicroTriangle {
    /// The three vertices as integer barycentric weights `(w0, w1, w2)`.
    pub vertices: [[u32; 3]; 3],
    /// `true` for an upright micro-triangle (same winding as the base), `false`
    /// for an inverted one.
    pub upright: bool,
}

impl MicroTriangle {
    /// Returns the three vertices as fractional barycentric weights in `0..=1`,
    /// dividing each integer weight by `n = 2^level`.
    #[must_use]
    pub fn barycentric_f32(&self, level: SubdivisionLevel) -> [[f32; 3]; 3] {
        let inv_n = 1.0 / level.segments() as f32;
        let mut out = [[0.0f32; 3]; 3];
        let mut v = 0;
        while v < 3 {
            let mut c = 0;
            while c < 3 {
                out[v][c] = self.vertices[v][c] as f32 * inv_n;
                c += 1;
            }
            v += 1;
        }
        out
    }
}

/// Builds a lattice vertex `(n - a - b, a, b)` for edge-step coordinates
/// `(a, b)` at `n` segments.
#[inline]
const fn lattice(n: u32, a: u32, b: u32) -> [u32; 3] {
    [n - a - b, a, b]
}

#[inline]
const fn upright(n: u32, a: u32, b: u32) -> MicroTriangle {
    MicroTriangle {
        vertices: [lattice(n, a, b), lattice(n, a + 1, b), lattice(n, a, b + 1)],
        upright: true,
    }
}

#[inline]
const fn inverted(n: u32, a: u32, b: u32) -> MicroTriangle {
    MicroTriangle {
        vertices: [
            lattice(n, a + 1, b),
            lattice(n, a, b + 1),
            lattice(n, a + 1, b + 1),
        ],
        upright: false,
    }
}

/// Number of micro-triangles in row `b` for `n` segments: `2 * (n - b) - 1`.
#[inline]
const fn row_len(n: u32, b: u32) -> u32 {
    2 * (n - b) - 1
}

/// Returns the micro-triangle at canonical `index`, or [`None`] when `index`
/// is out of range for `level`.
#[must_use]
pub fn micro_triangle_at(level: SubdivisionLevel, index: u32) -> Option<MicroTriangle> {
    let n = level.segments();
    if index >= level.micro_triangle_count() {
        return None;
    }
    let mut remaining = index;
    let mut b = 0;
    loop {
        let len = row_len(n, b);
        if remaining < len {
            break;
        }
        remaining -= len;
        b += 1;
    }
    // Within the row, even positions are upright, odd positions inverted.
    let a = remaining / 2;
    if remaining.is_multiple_of(2) {
        Some(upright(n, a, b))
    } else {
        Some(inverted(n, a, b))
    }
}

/// Materialises every micro-triangle in canonical order.
///
/// Allocates a [`Vec`] of length `4^level`; prefer [`micro_triangle_at`] for
/// random access at high levels.
#[must_use]
pub fn micro_triangles(level: SubdivisionLevel) -> Vec<MicroTriangle> {
    let n = level.segments();
    let count = level.micro_triangle_count() as usize;
    let mut out = Vec::with_capacity(count);
    let mut b = 0;
    while b < n {
        let mut a = 0;
        while a < n - b {
            out.push(upright(n, a, b));
            if a < n - 1 - b {
                out.push(inverted(n, a, b));
            }
            a += 1;
        }
        b += 1;
    }
    out
}
