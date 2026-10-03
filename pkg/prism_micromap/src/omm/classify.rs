//! Conservative tri-state classification of a micro-triangle against a mask.

use crate::omm::mask::AlphaMask;
use crate::omm::state::OpacityState;

/// How a micro-triangle's coverage is sampled during classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SampleStrategy {
    /// Sample a uniform triangular grid of `samples_per_edge^2` interior
    /// centroids plus the three vertices and edge midpoints. Fully
    /// deterministic and trivially portable to a compute kernel. A value of
    /// `0` is treated as `1`.
    Uniform {
        /// Sub-subdivision segments per micro-triangle edge.
        samples_per_edge: u32,
    },
    /// Sample every mask texel whose center falls inside the micro-triangle
    /// (plus the three vertices as a fallback for sub-texel micro-triangles).
    /// This matches exactly the texels a hardware alpha test can read, giving
    /// the most conservative `Unknown` detection.
    TexelConservative,
}

impl Default for SampleStrategy {
    fn default() -> Self {
        Self::Uniform { samples_per_edge: 8 }
    }
}

/// Running tri-state accumulator folded over coverage samples.
#[derive(Debug, Clone, Copy)]
struct Coverage {
    opaque: u32,
    transparent: u32,
}

impl Coverage {
    const fn new() -> Self {
        Self {
            opaque: 0,
            transparent: 0,
        }
    }

    fn observe(&mut self, is_opaque: bool) {
        if is_opaque {
            self.opaque += 1;
        } else {
            self.transparent += 1;
        }
    }

    fn resolve(self) -> OpacityState {
        match (self.opaque, self.transparent) {
            (0, 0) => OpacityState::Transparent,
            (_, 0) => OpacityState::Opaque,
            (0, _) => OpacityState::Transparent,
            (o, t) => {
                if o >= t {
                    OpacityState::UnknownOpaque
                } else {
                    OpacityState::UnknownTransparent
                }
            }
        }
    }
}

/// Linear interpolation of a `UV` from barycentric weights over the three
/// micro-triangle `UV` corners.
#[inline]
fn interp_uv(uv: &[[f32; 2]; 3], b0: f32, b1: f32, b2: f32) -> [f32; 2] {
    [
        uv[0][0] * b0 + uv[1][0] * b1 + uv[2][0] * b2,
        uv[0][1] * b0 + uv[1][1] * b1 + uv[2][1] * b2,
    ]
}

/// Twice the signed area of triangle `(a, b, c)` in `UV` space.
#[inline]
fn orient(a: [f32; 2], b: [f32; 2], c: [f32; 2]) -> f32 {
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
}

/// Inclusive point-in-triangle test tolerant of either winding.
fn point_in_triangle(p: [f32; 2], a: [f32; 2], b: [f32; 2], c: [f32; 2]) -> bool {
    const EPS: f32 = 1e-6;
    let d0 = orient(p, a, b);
    let d1 = orient(p, b, c);
    let d2 = orient(p, c, a);
    let has_neg = d0 < -EPS || d1 < -EPS || d2 < -EPS;
    let has_pos = d0 > EPS || d1 > EPS || d2 > EPS;
    !(has_neg && has_pos)
}

/// Classifies a micro-triangle whose three corners carry the `UV` coordinates
/// `uv` using the chosen [`SampleStrategy`].
#[must_use]
pub fn classify_micro_triangle<M: AlphaMask + ?Sized>(
    mask: &M,
    uv: &[[f32; 2]; 3],
    strategy: SampleStrategy,
) -> OpacityState {
    let mut cov = Coverage::new();
    // The three vertices and edge midpoints anchor the classification for both
    // strategies, catching slivers that a sparse interior grid might miss.
    for &(b0, b1, b2) in &[
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.5, 0.5, 0.0),
        (0.0, 0.5, 0.5),
        (0.5, 0.0, 0.5),
    ] {
        let p = interp_uv(uv, b0, b1, b2);
        cov.observe(mask.is_opaque(p[0], p[1]));
    }

    match strategy {
        SampleStrategy::Uniform { samples_per_edge } => {
            classify_uniform(mask, uv, samples_per_edge.max(1), &mut cov);
        }
        SampleStrategy::TexelConservative => {
            classify_texels(mask, uv, &mut cov);
        }
    }

    cov.resolve()
}

/// Samples the centroids of an `s`-segment sub-subdivision of the
/// micro-triangle.
fn classify_uniform<M: AlphaMask + ?Sized>(
    mask: &M,
    uv: &[[f32; 2]; 3],
    s: u32,
    cov: &mut Coverage,
) {
    let inv_s = 1.0 / s as f32;
    let mut b = 0;
    while b < s {
        let mut a = 0;
        while a < s - b {
            // Upright sub-cell centroid.
            emit_centroid(mask, uv, inv_s, a, b, a + 1, b, a, b + 1, cov);
            if a < s - 1 - b {
                // Inverted sub-cell centroid.
                emit_centroid(mask, uv, inv_s, a + 1, b, a, b + 1, a + 1, b + 1, cov);
            }
            a += 1;
        }
        b += 1;
    }
}

/// Averages three sub-lattice vertices into a centroid barycentric coordinate
/// and samples the mask there.
#[inline]
#[expect(
    clippy::too_many_arguments,
    reason = "flat scalar sub-lattice coordinates avoid intermediate array allocation on the hot classification path"
)]
fn emit_centroid<M: AlphaMask + ?Sized>(
    mask: &M,
    uv: &[[f32; 2]; 3],
    inv_s: f32,
    a0: u32,
    b0: u32,
    a1: u32,
    b1: u32,
    a2: u32,
    b2: u32,
    cov: &mut Coverage,
) {
    // Each sub-vertex `(a, b)` maps to micro-triangle barycentric
    // `(1 - a*inv_s - b*inv_s, a*inv_s, b*inv_s)`; the centroid is their mean.
    let third = 1.0 / 3.0;
    let s1 = (a0 + a1 + a2) as f32 * inv_s * third;
    let s2 = (b0 + b1 + b2) as f32 * inv_s * third;
    let s0 = 1.0 - s1 - s2;
    let p = interp_uv(uv, s0, s1, s2);
    cov.observe(mask.is_opaque(p[0], p[1]));
}

/// Samples the center of every mask texel whose center lies inside the
/// micro-triangle, where `mask` exposes its grid resolution.
fn classify_texels<M: AlphaMask + ?Sized>(mask: &M, uv: &[[f32; 2]; 3], cov: &mut Coverage) {
    let Some((width, height)) = mask.grid_resolution() else {
        // Resolution-free mask: fall back to a dense uniform interior grid.
        classify_uniform(mask, uv, 16, cov);
        return;
    };

    let min_u = uv[0][0].min(uv[1][0]).min(uv[2][0]).clamp(0.0, 1.0);
    let max_u = uv[0][0].max(uv[1][0]).max(uv[2][0]).clamp(0.0, 1.0);
    let min_v = uv[0][1].min(uv[1][1]).min(uv[2][1]).clamp(0.0, 1.0);
    let max_v = uv[0][1].max(uv[1][1]).max(uv[2][1]).clamp(0.0, 1.0);

    let x0 = libm::floorf(min_u * width as f32).max(0.0) as u32;
    let y0 = libm::floorf(min_v * height as f32).max(0.0) as u32;
    let x1 = (libm::floorf(max_u * width as f32) as u32).min(width - 1);
    let y1 = (libm::floorf(max_v * height as f32) as u32).min(height - 1);

    let inv_w = 1.0 / width as f32;
    let inv_h = 1.0 / height as f32;
    let threshold = mask.threshold();
    let mut y = y0;
    while y <= y1 {
        let mut x = x0;
        while x <= x1 {
            let center = [(x as f32 + 0.5) * inv_w, (y as f32 + 0.5) * inv_h];
            if point_in_triangle(center, uv[0], uv[1], uv[2]) {
                cov.observe(mask.fetch_texel(x, y) >= threshold);
            }
            x += 1;
        }
        y += 1;
    }
}
