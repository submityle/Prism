//! Perspective-correct barycentric coordinates and their screen-space
//! derivatives, reconstructed from clip-space triangle vertices.
//!
//! This is the core of Deferred Attribute Interpolation Shading (DAIS, Schied
//! & Dachsbacher 2015). A visibility-buffer pixel stores only an instance and
//! triangle id; to shade it we must recover, analytically, the perspective-
//! correct barycentric weights `b = (b0, b1, b2)` at the pixel centre *and*
//! their partial derivatives with respect to screen `x` and `y`. Those
//! derivatives let the deferred pass interpolate any vertex attribute and get
//! its screen gradients "for free", which in turn drive texture `LOD` selection
//! exactly as a forward rasterizer's hardware quad derivatives would.
//!
//! All math is done in `f32` to stay bit-faithful to a GPU implementation and
//! verifiable against finite differences.

/// A clip-space vertex: homogeneous position straight out of the vertex shader
/// (before the perspective divide).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClipVertex {
    /// Homogeneous clip-space position `(x, y, z, w)`.
    pub position: [f32; 4],
}

impl ClipVertex {
    /// Builds a clip-space vertex from its homogeneous coordinates.
    #[must_use]
    pub const fn new(position: [f32; 4]) -> Self {
        Self { position }
    }
}

/// Screen viewport in pixels. The origin is the top-left corner; `y` grows
/// downward, matching the usual framebuffer convention.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Viewport {
    /// Width in pixels.
    pub width: f32,
    /// Height in pixels.
    pub height: f32,
}

impl Viewport {
    /// Builds a viewport from integer pixel dimensions.
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width: pixels_to_f32(width),
            height: pixels_to_f32(height),
        }
    }
}

/// Converts a pixel count to `f32` without a lossy-cast lint.
///
/// Resolutions comfortably fit in `f32`'s exact-integer range (`< 2^24`), so
/// this is exact for any sane viewport.
#[must_use]
fn pixels_to_f32(pixels: u32) -> f32 {
    u16::try_from(pixels).map_or_else(
        |_| {
            // Extremely large viewport; clamp to the exact-integer ceiling.
            let hi = u16::try_from(pixels >> 16).unwrap_or(u16::MAX);
            let lo = u16::try_from(pixels & 0xFFFF).unwrap_or(u16::MAX);
            f32::from(hi) * 65_536.0 + f32::from(lo)
        },
        f32::from,
    )
}

/// Perspective-correct barycentric weights and their screen derivatives.
///
/// `lambda` are the true barycentric coordinates at the sample point (they sum
/// to 1). `ddx`/`ddy` are `∂lambda/∂x` and `∂lambda/∂y` in pixel units.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BarycentricDerivatives {
    /// Perspective-correct barycentric weights `(b0, b1, b2)`, summing to 1.
    pub lambda: [f32; 3],
    /// `∂lambda/∂x` per component, in units of 1/pixel.
    pub ddx: [f32; 3],
    /// `∂lambda/∂y` per component, in units of 1/pixel.
    pub ddy: [f32; 3],
}

/// Screen-space pixel position of a projected clip vertex.
#[derive(Clone, Copy)]
struct ScreenVertex {
    /// Pixel-space position.
    pos: [f32; 2],
    /// Reciprocal of clip `w` (`1/w`), used for perspective correction.
    inv_w: f32,
}

/// Projects a clip-space vertex to pixel space, returning `None` if the vertex
/// is behind or on the camera plane (`w <= 0`).
fn project(vertex: &ClipVertex, viewport: Viewport) -> Option<ScreenVertex> {
    let [x, y, _z, w] = vertex.position;
    if w <= 0.0 {
        return None;
    }
    let inv_w = 1.0 / w;
    // Clip -> NDC in [-1, 1], then NDC -> pixel with a y-flip (top-left origin).
    let ndc_x = x * inv_w;
    let ndc_y = y * inv_w;
    let px = (ndc_x * 0.5 + 0.5) * viewport.width;
    let py = (0.5 - ndc_y * 0.5) * viewport.height;
    Some(ScreenVertex {
        pos: [px, py],
        inv_w,
    })
}

/// Computes perspective-correct barycentric weights and their screen-space
/// derivatives for `pixel` inside the triangle `tri`.
///
/// Returns `None` when any vertex is behind the camera (`w <= 0`) or the
/// projected triangle is degenerate (zero screen area).
///
/// The derivatives are exact (closed-form), not finite differences, and hold
/// everywhere on the triangle's plane, including outside the triangle, which is
/// what a `2x2` shading quad straddling an edge needs.
#[must_use]
pub fn barycentric_derivatives(
    tri: &[ClipVertex; 3],
    viewport: Viewport,
    pixel: [f32; 2],
) -> Option<BarycentricDerivatives> {
    let s0 = project(&tri[0], viewport)?;
    let s1 = project(&tri[1], viewport)?;
    let s2 = project(&tri[2], viewport)?;

    // Edge vectors of the projected triangle in pixel space.
    let d1 = [s1.pos[0] - s0.pos[0], s1.pos[1] - s0.pos[1]];
    let d2 = [s2.pos[0] - s0.pos[0], s2.pos[1] - s0.pos[1]];

    // Twice the signed screen area. A near-zero value relative to the edge
    // lengths means a degenerate (edge-on or sub-pixel) triangle; a plain
    // `== 0.0` test misses collinear vertices whose cross product is a tiny
    // but non-zero float, so compare against a relative epsilon.
    let denom = d1[0] * d2[1] - d2[0] * d1[1];
    let edge_scale = (d1[0] * d1[0] + d1[1] * d1[1]).sqrt()
        * (d2[0] * d2[0] + d2[1] * d2[1]).sqrt();
    if denom.abs() <= 1e-6 * edge_scale.max(f32::MIN_POSITIVE) {
        return None;
    }
    let inv_denom = 1.0 / denom;

    // Offset of the sample from vertex 0.
    let dp = [pixel[0] - s0.pos[0], pixel[1] - s0.pos[1]];

    // Affine (screen-linear) barycentrics of the *projected* triangle. These
    // are linear in the pixel coordinate, so their derivatives are constant.
    let f1 = (dp[0] * d2[1] - d2[0] * dp[1]) * inv_denom;
    let f2 = (d1[0] * dp[1] - dp[0] * d1[1]) * inv_denom;
    let f0 = 1.0 - f1 - f2;
    let f = [f0, f1, f2];

    // Constant screen gradients of the affine barycentrics.
    let df1_dx = d2[1] * inv_denom;
    let df1_dy = -d2[0] * inv_denom;
    let df2_dx = -d1[1] * inv_denom;
    let df2_dy = d1[0] * inv_denom;
    let df_dx = [-(df1_dx + df2_dx), df1_dx, df2_dx];
    let df_dy = [-(df1_dy + df2_dy), df1_dy, df2_dy];

    let inv_w = [s0.inv_w, s1.inv_w, s2.inv_w];

    // Perspective correction: b_i = f_i/w_i / sum_j(f_j/w_j).
    let big_d = f[0] * inv_w[0] + f[1] * inv_w[1] + f[2] * inv_w[2];
    if big_d == 0.0 {
        return None;
    }
    let inv_big_d = 1.0 / big_d;

    // Derivatives of the denominator D.
    let dd_dx = df_dx[0] * inv_w[0] + df_dx[1] * inv_w[1] + df_dx[2] * inv_w[2];
    let dd_dy = df_dy[0] * inv_w[0] + df_dy[1] * inv_w[1] + df_dy[2] * inv_w[2];

    let mut lambda = [0.0_f32; 3];
    let mut ddx = [0.0_f32; 3];
    let mut ddy = [0.0_f32; 3];
    let inv_big_d_sq = inv_big_d * inv_big_d;
    for i in 0..3 {
        let num = f[i] * inv_w[i];
        lambda[i] = num * inv_big_d;
        // Quotient rule: d(num/D) = (dnum*D - num*dD)/D^2.
        let dnum_dx = df_dx[i] * inv_w[i];
        let dnum_dy = df_dy[i] * inv_w[i];
        ddx[i] = (dnum_dx * big_d - num * dd_dx) * inv_big_d_sq;
        ddy[i] = (dnum_dy * big_d - num * dd_dy) * inv_big_d_sq;
    }

    Some(BarycentricDerivatives { lambda, ddx, ddy })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A screen-facing triangle at a constant depth `w` (an orthographic-ish
    /// setup where perspective correction is a no-op) laid out so pixel space
    /// is easy to reason about.
    fn flat_triangle(w: f32) -> [ClipVertex; 3] {
        [
            ClipVertex::new([-1.0 * w, -1.0 * w, 0.0, w]),
            ClipVertex::new([1.0 * w, -1.0 * w, 0.0, w]),
            ClipVertex::new([-1.0 * w, 1.0 * w, 0.0, w]),
        ]
    }

    fn central_difference<F: Fn([f32; 2]) -> [f32; 3]>(
        f: &F,
        pixel: [f32; 2],
        axis: usize,
        h: f32,
    ) -> [f32; 3] {
        let mut plus = pixel;
        let mut minus = pixel;
        plus[axis] += h;
        minus[axis] -= h;
        let fp = f(plus);
        let fm = f(minus);
        [
            (fp[0] - fm[0]) / (2.0 * h),
            (fp[1] - fm[1]) / (2.0 * h),
            (fp[2] - fm[2]) / (2.0 * h),
        ]
    }

    #[test]
    fn lambda_sums_to_one_everywhere() {
        let tri = flat_triangle(1.0);
        let vp = Viewport::new(256, 256);
        for &p in &[[64.0, 64.0], [10.0, 200.0], [128.0, 30.0]] {
            let bary = barycentric_derivatives(&tri, vp, p).unwrap();
            let sum: f32 = bary.lambda.iter().sum();
            assert!((sum - 1.0).abs() < 1e-5, "sum={sum}");
        }
    }

    #[test]
    fn vertices_recover_canonical_barycentrics() {
        let tri = flat_triangle(1.0);
        let vp = Viewport::new(256, 256);
        // Vertex 0 projects to pixel-space top-left (0, 256) under the y-flip.
        let b0 = barycentric_derivatives(&tri, vp, [0.0, 256.0]).unwrap();
        assert!((b0.lambda[0] - 1.0).abs() < 1e-4);
        assert!(b0.lambda[1].abs() < 1e-4);
        assert!(b0.lambda[2].abs() < 1e-4);
    }

    #[test]
    fn derivatives_match_finite_difference_flat() {
        let tri = flat_triangle(1.0);
        let vp = Viewport::new(256, 256);
        let eval = |p: [f32; 2]| barycentric_derivatives(&tri, vp, p).unwrap().lambda;
        let p = [100.0, 90.0];
        let bary = barycentric_derivatives(&tri, vp, p).unwrap();
        let fd_x = central_difference(&eval, p, 0, 0.01);
        let fd_y = central_difference(&eval, p, 1, 0.01);
        for i in 0..3 {
            assert!(
                (bary.ddx[i] - fd_x[i]).abs() < 1e-3,
                "ddx[{i}] analytic={} fd={}",
                bary.ddx[i],
                fd_x[i]
            );
            assert!(
                (bary.ddy[i] - fd_y[i]).abs() < 1e-3,
                "ddy[{i}] analytic={} fd={}",
                bary.ddy[i],
                fd_y[i]
            );
        }
    }

    #[test]
    fn derivatives_match_finite_difference_perspective() {
        // A triangle tilted in depth so the three vertices have different w,
        // exercising the perspective-correction derivative terms.
        let tri = [
            ClipVertex::new([-0.5, -0.5, 0.0, 1.0]),
            ClipVertex::new([2.0, -1.0, 0.0, 2.0]),
            ClipVertex::new([-1.5, 3.0, 0.0, 3.0]),
        ];
        let vp = Viewport::new(512, 512);
        let eval = |p: [f32; 2]| barycentric_derivatives(&tri, vp, p).unwrap().lambda;
        let p = [240.0, 260.0];
        let bary = barycentric_derivatives(&tri, vp, p).unwrap();

        // Perspective weights still sum to one.
        let sum: f32 = bary.lambda.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "sum={sum}");

        let fd_x = central_difference(&eval, p, 0, 0.02);
        let fd_y = central_difference(&eval, p, 1, 0.02);
        for i in 0..3 {
            let rel_x = (bary.ddx[i] - fd_x[i]).abs() / fd_x[i].abs().max(1e-4);
            let rel_y = (bary.ddy[i] - fd_y[i]).abs() / fd_y[i].abs().max(1e-4);
            assert!(rel_x < 1e-2, "ddx[{i}] rel={rel_x}");
            assert!(rel_y < 1e-2, "ddy[{i}] rel={rel_y}");
        }
    }

    #[test]
    fn degenerate_triangle_returns_none() {
        // Three collinear clip vertices -> zero screen area.
        let tri = [
            ClipVertex::new([0.0, 0.0, 0.0, 1.0]),
            ClipVertex::new([0.2, 0.2, 0.0, 1.0]),
            ClipVertex::new([0.4, 0.4, 0.0, 1.0]),
        ];
        let vp = Viewport::new(128, 128);
        assert!(barycentric_derivatives(&tri, vp, [64.0, 64.0]).is_none());
    }

    #[test]
    fn behind_camera_vertex_returns_none() {
        let tri = [
            ClipVertex::new([-1.0, -1.0, 0.0, 1.0]),
            ClipVertex::new([1.0, -1.0, 0.0, 1.0]),
            ClipVertex::new([-1.0, 1.0, 0.0, -0.5]),
        ];
        let vp = Viewport::new(128, 128);
        assert!(barycentric_derivatives(&tri, vp, [10.0, 10.0]).is_none());
    }
}
