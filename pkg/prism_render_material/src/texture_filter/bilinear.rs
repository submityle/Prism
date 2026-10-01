//! Manual bilinear fetch from a user-supplied texel source.
//!
//! A visibility-buffer or ray-traced shading path has no fixed-function
//! sampler, so it must fetch the four neighbour texels around a sample position
//! and blend them itself. This module performs that blend in closed form from
//! raw texel reads provided through [`TexelSource`], folding each corner's
//! integer index through [`super::texel_wrap`] so tiling and border textures
//! match hardware. Pure `f32` arithmetic, no AI/ML -- a CPU golden reproduces a
//! GPU twin exactly.
//!
//! # Conventions
//! * Texel centres sit at half-integer positions: the sample position in texels
//!   is `uv * dim - 0.5`, so `uv = (k + 0.5)/dim` reads texel `k` exactly.
//! * The axis dimension at `mip` is `max(1, base_dim >> mip)`, matching
//!   [`super::super::texture_lod`] mip sizing.
//! * A [`WrapMode::ClampToBorder`] corner contributes the caller's border
//!   colour (passed to [`bilinear`]) with its bilinear weight.
//! * Non-finite UVs collapse to `0.0`, never `NaN`.
//!
//! # References
//! * Akenine-Moller et al., *Real-Time Rendering* 4th ed., Section 6.2.1.
//! * OpenGL/Vulkan `LINEAR` minification/magnification filter definition.

use super::super::texture_addressing::WrapMode;
use super::texel_wrap::{wrap_texel, TexelAddr};

/// A read-only source of raw RGBA texels for the manual sampler.
///
/// Implementors expose base (mip-0) dimensions and a single-texel read at an
/// integer coordinate on a mip level. Coordinates passed to [`TexelSource::texel`]
/// are already wrapped into `[0, dim_at_mip)`; the source never has to clamp.
pub trait TexelSource {
    /// Base (mip-0) dimensions `(width, height)` in texels.
    fn dimensions(&self) -> (u32, u32);

    /// Read one RGBA texel at `(x, y)` on `mip`. Inputs are pre-wrapped into the
    /// valid range for that mip by the caller.
    fn texel(&self, mip: u32, x: u32, y: u32) -> [f32; 4];
}

/// Axis dimension (texels) at `mip`, floored to `>= 1`.
#[inline]
#[must_use]
pub(crate) fn dim_at(base: u32, mip: u32) -> u32 {
    (base >> mip.min(31)).max(1)
}

#[inline]
fn finite_or_zero(x: f32) -> f32 {
    if x.is_finite() { x } else { 0.0 }
}

#[inline]
fn lerp4(a: [f32; 4], b: [f32; 4], t: f32) -> [f32; 4] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
        a[3] + (b[3] - a[3]) * t,
    ]
}

/// Bilinearly sample `src` at mip `mip` for UV `uv`, folding corner indices
/// through the per-axis wrap modes and substituting `border_color` for any
/// [`WrapMode::ClampToBorder`] corner out of range.
#[must_use]
pub fn bilinear<S: TexelSource>(
    src: &S,
    mip: u32,
    uv: [f32; 2],
    wrap_u: WrapMode,
    wrap_v: WrapMode,
    border_color: [f32; 4],
) -> [f32; 4] {
    let (bw, bh) = src.dimensions();
    let w = dim_at(bw.max(1), mip);
    let h = dim_at(bh.max(1), mip);

    // Sample position in texel space with the half-texel centre offset.
    let fx = finite_or_zero(uv[0]) * w as f32 - 0.5;
    let fy = finite_or_zero(uv[1]) * h as f32 - 0.5;
    let x0 = fx.floor();
    let y0 = fy.floor();
    let tx = fx - x0;
    let ty = fy - y0;
    let (x0, y0) = (x0 as i64, y0 as i64);

    // Resolve the four corner texels (or border contributions).
    let fetch = |ix: i64, iy: i64| -> [f32; 4] {
        match (wrap_texel(ix, w, wrap_u), wrap_texel(iy, h, wrap_v)) {
            (TexelAddr::In(cx), TexelAddr::In(cy)) => src.texel(mip, cx, cy),
            _ => border_color,
        }
    };
    let c00 = fetch(x0, y0);
    let c10 = fetch(x0 + 1, y0);
    let c01 = fetch(x0, y0 + 1);
    let c11 = fetch(x0 + 1, y0 + 1);

    let top = lerp4(c00, c10, tx);
    let bottom = lerp4(c01, c11, tx);
    lerp4(top, bottom, ty)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny 2x2 (mip 0) source: distinct colour per texel, constant per mip.
    struct Checker;
    impl TexelSource for Checker {
        fn dimensions(&self) -> (u32, u32) {
            (2, 2)
        }
        fn texel(&self, _mip: u32, x: u32, y: u32) -> [f32; 4] {
            let v = (x + y * 2) as f32;
            [v, 0.0, 0.0, 1.0]
        }
    }

    #[test]
    fn texel_center_reads_exact_texel() {
        // uv = (0.5)/2 = 0.25 -> texel 0 centre on each axis.
        let c = bilinear(&Checker, 0, [0.25, 0.25], WrapMode::ClampToEdge, WrapMode::ClampToEdge, [0.0; 4]);
        assert!((c[0] - 0.0).abs() < 1.0e-6, "{c:?}");
        // uv = 1.5/2 = 0.75 -> texel 1 on x, texel 0 on y -> value 1.
        let c = bilinear(&Checker, 0, [0.75, 0.25], WrapMode::ClampToEdge, WrapMode::ClampToEdge, [0.0; 4]);
        assert!((c[0] - 1.0).abs() < 1.0e-6, "{c:?}");
    }

    #[test]
    fn midpoint_blends_two_texels() {
        // uv.x = 0.5 -> fx = 0.5 -> between texel 0 (val 0) and texel 1 (val 1).
        let c = bilinear(&Checker, 0, [0.5, 0.25], WrapMode::ClampToEdge, WrapMode::ClampToEdge, [0.0; 4]);
        assert!((c[0] - 0.5).abs() < 1.0e-6, "{c:?}");
    }

    #[test]
    fn center_blends_all_four() {
        // uv = (0.5, 0.5) -> fx=fy=0.5 -> mean of 0,1,2,3 = 1.5.
        let c = bilinear(&Checker, 0, [0.5, 0.5], WrapMode::ClampToEdge, WrapMode::ClampToEdge, [0.0; 4]);
        assert!((c[0] - 1.5).abs() < 1.0e-6, "{c:?}");
    }

    #[test]
    fn border_corner_uses_border_color() {
        // uv.x just inside 0 -> left neighbour index -1 -> border under ClampToBorder.
        let border = [9.0, 9.0, 9.0, 9.0];
        let c = bilinear(&Checker, 0, [0.0, 0.25], WrapMode::ClampToBorder, WrapMode::ClampToEdge, border);
        // x0 = floor(-0.5) = -1 (border), x1 = 0 (texel). tx = 0.5.
        // top = lerp(border=9, texel0=0, 0.5) = 4.5 on red.
        assert!((c[0] - 4.5).abs() < 1.0e-6, "{c:?}");
    }

    #[test]
    fn non_finite_uv_is_safe() {
        let c = bilinear(&Checker, 0, [f32::NAN, f32::INFINITY], WrapMode::Repeat, WrapMode::Repeat, [0.0; 4]);
        assert!(c.iter().all(|v| v.is_finite()));
    }
}
