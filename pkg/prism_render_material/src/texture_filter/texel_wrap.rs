//! Integer-texel wrap addressing for manual bilinear fetch.
//!
//! [`super::super::texture_addressing`] folds a *normalized* UV through a wrap
//! mode before mip/page math. Bilinear filtering then needs the four integer
//! neighbour texels around the sample position, and those indices can fall just
//! outside `[0, dim)` even for an in-range UV (the `-0.5` texel-centre offset
//! straddles an edge). This module folds a single integer texel index through
//! the same wrap semantics at texel granularity, matching how hardware fetches
//! each bilinear corner, so a CPU golden reproduces a GPU twin exactly.
//!
//! # Conventions
//! * `dim` is the axis size at the sampled mip, clamped to `>= 1` by callers.
//! * The returned [`TexelAddr`] is either an in-range index or
//!   [`TexelAddr::Border`], which only [`WrapMode::ClampToBorder`] can produce
//!   and which tells bilinear to blend the sampler border colour for that
//!   corner.
//! * Semantics mirror [`WrapMode`] / `wrap_coord` exactly, lifted to integers.
//!
//! # References
//! * Vulkan `VkSamplerAddressMode`; OpenGL `GL_TEXTURE_WRAP_*`.

use super::super::texture_addressing::WrapMode;

/// A wrapped texel index, or a signal to use the border colour.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TexelAddr {
    /// In-range texel index within `[0, dim)`.
    In(u32),
    /// Out-of-range under [`WrapMode::ClampToBorder`]: use the border colour.
    Border,
}

/// Fold a (possibly out-of-range) integer texel index through `mode`.
///
/// `dim` is clamped to `>= 1`. Every mode except [`WrapMode::ClampToBorder`]
/// always yields [`TexelAddr::In`].
#[must_use]
pub fn wrap_texel(i: i64, dim: u32, mode: WrapMode) -> TexelAddr {
    let dim = dim.max(1) as i64;
    match mode {
        WrapMode::Repeat => TexelAddr::In(i.rem_euclid(dim) as u32),
        WrapMode::ClampToEdge => TexelAddr::In(i.clamp(0, dim - 1) as u32),
        WrapMode::MirroredRepeat => {
            // Period 2*dim triangle wave over the integer lattice.
            let period = 2 * dim;
            let m = i.rem_euclid(period);
            let folded = if m >= dim { period - 1 - m } else { m };
            TexelAddr::In(folded.clamp(0, dim - 1) as u32)
        }
        WrapMode::MirrorClampToEdge => {
            // Reflect negatives about -0.5 (so -1 -> 0), then clamp to the edge.
            let reflected = if i < 0 { -1 - i } else { i };
            TexelAddr::In(reflected.clamp(0, dim - 1) as u32)
        }
        WrapMode::ClampToBorder => {
            if i < 0 || i >= dim {
                TexelAddr::Border
            } else {
                TexelAddr::In(i as u32)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repeat_wraps_both_directions() {
        assert_eq!(wrap_texel(-1, 4, WrapMode::Repeat), TexelAddr::In(3));
        assert_eq!(wrap_texel(4, 4, WrapMode::Repeat), TexelAddr::In(0));
        assert_eq!(wrap_texel(5, 4, WrapMode::Repeat), TexelAddr::In(1));
    }

    #[test]
    fn clamp_to_edge_pins_to_endpoints() {
        assert_eq!(wrap_texel(-3, 4, WrapMode::ClampToEdge), TexelAddr::In(0));
        assert_eq!(wrap_texel(99, 4, WrapMode::ClampToEdge), TexelAddr::In(3));
    }

    #[test]
    fn mirrored_repeat_reflects() {
        // dim 4: indices 0,1,2,3 then mirror 3,2,1,0 ...
        assert_eq!(wrap_texel(4, 4, WrapMode::MirroredRepeat), TexelAddr::In(3));
        assert_eq!(wrap_texel(5, 4, WrapMode::MirroredRepeat), TexelAddr::In(2));
        assert_eq!(wrap_texel(-1, 4, WrapMode::MirroredRepeat), TexelAddr::In(0));
    }

    #[test]
    fn mirror_clamp_to_edge_reflects_once_then_clamps() {
        assert_eq!(wrap_texel(-1, 4, WrapMode::MirrorClampToEdge), TexelAddr::In(0));
        assert_eq!(wrap_texel(-2, 4, WrapMode::MirrorClampToEdge), TexelAddr::In(1));
        assert_eq!(wrap_texel(99, 4, WrapMode::MirrorClampToEdge), TexelAddr::In(3));
    }

    #[test]
    fn clamp_to_border_flags_out_of_range() {
        assert_eq!(wrap_texel(-1, 4, WrapMode::ClampToBorder), TexelAddr::Border);
        assert_eq!(wrap_texel(4, 4, WrapMode::ClampToBorder), TexelAddr::Border);
        assert_eq!(wrap_texel(2, 4, WrapMode::ClampToBorder), TexelAddr::In(2));
    }

    #[test]
    fn zero_dim_is_safe() {
        assert_eq!(wrap_texel(7, 0, WrapMode::Repeat), TexelAddr::In(0));
    }
}
