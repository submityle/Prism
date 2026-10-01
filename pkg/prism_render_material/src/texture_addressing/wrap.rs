//! Per-axis wrap-mode transforms and the UV-level [`address_uv`] helper.

/// Sampler address (wrap) mode for one texture axis, mirroring the standard
/// graphics-API addressing modes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WrapMode {
    /// Tile the texture: `c -> c - floor(c)` (period 1).
    Repeat,
    /// Clamp to the `[0, 1]` edge.
    ClampToEdge,
    /// Tile with every other copy mirrored (period 2 triangle wave).
    MirroredRepeat,
    /// Mirror once about 0 then clamp: `clamp(|c|, 0, 1)`.
    MirrorClampToEdge,
    /// Clamp coordinate but flag out-of-range samples for a border-colour fetch.
    ClampToBorder,
}

/// Result of addressing one UV: the folded coordinate plus whether the sample
/// falls on the sampler border colour.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AddressResult {
    /// Folded UV, each component in `[0, 1]`.
    pub uv: [f32; 2],
    /// `true` when a [`WrapMode::ClampToBorder`] axis was out of `[0, 1]`.
    pub border: bool,
}

/// Fold one normalized coordinate through `mode`.
///
/// Returns `(coord, border)` where `coord` is in `[0, 1]` and `border` is only
/// ever `true` for [`WrapMode::ClampToBorder`] out-of-range inputs. A non-finite
/// input collapses to `0.0` (non-border).
#[must_use]
pub fn wrap_coord(c: f32, mode: WrapMode) -> (f32, bool) {
    if !c.is_finite() {
        return (0.0, false);
    }
    match mode {
        WrapMode::Repeat => {
            // rem_euclid keeps negatives periodic; result lands in [0, 1).
            (c.rem_euclid(1.0), false)
        }
        WrapMode::ClampToEdge => (c.clamp(0.0, 1.0), false),
        WrapMode::MirroredRepeat => {
            // Period-2 triangle wave: fold [1, 2) back down to (0, 1].
            let t = c.rem_euclid(2.0);
            let m = if t > 1.0 { 2.0 - t } else { t };
            (m.clamp(0.0, 1.0), false)
        }
        WrapMode::MirrorClampToEdge => (c.abs().clamp(0.0, 1.0), false),
        WrapMode::ClampToBorder => {
            let border = c < 0.0 || c > 1.0;
            (c.clamp(0.0, 1.0), border)
        }
    }
}

/// Address a full UV with independent per-axis wrap modes.
///
/// The `border` flag is set when *either* axis reports a border sample, since a
/// single border axis forces the border colour for the whole fetch.
#[must_use]
pub fn address_uv(uv: [f32; 2], mode_u: WrapMode, mode_v: WrapMode) -> AddressResult {
    let (u, bu) = wrap_coord(uv[0], mode_u);
    let (v, bv) = wrap_coord(uv[1], mode_v);
    AddressResult {
        uv: [u, v],
        border: bu || bv,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1.0e-6;

    #[test]
    fn repeat_is_periodic() {
        assert!((wrap_coord(1.25, WrapMode::Repeat).0 - 0.25).abs() < EPS);
        assert!((wrap_coord(-0.25, WrapMode::Repeat).0 - 0.75).abs() < EPS);
        assert!((wrap_coord(3.0, WrapMode::Repeat).0).abs() < EPS);
    }

    #[test]
    fn clamp_to_edge_saturates() {
        assert_eq!(wrap_coord(-2.0, WrapMode::ClampToEdge), (0.0, false));
        assert_eq!(wrap_coord(5.0, WrapMode::ClampToEdge), (1.0, false));
        assert!((wrap_coord(0.4, WrapMode::ClampToEdge).0 - 0.4).abs() < EPS);
    }

    #[test]
    fn mirrored_repeat_is_triangle_wave() {
        // 1.25 mirrors to 0.75; 1.75 mirrors to 0.25.
        assert!((wrap_coord(1.25, WrapMode::MirroredRepeat).0 - 0.75).abs() < EPS);
        assert!((wrap_coord(1.75, WrapMode::MirroredRepeat).0 - 0.25).abs() < EPS);
        // Negative side mirrors symmetrically: -0.25 -> 0.25.
        assert!((wrap_coord(-0.25, WrapMode::MirroredRepeat).0 - 0.25).abs() < EPS);
    }

    #[test]
    fn mirror_clamp_to_edge_folds_negative_then_clamps() {
        assert!((wrap_coord(-0.3, WrapMode::MirrorClampToEdge).0 - 0.3).abs() < EPS);
        assert_eq!(wrap_coord(-9.0, WrapMode::MirrorClampToEdge), (1.0, false));
    }

    #[test]
    fn clamp_to_border_flags_out_of_range() {
        assert_eq!(wrap_coord(0.5, WrapMode::ClampToBorder), (0.5, false));
        assert_eq!(wrap_coord(1.5, WrapMode::ClampToBorder), (1.0, true));
        assert_eq!(wrap_coord(-0.1, WrapMode::ClampToBorder), (0.0, true));
    }

    #[test]
    fn non_finite_collapses_to_zero() {
        assert_eq!(wrap_coord(f32::NAN, WrapMode::Repeat), (0.0, false));
        assert_eq!(wrap_coord(f32::INFINITY, WrapMode::ClampToBorder), (0.0, false));
    }

    #[test]
    fn address_uv_combines_axes_and_border() {
        let r = address_uv([1.25, -0.25], WrapMode::Repeat, WrapMode::Repeat);
        assert!((r.uv[0] - 0.25).abs() < EPS);
        assert!((r.uv[1] - 0.75).abs() < EPS);
        assert!(!r.border);

        // One border axis forces the whole fetch to border.
        let b = address_uv([0.5, 2.0], WrapMode::ClampToEdge, WrapMode::ClampToBorder);
        assert!(b.border);
        assert_eq!(b.uv, [0.5, 1.0]);
    }
}
