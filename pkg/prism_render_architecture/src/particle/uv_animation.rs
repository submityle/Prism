//! Continuous `UV`-coordinate animation for `Sprite`/material sampling: scroll,
//! tile, rotate, and flowmap-style dual panners (design §15).
//!
//! This module owns exactly one contract — the `CPU`-verifiable maths that maps
//! an incoming texture coordinate `uv` to an outgoing one — and it is
//! deliberately orthogonal to the four neighbours it composes with:
//!
//! * [`super::renderers`] owns *quad orientation*: how a billboard faces the
//!   camera or aligns to velocity. It decides where the quad's corners land in
//!   clip space; it never touches the `UV`s sampled inside those corners. (This
//!   module intentionally does **not** import it.)
//! * [`super::renderers`]'s `Flipbook` owns *discrete frames*: it advances an
//!   integer sub-image index over time and sets the cell a sprite reads. That
//!   is a stepwise atlas selection, not a continuous coordinate warp.
//! * [`super::curves`] owns *scalar over-life* look-up tables: it sweeps a
//!   single authored value along a particle's age. It is 1D and stepless in
//!   value, but it is not a 2D coordinate transform.
//! * `sprite_stretch` owns *quad size*: how wide/tall the billboard is drawn.
//!   It scales geometry, not texture coordinates.
//!
//! The missing orthogonal piece is *continuous `UV` transformation*: sliding,
//! scaling-about-a-pivot, and rotating the sample coordinate every frame, plus
//! the twin-offset blend a flowmap needs. A scrolling lava crust, a swirling
//! portal, a panning cloud sheet, and a distortion flowmap are all this module
//! layered on top of — never inside — the four neighbours above. Because every
//! routine is a pure function of `[f32; 2]` coordinates, the same transform
//! composes with any orientation, any flipbook frame, any over-life curve, and
//! any quad size without either side knowing about the other.
//!
//! # Coordinate convention
//!
//! `UV`s are plain `[f32; 2]` arrays (`[u, v]`); no `Vec3`/`Vec2` type is
//! imported, since a 2D coordinate warp never needs 3D vector algebra. The few
//! 2D operations used (componentwise add/scale and a 2x2 rotation) are written
//! inline as array arithmetic.
//!
//! # Determinism
//!
//! Rotation is expressed with a caller-supplied `(cos_rot, sin_rot)` pair: this
//! module performs **no** transcendental maths and never calls `sin`/`cos`/
//! `tan` itself. The only floating-point primitive beyond ordinary arithmetic
//! is `f32::floor` (used by [`wrap01`] to fold a coordinate back into the
//! half-open unit range via `x - x.floor()`). With no transcendental calls the
//! `CPU` reference stays bit-reproducible against a future `GPU` sampler,
//! matching the determinism contract of the sibling [`super::curves`] module.

/// Absolute tolerance for the `f32` decisions in this module.
///
/// Used to snap a folded coordinate that lands on (or a hair above) `1.0` back
/// to `0.0` so [`wrap01`] stays strictly half-open, and by the tests to assert
/// coordinate equality with slack rather than a forbidden bare `==`.
pub const EPS: f32 = 1e-6;

/// A continuous `UV`-coordinate transform: pivoted tile + rotate, then a global
/// scroll offset.
///
/// Rotation is stored as its already-evaluated `(cos_rot, sin_rot)` pair rather
/// than an angle, so applying the transform needs no trigonometry (see the
/// module-level determinism note). The intended application order is encoded by
/// [`UvTransform::apply`]: subtract `pivot`, scale by `tiling`, rotate by
/// `(cos_rot, sin_rot)`, add `pivot` back, then add `scroll`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UvTransform {
    /// Global translation added last, after the pivoted tile/rotate. A
    /// time-driven scroll (see [`scroll_offset`]) feeds this field.
    pub scroll: [f32; 2],
    /// Per-axis scale applied about `pivot` (`> 1` repeats the texture more
    /// often, `< 1` zooms in). Componentwise on `[u, v]`.
    pub tiling: [f32; 2],
    /// The fixed point that `tiling` and the rotation act about; it maps to
    /// itself under the tile+rotate stage.
    pub pivot: [f32; 2],
    /// Cosine of the rotation angle, supplied by the caller.
    pub cos_rot: f32,
    /// Sine of the rotation angle, supplied by the caller.
    pub sin_rot: f32,
}

impl UvTransform {
    /// The identity transform: no scroll, unit `tiling`, `pivot` at the origin,
    /// and a zero-angle rotation (`cos_rot = 1`, `sin_rot = 0`).
    #[must_use]
    pub const fn identity() -> Self {
        Self {
            scroll: [0.0, 0.0],
            tiling: [1.0, 1.0],
            pivot: [0.0, 0.0],
            cos_rot: 1.0,
            sin_rot: 0.0,
        }
    }

    /// Builds a transform from its parts.
    ///
    /// `cos_rot`/`sin_rot` are taken as-is; the caller is responsible for
    /// passing a valid `(cos, sin)` pair (this module computes no trig).
    #[must_use]
    pub const fn new(
        scroll: [f32; 2],
        tiling: [f32; 2],
        pivot: [f32; 2],
        cos_rot: f32,
        sin_rot: f32,
    ) -> Self {
        Self {
            scroll,
            tiling,
            pivot,
            cos_rot,
            sin_rot,
        }
    }

    /// Applies the full transform to `uv`.
    ///
    /// The order is fixed: subtract `pivot`, scale by `tiling`, rotate by the
    /// stored `(cos_rot, sin_rot)`, add `pivot` back, then add `scroll`. The
    /// `pivot` is therefore a fixed point of the tile+rotate stage, so a `uv`
    /// equal to `pivot` returns `pivot + scroll`.
    #[must_use]
    pub fn apply(&self, uv: [f32; 2]) -> [f32; 2] {
        // Into pivot-local space, then scale about the pivot.
        let su = (uv[0] - self.pivot[0]) * self.tiling[0];
        let sv = (uv[1] - self.pivot[1]) * self.tiling[1];
        // 2x2 rotation with the caller-supplied cosine/sine.
        let ru = self.cos_rot * su - self.sin_rot * sv;
        let rv = self.sin_rot * su + self.cos_rot * sv;
        // Back out of pivot-local space, then apply the global scroll.
        [
            ru + self.pivot[0] + self.scroll[0],
            rv + self.pivot[1] + self.scroll[1],
        ]
    }
}

/// Translates `uv` by `offset` (the atomic scroll/pan step).
///
/// This is the componentwise addition that [`UvTransform::apply`] performs last;
/// exposed on its own so a caller can pan a coordinate without building a full
/// [`UvTransform`].
#[must_use]
pub fn uv_scroll(uv: [f32; 2], offset: [f32; 2]) -> [f32; 2] {
    [uv[0] + offset[0], uv[1] + offset[1]]
}

/// Scales `uv` by `tiling` about `pivot` (the atomic tile step).
///
/// `pivot` is a fixed point: `uv_tile(pivot, tiling, pivot) == pivot` for any
/// `tiling`. Larger `tiling` values repeat the source texture more often.
#[must_use]
pub fn uv_tile(uv: [f32; 2], tiling: [f32; 2], pivot: [f32; 2]) -> [f32; 2] {
    [
        (uv[0] - pivot[0]) * tiling[0] + pivot[0],
        (uv[1] - pivot[1]) * tiling[1] + pivot[1],
    ]
}

/// Rotates `uv` about `pivot` using the caller-supplied `(cos_rot, sin_rot)`
/// (the atomic rotate step).
///
/// Uses the standard 2x2 rotation `u' = c*u - s*v`, `v' = s*u + c*v` on the
/// pivot-local coordinate. No trigonometry is computed here; pass `cos_rot = 1`,
/// `sin_rot = 0` for the identity.
#[must_use]
pub fn uv_rotate(uv: [f32; 2], cos_rot: f32, sin_rot: f32, pivot: [f32; 2]) -> [f32; 2] {
    let lu = uv[0] - pivot[0];
    let lv = uv[1] - pivot[1];
    [
        cos_rot * lu - sin_rot * lv + pivot[0],
        sin_rot * lu + cos_rot * lv + pivot[1],
    ]
}

/// Computes a time-driven scroll offset as `velocity * time` (a linear panner).
///
/// Feed the result into [`UvTransform::scroll`] or [`uv_scroll`] to slide a
/// texture at a constant per-axis speed. Linear in `time`, so doubling `time`
/// doubles the offset.
#[must_use]
pub fn scroll_offset(velocity: [f32; 2], time: f32) -> [f32; 2] {
    [velocity[0] * time, velocity[1] * time]
}

/// Folds a single coordinate back into the half-open unit range `[0, 1)`.
///
/// Uses `x - x.floor()` (no transcendental maths), which is correct for
/// negative inputs: `wrap01(-0.25) == 0.75`. Floating-point rounding can make
/// that difference land on `1.0` for tiny negative inputs, so a result within
/// [`EPS`] of `1.0` (or above it) is snapped to `0.0` to keep the range strictly
/// half-open.
#[must_use]
pub fn wrap01(x: f32) -> f32 {
    let f = x - x.floor();
    if f >= 1.0 - EPS {
        0.0
    } else {
        f
    }
}

/// Folds both components of `uv` into `[0, 1)` via [`wrap01`].
///
/// Applied after scrolling/tiling so a repeating texture samples inside its
/// atlas cell regardless of how far the coordinate has drifted.
#[must_use]
pub fn wrap_uv(uv: [f32; 2]) -> [f32; 2] {
    [wrap01(uv[0]), wrap01(uv[1])]
}

/// Prepares a flowmap-style dual panner: two independently scrolled copies of
/// `uv` plus a blend weight to `lerp` between the two samples.
///
/// Returns `(uv + offset_a, uv + offset_b, blend)` where `blend` is clamped to
/// `[0, 1]`. The caller samples the texture twice (once per returned `UV`) and
/// mixes the results with `blend` as `a * (1 - blend) + b * blend`; alternating
/// the two offsets and crossfading hides the seam of a single scroll wrapping
/// around. This module only produces the coordinates and weight — it never
/// samples a texture itself.
#[must_use]
pub fn dual_panner(
    uv: [f32; 2],
    offset_a: [f32; 2],
    offset_b: [f32; 2],
    blend: f32,
) -> ([f32; 2], [f32; 2], f32) {
    (
        uv_scroll(uv, offset_a),
        uv_scroll(uv, offset_b),
        blend.clamp(0.0, 1.0),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Asserts two `f32`s are within [`EPS`] (avoids a forbidden bare `==`).
    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() <= EPS
    }

    /// Asserts two `UV`s match componentwise within [`EPS`].
    fn approx_uv(a: [f32; 2], b: [f32; 2]) -> bool {
        approx(a[0], b[0]) && approx(a[1], b[1])
    }

    #[test]
    fn scroll_translates_componentwise() {
        let out = uv_scroll([0.2, 0.3], [0.5, -0.1]);
        assert!(approx_uv(out, [0.7, 0.2]));
    }

    #[test]
    fn scroll_offset_is_linear_in_time() {
        let v = [0.25, -0.5];
        let o1 = scroll_offset(v, 1.0);
        let o2 = scroll_offset(v, 2.0);
        assert!(approx_uv(o1, [0.25, -0.5]));
        // Doubling time doubles the offset.
        assert!(approx_uv(o2, [o1[0] * 2.0, o1[1] * 2.0]));
        // Zero time yields zero offset.
        assert!(approx_uv(scroll_offset(v, 0.0), [0.0, 0.0]));
    }

    #[test]
    fn tile_scales_about_pivot() {
        let pivot = [0.5, 0.5];
        // A point one unit from the pivot doubles its distance under 2x tiling.
        let out = uv_tile([1.0, 0.5], [2.0, 2.0], pivot);
        assert!(approx_uv(out, [1.5, 0.5]));
    }

    #[test]
    fn tile_leaves_pivot_fixed() {
        let pivot = [0.3, 0.7];
        let out = uv_tile(pivot, [4.0, 0.25], pivot);
        assert!(approx_uv(out, pivot));
    }

    #[test]
    fn rotate_90_degrees_maps_x_to_y() {
        // cos = 0, sin = 1 is a +90 degree rotation: (1, 0) -> (0, 1).
        let out = uv_rotate([1.0, 0.0], 0.0, 1.0, [0.0, 0.0]);
        assert!(approx_uv(out, [0.0, 1.0]));
    }

    #[test]
    fn rotate_identity_is_no_op() {
        // cos = 1, sin = 0 leaves the coordinate untouched.
        let uv = [0.42, -0.17];
        let out = uv_rotate(uv, 1.0, 0.0, [0.5, 0.5]);
        assert!(approx_uv(out, uv));
    }

    #[test]
    fn rotate_leaves_pivot_fixed() {
        let pivot = [0.25, 0.75];
        let out = uv_rotate(pivot, 0.0, 1.0, pivot);
        assert!(approx_uv(out, pivot));
    }

    #[test]
    fn apply_follows_pivot_scale_rotate_scroll_order() {
        // pivot at origin, 3x uniform tiling, +90 degree rotation, then scroll.
        let t = UvTransform::new([0.1, 0.2], [3.0, 3.0], [0.0, 0.0], 0.0, 1.0);
        // uv = (1, 0): scale -> (3, 0); rotate +90 -> (0, 3); +scroll -> (0.1, 3.2).
        let out = t.apply([1.0, 0.0]);
        assert!(approx_uv(out, [0.1, 3.2]));
    }

    #[test]
    fn apply_pivot_maps_to_pivot_plus_scroll() {
        let pivot = [0.5, 0.5];
        let scroll = [0.05, -0.05];
        // Non-trivial tiling/rotation must not move the pivot before scroll.
        let t = UvTransform::new(scroll, [7.0, 0.5], pivot, 0.0, 1.0);
        let out = t.apply(pivot);
        assert!(approx_uv(out, [pivot[0] + scroll[0], pivot[1] + scroll[1]]));
    }

    #[test]
    fn identity_transform_is_no_op() {
        let uv = [0.33, 0.66];
        assert!(approx_uv(UvTransform::identity().apply(uv), uv));
    }

    #[test]
    fn wrap01_folds_positive_values() {
        assert!(approx(wrap01(0.25), 0.25));
        assert!(approx(wrap01(1.25), 0.25));
        assert!(approx(wrap01(3.5), 0.5));
        // Exact integers fold to 0, staying inside the half-open range.
        assert!(approx(wrap01(1.0), 0.0));
        assert!(approx(wrap01(0.0), 0.0));
    }

    #[test]
    fn wrap01_folds_negative_values() {
        assert!(approx(wrap01(-0.25), 0.75));
        assert!(approx(wrap01(-1.25), 0.75));
        assert!(approx(wrap01(-1.0), 0.0));
    }

    #[test]
    fn wrap01_result_is_half_open() {
        // The result must never reach 1.0 for any of a spread of inputs.
        let samples = [-2.0, -0.5, -1e-7, 0.0, 0.999_999, 1.0, 2.75, 10.5];
        for x in samples {
            let w = wrap01(x);
            // Strictly half-open: 0.0 <= w < 1.0 for every input.
            assert!((0.0..1.0).contains(&w));
        }
    }

    #[test]
    fn wrap_uv_folds_both_axes() {
        let out = wrap_uv([1.25, -0.25]);
        assert!(approx_uv(out, [0.25, 0.75]));
    }

    #[test]
    fn dual_panner_returns_two_offsets_and_clamped_weight() {
        let (a, b, w) = dual_panner([0.1, 0.2], [0.5, 0.0], [0.0, 0.5], 0.3);
        assert!(approx_uv(a, [0.6, 0.2]));
        assert!(approx_uv(b, [0.1, 0.7]));
        assert!(approx(w, 0.3));
    }

    #[test]
    fn dual_panner_clamps_blend_weight() {
        let (_, _, hi) = dual_panner([0.0, 0.0], [0.0, 0.0], [0.0, 0.0], 1.5);
        let (_, _, lo) = dual_panner([0.0, 0.0], [0.0, 0.0], [0.0, 0.0], -0.5);
        assert!(approx(hi, 1.0));
        assert!(approx(lo, 0.0));
    }
}
