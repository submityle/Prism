//! Flipbook sub-frame interpolation, sub-`UV` atlas rectangles, and
//! motion-blur two-frame blending (design §15).
//!
//! This module is the *orthogonal complement* of
//! [`super::renderers`]'s `Flipbook`. That type answers a single question —
//! "which cell is a particle showing *right now*?" — by returning one nearest
//! `u32` frame index (`frame_for_age` / `frame_for_normalized`). It deliberately
//! owns *no* atlas geometry and *no* interpolation: a hard cut between cells.
//!
//! Smooth flipbooks and motion blur need two extra pieces that `Flipbook`
//! intentionally leaves out, and which this module supplies:
//!
//! 1. **Atlas `UV` rectangles.** [`AtlasLayout`] maps a flat cell index onto the
//!    `[u_min, v_min, u_max, v_max]` rectangle of a `columns × rows` sprite
//!    sheet, so a `GPU` sampler (or the `CPU` reference) can address one cell.
//! 2. **Two-frame blend weights.** [`FlipbookBlend`] and the `blend_for_*`
//!    functions expose the *fractional* phase between two adjacent frames, so a
//!    shader can sample both cells and `lerp` — the sub-`UV` blend that turns a
//!    stepped flipbook into a smooth, motion-blurred animation.
//!
//! Nothing here re-derives the single-index logic and nothing imports
//! `renderers`, keeping the two modules decoupled. [`WrapMode`] mirrors the
//! semantics of `renderers::FlipbookWrap` (`Clamp` holds the last frame, `Loop`
//! wraps modulo the frame count) without depending on that enum.
//!
//! Only `floor` is used from the transcendental-adjacent surface; every other
//! operation is plain arithmetic, so this `CPU` reference stays bit-reproducible
//! against a future `GPU` kernel.

/// Absolute tolerance for `f32` comparisons in this module's tests and any
/// caller that needs to treat blend weights as "effectively equal".
///
/// Bare `==` / `!=` on `f32` is avoided throughout; compare against this
/// epsilon instead.
pub const EPS: f32 = 1.0e-6;

/// The `columns × rows` cell grid of a flipbook sprite sheet (design §15).
///
/// A flat frame index is laid out row-major: `col = frame % columns` and
/// `row = frame / columns`. The `V` axis runs *top to bottom*, so `row 0`
/// is the top strip of the sheet and increasing `row` moves downward — the
/// convention used by most texture atlases and `GPU` samplers.
///
/// Every field is a `u32`, so this type derives [`Eq`] and [`Hash`] and can key
/// an atlas cache. A `0` in either dimension is treated as `1` for all
/// divisions, so the layout can never divide by zero.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct AtlasLayout {
    /// Number of cells across one row (`0` is treated as `1`).
    pub columns: u32,
    /// Number of cell rows in the sheet (`0` is treated as `1`).
    pub rows: u32,
}

impl AtlasLayout {
    /// Total addressable cells, `effective_columns * effective_rows`.
    ///
    /// A `0` dimension counts as `1`, so the count is always at least `1`.
    #[must_use]
    pub fn cell_count(self) -> u32 {
        self.columns.max(1) * self.rows.max(1)
    }

    /// The `[u_min, v_min, u_max, v_max]` `UV` rectangle of `frame_index`.
    ///
    /// `col = frame_index % columns` and `row = frame_index / columns`, both
    /// using the effective (non-zero) column count. Each cell spans
    /// `1.0 / columns` in `U` and `1.0 / rows` in `V`. Because `V` runs top
    /// to bottom, `row 0` yields `v_min = 0.0` (the top of the sheet).
    #[must_use]
    pub fn uv_rect(self, frame_index: u32) -> [f32; 4] {
        let columns = self.columns.max(1);
        let rows = self.rows.max(1);
        let col = frame_index % columns;
        let row = frame_index / columns;
        let inv_columns = 1.0 / columns as f32;
        let inv_rows = 1.0 / rows as f32;
        let u_min = col as f32 * inv_columns;
        let v_min = row as f32 * inv_rows;
        [u_min, v_min, u_min + inv_columns, v_min + inv_rows]
    }
}

/// How a flipbook behaves once the phase runs past the last frame (design §15).
///
/// These variants mirror `super::renderers`'s `FlipbookWrap` one-for-one —
/// [`WrapMode::Clamp`] corresponds to `FlipbookWrap::Clamp` and
/// [`WrapMode::Loop`] to `FlipbookWrap::Loop` — but this module keeps its own
/// field-less enum so the two files stay decoupled. Being field-less, it
/// derives [`Eq`] and [`Hash`].
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum WrapMode {
    /// Hold the final frame; the second blend frame collapses onto the first.
    Clamp,
    /// Wrap back to the first frame, blending the last cell into cell `0`.
    Loop,
}

/// A resolved two-frame blend: the pair of cells to sample and the weight
/// between them (design §15).
///
/// A shader samples `frame_a` and `frame_b` and mixes them with
/// `lerp(a, b, blend)`, where `blend` is the fractional phase in `0.0..1.0`.
/// `blend == 0.0` shows `frame_a` exactly; values approaching `1.0` approach
/// `frame_b`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FlipbookBlend {
    /// The floor frame — the cell the phase currently sits on.
    pub frame_a: u32,
    /// The next frame — the cell the phase is advancing toward.
    pub frame_b: u32,
    /// Fractional weight in `0.0..=1.0` from `frame_a` toward `frame_b`.
    pub blend: f32,
}

/// Resolves a continuous frame `phase` into a two-frame [`FlipbookBlend`].
///
/// `phase` is a floating frame counter (for example `age_seconds * fps`).
/// Negative phases clamp to `0.0`. `frame_a = floor(phase)`,
/// `frame_b = frame_a + 1`, and `blend = phase - floor(phase)` (the fract,
/// computed without a `fract` intrinsic). The wrap mode resolves the boundary:
/// [`WrapMode::Clamp`] pins both frames to the last cell, while
/// [`WrapMode::Loop`] takes each frame modulo `frames`.
///
/// With `frames == 0` the sheet is a single static cell: both frames are `0`
/// and `blend` is `0.0`.
#[must_use]
pub fn blend_for_phase(phase: f32, frames: u32, wrap: WrapMode) -> FlipbookBlend {
    if frames == 0 {
        return FlipbookBlend {
            frame_a: 0,
            frame_b: 0,
            blend: 0.0,
        };
    }
    let clamped = phase.max(0.0);
    let base = clamped.floor();
    // `clamped >= 0` and `base <= clamped`, so the fract lands in `0.0..1.0`.
    let fract = (clamped - base).clamp(0.0, 1.0);
    // `base >= 0`, so the cast floors toward zero safely.
    let raw = base as u32;
    let last = frames - 1;
    let (frame_a, frame_b) = match wrap {
        WrapMode::Clamp => (raw.min(last), raw.saturating_add(1).min(last)),
        WrapMode::Loop => (raw % frames, raw.saturating_add(1) % frames),
    };
    FlipbookBlend {
        frame_a,
        frame_b,
        blend: fract,
    }
}

/// Resolves a two-frame blend from a particle `age_seconds` at a given `FPS`.
///
/// The phase is `max(age_seconds, 0) * max(fps, 0)`, so negative ages and
/// negative rates both clamp to a still first frame. See [`blend_for_phase`]
/// for the wrap and `frames == 0` behaviour.
#[must_use]
pub fn blend_for_age(age_seconds: f32, fps: f32, frames: u32, wrap: WrapMode) -> FlipbookBlend {
    let phase = age_seconds.max(0.0) * fps.max(0.0);
    blend_for_phase(phase, frames, wrap)
}

/// Resolves a two-frame blend from a normalized life fraction in `0.0..=1.0`.
///
/// `life` is clamped to `0.0..=1.0` and scaled by `frames` to form the
/// phase, matching `renderers::Flipbook::frame_for_normalized`'s mapping.
/// At `life == 1.0` the phase equals `frames`, which [`WrapMode::Loop`]
/// returns to cell `0` and [`WrapMode::Clamp`] holds on the last cell.
#[must_use]
pub fn blend_for_life(life: f32, frames: u32, wrap: WrapMode) -> FlipbookBlend {
    let phase = life.clamp(0.0, 1.0) * frames as f32;
    blend_for_phase(phase, frames, wrap)
}

/// Combines an [`AtlasLayout`] with a [`FlipbookBlend`] into the two `UV`
/// rectangles and the blend weight a shader needs for a two-sample `lerp`.
///
/// Returns `(rect_a, rect_b, blend)`, where each rect is the
/// `[u_min, v_min, u_max, v_max]` of the corresponding frame and `blend` is the
/// weight from `rect_a` toward `rect_b`.
#[must_use]
pub fn sample_rects(layout: &AtlasLayout, blend: &FlipbookBlend) -> ([f32; 4], [f32; 4], f32) {
    (
        layout.uv_rect(blend.frame_a),
        layout.uv_rect(blend.frame_b),
        blend.blend,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() <= EPS
    }

    fn rect_approx(a: [f32; 4], b: [f32; 4]) -> bool {
        approx(a[0], b[0]) && approx(a[1], b[1]) && approx(a[2], b[2]) && approx(a[3], b[3])
    }

    #[test]
    fn uv_rect_corners_and_center() {
        let layout = AtlasLayout {
            columns: 4,
            rows: 4,
        };
        // Top-left cell.
        assert!(rect_approx(layout.uv_rect(0), [0.0, 0.0, 0.25, 0.25]));
        // Centre-ish cell: col 1, row 1.
        assert!(rect_approx(layout.uv_rect(5), [0.25, 0.25, 0.5, 0.5]));
        // Bottom-right cell: col 3, row 3.
        assert!(rect_approx(layout.uv_rect(15), [0.75, 0.75, 1.0, 1.0]));
        // Second cell of the top row confirms V runs top-to-bottom.
        assert!(rect_approx(layout.uv_rect(1), [0.25, 0.0, 0.5, 0.25]));
    }

    #[test]
    fn cell_count_multiplies_dimensions() {
        assert_eq!(
            AtlasLayout {
                columns: 4,
                rows: 4
            }
            .cell_count(),
            16
        );
        assert_eq!(
            AtlasLayout {
                columns: 8,
                rows: 1
            }
            .cell_count(),
            8
        );
    }

    #[test]
    fn zero_dimensions_avoid_division_by_zero() {
        let layout = AtlasLayout {
            columns: 0,
            rows: 0,
        };
        assert_eq!(layout.cell_count(), 1);
        assert!(rect_approx(layout.uv_rect(0), [0.0, 0.0, 1.0, 1.0]));
    }

    #[test]
    fn blend_phase_zero_half_and_near_one() {
        let zero = blend_for_phase(0.0, 8, WrapMode::Loop);
        assert_eq!((zero.frame_a, zero.frame_b), (0, 1));
        assert!(approx(zero.blend, 0.0));

        let half = blend_for_phase(0.5, 8, WrapMode::Loop);
        assert_eq!((half.frame_a, half.frame_b), (0, 1));
        assert!(approx(half.blend, 0.5));

        let near_one = blend_for_phase(0.999, 8, WrapMode::Loop);
        assert_eq!((near_one.frame_a, near_one.frame_b), (0, 1));
        assert!(approx(near_one.blend, 0.999));
    }

    #[test]
    fn loop_wraps_last_frame_to_zero() {
        let b = blend_for_phase(7.5, 8, WrapMode::Loop);
        assert_eq!((b.frame_a, b.frame_b), (7, 0));
        assert!(approx(b.blend, 0.5));
    }

    #[test]
    fn clamp_holds_last_frame() {
        let b = blend_for_phase(7.5, 8, WrapMode::Clamp);
        assert_eq!((b.frame_a, b.frame_b), (7, 7));
        assert!(approx(b.blend, 0.5));

        let past = blend_for_phase(20.0, 8, WrapMode::Clamp);
        assert_eq!((past.frame_a, past.frame_b), (7, 7));
    }

    #[test]
    fn zero_frames_is_static() {
        let b = blend_for_phase(3.0, 0, WrapMode::Loop);
        assert_eq!((b.frame_a, b.frame_b), (0, 0));
        assert!(approx(b.blend, 0.0));
    }

    #[test]
    fn negative_phase_clamps_to_zero() {
        let b = blend_for_phase(-5.0, 8, WrapMode::Loop);
        assert_eq!((b.frame_a, b.frame_b), (0, 1));
        assert!(approx(b.blend, 0.0));
    }

    #[test]
    fn blend_for_age_scales_by_fps() {
        let b = blend_for_age(1.0, 4.0, 8, WrapMode::Loop);
        assert_eq!((b.frame_a, b.frame_b), (4, 5));
        assert!(approx(b.blend, 0.0));

        // Negative age clamps to a still first frame.
        let neg = blend_for_age(-1.0, 4.0, 8, WrapMode::Loop);
        assert_eq!((neg.frame_a, neg.frame_b), (0, 1));
        assert!(approx(neg.blend, 0.0));
    }

    #[test]
    fn blend_for_life_maps_normalized_range() {
        let mid = blend_for_life(0.5, 8, WrapMode::Loop);
        assert_eq!((mid.frame_a, mid.frame_b), (4, 5));
        assert!(approx(mid.blend, 0.0));

        // life == 1.0 → phase == frames, wrapping to cell 0 under Loop.
        let end_loop = blend_for_life(1.0, 8, WrapMode::Loop);
        assert_eq!((end_loop.frame_a, end_loop.frame_b), (0, 1));

        // …and holding the last cell under Clamp.
        let end_clamp = blend_for_life(1.0, 8, WrapMode::Clamp);
        assert_eq!((end_clamp.frame_a, end_clamp.frame_b), (7, 7));
    }

    #[test]
    fn sample_rects_pairs_frames_and_weight() {
        let layout = AtlasLayout {
            columns: 4,
            rows: 4,
        };
        let blend = FlipbookBlend {
            frame_a: 0,
            frame_b: 1,
            blend: 0.5,
        };
        let (rect_a, rect_b, weight) = sample_rects(&layout, &blend);
        assert!(rect_approx(rect_a, [0.0, 0.0, 0.25, 0.25]));
        assert!(rect_approx(rect_b, [0.25, 0.0, 0.5, 0.25]));
        assert!(approx(weight, 0.5));
    }
}
