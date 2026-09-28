//! Backend-neutral reference for weighted-blended order-independent
//! transparency (WBOIT).
//!
//! A visibility-buffer deferred renderer can only store one opaque surface per
//! pixel, so transparent geometry is composited in a separate forward pass.
//! Sorting every transparent triangle per pixel is impractical on the GPU, so
//! Prism uses `McGuire` & Bavoil's weighted-blended OIT ("Weighted Blended
//! Order-Independent Transparency", JCGT 2013): each fragment is accumulated
//! with a depth-derived weight into two render targets, and a final pass
//! normalises the accumulation and composites it over the opaque background.
//!
//! The scheme is *order independent* because accumulation is a commutative sum
//! and the revealage is a commutative product; this module encodes exactly the
//! arithmetic the GPU MRT blend + composite shader must reproduce, so it is the
//! golden reference for `oit.wesl`.

/// A premultiplied transparent fragment awaiting accumulation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OitFragment {
    /// Straight (non-premultiplied) linear RGB radiance leaving the fragment.
    pub color: [f32; 3],
    /// Coverage / opacity in `[0, 1]`.
    pub alpha: f32,
    /// Positive view-space distance from the camera, used by the weight.
    pub view_depth: f32,
}

impl OitFragment {
    /// Builds a fragment, clamping alpha into `[0, 1]` and depth to non-negative.
    pub fn new(color: [f32; 3], alpha: f32, view_depth: f32) -> Self {
        Self {
            color,
            alpha: alpha.clamp(0.0, 1.0),
            view_depth: view_depth.max(0.0),
        }
    }
}

/// Lower clamp on the depth weight; keeps distant fragments from vanishing.
const WEIGHT_MIN: f32 = 1.0e-2;
/// Upper clamp on the depth weight; keeps near fragments from saturating f16.
const WEIGHT_MAX: f32 = 3.0e3;
/// Guards the average-colour division when nothing has been accumulated.
const ACCUM_EPSILON: f32 = 1.0e-5;

/// Depth-based fragment weight from `McGuire` & Bavoil (2013), equation 10.
///
/// Nearer fragments (smaller `view_depth`) receive a larger weight so they
/// dominate the blend, approximating a front-to-back sort without ordering.
/// The result is scaled by `alpha` and clamped to a range that stays finite in
/// an `f16` accumulation target.
pub fn oit_weight(view_depth: f32, alpha: f32) -> f32 {
    let depth = view_depth.max(0.0);
    // (depth / 200)^4 in the denominator: unit-independent falloff tuned by
    // McGuire for metres-scale scenes.
    let scaled = depth * (1.0 / 200.0);
    let falloff = scaled * scaled * scaled * scaled;
    let weight = (0.03 / (ACCUM_EPSILON + falloff)).clamp(WEIGHT_MIN, WEIGHT_MAX);
    alpha.clamp(0.0, 1.0) * weight
}

/// The two-target WBOIT accumulation state for one pixel.
///
/// `accum` mirrors the RGBA16F accumulation target (premultiplied,
/// weighted colour in `rgb`, summed weighted alpha in `a`); `revealage`
/// mirrors the R16F revealage target (the running product of `1 - alpha`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OitAccumulation {
    /// Weighted premultiplied colour (`rgb`) and summed weighted alpha (`a`).
    pub accum: [f32; 4],
    /// Running product of `1 - alpha` over all accumulated fragments.
    pub revealage: f32,
}

impl Default for OitAccumulation {
    fn default() -> Self {
        Self::CLEAR
    }
}

impl OitAccumulation {
    /// The cleared state before any fragment is accumulated: no colour, and a
    /// revealage of one (the background is fully revealed).
    pub const CLEAR: Self = Self {
        accum: [0.0, 0.0, 0.0, 0.0],
        revealage: 1.0,
    };

    /// Accumulates one transparent fragment with its depth weight.
    ///
    /// This is the exact per-fragment blend the GPU performs with additive
    /// blending on the accumulation target and multiplicative blending on the
    /// revealage target.
    pub fn accumulate(&mut self, fragment: OitFragment) {
        let alpha = fragment.alpha.clamp(0.0, 1.0);
        let weight = oit_weight(fragment.view_depth, alpha);
        // Premultiply colour by alpha, then by the depth weight.
        let premultiplied = alpha * weight;
        self.accum[0] += fragment.color[0] * premultiplied;
        self.accum[1] += fragment.color[1] * premultiplied;
        self.accum[2] += fragment.color[2] * premultiplied;
        self.accum[3] += alpha * weight;
        self.revealage *= 1.0 - alpha;
    }

    /// Fraction of the background still visible through the transparent stack.
    pub fn revealage(&self) -> f32 {
        self.revealage.clamp(0.0, 1.0)
    }

    /// Resolves the accumulated transparency over an opaque `background`.
    ///
    /// The weighted colour is normalised by the summed weighted alpha to undo
    /// the depth weighting, then blended over the background by the revealage.
    pub fn resolve(&self, background: [f32; 3]) -> [f32; 3] {
        let normaliser = self.accum[3].max(ACCUM_EPSILON);
        let average = [
            self.accum[0] / normaliser,
            self.accum[1] / normaliser,
            self.accum[2] / normaliser,
        ];
        let reveal = self.revealage.clamp(0.0, 1.0);
        let coverage = 1.0 - reveal;
        [
            average[0] * coverage + background[0] * reveal,
            average[1] * coverage + background[1] * reveal,
            average[2] * coverage + background[2] * reveal,
        ]
    }

    /// The source colour the hardware-blended composite pass emits so that
    /// `SrcAlpha`/`OneMinusSrcAlpha` blending over the opaque background
    /// reproduces [`resolve`](Self::resolve) exactly, without the pass having
    /// to sample the background itself.
    ///
    /// Returns `[average_r, average_g, average_b, coverage]` where
    /// `coverage = 1 - revealage`.  Fixed-function blending then computes
    /// `out = src.rgb * src.a + dst * (1 - src.a) = average * coverage +
    /// background * revealage`, which is byte-for-byte the arithmetic in
    /// [`resolve`](Self::resolve).
    pub fn composite_source(&self) -> [f32; 4] {
        let normaliser = self.accum[3].max(ACCUM_EPSILON);
        let reveal = self.revealage.clamp(0.0, 1.0);
        [
            self.accum[0] / normaliser,
            self.accum[1] / normaliser,
            self.accum[2] / normaliser,
            1.0 - reveal,
        ]
    }
}

/// Convenience: accumulates a slice of fragments and resolves them over a
/// background in one call.  Handy for tests and CPU parity checks.
pub fn composite_transparency(fragments: &[OitFragment], background: [f32; 3]) -> [f32; 3] {
    let mut accumulation = OitAccumulation::CLEAR;
    for &fragment in fragments {
        accumulation.accumulate(fragment);
    }
    accumulation.resolve(background)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: [f32; 3], b: [f32; 3], eps: f32) -> bool {
        (0..3).all(|i| (a[i] - b[i]).abs() <= eps)
    }

    #[test]
    fn empty_stack_reveals_the_background() {
        let accumulation = OitAccumulation::CLEAR;
        assert_eq!(accumulation.resolve([0.2, 0.4, 0.6]), [0.2, 0.4, 0.6]);
        assert_eq!(accumulation.revealage(), 1.0);
    }

    #[test]
    fn opaque_fragment_hides_the_background() {
        // A single alpha=1 fragment fully covers the pixel: revealage -> 0 and
        // the resolved colour is the fragment colour regardless of its weight.
        let result = composite_transparency(
            &[OitFragment::new([0.8, 0.1, 0.3], 1.0, 12.0)],
            [0.0, 0.0, 0.0],
        );
        assert!(approx(result, [0.8, 0.1, 0.3], 1.0e-4), "{result:?}");
    }

    #[test]
    fn accumulation_is_order_independent() {
        let a = OitFragment::new([0.9, 0.1, 0.2], 0.4, 3.0);
        let b = OitFragment::new([0.1, 0.8, 0.3], 0.6, 9.0);
        let c = OitFragment::new([0.2, 0.2, 0.9], 0.5, 30.0);
        let background = [0.05, 0.05, 0.05];
        let forward = composite_transparency(&[a, b, c], background);
        let shuffled = composite_transparency(&[c, a, b], background);
        let reversed = composite_transparency(&[c, b, a], background);
        assert!(approx(forward, shuffled, 1.0e-6), "{forward:?} vs {shuffled:?}");
        assert!(approx(forward, reversed, 1.0e-6), "{forward:?} vs {reversed:?}");
    }

    #[test]
    fn more_layers_reveal_less_background() {
        let layer = OitFragment::new([1.0, 1.0, 1.0], 0.5, 10.0);
        let one = {
            let mut acc = OitAccumulation::CLEAR;
            acc.accumulate(layer);
            acc.revealage()
        };
        let three = {
            let mut acc = OitAccumulation::CLEAR;
            for _ in 0..3 {
                acc.accumulate(layer);
            }
            acc.revealage()
        };
        assert!(three < one, "more layers must occlude more: {three} !< {one}");
        assert!((one - 0.5).abs() < 1.0e-6);
        assert!((three - 0.125).abs() < 1.0e-6);
    }

    #[test]
    fn nearer_fragments_weigh_more_than_far_ones() {
        let near = oit_weight(1.0, 1.0);
        let far = oit_weight(500.0, 1.0);
        assert!(near >= far, "near {near} should outweigh far {far}");
        // Weight stays within the finite f16-safe clamp range.
        for depth in [0.0_f32, 1.0, 50.0, 200.0, 1000.0, 5000.0] {
            let weight = oit_weight(depth, 1.0);
            assert!(weight >= WEIGHT_MIN - 1.0e-6 && weight <= WEIGHT_MAX + 1.0e-6);
            assert!(weight.is_finite());
        }
    }

    #[test]
    fn resolved_colour_stays_within_inputs() {
        // Blending translucent white over black must land between them.
        let result = composite_transparency(
            &[OitFragment::new([1.0, 1.0, 1.0], 0.5, 10.0)],
            [0.0, 0.0, 0.0],
        );
        for channel in result {
            assert!((0.0..=1.0).contains(&channel), "channel {channel} out of range");
        }
        // Half coverage of white over black is mid-grey.
        assert!(approx(result, [0.5, 0.5, 0.5], 1.0e-4), "{result:?}");
    }

    #[test]
    fn zero_alpha_fragment_is_transparent() {
        let result = composite_transparency(
            &[OitFragment::new([1.0, 0.0, 0.0], 0.0, 5.0)],
            [0.1, 0.2, 0.3],
        );
        assert!(approx(result, [0.1, 0.2, 0.3], 1.0e-5), "{result:?}");
    }

    #[test]
    fn composite_source_matches_resolve_under_fixed_function_blend() {
        // The GPU composites transparency with fixed-function
        // SrcAlpha/OneMinusSrcAlpha blending over the view target instead of
        // sampling the background in the shader.  Emulate that blend and prove
        // it reproduces resolve() bit-for-bit for several stacks/backgrounds.
        let stacks: [&[OitFragment]; 3] = [
            &[],
            &[OitFragment::new([0.9, 0.1, 0.2], 0.4, 3.0)],
            &[
                OitFragment::new([0.9, 0.1, 0.2], 0.4, 3.0),
                OitFragment::new([0.1, 0.8, 0.3], 0.6, 9.0),
                OitFragment::new([0.2, 0.2, 0.9], 0.5, 30.0),
            ],
        ];
        for background in [[0.05, 0.05, 0.05], [0.2, 0.4, 0.6], [0.0, 0.0, 0.0]] {
            for fragments in stacks {
                let mut acc = OitAccumulation::CLEAR;
                for &fragment in fragments {
                    acc.accumulate(fragment);
                }
                let src = acc.composite_source();
                let coverage = src[3];
                // out = src.rgb * src.a + dst * (1 - src.a)
                let blended = [
                    src[0] * coverage + background[0] * (1.0 - coverage),
                    src[1] * coverage + background[1] * (1.0 - coverage),
                    src[2] * coverage + background[2] * (1.0 - coverage),
                ];
                let resolved = acc.resolve(background);
                assert!(
                    approx(blended, resolved, 1.0e-6),
                    "blend {blended:?} != resolve {resolved:?}"
                );
                assert!((0.0..=1.0).contains(&coverage), "coverage {coverage} out of range");
            }
        }
    }
}
