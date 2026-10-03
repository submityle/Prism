//! Weighted Blended Order-Independent Transparency (`WBOIT`).
//!
//! This is the device-free numeric core behind the
//! [`TransparencyPath::WeightedOit`](super::TransparencyPath::WeightedOit)
//! bucket produced by [`routing`](super::routing). It implements the
//! `McGuire`-`Bavoil` *"Weighted Blended Order-Independent Transparency"* (Journal of
//! Computer Graphics Techniques, 2013): instead of sorting transparent
//! fragments back-to-front, every fragment is accumulated into two
//! order-independent buffers and resolved in a single pass.
//!
//! For each fragment the accumulation is
//!
//! ```text
//! accum.rgb += color * alpha * w(z, alpha)
//! accum.a   += alpha * w(z, alpha)
//! revealage *= (1 - alpha)
//! ```
//!
//! and the resolve composites the weighted average colour over the background
//! using the product of transmittances:
//!
//! ```text
//! average   = accum.rgb / max(accum.a, EPSILON)
//! out.rgb   = average * (1 - revealage) + background * revealage
//! ```
//!
//! Both the sum (`accum`) and the product (`revealage`) are commutative, so the
//! resolved colour does not depend on the order fragments are submitted — that
//! is the whole point of the technique. The depth weight `w(z, alpha)` is a
//! heuristic that biases nearer, more opaque fragments to dominate the average;
//! the paper gives several closed forms selected by [`WeightFunction`].
//!
//! The approximation is exact for a single transparent layer (it reduces to
//! ordinary `src-over` alpha compositing) and degrades gracefully as layers
//! overlap, which is why it is the default fallback for general blended
//! surfaces that cannot afford per-pixel sorting.

/// Linear, straight-alpha (non-premultiplied) `RGB` colour.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rgb {
    /// Red channel in linear space.
    pub r: f32,
    /// Green channel in linear space.
    pub g: f32,
    /// Blue channel in linear space.
    pub b: f32,
}

impl Rgb {
    /// A black colour (all channels zero).
    pub const BLACK: Self = Self {
        r: 0.0,
        g: 0.0,
        b: 0.0,
    };

    /// Construct a colour from its three linear channels.
    #[inline]
    pub const fn new(r: f32, g: f32, b: f32) -> Self {
        Self { r, g, b }
    }
}

/// A single transparent fragment competing for one pixel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OitFragment {
    /// Straight (non-premultiplied) linear colour of the fragment.
    pub color: Rgb,
    /// Coverage / opacity in `[0, 1]`.
    pub alpha: f32,
    /// Positive view-space distance to the fragment (eye-space `Z`), used by
    /// the depth-based weight functions. Larger is farther from the camera.
    pub depth: f32,
}

impl OitFragment {
    /// Construct a fragment from its colour, alpha, and view-space depth.
    #[inline]
    pub const fn new(color: Rgb, alpha: f32, depth: f32) -> Self {
        Self {
            color,
            alpha,
            depth,
        }
    }
}

/// Depth-weight heuristic `w(z, alpha)` from the paper.
///
/// The depth-based variants ([`Eq7`](WeightFunction::Eq7),
/// [`Eq8`](WeightFunction::Eq8), [`Eq9`](WeightFunction::Eq9)) take the
/// view-space distance stored in [`OitFragment::depth`] and are tuned for
/// different scene scales. [`Eq10`](WeightFunction::Eq10) instead expects a
/// window-space depth already normalized to `[0, 1]`; callers that use it must
/// pre-normalize [`OitFragment::depth`] into that range (0 at the near plane,
/// 1 at the far plane).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Default)]
pub enum WeightFunction {
    /// Paper equation (7): tuned for small depth ranges.
    Eq7,
    /// Paper equation (8): tuned for medium depth ranges.
    Eq8,
    /// Paper equation (9): tuned for large depth ranges. A robust default.
    #[default]
    Eq9,
    /// Paper equation (10): operates on a normalized `[0, 1]` depth instead of
    /// a view-space distance.
    Eq10,
}

/// Lower clamp applied to every depth weight, from the paper.
const WEIGHT_MIN: f32 = 1e-2;
/// Upper clamp applied to every depth weight, from the paper.
const WEIGHT_MAX: f32 = 3e3;
/// Guard added to weight denominators to avoid a divide-by-zero at `z = 0`.
const WEIGHT_DENOM_GUARD: f32 = 1e-5;
/// Guard used when dividing the accumulated colour by its accumulated weight.
const RESOLVE_EPSILON: f32 = 1e-5;

/// `x` raised to the fourth power via multiplication (avoids `f32::powi`,
/// which is disallowed in this crate for `libm` determinism).
#[inline]
fn pow4(x: f32) -> f32 {
    let x2 = x * x;
    x2 * x2
}

/// `x` raised to the sixth power via multiplication.
#[inline]
fn pow6(x: f32) -> f32 {
    let x2 = x * x;
    x2 * x2 * x2
}

impl WeightFunction {
    /// Evaluate the (depth-only) part of the weight for a fragment depth.
    ///
    /// The returned value is clamped to `[WEIGHT_MIN, WEIGHT_MAX]` exactly as
    /// the paper specifies; it is multiplied by `alpha` inside
    /// [`WeightedOitAccumulator::add`] to form the full `w(z, alpha)`.
    #[inline]
    fn depth_weight(self, depth: f32) -> f32 {
        let z = depth.abs();
        let raw = match self {
            WeightFunction::Eq7 => {
                let t = z / 200.0;
                0.03 / (WEIGHT_DENOM_GUARD + pow4(t))
            }
            WeightFunction::Eq8 => {
                let a = z / 10.0;
                let b = z / 200.0;
                10.0 / (WEIGHT_DENOM_GUARD + a * a * a + pow6(b))
            }
            WeightFunction::Eq9 => {
                let a = z / 5.0;
                let b = z / 200.0;
                10.0 / (WEIGHT_DENOM_GUARD + a * a + pow6(b))
            }
            WeightFunction::Eq10 => {
                // `z` is a normalized `[0, 1]` window-space depth here.
                let d = z.clamp(0.0, 1.0);
                0.3 / (WEIGHT_DENOM_GUARD + pow4(d))
            }
        };
        raw.clamp(WEIGHT_MIN, WEIGHT_MAX)
    }
}

/// The two order-independent buffers `WBOIT` accumulates into.
///
/// This mirrors the two render targets used on the `GPU`: a four-channel
/// additive `accum` target and a single-channel multiplicative `revealage`
/// target. Accumulating is associative and commutative, so fragments may be
/// added in any order and even merged across tiles with
/// [`merge`](WeightedOitAccumulator::merge).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WeightedOitAccumulator {
    /// `(sum color*alpha*w, sum alpha*w)` — the weighted colour sum and the
    /// weight normalizer in the fourth channel.
    accum: [f32; 4],
    /// Product of `(1 - alpha)` over every fragment: the net transmittance of
    /// the transparent stack, i.e. how much background shows through.
    revealage: f32,
    /// Weight function used by [`add`](WeightedOitAccumulator::add).
    weight: WeightFunction,
}

impl WeightedOitAccumulator {
    /// Create an empty accumulator using the given depth-weight function.
    ///
    /// The empty state has zero accumulated colour and a revealage of `1.0`
    /// (the background is fully visible until fragments cover it).
    #[inline]
    pub const fn new(weight: WeightFunction) -> Self {
        Self {
            accum: [0.0; 4],
            revealage: 1.0,
            weight,
        }
    }

    /// Accumulate one transparent fragment.
    ///
    /// The fragment alpha is clamped to `[0, 1]` so stray values outside the
    /// valid coverage range cannot drive the revealage negative.
    pub fn add(&mut self, fragment: OitFragment) {
        let alpha = fragment.alpha.clamp(0.0, 1.0);
        let w = alpha * self.weight.depth_weight(fragment.depth);
        self.accum[0] += fragment.color.r * w;
        self.accum[1] += fragment.color.g * w;
        self.accum[2] += fragment.color.b * w;
        self.accum[3] += w;
        self.revealage *= 1.0 - alpha;
    }

    /// Accumulate every fragment in an iterator.
    pub fn extend<I>(&mut self, fragments: I)
    where
        I: IntoIterator<Item = OitFragment>,
    {
        for fragment in fragments {
            self.add(fragment);
        }
    }

    /// Merge another accumulator built with the same weight function.
    ///
    /// This is the tile-combine operation: the colour sums add and the
    /// revealages multiply, preserving order independence across partitions.
    ///
    /// # Panics
    /// Panics if the two accumulators use different [`WeightFunction`]s, since
    /// mixing weight heuristics would make the averaged colour meaningless.
    pub fn merge(&mut self, other: &WeightedOitAccumulator) {
        assert_eq!(
            self.weight, other.weight,
            "cannot merge WBOIT accumulators built with different weight functions",
        );
        for i in 0..4 {
            self.accum[i] += other.accum[i];
        }
        self.revealage *= other.revealage;
    }

    /// Net transmittance of the transparent stack in `[0, 1]`.
    ///
    /// `1.0` means no fragment covered the pixel; `0.0` means the stack is
    /// fully opaque and the background is hidden.
    #[inline]
    pub fn revealage(&self) -> f32 {
        self.revealage
    }

    /// Resolve the accumulated buffers over a background colour.
    ///
    /// Returns `background` unchanged when nothing was accumulated.
    pub fn resolve(&self, background: Rgb) -> Rgb {
        let coverage = 1.0 - self.revealage;
        let denom = self.accum[3].max(RESOLVE_EPSILON);
        let avg = Rgb {
            r: self.accum[0] / denom,
            g: self.accum[1] / denom,
            b: self.accum[2] / denom,
        };
        Rgb {
            r: avg.r * coverage + background.r * self.revealage,
            g: avg.g * coverage + background.g * self.revealage,
            b: avg.b * coverage + background.b * self.revealage,
        }
    }
}

/// Composite a slice of transparent fragments over a background in one call.
///
/// This is the convenience wrapper over [`WeightedOitAccumulator`]: it builds
/// an accumulator, adds every fragment, and resolves. Because accumulation is
/// order independent, the result is invariant under any permutation of
/// `fragments`.
pub fn composite(fragments: &[OitFragment], weight: WeightFunction, background: Rgb) -> Rgb {
    let mut acc = WeightedOitAccumulator::new(weight);
    acc.extend(fragments.iter().copied());
    acc.resolve(background)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    fn rgb_approx(a: Rgb, b: Rgb, eps: f32) -> bool {
        approx(a.r, b.r, eps) && approx(a.g, b.g, eps) && approx(a.b, b.b, eps)
    }

    /// `src-over` reference compositing of a single straight-alpha layer.
    fn over(src: Rgb, alpha: f32, dst: Rgb) -> Rgb {
        Rgb {
            r: src.r * alpha + dst.r * (1.0 - alpha),
            g: src.g * alpha + dst.g * (1.0 - alpha),
            b: src.b * alpha + dst.b * (1.0 - alpha),
        }
    }

    #[test]
    fn empty_stack_shows_background() {
        let bg = Rgb::new(0.2, 0.4, 0.6);
        let out = composite(&[], WeightFunction::Eq9, bg);
        assert!(rgb_approx(out, bg, 1e-6));
    }

    #[test]
    fn single_layer_matches_exact_alpha_compositing() {
        // WBOIT is exact for one layer regardless of weight/depth.
        let bg = Rgb::new(0.1, 0.1, 0.1);
        for &weight in &[
            WeightFunction::Eq7,
            WeightFunction::Eq8,
            WeightFunction::Eq9,
        ] {
            for &alpha in &[0.0_f32, 0.25, 0.5, 0.75, 1.0] {
                for &depth in &[0.5_f32, 7.0, 50.0, 500.0] {
                    let frag = OitFragment::new(Rgb::new(0.9, 0.3, 0.2), alpha, depth);
                    let out = composite(&[frag], weight, bg);
                    let want = over(frag.color, alpha, bg);
                    assert!(
                        rgb_approx(out, want, 1e-4),
                        "weight={weight:?} alpha={alpha} depth={depth} out={out:?} want={want:?}",
                    );
                }
            }
        }
    }

    #[test]
    fn fully_opaque_layer_hides_background() {
        let bg = Rgb::new(0.9, 0.9, 0.9);
        let color = Rgb::new(0.2, 0.4, 0.1);
        let out = composite(
            &[OitFragment::new(color, 1.0, 10.0)],
            WeightFunction::Eq9,
            bg,
        );
        assert!(rgb_approx(out, color, 1e-4), "out={out:?}");
    }

    #[test]
    fn result_is_order_independent() {
        let bg = Rgb::new(0.05, 0.05, 0.2);
        let frags = [
            OitFragment::new(Rgb::new(0.9, 0.1, 0.1), 0.6, 3.0),
            OitFragment::new(Rgb::new(0.1, 0.9, 0.1), 0.4, 12.0),
            OitFragment::new(Rgb::new(0.1, 0.1, 0.9), 0.8, 1.5),
            OitFragment::new(Rgb::new(0.8, 0.8, 0.2), 0.3, 42.0),
        ];
        let base = composite(&frags, WeightFunction::Eq9, bg);

        // Every rotation/permutation must resolve to the same colour.
        let mut permuted = frags;
        permuted.reverse();
        let rev = composite(&permuted, WeightFunction::Eq9, bg);
        assert!(rgb_approx(base, rev, 1e-5), "base={base:?} rev={rev:?}");

        let shuffled = [frags[2], frags[0], frags[3], frags[1]];
        let sh = composite(&shuffled, WeightFunction::Eq9, bg);
        assert!(rgb_approx(base, sh, 1e-5), "base={base:?} shuffled={sh:?}");
    }

    #[test]
    fn merge_matches_single_accumulator() {
        let bg = Rgb::new(0.3, 0.2, 0.1);
        let a = [
            OitFragment::new(Rgb::new(0.7, 0.2, 0.5), 0.5, 4.0),
            OitFragment::new(Rgb::new(0.2, 0.6, 0.9), 0.35, 20.0),
        ];
        let b = [
            OitFragment::new(Rgb::new(0.9, 0.9, 0.1), 0.6, 2.0),
            OitFragment::new(Rgb::new(0.1, 0.4, 0.4), 0.2, 60.0),
        ];

        let mut whole = WeightedOitAccumulator::new(WeightFunction::Eq9);
        whole.extend(a.iter().copied());
        whole.extend(b.iter().copied());

        let mut part_a = WeightedOitAccumulator::new(WeightFunction::Eq9);
        part_a.extend(a.iter().copied());
        let mut part_b = WeightedOitAccumulator::new(WeightFunction::Eq9);
        part_b.extend(b.iter().copied());
        part_a.merge(&part_b);

        assert!(rgb_approx(whole.resolve(bg), part_a.resolve(bg), 1e-6));
        assert!(approx(whole.revealage(), part_a.revealage(), 1e-6));
    }

    #[test]
    fn revealage_tracks_stacked_transmittance() {
        let mut acc = WeightedOitAccumulator::new(WeightFunction::Eq9);
        acc.add(OitFragment::new(Rgb::BLACK, 0.5, 1.0));
        acc.add(OitFragment::new(Rgb::BLACK, 0.5, 2.0));
        // (1 - 0.5) * (1 - 0.5) = 0.25 of the background survives.
        assert!(approx(acc.revealage(), 0.25, 1e-6));
    }

    #[test]
    fn alpha_is_clamped() {
        // Out-of-range alpha must not drive revealage negative.
        let mut acc = WeightedOitAccumulator::new(WeightFunction::Eq9);
        acc.add(OitFragment::new(Rgb::new(1.0, 1.0, 1.0), 2.0, 1.0));
        assert!(approx(acc.revealage(), 0.0, 1e-6));
        let out = acc.resolve(Rgb::new(0.5, 0.5, 0.5));
        assert!(
            rgb_approx(out, Rgb::new(1.0, 1.0, 1.0), 1e-4),
            "out={out:?}"
        );
    }

    #[test]
    fn eq10_uses_normalized_depth() {
        // Nearer (smaller normalized depth) must weigh at least as much as
        // farther, so a near opaque layer dominates the average.
        let bg = Rgb::BLACK;
        let near = OitFragment::new(Rgb::new(1.0, 0.0, 0.0), 0.5, 0.01);
        let far = OitFragment::new(Rgb::new(0.0, 0.0, 1.0), 0.5, 0.99);
        let out = composite(&[near, far], WeightFunction::Eq10, bg);
        // Red (near) should dominate blue (far).
        assert!(out.r > out.b, "out={out:?}");
    }
}
