//! Screen-space transparency composite / reconstruction layer for design §12.
//!
//! Design §12 is split across two sibling modules. [`super::sort_cull`] owns the
//! *strategy* half: given a blend mode and a live count it decides *whether* to
//! sort and *how* (a standalone radix/bitonic key, or a hand-off to the scene's
//! shared order-independent transparency path via
//! [`super::SortStrategy::SharedOit`]). This module owns the half that layer
//! deliberately leaves open: the *composite / reconstruction math* an
//! order-independent transparency (`OIT`) resolve actually runs once the
//! fragments of a translucent draw have been gathered.
//!
//! It is the `CPU`-verifiable contract for four production `OIT` techniques,
//! modelled at the algorithm level (no vendor code is pulled in):
//!
//! - **Weighted-Blended `OIT`** (`WBOIT`, `McGuire` & Bavoil): a single additive
//!   accumulation buffer plus a revealage product, resolved in one pass. Cheap,
//!   order-independent, approximate. See [`WboitAccumulator`] and
//!   [`wboit_weight`].
//! - **Moment-Based `OIT`** (`MBOIT`, Muennstermann et al., built on the
//!   `MSM` four-power-moment reconstruction of Peters & Klein): accumulate the
//!   power moments of the per-fragment optical-depth distribution, then
//!   reconstruct the fraction of absorbance in front of any depth. See
//!   [`MomentAccumulator`], [`PowerMoments`], and [`moment_occlusion`].
//! - **Per-pixel linked list**: gather every fragment, sort back-to-front, and
//!   composite exactly. The reference resolve any approximate tier is measured
//!   against. See [`FragmentList`].
//! - **Soft-particle depth fade**: fade a translucent fragment as it approaches
//!   the shared depth prepass surface so volumetric sprites do not show a hard
//!   intersection seam. See [`soft_particle_fade`].
//!
//! Division of responsibility: this module consumes [`super::SortStrategy`] and
//! [`super::renderers::ParticleBlend`] to pick a composite path
//! ([`composite_method`]) but never re-derives the sort decision — it only
//! covers the compositing the sort decision routes into. It is distinct from
//! [`super::volumetrics`], which handles *3D* volumetric light transport
//! (froxels, six-way baking); this module is purely *screen-space* fragment
//! compositing.
//!
//! Determinism rules match the sibling particle modules: the only non-`+ - * /`
//! primitive is `sqrt` (through [`Vec3`] or [`f32::sqrt`]); there are no
//! transcendental calls. The two transforms that are physically `exp`/`log`
//! (alpha to optical depth, optical depth to transmittance) are provided as
//! documented multiply-only polynomial approximations
//! ([`approx_absorbance`] / [`approx_transmittance`]) for the `CPU` reference,
//! while a production `GPU` resolve would use the hardware transcendental. Every
//! comparison of two `f32` values goes through an epsilon band rather than a
//! bare `==`.

use super::renderers::ParticleBlend;
use super::{SortStrategy, Vec3};
use alloc::vec::Vec;
use core::cmp::Ordering;

/// Shared epsilon guarding divisions and `f32` equality bands.
pub const OIT_EPS: f32 = 1.0e-6;

/// Clamps a scalar to the closed unit interval.
#[must_use]
fn clamp01(v: f32) -> f32 {
    v.clamp(0.0, 1.0)
}

/// Linear interpolation `(1 - t) * a + t * b`.
#[must_use]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Hermite smoothstep `3t^2 - 2t^3` on an already-normalized `t` in `[0, 1]`.
#[must_use]
fn smoothstep01(t: f32) -> f32 {
    let t = clamp01(t);
    t * t * (3.0 - 2.0 * t)
}

// ---------------------------------------------------------------------------
// Composite-method routing (bridges the §12 strategy decision)
// ---------------------------------------------------------------------------

/// The quality tier a shared-`OIT` resolve runs at (design §12).
///
/// The tiers trade accuracy for bandwidth: `Fast` is a single `WBOIT`
/// accumulation, `Balanced` is `MBOIT`, and `Reference` is the exact per-pixel
/// linked list. A `LOD` system picks a lower tier for distant emitters.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum OitQuality {
    /// Weighted-blended: one accumulation pass, cheapest, most approximate.
    Fast,
    /// Moment-based: a handful of moments, good accuracy at moderate cost.
    Balanced,
    /// Exact per-pixel linked list: the ground-truth resolve.
    Reference,
}

/// The composite path a translucent draw resolves through (design §12).
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum OitMethod {
    /// No `OIT`: opaque / alpha-masked draws depth-test and write directly.
    None,
    /// Order-independent additive / premultiplied blending: commutative, so no
    /// sort or accumulation buffer is needed.
    Additive,
    /// Weighted-blended `OIT` (`WBOIT`).
    WeightedBlended,
    /// Moment-based `OIT` (`MBOIT`).
    MomentBased,
    /// Exact per-pixel linked-list resolve.
    PerPixelLinkedList,
}

impl OitMethod {
    /// Whether this method needs the fragments gathered into an accumulation or
    /// list structure (as opposed to blending straight into the target).
    #[must_use]
    pub fn needs_gather(self) -> bool {
        matches!(
            self,
            OitMethod::WeightedBlended | OitMethod::MomentBased | OitMethod::PerPixelLinkedList
        )
    }
}

/// Picks the composite path for a translucent draw (design §12).
///
/// The sort *decision* is taken upstream by [`super::sort_cull`]; this bridge
/// only maps its result to the compositing side. An order-independent blend
/// (`SortStrategy::None`) resolves as [`OitMethod::Additive`] for
/// additive/premultiplied surfaces or [`OitMethod::None`] otherwise. A
/// standalone explicit sort composites with a plain back-to-front `over`, so it
/// also maps to [`OitMethod::None`]. A [`SortStrategy::SharedOit`] hand-off
/// selects the `OIT` technique from `tier`.
#[must_use]
pub fn composite_method(
    strategy: SortStrategy,
    blend: ParticleBlend,
    tier: OitQuality,
) -> OitMethod {
    match strategy {
        SortStrategy::None => match blend {
            ParticleBlend::Additive | ParticleBlend::Premultiplied => OitMethod::Additive,
            ParticleBlend::Opaque | ParticleBlend::AlphaMask | ParticleBlend::AlphaBlend => {
                OitMethod::None
            }
        },
        SortStrategy::ViewDepthRadix | SortStrategy::ViewDepthBitonic => OitMethod::None,
        SortStrategy::SharedOit => match tier {
            OitQuality::Fast => OitMethod::WeightedBlended,
            OitQuality::Balanced => OitMethod::MomentBased,
            OitQuality::Reference => OitMethod::PerPixelLinkedList,
        },
    }
}

// ---------------------------------------------------------------------------
// Weighted-Blended OIT (`McGuire` & Bavoil)
// ---------------------------------------------------------------------------

/// The depth/alpha weight for weighted-blended `OIT` (`McGuire` & Bavoil).
///
/// Nearer fragments must dominate, so the weight decreases monotonically with
/// view-space depth. This is the multiply-only form of `McGuire`'s tuned curve:
/// with `u = clamp(depth, near, far) / far`, the weight is
/// `alpha * clamp(0.03 / (1e-5 + u^4), 1e-2, 3e3)`. The `u^4` term is written as
/// repeated multiplication, so no transcendental is used. Larger depth means a
/// larger denominator and a smaller weight.
#[must_use]
pub fn wboit_weight(view_depth: f32, alpha: f32, near: f32, far: f32) -> f32 {
    let far = far.max(near.max(OIT_EPS));
    let d = view_depth.max(near.max(0.0));
    let u = d / far;
    let u2 = u * u;
    let u4 = u2 * u2;
    let raw = 0.03 / (1.0e-5 + u4);
    clamp01(alpha) * raw.clamp(1.0e-2, 3.0e3)
}

/// The premultiplied colour and coverage a [`WboitAccumulator`] resolves to.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResolvedTransparency {
    /// The resolved colour to blend over the background, already scaled by
    /// `coverage` (i.e. it is the foreground contribution, not the full frame).
    pub color: Vec3,
    /// Total coverage `1 - revealage` in `[0, 1]`: how much of the pixel the
    /// translucent fragments cover.
    pub coverage: f32,
}

impl ResolvedTransparency {
    /// Composites the resolved transparency over an opaque `background`.
    ///
    /// `color` is the coverage-weighted foreground, so the full frame colour is
    /// `color + background * revealage = color + background * (1 - coverage)`.
    #[must_use]
    pub fn over(self, background: Vec3) -> Vec3 {
        let revealage = 1.0 - clamp01(self.coverage);
        self.color.add(background.scale(revealage))
    }
}

/// The weighted-blended `OIT` accumulator (`McGuire` & Bavoil).
///
/// It stores the running sums a single-pass `WBOIT` resolve needs: the
/// weighted premultiplied colour, the weighted alpha, and the revealage
/// product. Because sums and products are commutative, the resolved result is
/// independent of fragment submission order (up to floating-point rounding).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WboitAccumulator {
    /// `sum(color_i * alpha_i * weight_i)` — the accumulation buffer `rgb`.
    color_weight: Vec3,
    /// `sum(alpha_i * weight_i)` — the accumulation buffer `alpha`.
    alpha_weight: f32,
    /// `product(1 - alpha_i)` — the revealage channel.
    revealage: f32,
}

impl Default for WboitAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

impl WboitAccumulator {
    /// The empty accumulator: no colour, no coverage, full revealage (`1`).
    #[must_use]
    pub const fn new() -> Self {
        Self {
            color_weight: Vec3::ZERO,
            alpha_weight: 0.0,
            revealage: 1.0,
        }
    }

    /// Folds one straight-alpha fragment in with an explicit `weight`.
    ///
    /// `color` is the straight (non-premultiplied) `RGB`; it is premultiplied by
    /// `alpha` and scaled by `weight` into the colour buffer, `alpha * weight`
    /// into the alpha buffer, and `(1 - alpha)` into the revealage product.
    pub fn accumulate(&mut self, color: Vec3, alpha: f32, weight: f32) {
        let a = clamp01(alpha);
        let w = weight.max(0.0);
        self.color_weight = self.color_weight.add(color.scale(a * w));
        self.alpha_weight += a * w;
        self.revealage *= 1.0 - a;
    }

    /// Folds one fragment in, deriving the weight from its view-space depth via
    /// [`wboit_weight`].
    pub fn accumulate_shaded(
        &mut self,
        color: Vec3,
        alpha: f32,
        view_depth: f32,
        near: f32,
        far: f32,
    ) {
        let w = wboit_weight(view_depth, alpha, near, far);
        // `wboit_weight` already folds `alpha`; divide it back out so the shared
        // `accumulate` path re-applies exactly one factor of `alpha`.
        let a = clamp01(alpha).max(OIT_EPS);
        self.accumulate(color, alpha, w / a);
    }

    /// The revealage product `product(1 - alpha_i)` — the fraction of the pixel
    /// still showing through all fragments.
    #[must_use]
    pub fn revealage(self) -> f32 {
        clamp01(self.revealage)
    }

    /// Resolves the accumulated fragments into a coverage-weighted colour.
    ///
    /// The average colour is `color_weight / max(alpha_weight, eps)`; the
    /// coverage is `1 - revealage`; the returned colour is
    /// `average * coverage`, ready to composite over the background with
    /// [`ResolvedTransparency::over`].
    #[must_use]
    pub fn resolve(self) -> ResolvedTransparency {
        let average = self
            .color_weight
            .scale(1.0 / self.alpha_weight.max(OIT_EPS));
        let coverage = 1.0 - clamp01(self.revealage);
        ResolvedTransparency {
            color: average.scale(coverage),
            coverage,
        }
    }
}

// ---------------------------------------------------------------------------
// Physical alpha <-> optical-depth transforms (multiply-only approximations)
// ---------------------------------------------------------------------------

/// Approximates the optical depth `-ln(1 - alpha)` of one fragment.
///
/// The exact absorbance is `-ln(1 - alpha)`, which is transcendental. For the
/// `CPU` reference this uses the truncated Mercator series
/// `alpha + alpha^2/2 + ... + alpha^8/8` (multiply/divide only), with `alpha`
/// clamped just below `1` so the series stays finite. A production `GPU`
/// `MBOIT` pass would use the hardware `log`; the series is accurate for small
/// to moderate `alpha` and monotonically increasing throughout.
#[must_use]
pub fn approx_absorbance(alpha: f32) -> f32 {
    let a = clamp01(alpha).min(0.999);
    let a2 = a * a;
    let a3 = a2 * a;
    let a4 = a3 * a;
    let a5 = a4 * a;
    let a6 = a5 * a;
    let a7 = a6 * a;
    let a8 = a7 * a;
    a + a2 / 2.0 + a3 / 3.0 + a4 / 4.0 + a5 / 5.0 + a6 / 6.0 + a7 / 7.0 + a8 / 8.0
}

/// Approximates the transmittance `exp(-optical_depth)` behind a depth.
///
/// The exact transmittance is `exp(-x)`, transcendental. For the `CPU`
/// reference this uses the reciprocal of the truncated exponential series
/// `1 / (1 + x + x^2/2 + x^3/6 + x^4/24)` (multiply/divide only). For `x >= 0`
/// it is positive, at most `1`, equals `1` at `x = 0`, and decreases
/// monotonically toward `0` as `x` grows, matching the qualitative behaviour of
/// `exp(-x)`. A production `GPU` resolve would use the hardware `exp`.
#[must_use]
pub fn approx_transmittance(optical_depth: f32) -> f32 {
    let x = optical_depth.max(0.0);
    let x2 = x * x;
    let x3 = x2 * x;
    let x4 = x3 * x;
    let denom = 1.0 + x + x2 / 2.0 + x3 / 6.0 + x4 / 24.0;
    clamp01(1.0 / denom)
}

// ---------------------------------------------------------------------------
// Moment-Based OIT (power moments + MSM reconstruction)
// ---------------------------------------------------------------------------

/// The normalized power moments of a per-pixel optical-depth distribution.
///
/// `total` is the sum of all fragment absorbances (the zeroth moment); `b` holds
/// the four normalized power moments `E[d]`, `E[d^2]`, `E[d^3]`, `E[d^4]` of the
/// (absorbance-weighted) normalized depth `d` in `[0, 1]`. These feed the four
/// power-moment `MSM` reconstruction in [`moment_occlusion`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PowerMoments {
    /// Total accumulated absorbance (zeroth moment).
    pub total: f32,
    /// Normalized power moments `[E[d], E[d^2], E[d^3], E[d^4]]`.
    pub b: [f32; 4],
}

/// Accumulates the power moments of a pixel's fragment stream for `MBOIT`.
///
/// Each fragment contributes its absorbance (via [`approx_absorbance`]) weighted
/// by increasing powers of its normalized depth. Because every field is an
/// additive sum, accumulation is order-independent.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MomentAccumulator {
    total: f32,
    m1: f32,
    m2: f32,
    m3: f32,
    m4: f32,
}

impl Default for MomentAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

impl MomentAccumulator {
    /// The empty accumulator (all moments zero).
    #[must_use]
    pub const fn new() -> Self {
        Self {
            total: 0.0,
            m1: 0.0,
            m2: 0.0,
            m3: 0.0,
            m4: 0.0,
        }
    }

    /// Folds one fragment in at normalized depth `depth` in `[0, 1]` with the
    /// given straight `alpha`.
    pub fn accumulate(&mut self, depth: f32, alpha: f32) {
        let a = approx_absorbance(alpha);
        let d = clamp01(depth);
        let d2 = d * d;
        let d3 = d2 * d;
        let d4 = d3 * d;
        self.total += a;
        self.m1 += a * d;
        self.m2 += a * d2;
        self.m3 += a * d3;
        self.m4 += a * d4;
    }

    /// The total accumulated absorbance (zeroth moment).
    #[must_use]
    pub fn total(self) -> f32 {
        self.total
    }

    /// Normalizes the raw sums into [`PowerMoments`], or `None` if no absorbance
    /// was accumulated (an empty pixel).
    #[must_use]
    pub fn normalized(self) -> Option<PowerMoments> {
        if self.total < OIT_EPS {
            return None;
        }
        let inv = 1.0 / self.total;
        Some(PowerMoments {
            total: self.total,
            b: [self.m1 * inv, self.m2 * inv, self.m3 * inv, self.m4 * inv],
        })
    }
}

/// Reconstructs the fraction of absorbance lying in front of `depth` from four
/// power moments (the `MSM` four-moment estimator of Peters & Klein, as adopted
/// by `MBOIT`).
///
/// `moments` are the normalized power moments `[E[d], E[d^2], E[d^3], E[d^4]]`;
/// `depth` is the query depth in `[0, 1]`; `bias` in `[0, 1]` mixes the moments
/// toward a well-conditioned reference distribution to keep the Hankel system
/// non-singular (a small value such as `3e-4` is typical). The result is the
/// occluded fraction in `[0, 1]`, monotonically non-decreasing in `depth`. When
/// the (unbiased) system is singular — e.g. a single opaque occluder — it
/// degrades to an exact step at the first moment.
#[must_use]
pub fn moment_occlusion(moments: [f32; 4], depth: f32, bias: f32) -> f32 {
    let bias = clamp01(bias);
    // Bias toward the moments of a symmetric reference distribution to
    // condition the Hankel matrix (`E[d] = 0`, `E[d^2] = 0.375`, ...).
    let b0 = lerp(moments[0], 0.0, bias);
    let b1 = lerp(moments[1], 0.375, bias);
    let b2 = lerp(moments[2], 0.0, bias);
    let b3 = lerp(moments[3], 0.375, bias);

    // Cholesky-related entries of the Hankel matrix
    // [[1, b0, b1], [b0, b1, b2], [b1, b2, b3]].
    let l32d22 = b2 - b0 * b1;
    let d22 = b1 - b0 * b0;
    let sq_depth_var = b3 - b1 * b1;
    let d33d22 = sq_depth_var * d22 - l32d22 * l32d22;

    // A singular system means a (near-)degenerate distribution: fall back to an
    // exact step at the mean depth `b0`.
    if d22.abs() < OIT_EPS || d33d22.abs() < OIT_EPS {
        return if depth <= b0 { 0.0 } else { 1.0 };
    }

    let inv_d22 = 1.0 / d22;
    let l32 = l32d22 * inv_d22;

    // Solve the Hankel system for the coefficients of `(1, z, z^2)`.
    let mut c0 = 1.0_f32;
    let mut c1 = depth - b0;
    let mut c2 = depth * depth - b1 - l32 * c1;
    c1 *= inv_d22;
    c2 *= d22 / d33d22;
    c1 -= l32 * c2;
    c0 -= c1 * b0 + c2 * b1;

    // Roots of `c2 z^2 + c1 z + c0 = 0` are the support points `z1 <= z2`.
    if c2.abs() < OIT_EPS {
        return if depth <= b0 { 0.0 } else { 1.0 };
    }
    let p = c1 / c2;
    let q = c0 / c2;
    let disc = (p * p * 0.25 - q).max(0.0);
    let r = disc.sqrt();
    let z1 = -p * 0.5 - r;
    let z2 = -p * 0.5 + r;

    // Piecewise closed-form occlusion depending on where `depth` falls relative
    // to the two support points.
    let (sx, sy, sz, sw) = if z2 < depth {
        (z1, depth, 1.0, 1.0)
    } else if z1 < depth {
        (depth, z1, 0.0, 1.0)
    } else {
        (0.0, 0.0, 0.0, 0.0)
    };
    let denom = (z2 - sy) * (depth - z1);
    let quotient = if denom.abs() < OIT_EPS {
        0.0
    } else {
        (sx * z2 - b0 * (sx + z2) + b1) / denom
    };
    clamp01(sz + sw * quotient)
}

/// The optical depth in front of `depth` reconstructed from [`PowerMoments`].
///
/// This is `total * occlusion`, where `occlusion` comes from
/// [`moment_occlusion`]. Feed the result to [`approx_transmittance`] to get the
/// transmittance a fragment at `depth` sees.
#[must_use]
pub fn reconstruct_optical_depth(moments: PowerMoments, depth: f32, bias: f32) -> f32 {
    moments.total * moment_occlusion(moments.b, depth, bias)
}

/// The transmittance a fragment at `depth` sees, reconstructed from
/// [`PowerMoments`] (the `MBOIT` per-fragment visibility).
#[must_use]
pub fn moment_transmittance(moments: PowerMoments, depth: f32, bias: f32) -> f32 {
    approx_transmittance(reconstruct_optical_depth(moments, depth, bias))
}

// ---------------------------------------------------------------------------
// Exact per-pixel linked-list resolve
// ---------------------------------------------------------------------------

/// One translucent fragment gathered for an exact resolve.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OitFragment {
    /// Straight (non-premultiplied) `RGB` colour.
    pub color: Vec3,
    /// Straight alpha (coverage) in `[0, 1]`.
    pub alpha: f32,
    /// View-space depth (smaller is nearer the camera).
    pub depth: f32,
}

impl OitFragment {
    /// Builds a fragment, clamping alpha to `[0, 1]`.
    #[must_use]
    pub fn new(color: Vec3, alpha: f32, depth: f32) -> Self {
        Self {
            color,
            alpha: clamp01(alpha),
            depth,
        }
    }
}

/// The rule for discarding a fragment when a bounded [`FragmentList`] overflows.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum OverflowPolicy {
    /// Keep the nearest fragments: drop whichever stored (or incoming) fragment
    /// is farthest from the camera.
    DropFarthest,
    /// Keep the most opaque fragments: drop whichever stored (or incoming)
    /// fragment has the lowest alpha.
    DropMostTransparent,
}

/// A bounded per-pixel fragment list resolved by an exact back-to-front `over`.
///
/// This is the ground-truth `OIT` resolve: gather up to `capacity` fragments,
/// then composite them strictly far-to-near. Because the resolve sorts by depth
/// first, the composited result is independent of insertion order. When the
/// list is full, [`FragmentList::insert`] applies an [`OverflowPolicy`]
/// deterministically (ties break toward the lowest stored index).
#[derive(Clone, Debug, PartialEq)]
pub struct FragmentList {
    fragments: Vec<OitFragment>,
    capacity: usize,
}

impl FragmentList {
    /// Creates an empty list bounded to `capacity` fragments (at least `1`).
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            fragments: Vec::new(),
            capacity: capacity.max(1),
        }
    }

    /// The number of stored fragments.
    #[must_use]
    pub fn len(&self) -> usize {
        self.fragments.len()
    }

    /// Whether no fragments are stored.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.fragments.is_empty()
    }

    /// The bounded capacity.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Finds the stored index the given `policy` would evict, if any.
    fn victim_index(&self, policy: OverflowPolicy) -> Option<usize> {
        let mut victim: Option<usize> = None;
        for (i, frag) in self.fragments.iter().enumerate() {
            let replace = match victim {
                None => true,
                Some(v) => {
                    let cur = &self.fragments[v];
                    match policy {
                        // Strict `>` keeps the earliest (lowest-index) extreme
                        // on ties, so the choice is deterministic.
                        OverflowPolicy::DropFarthest => frag.depth > cur.depth,
                        OverflowPolicy::DropMostTransparent => frag.alpha < cur.alpha,
                    }
                }
            };
            if replace {
                victim = Some(i);
            }
        }
        victim
    }

    /// Inserts a fragment, honouring the capacity bound.
    ///
    /// Below capacity the fragment is stored and `true` is returned. At capacity
    /// the `policy` picks the least-important stored fragment; the incoming
    /// fragment replaces it only if the incoming fragment is *more* important
    /// (nearer for [`OverflowPolicy::DropFarthest`], more opaque for
    /// [`OverflowPolicy::DropMostTransparent`]). Returns whether the incoming
    /// fragment ended up stored.
    pub fn insert(&mut self, fragment: OitFragment, policy: OverflowPolicy) -> bool {
        if self.fragments.len() < self.capacity {
            self.fragments.push(fragment);
            return true;
        }
        let Some(victim) = self.victim_index(policy) else {
            return false;
        };
        let evict = &self.fragments[victim];
        let incoming_wins = match policy {
            OverflowPolicy::DropFarthest => fragment.depth < evict.depth,
            OverflowPolicy::DropMostTransparent => fragment.alpha > evict.alpha,
        };
        if incoming_wins {
            self.fragments[victim] = fragment;
            true
        } else {
            false
        }
    }

    /// Composites the stored fragments over `background` with an exact
    /// back-to-front `over` (farthest fragment first).
    #[must_use]
    pub fn resolve(&self, background: Vec3) -> Vec3 {
        let mut ordered = self.fragments.clone();
        ordered.sort_by(|a, b| b.depth.partial_cmp(&a.depth).unwrap_or(Ordering::Equal));
        let mut out = background;
        for frag in &ordered {
            let a = clamp01(frag.alpha);
            out = frag.color.scale(a).add(out.scale(1.0 - a));
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Soft-particle depth fade
// ---------------------------------------------------------------------------

/// The soft-particle fade factor for a translucent fragment (design §12).
///
/// Volumetric sprites intersecting opaque geometry show a hard seam unless their
/// alpha is faded out as they approach the shared depth prepass surface. Given
/// the opaque `scene_depth` behind the fragment and the fragment's own
/// `particle_depth` (both view-space, larger is farther), the fade is
/// `smoothstep(0, contrast, scene_depth - particle_depth)`: `0` when the
/// fragment is at or behind the surface, ramping smoothly to `1` once it is
/// `contrast` in front. The Hermite smoothstep uses only multiplies, so no
/// transcendental is involved.
#[must_use]
pub fn soft_particle_fade(scene_depth: f32, particle_depth: f32, contrast: f32) -> f32 {
    let contrast = contrast.max(OIT_EPS);
    let delta = scene_depth - particle_depth;
    smoothstep01(delta / contrast)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v3(x: f32, y: f32, z: f32) -> Vec3 {
        Vec3::new(x, y, z)
    }

    fn approx(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    // --- routing bridge -----------------------------------------------------

    #[test]
    fn additive_blend_without_sort_routes_to_additive() {
        assert_eq!(
            composite_method(
                SortStrategy::None,
                ParticleBlend::Additive,
                OitQuality::Fast
            ),
            OitMethod::Additive
        );
        assert_eq!(
            composite_method(
                SortStrategy::None,
                ParticleBlend::Premultiplied,
                OitQuality::Reference
            ),
            OitMethod::Additive
        );
    }

    #[test]
    fn opaque_without_sort_routes_to_none() {
        assert_eq!(
            composite_method(SortStrategy::None, ParticleBlend::Opaque, OitQuality::Fast),
            OitMethod::None
        );
        assert_eq!(
            composite_method(
                SortStrategy::None,
                ParticleBlend::AlphaMask,
                OitQuality::Balanced
            ),
            OitMethod::None
        );
    }

    #[test]
    fn explicit_sort_composites_without_oit() {
        assert_eq!(
            composite_method(
                SortStrategy::ViewDepthRadix,
                ParticleBlend::AlphaBlend,
                OitQuality::Reference
            ),
            OitMethod::None
        );
        assert_eq!(
            composite_method(
                SortStrategy::ViewDepthBitonic,
                ParticleBlend::AlphaBlend,
                OitQuality::Reference
            ),
            OitMethod::None
        );
    }

    #[test]
    fn shared_oit_tier_selects_technique() {
        assert_eq!(
            composite_method(
                SortStrategy::SharedOit,
                ParticleBlend::AlphaBlend,
                OitQuality::Fast
            ),
            OitMethod::WeightedBlended
        );
        assert_eq!(
            composite_method(
                SortStrategy::SharedOit,
                ParticleBlend::AlphaBlend,
                OitQuality::Balanced
            ),
            OitMethod::MomentBased
        );
        assert_eq!(
            composite_method(
                SortStrategy::SharedOit,
                ParticleBlend::AlphaBlend,
                OitQuality::Reference
            ),
            OitMethod::PerPixelLinkedList
        );
    }

    #[test]
    fn needs_gather_is_true_only_for_accumulating_methods() {
        assert!(!OitMethod::None.needs_gather());
        assert!(!OitMethod::Additive.needs_gather());
        assert!(OitMethod::WeightedBlended.needs_gather());
        assert!(OitMethod::MomentBased.needs_gather());
        assert!(OitMethod::PerPixelLinkedList.needs_gather());
    }

    // --- WBOIT --------------------------------------------------------------

    #[test]
    fn wboit_weight_decreases_with_depth() {
        let near = wboit_weight(1.0, 1.0, 0.1, 100.0);
        let mid = wboit_weight(20.0, 1.0, 0.1, 100.0);
        let far = wboit_weight(90.0, 1.0, 0.1, 100.0);
        assert!(near > mid, "near {near} should exceed mid {mid}");
        assert!(mid > far, "mid {mid} should exceed far {far}");
    }

    #[test]
    fn wboit_weight_is_clamped_and_scales_with_alpha() {
        // Very near depth saturates the upper clamp before the alpha factor.
        let full = wboit_weight(0.0, 1.0, 0.1, 100.0);
        let half = wboit_weight(0.0, 0.5, 0.1, 100.0);
        assert!(
            approx(full, 3.0e3, 1.0e-3),
            "full weight clamps to 3e3: {full}"
        );
        assert!(approx(half, 1.5e3, 1.0e-3), "half alpha halves it: {half}");
    }

    #[test]
    fn wboit_single_opaque_fragment_resolves_to_its_color() {
        let mut acc = WboitAccumulator::new();
        acc.accumulate(v3(0.2, 0.4, 0.6), 1.0, 5.0);
        let resolved = acc.resolve();
        assert!(approx(resolved.coverage, 1.0, 1.0e-6));
        let framed = resolved.over(v3(1.0, 1.0, 1.0));
        assert!(approx(framed.x, 0.2, 1.0e-5));
        assert!(approx(framed.y, 0.4, 1.0e-5));
        assert!(approx(framed.z, 0.6, 1.0e-5));
    }

    #[test]
    fn wboit_is_order_independent() {
        let frags = [
            (v3(0.9, 0.1, 0.0), 0.5, 2.0_f32),
            (v3(0.0, 0.8, 0.2), 0.3, 8.0),
            (v3(0.1, 0.1, 0.7), 0.7, 20.0),
        ];
        let mut forward = WboitAccumulator::new();
        for (c, a, d) in frags {
            forward.accumulate_shaded(c, a, d, 0.1, 100.0);
        }
        let mut reverse = WboitAccumulator::new();
        for (c, a, d) in frags.iter().rev() {
            reverse.accumulate_shaded(*c, *a, *d, 0.1, 100.0);
        }
        let f = forward.resolve().over(Vec3::ZERO);
        let r = reverse.resolve().over(Vec3::ZERO);
        assert!(approx(f.x, r.x, 1.0e-5));
        assert!(approx(f.y, r.y, 1.0e-5));
        assert!(approx(f.z, r.z, 1.0e-5));
    }

    #[test]
    fn wboit_empty_reveals_background() {
        let resolved = WboitAccumulator::new().resolve();
        assert!(approx(resolved.coverage, 0.0, 1.0e-6));
        let framed = resolved.over(v3(0.3, 0.3, 0.3));
        assert!(approx(framed.x, 0.3, 1.0e-6));
    }

    // --- absorbance / transmittance approximations --------------------------

    #[test]
    fn approx_absorbance_matches_series_for_small_alpha() {
        // For small alpha, -ln(1-a) ~= a + a^2/2.
        let a = 0.1;
        let expected = a + a * a / 2.0;
        assert!(approx(approx_absorbance(a), expected, 5.0e-4));
    }

    #[test]
    fn approx_absorbance_is_monotonic_and_zero_at_zero() {
        assert!(approx(approx_absorbance(0.0), 0.0, 1.0e-7));
        let mut prev = 0.0;
        for i in 0..=10 {
            let a = i as f32 / 10.0;
            let cur = approx_absorbance(a);
            assert!(cur >= prev, "absorbance must be non-decreasing at a={a}");
            prev = cur;
        }
    }

    #[test]
    fn approx_transmittance_bounds_and_monotonicity() {
        assert!(approx(approx_transmittance(0.0), 1.0, 1.0e-7));
        let mut prev = approx_transmittance(0.0);
        for i in 1..=20 {
            let x = i as f32 * 0.5;
            let cur = approx_transmittance(x);
            assert!(
                (0.0..=1.0).contains(&cur),
                "transmittance in [0,1] at x={x}: {cur}"
            );
            assert!(cur <= prev, "transmittance must be non-increasing at x={x}");
            prev = cur;
        }
    }

    // --- MBOIT --------------------------------------------------------------

    #[test]
    fn moment_accumulator_normalizes_or_returns_none() {
        assert!(MomentAccumulator::new().normalized().is_none());
        let mut acc = MomentAccumulator::new();
        acc.accumulate(0.5, 0.6);
        let pm = acc.normalized().expect("non-empty");
        assert!(pm.total > 0.0);
        // Single fragment at d=0.5 -> all normalized moments are 0.5^k.
        assert!(approx(pm.b[0], 0.5, 1.0e-6));
        assert!(approx(pm.b[1], 0.25, 1.0e-6));
        assert!(approx(pm.b[2], 0.125, 1.0e-6));
        assert!(approx(pm.b[3], 0.0625, 1.0e-6));
    }

    #[test]
    fn moment_occlusion_single_occluder_is_a_step_when_unbiased() {
        // Degenerate single-point distribution at d=0.5: unbiased system is
        // singular, so it falls back to an exact step.
        let b = [0.5, 0.25, 0.125, 0.0625];
        assert!(approx(moment_occlusion(b, 0.3, 0.0), 0.0, 1.0e-6));
        assert!(approx(moment_occlusion(b, 0.7, 0.0), 1.0, 1.0e-6));
    }

    #[test]
    fn moment_occlusion_is_monotonic_in_depth() {
        let mut acc = MomentAccumulator::new();
        acc.accumulate(0.25, 0.5);
        acc.accumulate(0.5, 0.5);
        acc.accumulate(0.75, 0.5);
        let pm = acc.normalized().expect("non-empty");
        let mut prev = -1.0;
        for i in 0..=10 {
            let z = i as f32 / 10.0;
            let occ = moment_occlusion(pm.b, z, 3.0e-4);
            assert!(
                (0.0..=1.0).contains(&occ),
                "occlusion in [0,1] at z={z}: {occ}"
            );
            assert!(occ >= prev - 1.0e-4, "occlusion non-decreasing at z={z}");
            prev = occ;
        }
    }

    #[test]
    fn moment_transmittance_drops_across_occluders() {
        let mut acc = MomentAccumulator::new();
        acc.accumulate(0.4, 0.8);
        acc.accumulate(0.6, 0.8);
        let pm = acc.normalized().expect("non-empty");
        let front = moment_transmittance(pm, 0.1, 3.0e-4);
        let behind = moment_transmittance(pm, 0.9, 3.0e-4);
        assert!(
            front > behind,
            "front {front} should transmit more than behind {behind}"
        );
        assert!(behind >= 0.0 && front <= 1.0);
    }

    #[test]
    fn reconstruct_optical_depth_scales_with_total() {
        let pm = PowerMoments {
            total: 4.0,
            b: [0.5, 0.25, 0.125, 0.0625],
        };
        // Behind the single occluder, all absorbance is in front.
        let od = reconstruct_optical_depth(pm, 0.9, 0.0);
        assert!(approx(od, 4.0, 1.0e-5));
    }

    // --- per-pixel linked list ----------------------------------------------

    #[test]
    fn linked_list_resolve_is_insertion_order_independent() {
        let frags = [
            OitFragment::new(v3(1.0, 0.0, 0.0), 0.5, 5.0),
            OitFragment::new(v3(0.0, 1.0, 0.0), 0.4, 2.0),
            OitFragment::new(v3(0.0, 0.0, 1.0), 0.6, 9.0),
        ];
        let mut a = FragmentList::new(8);
        for f in frags {
            a.insert(f, OverflowPolicy::DropFarthest);
        }
        let mut b = FragmentList::new(8);
        for f in frags.iter().rev() {
            b.insert(*f, OverflowPolicy::DropFarthest);
        }
        let ra = a.resolve(Vec3::ZERO);
        let rb = b.resolve(Vec3::ZERO);
        assert!(approx(ra.x, rb.x, 1.0e-6));
        assert!(approx(ra.y, rb.y, 1.0e-6));
        assert!(approx(ra.z, rb.z, 1.0e-6));
    }

    #[test]
    fn linked_list_matches_manual_over() {
        // Two fragments: near red (a=0.5, z=1), far green (a=0.5, z=3), on black.
        let mut list = FragmentList::new(4);
        list.insert(
            OitFragment::new(v3(1.0, 0.0, 0.0), 0.5, 1.0),
            OverflowPolicy::DropFarthest,
        );
        list.insert(
            OitFragment::new(v3(0.0, 1.0, 0.0), 0.5, 3.0),
            OverflowPolicy::DropFarthest,
        );
        // Back-to-front: green over black -> (0,0.5,0); red over that ->
        // (0.5, 0.25, 0).
        let out = list.resolve(Vec3::ZERO);
        assert!(approx(out.x, 0.5, 1.0e-6));
        assert!(approx(out.y, 0.25, 1.0e-6));
        assert!(approx(out.z, 0.0, 1.0e-6));
    }

    #[test]
    fn linked_list_drop_farthest_keeps_nearest() {
        let mut list = FragmentList::new(2);
        assert!(list.insert(
            OitFragment::new(Vec3::ZERO, 0.5, 10.0),
            OverflowPolicy::DropFarthest
        ));
        assert!(list.insert(
            OitFragment::new(Vec3::ZERO, 0.5, 20.0),
            OverflowPolicy::DropFarthest
        ));
        // Nearer fragment evicts the farthest (z=20).
        assert!(list.insert(
            OitFragment::new(Vec3::ZERO, 0.5, 5.0),
            OverflowPolicy::DropFarthest
        ));
        // A farther-than-all fragment is rejected.
        assert!(!list.insert(
            OitFragment::new(Vec3::ZERO, 0.5, 30.0),
            OverflowPolicy::DropFarthest
        ));
        assert_eq!(list.len(), 2);
        let mut depths: Vec<f32> = list.fragments.iter().map(|f| f.depth).collect();
        depths.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        assert!(approx(depths[0], 5.0, 1.0e-6));
        assert!(approx(depths[1], 10.0, 1.0e-6));
    }

    #[test]
    fn linked_list_drop_most_transparent_keeps_opaque() {
        let mut list = FragmentList::new(2);
        list.insert(
            OitFragment::new(Vec3::ZERO, 0.2, 1.0),
            OverflowPolicy::DropMostTransparent,
        );
        list.insert(
            OitFragment::new(Vec3::ZERO, 0.9, 2.0),
            OverflowPolicy::DropMostTransparent,
        );
        // More opaque incoming (0.6) evicts the most transparent stored (0.2).
        assert!(list.insert(
            OitFragment::new(Vec3::ZERO, 0.6, 3.0),
            OverflowPolicy::DropMostTransparent
        ));
        // A more transparent incoming (0.1) is rejected.
        assert!(!list.insert(
            OitFragment::new(Vec3::ZERO, 0.1, 4.0),
            OverflowPolicy::DropMostTransparent
        ));
        let mut alphas: Vec<f32> = list.fragments.iter().map(|f| f.alpha).collect();
        alphas.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        assert!(approx(alphas[0], 0.6, 1.0e-6));
        assert!(approx(alphas[1], 0.9, 1.0e-6));
    }

    #[test]
    fn empty_list_resolves_to_background() {
        let list = FragmentList::new(4);
        assert!(list.is_empty());
        let out = list.resolve(v3(0.7, 0.2, 0.1));
        assert!(approx(out.x, 0.7, 1.0e-6));
        assert!(approx(out.y, 0.2, 1.0e-6));
        assert!(approx(out.z, 0.1, 1.0e-6));
    }

    #[test]
    fn capacity_is_at_least_one() {
        let list = FragmentList::new(0);
        assert_eq!(list.capacity(), 1);
    }

    // --- soft particles -----------------------------------------------------

    #[test]
    fn soft_fade_is_zero_at_surface_and_one_far_in_front() {
        // Fragment exactly at the surface fades to zero.
        assert!(approx(soft_particle_fade(10.0, 10.0, 2.0), 0.0, 1.0e-6));
        // Fragment behind the surface also fades to zero.
        assert!(approx(soft_particle_fade(10.0, 12.0, 2.0), 0.0, 1.0e-6));
        // Fragment well in front is fully visible.
        assert!(approx(soft_particle_fade(10.0, 2.0, 2.0), 1.0, 1.0e-6));
    }

    #[test]
    fn soft_fade_is_monotonic_across_the_band() {
        let mut prev = -1.0;
        for i in 0..=8 {
            let particle = 10.0 - i as f32 * 0.25;
            let fade = soft_particle_fade(10.0, particle, 2.0);
            assert!((0.0..=1.0).contains(&fade));
            assert!(
                fade >= prev,
                "fade must be non-decreasing as the fragment nears the camera"
            );
            prev = fade;
        }
    }

    #[test]
    fn soft_fade_midpoint_is_one_half() {
        // At half the contrast band the Hermite smoothstep passes through 0.5.
        assert!(approx(soft_particle_fade(10.0, 9.0, 2.0), 0.5, 1.0e-6));
    }
}
