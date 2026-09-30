//! Near-field compensation (NFC) filtering for higher-order Ambisonics.
//!
//! A point source at a finite distance radiates a spherical wave whose
//! higher-order spherical-harmonic components carry an increasing low-frequency
//! emphasis (the acoustic *near-field effect*). Encoding a source with the plain
//! far-field encoder in [`crate::hoa`] therefore omits a per-order transfer
//! function `F_m(kr)` that, taken literally, diverges at DC and cannot be
//! realised as a stable filter. The standard fix is **near-field compensated**
//! HOA: express every filter relative to a finite reference distance so the
//! result is a stable, finite-DC-gain shelving filter.
//!
//! This module builds the per-order compensation filter
//!
//! ```text
//! H_m(s) = theta_m(s * r_src / c) / theta_m(s * r_ref / c)
//! ```
//!
//! where `theta_m` is the reverse Bessel polynomial of degree `m`, `r_src` is
//! the source distance, `r_ref` the decoder reference distance, and `c` the
//! speed of sound. Because the numerator and denominator share the same monic
//! polynomial, the leading gains cancel: the filter is unity at high frequency
//! and has the finite DC gain `(r_ref / r_src)^m`. Sources closer than the
//! reference (`r_src < r_ref`) therefore receive a low-frequency boost that
//! grows with order (the reconstructed near-field bass lift); sources farther
//! away are attenuated. The poles are the roots of `theta_m(s * r_ref / c)`,
//! which lie in the open left half-plane, so every section is stable.
//!
//! Each analog section is mapped to a discrete IIR by the **bilinear transform**
//! `s = 2 * fs * (1 - z^-1) / (1 + z^-1)` (no frequency pre-warping, as the
//! near-field shelf is not tuned to a single corner frequency): a real root
//! yields a first-order section and a complex-conjugate root pair yields a
//! biquad. Degree `m` uses at most one first-order section and one biquad, so
//! orders up to [`MAX_NFC_ORDER`] need only fixed, small stack storage.
//!
//! # Provenance
//!
//! This module is engine-agnostic and contains **no Unreal Engine, Unity,
//! Godot, Wwise, FMOD, Steam Audio, or Google Resonance Audio source or derived
//! code**. Near-field coding and its stabilised near-field-compensated form are
//! publicly documented Ambisonic theory: see J. Daniel, "Representation de
//! champs acoustiques" (2001) and J. Daniel, "Further Study of Sound Field
//! Coding with Higher Order Ambisonics" (AES 116, 2004), which give the reverse
//! Bessel polynomial pole model and the reference-distance stabilisation used
//! here. Everything is implemented from that public literature.
//!
//! # Reverse Bessel polynomials and their roots
//!
//! The reverse Bessel polynomials are `theta_0 = 1`, `theta_1 = x + 1`,
//! `theta_2 = x^2 + 3x + 3`, `theta_3 = x^3 + 6x^2 + 15x + 15`. Their roots
//! (used to place the analog poles/zeros) are hard-coded closed-form values
//! rather than numerically solved, keeping the design bit-reproducible:
//!
//! - `theta_1`: `x = -1`.
//! - `theta_2`: `x = -1.5 +/- j * sqrt(3)/2`.
//! - `theta_3`: `x = -2.3221854` and `x = -1.8389073 +/- j * 1.7543810`.
//!
//! # Real-time contract
//!
//! Filter *design* ([`NfcCoeffs::design`]) runs at control rate and only
//! evaluates ratios of small hard-coded polynomials. The *processing* path
//! ([`NfcFilter::process_channel`] / [`NfcFilter::process_block`]) is a
//! sample-by-sample IIR over fixed-size persistent state: **allocation free,
//! lock free, and panic free**. Degenerate inputs (zero or negative distances,
//! zero sample rate, out-of-range order or channel index) are clamped or
//! ignored, never panicked.
//!
//! # Determinism
//!
//! The design and processing use only add/multiply/divide on [`Sample`]; there
//! are no `f32` intrinsics, so results are bit-reproducible across targets and
//! can be golden-compared sample-for-sample. This matches the workspace lints.

use prism_audio_core::math::Sample;

use crate::hoa::{MAX_HOA_CHANNELS, MAX_HOA_ORDER};

/// The highest Ambisonic order this module builds compensation filters for,
/// equal to [`MAX_HOA_ORDER`].
pub const MAX_NFC_ORDER: usize = MAX_HOA_ORDER;

/// Minimum distance (metres) a source/reference distance is clamped to, keeping
/// the design finite for a coincident or zero-distance input.
const MIN_DISTANCE: Sample = 1.0e-3;

/// Minimum sample rate (Hz) the design is clamped to.
const MIN_SAMPLE_RATE: Sample = 1.0;

/// Fallback speed of sound (m/s) used when a non-positive value is supplied.
const DEFAULT_SOUND_SPEED: Sample = 343.0;

/// Guard threshold below which a denominator is treated as zero (f64 design).
const DESIGN_EPSILON_F64: f64 = 1.0e-12;

// Reverse Bessel polynomial roots (closed form; see module docs).
const R1_REAL: Sample = -1.0;
const R2_RE: Sample = -1.5;
const R2_IM: Sample = 0.866_025_4;
const R3_REAL: Sample = -2.322_185_4;
const R3_RE: Sample = -1.838_907_3;
const R3_IM: Sample = 1.754_381;

/// A first-order IIR section `(b0 + b1 z^-1) / (1 + a1 z^-1)`.
#[derive(Debug, Clone, Copy)]
struct FirstOrder {
    b0: Sample,
    b1: Sample,
    a1: Sample,
}

impl FirstOrder {
    /// The pass-through section `H(z) = 1`.
    const IDENTITY: Self = Self { b0: 1.0, b1: 0.0, a1: 0.0 };
}

/// A biquad section `(b0 + b1 z^-1 + b2 z^-2) / (1 + a1 z^-1 + a2 z^-2)`.
#[derive(Debug, Clone, Copy)]
struct Biquad {
    b0: Sample,
    b1: Sample,
    b2: Sample,
    a1: Sample,
    a2: Sample,
}

impl Biquad {
    /// The pass-through section `H(z) = 1`.
    const IDENTITY: Self = Self { b0: 1.0, b1: 0.0, b2: 0.0, a1: 0.0, a2: 0.0 };
}

/// The cascaded sections realising one Ambisonic order's compensation filter.
///
/// Order 1 uses only [`FirstOrder`]; order 2 only [`Biquad`]; order 3 both.
/// Unused sections are [`FirstOrder::IDENTITY`] / [`Biquad::IDENTITY`], so the
/// processing path can always apply both without branching.
#[derive(Debug, Clone, Copy)]
struct OrderSections {
    fos: FirstOrder,
    biquad: Biquad,
}

impl OrderSections {
    const IDENTITY: Self = Self { fos: FirstOrder::IDENTITY, biquad: Biquad::IDENTITY };
}

/// Designs the discrete first-order section for a real analog root `x_root`.
///
/// The analog zero sits at `x_root * c / r_src` and the pole at
/// `x_root * c / r_ref`; both are in the left half-plane, so the bilinear map
/// yields a stable section.
fn design_first_order(
    x_root: Sample,
    r_src: Sample,
    r_ref: Sample,
    c: Sample,
    k: Sample,
) -> FirstOrder {
    // The design runs at control rate; it is carried out in f64 so the large
    // `k = 2 fs` terms do not lose the small pole/zero contributions to f32
    // catastrophic cancellation. Only add/sub/mul/div are used, so the result
    // stays deterministic.
    let x_root = f64::from(x_root);
    let c = f64::from(c);
    let r_src = f64::from(r_src);
    let r_ref = f64::from(r_ref);
    let k = f64::from(k);
    let zero = x_root * c / r_src;
    let pole = x_root * c / r_ref;
    let denom = k - pole;
    let inv = if (-DESIGN_EPSILON_F64..=DESIGN_EPSILON_F64).contains(&denom) {
        0.0
    } else {
        1.0 / denom
    };
    FirstOrder {
        b0: ((k - zero) * inv) as Sample,
        b1: (-(k + zero) * inv) as Sample,
        a1: (-(k + pole) * inv) as Sample,
    }
}

/// Designs the discrete biquad for a complex analog root pair `x_re +/- j x_im`.
///
/// The analog numerator is `s^2 - 2 Re(z) s + |z|^2` and denominator
/// `s^2 - 2 Re(p) s + |p|^2`, with the zero/pole obtained by scaling the root by
/// `c / r_src` and `c / r_ref` respectively.
fn design_biquad(
    x_re: Sample,
    x_im: Sample,
    r_src: Sample,
    r_ref: Sample,
    c: Sample,
    k: Sample,
) -> Biquad {
    // Control-rate design in f64 (see `design_first_order`): the `k2` term is on
    // the order of 1e10 while `den_a0` is on the order of 1e5, so f32 would lose
    // the small term to cancellation and skew the DC gain. Only elementary
    // arithmetic is used, so the design remains deterministic.
    let x_re = f64::from(x_re);
    let x_im = f64::from(x_im);
    let c = f64::from(c);
    let r_src = f64::from(r_src);
    let r_ref = f64::from(r_ref);
    let k = f64::from(k);
    let zr = x_re * c / r_src;
    let zi = x_im * c / r_src;
    let pr = x_re * c / r_ref;
    let pi = x_im * c / r_ref;
    let num_b1 = -2.0 * zr;
    let num_b0 = zr * zr + zi * zi;
    let den_a1 = -2.0 * pr;
    let den_a0 = pr * pr + pi * pi;
    let k2 = k * k;
    let da0 = k2 + den_a1 * k + den_a0;
    let inv = if (-DESIGN_EPSILON_F64..=DESIGN_EPSILON_F64).contains(&da0) {
        0.0
    } else {
        1.0 / da0
    };
    Biquad {
        b0: ((k2 + num_b1 * k + num_b0) * inv) as Sample,
        b1: ((-2.0 * k2 + 2.0 * num_b0) * inv) as Sample,
        b2: ((k2 - num_b1 * k + num_b0) * inv) as Sample,
        a1: ((-2.0 * k2 + 2.0 * den_a0) * inv) as Sample,
        a2: ((k2 - den_a1 * k + den_a0) * inv) as Sample,
    }
}

/// Precomputed near-field compensation filter coefficients, one section set per
/// Ambisonic order.
///
/// Build once at control rate with [`NfcCoeffs::design`], then hand to a
/// [`NfcFilter`]. Copyable and allocation free.
#[derive(Debug, Clone, Copy)]
pub struct NfcCoeffs {
    order: usize,
    orders: [OrderSections; MAX_NFC_ORDER + 1],
}

impl Default for NfcCoeffs {
    fn default() -> Self {
        Self::identity()
    }
}

impl NfcCoeffs {
    /// Coefficients that pass every channel through unchanged.
    #[must_use]
    pub const fn identity() -> Self {
        Self { order: MAX_NFC_ORDER, orders: [OrderSections::IDENTITY; MAX_NFC_ORDER + 1] }
    }

    /// Designs the stabilised NFC filter for a source at `source_distance`
    /// decoded against `reference_distance`, at `sample_rate` Hz with the given
    /// `sound_speed` (m/s).
    ///
    /// `order` is clamped to [`MAX_NFC_ORDER`]; distances to [`MIN_DISTANCE`];
    /// `sample_rate` to at least [`MIN_SAMPLE_RATE`]; a non-positive
    /// `sound_speed` falls back to [`DEFAULT_SOUND_SPEED`]. Order 0 is always the
    /// identity (the omnidirectional component has no near-field filter).
    ///
    /// # Examples
    ///
    /// ```
    /// use prism_audio_spatial::nfc::{NfcCoeffs, NfcFilter};
    ///
    /// // A source one metre away, decoded for a two-metre reference.
    /// let coeffs = NfcCoeffs::design(3, 1.0, 2.0, 48_000.0, 343.0);
    /// let mut filter = NfcFilter::new(3);
    /// filter.set_coeffs(&coeffs);
    ///
    /// // The omnidirectional (order-0) channel is unaffected.
    /// let mut w = [1.0f32, 0.0, 0.0, 0.0];
    /// filter.process_channel(0, &mut w);
    /// assert!((w[0] - 1.0).abs() < 1e-6);
    /// ```
    #[must_use]
    pub fn design(
        order: usize,
        source_distance: Sample,
        reference_distance: Sample,
        sample_rate: Sample,
        sound_speed: Sample,
    ) -> Self {
        let order = order.min(MAX_NFC_ORDER);
        let r_src = if source_distance > MIN_DISTANCE { source_distance } else { MIN_DISTANCE };
        let r_ref =
            if reference_distance > MIN_DISTANCE { reference_distance } else { MIN_DISTANCE };
        let fs = if sample_rate > MIN_SAMPLE_RATE { sample_rate } else { MIN_SAMPLE_RATE };
        let c = if sound_speed > 0.0 { sound_speed } else { DEFAULT_SOUND_SPEED };
        let k = 2.0 * fs;

        let mut orders = [OrderSections::IDENTITY; MAX_NFC_ORDER + 1];
        if order >= 1 {
            orders[1] = OrderSections {
                fos: design_first_order(R1_REAL, r_src, r_ref, c, k),
                biquad: Biquad::IDENTITY,
            };
        }
        if order >= 2 {
            orders[2] = OrderSections {
                fos: FirstOrder::IDENTITY,
                biquad: design_biquad(R2_RE, R2_IM, r_src, r_ref, c, k),
            };
        }
        if order >= 3 {
            orders[3] = OrderSections {
                fos: design_first_order(R3_REAL, r_src, r_ref, c, k),
                biquad: design_biquad(R3_RE, R3_IM, r_src, r_ref, c, k),
            };
        }
        Self { order, orders }
    }

    /// The Ambisonic order these coefficients were designed for.
    #[must_use]
    pub const fn order(&self) -> usize {
        self.order
    }
}

/// Per-channel IIR state for one ACN component.
#[derive(Debug, Clone, Copy, Default)]
struct ChannelState {
    fos_x1: Sample,
    fos_y1: Sample,
    bq_x1: Sample,
    bq_x2: Sample,
    bq_y1: Sample,
    bq_y2: Sample,
}

/// Returns the Ambisonic degree (order) of ACN channel `acn`, i.e.
/// `floor(sqrt(acn))`, clamped to [`MAX_NFC_ORDER`]. Computed with an integer
/// loop for determinism (no floating-point square root).
const fn order_of_channel(acn: usize) -> usize {
    let mut n = 0usize;
    while n < MAX_NFC_ORDER && (n + 1) * (n + 1) <= acn {
        n += 1;
    }
    n
}

/// A stateful near-field compensation filter over a full Ambisonic buffer.
///
/// Holds independent IIR state for each of the [`MAX_HOA_CHANNELS`] ACN
/// channels; all `2n + 1` channels of a given order share that order's
/// coefficients but filter independently. Build with [`NfcFilter::new`], load
/// coefficients with [`NfcFilter::set_coeffs`], then call
/// [`NfcFilter::process_channel`] or [`NfcFilter::process_block`] on each audio
/// block. [`NfcFilter::reset`] clears all state.
#[derive(Debug, Clone)]
pub struct NfcFilter {
    order: usize,
    coeffs: NfcCoeffs,
    states: [ChannelState; MAX_HOA_CHANNELS],
}

impl NfcFilter {
    /// Creates a filter for `order` (clamped to [`MAX_NFC_ORDER`]) with identity
    /// coefficients and zeroed state.
    #[must_use]
    pub fn new(order: usize) -> Self {
        let order = order.min(MAX_NFC_ORDER);
        Self {
            order,
            coeffs: NfcCoeffs::identity(),
            states: [ChannelState::default(); MAX_HOA_CHANNELS],
        }
    }

    /// Replaces the active coefficients (control rate). State is preserved so a
    /// coefficient swap does not click; call [`NfcFilter::reset`] to also clear
    /// history.
    pub fn set_coeffs(&mut self, coeffs: &NfcCoeffs) {
        self.coeffs = *coeffs;
    }

    /// The filter's Ambisonic order.
    #[must_use]
    pub const fn order(&self) -> usize {
        self.order
    }

    /// Clears all per-channel IIR history.
    pub fn reset(&mut self) {
        self.states = [ChannelState::default(); MAX_HOA_CHANNELS];
    }

    /// Filters `buf` in place as ACN channel `acn_index`.
    ///
    /// The channel's Ambisonic order selects the section set; order-0 (the `W`
    /// channel) and any channel whose order exceeds the design order pass
    /// through unchanged. An out-of-range `acn_index` is ignored. Real-time
    /// safe: no allocation, locking, or panics.
    pub fn process_channel(&mut self, acn_index: usize, buf: &mut [Sample]) {
        if acn_index >= MAX_HOA_CHANNELS {
            return;
        }
        let sections = self.coeffs.orders[order_of_channel(acn_index)];
        let state = &mut self.states[acn_index];
        for sample in buf.iter_mut() {
            // First-order section.
            let x0 = *sample;
            let y0 = sections.fos.b0 * x0 + sections.fos.b1 * state.fos_x1
                - sections.fos.a1 * state.fos_y1;
            state.fos_x1 = x0;
            state.fos_y1 = y0;
            // Biquad section.
            let y1 = sections.biquad.b0 * y0
                + sections.biquad.b1 * state.bq_x1
                + sections.biquad.b2 * state.bq_x2
                - sections.biquad.a1 * state.bq_y1
                - sections.biquad.a2 * state.bq_y2;
            state.bq_x2 = state.bq_x1;
            state.bq_x1 = y0;
            state.bq_y2 = state.bq_y1;
            state.bq_y1 = y1;
            *sample = y1;
        }
    }

    /// Filters an entire Ambisonic block in place: `channels[c]` is treated as
    /// ACN channel `c`. Channels beyond [`MAX_HOA_CHANNELS`] are ignored. Real-
    /// time safe.
    pub fn process_block(&mut self, channels: &mut [&mut [Sample]]) {
        for (acn, channel) in channels.iter_mut().enumerate() {
            if acn >= MAX_HOA_CHANNELS {
                break;
            }
            self.process_channel(acn, channel);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_math::ops;

    const FS: Sample = 48_000.0;
    const C: Sample = 343.0;

    fn fos_dc_gain(s: &FirstOrder) -> Sample {
        (s.b0 + s.b1) / (1.0 + s.a1)
    }

    fn biquad_dc_gain(s: &Biquad) -> Sample {
        (s.b0 + s.b1 + s.b2) / (1.0 + s.a1 + s.a2)
    }

    fn order_dc_gain(coeffs: &NfcCoeffs, order: usize) -> Sample {
        fos_dc_gain(&coeffs.orders[order].fos) * biquad_dc_gain(&coeffs.orders[order].biquad)
    }

    fn max_pole_magnitude(coeffs: &NfcCoeffs, order: usize) -> Sample {
        let mut worst = 0.0 as Sample;
        // First-order pole at z = -a1.
        let fos_pole = ops::abs(coeffs.orders[order].fos.a1);
        if fos_pole > worst {
            worst = fos_pole;
        }
        // Biquad poles: roots of z^2 + a1 z + a2.
        let a1 = coeffs.orders[order].biquad.a1;
        let a2 = coeffs.orders[order].biquad.a2;
        let disc = a1 * a1 - 4.0 * a2;
        if disc >= 0.0 {
            let root = ops::sqrt(disc);
            let m1 = ops::abs((-a1 + root) * 0.5);
            let m2 = ops::abs((-a1 - root) * 0.5);
            worst = worst.max(m1).max(m2);
        } else {
            // Complex pair: magnitude^2 = a2.
            let mag = ops::sqrt(ops::abs(a2));
            worst = worst.max(mag);
        }
        worst
    }

    #[test]
    fn order_zero_channel_is_identity() {
        let coeffs = NfcCoeffs::design(3, 1.0, 2.0, FS, C);
        let mut filter = NfcFilter::new(3);
        filter.set_coeffs(&coeffs);
        let input = [0.3, -0.7, 1.0, 0.2, -0.4];
        let mut buf = input;
        filter.process_channel(0, &mut buf);
        for (a, b) in buf.iter().zip(input.iter()) {
            assert!((a - b).abs() < 1.0e-6);
        }
    }

    #[test]
    fn equal_distance_impulse_response_is_delta() {
        // r_src == r_ref => every order collapses to a pure pass-through.
        let coeffs = NfcCoeffs::design(3, 1.5, 1.5, FS, C);
        let mut filter = NfcFilter::new(3);
        filter.set_coeffs(&coeffs);
        for acn in 0..MAX_HOA_CHANNELS {
            let mut buf = [0.0 as Sample; 8];
            buf[0] = 1.0;
            filter.process_channel(acn, &mut buf);
            assert!((buf[0] - 1.0).abs() < 1.0e-4, "acn {acn}: h[0]={}", buf[0]);
            for (i, v) in buf.iter().enumerate().skip(1) {
                assert!(v.abs() < 1.0e-4, "acn {acn}: h[{i}]={v} not ~0");
            }
        }
    }

    #[test]
    fn dc_gain_matches_distance_ratio() {
        let coeffs = NfcCoeffs::design(3, 1.0, 2.0, FS, C);
        // Expected DC gain per order is (r_ref / r_src)^m = 2^m. The design is
        // exact (carried out in f64), but the coefficients are stored in f32
        // direct form, where the biquad numerator taps b0 + b1 + b2 nearly
        // cancel (~1, -2, 1). The realised DC gain is therefore limited to about
        // 0.2% relative error by f32 quantisation, which is asserted here rather
        // than a tighter figure the f32 filter cannot actually deliver.
        for (order, expected) in [(1, 2.0 as Sample), (2, 4.0), (3, 8.0)] {
            let g = order_dc_gain(&coeffs, order);
            let rel = ops::abs(g - expected) / expected;
            assert!(rel < 5.0e-3, "order {order}: DC {g} != {expected} (rel {rel})");
        }
    }

    #[test]
    fn near_source_boosts_low_frequency_with_order() {
        // Source closer than the reference: DC gain > 1 and grows with order.
        let coeffs = NfcCoeffs::design(3, 0.5, 2.0, FS, C);
        let g1 = order_dc_gain(&coeffs, 1);
        let g2 = order_dc_gain(&coeffs, 2);
        let g3 = order_dc_gain(&coeffs, 3);
        assert!(g1 > 1.0);
        assert!(g2 > g1, "g2 {g2} !> g1 {g1}");
        assert!(g3 > g2, "g3 {g3} !> g2 {g2}");
    }

    #[test]
    fn far_source_attenuates_low_frequency_with_order() {
        // Source farther than the reference: DC gain < 1 and shrinks with order.
        let coeffs = NfcCoeffs::design(3, 3.0, 1.0, FS, C);
        let g1 = order_dc_gain(&coeffs, 1);
        let g2 = order_dc_gain(&coeffs, 2);
        let g3 = order_dc_gain(&coeffs, 3);
        assert!(g1 < 1.0);
        assert!(g2 < g1, "g2 {g2} !< g1 {g1}");
        assert!(g3 < g2, "g3 {g3} !< g2 {g2}");
    }

    #[test]
    fn all_poles_are_inside_the_unit_circle() {
        for (rs, rr) in [(1.0, 2.0), (2.0, 1.0), (0.5, 3.0), (4.0, 0.25)] {
            let coeffs = NfcCoeffs::design(3, rs, rr, FS, C);
            for order in 1..=3 {
                let m = max_pole_magnitude(&coeffs, order);
                assert!(m < 1.0, "rs {rs} rr {rr} order {order}: pole mag {m} >= 1");
            }
        }
    }

    #[test]
    fn impulse_response_is_finite_and_settles() {
        let coeffs = NfcCoeffs::design(3, 0.5, 2.0, FS, C);
        let mut filter = NfcFilter::new(3);
        filter.set_coeffs(&coeffs);
        // Third-order channel (acn 15) exercises both cascaded sections.
        let mut buf = [0.0 as Sample; 8192];
        buf[0] = 1.0;
        filter.process_channel(15, &mut buf);
        for v in &buf {
            assert!(v.is_finite());
        }
        // Late tail should have decayed toward zero (stable IIR).
        let tail = buf[buf.len() - 1].abs();
        assert!(tail < 1.0e-3, "tail {tail} did not settle");
    }

    #[test]
    fn steady_state_matches_analytic_dc_gain() {
        // Feed a DC step through the order-1 channel and compare to the closed
        // form (r_ref / r_src) = 2.
        let coeffs = NfcCoeffs::design(3, 1.0, 2.0, FS, C);
        let mut filter = NfcFilter::new(3);
        filter.set_coeffs(&coeffs);
        let mut buf = [1.0 as Sample; 60_000];
        filter.process_channel(1, &mut buf);
        let settled = buf[buf.len() - 1];
        assert!((settled - 2.0).abs() < 2.0e-2, "settled {settled} != 2.0");
    }

    #[test]
    fn reset_makes_processing_reproducible() {
        let coeffs = NfcCoeffs::design(3, 0.7, 1.3, FS, C);
        let mut filter = NfcFilter::new(3);
        filter.set_coeffs(&coeffs);
        let input = [0.1, -0.5, 0.9, 0.2, -0.8, 0.3, 0.0, 0.4];
        let mut first = input;
        filter.process_channel(6, &mut first);
        filter.reset();
        let mut second = input;
        filter.process_channel(6, &mut second);
        for (a, b) in first.iter().zip(second.iter()) {
            assert!((a - b).abs() < 1.0e-7);
        }
    }

    #[test]
    fn degenerate_inputs_do_not_panic() {
        // Zero distances, zero sample rate, huge order.
        let coeffs = NfcCoeffs::design(99, 0.0, 0.0, 0.0, -1.0);
        assert_eq!(coeffs.order(), MAX_NFC_ORDER);
        let mut filter = NfcFilter::new(99);
        assert_eq!(filter.order(), MAX_NFC_ORDER);
        filter.set_coeffs(&coeffs);
        // Empty buffer.
        filter.process_channel(2, &mut []);
        // Out-of-range channel is a no-op.
        let mut buf = [1.0, 2.0, 3.0];
        let original = buf;
        filter.process_channel(999, &mut buf);
        assert_eq!(buf, original);
    }

    #[test]
    fn process_block_routes_channels() {
        let coeffs = NfcCoeffs::design(2, 1.0, 2.0, FS, C);
        let mut block = NfcFilter::new(2);
        block.set_coeffs(&coeffs);
        let mut single = NfcFilter::new(2);
        single.set_coeffs(&coeffs);

        let mut c0 = [1.0 as Sample, 0.0, 0.0, 0.0];
        let mut c1 = [1.0 as Sample, 0.0, 0.0, 0.0];
        let mut ref0 = c0;
        let mut ref1 = c1;

        {
            let mut chans: [&mut [Sample]; 2] = [&mut c0, &mut c1];
            block.process_block(&mut chans);
        }
        single.process_channel(0, &mut ref0);
        single.process_channel(1, &mut ref1);

        for (a, b) in c0.iter().zip(ref0.iter()) {
            assert!((a - b).abs() < 1.0e-7);
        }
        for (a, b) in c1.iter().zip(ref1.iter()) {
            assert!((a - b).abs() < 1.0e-7);
        }
    }
}
