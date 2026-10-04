//! [`DriftCorrector`]: bounded-slew reconciliation of a local monotonic clock
//! toward an external reference, without breaking monotonicity or determinism.

use crate::Duration;

/// Nanoseconds in one second.
const NANOS_PER_SEC: u128 = 1_000_000_000;

/// Parts per million, the unit the slew rate is expressed in.
const PPM: i128 = 1_000_000;

/// Convert a `u128` nanosecond count into a [`Duration`], saturating the
/// seconds field rather than overflowing.
#[inline]
fn duration_from_nanos_u128(nanos: u128) -> Duration {
    let secs = (nanos / NANOS_PER_SEC).min(u64::MAX as u128) as u64;
    let sub = (nanos % NANOS_PER_SEC) as u32;
    Duration::new(secs, sub)
}

/// Slowly steers a local monotonic clock toward an external reference by
/// adjusting its *rate*, never by stepping it.
///
/// The corrector tracks a signed `residual` — the outstanding offset
/// `reference - corrected` still to be absorbed. Each [`advance`](Self::advance)
/// adds the real delta plus a bounded *slew* to the corrected clock, where the
/// slew magnitude is capped at [`max_slew_ppm`](Self::max_slew_ppm) parts per
/// million of the real delta. Because the cap is strictly below `1_000_000`
/// ppm, the applied correction is always strictly smaller than the real delta,
/// so the corrected clock **always moves forward** (monotonic) and never jumps
/// — the invariant the fixed-step simulation depends on. Over many frames the
/// residual converges to zero.
///
/// For a genuine discontinuity — the very first sync, or resuming from a long
/// process suspend where slewing would take too long — [`resync`](Self::resync)
/// performs an explicit, acknowledged step and reports the jump, so callers can
/// treat it as a session boundary rather than a per-frame correction.
///
/// All arithmetic is integer (`u128` / `i128` nanoseconds); no floating point
/// enters the correction, so the corrected timeline is deterministic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DriftCorrector {
    /// Corrected elapsed time in nanoseconds (monotonic).
    corrected: u128,
    /// Outstanding offset `reference - corrected` still to be absorbed; a
    /// positive value means the corrected clock is *behind* the reference and
    /// should speed up.
    residual: i128,
    /// Maximum slew as parts per million of the real delta (clamped to
    /// `0..=999_999`).
    max_slew_ppm: u32,
}

impl DriftCorrector {
    /// Default slew cap: `500` ppm (0.05% of real time), a conventional,
    /// imperceptible clock-discipline rate.
    pub const DEFAULT_MAX_SLEW_PPM: u32 = 500;

    /// Largest permitted slew cap. Kept strictly below `1_000_000` ppm so the
    /// corrected clock stays strictly monotonic.
    pub const MAX_SLEW_PPM: u32 = 999_999;

    /// A corrector at corrected time zero, no outstanding offset, and the
    /// [default slew cap](Self::DEFAULT_MAX_SLEW_PPM).
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            corrected: 0,
            residual: 0,
            max_slew_ppm: Self::DEFAULT_MAX_SLEW_PPM,
        }
    }

    /// A corrector with an explicit slew cap (clamped to
    /// `0..=`[`MAX_SLEW_PPM`](Self::MAX_SLEW_PPM)).
    #[inline]
    #[must_use]
    pub const fn with_max_slew_ppm(mut self, ppm: u32) -> Self {
        self.max_slew_ppm = if ppm > Self::MAX_SLEW_PPM {
            Self::MAX_SLEW_PPM
        } else {
            ppm
        };
        self
    }

    /// The current slew cap in parts per million.
    #[inline]
    #[must_use]
    pub const fn max_slew_ppm(&self) -> u32 {
        self.max_slew_ppm
    }

    /// Set the slew cap (clamped to `0..=`[`MAX_SLEW_PPM`](Self::MAX_SLEW_PPM)).
    #[inline]
    pub fn set_max_slew_ppm(&mut self, ppm: u32) {
        self.max_slew_ppm = ppm.min(Self::MAX_SLEW_PPM);
    }

    /// Record a fresh reference sample. Sets the outstanding offset to
    /// `reference - corrected` at the current corrected time; subsequent
    /// [`advance`](Self::advance) calls slew to absorb it.
    #[inline]
    pub fn observe(&mut self, reference: Duration) {
        self.residual = reference.as_nanos() as i128 - self.corrected as i128;
    }

    /// Record a fresh reference *offset* directly (`reference - corrected`), for
    /// callers that measure the offset themselves (e.g. the net layer's
    /// round-trip estimator).
    #[inline]
    pub fn observe_offset(&mut self, offset: Duration, reference_ahead: bool) {
        let mag = offset.as_nanos() as i128;
        self.residual = if reference_ahead { mag } else { -mag };
    }

    /// Advance the corrected clock by a real delta, applying a bounded slew to
    /// absorb part of the outstanding offset. Returns the *corrected* delta
    /// actually applied (always `>= 0`, and `> 0` whenever `real_delta > 0`).
    pub fn advance(&mut self, real_delta: Duration) -> Duration {
        let rd = real_delta.as_nanos() as i128;
        // Max slew this step: a bounded fraction of the real delta. Floor
        // division keeps it strictly below `rd` for any cap < 1e6 ppm.
        let max_slew = rd.saturating_mul(self.max_slew_ppm as i128) / PPM;
        let slew = self.residual.clamp(-max_slew, max_slew);
        self.residual -= slew;
        // corrected_delta = rd + slew; |slew| <= max_slew < rd => >= 0.
        let corrected_delta = (rd + slew).max(0) as u128;
        self.corrected = self.corrected.saturating_add(corrected_delta);
        duration_from_nanos_u128(corrected_delta)
    }

    /// Explicitly step the corrected clock to `reference` and clear the
    /// outstanding offset. Returns the signed jump applied in nanoseconds
    /// (`reference - old_corrected`). Use only at session boundaries (first
    /// sync, resume-from-suspend); a per-frame step would break monotonicity
    /// and fixed-step determinism.
    #[inline]
    pub fn resync(&mut self, reference: Duration) -> i128 {
        let r = reference.as_nanos();
        let jump = r as i128 - self.corrected as i128;
        self.corrected = r;
        self.residual = 0;
        jump
    }

    /// Corrected elapsed time as a [`Duration`].
    #[inline]
    #[must_use]
    pub fn corrected(&self) -> Duration {
        duration_from_nanos_u128(self.corrected)
    }

    /// Corrected elapsed time in exact nanoseconds.
    #[inline]
    #[must_use]
    pub const fn corrected_nanos(&self) -> u128 {
        self.corrected
    }

    /// The outstanding offset still to be absorbed, in signed nanoseconds
    /// (`reference - corrected`). Positive means the clock is behind.
    #[inline]
    #[must_use]
    pub const fn residual_nanos(&self) -> i128 {
        self.residual
    }

    /// Whether the outstanding offset is within `tolerance` of zero.
    #[inline]
    #[must_use]
    pub fn is_converged(&self, tolerance: Duration) -> bool {
        self.residual.unsigned_abs() <= tolerance.as_nanos()
    }

    /// Reset corrected time and the outstanding offset to zero, keeping the
    /// slew cap.
    #[inline]
    pub fn reset(&mut self) {
        self.corrected = 0;
        self.residual = 0;
    }
}

impl Default for DriftCorrector {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}
