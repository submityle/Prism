//! **§24.2 Frame pacing / low-latency + §24.3 VRR awareness.** Pure-CPU,
//! deterministic *data models* the design doc §24.2/§24.3 call for — no
//! wall-clock reads, no OS/display calls, integer-only arithmetic.
//!
//! Three cooperating pieces, each in its own file:
//!
//! - [`FramePacer`] ([`pacer`]) — phase-locked present-cadence smoothing. Fed a
//!   stream of caller-supplied present timestamps, it keeps a steady predicted
//!   cadence (`next_present_ns`) that absorbs small jitter so micro-stutter does
//!   not propagate into the predicted present time, and only hard re-anchors on
//!   a large deviation. Mirrors console frame-pacing.
//! - [`ReflexGate`] ([`pacer`]) — the "begin work just-in-time, do not queue too
//!   many frames ahead" decision (NVIDIA Reflex / AMD Anti-Lag *form*). Given
//!   the present deadline, an estimate of CPU+GPU work time, and the current
//!   frames-in-flight, it decides whether to begin now, late-latch (wait so
//!   input is sampled as close to simulation start as possible), or hold because
//!   the render queue is full.
//! - [`VrrWindow`] ([`vrr`]) — a variable-refresh (G-Sync/FreeSync) display
//!   window. Clamps a desired present interval into the panel's `[min, max]`
//!   refresh window, applying Low-Framerate-Compensation (frame duplication)
//!   below the window, and computes the earliest allowed present time so the
//!   fixed-step simulation (§7) stays decoupled from the variable display.
//!
//! ## Honest boundary
//! These are the deterministic scheduling *models*. Reading real present
//! timestamps, the display's advertised VRR range, and actually sleeping /
//! submitting to the RHI are the `prism_app` main-loop and platform/RHI layers'
//! job (design doc §24.2/§24.3). Nothing here touches a clock or the OS, so a
//! given input sequence yields a bit-identical schedule across runs — safe to
//! drive from a deterministic replay. `no_std + alloc`, no `unsafe`.

mod pacer;
mod vrr;

pub use pacer::{BeginDecision, FramePacer, PresentInfo, ReflexGate};
pub use vrr::{VrrPresent, VrrWindow};

/// Integer exponential-moving-average step, rounded to nearest, saturating.
///
/// Returns `old + (sample - old) * num / den`. Used to smooth present intervals
/// without floating point so the result is bit-identical across runs. `den`
/// must be non-zero (`num <= den` for a conventional `0..=1` smoothing factor,
/// though that is not required).
#[inline]
#[must_use]
pub(crate) fn ema_step(old: u64, sample: u64, num: u32, den: u32) -> u64 {
    debug_assert!(den != 0, "ema_step denominator must be non-zero");
    if den == 0 {
        return old;
    }
    let diff = i128::from(sample) - i128::from(old);
    let scaled = diff * i128::from(num);
    let half = i128::from(den) / 2;
    // Round half away from zero so positive and negative drift are symmetric.
    let delta = if scaled >= 0 {
        (scaled + half) / i128::from(den)
    } else {
        (scaled - half) / i128::from(den)
    };
    let result = i128::from(old) + delta;
    result.clamp(0, i128::from(u64::MAX)) as u64
}

#[cfg(test)]
mod ema_tests {
    use super::ema_step;

    #[test]
    fn converges_toward_sample() {
        // Half-weight EMA from 100 toward 200 climbs and settles at 200.
        let mut v = 100u64;
        for _ in 0..40 {
            v = ema_step(v, 200, 1, 2);
        }
        assert_eq!(v, 200);
    }

    #[test]
    fn full_weight_tracks_sample_exactly() {
        assert_eq!(ema_step(100, 250, 1, 1), 250);
    }

    #[test]
    fn zero_weight_holds() {
        assert_eq!(ema_step(100, 250, 0, 1), 100);
    }

    #[test]
    fn rounds_to_nearest_symmetrically() {
        // +5 * 1/2 = +2.5 -> +3 (away from zero).
        assert_eq!(ema_step(100, 105, 1, 2), 103);
        // -5 * 1/2 = -2.5 -> -3 (away from zero).
        assert_eq!(ema_step(100, 95, 1, 2), 97);
    }

    #[test]
    fn deterministic_sequence() {
        let run = || {
            let mut v = 16_000_000u64;
            let samples = [16_666_667u64, 15_500_000, 17_200_000, 16_000_000];
            for s in samples {
                v = ema_step(v, s, 1, 4);
            }
            v
        };
        assert_eq!(run(), run());
    }
}
