//! [`InterpolationBuffer`]: a buffered snapshot timeline for smooth remote
//! playback.
//!
//! Remote entity state arrives as discrete, jittery, occasionally-dropped
//! snapshots. Rendering the newest snapshot directly stutters and teleports on
//! loss. The standard fix (Valve-/Overwatch-style) is to render in the *recent
//! past*: buffer snapshots and sample the timeline at `now - delay`,
//! interpolating between the two bracketing snapshots. The interpolation delay
//! (e.g. `100 ms`) trades a little latency for smoothness that absorbs jitter
//! and a dropped packet or two.
//!
//! The buffer is a fixed-capacity (`N`) timeline kept sorted by timestamp with
//! no heap allocation, so it is `no_std` and bounded. Sampling is deterministic
//! integer/`f64` math.

use core::array;

/// A single timestamped snapshot in the timeline.
struct Entry<T> {
    time_nanos: u128,
    value: T,
}

/// The result of sampling the timeline at a render time.
///
/// `from`/`to` bracket the render time with `alpha` in `[0, 1]`; callers lerp
/// `from -> to` by `alpha`. When the render time falls outside the buffered
/// range the result is [`Sampled::Clamped`] (hold the nearest snapshot) — never
/// an extrapolation, which would overshoot and jitter.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Sampled<'a, T> {
    /// The buffer is empty; nothing to sample.
    Empty,
    /// Render time is bracketed; interpolate `from -> to` by `alpha`.
    Interpolated {
        /// The earlier bracketing snapshot.
        from: &'a T,
        /// The later bracketing snapshot.
        to: &'a T,
        /// Interpolation factor in `[0, 1]`.
        alpha: f64,
    },
    /// Render time is outside the buffered range; hold this nearest snapshot.
    Clamped(&'a T),
}

/// Linear interpolation for snapshot payloads sampled via
/// [`InterpolationBuffer::sample_lerp`].
///
/// Implemented for `f32`/`f64`; implement it for your own transform/state type
/// to use `sample_lerp`, or use the closure-based [`InterpolationBuffer::sample_with`]
/// for arbitrary payloads.
pub trait Lerp {
    /// Interpolate from `self` to `other` by `t` in `[0, 1]`.
    fn lerp(&self, other: &Self, t: f64) -> Self;
}

impl Lerp for f32 {
    #[inline]
    fn lerp(&self, other: &Self, t: f64) -> Self {
        (*self as f64 + (*other as f64 - *self as f64) * t) as f32
    }
}

impl Lerp for f64 {
    #[inline]
    fn lerp(&self, other: &Self, t: f64) -> Self {
        self + (other - self) * t
    }
}

/// A fixed-capacity, time-ordered snapshot buffer rendered at a configurable
/// interpolation delay.
///
/// Push snapshots with [`push`](Self::push) (timestamped on the source clock,
/// e.g. server time), then sample the timeline at a render time with
/// [`sample`](Self::sample) / [`sample_lerp`](Self::sample_lerp). The buffer
/// holds at most `N` snapshots, evicting the oldest when full.
pub struct InterpolationBuffer<T, const N: usize> {
    /// Slots `0..len` are `Some`, sorted ascending by `time_nanos`.
    entries: [Option<Entry<T>>; N],
    len: usize,
    delay_nanos: u128,
}

impl<T, const N: usize> InterpolationBuffer<T, N> {
    /// Create an empty buffer with the given interpolation `delay`.
    #[inline]
    #[must_use]
    pub fn new(delay: core::time::Duration) -> Self {
        const { assert!(N >= 2, "InterpolationBuffer needs capacity >= 2") };
        Self {
            entries: array::from_fn(|_| None),
            len: 0,
            delay_nanos: delay.as_nanos(),
        }
    }

    /// The configured interpolation delay.
    #[inline]
    #[must_use]
    pub fn delay(&self) -> core::time::Duration {
        duration_from_nanos_u128(self.delay_nanos)
    }

    /// Set the interpolation delay (how far in the past the timeline is
    /// sampled).
    #[inline]
    pub fn set_delay(&mut self, delay: core::time::Duration) {
        self.delay_nanos = delay.as_nanos();
    }

    /// Number of buffered snapshots.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the buffer is empty.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Capacity (`N`).
    #[inline]
    #[must_use]
    pub fn capacity(&self) -> usize {
        N
    }

    /// Drop all snapshots.
    #[inline]
    pub fn clear(&mut self) {
        for slot in &mut self.entries[..self.len] {
            *slot = None;
        }
        self.len = 0;
    }

    /// The oldest buffered timestamp, or `None` if empty.
    #[inline]
    #[must_use]
    pub fn oldest(&self) -> Option<core::time::Duration> {
        self.entries[0]
            .as_ref()
            .map(|e| duration_from_nanos_u128(e.time_nanos))
    }

    /// The newest buffered timestamp, or `None` if empty.
    #[inline]
    #[must_use]
    pub fn newest(&self) -> Option<core::time::Duration> {
        if self.len == 0 {
            None
        } else {
            self.entries[self.len - 1]
                .as_ref()
                .map(|e| duration_from_nanos_u128(e.time_nanos))
        }
    }

    /// Insert a snapshot stamped at `time` on the source clock.
    ///
    /// Snapshots are kept sorted by timestamp, so out-of-order arrivals are
    /// placed correctly. A snapshot with a timestamp equal to an existing one
    /// replaces that entry's value (newer data for the same tick wins). When
    /// the buffer is full the oldest snapshot is evicted; an arrival older than
    /// everything buffered while full is dropped.
    pub fn push(&mut self, time: core::time::Duration, value: T) {
        let time_nanos = time.as_nanos();

        // Replace in place if a snapshot with this exact timestamp exists.
        for entry in self.entries[..self.len].iter_mut().flatten() {
            if entry.time_nanos == time_nanos {
                entry.value = value;
                return;
            }
        }

        if self.len == N {
            // Full: drop the incoming sample if it is older than the oldest.
            let oldest = self.entries[0].as_ref().map(|e| e.time_nanos).unwrap_or(0);
            if time_nanos <= oldest {
                return;
            }
            // Evict the oldest by shifting left one slot.
            for i in 0..self.len - 1 {
                self.entries.swap(i, i + 1);
            }
            self.entries[self.len - 1] = None;
            self.len -= 1;
        }

        // Find the sorted insert position.
        let mut pos = self.len;
        for i in 0..self.len {
            let t = self.entries[i].as_ref().map(|e| e.time_nanos).unwrap_or(0);
            if t > time_nanos {
                pos = i;
                break;
            }
        }

        // Shift the tail right by one to open slot `pos`.
        let mut i = self.len;
        while i > pos {
            self.entries.swap(i, i - 1);
            i -= 1;
        }
        self.entries[pos] = Some(Entry { time_nanos, value });
        self.len += 1;
    }

    /// Sample the timeline at render time `now - delay`.
    ///
    /// Returns an interpolation bracket, a clamp to the nearest snapshot when
    /// the render time is outside the buffered range, or [`Sampled::Empty`].
    pub fn sample(&self, now: core::time::Duration) -> Sampled<'_, T> {
        if self.len == 0 {
            return Sampled::Empty;
        }
        let render = now.as_nanos().saturating_sub(self.delay_nanos);

        let first = self.entries[0].as_ref().expect("len invariant");
        if render <= first.time_nanos {
            return Sampled::Clamped(&first.value);
        }
        let last = self.entries[self.len - 1].as_ref().expect("len invariant");
        if render >= last.time_nanos {
            return Sampled::Clamped(&last.value);
        }

        // Find the segment [i, i+1] bracketing `render`.
        for i in 0..self.len - 1 {
            let a = self.entries[i].as_ref().expect("len invariant");
            let b = self.entries[i + 1].as_ref().expect("len invariant");
            if render >= a.time_nanos && render < b.time_nanos {
                let span = (b.time_nanos - a.time_nanos) as f64;
                let into = (render - a.time_nanos) as f64;
                let alpha = if span > 0.0 { into / span } else { 0.0 };
                return Sampled::Interpolated {
                    from: &a.value,
                    to: &b.value,
                    alpha,
                };
            }
        }
        // Unreachable given the range checks above, but stay total: hold newest.
        Sampled::Clamped(&last.value)
    }

    /// Sample and apply a caller-supplied interpolation function.
    ///
    /// `f(from, to, alpha)` is called for a bracketed sample; a clamped sample
    /// calls `f(value, value, 0.0)`. Returns `None` only when the buffer is
    /// empty. Works for any payload type.
    pub fn sample_with<F, R>(&self, now: core::time::Duration, f: F) -> Option<R>
    where
        F: FnOnce(&T, &T, f64) -> R,
    {
        match self.sample(now) {
            Sampled::Empty => None,
            Sampled::Clamped(v) => Some(f(v, v, 0.0)),
            Sampled::Interpolated { from, to, alpha } => Some(f(from, to, alpha)),
        }
    }
}

impl<T: Lerp + Clone, const N: usize> InterpolationBuffer<T, N> {
    /// Sample the timeline and return the interpolated value using the payload's
    /// [`Lerp`] implementation. Returns `None` only when the buffer is empty.
    #[inline]
    pub fn sample_lerp(&self, now: core::time::Duration) -> Option<T> {
        self.sample_with(now, Lerp::lerp)
    }
}

/// Convert a `u128` nanosecond count to a [`Duration`](core::time::Duration),
/// saturating the seconds field.
#[inline]
fn duration_from_nanos_u128(nanos: u128) -> core::time::Duration {
    let secs = (nanos / 1_000_000_000).min(u64::MAX as u128) as u64;
    let sub = (nanos % 1_000_000_000) as u32;
    core::time::Duration::new(secs, sub)
}

#[cfg(test)]
mod tests {
    use super::{InterpolationBuffer, Sampled};
    use crate::Duration;

    fn ms(n: u64) -> Duration {
        Duration::from_millis(n)
    }

    #[test]
    fn empty_buffer_samples_empty() {
        let buf = InterpolationBuffer::<f64, 8>::new(ms(100));
        assert!(buf.is_empty());
        assert_eq!(buf.sample(ms(500)), Sampled::Empty);
        assert_eq!(buf.sample_lerp(ms(500)), None);
    }

    #[test]
    fn interpolates_at_delayed_render_time() {
        let mut buf = InterpolationBuffer::<f64, 8>::new(ms(100));
        buf.push(ms(0), 0.0);
        buf.push(ms(100), 10.0);
        buf.push(ms(200), 20.0);
        // now=250 -> render=150 -> between 100(10.0) and 200(20.0), alpha 0.5.
        match buf.sample(ms(250)) {
            Sampled::Interpolated { from, to, alpha } => {
                assert_eq!(*from, 10.0);
                assert_eq!(*to, 20.0);
                assert!((alpha - 0.5).abs() < 1e-12);
            }
            other => panic!("expected interpolation, got {other:?}"),
        }
        assert_eq!(buf.sample_lerp(ms(250)), Some(15.0));
    }

    #[test]
    fn clamps_before_oldest_and_after_newest() {
        let mut buf = InterpolationBuffer::<f64, 8>::new(ms(100));
        buf.push(ms(100), 1.0);
        buf.push(ms(200), 2.0);
        // render = 0 -> before oldest(100): hold oldest.
        assert_eq!(buf.sample(ms(100)), Sampled::Clamped(&1.0));
        // render = 300 -> after newest(200): hold newest.
        assert_eq!(buf.sample(ms(400)), Sampled::Clamped(&2.0));
    }

    #[test]
    fn sampled_value_is_monotonic_for_monotonic_input() {
        let mut buf = InterpolationBuffer::<f64, 16>::new(ms(100));
        for i in 0..10 {
            buf.push(ms(i * 100), i as f64 * 10.0);
        }
        let mut last = f64::NEG_INFINITY;
        // Sweep render time forward; sampled value must never decrease.
        for now in (100..=1000).step_by(10) {
            if let Some(v) = buf.sample_lerp(ms(now)) {
                assert!(v + 1e-9 >= last, "non-monotonic at now={now}: {v} < {last}");
                last = v;
            }
        }
    }

    #[test]
    fn out_of_order_push_is_sorted() {
        let mut buf = InterpolationBuffer::<f64, 8>::new(ms(0));
        buf.push(ms(300), 3.0);
        buf.push(ms(100), 1.0);
        buf.push(ms(200), 2.0);
        assert_eq!(buf.oldest(), Some(ms(100)));
        assert_eq!(buf.newest(), Some(ms(300)));
        // With zero delay, render=now: midpoint of 100..200 interpolates.
        assert_eq!(buf.sample_lerp(ms(150)), Some(1.5));
    }

    #[test]
    fn duplicate_timestamp_replaces_value() {
        let mut buf = InterpolationBuffer::<f64, 8>::new(ms(0));
        buf.push(ms(100), 1.0);
        buf.push(ms(100), 9.0);
        assert_eq!(buf.len(), 1);
        assert_eq!(buf.sample(ms(100)), Sampled::Clamped(&9.0));
    }

    #[test]
    fn full_buffer_evicts_oldest() {
        let mut buf = InterpolationBuffer::<f64, 4>::new(ms(0));
        for i in 0..6 {
            buf.push(ms(i * 100), i as f64);
        }
        assert_eq!(buf.len(), 4);
        // Oldest two (0,100) evicted; window is now 200..500.
        assert_eq!(buf.oldest(), Some(ms(200)));
        assert_eq!(buf.newest(), Some(ms(500)));
    }

    #[test]
    fn stale_push_into_full_buffer_is_dropped() {
        let mut buf = InterpolationBuffer::<f64, 4>::new(ms(0));
        for i in 1..=4 {
            buf.push(ms(i * 100), i as f64);
        }
        // Buffer full with 100..400. A very old arrival is dropped.
        buf.push(ms(10), -1.0);
        assert_eq!(buf.len(), 4);
        assert_eq!(buf.oldest(), Some(ms(100)));
    }

    #[test]
    fn sample_with_handles_arbitrary_payload() {
        // Non-Lerp payload: a 2D point interpolated via closure.
        let mut buf = InterpolationBuffer::<(f64, f64), 8>::new(ms(0));
        buf.push(ms(0), (0.0, 0.0));
        buf.push(ms(100), (10.0, 20.0));
        let p = buf
            .sample_with(ms(50), |a, b, t| {
                (a.0 + (b.0 - a.0) * t, a.1 + (b.1 - a.1) * t)
            })
            .unwrap();
        assert!((p.0 - 5.0).abs() < 1e-12);
        assert!((p.1 - 10.0).abs() < 1e-12);
    }

    #[test]
    fn clear_empties_buffer() {
        let mut buf = InterpolationBuffer::<f64, 8>::new(ms(100));
        buf.push(ms(0), 1.0);
        buf.push(ms(100), 2.0);
        buf.clear();
        assert!(buf.is_empty());
        assert_eq!(buf.sample(ms(200)), Sampled::Empty);
    }

    #[test]
    fn determinism_double_run_is_bit_equivalent() {
        let run = || {
            let mut buf = InterpolationBuffer::<f64, 16>::new(ms(100));
            let mut trace = [0.0f64; 50];
            for (i, slot) in trace.iter_mut().enumerate() {
                buf.push(ms(i as u64 * 33), (i as f64 * 7.0) % 13.0);
                *slot = buf.sample_lerp(ms(i as u64 * 33)).unwrap_or(0.0);
            }
            trace
        };
        let a = run();
        let b = run();
        assert_eq!(a, b);
    }
}
