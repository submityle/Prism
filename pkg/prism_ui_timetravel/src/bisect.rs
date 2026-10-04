//! `git bisect`-style regression finding over a recorded [`Timeline`].
//!
//! When a bug first appears somewhere in a long history, scanning every frame
//! is `O(n)` predicate evaluations. If the property under test is *monotone* —
//! once it becomes true it stays true, exactly like "this frame is broken" for
//! a regression that never heals — the first offending frame can be located in
//! `O(log n)` evaluations by binary search. This is the same strategy a
//! developer uses with `git bisect`, lifted onto the UI time-travel timeline.
//!
//! Two entry points are provided:
//!
//! * [`first_matching`] — a one-shot search that evaluates the predicate itself
//!   and returns the first matching frame index.
//! * [`Bisection`] — an *interactive* session mirroring `git bisect good/bad`:
//!   it proposes a candidate frame, the caller inspects it and reports
//!   [`Bisection::mark_good`] or [`Bisection::mark_bad`], and the session
//!   narrows until it pins the first bad frame. This suits a human-in-the-loop
//!   debugger where "is this frame broken?" is a judgement call.
//!
//! Both assume the predicate is monotone over the timeline (a run of
//! non-matching frames followed by a run of matching frames). The result is
//! only meaningful under that assumption, which the documentation on each item
//! spells out.
//!
//! All arithmetic is integer index math — deterministic and free of
//! floating-point — so results are perfectly reproducible.

use crate::frame::Frame;
use crate::timeline::Timeline;

/// Returns the index of the first frame in `timeline` for which `pred` is
/// `true`, using binary search, or [`None`] when no frame matches.
///
/// `pred` must be **monotone** across the timeline: there is a (possibly empty)
/// prefix of frames for which it is `false` followed by a (possibly empty)
/// suffix for which it is `true`. Under that contract the function evaluates
/// `pred` at most `ceil(log2(len)) + 1` times. If the predicate is not monotone
/// the returned index is unspecified (but always in range or [`None`]).
///
/// # Example
///
/// ```
/// use prism_ui::Element;
/// use prism_ui_timetravel::{Frame, Timeline};
/// use prism_ui_timetravel::bisect::first_matching;
///
/// let mut timeline = Timeline::new();
/// for i in 0..8 {
///     let text = if i >= 5 { "broken" } else { "ok" };
///     timeline.record(Frame::capture("f", &Element::box_().child(Element::text(text))));
/// }
/// // The regression first appears at frame 5.
/// let first = first_matching(&timeline, |f| {
///     f.snapshot().find(|n| n.text.as_deref() == Some("broken")).is_some()
/// });
/// assert_eq!(first, Some(5));
/// ```
pub fn first_matching<F>(timeline: &Timeline, mut pred: F) -> Option<usize>
where
    F: FnMut(&Frame) -> bool,
{
    let frames = timeline.frames();
    let index = first_matching_index(frames.len(), |i| pred(&frames[i]))?;
    Some(index)
}

/// Index-only core of [`first_matching`]: returns the smallest `i < len` for
/// which `pred(i)` is `true`, or [`None`] when none is.
///
/// `pred` must be monotone in `i` (see [`first_matching`]). This is the classic
/// partition-point binary search and is independent of [`Frame`], so it can
/// bisect any monotone indexed sequence.
pub fn first_matching_index<F>(len: usize, mut pred: F) -> Option<usize>
where
    F: FnMut(usize) -> bool,
{
    let mut lo = 0;
    let mut hi = len;
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if pred(mid) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    if lo < len {
        Some(lo)
    } else {
        None
    }
}

/// An interactive `git bisect` session narrowing in on the first "bad" frame.
///
/// The session tracks a known-good index (predicate `false`) and a known-bad
/// index (predicate `true`) with `good < bad`, so the first bad frame always
/// lies in `good + 1 ..= bad`. [`Bisection::next_candidate`] proposes the frame
/// to inspect next; the caller reports the verdict with [`Bisection::mark_good`]
/// or [`Bisection::mark_bad`], shrinking the window roughly in half each time.
/// When the window closes to a single step the search is done and
/// [`Bisection::first_bad`] yields the answer.
///
/// This mirrors a human-in-the-loop debugger: unlike [`first_matching`], the
/// verdict for each candidate comes from outside, so a developer can make the
/// "is this frame broken?" call by eye.
///
/// # Example
///
/// ```
/// use prism_ui_timetravel::bisect::Bisection;
///
/// // Frames 0..8; the regression is first present at frame 5. We know frame 0
/// // is good and frame 7 is bad.
/// let broken = |i: usize| i >= 5;
/// let mut session = Bisection::new(0, 7);
/// while let Some(candidate) = session.next_candidate() {
///     if broken(candidate) {
///         session.mark_bad(candidate);
///     } else {
///         session.mark_good(candidate);
///     }
/// }
/// assert_eq!(session.first_bad(), Some(5));
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bisection {
    /// Highest index known to be good (predicate `false`).
    good: usize,
    /// Lowest index known to be bad (predicate `true`); always `> good`.
    bad: usize,
}

impl Bisection {
    /// Starts a session given a known-good and known-bad index.
    ///
    /// # Panics
    ///
    /// Panics unless `good < bad`. The caller is asserting that the frame at
    /// `good` does not exhibit the regression and the frame at `bad` does.
    #[must_use]
    pub fn new(good: usize, bad: usize) -> Self {
        assert!(good < bad, "bisection requires good < bad");
        Self { good, bad }
    }

    /// The current known-good index (predicate `false`).
    #[must_use]
    pub const fn good(&self) -> usize {
        self.good
    }

    /// The current known-bad index (predicate `true`).
    #[must_use]
    pub const fn bad(&self) -> usize {
        self.bad
    }

    /// The number of candidate frames still unclassified between the good and
    /// bad bounds (`bad - good - 1`).
    #[must_use]
    pub const fn remaining(&self) -> usize {
        self.bad - self.good - 1
    }

    /// The next frame index to inspect, or [`None`] once the window has closed
    /// to a single step (search complete).
    ///
    /// The candidate is always strictly between [`Bisection::good`] and
    /// [`Bisection::bad`].
    #[must_use]
    pub const fn next_candidate(&self) -> Option<usize> {
        if self.bad - self.good <= 1 {
            None
        } else {
            Some(self.good + (self.bad - self.good) / 2)
        }
    }

    /// Records that the frame at `index` is good (does not exhibit the
    /// regression), raising the good bound.
    ///
    /// # Panics
    ///
    /// Panics unless `good < index < bad`, i.e. `index` is a currently
    /// unclassified candidate.
    pub fn mark_good(&mut self, index: usize) {
        assert!(
            self.good < index && index < self.bad,
            "marked index must be an unclassified candidate"
        );
        self.good = index;
    }

    /// Records that the frame at `index` is bad (exhibits the regression),
    /// lowering the bad bound.
    ///
    /// # Panics
    ///
    /// Panics unless `good < index < bad`, i.e. `index` is a currently
    /// unclassified candidate.
    pub fn mark_bad(&mut self, index: usize) {
        assert!(
            self.good < index && index < self.bad,
            "marked index must be an unclassified candidate"
        );
        self.bad = index;
    }

    /// The first bad frame once the search has converged, or [`None`] while
    /// candidates remain.
    ///
    /// Convergence means the good and bad bounds are adjacent (`bad == good +
    /// 1`); the first bad frame is then the bad bound.
    #[must_use]
    pub const fn first_bad(&self) -> Option<usize> {
        if self.bad - self.good == 1 {
            Some(self.bad)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::Frame;
    use prism_ui::Element;

    struct SplitMix64(u64);
    impl SplitMix64 {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        /// A `usize` in `[lo, hi)`; `hi` must exceed `lo`.
        fn below(&mut self, lo: usize, hi: usize) -> usize {
            lo + (self.next_u64() % (hi - lo) as u64) as usize
        }
    }

    /// Linear-scan oracle: the first index whose boolean is `true`.
    fn linear_first_true(bits: &[bool]) -> Option<usize> {
        bits.iter().position(|&b| b)
    }

    /// `ceil(log2(span))`, computed by an independent halving loop.
    fn ceil_log2(mut span: usize) -> u32 {
        let mut c = 0;
        while span > 1 {
            span = span.div_ceil(2);
            c += 1;
        }
        c
    }

    #[test]
    fn first_matching_index_matches_linear_scan() {
        let mut rng = SplitMix64(0x1111_2222_3333_0001);
        for _ in 0..1000 {
            let len = rng.below(1, 64);
            // Build a monotone boolean array: false up to `threshold`, true after.
            let threshold = rng.below(0, len + 1); // 0..=len
            let bits: Vec<bool> = (0..len).map(|i| i >= threshold).collect();

            let mut evals = 0u32;
            let got = first_matching_index(len, |i| {
                evals += 1;
                bits[i]
            });
            assert_eq!(got, linear_first_true(&bits), "bisect disagreed with scan");
            // Logarithmic evaluation budget.
            assert!(
                evals <= ceil_log2(len) + 1,
                "too many evals: {} for len {}",
                evals,
                len
            );
        }
    }

    #[test]
    fn first_matching_index_edge_cases() {
        assert_eq!(first_matching_index(0, |_| true), None);
        assert_eq!(first_matching_index(5, |_| false), None); // all good
        assert_eq!(first_matching_index(5, |_| true), Some(0)); // all bad
        assert_eq!(first_matching_index(1, |_| true), Some(0));
    }

    #[test]
    fn interactive_session_converges_like_scan() {
        let mut rng = SplitMix64(0xAAAA_BBBB_CCCC_0002);
        for _ in 0..1000 {
            let len = rng.below(2, 128);
            // Threshold in 1..len so that frame 0 is good and frame len-1 is bad.
            let threshold = rng.below(1, len);
            let broken = |i: usize| i >= threshold;

            let mut session = Bisection::new(0, len - 1);
            let mut steps = 0u32;
            while let Some(candidate) = session.next_candidate() {
                // The candidate is always a fresh, in-range unclassified frame.
                assert!(session.good() < candidate && candidate < session.bad());
                if broken(candidate) {
                    session.mark_bad(candidate);
                } else {
                    session.mark_good(candidate);
                }
                steps += 1;
            }
            assert_eq!(session.first_bad(), Some(threshold), "wrong first-bad frame");
            assert_eq!(session.first_bad(), linear_first_true_from(len, threshold));
            // Logarithmic step budget over the initial span.
            assert!(
                steps <= ceil_log2(len - 1),
                "too many steps: {} for span {}",
                steps,
                len - 1
            );
        }
    }

    /// Oracle mirror of the interactive setup: first bad is simply `threshold`.
    fn linear_first_true_from(len: usize, threshold: usize) -> Option<usize> {
        (0..len).find(|&i| i >= threshold)
    }

    #[test]
    fn session_invariants_hold_throughout() {
        let broken = |i: usize| i >= 9;
        let mut session = Bisection::new(0, 15);
        let mut prev_remaining = session.remaining();
        while let Some(candidate) = session.next_candidate() {
            if broken(candidate) {
                session.mark_bad(candidate);
            } else {
                session.mark_good(candidate);
            }
            // Each step strictly shrinks the unclassified window.
            assert!(session.remaining() < prev_remaining, "window did not shrink");
            prev_remaining = session.remaining();
            // Good bound stays good, bad bound stays bad.
            assert!(!broken(session.good()), "good bound became bad");
            assert!(broken(session.bad()), "bad bound became good");
        }
        assert_eq!(session.first_bad(), Some(9));
    }

    #[test]
    fn first_matching_on_timeline_finds_regression() {
        let mut timeline = Timeline::new();
        for i in 0..16 {
            let text = if i >= 11 { "broken" } else { "ok" };
            timeline.record(Frame::capture("f", &Element::box_().child(Element::text(text))));
        }
        let got = first_matching(&timeline, |f| {
            f.snapshot().find(|n| n.text.as_deref() == Some("broken")).is_some()
        });
        assert_eq!(got, Some(11));

        // No matching frame -> None.
        let none = first_matching(&timeline, |f| {
            f.snapshot().find(|n| n.text.as_deref() == Some("nope")).is_some()
        });
        assert_eq!(none, None);
    }

    #[test]
    #[should_panic(expected = "good < bad")]
    fn new_rejects_bad_bounds() {
        let _ = Bisection::new(3, 3);
    }

    #[test]
    #[should_panic(expected = "unclassified candidate")]
    fn mark_rejects_out_of_window_index() {
        let mut session = Bisection::new(0, 10);
        session.mark_bad(10); // == bad bound, not a candidate
    }
}
