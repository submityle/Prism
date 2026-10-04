//! System profiling: nested span timing and flame-graph export
//! (design §16.6 "系统火焰图").
//!
//! The kernel does not own a clock or a thread pool, so profiling is split
//! into two layers:
//!
//! * a **`no_std` core** — [`SystemInstrument`] (the hook the executor calls
//!   around each system), plus [`SpanNode`] / [`FlameGraph`], the timing tree
//!   and its exports. These are pure data: a caller on any platform can build
//!   a [`FlameGraph`] from [`Duration`](core::time::Duration) values it
//!   measured however it likes.
//! * a **`std` recorder** — [`SpanRecorder`] (behind the `std` feature) which
//!   implements [`SystemInstrument`] using [`std::time::Instant`], turning the
//!   executor's `begin`/`end` calls into a nested [`SpanNode`] tree.
//!
//! The default executor path is zero-overhead: [`SystemInstrument`] is
//! implemented for `()` as a no-op, and
//! [`SingleThreadedExecutor::run`](crate::schedule::SingleThreadedExecutor::run)
//! instruments with `()`. Only
//! [`run_instrumented`](crate::schedule::SingleThreadedExecutor::run_instrumented)
//! (also `std`-gated) threads a real [`SpanRecorder`] through.
//!
//! Timings describe wall-clock span durations only; they carry no allocation
//! or cache accounting. The flame graph's weights use *self time* (a span's
//! own duration minus its children), matching the folded-stack convention
//! consumed by common flame-graph renderers.

use alloc::string::String;
use alloc::vec::Vec;
use core::time::Duration;

/// A hook the executor calls around each system so a profiler can time nested
/// spans without the executor knowing how timing is implemented.
///
/// `begin`/`end` calls are perfectly nested (stack discipline): the executor
/// calls [`begin`](SystemInstrument::begin) before a system runs and
/// [`end`](SystemInstrument::end) after it returns, so a recorder can treat
/// them as push/pop on a span stack. The blanket no-op implementation for `()`
/// keeps the default (uninstrumented) run path free of overhead.
pub trait SystemInstrument {
    /// Open a new span labelled `label` (e.g. a system name). Must be paired
    /// with a later [`end`](SystemInstrument::end).
    fn begin(&mut self, label: &str);

    /// Close the most recently opened span.
    fn end(&mut self);
}

/// A no-op instrument: the zero-overhead default used by the uninstrumented
/// executor path.
impl SystemInstrument for () {
    #[inline]
    fn begin(&mut self, _label: &str) {}

    #[inline]
    fn end(&mut self) {}
}

/// One node in a profiling tree: a labelled span, its total wall-clock
/// duration, and the child spans opened while it was active.
///
/// `duration` is the span's *inclusive* time (it covers the children);
/// [`self_duration`](SpanNode::self_duration) subtracts the children to give
/// the span's own exclusive cost, which is what flame-graph weights use.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpanNode {
    label: String,
    duration: Duration,
    children: Vec<SpanNode>,
}

impl SpanNode {
    /// Build a span node from its parts.
    #[must_use]
    pub fn new(label: String, duration: Duration, children: Vec<SpanNode>) -> Self {
        Self {
            label,
            duration,
            children,
        }
    }

    /// The span's label (e.g. the system name).
    #[must_use]
    pub fn label(&self) -> &str {
        &self.label
    }

    /// The span's inclusive wall-clock duration (covers its children).
    #[must_use]
    pub fn duration(&self) -> Duration {
        self.duration
    }

    /// The child spans opened while this span was active.
    #[must_use]
    pub fn children(&self) -> &[SpanNode] {
        &self.children
    }

    /// The span's *exclusive* duration: its inclusive duration minus the sum of
    /// its children's inclusive durations, saturating at zero.
    ///
    /// Clock jitter can make the children's measured total momentarily exceed
    /// the parent's; the saturating subtraction keeps self-time non-negative.
    #[must_use]
    pub fn self_duration(&self) -> Duration {
        let children: Duration = self.children.iter().map(SpanNode::duration).sum();
        self.duration.saturating_sub(children)
    }
}

/// A completed profiling tree: the root spans of one instrumented run plus the
/// exports an external flame-graph / hotspot panel consumes (design §16.6).
///
/// Build one from a [`SpanRecorder`] via
/// [`SpanRecorder::flame_graph`](SpanRecorder::flame_graph), or directly from
/// span nodes measured elsewhere via [`from_roots`](FlameGraph::from_roots).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FlameGraph {
    roots: Vec<SpanNode>,
}

impl FlameGraph {
    /// Build a flame graph from its root spans.
    #[must_use]
    pub fn from_roots(roots: Vec<SpanNode>) -> Self {
        Self { roots }
    }

    /// The root spans (one per top-level system in a schedule run).
    #[must_use]
    pub fn roots(&self) -> &[SpanNode] {
        &self.roots
    }

    /// Whether the graph has no spans.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.roots.is_empty()
    }

    /// The number of root spans.
    #[must_use]
    pub fn len(&self) -> usize {
        self.roots.len()
    }

    /// The total inclusive duration across all root spans.
    #[must_use]
    pub fn total_duration(&self) -> Duration {
        self.roots.iter().map(SpanNode::duration).sum()
    }

    /// Export folded stacks in Brendan Gregg's `flamegraph.pl` format: one line
    /// per leaf path `root;child;grandchild <nanos>`, where the weight is the
    /// *self time* (nanoseconds) of the deepest span on that path.
    ///
    /// Every span contributes exactly one line carrying its own self time, so
    /// summing all weights reproduces the total measured time. The output order
    /// is a stable depth-first pre-order walk of the tree.
    #[must_use]
    pub fn folded(&self) -> Vec<String> {
        let mut lines = Vec::new();
        let mut stack: String = String::new();
        for root in &self.roots {
            Self::fold_into(root, &mut stack, &mut lines);
        }
        lines
    }

    fn fold_into(node: &SpanNode, stack: &mut String, lines: &mut Vec<String>) {
        let base = stack.len();
        if base != 0 {
            stack.push(';');
        }
        stack.push_str(node.label());

        let mut line = String::with_capacity(stack.len() + 16);
        line.push_str(stack);
        line.push(' ');
        push_u128(&mut line, node.self_duration().as_nanos());
        lines.push(line);

        for child in node.children() {
            Self::fold_into(child, stack, lines);
        }

        stack.truncate(base);
    }

    /// Aggregate self time by label across the whole tree, returned as
    /// `(label, self_time)` pairs sorted by descending self time (ties broken
    /// by label for determinism). This is the "hottest systems" view.
    #[must_use]
    pub fn hotspots(&self) -> Vec<(String, Duration)> {
        let mut acc: Vec<(String, Duration)> = Vec::new();
        for root in &self.roots {
            Self::accumulate(root, &mut acc);
        }
        acc.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        acc
    }

    fn accumulate(node: &SpanNode, acc: &mut Vec<(String, Duration)>) {
        let self_time = node.self_duration();
        match acc.iter_mut().find(|(label, _)| label == node.label()) {
            Some((_, total)) => *total = total.saturating_add(self_time),
            None => acc.push((String::from(node.label()), self_time)),
        }
        for child in node.children() {
            Self::accumulate(child, acc);
        }
    }
}

/// Append the decimal digits of `value` to `out` without allocating or needing
/// `std`'s formatting machinery.
fn push_u128(out: &mut String, value: u128) {
    if value == 0 {
        out.push('0');
        return;
    }
    // u128 max is 39 decimal digits.
    let mut buf = [0u8; 39];
    let mut i = buf.len();
    let mut v = value;
    while v != 0 {
        i -= 1;
        buf[i] = b'0' + (v % 10) as u8;
        v /= 10;
    }
    // Every byte written above is an ASCII digit, so the cast to `char` is
    // a valid 1:1 code-point mapping.
    for &byte in &buf[i..] {
        out.push(byte as char);
    }
}

#[cfg(feature = "std")]
pub use std_impl::SpanRecorder;

#[cfg(feature = "std")]
mod std_impl {
    use super::{FlameGraph, SpanNode, SystemInstrument};
    use alloc::string::{String, ToString};
    use alloc::vec::Vec;
    use std::time::Instant;

    /// An open (not-yet-closed) span on the recorder's stack.
    struct Open {
        label: String,
        start: Instant,
        children: Vec<SpanNode>,
    }

    /// A `std` [`SystemInstrument`] that times perfectly-nested spans with
    /// [`Instant`] and assembles them into a [`SpanNode`] tree.
    ///
    /// Drive it through
    /// [`run_instrumented`](crate::schedule::SingleThreadedExecutor::run_instrumented):
    /// each [`begin`](SystemInstrument::begin) pushes a span (stamping
    /// [`Instant::now`]), and each [`end`](SystemInstrument::end) pops it,
    /// recording [`Instant::elapsed`] and attaching the finished [`SpanNode`]
    /// to its parent's children (or to the roots at depth zero). After a run,
    /// read [`roots`](SpanRecorder::roots) or build a
    /// [`FlameGraph`](SpanRecorder::flame_graph).
    #[derive(Default)]
    pub struct SpanRecorder {
        stack: Vec<Open>,
        roots: Vec<SpanNode>,
    }

    impl SpanRecorder {
        /// Create an empty recorder.
        #[must_use]
        pub fn new() -> Self {
            Self::default()
        }

        /// The completed root spans recorded so far.
        #[must_use]
        pub fn roots(&self) -> &[SpanNode] {
            &self.roots
        }

        /// Consume the recorder, returning its completed root spans.
        #[must_use]
        pub fn into_roots(self) -> Vec<SpanNode> {
            self.roots
        }

        /// Build a [`FlameGraph`] from the completed root spans (clones them,
        /// leaving the recorder usable).
        #[must_use]
        pub fn flame_graph(&self) -> FlameGraph {
            FlameGraph::from_roots(self.roots.clone())
        }

        /// Whether every opened span has been closed (the span stack is empty).
        /// A balanced recorder after a run means `begin`/`end` were perfectly
        /// paired.
        #[must_use]
        pub fn is_balanced(&self) -> bool {
            self.stack.is_empty()
        }
    }

    impl SystemInstrument for SpanRecorder {
        fn begin(&mut self, label: &str) {
            self.stack.push(Open {
                label: label.to_string(),
                start: Instant::now(),
                children: Vec::new(),
            });
        }

        fn end(&mut self) {
            let Some(open) = self.stack.pop() else {
                // Unbalanced `end` without a matching `begin`: ignore rather
                // than panic, so a misuse never aborts a running schedule.
                return;
            };
            let node = SpanNode::new(open.label, open.start.elapsed(), open.children);
            match self.stack.last_mut() {
                Some(parent) => parent.children.push(node),
                None => self.roots.push(node),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;
    use alloc::vec;

    fn node(label: &str, nanos: u64, children: Vec<SpanNode>) -> SpanNode {
        SpanNode::new(label.to_string(), Duration::from_nanos(nanos), children)
    }

    #[test]
    fn self_duration_subtracts_children_saturating() {
        let n = node(
            "parent",
            100,
            vec![node("a", 30, Vec::new()), node("b", 20, Vec::new())],
        );
        assert_eq!(n.self_duration(), Duration::from_nanos(50));

        // Children exceeding the parent saturate to zero rather than underflow.
        let skewed = node("p", 10, vec![node("c", 40, Vec::new())]);
        assert_eq!(skewed.self_duration(), Duration::ZERO);
    }

    #[test]
    fn folded_emits_one_line_per_span_with_self_time() {
        let graph = FlameGraph::from_roots(vec![node(
            "update",
            100,
            vec![node(
                "physics",
                40,
                vec![node("broadphase", 10, Vec::new())],
            )],
        )]);
        let folded = graph.folded();
        assert_eq!(
            folded,
            vec![
                // update self = 100 - 40 = 60
                "update 60".to_string(),
                // physics self = 40 - 10 = 30
                "update;physics 30".to_string(),
                // broadphase is a leaf: self = 10
                "update;physics;broadphase 10".to_string(),
            ]
        );
    }

    #[test]
    fn hotspots_rank_by_self_time_descending() {
        let graph = FlameGraph::from_roots(vec![
            node("a", 100, vec![node("inner", 90, Vec::new())]),
            node("b", 70, Vec::new()),
        ]);
        let hot = graph.hotspots();
        // a self = 10, inner self = 90, b self = 70 → inner, b, a.
        assert_eq!(
            hot,
            vec![
                ("inner".to_string(), Duration::from_nanos(90)),
                ("b".to_string(), Duration::from_nanos(70)),
                ("a".to_string(), Duration::from_nanos(10)),
            ]
        );
    }

    #[test]
    fn hotspots_merge_repeated_labels() {
        let graph = FlameGraph::from_roots(vec![
            node("sys", 30, Vec::new()),
            node("sys", 20, Vec::new()),
        ]);
        let hot = graph.hotspots();
        assert_eq!(hot, vec![("sys".to_string(), Duration::from_nanos(50))]);
    }

    #[test]
    fn total_duration_sums_roots() {
        let graph = FlameGraph::from_roots(vec![
            node("a", 30, vec![node("x", 10, Vec::new())]),
            node("b", 20, Vec::new()),
        ]);
        assert_eq!(graph.total_duration(), Duration::from_nanos(50));
        assert_eq!(graph.len(), 2);
        assert!(!graph.is_empty());
    }

    #[test]
    fn unit_instrument_is_noop() {
        let mut noop = ();
        noop.begin("x");
        noop.end();
        // Nothing to assert beyond "does not panic / compiles as a no-op".
    }

    #[cfg(feature = "std")]
    #[test]
    fn recorder_builds_nested_tree() {
        let mut rec = SpanRecorder::new();
        rec.begin("outer");
        rec.begin("inner");
        rec.end();
        rec.end();
        assert!(rec.is_balanced());

        let roots = rec.roots();
        assert_eq!(roots.len(), 1);
        assert_eq!(roots[0].label(), "outer");
        assert_eq!(roots[0].children().len(), 1);
        assert_eq!(roots[0].children()[0].label(), "inner");

        let graph = rec.flame_graph();
        assert_eq!(graph.roots().len(), 1);
        assert!(!graph.folded().is_empty());
    }

    #[cfg(feature = "std")]
    #[test]
    fn recorder_tolerates_unbalanced_end() {
        let mut rec = SpanRecorder::new();
        rec.end(); // stray end: ignored
        assert!(rec.is_balanced());
        assert!(rec.roots().is_empty());
    }
}
