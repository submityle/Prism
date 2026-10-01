//! Aggregating op traces and tree shape into human-readable reports.
//!
//! [`PerfReport`] turns an [`OpTrace`] into per-category counts plus derived
//! integer metrics such as *churn* (structural mutations) and whole-percent
//! shares of the total. [`TreeMetrics`] summarises the shape of a
//! [`TreeSnapshot`]: node count, depth, a per-kind histogram, and the widest
//! fan-out. Both avoid floating point entirely — percentages are computed with
//! integer arithmetic — so their output is deterministic and portable to
//! `no_std` targets.

use alloc::collections::BTreeMap;
use alloc::string::String;
use core::fmt::Write as _;

use prism_ui_devtools::{OpTrace, SnapshotNode, TreeSnapshot};

/// Per-category counts and derived metrics for a flush's [`OpTrace`].
///
/// Build one with [`PerfReport::from_trace`]. The six count fields mirror the
/// [`OpTrace`] variants; [`PerfReport::total`] is their sum.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PerfReport {
    /// Number of create ops.
    pub creates: usize,
    /// Number of remove ops.
    pub removes: usize,
    /// Number of text-update ops.
    pub set_texts: usize,
    /// Number of layout-update ops.
    pub set_layouts: usize,
    /// Number of paint-update ops.
    pub set_paints: usize,
    /// Number of reorder ops.
    pub reorders: usize,
    /// Grand total of all ops.
    pub total: usize,
}

impl PerfReport {
    /// Aggregates an [`OpTrace`] into a report.
    #[must_use]
    pub fn from_trace(trace: &OpTrace) -> Self {
        Self {
            creates: trace.creates(),
            removes: trace.removes(),
            set_texts: trace.set_texts(),
            set_layouts: trace.set_layouts(),
            set_paints: trace.set_paints(),
            reorders: trace.reorders(),
            total: trace.total(),
        }
    }

    /// Structural churn: the number of create, remove, and reorder ops.
    ///
    /// These are the ops that change the shape of the retained tree, as opposed
    /// to text/layout/paint updates that only refresh existing nodes.
    #[must_use]
    pub fn churn(&self) -> usize {
        self.creates + self.removes + self.reorders
    }

    /// Whole-percent share of `count` relative to [`PerfReport::total`].
    ///
    /// Returns `0` when the total is `0`, avoiding division by zero. The result
    /// is truncated toward zero and lies in `0..=100`.
    #[must_use]
    pub fn percent_of_total(&self, count: usize) -> usize {
        (count.saturating_mul(100))
            .checked_div(self.total)
            .unwrap_or(0)
    }

    /// Whole-percent share of [`PerfReport::churn`] relative to the total.
    #[must_use]
    pub fn churn_percent(&self) -> usize {
        self.percent_of_total(self.churn())
    }

    /// Renders a deterministic, multi-line human-readable summary.
    #[must_use]
    pub fn report(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(out, "ops total: {}", self.total);
        let _ = writeln!(
            out,
            "  create={} ({}%)",
            self.creates,
            self.percent_of_total(self.creates)
        );
        let _ = writeln!(
            out,
            "  remove={} ({}%)",
            self.removes,
            self.percent_of_total(self.removes)
        );
        let _ = writeln!(
            out,
            "  set_text={} ({}%)",
            self.set_texts,
            self.percent_of_total(self.set_texts)
        );
        let _ = writeln!(
            out,
            "  set_layout={} ({}%)",
            self.set_layouts,
            self.percent_of_total(self.set_layouts)
        );
        let _ = writeln!(
            out,
            "  set_paint={} ({}%)",
            self.set_paints,
            self.percent_of_total(self.set_paints)
        );
        let _ = writeln!(
            out,
            "  reorder={} ({}%)",
            self.reorders,
            self.percent_of_total(self.reorders)
        );
        let _ = writeln!(out, "churn: {} ({}%)", self.churn(), self.churn_percent());
        out
    }
}

/// Structural metrics describing the shape of a [`TreeSnapshot`].
///
/// Build one with [`TreeMetrics::from_snapshot`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TreeMetrics {
    /// Total number of nodes, including the root.
    pub node_count: usize,
    /// Depth of the tree; a lone root has depth `1`.
    pub depth: usize,
    /// Count of nodes per kind label, in sorted key order.
    pub kind_histogram: BTreeMap<String, usize>,
    /// The largest number of direct children held by any single node.
    pub max_fan_out: usize,
}

impl TreeMetrics {
    /// Computes metrics for `tree` in a single depth-first walk.
    #[must_use]
    pub fn from_snapshot(tree: &TreeSnapshot) -> Self {
        let mut metrics = Self {
            node_count: 0,
            depth: tree.depth(),
            kind_histogram: BTreeMap::new(),
            max_fan_out: 0,
        };
        metrics.visit(&tree.root);
        metrics
    }

    /// Depth-first accumulation of counts, histogram, and fan-out.
    fn visit(&mut self, node: &SnapshotNode) {
        self.node_count += 1;
        *self.kind_histogram.entry(node.kind.clone()).or_insert(0) += 1;
        let fan_out = node.children.len();
        if fan_out > self.max_fan_out {
            self.max_fan_out = fan_out;
        }
        for child in &node.children {
            self.visit(child);
        }
    }

    /// Renders a deterministic, multi-line human-readable summary.
    #[must_use]
    pub fn report(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(out, "nodes: {}", self.node_count);
        let _ = writeln!(out, "depth: {}", self.depth);
        let _ = writeln!(out, "max_fan_out: {}", self.max_fan_out);
        let _ = writeln!(out, "kinds:");
        for (kind, count) in &self.kind_histogram {
            let _ = writeln!(out, "  {kind}={count}");
        }
        out
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    extern crate alloc;

    use alloc::collections::BTreeMap;
    use alloc::string::ToString;

    use prism_ui::layout::{AvailableSpace, Size};
    use prism_ui::{Element, RecordingBackend, Ui};
    use prism_ui_devtools::{snapshot, OpTrace};

    use super::{PerfReport, TreeMetrics};

    fn definite(w: f32, h: f32) -> Size<AvailableSpace> {
        Size::new(AvailableSpace::Definite(w), AvailableSpace::Definite(h))
    }

    fn mounted_trace(view: &Element, layout: bool) -> OpTrace {
        let mut ui = Ui::new(RecordingBackend::new());
        ui.mount(view);
        if layout {
            ui.compute_layout(definite(800.0, 600.0));
        }
        OpTrace::from_ops(ui.backend().ops())
    }

    #[test]
    fn empty_trace_report() {
        let report = PerfReport::from_trace(&OpTrace::from_ops(&[]));
        assert_eq!(report.total, 0);
        assert_eq!(report.churn(), 0);
        assert_eq!(report.churn_percent(), 0);
        assert_eq!(report.percent_of_total(0), 0);
    }

    #[test]
    fn counts_match_trace() {
        let view = Element::box_()
            .child(Element::text("a"))
            .child(Element::text("b"));
        let trace = mounted_trace(&view, true);
        let report = PerfReport::from_trace(&trace);
        assert_eq!(report.creates, 3);
        assert_eq!(report.set_texts, 2);
        assert_eq!(report.set_layouts, 3);
        assert_eq!(report.removes, 0);
        assert_eq!(report.reorders, 0);
        assert_eq!(report.total, trace.total());
    }

    #[test]
    fn churn_is_structural_ops() {
        let view = Element::box_().child(Element::text("a"));
        let report = PerfReport::from_trace(&mounted_trace(&view, false));
        // Two creates, no removes/reorders -> churn == creates.
        assert_eq!(report.churn(), report.creates);
    }

    #[test]
    fn percent_of_total_is_integer_share() {
        let report = PerfReport {
            creates: 1,
            removes: 1,
            set_texts: 1,
            set_layouts: 1,
            set_paints: 0,
            reorders: 0,
            total: 4,
        };
        assert_eq!(report.percent_of_total(1), 25);
        assert_eq!(report.percent_of_total(4), 100);
        assert_eq!(report.churn(), 2);
        assert_eq!(report.churn_percent(), 50);
    }

    #[test]
    fn percent_truncates_toward_zero() {
        let report = PerfReport {
            creates: 1,
            removes: 0,
            set_texts: 0,
            set_layouts: 0,
            set_paints: 0,
            reorders: 0,
            total: 3,
        };
        // 1/3 -> 33% after truncation.
        assert_eq!(report.percent_of_total(1), 33);
    }

    #[test]
    fn report_string_contains_total_and_churn() {
        let view = Element::box_().child(Element::text("a"));
        let report = PerfReport::from_trace(&mounted_trace(&view, false));
        let text = report.report();
        assert!(text.contains("ops total:"));
        assert!(text.contains("churn:"));
    }

    #[test]
    fn tree_metrics_basic() {
        let view = Element::box_()
            .child(Element::box_().child(Element::text("deep")))
            .child(Element::text("sib"));
        let snap = snapshot(&view);
        let metrics = TreeMetrics::from_snapshot(&snap);
        assert_eq!(metrics.node_count, snap.node_count());
        assert_eq!(metrics.node_count, 4);
        assert_eq!(metrics.depth, 3);
        assert_eq!(metrics.max_fan_out, 2);
    }

    #[test]
    fn tree_metrics_histogram() {
        let view = Element::box_()
            .child(Element::text("a"))
            .child(Element::text("b"))
            .child(Element::custom("Canvas"));
        let snap = snapshot(&view);
        let metrics = TreeMetrics::from_snapshot(&snap);
        let mut expected = BTreeMap::new();
        expected.insert("Box".to_string(), 1);
        expected.insert("Text".to_string(), 2);
        expected.insert("Canvas".to_string(), 1);
        assert_eq!(metrics.kind_histogram, expected);
    }

    #[test]
    fn tree_metrics_single_node() {
        let snap = snapshot(&Element::box_());
        let metrics = TreeMetrics::from_snapshot(&snap);
        assert_eq!(metrics.node_count, 1);
        assert_eq!(metrics.depth, 1);
        assert_eq!(metrics.max_fan_out, 0);
    }

    #[test]
    fn tree_metrics_report_lists_kinds() {
        let view = Element::box_().child(Element::text("a"));
        let snap = snapshot(&view);
        let text = TreeMetrics::from_snapshot(&snap).report();
        assert!(text.contains("nodes: 2"));
        assert!(text.contains("Box=1"));
        assert!(text.contains("Text=1"));
    }
}
