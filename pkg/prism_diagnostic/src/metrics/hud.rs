//! HUD data provider.
//!
//! This module turns metric and frame-timing snapshots into plain text lines
//! suitable for an on-screen overlay. It performs **no rendering**: this crate
//! has no GPU dependency, so drawing the returned [`HudSnapshot`] lines (choice
//! of font, placement, color) is entirely the consumer's responsibility.
//!
//! Use [`hud_lines`] for a one-shot `Vec<String>`, or [`Hud`] to carry a title
//! and build a structured [`HudSnapshot`].

extern crate alloc;

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use crate::metrics::counter::RegistrySnapshot;
use crate::metrics::frame::FrameStatsSnapshot;

/// A structured HUD snapshot: an ordered list of text lines ready to draw.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct HudSnapshot {
    /// The formatted overlay lines, top to bottom.
    pub lines: Vec<String>,
}

impl HudSnapshot {
    /// Borrow the formatted lines.
    pub fn lines(&self) -> &[String] {
        &self.lines
    }

    /// Consume the snapshot, yielding its lines.
    pub fn into_lines(self) -> Vec<String> {
        self.lines
    }

    /// Join the lines with `\n` into a single block of text.
    pub fn as_text(&self) -> String {
        self.lines.join("\n")
    }
}

/// A HUD data provider carrying a title.
///
/// [`Hud::snapshot`] combines a [`FrameStatsSnapshot`] and a
/// [`RegistrySnapshot`] into an ordered [`HudSnapshot`]. The provider is
/// stateless beyond its title, so it is cheap to construct per frame.
#[derive(Clone, Debug)]
pub struct Hud {
    title: String,
}

impl Default for Hud {
    fn default() -> Self {
        Self {
            title: "Prism HUD".to_string(),
        }
    }
}

impl Hud {
    /// Create a HUD with the default title.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a HUD with a custom title line.
    pub fn with_title(title: impl Into<String>) -> Self {
        Self {
            title: title.into(),
        }
    }

    /// Build a [`HudSnapshot`] from frame stats and registered metrics.
    pub fn snapshot(
        &self,
        frame: &FrameStatsSnapshot,
        metrics: &RegistrySnapshot,
    ) -> HudSnapshot {
        let mut lines = Vec::new();
        lines.push(self.title.clone());
        lines.extend(frame_and_metric_lines(frame, metrics));
        HudSnapshot { lines }
    }
}

/// Format frame stats plus registered metrics into HUD lines (without a title).
///
/// Rendering the returned lines is the consumer's job; this function only
/// produces text.
pub fn hud_lines(frame: &FrameStatsSnapshot, metrics: &RegistrySnapshot) -> Vec<String> {
    frame_and_metric_lines(frame, metrics)
}

fn frame_and_metric_lines(
    frame: &FrameStatsSnapshot,
    metrics: &RegistrySnapshot,
) -> Vec<String> {
    let mut lines = Vec::new();
    lines.push(format!(
        "FPS {:.1} (min {:.1} / max {:.1})",
        frame.fps, frame.min_fps, frame.max_fps
    ));
    lines.push(format!(
        "frame {:.2} ms (min {:.2} / avg {:.2} / max {:.2})",
        frame.last_delta_ms, frame.min_ms, frame.avg_ms, frame.max_ms
    ));
    for (name, value) in &metrics.gauges {
        lines.push(format!("{name}: {value:.3}"));
    }
    for (name, value) in &metrics.counters {
        lines.push(format!("{name}: {value}"));
    }
    for (name, hist) in &metrics.histograms {
        lines.push(format!(
            "{name}: n={} mean={:.3} p50={:.3} p99={:.3} min={:.3} max={:.3}",
            hist.count,
            hist.mean(),
            hist.percentile(0.5),
            hist.percentile(0.99),
            hist.min,
            hist.max
        ));
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::counter::MetricRegistry;
    use crate::metrics::frame::FrameTimer;

    #[test]
    fn hud_snapshot_contains_expected_lines() {
        let reg = MetricRegistry::new();
        reg.gauge("mem_mb").set(512.0);
        reg.counter("entities").add(128);
        let hist = reg.histogram("latency_ms", &[1.0, 5.0, 10.0]);
        hist.record(2.0);
        hist.record(8.0);

        let mut timer = FrameTimer::new();
        for d in [16.0, 16.0, 16.0] {
            timer.record_frame_delta_ms(d);
        }

        let hud = Hud::with_title("Diag");
        let snap = hud.snapshot(&timer.stats(), &reg.snapshot());
        let text = snap.as_text();

        assert_eq!(snap.lines()[0], "Diag");
        assert!(text.contains("FPS 62.5"), "missing fps line: {text}");
        assert!(text.contains("mem_mb: 512.000"), "missing gauge: {text}");
        assert!(text.contains("entities: 128"), "missing counter: {text}");
        assert!(text.contains("latency_ms: n=2"), "missing histogram: {text}");
    }

    #[test]
    fn hud_lines_free_function_matches_snapshot_body() {
        let reg = MetricRegistry::new();
        reg.counter("frames").add(1);
        let timer = FrameTimer::new();
        let frame = timer.stats();
        let metrics = reg.snapshot();

        let free = hud_lines(&frame, &metrics);
        let hud = Hud::new();
        let via_hud = hud.snapshot(&frame, &metrics).into_lines();
        // The free function omits only the leading title line.
        assert_eq!(&via_hud[1..], free.as_slice());
    }
}
