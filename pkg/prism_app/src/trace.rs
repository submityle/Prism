//! Per-frame phase trace for flamegraphs (design §16 observability, M6 tooling).
//!
//! [`FrameDiagnostics`](crate::diagnostics::FrameDiagnostics) keeps *rolling
//! statistics* per phase — enough for a HUD, not enough for a flamegraph. A
//! flamegraph needs the *ordered, per-frame* phases with their offsets within
//! the frame, laid on a global timeline. [`PhaseTrace`] is that log: the
//! instrumented frame loop calls [`begin_frame`](PhaseTrace::begin_frame), then
//! [`record_phase`](PhaseTrace::record_phase) once per phase in order, then
//! [`end_frame`](PhaseTrace::end_frame). The retained events export to
//! [`to_chrome_trace_json`](PhaseTrace::to_chrome_trace_json), which a
//! Chrome/Perfetto viewer renders as a flamegraph.
//!
//! Memory is bounded: only the most recent [`window`](PhaseTrace::window) frames
//! are kept, so a long session costs O(window) regardless of run length.

use std::collections::VecDeque;
use std::fmt::Write as _;
use std::time::Duration;

use prism_ecs::resource::Resource;

/// One timed phase within a frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PhaseEvent {
    /// The frame this phase belongs to (`0`-based, in run order).
    pub frame: u64,
    /// The phase label (e.g. `"Update"`, `"RunFixedMainLoop"`).
    pub name: &'static str,
    /// Start offset from the start of the run, in microseconds.
    pub start_us: f64,
    /// Phase duration, in microseconds.
    pub dur_us: f64,
}

/// A bounded per-frame phase log for flamegraph export.
///
/// Installed as a main-world resource by
/// [`App::init_phase_trace`](crate::app::App::init_phase_trace); the
/// instrumented frame loop then fills it in automatically.
pub struct PhaseTrace {
    window: usize,
    next_frame: u64,
    frame: u64,
    frame_base_us: f64,
    offset_us: f64,
    events: VecDeque<PhaseEvent>,
}

impl Resource for PhaseTrace {}

impl PhaseTrace {
    /// A trace retaining at most `window` frames. A `window` of `0` is clamped
    /// to `1` so the current frame always fits.
    #[must_use]
    pub fn new(window: usize) -> Self {
        Self {
            window: window.max(1),
            next_frame: 0,
            frame: 0,
            frame_base_us: 0.0,
            offset_us: 0.0,
            events: VecDeque::new(),
        }
    }

    /// The number of frames retained at most.
    #[must_use]
    pub fn window(&self) -> usize {
        self.window
    }

    /// How many frames have been traced so far.
    #[must_use]
    pub fn frame_count(&self) -> u64 {
        self.next_frame
    }

    /// The number of retained phase events.
    #[must_use]
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Whether no phase has been recorded yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// The retained phase events, oldest first.
    pub fn events(&self) -> impl Iterator<Item = &PhaseEvent> {
        self.events.iter()
    }

    /// Begin a new frame: assign it the next frame number and reset the
    /// intra-frame offset to zero.
    pub fn begin_frame(&mut self) {
        self.frame = self.next_frame;
        self.next_frame += 1;
        self.offset_us = 0.0;
    }

    /// Record a phase in the current frame, at the running intra-frame offset.
    pub fn record_phase(&mut self, name: &'static str, elapsed: Duration) {
        let dur_us = elapsed.as_secs_f64() * 1_000_000.0;
        self.events.push_back(PhaseEvent {
            frame: self.frame,
            name,
            start_us: self.frame_base_us + self.offset_us,
            dur_us,
        });
        self.offset_us += dur_us;
    }

    /// Close the current frame; `total` advances the global timeline so the next
    /// frame starts after this one, and evicts frames past the window.
    pub fn end_frame(&mut self, total: Duration) {
        self.frame_base_us += total.as_secs_f64() * 1_000_000.0;
        self.trim();
    }

    /// Drop events older than the retained window.
    fn trim(&mut self) {
        let keep_from = self
            .frame
            .saturating_add(1)
            .saturating_sub(self.window as u64);
        while let Some(front) = self.events.front() {
            if front.frame < keep_from {
                self.events.pop_front();
            } else {
                break;
            }
        }
    }

    /// Serialize the retained events as a Chrome trace-event JSON document
    /// (`{"traceEvents":[...]}`) for a flamegraph viewer.
    ///
    /// Phase labels are `&'static str` labels chosen by the engine (no quotes or
    /// backslashes), so no JSON escaping is required.
    #[must_use]
    pub fn to_chrome_trace_json(&self) -> String {
        let mut out = String::from("{\"traceEvents\":[");
        for (i, event) in self.events.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            let _ = write!(
                out,
                "{{\"name\":\"{}\",\"ph\":\"X\",\"ts\":{},\"dur\":{},\"pid\":1,\"tid\":1}}",
                event.name, event.start_us, event.dur_us
            );
        }
        out.push_str("]}");
        out
    }
}

impl crate::app::App {
    /// Install a bounded per-frame phase trace keeping the last `window` frames.
    ///
    /// Once installed, the frame loop runs its instrumented path and fills the
    /// trace automatically (design §16, M6 flamegraph). Replaces any previous
    /// trace.
    pub fn init_phase_trace(&mut self, window: usize) -> &mut Self {
        self.insert_resource(PhaseTrace::new(window));
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_phases_with_running_offsets() {
        let mut trace = PhaseTrace::new(8);
        trace.begin_frame();
        trace.record_phase("First", Duration::from_micros(10));
        trace.record_phase("Update", Duration::from_micros(30));
        trace.end_frame(Duration::from_micros(50));

        let events: Vec<PhaseEvent> = trace.events().copied().collect();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].frame, 0);
        assert_eq!(events[0].name, "First");
        assert_eq!(events[0].start_us, 0.0);
        assert_eq!(events[0].dur_us, 10.0);
        // The second phase starts where the first ended.
        assert_eq!(events[1].start_us, 10.0);
        assert_eq!(events[1].dur_us, 30.0);
        assert_eq!(trace.frame_count(), 1);
    }

    #[test]
    fn next_frame_starts_after_the_previous_frame_total() {
        let mut trace = PhaseTrace::new(8);
        trace.begin_frame();
        trace.record_phase("Update", Duration::from_micros(40));
        trace.end_frame(Duration::from_micros(100)); // 60 us idle
        trace.begin_frame();
        trace.record_phase("Update", Duration::from_micros(5));

        let events: Vec<PhaseEvent> = trace.events().copied().collect();
        assert_eq!(events[1].frame, 1);
        assert_eq!(events[1].start_us, 100.0); // starts after frame 0's total
        assert_eq!(trace.frame_count(), 2);
    }

    #[test]
    fn window_evicts_old_frames_only() {
        let mut trace = PhaseTrace::new(2);
        for _ in 0..3 {
            trace.begin_frame();
            trace.record_phase("Update", Duration::from_micros(1));
            trace.end_frame(Duration::from_micros(1));
        }
        let frames: Vec<u64> = trace.events().map(|e| e.frame).collect();
        assert_eq!(frames, vec![1, 2]); // frame 0 evicted
        assert_eq!(trace.frame_count(), 3);
    }

    #[test]
    fn chrome_json_lists_the_events() {
        let mut trace = PhaseTrace::new(4);
        trace.begin_frame();
        trace.record_phase("Update", Duration::from_micros(7));
        trace.end_frame(Duration::from_micros(9));
        let json = trace.to_chrome_trace_json();
        assert!(json.starts_with("{\"traceEvents\":["));
        assert!(json.ends_with("]}"));
        assert!(json.contains("\"name\":\"Update\""));
        assert!(json.contains("\"ph\":\"X\""));
        assert!(json.contains("\"dur\":7"));
    }

    #[test]
    fn installing_the_trace_instruments_the_frame_loop() {
        let mut app = crate::app::App::new();
        app.init_phase_trace(4);
        app.update();

        let trace = app
            .world()
            .get_resource::<PhaseTrace>()
            .expect("trace installed");
        assert_eq!(trace.frame_count(), 1);
        let names: Vec<&'static str> = trace.events().map(|e| e.name).collect();
        // The instrumented frame loop records the core phases in order.
        assert!(names.contains(&"First"));
        assert!(names.contains(&"RunFixedMainLoop"));
        assert!(names.contains(&"Update"));
        assert!(names.contains(&"Last"));
    }
}
