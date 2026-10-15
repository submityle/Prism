use std::collections::VecDeque;

#[cfg(any(test, feature = "diagnostic"))]
use super::CrashReporter;

pub(super) struct RecentLogs {
    entries: VecDeque<String>,
    capacity: usize,
    max_line_bytes: usize,
}

impl RecentLogs {
    pub(super) fn new(capacity: usize, max_line_bytes: usize) -> Self {
        Self {
            entries: VecDeque::new(),
            capacity,
            max_line_bytes,
        }
    }

    pub(super) fn push(&mut self, line: &str) {
        if self.capacity == 0 || self.max_line_bytes == 0 {
            return;
        }
        let mut end = line.len().min(self.max_line_bytes);
        while !line.is_char_boundary(end) {
            end -= 1;
        }
        let line = line[..end].to_owned();
        if self.entries.len() == self.capacity {
            self.entries.pop_front();
        }
        self.entries.push_back(line);
    }

    pub(super) fn snapshot(&self) -> Vec<String> {
        self.entries.iter().cloned().collect()
    }
}

#[cfg(feature = "diagnostic")]
mod diagnostic {
    use std::fmt::{self, Write};
    use std::sync::Arc;

    use super::CrashReporter;
    use prism_diagnostic::{Event, FieldValue, Sink};

    struct BoundedLine {
        text: String,
        remaining: usize,
    }
    impl Write for BoundedLine {
        fn write_str(&mut self, text: &str) -> fmt::Result {
            let mut end = text.len().min(self.remaining);
            while !text.is_char_boundary(end) {
                end -= 1;
            }
            self.text.push_str(&text[..end]);
            self.remaining -= end;
            // Once a scalar does not fit, no later text may overtake it.
            if end < text.len() {
                self.remaining = 0;
            }
            Ok(())
        }
    }

    impl Sink for CrashReporter {
        fn emit(&self, event: &Event) {
            let (capacity, max_line_bytes) = {
                let logs = self.logs.lock().unwrap_or_else(|e| e.into_inner());
                (logs.capacity, logs.max_line_bytes)
            };
            if capacity == 0 || max_line_bytes == 0 {
                return;
            }
            let mut line = BoundedLine {
                text: String::new(),
                remaining: max_line_bytes,
            };
            let _ = write!(
                line,
                "[{}] {}: {}",
                event.level.label(),
                event.target,
                event.message
            );
            for index in 0..event.fields.len() {
                if line.remaining == 0 {
                    break;
                }
                let Some(field) = event.fields.get(index) else {
                    continue;
                };
                let _ = write!(line, " {}=", field.key);
                match &field.value {
                    FieldValue::Str(value) => {
                        let _ = line.write_str(value);
                    }
                    FieldValue::I64(value) => {
                        let _ = write!(line, "{value}");
                    }
                    FieldValue::U64(value) => {
                        let _ = write!(line, "{value}");
                    }
                    FieldValue::F64(value) => {
                        let _ = write!(line, "{value}");
                    }
                    FieldValue::Bool(value) => {
                        let _ = write!(line, "{value}");
                    }
                }
            }
            self.record_log(&line.text);
        }
    }

    struct ForwardingSink {
        reporter: CrashReporter,
        downstream: Arc<dyn Sink>,
    }
    impl Sink for ForwardingSink {
        fn emit(&self, event: &Event) {
            self.reporter.emit(event);
            self.downstream.emit(event);
        }
    }

    impl CrashReporter {
        /// A diagnostic sink that records first, then forwards the original
        /// event unchanged. Install explicitly with `prism_diagnostic::set_sink`
        /// or compose into the caller's logger. Construction has no global effects.
        #[must_use]
        pub fn diagnostic_sink(&self, downstream: Arc<dyn Sink>) -> Arc<dyn Sink> {
            Arc::new(ForwardingSink {
                reporter: self.clone(),
                downstream,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crash::{SystemFailures, SystemPanicPolicy};
    use crate::{App, Update};
    use prism_ecs::{resource::Resource, system::ResMut};
    use std::sync::{Arc, Mutex};

    #[test]
    fn logs_are_bounded_shared_and_embedded_in_the_actual_dump() {
        let dumps = Arc::new(Mutex::new(Vec::new()));
        let output = dumps.clone();
        let reporter = CrashReporter::with_log_capacity(
            Arc::new(move |dump| output.lock().unwrap().push(dump.to_string())),
            2,
            5,
        );
        reporter.record_log("old");
        reporter.clone().record_log("界界界");
        reporter.record_log("last");
        assert_eq!(reporter.recent_logs(), ["界", "last"]);
        let dump = reporter.report("caught", None, None);
        assert!(dump.contains("recent logs:\n界\nlast\n"));
        assert_eq!(dumps.lock().unwrap().as_slice(), [dump]);
        let disabled = CrashReporter::with_log_capacity(Arc::new(|_| {}), 0, 0);
        disabled.record_log("discarded");
        assert!(disabled.recent_logs().is_empty());
    }

    #[test]
    fn cloned_reporters_capture_parallel_logs_without_loss() {
        let reporter = CrashReporter::with_log_capacity(Arc::new(|_| {}), 64, 64);
        std::thread::scope(|scope| {
            for index in 0..8 {
                let reporter = reporter.clone();
                scope.spawn(move || {
                    for line in 0..8 {
                        reporter.record_log(&format!("{index}:{line}"));
                    }
                });
            }
        });
        let mut logs = reporter.recent_logs();
        logs.sort();
        logs.dedup();
        assert_eq!(logs.len(), 64);
    }

    #[test]
    fn app_policy_retires_only_the_failed_system_across_frames() {
        struct Count(usize);
        impl Resource for Count {}
        let mut app = App::new();
        app.world_mut().insert_resource(Count(0));
        app.set_system_panic_policy(SystemPanicPolicy::DisableAndContinue);
        app.add_systems(Update, || panic!("isolated app system"));
        app.add_systems(Update, |mut count: ResMut<Count>| count.0 += 1);
        app.update();
        app.update();
        assert_eq!(app.world().resource::<Count>().0, 2);
        assert_eq!(app.world().resource::<SystemFailures>().total(), 1);
    }

    #[cfg(feature = "diagnostic")]
    #[test]
    fn diagnostic_adapter_captures_structured_events_and_forwards_unchanged() {
        use prism_diagnostic::{CaptureSink, Event, FieldValue, Level};
        let reporter = CrashReporter::with_log_capacity(Arc::new(|_| {}), 2, 40);
        let downstream = Arc::new(CaptureSink::new());
        let sink = reporter.diagnostic_sink(downstream.clone());
        sink.emit(&Event::new(Level::Warn, "test", "fault").with_field("id", FieldValue::U64(42)));
        assert_eq!(reporter.recent_logs(), ["[WARN] test: fault id=42"]);
        sink.emit(&Event::new(Level::Info, "test", "界".repeat(100)));
        assert!(reporter.recent_logs().last().unwrap().len() <= 40);
        assert_eq!(downstream.len(), 2);
        assert_eq!(downstream.events()[1].message.len(), 300);
    }
}
