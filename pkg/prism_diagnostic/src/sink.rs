//! Event sinks (backends) and the global logger.

extern crate alloc;

use alloc::sync::Arc;
use alloc::vec::Vec;
use std::sync::{Mutex, OnceLock, RwLock};

use crate::fmt::format_line;
use crate::model::Event;

/// A destination for diagnostic events.
pub trait Sink: Send + Sync {
    /// Consume one event.
    fn emit(&self, event: &Event);
}

/// Writes formatted lines to standard error.
#[derive(Default)]
pub struct ConsoleSink;

impl Sink for ConsoleSink {
    #[allow(
        clippy::print_stderr,
        reason = "the console sink is the one place diagnostics are intentionally written to stderr"
    )]
    fn emit(&self, event: &Event) {
        eprintln!("{}", format_line(event));
    }
}

/// Appends formatted lines to a file.
pub struct FileSink {
    file: Mutex<std::fs::File>,
}

impl FileSink {
    /// Open (create/append) a log file.
    pub fn create(path: impl AsRef<std::path::Path>) -> std::io::Result<Self> {
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        Ok(Self {
            file: Mutex::new(file),
        })
    }
}

impl Sink for FileSink {
    fn emit(&self, event: &Event) {
        use std::io::Write as _;
        if let Ok(mut f) = self.file.lock() {
            let _ = writeln!(f, "{}", format_line(event));
        }
    }
}

/// Collects events in memory; useful for tests and in-process inspection.
#[derive(Default, Clone)]
pub struct CaptureSink {
    events: Arc<Mutex<Vec<Event>>>,
}

impl CaptureSink {
    /// Create an empty capture sink.
    pub fn new() -> Self {
        Self::default()
    }

    /// Snapshot the captured events.
    pub fn events(&self) -> Vec<Event> {
        self.events.lock().unwrap().clone()
    }

    /// Number of captured events.
    pub fn len(&self) -> usize {
        self.events.lock().unwrap().len()
    }

    /// Whether nothing has been captured.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Sink for CaptureSink {
    fn emit(&self, event: &Event) {
        self.events.lock().unwrap().push(event.clone());
    }
}

type SinkSlot = RwLock<Option<Arc<dyn Sink>>>;

fn global() -> &'static SinkSlot {
    static GLOBAL: OnceLock<SinkSlot> = OnceLock::new();
    GLOBAL.get_or_init(|| RwLock::new(None))
}

/// Install `sink` as the global diagnostic sink, replacing any previous one.
pub fn set_sink(sink: Arc<dyn Sink>) {
    *global().write().unwrap() = Some(sink);
}

/// Remove the global sink (events are then dropped).
pub fn clear_sink() {
    *global().write().unwrap() = None;
}

/// Dispatch an event to the installed sink, if any.
pub fn dispatch(event: &Event) {
    if let Some(sink) = global().read().unwrap().as_ref() {
        sink.emit(event);
    }
}
