//! Crash reporting (design §24.7): assemble a structured crash dump from the
//! shell's live state and, opt-in, route process panics through it.
//!
//! When something goes wrong in a shipped build, "it panicked" is not enough —
//! an AAA post-mortem needs *context*: what role the process was playing, what
//! quality tier it ran at, how large the world was, and which settings were in
//! force. This module captures that context as a cheap [`CrashSnapshot`] and
//! pairs it with the panic payload / location / [`Backtrace`] into a rendered
//! [`CrashReport`] (design §24.7 "崩溃转储:panic hook 收集堆栈、最近日志、World
//! 摘要、cvar 快照").
//!
//! # The snapshot / hook split
//!
//! A `std::panic` hook is a global `Fn` that **cannot borrow the live
//! [`World`](prism_ecs::world::World)** — it runs on the unwinding thread with
//! no handle to the app. Real engines solve this the way [`CrashReporter`]
//! does: the app publishes a lightweight, owned snapshot into a shared slot
//! ([`publish`](CrashReporter::publish)) whenever its state changes, and the
//! installed hook reads that last-published snapshot. The hook never touches
//! the `World`, so it is sound to run from an arbitrary panic.
//!
//! # Opt-in, process-global
//!
//! [`CrashReporter::install`] calls [`std::panic::set_hook`], which is
//! process-global state. It is therefore **explicitly opt-in** (never called by
//! [`App::new`](crate::app::App::new)) and chains the previously-installed hook
//! so default panic output is preserved. The dump is written through a
//! [`CrashSink`], defaulting to standard error; a custom sink (file, log
//! pipeline, telemetry) is a plain closure.
//!
//! # Honestly deferred
//!
//! - **Per-system panic isolation** (design §24.7 "单 system panic 可捕获…隔离
//!   该 system") must happen inside the `prism_ecs` executor that owns the
//!   per-system call site; `prism_app` cannot catch a panic *around one system*
//!   from the outside without the executor's cooperation, so it is not faked
//!   here.
//! - **Watchdog / frame-timeout detection** (design §24.7 "主循环卡死检测")
//!   needs a background thread observing a main-loop heartbeat; it is a later
//!   tooling-milestone (M6) increment and is documented as absent, not stubbed.
//! - **"Recent logs" capture** depends on a `prism_log` ring buffer that does
//!   not exist in this crate; the report carries the state `prism_app` actually
//!   owns (run mode, tier, world size, settings) rather than inventing a log
//!   feed.

use std::backtrace::{Backtrace, BacktraceStatus};
use std::fmt::Write as _;
use std::panic::{self, PanicHookInfo};
use std::sync::{Arc, Mutex};

use prism_ecs::resource::Resource;

use crate::app::App;
use crate::capability::QualityTier;
use crate::run_mode::RunMode;
use crate::settings::{SettingValue, Settings};

/// A cheap, owned snapshot of the shell's state at a point in time, suitable
/// for a panic hook to read without borrowing the [`World`](prism_ecs::world::World).
///
/// Built by [`App::capture_crash_snapshot`] and published into a
/// [`CrashReporter`] via [`publish`](CrashReporter::publish). Every field is
/// state `prism_app` genuinely owns; nothing is fabricated.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CrashSnapshot {
    /// The [`RunMode`] the app was launched as, if a [`RunMode`] resource was
    /// present (it is seeded by [`App::new`](crate::app::App::new)).
    pub run_mode: Option<RunMode>,
    /// The [`QualityTier`] the app resolved from its probed capabilities.
    pub quality_tier: Option<QualityTier>,
    /// The number of live entities in the main world at capture time.
    pub main_entity_count: u32,
    /// The resolved settings (`key`, value) pairs, sorted by key. Empty when no
    /// [`Settings`] resource is installed.
    pub settings: Vec<(String, SettingValue)>,
}

impl CrashSnapshot {
    /// Append a human-readable rendering of this snapshot to `out`.
    ///
    /// Used by [`CrashReport::render`]; exposed so a caller can embed the
    /// context block in its own report layout.
    pub fn render_into(&self, out: &mut String) {
        match self.run_mode {
            Some(mode) => {
                let _ = writeln!(out, "run mode:      {mode:?}");
            }
            None => out.push_str("run mode:      <unset>\n"),
        }
        match self.quality_tier {
            Some(tier) => {
                let _ = writeln!(out, "quality tier:  {tier:?}");
            }
            None => out.push_str("quality tier:  <unset>\n"),
        }
        let _ = writeln!(out, "main entities: {}", self.main_entity_count);
        if self.settings.is_empty() {
            out.push_str("settings:      <none installed>\n");
        } else {
            let _ = writeln!(out, "settings:      {} key(s)", self.settings.len());
            for (key, value) in &self.settings {
                let _ = writeln!(out, "  - {key} = {value:?}");
            }
        }
    }

    /// Render this snapshot on its own (without a panic payload) — handy for a
    /// voluntary state dump that is not tied to a crash.
    #[must_use]
    pub fn render(&self) -> String {
        let mut out = String::new();
        self.render_into(&mut out);
        out
    }
}

/// A fully-assembled crash dump: the panic context plus the shell
/// [`CrashSnapshot`].
#[derive(Clone, Debug)]
pub struct CrashReport {
    /// The panic message / payload, best-effort decoded to text.
    pub message: String,
    /// The source location of the panic, when the runtime provided one.
    pub location: Option<String>,
    /// A captured backtrace, present only when backtrace capture was enabled
    /// for the process (e.g. `RUST_BACKTRACE=1`) and succeeded.
    pub backtrace: Option<String>,
    /// The last-published shell state at the time of the crash.
    pub snapshot: CrashSnapshot,
}

impl CrashReport {
    /// Render the whole dump to a single multi-line string.
    #[must_use]
    pub fn render(&self) -> String {
        let mut out = String::new();
        out.push_str("=== prism_app crash report ===\n");
        let _ = writeln!(out, "panic:         {}", self.message);
        match &self.location {
            Some(loc) => {
                let _ = writeln!(out, "location:      {loc}");
            }
            None => out.push_str("location:      <unknown>\n"),
        }
        self.snapshot.render_into(&mut out);
        match &self.backtrace {
            Some(bt) => {
                out.push_str("backtrace:\n");
                out.push_str(bt);
                if !bt.ends_with('\n') {
                    out.push('\n');
                }
            }
            None => out.push_str("backtrace:     <unavailable; set RUST_BACKTRACE=1>\n"),
        }
        out.push_str("=== end crash report ===\n");
        out
    }
}

/// Where a rendered [`CrashReport`] is written. A plain `Fn(&str)`, so a file
/// writer, log pipeline, or in-memory buffer all fit without a bespoke trait.
pub type CrashSink = Arc<dyn Fn(&str) + Send + Sync>;

/// Builds the default [`CrashSink`] that writes to standard error.
#[must_use]
fn stderr_sink() -> CrashSink {
    Arc::new(|dump: &str| {
        eprint!("{dump}");
    })
}

/// Holds the last-published [`CrashSnapshot`] and the [`CrashSink`], and can
/// install a process panic hook that renders a [`CrashReport`] on any panic.
///
/// Cloneable and `Send + Sync`: the clone kept inside the installed hook and
/// the clone held by the app (e.g. as a resource) share the same snapshot slot,
/// so a [`publish`](CrashReporter::publish) from the app is visible to the hook.
#[derive(Clone)]
pub struct CrashReporter {
    snapshot: Arc<Mutex<CrashSnapshot>>,
    sink: CrashSink,
}

impl Resource for CrashReporter {}

impl Default for CrashReporter {
    fn default() -> Self {
        Self::new()
    }
}

impl CrashReporter {
    /// A reporter with an empty snapshot that writes dumps to standard error.
    #[must_use]
    pub fn new() -> Self {
        Self {
            snapshot: Arc::new(Mutex::new(CrashSnapshot::default())),
            sink: stderr_sink(),
        }
    }

    /// A reporter that writes dumps through `sink` instead of standard error.
    #[must_use]
    pub fn with_sink(sink: CrashSink) -> Self {
        Self {
            snapshot: Arc::new(Mutex::new(CrashSnapshot::default())),
            sink,
        }
    }

    /// Replace the last-published snapshot. The next crash dump reads this.
    ///
    /// Cheap enough to call every frame; the app owns when to refresh (there is
    /// no hidden per-frame system). A poisoned lock is recovered from, since a
    /// crash reporter must not itself panic.
    pub fn publish(&self, snapshot: CrashSnapshot) {
        let mut slot = self.snapshot.lock().unwrap_or_else(|e| e.into_inner());
        *slot = snapshot;
    }

    /// A clone of the last-published snapshot.
    #[must_use]
    pub fn snapshot(&self) -> CrashSnapshot {
        self.snapshot
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// Assemble a [`CrashReport`] from the given panic details plus the
    /// last-published snapshot, write its rendering to the sink, and return the
    /// rendered dump.
    ///
    /// This is the exact body the installed panic hook runs; it is public so a
    /// caller that catches a panic itself (e.g. an authoritative server wrapping
    /// a tick in [`catch_unwind`](std::panic::catch_unwind)) can produce the
    /// same dump without installing a global hook.
    pub fn report(
        &self,
        message: impl Into<String>,
        location: Option<String>,
        backtrace: Option<String>,
    ) -> String {
        let report = CrashReport {
            message: message.into(),
            location,
            backtrace,
            snapshot: self.snapshot(),
        };
        let dump = report.render();
        (self.sink)(&dump);
        dump
    }

    /// Install a process panic hook that renders a [`CrashReport`] for every
    /// panic, then delegates to the previously-installed hook.
    ///
    /// This mutates **process-global** state via [`std::panic::set_hook`]; call
    /// it once, early, and only when you want crash dumps. The prior hook is
    /// preserved and still runs (so the standard panic message is not lost).
    pub fn install(&self) {
        let reporter = self.clone();
        let previous = panic::take_hook();
        panic::set_hook(Box::new(move |info: &PanicHookInfo<'_>| {
            let message = panic_message(info);
            let location = info.location().map(|loc| loc.to_string());
            let backtrace = capture_backtrace();
            reporter.report(message, location, backtrace);
            previous(info);
        }));
    }
}

/// Best-effort decode of a panic payload to text (`&str` / `String` payloads;
/// otherwise a placeholder).
fn panic_message(info: &PanicHookInfo<'_>) -> String {
    let payload = info.payload();
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "<non-string panic payload>".to_string()
    }
}

/// Capture a backtrace, returning `Some` only when capture is actually enabled
/// and succeeded (so a disabled backtrace is reported as unavailable rather
/// than as an empty string).
fn capture_backtrace() -> Option<String> {
    let bt = Backtrace::capture();
    match bt.status() {
        BacktraceStatus::Captured => Some(bt.to_string()),
        _ => None,
    }
}

impl App {
    /// Capture the current shell state as a [`CrashSnapshot`] (design §24.7).
    ///
    /// Reads the main world's [`RunMode`] and [`QualityTier`], its live entity
    /// count, and the resolved [`Settings`] (if installed). Pure and cheap; it
    /// does not install anything.
    #[must_use]
    pub fn capture_crash_snapshot(&self) -> CrashSnapshot {
        let world = self.world();
        let run_mode = world.get_resource::<RunMode>().copied();
        let quality_tier = Some(self.quality_tier());
        let main_entity_count = world.entity_count();
        let settings = world
            .get_resource::<Settings>()
            .map(|settings| {
                let mut pairs: Vec<(String, SettingValue)> = settings
                    .iter()
                    .map(|(key, value)| (key.to_string(), value.clone()))
                    .collect();
                pairs.sort_by(|a, b| a.0.cmp(&b.0));
                pairs
            })
            .unwrap_or_default();
        CrashSnapshot {
            run_mode,
            quality_tier,
            main_entity_count,
            settings,
        }
    }

    /// Install a [`CrashReporter`], seed it with the current snapshot, register
    /// its process panic hook, and store it as a main-world resource.
    ///
    /// Returns a clone of the reporter so the caller can keep publishing fresh
    /// snapshots. This installs a **process-global** panic hook (see
    /// [`CrashReporter::install`]); it is explicit and never called implicitly.
    pub fn install_crash_reporter(&mut self) -> CrashReporter {
        let reporter = CrashReporter::new();
        reporter.publish(self.capture_crash_snapshot());
        reporter.install();
        self.world_mut().insert_resource(reporter.clone());
        reporter
    }

    /// Re-capture the current snapshot and publish it to the installed
    /// [`CrashReporter`] resource, so a subsequent crash dump reflects the
    /// latest state. A no-op when no reporter was installed.
    pub fn refresh_crash_snapshot(&mut self) -> &mut Self {
        let snapshot = self.capture_crash_snapshot();
        if let Some(reporter) = self.world().get_resource::<CrashReporter>() {
            reporter.publish(snapshot);
        }
        self
    }
}
