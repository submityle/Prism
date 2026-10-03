//! Main-loop hang detection (design §24.7: *"看门狗:主循环卡死检测(帧超时),
//! 导出诊断"*).
//!
//! An AAA / certification target must not silently freeze: when the main loop
//! wedges — an infinite loop in a system, a deadlocked job, a blocking I/O call
//! that never returns — the process should *notice* and emit diagnostics rather
//! than appear alive-but-stuck to the player and the OS. This module provides a
//! real, self-contained watchdog built on nothing but the standard library.
//!
//! # How it works
//!
//! The watchdog runs a dedicated background thread that owns its own monotonic
//! clock. The main loop calls [`FrameWatchdog::beat`] once per completed frame,
//! which bumps a shared atomic counter. The background thread wakes every
//! [`poll_interval`](WatchdogConfig::poll_interval) and compares the counter to
//! what it last saw:
//!
//! - counter advanced → the loop is making progress; reset the stall timer.
//! - counter unchanged for at least [`timeout`](WatchdogConfig::timeout) → the
//!   loop is stuck; invoke the stall handler **once** for this stall (it will
//!   not fire again until the loop beats and then stalls afresh).
//!
//! Because the background thread measures elapsed time with its *own* clock and
//! the main loop only ever stores into an atomic, there is no cross-thread
//! timestamp to keep coherent and the main-loop cost of a beat is a single
//! relaxed-ordering atomic add.
//!
//! # The timeout is a frame budget, not a frame period
//!
//! The watchdog measures wall-clock gaps *between frame completions*, so the gap
//! legitimately includes any intentional pacing sleep (a
//! [`FrameLimit`](crate::pacing::FrameLimit) cap, a dedicated-server tick
//! interval). Set [`timeout`](WatchdogConfig::timeout) comfortably above your
//! longest *expected* frame (target period + headroom), not at the frame period
//! itself, or an intentionally slow tick will be flagged as a hang. The default
//! of five seconds suits an interactive loop; a low-tickrate server should pick
//! a timeout above its tick interval.
//!
//! # What the handler can do
//!
//! The stall handler runs on the watchdog thread and therefore **cannot borrow
//! the [`World`](prism_ecs::world::World)** — the main thread is, by definition,
//! wedged. It is handed a [`FrameStall`] describing which frame was last seen
//! and how long the loop has been quiet. The honest, panic-safe pairing is to
//! render a previously-published [`CrashSnapshot`](crate::crash::CrashSnapshot)
//! (the same owned-snapshot discipline the [`crash`](crate::crash) module uses):
//! publish state as it changes, and have the watchdog handler read and dump the
//! last snapshot. The default handler writes a one-line notice to standard
//! error.
//!
//! # Scope and honesty
//!
//! This module detects a stall and reports it. It deliberately does **not** try
//! to *kill* or *recover* the stuck thread: there is no portable, sound way to
//! unwind an arbitrary wedged thread from the outside, and faking recovery would
//! be worse than an honest diagnostic. Automatic recovery belongs to a
//! supervising process (a crash handler restarting the app), which is out of
//! this crate's scope.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread::{self, JoinHandle};

use prism_time::{Duration, Instant};

/// Details of a detected main-loop stall, handed to the stall handler.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameStall {
    /// The value of the beat counter when the stall was detected — i.e. the
    /// number of frames that had completed before the loop went quiet.
    pub frame: u64,
    /// How long the loop had been quiet (no [`beat`](FrameWatchdog::beat)) when
    /// the stall fired. At least [`timeout`](WatchdogConfig::timeout).
    pub stalled_for: Duration,
}

/// A stall handler: a `Send + Sync` closure invoked on the watchdog thread when
/// a stall is detected. Shared behind an [`Arc`] so a [`WatchdogConfig`] stays
/// cheaply cloneable.
pub type StallHandler = Arc<dyn Fn(FrameStall) + Send + Sync>;

/// Configuration for a [`FrameWatchdog`], cloneable so it can be stored in a
/// [`HeadlessRunner`](crate::runner::HeadlessRunner) (which must stay `Clone`)
/// and started lazily inside the run loop.
#[derive(Clone)]
pub struct WatchdogConfig {
    timeout: Duration,
    poll_interval: Duration,
    handler: StallHandler,
}

impl WatchdogConfig {
    /// Default stall timeout: five seconds, suitable for an interactive loop.
    pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(5);

    /// Default poll interval: how often the watchdog thread checks the beat
    /// counter. A quarter second keeps the thread near-idle while still
    /// reporting a hang promptly relative to the multi-second default timeout.
    pub const DEFAULT_POLL_INTERVAL: Duration = Duration::from_millis(250);

    /// A config that fires after `timeout` of main-loop silence, polling at the
    /// [default interval](WatchdogConfig::DEFAULT_POLL_INTERVAL) and reporting to
    /// standard error.
    #[must_use]
    pub fn new(timeout: Duration) -> Self {
        Self {
            timeout,
            poll_interval: Self::DEFAULT_POLL_INTERVAL,
            handler: default_stall_handler(),
        }
    }

    /// Set how often the watchdog thread wakes to check for progress.
    ///
    /// The effective interval is clamped at [`start`](WatchdogConfig::start) to
    /// `[1ms, timeout]`: never a busy-spin, never coarser than the timeout it is
    /// trying to observe. Builder-style.
    #[must_use]
    pub fn with_poll_interval(mut self, poll_interval: Duration) -> Self {
        self.poll_interval = poll_interval;
        self
    }

    /// Replace the stall handler. The handler runs on the watchdog thread and
    /// must be `Send + Sync`; it must not touch the (wedged) main-thread
    /// [`World`](prism_ecs::world::World). Builder-style.
    #[must_use]
    pub fn with_handler<H>(mut self, handler: H) -> Self
    where
        H: Fn(FrameStall) + Send + Sync + 'static,
    {
        self.handler = Arc::new(handler);
        self
    }

    /// The configured stall timeout.
    #[must_use]
    pub fn timeout(&self) -> Duration {
        self.timeout
    }

    /// The configured poll interval (before the `[1ms, timeout]` clamp applied
    /// at [`start`](WatchdogConfig::start)).
    #[must_use]
    pub fn poll_interval(&self) -> Duration {
        self.poll_interval
    }

    /// Spawn the watchdog thread and return a live [`FrameWatchdog`].
    ///
    /// The returned handle must be beaten once per frame via
    /// [`beat`](FrameWatchdog::beat); dropping it (or calling
    /// [`stop`](FrameWatchdog::stop)) signals the thread to exit and joins it.
    ///
    /// # Panics
    ///
    /// Panics if the OS refuses to spawn the watchdog thread (e.g. thread
    /// exhaustion). A watchdog that cannot start is a hard configuration
    /// failure, not a condition to paper over.
    #[must_use]
    pub fn start(self) -> FrameWatchdog {
        let shared = Arc::new(Shared {
            beats: AtomicU64::new(0),
            running: AtomicBool::new(true),
            stalls: AtomicU64::new(0),
        });
        let timeout = self.timeout;
        // Never busy-spin, never poll coarser than the timeout we observe.
        let floor = Duration::from_millis(1);
        let poll_interval = self.poll_interval.clamp(floor, timeout.max(floor));
        let handler = self.handler;
        let thread_shared = Arc::clone(&shared);
        let handle = thread::Builder::new()
            .name("prism-frame-watchdog".to_owned())
            .spawn(move || watch(&thread_shared, timeout, poll_interval, &handler))
            .expect("spawn prism frame-watchdog thread");
        FrameWatchdog {
            shared,
            handle: Some(handle),
        }
    }
}

impl Default for WatchdogConfig {
    fn default() -> Self {
        Self::new(Self::DEFAULT_TIMEOUT)
    }
}

impl core::fmt::Debug for WatchdogConfig {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("WatchdogConfig")
            .field("timeout", &self.timeout)
            .field("poll_interval", &self.poll_interval)
            .field("handler", &"<stall handler>")
            .finish()
    }
}

/// State shared between the main loop and the watchdog thread.
#[derive(Debug)]
struct Shared {
    /// Monotonically increasing frame-beat counter, bumped by
    /// [`FrameWatchdog::beat`].
    beats: AtomicU64,
    /// Cleared to signal the watchdog thread to exit.
    running: AtomicBool,
    /// Count of stalls the watchdog has reported so far.
    stalls: AtomicU64,
}

/// A live main-loop watchdog: beat it once per frame; drop it to stop.
///
/// See the [module docs](crate::watchdog) for the model. Created by
/// [`WatchdogConfig::start`].
#[derive(Debug)]
pub struct FrameWatchdog {
    shared: Arc<Shared>,
    handle: Option<JoinHandle<()>>,
}

impl FrameWatchdog {
    /// Record one completed frame. Call this once per main-loop iteration; the
    /// watchdog treats a timeout's worth of missing beats as a stall.
    ///
    /// The cost is a single atomic add, cheap enough to sit on the hot path.
    #[inline]
    pub fn beat(&self) {
        self.shared.beats.fetch_add(1, Ordering::Relaxed);
    }

    /// The number of beats recorded so far.
    #[must_use]
    pub fn beats(&self) -> u64 {
        self.shared.beats.load(Ordering::Relaxed)
    }

    /// The number of stalls the watchdog has reported so far.
    #[must_use]
    pub fn observed_stalls(&self) -> u64 {
        self.shared.stalls.load(Ordering::Relaxed)
    }

    /// Whether the watchdog thread is still active (has not been stopped).
    #[must_use]
    pub fn is_running(&self) -> bool {
        self.shared.running.load(Ordering::Relaxed)
    }

    /// Stop the watchdog thread and join it. Equivalent to dropping the handle,
    /// but explicit; useful to stop watching *before* a known-slow phase (e.g.
    /// graceful shutdown) that should not be mistaken for a hang.
    pub fn stop(self) {
        // Dropping runs the join in `Drop`.
        drop(self);
    }
}

impl Drop for FrameWatchdog {
    fn drop(&mut self) {
        self.shared.running.store(false, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// The watchdog thread body: poll the beat counter and fire the handler once
/// per stall.
fn watch(shared: &Arc<Shared>, timeout: Duration, poll_interval: Duration, handler: &StallHandler) {
    let mut last_seen = shared.beats.load(Ordering::Relaxed);
    let mut last_change = Instant::now();
    let mut reported = false;
    while shared.running.load(Ordering::Relaxed) {
        sleep_responsive(shared, poll_interval);
        if !shared.running.load(Ordering::Relaxed) {
            break;
        }
        let now = Instant::now();
        let current = shared.beats.load(Ordering::Relaxed);
        if current != last_seen {
            // Progress: reset the stall timer and re-arm the one-shot report.
            last_seen = current;
            last_change = now;
            reported = false;
        } else {
            let waited = now.saturating_duration_since(last_change);
            if !reported && waited >= timeout {
                reported = true;
                shared.stalls.fetch_add(1, Ordering::Relaxed);
                handler(FrameStall {
                    frame: current,
                    stalled_for: waited,
                });
            }
        }
    }
}

/// Sleep up to `total`, but in short steps so a stop request is honored
/// promptly (the join in `Drop` then returns within one step rather than one
/// full poll interval).
fn sleep_responsive(shared: &Shared, total: Duration) {
    const STEP: Duration = Duration::from_millis(20);
    let mut remaining = total;
    while !remaining.is_zero() {
        if !shared.running.load(Ordering::Relaxed) {
            return;
        }
        let step = remaining.min(STEP);
        thread::sleep(step);
        remaining -= step;
    }
}

/// The default stall handler: a one-line notice on standard error.
fn default_stall_handler() -> StallHandler {
    Arc::new(|stall: FrameStall| {
        eprintln!(
            "prism watchdog: main loop stalled on frame {} (quiet for {:?})",
            stall.frame, stall.stalled_for
        );
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use std::sync::atomic::AtomicU64;

    #[test]
    fn beats_are_counted_and_no_false_stall() {
        // A generous timeout the test will never cross: beating a few times must
        // not trip the watchdog.
        let fired = Arc::new(AtomicU64::new(0));
        let f = Arc::clone(&fired);
        let dog = WatchdogConfig::new(Duration::from_secs(60))
            .with_poll_interval(Duration::from_millis(5))
            .with_handler(move |_| {
                f.fetch_add(1, Ordering::Relaxed);
            })
            .start();
        assert!(dog.is_running());
        for _ in 0..4 {
            dog.beat();
            thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(dog.beats(), 4);
        assert_eq!(dog.observed_stalls(), 0);
        assert_eq!(fired.load(Ordering::Relaxed), 0);
        dog.stop();
    }

    #[test]
    fn a_quiet_loop_is_reported_once_with_the_last_frame() {
        let seen = Arc::new(Mutex::new(Vec::<FrameStall>::new()));
        let s = Arc::clone(&seen);
        let dog = WatchdogConfig::new(Duration::from_millis(40))
            .with_poll_interval(Duration::from_millis(5))
            .with_handler(move |stall| {
                s.lock().expect("stall log not poisoned").push(stall);
            })
            .start();
        // Two beats, then go quiet well past the timeout.
        dog.beat();
        dog.beat();
        thread::sleep(Duration::from_millis(250));

        let reports = seen.lock().expect("stall log not poisoned").clone();
        assert_eq!(dog.observed_stalls(), 1, "stall should fire exactly once");
        assert_eq!(reports.len(), 1);
        assert_eq!(reports[0].frame, 2, "last beat before the stall was frame 2");
        assert!(
            reports[0].stalled_for >= Duration::from_millis(40),
            "stall duration should be at least the timeout: {:?}",
            reports[0].stalled_for
        );
        dog.stop();
    }

    #[test]
    fn recovery_rearms_the_one_shot_report() {
        let count = Arc::new(AtomicU64::new(0));
        let c = Arc::clone(&count);
        let dog = WatchdogConfig::new(Duration::from_millis(40))
            .with_poll_interval(Duration::from_millis(5))
            .with_handler(move |_| {
                c.fetch_add(1, Ordering::Relaxed);
            })
            .start();
        // First stall.
        dog.beat();
        thread::sleep(Duration::from_millis(200));
        assert_eq!(count.load(Ordering::Relaxed), 1);
        // Recover (a beat) then stall again: a second report must fire.
        dog.beat();
        thread::sleep(Duration::from_millis(200));
        assert_eq!(
            count.load(Ordering::Relaxed),
            2,
            "a beat between stalls must re-arm the one-shot report"
        );
        assert_eq!(dog.observed_stalls(), 2);
        dog.stop();
    }

    #[test]
    fn poll_interval_is_clamped_to_the_timeout() {
        // A poll interval coarser than the timeout would never observe the
        // stall promptly; `start` clamps it. Here the timeout is small and the
        // requested poll interval is huge; the stall must still be reported.
        let fired = Arc::new(AtomicU64::new(0));
        let f = Arc::clone(&fired);
        let config = WatchdogConfig::new(Duration::from_millis(30))
            .with_poll_interval(Duration::from_secs(10))
            .with_handler(move |_| {
                f.fetch_add(1, Ordering::Relaxed);
            });
        assert_eq!(config.poll_interval(), Duration::from_secs(10));
        assert_eq!(config.timeout(), Duration::from_millis(30));
        let dog = config.start();
        dog.beat();
        thread::sleep(Duration::from_millis(250));
        assert_eq!(fired.load(Ordering::Relaxed), 1);
        dog.stop();
    }
}
