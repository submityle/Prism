//! Process-signal runner integration tests.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use prism_app::exit::AppExit;
use prism_app::runner::{run_once, DedicatedServerRunner, HeadlessRunner};
use prism_app::schedule::Shutdown;
use prism_app::App;
use prism_platform::process::signal::{clear, notify, ProcessSignal};

// Signal state is process-global, so keep the two runner scenarios serial.
static SIGNAL_TEST_LOCK: Mutex<()> = Mutex::new(());

#[test]
fn headless_runner_forwards_process_signal_and_shuts_down_once() {
    let _guard = SIGNAL_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    clear();

    let shutdowns = Arc::new(AtomicU64::new(0));
    let s = shutdowns.clone();
    let mut app = App::new();
    app.add_systems(Shutdown, move || {
        s.fetch_add(1, Ordering::Relaxed);
    });
    app.set_runner(|app| HeadlessRunner::new().run(app));
    notify(ProcessSignal::Interrupt);

    let exit = app.run();
    assert_eq!(exit, AppExit::Error(ProcessSignal::Interrupt.exit_code()));
    assert_eq!(shutdowns.load(Ordering::Relaxed), 1);
    clear();
}

#[test]
fn run_once_forwards_process_signal() {
    let _guard = SIGNAL_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    clear();
    notify(ProcessSignal::Terminate);

    let exit = run_once(App::new());
    assert_eq!(exit, AppExit::Error(ProcessSignal::Terminate.exit_code()));
    clear();
}

#[test]
fn dedicated_server_forwards_process_signal_and_shuts_down_once() {
    let _guard = SIGNAL_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    clear();

    let shutdowns = Arc::new(AtomicU64::new(0));
    let s = shutdowns.clone();
    let mut app = App::new();
    app.add_systems(Shutdown, move || {
        s.fetch_add(1, Ordering::Relaxed);
    });
    app.set_runner(|app| {
        DedicatedServerRunner::new(1_000)
            .without_real_time_pacing()
            .run(app)
    });
    notify(ProcessSignal::Hangup);

    let exit = app.run();
    assert_eq!(exit, AppExit::Error(ProcessSignal::Hangup.exit_code()));
    assert_eq!(shutdowns.load(Ordering::Relaxed), 1);
    clear();
}
