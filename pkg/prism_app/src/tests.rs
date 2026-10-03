//! M0 integration tests: the frame loop, startup-once semantics, plugin
//! lifecycle ordering, exit handling, and runner behavior — all against an
//! otherwise empty world.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use prism_ecs::resource::Resource;
use prism_ecs::system::ResMut;

use crate::app::{App, PluginsState};
use crate::exit::{AppExit, AppExitRequest};
use crate::plugin::Plugin;
use crate::runner::HeadlessRunner;
use crate::schedule_label::CoreSchedule;

/// `HeadlessRunner` with a frame cap runs `Update` exactly `cap` times.
#[test]
fn headless_runner_drives_requested_frames() {
    let frames = Arc::new(AtomicU64::new(0));
    let f = frames.clone();

    let mut app = App::new();
    app.add_systems(CoreSchedule::Update, move || {
        f.fetch_add(1, Ordering::Relaxed);
    });
    app.set_runner(|app| HeadlessRunner::with_max_frames(5).run(app));

    let exit = app.run();
    assert_eq!(exit, AppExit::Success);
    assert_eq!(frames.load(Ordering::Relaxed), 5);
}

/// Startup schedules run exactly once; frame schedules run every frame.
#[test]
fn startup_runs_once_update_runs_each_frame() {
    let startups = Arc::new(AtomicU64::new(0));
    let frames = Arc::new(AtomicU64::new(0));
    let s = startups.clone();
    let fr = frames.clone();

    let mut app = App::new();
    app.add_systems(CoreSchedule::Startup, move || {
        s.fetch_add(1, Ordering::Relaxed);
    });
    app.add_systems(CoreSchedule::Update, move || {
        fr.fetch_add(1, Ordering::Relaxed);
    });
    app.set_runner(|app| HeadlessRunner::with_max_frames(3).run(app));

    app.run();
    assert_eq!(startups.load(Ordering::Relaxed), 1, "startup must run once");
    assert_eq!(frames.load(Ordering::Relaxed), 3, "update runs every frame");
}

/// The per-frame phases run in the invariant order
/// `First → PreUpdate → Update → PostUpdate → Last` (design §7 M0 subset).
#[test]
fn frame_phases_run_in_order() {
    let order = Arc::new(Mutex::new(Vec::<&'static str>::new()));

    let mut app = App::new();
    for (label, name) in [
        (CoreSchedule::First, "first"),
        (CoreSchedule::PreUpdate, "pre"),
        (CoreSchedule::Update, "update"),
        (CoreSchedule::PostUpdate, "post"),
        (CoreSchedule::Last, "last"),
    ] {
        let o = order.clone();
        app.add_systems(label, move || o.lock().unwrap().push(name));
    }
    app.set_runner(|app| HeadlessRunner::with_max_frames(1).run(app));
    app.run();

    assert_eq!(
        *order.lock().unwrap(),
        vec!["first", "pre", "update", "post", "last"]
    );
}

/// A system requesting exit stops an otherwise-unbounded `HeadlessRunner`, and
/// the exit code is propagated.
#[test]
fn app_exit_request_stops_headless_runner() {
    let count = Arc::new(AtomicU64::new(0));
    let c = count.clone();

    let mut app = App::new();
    app.add_systems(CoreSchedule::Update, move |mut exit: ResMut<AppExitRequest>| {
        let n = c.fetch_add(1, Ordering::Relaxed) + 1;
        if n == 3 {
            exit.send_error();
        }
    });
    app.set_runner(|app| HeadlessRunner::new().run(app));

    let exit = app.run();
    assert_eq!(exit, AppExit::error());
    assert_eq!(count.load(Ordering::Relaxed), 3);
}

/// With no runner set, `run` defaults to a single frame.
#[test]
fn default_runner_runs_single_frame() {
    let frames = Arc::new(AtomicU64::new(0));
    let f = frames.clone();

    let mut app = App::new();
    app.add_systems(CoreSchedule::Update, move || {
        f.fetch_add(1, Ordering::Relaxed);
    });

    let exit = app.run();
    assert_eq!(exit, AppExit::Success);
    assert_eq!(frames.load(Ordering::Relaxed), 1);
}

// ---- plugin lifecycle --------------------------------------------------

/// Append-only lifecycle log stored in the world so plugin callbacks (which
/// only get `&mut App`) can record their order.
#[derive(Default)]
struct Log(Vec<&'static str>);
impl Resource for Log {}

struct LifecyclePlugin;
impl Plugin for LifecyclePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Log>();
        app.world_mut().resource_mut::<Log>().0.push("build");
    }
    fn finish(&self, app: &mut App) {
        app.world_mut().resource_mut::<Log>().0.push("finish");
    }
    fn cleanup(&self, app: &mut App) {
        app.world_mut().resource_mut::<Log>().0.push("cleanup");
    }
}

/// `build` runs on add, `finish`/`cleanup` run on the matching lifecycle calls,
/// in order, and the assembly state machine advances monotonically.
#[test]
fn plugin_lifecycle_runs_in_order() {
    let mut app = App::new();
    assert_eq!(app.plugins_state(), PluginsState::Adding);

    app.add_plugins(LifecyclePlugin);
    assert_eq!(app.world().resource::<Log>().0, vec!["build"]);
    assert_eq!(app.plugins_state(), PluginsState::Adding);

    app.finish();
    assert_eq!(app.plugins_state(), PluginsState::Finished);
    assert_eq!(app.world().resource::<Log>().0, vec!["build", "finish"]);

    app.cleanup();
    assert_eq!(app.plugins_state(), PluginsState::Cleaned);
    assert_eq!(
        app.world().resource::<Log>().0,
        vec!["build", "finish", "cleanup"]
    );
}

struct NoopPlugin;
impl Plugin for NoopPlugin {
    fn build(&self, _app: &mut App) {}
}

/// A unique plugin added twice panics.
#[test]
#[should_panic(expected = "added more than once")]
fn duplicate_unique_plugin_panics() {
    let mut app = App::new();
    app.add_plugins(NoopPlugin);
    app.add_plugins(NoopPlugin);
}

/// A tuple of plugins is added in order.
#[test]
fn tuple_of_plugins_all_build() {
    #[derive(Default)]
    struct Marks(Vec<&'static str>);
    impl Resource for Marks {}

    struct A;
    impl Plugin for A {
        fn build(&self, app: &mut App) {
            app.init_resource::<Marks>();
            app.world_mut().resource_mut::<Marks>().0.push("a");
        }
    }
    struct B;
    impl Plugin for B {
        fn build(&self, app: &mut App) {
            app.init_resource::<Marks>();
            app.world_mut().resource_mut::<Marks>().0.push("b");
        }
    }

    let mut app = App::new();
    app.add_plugins((A, B));
    assert_eq!(app.world().resource::<Marks>().0, vec!["a", "b"]);
}
