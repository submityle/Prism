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
use crate::schedule::{First, Last, PostUpdate, PreUpdate, Startup, Update};

/// `HeadlessRunner` with a frame cap runs `Update` exactly `cap` times.
#[test]
fn headless_runner_drives_requested_frames() {
    let frames = Arc::new(AtomicU64::new(0));
    let f = frames.clone();

    let mut app = App::new();
    app.add_systems(Update, move || {
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
    app.add_systems(Startup, move || {
        s.fetch_add(1, Ordering::Relaxed);
    });
    app.add_systems(Update, move || {
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
    {
        let o = order.clone();
        app.add_systems(First, move || o.lock().unwrap().push("first"));
    }
    {
        let o = order.clone();
        app.add_systems(PreUpdate, move || o.lock().unwrap().push("pre"));
    }
    {
        let o = order.clone();
        app.add_systems(Update, move || o.lock().unwrap().push("update"));
    }
    {
        let o = order.clone();
        app.add_systems(PostUpdate, move || o.lock().unwrap().push("post"));
    }
    {
        let o = order.clone();
        app.add_systems(Last, move || o.lock().unwrap().push("last"));
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
    app.add_systems(Update, move |mut exit: ResMut<AppExitRequest>| {
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
    app.add_systems(Update, move || {
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

// ---- states ------------------------------------------------------------

use prism_ecs::schedule::{in_state, IntoSystemConfigs, NextState, OnEnter, OnExit, State, States};

/// A tiny two-mode state machine used by the state tests.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
enum Mode {
    #[default]
    Menu,
    Game,
}
impl States for Mode {}

/// `insert_state` uses the deferred-entry model: no `State<S>` resource exists
/// until the first `StateTransition` phase, and that first phase inserts the
/// initial state and runs its `OnEnter`.
#[test]
fn insert_state_enters_initial_on_first_frame() {
    let log = Arc::new(Mutex::new(Vec::<&'static str>::new()));

    let mut app = App::new();
    app.insert_state(Mode::Menu);
    {
        let l = log.clone();
        app.add_systems(OnEnter(Mode::Menu), move || {
            l.lock().unwrap().push("enter_menu");
        });
    }

    // Deferred: nothing is live before the first frame.
    assert!(
        app.world().get_resource::<State<Mode>>().is_none(),
        "State<Mode> must not exist before the first StateTransition"
    );
    assert!(log.lock().unwrap().is_empty());

    app.update();

    assert_eq!(
        app.world().get_resource::<State<Mode>>().map(|s| *s.get()),
        Some(Mode::Menu),
        "first frame installs the initial state"
    );
    assert_eq!(*log.lock().unwrap(), vec!["enter_menu"]);
}

/// A queued `NextState` transition runs `OnExit(old)` then `OnEnter(new)` on the
/// next `StateTransition`, in that order, and updates `State<S>`.
#[test]
fn queued_transition_runs_exit_then_enter() {
    let log = Arc::new(Mutex::new(Vec::<&'static str>::new()));

    let mut app = App::new();
    app.insert_state(Mode::Menu);
    for (label_mode, tag) in [(Mode::Menu, "enter_menu"), (Mode::Game, "enter_game")] {
        let l = log.clone();
        app.add_systems(OnEnter(label_mode), move || l.lock().unwrap().push(tag));
    }
    for (label_mode, tag) in [(Mode::Menu, "exit_menu"), (Mode::Game, "exit_game")] {
        let l = log.clone();
        app.add_systems(OnExit(label_mode), move || l.lock().unwrap().push(tag));
    }

    app.update(); // first entry -> Menu
    assert_eq!(*log.lock().unwrap(), vec!["enter_menu"]);

    app.world_mut()
        .resource_mut::<NextState<Mode>>()
        .set(Mode::Game);
    app.update(); // transition Menu -> Game

    assert_eq!(
        app.world().get_resource::<State<Mode>>().map(|s| *s.get()),
        Some(Mode::Game)
    );
    assert_eq!(
        *log.lock().unwrap(),
        vec!["enter_menu", "exit_menu", "enter_game"]
    );
}

/// `run_if(in_state(..))` gates a system on the current mode. Because
/// `StateTransition` runs before `Update` in a frame, a transition requested
/// before `update()` is visible to that same frame's gated `Update` systems.
#[test]
fn in_state_gates_update_systems() {
    let runs = Arc::new(AtomicU64::new(0));
    let r = runs.clone();

    let mut app = App::new();
    app.insert_state(Mode::Menu);
    app.add_systems(
        Update,
        (move || {
            r.fetch_add(1, Ordering::Relaxed);
        })
        .run_if(in_state(Mode::Game)),
    );

    app.update(); // enters Menu this frame; Update gate (Game) is false
    assert_eq!(runs.load(Ordering::Relaxed), 0, "gated out while in Menu");

    app.world_mut()
        .resource_mut::<NextState<Mode>>()
        .set(Mode::Game);
    app.update(); // StateTransition -> Game, then gated Update runs
    assert_eq!(runs.load(Ordering::Relaxed), 1, "runs once now in Game");
}

/// `init_state` seeds the machine from `S::default()` (here `Mode::Menu`).
#[test]
fn init_state_uses_default_mode() {
    let mut app = App::new();
    app.init_state::<Mode>();
    app.update();
    assert_eq!(
        app.world().get_resource::<State<Mode>>().map(|s| *s.get()),
        Some(Mode::Menu)
    );
}

/// Inserting the same state type twice wires the transition system only once
/// (so the initial `OnEnter` fires exactly once), while re-queuing the latest
/// initial value.
#[test]
fn repeated_insert_state_wires_transition_once() {
    let enters = Arc::new(AtomicU64::new(0));
    let e = enters.clone();

    let mut app = App::new();
    app.insert_state(Mode::Menu);
    app.insert_state(Mode::Game); // re-queues initial as Game; no second system
    {
        let e = e.clone();
        app.add_systems(OnEnter(Mode::Game), move || {
            e.fetch_add(1, Ordering::Relaxed);
        });
    }

    app.update();
    assert_eq!(
        app.world().get_resource::<State<Mode>>().map(|s| *s.get()),
        Some(Mode::Game),
        "latest insert_state wins as the initial mode"
    );
    assert_eq!(
        enters.load(Ordering::Relaxed),
        1,
        "OnEnter(Game) fires exactly once (transition system not double-wired)"
    );
}

// ---- events ----

use prism_ecs::event::{Event, EventCursor, Events};
use prism_ecs::system::{Local, Res};

/// A trivial buffered event carrying a payload for the event tests.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Ping(u32);
impl Event for Ping {}

/// `add_event` installs an empty `Events<E>` resource immediately, before any
/// frame has run.
#[test]
fn add_event_installs_empty_resource() {
    let mut app = App::new();
    app.add_event::<Ping>();
    let events = app
        .world()
        .get_resource::<Events<Ping>>()
        .expect("add_event should install the Events<Ping> resource");
    assert!(events.is_empty(), "freshly added event buffer is empty");
}

/// An event sent during a frame stays buffered that frame and the next, then is
/// retired by the `First`-phase rotation on the following frame.
#[test]
fn event_is_readable_for_one_frame_of_grace_then_retired() {
    let mut app = App::new();
    app.add_event::<Ping>();

    // Send exactly one Ping, on the first `Update` only.
    let sent = Arc::new(AtomicU64::new(0));
    let s = sent.clone();
    app.add_systems(Update, move |mut events: ResMut<Events<Ping>>| {
        if s.fetch_add(1, Ordering::Relaxed) == 0 {
            events.send(Ping(7));
        }
    });

    // Frame 1: First rotates empty buffers, then Update sends the Ping.
    app.update();
    assert_eq!(
        app.world().get_resource::<Events<Ping>>().unwrap().len(),
        1,
        "the event is buffered the frame it is sent"
    );

    // Frame 2: First rotates it into the read buffer; still readable.
    app.update();
    assert_eq!(
        app.world().get_resource::<Events<Ping>>().unwrap().len(),
        1,
        "the event survives one frame of grace"
    );

    // Frame 3: First rotation retires it.
    app.update();
    assert_eq!(
        app.world().get_resource::<Events<Ping>>().unwrap().len(),
        0,
        "the event is retired on the following frame"
    );
}

/// A reader cursor observes every sent event exactly once across frames and
/// never re-observes retired events.
#[test]
fn reader_cursor_sees_each_event_once() {
    let mut app = App::new();
    app.add_event::<Ping>();

    // Send two pings on the first `Update` only.
    let sent = Arc::new(AtomicU64::new(0));
    let s = sent.clone();
    app.add_systems(Update, move |mut events: ResMut<Events<Ping>>| {
        if s.fetch_add(1, Ordering::Relaxed) == 0 {
            events.send(Ping(1));
            events.send(Ping(2));
        }
    });

    // A reader in `PostUpdate` (after the `Update` sender) accumulates what it
    // observes, carrying its cursor across frames via `Local`.
    let seen = Arc::new(Mutex::new(Vec::<u32>::new()));
    let seen_sys = seen.clone();
    app.add_systems(
        PostUpdate,
        move |mut cursor: Local<EventCursor<Ping>>, events: Res<Events<Ping>>| {
            for ping in cursor.read(&events) {
                seen_sys.lock().unwrap().push(ping.0);
            }
        },
    );

    for _ in 0..3 {
        app.update();
    }

    assert_eq!(
        *seen.lock().unwrap(),
        vec![1, 2],
        "each event is observed exactly once, in send order, with no duplicates"
    );
}

/// Calling `add_event` twice for the same type neither resets the buffer (losing
/// already-sent events) nor double-registers the rotation system.
#[test]
fn repeated_add_event_is_idempotent() {
    let mut app = App::new();
    app.add_event::<Ping>();

    // Seed one event directly, then re-register the type.
    app.world_mut()
        .resource_mut::<Events<Ping>>()
        .send(Ping(99));
    app.add_event::<Ping>();

    assert_eq!(
        app.world().get_resource::<Events<Ping>>().unwrap().len(),
        1,
        "re-adding the event type must not reset the buffer"
    );

    // One frame: a single rotation moves the event into the read buffer (still
    // present). If the rotation system were double-wired, two `update()` calls
    // this frame would retire it immediately.
    app.update();
    assert_eq!(
        app.world().get_resource::<Events<Ping>>().unwrap().len(),
        1,
        "exactly one rotation per frame (rotation system wired once)"
    );

    // Next frame retires it.
    app.update();
    assert_eq!(
        app.world().get_resource::<Events<Ping>>().unwrap().len(),
        0,
        "the seeded event retires on the following frame"
    );
}
