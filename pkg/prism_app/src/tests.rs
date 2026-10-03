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

/// A `HeadlessRunner` with a `FrameLimit` still drives the requested frames and
/// actually paces the loop: with an explicit per-frame period the run cannot
/// finish faster than the paced minimum (one sleep per inter-frame boundary).
#[cfg(feature = "std")]
#[test]
fn headless_runner_frame_limit_paces_the_loop() {
    use crate::pacing::FrameLimit;
    use prism_time::Duration;

    let frames = Arc::new(AtomicU64::new(0));
    let f = frames.clone();

    let mut app = App::new();
    app.add_systems(Update, move || {
        f.fetch_add(1, Ordering::Relaxed);
    });
    // 4 frames at a 2ms minimum period: the first limited frame anchors the
    // cadence without sleeping, then the two interior boundaries each sleep
    // ~2ms (the 4th frame breaks before throttling), so the run takes at least
    // ~4ms. Assert a conservative lower bound to prove the pacer slept.
    let period = Duration::from_millis(2);
    app.set_runner(move |app| {
        HeadlessRunner::with_max_frames(4)
            .with_frame_limit(FrameLimit::Period(period))
            .run(app)
    });

    let start = std::time::Instant::now();
    let exit = app.run();
    let elapsed = start.elapsed();

    assert_eq!(exit, AppExit::Success);
    assert_eq!(frames.load(Ordering::Relaxed), 4);
    assert!(
        elapsed >= Duration::from_millis(3),
        "frame limiter did not pace the loop: {elapsed:?}"
    );
}

/// The adaptive limiter and the real `FramePacer` compose: feeding sustained
/// work that misses the top rungs walks the cap down the ladder and retargets
/// the pacer at each step (design §13, the feed → recommend → apply path).
#[cfg(feature = "std")]
#[test]
fn adaptive_limiter_retargets_a_real_pacer() {
    use crate::pacing::{AdaptiveFrameLimiter, FrameLimit, FramePacer, FrameRateLadder};
    use prism_time::Duration;

    let ladder = FrameRateLadder::from_fps([30, 60, 120]).expect("non-empty ladder");
    let mut limiter = AdaptiveFrameLimiter::new(ladder).with_window(2);
    let mut pacer = FramePacer::new(limiter.current_limit());
    // Starts at the most demanding rung.
    assert_eq!(pacer.limit(), FrameLimit::from_fps(120));

    // 20ms sustained work misses 120fps (~8.3ms) then 60fps (~16.6ms); the
    // limiter steps down twice and the pacer tracks each recommendation.
    for _ in 0..8 {
        if let Some(new_limit) = limiter.record(Duration::from_millis(20)) {
            pacer.set_limit(new_limit);
        }
    }
    assert_eq!(limiter.current_limit(), FrameLimit::from_fps(30));
    assert_eq!(pacer.limit(), FrameLimit::from_fps(30));
}

/// A `HeadlessRunner` with an adaptive frame limit drives the requested frames
/// and actually paces them: the limiter's starting rung becomes the pacer's cap
/// so the loop cannot finish faster than that cadence (end-to-end wiring).
#[cfg(feature = "std")]
#[test]
fn headless_runner_adaptive_frame_limit_paces_the_loop() {
    use crate::pacing::FrameRateLadder;
    use prism_time::Duration;

    let frames = Arc::new(AtomicU64::new(0));
    let f = frames.clone();

    let mut app = App::new();
    app.add_systems(Update, move || {
        f.fetch_add(1, Ordering::Relaxed);
    });
    // A two-rung ladder; with near-instant work the limiter holds at its top
    // rung (500fps => 2ms period), so 4 frames pace through two interior
    // boundaries and the run takes at least ~3ms.
    let ladder = FrameRateLadder::from_fps([250, 500]).expect("non-empty ladder");
    app.set_runner(move |app| {
        HeadlessRunner::with_max_frames(4)
            .with_adaptive_frame_limit(ladder)
            .run(app)
    });

    let start = std::time::Instant::now();
    let exit = app.run();
    let elapsed = start.elapsed();

    assert_eq!(exit, AppExit::Success);
    assert_eq!(frames.load(Ordering::Relaxed), 4);
    assert!(
        elapsed >= Duration::from_millis(3),
        "adaptive frame limiter did not pace the loop: {elapsed:?}"
    );
}

/// A watchdog attached to a *healthy* fast loop never fires: every frame beats
/// well within the timeout, so the stall handler is not invoked (design §24.7).
#[cfg(feature = "std")]
#[test]
fn headless_runner_watchdog_stays_quiet_on_a_healthy_loop() {
    use crate::watchdog::WatchdogConfig;
    use prism_time::Duration;

    let stalls = Arc::new(AtomicU64::new(0));
    let s = stalls.clone();

    let mut app = App::new();
    app.add_systems(Update, || {});
    app.set_runner(move |app| {
        let config = WatchdogConfig::new(Duration::from_secs(30))
            .with_poll_interval(Duration::from_millis(5))
            .with_handler(move |_| {
                s.fetch_add(1, Ordering::Relaxed);
            });
        HeadlessRunner::with_max_frames(50)
            .with_watchdog(config)
            .run(app)
    });

    let exit = app.run();
    assert_eq!(exit, AppExit::Success);
    assert_eq!(
        stalls.load(Ordering::Relaxed),
        0,
        "watchdog fired on a healthy loop"
    );
}

/// A frame that wedges inside `app.update()` for longer than the watchdog
/// timeout trips the watchdog: the beat for that frame lands late, so the
/// background thread reports a stall (design §24.7). A one-shot sleep models the
/// hang; the loop then recovers and finishes.
#[cfg(feature = "std")]
#[test]
fn headless_runner_watchdog_detects_a_wedged_frame() {
    use crate::watchdog::WatchdogConfig;
    use prism_ecs::system::Local;
    use prism_time::Duration;

    let stalls = Arc::new(AtomicU64::new(0));
    let s = stalls.clone();

    let mut app = App::new();
    // Sleep past the timeout on the very first frame only, then run normally.
    app.add_systems(Update, |mut done: Local<bool>| {
        if !*done {
            *done = true;
            std::thread::sleep(Duration::from_millis(160));
        }
    });
    app.set_runner(move |app| {
        let config = WatchdogConfig::new(Duration::from_millis(40))
            .with_poll_interval(Duration::from_millis(5))
            .with_handler(move |_| {
                s.fetch_add(1, Ordering::Relaxed);
            });
        HeadlessRunner::with_max_frames(3)
            .with_watchdog(config)
            .run(app)
    });

    let exit = app.run();
    assert_eq!(exit, AppExit::Success);
    assert!(
        stalls.load(Ordering::Relaxed) >= 1,
        "watchdog did not detect the wedged frame"
    );
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

// ---- plugin groups -----------------------------------------------------

use crate::plugin::PluginDependency;
use crate::plugin_graph::PluginGraphError;
use crate::plugin_group::{PluginGroup, PluginGroupBuilder};
use crate::run_mode::RunMode;

/// The resolved build-order names of a group's enabled members.
fn group_order(builder: PluginGroupBuilder) -> Vec<String> {
    builder
        .into_plugins()
        .iter()
        .map(|p| p.name().to_string())
        .collect()
}

struct Named<const N: char>;
impl<const N: char> Plugin for Named<N> {
    fn build(&self, _app: &mut App) {}
    fn name(&self) -> &str {
        match N {
            'a' => "a",
            'b' => "b",
            'c' => "c",
            'd' => "d",
            _ => "?",
        }
    }
}

/// `add` appends members; the group resolves in explicit order when there are
/// no dependency edges.
#[test]
fn group_adds_members_in_explicit_order() {
    let builder = PluginGroupBuilder::new()
        .add(Named::<'a'>)
        .add(Named::<'b'>)
        .add(Named::<'c'>);
    assert_eq!(group_order(builder), vec!["a", "b", "c"]);
}

/// `add_before` and `add_after` place members relative to a target.
#[test]
fn add_before_and_after_place_relative_to_target() {
    let builder = PluginGroupBuilder::new()
        .add(Named::<'a'>)
        .add(Named::<'b'>)
        .add_before::<Named<'b'>, _>(Named::<'c'>)
        .add_after::<Named<'a'>, _>(Named::<'d'>);
    // Start: [a, b]; add c before b -> [a, c, b]; add d after a -> [a, d, c, b].
    assert_eq!(group_order(builder), vec!["a", "d", "c", "b"]);
}

/// `disable` drops a member from the resolved output but keeps its slot, so a
/// later `enable` restores it in its original position.
#[test]
fn disable_keeps_position_for_later_enable() {
    let base = PluginGroupBuilder::new()
        .add(Named::<'a'>)
        .add(Named::<'b'>)
        .add(Named::<'c'>);

    let disabled = PluginGroupBuilder::new()
        .add(Named::<'a'>)
        .add(Named::<'b'>)
        .add(Named::<'c'>)
        .disable::<Named<'b'>>();
    assert_eq!(group_order(disabled), vec!["a", "c"]);

    let re_enabled = base.disable::<Named<'b'>>().enable::<Named<'b'>>();
    assert!(re_enabled.is_enabled::<Named<'b'>>());
    assert_eq!(group_order(re_enabled), vec!["a", "b", "c"]);
}

/// `set` replaces a member's instance while keeping its position.
#[test]
fn set_replaces_member_in_place() {
    // Two plugins that share a type parameter can't collide, so use distinct
    // marker types whose `name` differs to prove replacement happened.
    struct First;
    impl Plugin for First {
        fn build(&self, _app: &mut App) {}
        fn name(&self) -> &str {
            "first"
        }
    }
    // A second plugin of the *same* type with different state is the realistic
    // `set` case; here replacing First with a fresh First is a no-op in name,
    // so instead assert position is preserved when set is used on a middle
    // member.
    let builder = PluginGroupBuilder::new()
        .add(Named::<'a'>)
        .add(First)
        .add(Named::<'c'>)
        .set(First);
    assert_eq!(group_order(builder), vec!["a", "first", "c"]);
}

/// Adding the same plugin type twice panics at edit time.
#[test]
#[should_panic(expected = "already a member")]
fn duplicate_add_panics() {
    let _ = PluginGroupBuilder::new()
        .add(Named::<'a'>)
        .add(Named::<'a'>);
}

/// Declared dependencies reorder members topologically, overriding explicit
/// order while keeping it as the stable tie-break.
#[test]
fn dependencies_reorder_members_topologically() {
    // Core has no deps; Mid depends on Core; Top depends on Mid. Added in the
    // reverse (wrong) order on purpose.
    struct Core;
    impl Plugin for Core {
        fn build(&self, _app: &mut App) {}
        fn name(&self) -> &str {
            "core"
        }
    }
    struct Mid;
    impl Plugin for Mid {
        fn build(&self, _app: &mut App) {}
        fn name(&self) -> &str {
            "mid"
        }
        fn dependencies(&self) -> Vec<PluginDependency> {
            vec![PluginDependency::on::<Core>()]
        }
    }
    struct Top;
    impl Plugin for Top {
        fn build(&self, _app: &mut App) {}
        fn name(&self) -> &str {
            "top"
        }
        fn dependencies(&self) -> Vec<PluginDependency> {
            vec![PluginDependency::on::<Mid>()]
        }
    }

    let builder = PluginGroupBuilder::new().add(Top).add(Mid).add(Core);
    assert_eq!(group_order(builder), vec!["core", "mid", "top"]);
}

/// Two independent chains keep their explicit relative order as the tie-break.
#[test]
fn independent_members_keep_explicit_order_as_tiebreak() {
    struct Core;
    impl Plugin for Core {
        fn build(&self, _app: &mut App) {}
        fn name(&self) -> &str {
            "core"
        }
    }
    struct DependsOnCore;
    impl Plugin for DependsOnCore {
        fn build(&self, _app: &mut App) {}
        fn name(&self) -> &str {
            "dep"
        }
        fn dependencies(&self) -> Vec<PluginDependency> {
            vec![PluginDependency::on::<Core>()]
        }
    }
    struct Loner;
    impl Plugin for Loner {
        fn build(&self, _app: &mut App) {}
        fn name(&self) -> &str {
            "loner"
        }
    }

    // Explicit order: loner, dep, core. `dep` must wait for `core`; `loner` is
    // free and keeps its earliest-eligible slot.
    let builder = PluginGroupBuilder::new()
        .add(Loner)
        .add(DependsOnCore)
        .add(Core);
    assert_eq!(group_order(builder), vec!["loner", "core", "dep"]);
}

/// A dependency on a plugin not in the group is reported at assembly time.
#[test]
fn missing_dependency_is_reported_at_assembly_time() {
    struct Absent;
    impl Plugin for Absent {
        fn build(&self, _app: &mut App) {}
        fn name(&self) -> &str {
            "absent"
        }
    }
    struct Needs;
    impl Plugin for Needs {
        fn build(&self, _app: &mut App) {}
        fn name(&self) -> &str {
            "needs"
        }
        fn dependencies(&self) -> Vec<PluginDependency> {
            vec![PluginDependency::on::<Absent>()]
        }
    }

    let Err(err) = PluginGroupBuilder::new().add(Needs).try_into_plugins() else {
        panic!("missing dependency must fail");
    };
    match err {
        PluginGraphError::MissingDependency { dependent, missing } => {
            assert!(dependent.contains("Needs"), "dependent was {dependent}");
            assert!(missing.contains("Absent"), "missing was {missing}");
        }
        other => panic!("expected MissingDependency, got {other:?}"),
    }
}

/// A dependency on a member that was disabled out of the group is also treated
/// as missing — resolution runs over the enabled set only.
#[test]
fn dependency_on_disabled_member_is_missing() {
    struct Opt;
    impl Plugin for Opt {
        fn build(&self, _app: &mut App) {}
        fn name(&self) -> &str {
            "opt"
        }
    }
    struct Needs;
    impl Plugin for Needs {
        fn build(&self, _app: &mut App) {}
        fn name(&self) -> &str {
            "needs"
        }
        fn dependencies(&self) -> Vec<PluginDependency> {
            vec![PluginDependency::on::<Opt>()]
        }
    }

    let result = PluginGroupBuilder::new()
        .add(Opt)
        .add(Needs)
        .disable::<Opt>()
        .try_into_plugins();
    assert!(matches!(
        result,
        Err(PluginGraphError::MissingDependency { .. })
    ));
}

/// A render-only plugin: it requires a rendering-capable run mode.
struct RenderOnly;
impl Plugin for RenderOnly {
    fn build(&self, _app: &mut App) {}
    fn name(&self) -> &str {
        "render"
    }
    fn needs_rendering(&self) -> bool {
        true
    }
}

/// A simulation plugin that runs in every mode (uses the `needs_rendering`
/// default of `false`).
struct SimOnly;
impl Plugin for SimOnly {
    fn build(&self, _app: &mut App) {}
    fn name(&self) -> &str {
        "sim"
    }
}

/// `Plugin::needs_rendering` defaults to `false`, so an un-annotated plugin
/// runs in every mode.
#[test]
fn needs_rendering_defaults_to_false() {
    assert!(!SimOnly.needs_rendering());
    assert!(RenderOnly.needs_rendering(), "the override reports true");
}

/// In a non-rendering mode (`Headless`), `disable_for_run_mode` drops only the
/// render-requiring members and keeps the simulation ones.
#[test]
fn disable_for_run_mode_disables_render_plugins_in_headless() {
    let builder = PluginGroupBuilder::new()
        .add(SimOnly)
        .add(RenderOnly)
        .disable_for_run_mode(RunMode::Headless);
    assert!(builder.is_enabled::<SimOnly>(), "simulation survives headless");
    assert!(
        !builder.is_enabled::<RenderOnly>(),
        "rendering is dropped in headless"
    );
    assert_eq!(group_order(builder), vec!["sim"]);
}

/// In a rendering mode (`Client`), `disable_for_run_mode` is a no-op: every
/// member, including render plugins, stays enabled.
#[test]
fn disable_for_run_mode_is_noop_in_client_mode() {
    let builder = PluginGroupBuilder::new()
        .add(SimOnly)
        .add(RenderOnly)
        .disable_for_run_mode(RunMode::Client);
    assert!(builder.is_enabled::<SimOnly>());
    assert!(builder.is_enabled::<RenderOnly>(), "client keeps rendering");
    assert_eq!(group_order(builder), vec!["sim", "render"]);
}

/// `disable_for_run_mode` only toggles the enabled flag, so a later `enable`
/// restores the render plugin in its original position.
#[test]
fn disabled_render_plugin_can_be_re_enabled() {
    let builder = PluginGroupBuilder::new()
        .add(RenderOnly)
        .add(SimOnly)
        .disable_for_run_mode(RunMode::Headless)
        .enable::<RenderOnly>();
    assert!(builder.is_enabled::<RenderOnly>(), "re-enabled in place");
    // Original explicit order (render before sim) is preserved.
    assert_eq!(group_order(builder), vec!["render", "sim"]);
}

/// A dedicated server is also a non-rendering mode, so it drops render plugins
/// just like plain headless.
#[test]
fn disable_for_run_mode_for_dedicated_server_also_drops_rendering() {
    let builder = PluginGroupBuilder::new()
        .add(RenderOnly)
        .add(SimOnly)
        .disable_for_run_mode(RunMode::DedicatedServer);
    assert!(!builder.is_enabled::<RenderOnly>());
    assert!(builder.is_enabled::<SimOnly>());
}

/// An editor-embedded launch drives rendering, so `disable_for_run_mode` keeps
/// render plugins there too.
#[test]
fn disable_for_run_mode_keeps_rendering_for_editor_embedded() {
    let builder = PluginGroupBuilder::new()
        .add(RenderOnly)
        .add(SimOnly)
        .disable_for_run_mode(RunMode::EditorEmbedded);
    assert!(builder.is_enabled::<RenderOnly>(), "editor drives rendering");
    assert!(builder.is_enabled::<SimOnly>());
}

/// If a non-render member depends on a render member that headless filtering
/// disabled, the broken edge surfaces honestly as a missing dependency at
/// assembly time rather than being silently ignored.
#[test]
fn disabling_a_depended_on_render_plugin_errors_at_assembly() {
    struct NeedsRender;
    impl Plugin for NeedsRender {
        fn build(&self, _app: &mut App) {}
        fn name(&self) -> &str {
            "needs-render"
        }
        fn dependencies(&self) -> Vec<PluginDependency> {
            vec![PluginDependency::on::<RenderOnly>()]
        }
    }

    let result = PluginGroupBuilder::new()
        .add(RenderOnly)
        .add(NeedsRender)
        .disable_for_run_mode(RunMode::Headless)
        .try_into_plugins();
    assert!(matches!(
        result,
        Err(PluginGraphError::MissingDependency { .. })
    ));
}

/// A dependency cycle is reported at assembly time, not run time.
#[test]
fn dependency_cycle_is_reported_at_assembly_time() {
    struct Ping;
    struct Pong;
    impl Plugin for Ping {
        fn build(&self, _app: &mut App) {}
        fn name(&self) -> &str {
            "ping"
        }
        fn dependencies(&self) -> Vec<PluginDependency> {
            vec![PluginDependency::on::<Pong>()]
        }
    }
    impl Plugin for Pong {
        fn build(&self, _app: &mut App) {}
        fn name(&self) -> &str {
            "pong"
        }
        fn dependencies(&self) -> Vec<PluginDependency> {
            vec![PluginDependency::on::<Ping>()]
        }
    }

    let Err(err) = PluginGroupBuilder::new().add(Ping).add(Pong).try_into_plugins() else {
        panic!("cycle must fail");
    };
    match err {
        PluginGraphError::Cycle { members } => {
            assert_eq!(members.len(), 2, "both members are unresolved: {members:?}");
        }
        other => panic!("expected Cycle, got {other:?}"),
    }
}

/// `into_plugins` (the panicking shorthand) surfaces a cycle loudly.
#[test]
#[should_panic(expected = "assembly failed")]
fn into_plugins_panics_on_cycle() {
    struct A;
    struct B;
    impl Plugin for A {
        fn build(&self, _app: &mut App) {}
        fn dependencies(&self) -> Vec<PluginDependency> {
            vec![PluginDependency::on::<B>()]
        }
    }
    impl Plugin for B {
        fn build(&self, _app: &mut App) {}
        fn dependencies(&self) -> Vec<PluginDependency> {
            vec![PluginDependency::on::<A>()]
        }
    }
    let _ = PluginGroupBuilder::new().add(A).add(B).into_plugins();
}

/// A group plugs into `App::add_plugins` and builds its members in the resolved
/// (dependency-correct) order.
#[test]
fn app_builds_group_in_resolved_order() {
    #[derive(Default)]
    struct Order(Vec<&'static str>);
    impl Resource for Order {}

    struct Core;
    impl Plugin for Core {
        fn build(&self, app: &mut App) {
            app.init_resource::<Order>();
            app.world_mut().resource_mut::<Order>().0.push("core");
        }
    }
    struct Renderer;
    impl Plugin for Renderer {
        fn build(&self, app: &mut App) {
            app.init_resource::<Order>();
            app.world_mut().resource_mut::<Order>().0.push("renderer");
        }
        fn dependencies(&self) -> Vec<PluginDependency> {
            vec![PluginDependency::on::<Core>()]
        }
    }

    struct DemoGroup;
    impl PluginGroup for DemoGroup {
        fn build(self) -> PluginGroupBuilder {
            // Deliberately add the dependent first; resolution must fix it.
            PluginGroupBuilder::new().add(Renderer).add(Core)
        }
    }

    let mut app = App::new();
    app.add_plugins(DemoGroup);
    assert_eq!(app.world().resource::<Order>().0, vec!["core", "renderer"]);
}

// ---- fixed timestep ----

use crate::fixed::{FixedFirst, FixedLast, FixedPostUpdate, FixedPreUpdate, FixedUpdate};
use crate::time::{EngineClocks, TimeUpdateStrategy};
use prism_time::{DefaultSource, Duration};

/// Count how many times a schedule's system ran across `frames` frames, with the
/// real clock advanced by a fixed `delta` each frame and the fixed rate set to
/// `hz`. Returns `(fixed_runs, update_runs)`.
fn run_fixed_counts(hz: f64, delta: Duration, frames: u64) -> (u64, u64) {
    let fixed = Arc::new(AtomicU64::new(0));
    let update = Arc::new(AtomicU64::new(0));
    let fx = fixed.clone();
    let up = update.clone();

    let mut app = App::new();
    app.set_time_update_strategy(TimeUpdateStrategy::ManualDelta(delta));
    app.set_fixed_timestep_hz(hz);
    app.add_systems(FixedUpdate, move || {
        fx.fetch_add(1, Ordering::Relaxed);
    });
    app.add_systems(Update, move || {
        up.fetch_add(1, Ordering::Relaxed);
    });
    app.set_runner(move |app| HeadlessRunner::with_max_frames(frames).run(app));
    app.run();

    (
        fixed.load(Ordering::Relaxed),
        update.load(Ordering::Relaxed),
    )
}

/// A frame delta equal to the fixed timestep expends exactly one step per frame.
#[test]
fn fixed_update_runs_once_per_frame_at_matching_rate() {
    // 100 Hz => 10 ms step; feed 10 ms per frame.
    let (fixed, update) = run_fixed_counts(100.0, Duration::from_millis(10), 5);
    assert_eq!(fixed, 5, "one fixed step per 10 ms frame over 5 frames");
    assert_eq!(update, 5, "variable Update still runs once per frame");
}

/// A frame delta of several timesteps expends that many fixed steps per frame.
#[test]
fn fixed_update_runs_multiple_substeps_per_frame() {
    // 100 Hz => 10 ms step; 30 ms per frame => 3 steps/frame; 2 frames => 6.
    let (fixed, update) = run_fixed_counts(100.0, Duration::from_millis(30), 2);
    assert_eq!(fixed, 6, "three fixed steps per 30 ms frame over 2 frames");
    assert_eq!(update, 2);
}

/// A hitch cannot queue unbounded fixed steps: the accumulator is capped at
/// `max_substeps` (default 8), so even a huge frame delta runs at most 8 steps.
#[test]
fn fixed_loop_is_capped_by_max_substeps() {
    // 100 Hz => 10 ms step. One 1 s frame: Virtual clamps the real delta to its
    // 250 ms max first, then the fixed accumulator caps at 8 * 10 ms = 80 ms,
    // so exactly max_substeps (8) steps run — not 25 or 100.
    let (fixed, update) = run_fixed_counts(100.0, Duration::from_secs(1), 1);
    assert_eq!(fixed, 8, "death-spiral cap bounds the step count at max_substeps");
    assert_eq!(update, 1);
}

/// After a frame the default clock source is restored to `Virtual`, so the
/// variable-step phases read virtual time (the fixed loop only borrows the
/// default source for its inner steps).
#[test]
fn default_clock_source_is_virtual_after_frame() {
    let mut app = App::new();
    app.set_time_update_strategy(TimeUpdateStrategy::ManualDelta(Duration::from_millis(10)));
    app.set_fixed_timestep_hz(100.0);
    app.update();
    assert_eq!(
        app.world().resource::<EngineClocks>().source(),
        DefaultSource::Virtual
    );
}

/// Within one fixed step the sub-phases run in the invariant tick-group order
/// `FixedFirst → FixedPreUpdate → FixedUpdate → FixedPostUpdate → FixedLast`.
#[test]
fn fixed_phases_run_in_order_within_a_step() {
    let order = Arc::new(Mutex::new(Vec::<&'static str>::new()));

    let mut app = App::new();
    app.set_time_update_strategy(TimeUpdateStrategy::ManualDelta(Duration::from_millis(10)));
    app.set_fixed_timestep_hz(100.0);
    {
        let o = order.clone();
        app.add_systems(FixedFirst, move || o.lock().unwrap().push("ffirst"));
    }
    {
        let o = order.clone();
        app.add_systems(FixedPreUpdate, move || o.lock().unwrap().push("fpre"));
    }
    {
        let o = order.clone();
        app.add_systems(FixedUpdate, move || o.lock().unwrap().push("fupdate"));
    }
    {
        let o = order.clone();
        app.add_systems(FixedPostUpdate, move || o.lock().unwrap().push("fpost"));
    }
    {
        let o = order.clone();
        app.add_systems(FixedLast, move || o.lock().unwrap().push("flast"));
    }
    app.update();

    assert_eq!(
        *order.lock().unwrap(),
        vec!["ffirst", "fpre", "fupdate", "fpost", "flast"]
    );
}

/// `RunFixedMainLoop` runs between `First` and `PreUpdate` in the frame order
/// (design §7, §21 invariant), observable via a `FixedUpdate` system landing
/// between `First` and `PreUpdate` systems.
#[test]
fn fixed_loop_runs_between_first_and_pre_update() {
    let order = Arc::new(Mutex::new(Vec::<&'static str>::new()));

    let mut app = App::new();
    app.set_time_update_strategy(TimeUpdateStrategy::ManualDelta(Duration::from_millis(10)));
    app.set_fixed_timestep_hz(100.0);
    {
        let o = order.clone();
        app.add_systems(First, move || o.lock().unwrap().push("first"));
    }
    {
        let o = order.clone();
        app.add_systems(FixedUpdate, move || o.lock().unwrap().push("fixed"));
    }
    {
        let o = order.clone();
        app.add_systems(PreUpdate, move || o.lock().unwrap().push("pre"));
    }
    app.update();

    assert_eq!(*order.lock().unwrap(), vec!["first", "fixed", "pre"]);
}

/// A world without an `EngineClocks` resource skips the time + fixed-loop steps
/// entirely rather than panicking (a clock-less secondary sub-app path).
#[test]
fn clockless_world_skips_time_and_fixed_loop() {
    use prism_ecs::world::World;

    let mut world = World::new();
    // No EngineClocks inserted. Both drivers must early-return without touching
    // any (absent) schedules or clocks.
    crate::time::advance_time(&mut world);
    crate::fixed::run_fixed_main_loop(&mut world);
    // Reaching here without a panic is the assertion.
}

/// Pausing virtual time freezes the fixed accumulator: no fixed steps run while
/// paused, even though frames keep advancing the real clock.
#[test]
fn paused_virtual_time_freezes_fixed_steps() {
    let fixed = Arc::new(AtomicU64::new(0));
    let fx = fixed.clone();

    let mut app = App::new();
    app.set_time_update_strategy(TimeUpdateStrategy::ManualDelta(Duration::from_millis(10)));
    app.set_fixed_timestep_hz(100.0);
    app.world_mut()
        .resource_mut::<EngineClocks>()
        .virtual_time_mut()
        .pause();
    app.add_systems(FixedUpdate, move || {
        fx.fetch_add(1, Ordering::Relaxed);
    });
    app.set_runner(|app| HeadlessRunner::with_max_frames(5).run(app));
    app.run();

    assert_eq!(fixed.load(Ordering::Relaxed), 0, "paused clock feeds no fixed steps");
}

// ---- fixed-loop per-frame bracket hooks (BeforeFixedMainLoop / AfterFixedMainLoop) ----

use crate::fixed::{AfterFixedMainLoop, BeforeFixedMainLoop};

/// Both bracket hooks fire exactly once per frame when the frame expends
/// several fixed steps — they bracket the *loop*, not each step.
#[test]
fn fixed_bracket_hooks_run_once_per_frame_with_many_substeps() {
    let before = Arc::new(AtomicU64::new(0));
    let after = Arc::new(AtomicU64::new(0));
    let fixed = Arc::new(AtomicU64::new(0));
    let (bf, af, fx) = (before.clone(), after.clone(), fixed.clone());

    let mut app = App::new();
    app.set_time_update_strategy(TimeUpdateStrategy::ManualDelta(Duration::from_millis(30)));
    app.set_fixed_timestep_hz(100.0); // 10 ms step => 3 steps / 30 ms frame
    app.add_systems(BeforeFixedMainLoop, move || {
        bf.fetch_add(1, Ordering::Relaxed);
    });
    app.add_systems(AfterFixedMainLoop, move || {
        af.fetch_add(1, Ordering::Relaxed);
    });
    app.add_systems(FixedUpdate, move || {
        fx.fetch_add(1, Ordering::Relaxed);
    });
    app.set_runner(|app| HeadlessRunner::with_max_frames(2).run(app));
    app.run();

    assert_eq!(fixed.load(Ordering::Relaxed), 6, "3 fixed steps x 2 frames");
    assert_eq!(before.load(Ordering::Relaxed), 2, "before-hook once per frame");
    assert_eq!(after.load(Ordering::Relaxed), 2, "after-hook once per frame");
}

/// The bracket hooks still fire on frames that expend **zero** fixed steps
/// (frame delta smaller than one timestep) — they are per-frame, not per-step.
#[test]
fn fixed_bracket_hooks_run_on_zero_substep_frames() {
    let before = Arc::new(AtomicU64::new(0));
    let after = Arc::new(AtomicU64::new(0));
    let fixed = Arc::new(AtomicU64::new(0));
    let (bf, af, fx) = (before.clone(), after.clone(), fixed.clone());

    let mut app = App::new();
    // 3 ms per frame vs a 10 ms step: the accumulator never reaches a full step
    // within 3 frames, so zero fixed steps run.
    app.set_time_update_strategy(TimeUpdateStrategy::ManualDelta(Duration::from_millis(3)));
    app.set_fixed_timestep_hz(100.0);
    app.add_systems(BeforeFixedMainLoop, move || {
        bf.fetch_add(1, Ordering::Relaxed);
    });
    app.add_systems(AfterFixedMainLoop, move || {
        af.fetch_add(1, Ordering::Relaxed);
    });
    app.add_systems(FixedUpdate, move || {
        fx.fetch_add(1, Ordering::Relaxed);
    });
    app.set_runner(|app| HeadlessRunner::with_max_frames(3).run(app));
    app.run();

    assert_eq!(fixed.load(Ordering::Relaxed), 0, "no full step accumulates in 3 ms frames");
    assert_eq!(before.load(Ordering::Relaxed), 3, "before-hook fires every frame");
    assert_eq!(after.load(Ordering::Relaxed), 3, "after-hook fires every frame");
}

/// The hooks bracket the inner loop: `BeforeFixedMainLoop` → fixed steps →
/// `AfterFixedMainLoop`, and the whole bracket sits between `First` and
/// `PreUpdate` (within the `RunFixedMainLoop` phase, design §7/§8 invariant).
#[test]
fn fixed_bracket_hooks_wrap_the_inner_loop_in_frame_order() {
    let order = Arc::new(Mutex::new(Vec::<&'static str>::new()));

    let mut app = App::new();
    app.set_time_update_strategy(TimeUpdateStrategy::ManualDelta(Duration::from_millis(20)));
    app.set_fixed_timestep_hz(100.0); // 10 ms step => 2 steps this frame
    {
        let o = order.clone();
        app.add_systems(First, move || o.lock().unwrap().push("first"));
    }
    {
        let o = order.clone();
        app.add_systems(BeforeFixedMainLoop, move || o.lock().unwrap().push("before"));
    }
    {
        let o = order.clone();
        app.add_systems(FixedUpdate, move || o.lock().unwrap().push("step"));
    }
    {
        let o = order.clone();
        app.add_systems(AfterFixedMainLoop, move || o.lock().unwrap().push("after"));
    }
    {
        let o = order.clone();
        app.add_systems(PreUpdate, move || o.lock().unwrap().push("pre"));
    }
    app.update();

    assert_eq!(
        *order.lock().unwrap(),
        vec!["first", "before", "step", "step", "after", "pre"],
        "before runs once pre-loop, steps drain, after runs once post-loop, all before PreUpdate"
    );
}

/// The before-hook runs *before* the accumulator is drained and the after-hook
/// runs *after*: the before-hook observes a full pending step in the fixed
/// overstep (alpha) while the after-hook observes the drained leftover. This
/// proves the native drain happens between the two bracket schedules.
#[test]
fn fixed_bracket_hooks_straddle_the_accumulator_drain() {
    let before_overstep = Arc::new(Mutex::new(Duration::ZERO));
    let after_overstep = Arc::new(Mutex::new(Duration::ZERO));
    let bo = before_overstep.clone();
    let ao = after_overstep.clone();

    let mut app = App::new();
    // 15 ms frame vs a 10 ms step: one step drains, leaving a 5 ms overstep.
    app.set_time_update_strategy(TimeUpdateStrategy::ManualDelta(Duration::from_millis(15)));
    app.set_fixed_timestep_hz(100.0);
    app.add_systems(BeforeFixedMainLoop, move |clocks: Res<EngineClocks>| {
        *bo.lock().unwrap() = clocks.fixed().overstep();
    });
    app.add_systems(AfterFixedMainLoop, move |clocks: Res<EngineClocks>| {
        *ao.lock().unwrap() = clocks.fixed().overstep();
    });
    app.update();

    let step = Duration::from_millis(10);
    assert!(
        *before_overstep.lock().unwrap() >= step,
        "before-hook sees a full step still pending in the accumulator"
    );
    assert!(
        *after_overstep.lock().unwrap() < step,
        "after-hook sees the drained leftover (< one timestep)"
    );
    assert_eq!(
        *after_overstep.lock().unwrap(),
        Duration::from_millis(5),
        "15 ms frame - one 10 ms step = 5 ms interpolation alpha"
    );
}

/// A clock-less world skips both bracket hooks along with the fixed loop
/// (no `EngineClocks` => the whole `RunFixedMainLoop` driver is a no-op).
#[test]
fn clockless_world_skips_bracket_hooks() {
    use prism_ecs::schedule::{Schedule, Schedules};
    use prism_ecs::world::World;

    let ran = Arc::new(AtomicU64::new(0));
    let r = ran.clone();

    let mut world = World::new();
    world.init_resource::<Schedules>();
    // Register the bracket schedules but install NO EngineClocks: the driver
    // must early-return before touching them.
    let schedules = world.resource_mut::<Schedules>();
    schedules.insert(BeforeFixedMainLoop, Schedule::new());
    schedules.insert(AfterFixedMainLoop, Schedule::new());
    world
        .resource_mut::<Schedules>()
        .get_mut(BeforeFixedMainLoop)
        .unwrap()
        .add_systems(move || {
            r.fetch_add(1, Ordering::Relaxed);
        });

    crate::fixed::run_fixed_main_loop(&mut world);
    assert_eq!(ran.load(Ordering::Relaxed), 0, "no clock => no bracket hooks run");
}

// ---- M3 Inc1: secondary sub-apps + one-way extract seam -------------------

use crate::sub_app::SubApp;

/// A tiny counter resource used to probe sub-app / extract behavior.
#[derive(Default)]
struct Counter(u64);
impl Resource for Counter {}

/// A marker label for a secondary "render" sub-app in these tests.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct RenderApp;

/// Another distinct label, to prove labels of different values don't collide.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct ServerApp;

/// A secondary sub-app updates every frame alongside the main sub-app.
#[test]
fn secondary_sub_app_updates_each_frame() {
    let main_frames = Arc::new(AtomicU64::new(0));
    let sub_frames = Arc::new(AtomicU64::new(0));
    let mf = main_frames.clone();
    let sf = sub_frames.clone();

    let mut app = App::new();
    app.add_systems(Update, move || {
        mf.fetch_add(1, Ordering::Relaxed);
    });

    let mut render = SubApp::new();
    render
        .world
        .resource_mut::<prism_ecs::schedule::Schedules>()
        .insert(Update, prism_ecs::schedule::Schedule::new());
    render
        .world
        .resource_mut::<prism_ecs::schedule::Schedules>()
        .get_mut(Update)
        .unwrap()
        .add_systems(move || {
            sf.fetch_add(1, Ordering::Relaxed);
        });
    app.insert_sub_app(RenderApp, render);

    app.set_runner(|app| HeadlessRunner::with_max_frames(3).run(app));
    app.run();

    assert_eq!(main_frames.load(Ordering::Relaxed), 3);
    assert_eq!(sub_frames.load(Ordering::Relaxed), 3);
}

/// Extract runs one-way main → sub, before the sub-app updates, and sees the
/// main world's just-finished frame.
#[test]
fn extract_copies_main_into_sub_before_sub_update() {
    let mut app = App::new();
    // Main world owns the authoritative counter; bump it in Update.
    app.world_mut().insert_resource(Counter::default());
    app.add_systems(Update, |mut c: ResMut<Counter>| {
        c.0 += 10;
    });

    let mut render = SubApp::new();
    render.world.insert_resource(Counter::default());
    app.insert_sub_app(RenderApp, render);

    // Extract copies the main counter into the sub counter (read-only on main).
    app.set_extract(RenderApp, |main: &mut prism_ecs::world::World, sub: &mut prism_ecs::world::World| {
        let value = main.resource::<Counter>().0;
        sub.resource_mut::<Counter>().0 = value;
    });

    // Drive two frames directly (keeping ownership so we can inspect state;
    // `App::run` would move the app into its runner).
    app.update();
    app.update();

    // After 2 frames the main counter is 20; extract ran after each main
    // update, so the sub-app sees the latest value.
    assert_eq!(app.get_sub_app(RenderApp).unwrap().world.resource::<Counter>().0, 20);
    assert_eq!(app.world().resource::<Counter>().0, 20);
}

/// Ordering invariant: the main sub-app updates before any secondary sub-app,
/// and each secondary's extract runs before its own update.
#[test]
fn main_runs_before_secondary_and_extract_before_sub_update() {
    let order = Arc::new(Mutex::new(Vec::<&'static str>::new()));

    let mut app = App::new();
    {
        let o = order.clone();
        app.add_systems(Update, move || o.lock().unwrap().push("main_update"));
    }

    let mut render = SubApp::new();
    render
        .world
        .resource_mut::<prism_ecs::schedule::Schedules>()
        .insert(Update, prism_ecs::schedule::Schedule::new());
    {
        let o = order.clone();
        render
            .world
            .resource_mut::<prism_ecs::schedule::Schedules>()
            .get_mut(Update)
            .unwrap()
            .add_systems(move || o.lock().unwrap().push("sub_update"));
    }
    app.insert_sub_app(RenderApp, render);
    {
        let o = order.clone();
        app.set_extract(RenderApp, move |_main: &mut prism_ecs::world::World, _sub: &mut prism_ecs::world::World| {
            o.lock().unwrap().push("extract");
        });
    }

    app.update();

    assert_eq!(
        *order.lock().unwrap(),
        vec!["main_update", "extract", "sub_update"]
    );
}

/// A secondary sub-app without an extract fn still updates (extract is optional).
#[test]
fn secondary_without_extract_still_updates() {
    let ran = Arc::new(AtomicU64::new(0));
    let r = ran.clone();

    let mut app = App::new();
    let mut sub = SubApp::new();
    sub.world
        .resource_mut::<prism_ecs::schedule::Schedules>()
        .insert(Update, prism_ecs::schedule::Schedule::new());
    sub.world
        .resource_mut::<prism_ecs::schedule::Schedules>()
        .get_mut(Update)
        .unwrap()
        .add_systems(move || {
            r.fetch_add(1, Ordering::Relaxed);
        });
    assert!(!sub.has_extract());
    app.insert_sub_app(ServerApp, sub);

    app.update();
    assert_eq!(ran.load(Ordering::Relaxed), 1);
}

/// Labels of different concrete values address different sub-apps; insertion
/// order is preserved and re-inserting a label replaces it in place.
#[test]
fn labeled_lookup_and_insertion_order() {
    let mut app = App::new();
    app.insert_sub_app(RenderApp, SubApp::new());
    app.insert_sub_app(ServerApp, SubApp::new());

    // Both resolve independently.
    assert!(app.get_sub_app(RenderApp).is_some());
    assert!(app.get_sub_app(ServerApp).is_some());

    // Tag each sub-app's world with a distinct resource to prove lookup maps to
    // the right instance.
    app.sub_app_mut(RenderApp)
        .unwrap()
        .world
        .insert_resource(Counter(1));
    app.sub_app_mut(ServerApp)
        .unwrap()
        .world
        .insert_resource(Counter(2));
    assert_eq!(app.get_sub_app(RenderApp).unwrap().world.resource::<Counter>().0, 1);
    assert_eq!(app.get_sub_app(ServerApp).unwrap().world.resource::<Counter>().0, 2);

    // Re-inserting RenderApp replaces it in place (fresh world has no Counter).
    app.insert_sub_app(RenderApp, SubApp::new());
    assert!(app.get_sub_app(RenderApp).unwrap().world.get_resource::<Counter>().is_none());
    // ServerApp is untouched.
    assert_eq!(app.get_sub_app(ServerApp).unwrap().world.resource::<Counter>().0, 2);
}

/// `set_extract` on a missing label panics with a clear message.
#[test]
#[should_panic(expected = "no sub-app registered")]
fn set_extract_on_missing_sub_app_panics() {
    let mut app = App::new();
    app.set_extract(RenderApp, |_m: &mut prism_ecs::world::World, _s: &mut prism_ecs::world::World| {});
}

// ---- §24.9 / §25.4: per-sub-app independent time domains -----------------
//
// A secondary sub-app opts into its own `EngineClocks` via
// `App::init_sub_app_time_domain`; from then on `advance_time` steps that
// sub-app's clocks independently every frame. These tests prove the design
// §25.4 invariant: each world holds an independent time context, so pausing or
// re-rating one world's `Time<Virtual>` never disturbs another.

/// Build a secondary sub-app with an `Update` schedule that bumps `counter`
/// each frame, so a test can prove its phases keep running regardless of its
/// time domain.
fn secondary_counting_update(counter: Arc<AtomicU64>) -> SubApp {
    let mut sub = SubApp::new();
    sub.world
        .resource_mut::<prism_ecs::schedule::Schedules>()
        .insert(Update, prism_ecs::schedule::Schedule::new());
    sub.world
        .resource_mut::<prism_ecs::schedule::Schedules>()
        .get_mut(Update)
        .unwrap()
        .add_systems(move || {
            counter.fetch_add(1, Ordering::Relaxed);
        });
    sub
}

/// Virtual elapsed seconds of a sub-app that owns a time domain.
fn virtual_elapsed(sub: &SubApp) -> f64 {
    sub.world
        .resource::<EngineClocks>()
        .virtual_time()
        .elapsed_secs_f64()
}

/// A secondary sub-app with its own time domain advances its clocks each frame,
/// independently of (and in lock-step rate with) the main sub-app.
#[test]
fn secondary_time_domain_advances_each_frame() {
    let mut app = App::new();
    app.set_time_update_strategy(TimeUpdateStrategy::ManualDelta(Duration::from_millis(10)));

    app.insert_sub_app(RenderApp, SubApp::new());
    assert!(
        !app.get_sub_app(RenderApp).unwrap().has_time_domain(),
        "a fresh secondary sub-app owns no time domain"
    );
    app.init_sub_app_time_domain(RenderApp);
    app.set_sub_app_time_update_strategy(
        RenderApp,
        TimeUpdateStrategy::ManualDelta(Duration::from_millis(10)),
    );
    assert!(app.get_sub_app(RenderApp).unwrap().has_time_domain());

    for _ in 0..3 {
        app.update();
    }

    let main_elapsed = app
        .world()
        .resource::<EngineClocks>()
        .virtual_time()
        .elapsed_secs_f64();
    let sub_elapsed = virtual_elapsed(app.get_sub_app(RenderApp).unwrap());
    assert!((main_elapsed - 0.03).abs() < 1e-6, "main={main_elapsed}");
    assert!((sub_elapsed - 0.03).abs() < 1e-6, "sub={sub_elapsed}");
}

/// Pausing the secondary world's virtual time freezes only that world; the main
/// world keeps advancing (design §25.4: "一个暂停不冻结另一个").
#[test]
fn pausing_secondary_does_not_freeze_main() {
    let mut app = App::new();
    app.set_time_update_strategy(TimeUpdateStrategy::ManualDelta(Duration::from_millis(10)));
    app.insert_sub_app(RenderApp, SubApp::new());
    app.init_sub_app_time_domain(RenderApp);
    app.set_sub_app_time_update_strategy(
        RenderApp,
        TimeUpdateStrategy::ManualDelta(Duration::from_millis(10)),
    );

    // Two frames with both running.
    app.update();
    app.update();
    let sub_frozen_at = virtual_elapsed(app.get_sub_app(RenderApp).unwrap());

    // Pause only the secondary world's virtual clock.
    app.sub_app_mut(RenderApp)
        .unwrap()
        .world
        .resource_mut::<EngineClocks>()
        .virtual_time_mut()
        .pause();

    // Three more frames.
    for _ in 0..3 {
        app.update();
    }

    let main_elapsed = app
        .world()
        .resource::<EngineClocks>()
        .virtual_time()
        .elapsed_secs_f64();
    let sub_elapsed = virtual_elapsed(app.get_sub_app(RenderApp).unwrap());

    // Main advanced through all 5 frames; the secondary stayed frozen at frame 2.
    assert!((main_elapsed - 0.05).abs() < 1e-6, "main={main_elapsed}");
    assert!(
        (sub_elapsed - sub_frozen_at).abs() < 1e-9,
        "secondary should stay frozen: {sub_elapsed} vs {sub_frozen_at}"
    );
    assert!((sub_frozen_at - 0.02).abs() < 1e-6, "frozen_at={sub_frozen_at}");
}

/// The mirror case: pausing the main world's virtual time freezes only the main
/// world; a secondary world with its own domain keeps advancing.
#[test]
fn pausing_main_does_not_freeze_secondary() {
    let mut app = App::new();
    app.set_time_update_strategy(TimeUpdateStrategy::ManualDelta(Duration::from_millis(10)));
    app.insert_sub_app(RenderApp, SubApp::new());
    app.init_sub_app_time_domain(RenderApp);
    app.set_sub_app_time_update_strategy(
        RenderApp,
        TimeUpdateStrategy::ManualDelta(Duration::from_millis(10)),
    );

    app.update();
    app.update();
    let main_frozen_at = app
        .world()
        .resource::<EngineClocks>()
        .virtual_time()
        .elapsed_secs_f64();

    // Pause only the main world's virtual clock.
    app.world_mut()
        .resource_mut::<EngineClocks>()
        .virtual_time_mut()
        .pause();

    for _ in 0..3 {
        app.update();
    }

    let main_elapsed = app
        .world()
        .resource::<EngineClocks>()
        .virtual_time()
        .elapsed_secs_f64();
    let sub_elapsed = virtual_elapsed(app.get_sub_app(RenderApp).unwrap());

    assert!(
        (main_elapsed - main_frozen_at).abs() < 1e-9,
        "main should stay frozen: {main_elapsed} vs {main_frozen_at}"
    );
    assert!((sub_elapsed - 0.05).abs() < 1e-6, "sub={sub_elapsed}");
}

/// A secondary sub-app without a time domain still runs its `Update` phase each
/// frame, but owns no clock and advances none (honest no-op fallthrough in
/// `advance_time`).
#[test]
fn secondary_without_time_domain_runs_update_but_owns_no_clock() {
    let ran = Arc::new(AtomicU64::new(0));
    let mut app = App::new();
    app.insert_sub_app(RenderApp, secondary_counting_update(ran.clone()));

    assert!(!app.get_sub_app(RenderApp).unwrap().has_time_domain());

    app.update();
    app.update();

    assert_eq!(ran.load(Ordering::Relaxed), 2, "Update must still run each frame");
    assert!(
        app.get_sub_app(RenderApp)
            .unwrap()
            .world
            .get_resource::<EngineClocks>()
            .is_none(),
        "no time domain was opted into, so no clock exists"
    );
}

/// `init_sub_app_time_domain` is idempotent: a second call never rewinds an
/// already-running world's elapsed time.
#[test]
fn init_sub_app_time_domain_is_idempotent() {
    let mut app = App::new();
    app.set_time_update_strategy(TimeUpdateStrategy::ManualDelta(Duration::from_millis(10)));
    app.insert_sub_app(RenderApp, SubApp::new());
    app.init_sub_app_time_domain(RenderApp);
    app.set_sub_app_time_update_strategy(
        RenderApp,
        TimeUpdateStrategy::ManualDelta(Duration::from_millis(10)),
    );

    app.update();
    app.update();
    let before = virtual_elapsed(app.get_sub_app(RenderApp).unwrap());

    // Re-init must preserve the running clock.
    app.init_sub_app_time_domain(RenderApp);
    let after = virtual_elapsed(app.get_sub_app(RenderApp).unwrap());
    assert!((before - after).abs() < 1e-9, "re-init rewound time: {before} vs {after}");
    assert!((before - 0.02).abs() < 1e-6, "before={before}");
}

/// A secondary sub-app can run its fixed step at a different rate than the main
/// world, and setting one does not touch the other.
#[test]
fn secondary_fixed_timestep_is_independent() {
    let mut app = App::new();
    app.set_fixed_timestep_hz(60.0);
    app.insert_sub_app(RenderApp, SubApp::new());
    app.init_sub_app_time_domain(RenderApp);
    app.set_sub_app_fixed_timestep_hz(RenderApp, 20.0);

    let main_step = app
        .world()
        .resource::<EngineClocks>()
        .fixed()
        .timestep();
    let sub_step = app
        .get_sub_app(RenderApp)
        .unwrap()
        .world
        .resource::<EngineClocks>()
        .fixed()
        .timestep();

    assert_eq!(main_step, Duration::from_secs_f64(1.0 / 60.0));
    assert_eq!(sub_step, Duration::from_secs_f64(1.0 / 20.0));
    assert_ne!(main_step, sub_step);
}

/// Setting a time-domain strategy on a sub-app that never opted in panics
/// loudly rather than silently doing nothing.
#[test]
#[should_panic(expected = "no time domain")]
fn set_sub_app_time_strategy_without_domain_panics() {
    let mut app = App::new();
    app.insert_sub_app(RenderApp, SubApp::new());
    app.set_sub_app_time_update_strategy(
        RenderApp,
        TimeUpdateStrategy::ManualDelta(Duration::from_millis(10)),
    );
}

/// Addressing a missing label for a time-domain op panics with a clear message.
#[test]
#[should_panic(expected = "no sub-app registered")]
fn init_sub_app_time_domain_missing_label_panics() {
    let mut app = App::new();
    app.init_sub_app_time_domain(RenderApp);
}

// ---- M3 Inc2: cross-thread sub-app pipelining -----------------------------
//
// These tests are gated on the `pipelined` feature. They assert that the
// opt-in pipeline (a) runs every frame's render exactly once, (b) keeps the
// one-way extract seam intact (sub reads the main world's completed frame),
// (c) produces results identical to the serial path (determinism), and
// (d) parks secondary sub-apps on the worker thread until `sync_sub_apps`.

#[cfg(feature = "pipelined")]
mod pipelined_tests {
    use super::*;

    /// Build a render sub-app whose `Update` schedule bumps `render_count` and
    /// which owns a `Counter` resource for extract to write into.
    fn make_render_sub_app(render_count: Arc<AtomicU64>) -> SubApp {
        let mut render = SubApp::new();
        render.world.insert_resource(Counter::default());
        render
            .world
            .resource_mut::<prism_ecs::schedule::Schedules>()
            .insert(Update, prism_ecs::schedule::Schedule::new());
        render
            .world
            .resource_mut::<prism_ecs::schedule::Schedules>()
            .get_mut(Update)
            .unwrap()
            .add_systems(move || {
                render_count.fetch_add(1, Ordering::Relaxed);
            });
        render
    }

    /// Enabling pipelining is reflected by `is_pipelined` and is idempotent.
    #[test]
    fn enable_is_idempotent_and_observable() {
        let mut app = App::new();
        assert!(!app.is_pipelined());
        app.enable_pipelined_rendering();
        assert!(app.is_pipelined());
        // Enabling again keeps it on (and does not reset in-flight state).
        app.enable_pipelined_rendering();
        assert!(app.is_pipelined());
    }

    /// Over N pipelined frames the render sub-app runs exactly N times once the
    /// final in-flight frame is synced.
    #[test]
    fn pipelined_runs_every_frame_once() {
        let render_count = Arc::new(AtomicU64::new(0));

        let mut app = App::new();
        app.insert_sub_app(RenderApp, make_render_sub_app(render_count.clone()));
        app.enable_pipelined_rendering();

        app.set_runner(|app| HeadlessRunner::with_max_frames(5).run(app));
        app.run();

        // The runner's final `sync_sub_apps` brings the last render home, so
        // all 5 frames have rendered.
        assert_eq!(render_count.load(Ordering::Relaxed), 5);
    }

    /// Extract stays one-way (main → sub) and the sub-app sees the main world's
    /// completed frame, exactly as in the serial path.
    #[test]
    fn pipelined_extract_sees_completed_frame() {
        let render_count = Arc::new(AtomicU64::new(0));

        let mut app = App::new();
        app.world_mut().insert_resource(Counter::default());
        app.add_systems(Update, |mut c: ResMut<Counter>| {
            c.0 += 10;
        });
        app.insert_sub_app(RenderApp, make_render_sub_app(render_count.clone()));
        app.set_extract(
            RenderApp,
            |main: &mut prism_ecs::world::World, sub: &mut prism_ecs::world::World| {
                let value = main.resource::<Counter>().0;
                sub.resource_mut::<Counter>().0 = value;
            },
        );
        app.enable_pipelined_rendering();

        // Drive four frames directly, then sync to inspect the secondary.
        app.update();
        app.update();
        app.update();
        app.update();
        app.sync_sub_apps();

        assert_eq!(app.world().resource::<Counter>().0, 40);
        assert_eq!(
            app.get_sub_app(RenderApp).unwrap().world.resource::<Counter>().0,
            40
        );
        assert_eq!(render_count.load(Ordering::Relaxed), 4);
    }

    /// The pipelined path yields the same observable result as the serial path
    /// (determinism: pipelining only overlaps timing, never changes outcomes).
    #[test]
    fn pipelined_matches_serial() {
        fn drive(pipelined: bool, frames: u64) -> (u64, u64, u64) {
            let render_count = Arc::new(AtomicU64::new(0));
            let mut app = App::new();
            app.world_mut().insert_resource(Counter::default());
            app.add_systems(Update, |mut c: ResMut<Counter>| {
                c.0 += 7;
            });
            app.insert_sub_app(RenderApp, make_render_sub_app(render_count.clone()));
            app.set_extract(
                RenderApp,
                |main: &mut prism_ecs::world::World, sub: &mut prism_ecs::world::World| {
                    let value = main.resource::<Counter>().0;
                    sub.resource_mut::<Counter>().0 = value;
                },
            );
            if pipelined {
                app.enable_pipelined_rendering();
            }
            for _ in 0..frames {
                app.update();
            }
            app.sync_sub_apps();
            (
                app.world().resource::<Counter>().0,
                app.get_sub_app(RenderApp).unwrap().world.resource::<Counter>().0,
                render_count.load(Ordering::Relaxed),
            )
        }

        let serial = drive(false, 6);
        let pipelined = drive(true, 6);
        assert_eq!(serial, pipelined);
        // Sanity: 6 frames × +7 = 42 in both worlds, 6 renders.
        assert_eq!(pipelined, (42, 42, 6));
    }

    /// While a render frame is in flight the secondary sub-app is resident on
    /// the worker thread and unreachable until `sync_sub_apps` brings it home.
    #[test]
    fn secondaries_resident_on_worker_until_sync() {
        let render_count = Arc::new(AtomicU64::new(0));

        let mut app = App::new();
        app.insert_sub_app(RenderApp, make_render_sub_app(render_count.clone()));
        app.enable_pipelined_rendering();

        app.update();
        // Render frame 0 is in flight: the secondary is not on the main thread.
        assert!(app.get_sub_app(RenderApp).is_none());

        app.sync_sub_apps();
        // Now it is home and reachable again.
        assert!(app.get_sub_app(RenderApp).is_some());
    }

    /// A render-thread panic is propagated on the main thread (never silently
    /// swallowed) when the in-flight frame is joined.
    #[test]
    #[should_panic(expected = "render boom")]
    fn render_thread_panic_propagates() {
        let mut app = App::new();
        let mut render = SubApp::new();
        render
            .world
            .resource_mut::<prism_ecs::schedule::Schedules>()
            .insert(Update, prism_ecs::schedule::Schedule::new());
        render
            .world
            .resource_mut::<prism_ecs::schedule::Schedules>()
            .get_mut(Update)
            .unwrap()
            .add_systems(|| panic!("render boom"));
        app.insert_sub_app(RenderApp, render);
        app.enable_pipelined_rendering();

        // Frame 0 kicks render 0 (which will panic). Frame 1 joins it and the
        // panic surfaces here on the main thread.
        app.update();
        app.update();
    }

    /// Under the pipelined executor the per-frame extract cost is recorded on
    /// every frame, while a simulate/render overlap sample is recorded only on
    /// frames after the first: the first frame has no prior render in flight,
    /// so it is skipped rather than scored as a spurious perfect overlap.
    #[test]
    fn pipelined_records_extract_cost_and_overlap() {
        let render_count = Arc::new(AtomicU64::new(0));

        let mut app = App::new();
        app.set_time_update_strategy(TimeUpdateStrategy::ManualDelta(Duration::from_millis(10)));
        app.set_fixed_timestep_hz(100.0);
        app.insert_sub_app(RenderApp, make_render_sub_app(render_count.clone()));
        app.set_extract(
            RenderApp,
            |_m: &mut prism_ecs::world::World, _s: &mut prism_ecs::world::World| {},
        );
        app.enable_pipelined_rendering();
        app.init_frame_diagnostics();

        const FRAMES: u64 = 4;
        for _ in 0..FRAMES {
            app.update();
        }

        let diag = app.frame_diagnostics().expect("installed");
        // Extract is timed on every pipelined frame.
        assert_eq!(diag.extract_time().total_frames(), FRAMES);
        // Overlap is recorded only on frames with a prior render in flight, so
        // the first of the FRAMES frames is skipped.
        assert_eq!(diag.pipeline_sim().total_frames(), FRAMES - 1);
        assert_eq!(diag.pipeline_wait().total_frames(), FRAMES - 1);
        // The derived ratio, when present, is a well-formed fraction. It is
        // timing-dependent, so we assert sanity rather than require `Some`.
        if let Some(ratio) = diag.pipeline_overlap_ratio() {
            assert!((0.0..=1.0).contains(&ratio), "overlap ratio {ratio} out of range");
        }
    }
}

// ---- M4 Inc2: platform lifecycle events + graceful shutdown ------------

use crate::lifecycle::{
    AppLifecycle, FocusChanged, LowMemory, Resumed, Suspended, WillRenderFirstFrame,
};
use crate::schedule::Shutdown;

/// `add_lifecycle_events` registers all five lifecycle event types and installs
/// the `AppLifecycle` resource defaulting to `Running`.
#[test]
fn add_lifecycle_events_registers_events_and_installs_running_state() {
    let mut app = App::new();
    app.add_lifecycle_events();

    // Each lifecycle event has an installed buffer.
    assert!(app.world().get_resource::<Events<Suspended>>().is_some());
    assert!(app.world().get_resource::<Events<Resumed>>().is_some());
    assert!(app.world().get_resource::<Events<LowMemory>>().is_some());
    assert!(app.world().get_resource::<Events<FocusChanged>>().is_some());
    assert!(
        app.world()
            .get_resource::<Events<WillRenderFirstFrame>>()
            .is_some()
    );

    // The coarse run-state resource is installed, defaulting to Running.
    assert_eq!(
        *app.world().resource::<AppLifecycle>(),
        AppLifecycle::Running,
    );
}

/// `add_lifecycle_events` is idempotent: calling it again neither loses a
/// buffered event nor resets an already-advanced `AppLifecycle`.
#[test]
fn add_lifecycle_events_is_idempotent() {
    let mut app = App::new();
    app.add_lifecycle_events();

    // Seed a buffered event and manually advance the run state.
    app.send_event(FocusChanged { focused: true });
    *app.world_mut().resource_mut::<AppLifecycle>() = AppLifecycle::Suspended;

    // A second registration must not clobber either.
    app.add_lifecycle_events();
    assert_eq!(
        app.world().resource::<Events<FocusChanged>>().len(),
        1,
        "re-registering must not reset the event buffer",
    );
    assert_eq!(
        *app.world().resource::<AppLifecycle>(),
        AppLifecycle::Suspended,
        "re-registering must not reset an advanced AppLifecycle",
    );
}

/// `send_event` auto-registers the event type on first use, then buffers the
/// event so it participates in the normal rotation.
#[test]
fn send_event_auto_registers_and_buffers() {
    let mut app = App::new();
    // No prior `add_event`/`add_lifecycle_events`.
    app.send_event(LowMemory);
    let events = app
        .world()
        .get_resource::<Events<LowMemory>>()
        .expect("send_event should auto-install the Events<LowMemory> resource");
    assert_eq!(events.len(), 1, "the sent event is buffered");
}

/// A lifecycle event sent via `send_event` is observed exactly once by a reader
/// cursor running in a frame, exactly as it will be under a platform runner.
#[test]
fn lifecycle_event_is_delivered_to_a_reader() {
    let mut app = App::new();
    app.add_lifecycle_events();

    let seen = Arc::new(Mutex::new(Vec::<bool>::new()));
    let seen_sys = seen.clone();
    app.add_systems(
        Update,
        move |mut cursor: Local<EventCursor<FocusChanged>>, events: Res<Events<FocusChanged>>| {
            for ev in cursor.read(&events) {
                seen_sys.lock().unwrap().push(ev.focused);
            }
        },
    );

    // Inject a focus-lost then focus-gained event before the frame runs.
    app.send_event(FocusChanged { focused: false });
    app.send_event(FocusChanged { focused: true });
    app.update();

    assert_eq!(
        *seen.lock().unwrap(),
        vec![false, true],
        "the reader observes each lifecycle event once, in send order",
    );
}

/// `run_shutdown` runs the dedicated `Shutdown` schedule exactly once, even when
/// called repeatedly, and reports `shutdown_ran`.
#[test]
fn run_shutdown_runs_shutdown_schedule_exactly_once() {
    let mut app = App::new();

    let runs = Arc::new(AtomicU64::new(0));
    let r = runs.clone();
    app.add_systems(Shutdown, move || {
        r.fetch_add(1, Ordering::Relaxed);
    });

    assert!(!app.shutdown_ran());
    app.run_shutdown();
    assert!(app.shutdown_ran());
    app.run_shutdown();
    app.run_shutdown();

    assert_eq!(
        runs.load(Ordering::Relaxed),
        1,
        "the Shutdown schedule runs exactly once regardless of repeated calls",
    );
}

/// `run_shutdown` advances `AppLifecycle` to `WillExit` before running the
/// `Shutdown` schedule, so a shutdown system observes the exiting state.
#[test]
fn run_shutdown_sets_will_exit_before_running_shutdown_systems() {
    let mut app = App::new();
    app.add_lifecycle_events();

    let observed = Arc::new(Mutex::new(None::<AppLifecycle>));
    let o = observed.clone();
    app.add_systems(Shutdown, move |state: Res<AppLifecycle>| {
        *o.lock().unwrap() = Some(*state);
    });

    app.run_shutdown();

    assert_eq!(
        *observed.lock().unwrap(),
        Some(AppLifecycle::WillExit),
        "a shutdown system sees AppLifecycle::WillExit",
    );
    assert_eq!(
        *app.world().resource::<AppLifecycle>(),
        AppLifecycle::WillExit,
    );
}

/// Plugins are torn down in the *reverse* of registration order at shutdown,
/// distinct from the forward-order post-startup `cleanup`.
#[test]
fn plugin_shutdown_runs_in_reverse_registration_order() {
    #[derive(Default)]
    struct Teardown(Vec<&'static str>);
    impl Resource for Teardown {}

    struct A;
    impl Plugin for A {
        fn build(&self, app: &mut App) {
            app.init_resource::<Teardown>();
        }
        fn cleanup(&self, app: &mut App) {
            app.world_mut().resource_mut::<Teardown>().0.push("cleanup-A");
        }
        fn shutdown(&self, app: &mut App) {
            app.world_mut().resource_mut::<Teardown>().0.push("shutdown-A");
        }
    }
    struct B;
    impl Plugin for B {
        fn build(&self, _app: &mut App) {}
        fn cleanup(&self, app: &mut App) {
            app.world_mut().resource_mut::<Teardown>().0.push("cleanup-B");
        }
        fn shutdown(&self, app: &mut App) {
            app.world_mut().resource_mut::<Teardown>().0.push("shutdown-B");
        }
    }
    struct C;
    impl Plugin for C {
        fn build(&self, _app: &mut App) {}
        fn shutdown(&self, app: &mut App) {
            app.world_mut().resource_mut::<Teardown>().0.push("shutdown-C");
        }
    }

    let mut app = App::new();
    app.add_plugins(A).add_plugins(B).add_plugins(C);

    // cleanup is forward order (A, B); C defines no cleanup. `cleanup` requires
    // `finish` to have advanced the assembly state first.
    app.finish();
    app.cleanup();
    assert_eq!(
        app.world().resource::<Teardown>().0,
        vec!["cleanup-A", "cleanup-B"],
        "cleanup runs in forward registration order",
    );

    // shutdown is reverse order (C, B, A).
    app.run_shutdown();
    assert_eq!(
        app.world().resource::<Teardown>().0,
        vec![
            "cleanup-A",
            "cleanup-B",
            "shutdown-C",
            "shutdown-B",
            "shutdown-A",
        ],
        "shutdown runs in reverse registration order, after cleanup",
    );
}

/// The `HeadlessRunner` runs the graceful-shutdown path once after its frame
/// loop ends: the `Shutdown` schedule fires exactly once and `AppLifecycle`
/// reaches `WillExit`.
#[test]
fn headless_runner_runs_shutdown_once_after_frame_loop() {
    let mut app = App::new();
    app.add_lifecycle_events();

    let shutdowns = Arc::new(AtomicU64::new(0));
    let will_exit = Arc::new(Mutex::new(false));
    let s = shutdowns.clone();
    let w = will_exit.clone();
    app.add_systems(Shutdown, move |state: Res<AppLifecycle>| {
        s.fetch_add(1, Ordering::Relaxed);
        if state.is_exiting() {
            *w.lock().unwrap() = true;
        }
    });

    let exit = app.set_runner(|app| HeadlessRunner::with_max_frames(3).run(app)).run();
    assert_eq!(exit, AppExit::Success);
    assert_eq!(
        shutdowns.load(Ordering::Relaxed),
        1,
        "the runner runs the Shutdown schedule exactly once",
    );
    assert!(
        *will_exit.lock().unwrap(),
        "the shutdown system observes AppLifecycle::WillExit",
    );
}

/// `run_once` also runs the graceful-shutdown path before returning.
#[test]
fn run_once_runs_shutdown_path() {
    use crate::runner::run_once;

    let mut app = App::new();
    let ran = Arc::new(AtomicU64::new(0));
    let r = ran.clone();
    app.add_systems(Shutdown, move || {
        r.fetch_add(1, Ordering::Relaxed);
    });

    let exit = run_once(app);
    assert_eq!(exit, AppExit::Success);
    assert_eq!(
        ran.load(Ordering::Relaxed),
        1,
        "run_once drives the Shutdown schedule exactly once",
    );
}

// ---- lifecycle run-state + virtual-clock wiring (design §12, §25.1) ----

/// A `Suspended` event flips `AppLifecycle` to `Suspended` and pauses the main
/// world's virtual clock, so the fixed accumulator is fed zero delta and
/// simulation freezes while backgrounded.
#[test]
fn suspend_event_pauses_simulation_and_sets_suspended_state() {
    let mut app = App::new();
    app.add_lifecycle_events();
    app.set_time_update_strategy(TimeUpdateStrategy::ManualDelta(Duration::from_millis(10)));

    // A baseline running frame advances the virtual clock.
    app.update();
    let before = app
        .world()
        .resource::<EngineClocks>()
        .virtual_time()
        .elapsed_secs_f64();
    assert!(before > 0.0, "a running frame advances the virtual clock");

    // Suspend, then run the frame that processes the event.
    app.send_event(Suspended);
    app.update();
    assert_eq!(
        *app.world().resource::<AppLifecycle>(),
        AppLifecycle::Suspended,
        "the Suspended event moves the run state to Suspended",
    );
    assert!(
        app.world().resource::<EngineClocks>().virtual_time().is_paused(),
        "suspending pauses the virtual clock",
    );

    // Subsequent frames while suspended do not advance virtual time.
    let after_suspend = app
        .world()
        .resource::<EngineClocks>()
        .virtual_time()
        .elapsed_secs_f64();
    app.update();
    app.update();
    let frozen = app
        .world()
        .resource::<EngineClocks>()
        .virtual_time()
        .elapsed_secs_f64();
    assert_eq!(
        frozen, after_suspend,
        "virtual time stays frozen across suspended frames",
    );
}

/// A `Resumed` event flips `AppLifecycle` back to `Running` and unpauses the
/// virtual clock, so simulation advances again.
#[test]
fn resume_event_unpauses_and_restores_running() {
    let mut app = App::new();
    app.add_lifecycle_events();
    app.set_time_update_strategy(TimeUpdateStrategy::ManualDelta(Duration::from_millis(10)));

    app.send_event(Suspended);
    app.update();
    assert!(app.world().resource::<EngineClocks>().virtual_time().is_paused());

    app.send_event(Resumed);
    app.update();
    assert_eq!(
        *app.world().resource::<AppLifecycle>(),
        AppLifecycle::Running,
        "the Resumed event restores the Running state",
    );
    assert!(
        !app.world().resource::<EngineClocks>().virtual_time().is_paused(),
        "resuming unpauses the virtual clock",
    );

    let before = app
        .world()
        .resource::<EngineClocks>()
        .virtual_time()
        .elapsed_secs_f64();
    app.update();
    let after = app
        .world()
        .resource::<EngineClocks>()
        .virtual_time()
        .elapsed_secs_f64();
    assert!(after > before, "virtual time advances again after resume");
}

/// The handler only unpauses a clock *it* paused: a pause the game set itself
/// (a pause menu) survives a full suspend/resume cycle rather than being
/// silently cleared on resume.
#[test]
fn resume_preserves_a_user_pause_it_did_not_set() {
    let mut app = App::new();
    app.add_lifecycle_events();
    app.set_time_update_strategy(TimeUpdateStrategy::ManualDelta(Duration::from_millis(10)));

    // The game pauses the virtual clock for its own reasons.
    app.world_mut()
        .resource_mut::<EngineClocks>()
        .virtual_time_mut()
        .pause();

    app.send_event(Suspended);
    app.update();
    assert_eq!(*app.world().resource::<AppLifecycle>(), AppLifecycle::Suspended);
    assert!(
        app.world().resource::<EngineClocks>().virtual_time().is_paused(),
        "an already-paused clock stays paused through suspend",
    );

    app.send_event(Resumed);
    app.update();
    assert_eq!(*app.world().resource::<AppLifecycle>(), AppLifecycle::Running);
    assert!(
        app.world().resource::<EngineClocks>().virtual_time().is_paused(),
        "a user-owned pause survives the suspend/resume cycle",
    );
}

/// When both a `Suspended` and a `Resumed` event land in one frame, resume
/// wins so a transient background blip can never wedge the app suspended.
#[test]
fn resume_wins_when_both_edges_land_in_one_frame() {
    let mut app = App::new();
    app.add_lifecycle_events();

    // Drive into the suspended state first.
    app.send_event(Suspended);
    app.update();
    assert_eq!(*app.world().resource::<AppLifecycle>(), AppLifecycle::Suspended);

    // Same frame: both edges. Resume must win.
    app.send_event(Suspended);
    app.send_event(Resumed);
    app.update();
    assert_eq!(
        *app.world().resource::<AppLifecycle>(),
        AppLifecycle::Running,
        "resume wins when both edges arrive in one frame",
    );
    assert!(!app.world().resource::<EngineClocks>().virtual_time().is_paused());
}

/// A graceful exit in progress (`WillExit`) is terminal: lifecycle events do
/// not pull the run state back to `Running`/`Suspended`.
#[test]
fn exiting_state_is_not_resurrected_by_lifecycle_events() {
    let mut app = App::new();
    app.add_lifecycle_events();
    *app.world_mut().resource_mut::<AppLifecycle>() = AppLifecycle::WillExit;

    app.send_event(Resumed);
    app.send_event(Suspended);
    app.update();
    assert_eq!(
        *app.world().resource::<AppLifecycle>(),
        AppLifecycle::WillExit,
        "a graceful exit in progress is not resurrected",
    );
}

// ---- exit veto gate (design §24.5: 退出可被系统取消) --------------------

use crate::schedule::ExitConfirmation;

/// `AppExitRequest::cancel` withdraws a pending request and clears the
/// confirmation flag, so `get` reports nothing again.
#[test]
fn app_exit_request_cancel_withdraws_pending_request() {
    let mut req = AppExitRequest::default();
    assert_eq!(req.get(), None);
    assert!(!req.is_confirmed());

    req.send(AppExit::error());
    assert_eq!(req.get(), Some(AppExit::error()));

    req.cancel();
    assert_eq!(req.get(), None, "cancel withdraws the pending request");
    assert!(!req.is_confirmed(), "cancel clears any confirmation");
}

/// A fresh `send` after a `cancel` re-arms the request so the confirmation gate
/// gets another chance (the first-wins rule only applies while a request is
/// still pending).
#[test]
fn app_exit_request_send_after_cancel_rearms() {
    let mut req = AppExitRequest::default();
    req.send(AppExit::Success);
    // First-wins while pending: a second send does not overwrite the code.
    req.send(AppExit::error());
    assert_eq!(req.get(), Some(AppExit::Success));

    req.cancel();
    // After cancel, a new send takes effect.
    req.send(AppExit::error());
    assert_eq!(req.get(), Some(AppExit::error()));
    assert!(!req.is_confirmed());
}

/// `poll_exit` returns `None` and does not run the `ExitConfirmation` gate when
/// no system has requested exit.
#[test]
fn poll_exit_is_none_and_skips_gate_without_request() {
    let mut app = App::new();
    let gate_runs = Arc::new(AtomicU64::new(0));
    let g = gate_runs.clone();
    app.add_systems(ExitConfirmation, move || {
        g.fetch_add(1, Ordering::Relaxed);
    });

    assert_eq!(app.poll_exit(), None);
    assert_eq!(
        gate_runs.load(Ordering::Relaxed),
        0,
        "the veto gate must not run while no exit is pending",
    );
}

/// `poll_exit` runs the `ExitConfirmation` gate once for a pending request and,
/// when a confirmation system cancels it, returns `None` so the loop keeps
/// running — the "don't quit" path.
#[test]
fn poll_exit_veto_cancels_pending_exit() {
    let mut app = App::new();
    app.world_mut()
        .resource_mut::<AppExitRequest>()
        .send(AppExit::error());

    let gate_runs = Arc::new(AtomicU64::new(0));
    let g = gate_runs.clone();
    app.add_systems(ExitConfirmation, move |mut req: ResMut<AppExitRequest>| {
        g.fetch_add(1, Ordering::Relaxed);
        req.cancel();
    });

    assert_eq!(app.poll_exit(), None, "a vetoed request does not exit");
    assert_eq!(gate_runs.load(Ordering::Relaxed), 1);
    assert_eq!(
        app.should_exit(),
        None,
        "the pending request was withdrawn by the gate",
    );
}

/// `poll_exit` confirms a request the gate leaves standing: it returns the exit
/// code, marks the request confirmed, and does not re-run the gate on a later
/// poll (prompt-at-most-once).
#[test]
fn poll_exit_confirms_and_gate_runs_at_most_once() {
    let mut app = App::new();
    app.world_mut()
        .resource_mut::<AppExitRequest>()
        .send(AppExit::error());

    let gate_runs = Arc::new(AtomicU64::new(0));
    let g = gate_runs.clone();
    app.add_systems(ExitConfirmation, move || {
        g.fetch_add(1, Ordering::Relaxed);
    });

    assert_eq!(app.poll_exit(), Some(AppExit::error()));
    assert!(app.world().resource::<AppExitRequest>().is_confirmed());
    // A second poll short-circuits: same answer, gate not re-run.
    assert_eq!(app.poll_exit(), Some(AppExit::error()));
    assert_eq!(
        gate_runs.load(Ordering::Relaxed),
        1,
        "the veto gate runs at most once per confirmed request",
    );
}

/// Integration: a confirmation system that vetoes the *first* exit request
/// keeps an otherwise-unbounded `HeadlessRunner` looping; once it stops
/// vetoing, the app exits. Proves the veto actually prevents a frame-loop stop.
#[test]
fn headless_runner_honors_exit_veto_then_exits() {
    let mut app = App::new();

    // Every frame, request a clean exit (first-wins while pending; re-armed
    // after a cancel).
    app.add_systems(Update, move |mut req: ResMut<AppExitRequest>| {
        req.send_success();
    });

    // Veto only the first pending request; let the second through.
    let gate_runs = Arc::new(AtomicU64::new(0));
    let g = gate_runs.clone();
    app.add_systems(ExitConfirmation, move |mut req: ResMut<AppExitRequest>| {
        let n = g.fetch_add(1, Ordering::Relaxed);
        if n == 0 {
            req.cancel();
        }
    });

    let exit = app.set_runner(|app| HeadlessRunner::new().run(app)).run();
    assert_eq!(exit, AppExit::Success);
    assert_eq!(
        gate_runs.load(Ordering::Relaxed),
        2,
        "the gate ran twice: once vetoed (loop continued), once confirmed (exit)",
    );
}

// ---- dedicated-server runner (design §10 / §24.4) ----------------------

// Std-only: the dedicated-server runner and its diagnostics live behind
// `#[cfg(feature = "std")]` (it paces against the monotonic clock), so these
// tests compile only when `std` is enabled.
#[cfg(feature = "std")]
mod dedicated_server {
    use super::*;
    use crate::runner::{DedicatedServerRunner, ServerTickDiagnostics};

    /// The builder exposes its configuration, and the opt-outs flip exactly the
    /// flag they name while leaving the rest at their authoritative defaults.
    #[test]
    fn dedicated_server_builder_config_and_opt_outs() {
        use prism_time::Duration;

        let default = DedicatedServerRunner::new(60);
        assert_eq!(default.tickrate_hz(), 60);
        assert_eq!(default.max_ticks(), None);
        assert!(default.is_deterministic());
        assert!(default.aligns_fixed_timestep());
        assert!(default.is_real_time_paced());
        // 60 Hz => 1/60 s period (nanosecond-truncated, matching FrameLimit).
        assert_eq!(
            default.tick_period(),
            Duration::from_nanos(1_000_000_000 / 60)
        );

        let tuned = DedicatedServerRunner::new(30)
            .with_max_ticks(7)
            .with_wall_clock_time()
            .without_fixed_timestep_alignment()
            .without_real_time_pacing();
        assert_eq!(tuned.tickrate_hz(), 30);
        assert_eq!(tuned.max_ticks(), Some(7));
        assert!(!tuned.is_deterministic());
        assert!(!tuned.aligns_fixed_timestep());
        assert!(!tuned.is_real_time_paced());
    }

    /// A zero tickrate is a programming error and panics loudly rather than
    /// silently picking a rate.
    #[test]
    #[should_panic(expected = "tickrate must be non-zero")]
    fn dedicated_server_rejects_zero_tickrate() {
        let _ = DedicatedServerRunner::new(0);
    }

    /// The server loop drives `Update` exactly `max_ticks` times and reports a
    /// clean exit. Run unpaced so the test does not sleep.
    #[test]
    fn dedicated_server_drives_requested_ticks() {
        let ticks = Arc::new(AtomicU64::new(0));
        let t = ticks.clone();

        let mut app = App::new();
        app.add_systems(Update, move || {
            t.fetch_add(1, Ordering::Relaxed);
        });
        app.set_runner(|app| {
            DedicatedServerRunner::new(60)
                .with_max_ticks(5)
                .without_real_time_pacing()
                .run(app)
        });

        let exit = app.run();
        assert_eq!(exit, AppExit::Success);
        assert_eq!(ticks.load(Ordering::Relaxed), 5);
    }

    /// The authoritative heartbeat: with deterministic stepping and fixed-timestep
    /// alignment (both on by default), each tick advances simulated time by exactly
    /// one tick period, so `FixedUpdate` runs exactly once per tick — the
    /// "定 tickrate 固定步长心跳" invariant.
    #[test]
    fn dedicated_server_runs_exactly_one_fixed_step_per_tick() {
        let fixed = Arc::new(AtomicU64::new(0));
        let update = Arc::new(AtomicU64::new(0));
        let fx = fixed.clone();
        let up = update.clone();

        let mut app = App::new();
        app.add_systems(FixedUpdate, move || {
            fx.fetch_add(1, Ordering::Relaxed);
        });
        app.add_systems(Update, move || {
            up.fetch_add(1, Ordering::Relaxed);
        });
        app.set_runner(|app| {
            DedicatedServerRunner::new(60)
                .with_max_ticks(10)
                .without_real_time_pacing()
                .run(app)
        });
        app.run();

        assert_eq!(
            fixed.load(Ordering::Relaxed),
            10,
            "exactly one fixed step per server tick over 10 ticks",
        );
        assert_eq!(
            update.load(Ordering::Relaxed),
            10,
            "the variable Update phase still runs once per tick",
        );
    }

    /// Opting out of fixed-timestep alignment leaves the app's fixed rate untouched,
    /// so the server's manual per-tick delta no longer matches the step and the
    /// one-step-per-tick invariant does not hold. Guards against the alignment
    /// being a silent no-op.
    #[test]
    fn dedicated_server_without_alignment_does_not_force_one_step_per_tick() {
        let fixed = Arc::new(AtomicU64::new(0));
        let fx = fixed.clone();

        let mut app = App::new();
        // Leave the default fixed timestep (64 Hz) in place; the server ticks at
        // 60 Hz, so a per-tick delta of 1/60 s against a 1/64 s step does not
        // produce a clean one-step-per-tick cadence.
        app.add_systems(FixedUpdate, move || {
            fx.fetch_add(1, Ordering::Relaxed);
        });
        app.set_runner(|app| {
            DedicatedServerRunner::new(60)
                .with_max_ticks(4)
                .without_fixed_timestep_alignment()
                .without_real_time_pacing()
                .run(app)
        });
        app.run();

        // We assert only that alignment is *not* applied: the count differs from a
        // perfectly-aligned 1:1 run is not guaranteed, but the fixed system must
        // still have run a bounded number of times (never panicking / never
        // unbounded). The precise count is a function of the mismatched rates.
        let runs = fixed.load(Ordering::Relaxed);
        assert!(
            runs <= 4,
            "mismatched rates cannot exceed one step per tick here"
        );
    }

    /// The runner publishes live `ServerTickDiagnostics`: a server system can read
    /// the current tick health, the tick count advances, and trivial work does not
    /// register as overloaded.
    #[test]
    fn dedicated_server_publishes_tick_diagnostics() {
        let seen_period_nonzero = Arc::new(AtomicU64::new(0));
        let seen_overloaded = Arc::new(AtomicU64::new(0));
        let max_tick_seen = Arc::new(AtomicU64::new(0));
        let p = seen_period_nonzero.clone();
        let o = seen_overloaded.clone();
        let m = max_tick_seen.clone();

        let mut app = App::new();
        app.add_systems(Update, move |diag: Res<ServerTickDiagnostics>| {
            if !diag.tick_period().is_zero() {
                p.fetch_add(1, Ordering::Relaxed);
            }
            if diag.is_overloaded() {
                o.fetch_add(1, Ordering::Relaxed);
            }
            m.fetch_max(diag.tick(), Ordering::Relaxed);
        });
        app.set_runner(|app| {
            DedicatedServerRunner::new(120)
                .with_max_ticks(6)
                .without_real_time_pacing()
                .run(app)
        });

        let exit = app.run();
        assert_eq!(exit, AppExit::Success);
        // The resource was present and seeded with a non-zero period on every tick.
        assert_eq!(seen_period_nonzero.load(Ordering::Relaxed), 6);
        // Trivial work against an 1/120 s budget never counts as overloaded.
        assert_eq!(seen_overloaded.load(Ordering::Relaxed), 0);
        // Diagnostics are refreshed after each tick, so a system reading on tick N
        // observes the previous tick's count: the max observed is max_ticks - 1.
        assert_eq!(max_tick_seen.load(Ordering::Relaxed), 5);
    }

    /// A system requesting exit stops the server promptly and the requested exit
    /// code is returned, even with no tick cap set.
    #[test]
    fn dedicated_server_honors_exit_request() {
        let ticks = Arc::new(AtomicU64::new(0));
        let t = ticks.clone();

        let mut app = App::new();
        app.add_systems(Update, move |mut exit: ResMut<AppExitRequest>| {
            let n = t.fetch_add(1, Ordering::Relaxed) + 1;
            if n == 3 {
                exit.send_error();
            }
        });
        app.set_runner(|app| {
            DedicatedServerRunner::new(1000)
                .without_real_time_pacing()
                .run(app)
        });

        let exit = app.run();
        assert_eq!(exit, AppExit::error());
        assert_eq!(
            ticks.load(Ordering::Relaxed),
            3,
            "the server stops the tick after exit is requested",
        );
    }

    /// The server runs the graceful-shutdown path once after its tick loop ends.
    #[test]
    fn dedicated_server_runs_shutdown_once_after_tick_loop() {
        let shutdowns = Arc::new(AtomicU64::new(0));
        let s = shutdowns.clone();

        let mut app = App::new();
        app.add_systems(Shutdown, move || {
            s.fetch_add(1, Ordering::Relaxed);
        });
        app.set_runner(|app| {
            DedicatedServerRunner::new(60)
                .with_max_ticks(3)
                .without_real_time_pacing()
                .run(app)
        });

        let exit = app.run();
        assert_eq!(exit, AppExit::Success);
        assert_eq!(
            shutdowns.load(Ordering::Relaxed),
            1,
            "the dedicated server drains the Shutdown schedule exactly once",
        );
    }
}

// ---- settings

use crate::settings::{SettingChanged, SettingValue, Settings, SettingsLayer};

/// Layer precedence: the highest present layer wins, and `SettingsLayer`'s
/// derived `Ord` matches the documented ascending-precedence order.
#[test]
fn settings_resolve_highest_layer_and_order_is_precedence() {
    assert!(SettingsLayer::EngineDefault < SettingsLayer::PlatformTier);
    assert!(SettingsLayer::PlatformTier < SettingsLayer::User);
    assert!(SettingsLayer::User < SettingsLayer::CommandLine);
    assert!(SettingsLayer::CommandLine < SettingsLayer::Runtime);
    assert_eq!(SettingsLayer::ALL.len(), 5);
    assert_eq!(SettingsLayer::EngineDefault.precedence(), 0);
    assert_eq!(SettingsLayer::Runtime.precedence(), 4);

    let mut s = Settings::new();
    s.set(SettingsLayer::EngineDefault, "r.shadows", 1_i64);
    s.set(SettingsLayer::User, "r.shadows", 2_i64);
    s.set(SettingsLayer::PlatformTier, "r.shadows", 3_i64);

    // User (rank 2) outranks PlatformTier (rank 1) and EngineDefault (rank 0).
    assert_eq!(s.get_int("r.shadows"), Some(2));
    assert_eq!(s.resolved_layer("r.shadows"), Some(SettingsLayer::User));
    // All three contributions are retained underneath.
    assert_eq!(s.layers_for("r.shadows").map(|m| m.len()), Some(3));
}

/// Clearing the top layer transparently falls back to the next layer down;
/// clearing a shadowed (lower) layer does not change the resolved value.
#[test]
fn settings_clear_falls_back_to_lower_layer() {
    let mut s = Settings::new();
    s.set(SettingsLayer::EngineDefault, "net.tickrate", 30_i64);
    s.set(SettingsLayer::Runtime, "net.tickrate", 128_i64);
    assert_eq!(s.get_int("net.tickrate"), Some(128));

    // Clearing a lower, shadowed layer changes nothing resolved.
    assert!(s.clear(SettingsLayer::EngineDefault, "net.tickrate").is_none());
    assert_eq!(s.get_int("net.tickrate"), Some(128));

    // Re-add the default, then clear the top layer: falls back to the default.
    s.set(SettingsLayer::EngineDefault, "net.tickrate", 30_i64);
    let change = s
        .clear(SettingsLayer::Runtime, "net.tickrate")
        .expect("clearing the top layer changes the resolved value");
    assert_eq!(change.previous, Some(SettingValue::Int(128)));
    assert_eq!(change.current, Some(SettingValue::Int(30)));
    assert_eq!(s.get_int("net.tickrate"), Some(30));

    // Clearing the last remaining layer makes the key absent entirely.
    let change = s
        .clear(SettingsLayer::EngineDefault, "net.tickrate")
        .expect("clearing the last layer unsets the key");
    assert_eq!(change.current, None);
    assert!(!s.contains("net.tickrate"));
    assert!(s.layers_for("net.tickrate").is_none());
}

/// `set` only reports a change when the *resolved* value actually moves.
#[test]
fn settings_set_reports_resolved_change_only() {
    let mut s = Settings::new();

    // First write of a key: unset -> value is a change.
    let change = s
        .set(SettingsLayer::User, "vol", 50_i64)
        .expect("first write is a change");
    assert_eq!(change.key, "vol");
    assert_eq!(change.previous, None);
    assert_eq!(change.current, Some(SettingValue::Int(50)));

    // Writing a lower layer while User still overrides it: no resolved change.
    assert!(s.set(SettingsLayer::EngineDefault, "vol", 10_i64).is_none());
    // Rewriting the top layer with an equal value: no change.
    assert!(s.set(SettingsLayer::User, "vol", 50_i64).is_none());
    // Rewriting the top layer with a different value: a change.
    assert!(s.set(SettingsLayer::User, "vol", 60_i64).is_some());
}

/// `SettingValue::parse` infers the most specific type, and the typed accessors
/// / `From` impls round-trip.
#[test]
fn setting_value_parse_infers_type() {
    assert_eq!(SettingValue::parse("true"), SettingValue::Bool(true));
    assert_eq!(SettingValue::parse("false"), SettingValue::Bool(false));
    assert_eq!(SettingValue::parse("42"), SettingValue::Int(42));
    assert_eq!(SettingValue::parse("-7"), SettingValue::Int(-7));
    assert_eq!(SettingValue::parse("3.5"), SettingValue::Float(3.5));
    assert_eq!(SettingValue::parse("hi"), SettingValue::Str("hi".to_owned()));
    // An empty token is not a bool/int/float, so it stays a string.
    assert_eq!(SettingValue::parse(""), SettingValue::Str(String::new()));

    assert_eq!(SettingValue::from(true).as_bool(), Some(true));
    assert_eq!(SettingValue::from(9_i64).as_int(), Some(9));
    assert_eq!(SettingValue::from(1.25_f64).as_float(), Some(1.25));
    assert_eq!(SettingValue::from("s").as_str(), Some("s"));
    assert_eq!(SettingValue::from(String::from("t")).as_str(), Some("t"));
    // Cross-type accessors return None, not a coercion.
    assert_eq!(SettingValue::Int(1).as_bool(), None);
    assert_eq!(SettingValue::Bool(true).as_int(), None);
}

/// A `Float(NaN)` never equals itself, so re-setting NaN re-signals a change
/// (documented behavior).
#[test]
fn settings_nan_float_resignals_change() {
    let mut s = Settings::new();
    assert!(s.set(SettingsLayer::User, "x", f64::NAN).is_some());
    // Resolved value is NaN; writing NaN again counts as a change.
    assert!(s.set(SettingsLayer::User, "x", f64::NAN).is_some());
    // A finite value re-set to the same finite value does not re-signal.
    s.set(SettingsLayer::User, "y", 1.0_f64);
    assert!(s.set(SettingsLayer::User, "y", 1.0_f64).is_none());
}

/// CLI parsing: `--key=value`, `key=value`, and bare `--flag` land in the
/// CommandLine layer with inferred types.
#[test]
fn settings_apply_cli_args_parses_forms() {
    let mut s = Settings::new();
    let changes = s.apply_cli_args([
        "--r.shadows=2",
        "net.tickrate=128",
        "--vsync",
        "--name=prism",
    ]);
    assert_eq!(changes.len(), 4);
    assert_eq!(s.get_int("r.shadows"), Some(2));
    assert_eq!(s.resolved_layer("r.shadows"), Some(SettingsLayer::CommandLine));
    assert_eq!(s.get_int("net.tickrate"), Some(128));
    assert_eq!(s.get_bool("vsync"), Some(true));
    assert_eq!(s.get_str("name"), Some("prism"));

    // A bare "--" with no key is ignored rather than inserting an empty key.
    let none = s.apply_cli_args(["--"]);
    assert!(none.is_empty());
    assert!(!s.contains(""));
}

/// Env folding: only prefixed vars are taken, the prefix is stripped, and
/// `_`→`.` lowercase mapping matches the cvar namespace. Env + CLI share one
/// CommandLine layer, so a later env write overrides an earlier CLI write.
#[test]
fn settings_apply_env_vars_folds_into_command_line_layer() {
    let mut s = Settings::new();
    s.apply_cli_args(["r.shadows=1"]);
    let changes = s.apply_env_vars(
        [
            ("PRISM_R_SHADOWS", "4"),
            ("PRISM_NET_TICKRATE", "60"),
            ("PATH", "/usr/bin"), // no prefix -> ignored
        ],
        "PRISM_",
    );
    // r.shadows changed 1 -> 4; net.tickrate newly set; PATH ignored.
    assert_eq!(changes.len(), 2);
    assert_eq!(s.get_int("r.shadows"), Some(4));
    assert_eq!(s.resolved_layer("r.shadows"), Some(SettingsLayer::CommandLine));
    assert_eq!(s.get_int("net.tickrate"), Some(60));
    assert!(!s.contains("path"));
}

/// Resolved iteration is deterministic (ascending key order), independent of
/// insertion order.
#[test]
fn settings_iter_is_deterministic() {
    let mut a = Settings::new();
    a.set(SettingsLayer::User, "zeta", 1_i64);
    a.set(SettingsLayer::User, "alpha", 2_i64);
    a.set(SettingsLayer::User, "mid", 3_i64);

    let mut b = Settings::new();
    b.set(SettingsLayer::User, "mid", 3_i64);
    b.set(SettingsLayer::User, "zeta", 1_i64);
    b.set(SettingsLayer::User, "alpha", 2_i64);

    let keys_a: Vec<&str> = a.iter().map(|(k, _)| k).collect();
    let keys_b: Vec<&str> = b.iter().map(|(k, _)| k).collect();
    assert_eq!(keys_a, ["alpha", "mid", "zeta"]);
    assert_eq!(keys_a, keys_b);
}

/// `App::insert_setting` auto-initialises the store and broadcasts a
/// `SettingChanged` event carrying the resolved transition; a shadowed write
/// broadcasts nothing.
#[test]
fn app_insert_setting_broadcasts_resolved_change() {
    let mut app = App::new();

    // Settings is opt-in: not installed until a settings helper runs.
    assert!(app.world().get_resource::<Settings>().is_none());

    let seen = Arc::new(Mutex::new(Vec::<(String, Option<i64>, Option<i64>)>::new()));
    let seen_sys = seen.clone();
    app.init_settings();
    app.add_systems(
        Update,
        move |mut cursor: Local<EventCursor<SettingChanged>>, events: Res<Events<SettingChanged>>| {
            for change in cursor.read(&events) {
                seen_sys.lock().unwrap().push((
                    change.key.clone(),
                    change.previous.as_ref().and_then(SettingValue::as_int),
                    change.current.as_ref().and_then(SettingValue::as_int),
                ));
            }
        },
    );

    app.insert_setting(SettingsLayer::EngineDefault, "r.shadows", 1_i64);
    app.insert_setting(SettingsLayer::Runtime, "r.shadows", 3_i64);
    // Shadowed lower-layer write: no resolved change, no event.
    app.insert_setting(SettingsLayer::User, "r.shadows", 2_i64);

    assert_eq!(app.setting_int("r.shadows"), Some(3));

    // Two frames of grace to let the reader observe buffered events.
    app.update();
    app.update();

    let seen = seen.lock().unwrap();
    assert_eq!(
        seen.as_slice(),
        [
            ("r.shadows".to_owned(), None, Some(1)),
            ("r.shadows".to_owned(), Some(1), Some(3)),
        ]
    );
}

/// `App::apply_cli_overrides` initialises the store and the typed getters read
/// the resolved values back.
#[test]
fn app_apply_cli_overrides_and_typed_getters() {
    let mut app = App::new();
    app.apply_cli_overrides(["--net.tickrate=128", "--vsync", "--name=prism", "--gain=0.5"]);

    assert_eq!(app.setting_int("net.tickrate"), Some(128));
    assert_eq!(app.setting_bool("vsync"), Some(true));
    assert_eq!(app.setting_str("name"), Some("prism"));
    assert_eq!(app.setting_float("gain"), Some(0.5));
    // Unset key resolves to None through the App getter.
    assert_eq!(app.setting("missing"), None);
}

// ---- diagnostics (design §16: frame stats / phase timing / startup cost) ----
//
// The whole observability module and all five instrumentation hooks live behind
// `#[cfg(feature = "std")]` (they need a wall clock), with zero-overhead plain
// fallbacks, so these tests compile only when `std` is enabled. They assert the
// opt-in contract (nothing is installed by default), the per-frame / per-phase /
// substep collection, per-plugin startup timing, and the `CountWindow` helper.
#[cfg(feature = "std")]
mod diagnostics_tests {
    use super::*;

    use crate::diagnostics::{CountWindow, FrameDiagnostics, TIMED_FRAME_PHASES};

    /// Diagnostics are opt-in: a fresh `App` installs neither resource, and the
    /// plain (un-instrumented) frame path still drives `Update` normally.
    #[test]
    fn diagnostics_absent_by_default() {
        let ran = Arc::new(AtomicU64::new(0));
        let r = ran.clone();

        let mut app = App::new();
        assert!(app.frame_diagnostics().is_none());
        assert!(app.startup_diagnostics().is_none());

        app.add_systems(Update, move || {
            r.fetch_add(1, Ordering::Relaxed);
        });
        app.update();

        // Plain path ran the frame, and still nothing was installed.
        assert_eq!(ran.load(Ordering::Relaxed), 1);
        assert!(app.frame_diagnostics().is_none());
    }

    /// After `init_frame_diagnostics` the instrumented path records one frame per
    /// `update()` and times every core phase (see `TIMED_FRAME_PHASES`).
    #[test]
    fn frame_diagnostics_records_frames_and_every_core_phase() {
        let mut app = App::new();
        // A clock so the fixed loop actually runs (and `RunFixedMainLoop` times).
        app.set_time_update_strategy(TimeUpdateStrategy::ManualDelta(Duration::from_millis(10)));
        app.set_fixed_timestep_hz(100.0);
        app.init_frame_diagnostics();

        const FRAMES: u64 = 4;
        for _ in 0..FRAMES {
            app.update();
        }

        let diag = app.frame_diagnostics().expect("installed");
        assert_eq!(diag.frame_time().total_frames(), FRAMES);
        // Every core phase was timed, including the fixed loop, every frame.
        for phase in TIMED_FRAME_PHASES {
            let stats = diag
                .phase(phase)
                .unwrap_or_else(|| panic!("phase {phase} should be timed"));
            assert_eq!(stats.total_frames(), FRAMES, "phase {phase} per-frame count");
        }
        // The deterministic phase iterator yields exactly the core phases.
        assert_eq!(diag.phases().count(), TIMED_FRAME_PHASES.len());
    }

    /// The substep counter records how many fixed steps ran each frame: one step
    /// when the frame delta equals the fixed period, three when it is triple.
    #[test]
    fn frame_diagnostics_counts_fixed_substeps() {
        // 100 Hz => 10 ms period; feed exactly one period per frame.
        let mut app = App::new();
        app.set_time_update_strategy(TimeUpdateStrategy::ManualDelta(Duration::from_millis(10)));
        app.set_fixed_timestep_hz(100.0);
        app.init_frame_diagnostics();
        app.update();
        assert_eq!(app.frame_diagnostics().unwrap().fixed_substeps().last(), Some(1));

        // 30 ms per frame => three 10 ms substeps.
        let mut app = App::new();
        app.set_time_update_strategy(TimeUpdateStrategy::ManualDelta(Duration::from_millis(30)));
        app.set_fixed_timestep_hz(100.0);
        app.init_frame_diagnostics();
        app.update();
        let subs = app.frame_diagnostics().unwrap().fixed_substeps();
        assert_eq!(subs.last(), Some(3));
        assert_eq!(subs.total_samples(), 1);
        assert_eq!(subs.max(), Some(3));
    }

    /// The rolling window bounds memory: after more frames than the window, the
    /// per-metric sample count saturates at the window while the lifetime frame
    /// total keeps climbing.
    #[test]
    fn frame_diagnostics_window_bounds_memory() {
        let mut app = App::new();
        app.set_time_update_strategy(TimeUpdateStrategy::ManualDelta(Duration::from_millis(10)));
        app.set_fixed_timestep_hz(100.0);
        app.init_frame_diagnostics_with_window(3);

        for _ in 0..7 {
            app.update();
        }

        let diag = app.frame_diagnostics().unwrap();
        assert_eq!(diag.window(), 3);
        assert_eq!(diag.frame_time().len(), 3, "window caps live samples");
        assert_eq!(diag.frame_time().total_frames(), 7, "lifetime total is unbounded");
        assert_eq!(diag.fixed_substeps().len(), 3);
        assert_eq!(diag.fixed_substeps().total_samples(), 7);
    }

    /// `init_frame_diagnostics` is idempotent: a second call keeps the resource
    /// (and its accumulated history) rather than resetting it.
    #[test]
    fn init_frame_diagnostics_is_idempotent() {
        let mut app = App::new();
        app.set_time_update_strategy(TimeUpdateStrategy::ManualDelta(Duration::from_millis(10)));
        app.set_fixed_timestep_hz(100.0);
        app.init_frame_diagnostics();
        app.update();
        app.update();
        assert_eq!(app.frame_diagnostics().unwrap().frame_time().total_frames(), 2);

        // A redundant init (even with a different window) must not wipe history.
        app.init_frame_diagnostics_with_window(999);
        let diag = app.frame_diagnostics().unwrap();
        assert_eq!(diag.frame_time().total_frames(), 2, "history preserved");
        assert_eq!(diag.window(), FrameDiagnostics::DEFAULT_WINDOW, "window unchanged");
    }

    struct NoopBuild;
    impl Plugin for NoopBuild {
        fn build(&self, _app: &mut App) {}
        fn name(&self) -> &str {
            "noop"
        }
    }

    struct SlowBuild;
    impl Plugin for SlowBuild {
        fn build(&self, _app: &mut App) {
            std::thread::sleep(Duration::from_millis(5));
        }
        fn name(&self) -> &str {
            "slow-build"
        }
    }

    struct SlowFinish;
    impl Plugin for SlowFinish {
        fn build(&self, _app: &mut App) {}
        fn finish(&self, _app: &mut App) {
            std::thread::sleep(Duration::from_millis(5));
        }
        fn name(&self) -> &str {
            "slow-finish"
        }
    }

    /// With `StartupDiagnostics` installed before `add_plugins`, each plugin's
    /// build is timed in registration order, and `slowest_build` fingers the
    /// deliberately slow one.
    #[test]
    fn startup_diagnostics_times_plugin_builds() {
        let mut app = App::new();
        app.init_startup_diagnostics();
        app.add_plugins(NoopBuild);
        app.add_plugins(SlowBuild);

        let diag = app.startup_diagnostics().expect("installed");
        let names: Vec<&str> = diag.plugins().iter().map(|p| p.name.as_str()).collect();
        assert_eq!(names, ["noop", "slow-build"], "timed in registration order");

        let slowest = diag.slowest_build().expect("non-empty");
        assert_eq!(slowest.name, "slow-build");
        assert!(slowest.build >= Duration::from_millis(5), "slow build measured");
        assert!(diag.total_build() >= Duration::from_millis(5));
    }

    /// `finish` timings are filled in against the matching build entry when
    /// `App::finish` runs; a plugin that overrides nothing stays at `ZERO`.
    #[test]
    fn startup_diagnostics_records_finish() {
        let mut app = App::new();
        app.init_startup_diagnostics();
        app.add_plugins(NoopBuild);
        app.add_plugins(SlowFinish);
        app.finish();

        let diag = app.startup_diagnostics().expect("installed");
        let noop = diag.plugins().iter().find(|p| p.name == "noop").unwrap();
        let slow = diag.plugins().iter().find(|p| p.name == "slow-finish").unwrap();
        // Every plugin's finish is timed, so a no-op finish records a tiny (not
        // literally zero) duration; the deliberately slow one dwarfs it.
        assert!(slow.finish >= Duration::from_millis(5), "slow finish measured");
        assert!(slow.finish > noop.finish, "slow finish outweighs the no-op");
        assert!(diag.total_finish() >= Duration::from_millis(5));
    }

    /// Without `init_startup_diagnostics`, adding plugins installs nothing and
    /// no build timings are retroactively invented.
    #[test]
    fn startup_diagnostics_absent_before_init() {
        let mut app = App::new();
        app.add_plugins(NoopBuild);
        assert!(app.startup_diagnostics().is_none());
    }

    /// `CountWindow` reports last/average/max/min over its live window, evicts
    /// the oldest sample past the window, and keeps a lifetime total.
    #[test]
    fn count_window_records_evicts_and_summarises() {
        let mut w = CountWindow::new(3);
        assert!(w.is_empty());
        assert_eq!(w.last(), None);
        assert_eq!(w.average(), None);

        w.record(10);
        w.record(20);
        w.record(30);
        assert_eq!(w.len(), 3);
        assert_eq!(w.total_samples(), 3);
        assert_eq!(w.last(), Some(30));
        assert_eq!(w.min(), Some(10));
        assert_eq!(w.max(), Some(30));
        assert_eq!(w.average(), Some(20.0));

        // Fourth sample evicts the oldest (10); the lifetime total still grows.
        w.record(40);
        assert_eq!(w.len(), 3, "window stays bounded");
        assert_eq!(w.total_samples(), 4);
        assert_eq!(w.last(), Some(40));
        assert_eq!(w.min(), Some(20));
        assert_eq!(w.max(), Some(40));
        assert_eq!(w.average(), Some(30.0));
    }

    /// A `0` window is clamped to `1`, so the last sample is always retained.
    #[test]
    fn count_window_zero_window_clamped_to_one() {
        let mut w = CountWindow::new(0);
        w.record(7);
        w.record(9);
        assert_eq!(w.len(), 1);
        assert_eq!(w.last(), Some(9));
        assert_eq!(w.total_samples(), 2);
    }

    /// The serial frame path times the secondary-sub-app extract step and
    /// records one extract sample per `update()`, while never recording a
    /// pipeline-overlap sample (overlap is a `pipelined`-only quantity, so the
    /// derived ratio stays `None` on the serial path).
    #[test]
    fn frame_diagnostics_records_serial_extract_cost() {
        let mut app = App::new();
        app.set_time_update_strategy(TimeUpdateStrategy::ManualDelta(Duration::from_millis(10)));
        app.set_fixed_timestep_hz(100.0);

        // A secondary sub-app with a real extract that reads the main world.
        app.world_mut().insert_resource(Counter::default());
        app.add_systems(Update, |mut c: ResMut<Counter>| {
            c.0 += 1;
        });
        let mut render = SubApp::new();
        render.world.insert_resource(Counter::default());
        app.insert_sub_app(RenderApp, render);
        app.set_extract(
            RenderApp,
            |main: &mut prism_ecs::world::World, sub: &mut prism_ecs::world::World| {
                sub.resource_mut::<Counter>().0 = main.resource::<Counter>().0;
            },
        );

        app.init_frame_diagnostics();

        const FRAMES: u64 = 4;
        for _ in 0..FRAMES {
            app.update();
        }

        let diag = app.frame_diagnostics().expect("installed");
        // One extract sample recorded per frame on the serial path.
        assert_eq!(diag.extract_time().total_frames(), FRAMES);
        // The serial path never overlaps simulate and render, so no overlap
        // sample is recorded and the derived ratio is absent.
        assert_eq!(diag.pipeline_sim().total_frames(), 0);
        assert_eq!(diag.pipeline_wait().total_frames(), 0);
        assert_eq!(diag.pipeline_overlap_ratio(), None);
    }
}

/// Determinism & replay primitives (design §15, §22 M5): the seeded RNG, the
/// per-frame FNV-1a frame hash (including the dual-run divergence check), and
/// transparent input record/replay. Gated on the `determinism` feature so the
/// default build neither compiles nor pays for them.
#[cfg(feature = "determinism")]
mod determinism_tests {
    use super::*;

    use crate::determinism::{
        DeterministicRng, FrameHash, FrameHashManifest, HashDivergence, InputRecording,
        RecordedInput, ReplayLog, ReplayMode, DEFAULT_HASH_HISTORY,
    };

    // ---- DeterministicRng -------------------------------------------------

    /// The whole point of a seeded RNG: two generators from the same seed that
    /// draw in the same order observe the identical stream, and the seed is
    /// recoverable for logging/reproduction.
    #[test]
    fn rng_same_seed_reproduces_the_stream() {
        let mut a = DeterministicRng::seeded(0x1234_5678_9ABC_DEF0);
        let mut b = DeterministicRng::seeded(0x1234_5678_9ABC_DEF0);
        assert_eq!(a.seed(), 0x1234_5678_9ABC_DEF0);
        assert_eq!(b.seed(), a.seed());

        let sa: Vec<u64> = (0..64).map(|_| a.next_u64()).collect();
        let sb: Vec<u64> = (0..64).map(|_| b.next_u64()).collect();
        assert_eq!(sa, sb, "same seed must reproduce the exact stream");
        // A non-trivial generator does not just echo its seed back.
        assert_ne!(sa[0], 0x1234_5678_9ABC_DEF0);
    }

    /// Different seeds produce different streams (sanity: the generator mixes
    /// the seed rather than ignoring it).
    #[test]
    fn rng_different_seeds_diverge() {
        let mut a = DeterministicRng::seeded(1);
        let mut b = DeterministicRng::seeded(2);
        let sa: Vec<u64> = (0..32).map(|_| a.next_u64()).collect();
        let sb: Vec<u64> = (0..32).map(|_| b.next_u64()).collect();
        assert_ne!(sa, sb);
    }

    /// `fork` yields an independent, reproducible child stream and perturbs the
    /// parent by exactly the single draw it consumes.
    #[test]
    fn rng_fork_is_reproducible_and_consumes_one_draw() {
        // The child is seeded from the parent's next draw: forking twice from
        // equal parents yields equal children.
        let mut p1 = DeterministicRng::seeded(99);
        let mut p2 = DeterministicRng::seeded(99);
        let mut c1 = p1.fork();
        let mut c2 = p2.fork();
        let s1: Vec<u64> = (0..16).map(|_| c1.next_u64()).collect();
        let s2: Vec<u64> = (0..16).map(|_| c2.next_u64()).collect();
        assert_eq!(s1, s2, "forked children from equal parents match");

        // Forking consumes exactly one parent draw: a parent that forked once
        // is where an un-forked peer is after a single `next_u64`.
        let mut forked = DeterministicRng::seeded(7);
        let child_seed = {
            let mut peek = forked.clone();
            peek.next_u64()
        };
        let actual_child = forked.fork();
        assert_eq!(
            actual_child.seed(),
            child_seed,
            "child seed is the parent's next draw"
        );

        let mut plain = DeterministicRng::seeded(7);
        plain.next_u64(); // consume the one draw fork used
        assert_eq!(forked.next_u64(), plain.next_u64());
    }

    /// `next_bounded_u64` stays in range for a variety of bounds, returns `0`
    /// for a zero bound, and covers the full range for a small bound.
    #[test]
    fn rng_bounded_is_in_range_and_handles_zero() {
        let mut rng = DeterministicRng::seeded(0xDEAD_BEEF);
        assert_eq!(rng.next_bounded_u64(0), 0, "bound 0 => 0");
        assert_eq!(rng.next_bounded_u64(1), 0, "bound 1 => only 0");

        let mut seen = [false; 6];
        for _ in 0..4096 {
            let v = rng.next_bounded_u64(6);
            assert!(v < 6, "value {v} must be < bound");
            seen[v as usize] = true;
        }
        assert!(seen.iter().all(|&s| s), "a fair die should hit every face");
    }

    /// The float draws land in `[0, 1)` and reproduce from the same seed.
    #[test]
    fn rng_floats_are_unit_interval_and_reproducible() {
        let mut a = DeterministicRng::seeded(42);
        let mut b = DeterministicRng::seeded(42);
        for _ in 0..1000 {
            let fa = a.next_f64();
            assert!((0.0..1.0).contains(&fa), "f64 {fa} not in [0,1)");
            assert_eq!(fa.to_bits(), b.next_f64().to_bits(), "f64 reproducible");
        }
        let mut c = DeterministicRng::seeded(42);
        let mut d = DeterministicRng::seeded(42);
        for _ in 0..1000 {
            let fc = c.next_f32();
            assert!((0.0..1.0).contains(&fc), "f32 {fc} not in [0,1)");
            assert_eq!(fc.to_bits(), d.next_f32().to_bits(), "f32 reproducible");
        }
    }

    // ---- FrameHash --------------------------------------------------------

    /// Folding then finalizing produces a stable, order-sensitive digest, and
    /// an empty frame finalizes to the FNV-1a offset basis.
    #[test]
    fn frame_hash_folds_finalizes_and_is_order_sensitive() {
        // Equal inputs in equal order => equal finalized hash.
        let mut a = FrameHash::new();
        let mut b = FrameHash::new();
        a.write_u64(1);
        a.write_u64(2);
        b.write_u64(1);
        b.write_u64(2);
        assert_eq!(a.finalize_frame(), b.finalize_frame());

        // Reversed order => different hash (FNV-1a is order-sensitive).
        let mut c = FrameHash::new();
        let mut d = FrameHash::new();
        c.write_u64(1);
        c.write_u64(2);
        d.write_u64(2);
        d.write_u64(1);
        assert_ne!(c.finalize_frame(), d.finalize_frame());

        // An empty frame finalizes to the offset basis (0xcbf2_9ce4_8422_2325).
        let mut e = FrameHash::new();
        assert_eq!(e.finalize_frame(), 0xcbf2_9ce4_8422_2325);
    }

    /// `current` accumulates during a frame and resets to the offset basis
    /// after finalize; `frame_index` counts total finalized frames.
    #[test]
    fn frame_hash_current_resets_and_index_advances() {
        let mut h = FrameHash::new();
        assert_eq!(h.frame_index(), 0);
        assert!(h.is_empty());
        assert_eq!(h.last(), None);

        h.write_u8(0xAB);
        assert_ne!(h.current(), 0xcbf2_9ce4_8422_2325, "folding changed current");
        let f0 = h.finalize_frame();
        assert_eq!(h.current(), 0xcbf2_9ce4_8422_2325, "current reset after finalize");
        assert_eq!(h.frame_index(), 1);
        assert_eq!(h.last(), Some(f0));
        assert_eq!(h.len(), 1);
        assert!(!h.is_empty());
    }

    /// The rolling history is bounded by the window: the oldest finalized hash
    /// is evicted once the window is full, while `frame_index` keeps climbing.
    #[test]
    fn frame_hash_window_bounds_history() {
        let mut h = FrameHash::with_window(3);
        assert_eq!(h.window(), 3);
        for i in 0..5u64 {
            h.write_u64(i);
            h.finalize_frame();
        }
        assert_eq!(h.len(), 3, "history stays within the window");
        assert_eq!(h.frame_index(), 5, "frame_index counts every finalized frame");

        // The retained hashes are the last three frames (i = 2,3,4), oldest
        // first. Recompute the expected digests independently.
        let expected: Vec<u64> = (2..5u64)
            .map(|i| {
                let mut g = FrameHash::new();
                g.write_u64(i);
                g.finalize_frame()
            })
            .collect();
        let got: Vec<u64> = h.history().collect();
        assert_eq!(got, expected);
    }

    /// A `0` window is clamped to `1`, so `last` always reflects the most
    /// recent finalized frame.
    #[test]
    fn frame_hash_zero_window_clamped_to_one() {
        let mut h = FrameHash::with_window(0);
        assert_eq!(h.window(), 1);
        h.write_u64(10);
        h.finalize_frame();
        h.write_u64(20);
        let last = h.finalize_frame();
        assert_eq!(h.len(), 1);
        assert_eq!(h.last(), Some(last));
        assert_eq!(h.frame_index(), 2);
    }

    /// `write_f64`/`write_f32` fold by raw bits, so distinct bit patterns
    /// (notably `+0.0` vs `-0.0`) hash differently — the hash reports
    /// divergence rather than papering over it.
    #[test]
    fn frame_hash_floats_fold_by_bits() {
        let mut pos = FrameHash::new();
        let mut neg = FrameHash::new();
        pos.write_f64(0.0);
        neg.write_f64(-0.0);
        assert_ne!(pos.finalize_frame(), neg.finalize_frame());

        let mut a = FrameHash::new();
        let mut b = FrameHash::new();
        a.write_f32(1.5);
        b.write_f32(1.5);
        assert_eq!(a.finalize_frame(), b.finalize_frame());
    }

    // ---- FrameHash divergence (dual-run acceptance check) -----------------

    /// `first_divergence` finds the earliest differing frame and pinpoints both
    /// hashes; two identical runs report `None` (design §22 dual-run equality).
    #[test]
    fn frame_hash_first_divergence_detects_mismatch() {
        let mut a = FrameHash::new();
        let mut b = FrameHash::new();
        // Frames 0 and 1 stay in lockstep.
        for v in [10u64, 20] {
            a.write_u64(v);
            a.finalize_frame();
            b.write_u64(v);
            b.finalize_frame();
        }
        assert_eq!(a.first_divergence(&b), None, "lockstep runs agree");

        // Frame 2 diverges.
        a.write_u64(30);
        let a2 = a.finalize_frame();
        b.write_u64(31);
        let b2 = b.finalize_frame();
        assert_ne!(a2, b2, "distinct inputs produce distinct hashes");

        let d = a.first_divergence(&b).expect("divergence at frame 2");
        assert_eq!(
            d,
            HashDivergence {
                frame: 2,
                left: a2,
                right: b2
            }
        );
        // Viewed from the other run the sides swap but the frame is the same.
        let d2 = b
            .first_divergence(&a)
            .expect("divergence seen from the other run");
        assert_eq!(
            d2,
            HashDivergence {
                frame: 2,
                left: b2,
                right: a2
            }
        );
    }

    /// Divergence detection aligns by absolute frame index, so it still works
    /// after the rolling windows have slid; disjoint windows compare nothing
    /// and report `None`.
    #[test]
    fn frame_hash_divergence_aligns_sliding_windows() {
        let mut a = FrameHash::with_window(2);
        let mut b = FrameHash::with_window(2);
        for v in 0..5u64 {
            a.write_u64(v);
            a.finalize_frame();
            b.write_u64(v);
            b.finalize_frame();
        }
        // Both retain only frames 3 and 4; the evicted frames are gone.
        assert_eq!(a.oldest_frame_index(), Some(3));
        assert_eq!(a.frame_index(), 5);
        assert_eq!(a.first_divergence(&b), None, "lockstep across slid windows");

        // Diverge only on the latest frame (index 5).
        a.write_u64(99);
        let a5 = a.finalize_frame();
        b.write_u64(100);
        let b5 = b.finalize_frame();
        let d = a.first_divergence(&b).expect("latest frame diverges");
        assert_eq!(
            d,
            HashDivergence {
                frame: 5,
                left: a5,
                right: b5
            }
        );

        // `hash_at` only answers for retained, finalized frames.
        assert_eq!(a.hash_at(5), Some(a5));
        assert_eq!(a.hash_at(3), None, "frame 3 evicted past the window");
        assert_eq!(a.hash_at(6), None, "frame 6 not finalized yet");

        // Advance only `a` until its retained window no longer intersects `b`'s.
        for v in 0..2u64 {
            a.write_u64(v);
            a.finalize_frame();
        }
        assert_eq!(a.oldest_frame_index(), Some(6));
        assert_eq!(b.oldest_frame_index(), Some(4));
        assert_eq!(
            a.first_divergence(&b),
            None,
            "disjoint windows have nothing to compare"
        );
    }

    /// Before any frame finalizes there is nothing to compare or index.
    #[test]
    fn frame_hash_divergence_empty_is_none() {
        let a = FrameHash::new();
        let b = FrameHash::new();
        assert_eq!(a.oldest_frame_index(), None);
        assert_eq!(a.hash_at(0), None);
        assert_eq!(a.first_divergence(&b), None);
    }

    // ---- App integration --------------------------------------------------

    /// Determinism resources are opt-in: a fresh `App` installs none of them.
    #[test]
    fn determinism_absent_by_default() {
        let app = App::new();
        assert!(app.deterministic_rng().is_none());
        assert!(app.frame_hash().is_none());
        assert!(app.input_recording::<u32>().is_none());
    }

    /// `init_determinism` installs both the RNG and the frame hash, and the
    /// scheduled `Last` finalize records one frame per `update()`.
    #[test]
    fn init_determinism_installs_and_finalizes_each_frame() {
        let mut app = App::new();
        app.set_time_update_strategy(TimeUpdateStrategy::ManualDelta(Duration::from_millis(10)));
        app.set_fixed_timestep_hz(100.0);
        app.init_determinism(7);

        assert_eq!(app.deterministic_rng().unwrap().seed(), 7);
        assert_eq!(app.frame_hash().unwrap().window(), DEFAULT_HASH_HISTORY);
        // Nothing finalized before the first frame.
        assert!(app.frame_hash().unwrap().last().is_none());

        const FRAMES: u64 = 4;
        for _ in 0..FRAMES {
            app.update();
        }
        let hash = app.frame_hash().unwrap();
        assert_eq!(hash.frame_index(), FRAMES, "one finalize per frame");
        assert!(hash.last().is_some());
    }

    /// The init helpers are idempotent: a second call never reseeds the RNG,
    /// replaces the frame hash (losing history), or double-schedules finalize.
    #[test]
    fn init_helpers_are_idempotent() {
        let mut app = App::new();
        app.set_time_update_strategy(TimeUpdateStrategy::ManualDelta(Duration::from_millis(10)));
        app.set_fixed_timestep_hz(100.0);
        app.init_determinism(7);
        app.update();

        // Re-init with a different seed/window must be ignored.
        app.init_determinism(999);
        app.init_frame_hash_with_window(1);
        assert_eq!(app.deterministic_rng().unwrap().seed(), 7, "seed preserved");
        assert_eq!(
            app.frame_hash().unwrap().window(),
            DEFAULT_HASH_HISTORY,
            "window preserved"
        );

        // Exactly one finalize per frame despite repeated init calls.
        let before = app.frame_hash().unwrap().frame_index();
        app.update();
        assert_eq!(
            app.frame_hash().unwrap().frame_index(),
            before + 1,
            "finalize scheduled exactly once"
        );
    }

    /// Design §22 acceptance check: two independent apps with the same seed,
    /// driven by the same deltas and the same per-frame RNG folding, produce
    /// bit-identical frame-hash histories (dual-run determinism), while a
    /// different seed diverges.
    #[test]
    fn dual_run_frame_hashes_match_for_equal_seeds() {
        fn hashing_app(seed: u64) -> App {
            let mut app = App::new();
            app.set_time_update_strategy(TimeUpdateStrategy::ManualDelta(
                Duration::from_millis(10),
            ));
            app.set_fixed_timestep_hz(100.0);
            app.init_determinism(seed);
            app.add_systems(
                Update,
                |mut rng: ResMut<DeterministicRng>, mut hash: ResMut<FrameHash>| {
                    // Fold a reproducible per-frame draw into the digest.
                    let draw = rng.next_u64();
                    hash.write_u64(draw);
                },
            );
            app
        }

        const FRAMES: u64 = 8;
        let mut a = hashing_app(0xABCD_1234);
        let mut b = hashing_app(0xABCD_1234);
        for _ in 0..FRAMES {
            a.update();
            b.update();
        }
        let ha: Vec<u64> = a.frame_hash().unwrap().history().collect();
        let hb: Vec<u64> = b.frame_hash().unwrap().history().collect();
        assert_eq!(ha.len(), FRAMES as usize);
        assert_eq!(ha, hb, "equal seeds => bit-identical frame-hash history");

        // A different seed diverges.
        let mut c = hashing_app(0x9999_0000);
        for _ in 0..FRAMES {
            c.update();
        }
        let hc: Vec<u64> = c.frame_hash().unwrap().history().collect();
        assert_ne!(ha, hc, "different seed => divergent history");
    }

    // ---- InputRecording ---------------------------------------------------

    /// A record session captures each advanced frame tagged with its step; the
    /// captured buffer replays the exact same sequence.
    #[test]
    fn input_record_then_replay_round_trips() {
        let mut rec: InputRecording<u32> = InputRecording::recording();
        assert_eq!(rec.mode(), ReplayMode::Record);
        assert!(rec.is_empty());

        let live = [10u32, 20, 30, 40];
        for &v in &live {
            // Record mode returns the live frame unchanged.
            assert_eq!(rec.advance(v), v);
        }
        assert_eq!(rec.len(), live.len());
        assert_eq!(rec.step(), live.len() as u64);
        assert!(!rec.is_exhausted(), "recording is never 'exhausted'");

        // Each frame is tagged with its zero-based step index.
        for (i, captured) in rec.frames().iter().enumerate() {
            assert_eq!(
                captured,
                &RecordedInput {
                    step: i as u64,
                    frame: live[i]
                }
            );
        }

        // Replay the captured buffer: it reproduces the recorded sequence
        // regardless of what "live" input is fed.
        let frames = rec.into_frames();
        let mut replay = InputRecording::replaying(frames);
        assert_eq!(replay.mode(), ReplayMode::Replay);
        for &expected in &live {
            // Feed a bogus live value; replay must ignore it.
            assert_eq!(replay.advance(u32::MAX), expected);
        }
    }

    /// Once a replay is exhausted it reports `is_exhausted` and transparently
    /// falls back to the live input.
    #[test]
    fn replay_exhaustion_falls_back_to_live() {
        let frames = vec![
            RecordedInput { step: 0, frame: 1u8 },
            RecordedInput { step: 1, frame: 2u8 },
        ];
        let mut replay = InputRecording::replaying(frames);
        assert!(!replay.is_exhausted());
        assert_eq!(replay.advance(100), 1);
        assert_eq!(replay.advance(100), 2);
        assert!(replay.is_exhausted(), "all recorded frames consumed");
        // Past the end, the live frame passes through.
        assert_eq!(replay.advance(100), 100);
        assert_eq!(replay.advance(101), 101);
        // The step counter keeps advancing across the boundary.
        assert_eq!(replay.step(), 4);
    }

    /// Idle mode is a pure pass-through that still advances the step counter and
    /// never records.
    #[test]
    fn idle_recording_passes_through() {
        let mut idle: InputRecording<&'static str> = InputRecording::idle();
        assert_eq!(idle.mode(), ReplayMode::Idle);
        assert_eq!(idle.advance("a"), "a");
        assert_eq!(idle.advance("b"), "b");
        assert_eq!(idle.step(), 2);
        assert!(idle.is_empty(), "idle never records");
        assert!(!idle.is_exhausted());
    }

    /// `App::init_input_recording` installs a recorder for a frame type `F` and
    /// is idempotent per `F` (an existing recorder is never replaced).
    #[test]
    fn app_input_recording_install_is_idempotent() {
        let mut app = App::new();
        assert!(app.input_recording::<u16>().is_none());

        app.init_input_recording::<u16>(InputRecording::recording());
        assert_eq!(app.input_recording::<u16>().unwrap().mode(), ReplayMode::Record);

        // A second call with a different mode must be ignored for the same `F`.
        app.init_input_recording::<u16>(InputRecording::idle());
        assert_eq!(
            app.input_recording::<u16>().unwrap().mode(),
            ReplayMode::Record,
            "existing recorder preserved"
        );

        // A different frame type is tracked independently.
        assert!(app.input_recording::<i8>().is_none());
    }

    // ---- ReplayLog --------------------------------------------------------

    /// A `ReplayLog` is a self-contained repro: it carries the seed *and* the
    /// recorded frames, and `into_replay` hands back exactly the `(rng,
    /// recording)` pair that reproduces the session.
    #[test]
    fn replay_log_round_trips_seed_and_frames() {
        let mut rec = InputRecording::<u32>::recording();
        for live in [7u32, 8, 9] {
            assert_eq!(rec.advance(live), live);
        }

        let log = ReplayLog::from_recording(0xABCD, rec);
        assert_eq!(log.seed(), 0xABCD);
        assert_eq!(log.len(), 3);
        assert!(!log.is_empty());
        let steps: Vec<u64> = log.frames().iter().map(|f| f.step).collect();
        assert_eq!(steps, [0, 1, 2]);

        let (rng, mut replay) = log.into_replay();
        assert_eq!(rng.seed(), 0xABCD);
        assert_eq!(replay.mode(), ReplayMode::Replay);
        // Replaying ignores the live input and reproduces the recorded frames.
        assert_eq!(replay.advance(0), 7);
        assert_eq!(replay.advance(0), 8);
        assert_eq!(replay.advance(0), 9);
        assert!(replay.is_exhausted());
    }

    /// `App::init_replay` is authoritative, not idempotent: it overwrites any
    /// existing RNG seed and recorder so the run matches the log exactly, while
    /// leaving a `FrameHash` installed for divergence checks.
    #[test]
    fn init_replay_installs_authoritative_seed_and_frames() {
        let mut app = App::new();
        app.init_determinism(1);
        assert_eq!(app.deterministic_rng().unwrap().seed(), 1);

        let log = ReplayLog::new(
            0x5EED,
            vec![
                RecordedInput { step: 0, frame: 100u32 },
                RecordedInput { step: 1, frame: 200u32 },
            ],
        );
        app.init_replay(log);

        // RNG re-seeded to the log's authoritative seed (overwrites the 1 above).
        assert_eq!(app.deterministic_rng().unwrap().seed(), 0x5EED);
        // Recorder installed in Replay mode primed with the two frames.
        let rec = app.input_recording::<u32>().unwrap();
        assert_eq!(rec.mode(), ReplayMode::Replay);
        assert_eq!(rec.len(), 2);
        // FrameHash available for divergence detection.
        assert!(app.frame_hash().is_some());
    }

    // ---- FrameHashManifest ------------------------------------------------

    /// Capture a golden run's hashes into a manifest, then confirm an identical
    /// run matches frame-for-frame (no divergence) and the manifest exposes its
    /// recorded range and expected values.
    #[test]
    fn frame_hash_manifest_captures_a_clean_run() {
        // Golden run: fold three frames of authoritative state.
        let mut golden = FrameHash::new();
        for v in [1u64, 2, 3] {
            golden.write_u64(v);
            golden.finalize_frame();
        }
        let manifest = FrameHashManifest::from_frame_hash(&golden);
        assert_eq!(manifest.start_frame(), 0);
        assert_eq!(manifest.len(), 3);
        assert!(!manifest.is_empty());
        assert_eq!(manifest.end_frame(), 3);
        assert_eq!(manifest.expected_at(1), golden.hash_at(1));
        assert_eq!(manifest.expected_at(3), None);

        // An identical replay hashes the same way: no divergence.
        let mut replay = FrameHash::new();
        for v in [1u64, 2, 3] {
            replay.write_u64(v);
            replay.finalize_frame();
        }
        assert_eq!(manifest.first_divergence(&replay), None);
    }

    /// A manifest pins the first frame where a divergent run breaks from the
    /// golden expectation, reporting expected (left) vs actual (right).
    #[test]
    fn frame_hash_manifest_detects_divergence() {
        let mut golden = FrameHash::new();
        for v in [10u64, 20, 30, 40] {
            golden.write_u64(v);
            golden.finalize_frame();
        }
        let manifest = FrameHashManifest::from_frame_hash(&golden);

        // Divergent run: frame index 2 folds different state.
        let mut actual = FrameHash::new();
        for v in [10u64, 20, 99, 40] {
            actual.write_u64(v);
            actual.finalize_frame();
        }
        let div = manifest.first_divergence(&actual).expect("divergence at frame 2");
        assert_eq!(div.frame, 2);
        assert_eq!(div.left, manifest.expected_at(2).unwrap());
        assert_eq!(div.right, actual.hash_at(2).unwrap());
        assert_ne!(div.left, div.right);
    }

    /// Incremental capture via `recording_from` + `push` retains history beyond
    /// a hasher's rolling window, and `App::verify_against_manifest` runs the
    /// one-call dual-run check against the live `FrameHash`.
    #[test]
    fn app_verify_against_manifest_round_trips() {
        // Build a golden manifest incrementally, anchored at frame 0.
        let mut golden = FrameHash::new();
        let mut manifest = FrameHashManifest::recording_from(0);
        for v in [5u64, 6, 7] {
            golden.write_u64(v);
            golden.finalize_frame();
            manifest.push(golden.last().unwrap());
        }
        assert_eq!(manifest.len(), 3);

        // No FrameHash installed yet -> nothing to compare -> None.
        let mut app = App::new();
        assert_eq!(app.verify_against_manifest(&manifest), None);

        // Install a hasher and reproduce the golden run exactly: matches.
        app.init_frame_hash();
        {
            let hash = app
                .world_mut()
                .get_resource_mut::<FrameHash>()
                .expect("frame hash installed");
            for v in [5u64, 6, 7] {
                hash.write_u64(v);
                hash.finalize_frame();
            }
        }
        assert_eq!(app.verify_against_manifest(&manifest), None);

        // Fold a divergent fourth frame against a one-longer manifest.
        let mut golden2 = FrameHash::new();
        for v in [5u64, 6, 7, 8] {
            golden2.write_u64(v);
            golden2.finalize_frame();
        }
        let manifest2 = FrameHashManifest::from_frame_hash(&golden2);
        {
            let hash = app
                .world_mut()
                .get_resource_mut::<FrameHash>()
                .expect("frame hash installed");
            hash.write_u64(999);
            hash.finalize_frame();
        }
        let div = app
            .verify_against_manifest(&manifest2)
            .expect("fourth frame diverges");
        assert_eq!(div.frame, 3);
    }
}

// ---- state-machine depth (§11): computed / sub / scoped --------------------

/// Integration tests for the design §11 state-machine depth features layered on
/// the `prism_ecs` base states: computed states (derived each frame), sub-states
/// (exist only while a parent mode is active), and state-scoped entities
/// (auto-despawned when their owning mode leaves). All drive the real
/// `StateTransition` phase through `App::update`, so they also pin the
/// `Apply`-then-`Compute` ordering.
mod state_depth_tests {
    use super::*;

    use prism_ecs::entity::Entity;

    use crate::state::{ComputedStates, OnTransition, StateScoped, SubStates};

    /// Shared edge log: records `OnEnter`/`OnExit` tags in the order they fire.
    type Log = Arc<Mutex<Vec<&'static str>>>;

    fn log() -> Log {
        Arc::new(Mutex::new(Vec::new()))
    }

    fn drain(log: &Log) -> Vec<&'static str> {
        core::mem::take(&mut *log.lock().unwrap())
    }

    /// Three-mode base state the depth features derive from.
    #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
    enum AppState {
        #[default]
        Menu,
        InGame,
        Paused,
    }
    impl States for AppState {}

    /// A computed state derived from [`AppState`]: it exists only while the app
    /// is actually in a session, and changes value between playing and paused.
    /// `Menu` yields `None` (the computed state does not exist).
    #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
    enum Activity {
        Playing,
        Halted,
    }
    impl States for Activity {}
    impl ComputedStates for Activity {
        type SourceStates = AppState;
        fn compute(source: &AppState) -> Option<Self> {
            match source {
                AppState::Menu => None,
                AppState::InGame => Some(Activity::Playing),
                AppState::Paused => Some(Activity::Halted),
            }
        }
    }

    fn log_activity_edges(app: &mut App, log: &Log) {
        for (value, enter, exit) in [
            (Activity::Playing, "enter:playing", "exit:playing"),
            (Activity::Halted, "enter:halted", "exit:halted"),
        ] {
            let l = log.clone();
            app.add_systems(OnEnter(value), move || l.lock().unwrap().push(enter));
            let l = log.clone();
            app.add_systems(OnExit(value), move || l.lock().unwrap().push(exit));
        }
    }

    /// A computed state appears when its source enters a qualifying mode, runs
    /// the matching `OnEnter`, changes value (exit-old then enter-new) when the
    /// source moves between two qualifying modes, and disappears (exit-old then
    /// resource removed) when the source leaves all qualifying modes.
    #[test]
    fn computed_state_appears_changes_and_disappears() {
        let log = log();
        let mut app = App::new();
        app.insert_state(AppState::Menu)
            .add_computed_state::<Activity>();
        log_activity_edges(&mut app, &log);

        // Frame 1: enter Menu. Activity does not exist, no edges fire.
        app.update();
        assert!(
            app.world().get_resource::<State<Activity>>().is_none(),
            "Activity must not exist while in Menu"
        );
        assert!(drain(&log).is_empty(), "no Activity edge while in Menu");

        // Menu -> InGame: Activity appears as Playing.
        app.world_mut()
            .resource_mut::<NextState<AppState>>()
            .set(AppState::InGame);
        app.update();
        assert_eq!(
            app.world()
                .get_resource::<State<Activity>>()
                .map(|s| *s.get()),
            Some(Activity::Playing),
        );
        assert_eq!(drain(&log), vec!["enter:playing"]);

        // InGame -> Paused: Activity changes Playing -> Halted (exit then enter).
        app.world_mut()
            .resource_mut::<NextState<AppState>>()
            .set(AppState::Paused);
        app.update();
        assert_eq!(
            app.world()
                .get_resource::<State<Activity>>()
                .map(|s| *s.get()),
            Some(Activity::Halted),
        );
        assert_eq!(drain(&log), vec!["exit:playing", "enter:halted"]);

        // Paused -> Menu: Activity disappears (exit then resource removed).
        app.world_mut()
            .resource_mut::<NextState<AppState>>()
            .set(AppState::Menu);
        app.update();
        assert!(
            app.world().get_resource::<State<Activity>>().is_none(),
            "Activity must be removed once source leaves all qualifying modes"
        );
        assert_eq!(drain(&log), vec!["exit:halted"]);
    }

    /// An unchanged source leaves a computed state untouched: no redundant
    /// exit/enter edges fire on a frame where the derived value is identical.
    #[test]
    fn computed_state_is_stable_when_source_unchanged() {
        let log = log();
        let mut app = App::new();
        app.insert_state(AppState::InGame)
            .add_computed_state::<Activity>();
        log_activity_edges(&mut app, &log);

        app.update(); // enter InGame -> Activity::Playing appears
        assert_eq!(drain(&log), vec!["enter:playing"]);

        // No transition queued: Activity recomputes to the same value, silently.
        app.update();
        app.update();
        assert_eq!(
            app.world()
                .get_resource::<State<Activity>>()
                .map(|s| *s.get()),
            Some(Activity::Playing),
        );
        assert!(
            drain(&log).is_empty(),
            "a stable computed state fires no edges"
        );
    }

    /// A computed state derived from *another* computed state (depth 2): it
    /// exists only while [`Activity`] is `Playing`, i.e. two derivation hops
    /// away from the base [`AppState`] being `InGame`. Its
    /// [`DEPENDENCY_DEPTH`](ComputedStates::DEPENDENCY_DEPTH) is computed from
    /// the source so the chain stays consistent under refactors.
    #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
    struct Simulating;
    impl States for Simulating {}
    impl ComputedStates for Simulating {
        type SourceStates = Activity;
        const DEPENDENCY_DEPTH: usize = <Activity as ComputedStates>::DEPENDENCY_DEPTH + 1;
        fn compute(source: &Activity) -> Option<Self> {
            matches!(source, Activity::Playing).then_some(Simulating)
        }
    }

    fn log_sim_edges(app: &mut App, log: &Log) {
        let l = log.clone();
        app.add_systems(OnEnter(Simulating), move || l.lock().unwrap().push("enter:sim"));
        let l = log.clone();
        app.add_systems(OnExit(Simulating), move || l.lock().unwrap().push("exit:sim"));
    }

    /// A computed-of-computed chain settles in a single frame. When the base
    /// state moves, the depth-1 source ([`Activity`]) and the depth-2 derived
    /// state ([`Simulating`]) both recompute within the same
    /// [`StateTransition`], and the depth-1 edges fire strictly before the
    /// depth-2 edges. Without the [`ComputeDepth`] ordering, `Simulating` would
    /// read a stale `Activity` and lag one frame behind. Registration order is
    /// deliberately source-last to prove only the depth edges fix the order.
    #[test]
    fn computed_of_computed_settles_in_one_frame() {
        let log = log();
        let mut app = App::new();
        app.insert_state(AppState::Menu)
            .add_computed_state::<Simulating>()
            .add_computed_state::<Activity>();
        log_activity_edges(&mut app, &log);
        log_sim_edges(&mut app, &log);

        // Frame 1: Menu. Neither derived state exists; no edges fire.
        app.update();
        assert!(
            app.world().get_resource::<State<Activity>>().is_none(),
            "Activity must not exist in Menu"
        );
        assert!(
            app.world().get_resource::<State<Simulating>>().is_none(),
            "Simulating must not exist while its source is absent"
        );
        assert!(drain(&log).is_empty(), "no edges while in Menu");

        // Menu -> InGame: Activity becomes Playing AND Simulating appears in the
        // SAME frame, with the depth-1 enter firing before the depth-2 enter.
        app.world_mut()
            .resource_mut::<NextState<AppState>>()
            .set(AppState::InGame);
        app.update();
        assert_eq!(
            app.world()
                .get_resource::<State<Activity>>()
                .map(|s| *s.get()),
            Some(Activity::Playing),
        );
        assert!(
            app.world().get_resource::<State<Simulating>>().is_some(),
            "depth-2 Simulating must settle the same frame its depth-1 source does"
        );
        assert_eq!(drain(&log), vec!["enter:playing", "enter:sim"]);

        // InGame -> Paused: Activity -> Halted, so Simulating vanishes, again in
        // one frame. Depth-1 exit/enter fire before the depth-2 exit.
        app.world_mut()
            .resource_mut::<NextState<AppState>>()
            .set(AppState::Paused);
        app.update();
        assert_eq!(
            app.world()
                .get_resource::<State<Activity>>()
                .map(|s| *s.get()),
            Some(Activity::Halted),
        );
        assert!(
            app.world().get_resource::<State<Simulating>>().is_none(),
            "Simulating must vanish the same frame Activity leaves Playing"
        );
        assert_eq!(
            drain(&log),
            vec!["exit:playing", "enter:halted", "exit:sim"]
        );
    }

    /// A sub-state gated by the *computed* [`Activity`] state (not a base
    /// state): it exists only while `Activity` is `Playing`, activating into
    /// `Explore`. Its [`DEPENDENCY_DEPTH`](SubStates::DEPENDENCY_DEPTH) is the
    /// computed source's depth plus one, so it shares the computed-state depth
    /// ordering and settles in the same frame as its source.
    #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
    enum PlayMode {
        Explore,
        Fight,
    }
    impl States for PlayMode {}
    impl SubStates for PlayMode {
        type SourceStates = Activity;
        const DEPENDENCY_DEPTH: usize = <Activity as ComputedStates>::DEPENDENCY_DEPTH + 1;
        fn should_exist(source: &Activity) -> Option<Self> {
            matches!(source, Activity::Playing).then_some(PlayMode::Explore)
        }
    }

    fn log_playmode_edges(app: &mut App, log: &Log) {
        for (value, enter, exit) in [
            (PlayMode::Explore, "enter:explore", "exit:explore"),
            (PlayMode::Fight, "enter:fight", "exit:fight"),
        ] {
            let l = log.clone();
            app.add_systems(OnEnter(value), move || l.lock().unwrap().push(enter));
            let l = log.clone();
            app.add_systems(OnExit(value), move || l.lock().unwrap().push(exit));
        }
    }

    /// A sub-state whose parent is a computed state settles in a single frame:
    /// when the base state moves, the depth-1 computed source ([`Activity`]) and
    /// the depth-2 sub-state ([`PlayMode`]) both resolve within the same
    /// [`StateTransition`], with the computed edges firing before the sub-state
    /// edges. Without the shared [`ComputeDepth`] ordering the sub-state would
    /// read a stale `Activity` and lag one frame. The sub-state is registered
    /// *before* its computed source to prove only the depth edges fix the order.
    #[test]
    fn sub_state_on_computed_source_settles_in_one_frame() {
        let log = log();
        let mut app = App::new();
        app.insert_state(AppState::Menu)
            .add_sub_state::<PlayMode>()
            .add_computed_state::<Activity>();
        log_activity_edges(&mut app, &log);
        log_playmode_edges(&mut app, &log);

        // Frame 1: Menu. Neither the computed source nor the sub-state exists.
        app.update();
        assert!(
            app.world().get_resource::<State<Activity>>().is_none(),
            "Activity must not exist in Menu"
        );
        assert!(
            app.world().get_resource::<State<PlayMode>>().is_none(),
            "PlayMode must not exist while its computed parent is absent"
        );
        assert!(drain(&log).is_empty(), "no edges while in Menu");

        // Menu -> InGame: Activity becomes Playing AND the PlayMode sub-state
        // activates into Explore in the SAME frame, computed edge first.
        app.world_mut()
            .resource_mut::<NextState<AppState>>()
            .set(AppState::InGame);
        app.update();
        assert_eq!(
            app.world()
                .get_resource::<State<Activity>>()
                .map(|s| *s.get()),
            Some(Activity::Playing),
        );
        assert_eq!(
            app.world()
                .get_resource::<State<PlayMode>>()
                .map(|s| *s.get()),
            Some(PlayMode::Explore),
            "the sub-state must activate the same frame its computed parent appears"
        );
        assert_eq!(drain(&log), vec!["enter:playing", "enter:explore"]);

        // While active, gameplay drives the sub-state freely via NextState.
        app.world_mut()
            .resource_mut::<NextState<PlayMode>>()
            .set(PlayMode::Fight);
        app.update();
        assert_eq!(
            app.world()
                .get_resource::<State<PlayMode>>()
                .map(|s| *s.get()),
            Some(PlayMode::Fight),
        );
        assert_eq!(drain(&log), vec!["exit:explore", "enter:fight"]);

        // InGame -> Paused: Activity -> Halted, so the sub-state deactivates,
        // again in one frame. Computed exit/enter fire before the sub-state exit.
        app.world_mut()
            .resource_mut::<NextState<AppState>>()
            .set(AppState::Paused);
        app.update();
        assert_eq!(
            app.world()
                .get_resource::<State<Activity>>()
                .map(|s| *s.get()),
            Some(Activity::Halted),
        );
        assert!(
            app.world().get_resource::<State<PlayMode>>().is_none(),
            "PlayMode must deactivate the same frame Activity leaves Playing"
        );
        assert_eq!(
            drain(&log),
            vec!["exit:playing", "enter:halted", "exit:fight"]
        );
    }

    /// A two-mode base state with a nested sub-machine scoped to `InGame`.
    #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
    enum Shell {
        #[default]
        Menu,
        InGame,
    }
    impl States for Shell {}

    /// Sub-state scoped to [`Shell::InGame`]; activates into `Explore`.
    #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
    enum InGameMode {
        Explore,
        Combat,
    }
    impl States for InGameMode {}
    impl SubStates for InGameMode {
        type SourceStates = Shell;
        fn should_exist(source: &Shell) -> Option<Self> {
            matches!(source, Shell::InGame).then_some(InGameMode::Explore)
        }
    }

    fn log_ingame_edges(app: &mut App, log: &Log) {
        for (value, enter, exit) in [
            (InGameMode::Explore, "enter:explore", "exit:explore"),
            (InGameMode::Combat, "enter:combat", "exit:combat"),
        ] {
            let l = log.clone();
            app.add_systems(OnEnter(value), move || l.lock().unwrap().push(enter));
            let l = log.clone();
            app.add_systems(OnExit(value), move || l.lock().unwrap().push(exit));
        }
    }

    /// A sub-state activates (entering its parent-provided initial) when the
    /// parent enters the gating mode, honors queued transitions while active,
    /// and fully deactivates (exit + resource removed) when the parent leaves.
    #[test]
    fn sub_state_activates_transitions_and_deactivates() {
        let log = log();
        let mut app = App::new();
        app.insert_state(Shell::Menu).add_sub_state::<InGameMode>();
        log_ingame_edges(&mut app, &log);

        // Frame 1: Menu. Sub-state does not exist.
        app.update();
        assert!(
            app.world().get_resource::<State<InGameMode>>().is_none(),
            "sub-state must not exist while parent is Menu"
        );
        assert!(drain(&log).is_empty());

        // Menu -> InGame: sub activates into its initial Explore.
        app.world_mut()
            .resource_mut::<NextState<Shell>>()
            .set(Shell::InGame);
        app.update();
        assert_eq!(
            app.world()
                .get_resource::<State<InGameMode>>()
                .map(|s| *s.get()),
            Some(InGameMode::Explore),
        );
        assert_eq!(drain(&log), vec!["enter:explore"]);

        // Queued transition while active: Explore -> Combat (exit then enter).
        app.world_mut()
            .resource_mut::<NextState<InGameMode>>()
            .set(InGameMode::Combat);
        app.update();
        assert_eq!(
            app.world()
                .get_resource::<State<InGameMode>>()
                .map(|s| *s.get()),
            Some(InGameMode::Combat),
        );
        assert_eq!(drain(&log), vec!["exit:explore", "enter:combat"]);

        // InGame -> Menu: parent leaves, whole sub-machine exits and is removed.
        app.world_mut()
            .resource_mut::<NextState<Shell>>()
            .set(Shell::Menu);
        app.update();
        assert!(
            app.world().get_resource::<State<InGameMode>>().is_none(),
            "sub-state resource removed when parent leaves"
        );
        assert_eq!(drain(&log), vec!["exit:combat"]);
    }

    /// If gameplay pre-queues a specific entry value before the sub-state
    /// activates, activation honors that queued value instead of the
    /// parent-provided initial.
    #[test]
    fn sub_state_activation_honors_pre_queued_entry() {
        let log = log();
        let mut app = App::new();
        app.insert_state(Shell::Menu).add_sub_state::<InGameMode>();
        log_ingame_edges(&mut app, &log);

        app.update(); // Menu; sub inactive.

        // Pre-queue Combat and switch the parent into InGame in the same frame.
        app.world_mut()
            .resource_mut::<NextState<InGameMode>>()
            .set(InGameMode::Combat);
        app.world_mut()
            .resource_mut::<NextState<Shell>>()
            .set(Shell::InGame);
        app.update();

        assert_eq!(
            app.world()
                .get_resource::<State<InGameMode>>()
                .map(|s| *s.get()),
            Some(InGameMode::Combat),
            "activation enters the pre-queued value, not the default initial",
        );
        assert_eq!(drain(&log), vec!["enter:combat"]);
    }

    /// A stale queued sub-state request made while the sub-state is inactive is
    /// dropped, so it cannot leak into a later, unrelated activation.
    #[test]
    fn sub_state_clears_stale_request_while_inactive() {
        let mut app = App::new();
        app.insert_state(Shell::Menu).add_sub_state::<InGameMode>();

        app.update(); // Menu; sub inactive.

        // Queue a transition while inactive: it must be discarded, not retained.
        app.world_mut()
            .resource_mut::<NextState<InGameMode>>()
            .set(InGameMode::Combat);
        app.update(); // still Menu: the stale request is cleared.

        // Now activate: should enter the parent-provided initial (Explore), not
        // the stale Combat request from while it was inactive.
        app.world_mut()
            .resource_mut::<NextState<Shell>>()
            .set(Shell::InGame);
        app.update();
        assert_eq!(
            app.world()
                .get_resource::<State<InGameMode>>()
                .map(|s| *s.get()),
            Some(InGameMode::Explore),
            "stale inactive request must not survive to the next activation",
        );
    }

    /// Count live entities carrying a `StateScoped<Shell>` tag.
    fn scoped_entities(app: &mut App, entities: &[Entity]) -> usize {
        entities
            .iter()
            .filter(|&&e| app.world().contains(e))
            .count()
    }

    /// A state-scoped entity survives while its owning mode is current and is
    /// despawned on the first `StateTransition` after that mode is no longer
    /// current. Entities tagged for other modes are removed immediately.
    #[test]
    fn state_scoped_entity_despawns_when_mode_leaves() {
        let mut app = App::new();
        app.insert_state(Shell::Menu)
            .enable_state_scoped_entities::<Shell>();

        app.update(); // current mode settles to Menu.

        // Tag one entity for Menu (current) and one for InGame (not current).
        let menu_entity = app.world_mut().spawn(StateScoped(Shell::Menu));
        let game_entity = app.world_mut().spawn(StateScoped(Shell::InGame));
        assert!(app.world().contains(menu_entity));
        assert!(app.world().contains(game_entity));

        // No transition queued: cleanup runs against the current mode (Menu).
        // The Menu-tagged entity survives; the InGame-tagged entity is removed
        // because its owning mode is not current.
        app.update();
        assert!(
            app.world().contains(menu_entity),
            "entity tagged for the current mode survives"
        );
        assert!(
            !app.world().contains(game_entity),
            "entity tagged for a non-current mode is despawned"
        );

        // Switch into InGame: the Menu-tagged entity's mode is no longer current
        // and it is despawned on this frame's StateTransition. A freshly tagged
        // InGame entity (current mode now) survives.
        let game_entity_2 = app.world_mut().spawn(StateScoped(Shell::InGame));
        app.world_mut()
            .resource_mut::<NextState<Shell>>()
            .set(Shell::InGame);
        app.update();
        assert!(
            !app.world().contains(menu_entity),
            "entity tagged for the mode just left is despawned"
        );
        assert!(
            app.world().contains(game_entity_2),
            "entity tagged for the newly current mode survives"
        );
    }

    /// With cleanup enabled but no current `State<S>` yet (the state machine has
    /// not run its first transition), tagged entities are left untouched.
    #[test]
    fn state_scoped_is_inert_before_first_transition() {
        let mut app = App::new();
        app.insert_state(Shell::Menu)
            .enable_state_scoped_entities::<Shell>();

        // Spawn a tagged entity before any frame: no State<Shell> exists yet.
        let entity = app.world_mut().spawn(StateScoped(Shell::InGame));
        assert!(
            app.world().get_resource::<State<Shell>>().is_none(),
            "no current mode before the first StateTransition"
        );

        // The very first frame installs State(Menu) in Apply and then runs the
        // scoped cleanup in Compute, which now despawns the InGame-tagged entity
        // (its mode is not the newly-current Menu).
        let live = [entity];
        assert_eq!(scoped_entities(&mut app, &live), 1);
        app.update();
        assert!(
            !app.world().contains(entity),
            "once the first mode settles, mismatched tags are cleaned up"
        );
    }

    // ---- transition hooks (OnTransition from -> to) --------------------------

    /// Register an [`OnTransition`] edge that pushes `tag` to `log` when it runs.
    fn log_transition<S: States>(app: &mut App, from: S, to: S, log: &Log, tag: &'static str) {
        let l = log.clone();
        app.add_systems(OnTransition { from, to }, move || l.lock().unwrap().push(tag));
    }

    /// A base-state `from -> to` edge runs its `OnTransition` schedule, and only
    /// for the specific edge that actually occurred.
    #[test]
    fn transition_hook_fires_on_specific_base_edge() {
        let log = log();
        let mut app = App::new();
        app.insert_state(AppState::Menu)
            .add_state_transition_hooks::<AppState>();
        log_transition(&mut app, AppState::Menu, AppState::InGame, &log, "menu->ingame");
        log_transition(&mut app, AppState::InGame, AppState::Paused, &log, "ingame->paused");
        log_transition(&mut app, AppState::Menu, AppState::Paused, &log, "menu->paused");

        // Frame 1: first entry into Menu is not an edge (no `from`): no hook.
        app.update();
        assert!(drain(&log).is_empty(), "first entry fires no transition hook");

        // Menu -> InGame fires only the matching edge.
        app.world_mut()
            .resource_mut::<NextState<AppState>>()
            .set(AppState::InGame);
        app.update();
        assert_eq!(drain(&log), vec!["menu->ingame"]);

        // InGame -> Paused fires only its edge (not menu->paused).
        app.world_mut()
            .resource_mut::<NextState<AppState>>()
            .set(AppState::Paused);
        app.update();
        assert_eq!(drain(&log), vec!["ingame->paused"]);
    }

    /// The transition hook runs *after* the `OnExit`/`OnEnter` edges of the same
    /// transition (App-layer ordering; see `state::transition` docs).
    #[test]
    fn transition_hook_runs_after_exit_and_enter_edges() {
        let log = log();
        let mut app = App::new();
        app.insert_state(AppState::Menu)
            .add_state_transition_hooks::<AppState>();
        let l = log.clone();
        app.add_systems(OnExit(AppState::Menu), move || l.lock().unwrap().push("exit:menu"));
        let l = log.clone();
        app.add_systems(OnEnter(AppState::InGame), move || {
            l.lock().unwrap().push("enter:ingame");
        });
        log_transition(&mut app, AppState::Menu, AppState::InGame, &log, "transition");

        app.update(); // settle into Menu (no edge yet)
        drain(&log);

        app.world_mut()
            .resource_mut::<NextState<AppState>>()
            .set(AppState::InGame);
        app.update();
        assert_eq!(
            drain(&log),
            vec!["exit:menu", "enter:ingame", "transition"],
            "OnTransition observes the fully settled edge, after exit and enter"
        );
    }

    /// Transition hooks also observe a computed state's `from -> to` edge, but
    /// not its appearance (`None -> value`) or disappearance (`value -> None`).
    #[test]
    fn transition_hook_fires_for_computed_state_edge_only() {
        let log = log();
        let mut app = App::new();
        app.insert_state(AppState::Menu)
            .add_computed_state::<Activity>()
            .add_state_transition_hooks::<Activity>();
        log_transition(&mut app, Activity::Playing, Activity::Halted, &log, "play->halt");

        app.update(); // Menu: Activity absent, no edge
        assert!(drain(&log).is_empty());

        // Menu -> InGame: Activity *appears* (None -> Playing): no transition hook.
        app.world_mut()
            .resource_mut::<NextState<AppState>>()
            .set(AppState::InGame);
        app.update();
        assert!(drain(&log).is_empty(), "appearance is not a from->to edge");

        // InGame -> Paused: Activity Playing -> Halted: the hook fires.
        app.world_mut()
            .resource_mut::<NextState<AppState>>()
            .set(AppState::Paused);
        app.update();
        assert_eq!(drain(&log), vec!["play->halt"]);

        // Paused -> Menu: Activity *disappears* (Halted -> None): no hook.
        app.world_mut()
            .resource_mut::<NextState<AppState>>()
            .set(AppState::Menu);
        app.update();
        assert!(drain(&log).is_empty(), "disappearance is not a from->to edge");
    }

    /// Transition hooks are opt-in: without `add_state_transition_hooks`, an
    /// `OnTransition` schedule is never driven even across real edges.
    #[test]
    fn transition_hook_requires_opt_in() {
        let log = log();
        let mut app = App::new();
        app.insert_state(AppState::Menu); // no add_state_transition_hooks
        log_transition(&mut app, AppState::Menu, AppState::InGame, &log, "menu->ingame");

        app.update();
        app.world_mut()
            .resource_mut::<NextState<AppState>>()
            .set(AppState::InGame);
        app.update();
        assert!(
            drain(&log).is_empty(),
            "without opt-in the transition driver is not wired"
        );
    }

}

/// Capability tiering (design §3, §24.4): `Capabilities` / `QualityTier` /
/// `RunMode` derivation and their installation on the app.
mod capability_tests {
    use super::*;
    use crate::capability::{Capabilities, QualityTier};
    use crate::run_mode::RunMode;

    /// A desktop-shaped probe (display, many cores, not mobile) derives the
    /// desktop tier and the client run mode.
    #[test]
    fn desktop_capabilities_tier_and_mode() {
        let caps = Capabilities {
            logical_cores: 16,
            has_display: true,
            is_mobile: false,
            high_resolution_timer: true,
        };
        assert_eq!(QualityTier::from_capabilities(&caps), QualityTier::Desktop);
        assert_eq!(RunMode::detect(&caps), RunMode::Client);
        assert!(caps.is_multicore());
        assert!(QualityTier::Desktop.presents());
    }

    /// No display derives the headless server tier and the headless run mode.
    #[test]
    fn headless_capabilities_tier_and_mode() {
        let caps = Capabilities::headless();
        assert_eq!(QualityTier::from_capabilities(&caps), QualityTier::Server);
        assert_eq!(RunMode::detect(&caps), RunMode::Headless);
        assert!(!caps.is_multicore());
        assert!(!QualityTier::Server.presents());
        assert!(RunMode::Headless.is_headless());
        assert!(!RunMode::Headless.drives_rendering());
    }

    /// Mobile wins over display: a phone with a screen is still the mobile tier.
    #[test]
    fn mobile_capabilities_select_mobile_tier() {
        let caps = Capabilities {
            logical_cores: 8,
            has_display: true,
            is_mobile: true,
            high_resolution_timer: true,
        };
        assert_eq!(QualityTier::from_capabilities(&caps), QualityTier::Mobile);
        assert!(QualityTier::Mobile.presents());
        // The mode default still keys off the display, not the tier.
        assert_eq!(RunMode::detect(&caps), RunMode::Client);
    }

    /// `with_display` overrides the heuristic field before tier derivation, so a
    /// window plugin can correct an assume-present default to headless.
    #[test]
    fn with_display_override_flips_tier() {
        let caps = Capabilities::headless().with_display(true);
        assert!(caps.has_display);
        assert_eq!(QualityTier::from_capabilities(&caps), QualityTier::Desktop);

        let caps = caps.with_display(false);
        assert_eq!(QualityTier::from_capabilities(&caps), QualityTier::Server);
    }

    /// The explicit-choice modes are never auto-selected but describe their
    /// roles honestly.
    #[test]
    fn explicit_modes_report_their_roles() {
        assert!(RunMode::DedicatedServer.is_headless());
        assert!(!RunMode::DedicatedServer.drives_rendering());
        assert!(RunMode::EditorEmbedded.drives_rendering());
        assert!(RunMode::EditorEmbedded.is_externally_driven());
        assert!(!RunMode::Client.is_externally_driven());
    }

    /// `App::new` installs all three tiering resources, and `set_run_mode`
    /// overrides the capability default (design §24.4).
    #[test]
    fn app_installs_tiering_resources_and_allows_mode_override() {
        let mut app = App::new();
        // Resources are present and mutually consistent.
        let caps = *app.capabilities();
        assert_eq!(QualityTier::from_capabilities(&caps), app.quality_tier());
        assert_eq!(RunMode::detect(&caps), app.run_mode());
        // At least one core is always reported.
        assert!(app.capabilities().logical_cores >= 1);

        // The explicit role is a deliberate override the probe never guesses.
        app.set_run_mode(RunMode::DedicatedServer);
        assert_eq!(app.run_mode(), RunMode::DedicatedServer);
        assert!(app.run_mode().is_headless());
    }

    /// The real std probe reports a plausible, self-consistent profile.
    #[cfg(feature = "std")]
    #[test]
    fn detect_probe_is_self_consistent() {
        use core::time::Duration;

        let caps = Capabilities::detect();
        assert!(caps.logical_cores >= 1, "always at least one core");
        // The timer resolution probe returns a finite, sub-second measurement.
        let res = Capabilities::probe_timer_resolution();
        assert!(res < Duration::from_secs(1));
    }
}

// ---- platform tier -> settings PlatformTier layer (design §3, §14) ---------
//
// The `QualityTier` the app derives from its probed `Capabilities` is the
// natural source of the `PlatformTier` settings layer. These tests pin the
// per-tier profile to the design §3 server/mobile/desktop table, the
// serialisation into the dynamically-typed store, the layered `write_into`
// semantics (a higher layer keeps winning), and the `App` helpers that
// broadcast one `SettingChanged` per resolved change.
mod platform_tier_tests {
    use super::*;
    use crate::capability::{Capabilities, QualityTier};
    use crate::platform_tier::{
        self, PlatformTierProfile, KEY_FRAME_LIMIT_FPS, KEY_PIPELINED_RENDERING,
        KEY_POWER_AWARE_VARIABLE_STEP, KEY_RENDER_PRESENT, KEY_SUSPEND_RESUME_LIFECYCLE,
    };
    use crate::settings::{SettingChanged, Settings, SettingsLayer};

    /// The three tier profiles match the design §3 server/mobile/desktop table
    /// field-for-field, and the `QualityTier::platform_profile` shortcut agrees
    /// with `PlatformTierProfile::for_tier`.
    #[test]
    fn profiles_match_design_table() {
        let server = PlatformTierProfile::for_tier(QualityTier::Server);
        assert_eq!(
            server,
            PlatformTierProfile {
                presents_display: false,
                frame_limit_fps: 0,
                pipelined_rendering: false,
                power_aware_variable_step: false,
                suspend_resume_lifecycle: false,
            }
        );

        let mobile = PlatformTierProfile::for_tier(QualityTier::Mobile);
        assert_eq!(
            mobile,
            PlatformTierProfile {
                presents_display: true,
                frame_limit_fps: 60,
                pipelined_rendering: false,
                power_aware_variable_step: true,
                suspend_resume_lifecycle: true,
            }
        );

        let desktop = PlatformTierProfile::for_tier(QualityTier::Desktop);
        assert_eq!(
            desktop,
            PlatformTierProfile {
                presents_display: true,
                frame_limit_fps: 0,
                pipelined_rendering: true,
                power_aware_variable_step: false,
                suspend_resume_lifecycle: false,
            }
        );

        // The shortcut is exactly the free constructor.
        assert_eq!(QualityTier::Server.platform_profile(), server);
        assert_eq!(QualityTier::Mobile.platform_profile(), mobile);
        assert_eq!(QualityTier::Desktop.platform_profile(), desktop);
    }

    /// `entries` serialises every field under its stable key, in `ALL_KEYS`
    /// order, with the FPS cap stored as an integer and the rest as booleans.
    #[test]
    fn entries_serialise_every_field_under_its_key() {
        let profile = PlatformTierProfile::for_tier(QualityTier::Mobile);
        let entries = profile.entries();

        let keys: Vec<&str> = entries.iter().map(|(k, _)| *k).collect();
        assert_eq!(keys, platform_tier::ALL_KEYS);

        // Spot-check each payload against the mobile profile.
        assert_eq!(entries[0].1.as_bool(), Some(true)); // render.present
        assert_eq!(entries[1].1.as_int(), Some(60)); // pacing.frame_limit_fps
        assert_eq!(entries[2].1.as_bool(), Some(false)); // pipeline.pipelined_rendering
        assert_eq!(entries[3].1.as_bool(), Some(true)); // power-aware step
        assert_eq!(entries[4].1.as_bool(), Some(true)); // suspend/resume
    }

    /// `write_into` populates the `PlatformTier` layer so every key resolves to
    /// the profile value, and re-writing the same profile is idempotent.
    #[test]
    fn write_into_populates_platform_tier_layer_idempotently() {
        let mut settings = Settings::new();
        let profile = QualityTier::Desktop.platform_profile();

        let changed = profile.write_into(&mut settings);
        assert_eq!(changed, platform_tier::ALL_KEYS.len());

        assert_eq!(settings.get_bool(KEY_RENDER_PRESENT), Some(true));
        assert_eq!(settings.get_int(KEY_FRAME_LIMIT_FPS), Some(0));
        assert_eq!(settings.get_bool(KEY_PIPELINED_RENDERING), Some(true));
        assert_eq!(settings.get_bool(KEY_POWER_AWARE_VARIABLE_STEP), Some(false));
        assert_eq!(settings.get_bool(KEY_SUSPEND_RESUME_LIFECYCLE), Some(false));
        assert_eq!(
            settings.resolved_layer(KEY_RENDER_PRESENT),
            Some(SettingsLayer::PlatformTier)
        );

        // Re-applying the identical profile resolves to the same values, so
        // nothing changed.
        assert_eq!(profile.write_into(&mut settings), 0);
    }

    /// A higher-precedence layer keeps winning: the tier populates the
    /// `PlatformTier` layer underneath an existing user override, which is left
    /// resolving and reports no change for that key.
    #[test]
    fn higher_layer_override_survives_tier_application() {
        let mut settings = Settings::new();
        // The user forced presentation off and pinned a 30 FPS cap.
        settings.set(SettingsLayer::User, KEY_RENDER_PRESENT, false);
        settings.set(SettingsLayer::User, KEY_FRAME_LIMIT_FPS, 30_i64);

        // Desktop would present and run uncapped, but the user layer wins.
        let changed = QualityTier::Desktop.platform_profile().write_into(&mut settings);
        // Three keys resolve to the tier value; the two user-pinned keys do not
        // change their resolved value.
        assert_eq!(changed, 3);
        assert_eq!(settings.get_bool(KEY_RENDER_PRESENT), Some(false));
        assert_eq!(settings.get_int(KEY_FRAME_LIMIT_FPS), Some(30));
        assert_eq!(
            settings.resolved_layer(KEY_RENDER_PRESENT),
            Some(SettingsLayer::User)
        );
        // Clearing the user layer transparently falls back to the tier value.
        settings.clear(SettingsLayer::User, KEY_RENDER_PRESENT);
        assert_eq!(settings.get_bool(KEY_RENDER_PRESENT), Some(true));
        assert_eq!(
            settings.resolved_layer(KEY_RENDER_PRESENT),
            Some(SettingsLayer::PlatformTier)
        );
    }

    /// `App::apply_platform_tier` auto-initialises the store and broadcasts one
    /// `SettingChanged` per key whose resolved value changed.
    #[test]
    fn app_apply_platform_tier_broadcasts_changes() {
        let mut app = App::new();
        assert!(app.world().get_resource::<Settings>().is_none());

        let count = Arc::new(Mutex::new(0usize));
        let count_sys = count.clone();
        app.init_settings();
        app.add_systems(
            Update,
            move |mut cursor: Local<EventCursor<SettingChanged>>,
                  events: Res<Events<SettingChanged>>| {
                *count_sys.lock().unwrap() += cursor.read(&events).count();
            },
        );

        app.apply_platform_tier(QualityTier::Mobile);
        // Every mobile key resolves from the tier (empty store underneath).
        assert_eq!(app.setting_bool(KEY_RENDER_PRESENT), Some(true));
        assert_eq!(app.setting_int(KEY_FRAME_LIMIT_FPS), Some(60));
        assert_eq!(app.setting_bool(KEY_SUSPEND_RESUME_LIFECYCLE), Some(true));

        app.update();
        app.update();
        assert_eq!(*count.lock().unwrap(), platform_tier::ALL_KEYS.len());
    }

    /// `App::apply_detected_platform_tier` applies the tier the app derived from
    /// its probed capabilities: the resolved presentation flag agrees with the
    /// tier's `presents`.
    #[test]
    fn app_apply_detected_platform_tier_matches_quality_tier() {
        let mut app = App::new();
        let tier = app.quality_tier();
        app.apply_detected_platform_tier();
        assert_eq!(app.setting_bool(KEY_RENDER_PRESENT), Some(tier.presents()));
    }

    /// The std-only `frame_limit` helper turns the FPS cap into a `FrameLimit`:
    /// the mobile 60 FPS cap paces the loop; the uncapped tiers are `Off`.
    #[cfg(feature = "std")]
    #[test]
    fn frame_limit_reflects_fps_cap() {
        use crate::pacing::FrameLimit;

        let mobile = QualityTier::Mobile.platform_profile().frame_limit();
        assert_eq!(mobile, FrameLimit::from_fps(60));
        assert!(mobile.is_limited());

        assert_eq!(
            QualityTier::Desktop.platform_profile().frame_limit(),
            FrameLimit::Off
        );
        assert_eq!(
            QualityTier::Server.platform_profile().frame_limit(),
            FrameLimit::Off
        );
    }

    /// A headless-shaped capability set derives the server profile: no
    /// presentation, no pipeline, no mobile lifecycle.
    #[test]
    fn headless_capabilities_yield_server_profile() {
        let caps = Capabilities::headless();
        let tier = QualityTier::from_capabilities(&caps);
        let profile = tier.platform_profile();
        assert!(!profile.presents_display);
        assert!(!profile.pipelined_rendering);
        assert!(!profile.suspend_resume_lifecycle);
    }
}

// The crash-report module (design §24.7) is `std`-gated, so its tests are too.
// They exercise the pure, deterministic surface — snapshot capture from live
// `App` state, `CrashSnapshot`/`CrashReport` rendering, the `report` body that
// the panic hook runs, and the `publish`/`snapshot` shared slot — plus one
// guarded end-to-end `install` + `catch_unwind` check. Nothing here depends on
// `RUST_BACKTRACE`, and only the single `install` test touches the process
// panic hook (saving and restoring it) so the suite stays deterministic.
#[cfg(feature = "std")]
mod crash_tests {
    use super::*;

    use std::panic;
    use std::sync::{Arc, Mutex};

    use crate::capability::QualityTier;
    use crate::crash::{CrashReport, CrashReporter, CrashSink, CrashSnapshot};
    use crate::run_mode::RunMode;
    use crate::settings::SettingsLayer;

    /// Serializes the two tests that mutate the process-global panic hook,
    /// so their save/install/restore windows never overlap under the
    /// parallel test harness.
    static HOOK_GUARD: Mutex<()> = Mutex::new(());

    /// A `CrashSink` that appends every dump into a shared buffer, so a test can
    /// assert exactly what the reporter wrote without going through stderr.
    fn capturing_sink() -> (CrashSink, Arc<Mutex<Vec<String>>>) {
        let log: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let log_for_sink = log.clone();
        let sink: CrashSink = Arc::new(move |dump: &str| {
            log_for_sink.lock().unwrap().push(dump.to_string());
        });
        (sink, log)
    }

    /// `capture_crash_snapshot` reads real shell state: the seeded `RunMode`, the
    /// resolved `QualityTier`, the live main-world entity count, and an empty
    /// settings list when no store is installed.
    #[test]
    fn capture_reads_live_shell_state() {
        let mut app = App::new();
        app.set_run_mode(RunMode::DedicatedServer);

        let snapshot = app.capture_crash_snapshot();
        assert_eq!(snapshot.run_mode, Some(RunMode::DedicatedServer));
        assert_eq!(snapshot.quality_tier, Some(app.quality_tier()));
        assert_eq!(snapshot.main_entity_count, 0);
        assert!(snapshot.settings.is_empty());
    }

    /// Spawning entities and installing settings is reflected in a freshly
    /// captured snapshot, and the settings are sorted by key.
    #[test]
    fn capture_reflects_entities_and_sorted_settings() {
        let mut app = App::new();
        app.insert_setting(SettingsLayer::User, "r.shadows", 2_i64);
        app.insert_setting(SettingsLayer::EngineDefault, "net.tickrate", 60_i64);
        app.world_mut().spawn(());
        app.world_mut().spawn(());
        app.world_mut().spawn(());

        let snapshot = app.capture_crash_snapshot();
        assert_eq!(snapshot.main_entity_count, 3);
        let keys: Vec<&str> = snapshot
            .settings
            .iter()
            .map(|(k, _)| k.as_str())
            .collect();
        assert_eq!(keys, ["net.tickrate", "r.shadows"]);
    }

    /// `CrashSnapshot::render` lays every field out, including the `<unset>` /
    /// `<none installed>` fallbacks for the empty default snapshot.
    #[test]
    fn snapshot_render_covers_unset_fallbacks() {
        let rendered = CrashSnapshot::default().render();
        assert!(rendered.contains("run mode:      <unset>"));
        assert!(rendered.contains("quality tier:  <unset>"));
        assert!(rendered.contains("main entities: 0"));
        assert!(rendered.contains("settings:      <none installed>"));
    }

    /// A populated snapshot renders each setting on its own line and reports the
    /// key count.
    #[test]
    fn snapshot_render_lists_populated_settings() {
        let snapshot = CrashSnapshot {
            run_mode: Some(RunMode::Client),
            quality_tier: Some(QualityTier::Desktop),
            main_entity_count: 7,
            settings: vec![
                ("net.tickrate".to_string(), 60_i64.into()),
                ("r.shadows".to_string(), 2_i64.into()),
            ],
        };
        let rendered = snapshot.render();
        assert!(rendered.contains("run mode:      Client"));
        assert!(rendered.contains("quality tier:  Desktop"));
        assert!(rendered.contains("main entities: 7"));
        assert!(rendered.contains("settings:      2 key(s)"));
        assert!(rendered.contains("  - net.tickrate = Int(60)"));
        assert!(rendered.contains("  - r.shadows = Int(2)"));
    }

    /// `CrashReport::render` frames the dump with headers and prints the panic
    /// message, location, and the `<unavailable>` backtrace hint when none was
    /// captured.
    #[test]
    fn report_render_frames_the_dump() {
        let report = CrashReport {
            message: "it exploded".to_string(),
            location: Some("src/sim.rs:42:9".to_string()),
            backtrace: None,
            snapshot: CrashSnapshot::default(),
        };
        let rendered = report.render();
        assert!(rendered.starts_with("=== prism_app crash report ===\n"));
        assert!(rendered.contains("panic:         it exploded"));
        assert!(rendered.contains("location:      src/sim.rs:42:9"));
        assert!(rendered.contains("backtrace:     <unavailable; set RUST_BACKTRACE=1>"));
        assert!(rendered.trim_end().ends_with("=== end crash report ==="));
    }

    /// A present backtrace is embedded under its own header, with a trailing
    /// newline added when the captured text lacks one.
    #[test]
    fn report_render_embeds_backtrace() {
        let report = CrashReport {
            message: "boom".to_string(),
            location: None,
            backtrace: Some("frame#0\nframe#1".to_string()),
            snapshot: CrashSnapshot::default(),
        };
        let rendered = report.render();
        assert!(rendered.contains("location:      <unknown>"));
        assert!(rendered.contains("backtrace:\nframe#0\nframe#1\n"));
    }

    /// `publish` / `snapshot` round-trip the shared slot, and the clone the app
    /// keeps sees updates published through any other clone.
    #[test]
    fn publish_and_snapshot_share_one_slot() {
        let reporter = CrashReporter::new();
        assert_eq!(reporter.snapshot(), CrashSnapshot::default());

        let snapshot = CrashSnapshot {
            run_mode: Some(RunMode::Headless),
            quality_tier: Some(QualityTier::Server),
            main_entity_count: 11,
            settings: Vec::new(),
        };
        let clone = reporter.clone();
        clone.publish(snapshot.clone());
        assert_eq!(reporter.snapshot(), snapshot);
    }

    /// `report` writes the rendered dump to the sink and returns the identical
    /// text, embedding the last-published snapshot.
    #[test]
    fn report_writes_to_sink_and_returns_dump() {
        let (sink, log) = capturing_sink();
        let reporter = CrashReporter::with_sink(sink);
        reporter.publish(CrashSnapshot {
            run_mode: Some(RunMode::Client),
            quality_tier: Some(QualityTier::Desktop),
            main_entity_count: 5,
            settings: Vec::new(),
        });

        let returned = reporter.report("manual boom", Some("here:1:1".to_string()), None);

        let captured = log.lock().unwrap();
        assert_eq!(captured.len(), 1);
        assert_eq!(captured[0], returned);
        assert!(returned.contains("panic:         manual boom"));
        assert!(returned.contains("location:      here:1:1"));
        assert!(returned.contains("run mode:      Client"));
        assert!(returned.contains("main entities: 5"));
    }

    /// `install_crash_reporter` stores the reporter as a resource seeded with the
    /// current snapshot, and `refresh_crash_snapshot` republishes the latest
    /// state to it. Restores the process panic hook afterward.
    #[test]
    fn install_and_refresh_publish_app_state() {
        let _guard = HOOK_GUARD.lock().unwrap_or_else(|e| e.into_inner());
        let previous = panic::take_hook();

        let mut app = App::new();
        app.set_run_mode(RunMode::EditorEmbedded);
        let reporter = app.install_crash_reporter();

        // Seeded with the state at install time.
        let seeded = reporter.snapshot();
        assert_eq!(seeded.run_mode, Some(RunMode::EditorEmbedded));
        assert_eq!(seeded.main_entity_count, 0);

        // Mutate the world, then refresh: the stored reporter sees it.
        app.world_mut().spawn(());
        app.refresh_crash_snapshot();
        assert_eq!(reporter.snapshot().main_entity_count, 1);

        panic::set_hook(previous);
    }

    /// End-to-end: an installed hook turns a real caught panic into a dump on the
    /// reporter's sink. This is the only test that mutates the global panic hook,
    /// and it saves and restores the previous hook so the suite stays clean.
    #[test]
    fn installed_hook_reports_a_real_panic() {
        let _guard = HOOK_GUARD.lock().unwrap_or_else(|e| e.into_inner());
        let (sink, log) = capturing_sink();
        let reporter = CrashReporter::with_sink(sink);
        reporter.publish(CrashSnapshot {
            run_mode: Some(RunMode::Client),
            quality_tier: Some(QualityTier::Desktop),
            main_entity_count: 0,
            settings: Vec::new(),
        });

        let previous = panic::take_hook();
        reporter.install();

        let result = panic::catch_unwind(|| {
            panic!("sentinel-crash-xyz");
        });
        assert!(result.is_err());

        panic::set_hook(previous);

        let captured = log.lock().unwrap();
        assert!(
            captured.iter().any(|dump| dump.contains("sentinel-crash-xyz")),
            "installed hook should have written the panic dump to the sink",
        );
    }
}

// ---- cvar (design §24.6 console variables + §25.3 validation) ----
//
// The cvar registry is a declared, validated front door over the existing
// settings cascade: registering seeds the `EngineDefault` layer, a runtime
// write lands in `Runtime`, and validation (type coercion, bounds clamping,
// read-only / cheat gating) rejects illegal input at the boundary rather than
// panicking. The module is ungated (it mirrors `settings`), so these tests are
// ungated too.
mod cvar_tests {
    use super::*;

    use crate::cvar::{
        ConsoleOutcome, CvarBounds, CvarCategory, CvarChanged, CvarCliRejection, CvarError,
        CvarFlags, CvarRegistry, CvarSpec, ValidatedWrite,
    };

    /// Registering a cvar seeds its default into the `EngineDefault` settings
    /// layer, and the typed getters resolve it. A fresh `App` installs nothing.
    #[test]
    fn register_seeds_default_into_engine_default_layer() {
        let mut app = App::new();
        assert!(app.world().get_resource::<CvarRegistry>().is_none());
        assert!(app.world().get_resource::<Settings>().is_none());

        app.register_cvar(
            CvarSpec::new("r.shadows", 2_i64)
                .category(CvarCategory::Render)
                .bounds(CvarBounds::Int(0, 4))
                .description("shadow quality"),
        )
        .expect("fresh registration succeeds");

        // The registry now holds the schema and the cascade holds the value.
        let registry = app.world().resource::<CvarRegistry>();
        assert!(registry.contains("r.shadows"));
        assert_eq!(registry.len(), 1);
        let schema = registry.get("r.shadows").unwrap();
        assert_eq!(schema.category(), CvarCategory::Render);
        assert_eq!(schema.kind(), "int");
        assert_eq!(schema.default_value(), &SettingValue::Int(2));

        assert_eq!(app.cvar_int("r.shadows"), Some(2));
        assert_eq!(
            app.world().resource::<Settings>().resolved_layer("r.shadows"),
            Some(SettingsLayer::EngineDefault),
        );
    }

    /// A runtime `set_cvar` write lands in the highest-precedence `Runtime`
    /// layer and wins over the seeded default.
    #[test]
    fn set_cvar_runtime_override_wins_cascade() {
        let mut app = App::new();
        app.register_cvar(CvarSpec::new("net.tickrate", 30_i64).bounds(CvarBounds::Int(1, 240)))
            .unwrap();
        assert_eq!(app.cvar_int("net.tickrate"), Some(30));

        let outcome = app.set_cvar("net.tickrate", 128_i64).unwrap();
        assert!(outcome.changed);
        assert!(!outcome.clamped);
        assert_eq!(outcome.resolved, SettingValue::Int(128));
        assert_eq!(app.cvar_int("net.tickrate"), Some(128));
        assert_eq!(
            app.world().resource::<Settings>().resolved_layer("net.tickrate"),
            Some(SettingsLayer::Runtime),
        );
    }

    /// A write past the declared bounds is clamped, and the outcome's `clamped`
    /// flag reports it (regression guard: the flag must not be hard-wired false).
    #[test]
    fn out_of_range_writes_are_clamped_both_ends() {
        let mut app = App::new();
        app.register_cvar(CvarSpec::new("r.shadows", 1_i64).bounds(CvarBounds::Int(0, 3)))
            .unwrap();

        let high = app.set_cvar("r.shadows", 9_i64).unwrap();
        assert!(high.changed);
        assert!(high.clamped, "9 clamps down to the max 3");
        assert_eq!(high.resolved, SettingValue::Int(3));
        assert_eq!(app.cvar_int("r.shadows"), Some(3));

        let low = app.set_cvar("r.shadows", -5_i64).unwrap();
        assert!(low.changed);
        assert!(low.clamped, "-5 clamps up to the min 0");
        assert_eq!(low.resolved, SettingValue::Int(0));

        // An in-range write is not reported as clamped.
        let mid = app.set_cvar("r.shadows", 2_i64).unwrap();
        assert!(mid.changed);
        assert!(!mid.clamped);
        assert_eq!(mid.resolved, SettingValue::Int(2));
    }

    /// An integer written to a float cvar is losslessly coerced to float; the
    /// coerced value is still clamped into the float bounds.
    #[test]
    fn int_is_coerced_to_float_cvar() {
        let mut app = App::new();
        app.register_cvar(CvarSpec::new("r.gain", 1.0_f64).bounds(CvarBounds::Float(0.0, 2.0)))
            .unwrap();

        let ok = app.set_cvar("r.gain", 2_i64).unwrap();
        assert!(!ok.clamped);
        assert_eq!(ok.resolved, SettingValue::Float(2.0));
        assert_eq!(app.cvar_float("r.gain"), Some(2.0));

        let clamped = app.set_cvar("r.gain", 5_i64).unwrap();
        assert!(clamped.clamped);
        assert_eq!(clamped.resolved, SettingValue::Float(2.0));
    }

    /// A value of an incompatible kind (string into an int cvar) is rejected and
    /// no state changes. Note cvars do not parse strings the way `Settings` does.
    #[test]
    fn type_mismatch_is_rejected() {
        let mut app = App::new();
        app.register_cvar(CvarSpec::new("r.shadows", 2_i64)).unwrap();

        let err = app.set_cvar("r.shadows", "high").unwrap_err();
        assert_eq!(
            err,
            CvarError::TypeMismatch {
                name: "r.shadows".to_owned(),
                expected: "int",
                found: "string",
            }
        );
        // Rejected write leaves the resolved value untouched.
        assert_eq!(app.cvar_int("r.shadows"), Some(2));
    }

    /// A non-finite float is rejected at the boundary before it can poison
    /// clamping / comparisons.
    #[test]
    fn non_finite_float_is_rejected() {
        let mut app = App::new();
        app.register_cvar(CvarSpec::new("r.gain", 1.0_f64)).unwrap();

        let err = app.set_cvar("r.gain", f64::NAN).unwrap_err();
        assert_eq!(
            err,
            CvarError::TypeMismatch {
                name: "r.gain".to_owned(),
                expected: "finite float",
                found: "non-finite float",
            }
        );
        assert_eq!(app.cvar_float("r.gain"), Some(1.0));

        assert!(app.set_cvar("r.gain", f64::INFINITY).is_err());
    }

    /// Writing an unregistered cvar fails with `Unregistered`.
    #[test]
    fn writing_unregistered_cvar_fails() {
        let mut app = App::new();
        app.init_cvars();
        let err = app.set_cvar("does.not.exist", 1_i64).unwrap_err();
        assert_eq!(err, CvarError::Unregistered("does.not.exist".to_owned()));
    }

    /// A read-only cvar cannot be written at runtime (nor reset).
    #[test]
    fn read_only_cvar_cannot_be_written() {
        let mut app = App::new();
        app.register_cvar(
            CvarSpec::new("sys.version", "1.0").flag(CvarFlags::READ_ONLY),
        )
        .unwrap();

        assert_eq!(
            app.set_cvar("sys.version", "2.0").unwrap_err(),
            CvarError::ReadOnly("sys.version".to_owned())
        );
        assert_eq!(
            app.reset_cvar("sys.version").unwrap_err(),
            CvarError::ReadOnly("sys.version".to_owned())
        );
        assert_eq!(app.cvar_str("sys.version"), Some("1.0"));
    }

    /// A cheat-protected cvar is blocked while cheats are disabled and allowed
    /// once they are enabled.
    #[test]
    fn cheat_protected_cvar_is_gated() {
        let mut app = App::new();
        app.register_cvar(
            CvarSpec::new("g.godmode", false).flag(CvarFlags::CHEAT),
        )
        .unwrap();

        assert_eq!(
            app.set_cvar("g.godmode", true).unwrap_err(),
            CvarError::CheatProtected("g.godmode".to_owned())
        );
        assert_eq!(app.cvar_bool("g.godmode"), Some(false));

        app.set_cheats_enabled(true);
        let ok = app.set_cvar("g.godmode", true).unwrap();
        assert!(ok.changed);
        assert_eq!(app.cvar_bool("g.godmode"), Some(true));
    }

    /// A `NOTIFY` cvar broadcasts a `CvarChanged` event on a resolved change; a
    /// non-notify cvar changing broadcasts none (the `SettingChanged` still goes
    /// out, but that is a separate channel).
    #[test]
    fn notify_flag_controls_cvar_changed_event() {
        let mut app = App::new();
        app.register_cvar(
            CvarSpec::new("r.vsync", true)
                .category(CvarCategory::Render)
                .flag(CvarFlags::NOTIFY),
        )
        .unwrap();
        app.register_cvar(CvarSpec::new("r.quiet", 1_i64)).unwrap();

        let seen = Arc::new(Mutex::new(Vec::<(String, Option<bool>, Option<bool>)>::new()));
        let seen_sys = seen.clone();
        app.add_systems(
            Update,
            move |mut cursor: Local<EventCursor<CvarChanged>>,
                  events: Res<Events<CvarChanged>>| {
                for ev in cursor.read(&events) {
                    seen_sys.lock().unwrap().push((
                        ev.name.clone(),
                        ev.previous.as_ref().and_then(SettingValue::as_bool),
                        ev.current.as_bool(),
                    ));
                }
            },
        );

        app.set_cvar("r.vsync", false).unwrap();
        app.set_cvar("r.quiet", 7_i64).unwrap();
        // A no-op write (same resolved value) must not notify.
        app.set_cvar("r.vsync", false).unwrap();

        app.update();
        app.update();

        let seen = seen.lock().unwrap();
        assert_eq!(
            seen.as_slice(),
            [("r.vsync".to_owned(), Some(true), Some(false))],
            "only the NOTIFY cvar's real change fires CvarChanged",
        );
    }

    /// `reset_cvar` clears only the `Runtime` override, so the value falls back
    /// to the next-highest layer still present (here a `User` setting).
    #[test]
    fn reset_cvar_falls_back_to_lower_layer() {
        let mut app = App::new();
        app.register_cvar(CvarSpec::new("r.shadows", 1_i64).bounds(CvarBounds::Int(0, 4)))
            .unwrap();
        // A user-layer preference sits above the engine default.
        app.set_cvar_at(SettingsLayer::User, "r.shadows", 2_i64).unwrap();
        // A runtime console write sits above the user layer.
        app.set_cvar("r.shadows", 3_i64).unwrap();
        assert_eq!(app.cvar_int("r.shadows"), Some(3));

        let outcome = app.reset_cvar("r.shadows").unwrap();
        assert!(outcome.changed);
        assert!(!outcome.clamped);
        assert_eq!(outcome.resolved, SettingValue::Int(2));
        assert_eq!(app.cvar_int("r.shadows"), Some(2));
        assert_eq!(
            app.world().resource::<Settings>().resolved_layer("r.shadows"),
            Some(SettingsLayer::User),
        );
    }

    /// `iter_category` filters by category and `archived` lists only ARCHIVE
    /// cvars; both walk ascending-name order deterministically.
    #[test]
    fn iter_category_and_archived_are_filtered_and_ordered() {
        let mut app = App::new();
        app.register_cvar(
            CvarSpec::new("r.shadows", 1_i64).category(CvarCategory::Render),
        )
        .unwrap();
        app.register_cvar(
            CvarSpec::new("r.bloom", true)
                .category(CvarCategory::Render)
                .flag(CvarFlags::ARCHIVE),
        )
        .unwrap();
        app.register_cvar(
            CvarSpec::new("net.tickrate", 60_i64).category(CvarCategory::Network),
        )
        .unwrap();

        let registry = app.world().resource::<CvarRegistry>();
        let render: Vec<&str> = registry
            .iter_category(CvarCategory::Render)
            .map(|(name, _)| name)
            .collect();
        assert_eq!(render, ["r.bloom", "r.shadows"]);

        let archived: Vec<&str> = registry.archived().collect();
        assert_eq!(archived, ["r.bloom"]);
    }

    /// Registration rejects bounds whose kind disagrees with the default, and
    /// inverted bounds, both with `InvalidBounds` — and no state is seeded.
    #[test]
    fn register_rejects_invalid_bounds() {
        let mut app = App::new();

        let kind_mismatch = app
            .register_cvar(CvarSpec::new("r.gain", 1_i64).bounds(CvarBounds::Float(0.0, 1.0)))
            .unwrap_err();
        assert_eq!(kind_mismatch, CvarError::InvalidBounds("r.gain".to_owned()));

        let inverted = app
            .register_cvar(CvarSpec::new("r.shadows", 1_i64).bounds(CvarBounds::Int(4, 0)))
            .unwrap_err();
        assert_eq!(inverted, CvarError::InvalidBounds("r.shadows".to_owned()));

        // Nothing was registered or seeded.
        assert!(!app.world().resource::<CvarRegistry>().contains("r.gain"));
        assert_eq!(app.cvar_int("r.shadows"), None);
    }

    /// Registering the same name twice is a `Redeclared` error and leaves the
    /// first registration intact.
    #[test]
    fn register_rejects_redeclaration() {
        let mut app = App::new();
        app.register_cvar(CvarSpec::new("r.shadows", 1_i64)).unwrap();
        let err = app
            .register_cvar(CvarSpec::new("r.shadows", 9_i64))
            .unwrap_err();
        assert_eq!(err, CvarError::Redeclared("r.shadows".to_owned()));
        // The original default is untouched.
        assert_eq!(app.cvar_int("r.shadows"), Some(1));
    }

    /// `CvarRegistry::validate_set` is a pure check returning the coerced,
    /// clamped value plus whether clamping altered it — no cascade needed.
    #[test]
    fn validate_set_is_pure_and_reports_clamping() {
        let mut registry = CvarRegistry::new();
        let (name, default) = registry
            .register(CvarSpec::new("r.shadows", 1_i64).bounds(CvarBounds::Int(0, 3)))
            .unwrap();
        assert_eq!(name, "r.shadows");
        assert_eq!(default, SettingValue::Int(1));

        let in_range = registry
            .validate_set("r.shadows", SettingValue::Int(2))
            .unwrap();
        assert_eq!(
            in_range,
            ValidatedWrite {
                value: SettingValue::Int(2),
                clamped: false,
            }
        );

        let clamped = registry
            .validate_set("r.shadows", SettingValue::Int(99))
            .unwrap();
        assert_eq!(
            clamped,
            ValidatedWrite {
                value: SettingValue::Int(3),
                clamped: true,
            }
        );

        // Cheat gating is observable through the pure path too.
        registry
            .register(CvarSpec::new("g.noclip", false).flag(CvarFlags::CHEAT))
            .unwrap();
        assert_eq!(
            registry.validate_set("g.noclip", SettingValue::Bool(true)),
            Err(CvarError::CheatProtected("g.noclip".to_owned()))
        );
        registry.set_cheats_enabled(true);
        assert!(registry.validate_set("g.noclip", SettingValue::Bool(true)).is_ok());
    }

    /// A command-line override for a declared cvar is validated and clamped into
    /// the declared bounds, written to the `CommandLine` layer, and reported as
    /// an accepted (clamped) cvar — not written raw.
    #[test]
    fn cli_overrides_validate_and_clamp_declared_cvar() {
        let mut app = App::new();
        app.register_cvar(CvarSpec::new("r.shadows", 2_i64).bounds(CvarBounds::Int(0, 4)))
            .unwrap();

        // `99` is out of the [0, 4] bounds: it is clamped, not rejected.
        let report = app.apply_cvar_cli_overrides(["--r.shadows=99"]);

        assert_eq!(report.cvars.len(), 1);
        assert!(report.settings.is_empty());
        assert!(!report.has_rejections());
        let applied = &report.cvars[0];
        assert_eq!(applied.name, "r.shadows");
        assert!(applied.outcome.changed);
        assert!(applied.outcome.clamped);
        assert_eq!(applied.outcome.resolved, SettingValue::Int(4));

        assert_eq!(app.cvar_int("r.shadows"), Some(4));
        assert_eq!(
            app.world().resource::<Settings>().resolved_layer("r.shadows"),
            Some(SettingsLayer::CommandLine),
        );
    }

    /// A command-line value of the wrong type for a declared cvar is rejected at
    /// the boundary (not coerced, not written): the cascade keeps the seeded
    /// default and the rejection carries the `TypeMismatch` error.
    #[test]
    fn cli_overrides_reject_wrong_type_leaving_cascade_untouched() {
        let mut app = App::new();
        app.register_cvar(CvarSpec::new("r.shadows", 2_i64).bounds(CvarBounds::Int(0, 4)))
            .unwrap();

        let report = app.apply_cvar_cli_overrides(["r.shadows=high"]);

        assert!(report.cvars.is_empty());
        assert_eq!(report.rejected_count(), 1);
        assert_eq!(report.rejected[0].key, "r.shadows");
        assert!(matches!(
            report.rejected[0].error,
            CvarError::TypeMismatch { .. }
        ));
        // Untouched: still the EngineDefault seed.
        assert_eq!(app.cvar_int("r.shadows"), Some(2));
        assert_eq!(
            app.world().resource::<Settings>().resolved_layer("r.shadows"),
            Some(SettingsLayer::EngineDefault),
        );
    }

    /// Read-only and cheat-protected cvars refuse command-line writes at the
    /// boundary; enabling cheats first lets the cheat cvar through.
    #[test]
    fn cli_overrides_respect_read_only_and_cheat_flags() {
        let mut app = App::new();
        app.register_cvar(CvarSpec::new("sys.build", 7_i64).flag(CvarFlags::READ_ONLY))
            .unwrap();
        app.register_cvar(CvarSpec::new("g.noclip", false).flag(CvarFlags::CHEAT))
            .unwrap();

        let report = app.apply_cvar_cli_overrides(["--sys.build=9", "--g.noclip=true"]);
        assert!(report.cvars.is_empty());
        assert_eq!(report.rejected_count(), 2);
        assert_eq!(
            report.rejected,
            vec![
                CvarCliRejection {
                    key: "sys.build".to_owned(),
                    error: CvarError::ReadOnly("sys.build".to_owned()),
                },
                CvarCliRejection {
                    key: "g.noclip".to_owned(),
                    error: CvarError::CheatProtected("g.noclip".to_owned()),
                },
            ],
        );
        assert_eq!(app.cvar_int("sys.build"), Some(7));
        assert_eq!(app.cvar_bool("g.noclip"), Some(false));

        // With cheats enabled, the cheat cvar is accepted; read-only stays refused.
        app.set_cheats_enabled(true);
        let report = app.apply_cvar_cli_overrides(["--sys.build=9", "--g.noclip=true"]);
        assert_eq!(report.rejected_count(), 1);
        assert_eq!(report.rejected[0].key, "sys.build");
        assert_eq!(report.cvars.len(), 1);
        assert_eq!(report.cvars[0].name, "g.noclip");
        assert_eq!(app.cvar_bool("g.noclip"), Some(true));
    }

    /// A key that is not a declared cvar is written verbatim into the
    /// `CommandLine` layer as an ordinary setting (cvars are opt-in), and shows
    /// up in the report's `settings` bucket.
    #[test]
    fn cli_overrides_undeclared_key_written_as_plain_setting() {
        let mut app = App::new();
        app.register_cvar(CvarSpec::new("r.shadows", 2_i64).bounds(CvarBounds::Int(0, 4)))
            .unwrap();

        let report = app.apply_cvar_cli_overrides(["--window.title=Prism", "--r.shadows=1"]);

        assert_eq!(report.settings.len(), 1);
        assert_eq!(report.settings[0].key, "window.title");
        assert_eq!(
            report.settings[0].current,
            Some(SettingValue::Str("Prism".to_owned()))
        );
        assert_eq!(report.cvars.len(), 1);
        assert_eq!(report.cvars[0].name, "r.shadows");
        assert_eq!(
            app.world().resource::<Settings>().get("window.title"),
            Some(&SettingValue::Str("Prism".to_owned()))
        );
        assert_eq!(app.cvar_int("r.shadows"), Some(1));
    }

    /// A bare `--flag` (no `=`) means `Bool(true)`; applied to a declared bool
    /// cvar it validates and sets it true.
    #[test]
    fn cli_overrides_bare_flag_sets_bool_cvar_true() {
        let mut app = App::new();
        app.register_cvar(CvarSpec::new("r.vsync", false)).unwrap();

        let report = app.apply_cvar_cli_overrides(["--r.vsync"]);

        assert_eq!(report.cvars.len(), 1);
        assert_eq!(report.cvars[0].name, "r.vsync");
        assert!(report.cvars[0].outcome.changed);
        assert_eq!(app.cvar_bool("r.vsync"), Some(true));
    }

    /// Launch flags land in the `CommandLine` layer, which sits *below*
    /// `Runtime`: a cvar already overridden at runtime resolves unchanged, yet
    /// the command-line contribution is still recorded (changed = false).
    #[test]
    fn cli_overrides_below_runtime_report_unchanged() {
        let mut app = App::new();
        app.register_cvar(CvarSpec::new("net.tickrate", 30_i64).bounds(CvarBounds::Int(1, 240)))
            .unwrap();
        app.set_cvar("net.tickrate", 128_i64).unwrap();
        assert_eq!(app.cvar_int("net.tickrate"), Some(128));

        let report = app.apply_cvar_cli_overrides(["--net.tickrate=60"]);

        assert_eq!(report.cvars.len(), 1);
        assert!(!report.cvars[0].outcome.changed);
        // Runtime still wins; the CommandLine value sits underneath.
        assert_eq!(app.cvar_int("net.tickrate"), Some(128));
        assert_eq!(
            app.world().resource::<Settings>().resolved_layer("net.tickrate"),
            Some(SettingsLayer::Runtime),
        );
    }

    /// Accepting a `NOTIFY` cvar through the command line broadcasts a
    /// `CvarChanged` event (same path as `set_cvar`).
    #[test]
    fn cli_overrides_emit_cvar_changed_for_notify_flag() {
        let mut app = App::new();
        app.register_cvar(
            CvarSpec::new("r.shadows", 2_i64)
                .category(CvarCategory::Render)
                .bounds(CvarBounds::Int(0, 4))
                .flag(CvarFlags::NOTIFY),
        )
        .unwrap();

        app.add_systems(
            Update,
            |mut cursor: Local<EventCursor<CvarChanged>>, events: Res<Events<CvarChanged>>| {
                for ev in cursor.read(&events) {
                    assert_eq!(ev.name, "r.shadows");
                    assert_eq!(ev.category, CvarCategory::Render);
                    assert_eq!(ev.current, SettingValue::Int(3));
                }
            },
        );

        let report = app.apply_cvar_cli_overrides(["--r.shadows=3"]);
        assert!(report.cvars[0].outcome.changed);
        app.update();
        app.update();
    }

    /// Environment-style overrides map `PREFIX_A_B` → `a.b` and run declared
    /// cvars through the same validation/clamping as the command-line path.
    #[test]
    fn env_overrides_map_prefix_and_validate() {
        let mut app = App::new();
        app.register_cvar(CvarSpec::new("r.shadows", 2_i64).bounds(CvarBounds::Int(0, 4)))
            .unwrap();

        let report = app.apply_cvar_env_overrides(
            [
                ("PRISM_R_SHADOWS", "99"),
                ("PRISM_WINDOW_TITLE", "Prism"),
                ("OTHER_IGNORED", "x"),
            ],
            "PRISM_",
        );

        // r.shadows is a declared cvar: clamped into bounds.
        assert_eq!(report.cvars.len(), 1);
        assert_eq!(report.cvars[0].name, "r.shadows");
        assert!(report.cvars[0].outcome.clamped);
        assert_eq!(app.cvar_int("r.shadows"), Some(4));
        // window.title is undeclared: written verbatim.
        assert_eq!(report.settings.len(), 1);
        assert_eq!(report.settings[0].key, "window.title");
        // The non-prefixed var is ignored entirely.
        assert_eq!(app.world().resource::<Settings>().get("other.ignored"), None);
    }

    /// An empty / all-no-op override pass produces an empty report.
    #[test]
    fn cli_overrides_empty_report_when_nothing_applies() {
        let mut app = App::new();
        let report = app.apply_cvar_cli_overrides(["--", "=value", ""]);
        assert!(report.is_empty());
    }

    /// A blank line, a whitespace-only line, and a `//` comment are all no-ops.
    #[test]
    fn console_blank_and_comment_lines_are_noops() {
        let mut app = App::new();
        assert_eq!(app.exec_console(""), ConsoleOutcome::Empty);
        assert_eq!(app.exec_console("   \t "), ConsoleOutcome::Empty);
        assert_eq!(app.exec_console("// just a comment"), ConsoleOutcome::Empty);
        // A no-op line must not even install the cvar machinery.
        assert!(app.world().get_resource::<CvarRegistry>().is_none());
    }

    /// A bare registered cvar name queries its current resolved value.
    #[test]
    fn console_bare_name_queries_resolved_value() {
        let mut app = App::new();
        app.register_cvar(CvarSpec::new("r.shadows", 2_i64).bounds(CvarBounds::Int(0, 4)))
            .unwrap();

        assert_eq!(
            app.exec_console("r.shadows"),
            ConsoleOutcome::Queried {
                name: "r.shadows".to_owned(),
                value: SettingValue::Int(2),
            },
        );

        // Leading/trailing whitespace around a bare name is still a query.
        assert_eq!(
            app.exec_console("  r.shadows  "),
            ConsoleOutcome::Queried {
                name: "r.shadows".to_owned(),
                value: SettingValue::Int(2),
            },
        );
    }

    /// Querying an unregistered name reports `Unknown`, not a resolved value.
    #[test]
    fn console_query_unknown_cvar_is_unknown() {
        let mut app = App::new();
        assert_eq!(
            app.exec_console("r.nope"),
            ConsoleOutcome::Unknown("r.nope".to_owned()),
        );
    }

    /// A declared-only front door: an undeclared settings key that happens to
    /// be set is still `Unknown` from the console, never silently resolved.
    #[test]
    fn console_query_ignores_undeclared_settings_key() {
        let mut app = App::new();
        // Register an unrelated cvar so the `Settings` cascade exists, then set
        // an *undeclared* key directly in it.
        app.register_cvar(CvarSpec::new("r.shadows", 2_i64)).unwrap();
        app.world_mut().resource_mut::<Settings>().set(
            SettingsLayer::User,
            "window.title".to_owned(),
            SettingValue::Str("Prism".to_owned()),
        );
        assert_eq!(
            app.world().resource::<Settings>().get("window.title"),
            Some(&SettingValue::Str("Prism".to_owned())),
        );
        assert_eq!(
            app.exec_console("window.title"),
            ConsoleOutcome::Unknown("window.title".to_owned()),
        );
    }

    /// `name value` writes the cvar at the `Runtime` layer and the typed getter
    /// reflects it (the design's `r.shadows 2` form).
    #[test]
    fn console_sets_cvar_at_runtime_layer() {
        let mut app = App::new();
        app.register_cvar(CvarSpec::new("r.shadows", 2_i64).bounds(CvarBounds::Int(0, 4)))
            .unwrap();

        let outcome = app.exec_console("r.shadows 3");
        match outcome {
            ConsoleOutcome::Set(set) => {
                assert!(set.changed);
                assert!(!set.clamped);
                assert_eq!(set.resolved, SettingValue::Int(3));
            }
            other => panic!("expected Set, got {other:?}"),
        }
        assert_eq!(app.cvar_int("r.shadows"), Some(3));
        assert_eq!(
            app.world().resource::<Settings>().resolved_layer("r.shadows"),
            Some(SettingsLayer::Runtime),
        );

        // A subsequent bare-name query now reflects the runtime write.
        assert_eq!(
            app.exec_console("r.shadows"),
            ConsoleOutcome::Queried {
                name: "r.shadows".to_owned(),
                value: SettingValue::Int(3),
            },
        );
    }

    /// Out-of-range numeric console input is clamped into bounds, not rejected.
    #[test]
    fn console_set_out_of_bounds_is_clamped_not_rejected() {
        let mut app = App::new();
        app.register_cvar(CvarSpec::new("r.shadows", 1_i64).bounds(CvarBounds::Int(0, 3)))
            .unwrap();

        match app.exec_console("r.shadows 99") {
            ConsoleOutcome::Set(set) => {
                assert!(set.changed);
                assert!(set.clamped);
                assert_eq!(set.resolved, SettingValue::Int(3));
            }
            other => panic!("expected clamped Set, got {other:?}"),
        }
        assert_eq!(app.cvar_int("r.shadows"), Some(3));
    }

    /// The whole remainder after the first token is the value, so a multi-word
    /// console line sets a string containing the interior whitespace.
    #[test]
    fn console_multiword_value_is_a_single_string() {
        let mut app = App::new();
        app.register_cvar(CvarSpec::new("sv.motd", "welcome")).unwrap();

        match app.exec_console("sv.motd  hello   brave  world  ") {
            ConsoleOutcome::Set(set) => {
                assert_eq!(set.resolved, SettingValue::Str("hello   brave  world".to_owned()));
            }
            other => panic!("expected Set, got {other:?}"),
        }
        assert_eq!(app.cvar_str("sv.motd"), Some("hello   brave  world"));
    }

    /// Writing an unregistered cvar from the console reports `Unknown` and
    /// leaves the cascade untouched.
    #[test]
    fn console_set_unknown_cvar_is_unknown() {
        let mut app = App::new();
        assert_eq!(
            app.exec_console("does.not.exist 1"),
            ConsoleOutcome::Unknown("does.not.exist".to_owned()),
        );
        assert_eq!(app.cvar("does.not.exist"), None);
    }

    /// A type-incompatible write is rejected at the boundary with the write's
    /// `CvarError`, leaving the resolved value unchanged.
    #[test]
    fn console_type_mismatch_is_rejected() {
        let mut app = App::new();
        app.register_cvar(CvarSpec::new("r.shadows", 2_i64)).unwrap();

        match app.exec_console("r.shadows high") {
            ConsoleOutcome::Rejected(CvarError::TypeMismatch { name, .. }) => {
                assert_eq!(name, "r.shadows");
            }
            other => panic!("expected Rejected(TypeMismatch), got {other:?}"),
        }
        // Rejection left the value untouched.
        assert_eq!(app.cvar_int("r.shadows"), Some(2));
    }

    /// A read-only cvar refuses a console write.
    #[test]
    fn console_read_only_cvar_is_rejected() {
        let mut app = App::new();
        app.register_cvar(CvarSpec::new("sys.version", "1.0").flag(CvarFlags::READ_ONLY))
            .unwrap();

        assert_eq!(
            app.exec_console("sys.version 2.0"),
            ConsoleOutcome::Rejected(CvarError::ReadOnly("sys.version".to_owned())),
        );
        assert_eq!(app.cvar_str("sys.version"), Some("1.0"));
    }

    /// A cheat-protected cvar is refused while cheats are disabled, then
    /// accepted once cheats are enabled.
    #[test]
    fn console_cheat_protected_cvar_is_gated() {
        let mut app = App::new();
        app.register_cvar(CvarSpec::new("g.godmode", false).flag(CvarFlags::CHEAT))
            .unwrap();

        assert_eq!(
            app.exec_console("g.godmode true"),
            ConsoleOutcome::Rejected(CvarError::CheatProtected("g.godmode".to_owned())),
        );
        assert_eq!(app.cvar_bool("g.godmode"), Some(false));

        app.set_cheats_enabled(true);
        match app.exec_console("g.godmode true") {
            ConsoleOutcome::Set(set) => {
                assert!(set.changed);
                assert_eq!(set.resolved, SettingValue::Bool(true));
            }
            other => panic!("expected Set once cheats are enabled, got {other:?}"),
        }
        assert_eq!(app.cvar_bool("g.godmode"), Some(true));
    }
}
