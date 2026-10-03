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

// ---- plugin groups -----------------------------------------------------

use crate::plugin::PluginDependency;
use crate::plugin_graph::PluginGraphError;
use crate::plugin_group::{PluginGroup, PluginGroupBuilder};

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
}
