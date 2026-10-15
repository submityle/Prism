//! Core API coverage for both std and no_std + alloc library builds.
//! Keep this caller no_std too, so it cannot depend on the std prelude.
#![no_std]

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;
use core::time::Duration;
use prism_app::prelude::*;
use prism_app::SnapshotHistory;

#[derive(Default)]
struct Calls(Vec<&'static str>);
impl Resource for Calls {}

struct Foundation;
impl Plugin for Foundation {
    fn build(&self, app: &mut App) {
        app.world_mut().resource_mut::<Calls>().0.push("foundation");
    }
}

struct Dependent;
impl Plugin for Dependent {
    fn dependencies(&self) -> Vec<PluginDependency> {
        vec![PluginDependency::on::<Foundation>()]
    }

    fn build(&self, app: &mut App) {
        app.world_mut().resource_mut::<Calls>().0.push("dependent");
    }
}

struct Group;
impl PluginGroup for Group {
    fn build(self) -> PluginGroupBuilder {
        PluginGroupBuilder::new().add(Dependent).add(Foundation)
    }
}

#[test]
fn plugin_graph_resolves_in_alloc_core_and_preserves_errors() {
    let mut app = App::new();
    app.init_resource::<Calls>().try_add_plugins(Group).unwrap();
    assert_eq!(
        app.world().resource::<Calls>().0,
        ["foundation", "dependent"]
    );
    assert!(matches!(
        PluginGroupBuilder::new().add(Dependent).try_into_plugins(),
        Err(PluginGraphError::MissingDependency { .. })
    ));
    assert!(app.try_finish());
    assert_eq!(app.plugins_state(), PluginsState::Cleaned);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Custom;

#[test]
fn manual_clock_phase_order_and_secondary_world_work_without_platform_services() {
    let mut app = App::new();
    app.init_resource::<Calls>()
        .set_time_update_strategy(TimeUpdateStrategy::ManualDelta(Duration::from_millis(25)))
        .set_fixed_timestep(Duration::from_millis(10))
        .add_systems(Startup, |mut calls: ResMut<Calls>| calls.0.push("startup"))
        .add_systems(FixedUpdate, |mut calls: ResMut<Calls>| {
            calls.0.push("fixed");
        })
        .add_systems(Custom, |mut calls: ResMut<Calls>| calls.0.push("custom"))
        .add_systems(Update, |mut calls: ResMut<Calls>| calls.0.push("update"));
    app.insert_phase_before(Update, Custom).unwrap();

    let mut secondary = SubApp::new();
    secondary.world.insert_resource(Calls::default());
    secondary.set_extract(|main, secondary| {
        secondary.resource_mut::<Calls>().0 = main.resource::<Calls>().0.clone();
    });
    app.insert_sub_app(Custom, secondary);

    assert!(app.try_finish());
    app.update();
    let expected = ["startup", "fixed", "fixed", "custom", "update"];
    assert_eq!(app.world().resource::<Calls>().0, expected);
    assert_eq!(
        app.get_sub_app(Custom).unwrap().world.resource::<Calls>().0,
        expected
    );
    app.world_mut().resource_mut::<Calls>().0.clear();
    app.update();
    assert_eq!(
        app.world().resource::<Calls>().0,
        ["fixed", "fixed", "fixed", "custom", "update"]
    );
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum Mode {
    Menu,
    Playing,
}
impl States for Mode {}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Active;
impl States for Active {}
impl ComputedStates for Active {
    type SourceStates = Mode;
    fn compute(source: &Mode) -> Option<Self> {
        (*source == Mode::Playing).then_some(Self)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Session;
impl States for Session {}
impl SubStates for Session {
    type SourceStates = Active;
    const DEPENDENCY_DEPTH: usize = 2;
    fn should_exist(_: &Active) -> Option<Self> {
        Some(Self)
    }
}

#[test]
fn state_computation_substates_and_scoped_cleanup_remain_available() {
    let mut app = App::new();
    app.insert_state(Mode::Menu)
        .add_computed_state::<Active>()
        .add_sub_state::<Session>()
        .enable_state_scoped_entities::<Mode>();
    let menu = app.world_mut().spawn(StateScoped(Mode::Menu));
    app.update();
    assert!(app.world().contains(menu));
    assert!(app.world().get_resource::<State<Active>>().is_none());

    app.world_mut()
        .resource_mut::<NextState<Mode>>()
        .set(Mode::Playing);
    app.update();
    assert!(!app.world().contains(menu));
    assert_eq!(app.world().resource::<State<Active>>().get(), &Active);
    assert_eq!(app.world().resource::<State<Session>>().get(), &Session);

    app.world_mut()
        .resource_mut::<NextState<Mode>>()
        .set(Mode::Menu);
    app.update();
    assert!(app.world().get_resource::<State<Active>>().is_none());
    assert!(app.world().get_resource::<State<Session>>().is_none());
}

#[test]
fn settings_use_caller_supplied_environment_and_ordered_cascade() {
    let mut settings = Settings::new();
    settings.set(SettingsLayer::EngineDefault, "render.quality", 1_i64);
    settings.apply_env_vars(
        [("PRISM_RENDER_QUALITY", "2"), ("OTHER", "ignored")],
        "PRISM_",
    );
    assert_eq!(settings.get_int("render.quality"), Some(2));
    settings.set(SettingsLayer::Runtime, "render.quality", 3_i64);
    assert_eq!(settings.get_int("render.quality"), Some(3));
    settings.clear(SettingsLayer::Runtime, "render.quality");
    assert_eq!(settings.get_int("render.quality"), Some(2));
    assert!(!settings.contains("other"));
}

#[test]
fn alloc_cvar_math_rounds_half_steps_and_archives_float_kinds() {
    let mut app = App::new();
    app.register_cvar(CvarSpec::new("step", 0_i64)).unwrap();
    for (delta, expected) in [(0.5, 1), (-0.5, -1)] {
        app.set_cvar("step", 0_i64).unwrap();
        app.increment_cvar("step", -10.0, 10.0, delta).unwrap();
        assert_eq!(app.cvar_int("step"), Some(expected));
    }
    for (name, value) in [("whole", 2.0), ("negative_zero", -0.0), ("fraction", -1.25)] {
        app.register_cvar(CvarSpec::new(name, value).flag(CvarFlags::ARCHIVE))
            .unwrap();
    }
    let archive = app.write_archive_config();
    app.set_cvar("whole", 7.25).unwrap();
    app.exec_console_script(&archive);
    assert_eq!(app.cvar("whole"), Some(&SettingValue::Float(2.0)));
    assert_eq!(app.cvar_float("fraction"), Some(-1.25));
    assert_eq!(
        app.cvar_float("negative_zero").unwrap().to_bits(),
        (-0.0_f64).to_bits()
    );
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Counter(i32);
impl Component for Counter {}

#[test]
fn snapshots_capture_restore_and_evict_with_alloc_only() {
    let mut world = World::new();
    world.register_snapshot_component_hashable::<Counter>();
    let entity = world.spawn(Counter(4));
    let mut history = SnapshotHistory::new(2);
    assert_eq!(history.try_capture(&world), Ok(0));
    world.get_mut::<Counter>(entity).unwrap().0 = 9;
    assert!(history.rollback(&mut world, 0));
    assert_eq!(world.get::<Counter>(entity).unwrap().0, 4);
    history.capture(&world);
    history.capture(&world);
    assert!(!history.rollback(&mut world, 0));
}

#[test]
fn platform_free_runners_still_drive_the_core() {
    let mut once = App::new();
    once.set_runner(ScheduleRunnerOnce::run);
    assert_eq!(once.run(), AppExit::Success);
    let mut bounded = App::new();
    bounded.set_runner(|app| HeadlessRunner::with_max_frames(2).run(app));
    assert_eq!(bounded.run(), AppExit::Success);
}

#[cfg(not(feature = "std"))]
#[test]
fn no_std_capabilities_are_conservative_and_time_is_explicit() {
    let mut app = App::new();
    assert_eq!(app.quality_tier(), QualityTier::Server);
    app.update();
    assert_eq!(
        app.world().resource::<EngineClocks>().real().delta(),
        Duration::ZERO
    );
    app.set_time_update_strategy(TimeUpdateStrategy::ManualDelta(Duration::from_millis(10)));
    app.update();
    assert_eq!(
        app.world().resource::<EngineClocks>().real().delta(),
        Duration::from_millis(10)
    );
}

#[cfg(feature = "determinism")]
#[test]
fn rng_input_replay_and_snapshot_reconciliation_remain_alloc_only() {
    use prism_app::{DeterministicRng, FrameHash, InputRecording, ReplayLog, RollbackReplay};

    let mut rng = DeterministicRng::seeded(17);
    let mut recording = InputRecording::recording();
    let mut original = FrameHash::new();
    for input in [2_i32, 3, 5] {
        original.write_i64(i64::from(recording.advance(input)));
        original.write_u64(rng.next_u64());
        original.finalize_frame();
    }
    let (mut rng, mut replay) = ReplayLog::from_recording(17, recording).into_replay();
    let mut replayed = FrameHash::new();
    for _ in 0..3 {
        replayed.write_i64(i64::from(replay.advance(0)));
        replayed.write_u64(rng.next_u64());
        replayed.finalize_frame();
    }
    assert_eq!(original.first_divergence(&replayed), None);

    let mut world = World::new();
    world.register_snapshot_component_hashable::<Counter>();
    let entity = world.spawn(Counter(0));
    let mut session = RollbackReplay::new(4);
    for input in [2_i32, 3] {
        session.commit(&world, input);
        world.get_mut::<Counter>(entity).unwrap().0 += input;
    }
    assert_eq!(
        session.reconcile(&mut world, 0, &[7, 3], |world, input| {
            world.get_mut::<Counter>(entity).unwrap().0 += input;
        }),
        Some(2)
    );
    assert_eq!(world.get::<Counter>(entity).unwrap().0, 10);
}
