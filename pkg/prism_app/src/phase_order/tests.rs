use super::*;
use crate::time::{EngineClocks, TimeUpdateStrategy};
use crate::App;
use prism_ecs::resource::Resource;
use prism_ecs::system::ResMut;
use prism_time::{DefaultSource, Duration};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Custom(u8);

#[derive(Default)]
struct Runs(Vec<&'static str>);
impl Resource for Runs {}

fn app() -> App {
    let mut app = App::empty();
    app.insert_resource(Runs::default());
    app
}

fn record(app: &mut App, label: impl ScheduleLabel + Clone, name: &'static str) {
    app.add_systems(label, move |mut runs: ResMut<Runs>| runs.0.push(name));
}

fn record_sub(sub_app: &mut SubApp, label: impl ScheduleLabel, name: &'static str) {
    let mut schedule = Schedule::new();
    schedule.add_systems(move |mut runs: ResMut<Runs>| runs.0.push(name));
    sub_app
        .world
        .resource_mut::<Schedules>()
        .insert(label, schedule);
}

#[test]
fn default_order_preserves_all_core_phases() {
    let order = PhaseOrder::default();
    assert_eq!(
        order.startup.iter().map(|p| p.name).collect::<Vec<_>>(),
        crate::schedule::STARTUP_PHASES
    );
    assert_eq!(
        order.frame.iter().map(|p| p.name).collect::<Vec<_>>(),
        crate::schedule::FRAME_PHASES
    );
}

#[test]
fn custom_phases_run_before_after_and_at_frame_boundaries() {
    let mut app = app();
    record(&mut app, First, "first");
    record(&mut app, PreUpdate, "pre");
    record(&mut app, StateTransition, "state");
    record(&mut app, Update, "update");
    record(&mut app, PostUpdate, "post");
    record(&mut app, Last, "last");
    // Existing custom schedules must survive registration into the phase order.
    for (id, name) in [(0, "head"), (1, "before"), (2, "after"), (3, "tail")] {
        record(&mut app, Custom(id), name);
    }
    app.insert_phase_before(First, Custom(0)).unwrap();
    app.insert_phase_before(Update, Custom(1)).unwrap();
    app.insert_phase_after(Update, Custom(2)).unwrap();
    app.insert_phase_after(Last, Custom(3)).unwrap();
    for _ in 0..2 {
        app.world_mut().resource_mut::<Runs>().0.clear();
        app.update();
        assert_eq!(
            app.world().resource::<Runs>().0,
            [
                "head", "first", "pre", "state", "before", "update", "after", "post", "last",
                "tail"
            ]
        );
    }
}

#[test]
fn custom_anchors_and_repeated_insertions_have_immediate_position_semantics() {
    let mut app = app();
    app.insert_phase_after(Update, Custom(0)).unwrap();
    app.insert_phase_after(Update, Custom(1)).unwrap();
    app.insert_phase_before(Custom(0), Custom(2)).unwrap();
    app.insert_phase_before(Custom(0), Custom(3)).unwrap();
    app.insert_phase_after(Custom(0), Custom(4)).unwrap();
    // Schedules created by insertion remain configurable through add_systems.
    for (id, name) in [(0, "a"), (1, "b"), (2, "c"), (3, "d"), (4, "e")] {
        record(&mut app, Custom(id), name);
    }
    record(&mut app, Update, "update");
    app.update();
    assert_eq!(
        app.world().resource::<Runs>().0,
        ["update", "b", "c", "d", "a", "e"]
    );
}

#[test]
fn startup_insertions_run_once_and_never_join_the_frame_sequence() {
    let mut app = app();
    record(&mut app, PreStartup, "pre");
    record(&mut app, Startup, "startup");
    record(&mut app, PostStartup, "post");
    app.insert_phase_before(PreStartup, Custom(0)).unwrap();
    app.insert_phase_after(Startup, Custom(1)).unwrap();
    app.insert_phase_after(PostStartup, Custom(2)).unwrap();
    record(&mut app, Custom(0), "head");
    record(&mut app, Custom(1), "middle");
    record(&mut app, Custom(2), "tail");
    record(&mut app, Update, "frame");
    assert!(app.try_finish());
    assert!(app.try_finish());
    app.update();
    app.update();
    assert_eq!(
        app.world().resource::<Runs>().0,
        ["head", "pre", "startup", "middle", "post", "tail", "frame", "frame"]
    );
}

#[test]
fn duplicate_and_missing_anchor_errors_are_atomic_and_ordered() {
    let mut app = app();
    app.insert_phase_before(Startup, Custom(0)).unwrap();
    let count = app.world().resource::<Schedules>().len();
    // A duplicate across startup/frame takes precedence over a missing anchor.
    assert_eq!(
        app.insert_phase_after(Custom(99), Custom(0)).err(),
        Some(PhaseOrderError::DuplicatePhase(BoxedScheduleLabel::new(
            Custom(0)
        )))
    );
    assert_eq!(
        app.insert_phase_before(Custom(99), Custom(1)).err(),
        Some(PhaseOrderError::MissingAnchor(BoxedScheduleLabel::new(
            Custom(99)
        )))
    );
    assert_eq!(app.world().resource::<Schedules>().len(), count);
    assert!(!app.world().resource::<Schedules>().contains(Custom(1)));
    // Failure did not reserve the label or insert an executable phase.
    record(&mut app, Custom(1), "accepted once");
    app.insert_phase_after(Update, Custom(1)).unwrap();
    app.update();
    assert_eq!(app.world().resource::<Runs>().0, ["accepted once"]);
}

#[test]
fn core_labels_cannot_be_reinserted_or_used_at_another_cadence() {
    let mut app = app();
    macro_rules! reject {
        ($($label:expr),* $(,)?) => {$(
            assert_eq!(
                app.insert_phase_before(Update, $label).err(),
                Some(PhaseOrderError::DuplicatePhase(BoxedScheduleLabel::new($label)))
            );
        )*};
    }
    reject!(
        PreStartup,
        Startup,
        PostStartup,
        First,
        RunFixedMainLoop,
        PreUpdate,
        StateTransition,
        Update,
        PostUpdate,
        Last,
        FixedFirst,
        FixedPreUpdate,
        FixedUpdate,
        FixedPostUpdate,
        FixedLast,
        BeforeFixedMainLoop,
        AfterFixedMainLoop,
        Shutdown,
        ExitConfirmation,
    );
    assert!(app.world().resource::<Schedules>().is_empty());
}

#[test]
fn schedules_outside_startup_and_frame_order_are_not_anchors() {
    let mut app = app();
    record(&mut app, Custom(99), "registered but not ordered");
    macro_rules! reject {
        ($($anchor:expr),* $(,)?) => {$(
            assert_eq!(
                app.insert_phase_after($anchor, Custom(0)).err(),
                Some(PhaseOrderError::MissingAnchor(BoxedScheduleLabel::new($anchor)))
            );
        )*};
    }
    reject!(
        FixedUpdate,
        BeforeFixedMainLoop,
        AfterFixedMainLoop,
        Shutdown,
        ExitConfirmation,
        Custom(99)
    );
    assert_eq!(app.world().resource::<Schedules>().len(), 1);
    app.update();
    assert!(app.world().resource::<Runs>().0.is_empty());
}

#[test]
fn custom_phases_bracket_fixed_loop_once_for_zero_one_many_and_capped_steps() {
    for (delta_ms, steps) in [(3, 0), (10, 1), (30, 3), (1000, 8)] {
        let mut app = app();
        app.main_mut().init_time_domain();
        app.set_time_update_strategy(TimeUpdateStrategy::ManualDelta(Duration::from_millis(
            delta_ms,
        )));
        app.set_fixed_timestep_hz(100.0);
        app.insert_phase_before(RunFixedMainLoop, Custom(0))
            .unwrap();
        app.insert_phase_after(RunFixedMainLoop, Custom(1)).unwrap();
        record(&mut app, Custom(0), "outer before");
        record(&mut app, BeforeFixedMainLoop, "hook before");
        record(&mut app, FixedFirst, "fixed first");
        record(&mut app, FixedPreUpdate, "fixed pre");
        record(&mut app, FixedUpdate, "fixed update");
        record(&mut app, FixedPostUpdate, "fixed post");
        record(&mut app, FixedLast, "fixed last");
        record(&mut app, AfterFixedMainLoop, "hook after");
        record(&mut app, Custom(1), "outer after");
        app.update();
        let mut expected = vec!["outer before", "hook before"];
        for _ in 0..steps {
            expected.extend([
                "fixed first",
                "fixed pre",
                "fixed update",
                "fixed post",
                "fixed last",
            ]);
        }
        expected.extend(["hook after", "outer after"]);
        assert_eq!(
            app.world().resource::<Runs>().0,
            expected,
            "delta {delta_ms} ms"
        );
        assert_eq!(
            app.world().resource::<EngineClocks>().source(),
            DefaultSource::Virtual
        );
    }
}

#[test]
fn sub_apps_own_independent_orders() {
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    struct Secondary;
    let mut app = app();
    app.insert_phase_before(Update, Custom(0)).unwrap();
    record(&mut app, Custom(0), "main custom");
    record(&mut app, Update, "main update");
    let mut secondary = SubApp::new();
    secondary.world.insert_resource(Runs::default());
    secondary.insert_phase_after(Update, Custom(0)).unwrap();
    record_sub(&mut secondary, Update, "secondary update");
    record_sub(&mut secondary, Custom(0), "secondary custom");
    app.insert_sub_app(Secondary, secondary);
    app.update();
    assert_eq!(
        app.world().resource::<Runs>().0,
        ["main custom", "main update"]
    );
    assert_eq!(
        app.get_sub_app(Secondary)
            .unwrap()
            .world
            .resource::<Runs>()
            .0,
        ["secondary update", "secondary custom"]
    );
}

#[cfg(feature = "std")]
#[test]
fn instrumented_and_plain_paths_execute_the_same_custom_order() {
    let mut app = app();
    app.insert_phase_after(Update, Custom(0)).unwrap();
    record(&mut app, Update, "update");
    record(&mut app, Custom(0), "custom");
    record(&mut app, Last, "last");
    app.update();
    let plain = app.world().resource::<Runs>().0.clone();
    app.world_mut().resource_mut::<Runs>().0.clear();
    app.init_frame_diagnostics();
    app.update();
    assert_eq!(app.world().resource::<Runs>().0, plain);
    let diag = app.frame_diagnostics().unwrap();
    assert_eq!(diag.phases().count(), 8);
    assert_eq!(
        diag.phase(std::any::type_name::<Custom>())
            .unwrap()
            .total_frames(),
        1
    );
    for name in crate::schedule::FRAME_PHASES {
        assert_eq!(diag.phase(name).unwrap().total_frames(), 1);
    }
}

#[cfg(feature = "std")]
#[test]
fn inserted_frame_phases_keep_budget_gating_and_startup_runs_completely() {
    use crate::budget::FrameScheduleBudget;
    use prism_ecs::schedule::{IntoSystemConfigs, SystemLane};
    use prism_time::FrameBudget;

    for instrumented in [false, true] {
        let mut app = app();
        #[cfg(feature = "multi_thread")]
        app.set_task_pool(prism_tasks::TaskPool::new());
        app.insert_resource(FrameScheduleBudget::new(FrameBudget::new(Duration::ZERO)));
        app.insert_phase_after(Startup, Custom(0)).unwrap();
        app.insert_phase_after(Update, Custom(1)).unwrap();
        app.add_systems(
            Custom(0),
            (|mut runs: ResMut<Runs>| runs.0.push("startup")).lane(SystemLane::Background),
        );
        app.add_systems(
            Custom(1),
            (|mut runs: ResMut<Runs>| runs.0.push("critical")).lane(SystemLane::Critical),
        );
        record(&mut app, Custom(1), "normal");
        app.add_systems(
            Custom(1),
            (|mut runs: ResMut<Runs>| runs.0.push("background")).lane(SystemLane::Background),
        );
        if instrumented {
            app.init_frame_diagnostics();
        }
        assert!(app.try_finish());
        app.update();
        let runs = &app.world().resource::<Runs>().0;
        assert_eq!(runs.len(), 3);
        assert_eq!(runs[0], "startup");
        assert!(runs.contains(&"critical") && runs.contains(&"normal"));
        assert_eq!(
            app.world()
                .resource::<FrameScheduleBudget>()
                .deferred_systems(),
            1
        );
        app.world_mut().resource_mut::<Runs>().0.clear();
        app.world_mut()
            .resource_mut::<FrameScheduleBudget>()
            .set_budget(FrameBudget::new(Duration::from_secs(3600)));
        app.update();
        let runs = &app.world().resource::<Runs>().0;
        assert_eq!(runs.len(), 3);
        assert!(
            runs.contains(&"critical") && runs.contains(&"normal") && runs.contains(&"background")
        );
        assert_eq!(
            app.world()
                .resource::<FrameScheduleBudget>()
                .deferred_systems(),
            0
        );
    }
}

#[cfg(feature = "trace")]
#[test]
fn trace_only_path_includes_custom_phases_in_execution_order() {
    let mut app = app();
    app.insert_phase_before(RunFixedMainLoop, Custom(0))
        .unwrap();
    app.insert_resource(crate::trace::PhaseTrace::new(2));
    app.update();
    let trace = app.world().resource::<crate::trace::PhaseTrace>();
    assert_eq!(trace.frame_count(), 1);
    assert_eq!(
        trace.events().map(|event| event.name).collect::<Vec<_>>(),
        [
            "First",
            std::any::type_name::<Custom>(),
            "RunFixedMainLoop",
            "PreUpdate",
            "StateTransition",
            "Update",
            "PostUpdate",
            "Last"
        ]
    );
}
