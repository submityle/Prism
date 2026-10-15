//! Real ECS diagnostics, reflection edits and history through Loom's UI tools.
#![cfg(feature = "ecs")]

use prism_ecs::{
    component::{Component, StorageType},
    component_hooks::{ComponentHooks, HookContext},
    diagnostics::profiler::SpanRecorder,
    prelude::{Query as EcsQuery, Schedule},
    resource::Resource,
    schedule::SingleThreadedExecutor,
    world::World,
};
use prism_reflect::{Patch, Reflect, StableTypeId};
use prism_ui::{RecordingBackend, Ui};
use prism_ui_devtools::{snapshot, OpTrace};
use prism_ui_inspector::{EcsInspectorError, EcsInspectorSession, PerfReport, Query};
use prism_ui_timetravel::EcsHistoryError;

#[derive(Clone, Reflect, Debug, PartialEq, Eq)]
struct Stats {
    hp: i32,
    mana: i32,
}
impl Component for Stats {}

#[derive(Clone, Reflect, Debug, PartialEq, Eq)]
struct Sparse {
    enabled: bool,
}
impl Component for Sparse {
    const STORAGE: StorageType = StorageType::SparseSet;
}

#[derive(Clone, Reflect)]
struct NotSnapshot(i32);
impl Component for NotSnapshot {}

fn setup() -> (World, prism_ecs::entity::Entity, StableTypeId, StableTypeId) {
    let mut world = World::new();
    let stats = world.register_reflect_component::<Stats>();
    let sparse = world.register_reflect_component::<Sparse>();
    world.register_snapshot_component::<Stats>();
    world.register_snapshot_component::<Sparse>();
    let entity = world.spawn((Stats { hp: 7, mana: 3 }, Sparse { enabled: true }));
    (world, entity, stats, sparse)
}

#[test]
fn live_properties_render_through_existing_ui_tools_without_marking_changed() {
    let (mut world, entity, stats, _) = setup();
    let tick = world.get_ref::<Stats>(entity).unwrap().changed_tick();
    world.increment_change_tick();
    let mut session = EcsInspectorSession::new(&world, 8).unwrap();
    session.select(&world, Some(entity)).unwrap();
    let panel = session.inspect(&world).unwrap();
    assert_eq!(panel.world.entity_count, 1);
    assert_eq!(panel.world.total_live_rows(), 1);
    assert_eq!(panel.components.len(), 2);
    let properties = &panel
        .components
        .iter()
        .find(|component| component.type_id == stats)
        .unwrap()
        .properties;
    assert_eq!(properties.children[0].label, "hp");
    assert_eq!(properties.children[0].value.as_deref(), Some("7"));
    let view = panel.element();
    let tree = snapshot(&view);
    assert_eq!(Query::new().with_class("ecs-component").count(&tree), 2);
    assert_eq!(Query::new().text_contains("entities=1").count(&tree), 1);
    assert_eq!(Query::new().text_contains("hp: i32 = 7").count(&tree), 1);
    let mut ui = Ui::new(RecordingBackend::new());
    ui.mount(&view);
    let report = PerfReport::from_trace(&OpTrace::from_ops(ui.backend().ops()));
    assert!(report.creates > 0);
    assert_eq!(
        session
            .read(&world, stats)
            .unwrap()
            .downcast_ref::<Stats>()
            .unwrap()
            .hp,
        7
    );
    assert_eq!(world.get_ref::<Stats>(entity).unwrap().changed_tick(), tick);
}

#[test]
fn reflected_table_and_sparse_edits_remove_undo_redo_and_seek_restore_real_world() {
    let (mut world, entity, stats, sparse) = setup();
    let mut session = EcsInspectorSession::new(&world, 16).unwrap();
    session.select(&world, Some(entity)).unwrap();
    world.increment_change_tick();
    let edit_tick = world.change_tick();
    assert!(session
        .edit_field(&mut world, stats, "hp", &42i32, "Set hp")
        .unwrap());
    assert_eq!(world.get::<Stats>(entity).unwrap().hp, 42);
    assert_eq!(
        world.get_ref::<Stats>(entity).unwrap().changed_tick(),
        edit_tick
    );
    let frame = session.history().current().unwrap();
    assert_eq!(frame.label(), "Set hp");
    assert_eq!(
        Query::new()
            .text_contains("hp: i32 = 42")
            .count(frame.ui().unwrap().snapshot()),
        1
    );
    assert!(session
        .edit_field(&mut world, sparse, "enabled", &false, "Disable")
        .unwrap());
    assert_eq!(
        world.get::<Sparse>(entity),
        Some(&Sparse { enabled: false })
    );
    assert!(session
        .remove_component(&mut world, stats, "Remove stats")
        .unwrap());
    assert!(world.get::<Stats>(entity).is_none());
    assert!(session.undo(&mut world).unwrap());
    assert_eq!(world.get::<Stats>(entity).unwrap().hp, 42);
    assert!(session.undo(&mut world).unwrap());
    assert_eq!(world.get::<Sparse>(entity), Some(&Sparse { enabled: true }));
    assert!(session.redo(&mut world).unwrap());
    assert_eq!(
        world.get::<Sparse>(entity),
        Some(&Sparse { enabled: false })
    );
    session.seek(&mut world, 0).unwrap();
    assert_eq!(world.get::<Stats>(entity).unwrap().hp, 7);
    assert!(session
        .set_component(&mut world, stats, &Stats { hp: 9, mana: 8 }, "Branch")
        .unwrap());
    assert!(!session.history().can_redo());
    assert_eq!(session.history().len(), 2);
}

#[test]
fn invalid_or_noop_edits_preserve_values_ticks_and_redo_even_for_partial_patch() {
    let (mut world, entity, stats, _) = setup();
    let mut session = EcsInspectorSession::new(&world, 8).unwrap();
    session.select(&world, Some(entity)).unwrap();
    session
        .edit_field(&mut world, stats, "hp", &99i32, "Edit")
        .unwrap();
    session.undo(&mut world).unwrap();
    world.increment_change_tick();
    let ticks = world.get_ref::<Stats>(entity).unwrap().changed_tick();
    let patch = Patch::Struct(vec![
        ("hp".into(), Patch::Replace(Box::new(100i32))),
        ("missing".into(), Patch::Replace(Box::new(1i32))),
    ]);
    assert!(matches!(
        session.apply_patch(&mut world, stats, &patch, "Invalid"),
        Err(EcsInspectorError::Patch(_))
    ));
    assert!(matches!(
        session.edit_field(&mut world, stats, "hp", &"wrong".to_string(), "Bad type"),
        Err(EcsInspectorError::Apply(_))
    ));
    assert!(matches!(
        session.edit_field(&mut world, stats, "missing", &1i32, "Bad path"),
        Err(EcsInspectorError::InvalidPath(_))
    ));
    assert!(!session
        .edit_field(&mut world, stats, "hp", &7i32, "No-op")
        .unwrap());
    assert!(!session
        .apply_patch(&mut world, stats, &Patch::Unchanged, "No-op patch")
        .unwrap());
    assert_eq!(world.get::<Stats>(entity), Some(&Stats { hp: 7, mana: 3 }));
    assert_eq!(
        world.get_ref::<Stats>(entity).unwrap().changed_tick(),
        ticks
    );
    assert_eq!(session.history().len(), 2);
    assert!(session.history().can_redo());
}

#[test]
fn missing_snapshot_glue_dead_selection_and_foreign_world_are_recoverable() {
    let (mut world, entity, stats, _) = setup();
    let opaque = world.register_reflect_component::<NotSnapshot>();
    let mut session = EcsInspectorSession::new(&world, 8).unwrap();
    session.select(&world, Some(entity)).unwrap();
    assert!(matches!(
        session.set_component(&mut world, opaque, &NotSnapshot(1), "Unsafe undo"),
        Err(EcsInspectorError::History(
            EcsHistoryError::MissingCloneGlue(_)
        ))
    ));
    assert!(world.get::<NotSnapshot>(entity).is_none());
    assert_eq!(session.history().len(), 1);
    let mut other = World::new();
    let foreign = other.spawn(Stats { hp: 88, mana: 1 });
    assert!(matches!(
        session.select(&other, Some(foreign)),
        Err(EcsInspectorError::History(EcsHistoryError::WrongWorld))
    ));
    assert!(matches!(
        session.undo(&mut other),
        Err(EcsInspectorError::History(EcsHistoryError::WrongWorld))
    ));
    assert_eq!(other.get::<Stats>(foreign).unwrap().hp, 88);
    world.despawn(entity);
    assert!(session.inspect(&world).unwrap().selected.is_none());
    assert!(matches!(
        session.edit_field(&mut world, stats, "hp", &1i32, "Dead"),
        Err(EcsInspectorError::DeadEntity(_))
    ));
}

fn grow(mut query: EcsQuery<&mut Stats>) {
    for mut stats in query.iter_mut() {
        stats.hp += 1;
    }
}

#[test]
fn actual_schedule_profile_and_world_changes_feed_the_panel_and_checkpoint() {
    let (mut world, entity, _, _) = setup();
    let mut session = EcsInspectorSession::new(&world, 8).unwrap();
    session.select(&world, Some(entity)).unwrap();
    let mut schedule = Schedule::new();
    schedule.add_systems(grow);
    schedule.initialize(&mut world);
    let mut recorder = SpanRecorder::new();
    SingleThreadedExecutor::run_instrumented(&mut schedule, &mut world, &mut recorder);
    session.set_profile(recorder.flame_graph());
    let report = session.inspect(&world).unwrap();
    assert_eq!(world.get::<Stats>(entity).unwrap().hp, 8);
    assert_eq!(report.profile.as_ref().unwrap().len(), 1);
    let tree = snapshot(&report.element());
    assert_eq!(Query::new().with_class("ecs-profile").count(&tree), 1);
    assert_eq!(Query::new().text_contains("grow").count(&tree), 1);
    session.checkpoint(&world, "Simulation frame").unwrap();
    session.undo(&mut world).unwrap();
    assert_eq!(world.get::<Stats>(entity).unwrap().hp, 7);
    assert!(session.inspect(&world).unwrap().profile.is_none());
}

#[test]
fn successful_patch_and_absent_component_insertions_are_undoable() {
    let (mut world, entity, stats, sparse) = setup();
    let mut session = EcsInspectorSession::new(&world, 8).unwrap();
    session.select(&world, Some(entity)).unwrap();
    let patch = Patch::Struct(vec![
        ("hp".into(), Patch::Replace(Box::new(12i32))),
        ("mana".into(), Patch::Replace(Box::new(15i32))),
    ]);
    session
        .apply_patch(&mut world, stats, &patch, "Patch")
        .unwrap();
    assert_eq!(
        world.get::<Stats>(entity),
        Some(&Stats { hp: 12, mana: 15 })
    );
    session.undo(&mut world).unwrap();
    assert_eq!(world.get::<Stats>(entity), Some(&Stats { hp: 7, mana: 3 }));
    session.redo(&mut world).unwrap();
    session
        .remove_component(&mut world, sparse, "Remove sparse")
        .unwrap();
    assert!(world.get::<Sparse>(entity).is_none());
    session
        .set_component(
            &mut world,
            sparse,
            &Sparse { enabled: false },
            "Insert sparse",
        )
        .unwrap();
    assert_eq!(
        world.get::<Sparse>(entity),
        Some(&Sparse { enabled: false })
    );
    session.undo(&mut world).unwrap();
    assert!(world.get::<Sparse>(entity).is_none());
    session.undo(&mut world).unwrap();
    assert_eq!(world.get::<Sparse>(entity), Some(&Sparse { enabled: true }));
    session
        .remove_component(&mut world, stats, "Remove table")
        .unwrap();
    session
        .set_component(
            &mut world,
            stats,
            &Stats { hp: 20, mana: 30 },
            "Insert table",
        )
        .unwrap();
    session.undo(&mut world).unwrap();
    assert!(world.get::<Stats>(entity).is_none());
    session.redo(&mut world).unwrap();
    assert_eq!(
        world.get::<Stats>(entity),
        Some(&Stats { hp: 20, mana: 30 })
    );
}

#[derive(Default)]
struct HookCalls(usize);
impl Resource for HookCalls {}

fn count_insert(ctx: HookContext<'_>) {
    ctx.world.resource_mut::<HookCalls>().0 += 1;
}

fn insert_unsnapshotable(ctx: HookContext<'_>) {
    ctx.world.resource_mut::<HookCalls>().0 += 1;
    ctx.world.spawn(NotSnapshot(10));
}

#[test]
fn edits_fire_hooks_but_seek_does_not_replay_them() {
    let (mut world, entity, stats, _) = setup();
    world.insert_resource(HookCalls::default());
    world.register_component_hooks::<Stats>(ComponentHooks::new().with_on_insert(count_insert));
    let mut session = EcsInspectorSession::new(&world, 8).unwrap();
    session.select(&world, Some(entity)).unwrap();
    session
        .edit_field(&mut world, stats, "hp", &20i32, "Edit")
        .unwrap();
    assert_eq!(world.resource::<HookCalls>().0, 1);
    session.undo(&mut world).unwrap();
    assert_eq!(world.get::<Stats>(entity).unwrap().hp, 7);
    session.redo(&mut world).unwrap();
    assert_eq!(world.get::<Stats>(entity).unwrap().hp, 20);
    assert_eq!(world.resource::<HookCalls>().0, 1);
}

#[test]
fn failed_post_edit_capture_restores_world_and_preserves_redo() {
    let (mut world, entity, stats, _) = setup();
    let mut session = EcsInspectorSession::new(&world, 8).unwrap();
    session.select(&world, Some(entity)).unwrap();
    session
        .edit_field(&mut world, stats, "hp", &50i32, "Edit")
        .unwrap();
    session.undo(&mut world).unwrap();
    let tick = world.get_ref::<Stats>(entity).unwrap().changed_tick();
    world.insert_resource(HookCalls::default());
    world.register_component_hooks::<Stats>(
        ComponentHooks::new().with_on_insert(insert_unsnapshotable),
    );
    assert!(matches!(
        session.edit_field(&mut world, stats, "hp", &30i32, "Unrecordable"),
        Err(EcsInspectorError::History(
            EcsHistoryError::MissingCloneGlue(_)
        ))
    ));
    assert_eq!(world.entity_count(), 1);
    assert_eq!(world.get::<Stats>(entity).unwrap().hp, 7);
    assert_eq!(world.get_ref::<Stats>(entity).unwrap().changed_tick(), tick);
    assert_eq!(session.history().len(), 2);
    assert_eq!(session.history().cursor(), Some(0));
    assert!(session.history().can_redo());
    // External resources are deliberately not rolled back; neither rollback
    // nor later navigation replays the hook.
    assert_eq!(world.resource::<HookCalls>().0, 1);
    session.redo(&mut world).unwrap();
    assert_eq!(world.get::<Stats>(entity).unwrap().hp, 50);
    assert_eq!(world.resource::<HookCalls>().0, 1);
}

fn replace_with_empty_world(ctx: HookContext<'_>) {
    *ctx.world = World::new();
}

fn replace_with_nonempty_world(ctx: HookContext<'_>) {
    let mut replacement = World::new();
    replacement.spawn(NotSnapshot(77));
    *ctx.world = replacement;
}

#[test]
fn hook_replacing_world_never_restores_source_history_into_replacement() {
    for hook in [
        replace_with_empty_world as prism_ecs::component_hooks::ComponentHook,
        replace_with_nonempty_world,
    ] {
        let (mut world, entity, stats, _) = setup();
        let original_id = world.id();
        let mut session = EcsInspectorSession::new(&world, 8).unwrap();
        session.select(&world, Some(entity)).unwrap();
        world.register_component_hooks::<Stats>(ComponentHooks::new().with_on_insert(hook));
        assert!(matches!(
            session.edit_field(&mut world, stats, "hp", &30i32, "Replaced"),
            Err(EcsInspectorError::History(EcsHistoryError::WrongWorld))
        ));
        assert_ne!(world.id(), original_id);
        assert_eq!(session.history().len(), 1);
        assert_eq!(session.history().cursor(), Some(0));
        // Identical generational handles may exist in the replacement, but it
        // must never receive the source World's Stats/Sparse checkpoints.
        assert!(world.get::<Stats>(entity).is_none());
        assert!(world.get::<Sparse>(entity).is_none());
        assert!(matches!(
            session.seek(&mut world, 0),
            Err(EcsInspectorError::History(EcsHistoryError::WrongWorld))
        ));
    }
}
