//! Bevy scheduler integration that drives an [`EcsBridge`] from a real frame
//! loop.
//!
//! The bridge owns `Box<dyn Fn>` reader/writer closures that are **not**
//! `Send + Sync`, so it cannot be stored as an ordinary parallel
//! [`Resource`](bevy_ecs::prelude::Resource). Instead it is installed as a
//! **non-send resource** (via [`World::insert_non_send`]) and driven by two
//! **exclusive systems** (`fn(&mut World)`): the pull pass only needs `&World`,
//! but the push pass needs `&mut World`, and an exclusive system is the only
//! system flavour with mutable access to the whole world.
//!
//! # Why remove-then-reinsert instead of `resource_scope`
//!
//! [`World::resource_scope`](bevy_ecs::prelude::World::resource_scope) is only
//! defined for `R: Resource` and therefore cannot temporarily take out a
//! non-send resource. To obtain both the bridge *and* the rest of the world at
//! once we use the safe `remove -> call -> reinsert` pattern: taking ownership
//! of the bridge releases the world borrow, so the subsequent
//! `pull_all(&World)` / `push_all(&mut World)` call type-checks with no
//! `unsafe` code.
//!
//! # Ordering
//!
//! The two systems are grouped into [`LoomSyncSet::Pull`] and
//! [`LoomSyncSet::Push`]; [`add_loom_sync_systems`] configures those sets so
//! that `Pull` always runs before `Push` within a frame. Pulling first imports
//! fresh ECS state into the signals, user systems react, and pushing last
//! writes the signals back into the components.
//!
//! # Example
//!
//! ```
//! use bevy_ecs::prelude::{Component, Schedule, World};
//! use prism_ui_ecs::schedule::{add_loom_sync_systems, insert_bridge};
//! use prism_ui_ecs::EcsBridge;
//! use prism_ui_reactive::Runtime;
//!
//! #[derive(Component)]
//! struct Counter {
//!     value: i32,
//! }
//!
//! let mut world = World::new();
//! let entity = world.spawn(Counter { value: 1 }).id();
//!
//! let rt = Runtime::new();
//! let signal = rt.signal(0i32);
//!
//! let mut bridge = EcsBridge::new();
//! bridge.bind_two_way::<Counter, i32>(
//!     entity,
//!     signal.clone(),
//!     |c| c.value,
//!     |c, v| c.value = *v,
//! );
//! insert_bridge(&mut world, bridge);
//!
//! let mut schedule = Schedule::default();
//! add_loom_sync_systems(&mut schedule);
//!
//! // Mutate the component before the frame; the pull pass imports it.
//! world.get_mut::<Counter>(entity).unwrap().value = 42;
//! schedule.run(&mut world);
//! assert_eq!(signal.get_untracked(), 42);
//!
//! // Mutate the signal before the next frame; the push pass writes it back.
//! signal.set(7);
//! schedule.run(&mut world);
//! assert_eq!(world.get::<Counter>(entity).unwrap().value, 7);
//! ```

use bevy_ecs::prelude::{IntoScheduleConfigs, Schedule, SystemSet, World};

use crate::bridge::EcsBridge;

/// System set labels for the Loom synchronisation passes.
///
/// Users can order their own systems relative to the bridge by referring to
/// these sets, e.g. a system that reacts to pulled values can be placed
/// `.after(LoomSyncSet::Pull)` and `.before(LoomSyncSet::Push)`.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LoomSyncSet {
    /// The frame-start pass that imports ECS component state into signals.
    Pull,
    /// The frame-end pass that writes signal state back into ECS components.
    Push,
}

/// Installs `bridge` into `world` as a non-send resource.
///
/// This is a thin, discoverable wrapper around [`World::insert_non_send`]; the
/// bridge must be installed before [`loom_pull_system`] or [`loom_push_system`]
/// can act on it. Installing a bridge replaces any previously installed one.
pub fn insert_bridge(world: &mut World, bridge: EcsBridge) {
    world.insert_non_send(bridge);
}

/// Removes the installed [`EcsBridge`] from `world`, if present.
///
/// Returns the owned bridge so callers can inspect or reconfigure it, mirroring
/// [`World::remove_non_send`].
pub fn remove_bridge(world: &mut World) -> Option<EcsBridge> {
    world.remove_non_send::<EcsBridge>()
}

/// Exclusive system that runs the bridge's frame-start pull pass.
///
/// It takes ownership of the installed [`EcsBridge`] (releasing the world
/// borrow), refreshes every bound signal from its component via
/// [`EcsBridge::pull_all`], and reinstalls the bridge. If no bridge is
/// installed the system is a no-op, so it is always safe to schedule.
pub fn loom_pull_system(world: &mut World) {
    if let Some(mut bridge) = world.remove_non_send::<EcsBridge>() {
        bridge.pull_all(world);
        world.insert_non_send(bridge);
    }
}

/// Exclusive system that runs the bridge's frame-end push pass.
///
/// It takes ownership of the installed [`EcsBridge`] (releasing the world
/// borrow), writes every two-way signal back into its component via
/// [`EcsBridge::push_all`], and reinstalls the bridge. If no bridge is
/// installed the system is a no-op, so it is always safe to schedule.
pub fn loom_push_system(world: &mut World) {
    if let Some(bridge) = world.remove_non_send::<EcsBridge>() {
        bridge.push_all(world);
        world.insert_non_send(bridge);
    }
}

/// Adds both Loom sync systems to `schedule` with the correct frame ordering.
///
/// [`loom_pull_system`] is placed in [`LoomSyncSet::Pull`] and
/// [`loom_push_system`] in [`LoomSyncSet::Push`], and the sets are chained so
/// that `Pull` always runs before `Push`. The schedule is returned by mutable
/// reference to allow further chained configuration.
pub fn add_loom_sync_systems(schedule: &mut Schedule) -> &mut Schedule {
    schedule.configure_sets((LoomSyncSet::Pull, LoomSyncSet::Push).chain());
    schedule.add_systems((
        loom_pull_system.in_set(LoomSyncSet::Pull),
        loom_push_system.in_set(LoomSyncSet::Push),
    ));
    schedule
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_ecs::prelude::{Component, Entity, World};
    use prism_ui_reactive::{Runtime, Signal};

    #[derive(Component)]
    struct Counter {
        value: i32,
    }

    #[derive(Component)]
    struct Label {
        text: i32,
    }

    fn setup_two_way(world: &mut World, rt: &Runtime, start: i32) -> (Entity, Signal<i32>) {
        let entity = world.spawn(Counter { value: start }).id();
        let signal = rt.signal(0i32);
        let mut bridge = EcsBridge::new();
        bridge.bind_two_way::<Counter, i32>(
            entity,
            signal.clone(),
            |c| c.value,
            |c, v| c.value = *v,
        );
        insert_bridge(world, bridge);
        (entity, signal)
    }

    #[test]
    fn insert_and_remove_bridge_roundtrip() {
        let mut world = World::new();
        assert!(remove_bridge(&mut world).is_none());
        insert_bridge(&mut world, EcsBridge::new());
        assert!(world.contains_non_send::<EcsBridge>());
        let taken = remove_bridge(&mut world);
        assert!(taken.is_some());
        assert!(!world.contains_non_send::<EcsBridge>());
    }

    #[test]
    fn pull_system_imports_component_into_signal() {
        let mut world = World::new();
        let rt = Runtime::new();
        let (entity, signal) = setup_two_way(&mut world, &rt, 5);
        world.get_mut::<Counter>(entity).unwrap().value = 11;
        loom_pull_system(&mut world);
        assert_eq!(signal.get_untracked(), 11);
    }

    #[test]
    fn pull_system_reinserts_bridge() {
        let mut world = World::new();
        let rt = Runtime::new();
        let _ = setup_two_way(&mut world, &rt, 1);
        loom_pull_system(&mut world);
        assert!(world.contains_non_send::<EcsBridge>());
    }

    #[test]
    fn push_system_writes_signal_into_component() {
        let mut world = World::new();
        let rt = Runtime::new();
        let (entity, signal) = setup_two_way(&mut world, &rt, 0);
        signal.set(99);
        loom_push_system(&mut world);
        assert_eq!(world.get::<Counter>(entity).unwrap().value, 99);
    }

    #[test]
    fn push_system_reinserts_bridge() {
        let mut world = World::new();
        let rt = Runtime::new();
        let _ = setup_two_way(&mut world, &rt, 1);
        loom_push_system(&mut world);
        assert!(world.contains_non_send::<EcsBridge>());
    }

    #[test]
    fn pull_system_without_bridge_is_noop() {
        let mut world = World::new();
        loom_pull_system(&mut world);
        assert!(!world.contains_non_send::<EcsBridge>());
    }

    #[test]
    fn push_system_without_bridge_is_noop() {
        let mut world = World::new();
        loom_push_system(&mut world);
        assert!(!world.contains_non_send::<EcsBridge>());
    }

    #[test]
    fn schedule_runs_pull_then_push_in_one_frame() {
        let mut world = World::new();
        let rt = Runtime::new();
        let (entity, signal) = setup_two_way(&mut world, &rt, 3);
        let mut schedule = Schedule::default();
        add_loom_sync_systems(&mut schedule);

        world.get_mut::<Counter>(entity).unwrap().value = 20;
        schedule.run(&mut world);
        assert_eq!(signal.get_untracked(), 20);
    }

    #[test]
    fn schedule_pushes_signal_change_back() {
        let mut world = World::new();
        let rt = Runtime::new();
        let (entity, signal) = setup_two_way(&mut world, &rt, 0);
        let mut schedule = Schedule::default();
        add_loom_sync_systems(&mut schedule);

        // Prime one frame so the binding's baseline tick is established and the
        // signal matches the component; an unchanged component is then not
        // re-pulled, letting the push pass carry the signal edit through.
        schedule.run(&mut world);
        signal.set(44);
        schedule.run(&mut world);
        assert_eq!(world.get::<Counter>(entity).unwrap().value, 44);
    }

    #[test]
    fn frame_roundtrip_does_not_oscillate() {
        let mut world = World::new();
        let rt = Runtime::new();
        let (entity, signal) = setup_two_way(&mut world, &rt, 0);
        let mut schedule = Schedule::default();
        add_loom_sync_systems(&mut schedule);

        // Prime a baseline frame, then let the signal drive the value; after
        // push the component equals the signal.
        schedule.run(&mut world);
        signal.set(8);
        schedule.run(&mut world);
        assert_eq!(world.get::<Counter>(entity).unwrap().value, 8);
        assert_eq!(signal.get_untracked(), 8);

        // A quiet frame changes nothing in either direction.
        schedule.run(&mut world);
        assert_eq!(world.get::<Counter>(entity).unwrap().value, 8);
        assert_eq!(signal.get_untracked(), 8);
    }

    #[test]
    fn component_change_wins_within_a_frame() {
        let mut world = World::new();
        let rt = Runtime::new();
        let (entity, signal) = setup_two_way(&mut world, &rt, 0);
        let mut schedule = Schedule::default();
        add_loom_sync_systems(&mut schedule);

        // With pull-before-push, a fresh component edit propagates to the
        // signal and the push pass writes the same value back (no change).
        world.get_mut::<Counter>(entity).unwrap().value = 123;
        schedule.run(&mut world);
        assert_eq!(signal.get_untracked(), 123);
        assert_eq!(world.get::<Counter>(entity).unwrap().value, 123);
    }

    #[test]
    fn multiple_entities_sync_independently() {
        let mut world = World::new();
        let rt = Runtime::new();
        let e1 = world.spawn(Counter { value: 1 }).id();
        let e2 = world.spawn(Counter { value: 2 }).id();
        let s1 = rt.signal(0i32);
        let s2 = rt.signal(0i32);
        let mut bridge = EcsBridge::new();
        bridge.bind_two_way::<Counter, i32>(e1, s1.clone(), |c| c.value, |c, v| c.value = *v);
        bridge.bind_two_way::<Counter, i32>(e2, s2.clone(), |c| c.value, |c, v| c.value = *v);
        insert_bridge(&mut world, bridge);

        let mut schedule = Schedule::default();
        add_loom_sync_systems(&mut schedule);

        world.get_mut::<Counter>(e1).unwrap().value = 10;
        world.get_mut::<Counter>(e2).unwrap().value = 20;
        schedule.run(&mut world);
        assert_eq!(s1.get_untracked(), 10);
        assert_eq!(s2.get_untracked(), 20);
    }

    #[test]
    fn multiple_bindings_on_one_entity() {
        let mut world = World::new();
        let rt = Runtime::new();
        let entity = world.spawn((Counter { value: 1 }, Label { text: 2 })).id();
        let counter_sig = rt.signal(0i32);
        let label_sig = rt.signal(0i32);
        let mut bridge = EcsBridge::new();
        bridge.bind_two_way::<Counter, i32>(
            entity,
            counter_sig.clone(),
            |c| c.value,
            |c, v| c.value = *v,
        );
        bridge.bind_two_way::<Label, i32>(
            entity,
            label_sig.clone(),
            |l| l.text,
            |l, v| l.text = *v,
        );
        insert_bridge(&mut world, bridge);

        let mut schedule = Schedule::default();
        add_loom_sync_systems(&mut schedule);

        schedule.run(&mut world);
        counter_sig.set(7);
        label_sig.set(9);
        schedule.run(&mut world);
        assert_eq!(world.get::<Counter>(entity).unwrap().value, 7);
        assert_eq!(world.get::<Label>(entity).unwrap().text, 9);
    }

    #[test]
    fn repeated_frames_are_stable() {
        let mut world = World::new();
        let rt = Runtime::new();
        let (entity, signal) = setup_two_way(&mut world, &rt, 0);
        let mut schedule = Schedule::default();
        add_loom_sync_systems(&mut schedule);

        schedule.run(&mut world);
        signal.set(5);
        for _ in 0..5 {
            schedule.run(&mut world);
        }
        assert_eq!(world.get::<Counter>(entity).unwrap().value, 5);
        assert_eq!(signal.get_untracked(), 5);
    }

    #[test]
    fn remove_reinsert_preserves_binding_count() {
        let mut world = World::new();
        let rt = Runtime::new();
        let _ = setup_two_way(&mut world, &rt, 0);
        loom_pull_system(&mut world);
        let bridge = remove_bridge(&mut world).unwrap();
        assert_eq!(bridge.len(), 1);
    }

    #[test]
    fn one_way_binding_pulls_but_does_not_push() {
        let mut world = World::new();
        let rt = Runtime::new();
        let entity = world.spawn(Counter { value: 4 }).id();
        let signal = rt.signal(0i32);
        let mut bridge = EcsBridge::new();
        bridge.bind::<Counter, i32>(entity, signal.clone(), |c| c.value);
        insert_bridge(&mut world, bridge);

        let mut schedule = Schedule::default();
        add_loom_sync_systems(&mut schedule);

        // Pull imports the component value.
        world.get_mut::<Counter>(entity).unwrap().value = 15;
        schedule.run(&mut world);
        assert_eq!(signal.get_untracked(), 15);

        // Signal edits are not written back for a one-way binding.
        signal.set(1000);
        schedule.run(&mut world);
        assert_eq!(world.get::<Counter>(entity).unwrap().value, 15);
    }

    #[test]
    fn add_loom_sync_systems_returns_schedule() {
        let mut schedule = Schedule::default();
        let returned = add_loom_sync_systems(&mut schedule);
        // The returned reference points at the same schedule and can be reused.
        returned.set_apply_final_deferred(true);
    }
}
