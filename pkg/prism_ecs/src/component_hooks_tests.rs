//! Behavioural tests for component lifecycle hooks (design §12).
//!
//! Hooks are stateless `fn` pointers, so these tests route their observations
//! through a [`HookLog`] resource that each hook mutates via its
//! [`HookContext::world`](crate::component_hooks::HookContext). This exercises
//! the full trigger contract — ordering, add-vs-overwrite distinction,
//! before-overwrite/before-drop timing, sparse storage, and the hook-free
//! fast path — against the real [`World`] structural operations.

use alloc::vec;
use alloc::vec::Vec;

use crate::component::{Component, StorageType};
use crate::component_hooks::{ComponentHooks, HookContext};
use crate::resource::Resource;
use crate::world::World;

/// Records every hook firing in call order, plus any outgoing value a
/// before-drop hook managed to read.
#[derive(Default)]
struct HookLog {
    /// One entry per hook firing, in the order the world fired them.
    events: Vec<&'static str>,
    /// Values observed by `on_replace` / `on_remove` before the component was
    /// overwritten or dropped. Proves outgoing values are still readable.
    outgoing: Vec<i32>,
}

impl Resource for HookLog {}

impl HookLog {
    fn count(&self, which: &str) -> usize {
        self.events.iter().filter(|e| **e == which).count()
    }
}

/// A table-stored component carrying a payload the hooks read back.
#[derive(Debug, PartialEq)]
struct Tracked(i32);
impl Component for Tracked {}

/// A sparse-stored component to prove hooks fire on the out-of-band path too.
#[derive(Debug, PartialEq)]
struct SparseTracked(i32);
impl Component for SparseTracked {
    const STORAGE: StorageType = StorageType::SparseSet;
}

/// A component with no hooks registered, used to prove the hook-free path is
/// inert.
#[derive(Debug, PartialEq)]
struct Plain(i32);
impl Component for Plain {}

fn on_add(ctx: HookContext<'_>) {
    ctx.world.resource_mut::<HookLog>().events.push("add");
}

fn on_insert(ctx: HookContext<'_>) {
    ctx.world.resource_mut::<HookLog>().events.push("insert");
}

/// `on_replace` fires before the old value is overwritten/dropped, so it must
/// still be able to read the outgoing `Tracked` value.
fn on_replace(ctx: HookContext<'_>) {
    let outgoing = ctx.world.get::<Tracked>(ctx.entity).map(|t| t.0);
    let log = ctx.world.resource_mut::<HookLog>();
    log.events.push("replace");
    if let Some(v) = outgoing {
        log.outgoing.push(v);
    }
}

fn on_remove(ctx: HookContext<'_>) {
    ctx.world.resource_mut::<HookLog>().events.push("remove");
}

fn full_hooks() -> ComponentHooks {
    ComponentHooks::new()
        .with_on_add(on_add)
        .with_on_insert(on_insert)
        .with_on_replace(on_replace)
        .with_on_remove(on_remove)
}

/// Build a world with `HookLog` installed and `Tracked` hooks registered.
fn world_with_tracked_hooks() -> World {
    let mut w = World::new();
    w.init_resource::<HookLog>();
    w.register_component_hooks::<Tracked>(full_hooks());
    w
}

#[test]
fn spawn_fires_add_then_insert_once() {
    let mut w = world_with_tracked_hooks();
    let _e = w.spawn(Tracked(1));
    let log = w.resource::<HookLog>();
    assert_eq!(log.events, vec!["add", "insert"]);
    assert_eq!(log.count("replace"), 0);
    assert_eq!(log.count("remove"), 0);
}

#[test]
fn overwrite_fires_replace_then_insert_not_add() {
    let mut w = world_with_tracked_hooks();
    let e = w.spawn(Tracked(1));
    w.resource_mut::<HookLog>().events.clear();

    assert!(w.insert(e, Tracked(2)));
    let log = w.resource::<HookLog>();
    // Overwrite: replace before the write, insert after; no second add.
    assert_eq!(log.events, vec!["replace", "insert"]);
    assert_eq!(log.count("add"), 0);
    // The outgoing value (1) was readable inside on_replace.
    assert_eq!(log.outgoing, vec![1]);
    // The new value is in place afterwards.
    assert_eq!(w.get::<Tracked>(e), Some(&Tracked(2)));
}

#[test]
fn insert_new_component_on_existing_entity_fires_add() {
    let mut w = world_with_tracked_hooks();
    // Spawn without Tracked, then add it via a structural insert.
    let e = w.spawn(Plain(0));
    assert!(w.insert(e, Tracked(7)));
    let log = w.resource::<HookLog>();
    assert_eq!(log.events, vec!["add", "insert"]);
    assert_eq!(log.count("replace"), 0);
}

#[test]
fn remove_fires_replace_then_remove() {
    let mut w = world_with_tracked_hooks();
    let e = w.spawn(Tracked(9));
    w.resource_mut::<HookLog>().events.clear();
    w.resource_mut::<HookLog>().outgoing.clear();

    assert!(w.remove::<Tracked>(e));
    let log = w.resource::<HookLog>();
    assert_eq!(log.events, vec!["replace", "remove"]);
    // on_replace read the outgoing value before it was dropped.
    assert_eq!(log.outgoing, vec![9]);
    assert_eq!(w.get::<Tracked>(e), None);
}

#[test]
fn despawn_fires_replace_then_remove() {
    let mut w = world_with_tracked_hooks();
    let e = w.spawn(Tracked(42));
    w.resource_mut::<HookLog>().events.clear();
    w.resource_mut::<HookLog>().outgoing.clear();

    assert!(w.despawn(e));
    let log = w.resource::<HookLog>();
    assert_eq!(log.events, vec!["replace", "remove"]);
    assert_eq!(log.outgoing, vec![42]);
    assert!(!w.contains(e));
}

#[test]
fn sparse_component_hooks_fire_on_insert_and_remove() {
    let mut w = World::new();
    w.init_resource::<HookLog>();
    w.register_component_hooks::<SparseTracked>(
        ComponentHooks::new()
            .with_on_add(on_add)
            .with_on_insert(on_insert)
            .with_on_remove(on_remove),
    );

    let e = w.spawn(SparseTracked(3));
    assert_eq!(w.resource::<HookLog>().events, vec!["add", "insert"]);

    w.resource_mut::<HookLog>().events.clear();
    assert!(w.remove::<SparseTracked>(e));
    assert_eq!(w.resource::<HookLog>().events, vec!["remove"]);
}

#[test]
fn hook_free_components_are_inert() {
    let mut w = World::new();
    w.init_resource::<HookLog>();
    // No hooks registered anywhere: the global gate must stay off.
    let e = w.spawn(Plain(1));
    assert!(w.insert(e, Plain(2)));
    assert!(w.remove::<Plain>(e));
    assert!(w.despawn(e));
    assert!(w.resource::<HookLog>().events.is_empty());
}

#[test]
fn hooks_only_fire_for_their_own_component() {
    let mut w = world_with_tracked_hooks();
    // Spawning an unrelated hook-free component must not touch the log, even
    // though some component in the world does have hooks.
    let e = w.spawn(Plain(5));
    assert!(w.resource::<HookLog>().events.is_empty());
    // Adding the hooked component to the same entity fires only its hooks.
    assert!(w.insert(e, Tracked(5)));
    assert_eq!(w.resource::<HookLog>().events, vec!["add", "insert"]);
}
