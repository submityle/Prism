//! Behavioural tests for the canonical hierarchy relation and its operations
//! (design §23.6).

use alloc::vec::Vec;

use crate::command::{CommandQueue, Commands};
use crate::component::Component;
use crate::entity::Entity;
use crate::hierarchy::ChildOf;
use crate::world::World;

/// A leaf marker used to give children something to carry.
#[derive(Debug, PartialEq, Clone, Copy)]
struct Part(u32);
impl Component for Part {}

#[test]
fn scoped_children_spawn_and_attach_in_one_scope() {
    let mut world = World::new();
    let root = world.spawn(Part(0));

    let mut queue = CommandQueue::new();
    {
        let mut commands: Commands<'_, '_> = queue.commands(world.entities());
        commands.entity(root).with_children(|children| {
            assert_eq!(children.parent(), root);
            children.spawn(Part(1));
            children.spawn(Part(2));
        });
    }
    queue.apply(&mut world);

    let children = world.relation_sources::<ChildOf>(root);
    assert_eq!(
        children.len(),
        2,
        "both scoped spawns are attached to the parent"
    );
    for child in children {
        assert_eq!(
            world.relation_targets::<ChildOf>(*child),
            [root],
            "each child points at the scope parent"
        );
    }
    // The children really exist with their bundles.
    assert!(children.iter().all(|c| world.contains(*c)));
}

#[test]
fn add_child_attaches_an_existing_entity_and_moves_it_off_its_old_parent() {
    let mut world = World::new();
    let first = world.spawn(Part(0));
    let second = world.spawn(Part(1));
    let child = world.spawn(Part(2));

    let mut queue = CommandQueue::new();
    {
        let mut commands: Commands<'_, '_> = queue.commands(world.entities());
        commands.entity(first).add_child(child);
    }
    queue.apply(&mut world);
    assert_eq!(world.relation_targets::<ChildOf>(child), [first]);

    // `ChildOf` is exclusive: re-parenting evicts the old edge rather than
    // double-parenting (design §23.3).
    let mut queue = CommandQueue::new();
    {
        let mut commands: Commands<'_, '_> = queue.commands(world.entities());
        commands.entity(second).add_child(child);
    }
    queue.apply(&mut world);
    assert_eq!(world.relation_targets::<ChildOf>(child), [second]);
    assert!(world.relation_sources::<ChildOf>(first).is_empty());
}

#[test]
fn despawn_recursive_removes_the_whole_subtree() {
    let mut world = World::new();
    let root = world.spawn(Part(0));
    let child = world.spawn(Part(1));
    let grandchild = world.spawn(Part(2));
    let sibling = world.spawn(Part(3));
    world.add_relation::<ChildOf>(child, root);
    world.add_relation::<ChildOf>(grandchild, child);
    world.add_relation::<ChildOf>(sibling, root);
    let unrelated = world.spawn(Part(4));

    let removed = world.despawn_recursive(root);

    assert_eq!(removed, 4, "root + child + grandchild + sibling");
    for entity in [root, child, grandchild, sibling] {
        assert!(!world.contains(entity), "the subtree must be gone");
    }
    assert!(
        world.contains(unrelated),
        "nothing outside the subtree is touched"
    );
}

#[test]
fn despawning_a_parent_cascades_through_the_delete_policy() {
    let mut world = World::new();
    world.register_child_of();
    let parent = world.spawn(Part(0));
    let child = world.spawn(Part(1));
    let grandchild = world.spawn(Part(2));
    world.add_relation::<ChildOf>(child, parent);
    world.add_relation::<ChildOf>(grandchild, child);

    // `ChildOf` declares `on_delete_target = Delete`, so the relation cascade
    // (design §23.2) removes descendants without any explicit recursion.
    assert!(world.despawn(parent));
    assert!(!world.contains(parent));
    assert!(!world.contains(child), "the cascade follows the child edge");
    assert!(!world.contains(grandchild), "and keeps following it");
}

#[test]
fn despawn_recursive_is_cycle_safe_and_idempotent() {
    let mut world = World::new();
    let a = world.spawn(Part(0));
    let b = world.spawn(Part(1));
    // A malformed cycle: a is b's child and b is a's child.
    world.add_relation::<ChildOf>(a, b);
    world.add_relation::<ChildOf>(b, a);

    assert_eq!(
        world.despawn_recursive(a),
        2,
        "the walk visits each entity once"
    );
    assert!(!world.contains(a) && !world.contains(b));

    // A second call on a dead root is a no-op, not a panic.
    assert_eq!(world.despawn_recursive(a), 0);
}

#[test]
fn scoped_children_nest() {
    let mut world = World::new();
    let root = world.spawn(Part(0));

    let child: Entity;
    let mut queue = CommandQueue::new();
    {
        let mut commands: Commands<'_, '_> = queue.commands(world.entities());
        let mut root_commands = commands.entity(root);
        let mut spawned: Vec<Entity> = Vec::new();
        root_commands.with_children(|children| {
            spawned.push(children.spawn(Part(1)));
        });
        child = spawned[0];
    }
    queue.apply(&mut world);

    // The nested scope attaches to the child, not the root.
    let mut queue = CommandQueue::new();
    {
        let mut commands: Commands<'_, '_> = queue.commands(world.entities());
        commands.entity(child).with_children(|children| {
            children.spawn(Part(2));
        });
    }
    queue.apply(&mut world);

    assert_eq!(world.relation_targets::<ChildOf>(child), [root]);
    assert_eq!(world.relation_sources::<ChildOf>(child).len(), 1);
}
