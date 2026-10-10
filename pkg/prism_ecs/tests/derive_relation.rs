//! End-to-end integration test for `#[derive(Relation)]` exposed through the
//! `prism_ecs` prelude.
//!
//! The macro-crate unit tests check the *generated token stream*; this test
//! checks that the generated `impl prism_ecs::relation::Relation` compiles and
//! that its `KIND` constant is actually consumed by
//! [`World::register_relation_type`], producing the same registry state and
//! runtime behaviour (exclusive eviction, cascade policy) as a hand-written
//! `register_relation` call with an explicit [`RelationKind`].

use prism_ecs::prelude::*;

/// `ChildOf`: exclusive hierarchy edge whose target deletion cascades.
#[derive(Component, Relation)]
#[relation(fragmenting, exclusive, on_delete_target = "Delete")]
struct ChildOf;

/// `EquippedBy`: all-default relation (non-exclusive, `Remove` on both slots).
#[derive(Component, Relation)]
struct EquippedBy;

/// `LocatedIn`: transitive containment, panic-on-dangling for debugging.
#[derive(Component, Relation)]
#[relation(transitive, on_delete = "Panic")]
struct LocatedIn;

#[test]
fn kind_constant_matches_attributes() {
    // Compare the whole derived `KIND` against the equivalent fluent-builder
    // value so each field is checked without `assert!(<const bool>)` (which
    // clippy flags as an assertion on a constant).
    assert_eq!(
        ChildOf::KIND,
        RelationKind::new()
            .with_fragmenting(true)
            .with_exclusive(true)
            .with_on_delete_target(CleanupPolicy::Delete),
    );

    assert_eq!(EquippedBy::KIND, RelationKind::new());

    assert_eq!(
        LocatedIn::KIND,
        RelationKind::new()
            .with_transitive(true)
            .with_on_delete(CleanupPolicy::Panic),
    );
}

#[test]
fn register_relation_type_installs_kind_in_registry() {
    let mut world = World::new();
    world.register_relation_type::<ChildOf>();
    world.register_relation_type::<EquippedBy>();
    world.register_relation_type::<LocatedIn>();

    let child_of = world.components().id_of::<ChildOf>().unwrap();
    let equipped_by = world.components().id_of::<EquippedBy>().unwrap();
    let located_in = world.components().id_of::<LocatedIn>().unwrap();

    assert!(world.relations().is_registered(child_of));
    assert!(world.relations().is_exclusive(child_of));
    assert!(!world.relations().is_transitive(child_of));
    assert_eq!(world.relations().kind(child_of), Some(&ChildOf::KIND));

    assert!(world.relations().is_registered(equipped_by));
    assert!(!world.relations().is_exclusive(equipped_by));

    assert!(world.relations().is_transitive(located_in));
}

#[test]
fn derived_exclusive_relation_evicts_previous_target() {
    let mut world = World::new();
    world.register_relation_type::<ChildOf>();

    let child = world.spawn(());
    let p1 = world.spawn(());
    let p2 = world.spawn(());

    // First edge establishes the (single) target.
    assert!(world.add_relation::<ChildOf>(child, p1).is_none());
    // Because `ChildOf` derived `exclusive`, adding a second target evicts the
    // first and returns it — identical to the hand-written `with_exclusive`
    // path, but driven entirely by the derived `KIND`.
    assert_eq!(world.add_relation::<ChildOf>(child, p2), Some(p1));
    assert_eq!(world.relation_targets::<ChildOf>(child), &[p2]);
}

#[test]
fn derived_default_relation_is_non_exclusive() {
    let mut world = World::new();
    world.register_relation_type::<EquippedBy>();

    let hero = world.spawn(());
    let sword = world.spawn(());
    let shield = world.spawn(());

    // Non-exclusive: a wielder can hold several equipment edges at once.
    assert!(world.add_relation::<EquippedBy>(hero, sword).is_none());
    assert!(world.add_relation::<EquippedBy>(hero, shield).is_none());
    let targets = world.relation_targets::<EquippedBy>(hero);
    assert!(targets.contains(&sword));
    assert!(targets.contains(&shield));
    assert_eq!(targets.len(), 2);
}
