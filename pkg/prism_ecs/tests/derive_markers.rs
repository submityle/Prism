//! End-to-end integration test for the marker-trait derive macros
//! (`#[derive(Resource)]`, `#[derive(Event)]`) exposed through the
//! `prism_ecs` prelude.
//!
//! The macro-crate unit tests check the *generated token stream*; this test
//! checks that the generated `impl prism_ecs::...` paths actually resolve and
//! satisfy the real trait bounds when the derive is invoked from a downstream
//! crate (here, `prism_ecs` itself via its `prism_ecs` dev-dependency), by
//! driving a live `World` and `Events<E>` with the derived types.

use prism_ecs::prelude::*;

#[derive(Resource)]
struct FrameClock {
    tick: u64,
}

#[derive(Resource, Default)]
struct Score(u32);

#[derive(Resource)]
struct Paused;

#[derive(Event)]
struct Collision {
    a: u32,
    b: u32,
}

#[derive(Event)]
struct AppExit;

#[test]
fn derived_resource_round_trips_through_world() {
    let mut world = World::new();

    // `insert_resource` is bounded `R: Resource`; this only compiles because
    // `#[derive(Resource)]` produced a real `impl Resource for FrameClock`.
    world.insert_resource(FrameClock { tick: 7 });
    world.insert_resource(Paused);
    world.init_resource::<Score>(); // requires `Resource + Default`

    assert_eq!(world.get_resource::<FrameClock>().map(|c| c.tick), Some(7));
    assert_eq!(world.get_resource::<Score>().map(|s| s.0), Some(0));
    assert!(world.get_resource::<Paused>().is_some());

    world.get_resource_mut::<Score>().unwrap().0 = 42;
    assert_eq!(world.resource::<Score>().0, 42);
}

#[test]
fn derived_event_flows_through_double_buffer() {
    // `Events<E>` is bounded `E: Event`; this only compiles because
    // `#[derive(Event)]` produced a real `impl Event for Collision`.
    let mut events: Events<Collision> = Events::new();
    events.send(Collision { a: 1, b: 2 });
    events.send(Collision { a: 3, b: 4 });

    let drained: Vec<(u32, u32)> = events.drain().map(|c| (c.a, c.b)).collect();
    assert_eq!(drained, vec![(1, 2), (3, 4)]);

    // Unit-struct event type is also a valid `Event`.
    let mut exits: Events<AppExit> = Events::new();
    let _id = exits.send(AppExit);
}
