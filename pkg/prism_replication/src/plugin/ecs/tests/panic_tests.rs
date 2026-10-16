use std::panic::{catch_unwind, AssertUnwindSafe};

use super::*;
use crate::transport::memory::InMemoryTransport;
use prism_ecs::world::World;

#[test]
fn bridge_resource_is_restored_when_reconcile_callback_unwinds() {
    let mut world = World::new();
    let entity = world.spawn(Counter(0));
    let (transport, _) = InMemoryTransport::pair_secure();
    let bridge = EcsReplication::new(
        schema(),
        4,
        &seed(),
        ClientStrategy { entity },
        Box::new(transport),
    )
    .expect("test bridge");
    world.insert_resource(bridge);

    let result = catch_unwind(AssertUnwindSafe(|| {
        with_ecs_bridge::<ClientStrategy>(&mut world, |_world, _bridge| {
            panic!("injected callback panic");
        });
    }));

    assert!(result.is_err());
    assert!(world
        .get_resource::<EcsReplication<ClientStrategy>>()
        .is_some());
}
