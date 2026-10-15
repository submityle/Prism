//! Send-safe invalidation routes. Observers hold weak routes, never Signals or
//! field closures. The World owns removal tokens; the bridge owns their lease.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex, MutexGuard, Weak};

use prism_ecs::observer::{LifecycleEvent, ObserverId};
use prism_ecs::prelude::{Component, Entity, Resource, World};

use crate::bridge::BindingId;

#[derive(Default)]
pub(crate) struct Routes {
    pub(crate) entities: BTreeMap<Entity, BTreeSet<BindingId>>,
    pub(crate) dirty: BTreeSet<BindingId>,
}

impl Routes {
    fn notify(&mut self, entity: Entity) {
        if let Some(bindings) = self.entities.get(&entity) {
            self.dirty.extend(bindings);
        }
    }
}

pub(crate) fn lock(routes: &Mutex<Routes>) -> MutexGuard<'_, Routes> {
    // A user projection panic never runs under this lock. Recovering poison
    // still allows detach/cleanup after an unwinding allocation or callback.
    routes
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

pub(crate) struct Subscription {
    pub(crate) routes: Arc<Mutex<Routes>>,
    installed: bool,
    register: fn(&mut World, &Arc<Mutex<Routes>>),
}

impl Subscription {
    pub(crate) fn new<C: Component>() -> Self {
        Self {
            routes: Arc::new(Mutex::new(Routes::default())),
            installed: false,
            register: register::<C>,
        }
    }

    pub(crate) fn reset(&mut self) {
        let entities = lock(&self.routes).entities.clone();
        let dirty = entities.values().flatten().copied().collect();
        // Invalidate callbacks in the old World, even if it was replaced or the
        // bridge was moved without a chance to borrow that World for cleanup.
        self.routes = Arc::new(Mutex::new(Routes { entities, dirty }));
        self.installed = false;
    }

    pub(crate) fn install(&mut self, world: &mut World) {
        if !self.installed {
            (self.register)(world, &self.routes);
            self.installed = true;
        }
    }
}

struct ObserverOwner {
    routes: Weak<Mutex<Routes>>,
    observers: [ObserverId; 3],
}

#[derive(Resource, Default)]
struct ObserverOwners(Vec<ObserverOwner>);

fn register<C: Component>(world: &mut World, routes: &Arc<Mutex<Routes>>) {
    let weak = Arc::downgrade(routes);
    let changed = world.observe_changed::<C, _>(move |ctx| {
        if let Some(routes) = weak.upgrade() {
            let mut routes = lock(&routes);
            for &entity in ctx.changed() {
                routes.notify(entity);
            }
        }
    });
    let weak = Arc::downgrade(routes);
    let inserted = world.observe::<C, _>(LifecycleEvent::Insert, move |ctx| {
        if let Some(routes) = weak.upgrade() {
            lock(&routes).notify(ctx.entity);
        }
    });
    let weak = Arc::downgrade(routes);
    let removed = world.observe::<C, _>(LifecycleEvent::Remove, move |ctx| {
        if let Some(routes) = weak.upgrade() {
            lock(&routes).notify(ctx.entity);
        }
    });
    world.init_resource::<ObserverOwners>();
    world
        .resource_mut::<ObserverOwners>()
        .0
        .push(ObserverOwner {
            routes: Arc::downgrade(routes),
            observers: [changed, inserted, removed],
        });
}

/// Remove expired observer leases, including ones left by a dropped standalone
/// bridge. Automatic sync systems call this even if no bridge is installed.
/// Direct bridge users can call it after dropping a bridge without detaching.
pub fn cleanup_bridge_observers(world: &mut World) {
    let Some(mut owners) = world.remove_resource::<ObserverOwners>() else {
        return;
    };
    owners.0.retain(|owner| {
        if owner.routes.strong_count() != 0 {
            return true;
        }
        for observer in owner.observers {
            world.remove_observer(observer);
        }
        false
    });
    if !owners.0.is_empty() {
        world.insert_resource(owners);
    }
}

#[cfg(test)]
pub(crate) fn observer_count(world: &World) -> usize {
    world.get_resource::<ObserverOwners>().map_or(0, |owners| {
        owners.0.iter().map(|owner| owner.observers.len()).sum()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{insert_bridge, loom_pull_system, remove_bridge, EcsBridge};
    use prism_ui_reactive::Runtime;

    #[derive(Component)]
    struct Value(i32);

    #[test]
    fn observers_are_shared_per_type_and_released_on_last_unbind() {
        let mut world = World::new();
        let entity = world.spawn(Value(1));
        let runtime = Runtime::new();
        let mut bridge = EcsBridge::new();
        let a = bridge.bind::<Value, _>(entity, runtime.signal(0), |value| value.0);
        let b = bridge.bind::<Value, _>(entity, runtime.signal(0), |value| value.0);
        bridge.pull_changed(&mut world).unwrap();
        assert_eq!(observer_count(&world), 3);
        assert!(bridge.unbind(&mut world, a));
        assert_eq!(observer_count(&world), 3);
        let observers = world.resource::<ObserverOwners>().0[0].observers;
        assert!(bridge.unbind(&mut world, b));
        assert_eq!(observer_count(&world), 0);
        for observer in observers {
            assert!(
                !world.remove_observer(observer),
                "observer already unregistered"
            );
        }
    }

    #[test]
    fn detach_drop_and_replace_release_only_their_own_observers() {
        let mut world = World::new();
        let entity = world.spawn(Value(1));
        let runtime = Runtime::new();
        let external_calls = Arc::new(Mutex::new(0));
        let calls = external_calls.clone();
        world.observe_changed::<Value, _>(move |_| *calls.lock().unwrap() += 1);
        let mut first = EcsBridge::new();
        let mut second = EcsBridge::new();
        first.bind::<Value, _>(entity, runtime.signal(0), |value| value.0);
        second.bind::<Value, _>(entity, runtime.signal(0), |value| value.0);
        first.pull_changed(&mut world).unwrap();
        second.pull_changed(&mut world).unwrap();
        assert_eq!(observer_count(&world), 6);
        first.detach(&mut world);
        assert_eq!(observer_count(&world), 3);
        drop(second);
        // Automatic cleanup also works with no installed bridge.
        loom_pull_system(&mut world);
        assert_eq!(observer_count(&world), 0);
        insert_bridge(&mut world, first);
        loom_pull_system(&mut world);
        assert_eq!(observer_count(&world), 3);
        insert_bridge(&mut world, EcsBridge::new());
        assert_eq!(observer_count(&world), 0);
        let mut bridge = remove_bridge(&mut world).unwrap();
        bridge.bind::<Value, _>(entity, runtime.signal(0), |value| value.0);
        insert_bridge(&mut world, bridge);
        loom_pull_system(&mut world);
        drop(remove_bridge(&mut world));
        assert_eq!(observer_count(&world), 0);
        world.get_mut::<Value>(entity).unwrap().0 = 2;
        world.flush_changed_observers();
        assert_eq!(*external_calls.lock().unwrap(), 1);
    }
}
