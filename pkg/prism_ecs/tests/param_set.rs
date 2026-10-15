//! ParamSet through the public system API, including nested elided query types.

use prism_ecs::prelude::*;
use prism_ecs::query::Access;
use prism_ecs::system::ParamSet;
use std::panic::{catch_unwind, AssertUnwindSafe};

#[derive(Component, Debug)]
struct Value(i32);

#[derive(Component)]
struct Other;

#[derive(Resource, Default, Debug)]
struct Counter(i32);

#[derive(Resource, Default)]
struct Seen(Vec<usize>);

#[test]
fn conflicting_queries_are_sequential_and_observe_prior_writes() {
    fn step(mut set: ParamSet<(Query<&mut Value>, Query<&Value>, Query<&mut Value>)>) {
        set.p0().for_each_mut(|mut value| value.0 += 2);
        let sum: i32 = set.p1().iter().map(|value| value.0).sum();
        set.p2().for_each_mut(|mut value| value.0 *= sum);
    }

    let mut world = World::new();
    let a = world.spawn(Value(1));
    let b = world.spawn(Value(2));
    let mut system = IntoSystem::into_system(step);
    system.initialize(&mut world);
    system.run(&mut world);
    assert_eq!(world.get::<Value>(a).unwrap().0, 21);
    assert_eq!(world.get::<Value>(b).unwrap().0, 28);

    // Refresh cached matches after an archetype appears between runs.
    let c = world.spawn((Value(0), Other));
    system.run(&mut world);
    assert_eq!(world.get::<Value>(a).unwrap().0, 23 * 55);
    assert_eq!(world.get::<Value>(b).unwrap().0, 30 * 55);
    assert_eq!(world.get::<Value>(c).unwrap().0, 2 * 55);
}

#[test]
fn resource_items_and_into_inner_can_be_reborrowed_sequentially() {
    fn step(mut set: ParamSet<(ResMut<Counter>, Res<Counter>, ResMut<Counter>)>) {
        set.p0().into_inner().0 += 3;
        let previous = set.p1().into_inner().0;
        set.p2().into_inner().0 += previous;
    }

    let mut world = World::new();
    world.insert_resource(Counter(1));
    let mut system = IntoSystem::into_system(step);
    system.initialize(&mut world);
    system.run(&mut world);
    assert_eq!(world.resource::<Counter>().0, 8);
    system.run(&mut world);
    assert_eq!(world.resource::<Counter>().0, 22);
}

#[test]
fn optional_and_unused_members_are_fetched_lazily() {
    fn step(mut set: ParamSet<(Res<Counter>, Option<ResMut<Counter>>, Local<u32>)>) {
        // p0 would panic: Counter is absent. Constructing the set must not
        // fetch it, even though it still contributes static resource access.
        assert!(set.p1().is_none());
        *set.p2() += 1;
        assert!(*set.p2() > 0);
    }

    let mut world = World::new();
    let mut system = IntoSystem::into_system(step);
    system.initialize(&mut world);
    system.run(&mut world);
    system.run(&mut world);
}

#[test]
fn union_includes_unused_members_and_both_access_domains() {
    fn step(
        _set: ParamSet<(
            Query<&Value>,
            Query<&mut Value>,
            Query<&Other>,
            Res<Counter>,
            ResMut<Counter>,
            Res<Seen>,
        )>,
    ) {
    }

    let mut world = World::new();
    let mut system = IntoSystem::into_system(step);
    system.initialize(&mut world);
    let value = world.register_component::<Value>();
    let other = world.register_component::<Other>();
    let counter = world.resources().id_of::<Counter>().unwrap();
    let seen = world.resources().id_of::<Seen>().unwrap();
    let access = system.access();
    assert!(access.reads().contains(&value));
    assert!(access.writes().contains(&value));
    assert!(access.reads().contains(&other));
    assert!(access.resource_reads().contains(&counter));
    assert!(access.resource_writes().contains(&counter));
    assert!(access.resource_reads().contains(&seen));

    let mut reader = Access::new();
    reader.add_read(value);
    let mut writer = Access::new();
    writer.add_write(other);
    let mut resource_reader = Access::new();
    resource_reader.add_resource_read(counter);
    let mut resource_writer = Access::new();
    resource_writer.add_resource_write(seen);
    for conflicting in [&reader, &writer, &resource_reader, &resource_writer] {
        assert!(!access.is_compatible(conflicting));
        assert!(!conflicting.is_compatible(access));
    }
    assert!(access.is_compatible(&Access::new()));
}

fn assert_rejected<Out, Marker>(function: impl IntoSystem<Out, Marker>) {
    let mut world = World::new();
    let mut system = function.into_system();
    assert!(catch_unwind(AssertUnwindSafe(|| system.initialize(&mut world))).is_err());
}

#[test]
fn outside_component_conflicts_are_rejected_in_both_orders() {
    fn before(_outside: Query<&Value>, _set: ParamSet<(Query<&mut Value>,)>) {}
    fn after(_set: ParamSet<(Query<&mut Value>,)>, _outside: Query<&Value>) {}
    fn writes(_set: ParamSet<(Query<&mut Value>,)>, _outside: Query<&mut Value>) {}
    assert_rejected(before);
    assert_rejected(after);
    assert_rejected(writes);
}

#[test]
fn outside_resource_conflicts_are_rejected_in_both_orders() {
    fn before(_outside: Res<Counter>, _set: ParamSet<(ResMut<Counter>,)>) {}
    fn after(_set: ParamSet<(ResMut<Counter>,)>, _outside: Res<Counter>) {}
    fn writes(_outside: ResMut<Counter>, _set: ParamSet<(ResMut<Counter>,)>) {}
    assert_rejected(before);
    assert_rejected(after);
    assert_rejected(writes);
}

#[test]
fn two_sets_cannot_hide_conflicts_from_each_other() {
    fn components(_a: ParamSet<(Query<&mut Value>,)>, _b: ParamSet<(Query<&Value>,)>) {}
    fn resources(_a: ParamSet<(Res<Counter>,)>, _b: ParamSet<(ResMut<Counter>,)>) {}
    assert_rejected(components);
    assert_rejected(resources);
}

#[test]
fn each_member_must_remain_internally_conflict_free() {
    fn query(_set: ParamSet<(Query<(&mut Value, &Value)>,)>) {}
    fn tuple(_set: ParamSet<((ResMut<Counter>, Res<Counter>),)>) {}
    assert_rejected(query);
    assert_rejected(tuple);
}

#[test]
fn shared_and_disjoint_outside_params_remain_compatible() {
    fn step(
        mut set: ParamSet<(Query<&Value>, Res<Counter>)>,
        query: Query<&Value>,
        counter: Res<Counter>,
        mut seen: ResMut<Seen>,
    ) {
        assert_eq!(set.p0().single().unwrap().0, query.single().unwrap().0);
        assert_eq!(set.p1().0, counter.0);
        seen.0.push(1);
    }

    let mut world = World::new();
    world.spawn(Value(7));
    world.insert_resource(Counter(9));
    world.insert_resource(Seen::default());
    let mut system = IntoSystem::into_system(step);
    system.initialize(&mut world);
    system.run(&mut world);
    assert_eq!(world.resource::<Seen>().0, [1]);
}

#[test]
fn nested_sets_and_tuple_members_preserve_state_and_access() {
    fn step(
        mut set: ParamSet<(
            ParamSet<(ResMut<Counter>, Res<Counter>)>,
            (ResMut<Counter>, Local<u32>),
        )>,
    ) {
        {
            let mut inner = set.p0();
            inner.p0().0 += 1;
            assert!(inner.p1().0 > 0);
        }
        let (mut counter, mut runs) = set.p1();
        *runs += 1;
        counter.0 += *runs as i32;
    }

    let mut world = World::new();
    world.insert_resource(Counter::default());
    let mut system = IntoSystem::into_system(step);
    system.initialize(&mut world);
    system.run(&mut world);
    system.run(&mut world);
    assert_eq!(world.resource::<Counter>().0, 5);
}

#[derive(SystemParam)]
struct Context<'w, 's> {
    set: ParamSet<
        'w,
        's,
        (
            Query<'static, 'static, &'static mut Value>,
            Query<'static, 'static, &'static Value>,
        ),
    >,
}

#[test]
fn sets_work_inside_derived_params() {
    fn step(mut context: Context) {
        context.set.p0().single_mut().unwrap().0 += 5;
        assert_eq!(context.set.p1().single().unwrap().0, 8);
    }

    let mut world = World::new();
    let entity = world.spawn(Value(3));
    let mut system = IntoSystem::into_system(step);
    system.initialize(&mut world);
    system.run(&mut world);
    assert_eq!(world.get::<Value>(entity).unwrap().0, 8);
}

#[test]
fn every_members_deferred_work_is_applied_in_tuple_order() {
    fn enqueue(mut set: ParamSet<(Commands, Commands, Commands)>) {
        // Fetch order deliberately differs from apply order. Dropping and
        // refetching Commands must keep appending to the same member buffer.
        set.p1().queue(|world| {
            world.insert_resource(Counter(20));
        });
        set.p0().queue(|world| {
            world.insert_resource(Counter(10));
        });
        set.p0().spawn(Value(1));
        set.p1().spawn(Value(2));
        // p2 is never fetched but its empty buffer is still applied.
    }

    let mut world = World::new();
    let mut system = IntoSystem::into_system(enqueue);
    system.initialize(&mut world);
    system.run(&mut world);
    assert_eq!(world.resource::<Counter>().0, 20);
    assert_eq!(world.query::<&Value>().iter(&world).count(), 2);
}

#[test]
fn member_queries_keep_the_system_change_detection_window() {
    fn step(
        mut set: ParamSet<(Query<&mut Value>, Query<&Value, Changed<Value>>)>,
        mut done: Local<bool>,
        mut seen: ResMut<Seen>,
    ) {
        if !*done {
            set.p0().single_mut().unwrap().0 += 1;
            *done = true;
        }
        seen.0.push(set.p1().count());
    }

    let mut world = World::new();
    world.spawn(Value(1));
    world.insert_resource(Seen::default());
    let mut system = IntoSystem::into_system(step);
    system.initialize(&mut world);
    system.run(&mut world);
    system.run(&mut world);
    assert_eq!(world.resource::<Seen>().0, [1, 0]);
}

#[test]
fn zero_and_twelve_member_sets_work() {
    fn empty(_set: ParamSet<()>) {}
    let mut world = World::new();
    let mut system = IntoSystem::into_system(empty);
    system.initialize(&mut world);
    system.run(&mut world);
    assert!(system.access().reads().is_empty());
    assert!(system.access().writes().is_empty());

    fn twelve(
        mut set: ParamSet<(
            Local<u32>,
            Local<u32>,
            Local<u32>,
            Local<u32>,
            Local<u32>,
            Local<u32>,
            Local<u32>,
            Local<u32>,
            Local<u32>,
            Local<u32>,
            Local<u32>,
            Local<u32>,
        )>,
    ) -> u32 {
        *set.p0() += 1;
        *set.p1() += 1;
        *set.p2() += 1;
        *set.p3() += 1;
        *set.p4() += 1;
        *set.p5() += 1;
        *set.p6() += 1;
        *set.p7() += 1;
        *set.p8() += 1;
        *set.p9() += 1;
        *set.p10() += 1;
        *set.p11() += 1;
        *set.p0() + *set.p11()
    }
    let mut system = IntoSystem::into_system(twelve);
    system.initialize(&mut world);
    assert_eq!(system.run(&mut world), 2);
    assert_eq!(system.run(&mut world), 4);
}

#[test]
fn cross_world_reuse_is_rejected_and_reinitialization_works() {
    fn step(mut set: ParamSet<(Query<&mut Value>, Query<&Value>)>) {
        set.p0().single_mut().unwrap().0 += 1;
    }

    let mut a = World::new();
    a.spawn(Value(1));
    let mut b = World::new();
    // Deliberately assign a different component-id ordering in the target.
    b.register_component::<Other>();
    let entity = b.spawn(Value(2));
    let mut system = IntoSystem::into_system(step);
    system.initialize(&mut a);
    let tick = b.change_tick();
    assert!(catch_unwind(AssertUnwindSafe(|| system.run(&mut b))).is_err());
    assert_eq!(b.change_tick(), tick);
    assert_eq!(b.get::<Value>(entity).unwrap().0, 2);
    system.initialize(&mut b);
    system.run(&mut b);
    assert_eq!(b.get::<Value>(entity).unwrap().0, 3);
}

#[test]
fn erased_query_access_replay_never_reinterprets_another_querys_state() {
    // Both query params intentionally share the erased State type. The safe
    // access-replay API must inspect the owner's actual access, not cast it
    // to the query type used at this call site.
    let mut world = World::new();
    let state = <Query<&Value> as SystemParam>::init_state(&mut world);
    let mut access = Access::new();
    <Query<&mut Other> as SystemParam>::update_access(&state, &mut access);
    assert_eq!(access.reads(), &[world.register_component::<Value>()]);
    assert!(access.writes().is_empty());
}

#[test]
fn failed_member_fetch_releases_the_set_borrow_for_later_access() {
    fn step(mut set: ParamSet<(Res<Counter>, Local<u32>, Query<&mut Value>)>) {
        assert!(catch_unwind(AssertUnwindSafe(|| {
            let _ = set.p0();
        }))
        .is_err());
        *set.p1() += 1;
        let runs = *set.p1();
        set.p2().single_mut().unwrap().0 += runs as i32;
    }

    let mut world = World::new();
    let entity = world.spawn(Value(10));
    let mut system = IntoSystem::into_system(step);
    system.initialize(&mut world);
    system.run(&mut world);
    system.run(&mut world);
    assert_eq!(world.get::<Value>(entity).unwrap().0, 13);
}

#[test]
fn member_body_panic_releases_conflicting_query_and_resource_borrows() {
    fn step(
        mut set: ParamSet<(
            Query<&mut Value>,
            Query<&Value>,
            ResMut<Counter>,
            Res<Counter>,
        )>,
    ) {
        assert!(catch_unwind(AssertUnwindSafe(|| {
            let mut query = set.p0();
            let mut value = query.single_mut().unwrap();
            value.0 += 5;
            panic!("panic while holding a mutable query item");
        }))
        .is_err());
        assert_eq!(set.p1().single().unwrap().0, 15);
        assert!(catch_unwind(AssertUnwindSafe(|| {
            let counter = set.p2().into_inner();
            counter.0 += 3;
            panic!("panic while holding a consumed resource wrapper");
        }))
        .is_err());
        assert_eq!(set.p3().into_inner().0, 4);
        set.p0().single_mut().unwrap().0 += 1;
        set.p2().into_inner().0 += 1;
    }

    let mut world = World::new();
    let entity = world.spawn(Value(10));
    world.insert_resource(Counter(1));
    let mut system = IntoSystem::into_system(step);
    system.initialize(&mut world);
    system.run(&mut world);
    assert_eq!(world.get::<Value>(entity).unwrap().0, 16);
    assert_eq!(world.resource::<Counter>().0, 5);
}

#[cfg(feature = "multi_thread")]
#[test]
fn parallel_member_iteration_joins_before_the_next_conflicting_member() {
    use std::sync::atomic::{AtomicI32, Ordering};
    use std::sync::Arc;

    let pool = Arc::new(prism_tasks::TaskPool::with_threads(3));
    let mut world = World::new();
    let entities: Vec<_> = (0..257).map(|value| world.spawn(Value(value))).collect();
    let mut system = IntoSystem::into_system(
        move |mut set: ParamSet<(Query<&mut Value>, Query<&Value>, Query<&mut Value>)>| {
            set.p0()
                .par_for_each_mut(&pool, 7, |mut value| value.0 += 3);
            let sum = AtomicI32::new(0);
            set.p1().par_for_each(&pool, 9, |value| {
                sum.fetch_add(value.0, Ordering::Relaxed);
            });
            assert_eq!(sum.load(Ordering::Relaxed), (0..257).sum::<i32>() + 257 * 3);
            set.p2()
                .par_for_each_mut(&pool, 11, |mut value| value.0 *= 2);
        },
    );
    system.initialize(&mut world);
    system.run(&mut world);
    for (initial, entity) in entities.into_iter().enumerate() {
        assert_eq!(
            world.get::<Value>(entity).unwrap().0,
            (initial as i32 + 3) * 2
        );
    }
}
