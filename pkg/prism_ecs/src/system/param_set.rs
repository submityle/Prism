//! Sequential access to potentially conflicting system parameters.

use crate::query::Access;
use crate::system::{SystemParam, SystemParamItem, UnsafeWorldCell};
use crate::world::World;

/// A group of parameters whose potentially conflicting access is sequential.
///
/// `p0()` through `p11()` borrow the **whole set** exclusively. Both the world
/// and state lifetimes of the fetched item are shortened to that borrow, so
/// an item (including a resource's `into_inner()` result) cannot survive the
/// next member fetch. No member is fetched until its accessor is called.
///
/// Each member must be internally conflict-free. Conflicts between members
/// are allowed, but the union of their access still conflicts with parameters
/// outside the set and with other systems, even for members never fetched.
/// Tuples of zero through twelve parameters and nested sets are supported.
/// Deferred effects are applied for every member in tuple order.
///
/// In a `#[derive(SystemParam)]` struct, use `'static` lifetimes for the
/// member parameter descriptors (for example
/// `ParamSet<'w, 's, (ResMut<'static, Counter>, Res<'static, Counter>)>`).
/// These describe fetches, not live borrows: the accessors always replace
/// their world/state lifetimes with the exclusive set borrow.
///
/// ```
/// use prism_ecs::prelude::*;
/// use prism_ecs::system::ParamSet;
///
/// #[derive(Component)]
/// struct Position(i32);
///
/// fn step(mut queries: ParamSet<(Query<&mut Position>, Query<&Position>)>) {
///     for mut position in queries.p0().iter_mut() {
///         position.0 += 1;
///     }
///     assert!(queries.p1().iter().all(|position| position.0 > 0));
/// }
///
/// let mut world = World::new();
/// world.spawn(Position(0));
/// let mut system = IntoSystem::into_system(step);
/// system.initialize(&mut world);
/// system.run(&mut world);
/// ```
///
/// Simultaneous member access is rejected, even after consuming a wrapper:
///
/// ```compile_fail,E0499
/// use prism_ecs::prelude::*;
/// use prism_ecs::system::ParamSet;
/// #[derive(Resource)]
/// struct Counter(i32);
/// fn alias(mut set: ParamSet<(ResMut<Counter>, ResMut<Counter>)>) {
///     let first = set.p0().into_inner();
///     let second = set.p1().into_inner();
///     first.0 += second.0;
/// }
/// ```
///
/// The world lifetime of a shared query's consuming iterator is also bounded
/// by the set borrow, preventing a later writer from aliasing its results:
///
/// ```compile_fail,E0499
/// use prism_ecs::prelude::*;
/// use prism_ecs::system::ParamSet;
/// #[derive(Component)]
/// struct Position(i32);
/// fn alias(mut set: ParamSet<(Query<&Position>, Query<&mut Position>)>) {
///     let read = (&set.p0()).into_iter().next().unwrap();
///     set.p1().single_mut().unwrap().0 += 1;
///     assert_eq!(read.0, 0);
/// }
/// ```
///
/// Fetching the same member twice cannot produce two state borrows either:
///
/// ```compile_fail,E0499
/// use prism_ecs::prelude::*;
/// use prism_ecs::system::ParamSet;
/// fn alias(mut set: ParamSet<(Local<u32>,)>) {
///     let first = set.p0().into_inner();
///     let second = set.p0().into_inner();
///     *first += *second;
/// }
/// ```
///
/// Mutable query results also keep the whole set borrowed:
///
/// ```compile_fail,E0499
/// use prism_ecs::prelude::*;
/// use prism_ecs::system::ParamSet;
/// #[derive(Component)]
/// struct Position(i32);
/// fn alias(mut set: ParamSet<(Query<&mut Position>, Query<&mut Position>)>) {
///     let mut first_query = set.p0();
///     let mut first = (&mut first_query).into_iter().next().unwrap();
///     let mut second_query = set.p1();
///     first.0 += second_query.single_mut().unwrap().0;
/// }
/// ```
///
/// An inner set's items retain the borrow of the outer set:
///
/// ```compile_fail,E0499
/// use prism_ecs::prelude::*;
/// use prism_ecs::system::ParamSet;
/// #[derive(Resource)]
/// struct Counter(i32);
/// fn alias(mut set: ParamSet<(ParamSet<(ResMut<Counter>,)>, Res<Counter>)>) {
///     let mut inner = set.p0();
///     let first = inner.p0().into_inner();
///     let second = set.p1().into_inner();
///     first.0 += second.0;
/// }
/// ```
///
/// Consuming a member cannot recover the original world lifetime:
///
/// ```compile_fail,E0515
/// use prism_ecs::prelude::*;
/// use prism_ecs::system::ParamSet;
/// #[derive(Resource)]
/// struct Counter(i32);
/// fn escape<'w, 's>(
///     mut set: ParamSet<'w, 's, (ResMut<'static, Counter>,)>,
/// ) -> &'w mut Counter {
///     set.p0().into_inner()
/// }
/// ```
pub struct ParamSet<'w, 's, P: SystemParam> {
    state: &'s mut P::State,
    world: UnsafeWorldCell<'w>,
}

macro_rules! impl_param_set {
    ($(($param:ident, $index:tt, $accessor:ident)),* $(,)?) => {
        // SAFETY: each member validates its own access in isolation. Their
        // union is checked against all earlier outside params before merging;
        // later params see that same union. Only accessor calls fetch members,
        // and the exclusive borrow of the entire set bounds both lifetimes of
        // every item, so conflicting members can never be live together.
        #[allow(unused_variables, unused_mut, clippy::unused_unit)]
        unsafe impl<$($param: SystemParam),*> SystemParam
            for ParamSet<'_, '_, ($($param,)*)>
        {
            type State = ($($param::State,)*);
            type Item<'w, 's> = ParamSet<'w, 's, ($($param,)*)>;

            fn init_state(world: &mut World) -> Self::State {
                ($($param::init_state(world),)*)
            }

            fn update_access(state: &Self::State, access: &mut Access) {
                let mut union = Access::new();
                $(
                    let mut member = Access::new();
                    $param::update_access(&state.$index, &mut member);
                    assert!(
                        access.is_compatible(&member),
                        "ParamSet conflict: member {} conflicts with a parameter outside the set",
                        $index,
                    );
                    union.extend(&member);
                )*
                access.extend(&union);
            }

            unsafe fn get_param<'w, 's>(
                state: &'s mut Self::State,
                world: UnsafeWorldCell<'w>,
            ) -> Self::Item<'w, 's> {
                ParamSet { state, world }
            }

            fn apply(state: &mut Self::State, world: &mut World) {
                $($param::apply(&mut state.$index, world);)*
            }

            fn discard(state: &mut Self::State, world: &mut World) {
                $($param::discard(&mut state.$index, world);)*
            }
        }

        impl<'w, 's, $($param: SystemParam),*> ParamSet<'w, 's, ($($param,)*)> {
            $(
                /// Fetch this member, exclusively borrowing the whole set
                /// until the returned item and its derived borrows expire.
                #[inline]
                pub fn $accessor<'a>(&'a mut self) -> SystemParamItem<'a, 'a, $param> {
                    // Shorten the world capability as well as the state borrow.
                    // Returning Item<'w, 'a> would let ResMut::into_inner or a
                    // shared Query's IntoIterator escape this exclusive borrow.
                    let world: UnsafeWorldCell<'a> = self.world;
                    // SAFETY: the system grants the entire declared union;
                    // `&mut self` excludes all other live member items, and
                    // this state field was initialized for exactly this member.
                    unsafe { $param::get_param(&mut self.state.$index, world) }
                }
            )*
        }
    };
}

impl_param_set!();
impl_param_set!((P0, 0, p0));
impl_param_set!((P0, 0, p0), (P1, 1, p1));
impl_param_set!((P0, 0, p0), (P1, 1, p1), (P2, 2, p2));
impl_param_set!((P0, 0, p0), (P1, 1, p1), (P2, 2, p2), (P3, 3, p3));
impl_param_set!(
    (P0, 0, p0),
    (P1, 1, p1),
    (P2, 2, p2),
    (P3, 3, p3),
    (P4, 4, p4)
);
impl_param_set!(
    (P0, 0, p0),
    (P1, 1, p1),
    (P2, 2, p2),
    (P3, 3, p3),
    (P4, 4, p4),
    (P5, 5, p5)
);
impl_param_set!(
    (P0, 0, p0),
    (P1, 1, p1),
    (P2, 2, p2),
    (P3, 3, p3),
    (P4, 4, p4),
    (P5, 5, p5),
    (P6, 6, p6)
);
impl_param_set!(
    (P0, 0, p0),
    (P1, 1, p1),
    (P2, 2, p2),
    (P3, 3, p3),
    (P4, 4, p4),
    (P5, 5, p5),
    (P6, 6, p6),
    (P7, 7, p7)
);
impl_param_set!(
    (P0, 0, p0),
    (P1, 1, p1),
    (P2, 2, p2),
    (P3, 3, p3),
    (P4, 4, p4),
    (P5, 5, p5),
    (P6, 6, p6),
    (P7, 7, p7),
    (P8, 8, p8)
);
impl_param_set!(
    (P0, 0, p0),
    (P1, 1, p1),
    (P2, 2, p2),
    (P3, 3, p3),
    (P4, 4, p4),
    (P5, 5, p5),
    (P6, 6, p6),
    (P7, 7, p7),
    (P8, 8, p8),
    (P9, 9, p9)
);
impl_param_set!(
    (P0, 0, p0),
    (P1, 1, p1),
    (P2, 2, p2),
    (P3, 3, p3),
    (P4, 4, p4),
    (P5, 5, p5),
    (P6, 6, p6),
    (P7, 7, p7),
    (P8, 8, p8),
    (P9, 9, p9),
    (P10, 10, p10)
);
impl_param_set!(
    (P0, 0, p0),
    (P1, 1, p1),
    (P2, 2, p2),
    (P3, 3, p3),
    (P4, 4, p4),
    (P5, 5, p5),
    (P6, 6, p6),
    (P7, 7, p7),
    (P8, 8, p8),
    (P9, 9, p9),
    (P10, 10, p10),
    (P11, 11, p11)
);

#[cfg(all(test, feature = "multi_thread"))]
mod schedule_tests {
    use super::*;
    use crate::component::Component;
    use crate::schedule::parallel_executor::{compute_waves, effective_accesses};
    use crate::schedule::Schedule;
    use crate::system::Query;

    struct Value;
    impl Component for Value {}

    #[test]
    fn unused_conflicting_member_forces_separate_executor_waves() {
        fn unused(_set: ParamSet<(Query<&mut Value>, Query<&Value>)>) {}
        fn reader(_query: Query<&Value>) {}

        let mut world = World::new();
        let mut schedule = Schedule::new();
        schedule.add_systems((unused, reader));
        schedule.initialize(&mut world);
        let access = effective_accesses(&schedule);
        // There are deliberately no ordering edges: only the unused member's
        // declared write can keep these two systems in different waves.
        let reach = alloc::vec![alloc::vec![false; 2]; 2];
        assert_eq!(
            compute_waves(&schedule.order, &access, &reach),
            alloc::vec![alloc::vec![0], alloc::vec![1]],
        );
    }
}
