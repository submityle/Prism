//! The render-world "current view" lookup: [`CurrentView`] + [`ViewQuery`].
//!
//! This module is the render app's face of
//! [`prism_ecs::system::view_query`]. The types themselves are defined in
//! `prism_ecs` for two reasons that are properties of the engine, not of taste:
//!
//! - [`SystemParam`](prism_ecs::system::SystemParam) is an `unsafe trait` and
//!   this crate is `#![forbid(unsafe_code)]`, so the parameter's `unsafe impl`
//!   cannot be written here; and
//! - the engine that resolves one row from an `UnsafeWorldCell`
//!   (`QueryState::get_from_ptr`) is `pub(crate)` to `prism_ecs`, so a
//!   downstream crate cannot reach it at all.
//!
//! Re-exporting from here keeps the render-facing vocabulary in one place: a
//! port of `prism_render_scene` — which uses `bevy_render`'s `ViewQuery` at ~154
//! sites — imports it from the render app, exactly as it imported it from
//! `bevy_render::renderer`.
//!
//! # Port note: the skip is spelled `None`
//!
//! Bevy's `ViewQuery` *skips the whole system* when `CurrentView` is missing or
//! names a non-matching entity (`SystemParamValidationError::skipped`).
//! `prism_ecs`'s `SystemParam` has no validation-error/skip mechanism — its
//! `get_param` returns the item directly, and the only run-level gate is a
//! `run_if` condition on the schedule. The parameter is therefore
//! `Option<ViewQuery<..>>`, the same "absent means `None`" shape the crate
//! already uses for `Option<Res<T>>`:
//!
//! ```no_run
//! use prism_render_app::ViewQuery;
//!
//! #[derive(Debug)]
//! struct MaterialRenderClass(u32);
//! impl prism_ecs::prelude::Component for MaterialRenderClass {}
//!
//! fn record(views: Option<ViewQuery<&MaterialRenderClass>>) {
//!     // Bevy's implicit skip, written out: no current view (or a filtered
//!     // one) means this system contributes nothing this run.
//!     let Some(views) = views else { return };
//!     let _entity = views.entity();
//!     let _class = views.into_inner().0;
//! }
//! ```
//!
//! `ViewQuery::entity()` and `ViewQuery::into_inner()` keep Bevy's signatures
//! (`Entity` and `D::Item`, respectively), so only the parameter position
//! changes in a ported signature, not the body.
//!
//! A system that only needs the entity — no component fetch — takes
//! `Res<CurrentView>` (or `Option<Res<CurrentView>>`) instead.

pub use prism_ecs::system::view_query::{CurrentView, ViewQuery, ViewQueryState};

#[cfg(test)]
mod tests {
    use prism_ecs::prelude::{Component, Entity, With};
    use prism_ecs::system::{CurrentView, IntoSystem, System};
    use prism_ecs::world::World;

    use super::ViewQuery;

    #[derive(Debug, PartialEq)]
    struct Depth(i32);
    impl Component for Depth {}

    #[derive(Debug)]
    struct Tagged;
    impl Component for Tagged {}

    /// A system taking the view query reads the component of the entity named
    /// by `CurrentView`, and no other entity's.
    #[test]
    fn view_query_reads_the_component_of_the_current_view_entity() {
        let mut world = World::new();
        let view = world.spawn(Depth(7));
        // A decoy with the same component: reading it would be the failure mode
        // this test exists to catch.
        world.spawn(Depth(99));
        world.insert_resource(CurrentView(view));

        fn read(view: Option<ViewQuery<&Depth>>) -> Option<i32> {
            view.map(|view| view.into_inner().0)
        }

        let mut sys = IntoSystem::into_system(read);
        sys.initialize(&mut world);

        assert_eq!(
            sys.run(&mut world),
            Some(7),
            "the fetch must come from the CurrentView entity (7), not the decoy (99)"
        );
    }

    /// `entity()` reports the entity `CurrentView` named.
    #[test]
    fn view_query_entity_is_the_current_view_entity() {
        let mut world = World::new();
        world.spawn(Depth(1));
        let current = world.spawn(Depth(2));
        world.insert_resource(CurrentView(current));

        fn read(view: Option<ViewQuery<&Depth>>) -> Option<(Entity, i32)> {
            let view = view?;
            let entity = view.entity();
            Some((entity, view.into_inner().0))
        }

        let mut sys = IntoSystem::into_system(read);
        sys.initialize(&mut world);

        assert_eq!(
            sys.run(&mut world),
            Some((current, 2)),
            "entity() and the fetch must agree on the CurrentView entity"
        );
    }

    /// When the current view's entity does not satisfy the query's filter, the
    /// system observes that fact as `None` — it never falls back to another
    /// matching entity.
    #[test]
    fn view_query_is_none_when_the_current_view_does_not_match() {
        let mut world = World::new();
        let untagged = world.spawn(Depth(1));
        let tagged = world.spawn((Depth(2), Tagged));

        fn read(view: Option<ViewQuery<&Depth, With<Tagged>>>) -> Option<i32> {
            view.map(|view| view.into_inner().0)
        }

        let mut sys = IntoSystem::into_system(read);
        sys.initialize(&mut world);

        world.insert_resource(CurrentView(untagged));
        assert_eq!(
            sys.run(&mut world),
            None,
            "the untagged current view is filtered out; the tagged entity must not be substituted"
        );

        // Same system, same resolved state: pointing CurrentView at a matching
        // entity resolves, proving the `None` above came from the lookup rather
        // than from a broken state.
        world.insert_resource(CurrentView(tagged));
        assert_eq!(
            sys.run(&mut world),
            Some(2),
            "a matching current view resolves to that entity's component"
        );
    }

    /// A missing `CurrentView` is `None`, not a panic — the closest analogue of
    /// Bevy's "skip the system" that a `get_param`-without-`Result` engine
    /// permits.
    #[test]
    fn view_query_is_none_when_current_view_is_absent() {
        let mut world = World::new();
        world.spawn(Depth(5));
        // Deliberately no `insert_resource(CurrentView(..))`.

        fn read(view: Option<ViewQuery<&Depth>>) -> Option<i32> {
            view.map(|view| view.into_inner().0)
        }

        let mut sys = IntoSystem::into_system(read);
        sys.initialize(&mut world);

        assert_eq!(
            sys.run(&mut world),
            None,
            "no CurrentView resource means no view to record, never a panic"
        );
    }

    /// The fetch is a real borrow: a `&mut` view query writes through to the
    /// entity `CurrentView` names, and leaves every other entity alone.
    #[test]
    fn view_query_can_mutate_the_current_view_entity() {
        let mut world = World::new();
        let view = world.spawn(Depth(10));
        let untouched = world.spawn(Depth(99));
        world.insert_resource(CurrentView(view));

        fn bump(view: Option<ViewQuery<&mut Depth>>) {
            if let Some(view) = view {
                view.into_inner().0 += 1;
            }
        }

        let mut sys = IntoSystem::into_system(bump);
        sys.initialize(&mut world);
        sys.run(&mut world);
        sys.run(&mut world);

        assert_eq!(
            world.get::<Depth>(view).map(|depth| depth.0),
            Some(12),
            "two runs must have written the current view's component twice"
        );
        assert_eq!(
            world.get::<Depth>(untouched).map(|depth| depth.0),
            Some(99),
            "the non-current entity must be untouched"
        );
    }

    /// The parameter declares its accesses: a shared read of the `CurrentView`
    /// resource and the query's own term access. Without this the parallel
    /// executor's conflict analysis would not see the fetch at all.
    #[test]
    fn view_query_declares_the_current_view_resource_and_term_access() {
        let mut world = World::new();
        world.spawn(Depth(1));

        fn read(view: Option<ViewQuery<&mut Depth, With<Tagged>>>) {
            if let Some(view) = view {
                view.into_inner().0 += 1;
            }
        }

        let mut sys = IntoSystem::into_system(read);
        sys.initialize(&mut world);

        let view_resource = world
            .resources()
            .id_of::<CurrentView>()
            .expect("init_state registers CurrentView");
        let depth = world
            .components()
            .id_of::<Depth>()
            .expect("init_state registers the query's component");
        let access = sys.access();
        assert!(
            access.resource_reads().contains(&view_resource),
            "CurrentView must be declared as a resource read"
        );
        assert!(
            access.writes().contains(&depth),
            "the &mut Depth term must be declared as a write"
        );
        assert!(
            world.components().id_of::<Tagged>().is_some(),
            "the filter's component is registered by init_state"
        );
        assert!(
            access.reads().is_empty(),
            "With<..> is archetype-only: it registers Tagged but declares no tick read"
        );
    }
}
