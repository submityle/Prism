//! Ownership and failed-staging regressions through the public server API.

#![cfg(feature = "std")]

use prism_asset::{
    Asset, AssetLoader, AssetPath, AssetServer, AssetTypeId, DepRequest, LoadContext, LoadError,
    LoadScheduler, LoadState, MemSource, RecursiveDependencyLoadState,
};
use prism_tasks::{Priority, TaskPool};
use std::sync::Arc;

#[derive(Debug)]
struct Source(u8);

impl Asset for Source {
    const TYPE_NAME: &'static str = "reload_ownership_regressions::Source";
}

#[derive(Clone, Debug)]
struct Label(u8);

impl Asset for Label {
    const TYPE_NAME: &'static str = "reload_ownership_regressions::Label";
}

#[derive(Clone, Debug)]
struct Consumer;

impl Asset for Consumer {
    const TYPE_NAME: &'static str = "reload_ownership_regressions::Consumer";
}

struct SourceLoader;

impl AssetLoader for SourceLoader {
    type Asset = Source;

    fn extensions(&self) -> &'static [&'static str] {
        &["ownership"]
    }

    fn load(&self, bytes: &[u8], ctx: &mut LoadContext<'_>) -> Result<Source, LoadError> {
        let revision = bytes[0];
        match revision {
            1 => ctx.add_dependency(ctx.path().clone()),
            2 => {
                ctx.add_labeled_asset("b", Label(revision));
                ctx.add_labeled_asset_with_deps(
                    "a",
                    Label(revision),
                    vec![DepRequest::typed(
                        ctx.path().clone().with_label("b"),
                        AssetTypeId::of::<Label>(),
                    )],
                );
            }
            3 => ctx.add_labeled_asset_with_deps(
                "a",
                Label(revision),
                vec![DepRequest::typed(
                    ctx.path().clone(),
                    AssetTypeId::of::<Source>(),
                )],
            ),
            4 => {
                ctx.add_labeled_asset("b", Label(revision));
                ctx.add_labeled_asset("a", Label(revision));
            }
            5 | 6 => {
                ctx.add_labeled_asset("b", Label(revision));
                ctx.add_labeled_asset("provisional", Label(revision));
                if revision == 6 {
                    // This nested successful load keeps the provisional slot
                    // alive after rollback and depends on its recursive state.
                    ctx.add_labeled_asset_with_deps(
                        "trigger",
                        Label(revision),
                        vec![DepRequest::typed(
                            AssetPath::parse("nested.consumer"),
                            AssetTypeId::of::<Consumer>(),
                        )],
                    );
                }
                ctx.add_labeled_asset("b", Label(revision));
            }
            _ => {}
        }
        Ok(Source(revision))
    }
}

struct ConsumerLoader;

impl AssetLoader for ConsumerLoader {
    type Asset = Consumer;

    fn extensions(&self) -> &'static [&'static str] {
        &["consumer"]
    }

    fn load(&self, _: &[u8], ctx: &mut LoadContext<'_>) -> Result<Consumer, LoadError> {
        ctx.add_dependency(AssetPath::parse("scene.ownership#provisional"));
        Ok(Consumer)
    }
}

fn mount(server: &AssetServer, revision: u8) {
    let mut source = MemSource::new();
    source.insert("scene.ownership", [revision]);
    source.insert("nested.consumer", [0]);
    server.mount("mem", Arc::new(source), 0);
}

fn server(revision: u8) -> AssetServer {
    let server = AssetServer::new();
    server.register_loader(SourceLoader);
    server.register_loader(ConsumerLoader);
    mount(&server, revision);
    server
}

fn collect(server: &AssetServer) {
    for _ in 0..8 {
        server.collect_releases();
    }
}

#[test]
fn primary_self_dependency_does_not_keep_initial_or_reloaded_source_alive() {
    for reload in [false, true] {
        let server = server(if reload { 0 } else { 1 });
        let source = server.load::<Source>("scene.ownership");
        if reload {
            mount(&server, 1);
            drop(server.reload("scene.ownership").unwrap());
        }
        assert_eq!(server.with_asset(source.id(), |value| value.0), Some(1));
        let weak = source.downgrade();
        drop(source);
        collect(&server);
        assert!(weak.upgrade().is_none(), "source retained its own handle");
        assert_eq!(server.total_assets(), 0);
    }
}

#[test]
fn async_reload_self_dependency_does_not_keep_the_source_alive() {
    let server = Arc::new(server(0));
    let pool = TaskPool::with_threads(0);
    let scheduler = LoadScheduler::new(Arc::clone(&server), pool.clone(), 1);
    let source = pool
        .block_on(
            scheduler
                .schedule("scene.ownership", Priority::Normal)
                .unwrap(),
        )
        .unwrap();
    let weak = source.downgrade();
    mount(&server, 1);
    let reloaded = pool
        .block_on(
            scheduler
                .schedule_reload("scene.ownership", Priority::Normal)
                .unwrap(),
        )
        .unwrap();
    assert_eq!(source.handle_id(), reloaded.handle_id());
    assert_eq!(
        server.with_asset(source.typed::<Source>().unwrap().id(), |value| value.0),
        Some(1)
    );
    drop(reloaded);
    drop(source);
    drop(scheduler);
    collect(&server);
    assert!(
        weak.upgrade().is_none(),
        "async reload retained its own handle"
    );
    assert_eq!(server.total_assets(), 0);
}

#[test]
fn async_initial_self_dependency_fails_without_leaking_a_loading_slot() {
    let server = Arc::new(server(1));
    let pool = TaskPool::with_threads(0);
    let scheduler = LoadScheduler::new(Arc::clone(&server), pool.clone(), 1);
    assert!(pool
        .block_on(
            scheduler
                .schedule("scene.ownership", Priority::Normal)
                .unwrap()
        )
        .is_err());
    drop(scheduler);
    collect(&server);
    assert_eq!(server.total_assets(), 0);
}

#[test]
fn sibling_label_dependency_keeps_values_live_but_releases_the_whole_closure() {
    let server = server(2);
    let source = server.load::<Source>("scene.ownership");
    let source_weak = source.downgrade();
    collect(&server);
    assert_eq!(server.total_assets(), 3);
    let a = server.load::<Label>("scene.ownership#a");
    let b = server.load::<Label>("scene.ownership#b");
    let b_weak = b.downgrade();
    drop(b);
    drop(source);
    collect(&server);
    assert!(source_weak.upgrade().is_some());
    assert!(b_weak.upgrade().is_some());
    assert_eq!(server.with_asset(a.id(), |value| value.0), Some(2));
    drop(a);
    collect(&server);
    assert!(
        source_weak.upgrade().is_none(),
        "sibling dependency pins source"
    );
    assert!(b_weak.upgrade().is_none());
    assert_eq!(server.total_assets(), 0);
}

#[test]
fn promoted_sibling_cycle_is_reclaimed_without_followup_label_resolution() {
    let server = server(2);
    let source = server.load::<Source>("scene.ownership");
    let source_weak = source.downgrade();
    let a = server.load::<Label>("scene.ownership#a");
    let b = server.load::<Label>("scene.ownership#b");
    let b_weak = b.downgrade();

    // A's staged dependency promotes B while the source still hoists B into
    // its committed closure. Both live roots must therefore keep the entire
    // source/label closure resident.
    assert_eq!(server.with_asset(a.id(), |value| value.0), Some(2));
    assert!(source_weak.upgrade().is_some());
    assert!(b_weak.upgrade().is_some());

    // No load/resolve occurs after the roots are dropped. Collection itself
    // must break the server-owned source <-> promoted-label cycle.
    drop(a);
    drop(b);
    drop(source);
    collect(&server);
    assert!(source_weak.upgrade().is_none());
    assert!(b_weak.upgrade().is_none());
    assert_eq!(server.total_assets(), 0);
}

#[test]
fn label_dependency_on_its_primary_does_not_self_pin_the_source() {
    let server = server(3);
    let source = server.load::<Source>("scene.ownership");
    assert_eq!(server.total_assets(), 2);
    drop(source);
    collect(&server);
    assert_eq!(
        server.total_assets(),
        0,
        "label dependency retains source itself"
    );
}

#[test]
fn reload_keeps_a_previously_promoted_sibling_alive_for_a_live_source() {
    let server = server(4);
    let source = server.load::<Source>("scene.ownership");
    let b = server.load::<Label>("scene.ownership#b");
    let b_weak = b.downgrade();
    mount(&server, 2);
    drop(server.reload("scene.ownership").unwrap());
    drop(b);
    collect(&server);
    let b = b_weak
        .upgrade()
        .expect("live source lost its label dependency");
    assert_eq!(server.with_asset(b.id(), |value| value.0), Some(2));
    drop(b);
    drop(source);
    collect(&server);
    assert_eq!(
        server.total_assets(),
        0,
        "reload leaves a promoted sibling cycle"
    );
}

#[test]
fn rollback_failed_slot_has_matching_recursive_state_until_arena_reclaims_it() {
    let server = server(4);
    let source = server.load::<Source>("scene.ownership");
    server.drain_events_with_snapshots::<Label>().unwrap();
    mount(&server, 5);
    drop(server.reload("scene.ownership").unwrap());
    let (events, _) = server.drain_events_with_snapshots::<Label>().unwrap();
    let failed = events
        .iter()
        .find(|event| event.is_failed())
        .unwrap()
        .id()
        .untyped();
    let error = server.load_state(failed).error().unwrap();
    assert_eq!(
        server.recursive_state(failed),
        RecursiveDependencyLoadState::Failed(error)
    );
    collect(&server);
    assert_eq!(server.load_state(failed), LoadState::NotLoaded);
    assert_eq!(
        server.recursive_state(failed),
        RecursiveDependencyLoadState::NotLoaded
    );
    assert_eq!(server.with_asset(source.id(), |value| value.0), Some(4));
    drop(source);
    collect(&server);
    assert_eq!(server.total_assets(), 0);
}

#[test]
fn rollback_propagates_failure_to_nested_consumer_and_releases_promoted_closure() {
    let server = server(4);
    let source = server.load::<Source>("scene.ownership");
    mount(&server, 6);
    drop(server.reload("scene.ownership").unwrap());
    // Do not load again: the temporary consumer handle was dropped on
    // rollback, so another load could create an unrelated allocation.
    let (events, _) = server.drain_events_with_snapshots::<Consumer>().unwrap();
    let consumer_id = events.iter().find(|event| event.is_added()).unwrap().id();
    assert_eq!(server.load_state(consumer_id.untyped()), LoadState::Loaded);
    assert!(matches!(
        server.recursive_state(consumer_id.untyped()),
        RecursiveDependencyLoadState::Failed(_)
    ));
    drop(source);
    collect(&server);
    assert_eq!(server.total_assets(), 0);
}
