//! Public authoring compilation, publication and incremental extraction regressions.

use prism_ecs::world::World;
use prism_render_app::{
    extract_ember_render_assets, EmberCompiledGraphAssets, EmberGraphCompileBudget,
    EmberGraphCompileError, EmberGraphCompilePhase, EmberGraphCompileRequestError,
    EmberGraphSource, EmberGraphStageModule, EmberRenderAssets, ExtractedEmberRenderAssets,
};
use prism_render_architecture::particle::{
    asset_registry::EmberAssetRevision,
    attributes::AttributeSemantic,
    authoring::{
        EmberAsset, EmberAssetHandle, GraphAssetRef, RendererConfig, RendererConfigId,
        RendererKind, SerializationFormat, ShadingSettings,
    },
    compiler::compile_with_data_interfaces,
    graph::{GraphIr, PlatformProfile, StageIr},
    modules::{CodegenCtx, DataInterface, EmberModule, ResourceDecl, StageKind},
    EmberShadingModel, IterationDomain,
};
use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc, Arc, Condvar, Mutex, MutexGuard,
    },
    thread,
    time::{Duration, Instant},
};

static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|poison| poison.into_inner())
}

struct Gate {
    released: Mutex<bool>,
    changed: Condvar,
}

struct Release(Arc<Gate>);

impl Drop for Release {
    fn drop(&mut self) {
        *self.0.released.lock().unwrap() = true;
        self.0.changed.notify_all();
    }
}

struct Module {
    fragment: &'static str,
    entered: Option<mpsc::SyncSender<thread::ThreadId>>,
    gate: Option<Arc<Gate>>,
    panic_once: AtomicBool,
}

impl EmberModule for Module {
    fn stage(&self) -> StageKind {
        StageKind::Update
    }
    fn required_attributes(&self) -> &[AttributeSemantic] {
        &[]
    }
    fn resources(&self) -> &[ResourceDecl] {
        &[]
    }
    fn emit_wesl(&self, ctx: &mut CodegenCtx) {
        if let Some(entered) = &self.entered {
            entered.send(thread::current().id()).unwrap();
        }
        if let Some(gate) = &self.gate {
            let (released, timeout) = gate
                .changed
                .wait_timeout_while(
                    gate.released.lock().unwrap(),
                    Duration::from_secs(10),
                    |released| !*released,
                )
                .unwrap();
            assert!(
                *released && !timeout.timed_out(),
                "test must release worker"
            );
        }
        assert!(
            !self.panic_once.swap(false, Ordering::Relaxed),
            "injected plugin panic"
        );
        ctx.push_fragment(self.fragment);
    }
}

struct Interface;

impl DataInterface for Interface {
    fn bindings(&self) -> &[ResourceDecl] {
        &[]
    }
    fn emit_wesl_functions(&self, ctx: &mut CodegenCtx) {
        assert!(ctx.push_function(
            "authoring_value",
            "fn authoring_value() -> u32 { return 17u; }"
        ));
    }
}

fn source(name: &str, fragment: &'static str) -> Arc<EmberGraphSource> {
    source_with_module(
        name,
        Arc::new(Module {
            fragment,
            entered: None,
            gate: None,
            panic_once: AtomicBool::new(false),
        }),
    )
}

fn source_with_module(name: &str, module: Arc<Module>) -> Arc<EmberGraphSource> {
    Arc::new(EmberGraphSource {
        graph: GraphIr::new(128, EmberShadingModel::Unlit, PlatformProfile::Desktop)
            .with_stage(StageIr::new(name, IterationDomain::PerParticle, 1)),
        modules: vec![EmberGraphStageModule {
            stage_index: 0,
            module,
        }],
        data_interfaces: vec![Arc::new(Interface)],
    })
}

fn blocked_source(
    name: &str,
) -> (
    Arc<EmberGraphSource>,
    mpsc::Receiver<thread::ThreadId>,
    Release,
) {
    let gate = Arc::new(Gate {
        released: Mutex::new(false),
        changed: Condvar::new(),
    });
    let (sender, receiver) = mpsc::sync_channel(1);
    let source = source_with_module(
        name,
        Arc::new(Module {
            fragment: "let authored_value = authoring_value();",
            entered: Some(sender),
            gate: Some(gate.clone()),
            panic_once: AtomicBool::new(false),
        }),
    );
    (source, receiver, Release(gate))
}

fn terminal(assets: &mut EmberCompiledGraphAssets, graph: GraphAssetRef) -> EmberGraphCompilePhase {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        assets.poll_compilations();
        let phase = assets.compile_status(graph).unwrap().phase.clone();
        if matches!(
            phase,
            EmberGraphCompilePhase::Ready { .. } | EmberGraphCompilePhase::Failed(_)
        ) {
            return phase;
        }
        assert!(
            Instant::now() < deadline,
            "compiler did not finish: {phase:?}"
        );
        thread::yield_now();
    }
}

fn ready(assets: &mut EmberCompiledGraphAssets, graph: GraphAssetRef) -> EmberAssetRevision {
    match terminal(assets, graph) {
        EmberGraphCompilePhase::Ready { publication } => publication,
        phase => panic!("expected ready, got {phase:?}"),
    }
}

fn drain(assets: &mut EmberCompiledGraphAssets) {
    let barrier = GraphAssetRef(u32::MAX);
    assets
        .request_compile(barrier, 1, source("barrier", ""))
        .unwrap();
    ready(assets, barrier);
    assets.remove_graph(barrier).unwrap();
}

#[test]
fn authoring_modules_and_interfaces_compile_off_thread_and_publish_validated_wgsl() {
    let _serial = serial();
    let mut assets = EmberCompiledGraphAssets::new();
    let graph = GraphAssetRef(1);
    let (source, entered, release) = blocked_source("advance");
    let ticket = assets.request_compile(graph, 4, source).unwrap();
    assert_eq!(
        assets.compile_status(graph).unwrap().phase,
        EmberGraphCompilePhase::Queued
    );
    assert_eq!(assets.poll_compilations(), 0);
    assert_ne!(
        entered.recv_timeout(Duration::from_secs(5)).unwrap(),
        thread::current().id()
    );
    for _ in 0..16 {
        assert_eq!(assets.poll_compilations(), 0);
        assert!(assets.registry().get(graph).is_none());
    }
    drop(release);
    let publication = ready(&mut assets, graph);
    let compiled = assets.registry().get(graph).unwrap();
    assert_eq!(compiled.revision(), publication);
    assert!(compiled.graph().source.contains("authoring_value()"));
    assert!(compiled.graph().source.contains("authored_value"));
    assert_eq!(assets.compile_status(graph).unwrap().ticket, ticket);
    assert_eq!(assets.poll_compilations(), 0);
}

#[test]
fn newest_source_coalesces_queued_revisions_and_discards_inflight_result() {
    let _serial = serial();
    let mut assets = EmberCompiledGraphAssets::new();
    let graph = GraphAssetRef(2);
    let (old, entered, release) = blocked_source("old");
    let old_ticket = assets.request_compile(graph, 1, old).unwrap();
    assets.poll_compilations();
    entered.recv_timeout(Duration::from_secs(5)).unwrap();
    assets
        .request_compile(graph, 2, source("obsolete", "invalid syntax"))
        .unwrap();
    let latest = assets
        .request_compile(graph, 3, source("newest", "let newest = 3u;"))
        .unwrap();
    assert!(!assets.retry_compile(old_ticket));
    drop(release);
    assert_eq!(
        ready(&mut assets, graph).value(),
        1,
        "obsolete results never publish"
    );
    assert_eq!(assets.compile_status(graph).unwrap().ticket, latest);
    assert!(assets
        .registry()
        .get(graph)
        .unwrap()
        .graph()
        .source
        .contains("let newest = 3u;"));
}

#[test]
fn equal_revision_requires_exact_source_identity_not_graph_hash() {
    let _serial = serial();
    let mut assets = EmberCompiledGraphAssets::new();
    let graph = GraphAssetRef(3);
    let original = source("same", "let value = 1u;");
    let changed_plugin = source("same", "let value = 2u;");
    assert_eq!(
        original.graph.specialization_key(),
        changed_plugin.graph.specialization_key()
    );
    let ticket = assets.request_compile(graph, 9, original.clone()).unwrap();
    assert_eq!(assets.request_compile(graph, 9, original), Ok(ticket));
    assert_eq!(
        assets.request_compile(graph, 9, changed_plugin),
        Err(EmberGraphCompileRequestError::RevisionConflict)
    );
    assert_eq!(
        assets.request_compile(graph, 8, source("older", "")),
        Err(EmberGraphCompileRequestError::StaleRevision {
            expected: 9,
            actual: 8
        })
    );
    ready(&mut assets, graph);
}

#[test]
fn cancel_and_reinsert_never_reuses_an_inflight_ticket() {
    let _serial = serial();
    let mut assets = EmberCompiledGraphAssets::new();
    let graph = GraphAssetRef(4);
    let (old, entered, release) = blocked_source("removed");
    let old_ticket = assets.request_compile(graph, 1, old.clone()).unwrap();
    assets.poll_compilations();
    entered.recv_timeout(Duration::from_secs(5)).unwrap();
    assert!(assets.cancel_compile(graph));
    let new_ticket = assets
        .request_compile(graph, 1, source("reinserted", "let replacement = 1u;"))
        .unwrap();
    assert_ne!(old_ticket, new_ticket);
    drop(release);
    assert_eq!(ready(&mut assets, graph).value(), 1);
    assert!(assets
        .registry()
        .get(graph)
        .unwrap()
        .graph()
        .source
        .contains("replacement"));
}

#[test]
fn mutable_registry_boundary_cancels_workers_and_preserves_manual_publication() {
    let _serial = serial();
    let mut assets = EmberCompiledGraphAssets::new();
    let graph = GraphAssetRef(5);
    let (source, entered, release) = blocked_source("stale");
    assets.request_compile(graph, 1, source).unwrap();
    assets.poll_compilations();
    entered.recv_timeout(Duration::from_secs(5)).unwrap();
    let manual =
        compile_with_data_interfaces(&crate::source("manual", "").graph, &[], &[]).unwrap();
    let revision = assets
        .registry_mut()
        .publish(graph, manual.clone())
        .unwrap();
    assert!(assets.compile_status(graph).is_none());
    drop(release);
    drain(&mut assets);
    assert_eq!(assets.registry().get(graph).unwrap().revision(), revision);
    assert_eq!(assets.registry().get(graph).unwrap().graph(), &manual);
}

#[test]
fn failed_compile_and_plugin_panic_keep_last_graph_and_allow_explicit_retry() {
    let _serial = serial();
    let mut assets = EmberCompiledGraphAssets::new();
    let graph = GraphAssetRef(6);
    assets
        .request_compile(graph, 1, source("good", "let value = 1u;"))
        .unwrap();
    let revision = ready(&mut assets, graph);
    let previous = assets.registry().get(graph).unwrap().graph().clone();
    assets
        .request_compile(graph, 2, source("bad", "not wgsl syntax"))
        .unwrap();
    assert!(matches!(
        terminal(&mut assets, graph),
        EmberGraphCompilePhase::Failed(EmberGraphCompileError::Compile(_))
    ));
    assert_eq!(assets.registry().get(graph).unwrap().graph(), &previous);
    let transient = source_with_module(
        "retry",
        Arc::new(Module {
            fragment: "let recovered = 1u;",
            entered: None,
            gate: None,
            panic_once: AtomicBool::new(true),
        }),
    );
    let ticket = assets.request_compile(graph, 3, transient).unwrap();
    assert_eq!(
        terminal(&mut assets, graph),
        EmberGraphCompilePhase::Failed(EmberGraphCompileError::PluginPanicked)
    );
    for _ in 0..16 {
        assert_eq!(assets.poll_compilations(), 0);
    }
    assert_eq!(assets.registry().get(graph).unwrap().revision(), revision);
    assert_eq!(assets.registry().get(graph).unwrap().graph(), &previous);
    assert!(assets.retry_compile(ticket));
    assert!(!assets.retry_compile(ticket));
    assert!(ready(&mut assets, graph) > revision);
    assert!(!assets.retry_compile(ticket));
}

#[test]
fn admission_limits_are_atomic_and_cancellation_releases_capacity() {
    let _serial = serial();
    assert!(matches!(
        EmberCompiledGraphAssets::with_compile_budget(EmberGraphCompileBudget {
            max_graphs: 0,
            ..Default::default()
        }),
        Err(EmberGraphCompileRequestError::InvalidBudget)
    ));
    let mut assets = EmberCompiledGraphAssets::with_compile_budget(EmberGraphCompileBudget {
        max_graphs: 1,
        max_stages: 1,
        max_modules: 1,
        max_data_interfaces: 1,
        ..Default::default()
    })
    .unwrap();
    let graph = GraphAssetRef(7);
    let original = source("good", "");
    let ticket = assets.request_compile(graph, 1, original.clone()).unwrap();
    assert_eq!(
        assets.request_compile(GraphAssetRef(8), 1, original),
        Err(EmberGraphCompileRequestError::GraphCapacity)
    );
    for dimension in ["stages", "modules", "data_interfaces"] {
        let mut too_big = source("bad", "");
        let too_big_mut = Arc::get_mut(&mut too_big).unwrap();
        match dimension {
            "stages" => too_big_mut.graph.stages.push(StageIr::new(
                "second",
                IterationDomain::PerParticle,
                1,
            )),
            "modules" => too_big_mut.modules.push(EmberGraphStageModule {
                stage_index: 0,
                module: Arc::new(Module {
                    fragment: "",
                    entered: None,
                    gate: None,
                    panic_once: AtomicBool::new(false),
                }),
            }),
            _ => too_big_mut.data_interfaces.push(Arc::new(Interface)),
        }
        assert_eq!(
            assets.request_compile(graph, 2, too_big),
            Err(EmberGraphCompileRequestError::SourceLimit {
                kind: dimension,
                actual: 2,
                limit: 1
            })
        );
        assert_eq!(assets.compile_status(graph).unwrap().ticket, ticket);
    }
    assert!(assets.cancel_compile(graph));
    assert!(!assets.cancel_compile(graph));
    assets
        .request_compile(GraphAssetRef(8), 1, source("admitted", ""))
        .unwrap();
    ready(&mut assets, GraphAssetRef(8));
}

#[test]
fn oversized_generated_source_cannot_replace_a_live_graph() {
    let _serial = serial();
    let mut assets = EmberCompiledGraphAssets::with_compile_budget(EmberGraphCompileBudget {
        max_generated_source_bytes: 1,
        ..Default::default()
    })
    .unwrap();
    let graph = GraphAssetRef(9);
    let previous = compile_with_data_interfaces(&source("previous", "").graph, &[], &[]).unwrap();
    let revision = assets
        .registry_mut()
        .publish(graph, previous.clone())
        .unwrap();
    assets
        .request_compile(graph, 1, source("oversized", ""))
        .unwrap();
    assert!(
        matches!(terminal(&mut assets, graph), EmberGraphCompilePhase::Failed(EmberGraphCompileError::GeneratedSourceLimit { limit: 1, actual }) if actual > 1)
    );
    assert_eq!(assets.registry().get(graph).unwrap().revision(), revision);
    assert_eq!(assets.registry().get(graph).unwrap().graph(), &previous);
}

#[test]
fn resource_replacement_never_multiplies_running_workers_or_waits_for_plugins() {
    let _serial = serial();
    let mut old = EmberCompiledGraphAssets::new();
    let graph = GraphAssetRef(11);
    let (source, entered, release) = blocked_source("detached");
    old.request_compile(graph, 1, source).unwrap();
    old.poll_compilations();
    entered.recv_timeout(Duration::from_secs(5)).unwrap();
    drop(old);
    let mut replacement = EmberCompiledGraphAssets::new();
    replacement
        .request_compile(
            graph,
            1,
            crate::source("replacement", "let replacement = 7u;"),
        )
        .unwrap();
    for _ in 0..16 {
        assert_eq!(replacement.poll_compilations(), 0);
        assert_eq!(
            replacement.compile_status(graph).unwrap().phase,
            EmberGraphCompilePhase::Queued
        );
    }
    drop(release);
    ready(&mut replacement, graph);
    assert!(replacement
        .registry()
        .get(graph)
        .unwrap()
        .graph()
        .source
        .contains("replacement = 7u"));
}

#[test]
fn extraction_advances_compilation_without_instances_or_camera() {
    let _serial = serial();
    let mut main = World::new();
    let mut render = World::new();
    let graph = GraphAssetRef(12);
    let mut assets = EmberCompiledGraphAssets::new();
    assets
        .request_compile(graph, 1, source("headless", "let headless = 1u;"))
        .unwrap();
    main.insert_resource(assets);
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        extract_ember_render_assets(&mut main, &mut render);
        if main
            .resource::<EmberCompiledGraphAssets>()
            .registry()
            .get(graph)
            .is_some()
        {
            break;
        }
        assert!(Instant::now() < deadline);
        thread::yield_now();
    }
    assert!(matches!(
        main.resource::<EmberCompiledGraphAssets>()
            .compile_status(graph)
            .unwrap()
            .phase,
        EmberGraphCompilePhase::Ready { .. }
    ));
}

#[test]
fn graph_removal_cancels_pending_publication() {
    let _serial = serial();
    let mut assets = EmberCompiledGraphAssets::new();
    let graph = GraphAssetRef(13);
    assets
        .request_compile(graph, 1, source("published", ""))
        .unwrap();
    ready(&mut assets, graph);
    let (pending, entered, release) = blocked_source("removed");
    assets.request_compile(graph, 2, pending).unwrap();
    assets.poll_compilations();
    entered.recv_timeout(Duration::from_secs(5)).unwrap();
    assets.remove_graph(graph).unwrap();
    drop(release);
    drain(&mut assets);
    assert!(assets.registry().get(graph).is_none());
    assert!(assets.compile_status(graph).is_none());
}

fn authored_asset(graph: GraphAssetRef) -> EmberAsset {
    EmberAsset {
        format_version: 1,
        graph,
        params: vec![],
        renderer_config: RendererConfig {
            id: RendererConfigId(0),
            kind: RendererKind::Sprite,
            writes_motion_vectors: false,
        },
        shading: ShadingSettings {
            model: EmberShadingModel::Unlit,
            receive_shadows: false,
            cast_shadows: false,
        },
        format: SerializationFormat::Binary,
    }
}

fn extracted_graph(
    render: &World,
    handle: EmberAssetHandle,
) -> Arc<prism_render_architecture::particle::compiler::CompiledGraph> {
    render
        .resource::<ExtractedEmberRenderAssets>()
        .get(handle)
        .unwrap()
        .compiled
        .as_ref()
        .unwrap()
        .clone()
}

#[test]
fn instances_share_registry_publication_across_payload_changes_and_graph_reload() {
    let _serial = serial();
    let graph = GraphAssetRef(14);
    let first = EmberAssetHandle(1);
    let second = EmberAssetHandle(2);
    let mut main = World::new();
    let mut render = World::new();
    let mut compiled = EmberCompiledGraphAssets::new();
    compiled
        .request_compile(graph, 1, source("initial", "let initial = 1u;"))
        .unwrap();
    let initial_revision = ready(&mut compiled, graph);
    let initial = compiled
        .registry()
        .get(graph)
        .unwrap()
        .shared_graph()
        .clone();
    main.insert_resource(compiled);
    main.insert_resource(EmberRenderAssets::default());
    for handle in [second, first] {
        main.resource_mut::<EmberRenderAssets>().upsert(
            handle,
            1,
            authored_asset(graph),
            vec![],
            vec![128],
            vec![],
        );
    }
    extract_ember_render_assets(&mut main, &mut render);
    assert!(Arc::ptr_eq(&initial, &extracted_graph(&render, first)));
    assert!(Arc::ptr_eq(&initial, &extracted_graph(&render, second)));
    assert_eq!(
        render
            .resource::<ExtractedEmberRenderAssets>()
            .iter()
            .map(|(handle, _)| handle)
            .collect::<Vec<_>>(),
        [first, second]
    );

    main.resource_mut::<EmberRenderAssets>().upsert(
        first,
        2,
        authored_asset(graph),
        vec![],
        vec![64],
        vec![1, 2, 3],
    );
    extract_ember_render_assets(&mut main, &mut render);
    let changed = render
        .resource::<ExtractedEmberRenderAssets>()
        .get(first)
        .unwrap();
    assert_eq!(changed.asset_revision.value(), 2);
    assert_eq!(changed.graph_revision, Some(initial_revision));
    assert_eq!(changed.stage_counts.as_slice(), [64]);
    assert_eq!(changed.attribute_bytes.as_slice(), [1, 2, 3]);
    assert!(Arc::ptr_eq(&initial, changed.compiled.as_ref().unwrap()));

    main.resource_mut::<EmberCompiledGraphAssets>()
        .request_compile(graph, 2, source("replacement", "let replacement = 2u;"))
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        extract_ember_render_assets(&mut main, &mut render);
        if extracted_graph(&render, first)
            .source
            .contains("replacement = 2u")
        {
            break;
        }
        assert!(Instant::now() < deadline);
        thread::yield_now();
    }
    let replacement = extracted_graph(&render, first);
    assert!(!Arc::ptr_eq(&initial, &replacement));
    assert!(Arc::ptr_eq(&replacement, &extracted_graph(&render, second)));
    let registry = main
        .resource::<EmberCompiledGraphAssets>()
        .registry()
        .get(graph)
        .unwrap();
    assert!(Arc::ptr_eq(&replacement, registry.shared_graph()));
    assert_ne!(
        Some(initial_revision),
        render
            .resource::<ExtractedEmberRenderAssets>()
            .get(first)
            .unwrap()
            .graph_revision
    );
    assert!(
        initial.source.contains("initial = 1u"),
        "old render/build readers retain immutable source"
    );

    main.resource_mut::<EmberCompiledGraphAssets>()
        .request_compile(graph, 3, source("invalid", "this is invalid WGSL"))
        .unwrap();
    terminal(&mut main.resource_mut::<EmberCompiledGraphAssets>(), graph);
    extract_ember_render_assets(&mut main, &mut render);
    assert!(Arc::ptr_eq(&replacement, &extracted_graph(&render, first)));
    assert!(Arc::ptr_eq(&replacement, &extracted_graph(&render, second)));
}

#[test]
fn extraction_unresolves_removed_graph_and_prunes_instances_without_invalidating_readers() {
    let _serial = serial();
    let graph = GraphAssetRef(15);
    let handle = EmberAssetHandle(1);
    let mut main = World::new();
    let mut render = World::new();
    let mut compiled = EmberCompiledGraphAssets::new();
    compiled
        .request_compile(graph, 1, source("retained", "let retained = 1u;"))
        .unwrap();
    ready(&mut compiled, graph);
    main.insert_resource(compiled);
    main.insert_resource(EmberRenderAssets::default());
    main.resource_mut::<EmberRenderAssets>().upsert(
        handle,
        1,
        authored_asset(graph),
        vec![],
        vec![],
        vec![],
    );
    extract_ember_render_assets(&mut main, &mut render);
    let retained = extracted_graph(&render, handle);

    main.resource_mut::<EmberCompiledGraphAssets>()
        .remove_graph(graph)
        .unwrap();
    extract_ember_render_assets(&mut main, &mut render);
    let unresolved = render
        .resource::<ExtractedEmberRenderAssets>()
        .get(handle)
        .unwrap();
    assert!(unresolved.graph_revision.is_none());
    assert!(unresolved.compiled.is_none());
    assert!(retained.source.contains("retained = 1u"));

    main.resource_mut::<EmberRenderAssets>().remove(handle);
    extract_ember_render_assets(&mut main, &mut render);
    assert!(render
        .resource::<ExtractedEmberRenderAssets>()
        .get(handle)
        .is_none());
    assert_eq!(
        render
            .resource::<ExtractedEmberRenderAssets>()
            .iter()
            .count(),
        0
    );
}
