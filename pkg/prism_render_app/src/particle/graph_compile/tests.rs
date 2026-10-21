use super::*;
use crate::particle::EmberCompiledGraphAssets;
use prism_render_architecture::particle::{
    graph::{PlatformProfile, StageIr},
    EmberShadingModel, IterationDomain,
};
use std::time::{Duration, Instant};

fn source() -> Arc<EmberGraphSource> {
    Arc::new(EmberGraphSource {
        graph: GraphIr::new(128, EmberShadingModel::Unlit, PlatformProfile::Desktop)
            .with_stage(StageIr::new("advance", IterationDomain::PerParticle, 1)),
        modules: Vec::new(),
        data_interfaces: Vec::new(),
    })
}

#[test]
fn publication_changed_cannot_overwrite_the_registry_or_be_retried() {
    let mut assets = EmberCompiledGraphAssets::new();
    let graph = GraphAssetRef(10);
    let source = source();
    let ticket = assets.request_compile(graph, 1, source.clone()).unwrap();
    let previous = compile_with_data_interfaces(&source.graph, &[], &[]).unwrap();
    assets.registry.publish(graph, previous.clone()).unwrap();
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        assets.poll_compilations();
        if matches!(
            assets.compile_status(graph).unwrap().phase,
            EmberGraphCompilePhase::Failed(_)
        ) {
            break;
        }
        assert!(Instant::now() < deadline);
        thread::yield_now();
    }
    assert_eq!(
        assets.compile_status(graph).unwrap().phase,
        EmberGraphCompilePhase::Failed(EmberGraphCompileError::PublicationChanged)
    );
    assert!(!assets.retry_compile(ticket));
    assert_eq!(assets.registry().get(graph).unwrap().graph(), &previous);
}

#[test]
fn ticket_exhaustion_leaves_the_latest_desired_source_unchanged() {
    let mut assets = EmberCompiledGraphAssets::new();
    let graph = GraphAssetRef(13);
    let ticket = assets.request_compile(graph, 1, source()).unwrap();
    assets.compiler.sequence = u64::MAX;
    assert_eq!(
        assets.request_compile(graph, 2, source()),
        Err(EmberGraphCompileRequestError::TicketExhausted)
    );
    assert_eq!(assets.compile_status(graph).unwrap().ticket, ticket);
}
