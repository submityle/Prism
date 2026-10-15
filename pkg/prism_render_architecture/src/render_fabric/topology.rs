//! Thin semantic-ID adapter, not a scheduler. Never attach resource accesses:
//! hazards, frame versions and barriers remain the parent's frame-graph concern.

use alloc::{
    borrow::Cow,
    collections::{BTreeMap, BTreeSet},
    vec::Vec,
};

use crate::frame_graph::{CompileError, GpuFrameGraphBuilder, PassDescriptor, PassId};

use super::{CatalogError, FeatureCatalog, FeatureId, QueueClass};

pub(super) fn order(
    catalog: &FeatureCatalog,
    closure: &BTreeSet<FeatureId>,
) -> Result<Vec<FeatureId>, CatalogError> {
    let mut nodes: Vec<_> = closure.iter().copied().collect();
    nodes.sort_by_key(|&id| {
        (
            !catalog
                .feature(id)
                .expect("validated feature")
                .kind
                .is_service(),
            id,
        )
    });
    if nodes.len() > u32::MAX as usize {
        return Err(CatalogError::TopologyCapacityExceeded);
    }
    let indices: BTreeMap<_, _> = nodes
        .iter()
        .enumerate()
        .map(|(index, &id)| (id, PassId(index as u32)))
        .collect();
    let mut graph = GpuFrameGraphBuilder::default();
    for &id in &nodes {
        let feature = catalog.feature(id).expect("validated feature");
        let mut dependencies = BTreeSet::new();
        for &service in feature.requires_services.as_slice() {
            let provider = catalog.provider(service).expect("validated provider");
            if provider == id {
                return Err(CatalogError::CyclicHardDependencies);
            }
            dependencies.insert(provider);
        }
        graph
            .try_add_pass(PassDescriptor {
                name: Cow::Owned(format!("render-fabric-feature-{}", id.0)),
                queue: QueueClass::Graphics,
                accesses: Vec::new(),
                depends_on: dependencies
                    .iter()
                    .map(|provider| indices[provider])
                    .collect(),
            })
            .map_err(map_compile_error)?;
    }
    match graph.compile() {
        Ok(compiled) => Ok(compiled.order.iter().map(|&pass| nodes[pass]).collect()),
        Err(error) => Err(map_compile_error(error)),
    }
}

fn map_compile_error(error: CompileError) -> CatalogError {
    match error {
        CompileError::Cycle { .. } => CatalogError::CyclicHardDependencies,
        CompileError::Capacity { .. } => CatalogError::TopologyCapacityExceeded,
        other => CatalogError::TopologyCompilationFailed(other.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn authoritative_graph_admission_and_compile_failures_are_reported() {
        assert_eq!(
            map_compile_error(CompileError::Capacity {
                kind: "passes",
                count: 2,
                limit: 1,
            }),
            CatalogError::TopologyCapacityExceeded
        );
        assert_eq!(
            map_compile_error(CompileError::Cycle {
                unscheduled: 2,
                blocked: vec![0, 1],
                cycle: vec![0, 1, 0],
                names: vec!["a".into(), "b".into(), "a".into()],
            }),
            CatalogError::CyclicHardDependencies
        );
        assert!(matches!(
            map_compile_error(CompileError::MemoryOverflow),
            CatalogError::TopologyCompilationFailed(_)
        ));
    }
}
