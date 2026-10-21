//! Deterministic runtime registry for versioned `.ember` assets.
//!
//! Registries own CPU-side assets and immutable compiled-graph publications,
//! with revisions used to reject stale asynchronous reload results. Extracted
//! readers share publications without copying their shader source or metadata.
//! GPU buffers and pipeline objects remain owned by the render layer.

use alloc::collections::BTreeMap;
use alloc::sync::Arc;

use super::authoring::{
    decode_binary, EmberAsset, EmberAssetHandle, EmberCodecError, GraphAssetRef, HotReload,
    ReloadEvent, ReloadState,
};
use super::compiler::CompiledGraph;

/// Monotonic revision of one registered asset.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct EmberAssetRevision(u64);

impl EmberAssetRevision {
    /// Creates a revision received from an external asset database or an
    /// extracted render record.
    ///
    /// Revision values are opaque to the registry, but render-world bridges
    /// must be able to carry the value across the main/render world boundary.
    #[must_use]
    pub const fn from_value(value: u64) -> Self {
        Self(value)
    }

    /// Returns the numeric revision used by render caches and diagnostics.
    #[must_use]
    pub const fn value(self) -> u64 {
        self.0
    }
}

/// A generation-checked asynchronous reload ticket.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct EmberReloadTicket {
    handle: EmberAssetHandle,
    revision: EmberAssetRevision,
}

impl EmberReloadTicket {
    /// Asset handle captured when the reload started.
    #[must_use]
    pub const fn handle(self) -> EmberAssetHandle {
        self.handle
    }

    /// Live revision captured when the reload started.
    #[must_use]
    pub const fn revision(self) -> EmberAssetRevision {
        self.revision
    }
}

/// Immutable view of one live asset and its reload metadata.
pub struct EmberAssetEntry {
    asset: EmberAsset,
    revision: EmberAssetRevision,
    reload: HotReload,
}

impl EmberAssetEntry {
    /// The currently published asset.
    #[must_use]
    pub const fn asset(&self) -> &EmberAsset {
        &self.asset
    }

    /// The current registry revision.
    #[must_use]
    pub const fn revision(&self) -> EmberAssetRevision {
        self.revision
    }

    /// Current reload state for this asset.
    #[must_use]
    pub const fn reload_state(&self) -> ReloadState {
        self.reload.state
    }

    /// Pipeline generation tracked by the hot-reload state machine.
    #[must_use]
    pub const fn pipeline_generation(&self) -> u32 {
        self.reload.generation
    }
}

/// Errors returned by [`EmberAssetRegistry`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EmberAssetRegistryError {
    /// The requested handle is already present.
    DuplicateHandle(EmberAssetHandle),
    /// The requested handle is not present.
    UnknownHandle(EmberAssetHandle),
    /// No free `u32` handle remains.
    HandleSpaceExhausted,
    /// An asynchronous result belongs to an older registry revision.
    StaleReload {
        handle: EmberAssetHandle,
        expected: EmberAssetRevision,
        actual: EmberAssetRevision,
    },
    /// The requested hot-reload transition is not valid in the current state.
    InvalidReloadState {
        handle: EmberAssetHandle,
        expected: ReloadState,
        actual: ReloadState,
    },
    /// The incoming asset bytes are not a valid `.ember` binary.
    Decode(EmberCodecError),
}

/// An immutable compiled graph together with its published revision.
pub struct EmberCompiledGraphEntry {
    graph: Arc<CompiledGraph>,
    revision: EmberAssetRevision,
}

impl EmberCompiledGraphEntry {
    /// Returns the validated compiled graph.
    #[must_use]
    pub fn graph(&self) -> &CompiledGraph {
        &self.graph
    }

    /// Shares this exact immutable publication without copying shader source,
    /// reflection or dispatch metadata. Older publications remain alive until
    /// all extracted instances and pending GPU builds release their references.
    #[must_use]
    pub const fn shared_graph(&self) -> &Arc<CompiledGraph> {
        &self.graph
    }

    /// Returns the graph publication revision.
    #[must_use]
    pub const fn revision(&self) -> EmberAssetRevision {
        self.revision
    }
}

/// Errors returned by [`EmberCompiledGraphRegistry`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EmberCompiledGraphRegistryError {
    /// A graph handle is already present and an explicit duplicate insert was
    /// attempted.
    DuplicateGraph(GraphAssetRef),
    /// A publication result belongs to an older or equal graph revision.
    StaleGraph {
        /// Graph whose publication was rejected.
        graph: GraphAssetRef,
        /// Revision currently published.
        expected: EmberAssetRevision,
        /// Revision supplied by the caller.
        actual: EmberAssetRevision,
    },
    /// The requested graph handle is not present.
    UnknownGraph(GraphAssetRef),
    /// The registry revision counter wrapped around.
    RevisionExhausted,
}

/// Formal `GraphAssetRef → CompiledGraph` registry.
///
/// Compilation publishes an already validated [`CompiledGraph`] atomically.
/// The registry never exposes a partially built graph, and an older async
/// result cannot replace a newer publication. GPU resource realization remains
/// the responsibility of the render layer.
#[derive(Default)]
pub struct EmberCompiledGraphRegistry {
    graphs: BTreeMap<GraphAssetRef, EmberCompiledGraphEntry>,
    next_revision: u64,
}

impl EmberCompiledGraphRegistry {
    /// Creates an empty compiled-graph registry.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            graphs: BTreeMap::new(),
            next_revision: 0,
        }
    }

    /// Number of published compiled graphs.
    #[must_use]
    pub fn len(&self) -> usize {
        self.graphs.len()
    }

    /// Returns the graph entry for a stable graph reference.
    #[must_use]
    pub fn get(&self, graph: GraphAssetRef) -> Option<&EmberCompiledGraphEntry> {
        self.graphs.get(&graph)
    }

    /// Returns deterministic graph/revision pairs.
    pub fn iter(&self) -> impl Iterator<Item = (GraphAssetRef, &EmberCompiledGraphEntry)> {
        self.graphs.iter().map(|(graph, entry)| (*graph, entry))
    }

    /// Publishes a new compiled graph with a registry-owned revision.
    pub fn publish(
        &mut self,
        graph: GraphAssetRef,
        compiled: CompiledGraph,
    ) -> Result<EmberAssetRevision, EmberCompiledGraphRegistryError> {
        let revision = self
            .next_revision
            .checked_add(1)
            .ok_or(EmberCompiledGraphRegistryError::RevisionExhausted)?;
        self.next_revision = revision;
        let revision = EmberAssetRevision::from_value(revision);
        self.graphs.insert(
            graph,
            EmberCompiledGraphEntry {
                graph: Arc::new(compiled),
                revision,
            },
        );
        Ok(revision)
    }

    /// Inserts a graph under an explicit revision without replacing a live
    /// graph.
    pub fn insert(
        &mut self,
        graph: GraphAssetRef,
        revision: EmberAssetRevision,
        compiled: CompiledGraph,
    ) -> Result<(), EmberCompiledGraphRegistryError> {
        if self.graphs.contains_key(&graph) {
            return Err(EmberCompiledGraphRegistryError::DuplicateGraph(graph));
        }
        self.next_revision = self.next_revision.max(revision.value());
        self.graphs.insert(
            graph,
            EmberCompiledGraphEntry {
                graph: Arc::new(compiled),
                revision,
            },
        );
        Ok(())
    }

    /// Atomically replaces a graph if the incoming revision is newer.
    pub fn replace_if_newer(
        &mut self,
        graph: GraphAssetRef,
        revision: EmberAssetRevision,
        compiled: CompiledGraph,
    ) -> Result<(), EmberCompiledGraphRegistryError> {
        if let Some(current) = self.graphs.get(&graph) {
            if revision <= current.revision {
                return Err(EmberCompiledGraphRegistryError::StaleGraph {
                    graph,
                    expected: current.revision,
                    actual: revision,
                });
            }
        }
        self.next_revision = self.next_revision.max(revision.value());
        self.graphs.insert(
            graph,
            EmberCompiledGraphEntry {
                graph: Arc::new(compiled),
                revision,
            },
        );
        Ok(())
    }

    /// Removes and returns the shared publication without copying it, even
    /// while extracted readers or pending device builds retain the same graph.
    pub fn remove(
        &mut self,
        graph: GraphAssetRef,
    ) -> Result<Arc<CompiledGraph>, EmberCompiledGraphRegistryError> {
        self.graphs
            .remove(&graph)
            .map(|entry| entry.graph)
            .ok_or(EmberCompiledGraphRegistryError::UnknownGraph(graph))
    }
}

/// CPU-side registry for live `.ember` assets.
///
/// `BTreeMap` makes iteration and lookup order deterministic. Replacing an
/// asset always advances its revision, so render-side caches can compare one
/// integer and never observe a partially published asset.
#[derive(Default)]
pub struct EmberAssetRegistry {
    assets: BTreeMap<EmberAssetHandle, EmberAssetEntry>,
    next_handle: u32,
    next_revision: u64,
}

impl EmberAssetRegistry {
    /// Creates an empty registry.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            assets: BTreeMap::new(),
            next_handle: 0,
            next_revision: 0,
        }
    }

    /// Number of live assets.
    #[must_use]
    pub fn len(&self) -> usize {
        self.assets.len()
    }

    /// Whether the registry has no live assets.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.assets.is_empty()
    }

    /// Returns a deterministic handle/revision/asset iterator.
    pub fn iter(&self) -> impl Iterator<Item = (EmberAssetHandle, &EmberAssetEntry)> {
        self.assets.iter().map(|(handle, entry)| (*handle, entry))
    }

    /// Looks up a live asset entry.
    #[must_use]
    pub fn get(&self, handle: EmberAssetHandle) -> Option<&EmberAssetEntry> {
        self.assets.get(&handle)
    }

    /// Inserts an asset under an explicit stable handle.
    pub fn insert(
        &mut self,
        handle: EmberAssetHandle,
        asset: EmberAsset,
    ) -> Result<EmberAssetRevision, EmberAssetRegistryError> {
        if self.assets.contains_key(&handle) {
            return Err(EmberAssetRegistryError::DuplicateHandle(handle));
        }
        let revision = self.next_revision();
        self.assets.insert(
            handle,
            EmberAssetEntry {
                asset,
                revision,
                reload: HotReload::default(),
            },
        );
        self.next_handle = self.next_handle.max(handle.0.saturating_add(1));
        Ok(revision)
    }

    /// Allocates the next unused stable handle and inserts an asset.
    pub fn allocate(
        &mut self,
        asset: EmberAsset,
    ) -> Result<(EmberAssetHandle, EmberAssetRevision), EmberAssetRegistryError> {
        let start = self.next_handle;
        loop {
            let handle = EmberAssetHandle(self.next_handle);
            self.next_handle = self.next_handle.wrapping_add(1);
            if !self.assets.contains_key(&handle) {
                let revision = self.insert(handle, asset)?;
                return Ok((handle, revision));
            }
            if self.next_handle == start {
                return Err(EmberAssetRegistryError::HandleSpaceExhausted);
            }
        }
    }

    /// Decodes and inserts a binary `.ember` asset under an explicit handle.
    pub fn insert_binary(
        &mut self,
        handle: EmberAssetHandle,
        bytes: &[u8],
    ) -> Result<EmberAssetRevision, EmberAssetRegistryError> {
        let asset = decode_binary(bytes).map_err(EmberAssetRegistryError::Decode)?;
        self.insert(handle, asset)
    }

    /// Removes an asset and returns its owned payload.
    pub fn remove(
        &mut self,
        handle: EmberAssetHandle,
    ) -> Result<EmberAsset, EmberAssetRegistryError> {
        self.assets
            .remove(&handle)
            .map(|entry| entry.asset)
            .ok_or(EmberAssetRegistryError::UnknownHandle(handle))
    }

    /// Starts a graph/shading reload and captures the current revision.
    pub fn begin_reload(
        &mut self,
        handle: EmberAssetHandle,
    ) -> Result<EmberReloadTicket, EmberAssetRegistryError> {
        let entry = self
            .assets
            .get_mut(&handle)
            .ok_or(EmberAssetRegistryError::UnknownHandle(handle))?;
        let actual = entry.reload.state;
        if actual != ReloadState::Idle && actual != ReloadState::Failed {
            return Err(EmberAssetRegistryError::InvalidReloadState {
                handle,
                expected: ReloadState::Idle,
                actual,
            });
        }
        let event = if actual == ReloadState::Failed {
            ReloadEvent::Retry
        } else {
            ReloadEvent::GraphEdited
        };
        entry.reload.on_event(event);
        Ok(EmberReloadTicket {
            handle,
            revision: entry.revision,
        })
    }

    /// Marks a generation-checked reload compile as successful.
    pub fn mark_compile_succeeded(
        &mut self,
        ticket: EmberReloadTicket,
    ) -> Result<(), EmberAssetRegistryError> {
        let entry = self.entry_for_ticket(ticket)?;
        let actual = entry.reload.state;
        if actual != ReloadState::Recompiling {
            return Err(EmberAssetRegistryError::InvalidReloadState {
                handle: ticket.handle,
                expected: ReloadState::Recompiling,
                actual,
            });
        }
        entry.reload.on_event(ReloadEvent::CompileSucceeded);
        Ok(())
    }

    /// Records a generation-checked reload compile failure while keeping the
    /// previous asset live.
    pub fn mark_compile_failed(
        &mut self,
        ticket: EmberReloadTicket,
    ) -> Result<(), EmberAssetRegistryError> {
        let entry = self.entry_for_ticket(ticket)?;
        let actual = entry.reload.state;
        if actual != ReloadState::Recompiling {
            return Err(EmberAssetRegistryError::InvalidReloadState {
                handle: ticket.handle,
                expected: ReloadState::Recompiling,
                actual,
            });
        }
        entry.reload.on_event(ReloadEvent::CompileFailed);
        Ok(())
    }

    /// Atomically publishes a successfully compiled asset.
    pub fn commit_reload(
        &mut self,
        ticket: EmberReloadTicket,
        asset: EmberAsset,
    ) -> Result<EmberAssetRevision, EmberAssetRegistryError> {
        {
            let entry = self.entry_for_ticket(ticket)?;
            let actual = entry.reload.state;
            if actual != ReloadState::Swapping {
                return Err(EmberAssetRegistryError::InvalidReloadState {
                    handle: ticket.handle,
                    expected: ReloadState::Swapping,
                    actual,
                });
            }
        }
        let revision = self.next_revision();
        let entry = self.entry_for_ticket(ticket)?;
        entry.asset = asset;
        entry.revision = revision;
        entry.reload.on_event(ReloadEvent::SwapCompleted);
        Ok(revision)
    }

    fn entry_for_ticket(
        &mut self,
        ticket: EmberReloadTicket,
    ) -> Result<&mut EmberAssetEntry, EmberAssetRegistryError> {
        let entry = self
            .assets
            .get_mut(&ticket.handle)
            .ok_or(EmberAssetRegistryError::UnknownHandle(ticket.handle))?;
        if entry.revision != ticket.revision {
            return Err(EmberAssetRegistryError::StaleReload {
                handle: ticket.handle,
                expected: ticket.revision,
                actual: entry.revision,
            });
        }
        Ok(entry)
    }

    fn next_revision(&mut self) -> EmberAssetRevision {
        self.next_revision = self.next_revision.wrapping_add(1);
        EmberAssetRevision(self.next_revision)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::particle::attributes::AttributeLayoutPlan;
    use crate::particle::authoring::{
        EmberAsset, GraphAssetRef, RendererConfig, RendererConfigId, RendererKind,
        SerializationFormat, ShadingSettings,
    };
    use crate::particle::compiler::CompiledGraph;
    use crate::particle::EmberShadingModel;
    use alloc::vec::Vec;

    fn asset(graph: u32) -> EmberAsset {
        EmberAsset {
            format_version: 1,
            graph: GraphAssetRef(graph),
            params: Vec::new(),
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

    fn compiled_graph() -> CompiledGraph {
        CompiledGraph {
            source: String::new(),
            stage_order: Vec::new(),
            entry_points: Vec::new(),
            dispatches: Vec::new(),
            bindings: Vec::new(),
            parameter_blocks: Vec::new(),
            attribute_layout: AttributeLayoutPlan::build(0, &[]),
            attribute_bindings: Vec::new(),
        }
    }

    #[test]
    fn explicit_and_allocated_handles_are_deterministic() {
        let mut registry = EmberAssetRegistry::new();
        let explicit = registry.insert(EmberAssetHandle(7), asset(7)).unwrap();
        assert_eq!(explicit.value(), 1);
        let (allocated, revision) = registry.allocate(asset(8)).unwrap();
        assert_eq!(allocated, EmberAssetHandle(8));
        assert_eq!(revision.value(), 2);
        assert_eq!(
            registry
                .iter()
                .map(|(handle, _)| handle.0)
                .collect::<Vec<_>>(),
            [7, 8]
        );
    }

    #[test]
    fn binary_insert_and_duplicate_handle_are_checked() {
        let bytes = super::super::authoring::encode_binary(&asset(11)).unwrap();
        let mut registry = EmberAssetRegistry::new();
        registry.insert_binary(EmberAssetHandle(3), &bytes).unwrap();
        assert_eq!(
            registry.get(EmberAssetHandle(3)).unwrap().asset().graph,
            GraphAssetRef(11)
        );
        assert_eq!(
            registry.insert_binary(EmberAssetHandle(3), &bytes),
            Err(EmberAssetRegistryError::DuplicateHandle(EmberAssetHandle(
                3
            )))
        );
    }

    #[test]
    fn stale_reload_cannot_replace_a_newer_asset() {
        let mut registry = EmberAssetRegistry::new();
        let handle = EmberAssetHandle(1);
        registry.insert(handle, asset(1)).unwrap();
        let ticket = registry.begin_reload(handle).unwrap();
        registry.mark_compile_succeeded(ticket).unwrap();
        let revision = registry.commit_reload(ticket, asset(2)).unwrap();
        assert_eq!(revision.value(), 2);
        assert_eq!(
            registry.get(handle).unwrap().asset().graph,
            GraphAssetRef(2)
        );
        assert_eq!(
            registry.mark_compile_succeeded(ticket),
            Err(EmberAssetRegistryError::StaleReload {
                handle,
                expected: ticket.revision(),
                actual: revision,
            })
        );
    }

    #[test]
    fn failed_reload_keeps_live_asset_and_can_retry() {
        let mut registry = EmberAssetRegistry::new();
        let handle = EmberAssetHandle(4);
        registry.insert(handle, asset(4)).unwrap();
        let ticket = registry.begin_reload(handle).unwrap();
        registry.mark_compile_failed(ticket).unwrap();
        assert_eq!(
            registry.get(handle).unwrap().asset().graph,
            GraphAssetRef(4)
        );
        let retry = registry.begin_reload(handle).unwrap();
        registry.mark_compile_succeeded(retry).unwrap();
        registry.commit_reload(retry, asset(5)).unwrap();
        assert_eq!(
            registry.get(handle).unwrap().asset().graph,
            GraphAssetRef(5)
        );
        assert_eq!(
            registry.get(handle).unwrap().reload_state(),
            ReloadState::Idle
        );
    }

    #[test]
    fn compiled_graph_registry_publishes_and_replaces_atomically() {
        let mut registry = EmberCompiledGraphRegistry::new();
        let graph = GraphAssetRef(9);
        let first = registry.publish(graph, compiled_graph()).unwrap();
        assert_eq!(first.value(), 1);
        assert_eq!(registry.get(graph).unwrap().revision(), first);

        let second = EmberAssetRevision::from_value(4);
        registry
            .replace_if_newer(graph, second, compiled_graph())
            .unwrap();
        assert_eq!(registry.get(graph).unwrap().revision(), second);
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn compiled_graph_registry_rejects_stale_results_and_keeps_previous_graph() {
        let mut registry = EmberCompiledGraphRegistry::new();
        let graph = GraphAssetRef(10);
        let current = EmberAssetRevision::from_value(8);
        registry.insert(graph, current, compiled_graph()).unwrap();
        let error = registry
            .replace_if_newer(graph, EmberAssetRevision::from_value(7), compiled_graph())
            .expect_err("older asynchronous compile must be rejected");
        assert_eq!(
            error,
            EmberCompiledGraphRegistryError::StaleGraph {
                graph,
                expected: current,
                actual: EmberAssetRevision::from_value(7),
            }
        );
        assert_eq!(registry.get(graph).unwrap().revision(), current);
    }

    #[test]
    fn compiled_graph_publications_survive_replacement_and_removal_without_copying() {
        let mut registry = EmberCompiledGraphRegistry::new();
        let graph = GraphAssetRef(19);
        let mut initial = compiled_graph();
        initial.source = String::from("first publication");
        registry.publish(graph, initial).unwrap();
        let first = Arc::clone(registry.get(graph).unwrap().shared_graph());
        assert_eq!(Arc::strong_count(&first), 2);

        let mut replacement = compiled_graph();
        replacement.source = String::from("replacement publication");
        registry.publish(graph, replacement).unwrap();
        let current = Arc::clone(registry.get(graph).unwrap().shared_graph());
        assert!(!Arc::ptr_eq(&first, &current));
        assert_eq!(first.source, "first publication");
        assert_eq!(Arc::strong_count(&first), 1);

        let removed = registry.remove(graph).unwrap();
        assert!(Arc::ptr_eq(&current, &removed));
        assert_eq!(Arc::strong_count(&current), 2);
        assert!(registry.get(graph).is_none());
        drop(registry);
        assert_eq!(current.source, "replacement publication");
        assert_eq!(first.source, "first publication");
    }

    #[test]
    fn compiled_graph_registry_remove_reports_unknown_graph() {
        let mut registry = EmberCompiledGraphRegistry::new();
        assert_eq!(
            registry.remove(GraphAssetRef(404)),
            Err(EmberCompiledGraphRegistryError::UnknownGraph(
                GraphAssetRef(404)
            ))
        );
        registry
            .insert(
                GraphAssetRef(1),
                EmberAssetRevision::from_value(1),
                compiled_graph(),
            )
            .unwrap();
        assert!(registry.remove(GraphAssetRef(1)).is_ok());
        assert!(registry.get(GraphAssetRef(1)).is_none());
    }
}
