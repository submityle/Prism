//! Deterministic runtime registry for versioned `.ember` assets.
//!
//! The registry owns the CPU-side asset lifetime and the generation used by
//! render preparation to reject stale asynchronous reload results. It does
//! not own compiled graphs, GPU buffers, or pipeline objects; those resources
//! consume an [`EmberAsset`] through the generation returned here.

use alloc::collections::BTreeMap;

use super::authoring::{
    decode_binary, EmberAsset, EmberAssetHandle, EmberCodecError, HotReload, ReloadEvent,
    ReloadState,
};

/// Monotonic revision of one registered asset.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct EmberAssetRevision(u64);

impl EmberAssetRevision {
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
    use crate::particle::authoring::{
        EmberAsset, GraphAssetRef, RendererConfig, RendererConfigId, RendererKind,
        SerializationFormat, ShadingSettings,
    };
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
}
