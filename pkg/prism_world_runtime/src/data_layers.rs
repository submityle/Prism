//! Engine-neutral runtime Data Layers admission and residency.
//!
//! A cell has one payload residency owned by [`crate::SceneHost`]. Data Layers
//! are a second, explicit admission state on top of that payload: an unloaded
//! cell has no host state, a loaded cell has a payload but no active layer, and
//! an activated cell has the deterministic intersection of the runtime-enabled
//! layers and the layers declared by its manifest.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use core::fmt;
use core::task::{Context, Poll};

use prism_world_stream::{CellManifest, DemandSource, SectorId, StreamBudget};

use crate::{CellFrameCoordinator, CellIo, FrameReport, LoadedCell, SceneHost};

/// Stable identity for one runtime Data Layer.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct DataLayerId(pub u32);

impl DataLayerId {
    /// Creates a layer identity.
    #[must_use]
    pub const fn new(value: u32) -> Self {
        Self(value)
    }
}

/// Runtime admission state for one Data Layer.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum DataLayerState {
    /// The layer is not resident and cells declaring it cannot stream.
    Unloaded,
    /// The layer is resident but not active in the scene host.
    Loaded,
    /// The layer is resident and active in the scene host.
    Activated,
}

/// Failure while validating an ordered layer set.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DataLayerConfigError {
    /// The same layer was supplied more than once.
    DuplicateLayer {
        /// Repeated layer identity.
        layer: DataLayerId,
    },
    /// Enabling the requested set would exceed the configured admission cap.
    CapacityExceeded {
        /// Maximum number of enabled layers.
        capacity: usize,
        /// Number of layers requested by the rejected operation.
        requested: usize,
    },
    /// An activated layer was not declared by the cell manifest.
    LayerNotDeclared {
        /// Layer that is not declared by the cell.
        layer: DataLayerId,
    },
}

impl fmt::Display for DataLayerConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateLayer { layer } => write!(formatter, "duplicate Data Layer {layer:?}"),
            Self::CapacityExceeded {
                capacity,
                requested,
            } => write!(
                formatter,
                "Data Layer admission capacity {capacity} exceeded by {requested} layers"
            ),
            Self::LayerNotDeclared { layer } => {
                write!(
                    formatter,
                    "Data Layer {layer:?} is not declared by the cell"
                )
            }
        }
    }
}

impl core::error::Error for DataLayerConfigError {}

fn canonical_layers(layers: &[DataLayerId]) -> Result<Vec<DataLayerId>, DataLayerConfigError> {
    let mut canonical = layers.to_vec();
    canonical.sort_unstable();
    for pair in canonical.windows(2) {
        if pair[0] == pair[1] {
            return Err(DataLayerConfigError::DuplicateLayer { layer: pair[0] });
        }
    }
    Ok(canonical)
}

/// Runtime-owned Data Layer states with an atomic activated-layer cap.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RuntimeDataLayers {
    max_enabled: usize,
    states: BTreeMap<DataLayerId, DataLayerState>,
    enabled: Vec<DataLayerId>,
}

impl RuntimeDataLayers {
    /// Creates an empty layer admission set.
    #[must_use]
    pub const fn new(max_enabled: usize) -> Self {
        Self {
            max_enabled,
            states: BTreeMap::new(),
            enabled: Vec::new(),
        }
    }

    /// Returns the configured maximum number of enabled layers.
    #[must_use]
    pub const fn max_enabled(&self) -> usize {
        self.max_enabled
    }

    /// Returns enabled layers in canonical order.
    #[must_use]
    pub fn enabled(&self) -> &[DataLayerId] {
        &self.enabled
    }

    /// Returns the state of a layer; undeclared/unknown layers are unloaded.
    #[must_use]
    pub fn state(&self, layer: DataLayerId) -> DataLayerState {
        self.states
            .get(&layer)
            .copied()
            .unwrap_or(DataLayerState::Unloaded)
    }

    /// Returns all non-unloaded layer states in deterministic order.
    pub fn states(&self) -> alloc::collections::btree_map::Iter<'_, DataLayerId, DataLayerState> {
        self.states.iter()
    }

    /// Replaces explicit layer states atomically.
    pub fn replace_states(
        &mut self,
        states: &[(DataLayerId, DataLayerState)],
    ) -> Result<(), DataLayerConfigError> {
        let mut canonical = states.to_vec();
        canonical.sort_unstable_by_key(|(layer, _)| *layer);
        for pair in canonical.windows(2) {
            if pair[0].0 == pair[1].0 {
                return Err(DataLayerConfigError::DuplicateLayer { layer: pair[0].0 });
            }
        }
        let activated = canonical
            .iter()
            .filter(|(_, state)| *state == DataLayerState::Activated)
            .count();
        if activated > self.max_enabled {
            return Err(DataLayerConfigError::CapacityExceeded {
                capacity: self.max_enabled,
                requested: activated,
            });
        }
        let mut next_states = BTreeMap::new();
        let mut next_enabled = Vec::new();
        for (layer, state) in canonical {
            if state != DataLayerState::Unloaded {
                next_states.insert(layer, state);
            }
            if state == DataLayerState::Activated {
                next_enabled.push(layer);
            }
        }
        self.states = next_states;
        self.enabled = next_enabled;
        Ok(())
    }

    /// Sets one layer state while preserving the old set on capacity failure.
    pub fn set_state(
        &mut self,
        layer: DataLayerId,
        state: DataLayerState,
    ) -> Result<(), DataLayerConfigError> {
        if state == DataLayerState::Activated && self.state(layer) != DataLayerState::Activated {
            let activated = self
                .states
                .values()
                .filter(|current| **current == DataLayerState::Activated)
                .count();
            if activated == self.max_enabled {
                return Err(DataLayerConfigError::CapacityExceeded {
                    capacity: self.max_enabled,
                    requested: activated + 1,
                });
            }
        }
        let mut next_states = self.states.clone();
        if state == DataLayerState::Unloaded {
            next_states.remove(&layer);
        } else {
            next_states.insert(layer, state);
        }
        let next_enabled = next_states
            .iter()
            .filter_map(|(layer, state)| (*state == DataLayerState::Activated).then_some(*layer))
            .collect();
        self.states = next_states;
        self.enabled = next_enabled;
        Ok(())
    }

    /// Replaces the enabled set atomically after validation and sorting.
    pub fn replace_enabled(&mut self, layers: &[DataLayerId]) -> Result<(), DataLayerConfigError> {
        let canonical = canonical_layers(layers)?;
        self.replace_states(
            &canonical
                .into_iter()
                .map(|layer| (layer, DataLayerState::Activated))
                .collect::<Vec<_>>(),
        )
    }

    /// Enables one layer, preserving the old set if admission fails.
    pub fn enable(&mut self, layer: DataLayerId) -> Result<(), DataLayerConfigError> {
        self.set_state(layer, DataLayerState::Activated)
    }

    /// Disables one layer. Disabling an absent layer is a no-op.
    pub fn disable(&mut self, layer: DataLayerId) {
        let _ = self.set_state(layer, DataLayerState::Unloaded);
    }
}

/// Cell manifest plus the Data Layers it can expose.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LayeredCellManifest {
    /// Base streaming manifest.
    pub manifest: CellManifest,
    layers: Vec<DataLayerId>,
}

impl LayeredCellManifest {
    /// Creates a manifest with a unique, canonical layer membership list.
    pub fn new(
        manifest: CellManifest,
        layers: &[DataLayerId],
    ) -> Result<Self, DataLayerConfigError> {
        Ok(Self {
            manifest,
            layers: canonical_layers(layers)?,
        })
    }

    /// Returns the cell's declared layers in canonical order.
    #[must_use]
    pub fn layers(&self) -> &[DataLayerId] {
        &self.layers
    }

    fn stream_admission(&self, runtime: &RuntimeDataLayers) -> (bool, Vec<DataLayerId>) {
        if self
            .layers
            .iter()
            .any(|layer| runtime.state(*layer) == DataLayerState::Unloaded)
        {
            return (false, Vec::new());
        }
        let active = self
            .layers
            .iter()
            .copied()
            .filter(|layer| runtime.state(*layer) == DataLayerState::Activated)
            .collect();
        (true, active)
    }
}

/// Initial host snapshot for a layered coordinator.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LayeredLoadedCell {
    /// Loaded cell identity and revision.
    pub manifest: CellManifest,
    active_layers: Vec<DataLayerId>,
}

impl LayeredLoadedCell {
    /// Creates a loaded snapshot and validates its active layer set.
    pub fn new(
        manifest: CellManifest,
        declared_layers: &[DataLayerId],
        active_layers: &[DataLayerId],
    ) -> Result<Self, DataLayerConfigError> {
        let declared = canonical_layers(declared_layers)?;
        let active = canonical_layers(active_layers)?;
        for layer in &active {
            if declared.binary_search(layer).is_err() {
                return Err(DataLayerConfigError::LayerNotDeclared { layer: *layer });
            }
        }
        Ok(Self {
            manifest,
            active_layers: active,
        })
    }

    /// Returns the layers currently active in the host snapshot.
    #[must_use]
    pub fn active_layers(&self) -> &[DataLayerId] {
        &self.active_layers
    }
}

/// Observable residency state for one cell.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LayeredCellState {
    /// No payload is resident in the scene host.
    Unloaded,
    /// The payload is resident, but no Data Layer is active.
    Loaded,
    /// The payload is resident and these layers are active.
    Activated(Vec<DataLayerId>),
}

/// Host boundary for atomic Data Layer replacement.
pub trait LayeredSceneHost: SceneHost {
    /// Replaces the complete active layer set for one loaded cell.
    ///
    /// The host must treat the operation as atomic: on `Err`, the previous
    /// active set remains authoritative. The coordinator only updates its
    /// state after `Ok(())`.
    fn set_active_layers(
        &mut self,
        sector: SectorId,
        layers: &[DataLayerId],
    ) -> Result<(), Self::Error>;
}

/// Error from layered manifest admission before streaming state is changed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LayeredCoordinatorError {
    /// Two layered manifests used the same sector identity.
    DuplicateManifestSector {
        /// Repeated manifest sector.
        sector: SectorId,
    },
    /// Two initial loaded snapshots used the same sector identity.
    DuplicateLoadedSector {
        /// Repeated loaded sector.
        sector: SectorId,
    },
}

impl fmt::Display for LayeredCoordinatorError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateManifestSector { sector } => {
                write!(formatter, "duplicate layered manifest sector {sector:?}")
            }
            Self::DuplicateLoadedSector { sector } => {
                write!(formatter, "duplicate layered loaded sector {sector:?}")
            }
        }
    }
}

impl core::error::Error for LayeredCoordinatorError {}

/// Outcome of an eviction attempted before cell streaming admission.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LayeredEvictionOutcome {
    /// The host evicted the cell and it was removed from the scheduler input.
    Evicted,
    /// The host rejected eviction; the cell remains resident and is not
    /// admitted to the scheduler for this frame.
    Failed,
}

/// Structured report for one pre-admission eviction.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LayeredEvictionReport {
    /// Cell requested for eviction.
    pub sector: SectorId,
    /// Host outcome.
    pub outcome: LayeredEvictionOutcome,
}

/// Result of one layered frame admission pass.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LayeredFrameReport {
    /// Underlying cell streaming report.
    pub streaming: FrameReport,
    /// Number of host layer-set calls attempted.
    pub layer_attempted: usize,
    /// Number of layer-set calls that committed.
    pub layer_committed: usize,
    /// Number of layer-set calls rejected by the host.
    pub layer_failed: usize,
    /// Evictions attempted before the cell scheduler was planned.
    pub evictions: Vec<LayeredEvictionReport>,
}

/// Result of polling one layered IO completion.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LayeredPollReport {
    /// Whether the underlying coordinator retired an IO completion.
    pub io_progressed: bool,
    /// Number of layer-set calls attempted during reconciliation.
    pub layer_attempted: usize,
    /// Number of layer-set calls that committed.
    pub layer_committed: usize,
    /// Number of layer-set calls rejected by the host.
    pub layer_failed: usize,
}

struct ResidentLayeredCell {
    active_layers: Vec<DataLayerId>,
}

struct LayeredHostAdapter<'a, H: LayeredSceneHost> {
    host: &'a mut H,
    resident: &'a mut BTreeMap<SectorId, ResidentLayeredCell>,
}

impl<H: LayeredSceneHost> SceneHost for LayeredHostAdapter<'_, H> {
    type Error = H::Error;

    fn load_cell(&mut self, manifest: &CellManifest, payload: &[u8]) -> Result<(), Self::Error> {
        self.host.load_cell(manifest, payload)?;
        let active_layers = self
            .resident
            .get(&manifest.sector)
            .map(|cell| cell.active_layers.clone())
            .unwrap_or_default();
        self.resident
            .insert(manifest.sector, ResidentLayeredCell { active_layers });
        Ok(())
    }

    fn evict_cell(&mut self, sector: SectorId) -> Result<(), Self::Error> {
        self.host.evict_cell(sector)?;
        self.resident.remove(&sector);
        Ok(())
    }
}

impl<H: LayeredSceneHost> LayeredSceneHost for LayeredHostAdapter<'_, H> {
    fn set_active_layers(
        &mut self,
        sector: SectorId,
        layers: &[DataLayerId],
    ) -> Result<(), Self::Error> {
        self.host.set_active_layers(sector, layers)?;
        if let Some(cell) = self.resident.get_mut(&sector) {
            cell.active_layers = layers.to_vec();
        }
        Ok(())
    }
}

/// Coordinates cell streaming and deterministic Data Layer admission.
pub struct LayeredCellFrameCoordinator<I: CellIo> {
    cells: CellFrameCoordinator<I>,
    data_layers: RuntimeDataLayers,
    resident: BTreeMap<SectorId, ResidentLayeredCell>,
    desired: BTreeMap<SectorId, Vec<DataLayerId>>,
}

impl<I: CellIo> LayeredCellFrameCoordinator<I> {
    /// Creates a layered coordinator around an existing cell IO backend.
    #[must_use]
    pub fn with_io(io: I, data_layers: RuntimeDataLayers) -> Self {
        Self {
            cells: CellFrameCoordinator::with_io(io),
            data_layers,
            resident: BTreeMap::new(),
            desired: BTreeMap::new(),
        }
    }

    /// Returns the mutable runtime layer admission policy.
    pub fn data_layers_mut(&mut self) -> &mut RuntimeDataLayers {
        &mut self.data_layers
    }

    /// Returns the current runtime layer admission policy.
    #[must_use]
    pub const fn data_layers(&self) -> &RuntimeDataLayers {
        &self.data_layers
    }

    /// Returns the current state of a cell, including unloaded cells.
    #[must_use]
    pub fn state(&self, sector: SectorId) -> LayeredCellState {
        self.resident
            .get(&sector)
            .map_or(LayeredCellState::Unloaded, |cell| {
                if cell.active_layers.is_empty() {
                    LayeredCellState::Loaded
                } else {
                    LayeredCellState::Activated(cell.active_layers.clone())
                }
            })
    }

    /// Plans streaming, applies resident-layer changes, and preserves state on
    /// all validation and host errors.
    pub fn plan_frame<H: LayeredSceneHost>(
        &mut self,
        demands: &[DemandSource],
        manifests: &[LayeredCellManifest],
        loaded: &[LayeredLoadedCell],
        budget: StreamBudget,
        host: &mut H,
    ) -> Result<LayeredFrameReport, LayeredCoordinatorError> {
        validate_unique_manifests(manifests)?;
        validate_unique_loaded(loaded)?;

        self.desired = manifests
            .iter()
            .filter_map(|cell| {
                let (streamable, active) = cell.stream_admission(&self.data_layers);
                streamable.then_some((cell.manifest.sector, active))
            })
            .collect();

        let mut evictions = Vec::new();
        let mut admitted_loaded = Vec::new();
        self.resident.clear();
        for cell in loaded {
            let streamable = manifests
                .iter()
                .find(|manifest| manifest.manifest.sector == cell.manifest.sector)
                .is_none_or(|manifest| manifest.stream_admission(&self.data_layers).0);
            if !streamable {
                let outcome = if host.evict_cell(cell.manifest.sector).is_ok() {
                    self.resident.remove(&cell.manifest.sector);
                    LayeredEvictionOutcome::Evicted
                } else {
                    self.resident.insert(
                        cell.manifest.sector,
                        ResidentLayeredCell {
                            active_layers: cell.active_layers.clone(),
                        },
                    );
                    LayeredEvictionOutcome::Failed
                };
                evictions.push(LayeredEvictionReport {
                    sector: cell.manifest.sector,
                    outcome,
                });
                continue;
            }
            self.resident.insert(
                cell.manifest.sector,
                ResidentLayeredCell {
                    active_layers: cell.active_layers.clone(),
                },
            );
            admitted_loaded.push(LoadedCell::new(cell.manifest));
        }
        let plain_manifests: Vec<_> = manifests
            .iter()
            .filter(|cell| cell.stream_admission(&self.data_layers).0)
            .map(|cell| cell.manifest)
            .collect();
        let streaming = {
            let mut adapter = LayeredHostAdapter {
                host,
                resident: &mut self.resident,
            };
            self.cells.plan_frame(
                demands,
                &plain_manifests,
                &admitted_loaded,
                budget,
                &mut adapter,
            )
        };
        let (layer_attempted, layer_committed, layer_failed) = {
            let mut adapter = LayeredHostAdapter {
                host,
                resident: &mut self.resident,
            };
            reconcile(&self.desired, &mut adapter)
        };
        Ok(LayeredFrameReport {
            streaming,
            layer_attempted,
            layer_committed,
            layer_failed,
            evictions,
        })
    }

    /// Polls one completion using the caller's task context.
    pub fn poll_io_with_context<H: LayeredSceneHost>(
        &mut self,
        cx: &mut Context<'_>,
        host: &mut H,
    ) -> Poll<Result<LayeredPollReport, I::Error>>
    where
        I::Error: fmt::Debug,
    {
        let progressed = {
            let mut adapter = LayeredHostAdapter {
                host,
                resident: &mut self.resident,
            };
            match self.cells.poll_io_with_context(cx, &mut adapter) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(Err(error)) => return Poll::Ready(Err(error)),
                Poll::Ready(Ok(())) => true,
            }
        };
        let (layer_attempted, layer_committed, layer_failed) = {
            let mut adapter = LayeredHostAdapter {
                host,
                resident: &mut self.resident,
            };
            reconcile(&self.desired, &mut adapter)
        };
        Poll::Ready(Ok(LayeredPollReport {
            io_progressed: progressed,
            layer_attempted,
            layer_committed,
            layer_failed,
        }))
    }

    /// Returns the underlying deterministic cell trace.
    #[must_use]
    pub fn trace(&self) -> &crate::Trace {
        self.cells.trace()
    }
}

impl<P: crate::ChunkProvider> LayeredCellFrameCoordinator<crate::IoRuntime<P>> {
    /// Creates a layered coordinator over the synchronous provider runtime.
    pub fn new(
        provider: P,
        io_capacity: usize,
        data_layers: RuntimeDataLayers,
    ) -> Result<Self, crate::SubmitError> {
        Ok(Self::with_io(
            crate::IoRuntime::new(provider, io_capacity)?,
            data_layers,
        ))
    }

    /// Polls one synchronous provider completion and reconciles layers.
    pub fn poll_io<H: LayeredSceneHost>(&mut self, host: &mut H) -> LayeredPollReport
    where
        P::Error: fmt::Debug,
    {
        let io_progressed = {
            let mut adapter = LayeredHostAdapter {
                host,
                resident: &mut self.resident,
            };
            self.cells.poll_io(&mut adapter)
        };
        let (layer_attempted, layer_committed, layer_failed) = {
            let mut adapter = LayeredHostAdapter {
                host,
                resident: &mut self.resident,
            };
            reconcile(&self.desired, &mut adapter)
        };
        LayeredPollReport {
            io_progressed,
            layer_attempted,
            layer_committed,
            layer_failed,
        }
    }
}

fn reconcile<H: LayeredSceneHost>(
    desired: &BTreeMap<SectorId, Vec<DataLayerId>>,
    adapter: &mut LayeredHostAdapter<'_, H>,
) -> (usize, usize, usize) {
    let sectors: Vec<_> = adapter.resident.keys().copied().collect();
    let mut attempted = 0;
    let mut committed = 0;
    let mut failed = 0;
    for sector in sectors {
        let Some(wanted) = desired.get(&sector).map(Vec::as_slice) else {
            continue;
        };
        let current = adapter
            .resident
            .get(&sector)
            .map_or(&[][..], |cell| cell.active_layers.as_slice());
        if current == wanted {
            continue;
        }
        attempted += 1;
        if adapter.set_active_layers(sector, wanted).is_ok() {
            committed += 1;
        } else {
            failed += 1;
        }
    }
    (attempted, committed, failed)
}

fn validate_unique_manifests(
    manifests: &[LayeredCellManifest],
) -> Result<(), LayeredCoordinatorError> {
    let mut sectors = Vec::with_capacity(manifests.len());
    for manifest in manifests {
        sectors.push(manifest.manifest.sector);
    }
    sectors.sort_unstable();
    for pair in sectors.windows(2) {
        if pair[0] == pair[1] {
            return Err(LayeredCoordinatorError::DuplicateManifestSector { sector: pair[0] });
        }
    }
    Ok(())
}

fn validate_unique_loaded(loaded: &[LayeredLoadedCell]) -> Result<(), LayeredCoordinatorError> {
    let mut sectors: Vec<_> = loaded.iter().map(|cell| cell.manifest.sector).collect();
    sectors.sort_unstable();
    for pair in sectors.windows(2) {
        if pair[0] == pair[1] {
            return Err(LayeredCoordinatorError::DuplicateLoadedSector { sector: pair[0] });
        }
    }
    Ok(())
}

impl From<u32> for DataLayerId {
    fn from(value: u32) -> Self {
        Self::new(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use core::sync::atomic::{AtomicUsize, Ordering};

    use prism_world_stream::{AabbI64, GlobalPos};

    use crate::{CancellationToken, CellProvider, ManifestHeader};

    fn manifest(x: i64, version: u64) -> CellManifest {
        CellManifest {
            sector: SectorId::new(0, x, 0, 0),
            bounds: AabbI64::new(GlobalPos::new(x, 0, 0), GlobalPos::new(x + 1, 1, 1))
                .expect("valid bounds"),
            scene_key: 7,
            content_version: version,
            derived_hash: version as u128 + 10,
            io_bytes: 1,
        }
    }

    fn bytes(manifest: CellManifest, payload: &[u8]) -> Vec<u8> {
        let mut result = ManifestHeader::from_manifest(manifest).encode();
        result.extend_from_slice(payload);
        result
    }

    struct Provider {
        bytes: BTreeMap<(SectorId, u64), Vec<u8>>,
    }

    impl CellProvider for Provider {
        type Error = ();

        fn read_cell(
            &self,
            request: &crate::CellReadRequest,
            cancel: CancellationToken,
        ) -> Result<Vec<u8>, Self::Error> {
            if cancel.is_cancelled() {
                return Err(());
            }
            self.bytes
                .get(&(request.sector(), request.manifest().content_version))
                .cloned()
                .ok_or(())
        }
    }

    #[derive(Default)]
    struct Host {
        loaded: Vec<SectorId>,
        evicted: Vec<SectorId>,
        layer_calls: Vec<(SectorId, Vec<DataLayerId>)>,
        fail_layers: bool,
        fail_evictions: bool,
        load_calls: AtomicUsize,
    }

    impl SceneHost for Host {
        type Error = ();

        fn load_cell(
            &mut self,
            manifest: &CellManifest,
            _payload: &[u8],
        ) -> Result<(), Self::Error> {
            self.load_calls.fetch_add(1, Ordering::Relaxed);
            self.loaded.push(manifest.sector);
            Ok(())
        }

        fn evict_cell(&mut self, sector: SectorId) -> Result<(), Self::Error> {
            if self.fail_evictions {
                return Err(());
            }
            self.evicted.push(sector);
            Ok(())
        }
    }

    impl LayeredSceneHost for Host {
        fn set_active_layers(
            &mut self,
            sector: SectorId,
            layers: &[DataLayerId],
        ) -> Result<(), Self::Error> {
            if self.fail_layers {
                return Err(());
            }
            self.layer_calls.push((sector, layers.to_vec()));
            Ok(())
        }
    }

    fn coordinator(
        cells: &[CellManifest],
        max_layers: usize,
    ) -> LayeredCellFrameCoordinator<crate::IoRuntime<Provider>> {
        let bytes = cells
            .iter()
            .map(|manifest| {
                (
                    (manifest.sector, manifest.content_version),
                    bytes(*manifest, b"payload"),
                )
            })
            .collect();
        LayeredCellFrameCoordinator::with_io(
            crate::IoRuntime::new(Provider { bytes }, 4).expect("runtime config"),
            RuntimeDataLayers::new(max_layers),
        )
    }

    #[test]
    fn unloaded_loaded_and_activated_states_are_distinct() {
        let cell = manifest(0, 1);
        let layered = LayeredCellManifest::new(cell, &[DataLayerId::new(2)]).unwrap();
        let mut coordinator = coordinator(&[cell], 2);
        let mut host = Host::default();
        let demand = DemandSource::new(cell.sector, 0, 1, 10).unwrap();
        coordinator
            .data_layers_mut()
            .set_state(DataLayerId::new(2), DataLayerState::Loaded)
            .unwrap();

        assert_eq!(coordinator.state(cell.sector), LayeredCellState::Unloaded);
        coordinator
            .plan_frame(
                &[demand],
                core::slice::from_ref(&layered),
                &[],
                StreamBudget {
                    io_bytes: 1,
                    max_actions: 1,
                },
                &mut host,
            )
            .unwrap();
        assert!(matches!(
            coordinator.state(cell.sector),
            LayeredCellState::Unloaded
        ));
        assert!(coordinator.poll_io(&mut host).io_progressed);
        assert_eq!(coordinator.state(cell.sector), LayeredCellState::Loaded);

        coordinator
            .data_layers_mut()
            .enable(DataLayerId::new(2))
            .unwrap();
        let report = coordinator
            .plan_frame(
                &[demand],
                &[layered],
                &[LayeredLoadedCell::new(cell, &[DataLayerId::new(2)], &[]).unwrap()],
                StreamBudget {
                    io_bytes: 1,
                    max_actions: 1,
                },
                &mut host,
            )
            .unwrap();
        assert_eq!(report.layer_committed, 1);
        assert_eq!(
            coordinator.state(cell.sector),
            LayeredCellState::Activated(vec![DataLayerId::new(2)])
        );
    }

    #[test]
    fn multiple_layers_are_canonical_and_deterministic() {
        let cell = manifest(1, 1);
        let layered = LayeredCellManifest::new(
            cell,
            &[
                DataLayerId::new(9),
                DataLayerId::new(1),
                DataLayerId::new(4),
            ],
        )
        .unwrap();
        let mut coordinator = coordinator(&[cell], 3);
        coordinator
            .data_layers_mut()
            .replace_states(&[
                (DataLayerId::new(1), DataLayerState::Loaded),
                (DataLayerId::new(4), DataLayerState::Activated),
                (DataLayerId::new(9), DataLayerState::Activated),
            ])
            .unwrap();
        let mut host = Host::default();
        let demand = DemandSource::new(cell.sector, 0, 1, 1).unwrap();
        coordinator
            .plan_frame(
                &[demand],
                &[layered],
                &[LayeredLoadedCell::new(cell, &[1.into(), 4.into(), 9.into()], &[]).unwrap()],
                StreamBudget {
                    io_bytes: 1,
                    max_actions: 1,
                },
                &mut host,
            )
            .unwrap();
        assert_eq!(
            host.layer_calls,
            vec![(cell.sector, vec![DataLayerId::new(4), DataLayerId::new(9)])]
        );
    }

    #[test]
    fn any_unloaded_declared_layer_blocks_streaming_until_all_are_loaded() {
        let cell = manifest(4, 1);
        let layered = LayeredCellManifest::new(
            cell,
            &[DataLayerId::new(1), DataLayerId::new(2)],
        )
        .unwrap();
        let mut coordinator = coordinator(&[cell], 2);
        coordinator
            .data_layers_mut()
            .set_state(DataLayerId::new(1), DataLayerState::Activated)
            .unwrap();
        let mut host = Host::default();
        let demand = DemandSource::new(cell.sector, 0, 1, 1).unwrap();
        let blocked = coordinator
            .plan_frame(
                &[demand],
                core::slice::from_ref(&layered),
                &[],
                StreamBudget {
                    io_bytes: 1,
                    max_actions: 1,
                },
                &mut host,
            )
            .unwrap();
        assert_eq!(blocked.streaming.submitted, 0);
        assert_eq!(host.load_calls.load(Ordering::Relaxed), 0);

        coordinator
            .data_layers_mut()
            .set_state(DataLayerId::new(2), DataLayerState::Loaded)
            .unwrap();
        let admitted = coordinator
            .plan_frame(
                &[demand],
                &[layered],
                &[],
                StreamBudget {
                    io_bytes: 1,
                    max_actions: 1,
                },
                &mut host,
            )
            .unwrap();
        assert_eq!(admitted.streaming.submitted, 1);
        coordinator.poll_io(&mut host);
        assert_eq!(
            coordinator.state(cell.sector),
            LayeredCellState::Activated(vec![DataLayerId::new(1)])
        );
    }

    #[test]
    fn loaded_layers_stream_without_activation_and_activation_is_selective() {
        let cell = manifest(5, 1);
        let layered = LayeredCellManifest::new(
            cell,
            &[DataLayerId::new(1), DataLayerId::new(2)],
        )
        .unwrap();
        let mut coordinator = coordinator(&[cell], 2);
        coordinator
            .data_layers_mut()
            .replace_states(&[
                (DataLayerId::new(1), DataLayerState::Loaded),
                (DataLayerId::new(2), DataLayerState::Loaded),
            ])
            .unwrap();
        let mut host = Host::default();
        let demand = DemandSource::new(cell.sector, 0, 1, 1).unwrap();
        let report = coordinator
            .plan_frame(
                &[demand],
                core::slice::from_ref(&layered),
                &[],
                StreamBudget {
                    io_bytes: 1,
                    max_actions: 1,
                },
                &mut host,
            )
            .unwrap();
        assert_eq!(report.streaming.submitted, 1);
        coordinator.poll_io(&mut host);
        assert_eq!(coordinator.state(cell.sector), LayeredCellState::Loaded);

        coordinator
            .data_layers_mut()
            .set_state(DataLayerId::new(2), DataLayerState::Activated)
            .unwrap();
        let report = coordinator
            .plan_frame(
                &[demand],
                &[layered],
                &[LayeredLoadedCell::new(cell, &[1.into(), 2.into()], &[]).unwrap()],
                StreamBudget {
                    io_bytes: 1,
                    max_actions: 1,
                },
                &mut host,
            )
            .unwrap();
        assert_eq!(report.layer_committed, 1);
        assert_eq!(
            coordinator.state(cell.sector),
            LayeredCellState::Activated(vec![DataLayerId::new(2)])
        );
    }

    #[test]
    fn eviction_failure_keeps_resident_state_and_reports_atomically() {
        let cell = manifest(6, 1);
        let layered = LayeredCellManifest::new(cell, &[DataLayerId::new(8)]).unwrap();
        let mut coordinator = coordinator(&[cell], 1);
        let loaded = [LayeredLoadedCell::new(
            cell,
            &[DataLayerId::new(8)],
            &[DataLayerId::new(8)],
        )
        .unwrap()];
        let mut host = Host {
            fail_evictions: true,
            ..Host::default()
        };
        coordinator
            .plan_frame(
                &[],
                core::slice::from_ref(&layered),
                &loaded,
                StreamBudget {
                    io_bytes: 1,
                    max_actions: 1,
                },
                &mut host,
            )
            .unwrap();
        assert_eq!(
            coordinator.state(cell.sector),
            LayeredCellState::Activated(vec![DataLayerId::new(8)])
        );

        host.fail_evictions = false;
        coordinator
            .data_layers_mut()
            .set_state(DataLayerId::new(8), DataLayerState::Loaded)
            .unwrap();
        let report = coordinator
            .plan_frame(
                &[],
                &[layered],
                &loaded,
                StreamBudget {
                    io_bytes: 1,
                    max_actions: 1,
                },
                &mut host,
            )
            .unwrap();
        assert_eq!(report.evictions, Vec::new());
        assert_eq!(coordinator.state(cell.sector), LayeredCellState::Loaded);
    }

    #[test]
    fn layer_failure_is_atomic_and_retryable() {
        let cell = manifest(2, 1);
        let layered = LayeredCellManifest::new(cell, &[DataLayerId::new(3)]).unwrap();
        let mut coordinator = coordinator(&[cell], 1);
        coordinator
            .data_layers_mut()
            .enable(DataLayerId::new(3))
            .unwrap();
        let mut host = Host {
            fail_layers: true,
            ..Host::default()
        };
        let loaded = [LayeredLoadedCell::new(cell, &[DataLayerId::new(3)], &[]).unwrap()];
        let demand = DemandSource::new(cell.sector, 0, 1, 1).unwrap();
        let report = coordinator
            .plan_frame(
                &[demand],
                core::slice::from_ref(&layered),
                &loaded,
                StreamBudget {
                    io_bytes: 1,
                    max_actions: 1,
                },
                &mut host,
            )
            .unwrap();
        assert_eq!(report.layer_failed, 1);
        assert_eq!(coordinator.state(cell.sector), LayeredCellState::Loaded);

        host.fail_layers = false;
        let report = coordinator
            .plan_frame(
                &[demand],
                &[layered],
                &loaded,
                StreamBudget {
                    io_bytes: 1,
                    max_actions: 1,
                },
                &mut host,
            )
            .unwrap();
        assert_eq!(report.layer_committed, 1);
        assert!(matches!(
            coordinator.state(cell.sector),
            LayeredCellState::Activated(_)
        ));
    }

    #[test]
    fn eviction_unloads_layers_and_reload_reactivates_the_desired_set() {
        let old = manifest(3, 1);
        let new = manifest(3, 2);
        let layered_old = LayeredCellManifest::new(old, &[DataLayerId::new(5)]).unwrap();
        let layered_new = LayeredCellManifest::new(new, &[DataLayerId::new(5)]).unwrap();
        let mut coordinator = coordinator(&[old, new], 1);
        coordinator
            .data_layers_mut()
            .enable(DataLayerId::new(5))
            .unwrap();
        let mut host = Host::default();
        let demand = DemandSource::new(old.sector, 0, 0, 1).unwrap();
        coordinator
            .plan_frame(
                &[demand],
                core::slice::from_ref(&layered_old),
                &[],
                StreamBudget {
                    io_bytes: 1,
                    max_actions: 1,
                },
                &mut host,
            )
            .unwrap();
        let poll = coordinator.poll_io(&mut host);
        assert!(
            poll.io_progressed,
            "poll={poll:?}, loaded={:?}, evicted={:?}, trace={:?}",
            host.loaded,
            host.evicted,
            coordinator.trace().as_slice()
        );
        let state = coordinator.state(old.sector);
        assert_eq!(
            state,
            LayeredCellState::Activated(vec![DataLayerId::new(5)]),
            "poll={poll:?}, loaded={:?}, evicted={:?}, trace={:?}",
            host.loaded,
            host.evicted,
            coordinator.trace().as_slice()
        );

        coordinator
            .plan_frame(
                &[],
                &[layered_old],
                &[
                    LayeredLoadedCell::new(old, &[DataLayerId::new(5)], &[DataLayerId::new(5)])
                        .unwrap(),
                ],
                StreamBudget {
                    io_bytes: 1,
                    max_actions: 1,
                },
                &mut host,
            )
            .unwrap();
        assert_eq!(coordinator.state(old.sector), LayeredCellState::Unloaded);

        let demand = DemandSource::new(new.sector, 0, 1, 1).unwrap();
        coordinator
            .plan_frame(
                &[demand],
                &[layered_new],
                &[],
                StreamBudget {
                    io_bytes: 1,
                    max_actions: 1,
                },
                &mut host,
            )
            .unwrap();
        coordinator.poll_io(&mut host);
        assert_eq!(
            coordinator.state(new.sector),
            LayeredCellState::Activated(vec![DataLayerId::new(5)])
        );
    }

    #[test]
    fn invalid_admission_does_not_mutate_enabled_layers() {
        let mut layers = RuntimeDataLayers::new(1);
        layers.enable(DataLayerId::new(7)).unwrap();
        assert_eq!(
            layers.replace_enabled(&[DataLayerId::new(1), DataLayerId::new(2)]),
            Err(DataLayerConfigError::CapacityExceeded {
                capacity: 1,
                requested: 2
            })
        );
        assert_eq!(layers.enabled(), &[DataLayerId::new(7)]);
    }
}
