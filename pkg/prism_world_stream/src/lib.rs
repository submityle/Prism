//! Deterministic spatial indexing and budgeted streaming for Prism worlds.
//!
//! This crate owns policy only: it turns demand sources and cell manifests into
//! a stable, budgeted action list. IO, decompression, GPU upload, and scene
//! mutation are providers outside the crate. The core is `no_std + alloc` and
//! contains no engine or operating-system dependency.
#![no_std]
#![forbid(unsafe_code)]
#![warn(missing_docs)]

extern crate alloc;

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;
use core::cmp::Ordering;

mod spatial_index;

pub use spatial_index::ManifestSpatialIndex;

/// Maximum number of canonical Data Layers represented by a cell manifest.
///
/// The mask is deliberately fixed-width so admission stays allocation-free
/// in the streaming hot path. Layer ids outside `0..64` are rejected at the
/// contract boundary instead of silently truncating into the mask.
pub const MAX_DATA_LAYERS: u32 = 64;

/// Stable Data Layer identity shared by world streaming and runtime adapters.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct DataLayerId(pub u32);

impl DataLayerId {
    /// Creates a Data Layer identity.
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    /// Returns the numeric identity.
    pub const fn get(self) -> u32 {
        self.0
    }
}

impl From<u32> for DataLayerId {
    fn from(value: u32) -> Self {
        Self::new(value)
    }
}

/// Error raised while constructing a fixed-width Data Layer mask.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DataLayerMaskError {
    /// A layer id cannot be represented by the canonical mask.
    OutOfRange {
        /// Rejected layer identity.
        layer: DataLayerId,
    },
    /// The same layer was supplied more than once.
    Duplicate {
        /// Repeated layer identity.
        layer: DataLayerId,
    },
}

impl core::fmt::Display for DataLayerMaskError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::OutOfRange { layer } => write!(
                formatter,
                "data layer {:?} is outside canonical range 0..{}",
                layer, MAX_DATA_LAYERS
            ),
            Self::Duplicate { layer } => write!(formatter, "duplicate data layer {layer:?}"),
        }
    }
}

impl core::error::Error for DataLayerMaskError {}

/// Allocation-free canonical membership set for Data Layers.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct DataLayerMask(u64);

impl DataLayerMask {
    /// Empty membership set.
    pub const EMPTY: Self = Self(0);

    /// Creates an empty mask.
    pub const fn empty() -> Self {
        Self::EMPTY
    }

    /// Builds a mask, rejecting duplicates and unrepresentable ids.
    pub fn from_layers(layers: &[DataLayerId]) -> Result<Self, DataLayerMaskError> {
        let mut bits = 0_u64;
        for &layer in layers {
            if layer.0 >= MAX_DATA_LAYERS {
                return Err(DataLayerMaskError::OutOfRange { layer });
            }
            let bit = 1_u64 << layer.0;
            if bits & bit != 0 {
                return Err(DataLayerMaskError::Duplicate { layer });
            }
            bits |= bit;
        }
        Ok(Self(bits))
    }

    /// Creates a mask from raw bits. The value is already canonical.
    pub const fn from_bits(bits: u64) -> Self {
        Self(bits)
    }

    /// Returns the raw canonical bits.
    pub const fn bits(self) -> u64 {
        self.0
    }

    /// Returns whether the mask contains a layer.
    pub const fn contains(self, layer: DataLayerId) -> bool {
        layer.0 < MAX_DATA_LAYERS && self.0 & (1_u64 << layer.0) != 0
    }

    /// Returns whether every layer in `self` is present in `available`.
    pub const fn is_subset_of(self, available: Self) -> bool {
        self.0 & !available.0 == 0
    }

    /// Returns the number of layers in the mask.
    pub const fn count(self) -> u32 {
        self.0.count_ones()
    }

    /// Iterates layer ids in ascending canonical order.
    pub const fn iter(self) -> DataLayerMaskIter {
        DataLayerMaskIter { remaining: self.0 }
    }
}

/// Iterator over a [`DataLayerMask`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DataLayerMaskIter {
    remaining: u64,
}

impl Iterator for DataLayerMaskIter {
    type Item = DataLayerId;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        let index = self.remaining.trailing_zeros();
        self.remaining &= self.remaining - 1;
        Some(DataLayerId(index))
    }
}

impl IntoIterator for DataLayerMask {
    type Item = DataLayerId;
    type IntoIter = DataLayerMaskIter;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// A signed, lossless world-space sector identity.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct SectorId {
    /// Quadtree/octree level, where zero is the root grid.
    pub level: u8,
    /// Signed sector coordinate along X.
    pub x: i64,
    /// Signed sector coordinate along Y.
    pub y: i64,
    /// Signed sector coordinate along Z.
    pub z: i64,
}

impl SectorId {
    /// Creates a sector identity.
    pub const fn new(level: u8, x: i64, y: i64, z: i64) -> Self {
        Self { level, x, y, z }
    }

    /// Returns the signed Chebyshev distance between two sectors at one level.
    pub fn distance(self, other: Self) -> u64 {
        let dx = self.x.abs_diff(other.x);
        let dy = self.y.abs_diff(other.y);
        let dz = self.z.abs_diff(other.z);
        dx.max(dy).max(dz)
    }

    /// Returns a locality-preserving 21-bit-per-axis Morton hint.
    ///
    /// The hint is deliberately not the identity: full signed coordinates are
    /// retained by [`SectorId`], while this compact value is used only for
    /// ordering nearby cells in caches and traces.
    pub fn morton_hint(self) -> u64 {
        let x = zigzag21(self.x);
        let y = zigzag21(self.y);
        let z = zigzag21(self.z);
        morton_encode_21(x, y, z) ^ (u64::from(self.level) << 63)
    }
}

fn zigzag21(value: i64) -> u64 {
    // Perform the shift in unsigned space: `i64::MIN << 1` is a debug-build
    // overflow even though its ZigZag representation is perfectly defined.
    let mapped = (value as u64).wrapping_shl(1) ^ ((value >> 63) as u64);
    mapped & ((1 << 21) - 1)
}

fn morton_encode_21(x: u64, y: u64, z: u64) -> u64 {
    let mut out = 0;
    let mut bit = 0;
    while bit < 21 {
        out |= ((x >> bit) & 1) << (bit * 3);
        out |= ((y >> bit) & 1) << (bit * 3 + 1);
        out |= ((z >> bit) & 1) << (bit * 3 + 2);
        bit += 1;
    }
    out
}

/// Encodes three unsigned 21-bit coordinates into a Morton key.
pub const fn morton_encode(x: u32, y: u32, z: u32) -> u64 {
    let x = (x & 0x1f_ffff) as u64;
    let y = (y & 0x1f_ffff) as u64;
    let z = (z & 0x1f_ffff) as u64;
    let mut out = 0;
    let mut bit = 0;
    while bit < 21 {
        out |= ((x >> bit) & 1) << (bit * 3);
        out |= ((y >> bit) & 1) << (bit * 3 + 1);
        out |= ((z >> bit) & 1) << (bit * 3 + 2);
        bit += 1;
    }
    out
}

/// Decodes a 21-bit-per-axis Morton key.
pub const fn morton_decode(key: u64) -> (u32, u32, u32) {
    let mut x = 0;
    let mut y = 0;
    let mut z = 0;
    let mut bit = 0;
    while bit < 21 {
        x |= ((key >> (bit * 3)) & 1) << bit;
        y |= ((key >> (bit * 3 + 1)) & 1) << bit;
        z |= ((key >> (bit * 3 + 2)) & 1) << bit;
        bit += 1;
    }
    (x as u32, y as u32, z as u32)
}

/// Global position in signed millimetres.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct GlobalPos {
    /// X millimetres.
    pub x: i64,
    /// Y millimetres.
    pub y: i64,
    /// Z millimetres.
    pub z: i64,
}

impl GlobalPos {
    /// Creates a global position.
    pub const fn new(x: i64, y: i64, z: i64) -> Self {
        Self { x, y, z }
    }
}

/// Sector-local position in millimetres.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct LocalPos {
    /// X millimetres from the sector origin.
    pub x: i64,
    /// Y millimetres from the sector origin.
    pub y: i64,
    /// Z millimetres from the sector origin.
    pub z: i64,
}

/// Fixed-size sector grid used for lossless global/local conversion.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SectorGrid {
    sector_size_mm: i64,
}

impl SectorGrid {
    /// Creates a grid; zero and negative sizes are rejected.
    pub const fn new(sector_size_mm: i64) -> Option<Self> {
        if sector_size_mm > 0 {
            Some(Self { sector_size_mm })
        } else {
            None
        }
    }

    /// Returns the configured sector width in millimetres.
    pub const fn sector_size_mm(self) -> i64 {
        self.sector_size_mm
    }

    /// Converts a global position into a sector and non-negative local offset.
    pub fn locate(self, level: u8, position: GlobalPos) -> (SectorId, LocalPos) {
        let size = self.sector_size_mm;
        let sector = SectorId::new(
            level,
            position.x.div_euclid(size),
            position.y.div_euclid(size),
            position.z.div_euclid(size),
        );
        let local = LocalPos {
            x: position.x.rem_euclid(size),
            y: position.y.rem_euclid(size),
            z: position.z.rem_euclid(size),
        };
        (sector, local)
    }

    /// Reconstructs a global position from a sector and local offset.
    pub fn global(self, sector: SectorId, local: LocalPos) -> Option<GlobalPos> {
        if local.x < 0
            || local.x >= self.sector_size_mm
            || local.y < 0
            || local.y >= self.sector_size_mm
            || local.z < 0
            || local.z >= self.sector_size_mm
        {
            return None;
        }
        Some(GlobalPos {
            x: sector
                .x
                .checked_mul(self.sector_size_mm)?
                .checked_add(local.x)?,
            y: sector
                .y
                .checked_mul(self.sector_size_mm)?
                .checked_add(local.y)?,
            z: sector
                .z
                .checked_mul(self.sector_size_mm)?
                .checked_add(local.z)?,
        })
    }
}

/// Inclusive integer bounds for a cell manifest.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AabbI64 {
    /// Minimum corner.
    pub min: GlobalPos,
    /// Maximum corner.
    pub max: GlobalPos,
}

impl AabbI64 {
    /// Creates bounds, rejecting inverted axes.
    pub const fn new(min: GlobalPos, max: GlobalPos) -> Option<Self> {
        if min.x <= max.x && min.y <= max.y && min.z <= max.z {
            Some(Self { min, max })
        } else {
            None
        }
    }
}

/// Stream metadata that does not own scene content.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CellManifest {
    /// Spatial identity.
    pub sector: SectorId,
    /// Authoritative bounds.
    pub bounds: AabbI64,
    /// Stable scene/template key.
    pub scene_key: u128,
    /// Scene schema/content version.
    pub content_version: u64,
    /// Hash of all derived payloads for cache invalidation.
    pub derived_hash: u128,
    /// Approximate bytes required by the requested representation.
    pub io_bytes: u64,
    /// Data Layers required by this cell payload.
    pub data_layers: DataLayerMask,
}

/// A source of world interest.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DemandSource {
    /// Center sector.
    pub center: SectorId,
    /// Radius in sectors at which cells are requested.
    pub load_radius: u16,
    /// Radius in sectors at which cells may be evicted.
    pub unload_radius: u16,
    /// Base priority, with larger values winning.
    pub priority: u16,
    /// Optional predicted next sector for one-step prefetch.
    pub predicted: Option<SectorId>,
    /// Data Layers available/requested for admission.
    pub data_layers: DataLayerMask,
}

impl DemandSource {
    /// Creates an interest source, rejecting an unload radius below load.
    pub const fn new(
        center: SectorId,
        load_radius: u16,
        unload_radius: u16,
        priority: u16,
    ) -> Option<Self> {
        if unload_radius >= load_radius {
            Some(Self {
                center,
                load_radius,
                unload_radius,
                priority,
                predicted: None,
                data_layers: DataLayerMask::EMPTY,
            })
        } else {
            None
        }
    }

    /// Adds a predicted sector used for prefetch.
    pub const fn with_prediction(mut self, predicted: SectorId) -> Self {
        self.predicted = Some(predicted);
        self
    }

    /// Restricts admission to manifests whose Data Layer mask is a subset of
    /// `layers`.
    pub const fn with_data_layers(mut self, layers: DataLayerMask) -> Self {
        self.data_layers = layers;
        self
    }
}

/// Priority class for one stream request.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct StreamPriority(pub u16);

/// Reason for an action.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum StreamKind {
    /// Required by the current load radius.
    Required,
    /// Predicted one-step prefetch.
    Prefetch,
}

/// A deterministic request candidate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StreamRequest {
    /// Target sector.
    pub sector: SectorId,
    /// Required IO bytes.
    pub io_bytes: u64,
    /// Priority class.
    pub priority: StreamPriority,
    /// Distance from the closest demand source.
    pub distance: u64,
    /// Request kind.
    pub kind: StreamKind,
}

/// Per-frame streaming budget.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StreamBudget {
    /// Maximum bytes admitted this frame.
    pub io_bytes: u64,
    /// Maximum action count admitted this frame.
    pub max_actions: usize,
}

/// A streaming operation emitted by the scheduler.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StreamAction {
    /// Begin loading a sector.
    Load(StreamRequest),
    /// Load a predicted sector when budget remains.
    Prefetch(StreamRequest),
    /// Evict a sector outside all unload radii.
    Unload(SectorId),
}

/// Result of one deterministic scheduling pass.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StreamPlan {
    /// Ordered actions.
    pub actions: Vec<StreamAction>,
    /// Total bytes admitted by load, reload, and prefetch actions.
    pub admitted_bytes: u64,
    /// The exact VM instruction stream that produced this plan.
    pub trace: VmTrace,
}

/// Invalid manifest input rejected before scheduler state changes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ManifestValidationError {
    /// Two entries claimed the same sector identity.
    DuplicateSector {
        /// The duplicated sector.
        sector: SectorId,
        /// Index of the first entry in the caller's slice.
        first_index: usize,
        /// Index of the duplicate entry in the caller's slice.
        duplicate_index: usize,
    },
}

impl core::fmt::Display for ManifestValidationError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::DuplicateSector {
                sector,
                first_index,
                duplicate_index,
            } => write!(
                formatter,
                "duplicate manifest sector {sector:?} at indices {first_index} and {duplicate_index}"
            ),
        }
    }
}

impl core::error::Error for ManifestValidationError {}

/// Validates that each sector has exactly one manifest entry.
pub fn validate_manifests(manifests: &[CellManifest]) -> Result<(), ManifestValidationError> {
    let mut indexed: Vec<_> = manifests
        .iter()
        .enumerate()
        .map(|(index, manifest)| (manifest.sector, index))
        .collect();
    indexed.sort_unstable_by_key(|(sector, index)| (*sector, *index));
    for pair in indexed.windows(2) {
        if pair[0].0 == pair[1].0 {
            return Err(ManifestValidationError::DuplicateSector {
                sector: pair[0].0,
                first_index: pair[0].1,
                duplicate_index: pair[1].1,
            });
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ResidentState {
    Loading,
    Prefetched(CellManifest),
    Resident(Option<CellManifest>),
}

impl ResidentState {
    fn is_stale(self, manifest: CellManifest) -> bool {
        matches!(
            self,
            Self::Prefetched(previous) | Self::Resident(Some(previous))
                if previous.content_version != manifest.content_version
                    || previous.derived_hash != manifest.derived_hash
                    || previous.scene_key != manifest.scene_key
                    || previous.data_layers != manifest.data_layers
        )
    }
}

/// Budgeted streaming VM state.
#[derive(Clone, Debug, Default)]
pub struct StreamScheduler {
    resident: BTreeMap<SectorId, ResidentState>,
    frame: u64,
}

impl StreamScheduler {
    /// Creates an empty scheduler.
    pub const fn new() -> Self {
        Self {
            resident: BTreeMap::new(),
            frame: 0,
        }
    }

    /// Marks a sector as resident after its provider completes IO.
    pub fn mark_resident(&mut self, sector: SectorId) {
        self.resident.insert(sector, ResidentState::Resident(None));
    }

    /// Marks a sector resident and records the manifest revision that was
    /// actually committed by the provider/host.
    pub fn mark_resident_manifest(&mut self, manifest: CellManifest) {
        self.resident
            .insert(manifest.sector, ResidentState::Resident(Some(manifest)));
    }

    /// Marks a validated prefetch as available without making the cell scene
    /// resident. A later required request promotes it through the runtime host.
    pub fn mark_prefetched_manifest(&mut self, manifest: CellManifest) {
        self.resident
            .insert(manifest.sector, ResidentState::Prefetched(manifest));
    }

    /// Marks a sector as no longer resident.
    pub fn mark_unloaded(&mut self, sector: SectorId) {
        self.resident.remove(&sector);
    }

    /// Returns whether a sector is resident or loading.
    pub fn contains(&self, sector: SectorId) -> bool {
        self.resident.contains_key(&sector)
    }

    /// Produces a deterministic plan from unique manifests.
    pub fn try_plan(
        &mut self,
        demands: &[DemandSource],
        manifests: &[CellManifest],
        budget: StreamBudget,
    ) -> Result<StreamPlan, ManifestValidationError> {
        validate_manifests(manifests)?;
        self.frame = self.frame.saturating_add(1);
        let mut trace = VmTrace::new();
        trace.push(VmInstruction::BeginFrame(self.frame));
        for demand in demands {
            trace.push(VmInstruction::Demand(demand.center));
        }
        let manifest_by_sector: BTreeMap<_, _> = manifests
            .iter()
            .copied()
            .map(|manifest| (manifest.sector, manifest))
            .collect();
        let mut requests = BTreeMap::<SectorId, StreamRequest>::new();
        let mut stale_sectors = BTreeSet::new();
        for manifest in manifests {
            let mut best: Option<(u16, u64, StreamKind)> = None;
            for demand in demands {
                if demand.center.level != manifest.sector.level {
                    continue;
                }
                if !manifest.data_layers.is_subset_of(demand.data_layers) {
                    continue;
                }
                let distance = demand.center.distance(manifest.sector);
                let requested_kind = if distance <= u64::from(demand.load_radius) {
                    StreamKind::Required
                } else if demand.predicted == Some(manifest.sector) {
                    StreamKind::Prefetch
                } else {
                    continue;
                };
                if requested_kind == StreamKind::Prefetch
                    && distance > u64::from(demand.unload_radius) + 1
                {
                    continue;
                }
                let stale = self
                    .resident
                    .get(&manifest.sector)
                    .copied()
                    .filter(|state| state.is_stale(*manifest))
                    .is_some();
                if stale {
                    stale_sectors.insert(manifest.sector);
                }
                let score = demand.priority.saturating_add(
                    demand
                        .load_radius
                        .saturating_sub(distance.min(u64::from(u16::MAX)) as u16),
                );
                if best.is_none_or(|(p, d, k)| {
                    score > p
                        || (score == p && distance < d)
                        || (score == p
                            && distance == d
                            && requested_kind == StreamKind::Required
                            && k == StreamKind::Prefetch)
                }) {
                    best = Some((score, distance, requested_kind));
                }
            }
            if let Some((score, distance, kind)) = best {
                requests.insert(
                    manifest.sector,
                    StreamRequest {
                        sector: manifest.sector,
                        io_bytes: manifest.io_bytes,
                        priority: StreamPriority(score),
                        distance,
                        kind,
                    },
                );
            }
        }

        let mut actions = Vec::new();
        let mut candidates: Vec<StreamRequest> = requests
            .values()
            .copied()
            .filter(|r| {
                self.resident.get(&r.sector).is_none_or(|state| {
                    stale_sectors.contains(&r.sector)
                        || (r.kind == StreamKind::Required
                            && matches!(state, ResidentState::Prefetched(_)))
                })
            })
            .collect();
        candidates.sort_by(request_order);
        let mut admitted_bytes: u64 = 0;
        for request in candidates {
            if actions.len() >= budget.max_actions
                || admitted_bytes.saturating_add(request.io_bytes) > budget.io_bytes
            {
                break;
            }
            self.resident.insert(request.sector, ResidentState::Loading);
            admitted_bytes = admitted_bytes.saturating_add(request.io_bytes);
            trace.push(VmInstruction::Admit(request.sector));
            actions.push(if request.kind == StreamKind::Required {
                StreamAction::Load(request)
            } else {
                StreamAction::Prefetch(request)
            });
        }

        let mut evictions: Vec<SectorId> = self
            .resident
            .keys()
            .copied()
            .filter(|sector| {
                !within_unload(*sector, manifest_by_sector.get(sector).copied(), demands)
            })
            .collect();
        evictions.sort_unstable();
        for sector in evictions {
            if actions.len() >= budget.max_actions {
                break;
            }
            self.resident.remove(&sector);
            trace.push(VmInstruction::Evict(sector));
            actions.push(StreamAction::Unload(sector));
        }
        trace.push(VmInstruction::EndFrame {
            frame: self.frame,
            admitted_bytes,
        });
        Ok(StreamPlan {
            actions,
            admitted_bytes,
            trace,
        })
    }

    /// Produces the same plan as [`Self::try_plan`] while obtaining load
    /// candidates from a prebuilt hierarchical manifest index.
    ///
    /// The index is an acceleration structure, not a second source of truth:
    /// resident revisions, layer admission, hysteresis, byte budgets, and
    /// action ordering remain identical to the linear planner. Build one index
    /// for an immutable manifest snapshot and replace it when that snapshot
    /// changes.
    pub fn try_plan_indexed(
        &mut self,
        demands: &[DemandSource],
        index: &ManifestSpatialIndex,
        budget: StreamBudget,
    ) -> StreamPlan {
        self.frame = self.frame.saturating_add(1);
        let mut trace = VmTrace::new();
        trace.push(VmInstruction::BeginFrame(self.frame));
        for demand in demands {
            trace.push(VmInstruction::Demand(demand.center));
        }

        let mut requests = BTreeMap::<SectorId, StreamRequest>::new();
        let mut stale_sectors = BTreeSet::new();
        for demand in demands {
            let radius = u64::from(demand.load_radius).max(u64::from(demand.unload_radius) + 1);
            for manifest in index.query_sector_cube(demand.center, radius) {
                if demand.center.level != manifest.sector.level
                    || !manifest.data_layers.is_subset_of(demand.data_layers)
                {
                    continue;
                }
                let distance = demand.center.distance(manifest.sector);
                let requested_kind = if distance <= u64::from(demand.load_radius) {
                    StreamKind::Required
                } else if demand.predicted == Some(manifest.sector) {
                    StreamKind::Prefetch
                } else {
                    continue;
                };
                if requested_kind == StreamKind::Prefetch
                    && distance > u64::from(demand.unload_radius) + 1
                {
                    continue;
                }
                let stale = self
                    .resident
                    .get(&manifest.sector)
                    .copied()
                    .filter(|state| state.is_stale(manifest))
                    .is_some();
                if stale {
                    stale_sectors.insert(manifest.sector);
                }
                let score = demand.priority.saturating_add(
                    demand
                        .load_radius
                        .saturating_sub(distance.min(u64::from(u16::MAX)) as u16),
                );
                let request = StreamRequest {
                    sector: manifest.sector,
                    io_bytes: manifest.io_bytes,
                    priority: StreamPriority(score),
                    distance,
                    kind: requested_kind,
                };
                if requests
                    .get(&manifest.sector)
                    .is_none_or(|current| request_order(&request, current) == Ordering::Less)
                {
                    requests.insert(manifest.sector, request);
                }
            }
        }

        let mut actions = Vec::new();
        let mut candidates: Vec<StreamRequest> = requests
            .values()
            .copied()
            .filter(|request| {
                self.resident.get(&request.sector).is_none_or(|state| {
                    stale_sectors.contains(&request.sector)
                        || (request.kind == StreamKind::Required
                            && matches!(state, ResidentState::Prefetched(_)))
                })
            })
            .collect();
        candidates.sort_by(request_order);
        let mut admitted_bytes = 0_u64;
        for request in candidates {
            if actions.len() >= budget.max_actions
                || admitted_bytes.saturating_add(request.io_bytes) > budget.io_bytes
            {
                break;
            }
            self.resident.insert(request.sector, ResidentState::Loading);
            admitted_bytes = admitted_bytes.saturating_add(request.io_bytes);
            trace.push(VmInstruction::Admit(request.sector));
            actions.push(if request.kind == StreamKind::Required {
                StreamAction::Load(request)
            } else {
                StreamAction::Prefetch(request)
            });
        }

        let mut evictions: Vec<SectorId> = self
            .resident
            .keys()
            .copied()
            .filter(|sector| {
                let manifest = index.get(*sector);
                !within_unload(*sector, manifest, demands)
            })
            .collect();
        evictions.sort_unstable();
        for sector in evictions {
            if actions.len() >= budget.max_actions {
                break;
            }
            self.resident.remove(&sector);
            trace.push(VmInstruction::Evict(sector));
            actions.push(StreamAction::Unload(sector));
        }
        trace.push(VmInstruction::EndFrame {
            frame: self.frame,
            admitted_bytes,
        });
        StreamPlan {
            actions,
            admitted_bytes,
            trace,
        }
    }

    /// Produces a deterministic plan, panicking on invalid duplicate input.
    ///
    /// New callers that receive external or dynamically assembled manifests
    /// should use [`Self::try_plan`] so the validation error can be propagated.
    pub fn plan(
        &mut self,
        demands: &[DemandSource],
        manifests: &[CellManifest],
        budget: StreamBudget,
    ) -> StreamPlan {
        self.try_plan(demands, manifests, budget)
            .expect("stream manifests must contain unique sectors")
    }

    /// Returns the frame counter used in VM traces.
    pub const fn frame(&self) -> u64 {
        self.frame
    }
}

fn request_order(a: &StreamRequest, b: &StreamRequest) -> Ordering {
    b.priority
        .cmp(&a.priority)
        .then_with(|| a.distance.cmp(&b.distance))
        .then_with(|| a.kind.cmp(&b.kind))
        .then_with(|| a.sector.cmp(&b.sector))
}

fn within_unload(
    sector: SectorId,
    manifest: Option<CellManifest>,
    demands: &[DemandSource],
) -> bool {
    demands.iter().any(|d| {
        d.center.level == sector.level
            && d.center.distance(sector) <= u64::from(d.unload_radius)
            && manifest.is_none_or(|manifest| manifest.data_layers.is_subset_of(d.data_layers))
    })
}

/// A replayable VM instruction.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VmInstruction {
    /// Starts a frame.
    BeginFrame(u64),
    /// Records a demand source.
    Demand(SectorId),
    /// Admits an action.
    Admit(SectorId),
    /// Evicts a sector.
    Evict(SectorId),
    /// Ends a frame with admitted bytes.
    EndFrame {
        /// Frame number.
        frame: u64,
        /// Bytes admitted by the frame plan.
        admitted_bytes: u64,
    },
}

/// Append-only deterministic VM trace.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct VmTrace {
    instructions: Vec<VmInstruction>,
}

impl VmTrace {
    /// Creates an empty trace.
    pub const fn new() -> Self {
        Self {
            instructions: Vec::new(),
        }
    }

    /// Records an instruction.
    pub fn push(&mut self, instruction: VmInstruction) {
        self.instructions.push(instruction);
    }

    /// Returns the recorded instruction stream.
    pub fn as_slice(&self) -> &[VmInstruction] {
        &self.instructions
    }

    /// Validates frame nesting and returns the number of valid frames.
    pub fn replay(&self) -> Result<usize, TraceError> {
        let mut open = None;
        let mut count = 0;
        for instruction in &self.instructions {
            match *instruction {
                VmInstruction::BeginFrame(frame) => {
                    if open.is_some() {
                        return Err(TraceError::NestedFrame);
                    }
                    open = Some(frame);
                }
                VmInstruction::EndFrame { frame, .. } => {
                    if open != Some(frame) {
                        return Err(TraceError::FrameMismatch);
                    }
                    open = None;
                    count += 1;
                }
                VmInstruction::Demand(_) | VmInstruction::Admit(_) | VmInstruction::Evict(_) => {
                    if open.is_none() {
                        return Err(TraceError::InstructionOutsideFrame);
                    }
                }
            }
        }
        if open.is_some() {
            return Err(TraceError::UnclosedFrame);
        }
        Ok(count)
    }
}

/// Errors detected while replaying a stream VM trace.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TraceError {
    /// A frame began while another frame was open.
    NestedFrame,
    /// An end marker did not match the active frame.
    FrameMismatch,
    /// An operation appeared outside a frame.
    InstructionOutsideFrame,
    /// The trace ended while a frame was open.
    UnclosedFrame,
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn morton_round_trip_and_negative_sector_hint_are_stable() {
        let key = morton_encode(0x1f_ffff, 17, 91);
        assert_eq!(morton_decode(key), (0x1f_ffff, 17, 91));
        assert_eq!(
            SectorId::new(2, -4, 7, 9).morton_hint(),
            SectorId::new(2, -4, 7, 9).morton_hint()
        );
        assert_ne!(
            SectorId::new(2, i64::MIN, i64::MAX, -1).morton_hint(),
            SectorId::new(2, 0, 0, 0).morton_hint()
        );
    }

    #[test]
    fn coordinate_conversion_is_lossless_across_zero() {
        let grid = SectorGrid::new(1000).unwrap();
        for position in [
            GlobalPos::new(-1001, -1, 1000),
            GlobalPos::new(0, 0, 0),
            GlobalPos::new(1999, 2000, -2001),
        ] {
            let (sector, local) = grid.locate(0, position);
            assert_eq!(grid.global(sector, local), Some(position));
        }
    }

    #[test]
    fn data_layer_mask_is_fixed_width_canonical_and_ordered() {
        let mask = DataLayerMask::from_layers(&[
            DataLayerId::new(63),
            DataLayerId::new(0),
            DataLayerId::new(7),
        ])
        .unwrap();
        assert_eq!(mask.bits(), (1_u64 << 63) | 1 | (1_u64 << 7));
        assert_eq!(mask.count(), 3);
        assert_eq!(
            mask.into_iter().collect::<Vec<_>>(),
            vec![
                DataLayerId::new(0),
                DataLayerId::new(7),
                DataLayerId::new(63),
            ]
        );
        assert_eq!(
            DataLayerMask::from_layers(&[DataLayerId::new(7), DataLayerId::new(7)]),
            Err(DataLayerMaskError::Duplicate {
                layer: DataLayerId::new(7)
            })
        );
        assert_eq!(
            DataLayerMask::from_layers(&[DataLayerId::new(MAX_DATA_LAYERS)]),
            Err(DataLayerMaskError::OutOfRange {
                layer: DataLayerId::new(MAX_DATA_LAYERS)
            })
        );
    }

    #[test]
    fn indexed_planner_matches_linear_planner_and_preserves_hysteresis() {
        let bounds = AabbI64::new(GlobalPos::new(0, 0, 0), GlobalPos::new(1, 1, 1)).unwrap();
        let manifests = [
            CellManifest {
                sector: SectorId::new(0, -2, 0, 0),
                bounds,
                scene_key: 1,
                content_version: 1,
                derived_hash: 2,
                io_bytes: 10,
                data_layers: DataLayerMask::EMPTY,
            },
            CellManifest {
                sector: SectorId::new(0, 0, 0, 0),
                bounds,
                scene_key: 2,
                content_version: 1,
                derived_hash: 3,
                io_bytes: 11,
                data_layers: DataLayerMask::EMPTY,
            },
            CellManifest {
                sector: SectorId::new(0, 2, 0, 0),
                bounds,
                scene_key: 3,
                content_version: 1,
                derived_hash: 4,
                io_bytes: 12,
                data_layers: DataLayerMask::EMPTY,
            },
        ];
        let demand = DemandSource::new(SectorId::new(0, 0, 0, 0), 1, 2, 7)
            .unwrap()
            .with_prediction(SectorId::new(0, 2, 0, 0));
        let budget = StreamBudget {
            io_bytes: 64,
            max_actions: 8,
        };
        let index = ManifestSpatialIndex::build(&manifests).unwrap();
        let mut linear = StreamScheduler::new();
        let mut indexed = StreamScheduler::new();
        let linear_plan = linear.try_plan(&[demand], &manifests, budget).unwrap();
        let indexed_plan = indexed.try_plan_indexed(&[demand], &index, budget);
        assert_eq!(indexed_plan.actions, linear_plan.actions);
        assert_eq!(indexed_plan.admitted_bytes, linear_plan.admitted_bytes);
        assert_eq!(indexed_plan.trace, linear_plan.trace);

        let outside = DemandSource::new(SectorId::new(0, 10, 0, 0), 0, 0, 7).unwrap();
        let linear_eviction = linear.try_plan(&[outside], &manifests, budget).unwrap();
        let indexed_eviction = indexed.try_plan_indexed(&[outside], &index, budget);
        assert_eq!(indexed_eviction.actions, linear_eviction.actions);
        assert_eq!(
            indexed_eviction.admitted_bytes,
            linear_eviction.admitted_bytes
        );
        assert_eq!(indexed_eviction.trace, linear_eviction.trace);
    }

    #[test]
    fn scheduler_admission_requires_manifest_layers_in_each_demand() {
        let sector = SectorId::new(0, 0, 0, 0);
        let manifest = CellManifest {
            sector,
            bounds: AabbI64::new(GlobalPos::new(0, 0, 0), GlobalPos::new(1, 1, 1)).unwrap(),
            scene_key: 1,
            content_version: 1,
            derived_hash: 2,
            io_bytes: 8,
            data_layers: DataLayerMask::from_layers(&[DataLayerId::new(3)]).unwrap(),
        };
        let demand = DemandSource::new(sector, 0, 1, 10).unwrap();
        let mut scheduler = StreamScheduler::new();
        let blocked = scheduler
            .try_plan(
                &[demand],
                &[manifest],
                StreamBudget {
                    io_bytes: 8,
                    max_actions: 1,
                },
            )
            .unwrap();
        assert!(blocked.actions.is_empty());
        assert_eq!(scheduler.frame(), 1);

        let enabled = demand.with_data_layers(manifest.data_layers);
        let admitted = scheduler
            .try_plan(
                &[enabled],
                &[manifest],
                StreamBudget {
                    io_bytes: 8,
                    max_actions: 1,
                },
            )
            .unwrap();
        assert!(matches!(
            admitted.actions.as_slice(),
            [StreamAction::Load(_)]
        ));
    }

    #[test]
    fn layer_unavailability_evicts_and_layer_revision_reloads() {
        let sector = SectorId::new(0, 0, 0, 0);
        let bounds = AabbI64::new(GlobalPos::new(0, 0, 0), GlobalPos::new(1, 1, 1)).unwrap();
        let old = CellManifest {
            sector,
            bounds,
            scene_key: 1,
            content_version: 1,
            derived_hash: 2,
            io_bytes: 8,
            data_layers: DataLayerMask::from_bits(1 << 1),
        };
        let mut scheduler = StreamScheduler::new();
        scheduler.mark_resident_manifest(old);
        let spatial_demand = DemandSource::new(sector, 0, 1, 10).unwrap();
        let unavailable = scheduler.plan(
            &[spatial_demand],
            &[old],
            StreamBudget {
                io_bytes: 8,
                max_actions: 1,
            },
        );
        assert_eq!(unavailable.actions, vec![StreamAction::Unload(sector)]);

        scheduler.mark_resident_manifest(old);
        let revised = CellManifest {
            data_layers: DataLayerMask::from_bits(1 << 2),
            ..old
        };
        let available = DataLayerMask::from_bits((1 << 1) | (1 << 2));
        let reload = scheduler.plan(
            &[spatial_demand.with_data_layers(available)],
            &[revised],
            StreamBudget {
                io_bytes: 8,
                max_actions: 1,
            },
        );
        assert!(matches!(reload.actions.as_slice(), [StreamAction::Load(_)]));
    }

    #[test]
    fn scheduler_is_budgeted_hysteretic_and_deterministic() {
        let center = SectorId::new(0, 0, 0, 0);
        let demand = DemandSource::new(center, 1, 2, 10).unwrap();
        let bounds = AabbI64::new(GlobalPos::new(0, 0, 0), GlobalPos::new(1, 1, 1)).unwrap();
        let manifests = [
            CellManifest {
                sector: center,
                bounds,
                scene_key: 1,
                content_version: 1,
                derived_hash: 2,
                io_bytes: 50,
                data_layers: DataLayerMask::EMPTY,
            },
            CellManifest {
                sector: SectorId::new(0, 1, 0, 0),
                bounds,
                scene_key: 2,
                content_version: 1,
                derived_hash: 3,
                io_bytes: 60,
                data_layers: DataLayerMask::EMPTY,
            },
            CellManifest {
                sector: SectorId::new(0, 4, 0, 0),
                bounds,
                scene_key: 3,
                content_version: 1,
                derived_hash: 4,
                io_bytes: 70,
                data_layers: DataLayerMask::EMPTY,
            },
        ];
        let mut scheduler = StreamScheduler::new();
        let plan = scheduler.plan(
            &[demand],
            &manifests,
            StreamBudget {
                io_bytes: 100,
                max_actions: 8,
            },
        );
        assert_eq!(plan.admitted_bytes, 50);
        assert_eq!(plan.trace.replay(), Ok(1));
        assert!(matches!(
            plan.actions[0],
            StreamAction::Load(StreamRequest {
                sector: SectorId { x: 0, .. },
                ..
            })
        ));
        scheduler.mark_resident(center);
        let plan = scheduler.plan(
            &[demand],
            &manifests,
            StreamBudget {
                io_bytes: 100,
                max_actions: 8,
            },
        );
        assert!(plan
            .actions
            .iter()
            .all(|a| !matches!(a, StreamAction::Unload(_))));
    }

    #[test]
    fn duplicate_manifests_fail_before_frame_state_changes() {
        let sector = SectorId::new(0, 3, 0, 0);
        let bounds = AabbI64::new(GlobalPos::new(0, 0, 0), GlobalPos::new(1, 1, 1)).unwrap();
        let first = CellManifest {
            sector,
            bounds,
            scene_key: 1,
            content_version: 1,
            derived_hash: 2,
            io_bytes: 1,
            data_layers: DataLayerMask::EMPTY,
        };
        let duplicate = CellManifest {
            scene_key: 9,
            io_bytes: 99,
            ..first
        };
        let demand = DemandSource::new(sector, 0, 1, 10).unwrap();
        let mut scheduler = StreamScheduler::new();
        assert_eq!(
            scheduler.try_plan(
                &[demand],
                &[first, duplicate],
                StreamBudget {
                    io_bytes: 100,
                    max_actions: 2,
                },
            ),
            Err(ManifestValidationError::DuplicateSector {
                sector,
                first_index: 0,
                duplicate_index: 1,
            })
        );
        assert_eq!(scheduler.frame(), 0);
        let valid = scheduler
            .try_plan(
                &[demand],
                &[first],
                StreamBudget {
                    io_bytes: 100,
                    max_actions: 2,
                },
            )
            .unwrap();
        assert_eq!(valid.trace.as_slice()[0], VmInstruction::BeginFrame(1));
    }

    #[test]
    fn trace_replay_rejects_unclosed_frames() {
        let mut trace = VmTrace::new();
        trace.push(VmInstruction::BeginFrame(1));
        assert_eq!(trace.replay(), Err(TraceError::UnclosedFrame));
        trace.push(VmInstruction::EndFrame {
            frame: 1,
            admitted_bytes: 0,
        });
        assert_eq!(trace.replay(), Ok(1));
    }

    #[test]
    fn resident_manifest_revision_emits_reload_without_duplicate_load() {
        let sector = SectorId::new(0, 0, 0, 0);
        let demand = DemandSource::new(sector, 0, 1, 10).expect("demand");
        let bounds = AabbI64::new(GlobalPos::new(0, 0, 0), GlobalPos::new(1, 1, 1)).unwrap();
        let old = CellManifest {
            sector,
            bounds,
            scene_key: 7,
            content_version: 1,
            derived_hash: 10,
            io_bytes: 4,
            data_layers: DataLayerMask::EMPTY,
        };
        let new = CellManifest {
            content_version: 2,
            derived_hash: 11,
            ..old
        };
        let mut scheduler = StreamScheduler::new();
        scheduler.mark_resident_manifest(old);

        let plan = scheduler.plan(
            &[demand],
            &[new],
            StreamBudget {
                io_bytes: 4,
                max_actions: 1,
            },
        );

        assert_eq!(plan.admitted_bytes, 4);
        assert_eq!(
            plan.actions,
            vec![StreamAction::Load(StreamRequest {
                sector,
                io_bytes: 4,
                priority: StreamPriority(10),
                distance: 0,
                kind: StreamKind::Required,
            })]
        );
        assert_eq!(plan.trace.replay(), Ok(1));
    }
}
