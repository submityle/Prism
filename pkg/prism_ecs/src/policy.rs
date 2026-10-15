//! Explicit, project-independent ECS scale policy (design §3).
//!
//! A quality label is not a hardware probe. The host supplies capabilities,
//! the project supplies a profile, and [`EcsPolicy::resolve`] checks them
//! against the features actually compiled into this crate. The conservative
//! default requests no optional facilities and imposes no project-size limit.
//!
//! Policies do not silently change existing worlds, schedulers, or allocators.
//! Entity admission is an explicit synchronization-point check, chunk batches
//! use the table's real layout, and partition factories create the existing
//! LOD/streaming implementations. Store a resolved policy as a resource if
//! systems need to share the same decisions.

use alloc::borrow::Cow;
use alloc::vec::Vec;
use core::fmt;
use core::iter::FusedIterator;
use core::ops::Range;

use crate::storage::{Table, TARGET_CHUNK_BYTES};
use crate::world::World;

/// Content-scale label, never used to infer hardware or select magic budgets.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum QualityTier {
    /// Stable, unrestricted core-only default.
    #[default]
    Baseline,
    /// Project-defined mobile content profile.
    Mobile,
    /// Project-defined console content profile.
    Console,
    /// Project-defined high-end desktop content profile.
    HighEndDesktop,
    /// An application-defined label with exactly the same policy semantics.
    Custom(Cow<'static, str>),
}

/// Host-reported capabilities, not guesses based on an OS or quality label.
///
/// A reported capability only makes a requirement eligible; this object does
/// not enable CPU instructions or prove that a particular GPU allocation
/// succeeded. SIMD and GPU users must still use their backend's safe dispatch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EcsCapabilities {
    /// Workers the host makes available to ECS, including the calling thread.
    pub worker_threads: usize,
    /// Usable SIMD width in bits; zero means scalar/unknown.
    pub simd_width_bits: u16,
    /// Whether the selected device/backend supports persistent resident mapping.
    pub gpu_resident_mapping: bool,
    /// Whether the host reports a usable virtual-memory backend.
    pub virtual_memory: bool,
}

impl Default for EcsCapabilities {
    fn default() -> Self {
        Self {
            worker_threads: 1,
            simd_width_bits: 0,
            gpu_resident_mapping: false,
            virtual_memory: false,
        }
    }
}

/// Facilities that must be available rather than silently downgraded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EcsRequirements {
    /// Minimum ECS workers; values above one require `multi_thread`.
    pub min_workers: usize,
    /// Minimum host-reported SIMD width; nonzero also requires `simd`.
    pub min_simd_width_bits: u16,
    /// Require both `gpu_resident` and host-reported resident mapping support.
    pub gpu_resident_mapping: bool,
    /// Require both `vm` and host-reported virtual-memory support.
    ///
    /// This is not a huge-page, allocation-success, or NUMA-binding guarantee.
    pub virtual_memory: bool,
    /// Require the optional determinism auditing facility.
    pub determinism: bool,
}

impl Default for EcsRequirements {
    fn default() -> Self {
        Self {
            min_workers: 1,
            min_simd_width_bits: 0,
            gpu_resident_mapping: false,
            virtual_memory: false,
            determinism: false,
        }
    }
}

/// Live-entity and metadata high-water budgets (not a byte-memory estimate).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EntityBudget {
    /// Maximum live count admitted through this policy, including reservations.
    pub max_live: u32,
    /// Maximum metadata slot high-water mark, including reusable dead slots.
    pub max_slots: u32,
}

impl Default for EntityBudget {
    fn default() -> Self {
        Self {
            max_live: u32::MAX,
            max_slots: u32::MAX,
        }
    }
}

/// Chunk work granularity, independent of the immutable physical row layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChunkPolicy {
    /// Required payload target, currently exactly [`TARGET_CHUNK_BYTES`].
    ///
    /// This is not an allocation-size cap: headers, alignment, and oversized
    /// component rows can make a physical block larger than the payload target.
    pub target_bytes: usize,
    /// Maximum complete physical chunks in one policy work batch.
    pub chunks_per_batch: usize,
}

impl Default for ChunkPolicy {
    fn default() -> Self {
        Self {
            target_bytes: TARGET_CHUNK_BYTES,
            chunks_per_batch: 1,
        }
    }
}

/// One project-selected LOD band, usable even when `partition` is not compiled.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LodBandProfile {
    /// Inclusive squared distance in world units squared; finite and nonnegative.
    pub max_distance_sq: f32,
    /// Update once every this many frames; must be nonzero.
    pub tick_period: u32,
    /// Project-defined precision tier (zero is highest precision).
    pub quality: u8,
}

/// Distance/cadence policy converted into the existing entity LOD processor.
#[derive(Clone, Debug, PartialEq)]
pub struct LodProfile {
    /// One to 256 bands, in strictly increasing squared-distance order.
    pub bands: Vec<LodBandProfile>,
    /// Dormancy beyond the final band; otherwise clamp to the final band.
    pub dormant_beyond_last: bool,
}

/// Streaming-grid policy converted into the existing world partition driver.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StreamingProfile {
    /// Positive finite cell edge length in world units.
    pub cell_size: f64,
    /// Chebyshev load radius in cells.
    pub load_radius: u32,
    /// Chebyshev retention radius, at least the load radius.
    pub unload_radius: u32,
    /// Maximum cells enumerated by one interest's load cube, `(2r + 1)^3`.
    ///
    /// This bounds per-interest enumeration, not the union of arbitrarily many
    /// interests, in-flight requests, resident entities, or I/O bytes.
    pub max_cells_per_interest: u64,
}

/// Editable application profile; resolution validates every field atomically.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct EcsProfile {
    /// Content label only; all numeric values remain project-controlled.
    pub tier: QualityTier,
    /// Explicit capability/feature requirements.
    pub requirements: EcsRequirements,
    /// Entity admission limits.
    pub entities: EntityBudget,
    /// Physical-layout-compatible work batching.
    pub chunks: ChunkPolicy,
    /// Optional project worker ceiling; `None` uses the host's reported count.
    ///
    /// This is an output for the pool owner, not a mutation of an existing pool.
    pub max_workers: Option<usize>,
    /// Optional LOD policy; requesting it requires `partition`.
    pub lod: Option<LodProfile>,
    /// Optional streaming policy; requesting it requires `partition`.
    pub streaming: Option<StreamingProfile>,
}

/// Optional facility whose compile-time gate a profile can require.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EcsFeature {
    /// Parallel scheduling.
    MultiThread,
    /// SIMD column kernels.
    Simd,
    /// Partition streaming and entity LOD.
    Partition,
    /// GPU-resident column support (not a GPU device itself).
    GpuResident,
    /// VM-backed chunk arenas.
    VirtualMemory,
    /// Determinism diagnostics.
    Determinism,
}

impl EcsFeature {
    /// Whether this facility is compiled into the current crate.
    pub const fn is_compiled(self) -> bool {
        match self {
            Self::MultiThread => cfg!(feature = "multi_thread"),
            Self::Simd => cfg!(feature = "simd"),
            Self::Partition => cfg!(feature = "partition"),
            Self::GpuResident => cfg!(feature = "gpu_resident"),
            Self::VirtualMemory => cfg!(feature = "vm"),
            Self::Determinism => cfg!(feature = "determinism"),
        }
    }
}

/// Compile-time arena strategy; never a claim about a particular allocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArenaStrategy {
    /// A global allocation per physical chunk.
    GlobalBlocks,
    /// VM segments where the platform backend exists, global blocks otherwise.
    ///
    /// Huge pages are opportunistic; no explicit NUMA node binding is exposed.
    VirtualMemoryWithGlobalFallback,
}

/// Why a profile cannot be resolved without changing its meaning.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PolicyError {
    /// Zero host workers, zero worker limit/minimum, or ceiling below minimum.
    InvalidWorkerCount,
    /// Live-count limit exceeds metadata slot limit.
    InvalidEntityBudget,
    /// Physical chunk retuning is not supported by the current storage layout.
    UnsupportedChunkSize,
    /// A batch must contain at least one chunk.
    EmptyChunkBatch,
    /// A requested optional facility was not compiled in.
    FeatureUnavailable(EcsFeature),
    /// Host-reported capabilities cannot satisfy the requested worker minimum.
    InsufficientWorkers,
    /// Host-reported SIMD width is below the requested minimum.
    InsufficientSimd,
    /// The host did not report resident mapping support.
    GpuMappingUnavailable,
    /// The host did not report virtual-memory support.
    VirtualMemoryUnavailable,
    /// LOD bands are empty/too numerous, non-finite, negative, unordered, or zero-rate.
    InvalidLodBands,
    /// Streaming grid size, radii, or per-interest enumeration budget is invalid.
    InvalidStreaming,
}

impl fmt::Display for PolicyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid ECS policy: {self:?}")
    }
}

impl core::error::Error for PolicyError {}

/// Capacity dimension that rejected an admission request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EntityBudgetKind {
    /// Live count, including the proposed new entities.
    Live,
    /// Metadata slot high-water mark after free-slot reuse.
    Slots,
}

/// Failure from an explicit entity admission check.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdmissionError {
    /// Reservations must be materialized/cancelled at the owning sync point first.
    OutstandingReservations,
    /// Supplied counts are inconsistent (`live > slots`).
    InvalidCounts,
    /// An existing or proposed count exceeds the configured limit.
    Exceeded {
        /// Capacity dimension checked.
        kind: EntityBudgetKind,
        /// Required count, using a wider integer to avoid wrapping.
        requested: u64,
        /// Configured maximum.
        limit: u32,
    },
}

impl fmt::Display for AdmissionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ECS entity admission rejected: {self:?}")
    }
}

impl core::error::Error for AdmissionError {}

impl EntityBudget {
    /// Checks a proposed allocation using live and metadata-slot counts.
    ///
    /// Counts must describe one coherent allocator state. Reusable slots are
    /// consumed before growing metadata, matching the core entity allocator.
    /// This does not allocate or reserve anything: concurrent producers must
    /// serialize admission or account for outstanding admissions in both counts.
    pub fn check(&self, live: u32, slots: u32, additional: u32) -> Result<(), AdmissionError> {
        if live > slots {
            return Err(AdmissionError::InvalidCounts);
        }
        let requested = u64::from(live) + u64::from(additional);
        if requested > u64::from(self.max_live) {
            return Err(AdmissionError::Exceeded {
                kind: EntityBudgetKind::Live,
                requested,
                limit: self.max_live,
            });
        }
        let requested = requested.max(u64::from(slots));
        if requested > u64::from(self.max_slots) {
            return Err(AdmissionError::Exceeded {
                kind: EntityBudgetKind::Slots,
                requested,
                limit: self.max_slots,
            });
        }
        Ok(())
    }
}

/// Validated immutable scale decisions, shareable as a world resource.
#[derive(Clone, Debug, PartialEq)]
pub struct EcsPolicy {
    profile: EcsProfile,
    capabilities: EcsCapabilities,
    workers: usize,
}

impl crate::resource::Resource for EcsPolicy {}

impl EcsPolicy {
    /// Validates a project profile against explicit capabilities and Cargo gates.
    ///
    /// No world, arena, scheduler, GPU device, or external state is modified.
    pub fn resolve(
        profile: EcsProfile,
        capabilities: EcsCapabilities,
    ) -> Result<Self, PolicyError> {
        let requirements = profile.requirements;
        if capabilities.worker_threads == 0
            || requirements.min_workers == 0
            || profile.max_workers == Some(0)
            || profile
                .max_workers
                .is_some_and(|max| max < requirements.min_workers)
        {
            return Err(PolicyError::InvalidWorkerCount);
        }
        if profile.entities.max_live > profile.entities.max_slots {
            return Err(PolicyError::InvalidEntityBudget);
        }
        if profile.chunks.target_bytes != TARGET_CHUNK_BYTES {
            return Err(PolicyError::UnsupportedChunkSize);
        }
        if profile.chunks.chunks_per_batch == 0 {
            return Err(PolicyError::EmptyChunkBatch);
        }
        let require = |feature: EcsFeature| {
            if feature.is_compiled() {
                Ok(())
            } else {
                Err(PolicyError::FeatureUnavailable(feature))
            }
        };
        if requirements.min_workers > 1 {
            require(EcsFeature::MultiThread)?;
        }
        let workers = if EcsFeature::MultiThread.is_compiled() {
            capabilities.worker_threads
        } else {
            1
        }
        .min(profile.max_workers.unwrap_or(usize::MAX));
        if workers < requirements.min_workers {
            return Err(PolicyError::InsufficientWorkers);
        }
        if requirements.min_simd_width_bits > 0 {
            require(EcsFeature::Simd)?;
            if capabilities.simd_width_bits < requirements.min_simd_width_bits {
                return Err(PolicyError::InsufficientSimd);
            }
        }
        if requirements.gpu_resident_mapping {
            require(EcsFeature::GpuResident)?;
            if !capabilities.gpu_resident_mapping {
                return Err(PolicyError::GpuMappingUnavailable);
            }
        }
        if requirements.virtual_memory {
            require(EcsFeature::VirtualMemory)?;
            if !capabilities.virtual_memory {
                return Err(PolicyError::VirtualMemoryUnavailable);
            }
        }
        if requirements.determinism {
            require(EcsFeature::Determinism)?;
        }
        if let Some(lod) = &profile.lod {
            let mut previous = None;
            if lod.bands.is_empty() || lod.bands.len() > 256 {
                return Err(PolicyError::InvalidLodBands);
            }
            for band in &lod.bands {
                if !band.max_distance_sq.is_finite()
                    || band.max_distance_sq < 0.0
                    || band.tick_period == 0
                    || previous.is_some_and(|p| band.max_distance_sq <= p)
                {
                    return Err(PolicyError::InvalidLodBands);
                }
                previous = Some(band.max_distance_sq);
            }
            require(EcsFeature::Partition)?;
        }
        if let Some(streaming) = profile.streaming {
            let diameter = u64::from(streaming.load_radius) * 2 + 1;
            let cells = diameter.checked_pow(3);
            if !streaming.cell_size.is_finite()
                || streaming.cell_size <= 0.0
                || streaming.unload_radius < streaming.load_radius
                || streaming.unload_radius > i32::MAX as u32
                || cells.is_none_or(|count| count > streaming.max_cells_per_interest)
            {
                return Err(PolicyError::InvalidStreaming);
            }
            require(EcsFeature::Partition)?;
        }
        Ok(Self {
            profile,
            capabilities,
            workers,
        })
    }

    /// The validated profile; cloning/editing it does not mutate this policy.
    pub fn profile(&self) -> &EcsProfile {
        &self.profile
    }

    /// The original host report, not an implicit platform probe.
    pub fn capabilities(&self) -> EcsCapabilities {
        self.capabilities
    }

    /// Total ECS thread ceiling, including the calling thread.
    ///
    /// A pool counting only background workers must reserve one for the caller:
    /// use `worker_limit() - 1`, or [`Self::task_pool_config`] with `multi_thread`.
    /// The owner must apply this ceiling; resolving a policy never resizes a pool.
    pub fn worker_limit(&self) -> usize {
        self.workers
    }

    /// Adapts a task-pool configuration to the resolved total thread budget.
    ///
    /// `prism_tasks` counts background threads, while this policy includes the
    /// caller that helps execute work. A total budget of one therefore selects
    /// the pool's zero-background-thread synchronous fallback. Explicit affinity
    /// preferences are preserved; no topology detection or thread creation runs
    /// here. Construct the ECS pool with the returned configuration, and avoid
    /// sharing its budget with independently running executors.
    #[cfg(feature = "multi_thread")]
    pub fn task_pool_config(
        &self,
        mut config: prism_tasks::TaskPoolConfig,
    ) -> prism_tasks::TaskPoolConfig {
        config.threads = self.workers - 1;
        config
    }

    /// Actual compiled arena selection rule, independent of the quality label.
    pub const fn arena_strategy(&self) -> ArenaStrategy {
        if cfg!(feature = "vm") {
            ArenaStrategy::VirtualMemoryWithGlobalFallback
        } else {
            ArenaStrategy::GlobalBlocks
        }
    }

    /// Checks capacity at a synchronization point without mutating the world.
    ///
    /// Outstanding reservations are rejected rather than undercounted. This is
    /// not a global `World` limit: the owner must route spawn batches through
    /// admission and account for hook/observer/command-generated entities too.
    /// No permit is retained and no promise about byte-memory use is made.
    pub fn check_world_admission(
        &self,
        world: &World,
        additional: u32,
    ) -> Result<(), AdmissionError> {
        if world.entities().has_outstanding_reservations() {
            return Err(AdmissionError::OutstandingReservations);
        }
        self.profile.entities.check(
            world.entity_count(),
            world.entities().capacity(),
            additional,
        )
    }

    /// Partitions an actual table into complete-chunk work batches.
    ///
    /// Uses `Table::rows_per_chunk`, not a guessed row-byte calculation. The
    /// shared borrow keeps the table structurally stable during enumeration.
    pub fn chunk_batches<'a>(&self, table: &'a Table) -> ChunkBatches<'a> {
        ChunkBatches {
            table,
            next_chunk: 0,
            chunks_per_batch: self.profile.chunks.chunks_per_batch,
        }
    }

    /// Constructs the real LOD schedule, or `None` when LOD is not requested.
    #[cfg(feature = "partition")]
    pub fn lod_schedule(&self) -> Option<crate::partition::lod::LodSchedule> {
        use crate::partition::lod::{LodBand, LodQuality, LodSchedule, OutOfRange};
        self.profile.lod.as_ref().map(|lod| {
            let bands = lod
                .bands
                .iter()
                .map(|b| {
                    LodBand::with_quality(b.max_distance_sq, b.tick_period, LodQuality(b.quality))
                })
                .collect();
            LodSchedule::with_out_of_range(
                bands,
                if lod.dormant_beyond_last {
                    OutOfRange::Dormant
                } else {
                    OutOfRange::ClampToLast
                },
            )
        })
    }

    /// Constructs the existing entity processor with the requested phase policy.
    #[cfg(feature = "partition")]
    pub fn lod_processor(
        &self,
        phase: crate::partition::processor::PhasePolicy,
    ) -> Option<crate::partition::processor::EntityLodProcessor> {
        self.lod_schedule().map(|schedule| {
            crate::partition::processor::EntityLodProcessor::new(schedule).with_phase_policy(phase)
        })
    }

    /// Constructs a fresh real streaming subsystem with a matching origin/grid.
    ///
    /// This does not replace an existing partition (which would lose residency
    /// state), perform I/O, or bypass the partition's coordinate-range contract.
    #[cfg(feature = "partition")]
    pub fn world_partition(&self) -> Option<crate::partition::world_partition::WorldPartition> {
        self.profile.streaming.map(|streaming| {
            crate::partition::world_partition::WorldPartition::new(
                streaming.cell_size,
                streaming.load_radius,
                streaming.unload_radius,
            )
        })
    }
}

/// One batch containing whole physical chunks and their live row range.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChunkBatch {
    /// Half-open physical chunk-index range.
    pub chunks: Range<usize>,
    /// Half-open row range, clipped to the final chunk's live rows.
    pub rows: Range<usize>,
}

/// Allocation-free, table-borrowing complete-chunk work planner.
#[derive(Clone)]
pub struct ChunkBatches<'a> {
    table: &'a Table,
    next_chunk: usize,
    chunks_per_batch: usize,
}

impl fmt::Debug for ChunkBatches<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ChunkBatches")
            .field("chunk_count", &self.table.chunk_count())
            .field("next_chunk", &self.next_chunk)
            .field("chunks_per_batch", &self.chunks_per_batch)
            .finish()
    }
}

impl Iterator for ChunkBatches<'_> {
    type Item = ChunkBatch;

    fn next(&mut self) -> Option<Self::Item> {
        let remaining = self.table.chunk_count() - self.next_chunk;
        if remaining == 0 {
            return None;
        }
        let start = self.next_chunk;
        let end = start + remaining.min(self.chunks_per_batch);
        self.next_chunk = end;
        let rows_per_chunk = self.table.rows_per_chunk();
        let end_row = if end == self.table.chunk_count() {
            self.table.len()
        } else {
            end * rows_per_chunk
        };
        Some(ChunkBatch {
            chunks: start..end,
            rows: start * rows_per_chunk..end_row,
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining =
            (self.table.chunk_count() - self.next_chunk).div_ceil(self.chunks_per_batch);
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for ChunkBatches<'_> {}
impl FusedIterator for ChunkBatches<'_> {}

#[cfg(test)]
mod tests;
