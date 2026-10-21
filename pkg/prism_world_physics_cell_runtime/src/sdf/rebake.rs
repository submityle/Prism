//! Bounded background SDF work tied to the cell host's publication lifecycle.

use alloc::{boxed::Box, sync::Arc, vec::Vec};
use core::{convert::Infallible, fmt};
use prism_tasks::{CancelToken, TaskPool};
use prism_world_physics_bridge::{
    SdfRebakeError, SdfRebakeLimits, SdfRebakePoint, SdfRebakeRegion, SdfRebakeReport,
    SdfRebakeTask, SdfRebakeTaskError,
};

use super::{
    CellManifest, SceneHost, SdfCellDecoder, SdfPhysicsCellHost, SdfPhysicsCellHostError, SectorId,
};

/// Aggregate bounds for this runtime's uncollected work, including ready results.
///
/// Scratch covers replacement samples and descriptors, not callback captures,
/// task-pool queues, tokens, allocator overhead or the preallocated task index.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SdfCellRebakeBudget {
    /// Maximum uncollected tasks. Zero disables admission.
    pub max_tasks: usize,
    /// Maximum samples across all uncollected tasks.
    pub max_samples: usize,
    /// Maximum logical scratch bytes across all uncollected tasks.
    pub max_scratch_bytes: usize,
}

/// Resource bounded at task admission or before resident publication.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SdfCellRebakeResource {
    /// Uncollected tasks.
    Tasks,
    /// Resident bricks across all cells.
    ResidentBricks,
    /// Resident samples across all cells.
    ResidentSamples,
    /// Resident logical bytes across all cells.
    ResidentBytes,
}

/// Opaque identity scoped to one runtime; collected identities are never reused.
#[derive(Clone, Debug)]
pub struct SdfCellRebakeId {
    owner: Arc<()>,
    serial: u64,
}

impl PartialEq for SdfCellRebakeId {
    fn eq(&self, other: &Self) -> bool {
        self.serial == other.serial && Arc::ptr_eq(&self.owner, &other.owner)
    }
}
impl Eq for SdfCellRebakeId {}

/// Admission, lifecycle, accounting or evaluator failure.
#[derive(Debug)]
pub enum SdfCellRebakeError<E> {
    /// Foreign, unknown or already collected task identity.
    UnknownTask,
    /// The requested cell has no committed SDF.
    MissingCell {
        /// Requested sector.
        sector: SectorId,
    },
    /// A successful reload, eviction or sibling commit superseded this work.
    Invalidated {
        /// Sector whose publication changed.
        sector: SectorId,
    },
    /// A runtime task or aggregate residency limit would be exceeded.
    BudgetExceeded {
        /// Resource whose limit was exceeded.
        resource: SdfCellRebakeResource,
        /// Requested count or logical bytes.
        required: usize,
        /// Configured maximum.
        limit: usize,
    },
    /// Checked aggregate counters overflowed or an internal resident was missing.
    AccountingOverflow,
    /// The bounded pending-task index could not be reserved.
    AllocationFailed,
    /// All identities in this runtime's u64 sequence have been issued.
    IdExhausted,
    /// Per-work preparation failed; limits include remaining aggregate capacity.
    Prepare(SdfRebakeError<Infallible>),
    /// Evaluator, cancellation, panic or bridge publication failed.
    Task(SdfRebakeTaskError<E>),
}

impl<E: fmt::Display> fmt::Display for SdfCellRebakeError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownTask => f.write_str("unknown or collected SDF cell task"),
            Self::MissingCell { sector } => write!(f, "no resident SDF for {sector:?}"),
            Self::Invalidated { sector } => write!(f, "SDF cell task superseded for {sector:?}"),
            Self::BudgetExceeded {
                resource,
                required,
                limit,
            } => {
                write!(
                    f,
                    "SDF cell rebake needs {required} {resource:?}; limit is {limit}"
                )
            }
            Self::AccountingOverflow => f.write_str("SDF cell rebake accounting overflow"),
            Self::AllocationFailed => f.write_str("SDF cell task index allocation failed"),
            Self::IdExhausted => f.write_str("SDF cell task identities exhausted"),
            Self::Prepare(error) => error.fmt(f),
            Self::Task(error) => error.fmt(f),
        }
    }
}

impl<E: core::error::Error + 'static> core::error::Error for SdfCellRebakeError<E> {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Prepare(error) => Some(error),
            Self::Task(error) => Some(error),
            _ => None,
        }
    }
}

struct Pending<'pool, E> {
    id: SdfCellRebakeId,
    sector: SectorId,
    report: SdfRebakeReport,
    task: SdfRebakeTask<'pool, E>,
    invalidated: bool,
}

/// Cell lifecycle plus bounded work on a caller-owned shared pool.
///
/// Queries through [`Self::physics`] see the old snapshot until explicit commit.
/// Reload/evict failures retain running work; successful transitions cancel it.
/// Even cancelled or superseded jobs remain charged until terminal collection.
/// Polling never waits or evaluates source callbacks. Zero-worker and single
/// pools evaluate inline during start, following the task pool's contract.
///
/// Drop requests cancellation without waiting or publishing. A running callback
/// can outlive the runtime: these are per-instance budgets, not global limits
/// across dropped/recreated runtimes. Cancel and collect all work before orderly
/// replacement. The external pool must outlive this runtime.
///
/// ```
/// use core::convert::Infallible;
/// use prism_tasks::{CancelToken, TaskPool};
/// use prism_world_physics_bridge::{SdfPhysicsBridgeBudget, SdfRebakeLimits,
///     SdfRebakeRegion, WorldSdfSnapshot, WorldSdfTransform};
/// use prism_world_physics_cell_runtime::{SdfCellCodec, SdfCellCodecLimits,
///     SdfCellRebakeBudget, SdfCellRebakeRuntime, SdfCellSnapshot,
///     SdfPhysicsCellHost, SdfResidencyBudget};
/// use prism_world_residency::SparseSdfBrickGrid;
/// use prism_world_runtime::SceneHost;
/// use prism_world_stream::{AabbI64, CellManifest, DataLayerMask, GlobalPos, SectorId};
/// struct Scene;
/// impl SceneHost for Scene {
///     type Error = Infallible;
///     fn load_cell(&mut self, _: &CellManifest, _: &[u8]) -> Result<(), Self::Error> { Ok(()) }
///     fn evict_cell(&mut self, _: SectorId) -> Result<(), Self::Error> { Ok(()) }
/// }
/// let pool = TaskPool::with_threads(0); // Deterministic inline example.
/// let sector = SectorId::new(0, 0, 0, 0);
/// let manifest = CellManifest {
///     sector,
///     bounds: AabbI64::new(GlobalPos::new(0, 0, 0), GlobalPos::new(200, 200, 200)).unwrap(),
///     scene_key: 1, content_version: 1, derived_hash: 1, io_bytes: 0,
///     data_layers: DataLayerMask::EMPTY,
/// };
/// let codec = SdfCellCodec::new(SdfCellCodecLimits::new(4096, 1, 8, 80));
/// let payload = codec.encode(&SdfCellSnapshot::new(sector, WorldSdfSnapshot::new(
///     WorldSdfTransform::new([0; 3], 100, 10).unwrap(), 1,
///     SparseSdfBrickGrid::new(2, 1).unwrap(),
/// ))).unwrap();
/// let mut scene = Scene;
/// let mut physics = SdfPhysicsCellHost::new(&mut scene, codec,
///     SdfPhysicsBridgeBudget::new(1, 8, 80).unwrap(),
///     SdfResidencyBudget::new(1, 1, 8, 80));
/// physics.load_cell(&manifest, &payload).unwrap();
/// let mut runtime = SdfCellRebakeRuntime::<_, _, Infallible>::new(physics, &pool,
///     SdfCellRebakeBudget { max_tasks: 1, max_samples: 8, max_scratch_bytes: 4096 }
/// ).unwrap();
/// let id = runtime.start(sector, 2,
///     SdfRebakeRegion { min_voxel: [0; 3], max_voxel: [0; 3], mip: 0 },
///     SdfRebakeLimits { max_bricks: 1, max_samples: 8, max_scratch_bytes: 4096 },
///     &CancelToken::new(), |_| Ok(7),
/// ).unwrap();
/// assert_eq!(runtime.physics().resident(sector).unwrap().revision(), 1);
/// assert_eq!(runtime.pending_usage().0, 1);
/// assert_eq!(runtime.try_commit(&id).unwrap().unwrap().revision, 2);
/// assert_eq!(runtime.pending_usage(), (0, 0, 0));
/// assert_eq!(runtime.physics().resident(sector).unwrap().revision(), 2);
/// ```
pub struct SdfCellRebakeRuntime<'pool, 'host, H: SceneHost, D: SdfCellDecoder, E> {
    physics: SdfPhysicsCellHost<'host, H, D>,
    pool: &'pool TaskPool,
    budget: SdfCellRebakeBudget,
    pending: Vec<Pending<'pool, E>>,
    pending_samples: usize,
    pending_scratch_bytes: usize,
    identity: Arc<()>,
    next_serial: u64,
}

impl<'pool, 'host, H: SceneHost, D: SdfCellDecoder, E: Send + 'static>
    SdfCellRebakeRuntime<'pool, 'host, H, D, E>
{
    /// Reserves the complete bounded task index without creating a pool.
    /// Preparation/spawning allocate and should run outside frame hot paths.
    pub fn new(
        physics: SdfPhysicsCellHost<'host, H, D>,
        pool: &'pool TaskPool,
        budget: SdfCellRebakeBudget,
    ) -> Result<Self, SdfCellRebakeError<E>> {
        let mut pending = Vec::new();
        pending
            .try_reserve_exact(budget.max_tasks)
            .map_err(|_| SdfCellRebakeError::AllocationFailed)?;
        Ok(Self {
            physics,
            pool,
            budget,
            pending,
            pending_samples: 0,
            pending_scratch_bytes: 0,
            identity: Arc::new(()),
            next_serial: 0,
        })
    }

    /// Read-only access to committed queries, revisions and aggregate residency.
    #[must_use]
    pub const fn physics(&self) -> &SdfPhysicsCellHost<'host, H, D> {
        &self.physics
    }

    /// Updates the decoder used by future cell loads without changing residents.
    pub fn decoder_mut(&mut self) -> &mut D {
        self.physics.decoder_mut()
    }

    /// Returns uncollected `(tasks, samples, logical scratch bytes)`.
    #[must_use]
    pub fn pending_usage(&self) -> (usize, usize, usize) {
        (
            self.pending.len(),
            self.pending_samples,
            self.pending_scratch_bytes,
        )
    }

    /// Starts a complete composed-source revision for one resident cell.
    ///
    /// Per-work sample/scratch limits are capped by remaining aggregate capacity
    /// before prepare allocates buffers. Resident growth is checked here and
    /// again at publication, since other cells can consume that capacity.
    /// Multiple works may share a base; the first successful commit invalidates
    /// its same-cell siblings, including disjoint edits. No automatic merge is
    /// performed. The source must own a coherent field including halo and parents.
    pub fn start(
        &mut self,
        sector: SectorId,
        revision: u64,
        region: SdfRebakeRegion,
        mut limits: SdfRebakeLimits,
        scope: &CancelToken,
        source: impl FnMut(SdfRebakePoint) -> Result<i16, E> + Send + 'static,
    ) -> Result<SdfCellRebakeId, SdfCellRebakeError<E>> {
        check_limit(
            SdfCellRebakeResource::Tasks,
            self.pending
                .len()
                .checked_add(1)
                .ok_or(SdfCellRebakeError::AccountingOverflow)?,
            self.budget.max_tasks,
        )?;
        let next_serial = self
            .next_serial
            .checked_add(1)
            .ok_or(SdfCellRebakeError::IdExhausted)?;
        limits.max_samples = limits.max_samples.min(
            self.budget
                .max_samples
                .checked_sub(self.pending_samples)
                .ok_or(SdfCellRebakeError::AccountingOverflow)?,
        );
        limits.max_scratch_bytes = limits.max_scratch_bytes.min(
            self.budget
                .max_scratch_bytes
                .checked_sub(self.pending_scratch_bytes)
                .ok_or(SdfCellRebakeError::AccountingOverflow)?,
        );
        let index = self.resident_index(sector)?;
        let work = self.physics.residents[index]
            .1
            .prepare_rebake(region, revision, limits)
            .map_err(SdfCellRebakeError::Prepare)?;
        let report = work.report();
        self.next_resident_usage(sector, report)?;
        let samples = self
            .pending_samples
            .checked_add(report.sampled_voxels)
            .ok_or(SdfCellRebakeError::AccountingOverflow)?;
        let scratch = self
            .pending_scratch_bytes
            .checked_add(report.scratch_bytes)
            .ok_or(SdfCellRebakeError::AccountingOverflow)?;
        let id = SdfCellRebakeId {
            owner: Arc::clone(&self.identity),
            serial: self.next_serial,
        };
        let task = work.spawn(self.pool, scope, source);
        self.pending.push(Pending {
            id: id.clone(),
            sector,
            report,
            task,
            invalidated: false,
        });
        self.next_serial = next_serial;
        self.pending_samples = samples;
        self.pending_scratch_bytes = scratch;
        Ok(id)
    }

    /// Tests completion without helping execute queued work or collecting it.
    pub fn is_finished(&self, id: &SdfCellRebakeId) -> Result<bool, SdfCellRebakeError<E>> {
        Ok(self.pending[self.task_index(id)?].task.is_finished())
    }

    /// Requests per-task cancellation; the budget stays charged until collected.
    pub fn cancel(&self, id: &SdfCellRebakeId) -> Result<(), SdfCellRebakeError<E>> {
        self.pending[self.task_index(id)?].task.cancel();
        Ok(())
    }

    /// Requests cancellation of all current tasks without waiting or collecting.
    pub fn cancel_all(&self) {
        for pending in &self.pending {
            pending.task.cancel();
        }
    }

    /// Collects a terminal result once, checking aggregate residency before write.
    ///
    /// Unfinished work returns `Ok(None)` and retains its charge. Terminal errors
    /// release the charge and retain all resident data. Captured source errors
    /// and panics remain observable even for invalidated or budget-rejected work;
    /// cancellation of invalidated work is reported as `Invalidated`. Successful
    /// publication changes derived physics only, without reloading scene content.
    pub fn try_commit(
        &mut self,
        id: &SdfCellRebakeId,
    ) -> Result<Option<SdfRebakeReport>, SdfCellRebakeError<E>> {
        let index = self.task_index(id)?;
        if !self.pending[index].task.is_finished() {
            return Ok(None);
        }
        let mut pending = self.pending.swap_remove(index);
        let report = pending.report;
        let result = self.collect_terminal(&mut pending);
        drop(pending);
        // Release only after collection and destruction, including every error
        // path. Admission cannot interleave with this exclusive owner operation.
        self.pending_samples -= report.sampled_voxels;
        self.pending_scratch_bytes -= report.scratch_bytes;
        result
    }

    fn collect_terminal(
        &mut self,
        pending: &mut Pending<'pool, E>,
    ) -> Result<Option<SdfRebakeReport>, SdfCellRebakeError<E>> {
        if pending.invalidated {
            match pending.task.try_discard() {
                Err(SdfRebakeTaskError::Rebake(SdfRebakeError::Cancelled)) | Ok(Some(())) => {}
                Err(error) => return Err(SdfCellRebakeError::Task(error)),
                Ok(None) => return Err(SdfCellRebakeError::AccountingOverflow),
            }
            return Err(SdfCellRebakeError::Invalidated {
                sector: pending.sector,
            });
        }
        let usage = match self.next_resident_usage(pending.sector, pending.report) {
            Ok(usage) => usage,
            Err(error) => {
                let discarded = pending
                    .task
                    .try_discard()
                    .map_err(SdfCellRebakeError::Task)?;
                if discarded.is_none() {
                    return Err(SdfCellRebakeError::AccountingOverflow);
                }
                return Err(error);
            }
        };
        let resident = self.resident_index(pending.sector)?;
        let report = pending
            .task
            .try_commit(&mut self.physics.residents[resident].1)
            .map_err(SdfCellRebakeError::Task)?;
        if report.is_none() {
            return Err(SdfCellRebakeError::AccountingOverflow);
        }
        (
            self.physics.resident_bricks,
            self.physics.resident_samples,
            self.physics.resident_bytes,
        ) = usage;
        self.invalidate(pending.sector);
        Ok(report)
    }

    /// Recovers the original host only after every task has been collected.
    /// On failure boxes the unchanged runtime so callers can cancel and drain.
    /// That failure path allocates; draining first keeps extraction allocation-free.
    pub fn try_into_host(self) -> Result<SdfPhysicsCellHost<'host, H, D>, Box<Self>> {
        if self.pending.is_empty() {
            Ok(self.physics)
        } else {
            Err(Box::new(self))
        }
    }

    fn task_index(&self, id: &SdfCellRebakeId) -> Result<usize, SdfCellRebakeError<E>> {
        if !Arc::ptr_eq(&id.owner, &self.identity) {
            return Err(SdfCellRebakeError::UnknownTask);
        }
        self.pending
            .iter()
            .position(|pending| pending.id == *id)
            .ok_or(SdfCellRebakeError::UnknownTask)
    }

    fn resident_index(&self, sector: SectorId) -> Result<usize, SdfCellRebakeError<E>> {
        self.physics
            .residents
            .binary_search_by_key(&sector, |(key, _)| *key)
            .map_err(|_| SdfCellRebakeError::MissingCell { sector })
    }

    fn next_resident_usage(
        &self,
        sector: SectorId,
        report: SdfRebakeReport,
    ) -> Result<(usize, usize, usize), SdfCellRebakeError<E>> {
        let old = self
            .physics
            .resident(sector)
            .ok_or(SdfCellRebakeError::MissingCell { sector })?;
        if old.revision() != report.previous_revision {
            return Err(SdfCellRebakeError::Invalidated { sector });
        }
        let bricks = self
            .physics
            .resident_bricks
            .checked_add(report.inserted_bricks)
            .ok_or(SdfCellRebakeError::AccountingOverflow)?;
        let samples = report
            .inserted_bricks
            .checked_mul(old.grid().sample_count())
            .and_then(|growth| self.physics.resident_samples.checked_add(growth))
            .ok_or(SdfCellRebakeError::AccountingOverflow)?;
        let bytes = self
            .physics
            .resident_bytes
            .checked_sub(old.logical_bytes())
            .and_then(|base| base.checked_add(report.logical_bytes))
            .ok_or(SdfCellRebakeError::AccountingOverflow)?;
        let budget = self.physics.residency_budget;
        check_limit(
            SdfCellRebakeResource::ResidentBricks,
            bricks,
            budget.max_bricks,
        )?;
        check_limit(
            SdfCellRebakeResource::ResidentSamples,
            samples,
            budget.max_samples,
        )?;
        check_limit(
            SdfCellRebakeResource::ResidentBytes,
            bytes,
            budget.max_bytes,
        )?;
        Ok((bricks, samples, bytes))
    }

    fn invalidate(&mut self, sector: SectorId) {
        for pending in &mut self.pending {
            if pending.sector == sector {
                pending.invalidated = true;
                pending.task.cancel();
            }
        }
    }
}

fn check_limit<E>(
    resource: SdfCellRebakeResource,
    required: usize,
    limit: usize,
) -> Result<(), SdfCellRebakeError<E>> {
    if required > limit {
        Err(SdfCellRebakeError::BudgetExceeded {
            resource,
            required,
            limit,
        })
    } else {
        Ok(())
    }
}

impl<H: SceneHost, D: SdfCellDecoder, E: Send + 'static> SceneHost
    for SdfCellRebakeRuntime<'_, '_, H, D, E>
{
    type Error = SdfPhysicsCellHostError<D::Error, H::Error>;

    fn load_cell(&mut self, manifest: &CellManifest, payload: &[u8]) -> Result<(), Self::Error> {
        self.physics.load_cell(manifest, payload)?;
        self.invalidate(manifest.sector);
        Ok(())
    }

    fn evict_cell(&mut self, sector: SectorId) -> Result<(), Self::Error> {
        self.physics.evict_cell(sector)?;
        self.invalidate(sector);
        Ok(())
    }
}
