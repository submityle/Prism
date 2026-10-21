//! Transactional object composition paired with a world-mm driven GPU ring.

use alloc::vec::Vec;
use core::{fmt, mem::size_of};
use prism_world_gpu::{GpuContext, GpuSdfQuery};
use prism_world_residency::BrickCoord;

use crate::{
    WorldSdfClipmapConfig, WorldSdfClipmapUploadStats, WorldSdfFocusConfig, WorldSdfFocusUpdate,
    WorldSdfGpuClipmapError, WorldSdfGpuFocus, WorldSdfGpuFocusError, WorldSdfObject,
    WorldSdfObjectBake, WorldSdfObjectBakeBudget, WorldSdfObjectBakeReport, WorldSdfObjectConfig,
    WorldSdfObjectError, WorldSdfObjectLimits, WorldSdfObjectResource, WorldSdfObjectStore,
    WorldSdfObjectUpdate,
};

/// Fixed geometry and independent CPU/GPU admission limits.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WorldSdfGpuObjectsConfig {
    /// CPU field geometry and common signed-distance quantization.
    pub objects: WorldSdfObjectConfig,
    /// Persistent index and per-edit CPU resource bounds.
    pub limits: WorldSdfObjectLimits,
    /// World-mm mapping. Origin, spacing and brick edge must match `objects`.
    pub focus: WorldSdfFocusConfig,
    /// Ring geometry. Mip count must match `objects`.
    pub clipmap: WorldSdfClipmapConfig,
    /// Fixed GPU query capacity.
    pub max_queries: usize,
    /// Maximum physical sample/key/metadata bytes staged per upload call.
    pub max_upload_bytes: u64,
    /// Additional logical coordinate staging bytes per CPU-to-ring commit.
    /// Detached work and the scheduler's mark staging have separate budgets.
    pub max_commit_scratch_bytes: usize,
}

/// Returned failures preserve the attempted publication and retained GPU data.
#[derive(Debug, PartialEq)]
pub enum WorldSdfGpuObjectsError {
    /// CPU source/index/work/readiness failure.
    Objects(WorldSdfObjectError),
    /// World-mm mapping or focus/ring update failure.
    Focus(WorldSdfGpuFocusError),
    /// GPU dirty marking, upload, query or readiness failure.
    Upload(WorldSdfGpuClipmapError),
    /// CPU and GPU origin, spacing, brick edge or mip count differ.
    GeometryMismatch,
}

impl fmt::Display for WorldSdfGpuObjectsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Objects(error) => write!(f, "SDF objects: {error}"),
            Self::Focus(error) => write!(f, "SDF objects focus: {error}"),
            Self::Upload(error) => write!(f, "SDF objects GPU: {error}"),
            Self::GeometryMismatch => f.write_str("SDF object and GPU focus geometry differ"),
        }
    }
}

impl core::error::Error for WorldSdfGpuObjectsError {}

impl From<WorldSdfObjectError> for WorldSdfGpuObjectsError {
    fn from(value: WorldSdfObjectError) -> Self {
        Self::Objects(value)
    }
}

/// One owner for immutable object sources, derived CPU samples and GPU focus.
///
/// Edits first close the CPU readiness fence. A commit marks resident changed
/// coordinates before publishing CPU samples, so dirty-budget failure cannot
/// split the CPU and GPU revision. Off-window changes need no upload; entering
/// windows load the latest CPU grid. After the final bake, the GPU dirty fence
/// remains closed until every resident change is uploaded. There is no mutable
/// escape to the store or ring which could bypass these fences.
///
/// This owner does not impose a worker pool or observe ECS automatically. The
/// host calls upsert/remove and budgets detached work, just as it supplies the
/// focus. The ring retains the existing per-mip XZ/single-Y-slice geometry.
pub struct WorldSdfGpuObjects<T> {
    store: WorldSdfObjectStore<T>,
    focus: WorldSdfGpuFocus,
    max_commit_scratch_bytes: usize,
}

impl<T> WorldSdfGpuObjects<T> {
    /// Checks matching geometry before any GPU allocation.
    pub fn new(
        context: &GpuContext,
        config: WorldSdfGpuObjectsConfig,
    ) -> Result<Self, WorldSdfGpuObjectsError> {
        if config.objects.origin_mm != config.focus.origin_mm
            || config.objects.voxel_size_mm != config.focus.mip0_voxel_size_mm
            || config.objects.brick_size != config.focus.brick_size
            || config.objects.mip_levels != config.clipmap.clipmap.mip_levels
        {
            return Err(WorldSdfGpuObjectsError::GeometryMismatch);
        }
        let store = WorldSdfObjectStore::new(config.objects, config.limits)?;
        let focus = WorldSdfGpuFocus::new(
            context,
            config.focus,
            config.clipmap,
            config.max_queries,
            config.max_upload_bytes,
        )
        .map_err(WorldSdfGpuObjectsError::Focus)?;
        Ok(Self {
            store,
            focus,
            max_commit_scratch_bytes: config.max_commit_scratch_bytes,
        })
    }

    /// Read-only source statistics and grid; `grid()` may be stale while dirty.
    pub fn store(&self) -> &WorldSdfObjectStore<T> {
        &self.store
    }

    /// Read-only focus/ring scheduler, GPU bytes and pending upload diagnostics.
    pub fn focus(&self) -> &WorldSdfGpuFocus {
        &self.focus
    }

    /// Inserts or replaces an immutable source and closes query readiness.
    pub fn upsert(
        &mut self,
        id: u64,
        object: WorldSdfObject<T>,
    ) -> Result<(), WorldSdfGpuObjectsError> {
        self.store.upsert(id, object).map_err(Into::into)
    }

    /// Removes a source; surviving overlapping contributors are re-evaluated.
    pub fn remove(&mut self, id: u64) -> Result<bool, WorldSdfGpuObjectsError> {
        self.store.remove(id).map_err(Into::into)
    }

    /// Preallocates detached bounded CPU work without changing either owner.
    pub fn prepare(
        &self,
        budget: WorldSdfObjectBakeBudget,
    ) -> Result<WorldSdfObjectBake<T>, WorldSdfGpuObjectsError> {
        self.store.prepare(budget).map_err(Into::into)
    }

    /// Atomically admits GPU invalidations, then commits fully evaluated CPU
    /// work. Returned errors consume the update but preserve owner state; the
    /// same pending source region can be prepared again with a smaller batch.
    /// Successful nonempty commits fence all older detached jobs.
    pub fn commit(
        &mut self,
        update: WorldSdfObjectUpdate<T>,
    ) -> Result<WorldSdfObjectBakeReport, WorldSdfGpuObjectsError> {
        self.store.validate_update(&update)?;
        let scheduler = self.focus.ring().scheduler();
        let count = update
            .changed_bricks()
            .filter(|&coord| scheduler.slot_index(coord).is_some())
            .count();
        let required = count
            .checked_mul(size_of::<BrickCoord>())
            .ok_or(WorldSdfObjectError::ArithmeticOverflow)?;
        if required > self.max_commit_scratch_bytes {
            return Err(WorldSdfObjectError::BudgetExceeded {
                resource: WorldSdfObjectResource::ScratchBytes,
                required,
                limit: self.max_commit_scratch_bytes,
            }
            .into());
        }
        let mut coords = Vec::new();
        coords
            .try_reserve_exact(count)
            .map_err(|_| WorldSdfObjectError::AllocationFailed)?;
        coords.extend(
            update
                .changed_bricks()
                .filter(|&coord| scheduler.slot_index(coord).is_some()),
        );
        // The mark API rejects mixed off-window requests; filtering above is
        // intentional here because the grid remains the source for later entry.
        if !coords.is_empty() {
            self.focus
                .mark_dirty(&coords)
                .map_err(WorldSdfGpuObjectsError::Upload)?;
        }
        // No recoverable failure after dirty publication. CPU work was checked
        // first, and no edit/revision mutation can interleave through &mut self.
        Ok(self.store.commit_validated(update))
    }

    /// Rolls the window from the fully baked source. Pending CPU work is an
    /// error even for stationary focus; partial generations never become queryable.
    pub fn update_focus(
        &mut self,
        position_mm: [i64; 3],
    ) -> Result<WorldSdfFocusUpdate<WorldSdfClipmapUploadStats>, WorldSdfGpuObjectsError> {
        let grid = self.store.ready_grid()?;
        self.focus
            .update_focus(grid, position_mm)
            .map_err(WorldSdfGpuObjectsError::Focus)
    }

    /// Uploads a bounded resident dirty suffix after all CPU bakes have completed.
    pub fn flush_dirty(
        &mut self,
        max_bricks: usize,
        max_bytes: u64,
    ) -> Result<WorldSdfClipmapUploadStats, WorldSdfGpuObjectsError> {
        let grid = self.store.ready_grid()?;
        self.focus
            .flush_dirty(grid, max_bricks, max_bytes)
            .map_err(WorldSdfGpuObjectsError::Upload)
    }

    /// Requires complete CPU composition and no pending resident GPU changes,
    /// encodes queries with a consumer, and immediately submits on the same queue.
    pub fn submit_queries(
        &mut self,
        queries: &[GpuSdfQuery],
        encoder: wgpu::CommandEncoder,
        consume: impl FnOnce(&wgpu::Buffer, &mut wgpu::CommandEncoder),
    ) -> Result<wgpu::SubmissionIndex, WorldSdfGpuObjectsError> {
        self.store.ready_grid()?;
        self.focus
            .submit_queries(queries, encoder, consume)
            .map_err(WorldSdfGpuObjectsError::Upload)
    }
}
