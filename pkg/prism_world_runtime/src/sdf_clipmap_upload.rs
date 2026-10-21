//! Runtime ownership and bounded publication of a derived SDF GPU ring.

use core::fmt;

use prism_world_residency::{BrickCoord, SdfError, SparseSdfBrickGrid};

use crate::sdf_clipmap::SdfClipmapApplyError;
use crate::{WorldSdfClipmapConfig, WorldSdfClipmapError, WorldSdfClipmapScheduler};

/// A runtime upload failure. Returned errors preserve the attempted batch.
#[derive(Debug, PartialEq)]
pub enum WorldSdfClipmapUploadError<E> {
    /// Runtime geometry, dirty capacity, allocation, or revision preflight failed.
    Scheduler(WorldSdfClipmapError),
    /// The GPU backend rejected the operation before publishing queue writes.
    Backend(E),
    /// A CPU grid coordinate could not be read.
    Grid(SdfError),
    /// Brick size must be in 1..=32 and upload capacity must be nonzero.
    InvalidConfiguration,
    /// The CPU grid does not match the ring's fixed brick size or mip count.
    GridMismatch,
    /// Initialize a center before flushing edits or submitting queries.
    Uninitialized,
    /// Dirty data must be flushed before exposing results to consumers.
    PendingDirty {
        /// Number of bricks still awaiting upload.
        bricks: usize,
    },
    /// A nonempty flush requires a nonzero brick-count budget.
    ZeroBrickBudget,
    /// The first pending brick cannot fit in this batch's effective byte budget.
    UploadBudgetTooSmall {
        /// Bytes needed by the first pending brick, including key and metadata.
        required: u64,
        /// Minimum of the per-call and configured backend upload budget.
        budget: u64,
    },
}

impl<E: fmt::Display> fmt::Display for WorldSdfClipmapUploadError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Scheduler(error) => write!(f, "SDF runtime scheduler: {error}"),
            Self::Backend(error) => write!(f, "SDF upload backend: {error}"),
            Self::Grid(error) => write!(f, "SDF CPU grid: {error}"),
            Self::InvalidConfiguration => f.write_str("invalid SDF ring upload configuration"),
            Self::GridMismatch => f.write_str("SDF CPU grid and GPU ring geometry differ"),
            Self::Uninitialized => f.write_str("initialize the SDF ring center first"),
            Self::PendingDirty { bricks } => {
                write!(f, "SDF ring has {bricks} pending dirty bricks")
            }
            Self::ZeroBrickBudget => f.write_str("SDF dirty flush brick budget is zero"),
            Self::UploadBudgetTooSmall { required, budget } => {
                write!(f, "SDF dirty flush needs {required} bytes, budget {budget}")
            }
        }
    }
}

#[cfg(feature = "std")]
impl<E: std::error::Error + 'static> std::error::Error for WorldSdfClipmapUploadError<E> {}

impl<E> From<SdfClipmapApplyError<E>> for WorldSdfClipmapUploadError<E> {
    fn from(error: SdfClipmapApplyError<E>) -> Self {
        match error {
            SdfClipmapApplyError::Scheduler(error) => Self::Scheduler(error),
            SdfClipmapApplyError::Upload(error) => Self::Backend(error),
        }
    }
}

/// Statistics for one successful center update or dirty upload batch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WorldSdfClipmapUploadStats {
    /// Physical key, metadata and packed sample bytes staged by this call.
    /// Does not include query uniforms, driver staging, or prior calls.
    pub uploaded_bytes: u64,
    /// Remaining dirty coordinates, not pending GPU command completions.
    pub pending_bricks: usize,
    /// Runtime revision after successful publication.
    pub revision: u64,
}

// Private backend contract: both operations must return errors before any
// state/queue mutation. Only the actual toroidal GPU owner and test oracle
// implement this trait; callers cannot pair a mismatching external planner.
trait UploadBackend {
    type Error;

    fn update_center(
        &mut self,
        grid: &SparseSdfBrickGrid,
        center: BrickCoord,
    ) -> Result<u64, Self::Error>;

    fn upload_bricks(
        &mut self,
        grid: &SparseSdfBrickGrid,
        coords: &[BrickCoord],
    ) -> Result<u64, Self::Error>;
}

struct UploadState<B> {
    scheduler: WorldSdfClipmapScheduler,
    backend: B,
    brick_size: u8,
    sample_stride: u64,
    max_upload_bytes: u64,
}

impl<B: UploadBackend> UploadState<B> {
    fn new(
        config: WorldSdfClipmapConfig,
        brick_size: u8,
        max_upload_bytes: u64,
        create_backend: impl FnOnce() -> Result<B, B::Error>,
    ) -> Result<Self, WorldSdfClipmapUploadError<B::Error>> {
        if !(1..=32).contains(&brick_size) || max_upload_bytes == 0 {
            return Err(WorldSdfClipmapUploadError::InvalidConfiguration);
        }
        let scheduler =
            WorldSdfClipmapScheduler::new(config).map_err(WorldSdfClipmapUploadError::Scheduler)?;
        let sample_count = u64::from(brick_size).pow(3);
        let backend = create_backend().map_err(WorldSdfClipmapUploadError::Backend)?;
        Ok(Self {
            scheduler,
            backend,
            brick_size,
            sample_stride: sample_count.div_ceil(2) * 4,
            max_upload_bytes,
        })
    }

    fn validate_grid(
        &self,
        grid: &SparseSdfBrickGrid,
    ) -> Result<(), WorldSdfClipmapUploadError<B::Error>> {
        if grid.brick_size() != self.brick_size
            || grid.mip_levels() != self.scheduler.config().clipmap.mip_levels
        {
            return Err(WorldSdfClipmapUploadError::GridMismatch);
        }
        Ok(())
    }

    fn stats(&self, uploaded_bytes: u64) -> WorldSdfClipmapUploadStats {
        WorldSdfClipmapUploadStats {
            uploaded_bytes,
            pending_bricks: self.scheduler.dirty_bricks().len(),
            revision: self.scheduler.revision(),
        }
    }

    fn update_center(
        &mut self,
        grid: &SparseSdfBrickGrid,
        center: BrickCoord,
    ) -> Result<WorldSdfClipmapUploadStats, WorldSdfClipmapUploadError<B::Error>> {
        self.validate_grid(grid)?;
        if self.scheduler.center() == Some(center) {
            return Ok(self.stats(0));
        }
        let backend = &mut self.backend;
        let bytes = self
            .scheduler
            .update_center_with(center, || backend.update_center(grid, center))?;
        Ok(self.stats(bytes))
    }

    fn flush_dirty(
        &mut self,
        grid: &SparseSdfBrickGrid,
        max_bricks: usize,
        max_bytes: u64,
    ) -> Result<WorldSdfClipmapUploadStats, WorldSdfClipmapUploadError<B::Error>> {
        self.validate_grid(grid)?;
        if self.scheduler.center().is_none() {
            return Err(WorldSdfClipmapUploadError::Uninitialized);
        }
        let dirty = self.scheduler.dirty_bricks();
        if dirty.is_empty() {
            return Ok(self.stats(0));
        }
        if max_bricks == 0 {
            return Err(WorldSdfClipmapUploadError::ZeroBrickBudget);
        }
        let budget = max_bytes.min(self.max_upload_bytes);
        let mut bytes = 0_u64;
        let mut count = 0;
        for &coord in dirty.iter().rev().take(max_bricks) {
            let present = grid
                .brick_samples(coord)
                .map_err(WorldSdfClipmapUploadError::Grid)?
                .is_some();
            // Same ABI as GpuSdfToroidalLookup: 16-byte key, 16-byte
            // metadata and, only for present bricks, packed i16 sample words.
            let required = 32 + if present { self.sample_stride } else { 0 };
            if required > budget - bytes {
                if count == 0 {
                    return Err(WorldSdfClipmapUploadError::UploadBudgetTooSmall {
                        required,
                        budget,
                    });
                }
                break;
            }
            bytes += required;
            count += 1;
        }
        let backend = &mut self.backend;
        let uploaded_bytes = self
            .scheduler
            .upload_dirty_suffix_with(count, |coords| backend.upload_bricks(grid, coords))?;
        Ok(self.stats(uploaded_bytes))
    }

    fn require_ready(&self) -> Result<(), WorldSdfClipmapUploadError<B::Error>> {
        if self.scheduler.center().is_none() {
            return Err(WorldSdfClipmapUploadError::Uninitialized);
        }
        let bricks = self.scheduler.dirty_bricks().len();
        if bricks != 0 {
            return Err(WorldSdfClipmapUploadError::PendingDirty { bricks });
        }
        Ok(())
    }
}

#[cfg(feature = "gpu_proxy")]
mod gpu {
    use super::*;
    use prism_world_gpu::{GpuContext, GpuSdfError, GpuSdfQuery, GpuSdfToroidalLookup};

    /// Concrete GPU ring upload error.
    pub type WorldSdfGpuClipmapError = WorldSdfClipmapUploadError<GpuSdfError>;

    impl UploadBackend for GpuSdfToroidalLookup {
        type Error = GpuSdfError;

        fn update_center(
            &mut self,
            grid: &SparseSdfBrickGrid,
            center: BrickCoord,
        ) -> Result<u64, Self::Error> {
            self.update_center(grid, center)
        }

        fn upload_bricks(
            &mut self,
            grid: &SparseSdfBrickGrid,
            coords: &[BrickCoord],
        ) -> Result<u64, Self::Error> {
            self.upload_bricks(grid, coords)
        }
    }

    /// Owns one runtime scheduler and its matching fixed-capacity GPU SDF ring.
    ///
    /// The borrowed CPU grid remains the sole sample source. Its edits must be
    /// marked with [`Self::mark_dirty`] before submitting more queries. This
    /// owner never consumes the CPU grid's dirty list (other consumers may use
    /// it). Missing bricks upload Unknown, not a synthetic empty-space value.
    ///
    /// Center rolls upload only entered keys and preserve retained edits. Dirty
    /// batches are acknowledged only after all queue writes are staged. This
    /// is staging acknowledgement, not proof of device completion or a bake
    /// completion protocol. Device loss requires rebuilding the derived owner.
    pub struct WorldSdfGpuClipmap {
        state: UploadState<GpuSdfToroidalLookup>,
        queue: wgpu::Queue,
    }

    impl WorldSdfGpuClipmap {
        /// Creates both planners with identical geometry, before either rolls.
        /// `max_upload_bytes` bounds each roll and each explicit dirty batch,
        /// not the sum of multiple calls or the fixed buffer allocation.
        pub fn new(
            context: &GpuContext,
            config: WorldSdfClipmapConfig,
            brick_size: u8,
            max_queries: usize,
            max_upload_bytes: u64,
        ) -> Result<Self, WorldSdfGpuClipmapError> {
            let state = UploadState::new(config, brick_size, max_upload_bytes, || {
                GpuSdfToroidalLookup::new(
                    context,
                    config.clipmap,
                    brick_size,
                    max_queries,
                    max_upload_bytes,
                )
            })?;
            Ok(Self {
                state,
                queue: context.queue().clone(),
            })
        }

        /// Returns the scheduler read-only; mutable access would split planners.
        pub fn scheduler(&self) -> &WorldSdfClipmapScheduler {
            &self.state.scheduler
        }

        /// Rolls the camera window, uploading newly entered CPU brick samples.
        /// Returned errors preserve both old centers and pending edits. Same
        /// center is O(1), with no planner copy, allocation or GPU upload.
        pub fn update_center(
            &mut self,
            grid: &SparseSdfBrickGrid,
            center: BrickCoord,
        ) -> Result<WorldSdfClipmapUploadStats, WorldSdfGpuClipmapError> {
            self.state.update_center(grid, center)
        }

        /// Marks edited resident coordinates, including repeated edits to an
        /// already-dirty brick. Off-window edits need no mark: future entering
        /// keys read their latest values from the CPU grid.
        pub fn mark_dirty(
            &mut self,
            coords: &[BrickCoord],
        ) -> Result<usize, WorldSdfGpuClipmapError> {
            self.state
                .scheduler
                .mark_dirty(coords)
                .map_err(WorldSdfClipmapUploadError::Scheduler)
        }

        /// Uploads one deterministic dirty suffix within both call budgets.
        /// A successful batch may leave work pending. Errors retain all work in
        /// this batch; earlier successful calls are not rolled back. No dirty
        /// coordinate copy or whole-grid export is performed by the adapter.
        /// A suffix is truncated in O(1), avoiding repeated prefix compaction.
        pub fn flush_dirty(
            &mut self,
            grid: &SparseSdfBrickGrid,
            max_bricks: usize,
            max_bytes: u64,
        ) -> Result<WorldSdfClipmapUploadStats, WorldSdfGpuClipmapError> {
            self.state.flush_dirty(grid, max_bricks, max_bytes)
        }

        /// Uploads queries, encodes lookup then consumer commands, and submits
        /// the supplied same-device encoder on this owner's queue. Submission
        /// occurs before the next mutable call, preventing later uploads from
        /// replacing samples/queries for an unsubmitted dispatch. The callback
        /// may record GPU consumers or readback copies; it must not submit or
        /// overwrite ring resources. No CPU wait or readback is performed here.
        /// Pending dirty data fails closed before any query upload or callback.
        pub fn submit_queries(
            &mut self,
            queries: &[GpuSdfQuery],
            mut encoder: wgpu::CommandEncoder,
            consume: impl FnOnce(&wgpu::Buffer, &mut wgpu::CommandEncoder),
        ) -> Result<wgpu::SubmissionIndex, WorldSdfGpuClipmapError> {
            self.state.require_ready()?;
            self.state
                .backend
                .write_queries(queries)
                .map_err(WorldSdfClipmapUploadError::Backend)?;
            self.state
                .backend
                .encode(&mut encoder)
                .map_err(WorldSdfClipmapUploadError::Backend)?;
            consume(self.state.backend.results(), &mut encoder);
            Ok(self.queue.submit([encoder.finish()]))
        }

        /// Logical owned GPU buffer bytes; excludes driver and staging memory.
        pub fn allocated_buffer_bytes(&self) -> u64 {
            self.state.backend.allocated_buffer_bytes()
        }

        /// Number of valid sample slots, excluding absent/Unknown source bricks.
        pub fn brick_count(&self) -> usize {
            self.state.backend.brick_count()
        }
    }
}

#[cfg(feature = "gpu_proxy")]
pub use gpu::{WorldSdfGpuClipmap, WorldSdfGpuClipmapError};

#[cfg(test)]
mod tests;
