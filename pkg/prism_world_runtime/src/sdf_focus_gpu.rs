//! Paired ownership of a world-mm focus driver and its SDF GPU clipmap.

use crate::{
    WorldSdfClipmapConfig, WorldSdfClipmapUploadStats, WorldSdfFocusConfig, WorldSdfFocusDriver,
    WorldSdfFocusUpdate, WorldSdfFocusUpdateError, WorldSdfGpuClipmap, WorldSdfGpuClipmapError,
};
use prism_world_gpu::{GpuContext, GpuSdfQuery};
use prism_world_residency::{BrickCoord, SparseSdfBrickGrid};

/// Errors from focus mapping or the owned runtime/GPU publication path.
pub type WorldSdfGpuFocusError = WorldSdfFocusUpdateError<WorldSdfGpuClipmapError>;

/// Drives the SDF ring from global integer-mm positions supplied by a host.
///
/// The host chooses the camera/player/streaming focus; this type does not read
/// renderer-specific camera state or impose a gameplay policy. Neither member
/// is exposed mutably, so a failed update cannot split focus and ring centers.
/// XZ windows and their single Y slice retain the existing clipmap geometry.
/// Coverage is enforced: Y hysteresis must be zero; nonzero X/Z hysteresis
/// requires radius >= 1. Missing source bricks still produce Unknown.
pub struct WorldSdfGpuFocus {
    focus: WorldSdfFocusDriver,
    ring: WorldSdfGpuClipmap,
}

impl WorldSdfGpuFocus {
    /// Validates focus before allocating a matching-brick-size GPU ring.
    pub fn new(
        context: &GpuContext,
        focus: WorldSdfFocusConfig,
        clipmap: WorldSdfClipmapConfig,
        max_queries: usize,
        max_upload_bytes: u64,
    ) -> Result<Self, WorldSdfGpuFocusError> {
        focus
            .validate_single_slice_coverage(clipmap.clipmap.ring_radius)
            .map_err(WorldSdfFocusUpdateError::Focus)?;
        let driver = WorldSdfFocusDriver::new(focus).map_err(WorldSdfFocusUpdateError::Focus)?;
        let ring = WorldSdfGpuClipmap::new(
            context,
            clipmap,
            focus.brick_size,
            max_queries,
            max_upload_bytes,
        )
        .map_err(WorldSdfFocusUpdateError::Apply)?;
        Ok(Self {
            focus: driver,
            ring,
        })
    }

    /// Read-only focus mapping and committed center.
    pub fn focus(&self) -> &WorldSdfFocusDriver {
        &self.focus
    }

    /// Read-only scheduler, resource accounting, and current sample count.
    pub fn ring(&self) -> &WorldSdfGpuClipmap {
        &self.ring
    }

    /// Maps a host focus and stages newly entered bricks before committing it.
    /// Retained focus performs constant work without allocation/upload/revision
    /// changes. A mismatched CPU grid is rejected even on retained focus.
    pub fn update_focus(
        &mut self,
        grid: &SparseSdfBrickGrid,
        position_mm: [i64; 3],
    ) -> Result<WorldSdfFocusUpdate<WorldSdfClipmapUploadStats>, WorldSdfGpuFocusError> {
        if grid.brick_size() != self.focus.config().brick_size
            || grid.mip_levels() != self.ring.scheduler().config().clipmap.mip_levels
        {
            return Err(WorldSdfFocusUpdateError::Apply(
                WorldSdfGpuClipmapError::GridMismatch,
            ));
        }
        self.focus
            .update_with(position_mm, |center| self.ring.update_center(grid, center))
    }

    /// Queues edited resident bricks independently of focus motion.
    pub fn mark_dirty(&mut self, coords: &[BrickCoord]) -> Result<usize, WorldSdfGpuClipmapError> {
        self.ring.mark_dirty(coords)
    }

    /// Services pending edits even when the focus is retained by hysteresis.
    pub fn flush_dirty(
        &mut self,
        grid: &SparseSdfBrickGrid,
        max_bricks: usize,
        max_bytes: u64,
    ) -> Result<WorldSdfClipmapUploadStats, WorldSdfGpuClipmapError> {
        self.ring.flush_dirty(grid, max_bricks, max_bytes)
    }

    /// Encodes queries and a GPU consumer then submits immediately, preserving
    /// the underlying ring's no-pending-dirty and queue ordering contract.
    pub fn submit_queries(
        &mut self,
        queries: &[GpuSdfQuery],
        encoder: wgpu::CommandEncoder,
        consume: impl FnOnce(&wgpu::Buffer, &mut wgpu::CommandEncoder),
    ) -> Result<wgpu::SubmissionIndex, WorldSdfGpuClipmapError> {
        self.ring.submit_queries(queries, encoder, consume)
    }
}
