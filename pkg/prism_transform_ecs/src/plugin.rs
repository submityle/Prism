use prism_app::{App, Plugin, PostUpdate};
use prism_ecs::prelude::{IntoSystemConfigs, Resource, SystemSet};
use prism_ecs::world::World;

use crate::{install, sync_spatial_indexes, SpatialIndexBatch, SpatialIndexError};

#[cfg(feature = "bvh")]
use crate::install_bvh;
#[cfg(feature = "grid")]
use crate::install_grid;

/// Default edge length for the optional XZ interest grid.
#[cfg(feature = "grid")]
pub const DEFAULT_GRID_CELL_SIZE: f32 = 4.0;

/// Backend policy installed by [`TransformSpatialPlugin`].
///
/// By default every backend compiled into `prism_transform_ecs` is enabled.
/// The base transform/spatial stream remains active when all optional backends
/// are disabled.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TransformSpatialConfig {
    #[cfg(feature = "bvh")]
    bvh: bool,
    #[cfg(feature = "grid")]
    grid_cell_size: Option<f32>,
}

impl Default for TransformSpatialConfig {
    fn default() -> Self {
        Self {
            #[cfg(feature = "bvh")]
            bvh: true,
            #[cfg(feature = "grid")]
            grid_cell_size: Some(DEFAULT_GRID_CELL_SIZE),
        }
    }
}

impl TransformSpatialConfig {
    /// Enables or disables the compiled BVH backend.
    #[cfg(feature = "bvh")]
    #[must_use]
    pub const fn with_bvh(mut self, enabled: bool) -> Self {
        self.bvh = enabled;
        self
    }

    /// Whether the plugin will install the compiled BVH backend.
    #[cfg(feature = "bvh")]
    #[must_use]
    pub const fn bvh_enabled(&self) -> bool {
        self.bvh
    }

    /// Enables the compiled grid backend with a validated cell size.
    #[cfg(feature = "grid")]
    pub fn with_grid_cell_size(
        mut self,
        cell_size: f32,
    ) -> Result<Self, prism_replication::aoi::AoiError> {
        if !cell_size.is_finite() || cell_size <= 0.0 {
            return Err(prism_replication::aoi::AoiError::BadCellSize);
        }
        self.grid_cell_size = Some(cell_size);
        Ok(self)
    }

    /// Disables the compiled grid backend.
    #[cfg(feature = "grid")]
    #[must_use]
    pub const fn without_grid(mut self) -> Self {
        self.grid_cell_size = None;
        self
    }

    /// Cell size used to install the grid, or `None` when it is disabled.
    #[cfg(feature = "grid")]
    #[must_use]
    pub const fn grid_cell_size(&self) -> Option<f32> {
        self.grid_cell_size
    }
}

/// Ordered anchors for the transform-to-spatial `PostUpdate` pipeline.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, SystemSet)]
pub enum TransformSpatialSystems {
    /// Propagates local and relationship edits into world transforms.
    Propagate,
    /// Flushes changed observers so propagation is visible to spatial tracking.
    FlushObservers,
    /// Drains spatial edits and commits every installed index backend.
    SyncSpatial,
}

/// Result published by the most recent scheduled spatial synchronization.
///
/// Invalid bounds are reported here instead of being discarded. The underlying
/// pending batch remains retained by [`sync_spatial_indexes`] and is retried on
/// the next frame after the offending component is repaired or removed.
#[derive(Clone, Debug, Default, PartialEq, Resource)]
pub struct SpatialIndexFrame {
    sequence: u64,
    batch: SpatialIndexBatch,
    error: Option<SpatialIndexError>,
}

impl SpatialIndexFrame {
    /// Number of scheduled synchronization attempts, wrapping at `u64::MAX`.
    #[must_use]
    pub const fn sequence(&self) -> u64 {
        self.sequence
    }

    /// Batch committed by the latest successful attempt.
    ///
    /// This is empty when the latest attempt failed.
    #[must_use]
    pub const fn batch(&self) -> &SpatialIndexBatch {
        &self.batch
    }

    /// Error from the latest attempt, if its pending batch was rejected.
    #[must_use]
    pub const fn error(&self) -> Option<SpatialIndexError> {
        self.error
    }
}

/// Flushes changed observers between propagation and spatial synchronization.
///
/// A schedule normally flushes only after all of its systems finish. Keeping
/// this explicit system in the chain makes propagated globals visible to the
/// spatial observer stream within the same `PostUpdate` schedule.
pub fn flush_transform_observers(world: &mut World) {
    world.flush_changed_observers();
}

/// Scheduled adapter around [`sync_spatial_indexes`].
///
/// The result is moved into [`SpatialIndexFrame`] so downstream systems can
/// inspect committed commands or a retained-batch error without another drain.
pub fn sync_spatial_indexes_system(world: &mut World) {
    let result = sync_spatial_indexes(world);
    world.init_resource::<SpatialIndexFrame>();
    let frame = world.resource_mut::<SpatialIndexFrame>();
    frame.sequence = frame.sequence.wrapping_add(1);
    match result {
        Ok(batch) => {
            frame.batch = batch;
            frame.error = None;
        }
        Err(error) => {
            frame.batch = SpatialIndexBatch::default();
            frame.error = Some(error);
        }
    }
}

/// Production App assembly for transform propagation and spatial indexes.
///
/// The plugin installs its resources immediately and registers one strict
/// `PostUpdate` chain:
/// `Propagate -> FlushObservers -> SyncSpatial`.
#[derive(Clone, Copy, Debug, Default)]
pub struct TransformSpatialPlugin {
    config: TransformSpatialConfig,
}

impl TransformSpatialPlugin {
    /// Creates a plugin with an explicit backend policy.
    #[must_use]
    pub const fn new(config: TransformSpatialConfig) -> Self {
        Self { config }
    }

    /// Backend policy used during plugin assembly.
    #[must_use]
    pub const fn config(&self) -> TransformSpatialConfig {
        self.config
    }
}

impl Plugin for TransformSpatialPlugin {
    fn build(&self, app: &mut App) {
        install(app.world_mut());
        #[cfg(feature = "bvh")]
        if self.config.bvh_enabled() {
            install_bvh(app.world_mut());
        }
        #[cfg(feature = "grid")]
        if let Some(cell_size) = self.config.grid_cell_size() {
            install_grid(app.world_mut(), cell_size)
                .expect("TransformSpatialConfig stores only valid grid cell sizes");
        }

        app.init_resource::<SpatialIndexFrame>().add_systems(
            PostUpdate,
            (
                prism_transform::ecs::transform_system.in_set(TransformSpatialSystems::Propagate),
                flush_transform_observers.in_set(TransformSpatialSystems::FlushObservers),
                sync_spatial_indexes_system.in_set(TransformSpatialSystems::SyncSpatial),
            )
                .chain(),
        );
    }
}
