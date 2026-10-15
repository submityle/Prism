//! Concrete incremental consumers for transform ECS spatial notifications.
//!
//! Run `prism_transform::ecs::transform_system` after local/relationship writers,
//! flush the ECS changed-observer bus (also done at schedule completion), then
//! call [`sync_spatial_indexes`]. One drain fans out to every installed backend;
//! do not independently drain the transform spatial stream while these adapters
//! own it. The returned commands remain available to additional consumers.
//!
//! `bvh` uses the existing physics `DynamicBvh`; `grid` uses replication's
//! existing XZ `AoiGrid`. The latter indexes the centre of each committed world
//! bound, for point-based interest queries, rather than volume overlap queries.
//! Both preserve full generational ECS identity and deterministic query order.
//! Transform spatial margins apply before consumption: a nonzero margin means
//! queries reflect the last committed fat bound, not every intermediate pose.
//! Snapshot restore reconciles the retained indexes against restored components;
//! edits pending from the abandoned timeline are discarded. Moving this resource
//! into another world clears its previous proxies while preserving backend setup.
//!
//! Base and grid integration are `no_std` + `alloc`. The optional BVH backend
//! enables `std`, required by the current geometry crate. No backend dependency
//! is added to the reusable transform algebra.
#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

extern crate alloc;

use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use prism_ecs::entity::Entity;
use prism_ecs::resource::Resource;
use prism_ecs::world::{World, WorldId};
use prism_math::Aabb3;
use prism_transform::ecs::{self, SpatialUpdate};

#[cfg(feature = "bvh")]
mod bvh;
#[cfg(feature = "bvh")]
pub use bvh::BvhIndex;
#[cfg(feature = "grid")]
mod grid;
#[cfg(feature = "grid")]
pub use grid::GridIndex;

/// World-owned consumers and their last successfully applied spatial baseline.
#[derive(Default)]
pub struct SpatialIndexes {
    source: Option<(WorldId, u64)>,
    committed: BTreeMap<Entity, Aabb3>,
    pending: BTreeMap<Entity, Option<Aabb3>>,
    #[cfg(feature = "bvh")]
    bvh: Option<BvhIndex>,
    #[cfg(feature = "grid")]
    grid: Option<GridIndex>,
}
impl Resource for SpatialIndexes {}

impl SpatialIndexes {
    fn clear_proxies(&mut self) {
        for &entity in self.committed.keys() {
            #[cfg(feature = "bvh")]
            if let Some(bvh) = &mut self.bvh {
                bvh.remove(entity);
            }
            #[cfg(feature = "grid")]
            if let Some(grid) = &mut self.grid {
                grid.remove(entity);
            }
            let _ = entity;
        }
        self.committed.clear();
        self.pending.clear();
    }

    /// Number of committed proxies, independent of the number of backends.
    pub fn len(&self) -> usize {
        self.committed.len()
    }

    /// Whether no proxy is committed.
    pub fn is_empty(&self) -> bool {
        self.committed.is_empty()
    }

    /// Last successfully committed bounds for a full entity identity.
    pub fn bounds(&self, entity: Entity) -> Option<Aabb3> {
        self.committed.get(&entity).copied()
    }

    /// Installed BVH consumer, if enabled for this world.
    #[cfg(feature = "bvh")]
    pub fn bvh(&self) -> Option<&BvhIndex> {
        self.bvh.as_ref()
    }

    /// Installed XZ point-interest grid, if enabled for this world.
    #[cfg(feature = "grid")]
    pub fn grid(&self) -> Option<&GridIndex> {
        self.grid.as_ref()
    }
}

/// Invalid bounds prevent the entire pending batch from reaching any backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpatialIndexError {
    /// Bounds must be finite, ordered and have finite extents on every axis.
    /// Fix the component or remove the proxy, flush, and retry. Pending edits
    /// are retained, and later signals replace earlier invalid edits.
    InvalidBounds(Entity),
}

/// Work performed while consuming one deterministic batch.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SpatialIndexStats {
    /// New committed proxies.
    pub inserts: usize,
    /// Changed committed bounds.
    pub updates: usize,
    /// Removed committed proxies.
    pub removes: usize,
    /// BVH updates that actually escaped its own internal fat margin.
    pub bvh_refits: usize,
    /// Grid positions inserted or moved (bounds-only changes can leave centres unchanged).
    pub grid_moves: usize,
}

/// Commands applied to all installed backends, available to other consumers.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SpatialIndexBatch {
    /// Minimal edits in ascending entity index and generation order.
    pub updates: Vec<SpatialUpdate>,
    /// Actual work, including backend-specific suppression.
    pub stats: SpatialIndexStats,
}

/// Install the bus integration and one fan-out resource, idempotently.
///
/// Late installation snapshots the last drained transform baseline; its first
/// sync catches up from that snapshot and any subsequent bus signals.
pub fn install(world: &mut World) {
    ecs::install(world);
    let source = (world.id(), world.archetype_generation());
    if world
        .get_resource::<SpatialIndexes>()
        .is_some_and(|indexes| indexes.source == Some(source))
    {
        return;
    }
    let baseline = ecs::spatial_snapshot(world);
    world.init_resource::<SpatialIndexes>();
    let indexes = world.resource_mut::<SpatialIndexes>();
    if indexes
        .source
        .is_some_and(|previous| previous.0 != source.0)
    {
        indexes.clear_proxies();
    }
    // A retained invalid batch can contain keys no longer tracked by the base
    // stream. Start from the committed backend state, then seed its last drain;
    // restore reconciliation in the next drain supplies authoritative corrections.
    indexes.pending.clear();
    indexes
        .pending
        .extend(indexes.committed.keys().map(|&entity| (entity, None)));
    indexes.pending.extend(
        baseline
            .into_iter()
            .map(|(entity, bounds)| (entity, Some(bounds))),
    );
    indexes.source = Some(source);
}

/// Install the existing dynamic BVH backend once, seeding committed proxies.
/// Returns whether a backend was added. Run sync to consume pending edits.
#[cfg(feature = "bvh")]
pub fn install_bvh(world: &mut World) -> bool {
    install(world);
    let indexes = world.resource_mut::<SpatialIndexes>();
    if indexes.bvh.is_some() {
        return false;
    }
    let mut bvh = BvhIndex::default();
    for (&entity, &bounds) in &indexes.committed {
        bvh.upsert(entity, bounds);
    }
    indexes.bvh = Some(bvh);
    true
}

/// Install the existing AOI grid once, seeding centres of committed bounds.
/// Returns whether a backend was added; an existing world's cell size is kept.
///
/// # Errors
/// Rejects a non-finite or non-positive cell size when creating a new grid.
#[cfg(feature = "grid")]
pub fn install_grid(
    world: &mut World,
    cell_size: f32,
) -> Result<bool, prism_replication::aoi::AoiError> {
    install(world);
    let indexes = world.resource_mut::<SpatialIndexes>();
    if indexes.grid.is_some() {
        return Ok(false);
    }
    let mut grid = GridIndex::new(cell_size)?;
    for (&entity, &bounds) in &indexes.committed {
        grid.upsert(entity, bounds);
    }
    indexes.grid = Some(grid);
    Ok(true)
}

fn valid(bounds: Aabb3) -> bool {
    let min = bounds.min;
    let max = bounds.max;
    [min.x, min.y, min.z, max.x, max.y, max.z]
        .iter()
        .all(|v| v.is_finite())
        && min.x <= max.x
        && min.y <= max.y
        && min.z <= max.z
        && (max.x - min.x).is_finite()
        && (max.y - min.y).is_finite()
        && (max.z - min.z).is_finite()
}

/// Drain the observer-driven spatial stream once and incrementally update all
/// installed indexes. No pending signals means no index mutations or refits.
///
/// This does not flush ECS observers: call after the schedule or a manual
/// `flush_changed_observers`. Installs the base integration on first use.
///
/// # Errors
/// Invalid bounds retain all pending edits for retry and leave every backend's
/// committed state unchanged. A later valid update or removal repairs the batch.
pub fn sync_spatial_indexes(world: &mut World) -> Result<SpatialIndexBatch, SpatialIndexError> {
    install(world);
    let updates = ecs::drain_spatial_updates(world);
    let indexes = world.resource_mut::<SpatialIndexes>();
    for update in updates {
        let bounds = match update {
            SpatialUpdate::Insert { bounds, .. } => Some(bounds),
            SpatialUpdate::Update { new, .. } => Some(new),
            SpatialUpdate::Remove { .. } => None,
        };
        indexes.pending.insert(update.entity(), bounds);
    }
    for (&entity, &bounds) in &indexes.pending {
        if bounds.is_some_and(|b| !valid(b)) {
            return Err(SpatialIndexError::InvalidBounds(entity));
        }
    }
    let mut batch = SpatialIndexBatch::default();
    for (entity, bounds) in core::mem::take(&mut indexes.pending) {
        let old = indexes.committed.get(&entity).copied();
        if old == bounds {
            continue;
        }
        if let Some(bounds) = bounds {
            #[cfg(feature = "bvh")]
            if let Some(bvh) = &mut indexes.bvh {
                batch.stats.bvh_refits += usize::from(bvh.upsert(entity, bounds));
            }
            #[cfg(feature = "grid")]
            if let Some(grid) = &mut indexes.grid {
                batch.stats.grid_moves += usize::from(grid.upsert(entity, bounds));
            }
            indexes.committed.insert(entity, bounds);
            if let Some(old) = old {
                batch.updates.push(SpatialUpdate::Update {
                    entity,
                    old,
                    new: bounds,
                });
                batch.stats.updates += 1;
            } else {
                batch.updates.push(SpatialUpdate::Insert { entity, bounds });
                batch.stats.inserts += 1;
            }
        } else if let Some(old) = old {
            #[cfg(feature = "bvh")]
            if let Some(bvh) = &mut indexes.bvh {
                bvh.remove(entity);
            }
            #[cfg(feature = "grid")]
            if let Some(grid) = &mut indexes.grid {
                grid.remove(entity);
            }
            indexes.committed.remove(&entity);
            batch.updates.push(SpatialUpdate::Remove { entity, old });
            batch.stats.removes += 1;
        }
    }
    Ok(batch)
}

/// Unsubscribe the transform integration and remove every owned backend proxy.
///
/// Returns the transform stream's final removal commands for external consumers,
/// plus the now-empty indexes for inspection or disposal. Cleans the actual
/// committed backend state even when an invalid batch or unflushed move/removal
/// is pending. Components remain available; reinstalling seeds them afresh.
pub fn uninstall(world: &mut World) -> (Vec<SpatialUpdate>, Option<SpatialIndexes>) {
    let updates = ecs::uninstall(world);
    let indexes = world
        .remove_resource::<SpatialIndexes>()
        .map(|mut indexes| {
            indexes.clear_proxies();
            indexes
        });
    (updates, indexes)
}
