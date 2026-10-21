//! Object identity, spatial invalidation and transactional narrow-band SDF union.

use alloc::{collections::BTreeMap, sync::Arc, vec::Vec};
use core::{fmt, mem::size_of};
use prism_world_residency::{BrickCoord, SparseSdfBrickGrid};

/// Physical interpretation shared by every object and every mip.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WorldSdfObjectConfig {
    /// World position of mip-zero voxel zero, in millimetres.
    pub origin_mm: [i64; 3],
    /// Positive mip-zero sample spacing in millimetres.
    pub voxel_size_mm: u32,
    /// Positive millimetres per signed-i16 distance unit, unchanged across mips.
    pub distance_scale_mm: u32,
    /// Cubic sample edge, in 1..=32.
    pub brick_size: u8,
    /// Number of mips, in 1..=32. Samples are evaluated independently per mip.
    pub mip_levels: u8,
}

/// Independent persistent and per-edit limits. Zero capacities are allowed.
/// Logical bounds exclude allocator/tree overhead, Arc payloads and detached
/// work retained by callers. Tree/Arc allocation failure can abort the process.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WorldSdfObjectLimits {
    /// Maximum live object identities.
    pub max_objects: usize,
    /// Maximum union of covered and still-materialized bricks, including mips.
    /// Moving a full field may require baking removals before inserting elsewhere.
    pub max_bricks: usize,
    /// Maximum object-to-brick links in the spatial index.
    pub max_links: usize,
    /// Maximum retained dirty coordinates.
    pub max_dirty_bricks: usize,
    /// Maximum raw old plus new brick coverage in one edit, before deduplication.
    pub max_edit_bricks: usize,
    /// Maximum logical edit staging bytes, including all temporary Vec payloads.
    pub max_edit_scratch_bytes: usize,
}

/// Per-work limits; callers also bound simultaneously detached jobs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WorldSdfObjectBakeBudget {
    /// Maximum deterministic dirty suffix to prepare. Zero is a no-op.
    pub max_bricks: usize,
    /// Maximum source evaluations, charged conservatively for whole bricks.
    pub max_evaluations: usize,
    /// Maximum logical work descriptors, source snapshots and sample storage.
    pub max_scratch_bytes: usize,
}

/// A measured resource in an admission failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WorldSdfObjectResource {
    /// Live identities.
    Objects,
    /// Union of covered and still-materialized bricks.
    Bricks,
    /// Spatial index links.
    Links,
    /// Pending dirty coordinates.
    DirtyBricks,
    /// Raw old plus new coverage of one edit.
    EditBricks,
    /// Source evaluator calls in a detached job.
    Evaluations,
    /// Logical temporary storage.
    ScratchBytes,
}

/// All returned admission/publication failures leave the owner unchanged.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WorldSdfObjectError {
    /// Physical scales, brick geometry or mip count are invalid.
    InvalidConfiguration,
    /// An object's local inclusive minimum exceeds its maximum.
    InvalidBounds,
    /// A world influence region exceeds signed-i32 brick addressing.
    CoordinateOverflow,
    /// Checked counts, byte charges or revision overflowed.
    ArithmeticOverflow,
    /// Work belongs to another owner or a superseded revision.
    StaleWork,
    /// Objects have changed but the derived grid is not yet completely rebuilt.
    PendingBake {
        /// Remaining dirty coordinates.
        bricks: usize,
    },
    /// A configured resource limit was exceeded.
    BudgetExceeded {
        /// Resource being charged.
        resource: WorldSdfObjectResource,
        /// Requested count or bytes.
        required: usize,
        /// Configured maximum.
        limit: usize,
    },
    /// Fallible Vec reservation failed.
    AllocationFailed,
}

impl fmt::Display for WorldSdfObjectError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "world SDF object field: {self:?}")
    }
}
impl core::error::Error for WorldSdfObjectError {}

/// Immutable local field and its placement. The inclusive influence bounds
/// must contain every sample where the quantized source differs from i16::MAX.
/// This is a truncated/narrow-band field contract, not a finite-support claim
/// about an untruncated mathematical SDF. Outside the region the source is far.
/// Rotation/scale or arbitrary geometry may live in the immutable source; its
/// evaluator receives exact local/world mm and must use the common distance unit.
#[derive(Debug)]
pub struct WorldSdfObject<T> {
    /// Translation applied before local sampling.
    pub translation_mm: [i64; 3],
    /// Inclusive local influence minimum, in millimetres.
    pub local_min_mm: [i64; 3],
    /// Inclusive local influence maximum, in millimetres.
    pub local_max_mm: [i64; 3],
    /// Immutable payload shared with detached jobs. Interior mutations must not
    /// alter sampling; publish changes through upsert instead.
    pub source: Arc<T>,
}

impl<T> Clone for WorldSdfObject<T> {
    fn clone(&self) -> Self {
        Self {
            translation_mm: self.translation_mm,
            local_min_mm: self.local_min_mm,
            local_max_mm: self.local_max_mm,
            source: self.source.clone(),
        }
    }
}

/// Exact coordinates supplied to an object's local-field evaluator.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WorldSdfObjectSample {
    /// Stable identity currently contributing to this sample.
    pub object_id: u64,
    /// Global physical sample position, before float narrowing.
    pub world_position_mm: [i128; 3],
    /// Global sample position minus the object's translation.
    pub local_position_mm: [i128; 3],
    /// Owning brick and mip.
    pub brick: BrickCoord,
    /// Physical millimetres represented by one returned i16 unit.
    pub distance_scale_mm: u32,
}

/// Evaluator failure; the resident grid and dirty work remain unchanged.
#[derive(Debug, Eq, PartialEq)]
pub struct WorldSdfObjectBakeError<E> {
    /// Exact object/sample which failed.
    pub sample: WorldSdfObjectSample,
    /// Source-specific error. External callback effects cannot be rolled back.
    pub source: E,
}

impl<E: fmt::Display> fmt::Display for WorldSdfObjectBakeError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SDF object {} sample failed: {}",
            self.sample.object_id, self.source
        )
    }
}
impl<E: core::error::Error + 'static> core::error::Error for WorldSdfObjectBakeError<E> {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        Some(&self.source)
    }
}

struct Coverage {
    coord: BrickCoord,
    ids: Vec<u64>,
}
struct BakeSource<T> {
    id: u64,
    object: WorldSdfObject<T>,
}
struct BakeBrick<T> {
    coord: BrickCoord,
    sources: Vec<BakeSource<T>>,
    samples: Option<Vec<i16>>,
}

/// Bounded, detached work; only affected source Arcs are retained, not the grid.
/// Preparation allocates all buffers before the first evaluator invocation.
/// Successful evaluation alone never changes owner state.
#[must_use]
pub struct WorldSdfObjectBake<T> {
    owner: Arc<()>,
    revision: u64,
    config: WorldSdfObjectConfig,
    bricks: Vec<BakeBrick<T>>,
    evaluations: usize,
    scratch_bytes: usize,
}

/// Fully evaluated work, fenced to one owner and exact edit/publication revision.
#[must_use]
pub struct WorldSdfObjectUpdate<T> {
    work: WorldSdfObjectBake<T>,
    sampled: usize,
}

/// Publication cost, not a device completion or wall-clock deadline.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WorldSdfObjectBakeReport {
    /// Owner revision after commit.
    pub revision: u64,
    /// Replaced or removed bricks.
    pub changed_bricks: usize,
    /// Actual source evaluator calls.
    pub evaluations: usize,
    /// Conservative prepared evaluator bound.
    pub evaluation_bound: usize,
    /// Logical detached work payload bytes, not allocator/RSS usage.
    pub scratch_bytes: usize,
    /// Work remaining before queries can see the new field.
    pub pending_bricks: usize,
}

/// Engine-neutral object-field owner with a per-brick contributor index.
/// Edits invalidate both old and new influence at every mip. Rebakes compute
/// pointwise min from all surviving contributors, so move/delete restores
/// occluded contributors instead of attempting to subtract from a min field.
/// Bricks outside every object are absent/Unknown; covered brick samples outside
/// all influence boxes are explicitly i16::MAX. No unknown base grid is invented.
pub struct WorldSdfObjectStore<T> {
    config: WorldSdfObjectConfig,
    limits: WorldSdfObjectLimits,
    objects: BTreeMap<u64, WorldSdfObject<T>>,
    coverage: BTreeMap<BrickCoord, Vec<u64>>,
    links: usize,
    reserved: usize,
    dirty: Vec<BrickCoord>,
    grid: SparseSdfBrickGrid,
    owner: Arc<()>,
    revision: u64,
}

fn check(
    resource: WorldSdfObjectResource,
    required: usize,
    limit: usize,
) -> Result<(), WorldSdfObjectError> {
    if required > limit {
        Err(WorldSdfObjectError::BudgetExceeded {
            resource,
            required,
            limit,
        })
    } else {
        Ok(())
    }
}
fn add(a: usize, b: usize) -> Result<usize, WorldSdfObjectError> {
    a.checked_add(b)
        .ok_or(WorldSdfObjectError::ArithmeticOverflow)
}
fn mul(a: usize, b: usize) -> Result<usize, WorldSdfObjectError> {
    a.checked_mul(b)
        .ok_or(WorldSdfObjectError::ArithmeticOverflow)
}
fn buffer<T>(count: usize) -> Result<Vec<T>, WorldSdfObjectError> {
    let mut result = Vec::new();
    result
        .try_reserve_exact(count)
        .map_err(|_| WorldSdfObjectError::AllocationFailed)?;
    Ok(result)
}

type Bounds = ([i32; 3], [i32; 3]);

impl WorldSdfObjectConfig {
    fn ranges<T>(
        &self,
        object: &WorldSdfObject<T>,
    ) -> Result<([Bounds; 32], usize), WorldSdfObjectError> {
        let mut ranges = [([0; 3], [0; 3]); 32];
        let mut count = 0;
        for axis in 0..3 {
            if object.local_min_mm[axis] > object.local_max_mm[axis] {
                return Err(WorldSdfObjectError::InvalidBounds);
            }
        }
        for mip in 0..self.mip_levels {
            let extent =
                i128::from(self.voxel_size_mm) * i128::from(self.brick_size) * (1_i128 << mip);
            let mut lo = [0; 3];
            let mut hi = [0; 3];
            let mut volume = 1;
            for axis in 0..3 {
                let origin =
                    i128::from(object.translation_mm[axis]) - i128::from(self.origin_mm[axis]);
                lo[axis] = i32::try_from(
                    (origin + i128::from(object.local_min_mm[axis])).div_euclid(extent),
                )
                .map_err(|_| WorldSdfObjectError::CoordinateOverflow)?;
                hi[axis] = i32::try_from(
                    (origin + i128::from(object.local_max_mm[axis])).div_euclid(extent),
                )
                .map_err(|_| WorldSdfObjectError::CoordinateOverflow)?;
                let width = usize::try_from(i64::from(hi[axis]) - i64::from(lo[axis]) + 1)
                    .map_err(|_| WorldSdfObjectError::ArithmeticOverflow)?;
                volume = mul(volume, width)?;
            }
            count = add(count, volume)?;
            ranges[usize::from(mip)] = (lo, hi);
        }
        Ok((ranges, count))
    }

    fn coords(
        &self,
        ranges: &[Bounds; 32],
        count: usize,
    ) -> Result<Vec<BrickCoord>, WorldSdfObjectError> {
        let mut coords = buffer(count)?;
        for mip in 0..self.mip_levels {
            let (lo, hi) = ranges[usize::from(mip)];
            for x in lo[0]..=hi[0] {
                for y in lo[1]..=hi[1] {
                    for z in lo[2]..=hi[2] {
                        coords.push(BrickCoord::new(x, y, z, mip));
                    }
                }
            }
        }
        coords.sort_unstable();
        Ok(coords)
    }
}

impl<T> WorldSdfObjectStore<T> {
    /// Creates an empty field without allocating sample storage.
    pub fn new(
        config: WorldSdfObjectConfig,
        limits: WorldSdfObjectLimits,
    ) -> Result<Self, WorldSdfObjectError> {
        if config.voxel_size_mm == 0
            || config.distance_scale_mm == 0
            || !(1..=32).contains(&config.brick_size)
            || !(1..=32).contains(&config.mip_levels)
        {
            return Err(WorldSdfObjectError::InvalidConfiguration);
        }
        Ok(Self {
            config,
            limits,
            objects: BTreeMap::new(),
            coverage: BTreeMap::new(),
            links: 0,
            reserved: 0,
            dirty: Vec::new(),
            grid: SparseSdfBrickGrid::new(config.brick_size, config.mip_levels)
                .map_err(|_| WorldSdfObjectError::InvalidConfiguration)?,
            owner: Arc::new(()),
            revision: 0,
        })
    }
    /// Immutable physical configuration.
    pub const fn config(&self) -> WorldSdfObjectConfig {
        self.config
    }
    /// Current edit/publication sequence.
    pub const fn revision(&self) -> u64 {
        self.revision
    }
    /// Live object count.
    pub fn object_count(&self) -> usize {
        self.objects.len()
    }
    /// Distinct spatial-index coverage count; pending old grid bricks may differ.
    pub fn covered_bricks(&self) -> usize {
        self.coverage.len()
    }
    /// Covered-or-materialized brick count charged to the persistent limit.
    pub const fn reserved_bricks(&self) -> usize {
        self.reserved
    }
    /// Number of retained source-to-brick links.
    pub const fn link_count(&self) -> usize {
        self.links
    }
    /// Sorted coordinates awaiting composition.
    pub fn dirty_bricks(&self) -> &[BrickCoord] {
        &self.dirty
    }
    /// Last committed grid, possibly a partial update while dirty is nonempty.
    /// Consumer owners must use ready_grid or enforce the same readiness fence.
    pub fn grid(&self) -> &SparseSdfBrickGrid {
        &self.grid
    }
    /// Exposes a field only after all source changes have been composed.
    pub fn ready_grid(&self) -> Result<&SparseSdfBrickGrid, WorldSdfObjectError> {
        if self.dirty.is_empty() {
            Ok(&self.grid)
        } else {
            Err(WorldSdfObjectError::PendingBake {
                bricks: self.dirty.len(),
            })
        }
    }
    /// Inserts/replaces an immutable source. Every accepted replacement fences
    /// older work even when its influence bounds and source Arc are unchanged.
    pub fn upsert(
        &mut self,
        id: u64,
        object: WorldSdfObject<T>,
    ) -> Result<(), WorldSdfObjectError> {
        self.edit(id, Some(object))
    }
    /// Removes one identity, preserving surviving min contributors. Missing ids
    /// return false without changing the revision or invalidating detached jobs.
    pub fn remove(&mut self, id: u64) -> Result<bool, WorldSdfObjectError> {
        if !self.objects.contains_key(&id) {
            return Ok(false);
        }
        self.edit(id, None)?;
        Ok(true)
    }

    fn edit(
        &mut self,
        id: u64,
        next: Option<WorldSdfObject<T>>,
    ) -> Result<(), WorldSdfObjectError> {
        use WorldSdfObjectResource as R;
        let exists = self.objects.contains_key(&id);
        let object_count = add(
            self.objects.len() - usize::from(exists),
            usize::from(next.is_some()),
        )?;
        check(R::Objects, object_count, self.limits.max_objects)?;
        let revision = self
            .revision
            .checked_add(1)
            .ok_or(WorldSdfObjectError::ArithmeticOverflow)?;
        let previous = self
            .objects
            .get(&id)
            .map(|o| self.config.ranges(o))
            .transpose()?;
        let following = next.as_ref().map(|o| self.config.ranges(o)).transpose()?;
        let raw = add(
            previous.as_ref().map_or(0, |r| r.1),
            following.as_ref().map_or(0, |r| r.1),
        )?;
        check(R::EditBricks, raw, self.limits.max_edit_bricks)?;
        // Old/new coordinates plus their merged copy coexist during staging.
        let coord_bytes = mul(mul(raw, 2)?, size_of::<BrickCoord>())?;
        check(
            R::ScratchBytes,
            coord_bytes,
            self.limits.max_edit_scratch_bytes,
        )?;
        let old = match previous {
            Some((r, n)) => self.config.coords(&r, n)?,
            None => Vec::new(),
        };
        let new = match following {
            Some((r, n)) => self.config.coords(&r, n)?,
            None => Vec::new(),
        };
        let mut touched = buffer(raw)?;
        touched.extend_from_slice(&old);
        touched.extend_from_slice(&new);
        touched.sort_unstable();
        touched.dedup();
        let mut reserved = self.reserved;
        let mut links = self.links;
        let mut staged_links = 0;
        let mut dirty_count = self.dirty.len();
        for &coord in &touched {
            let current = self.coverage.get(&coord).map_or(0, Vec::len);
            let count = add(
                current - usize::from(old.binary_search(&coord).is_ok()),
                usize::from(new.binary_search(&coord).is_ok()),
            )?;
            let resident = self
                .grid
                .brick_samples(coord)
                .expect("validated coverage mip")
                .is_some();
            reserved = add(
                reserved - usize::from(current != 0 || resident),
                usize::from(count != 0 || resident),
            )?;
            links = add(links - current, count)?;
            staged_links = add(staged_links, count)?;
            dirty_count = add(
                dirty_count,
                usize::from(self.dirty.binary_search(&coord).is_err()),
            )?;
        }
        check(R::Bricks, reserved, self.limits.max_bricks)?;
        check(R::Links, links, self.limits.max_links)?;
        check(R::DirtyBricks, dirty_count, self.limits.max_dirty_bricks)?;
        let scratch = add(
            coord_bytes,
            add(
                mul(touched.len(), size_of::<Coverage>())?,
                add(
                    mul(staged_links, size_of::<u64>())?,
                    mul(dirty_count, size_of::<BrickCoord>())?,
                )?,
            )?,
        )?;
        check(R::ScratchBytes, scratch, self.limits.max_edit_scratch_bytes)?;
        let mut staged = buffer(touched.len())?;
        for &coord in &touched {
            let current = self.coverage.get(&coord).map_or(&[][..], Vec::as_slice);
            let insert = new.binary_search(&coord).is_ok();
            let count = current.len() - usize::from(old.binary_search(&coord).is_ok())
                + usize::from(insert);
            let mut ids = buffer(count)?;
            ids.extend(current.iter().copied().filter(|&v| v != id));
            if insert {
                let at = ids.binary_search(&id).unwrap_or_else(|at| at);
                ids.insert(at, id);
            }
            staged.push(Coverage { coord, ids });
        }
        let mut dirty = buffer(dirty_count)?;
        dirty.extend_from_slice(&self.dirty);
        dirty.extend(
            touched
                .iter()
                .filter(|c| self.dirty.binary_search(c).is_err())
                .copied(),
        );
        dirty.sort_unstable();
        // No returned failures after this point. BTree nodes use Rust's fatal
        // allocation policy, not a recoverable error/rollback promise.
        for entry in staged {
            if entry.ids.is_empty() {
                self.coverage.remove(&entry.coord);
            } else {
                self.coverage.insert(entry.coord, entry.ids);
            }
        }
        if let Some(object) = next {
            self.objects.insert(id, object);
        } else {
            self.objects.remove(&id);
        }
        self.links = links;
        self.reserved = reserved;
        self.dirty = dirty;
        self.revision = revision;
        Ok(())
    }

    /// Snapshots a deterministic dirty suffix and preallocates all evaluation
    /// buffers. A failing budget does not silently produce a smaller job.
    pub fn prepare(
        &self,
        budget: WorldSdfObjectBakeBudget,
    ) -> Result<WorldSdfObjectBake<T>, WorldSdfObjectError> {
        let count = self.dirty.len().min(budget.max_bricks);
        let coords = &self.dirty[self.dirty.len() - count..];
        let samples = usize::from(self.config.brick_size).pow(3);
        let mut evaluations = 0;
        let mut scratch = mul(count, size_of::<BakeBrick<T>>())?;
        for coord in coords {
            let sources = self.coverage.get(coord).map_or(0, Vec::len);
            evaluations = add(evaluations, mul(samples, sources)?)?;
            scratch = add(scratch, mul(sources, size_of::<BakeSource<T>>())?)?;
            if sources != 0 {
                scratch = add(scratch, mul(samples, size_of::<i16>())?)?;
            } else {
                // Removal creates legacy parent flags in SparseSdfBrickGrid.
                // Charge the temporary take_dirty Vec before it is allocated.
                let parents = usize::from(self.config.mip_levels - coord.mip - 1);
                scratch = add(scratch, mul(parents, size_of::<BrickCoord>())?)?;
            }
        }
        check(
            WorldSdfObjectResource::Evaluations,
            evaluations,
            budget.max_evaluations,
        )?;
        check(
            WorldSdfObjectResource::ScratchBytes,
            scratch,
            budget.max_scratch_bytes,
        )?;
        let mut bricks = buffer(count)?;
        for &coord in coords {
            let ids = self.coverage.get(&coord).map_or(&[][..], Vec::as_slice);
            let mut sources = buffer(ids.len())?;
            for &id in ids {
                sources.push(BakeSource {
                    id,
                    object: self.objects[&id].clone(),
                });
            }
            let samples = if ids.is_empty() {
                None
            } else {
                let mut values = buffer(samples)?;
                values.resize(samples, i16::MAX);
                Some(values)
            };
            bricks.push(BakeBrick {
                coord,
                sources,
                samples,
            });
        }
        Ok(WorldSdfObjectBake {
            owner: self.owner.clone(),
            revision: self.revision,
            config: self.config,
            bricks,
            evaluations,
            scratch_bytes: scratch,
        })
    }

    pub(super) fn validate_update(
        &self,
        update: &WorldSdfObjectUpdate<T>,
    ) -> Result<(), WorldSdfObjectError> {
        if !Arc::ptr_eq(&self.owner, &update.work.owner) || self.revision != update.work.revision {
            return Err(WorldSdfObjectError::StaleWork);
        }
        if !update.work.bricks.is_empty() {
            self.revision
                .checked_add(1)
                .ok_or(WorldSdfObjectError::ArithmeticOverflow)?;
        }
        Ok(())
    }

    /// Publishes a fully evaluated batch after its owner/revision check. Every
    /// nonempty publication fences other jobs, including disjoint older work.
    pub fn commit(
        &mut self,
        update: WorldSdfObjectUpdate<T>,
    ) -> Result<WorldSdfObjectBakeReport, WorldSdfObjectError> {
        self.validate_update(&update)?;
        Ok(self.commit_validated(update))
    }

    pub(super) fn commit_validated(
        &mut self,
        update: WorldSdfObjectUpdate<T>,
    ) -> WorldSdfObjectBakeReport {
        let changed_bricks = update.work.bricks.len();
        for brick in update.work.bricks {
            if let Some(samples) = brick.samples {
                self.grid
                    .insert_brick(brick.coord, samples)
                    .expect("validated object brick geometry");
            } else if self
                .grid
                .remove_brick(brick.coord)
                .expect("validated object brick mip")
            {
                self.reserved -= 1;
            }
        }
        // Every mip is explicitly indexed/rebaked. Grid's legacy parent flags
        // are not this owner's scheduling channel and must not leak as stale.
        let _ = self.grid.take_dirty();
        self.dirty.truncate(self.dirty.len() - changed_bricks);
        if changed_bricks != 0 {
            self.revision += 1;
        }
        WorldSdfObjectBakeReport {
            revision: self.revision,
            changed_bricks,
            evaluations: update.sampled,
            evaluation_bound: update.work.evaluations,
            scratch_bytes: update.work.scratch_bytes,
            pending_bricks: self.dirty.len(),
        }
    }
}

impl<T> WorldSdfObjectBake<T> {
    /// Conservative number of callback invocations admitted before allocation.
    pub const fn evaluation_bound(&self) -> usize {
        self.evaluations
    }
    /// Logical descriptor, source snapshot and replacement sample bytes.
    pub const fn scratch_bytes(&self) -> usize {
        self.scratch_bytes
    }
    /// Number of brick replacements/removals in this batch.
    pub fn brick_count(&self) -> usize {
        self.bricks.len()
    }

    /// Evaluates all contributors in stable id order, with X-fastest samples.
    /// Callback errors or unwinding leave the store and its pending work intact.
    /// No allocations occur here. Implement source cancellation by returning an
    /// evaluator error; worker/task-pool policy stays outside this field owner.
    #[expect(
        clippy::result_large_err,
        reason = "retain exact i128 sample diagnostics without allocating on evaluator failure"
    )]
    pub fn evaluate<E>(
        mut self,
        mut sample: impl FnMut(&T, WorldSdfObjectSample) -> Result<i16, E>,
    ) -> Result<WorldSdfObjectUpdate<T>, WorldSdfObjectBakeError<E>> {
        let edge = usize::from(self.config.brick_size);
        let mut sampled = 0;
        for brick in &mut self.bricks {
            let Some(values) = &mut brick.samples else {
                continue;
            };
            let spacing = i128::from(self.config.voxel_size_mm) * (1_i128 << brick.coord.mip);
            for (index, value) in values.iter_mut().enumerate() {
                let local = [index % edge, (index / edge) % edge, index / (edge * edge)];
                let bc = [brick.coord.x, brick.coord.y, brick.coord.z];
                let world = core::array::from_fn(|axis| {
                    i128::from(self.config.origin_mm[axis])
                        + (i128::from(bc[axis]) * edge as i128 + local[axis] as i128) * spacing
                });
                for source in &brick.sources {
                    let position = core::array::from_fn(|axis| {
                        world[axis] - i128::from(source.object.translation_mm[axis])
                    });
                    if (0..3).any(|axis| {
                        position[axis] < i128::from(source.object.local_min_mm[axis])
                            || position[axis] > i128::from(source.object.local_max_mm[axis])
                    }) {
                        continue;
                    }
                    let point = WorldSdfObjectSample {
                        object_id: source.id,
                        world_position_mm: world,
                        local_position_mm: position,
                        brick: brick.coord,
                        distance_scale_mm: self.config.distance_scale_mm,
                    };
                    let distance = sample(&source.object.source, point).map_err(|source| {
                        WorldSdfObjectBakeError {
                            sample: point,
                            source,
                        }
                    })?;
                    *value = (*value).min(distance);
                    sampled += 1;
                }
            }
        }
        Ok(WorldSdfObjectUpdate {
            work: self,
            sampled,
        })
    }
}

impl<T> WorldSdfObjectUpdate<T> {
    /// Changed coordinates for consumer invalidation; not completion receipts.
    pub fn changed_bricks(&self) -> impl ExactSizeIterator<Item = BrickCoord> + '_ {
        self.work.bricks.iter().map(|brick| brick.coord)
    }
}

#[cfg(test)]
mod tests;
