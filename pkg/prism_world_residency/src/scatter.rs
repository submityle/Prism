//! Rebuildable, deterministic CPU scatter over a global integer-millimetre grid.
//!
//! There is one hash-jittered candidate per global 3D grid voxel. Requests own
//! positions in `[min_mm, max_mm)`, not grid voxels; a candidate straddling a
//! request boundary is therefore emitted exactly once. Splitting, merging,
//! reordering or concurrently evaluating requests does not change membership.
//!
//! Optional minimum distance uses **Matérn type-II thinning of this jittered
//! lattice**: retain a candidate iff no original candidate within the strict
//! exclusion radius has a smaller `(hash_priority, full_candidate_key)` rank.
//! Neighbours are regenerated from global keys, including unrequested cells.
//! This is neither a Poisson input process, guaranteed blue noise, a maximal
//! packing nor greedy sequential dart throwing. A rejected candidate can still
//! reject another candidate; no recursive neighbour evaluation is needed.
//!
//! Only integer arithmetic, `alloc`, and the crate's stable PCG hash are used.
//! No terrain, asset, gameplay, persistence or residency policy is assumed.
//! Results are disposable derived caches, not scene truth. Discard/rebuild them
//! when their full [`ScatterCacheKey`] changes. Budgets are execution limits,
//! not generation inputs: failure never returns a budget-dependent prefix.
#![forbid(unsafe_code)]

use alloc::vec::Vec;
use core::{fmt, mem::size_of};

use crate::{pcg_hash_u32, SectorCoord};

/// Algorithm revision, including hashing, jitter, ranking and distance rules.
///
/// Bump this if any generated position or membership rule changes. The caller's
/// independent recipe revision is [`ScatterConfig::version`].
pub const SCATTER_KERNEL_VERSION: u32 = 1;

/// Pure generation inputs; no implicit world seed or request-local RNG state.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ScatterConfig {
    /// World/domain seed.
    pub seed: u64,
    /// Caller-owned recipe revision.
    pub version: u32,
    /// Caller-owned stream namespace, e.g. a procedural layer identifier.
    pub salt: u64,
    /// Positive cubic global grid spacing in integer millimetres.
    pub grid_spacing_mm: u32,
    /// Strict exclusion radius in mm; zero disables thinning.
    ///
    /// Equality is allowed: retained points are at least this far apart.
    /// Changing this affects membership/cache keys, not candidate identities.
    pub min_distance_mm: u32,
}

/// Half-open, axis-aligned world-space request in integer millimetres.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ScatterRegion {
    /// Inclusive minimum on X/Y/Z.
    pub min_mm: [i64; 3],
    /// Exclusive maximum on X/Y/Z. Equal endpoints make the region empty.
    pub max_mm: [i64; 3],
}

impl ScatterRegion {
    /// Validates half-open bounds; inverted axes are errors, not empty regions.
    pub fn new(min_mm: [i64; 3], max_mm: [i64; 3]) -> Result<Self, ScatterError> {
        let region = Self { min_mm, max_mm };
        region.validate()?;
        Ok(region)
    }

    /// Builds a cubic cell request relative to world origin `(0, 0, 0)`.
    ///
    /// Uses widened arithmetic and checks both endpoints. A cell whose bounds
    /// cannot fit `i64` returns an error rather than wrapping/clamping ownership.
    pub fn from_cell(cell: SectorCoord, cell_size_mm: u32) -> Result<Self, ScatterError> {
        if cell_size_mm == 0 {
            return Err(ScatterError::InvalidCellSize);
        }
        let mut min_mm = [0; 3];
        let mut max_mm = [0; 3];
        for (axis, coord) in [cell.x, cell.y, cell.z].into_iter().enumerate() {
            let start = i128::from(coord) * i128::from(cell_size_mm);
            let end = start + i128::from(cell_size_mm);
            min_mm[axis] = i64::try_from(start).map_err(|_| ScatterError::CoordinateOverflow)?;
            max_mm[axis] = i64::try_from(end).map_err(|_| ScatterError::CoordinateOverflow)?;
        }
        Self::new(min_mm, max_mm)
    }

    /// Tests exact half-open position ownership without floating-point rounding.
    pub fn contains_mm(self, position_mm: [i64; 3]) -> bool {
        (0..3).all(|axis| {
            position_mm[axis] >= self.min_mm[axis] && position_mm[axis] < self.max_mm[axis]
        })
    }

    fn validate(self) -> Result<(), ScatterError> {
        if (0..3).any(|axis| self.min_mm[axis] > self.max_mm[axis]) {
            return Err(ScatterError::InvalidRegion);
        }
        Ok(())
    }

    fn is_empty(self) -> bool {
        (0..3).any(|axis| self.min_mm[axis] == self.max_mm[axis])
    }
}

/// Collision-free *structural* stable ID of a candidate, independent of cells.
///
/// Use the complete key as identity, not a truncated hash or a Vec index. Fields
/// encode the generation namespace and global grid coordinate, not a persistence
/// object ID. Hash collisions affect distribution/rank only, never identity.
/// The key deliberately excludes the thinning radius and request bounds.
/// Rust memory layout is not a serialization format.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ScatterCandidateKey {
    /// Kernel algorithm revision.
    pub kernel_version: u32,
    /// World/domain seed.
    pub seed: u64,
    /// Caller-owned recipe revision.
    pub version: u32,
    /// Stream namespace.
    pub salt: u64,
    /// Grid spacing used to derive the candidate.
    pub grid_spacing_mm: u32,
    /// Global grid voxel coordinate, using Euclidean division for negatives.
    pub grid: SectorCoord,
}

impl ScatterCandidateKey {
    /// Reconstructs this candidate's position without accessing any cache/cell.
    ///
    /// This does not prove membership in a request or survival of thinning.
    /// Arbitrary keys whose jittered position is outside `i64` world coordinates
    /// are rejected. Only this kernel revision can be reconstructed here.
    pub fn rebuild(self) -> Result<ScatterPoint, ScatterError> {
        if self.kernel_version != SCATTER_KERNEL_VERSION {
            return Err(ScatterError::UnsupportedKernelVersion);
        }
        if self.grid_spacing_mm == 0 {
            return Err(ScatterError::InvalidGridSpacing);
        }
        candidate(self)
            .map(|ranked| ranked.point)
            .ok_or(ScatterError::CoordinateOverflow)
    }
}

/// A content-neutral world-space sample and its stable structural identity.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ScatterPoint {
    /// Full candidate identity; no dependency on persistence types.
    pub key: ScatterCandidateKey,
    /// Absolute X/Y/Z world position in signed integer millimetres.
    pub position_mm: [i64; 3],
}

/// Full, collision-free cache descriptor; comparison uses all fields.
///
/// A cache may be evicted at any time and reconstructed using [`scatter`].
/// External projection/filter inputs, if added by a consumer, require their
/// own invalidation keys. This key describes this kernel's unmodified output.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ScatterCacheKey {
    /// Algorithm revision.
    pub kernel_version: u32,
    /// All generation/membership inputs.
    pub config: ScatterConfig,
    /// Exact position ownership bounds.
    pub region: ScatterRegion,
}

/// Hard per-call limits; all must be explicitly supplied by the caller.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ScatterBudget {
    /// Maximum candidate probes, including repeated cross-cell neighbour probes.
    ///
    /// Checked conservatively before any allocation/enumeration, not just against
    /// accepted output. See [`ScatterWorkEstimate`].
    pub max_candidate_evaluations: u64,
    /// Maximum accepted output count. Exceeding it fails the entire request.
    pub max_output_points: usize,
    /// Maximum returned Vec capacity times `size_of::<ScatterPoint>()`.
    ///
    /// Includes spare element capacity, excludes the Vec header, stack usage
    /// and allocator-private metadata/rounding. This is an in-memory payload
    /// budget, not a portable wire-encoding size.
    pub max_output_bytes: usize,
}

/// Preflight upper bounds, computed with checked arithmetic and no allocation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ScatterWorkEstimate {
    /// Number of grid voxels intersecting the requested position bounds.
    pub primary_candidates: u64,
    /// Maximum neighbour probes per primary candidate (excludes self).
    pub max_neighbors_per_candidate: u64,
    /// Primary plus neighbour probes, including repeated regeneration.
    ///
    /// For `N` primary voxels, spacing `s`, radius `r > 0`, `h = ceil(r/s)`:
    /// `N * (2*h + 1)^3`; for `r = 0`, just `N`. Boundary clipping, candidates
    /// outside the requested bounds, and early rejection can reduce actual work.
    /// There is no recursive thinning or unbounded rejection sampling.
    pub max_candidate_evaluations: u64,
}

/// Complete derived output; no partial result is exposed on an error.
#[derive(Debug, Eq, PartialEq)]
pub struct ScatterOutput {
    /// Samples in lexicographic global grid X/Y/Z order, never hash-map order.
    pub points: Vec<ScatterPoint>,
    /// Inputs needed to validate/rebuild this disposable derived cache.
    pub cache_key: ScatterCacheKey,
    /// Actual primary and neighbour probes; at most the preflight estimate.
    pub candidate_evaluations: u64,
}

/// Validation, arithmetic, allocation or hard-budget failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScatterError {
    /// Grid spacing must be positive.
    InvalidGridSpacing,
    /// Cell edge length must be positive.
    InvalidCellSize,
    /// A request axis is inverted.
    InvalidRegion,
    /// A requested cell endpoint or reconstructed position does not fit `i64`.
    CoordinateOverflow,
    /// A count, byte size or cost bound cannot fit its integer representation.
    ArithmeticOverflow,
    /// Rebuilding a candidate requires a different algorithm revision.
    UnsupportedKernelVersion,
    /// Preflight worst-case probes exceed the caller's work budget.
    CandidateBudgetExceeded {
        /// Conservative required probe count.
        required: u64,
        /// Supplied limit.
        limit: u64,
    },
    /// Accepting the next survivor would exceed the output count limit.
    OutputCountBudgetExceeded {
        /// Required count when the violation was detected.
        required: usize,
        /// Supplied limit.
        limit: usize,
    },
    /// Accepting a survivor or reserving capacity would exceed the byte limit.
    OutputBytesBudgetExceeded {
        /// Required in-memory payload bytes.
        required: usize,
        /// Supplied limit.
        limit: usize,
    },
    /// Fallible reservation failed, including an unrepresentable Vec capacity.
    AllocationFailed,
}

impl fmt::Display for ScatterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidGridSpacing => f.write_str("scatter grid spacing must be positive"),
            Self::InvalidCellSize => f.write_str("scatter cell size must be positive"),
            Self::InvalidRegion => f.write_str("scatter region has an inverted axis"),
            Self::CoordinateOverflow => {
                f.write_str("scatter coordinate is outside i64 millimetres")
            }
            Self::ArithmeticOverflow => f.write_str("scatter count or cost arithmetic overflow"),
            Self::UnsupportedKernelVersion => f.write_str("unsupported scatter kernel revision"),
            Self::CandidateBudgetExceeded { required, limit } => {
                write!(
                    f,
                    "scatter requires up to {required} probes, budget is {limit}"
                )
            }
            Self::OutputCountBudgetExceeded { required, limit } => {
                write!(
                    f,
                    "scatter needs {required} output points, budget is {limit}"
                )
            }
            Self::OutputBytesBudgetExceeded { required, limit } => {
                write!(
                    f,
                    "scatter needs {required} output bytes, budget is {limit}"
                )
            }
            Self::AllocationFailed => f.write_str("scatter output allocation failed"),
        }
    }
}

impl core::error::Error for ScatterError {}

/// Estimates bounded candidate work without generating or allocating anything.
pub fn estimate_scatter_work(
    config: ScatterConfig,
    region: ScatterRegion,
) -> Result<ScatterWorkEstimate, ScatterError> {
    Ok(Layout::new(config, region)?.estimate)
}

/// Generates a complete deterministic, content-neutral scatter request.
///
/// Allocates only the returned Vec with one fallible exact reservation, bounded
/// by `min(primary_count, max_output_points, max_output_bytes / point_size)`.
/// It may therefore reserve more than the final survivor count; there is no
/// hidden neighbour cache/allocation or unchecked Vec growth. An insufficient
/// budget produces an error, never thinning or truncation dependent on limits.
///
/// Candidate positions outside signed `i64` mm are absent from the global
/// candidate universe; edge voxels are not clamped into the world. Halo work is
/// clipped to grid voxels intersecting that universe. Because region maximums
/// are exclusive `i64`, the single coordinate `i64::MAX` cannot itself be owned
/// by a request, but valid candidates there still participate as neighbours.
pub fn scatter(
    config: ScatterConfig,
    region: ScatterRegion,
    budget: ScatterBudget,
) -> Result<ScatterOutput, ScatterError> {
    let layout = Layout::new(config, region)?;
    if layout.estimate.max_candidate_evaluations > budget.max_candidate_evaluations {
        return Err(ScatterError::CandidateBudgetExceeded {
            required: layout.estimate.max_candidate_evaluations,
            limit: budget.max_candidate_evaluations,
        });
    }
    let point_size = size_of::<ScatterPoint>();
    let capacity = usize::try_from(layout.estimate.primary_candidates)
        .unwrap_or(usize::MAX)
        .min(budget.max_output_points)
        .min(budget.max_output_bytes / point_size);
    let mut points = Vec::new();
    // Reservation is exact to avoid Vec's geometric growth exceeding byte caps.
    // Verify reported capacity too: allocators may grant additional capacity.
    checked_output_bytes(capacity, budget.max_output_bytes)?;
    points
        .try_reserve_exact(capacity)
        .map_err(|_| ScatterError::AllocationFailed)?;
    checked_output_bytes(points.capacity(), budget.max_output_bytes)?;
    let mut evaluations = 0_u64;
    if layout.estimate.primary_candidates != 0 {
        for x in layout.min[0]..=layout.max[0] {
            for y in layout.min[1]..=layout.max[1] {
                for z in layout.min[2]..=layout.max[2] {
                    count_probe(&mut evaluations)?;
                    let Some(current) = candidate(key(config, SectorCoord::new(x, y, z))) else {
                        continue;
                    };
                    if !region.contains_mm(current.point.position_mm) {
                        continue;
                    }
                    if config.min_distance_mm != 0
                        && !survives(config, current, layout.halo, &mut evaluations)?
                    {
                        continue;
                    }
                    let required = points
                        .len()
                        .checked_add(1)
                        .ok_or(ScatterError::ArithmeticOverflow)?;
                    if required > budget.max_output_points {
                        return Err(ScatterError::OutputCountBudgetExceeded {
                            required,
                            limit: budget.max_output_points,
                        });
                    }
                    checked_output_bytes(required, budget.max_output_bytes)?;
                    // Any survivor fitting both output caps fits the reservation:
                    // there cannot be more survivors than primary candidates.
                    debug_assert!(required <= points.capacity());
                    points.push(current.point);
                }
            }
        }
    }
    debug_assert!(evaluations <= layout.estimate.max_candidate_evaluations);
    Ok(ScatterOutput {
        points,
        cache_key: ScatterCacheKey {
            kernel_version: SCATTER_KERNEL_VERSION,
            config,
            region,
        },
        candidate_evaluations: evaluations,
    })
}

fn checked_output_bytes(count: usize, limit: usize) -> Result<usize, ScatterError> {
    let required = count
        .checked_mul(size_of::<ScatterPoint>())
        .ok_or(ScatterError::ArithmeticOverflow)?;
    if required > limit {
        return Err(ScatterError::OutputBytesBudgetExceeded { required, limit });
    }
    Ok(required)
}

fn count_probe(count: &mut u64) -> Result<(), ScatterError> {
    *count = count
        .checked_add(1)
        .ok_or(ScatterError::ArithmeticOverflow)?;
    Ok(())
}

struct Layout {
    min: [i64; 3],
    max: [i64; 3],
    halo: i64,
    estimate: ScatterWorkEstimate,
}

impl Layout {
    fn new(config: ScatterConfig, region: ScatterRegion) -> Result<Self, ScatterError> {
        if config.grid_spacing_mm == 0 {
            return Err(ScatterError::InvalidGridSpacing);
        }
        region.validate()?;
        let mut layout = Self {
            min: [0; 3],
            max: [0; 3],
            halo: 0,
            estimate: ScatterWorkEstimate {
                primary_candidates: 0,
                max_neighbors_per_candidate: 0,
                max_candidate_evaluations: 0,
            },
        };
        if region.is_empty() {
            return Ok(layout);
        }
        let spacing = i64::from(config.grid_spacing_mm);
        let mut count = 1_u64;
        for axis in 0..3 {
            layout.min[axis] = region.min_mm[axis].div_euclid(spacing);
            // Non-empty ordered bounds imply max > MIN; subtraction is safe.
            let last = region.max_mm[axis]
                .checked_sub(1)
                .ok_or(ScatterError::ArithmeticOverflow)?;
            layout.max[axis] = last.div_euclid(spacing);
            let length = i128::from(layout.max[axis]) - i128::from(layout.min[axis]) + 1;
            let length = u64::try_from(length).map_err(|_| ScatterError::ArithmeticOverflow)?;
            count = count
                .checked_mul(length)
                .ok_or(ScatterError::ArithmeticOverflow)?;
        }
        // If |grid difference| > ceil(r/s), even the closest jittered
        // integer-mm positions cannot be strictly within radius r.
        let halo = u64::from(config.min_distance_mm).div_ceil(u64::from(config.grid_spacing_mm));
        layout.halo = i64::try_from(halo).map_err(|_| ScatterError::ArithmeticOverflow)?;
        let side = halo
            .checked_mul(2)
            .and_then(|value| value.checked_add(1))
            .ok_or(ScatterError::ArithmeticOverflow)?;
        let probes = side
            .checked_mul(side)
            .and_then(|value| value.checked_mul(side))
            .ok_or(ScatterError::ArithmeticOverflow)?;
        layout.estimate = ScatterWorkEstimate {
            primary_candidates: count,
            max_neighbors_per_candidate: probes - 1,
            max_candidate_evaluations: count
                .checked_mul(probes)
                .ok_or(ScatterError::ArithmeticOverflow)?,
        };
        Ok(layout)
    }
}

fn key(config: ScatterConfig, grid: SectorCoord) -> ScatterCandidateKey {
    ScatterCandidateKey {
        kernel_version: SCATTER_KERNEL_VERSION,
        seed: config.seed,
        version: config.version,
        salt: config.salt,
        grid_spacing_mm: config.grid_spacing_mm,
        grid,
    }
}

fn hash64(value: u64) -> u64 {
    (u64::from(pcg_hash_u32(value)) << 32) | u64::from(pcg_hash_u32(value ^ 0x9e37_79b9_7f4a_7c15))
}

fn candidate_hash(key: ScatterCandidateKey) -> u64 {
    let mut hash = hash64(0x7363_6174_7465_7200 ^ u64::from(key.kernel_version));
    // Ordered, fixed-width, integer mixing: no native-endian byte casts or
    // platform Hash trait implementation participates in generation.
    for value in [
        key.seed,
        u64::from(key.version),
        key.salt,
        u64::from(key.grid_spacing_mm),
        key.grid.x as u64,
        key.grid.y as u64,
        key.grid.z as u64,
    ] {
        hash = hash64(hash ^ value);
    }
    hash
}

#[derive(Clone, Copy)]
struct RankedCandidate {
    point: ScatterPoint,
    priority: u64,
}

fn candidate(key: ScatterCandidateKey) -> Option<RankedCandidate> {
    let hash = candidate_hash(key);
    let mut position_mm = [0; 3];
    for (axis, (coord, tag)) in [key.grid.x, key.grid.y, key.grid.z]
        .into_iter()
        .zip([
            0x243f_6a88_85a3_08d3,
            0x1319_8a2e_0370_7344,
            0xa409_3822_299f_31d0,
        ])
        .enumerate()
    {
        // Multiply-high gives bounded [0, spacing) jitter in one hash step;
        // it is not rejection sampling and does not claim exact unbiasedness.
        let offset = (u128::from(hash64(hash ^ tag)) * u128::from(key.grid_spacing_mm)) >> 64;
        let position = i128::from(coord) * i128::from(key.grid_spacing_mm) + offset as i128;
        position_mm[axis] = i64::try_from(position).ok()?;
    }
    Some(RankedCandidate {
        point: ScatterPoint { key, position_mm },
        priority: hash64(hash ^ 0x082e_fa98_ec4e_6c89),
    })
}

fn precedes(a: RankedCandidate, b: RankedCandidate) -> bool {
    (a.priority, a.point.key) < (b.priority, b.point.key)
}

fn strictly_within(a: [i64; 3], b: [i64; 3], radius: u32) -> bool {
    let radius = u128::from(radius);
    let radius_squared = radius * radius;
    let mut squared = 0_u128;
    for axis in 0..3 {
        let delta = (i128::from(a[axis]) - i128::from(b[axis])).unsigned_abs();
        if delta >= radius {
            return false;
        }
        // delta < radius <= u32::MAX: three squares fit u128. Early exit
        // additionally ensures the running sum never exceeds two r^2 terms.
        squared += delta * delta;
        if squared >= radius_squared {
            return false;
        }
    }
    squared < radius_squared
}

fn survives(
    config: ScatterConfig,
    current: RankedCandidate,
    halo: i64,
    evaluations: &mut u64,
) -> Result<bool, ScatterError> {
    let spacing = i128::from(config.grid_spacing_mm);
    let world_min = i128::from(i64::MIN).div_euclid(spacing);
    let world_max = i128::from(i64::MAX).div_euclid(spacing);
    let mut min = [0_i64; 3];
    let mut max = [0_i64; 3];
    for (axis, coord) in [
        current.point.key.grid.x,
        current.point.key.grid.y,
        current.point.key.grid.z,
    ]
    .into_iter()
    .enumerate()
    {
        min[axis] = i64::try_from((i128::from(coord) - i128::from(halo)).max(world_min))
            .map_err(|_| ScatterError::CoordinateOverflow)?;
        max[axis] = i64::try_from((i128::from(coord) + i128::from(halo)).min(world_max))
            .map_err(|_| ScatterError::CoordinateOverflow)?;
    }
    for x in min[0]..=max[0] {
        for y in min[1]..=max[1] {
            for z in min[2]..=max[2] {
                let grid = SectorCoord::new(x, y, z);
                if grid == current.point.key.grid {
                    continue;
                }
                count_probe(evaluations)?;
                let Some(neighbor) = candidate(key(config, grid)) else {
                    continue;
                };
                if precedes(neighbor, current)
                    && strictly_within(
                        neighbor.point.position_mm,
                        current.point.position_mm,
                        config.min_distance_mm,
                    )
                {
                    return Ok(false);
                }
            }
        }
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::{collections::BTreeSet, vec};

    fn config() -> ScatterConfig {
        ScatterConfig {
            seed: 0x1234_5678_9abc_def0,
            version: 7,
            salt: 91,
            grid_spacing_mm: 17,
            min_distance_mm: 0,
        }
    }

    fn budget() -> ScatterBudget {
        ScatterBudget {
            max_candidate_evaluations: 5_000_000,
            max_output_points: 20_000,
            max_output_bytes: 20_000 * size_of::<ScatterPoint>(),
        }
    }

    fn region(min: [i64; 3], max: [i64; 3]) -> ScatterRegion {
        ScatterRegion::new(min, max).unwrap()
    }

    fn sorted(mut points: Vec<ScatterPoint>) -> Vec<ScatterPoint> {
        points.sort_unstable_by_key(|point| point.key);
        points
    }

    #[test]
    fn jitter_is_repeatable_bounded_and_rebuildable() {
        let config = config();
        let bounds = region([-51; 3], [51; 3]);
        let first = scatter(config, bounds, budget()).unwrap();
        assert_eq!(first, scatter(config, bounds, budget()).unwrap());
        assert_eq!(first.points.len(), 216);
        assert_eq!(first.candidate_evaluations, 216);
        assert!(first
            .points
            .windows(2)
            .all(|pair| pair[0].key < pair[1].key));
        for point in first.points {
            assert_eq!(point.key.rebuild(), Ok(point));
            assert!(bounds.contains_mm(point.position_mm));
            for (axis, grid) in [point.key.grid.x, point.key.grid.y, point.key.grid.z]
                .into_iter()
                .enumerate()
            {
                assert_eq!(point.position_mm[axis].div_euclid(17), grid);
            }
        }
    }

    #[test]
    fn negative_voxels_use_euclidean_division() {
        let config = ScatterConfig {
            grid_spacing_mm: 10,
            ..config()
        };
        let out = scatter(config, region([-10, 0, 0], [0, 10, 10]), budget()).unwrap();
        assert_eq!(out.points.len(), 1);
        assert_eq!(out.points[0].key.grid, SectorCoord::new(-1, 0, 0));
        assert!((-10..0).contains(&out.points[0].position_mm[0]));
        let bounds = region([-1, -1, -1], [1, 1, 1]);
        assert_eq!(
            estimate_scatter_work(config, bounds)
                .unwrap()
                .primary_candidates,
            8
        );
    }

    #[test]
    fn cell_bounds_are_half_open_and_checked() {
        let bounds = ScatterRegion::from_cell(SectorCoord::new(-2, 0, 1), 10).unwrap();
        assert_eq!(bounds, region([-20, 0, 10], [-10, 10, 20]));
        assert!(bounds.contains_mm([-20, 0, 10]));
        for point in [[-10, 0, 10], [-20, 10, 10], [-20, 0, 20]] {
            assert!(!bounds.contains_mm(point));
        }
        assert_eq!(
            ScatterRegion::from_cell(SectorCoord::new(0, 0, 0), 0),
            Err(ScatterError::InvalidCellSize)
        );
        assert_eq!(
            ScatterRegion::from_cell(SectorCoord::new(i64::MAX, 0, 0), 1),
            Err(ScatterError::CoordinateOverflow)
        );
        assert_eq!(
            ScatterRegion::from_cell(SectorCoord::new(i64::MIN, 0, 0), 2),
            Err(ScatterError::CoordinateOverflow)
        );
    }

    #[test]
    fn every_integer_boundary_has_exactly_one_owner() {
        let config = ScatterConfig {
            grid_spacing_mm: 1,
            ..config()
        };
        let whole = scatter(config, region([-2, 0, 0], [2, 1, 1]), budget()).unwrap();
        assert_eq!(whole.points.len(), 4);
        for x in -2..2 {
            let cell = scatter(config, region([x, 0, 0], [x + 1, 1, 1]), budget()).unwrap();
            assert_eq!(cell.points.len(), 1);
            assert_eq!(cell.points[0].position_mm, [x, 0, 0]);
            assert!(whole.points.contains(&cell.points[0]));
        }
    }

    #[test]
    fn arbitrary_partitions_match_monolithic_in_both_request_orders() {
        for radius in [0, 11, 29] {
            let config = ScatterConfig {
                min_distance_mm: radius,
                ..config()
            };
            let whole = scatter(config, region([-35, -23, -19], [40, 39, 31]), budget()).unwrap();
            assert!(!whole.points.is_empty());
            let mut requests = Vec::new();
            for xs in [-35, -3, 12, 40].windows(2) {
                for ys in [-23, 1, 39].windows(2) {
                    for zs in [-19, 4, 31].windows(2) {
                        requests.push(region([xs[0], ys[0], zs[0]], [xs[1], ys[1], zs[1]]));
                    }
                }
            }
            let mut forward = Vec::new();
            let mut reverse = Vec::new();
            for request in &requests {
                forward.extend(scatter(config, *request, budget()).unwrap().points);
            }
            for request in requests.iter().rev() {
                reverse.extend(scatter(config, *request, budget()).unwrap().points);
            }
            let unique: BTreeSet<_> = forward.iter().map(|point| point.key).collect();
            assert_eq!(unique.len(), forward.len());
            assert_eq!(sorted(forward), whole.points);
            assert_eq!(sorted(reverse), whole.points);
        }
    }

    #[test]
    fn thinning_matches_independent_oversized_global_reference() {
        let config = ScatterConfig {
            min_distance_mm: 29,
            ..config()
        };
        let bounds = region([-51; 3], [51; 3]);
        // Enumerate a deliberately oversized fixed global neighbourhood rather
        // than reuse Layout/survives. Every possible blocker is inside this cube.
        let mut universe = Vec::new();
        for x in -5..=5 {
            for y in -5..=5 {
                for z in -5..=5 {
                    universe.push(candidate(key(config, SectorCoord::new(x, y, z))).unwrap());
                }
            }
        }
        let expected: Vec<_> = universe
            .iter()
            .filter_map(|current| {
                if !bounds.contains_mm(current.point.position_mm) {
                    return None;
                }
                let blocked = universe.iter().any(|other| {
                    let mut squared = 0_i128;
                    for axis in 0..3 {
                        let delta = i128::from(current.point.position_mm[axis])
                            - i128::from(other.point.position_mm[axis]);
                        squared += delta * delta;
                    }
                    precedes(*other, *current) && squared < 29 * 29
                });
                (!blocked).then_some(current.point)
            })
            .collect();
        assert!(!expected.is_empty());
        assert_eq!(scatter(config, bounds, budget()).unwrap().points, expected);
    }

    #[test]
    fn blockers_in_unrequested_cells_are_not_ignored() {
        let config = ScatterConfig {
            min_distance_mm: 29,
            ..config()
        };
        // Singleton ownership regions expose mistakes that inspect only their
        // own request. At least one candidate must be removed by an outside key.
        let mut saw_external_blocker = false;
        for x in -2..=2 {
            for y in -2..=2 {
                let point = key(config, SectorCoord::new(x, y, 0)).rebuild().unwrap();
                let max = point.position_mm.map(|coord| coord + 1);
                let bounds = region(point.position_mm, max);
                let out = scatter(config, bounds, budget()).unwrap();
                if out.points.is_empty() {
                    saw_external_blocker = true;
                }
                assert_eq!(
                    out.points,
                    scatter(config, bounds, budget()).unwrap().points
                );
            }
        }
        assert!(saw_external_blocker);
    }

    #[test]
    fn retained_samples_obey_minimum_distance_across_cell_seams() {
        let config = ScatterConfig {
            min_distance_mm: 23,
            ..config()
        };
        let mut points = Vec::new();
        for x in -2..=1 {
            let bounds = ScatterRegion::from_cell(SectorCoord::new(x, 0, 0), 35).unwrap();
            points.extend(scatter(config, bounds, budget()).unwrap().points);
        }
        assert!(points.len() > 1);
        for (index, a) in points.iter().enumerate() {
            for b in &points[index + 1..] {
                let squared: i128 = (0..3)
                    .map(|axis| {
                        let delta =
                            i128::from(a.position_mm[axis]) - i128::from(b.position_mm[axis]);
                        delta * delta
                    })
                    .sum();
                assert!(squared >= 23 * 23);
                assert_ne!(a.key, b.key);
            }
        }
    }

    #[test]
    fn rank_hash_ties_use_the_full_global_key() {
        let a = RankedCandidate {
            point: key(config(), SectorCoord::new(-1, 0, 0)).rebuild().unwrap(),
            priority: 5,
        };
        let b = RankedCandidate {
            point: key(config(), SectorCoord::new(0, 0, 0)).rebuild().unwrap(),
            priority: 5,
        };
        assert!(precedes(a, b));
        assert!(!precedes(b, a));
        assert!(!precedes(a, a));
    }

    #[test]
    fn strict_distance_is_integer_exact_and_overflow_safe() {
        assert!(!strictly_within([0; 3], [3, 4, 0], 5));
        assert!(strictly_within([0; 3], [3, 4, 0], 6));
        assert!(!strictly_within([0; 3], [0; 3], 0));
        assert!(!strictly_within([i64::MIN; 3], [i64::MAX; 3], u32::MAX));
        assert!(!strictly_within([0; 3], [i64::from(u32::MAX); 3], u32::MAX));
        assert!(strictly_within(
            [0; 3],
            [i64::from(u32::MAX) - 1, 0, 0],
            u32::MAX
        ));
    }

    #[test]
    fn namespace_inputs_change_identity_and_jitter() {
        let original = config();
        let grid = SectorCoord::new(-7, 4, 13);
        let a = key(original, grid).rebuild().unwrap();
        for changed in [
            ScatterConfig {
                seed: original.seed + 1,
                ..original
            },
            ScatterConfig {
                version: original.version + 1,
                ..original
            },
            ScatterConfig {
                salt: original.salt + 1,
                ..original
            },
            ScatterConfig {
                grid_spacing_mm: 19,
                ..original
            },
        ] {
            let b = key(changed, grid).rebuild().unwrap();
            assert_ne!(a.key, b.key);
            assert_ne!(a.position_mm, b.position_mm);
        }
        let radius_changed = ScatterConfig {
            min_distance_mm: 19,
            ..original
        };
        assert_eq!(key(original, grid), key(radius_changed, grid));
    }

    #[test]
    fn cache_is_rebuildable_and_budget_independent() {
        let config = config();
        let bounds = region([-17; 3], [17; 3]);
        let a = scatter(config, bounds, budget()).unwrap();
        let exact = ScatterBudget {
            max_candidate_evaluations: 8,
            max_output_points: 8,
            max_output_bytes: 8 * size_of::<ScatterPoint>(),
        };
        let b = scatter(config, bounds, exact).unwrap();
        assert_eq!(a, b);
        let key = a.cache_key;
        assert_eq!(a, scatter(key.config, key.region, exact).unwrap());
        let changed = scatter(
            ScatterConfig {
                min_distance_mm: 1,
                ..config
            },
            bounds,
            budget(),
        )
        .unwrap();
        assert_ne!(a.cache_key, changed.cache_key);
        assert_eq!(a.points, changed.points);
    }

    #[test]
    fn work_estimate_counts_halo_and_actual_never_exceeds_bound() {
        for radius in [0, 1, 17, 18, 34, 35] {
            let config = ScatterConfig {
                min_distance_mm: radius,
                ..config()
            };
            let bounds = region([-17; 3], [17; 3]);
            let estimate = estimate_scatter_work(config, bounds).unwrap();
            let h = u64::from(radius).div_ceil(17);
            let probes = (2 * h + 1).pow(3);
            assert_eq!(estimate.primary_candidates, 8);
            assert_eq!(estimate.max_neighbors_per_candidate, probes - 1);
            assert_eq!(estimate.max_candidate_evaluations, 8 * probes);
            let out = scatter(config, bounds, budget()).unwrap();
            assert!(out.candidate_evaluations <= estimate.max_candidate_evaluations);
        }
    }

    #[test]
    fn candidate_budget_is_checked_before_output_allocation() {
        let config = ScatterConfig {
            min_distance_mm: 18,
            ..config()
        };
        let bounds = region([0; 3], [17; 3]);
        let limited = ScatterBudget {
            max_candidate_evaluations: 124,
            max_output_points: 0,
            max_output_bytes: 0,
        };
        assert_eq!(
            scatter(config, bounds, limited),
            Err(ScatterError::CandidateBudgetExceeded {
                required: 125,
                limit: 124
            })
        );
    }

    #[test]
    fn output_count_and_bytes_fail_instead_of_truncating() {
        let config = ScatterConfig {
            grid_spacing_mm: 1,
            ..config()
        };
        let bounds = region([0; 3], [2, 1, 1]);
        let by_count = ScatterBudget {
            max_output_points: 1,
            ..budget()
        };
        assert_eq!(
            scatter(config, bounds, by_count),
            Err(ScatterError::OutputCountBudgetExceeded {
                required: 2,
                limit: 1
            })
        );
        let size = size_of::<ScatterPoint>();
        let by_bytes = ScatterBudget {
            max_output_bytes: 2 * size - 1,
            ..budget()
        };
        assert_eq!(
            scatter(config, bounds, by_bytes),
            Err(ScatterError::OutputBytesBudgetExceeded {
                required: 2 * size,
                limit: 2 * size - 1
            })
        );
        let exact = ScatterBudget {
            max_output_points: 2,
            max_output_bytes: 2 * size,
            ..budget()
        };
        let out = scatter(config, bounds, exact).unwrap();
        assert_eq!(out.points.len(), 2);
        assert!(out.points.capacity() * size <= exact.max_output_bytes);
        assert_eq!(
            checked_output_bytes(usize::MAX, usize::MAX),
            Err(ScatterError::ArithmeticOverflow)
        );
    }

    #[test]
    fn empty_regions_need_no_work_or_output_budget() {
        let config = ScatterConfig {
            min_distance_mm: u32::MAX,
            ..config()
        };
        let zero = ScatterBudget {
            max_candidate_evaluations: 0,
            max_output_points: 0,
            max_output_bytes: 0,
        };
        for bounds in [
            region([i64::MIN; 3], [i64::MIN, i64::MAX, i64::MAX]),
            region([3; 3], [3; 3]),
        ] {
            let out = scatter(config, bounds, zero).unwrap();
            assert_eq!(out.points.capacity(), 0);
            assert_eq!(out.candidate_evaluations, 0);
        }
    }

    #[test]
    fn invalid_inputs_and_extreme_costs_return_errors_without_enumeration() {
        let bounds = region([0; 3], [1; 3]);
        assert_eq!(
            scatter(
                ScatterConfig {
                    grid_spacing_mm: 0,
                    ..config()
                },
                bounds,
                budget()
            ),
            Err(ScatterError::InvalidGridSpacing)
        );
        assert_eq!(
            ScatterRegion::new([1, 0, 0], [0, 1, 1]),
            Err(ScatterError::InvalidRegion)
        );
        let invalid = ScatterRegion {
            min_mm: [1; 3],
            max_mm: [0; 3],
        };
        assert_eq!(
            scatter(config(), invalid, budget()),
            Err(ScatterError::InvalidRegion)
        );
        let huge = region([i64::MIN; 3], [i64::MAX; 3]);
        assert_eq!(
            estimate_scatter_work(
                ScatterConfig {
                    grid_spacing_mm: 1,
                    ..config()
                },
                huge
            ),
            Err(ScatterError::ArithmeticOverflow)
        );
        assert_eq!(
            estimate_scatter_work(
                ScatterConfig {
                    grid_spacing_mm: 1,
                    min_distance_mm: u32::MAX,
                    ..config()
                },
                bounds
            ),
            Err(ScatterError::ArithmeticOverflow)
        );
    }

    #[test]
    fn i64_edges_are_not_wrapped_and_neighbor_halos_are_clipped() {
        let config = ScatterConfig {
            grid_spacing_mm: 1,
            ..config()
        };
        for start in [i64::MIN, i64::MAX - 2] {
            let bounds = region([start, 0, 0], [start + 2, 1, 1]);
            let out = scatter(config, bounds, budget()).unwrap();
            assert_eq!(out.points.len(), 2);
            assert_eq!(out.points[0].position_mm, [start, 0, 0]);
            assert_eq!(out.points[1].position_mm, [start + 1, 0, 0]);
            let thin = ScatterConfig {
                min_distance_mm: 2,
                ..config
            };
            let out = scatter(thin, bounds, budget()).unwrap();
            let estimate = estimate_scatter_work(thin, bounds).unwrap();
            assert!(out.candidate_evaluations <= estimate.max_candidate_evaluations);
        }
        let non_unit = ScatterConfig {
            grid_spacing_mm: u32::MAX,
            ..config
        };
        let bounds = region([i64::MIN, 0, 0], [i64::MIN + 1, 1, 1]);
        assert!(scatter(non_unit, bounds, budget()).is_ok());
        let invalid_key = ScatterCandidateKey {
            grid: SectorCoord::new(i64::MAX, 0, 0),
            ..key(non_unit, SectorCoord::new(0, 0, 0))
        };
        assert_eq!(invalid_key.rebuild(), Err(ScatterError::CoordinateOverflow));
    }

    #[test]
    fn unsupported_rebuild_versions_fail_explicitly() {
        let a = key(config(), SectorCoord::new(0, 0, 0));
        assert_eq!(
            ScatterCandidateKey {
                kernel_version: 0,
                ..a
            }
            .rebuild(),
            Err(ScatterError::UnsupportedKernelVersion)
        );
        assert_eq!(
            ScatterCandidateKey {
                grid_spacing_mm: 0,
                ..a
            }
            .rebuild(),
            Err(ScatterError::InvalidGridSpacing)
        );
    }

    #[test]
    fn structural_identity_does_not_depend_on_hash_uniqueness() {
        let a = key(config(), SectorCoord::new(0, 0, 0));
        let b = key(config(), SectorCoord::new(1, 0, 0));
        let keys: BTreeSet<_> = vec![a, b, a].into_iter().collect();
        assert_eq!(keys.len(), 2);
    }
}
