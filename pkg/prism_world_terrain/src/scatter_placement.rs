//! Deterministic two-dimensional candidates projected onto a real heightfield.
//!
//! Unlike heightfield thinning, placement changes Y and has its own identity,
//! output type, cache and horizontal distance rule. No three-dimensional source
//! point is relabelled as if it still rebuilt to the projected position.

use alloc::vec::Vec;
use core::{fmt, mem::size_of};

use prism_world_residency::{
    pcg_hash_u32, ScatterCandidateKey, ScatterDensityField, ScatterError, ScatterFieldSample,
    SectorCoord, SCATTER_FIELD_VERSION, SCATTER_KERNEL_VERSION,
};

use crate::{
    scatter_filter::slope_within, HeightfieldQueryError, HeightfieldSurfaceSample,
    PlanarHeightfieldSurface, HEIGHTFIELD_SURFACE_VERSION,
};

/// Revision of planar candidate selection, ranking, projection and rounding.
pub const HEIGHTFIELD_PLACEMENT_VERSION: u32 = 1;

/// Explicit recipe for one candidate per global XZ grid square.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HeightfieldPlacementConfig {
    /// Domain seed; no request-local random state is used.
    pub seed: u64,
    /// Caller-owned recipe revision.
    pub version: u32,
    /// Caller-owned stream namespace.
    pub salt: u64,
    /// Positive horizontal grid spacing in millimetres.
    pub grid_spacing_mm: u32,
    /// Strict horizontal exclusion radius; equality is allowed, zero disables it.
    ///
    /// Matérn-II thinning compares original XZ candidates, including unrequested
    /// and subsequently field/terrain-rejected candidates. It is not maximal
    /// filling, a Poisson input process, or a blue-noise quality guarantee.
    pub min_distance_mm: u32,
    /// Integer vertical offset added to the exact surface before rounding.
    pub vertical_offset_mm: i64,
}

/// Half-open XZ ownership only; projection has no request-local Y slab.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HeightfieldPlacementRegion {
    /// Inclusive world X/Z minimum.
    pub min_xz_mm: [i64; 2],
    /// Exclusive world X/Z maximum; either equal axis makes the region empty.
    pub max_xz_mm: [i64; 2],
}

impl HeightfieldPlacementRegion {
    /// Rejects inverted axes without allocating or querying terrain.
    pub fn new(
        min_xz_mm: [i64; 2],
        max_xz_mm: [i64; 2],
    ) -> Result<Self, HeightfieldPlacementError> {
        let region = Self {
            min_xz_mm,
            max_xz_mm,
        };
        region.validate()?;
        Ok(region)
    }

    /// Exact half-open horizontal position ownership.
    pub fn contains(self, xz_mm: [i64; 2]) -> bool {
        (0..2).all(|axis| xz_mm[axis] >= self.min_xz_mm[axis] && xz_mm[axis] < self.max_xz_mm[axis])
    }

    fn validate(self) -> Result<(), HeightfieldPlacementError> {
        if (0..2).any(|axis| self.min_xz_mm[axis] > self.max_xz_mm[axis]) {
            return Err(HeightfieldPlacementError::InvalidRegion);
        }
        Ok(())
    }
}

/// Stable placement identity, separate from an unmodified 3D scatter point.
///
/// The source key always has grid Y zero. Its X/Z jitter is reused, but its Y
/// jitter is discarded. Full placement version plus full source key, under the
/// scene-owned placement node, are identity; a hash or dense slot is not.
/// Terrain, offset, minimum distance and filters change geometry/membership,
/// not identity. This Rust structure is not a persistence wire encoding.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct HeightfieldPlacementKey {
    /// Placement algorithm revision.
    pub placement_version: u32,
    /// Full canonical raw candidate key, constrained to grid Y zero.
    pub source: ScatterCandidateKey,
}

impl HeightfieldPlacementKey {
    /// Rebuilds geometry against an explicit immutable terrain and offset.
    ///
    /// Does not prove request ownership, minimum-distance survival, filters or
    /// author-field acceptance. Calling `source.rebuild()` alone returns the
    /// original 3D jitter, not this projected geometry.
    pub fn rebuild(
        self,
        surface: &PlanarHeightfieldSurface,
        vertical_offset_mm: i64,
    ) -> Result<HeightfieldPlacementPoint, HeightfieldPlacementError> {
        if self.placement_version != HEIGHTFIELD_PLACEMENT_VERSION || self.source.grid.y != 0 {
            return Err(HeightfieldPlacementError::InvalidKey);
        }
        let raw = self.source.rebuild()?;
        project(
            self,
            [raw.position_mm[0], raw.position_mm[2]],
            surface,
            vertical_offset_mm,
        )
    }
}

/// Projected geometry; intentionally not a residency `ScatterPoint`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HeightfieldPlacementPoint {
    /// Full placement identity.
    pub key: HeightfieldPlacementKey,
    /// Integer world position, with Y rounded to nearest mm, ties away from zero.
    ///
    /// Rounding applies to exact surface height plus offset, with at most 0.5mm
    /// error. Overflow fails the entire request rather than clamping geometry.
    pub position_mm: [i64; 3],
    /// Exact unoffset height and upward geometric face normal at this XZ.
    pub surface_sample: HeightfieldSurfaceSample,
}

/// Exact closed surface-height and rational geometric-slope restrictions.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HeightfieldPlacementFilters {
    /// Allowed unrounded, unoffset world surface height in millimetres.
    pub surface_height_mm: [i64; 2],
    /// Maximum nonnegative rise numerator; zero permits horizontal faces only.
    pub max_slope_rise: u32,
    /// Positive horizontal run denominator.
    pub max_slope_run: u32,
}

impl HeightfieldPlacementFilters {
    fn validate(self) -> Result<(), HeightfieldPlacementError> {
        if self.surface_height_mm[0] > self.surface_height_mm[1] || self.max_slope_run == 0 {
            return Err(HeightfieldPlacementError::InvalidFilters);
        }
        Ok(())
    }

    fn accepts(self, sample: HeightfieldSurfaceSample) -> bool {
        let numerator = sample.height_numerator_mm();
        let denominator = i128::from(sample.height_denominator());
        numerator >= i128::from(self.surface_height_mm[0]) * denominator
            && numerator <= i128::from(self.surface_height_mm[1]) * denominator
            && slope_within(sample.normal_mm(), self.max_slope_rise, self.max_slope_run)
    }
}

/// Explicit per-request logical work and combined returned-capacity limits.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HeightfieldPlacementBudget {
    /// Primary plus repeated halo candidate probes, checked conservatively first.
    pub max_candidate_evaluations: u64,
    /// Conservative four reads per primary, admitted before any thinning.
    pub max_sample_reads: u64,
    /// Primary count times linear author-field sample/rank work.
    pub max_field_work: u64,
    /// Maximum final accepted placements; failure returns no partial prefix.
    pub max_output_points: usize,
    /// Combined actual capacity payload of geometry and attribute Vecs.
    ///
    /// Includes spare capacity, excludes allocator metadata, shared terrain/field
    /// storage, stack and consumer copies. This is not an RSS or time limit.
    pub max_output_bytes: usize,
}

/// Allocation-free, checked upper bounds, independent of later rejection.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HeightfieldPlacementWork {
    /// Number of XZ grid squares intersecting the request.
    pub primary_candidates: u64,
    /// `N * (2 * ceil(radius / spacing) + 1)^2`, or N with radius zero.
    pub max_candidate_evaluations: u64,
    /// Four times the primary count.
    pub max_sample_reads: u64,
    /// Primary count times the author's linear work per sample.
    pub max_field_work: u64,
}

/// Complete structural dependencies; cloning shares immutable payloads.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HeightfieldPlacementCacheKey {
    /// Placement algorithm revision.
    pub placement_version: u32,
    /// Source XZ jitter algorithm revision.
    pub source_kernel_version: u32,
    /// Exact terrain query algorithm revision.
    pub surface_version: u32,
    /// Author-field sample and rank algorithm revision.
    pub field_version: u32,
    /// All candidate, distance and projection inputs.
    pub config: HeightfieldPlacementConfig,
    /// Exact XZ ownership.
    pub region: HeightfieldPlacementRegion,
    /// All terrain-selection inputs, including inactive limits.
    pub filters: HeightfieldPlacementFilters,
    /// Complete immutable terrain and embedding, not a digest or revision alone.
    pub surface: PlanarHeightfieldSurface,
    /// Complete immutable author recipe, including inactive brushes/noise.
    pub field: ScatterDensityField,
}

/// Complete disposable placements in lexicographic global grid X/Z order.
#[derive(Debug, Eq, PartialEq)]
pub struct HeightfieldPlacementOutput {
    /// Projected points; no accepted point is left at its raw random source Y.
    pub points: Vec<HeightfieldPlacementPoint>,
    /// Density/species sampled at each final rounded, offset position.
    pub attributes: Vec<ScatterFieldSample>,
    /// All semantic inputs; budgets do not participate in cache identity.
    pub cache_key: HeightfieldPlacementCacheKey,
    /// Actual primary and halo probes, including out-of-i64 candidate attempts.
    pub candidate_evaluations: u64,
    /// Actual logical terrain reads, four per horizontal-distance survivor.
    pub surface_sample_reads: u64,
    /// Actual author sample/rank work, including density-rejected candidates.
    pub field_work: u64,
}

/// Input, admission, reservation or exact projection failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HeightfieldPlacementError {
    /// An XZ request axis is inverted.
    InvalidRegion,
    /// Height interval inverted or slope denominator zero.
    InvalidFilters,
    /// Unsupported placement revision or a nonzero raw grid Y.
    InvalidKey,
    /// Count, work or payload arithmetic overflow.
    ArithmeticOverflow,
    /// Exact height plus offset cannot round into an i64 world position.
    ProjectionOverflow,
    /// Conservative candidate work exceeds the supplied limit.
    CandidateBudgetExceeded {
        /// Required probes.
        required: u64,
        /// Supplied probes.
        limit: u64,
    },
    /// Conservative terrain reads exceed the supplied limit.
    SurfaceBudgetExceeded {
        /// Required reads.
        required: u64,
        /// Supplied reads.
        limit: u64,
    },
    /// Conservative author work exceeds the supplied limit.
    FieldBudgetExceeded {
        /// Required work.
        required: u64,
        /// Supplied work.
        limit: u64,
    },
    /// An additional final point exceeds the supplied count limit.
    OutputCountBudgetExceeded {
        /// Required points.
        required: usize,
        /// Supplied points.
        limit: usize,
    },
    /// Actual returned capacities or an additional row exceed the payload limit.
    OutputBytesBudgetExceeded {
        /// Required bytes.
        required: usize,
        /// Supplied bytes.
        limit: usize,
    },
    /// A fallible exact reservation failed.
    AllocationFailed,
    /// Raw jitter source validation/reconstruction failed.
    Source(ScatterError),
    /// Nonempty footprint lacks terrain coverage, or a query failed.
    Surface(HeightfieldQueryError),
}

impl fmt::Display for HeightfieldPlacementError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRegion => f.write_str("placement XZ region has an inverted axis"),
            Self::InvalidFilters => f.write_str("invalid placement height or slope filters"),
            Self::InvalidKey => f.write_str("unsupported placement key revision or nonzero grid Y"),
            Self::ArithmeticOverflow => {
                f.write_str("placement work or capacity arithmetic overflow")
            }
            Self::ProjectionOverflow => {
                f.write_str("projected placement Y exceeds i64 millimetres")
            }
            Self::CandidateBudgetExceeded { required, limit } => write!(
                f,
                "placement requires {required} candidate probes, budget {limit}"
            ),
            Self::SurfaceBudgetExceeded { required, limit } => write!(
                f,
                "placement requires {required} terrain reads, budget {limit}"
            ),
            Self::FieldBudgetExceeded { required, limit } => write!(
                f,
                "placement requires {required} field work, budget {limit}"
            ),
            Self::OutputCountBudgetExceeded { required, limit } => {
                write!(f, "placement requires {required} points, budget {limit}")
            }
            Self::OutputBytesBudgetExceeded { required, limit } => write!(
                f,
                "placement requires {required} capacity bytes, budget {limit}"
            ),
            Self::AllocationFailed => f.write_str("placement output reservation failed"),
            Self::Source(error) => write!(f, "placement raw source: {error}"),
            Self::Surface(error) => write!(f, "placement terrain: {error}"),
        }
    }
}

impl core::error::Error for HeightfieldPlacementError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Source(error) => Some(error),
            Self::Surface(error) => Some(error),
            _ => None,
        }
    }
}

impl From<ScatterError> for HeightfieldPlacementError {
    fn from(error: ScatterError) -> Self {
        Self::Source(error)
    }
}

impl From<HeightfieldQueryError> for HeightfieldPlacementError {
    fn from(error: HeightfieldQueryError) -> Self {
        Self::Surface(error)
    }
}

/// Estimates the full request before allocation or terrain/field enumeration.
pub fn estimate_heightfield_placement_work(
    config: HeightfieldPlacementConfig,
    region: HeightfieldPlacementRegion,
    field: &ScatterDensityField,
) -> Result<HeightfieldPlacementWork, HeightfieldPlacementError> {
    Ok(Layout::new(config, region, field)?.work)
}

/// Places one global XZ jitter candidate per square onto real triangle geometry.
///
/// The full nonempty footprint must be covered. After input/coverage/work
/// admission, raw horizontal minimum-distance thinning runs before projection,
/// exact terrain filters, then author-field sampling at the final position.
/// Projection overflow is an error even if filters would later reject the point.
///
/// All neighbours are regenerated without terrain/field queries, including
/// outside the request/surface. Identical global terrain and field snapshots
/// therefore give request-order and arbitrary XZ partition invariant membership.
/// XZ minimum distance implies at least that 3D distance after projection; no
/// claim is made for geodesic spacing or maximum fill. Orientation is supplied
/// as an exact face normal; renderer-specific transforms remain the consumer's.
///
/// Both output Vecs reserve at most once. Only final accepted rows count against
/// output-count admission; their spare capacities remain charged even if empty.
/// No index, implicit terrain halo, persistence or GPU renderer is created.
pub fn scatter_on_heightfield(
    config: HeightfieldPlacementConfig,
    region: HeightfieldPlacementRegion,
    field: &ScatterDensityField,
    surface: &PlanarHeightfieldSurface,
    filters: HeightfieldPlacementFilters,
    budget: HeightfieldPlacementBudget,
) -> Result<HeightfieldPlacementOutput, HeightfieldPlacementError> {
    filters.validate()?;
    let layout = Layout::new(config, region, field)?;
    if layout.work.primary_candidates != 0
        && (0..2).any(|axis| {
            region.min_xz_mm[axis] < surface.min_xz_mm()[axis]
                || region.max_xz_mm[axis] > surface.max_xz_mm()[axis]
        })
    {
        return Err(HeightfieldQueryError::OutsideSurface.into());
    }
    admit(layout.work, budget)?;
    let row_bytes = size_of::<HeightfieldPlacementPoint>() + size_of::<ScatterFieldSample>();
    let capacity = usize::try_from(layout.work.primary_candidates)
        .unwrap_or(usize::MAX)
        .min(budget.max_output_points)
        .min(budget.max_output_bytes / row_bytes);
    let mut points = Vec::new();
    let mut attributes = Vec::new();
    points
        .try_reserve_exact(capacity)
        .map_err(|_| HeightfieldPlacementError::AllocationFailed)?;
    attributes
        .try_reserve_exact(capacity)
        .map_err(|_| HeightfieldPlacementError::AllocationFailed)?;
    check_bytes(
        points.capacity(),
        attributes.capacity(),
        budget.max_output_bytes,
    )?;
    let mut candidate_evaluations = 0;
    let mut surface_sample_reads = 0;
    let mut field_work = 0;
    if layout.work.primary_candidates != 0 {
        for x in layout.min[0]..=layout.max[0] {
            for z in layout.min[1]..=layout.max[1] {
                candidate_evaluations += 1;
                let Some(current) = candidate(config, x, z) else {
                    continue;
                };
                if !region.contains(current.xz)
                    || !survives(config, current, layout.halo, &mut candidate_evaluations)
                {
                    continue;
                }
                let point = project(current.key, current.xz, surface, config.vertical_offset_mm)?;
                surface_sample_reads += 4;
                if !filters.accepts(point.surface_sample) {
                    continue;
                }
                let sample = field.sample(point.position_mm);
                field_work += field.work_per_sample();
                if !field.accepts(point.key.source, sample.density) {
                    continue;
                }
                let required = points
                    .len()
                    .checked_add(1)
                    .ok_or(HeightfieldPlacementError::ArithmeticOverflow)?;
                if required > budget.max_output_points {
                    return Err(HeightfieldPlacementError::OutputCountBudgetExceeded {
                        required,
                        limit: budget.max_output_points,
                    });
                }
                check_bytes(required, required, budget.max_output_bytes)?;
                debug_assert!(required <= capacity);
                points.push(point);
                attributes.push(sample);
            }
        }
    }
    debug_assert!(candidate_evaluations <= layout.work.max_candidate_evaluations);
    debug_assert!(surface_sample_reads <= layout.work.max_sample_reads);
    debug_assert!(field_work <= layout.work.max_field_work);
    Ok(HeightfieldPlacementOutput {
        points,
        attributes,
        cache_key: HeightfieldPlacementCacheKey {
            placement_version: HEIGHTFIELD_PLACEMENT_VERSION,
            source_kernel_version: SCATTER_KERNEL_VERSION,
            surface_version: HEIGHTFIELD_SURFACE_VERSION,
            field_version: SCATTER_FIELD_VERSION,
            config,
            region,
            filters,
            surface: surface.clone(),
            field: field.clone(),
        },
        candidate_evaluations,
        surface_sample_reads,
        field_work,
    })
}

fn admit(
    work: HeightfieldPlacementWork,
    budget: HeightfieldPlacementBudget,
) -> Result<(), HeightfieldPlacementError> {
    if work.max_candidate_evaluations > budget.max_candidate_evaluations {
        return Err(HeightfieldPlacementError::CandidateBudgetExceeded {
            required: work.max_candidate_evaluations,
            limit: budget.max_candidate_evaluations,
        });
    }
    if work.max_sample_reads > budget.max_sample_reads {
        return Err(HeightfieldPlacementError::SurfaceBudgetExceeded {
            required: work.max_sample_reads,
            limit: budget.max_sample_reads,
        });
    }
    if work.max_field_work > budget.max_field_work {
        return Err(HeightfieldPlacementError::FieldBudgetExceeded {
            required: work.max_field_work,
            limit: budget.max_field_work,
        });
    }
    Ok(())
}

fn check_bytes(
    points: usize,
    attributes: usize,
    limit: usize,
) -> Result<(), HeightfieldPlacementError> {
    let required = points
        .checked_mul(size_of::<HeightfieldPlacementPoint>())
        .and_then(|bytes| {
            attributes
                .checked_mul(size_of::<ScatterFieldSample>())
                .and_then(|extra| bytes.checked_add(extra))
        })
        .ok_or(HeightfieldPlacementError::ArithmeticOverflow)?;
    if required > limit {
        return Err(HeightfieldPlacementError::OutputBytesBudgetExceeded { required, limit });
    }
    Ok(())
}

struct Layout {
    min: [i64; 2],
    max: [i64; 2],
    halo: i64,
    work: HeightfieldPlacementWork,
}

impl Layout {
    fn new(
        config: HeightfieldPlacementConfig,
        region: HeightfieldPlacementRegion,
        field: &ScatterDensityField,
    ) -> Result<Self, HeightfieldPlacementError> {
        if config.grid_spacing_mm == 0 {
            return Err(ScatterError::InvalidGridSpacing.into());
        }
        region.validate()?;
        let mut result = Self {
            min: [0; 2],
            max: [0; 2],
            halo: 0,
            work: HeightfieldPlacementWork {
                primary_candidates: 0,
                max_candidate_evaluations: 0,
                max_sample_reads: 0,
                max_field_work: 0,
            },
        };
        if (0..2).any(|axis| region.min_xz_mm[axis] == region.max_xz_mm[axis]) {
            return Ok(result);
        }
        let spacing = i64::from(config.grid_spacing_mm);
        let mut count = 1_u64;
        for axis in 0..2 {
            result.min[axis] = region.min_xz_mm[axis].div_euclid(spacing);
            result.max[axis] = (region.max_xz_mm[axis] - 1).div_euclid(spacing);
            let length =
                u64::try_from(i128::from(result.max[axis]) - i128::from(result.min[axis]) + 1)
                    .map_err(|_| HeightfieldPlacementError::ArithmeticOverflow)?;
            count = count
                .checked_mul(length)
                .ok_or(HeightfieldPlacementError::ArithmeticOverflow)?;
        }
        let halo = u64::from(config.min_distance_mm).div_ceil(u64::from(config.grid_spacing_mm));
        let side = halo * 2 + 1;
        result.halo = halo as i64;
        result.work = HeightfieldPlacementWork {
            primary_candidates: count,
            max_candidate_evaluations: side
                .checked_mul(side)
                .and_then(|area| count.checked_mul(area))
                .ok_or(HeightfieldPlacementError::ArithmeticOverflow)?,
            max_sample_reads: count
                .checked_mul(4)
                .ok_or(HeightfieldPlacementError::ArithmeticOverflow)?,
            max_field_work: count
                .checked_mul(field.work_per_sample())
                .ok_or(HeightfieldPlacementError::ArithmeticOverflow)?,
        };
        Ok(result)
    }
}

#[derive(Clone, Copy)]
struct Candidate {
    key: HeightfieldPlacementKey,
    xz: [i64; 2],
    priority: u64,
}

fn mix(value: u64) -> u64 {
    (u64::from(pcg_hash_u32(value)) << 32) | u64::from(pcg_hash_u32(value ^ 0x9e37_79b9_7f4a_7c15))
}

fn candidate(config: HeightfieldPlacementConfig, x: i64, z: i64) -> Option<Candidate> {
    let key = HeightfieldPlacementKey {
        placement_version: HEIGHTFIELD_PLACEMENT_VERSION,
        source: ScatterCandidateKey {
            kernel_version: SCATTER_KERNEL_VERSION,
            seed: config.seed,
            version: config.version,
            salt: config.salt,
            grid_spacing_mm: config.grid_spacing_mm,
            grid: SectorCoord::new(x, 0, z),
        },
    };
    let raw = key.source.rebuild().ok()?;
    // No ranking is consumed when distance thinning is disabled. Avoid its
    // separate hash stream in the dense, radius-zero path without changing XZ.
    let mut priority = 0;
    if config.min_distance_mm != 0 {
        priority = mix(0x706c_6163_655f_787a ^ u64::from(HEIGHTFIELD_PLACEMENT_VERSION));
        for value in [
            u64::from(key.source.kernel_version),
            config.seed,
            u64::from(config.version),
            config.salt,
            u64::from(config.grid_spacing_mm),
            x as u64,
            z as u64,
        ] {
            priority = mix(priority ^ value);
        }
    }
    Some(Candidate {
        key,
        xz: [raw.position_mm[0], raw.position_mm[2]],
        priority,
    })
}

fn survives(
    config: HeightfieldPlacementConfig,
    current: Candidate,
    halo: i64,
    evaluations: &mut u64,
) -> bool {
    if config.min_distance_mm == 0 {
        return true;
    }
    let grid = current.key.source.grid;
    let radius = u128::from(config.min_distance_mm);
    for x in grid.x.saturating_sub(halo)..=grid.x.saturating_add(halo) {
        for z in grid.z.saturating_sub(halo)..=grid.z.saturating_add(halo) {
            if x == grid.x && z == grid.z {
                continue;
            }
            *evaluations += 1;
            let Some(other) = candidate(config, x, z) else {
                continue;
            };
            if (other.priority, other.key) >= (current.priority, current.key) {
                continue;
            }
            let dx = (i128::from(other.xz[0]) - i128::from(current.xz[0])).unsigned_abs();
            let dz = (i128::from(other.xz[1]) - i128::from(current.xz[1])).unsigned_abs();
            if dx < radius && dz < radius && dx * dx + dz * dz < radius * radius {
                return false;
            }
        }
    }
    true
}

fn project(
    key: HeightfieldPlacementKey,
    xz: [i64; 2],
    surface: &PlanarHeightfieldSurface,
    offset: i64,
) -> Result<HeightfieldPlacementPoint, HeightfieldPlacementError> {
    let sample = surface.sample_xz_mm(xz)?;
    let denominator = i128::from(sample.height_denominator());
    // Exact surface numerators fit <2^111; adding i64 * u32 remains in i128.
    let numerator = sample.height_numerator_mm() + i128::from(offset) * denominator;
    let mut rounded = numerator / denominator;
    if (numerator % denominator).unsigned_abs() * 2 >= denominator as u128 {
        rounded += numerator.signum();
    }
    let y = i64::try_from(rounded).map_err(|_| HeightfieldPlacementError::ProjectionOverflow)?;
    Ok(HeightfieldPlacementPoint {
        key,
        position_mm: [xz[0], y, xz[1]],
        surface_sample: sample,
    })
}
