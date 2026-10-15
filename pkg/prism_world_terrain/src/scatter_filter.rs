//! Exact heightfield filters over unmodified integer source scatter positions.
//!
//! This is thinning, not ground projection: no points are moved or added. A
//! narrow clearance band can yield few or no survivors. The source's minimum
//! distance and arbitrary three-axis request-partition invariance are retained.

use alloc::vec::Vec;
use core::fmt;

use prism_world_residency::{
    estimate_scatter_work, scatter_with_density_field, FieldScatterBudget, FieldScatterCacheKey,
    ScatterBudget, ScatterConfig, ScatterDensityField, ScatterFieldError, ScatterFieldSample,
    ScatterPoint, ScatterRegion,
};

use crate::{
    HeightfieldQueryError, HeightfieldSurfaceSample, PlanarHeightfieldSurface,
    HEIGHTFIELD_SURFACE_VERSION,
};

/// Revision of exact height/clearance/slope selection and stable compaction.
pub const HEIGHTFIELD_SCATTER_FILTER_VERSION: u32 = 1;

/// Closed height/clearance intervals and a nonnegative rational slope limit.
///
/// All heights are world +Y millimetres. Clearance is source Y minus exact
/// surface Y, not Euclidean distance to the plane. Slope is rise/horizontal run
/// of the geometric triangle, not an interpolated mesh vertex normal.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HeightfieldScatterFilters {
    /// Inclusive allowed world surface height, not the source point's Y.
    pub surface_height_mm: [i64; 2],
    /// Inclusive allowed vertical distance above/below the surface.
    pub clearance_mm: [i64; 2],
    /// Numerator of the maximum rise/run; zero permits horizontal faces only.
    pub max_slope_rise: u32,
    /// Positive denominator of the maximum rise/run.
    pub max_slope_run: u32,
}

impl HeightfieldScatterFilters {
    fn validate(self) -> Result<(), HeightfieldScatterError> {
        if self.surface_height_mm[0] > self.surface_height_mm[1]
            || self.clearance_mm[0] > self.clearance_mm[1]
            || self.max_slope_run == 0
        {
            return Err(HeightfieldScatterError::InvalidFilters);
        }
        Ok(())
    }

    fn accepts(self, point_y: i64, sample: HeightfieldSurfaceSample) -> bool {
        let denominator = i128::from(sample.height_denominator());
        let height = sample.height_numerator_mm();
        let clearance = i128::from(point_y) * denominator - height;
        let within = |value: i128, bounds: [i64; 2]| {
            value >= i128::from(bounds[0]) * denominator
                && value <= i128::from(bounds[1]) * denominator
        };
        within(height, self.surface_height_mm)
            && within(clearance, self.clearance_mm)
            && slope_within(sample.normal_mm(), self.max_slope_rise, self.max_slope_run)
    }
}

/// Additional terrain-probe admission, independent of source/field work limits.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HeightfieldScatterBudget {
    /// Four sample reads per primary candidate, checked before source generation.
    /// This is a conservative logical count, not CPU instructions or a time cap.
    pub max_sample_reads: u64,
}

/// Complete structural input identity, with an immutable shared terrain snapshot.
///
/// Equality compares the complete height payload, not tile/revision alone, an
/// allocation address, or a digest. Clone shares the snapshot, never deep-copies
/// it; full equality is worst-case O(terrain samples + field brushes).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HeightfieldScatterCacheKey {
    /// Full source scatter and author density/species field recipe.
    pub source: FieldScatterCacheKey,
    /// Exact geometric surface interpolation revision.
    pub surface_version: u32,
    /// Height, clearance and slope selection revision.
    pub filter_version: u32,
    /// Full immutable height data and world embedding.
    pub surface: PlanarHeightfieldSurface,
    /// All selection parameters, including inactive range endpoints.
    pub filters: HeightfieldScatterFilters,
}

/// Stable retained source points with their aligned author-field attributes.
#[derive(Debug, Eq, PartialEq)]
pub struct HeightfieldScatterOutput {
    /// Unmodified retained points, in source global-grid lexicographic order.
    pub points: Vec<ScatterPoint>,
    /// Density/species at each corresponding source point, not at a projection.
    pub attributes: Vec<ScatterFieldSample>,
    /// Full identity needed to invalidate or rebuild these derived results.
    pub cache_key: HeightfieldScatterCacheKey,
    /// Actual source primary and halo probes, before terrain selection.
    pub candidate_evaluations: u64,
    /// Actual author-field work, before terrain selection.
    pub field_work: u64,
    /// Four reads for each field survivor queried against the surface.
    pub surface_sample_reads: u64,
}

/// Admission, input, terrain-coverage, or source-generation failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HeightfieldScatterError {
    /// A closed interval is inverted or the slope denominator is zero.
    InvalidFilters,
    /// The conservative terrain sample-read count overflows u64.
    SurfaceWorkOverflow,
    /// Conservative terrain sample reads exceed the explicit limit.
    SurfaceWorkBudgetExceeded {
        /// Four times the source primary-candidate count.
        required: u64,
        /// Supplied maximum logical sample reads.
        limit: u64,
    },
    /// Source generation, density field or output capacity failure.
    Source(ScatterFieldError),
    /// Missing terrain coverage or invalid surface query.
    Surface(HeightfieldQueryError),
}

impl fmt::Display for HeightfieldScatterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidFilters => {
                f.write_str("invalid heightfield scatter filter interval or slope")
            }
            Self::SurfaceWorkOverflow => {
                f.write_str("heightfield scatter sample-read count overflow")
            }
            Self::SurfaceWorkBudgetExceeded { required, limit } => {
                write!(
                    f,
                    "heightfield scatter needs {required} sample reads, limit {limit}"
                )
            }
            Self::Source(error) => write!(f, "heightfield source scatter: {error}"),
            Self::Surface(error) => write!(f, "heightfield scatter surface: {error}"),
        }
    }
}

impl core::error::Error for HeightfieldScatterError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Source(error) => Some(error),
            Self::Surface(error) => Some(error),
            _ => None,
        }
    }
}

impl From<ScatterFieldError> for HeightfieldScatterError {
    fn from(error: ScatterFieldError) -> Self {
        Self::Source(error)
    }
}

impl From<HeightfieldQueryError> for HeightfieldScatterError {
    fn from(error: HeightfieldQueryError) -> Self {
        Self::Surface(error)
    }
}

/// Generates density-filtered scatter and stably thins it using a real heightfield.
///
/// A nonempty request's entire XZ footprint must lie within the surface's
/// half-open coverage, even if density would reject all points. Missing terrain
/// is an error, not an implicit hole. Empty requests need no terrain coverage.
/// Logical terrain reads are admitted against the original primary upper bound
/// before source enumeration/allocation; no budget-dependent prefix is returned.
///
/// This reuses the two source Vec allocations and compresses both in place.
/// Their unchanged capacities remain covered by `field_budget.max_output_bytes`.
/// No third per-point output or terrain copy is allocated. Shared terrain/field
/// residency, allocator metadata and RSS are separate caller-owned budgets.
/// Retaining the output retains its immutable terrain snapshot; rebaking must
/// publish a new snapshot rather than silently mutating/copying this one.
#[allow(clippy::too_many_arguments)]
pub fn scatter_with_heightfield(
    config: ScatterConfig,
    region: ScatterRegion,
    source_budget: ScatterBudget,
    field: &ScatterDensityField,
    field_budget: FieldScatterBudget,
    surface: &PlanarHeightfieldSurface,
    filters: HeightfieldScatterFilters,
    surface_budget: HeightfieldScatterBudget,
) -> Result<HeightfieldScatterOutput, HeightfieldScatterError> {
    filters.validate()?;
    let estimate = estimate_scatter_work(config, region).map_err(ScatterFieldError::from)?;
    let empty = region
        .min_mm
        .into_iter()
        .zip(region.max_mm)
        .any(|(min, max)| min == max);
    if !empty {
        let min = surface.min_xz_mm();
        let max = surface.max_xz_mm();
        if region.min_mm[0] < min[0]
            || region.min_mm[2] < min[1]
            || region.max_mm[0] > max[0]
            || region.max_mm[2] > max[1]
        {
            return Err(HeightfieldQueryError::OutsideSurface.into());
        }
    }
    let required = estimate
        .primary_candidates
        .checked_mul(4)
        .ok_or(HeightfieldScatterError::SurfaceWorkOverflow)?;
    if required > surface_budget.max_sample_reads {
        return Err(HeightfieldScatterError::SurfaceWorkBudgetExceeded {
            required,
            limit: surface_budget.max_sample_reads,
        });
    }
    let mut source =
        scatter_with_density_field(config, region, source_budget, field, field_budget)?;
    let surface_sample_reads = source.points.len() as u64 * 4;
    let mut retained = 0;
    for read in 0..source.points.len() {
        let point = source.points[read];
        let sample = surface.sample_xz_mm([point.position_mm[0], point.position_mm[2]])?;
        if filters.accepts(point.position_mm[1], sample) {
            source.points[retained] = point;
            source.attributes[retained] = source.attributes[read];
            retained += 1;
        }
    }
    source.points.truncate(retained);
    source.attributes.truncate(retained);
    Ok(HeightfieldScatterOutput {
        points: source.points,
        attributes: source.attributes,
        cache_key: HeightfieldScatterCacheKey {
            source: source.cache_key,
            surface_version: HEIGHTFIELD_SURFACE_VERSION,
            filter_version: HEIGHTFIELD_SCATTER_FILTER_VERSION,
            surface: surface.clone(),
            filters,
        },
        candidate_evaluations: source.candidate_evaluations,
        field_work: source.field_work,
        surface_sample_reads,
    })
}

// Avoid squaring arbitrary decoded i128 height differences. First compare each
// scaled component to spacing*rise (<2^64); overflow/oversize must reject. Only
// surviving u64-sized components are squared in u128, and subtraction avoids
// the potentially overflowing sum of two individually representable squares.
fn slope_within(normal: [i128; 3], rise: u32, run: u32) -> bool {
    let limit = normal[1] as u128 * u128::from(rise);
    let component = |value: i128| {
        value
            .unsigned_abs()
            .checked_mul(u128::from(run))
            .filter(|scaled| *scaled <= limit)
    };
    let (Some(x), Some(z)) = (component(normal[0]), component(normal[2])) else {
        return false;
    };
    z * z <= limit * limit - x * x
}
