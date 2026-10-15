//! Integer author density/species fields over the global millimetre domain.
//!
//! These are immutable derived recipes, not an editor document or a second
//! scene truth. All operations, including domain warp and brush falloff, have
//! versioned integer rounding rules. No floating-point world conversion occurs.
//!
//! ```
//! use prism_world_residency::{
//!     ScatterDensityField, ScatterFieldBuildBudget, ScatterFieldConfig,
//! };
//! let field = ScatterDensityField::new(
//!     ScatterFieldConfig {
//!         seed: 42, version: 1, base_density: 32768,
//!         base_species: 9, noise: None,
//!     },
//!     Vec::new(),
//!     ScatterFieldBuildBudget { max_brushes: 0, max_brush_bytes: 0 },
//! )?;
//! let sample = field.sample([i64::MIN, -1, i64::MAX]);
//! assert_eq!((sample.density, sample.species), (32768, 9));
//! # Ok::<(), prism_world_residency::ScatterFieldError>(())
//! ```

use alloc::{sync::Arc, vec::Vec};
use core::{fmt, mem::size_of};

use crate::{
    estimate_scatter_work, pcg_hash_u32, scatter, ScatterBudget, ScatterCacheKey,
    ScatterCandidateKey, ScatterConfig, ScatterError, ScatterPoint, ScatterRegion,
};

/// Revision of noise, brush composition, rank and selection arithmetic.
pub const SCATTER_FIELD_VERSION: u32 = 1;
const ONE: u64 = u16::MAX as u64;

/// A single smooth value-noise band with optional three-channel domain warp.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct DensityNoise {
    /// Positive millimetre cell size of the density noise.
    pub cell_size_mm: u32,
    /// Signed density excursion about the field's base value (0 disables noise).
    pub amplitude: u16,
    /// Positive millimetre cell size of the independent warp noise.
    pub warp_cell_size_mm: u32,
    /// Maximum signed warp displacement on each axis; zero disables warp.
    pub warp_amplitude_mm: u32,
}

/// Scene-owned inputs from which a disposable field may be constructed.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ScatterFieldConfig {
    /// Namespace for noise channels and independent candidate-selection ranks.
    pub seed: u64,
    /// Caller recipe revision, independent of [`SCATTER_FIELD_VERSION`].
    pub version: u32,
    /// Unpainted density, in the inclusive UNORM16 range.
    pub base_density: u16,
    /// Content-neutral species/asset-table index, not a fixed vegetation enum.
    pub base_species: u32,
    /// Optional band; both cell sizes must be positive even when disabled.
    pub noise: Option<DensityNoise>,
}

/// Density operation of an author brush, composed in increasing stable ID.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum DensityBrushOp {
    /// Blend toward this UNORM16 value using effective brush weight.
    Replace(u16),
    /// Add a signed weighted delta, saturating density to 0..=65535.
    /// The magnitude must not exceed 65535.
    Add(i32),
}

/// A spherical author stroke in global integer millimetres.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct DensityBrush {
    /// Unique stable composition order. Input Vec order is not significant.
    pub id: u64,
    /// Absolute brush centre; never converted to floating point.
    pub center_mm: [i64; 3],
    /// Full-weight inner radius; may equal the outer radius for a hard brush.
    pub inner_radius_mm: u32,
    /// Positive outer radius; exactly on the outer boundary has zero weight.
    pub outer_radius_mm: u32,
    /// UNORM16 opacity, multiplied after radial smoothstep.
    pub opacity: u16,
    /// Density edit; use Add(0) for a species-only brush.
    pub density: DensityBrushOp,
    /// Replace the species where effective weight is at least 32768.
    /// Categorical selection deliberately has a hard, documented boundary.
    pub species: Option<u32>,
}

/// Limits on the recipe's owned brush Vec (not allocator/Arc metadata or RSS).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ScatterFieldBuildBudget {
    /// Maximum number of strokes, including zero-opacity strokes.
    pub max_brushes: usize,
    /// Maximum supplied Vec capacity times `size_of::<DensityBrush>()`.
    pub max_brush_bytes: usize,
}

#[derive(Debug, Eq, PartialEq)]
struct FieldDefinition {
    config: ScatterFieldConfig,
    brushes: Vec<DensityBrush>,
}

/// Validated, immutable recipe with a canonical brush order.
///
/// Cloning is O(1); equality compares complete configuration and brush values,
/// not a digest or allocation identity. Arc metadata allocation uses normal
/// allocator OOM handling, not a recoverable budget error. Construction sorts
/// the caller-owned Vec in place and does not allocate another brush buffer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ScatterDensityField(Arc<FieldDefinition>);

/// Sampled field values; density zero and one are exactly representable.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ScatterFieldSample {
    /// UNORM16 density, zero to 65535 inclusive.
    pub density: u16,
    /// Caller-defined species/asset-table index.
    pub species: u32,
}

/// Full structural cache descriptor for the field-filtered scatter.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FieldScatterCacheKey {
    /// Unmodified generation inputs and spatial ownership bounds.
    pub scatter: ScatterCacheKey,
    /// Field algorithm revision, not the caller's recipe revision.
    pub field_version: u32,
    /// Complete field recipe, shared immutably, compared by value.
    pub field: ScatterDensityField,
}

/// Independent field-work and total returned Vec payload limits.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FieldScatterBudget {
    /// Logical work limit, preflighted against all primary candidates.
    /// Linear sampling multiplies by [`ScatterDensityField::work_per_sample`];
    /// indexed sampling multiplies by [`crate::ScatterBrushIndex::max_sample_work`].
    /// These are probe units, not instruction, sorting-comparison, or time bounds.
    pub max_field_work: u64,
    /// Total capacity bytes of points plus aligned field attributes.
    /// Includes spare capacity, excludes Vec/Arc headers, metadata and stack.
    pub max_output_bytes: usize,
}

/// Complete deterministic result; points and attributes have equal lengths.
#[derive(Debug, Eq, PartialEq)]
pub struct FieldScatterOutput {
    /// Retained points in the original lexicographic global-grid order.
    pub points: Vec<ScatterPoint>,
    /// Density and species, indexed in exactly the same order as `points`.
    pub attributes: Vec<ScatterFieldSample>,
    /// All generation and field inputs, not a truncated hash identity.
    pub cache_key: FieldScatterCacheKey,
    /// Actual source primary and halo candidate evaluations.
    pub candidate_evaluations: u64,
    /// Logical work for all sampled source survivors, including rejected ones.
    /// Linear sampling charges the fixed `work_per_sample()` bound per survivor;
    /// indexed sampling sums actual base noise/rank, node, and brush probes.
    pub field_work: u64,
}

/// Invalid author input, exceeded execution budgets or fallible Vec allocation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScatterFieldError {
    /// Density and warp cell sizes must be positive.
    InvalidNoiseCellSize,
    /// Outer radius must be positive and inner must not exceed outer.
    InvalidBrushRadius,
    /// Signed additive density delta is outside -65535..=65535.
    InvalidBrushDelta,
    /// Two strokes have the same composition identity.
    DuplicateBrushId(u64),
    /// Number of supplied brushes exceeds the recipe limit.
    BrushBudgetExceeded {
        /// Supplied brush count.
        required: usize,
        /// Maximum permitted brush count.
        limit: usize,
    },
    /// Brush Vec capacity exceeds the supplied byte limit.
    BrushBytesBudgetExceeded {
        /// Supplied Vec capacity payload in bytes.
        required: usize,
        /// Maximum permitted brush payload.
        limit: usize,
    },
    /// Conservative field work exceeds the execution limit.
    WorkBudgetExceeded {
        /// Primary candidates times logical work per sample.
        required: u64,
        /// Maximum permitted logical field work.
        limit: u64,
    },
    /// Combined returned Vec capacities exceed the payload limit.
    OutputBytesBudgetExceeded {
        /// Combined point and attribute capacity payload in bytes.
        required: usize,
        /// Maximum permitted returned payload.
        limit: usize,
    },
    /// Byte size or work multiplication does not fit its representation.
    ArithmeticOverflow,
    /// Fallible reservation of output attributes failed.
    AllocationFailed,
    /// The original scatter kernel rejected the request.
    Scatter(ScatterError),
}

impl fmt::Display for ScatterFieldError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidNoiseCellSize => f.write_str("field noise cell sizes must be positive"),
            Self::InvalidBrushRadius => f.write_str("brush requires 0 <= inner <= positive outer"),
            Self::InvalidBrushDelta => f.write_str("brush delta must be in -65535..=65535"),
            Self::DuplicateBrushId(id) => write!(f, "duplicate brush ID {id}"),
            Self::BrushBudgetExceeded { required, limit } => {
                write!(f, "field needs {required} brushes, limit {limit}")
            }
            Self::BrushBytesBudgetExceeded { required, limit } => {
                write!(f, "brush Vec needs {required} bytes, limit {limit}")
            }
            Self::WorkBudgetExceeded { required, limit } => {
                write!(f, "field needs {required} work units, limit {limit}")
            }
            Self::OutputBytesBudgetExceeded { required, limit } => {
                write!(f, "field output needs {required} bytes, limit {limit}")
            }
            Self::ArithmeticOverflow => f.write_str("field work or byte size overflow"),
            Self::AllocationFailed => f.write_str("field output reservation failed"),
            Self::Scatter(error) => write!(f, "scatter source: {error}"),
        }
    }
}

impl core::error::Error for ScatterFieldError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Scatter(error) => Some(error),
            _ => None,
        }
    }
}

impl From<ScatterError> for ScatterFieldError {
    fn from(error: ScatterError) -> Self {
        Self::Scatter(error)
    }
}

impl ScatterDensityField {
    /// Validates every brush (including inactive ones), then sorts by unique ID.
    /// Bounds the supplied Vec's capacity, not merely its current length.
    pub fn new(
        config: ScatterFieldConfig,
        mut brushes: Vec<DensityBrush>,
        budget: ScatterFieldBuildBudget,
    ) -> Result<Self, ScatterFieldError> {
        if config
            .noise
            .is_some_and(|n| n.cell_size_mm == 0 || n.warp_cell_size_mm == 0)
        {
            return Err(ScatterFieldError::InvalidNoiseCellSize);
        }
        if brushes.len() > budget.max_brushes {
            return Err(ScatterFieldError::BrushBudgetExceeded {
                required: brushes.len(),
                limit: budget.max_brushes,
            });
        }
        let bytes = brushes
            .capacity()
            .checked_mul(size_of::<DensityBrush>())
            .ok_or(ScatterFieldError::ArithmeticOverflow)?;
        if bytes > budget.max_brush_bytes {
            return Err(ScatterFieldError::BrushBytesBudgetExceeded {
                required: bytes,
                limit: budget.max_brush_bytes,
            });
        }
        for brush in &brushes {
            if brush.outer_radius_mm == 0 || brush.inner_radius_mm > brush.outer_radius_mm {
                return Err(ScatterFieldError::InvalidBrushRadius);
            }
            if let DensityBrushOp::Add(delta) = brush.density
                && !(-65535..=65535).contains(&delta)
            {
                return Err(ScatterFieldError::InvalidBrushDelta);
            }
        }
        brushes.sort_unstable_by_key(|brush| brush.id);
        if let Some(pair) = brushes.windows(2).find(|pair| pair[0].id == pair[1].id) {
            return Err(ScatterFieldError::DuplicateBrushId(pair[0].id));
        }
        Ok(Self(Arc::new(FieldDefinition { config, brushes })))
    }

    /// Complete field configuration, including inactive inputs for invalidation.
    pub fn config(&self) -> ScatterFieldConfig {
        self.0.config
    }

    /// Canonical, immutable brush list in ascending ID order.
    pub fn brushes(&self) -> &[DensityBrush] {
        &self.0.brushes
    }

    /// Conservative logical work per sample, including one selection-rank unit.
    /// Each brush costs one probe; a noise band costs 8 lattice probes; enabled
    /// warp adds 24. It is not CPU instruction count or a time bound. Sampling
    /// skips disabled noise, but still charges all configured brushes.
    pub fn work_per_sample(&self) -> u64 {
        self.base_sample_work() + self.0.brushes.len() as u64
    }

    pub(crate) fn base_sample_work(&self) -> u64 {
        let noise_work = match self.0.config.noise {
            Some(n) if n.amplitude != 0 => 8 + if n.warp_amplitude_mm == 0 { 0 } else { 24 },
            _ => 0,
        };
        1 + noise_work
    }

    /// Samples with integer Euclidean lattice coordinates and UNORM16 rounding.
    /// Warp lookup uses widened coordinates, never wrapping/clamping positions
    /// at world edges. The lattice hash has an explicit period of 2^64 cells
    /// per axis; widened virtual neighbours use their low 64 coordinate bits.
    pub fn sample(&self, position_mm: [i64; 3]) -> ScatterFieldSample {
        let mut sample = self.sample_base(position_mm);
        for brush in &self.0.brushes {
            sample.apply_brush(*brush, position_mm);
        }
        sample
    }

    pub(crate) fn sample_base(&self, position_mm: [i64; 3]) -> ScatterFieldSample {
        let config = self.0.config;
        let mut density = i64::from(config.base_density);
        if let Some(noise) = config.noise.filter(|n| n.amplitude != 0) {
            let origin = position_mm.map(i128::from);
            let mut warped = origin;
            if noise.warp_amplitude_mm != 0 {
                for (axis, coordinate) in warped.iter_mut().enumerate() {
                    let value = value_noise(
                        origin,
                        noise.warp_cell_size_mm,
                        config.seed,
                        axis as u64 + 1,
                    );
                    let signed = 2 * i64::from(value) - ONE as i64;
                    // Signed division truncates toward zero on all platforms.
                    let delta = signed * i64::from(noise.warp_amplitude_mm) / ONE as i64;
                    *coordinate += i128::from(delta);
                }
            }
            let value = value_noise(warped, noise.cell_size_mm, config.seed, 0);
            let signed = 2 * i64::from(value) - ONE as i64;
            density =
                (density + signed * i64::from(noise.amplitude) / ONE as i64).clamp(0, ONE as i64);
        }
        ScatterFieldSample {
            density: density as u16,
            species: config.base_species,
        }
    }

    /// Stable independent rank, using every structural candidate-key field.
    /// This is distribution data, never a collision-free candidate identity.
    pub fn candidate_rank(&self, key: ScatterCandidateKey) -> u32 {
        let mut hash = mix(self.0.config.seed ^ 0x6669_656c_645f_726b);
        for value in [
            u64::from(key.kernel_version),
            key.seed,
            u64::from(key.version),
            key.salt,
            u64::from(key.grid_spacing_mm),
            key.grid.x as u64,
            key.grid.y as u64,
            key.grid.z as u64,
        ] {
            hash = mix(hash ^ value);
        }
        pcg_hash_u32(hash)
    }

    /// Selects with a stable rank; zero rejects all and 65535 accepts all.
    /// Holding the key/rank fixed makes membership monotonic in density.
    pub fn accepts(&self, key: ScatterCandidateKey, density: u16) -> bool {
        density == u16::MAX
            || (density != 0 && self.candidate_rank(key) < u32::from(density) * 65537)
    }
}

impl ScatterFieldSample {
    pub(crate) fn apply_brush(&mut self, brush: DensityBrush, position_mm: [i64; 3]) {
        let weight = brush_weight(brush, position_mm);
        if weight == 0 {
            return;
        }
        self.density = match brush.density {
            DensityBrushOp::Replace(target) => lerp(self.density, target, weight),
            DensityBrushOp::Add(delta) => (i64::from(self.density)
                + i64::from(delta) * i64::from(weight) / ONE as i64)
                .clamp(0, ONE as i64) as u16,
        };
        if weight >= 32768
            && let Some(value) = brush.species
        {
            self.species = value;
        }
    }
}

/// Generates the base global scatter, then thins its survivors by the field.
///
/// Thinning cannot break minimum distance or request-partition invariance.
/// Density-rejected candidates can still have blocked source candidates; this
/// does not re-pack empty regions or change the underlying scatter algorithm.
///
/// Field work is checked against *all primary candidates* before enumeration.
/// Source count/byte limits are still enforced before field thinning. The
/// returned point buffer is reused in place; attributes reserve once for all
/// source survivors, including those subsequently rejected. Combined capacity
/// bytes are checked before reserve and again afterward. No partial output,
/// budget-dependent prefix, unchecked Vec growth or hidden per-point cache.
pub fn scatter_with_density_field(
    config: ScatterConfig,
    region: ScatterRegion,
    scatter_budget: ScatterBudget,
    field: &ScatterDensityField,
    budget: FieldScatterBudget,
) -> Result<FieldScatterOutput, ScatterFieldError> {
    let estimate = estimate_scatter_work(config, region)?;
    let per_sample = field.work_per_sample();
    let required = estimate
        .primary_candidates
        .checked_mul(per_sample)
        .ok_or(ScatterFieldError::ArithmeticOverflow)?;
    if required > budget.max_field_work {
        return Err(ScatterFieldError::WorkBudgetExceeded {
            required,
            limit: budget.max_field_work,
        });
    }
    let mut source = scatter(config, region, scatter_budget)?;
    let field_work = (source.points.len() as u64)
        .checked_mul(per_sample)
        .ok_or(ScatterFieldError::ArithmeticOverflow)?;
    let point_bytes = source
        .points
        .capacity()
        .checked_mul(size_of::<ScatterPoint>())
        .ok_or(ScatterFieldError::ArithmeticOverflow)?;
    check_output_bytes(point_bytes, source.points.len(), budget.max_output_bytes)?;
    let mut attributes = Vec::new();
    attributes
        .try_reserve_exact(source.points.len())
        .map_err(|_| ScatterFieldError::AllocationFailed)?;
    check_output_bytes(point_bytes, attributes.capacity(), budget.max_output_bytes)?;
    source.points.retain(|point| {
        let sample = field.sample(point.position_mm);
        if !field.accepts(point.key, sample.density) {
            return false;
        }
        attributes.push(sample);
        true
    });
    Ok(FieldScatterOutput {
        points: source.points,
        attributes,
        cache_key: FieldScatterCacheKey {
            scatter: source.cache_key,
            field_version: SCATTER_FIELD_VERSION,
            field: field.clone(),
        },
        candidate_evaluations: source.candidate_evaluations,
        field_work,
    })
}

fn check_output_bytes(
    points: usize,
    attributes: usize,
    limit: usize,
) -> Result<(), ScatterFieldError> {
    let required = attributes
        .checked_mul(size_of::<ScatterFieldSample>())
        .and_then(|bytes| bytes.checked_add(points))
        .ok_or(ScatterFieldError::ArithmeticOverflow)?;
    if required > limit {
        return Err(ScatterFieldError::OutputBytesBudgetExceeded { required, limit });
    }
    Ok(())
}

fn mix(value: u64) -> u64 {
    (u64::from(pcg_hash_u32(value)) << 32) | u64::from(pcg_hash_u32(value ^ 0x9e37_79b9_7f4a_7c15))
}

// Round-to-nearest with half upward, denominator 65535. The nonnegative
// weighted-sum form avoids asymmetric negative-difference rounding.
fn lerp(a: u16, b: u16, weight: u16) -> u16 {
    let weight = u64::from(weight);
    ((u64::from(a) * (ONE - weight) + u64::from(b) * weight + ONE / 2) / ONE) as u16
}

fn smoothstep(weight: u16) -> u16 {
    let t = u64::from(weight);
    let denominator = ONE * ONE;
    ((t * t * (3 * ONE - 2 * t) + denominator / 2) / denominator) as u16
}

fn value_noise(position: [i128; 3], cell_size_mm: u32, seed: u64, channel: u64) -> u16 {
    let size = i128::from(cell_size_mm);
    let grid = position.map(|p| p.div_euclid(size));
    let weights = position
        .map(|p| smoothstep((p.rem_euclid(size) as u64 * ONE / u64::from(cell_size_mm)) as u16));
    let base = mix(seed ^ channel.wrapping_mul(0xd1b5_4a32_d192_ed03));
    let mut corners = [0_u16; 8];
    for x in 0..2 {
        // Adjacent corners share hash prefixes. Hoisting them preserves every
        // bit while reducing 24 mixes per band to 2 + 4 + 8, with no cache.
        let hash_x = mix(base ^ (grid[0] + x as i128) as u64);
        for y in 0..2 {
            let hash_xy = mix(hash_x ^ (grid[1] + y as i128) as u64);
            for z in 0..2 {
                let hash = mix(hash_xy ^ (grid[2] + z as i128) as u64);
                corners[x * 4 + y * 2 + z] = (pcg_hash_u32(hash) >> 16) as u16;
            }
        }
    }
    let x0 = lerp(
        lerp(corners[0], corners[1], weights[2]),
        lerp(corners[2], corners[3], weights[2]),
        weights[1],
    );
    let x1 = lerp(
        lerp(corners[4], corners[5], weights[2]),
        lerp(corners[6], corners[7], weights[2]),
        weights[1],
    );
    lerp(x0, x1, weights[0])
}

fn brush_weight(brush: DensityBrush, position: [i64; 3]) -> u16 {
    if brush.opacity == 0 {
        return 0;
    }
    let outer = u128::from(brush.outer_radius_mm).pow(2);
    let mut distance = 0_u128;
    for (p, center) in position.into_iter().zip(brush.center_mm) {
        let difference = (i128::from(p) - i128::from(center)).unsigned_abs();
        // Each difference fits u64, its square fits u128. Compare before
        // adding so even three opposite-world-corner squares cannot overflow.
        let square = difference * difference;
        if square >= outer - distance {
            return 0;
        }
        distance += square;
    }
    let inner = u128::from(brush.inner_radius_mm).pow(2);
    let radial = if distance <= inner {
        u16::MAX
    } else {
        smoothstep(((outer - distance) * u128::from(u16::MAX) / (outer - inner)) as u16)
    };
    ((u64::from(radial) * u64::from(brush.opacity) + ONE / 2) / ONE) as u16
}

#[cfg(test)]
mod tests;
