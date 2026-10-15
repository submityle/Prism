//! Disposable CPU spatial acceleration for an immutable author-density field.
//!
//! A deterministic median BVH uses the largest centre-coordinate spread (X,
//! then Y, then Z on ties) and canonical brush index to break coordinate ties.
//! Flat preorder nodes skip a missed subtree using its escape offset. Every
//! brush, including inactive strokes, is indexed and retained in the shared
//! field; neither the tree nor its spatial ordering is a new content truth.
//!
//! Queries allocate nothing. With K AABB hits, work in time is
//! O(visited_nodes + K log K + K), including restoring canonical ID order before
//! noncommutative composition. Overlapping brushes can degenerate to a full
//! scan plus sorting. Logical probe budgets do not bound sorting comparisons,
//! CPU instructions, wall time, allocator metadata, or stack space.

use alloc::vec::Vec;
use core::{fmt, mem::size_of};

use crate::{
    estimate_scatter_work, scatter, DensityBrush, FieldScatterBudget, FieldScatterCacheKey,
    FieldScatterOutput, ScatterBudget, ScatterConfig, ScatterDensityField, ScatterError,
    ScatterFieldError, ScatterFieldSample, ScatterPoint, ScatterRegion, SCATTER_FIELD_VERSION,
};

/// Limits on owned index buffers, excluding the shared field and Vec headers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ScatterBrushIndexBudget {
    /// Maximum complete brush count, including inactive strokes.
    pub max_brushes: usize,
    /// Combined actual capacity bytes of nodes and canonical-index storage.
    pub max_index_bytes: usize,
}

/// Index construction, scratch, query-budget, or underlying field failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScatterBrushIndexError {
    /// The underlying field/scatter operation rejected the request.
    Field(ScatterFieldError),
    /// The complete field has too many brushes.
    BrushBudgetExceeded {
        /// Complete brush count.
        required: usize,
        /// Maximum permitted count.
        limit: usize,
    },
    /// Owned node and canonical-index capacities exceed the index byte budget.
    IndexBytesBudgetExceeded {
        /// Required capacity payload bytes.
        required: usize,
        /// Maximum permitted payload bytes.
        limit: usize,
    },
    /// Scratch capacity exceeds its byte budget.
    ScratchBytesBudgetExceeded {
        /// Required capacity payload bytes.
        required: usize,
        /// Maximum permitted payload bytes.
        limit: usize,
    },
    /// Scratch lacks room for the complete field's canonical indices.
    ScratchCapacityExceeded {
        /// Complete brush count, including inactive strokes.
        required: usize,
        /// Actual supplied index capacity.
        limit: usize,
    },
    /// A query's next probe or a whole request's preflight exceeds its budget.
    WorkBudgetExceeded {
        /// Cumulative work needed for the next probe, or preflight upper bound.
        required: u64,
        /// Maximum permitted logical probe units.
        limit: u64,
    },
    /// A count, capacity byte size, or logical work total cannot be represented.
    ArithmeticOverflow,
    /// A fallible fixed-capacity reservation failed.
    AllocationFailed,
}

impl fmt::Display for ScatterBrushIndexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Field(error) => write!(f, "indexed field: {error}"),
            Self::BrushBudgetExceeded { required, limit } => {
                write!(f, "brush index needs {required} brushes, limit {limit}")
            }
            Self::IndexBytesBudgetExceeded { required, limit } => {
                write!(f, "brush index needs {required} bytes, limit {limit}")
            }
            Self::ScratchBytesBudgetExceeded { required, limit } => {
                write!(f, "brush scratch needs {required} bytes, limit {limit}")
            }
            Self::ScratchCapacityExceeded { required, limit } => {
                write!(
                    f,
                    "brush scratch needs {required} indices, capacity {limit}"
                )
            }
            Self::WorkBudgetExceeded { required, limit } => {
                write!(
                    f,
                    "indexed field needs {required} work units, limit {limit}"
                )
            }
            Self::ArithmeticOverflow => f.write_str("brush index count, bytes or work overflow"),
            Self::AllocationFailed => f.write_str("brush index or output reservation failed"),
        }
    }
}

impl core::error::Error for ScatterBrushIndexError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Field(error) => Some(error),
            _ => None,
        }
    }
}

impl From<ScatterFieldError> for ScatterBrushIndexError {
    fn from(error: ScatterFieldError) -> Self {
        Self::Field(error)
    }
}

impl From<ScatterError> for ScatterBrushIndexError {
    fn from(error: ScatterError) -> Self {
        Self::Field(error.into())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Bounds {
    min: [i64; 3],
    max: [i64; 3],
}

impl Bounds {
    fn brush(brush: &DensityBrush) -> Self {
        // Widen before arithmetic. Closed, clamped bounds remain conservative
        // at both world extremes and at the zero-weight spherical boundary.
        let radius = i128::from(brush.outer_radius_mm);
        let clamp = |value: i128| value.clamp(i128::from(i64::MIN), i128::from(i64::MAX)) as i64;
        Self {
            min: brush.center_mm.map(|c| clamp(i128::from(c) - radius)),
            max: brush.center_mm.map(|c| clamp(i128::from(c) + radius)),
        }
    }

    fn contains(self, position: [i64; 3]) -> bool {
        (0..3).all(|axis| self.min[axis] <= position[axis] && position[axis] <= self.max[axis])
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Node {
    bounds: Bounds,
    // Exclusive subtree end. A leaf's escape is its own index plus one.
    escape: usize,
    // Spatial permutation slot, used only for leaves.
    brush_slot: usize,
}

/// Immutable rebuildable spatial cache over the complete shared field recipe.
///
/// Construction reserves nodes and indices once each, fallibly, and rechecks
/// actual capacities against the budget. Balanced median construction uses
/// O(B log B) time and O(log B) stack for B brushes, with no temporary heap
/// buffers or subsequent Vec growth. The field's Arc is shared, not copied.
#[derive(Debug)]
pub struct ScatterBrushIndex {
    field: ScatterDensityField,
    nodes: Vec<Node>,
    brush_indices: Vec<usize>,
    allocated_bytes: usize,
    max_sample_work: u64,
}

/// Caller-owned query storage, reusable across indices with sufficient capacity.
///
/// Its contents are private, have no semantic identity, and may change even
/// when a query fails. It never grows during sampling.
#[derive(Debug)]
pub struct ScatterBrushScratch {
    indices: Vec<usize>,
    allocated_bytes: usize,
}

impl ScatterBrushScratch {
    /// Actual number of canonical brush indices that can be stored.
    pub fn capacity(&self) -> usize {
        self.indices.capacity()
    }

    /// Actual capacity payload bytes, excluding the Vec header and metadata.
    pub fn allocated_bytes(&self) -> usize {
        self.allocated_bytes
    }
}

/// A complete sample and its actual logical probe costs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IndexedFieldSample {
    /// Exactly the same integer density/species as linear field sampling.
    pub sample: ScatterFieldSample,
    /// Number of AABB node tests, including misses.
    pub node_tests: u64,
    /// Number of canonical brush-weight/composition probes after AABB filtering.
    pub brush_tests: u64,
    /// Base noise/rank units plus actual node and brush probes.
    /// Sorting comparisons and CPU time are not represented by this count.
    pub field_work: u64,
}

impl ScatterBrushIndex {
    /// Builds a deterministic median BVH without changing field/cache semantics.
    pub fn new(
        field: ScatterDensityField,
        budget: ScatterBrushIndexBudget,
    ) -> Result<Self, ScatterBrushIndexError> {
        let count = field.brushes().len();
        if count > budget.max_brushes {
            return Err(ScatterBrushIndexError::BrushBudgetExceeded {
                required: count,
                limit: budget.max_brushes,
            });
        }
        let node_count = if count == 0 {
            0
        } else {
            count
                .checked_add(count - 1)
                .ok_or(ScatterBrushIndexError::ArithmeticOverflow)?
        };
        let planned_bytes = index_bytes(node_count, count)?;
        check_index_bytes(planned_bytes, budget.max_index_bytes)?;
        let max_sample_work = field
            .base_sample_work()
            .checked_add(as_work(node_count)?)
            .and_then(|work| work.checked_add(u64::try_from(count).ok()?))
            .ok_or(ScatterBrushIndexError::ArithmeticOverflow)?;
        let mut nodes = Vec::new();
        nodes
            .try_reserve_exact(node_count)
            .map_err(|_| ScatterBrushIndexError::AllocationFailed)?;
        check_index_bytes(
            index_bytes(nodes.capacity(), count)?,
            budget.max_index_bytes,
        )?;
        let mut brush_indices = Vec::new();
        brush_indices
            .try_reserve_exact(count)
            .map_err(|_| ScatterBrushIndexError::AllocationFailed)?;
        let allocated_bytes = index_bytes(nodes.capacity(), brush_indices.capacity())?;
        check_index_bytes(allocated_bytes, budget.max_index_bytes)?;
        brush_indices.extend(0..count);
        if count != 0 {
            build_nodes(field.brushes(), &mut brush_indices, 0, &mut nodes)?;
        }
        Ok(Self {
            field,
            nodes,
            brush_indices,
            allocated_bytes,
            max_sample_work,
        })
    }

    /// Complete immutable recipe, including every inactive/cache dependency.
    pub fn field(&self) -> &ScatterDensityField {
        &self.field
    }

    /// Number of preorder BVH nodes (zero for an empty field).
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Actual combined owned capacity bytes, excluding the shared field.
    pub fn allocated_bytes(&self) -> usize {
        self.allocated_bytes
    }

    /// Conservative base noise/rank units plus all nodes plus all brushes.
    /// This logical bound does not constrain sorting comparisons or CPU time.
    pub fn max_sample_work(&self) -> u64 {
        self.max_sample_work
    }

    /// Reserves room for all B canonical indices once, including inactive ones.
    /// Both requested bytes and actual capacity bytes must fit the byte limit.
    pub fn new_scratch(
        &self,
        max_scratch_bytes: usize,
    ) -> Result<ScatterBrushScratch, ScatterBrushIndexError> {
        let count = self.field.brushes().len();
        let check = |capacity| {
            let required = capacity_bytes::<usize>(capacity)?;
            if required > max_scratch_bytes {
                return Err(ScatterBrushIndexError::ScratchBytesBudgetExceeded {
                    required,
                    limit: max_scratch_bytes,
                });
            }
            Ok(())
        };
        check(count)?;
        let mut indices = Vec::new();
        indices
            .try_reserve_exact(count)
            .map_err(|_| ScatterBrushIndexError::AllocationFailed)?;
        check(indices.capacity())?;
        let allocated_bytes = capacity_bytes::<usize>(indices.capacity())?;
        Ok(ScatterBrushScratch {
            indices,
            allocated_bytes,
        })
    }

    /// Samples without allocation, charging actual work incrementally.
    ///
    /// Scratch capacity is validated first, even if the query would miss the
    /// tree. Base noise/rank work and each node/brush probe are charged before
    /// execution. A budget smaller than `max_sample_work()` can succeed for a
    /// sparse query. Failure returns no partial sample; scratch may be changed.
    /// All node traversal precedes sorting and canonical brush composition.
    pub fn sample(
        &self,
        position_mm: [i64; 3],
        scratch: &mut ScatterBrushScratch,
        max_work: u64,
    ) -> Result<IndexedFieldSample, ScatterBrushIndexError> {
        self.check_scratch(scratch)?;
        scratch.indices.clear();
        let mut field_work = 0;
        charge(&mut field_work, self.field.base_sample_work(), max_work)?;
        let mut sample = self.field.sample_base(position_mm);
        let mut node_tests = 0_u64;
        let mut cursor = 0_usize;
        while let Some(node) = self.nodes.get(cursor) {
            charge(&mut field_work, 1, max_work)?;
            node_tests = node_tests
                .checked_add(1)
                .ok_or(ScatterBrushIndexError::ArithmeticOverflow)?;
            let next = cursor
                .checked_add(1)
                .ok_or(ScatterBrushIndexError::ArithmeticOverflow)?;
            if !node.bounds.contains(position_mm) {
                cursor = node.escape;
            } else if node.escape == next {
                // At most B leaves can hit, and capacity >= B was checked.
                scratch.indices.push(self.brush_indices[node.brush_slot]);
                cursor = node.escape;
            } else {
                cursor = next;
            }
        }
        scratch.indices.sort_unstable();
        let mut brush_tests = 0_u64;
        for &index in &scratch.indices {
            charge(&mut field_work, 1, max_work)?;
            brush_tests = brush_tests
                .checked_add(1)
                .ok_or(ScatterBrushIndexError::ArithmeticOverflow)?;
            sample.apply_brush(self.field.brushes()[index], position_mm);
        }
        Ok(IndexedFieldSample {
            sample,
            node_tests,
            brush_tests,
            field_work,
        })
    }

    fn check_scratch(&self, scratch: &ScatterBrushScratch) -> Result<(), ScatterBrushIndexError> {
        let required = self.field.brushes().len();
        let limit = scratch.capacity();
        if limit < required {
            return Err(ScatterBrushIndexError::ScratchCapacityExceeded { required, limit });
        }
        Ok(())
    }
}

/// Generates and filters the same points/attributes/cache keys as linear fields.
///
/// Before source enumeration, checks `primary_candidates * max_sample_work()`
/// with checked arithmetic. The reported `field_work` instead sums actual
/// base/node/brush probe units for every source survivor, including rejected
/// ones. Scratch capacity is checked even for empty requests. Source budgets
/// apply before thinning, exactly as for [`crate::scatter_with_density_field`].
/// The original point Vec is compacted in place in one pass; attributes reserve
/// once for all source survivors. Combined actual capacity bytes are bounded
/// before and after that reservation. Failure exposes no partial output.
pub fn scatter_with_indexed_density_field(
    config: ScatterConfig,
    region: ScatterRegion,
    scatter_budget: ScatterBudget,
    index: &ScatterBrushIndex,
    scratch: &mut ScatterBrushScratch,
    budget: FieldScatterBudget,
) -> Result<FieldScatterOutput, ScatterBrushIndexError> {
    index.check_scratch(scratch)?;
    let estimate = estimate_scatter_work(config, region)?;
    let required = estimate
        .primary_candidates
        .checked_mul(index.max_sample_work())
        .ok_or(ScatterBrushIndexError::ArithmeticOverflow)?;
    if required > budget.max_field_work {
        return Err(ScatterBrushIndexError::WorkBudgetExceeded {
            required,
            limit: budget.max_field_work,
        });
    }
    let mut source = scatter(config, region, scatter_budget)?;
    let point_bytes = capacity_bytes::<ScatterPoint>(source.points.capacity())?;
    check_output_bytes(point_bytes, source.points.len(), budget.max_output_bytes)?;
    let mut attributes = Vec::new();
    attributes
        .try_reserve_exact(source.points.len())
        .map_err(|_| ScatterBrushIndexError::Field(ScatterFieldError::AllocationFailed))?;
    check_output_bytes(point_bytes, attributes.capacity(), budget.max_output_bytes)?;
    let mut retained = 0_usize;
    let mut field_work = 0_u64;
    for read in 0..source.points.len() {
        let point = source.points[read];
        let result = index.sample(point.position_mm, scratch, index.max_sample_work())?;
        charge(&mut field_work, result.field_work, budget.max_field_work)?;
        if index.field.accepts(point.key, result.sample.density) {
            source.points[retained] = point;
            retained = retained
                .checked_add(1)
                .ok_or(ScatterBrushIndexError::ArithmeticOverflow)?;
            attributes.push(result.sample);
        }
    }
    source.points.truncate(retained);
    Ok(FieldScatterOutput {
        points: source.points,
        attributes,
        cache_key: FieldScatterCacheKey {
            scatter: source.cache_key,
            field_version: SCATTER_FIELD_VERSION,
            field: index.field.clone(),
        },
        candidate_evaluations: source.candidate_evaluations,
        field_work,
    })
}

fn capacity_bytes<T>(capacity: usize) -> Result<usize, ScatterBrushIndexError> {
    capacity
        .checked_mul(size_of::<T>())
        .ok_or(ScatterBrushIndexError::ArithmeticOverflow)
}

fn as_work(count: usize) -> Result<u64, ScatterBrushIndexError> {
    u64::try_from(count).map_err(|_| ScatterBrushIndexError::ArithmeticOverflow)
}

fn index_bytes(nodes: usize, indices: usize) -> Result<usize, ScatterBrushIndexError> {
    capacity_bytes::<Node>(nodes)?
        .checked_add(capacity_bytes::<usize>(indices)?)
        .ok_or(ScatterBrushIndexError::ArithmeticOverflow)
}

fn check_index_bytes(required: usize, limit: usize) -> Result<(), ScatterBrushIndexError> {
    if required > limit {
        return Err(ScatterBrushIndexError::IndexBytesBudgetExceeded { required, limit });
    }
    Ok(())
}

fn check_output_bytes(
    points: usize,
    attributes: usize,
    limit: usize,
) -> Result<(), ScatterBrushIndexError> {
    let required = capacity_bytes::<ScatterFieldSample>(attributes)?
        .checked_add(points)
        .ok_or(ScatterBrushIndexError::ArithmeticOverflow)?;
    if required > limit {
        return Err(ScatterFieldError::OutputBytesBudgetExceeded { required, limit }.into());
    }
    Ok(())
}

fn charge(total: &mut u64, amount: u64, limit: u64) -> Result<(), ScatterBrushIndexError> {
    let required = total
        .checked_add(amount)
        .ok_or(ScatterBrushIndexError::ArithmeticOverflow)?;
    if required > limit {
        return Err(ScatterBrushIndexError::WorkBudgetExceeded { required, limit });
    }
    *total = required;
    Ok(())
}

// Median splits guarantee logarithmic recursion depth, even with coincident
// centres. No sort/build call allocates; all pushes stay within fixed capacity.
fn build_nodes(
    brushes: &[DensityBrush],
    indices: &mut [usize],
    slot: usize,
    nodes: &mut Vec<Node>,
) -> Result<(), ScatterBrushIndexError> {
    let first = indices
        .first()
        .ok_or(ScatterBrushIndexError::ArithmeticOverflow)?;
    let mut bounds = Bounds::brush(&brushes[*first]);
    let mut center_min = brushes[*first].center_mm;
    let mut center_max = center_min;
    for &index in indices.iter().skip(1) {
        let brush = &brushes[index];
        let brush_bounds = Bounds::brush(brush);
        for axis in 0..3 {
            bounds.min[axis] = bounds.min[axis].min(brush_bounds.min[axis]);
            bounds.max[axis] = bounds.max[axis].max(brush_bounds.max[axis]);
            center_min[axis] = center_min[axis].min(brush.center_mm[axis]);
            center_max[axis] = center_max[axis].max(brush.center_mm[axis]);
        }
    }
    let parent = nodes.len();
    if parent == nodes.capacity() {
        return Err(ScatterBrushIndexError::ArithmeticOverflow);
    }
    nodes.push(Node {
        bounds,
        escape: 0,
        brush_slot: slot,
    });
    if indices.len() > 1 {
        let spread = |axis| i128::from(center_max[axis]) - i128::from(center_min[axis]);
        let mut axis = 0;
        for candidate in 1..3 {
            if spread(candidate) > spread(axis) {
                axis = candidate;
            }
        }
        let middle = indices.len() / 2;
        indices.select_nth_unstable_by(middle, |a, b| {
            brushes[*a].center_mm[axis]
                .cmp(&brushes[*b].center_mm[axis])
                .then_with(|| a.cmp(b))
        });
        let right_slot = slot
            .checked_add(middle)
            .ok_or(ScatterBrushIndexError::ArithmeticOverflow)?;
        let (left, right) = indices.split_at_mut(middle);
        build_nodes(brushes, left, slot, nodes)?;
        build_nodes(brushes, right, right_slot, nodes)?;
    }
    nodes[parent].escape = nodes.len();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DensityBrushOp, DensityNoise, ScatterFieldBuildBudget, ScatterFieldConfig};
    use alloc::{vec, vec::Vec};

    fn brush(id: u64, center_mm: [i64; 3]) -> DensityBrush {
        DensityBrush {
            id,
            center_mm,
            inner_radius_mm: 0,
            outer_radius_mm: 10,
            opacity: u16::MAX,
            density: DensityBrushOp::Replace(12345),
            species: Some(id as u32),
        }
    }

    fn index_with_noise(
        brushes: Vec<DensityBrush>,
        noise: Option<DensityNoise>,
    ) -> ScatterBrushIndex {
        let field = ScatterDensityField::new(
            ScatterFieldConfig {
                seed: 47,
                version: 2,
                base_density: 23456,
                base_species: 7,
                noise,
            },
            brushes,
            ScatterFieldBuildBudget {
                max_brushes: usize::MAX,
                max_brush_bytes: usize::MAX,
            },
        )
        .unwrap();
        ScatterBrushIndex::new(
            field,
            ScatterBrushIndexBudget {
                max_brushes: usize::MAX,
                max_index_bytes: usize::MAX,
            },
        )
        .unwrap()
    }

    fn index(brushes: Vec<DensityBrush>) -> ScatterBrushIndex {
        index_with_noise(brushes, None)
    }

    #[test]
    fn preorder_escapes_and_leaf_slots_form_complete_deterministic_tree() {
        let brushes: Vec<_> = (0..37)
            .map(|id| brush(id, [id as i64 % 5, (id as i64 * 17) % 13, id as i64 * 100]))
            .collect();
        let first = index(brushes.clone());
        let mut reversed = brushes;
        reversed.reverse();
        let second = index(reversed);
        assert_eq!(first.nodes, second.nodes);
        assert_eq!(first.brush_indices, second.brush_indices);
        assert_eq!(first.node_count(), 73);
        assert_eq!(first.nodes[0].escape, 73);
        let mut seen_slots = vec![false; first.brush_indices.len()];
        for (offset, node) in first.nodes.iter().enumerate() {
            assert!(offset < node.escape && node.escape <= first.nodes.len());
            if node.escape == offset + 1 {
                assert!(!seen_slots[node.brush_slot]);
                seen_slots[node.brush_slot] = true;
                assert_eq!(
                    node.bounds,
                    Bounds::brush(&first.field.brushes()[first.brush_indices[node.brush_slot]])
                );
            } else {
                let left = offset + 1;
                let right = first.nodes[left].escape;
                assert!(right < node.escape);
                assert_eq!(first.nodes[right].escape, node.escape);
                for child in [left, right] {
                    assert!(node.bounds.contains(first.nodes[child].bounds.min));
                    assert!(node.bounds.contains(first.nodes[child].bounds.max));
                }
            }
        }
        assert!(seen_slots.into_iter().all(|seen| seen));
        let mut canonical = first.brush_indices.clone();
        canonical.sort_unstable();
        assert_eq!(canonical, (0..37).collect::<Vec<_>>());
        assert_eq!(
            first.allocated_bytes(),
            first.nodes.capacity() * size_of::<Node>()
                + first.brush_indices.capacity() * size_of::<usize>()
        );
    }

    #[test]
    fn median_selects_max_spread_axis_and_breaks_axis_and_coordinate_ties() {
        let widest_z = index(vec![
            brush(0, [0, 100, 900]),
            brush(1, [30, 0, 800]),
            brush(2, [10, -100, -800]),
            brush(3, [20, 1, -900]),
        ]);
        let mut left = widest_z.brush_indices[..2].to_vec();
        left.sort_unstable();
        assert_eq!(left, [2, 3]);
        let tied_axes = index(vec![
            brush(0, [1, -1, 0]),
            brush(1, [-1, 1, 0]),
            brush(2, [0, 0, 0]),
        ]);
        assert_eq!(tied_axes.brush_indices[0], 1); // X wins the X/Y tie.
        let coincident = index((0..33).rev().map(|id| brush(id, [0; 3])).collect());
        assert_eq!(coincident.brush_indices, (0..33).collect::<Vec<_>>());
    }

    #[test]
    fn clamped_bounds_keep_both_world_extremes_and_large_radii() {
        for center in [i64::MIN, i64::MAX] {
            let mut b = brush(0, [center; 3]);
            b.outer_radius_mm = u32::MAX;
            let tree = index(vec![b]);
            let mut scratch = tree.new_scratch(usize::MAX).unwrap();
            let bounds = tree.nodes[0].bounds;
            assert!(bounds.contains([center; 3]));
            if center == i64::MIN {
                assert_eq!(bounds.min, [i64::MIN; 3]);
            } else {
                assert_eq!(bounds.max, [i64::MAX; 3]);
            }
            for position in [[center; 3], bounds.min, bounds.max, [0; 3]] {
                let result = tree
                    .sample(position, &mut scratch, tree.max_sample_work())
                    .unwrap();
                assert_eq!(result.sample, tree.field.sample(position));
            }
        }
    }

    #[test]
    fn inactive_brushes_are_full_index_and_cache_dependencies() {
        let mut brushes: Vec<_> = (0..3).map(|id| brush(id, [0; 3])).collect();
        for b in &mut brushes {
            b.opacity = 0;
        }
        let tree = index(brushes);
        let mut scratch = tree.new_scratch(usize::MAX).unwrap();
        assert!(scratch.capacity() >= 3);
        assert_eq!(tree.node_count(), 5);
        assert_eq!(tree.max_sample_work(), 9);
        let result = tree.sample([0; 3], &mut scratch, 9).unwrap();
        assert_eq!(
            (result.node_tests, result.brush_tests, result.field_work),
            (5, 3, 9)
        );
        assert_eq!(result.sample, tree.field.sample([0; 3]));
        assert_eq!(scratch.indices, [0, 1, 2]);
    }

    #[test]
    fn incremental_budget_rejects_the_next_probe_and_can_use_less_than_max() {
        let tree = index(vec![brush(0, [0; 3])]);
        let mut scratch = tree.new_scratch(usize::MAX).unwrap();
        assert_eq!(tree.max_sample_work(), 3);
        assert_eq!(
            tree.sample([0; 3], &mut scratch, 0),
            Err(ScatterBrushIndexError::WorkBudgetExceeded {
                required: 1,
                limit: 0
            })
        );
        assert_eq!(
            tree.sample([0; 3], &mut scratch, 1),
            Err(ScatterBrushIndexError::WorkBudgetExceeded {
                required: 2,
                limit: 1
            })
        );
        assert_eq!(
            tree.sample([0; 3], &mut scratch, 2),
            Err(ScatterBrushIndexError::WorkBudgetExceeded {
                required: 3,
                limit: 2
            })
        );
        let miss = tree.sample([100; 3], &mut scratch, 2).unwrap();
        assert_eq!(
            (miss.node_tests, miss.brush_tests, miss.field_work),
            (1, 0, 2)
        );
        assert!(scratch.indices.is_empty());
        let boundary = tree.sample([10, 0, 0], &mut scratch, 3).unwrap();
        assert_eq!(boundary.brush_tests, 1); // Closed AABB, zero sphere weight.
        assert_eq!(boundary.sample, tree.field.sample([10, 0, 0]));
    }

    #[test]
    fn scratch_checks_precede_work_and_reuse_never_grows_owned_buffers() {
        let small = index(vec![]);
        let tree = index(
            (0..17)
                .map(|id| brush(id, [id as i64 * 100, 0, 0]))
                .collect(),
        );
        let mut insufficient = small.new_scratch(0).unwrap();
        assert_eq!(
            tree.sample([i64::MAX; 3], &mut insufficient, 0),
            Err(ScatterBrushIndexError::ScratchCapacityExceeded {
                required: 17,
                limit: 0
            })
        );
        let mut scratch = tree.new_scratch(usize::MAX).unwrap();
        let capacity = scratch.capacity();
        let bytes = scratch.allocated_bytes();
        let pointer = scratch.indices.as_ptr();
        let nodes = (tree.nodes.as_ptr(), tree.nodes.capacity());
        let indices = (tree.brush_indices.as_ptr(), tree.brush_indices.capacity());
        for position in [[0; 3], [150, 0, 0], [1600, 0, 0], [i64::MIN; 3]] {
            tree.sample(position, &mut scratch, tree.max_sample_work())
                .unwrap();
            small.sample(position, &mut scratch, 1).unwrap();
            assert!(scratch.indices.is_empty());
            assert_eq!(
                (scratch.capacity(), scratch.allocated_bytes()),
                (capacity, bytes)
            );
            assert_eq!(scratch.indices.as_ptr(), pointer);
            assert_eq!((tree.nodes.as_ptr(), tree.nodes.capacity()), nodes);
            assert_eq!(
                (tree.brush_indices.as_ptr(), tree.brush_indices.capacity()),
                indices
            );
        }
    }

    #[test]
    fn empty_tree_charges_enabled_noise_and_rank_without_node_probes() {
        for (amplitude, warp_amplitude_mm, base) in [(0, 100, 1), (3, 0, 9), (3, 100, 33)] {
            let tree = index_with_noise(
                vec![],
                Some(DensityNoise {
                    cell_size_mm: 7,
                    amplitude,
                    warp_cell_size_mm: 11,
                    warp_amplitude_mm,
                }),
            );
            let mut scratch = tree.new_scratch(0).unwrap();
            assert_eq!(tree.allocated_bytes(), 0);
            assert_eq!(scratch.allocated_bytes(), 0);
            assert_eq!(tree.max_sample_work(), base);
            let position = [i64::MIN, 9, i64::MAX];
            let sample = tree.sample(position, &mut scratch, base).unwrap();
            assert_eq!(
                (sample.node_tests, sample.brush_tests, sample.field_work),
                (0, 0, base)
            );
            assert_eq!(sample.sample, tree.field.sample(position));
            assert_eq!(
                tree.sample(position, &mut scratch, base - 1),
                Err(ScatterBrushIndexError::WorkBudgetExceeded {
                    required: base,
                    limit: base - 1
                })
            );
        }
    }

    #[test]
    fn checked_size_and_charge_helpers_return_typed_overflow_errors() {
        assert_eq!(
            capacity_bytes::<Node>(usize::MAX),
            Err(ScatterBrushIndexError::ArithmeticOverflow)
        );
        assert_eq!(
            index_bytes(usize::MAX, 0),
            Err(ScatterBrushIndexError::ArithmeticOverflow)
        );
        assert_eq!(
            check_output_bytes(usize::MAX, 1, usize::MAX),
            Err(ScatterBrushIndexError::ArithmeticOverflow)
        );
        let mut total = u64::MAX;
        assert_eq!(
            charge(&mut total, 1, u64::MAX),
            Err(ScatterBrushIndexError::ArithmeticOverflow)
        );
        assert_eq!(total, u64::MAX);
    }
}
