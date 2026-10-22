//! Screen-space-error cut selection over a cluster LOD hierarchy (DAG).
//!
//! Paged geometry is authored as a hierarchy of cluster groups: the root holds
//! the coarsest simplification of a region and each descent step roughly halves
//! the geometric error of its children, down to the original triangles at the
//! leaves. Rendering one frame means choosing a *cut* through this hierarchy —
//! one cluster per surface region — such that every drawn cluster is the
//! coarsest simplification whose on-screen error still fits the pixel budget.
//! Because the error bound is monotonic from leaf to root, a single top-down
//! walk finds that cut without cracks: descend while a node is too coarse, stop
//! and draw as soon as a node is fine enough.
//!
//! This walk is the runtime heart of a Nanite-style pipeline and stays here in
//! the GPU-independent decision layer so it can be unit-tested against a
//! hand-built hierarchy. Like the rest of this module it is trig/sqrt-free: the
//! projected-error comparison is evaluated in squared form so the camera-to-
//! cluster distance never needs a square root, and frustum planes arrive
//! already normalized from the render layer.
//!
//! Frustum culling prunes whole subtrees during the walk: a node's bounds
//! enclose its children, so a node outside the frustum cannot expose a visible
//! child and its subtree is skipped. Occlusion is intentionally *not* applied
//! here — a parent being occluded does not prove every child is — and is left
//! to the per-cluster [`super::pipeline`] stage that runs after the cut is
//! known.

use super::lod::LodProjection;
use super::page_table::GeometryPageTable;
use super::{cull::Frustum, GeometryPageKey};
use crate::gpu_scene::SceneBounds;
use alloc::vec::Vec;

/// One node of a cluster LOD hierarchy.
///
/// Children occupy the half-open node-array range
/// `first_child .. first_child + child_count`; a node with `child_count == 0`
/// is a leaf holding original (finest) geometry. `self_error` is the node's
/// object-space geometric deviation bound in world units and must be
/// non-decreasing toward the root for the cut to be crack-free.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClusterNode {
    /// World-space bounds for culling and error projection.
    pub bounds: SceneBounds,
    /// Object-space geometric error bound of this node, in world units.
    pub self_error: f32,
    /// Page that must be resident to raster this node's cluster.
    pub page: GeometryPageKey,
    /// Index of the first child in the owning node array.
    pub first_child: u32,
    /// Number of contiguous children; `0` marks a leaf.
    pub child_count: u32,
}

impl ClusterNode {
    /// Builds a leaf node (no children) from its bounds, error and page.
    #[must_use]
    pub const fn leaf(bounds: SceneBounds, self_error: f32, page: GeometryPageKey) -> Self {
        Self {
            bounds,
            self_error,
            page,
            first_child: 0,
            child_count: 0,
        }
    }

    /// Builds an interior node spanning `child_count` children starting at
    /// `first_child` in the owning node array.
    #[must_use]
    pub const fn interior(
        bounds: SceneBounds,
        self_error: f32,
        page: GeometryPageKey,
        first_child: u32,
        child_count: u32,
    ) -> Self {
        Self {
            bounds,
            self_error,
            page,
            first_child,
            child_count,
        }
    }

    /// Whether this node holds original geometry with no finer refinement.
    #[must_use]
    pub const fn is_leaf(&self) -> bool {
        self.child_count == 0
    }
}

/// One entry of a selected cut: which node to draw and its page.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct CutCluster {
    /// Index of the drawn node in the hierarchy's node array.
    pub node: u32,
    /// Page backing the drawn cluster.
    pub page: GeometryPageKey,
}

/// A cluster LOD hierarchy: a node array plus the indices of its coarsest
/// (root) nodes. One physical asset may expose several roots when it is
/// partitioned into independent regions.
#[derive(Clone, Debug, PartialEq)]
pub struct ClusterHierarchy {
    nodes: Vec<ClusterNode>,
    roots: Vec<u32>,
    well_formed: bool,
}

impl ClusterHierarchy {
    /// Wraps a node array and its root indices.
    #[must_use]
    pub fn new(nodes: Vec<ClusterNode>, roots: Vec<u32>) -> Self {
        let well_formed = Self::validate_parts(&nodes, &roots);
        Self {
            nodes,
            roots,
            well_formed,
        }
    }

    /// The backing node array.
    #[must_use]
    pub fn nodes(&self) -> &[ClusterNode] {
        &self.nodes
    }

    /// The root (coarsest) node indices.
    #[must_use]
    pub fn roots(&self) -> &[u32] {
        &self.roots
    }

    /// Returns whether construction-time structural validation succeeded.
    ///
    /// Validation checks every root and child range and rejects cycles. It runs
    /// once in [`Self::new`], not per frame; cut selection therefore does not
    /// scan or allocate for the complete asset before applying view culling.
    #[must_use]
    pub fn is_well_formed(&self) -> bool {
        self.well_formed
    }

    /// Validates borrowed hierarchy tables without cloning their nodes.
    ///
    /// Asset admission uses this entry point so validating a large CLST asset
    /// does not temporarily duplicate its complete hierarchy.
    pub(super) fn validate_parts(nodes: &[ClusterNode], roots: &[u32]) -> bool {
        let len = nodes.len();
        let Ok(len) = u32::try_from(len) else {
            // More nodes than a u32 index can address; treat as malformed.
            return false;
        };
        if roots.iter().any(|&r| r >= len) {
            return false;
        }
        if !nodes.iter().all(|node| {
            let Some(end) = node.first_child.checked_add(node.child_count) else {
                return false;
            };
            node.child_count == 0 || end <= len
        }) {
            return false;
        }

        // A malformed cycle would make the cut walk revisit nodes forever. Use
        // iterative depth-first search so validation remains safe for deeply
        // nested assets without consuming the call stack.
        let mut state = alloc::vec![0_u8; nodes.len()];
        for start in 0..nodes.len() {
            if state[start] != 0 {
                continue;
            }
            state[start] = 1; // currently being visited
            let mut stack = alloc::vec![(start, 0_u32)];
            while let Some((index, next_child)) = stack.last_mut() {
                let node = nodes[*index];
                if *next_child == node.child_count {
                    state[*index] = 2; // completely visited
                    stack.pop();
                    continue;
                }

                let child = node.first_child + *next_child;
                *next_child += 1;
                match state[child as usize] {
                    0 => {
                        state[child as usize] = 1;
                        stack.push((child as usize, 0));
                    }
                    1 => return false, // back-edge: self-reference or cycle
                    _ => {}            // A shared child is valid in this DAG representation.
                }
            }
        }
        true
    }

    /// Selects the screen-space-error cut for one view.
    ///
    /// `view_origin` is the camera world position; `frustum` holds normalized
    /// inward planes; `projection` maps object-space error to pixels; and
    /// `target_error_pixels` is the per-frame pixel budget — the coarsest node
    /// whose projected error fits it is drawn. The walk descends only into
    /// frustum-visible interior nodes that are still too coarse, so the
    /// returned cut contains at most one cluster per visible surface region.
    ///
    /// Returns an empty cut when the hierarchy is malformed (see
    /// [`Self::is_well_formed`]) so a bad asset degrades to "draw nothing"
    /// rather than reading out of bounds.
    #[must_use]
    pub fn select_cut(
        &self,
        view_origin: [f32; 3],
        frustum: &Frustum,
        projection: LodProjection,
        target_error_pixels: f32,
    ) -> Vec<CutCluster> {
        self.walk_cut(view_origin, frustum, projection, target_error_pixels, None)
    }

    /// Selects the cut and, in the same walk, records a streaming request for
    /// every drawn cluster's page into `table`, stamped with `frame`.
    ///
    /// The request priority is the cluster's projected screen coverage
    /// (`radius * focal / distance`) so the residency budget favours the pages
    /// covering the most pixels; it is compared in squared form to stay
    /// square-root-free, which preserves the ordering because all terms are
    /// non-negative. A culled or refined-through node records nothing, so the
    /// table tracks exactly the visible working set the cut will raster.
    pub fn select_cut_streaming(
        &self,
        view_origin: [f32; 3],
        frustum: &Frustum,
        projection: LodProjection,
        target_error_pixels: f32,
        table: &mut GeometryPageTable,
        frame: u64,
    ) -> Vec<CutCluster> {
        self.walk_cut(
            view_origin,
            frustum,
            projection,
            target_error_pixels,
            Some((table, frame)),
        )
    }

    fn walk_cut(
        &self,
        view_origin: [f32; 3],
        frustum: &Frustum,
        projection: LodProjection,
        target_error_pixels: f32,
        mut streaming: Option<(&mut GeometryPageTable, u64)>,
    ) -> Vec<CutCluster> {
        let mut cut = Vec::new();
        if !self.is_well_formed() {
            return cut;
        }
        let budget = target_error_pixels.max(0.0);
        let mut stack = Vec::new();
        for &root in self.roots.iter().rev() {
            stack.push(root);
            while let Some(index) = stack.pop() {
                let node = self.nodes[index as usize];
                if !frustum.intersects_bounds(&node.bounds) {
                    // The node's bounds enclose every child, so nothing visible
                    // hides in this subtree: prune it wholesale.
                    continue;
                }
                if node.is_leaf()
                    || fits_budget(
                        node.self_error,
                        &node.bounds,
                        view_origin,
                        projection,
                        budget,
                    )
                {
                    // Coarsest acceptable simplification (or the finest available
                    // at a leaf): draw it and stop refining this branch.
                    if let Some((table, frame)) = streaming.as_mut() {
                        let priority = coverage_priority(&node.bounds, view_origin, projection);
                        table.request(node.page, priority, *frame);
                    }
                    cut.push(CutCluster {
                        node: index,
                        page: node.page,
                    });
                    continue;
                }
                let first = node.first_child;
                for offset in 0..node.child_count {
                    stack.push(first + offset);
                }
            }
        }
        cut
    }
}

impl Default for ClusterHierarchy {
    fn default() -> Self {
        Self::new(Vec::new(), Vec::new())
    }
}

/// Whether `self_error` projects to no more than `budget` pixels at the camera.
///
/// Evaluated in squared form to keep the layer square-root-free: the pixel test
/// `error * focal / distance <= budget` is multiplied through by the
/// (non-negative) distance and squared, so only the *squared* camera-to-center
/// distance is needed. When the camera sits on the center the squared distance
/// is zero and a non-zero error never fits, which correctly forces refinement
/// down to the finest leaf.
fn fits_budget(
    self_error: f32,
    bounds: &SceneBounds,
    view_origin: [f32; 3],
    projection: LodProjection,
    budget: f32,
) -> bool {
    let dx = bounds.center[0] - view_origin[0];
    let dy = bounds.center[1] - view_origin[1];
    let dz = bounds.center[2] - view_origin[2];
    let distance_sq = dx * dx + dy * dy + dz * dz;
    let projected = self_error.max(0.0) * projection.focal_length_pixels;
    // projected <= budget * distance  <=>  projected^2 <= budget^2 * distance^2
    projected * projected <= budget * budget * distance_sq
}

/// Squared projected screen coverage of `bounds` at the camera, used as a
/// streaming priority. The true coverage is `radius * focal / distance`; the
/// squared form `radius^2 * focal^2 / distance_sq` keeps the layer
/// square-root-free and preserves ordering because every term is non-negative.
/// A camera resting on the center yields the maximum priority so an enveloping
/// cluster is never starved.
pub fn coverage_priority(
    bounds: &SceneBounds,
    view_origin: [f32; 3],
    projection: LodProjection,
) -> f32 {
    let dx = bounds.center[0] - view_origin[0];
    let dy = bounds.center[1] - view_origin[1];
    let dz = bounds.center[2] - view_origin[2];
    let distance_sq = (dx * dx + dy * dy + dz * dz).max(f32::EPSILON);
    let extent = bounds.radius.max(0.0) * projection.focal_length_pixels;
    extent * extent / distance_sq
}

#[cfg(test)]
mod tests {
    use super::super::cull::Plane;
    use super::*;

    fn wide_frustum() -> Frustum {
        Frustum::from_planes([
            Plane::new([1.0, 0.0, 0.0], 1000.0),
            Plane::new([-1.0, 0.0, 0.0], 1000.0),
            Plane::new([0.0, 1.0, 0.0], 1000.0),
            Plane::new([0.0, -1.0, 0.0], 1000.0),
            Plane::new([0.0, 0.0, 1.0], 0.0),
            Plane::new([0.0, 0.0, -1.0], 10000.0),
        ])
    }

    fn bounds_at(z: f32, radius: f32) -> SceneBounds {
        SceneBounds {
            center: [0.0, 0.0, z],
            radius,
            half_extents: [radius, radius, radius],
            _padding: 0.0,
        }
    }

    // Root (coarse) with two leaf children (fine). Errors are monotonic:
    // root = 4.0 world units, leaves = 0.5.
    fn two_level_hierarchy() -> ClusterHierarchy {
        let root =
            ClusterNode::interior(bounds_at(50.0, 8.0), 4.0, GeometryPageKey::new(0, 0), 1, 2);
        let left = ClusterNode::leaf(bounds_at(50.0, 4.0), 0.5, GeometryPageKey::new(0, 1));
        let right = ClusterNode::leaf(bounds_at(50.0, 4.0), 0.5, GeometryPageKey::new(0, 2));
        ClusterHierarchy::new(alloc::vec![root, left, right], alloc::vec![0])
    }

    #[test]
    fn distant_view_draws_the_coarse_root() {
        let h = two_level_hierarchy();
        // 1000px focal, root error 4.0 at distance 50 => 80px projected; only a
        // very loose budget keeps the root, so pick 200px.
        let projection = LodProjection::from_focal_length_pixels(1000.0);
        let cut = h.select_cut([0.0, 0.0, 0.0], &wide_frustum(), projection, 200.0);
        assert_eq!(cut.len(), 1);
        assert_eq!(cut[0].node, 0);
        assert_eq!(cut[0].page, GeometryPageKey::new(0, 0));
    }

    #[test]
    fn near_view_refines_to_both_leaves() {
        let h = two_level_hierarchy();
        // Root projects to 80px which exceeds a tight 4px budget, so refine;
        // leaves project to 10px each, still over budget, but leaves are the
        // finest available and are always drawn.
        let projection = LodProjection::from_focal_length_pixels(1000.0);
        let cut = h.select_cut([0.0, 0.0, 0.0], &wide_frustum(), projection, 4.0);
        assert_eq!(cut.len(), 2);
        let mut pages: Vec<u32> = cut.iter().map(|c| c.page.page).collect();
        pages.sort_unstable();
        assert_eq!(pages, alloc::vec![1, 2]);
    }

    #[test]
    fn frustum_culled_root_prunes_the_whole_subtree() {
        let h = two_level_hierarchy();
        // A frustum entirely to the right (interior is x >= 500) rejects the
        // root at x = 0, so no child is even considered.
        let frustum = Frustum::from_planes([
            Plane::new([1.0, 0.0, 0.0], -500.0),
            Plane::new([-1.0, 0.0, 0.0], 1000.0),
            Plane::new([0.0, 1.0, 0.0], 1000.0),
            Plane::new([0.0, -1.0, 0.0], 1000.0),
            Plane::new([0.0, 0.0, 1.0], 0.0),
            Plane::new([0.0, 0.0, -1.0], 10000.0),
        ]);
        let projection = LodProjection::from_focal_length_pixels(1000.0);
        let cut = h.select_cut([0.0, 0.0, 0.0], &frustum, projection, 4.0);
        assert!(cut.is_empty());
    }

    #[test]
    fn malformed_child_range_yields_empty_cut() {
        // Child range runs past the end of the node array.
        let bad =
            ClusterNode::interior(bounds_at(50.0, 8.0), 4.0, GeometryPageKey::new(0, 0), 1, 5);
        let h = ClusterHierarchy::new(alloc::vec![bad], alloc::vec![0]);
        assert!(!h.is_well_formed());
        let projection = LodProjection::from_focal_length_pixels(1000.0);
        let cut = h.select_cut([0.0, 0.0, 0.0], &wide_frustum(), projection, 4.0);
        assert!(cut.is_empty());
    }

    #[test]
    fn streaming_variant_requests_only_drawn_pages() {
        use super::super::page_table::{GeometryPageTable, PageResidency};
        let h = two_level_hierarchy();
        let projection = LodProjection::from_focal_length_pixels(1000.0);
        let mut table = GeometryPageTable::new();
        // Tight budget refines past the root to both leaves.
        let cut = h.select_cut_streaming(
            [0.0, 0.0, 0.0],
            &wide_frustum(),
            projection,
            4.0,
            &mut table,
            9,
        );
        assert_eq!(cut.len(), 2);
        // Exactly the two drawn leaf pages are requested; the root is not.
        assert_eq!(table.len(), 2);
        assert_eq!(
            table.residency(GeometryPageKey::new(0, 1)),
            PageResidency::Requested
        );
        assert_eq!(
            table.residency(GeometryPageKey::new(0, 2)),
            PageResidency::Requested
        );
        assert_eq!(
            table.residency(GeometryPageKey::new(0, 0)),
            PageResidency::Unloaded
        );
    }

    #[test]
    fn self_reference_is_rejected() {
        let self_ref =
            ClusterNode::interior(bounds_at(50.0, 8.0), 4.0, GeometryPageKey::new(0, 0), 0, 1);
        let h = ClusterHierarchy::new(alloc::vec![self_ref], alloc::vec![0]);
        assert!(!h.is_well_formed());
        let projection = LodProjection::from_focal_length_pixels(1000.0);
        assert!(h
            .select_cut([0.0, 0.0, 0.0], &wide_frustum(), projection, 4.0)
            .is_empty());
    }

    #[test]
    fn indirect_cycle_is_rejected() {
        let first =
            ClusterNode::interior(bounds_at(50.0, 8.0), 4.0, GeometryPageKey::new(0, 0), 1, 1);
        let second =
            ClusterNode::interior(bounds_at(50.0, 4.0), 2.0, GeometryPageKey::new(0, 1), 0, 1);
        let h = ClusterHierarchy::new(alloc::vec![first, second], alloc::vec![0]);
        assert!(!h.is_well_formed());
    }

    #[test]
    fn out_of_range_root_is_rejected() {
        let leaf = ClusterNode::leaf(bounds_at(50.0, 4.0), 0.5, GeometryPageKey::new(0, 1));
        let h = ClusterHierarchy::new(alloc::vec![leaf], alloc::vec![3]);
        assert!(!h.is_well_formed());
    }
}
