//! Software `BVH`: primitive bounds, binned-`SAH` build, and a linear node layout.
//!
//! This is the `CPU`-verifiable golden reference for the bounding-volume
//! hierarchy that a `GPU` build kernel mirrors bit-for-bit. It owns three
//! things:
//!
//! - [`Aabb`] — an axis-aligned bounding box with the union / surface-area
//!   arithmetic the surface-area heuristic (`SAH`) needs.
//! - [`Triangle`] — a single indexed triangle plus its derived bounds/centroid.
//! - [`Bvh`] — a depth-first *linear* node array ([`LinearBvhNode`]) produced by
//!   a binned-`SAH` builder, laid out so a stackless or short-stack `GPU`
//!   traversal can walk it by simple index arithmetic.
//!
//! The layout deliberately matches the classic `pbrt`/`embree` flattening: an
//! interior node stores its *first* child immediately after itself and an
//! explicit offset to its *second* child, while a leaf stores a contiguous span
//! into a reordered primitive-index table. Traversal lives in
//! [`super::traversal`].
//!
//! All geometry is `f32` so the numbers agree with the `GPU` shader that
//! consumes this structure; only the `SAH` cost accumulator widens to `f64` to
//! keep bin comparisons stable, which never changes the resulting topology
//! relative to a matching `GPU` builder that does the same.

/// Axis index: `0 = x`, `1 = y`, `2 = z`.
pub type Axis = usize;

/// Axis-aligned bounding box in world space.
///
/// An *empty* box has `min` set to `+inf` and `max` to `-inf`, so
/// [`Aabb::union`] and [`Aabb::enclose`] behave as identity elements.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Aabb {
    /// Minimum corner (`x`, `y`, `z`).
    pub min: [f32; 3],
    /// Maximum corner (`x`, `y`, `z`).
    pub max: [f32; 3],
}

impl Aabb {
    /// The empty box: `min = +inf`, `max = -inf`.
    ///
    /// Unioning anything into this yields exactly that thing, which makes it the
    /// correct seed for a reduction over primitive bounds.
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            min: [f32::INFINITY; 3],
            max: [f32::NEG_INFINITY; 3],
        }
    }

    /// A box spanning `[min, max]` with no validity checks.
    #[must_use]
    pub const fn new(min: [f32; 3], max: [f32; 3]) -> Self {
        Self { min, max }
    }

    /// A degenerate box containing exactly one point.
    #[must_use]
    pub const fn point(p: [f32; 3]) -> Self {
        Self { min: p, max: p }
    }

    /// True when the box holds no volume (never had a point unioned in).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.min[0] > self.max[0] || self.min[1] > self.max[1] || self.min[2] > self.max[2]
    }

    /// Smallest box containing both `self` and `other`.
    #[must_use]
    pub fn union(&self, other: &Aabb) -> Aabb {
        Aabb {
            min: [
                self.min[0].min(other.min[0]),
                self.min[1].min(other.min[1]),
                self.min[2].min(other.min[2]),
            ],
            max: [
                self.max[0].max(other.max[0]),
                self.max[1].max(other.max[1]),
                self.max[2].max(other.max[2]),
            ],
        }
    }

    /// Smallest box containing `self` and the point `p`.
    #[must_use]
    pub fn enclose(&self, p: [f32; 3]) -> Aabb {
        Aabb {
            min: [
                self.min[0].min(p[0]),
                self.min[1].min(p[1]),
                self.min[2].min(p[2]),
            ],
            max: [
                self.max[0].max(p[0]),
                self.max[1].max(p[1]),
                self.max[2].max(p[2]),
            ],
        }
    }

    /// Per-axis extent `max - min`. Negative components indicate an empty box.
    #[must_use]
    pub fn extent(&self) -> [f32; 3] {
        [
            self.max[0] - self.min[0],
            self.max[1] - self.min[1],
            self.max[2] - self.min[2],
        ]
    }

    /// Geometric center of the box.
    #[must_use]
    pub fn centroid(&self) -> [f32; 3] {
        [
            0.5 * (self.min[0] + self.max[0]),
            0.5 * (self.min[1] + self.max[1]),
            0.5 * (self.min[2] + self.max[2]),
        ]
    }

    /// Index of the longest axis (`0`/`1`/`2`); ties prefer the lower axis.
    #[must_use]
    pub fn max_extent_axis(&self) -> Axis {
        let e = self.extent();
        if e[0] >= e[1] && e[0] >= e[2] {
            0
        } else if e[1] >= e[2] {
            1
        } else {
            2
        }
    }

    /// Total surface area of the box; an empty box has area `0`.
    ///
    /// This is the `SAH` cost weight: the probability a random ray that hits the
    /// parent also enters this box is proportional to its surface area.
    #[must_use]
    pub fn surface_area(&self) -> f32 {
        if self.is_empty() {
            return 0.0;
        }
        let e = self.extent();
        2.0 * (e[0] * e[1] + e[1] * e[2] + e[2] * e[0])
    }
}

/// A single indexed triangle in world space.
///
/// `primitive` is the caller's stable id (e.g. an index into a mesh's triangle
/// list); the builder reorders primitives internally but always reports hits by
/// this id so downstream shading can look up material/attributes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Triangle {
    /// First vertex.
    pub v0: [f32; 3],
    /// Second vertex.
    pub v1: [f32; 3],
    /// Third vertex.
    pub v2: [f32; 3],
    /// Stable caller-facing primitive id.
    pub primitive: u32,
}

impl Triangle {
    /// Builds a triangle carrying primitive id `primitive`.
    #[must_use]
    pub const fn new(v0: [f32; 3], v1: [f32; 3], v2: [f32; 3], primitive: u32) -> Self {
        Self {
            v0,
            v1,
            v2,
            primitive,
        }
    }

    /// Tight bounding box over the three vertices.
    #[must_use]
    pub fn bounds(&self) -> Aabb {
        Aabb::point(self.v0).enclose(self.v1).enclose(self.v2)
    }

    /// Centroid of the triangle's *bounds* (matches the build partition key).
    #[must_use]
    pub fn centroid(&self) -> [f32; 3] {
        self.bounds().centroid()
    }
}

/// One node of the flattened depth-first `BVH`.
///
/// Interior vs leaf is disambiguated by [`LinearBvhNode::primitive_count`]:
/// a value of `0` marks an interior node whose *first* child is the next node in
/// the array and whose *second* child sits at [`LinearBvhNode::second_child`].
/// A non-zero count marks a leaf owning `primitive_count` entries of the
/// reordered primitive-index table starting at [`LinearBvhNode::first_primitive`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LinearBvhNode {
    /// Node bounds.
    pub bounds: Aabb,
    /// Leaf: first index into the primitive-index table. Interior: unused (`0`).
    pub first_primitive: u32,
    /// Interior: array index of the second child. Leaf: unused (`0`).
    pub second_child: u32,
    /// Leaf primitive count; `0` marks an interior node.
    pub primitive_count: u16,
    /// Split axis of an interior node (`0`/`1`/`2`); used for ordered traversal.
    pub axis: u8,
}

impl LinearBvhNode {
    /// True when this node is a leaf (owns primitives).
    #[must_use]
    pub const fn is_leaf(&self) -> bool {
        self.primitive_count > 0
    }
}

/// Tuning knobs for the binned-`SAH` builder.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BvhBuildConfig {
    /// Maximum primitives allowed in a leaf before a split is forced.
    pub max_leaf_primitives: usize,
    /// Number of `SAH` bins evaluated along the split axis.
    pub sah_bins: usize,
    /// Estimated cost of visiting one interior node relative to one ray-triangle
    /// test (which costs `1.0`). Standard `pbrt` uses `1/8`.
    pub traversal_cost: f32,
}

impl Default for BvhBuildConfig {
    fn default() -> Self {
        Self {
            max_leaf_primitives: 4,
            sah_bins: 12,
            traversal_cost: 0.125,
        }
    }
}

/// A built bounding-volume hierarchy over a triangle soup.
///
/// Empty input yields an empty hierarchy ([`Bvh::is_empty`]); traversal of an
/// empty hierarchy simply never reports a hit.
#[derive(Clone, Debug, PartialEq)]
pub struct Bvh {
    pub(crate) nodes: Vec<LinearBvhNode>,
    pub(crate) primitives: Vec<Triangle>,
}

/// Per-primitive scratch used only during the build.
#[derive(Clone, Copy)]
struct PrimRef {
    bounds: Aabb,
    centroid: [f32; 3],
    index: usize,
}

impl Bvh {
    /// Builds a `BVH` over `triangles` with [`BvhBuildConfig::default`].
    #[must_use]
    pub fn build(triangles: &[Triangle]) -> Self {
        Self::build_with(triangles, BvhBuildConfig::default())
    }

    /// Builds a `BVH` over `triangles` with an explicit configuration.
    ///
    /// Degenerate triangles (`NaN`/inf vertices) still get valid — if wide —
    /// bounds; the builder never panics on pathological input.
    #[must_use]
    pub fn build_with(triangles: &[Triangle], config: BvhBuildConfig) -> Self {
        if triangles.is_empty() {
            return Self {
                nodes: Vec::new(),
                primitives: Vec::new(),
            };
        }
        let bins = config.sah_bins.max(1);
        let max_leaf = config.max_leaf_primitives.max(1);

        let mut refs: Vec<PrimRef> = triangles
            .iter()
            .enumerate()
            .map(|(index, tri)| {
                let bounds = tri.bounds();
                PrimRef {
                    bounds,
                    centroid: bounds.centroid(),
                    index,
                }
            })
            .collect();

        // Depth-first build into an explicit node arena so the emitted order is
        // exactly the traversal order (first child immediately follows parent).
        let mut nodes: Vec<LinearBvhNode> = Vec::with_capacity(2 * triangles.len());
        let mut ordered: Vec<Triangle> = Vec::with_capacity(triangles.len());
        Self::build_recursive(
            &mut refs,
            triangles,
            &mut nodes,
            &mut ordered,
            max_leaf,
            bins,
            f64::from(config.traversal_cost),
        );

        Self {
            nodes,
            primitives: ordered,
        }
    }

    /// Recursively partitions `refs[..]`, appending nodes depth-first and the
    /// reordered primitives into `ordered`. Returns the arena index of the
    /// subtree root it emitted.
    fn build_recursive(
        refs: &mut [PrimRef],
        triangles: &[Triangle],
        nodes: &mut Vec<LinearBvhNode>,
        ordered: &mut Vec<Triangle>,
        max_leaf: usize,
        bins: usize,
        traversal_cost: f64,
    ) -> u32 {
        let node_bounds = refs
            .iter()
            .fold(Aabb::empty(), |acc, r| acc.union(&r.bounds));

        let make_leaf = |nodes: &mut Vec<LinearBvhNode>, ordered: &mut Vec<Triangle>| -> u32 {
            let first = ordered.len() as u32;
            for r in refs.iter() {
                ordered.push(triangles[r.index]);
            }
            let node_index = nodes.len() as u32;
            nodes.push(LinearBvhNode {
                bounds: node_bounds,
                first_primitive: first,
                second_child: 0,
                primitive_count: refs.len() as u16,
                axis: 0,
            });
            node_index
        };

        if refs.len() <= max_leaf {
            return make_leaf(nodes, ordered);
        }

        // Partition on the axis with the widest *centroid* spread. A flat
        // centroid box means all primitives share a center: fall back to a leaf.
        let centroid_bounds = refs
            .iter()
            .fold(Aabb::empty(), |acc, r| acc.enclose(r.centroid));
        let axis = centroid_bounds.max_extent_axis();
        let axis_min = centroid_bounds.min[axis];
        let axis_max = centroid_bounds.max[axis];
        // A degenerate (flat) centroid box means all centers coincide on this
        // axis; no partition helps, so emit a leaf.
        if axis_max <= axis_min {
            return make_leaf(nodes, ordered);
        }

        // Bin primitives by centroid position along `axis`, then evaluate the
        // SAH cost of splitting after each bin boundary.
        let scale = bins as f32 / (axis_max - axis_min);
        let mut bin_counts = vec![0usize; bins];
        let mut bin_bounds = vec![Aabb::empty(); bins];
        for r in refs.iter() {
            let mut b = ((r.centroid[axis] - axis_min) * scale) as isize;
            if b < 0 {
                b = 0;
            }
            if b as usize >= bins {
                b = bins as isize - 1;
            }
            let b = b as usize;
            bin_counts[b] += 1;
            bin_bounds[b] = bin_bounds[b].union(&r.bounds);
        }

        // Forward/backward sweeps give, for each candidate split after bin `i`,
        // the count and bounds of each side in O(bins).
        let splits = bins - 1;
        let mut left_area = vec![0f64; splits];
        let mut left_count = vec![0usize; splits];
        let mut acc_bounds = Aabb::empty();
        let mut acc_count = 0usize;
        for i in 0..splits {
            acc_bounds = acc_bounds.union(&bin_bounds[i]);
            acc_count += bin_counts[i];
            left_area[i] = f64::from(acc_bounds.surface_area());
            left_count[i] = acc_count;
        }
        let mut right_area = vec![0f64; splits];
        let mut right_count = vec![0usize; splits];
        acc_bounds = Aabb::empty();
        acc_count = 0;
        for i in (0..splits).rev() {
            acc_bounds = acc_bounds.union(&bin_bounds[i + 1]);
            acc_count += bin_counts[i + 1];
            right_area[i] = f64::from(acc_bounds.surface_area());
            right_count[i] = acc_count;
        }

        let parent_area = f64::from(node_bounds.surface_area());
        let inv_parent_area = if parent_area > 0.0 {
            1.0 / parent_area
        } else {
            0.0
        };
        let leaf_cost = refs.len() as f64;
        let mut best_cost = f64::INFINITY;
        let mut best_split = usize::MAX;
        for i in 0..splits {
            if left_count[i] == 0 || right_count[i] == 0 {
                continue;
            }
            let cost = traversal_cost
                + (left_area[i] * left_count[i] as f64 + right_area[i] * right_count[i] as f64)
                    * inv_parent_area;
            if cost < best_cost {
                best_cost = cost;
                best_split = i;
            }
        }

        // Make a leaf when the SAH cannot beat it, but only if the leaf fits the
        // u16 primitive-count field. An oversized cluster is always split (below,
        // via the binned partition or the median fallback) so it never overflows.
        let leaf_is_legal = refs.len() <= u16::MAX as usize;
        if leaf_is_legal && (best_split == usize::MAX || best_cost >= leaf_cost) {
            return make_leaf(nodes, ordered);
        }

        // Partition `refs` in place: everything whose centroid bin is `<= split`
        // goes left. `partition_point`-style stable-ish partition via swapping.
        let split_bin = best_split;
        let mut mid = partition_refs(refs, axis, axis_min, scale, bins, split_bin);
        // Degenerate partition (all on one side) — fall back to a median split so
        // we still make progress and never recurse forever.
        if mid == 0 || mid == refs.len() {
            refs.sort_by(|a, b| {
                a.centroid[axis]
                    .partial_cmp(&b.centroid[axis])
                    .unwrap_or(core::cmp::Ordering::Equal)
            });
            mid = refs.len() / 2;
        }

        // Reserve this interior node's slot, emit first child (immediately
        // after), then the second child, and patch the offset.
        let node_index = nodes.len();
        nodes.push(LinearBvhNode {
            bounds: node_bounds,
            first_primitive: 0,
            second_child: 0,
            primitive_count: 0,
            axis: axis as u8,
        });
        let (left, right) = refs.split_at_mut(mid);
        let _first_child = Self::build_recursive(
            left, triangles, nodes, ordered, max_leaf, bins, traversal_cost,
        );
        let second_child = Self::build_recursive(
            right, triangles, nodes, ordered, max_leaf, bins, traversal_cost,
        );
        nodes[node_index].second_child = second_child;
        node_index as u32
    }

    /// Number of nodes in the flattened hierarchy.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of primitives referenced by the hierarchy.
    #[must_use]
    pub fn primitive_count(&self) -> usize {
        self.primitives.len()
    }

    /// True when the hierarchy holds no primitives.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Root bounds, or [`Aabb::empty`] when the hierarchy is empty.
    #[must_use]
    pub fn bounds(&self) -> Aabb {
        self.nodes.first().map_or(Aabb::empty(), |n| n.bounds)
    }

    /// Read-only view of the flattened nodes (depth-first order).
    #[must_use]
    pub fn nodes(&self) -> &[LinearBvhNode] {
        &self.nodes
    }

    /// Read-only view of the reordered primitive table.
    #[must_use]
    pub fn primitives(&self) -> &[Triangle] {
        &self.primitives
    }
}

/// Partitions `refs` so all primitives whose centroid falls in bin `<= split_bin`
/// come first. Returns the number of primitives placed on the left.
fn partition_refs(
    refs: &mut [PrimRef],
    axis: Axis,
    axis_min: f32,
    scale: f32,
    bins: usize,
    split_bin: usize,
) -> usize {
    let bin_of = |r: &PrimRef| -> usize {
        let mut b = ((r.centroid[axis] - axis_min) * scale) as isize;
        if b < 0 {
            b = 0;
        }
        if b as usize >= bins {
            b = bins as isize - 1;
        }
        b as usize
    };
    let mut i = 0usize;
    for j in 0..refs.len() {
        if bin_of(&refs[j]) <= split_bin {
            refs.swap(i, j);
            i += 1;
        }
    }
    i
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_box_is_identity_for_union() {
        let e = Aabb::empty();
        assert!(e.is_empty());
        assert_eq!(e.surface_area(), 0.0);
        let b = Aabb::new([-1.0, -2.0, -3.0], [4.0, 5.0, 6.0]);
        assert_eq!(e.union(&b), b);
        assert_eq!(b.union(&e), b);
    }

    #[test]
    fn enclose_grows_to_contain_point() {
        let b = Aabb::point([0.0, 0.0, 0.0]).enclose([1.0, -2.0, 3.0]);
        assert_eq!(b.min, [0.0, -2.0, 0.0]);
        assert_eq!(b.max, [1.0, 0.0, 3.0]);
    }

    #[test]
    fn surface_area_and_axis_are_correct() {
        let b = Aabb::new([0.0, 0.0, 0.0], [1.0, 2.0, 3.0]);
        // 2*(1*2 + 2*3 + 3*1) = 2*11 = 22
        assert_eq!(b.surface_area(), 22.0);
        assert_eq!(b.max_extent_axis(), 2);
        assert_eq!(b.centroid(), [0.5, 1.0, 1.5]);
    }

    fn tri(a: [f32; 3], b: [f32; 3], c: [f32; 3], id: u32) -> Triangle {
        Triangle::new(a, b, c, id)
    }

    #[test]
    fn empty_input_builds_empty_bvh() {
        let bvh = Bvh::build(&[]);
        assert!(bvh.is_empty());
        assert_eq!(bvh.node_count(), 0);
        assert_eq!(bvh.primitive_count(), 0);
        assert_eq!(bvh.bounds(), Aabb::empty());
    }

    #[test]
    fn single_triangle_is_one_leaf() {
        let t = tri([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 7);
        let bvh = Bvh::build(&[t]);
        assert_eq!(bvh.node_count(), 1);
        assert!(bvh.nodes()[0].is_leaf());
        assert_eq!(bvh.nodes()[0].primitive_count, 1);
        assert_eq!(bvh.bounds(), t.bounds());
        assert_eq!(bvh.primitives()[0].primitive, 7);
    }

    /// A simple deterministic xorshift so tests need no external rng crate.
    struct Rng(u64);
    impl Rng {
        fn next_f32(&mut self) -> f32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            // Map to [-5, 5].
            ((self.0 >> 40) as f32 / (1u32 << 24) as f32) * 10.0 - 5.0
        }
    }

    fn random_triangles(n: u32, seed: u64) -> Vec<Triangle> {
        let mut rng = Rng(seed);
        (0..n)
            .map(|id| {
                let base = [rng.next_f32(), rng.next_f32(), rng.next_f32()];
                let jitter = |r: &mut Rng| {
                    [
                        base[0] + r.next_f32() * 0.2,
                        base[1] + r.next_f32() * 0.2,
                        base[2] + r.next_f32() * 0.2,
                    ]
                };
                tri(base, jitter(&mut rng), jitter(&mut rng), id)
            })
            .collect()
    }

    #[test]
    fn build_preserves_every_primitive_exactly_once() {
        let tris = random_triangles(400, 0x1234_5678_9abc_def0);
        let bvh = Bvh::build(&tris);
        assert_eq!(bvh.primitive_count(), tris.len());
        let mut ids: Vec<u32> = bvh.primitives().iter().map(|t| t.primitive).collect();
        ids.sort_unstable();
        let expected: Vec<u32> = (0..tris.len() as u32).collect();
        assert_eq!(ids, expected);
    }

    #[test]
    fn interior_nodes_have_valid_children_and_enclosing_bounds() {
        let tris = random_triangles(300, 0x0f0f_0f0f_dead_beef);
        let bvh = Bvh::build(&tris);
        let nodes = bvh.nodes();
        for (i, node) in nodes.iter().enumerate() {
            if node.is_leaf() {
                let start = node.first_primitive as usize;
                let end = start + node.primitive_count as usize;
                assert!(end <= bvh.primitive_count());
                // Leaf bounds must enclose its primitives.
                for tri in &bvh.primitives()[start..end] {
                    let tb = tri.bounds();
                    for a in 0..3 {
                        assert!(node.bounds.min[a] <= tb.min[a] + 1e-4);
                        assert!(node.bounds.max[a] >= tb.max[a] - 1e-4);
                    }
                }
            } else {
                let first = i + 1;
                let second = node.second_child as usize;
                assert!(first < nodes.len());
                assert!(second < nodes.len());
                assert!(node.axis < 3);
                // Parent bounds enclose both children.
                for child in [first, second] {
                    let cb = nodes[child].bounds;
                    for a in 0..3 {
                        assert!(node.bounds.min[a] <= cb.min[a] + 1e-4);
                        assert!(node.bounds.max[a] >= cb.max[a] - 1e-4);
                    }
                }
            }
        }
    }

    #[test]
    fn leaves_respect_configured_max_when_splittable() {
        let tris = random_triangles(500, 0xabcd_ef01_2345_6789);
        let cfg = BvhBuildConfig {
            max_leaf_primitives: 4,
            ..Default::default()
        };
        let bvh = Bvh::build_with(&tris, cfg);
        // Every leaf either fits the budget or is an unsplittable coincident cluster.
        let over: usize = bvh
            .nodes()
            .iter()
            .filter(|n| n.is_leaf() && n.primitive_count as usize > 4)
            .count();
        // Random jittered triangles are splittable, so no oversized leaves.
        assert_eq!(over, 0);
    }
}
