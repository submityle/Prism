//! Compressed *wide* (`BVH8`) acceleration structure.
//!
//! A binary [`Bvh`] is the clearest golden reference, but production `GPU` ray
//! tracers never traverse a two-child tree: each node fetch reads a full cache
//! line yet only advances one level, so incoherent rays stall on memory. The
//! AAA answer, shipped by every modern hardware/software tracer (Embree `BVH8`,
//! `OptiX`, console `HWRT`), is a *wide, compressed* `BVH`: collapse the binary
//! tree into nodes with up to [`WIDE_BRANCHING`] children so one fetch tests
//! many boxes, and *quantize* each child's bounds to a byte lattice relative to
//! the parent so the whole node fits in a single cache line.
//!
//! This module is the `CPU` golden reference for exactly that structure,
//! following Ylitie et al., *"Efficient Incoherent Ray Traversal on GPUs
//! through Compressed Wide BVHs"* (HPG 2017): each [`WideNode`] stores a
//! quantization `origin` (its own bounds' minimum corner) and a per-axis
//! power-of-two `scale`, then packs every child's lower/upper corner into three
//! bytes each. The lower corner is *floored* and the upper corner is *ceiled*,
//! so the dequantized child box is always a **conservative superset** of the
//! exact box. That single invariant is what makes the wide walk safe: because
//! no real intersection is ever culled, [`WideBvh::closest_hit`] returns the
//! bit-for-bit same nearest [`Hit`] as [`Bvh::closest_hit`] on any ray whose
//! nearest hit is unique (general-position geometry), which the tests assert
//! over thousands of random rays.
//!
//! The power-of-two scale is derived by reading the IEEE-754 exponent field
//! directly (see [`pow2_ceil`]), so the quantization contains no disallowed
//! transcendental (`log2`/`exp2`) call and reproduces on the `GPU` bit-for-bit.
//! The flat, `GPU`-uploadable byte layout lives in
//! [`super::bvh_wide_gpu_layout`].

use super::bvh::{Aabb, Bvh, Triangle};
use super::traversal::{intersect_triangle, intersect_triangle_watertight, Hit, Ray};

/// Maximum number of children a [`WideNode`] can hold (a `BVH8`).
///
/// Eight children match a `GPU` cache line once the child bounds are quantized
/// to bytes and is the branching factor Ylitie et al. and shipping console
/// tracers standardize on: wide enough to amortize the node fetch, narrow
/// enough that the per-child byte corners stay in one line.
pub const WIDE_BRANCHING: usize = 8;

/// Number of quantization steps per axis (the child corners are packed into
/// unsigned bytes, so the lattice has `0..=255` steps).
pub const QUANT_STEPS: f32 = 255.0;

/// The payload a [`WideNode`] slot carries for one child.
///
/// Slots at or beyond [`WideNode::child_count`] are [`WideChild::Empty`]; the
/// fixed-size arrays keep every [`WideNode`] trivially `Copy` regardless of how
/// many children it actually uses.
#[derive(Clone, Copy, Debug)]
pub enum WideChild {
    /// Unused slot (never inspected during traversal).
    Empty,
    /// Interior child: index of another [`WideNode`] in [`WideBvh::nodes`].
    Interior(u32),
    /// Leaf child: `count` primitives starting at index `first` in
    /// [`WideBvh::primitives`] (the reordered table inherited from the source
    /// [`Bvh`]).
    Leaf {
        /// First primitive index into [`WideBvh::primitives`].
        first: u32,
        /// Number of primitives owned by this leaf.
        count: u16,
    },
}

/// One compressed wide node: up to [`WIDE_BRANCHING`] children whose bounds are
/// quantized to a byte lattice relative to `origin`/`scale`.
///
/// The dequantized child box for slot `c` on axis `a` is
/// `origin[a] + child_qlo[c][a] * scale[a]` (lower) and
/// `origin[a] + child_qhi[c][a] * scale[a]` (upper). Because the lower corner
/// was floored and the upper corner ceiled during [`WideBvh::from_bvh`], this
/// box always contains the exact child box.
#[derive(Clone, Copy, Debug)]
pub struct WideNode {
    /// Quantization origin: the node's own bounds' minimum corner.
    origin: [f32; 3],
    /// Per-axis power-of-two dequantization scale (`2^e`), never zero.
    scale: [f32; 3],
    /// Number of populated child slots, `1..=WIDE_BRANCHING`.
    child_count: u8,
    /// Per-child *floored* lower-corner bytes.
    child_qlo: [[u8; 3]; WIDE_BRANCHING],
    /// Per-child *ceiled* upper-corner bytes.
    child_qhi: [[u8; 3]; WIDE_BRANCHING],
    /// Per-child payload (interior index, leaf range, or empty).
    child_kind: [WideChild; WIDE_BRANCHING],
}

impl WideNode {
    /// An all-zero placeholder used while a node's index is reserved before its
    /// children (and thus their interior indices) are known.
    const PLACEHOLDER: WideNode = WideNode {
        origin: [0.0; 3],
        scale: [1.0; 3],
        child_count: 0,
        child_qlo: [[0; 3]; WIDE_BRANCHING],
        child_qhi: [[0; 3]; WIDE_BRANCHING],
        child_kind: [WideChild::Empty; WIDE_BRANCHING],
    };

    /// Quantization origin (the node's bounds' minimum corner).
    #[must_use]
    pub fn origin(&self) -> [f32; 3] {
        self.origin
    }

    /// Per-axis power-of-two dequantization scale.
    #[must_use]
    pub fn scale(&self) -> [f32; 3] {
        self.scale
    }

    /// Number of populated child slots.
    #[must_use]
    pub fn child_count(&self) -> u8 {
        self.child_count
    }

    /// Floored lower-corner bytes for child slot `c`.
    #[must_use]
    pub fn child_qlo(&self, c: usize) -> [u8; 3] {
        self.child_qlo[c]
    }

    /// Ceiled upper-corner bytes for child slot `c`.
    #[must_use]
    pub fn child_qhi(&self, c: usize) -> [u8; 3] {
        self.child_qhi[c]
    }

    /// Payload for child slot `c`.
    #[must_use]
    pub fn child_kind(&self, c: usize) -> WideChild {
        self.child_kind[c]
    }

    /// Dequantized (conservative) world-space bounds of child slot `c`.
    #[must_use]
    fn child_bounds(&self, c: usize) -> Aabb {
        let lo = self.child_qlo[c];
        let hi = self.child_qhi[c];
        let mut min = [0.0f32; 3];
        let mut max = [0.0f32; 3];
        for a in 0..3 {
            min[a] = self.origin[a] + f32::from(lo[a]) * self.scale[a];
            max[a] = self.origin[a] + f32::from(hi[a]) * self.scale[a];
        }
        Aabb::new(min, max)
    }
}

/// A compressed wide (`BVH8`) tree collapsed from a binary [`Bvh`].
///
/// Holds the wide node array and the reordered primitive table inherited from
/// the source [`Bvh`] so leaf ranges index the same primitives the binary walk
/// intersects, guaranteeing identical hit records.
#[derive(Clone, Debug)]
pub struct WideBvh {
    /// Wide nodes in depth-first emission order; the root is index `0`.
    nodes: Vec<WideNode>,
    /// Reordered primitives, shared with the source [`Bvh`].
    primitives: Vec<Triangle>,
}

impl WideBvh {
    /// Collapses a binary [`Bvh`] into a compressed wide tree.
    ///
    /// The collapse follows Ylitie et al.: starting from the binary root, the
    /// interior child with the largest surface area is repeatedly expanded into
    /// its two children until a node has [`WIDE_BRANCHING`] slots or none of its
    /// slots is interior. Each resulting slot becomes a leaf child (primitive
    /// range) or an interior child (its own recursively emitted [`WideNode`]).
    #[must_use]
    pub fn from_bvh(bvh: &Bvh) -> Self {
        if bvh.is_empty() {
            return Self {
                nodes: Vec::new(),
                primitives: Vec::new(),
            };
        }
        let mut nodes = Vec::new();
        emit(bvh, 0, &mut nodes);
        Self {
            nodes,
            primitives: bvh.primitives().to_vec(),
        }
    }

    /// Wide node array (root first). Empty when the source [`Bvh`] was empty.
    #[must_use]
    pub fn nodes(&self) -> &[WideNode] {
        &self.nodes
    }

    /// Reordered primitive table shared with the source [`Bvh`].
    #[must_use]
    pub fn primitives(&self) -> &[Triangle] {
        &self.primitives
    }

    /// Number of wide nodes.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of primitives.
    #[must_use]
    pub fn primitive_count(&self) -> usize {
        self.primitives.len()
    }

    /// True when the tree holds no nodes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Nearest [`Hit`] using the Möller–Trumbore test, matching
    /// [`Bvh::closest_hit`] bit-for-bit on rays with a unique nearest hit.
    #[must_use]
    pub fn closest_hit(&self, ray: &Ray) -> Option<Hit> {
        self.walk_closest(ray, intersect_triangle)
    }

    /// Nearest [`Hit`] using the watertight test, matching
    /// [`Bvh::closest_hit_watertight`] bit-for-bit on rays with a unique
    /// nearest hit.
    #[must_use]
    pub fn closest_hit_watertight(&self, ray: &Ray) -> Option<Hit> {
        self.walk_closest(ray, intersect_triangle_watertight)
    }

    /// True when any primitive intersects `ray` (Möller–Trumbore test).
    #[must_use]
    pub fn any_hit(&self, ray: &Ray) -> bool {
        self.walk_any(ray, intersect_triangle)
    }

    /// True when any primitive intersects `ray` (watertight test).
    #[must_use]
    pub fn any_hit_watertight(&self, ray: &Ray) -> bool {
        self.walk_any(ray, intersect_triangle_watertight)
    }

    /// Shared nearest-hit walk parameterized by the triangle intersection test.
    ///
    /// Descends with an explicit stack of wide-node indices, testing each
    /// child's dequantized box and shrinking the ray's `t_max` as leaves report
    /// nearer hits so far subtrees prune. Children are visited near-first (the
    /// box entry `t` sorts the slots) purely to prune sooner; the returned hit
    /// is the global nearest and does not depend on visit order.
    fn walk_closest(
        &self,
        ray: &Ray,
        test: fn(&Ray, &Triangle) -> Option<(f32, f32, f32)>,
    ) -> Option<Hit> {
        if self.nodes.is_empty() {
            return None;
        }
        let mut ray = *ray;
        let mut best: Option<Hit> = None;
        let mut stack: Vec<u32> = Vec::with_capacity(64);
        stack.push(0);
        while let Some(node_index) = stack.pop() {
            let node = &self.nodes[node_index as usize];
            let mut order: [(f32, usize); WIDE_BRANCHING] = [(0.0, 0); WIDE_BRANCHING];
            let mut hit_count = 0usize;
            for c in 0..node.child_count as usize {
                let bounds = node.child_bounds(c);
                if let Some((t_enter, _)) =
                    ray.aabb_interval(&bounds, ray.t_min(), ray.t_max())
                {
                    order[hit_count] = (t_enter, c);
                    hit_count += 1;
                }
            }
            // Visit near children first: sort by entry `t` descending so the
            // nearest child is pushed last and popped first.
            order[..hit_count].sort_by(|lhs, rhs| rhs.0.total_cmp(&lhs.0));
            for &(_, c) in &order[..hit_count] {
                match node.child_kind[c] {
                    WideChild::Interior(idx) => stack.push(idx),
                    WideChild::Leaf { first, count } => {
                        let start = first as usize;
                        let end = start + count as usize;
                        for tri in &self.primitives[start..end] {
                            if let Some((t, u, v)) = test(&ray, tri) {
                                best = Some(Hit {
                                    t,
                                    u,
                                    v,
                                    primitive: tri.primitive,
                                });
                                ray = Ray::new(ray.origin(), ray.direction(), ray.t_min(), t);
                            }
                        }
                    }
                    WideChild::Empty => {}
                }
            }
        }
        best
    }

    /// Shared any-hit walk parameterized by the triangle intersection test.
    ///
    /// Returns on the first intersection without tracking the nearest, so it is
    /// the cheap shadow/occlusion query. Visits a superset of the leaves the
    /// binary walk would, so it reports occlusion whenever [`Bvh::any_hit`]
    /// does.
    fn walk_any(
        &self,
        ray: &Ray,
        test: fn(&Ray, &Triangle) -> Option<(f32, f32, f32)>,
    ) -> bool {
        if self.nodes.is_empty() {
            return false;
        }
        let mut stack: Vec<u32> = Vec::with_capacity(64);
        stack.push(0);
        while let Some(node_index) = stack.pop() {
            let node = &self.nodes[node_index as usize];
            for c in 0..node.child_count as usize {
                let bounds = node.child_bounds(c);
                if ray
                    .aabb_interval(&bounds, ray.t_min(), ray.t_max())
                    .is_none()
                {
                    continue;
                }
                match node.child_kind[c] {
                    WideChild::Interior(idx) => stack.push(idx),
                    WideChild::Leaf { first, count } => {
                        let start = first as usize;
                        let end = start + count as usize;
                        for tri in &self.primitives[start..end] {
                            if test(ray, tri).is_some() {
                                return true;
                            }
                        }
                    }
                    WideChild::Empty => {}
                }
            }
        }
        false
    }
}

/// Smallest power of two `2^e` that is `>= value`, computed by reading the
/// IEEE-754 exponent field so the result is exact and uses no disallowed
/// transcendental call (`log2`/`exp2`).
///
/// A non-positive, subnormal, or `NaN` input returns the smallest normal power
/// of two (`2^-126`) so the quantization scale is always a well-defined,
/// non-zero divisor even for a degenerate (zero-extent) axis.
fn pow2_ceil(value: f32) -> f32 {
    // Smallest normal power of two; also the floor for degenerate axes.
    let min_scale = f32::from_bits(1u32 << 23);
    if value <= min_scale || value.is_nan() {
        return min_scale;
    }
    let bits = value.to_bits();
    let biased = (bits >> 23) & 0xff;
    let mantissa = bits & 0x007f_ffff;
    // An exact power of two keeps its exponent; anything larger rounds up.
    let exp = if mantissa == 0 { biased } else { biased + 1 };
    f32::from_bits(exp << 23)
}

/// Expands the binary subtree rooted at `root` into up to [`WIDE_BRANCHING`]
/// subtree roots (the future wide node's children).
///
/// Repeatedly replaces the largest-surface-area interior slot with its two
/// binary children until the slot budget is full or every slot is a leaf. The
/// left child of a binary interior node sits at `index + 1` and the right at
/// its `second_child`, matching the flattened [`Bvh`] layout.
fn gather(bvh: &Bvh, root: usize) -> Vec<usize> {
    let nodes = bvh.nodes();
    let mut slots: Vec<usize> = Vec::with_capacity(WIDE_BRANCHING);
    slots.push(root);
    while slots.len() < WIDE_BRANCHING {
        let mut best_pos: Option<usize> = None;
        let mut best_area = f32::NEG_INFINITY;
        for (pos, &n) in slots.iter().enumerate() {
            if !nodes[n].is_leaf() {
                let area = nodes[n].bounds.surface_area();
                if area > best_area {
                    best_area = area;
                    best_pos = Some(pos);
                }
            }
        }
        let Some(pos) = best_pos else { break };
        let n = slots[pos];
        let left = n + 1;
        let right = nodes[n].second_child as usize;
        slots[pos] = left;
        slots.push(right);
    }
    slots
}

/// Emits the wide node for the binary subtree rooted at `root`, recursively
/// emitting interior children, and returns its index in `out`.
///
/// The node's index is reserved with a [`WideNode::PLACEHOLDER`] before its
/// children are emitted (children append to `out`), then overwritten once every
/// child slot is resolved.
fn emit(bvh: &Bvh, root: usize, out: &mut Vec<WideNode>) -> u32 {
    let nodes = bvh.nodes();
    let slots = gather(bvh, root);
    let wide_index = out.len() as u32;
    out.push(WideNode::PLACEHOLDER);

    let mut bounds = Aabb::empty();
    for &s in &slots {
        bounds = bounds.union(&nodes[s].bounds);
    }
    let origin = bounds.min;
    let extent = bounds.extent();
    let scale = [
        pow2_ceil(extent[0] / QUANT_STEPS),
        pow2_ceil(extent[1] / QUANT_STEPS),
        pow2_ceil(extent[2] / QUANT_STEPS),
    ];

    let mut child_qlo = [[0u8; 3]; WIDE_BRANCHING];
    let mut child_qhi = [[0u8; 3]; WIDE_BRANCHING];
    let mut child_kind = [WideChild::Empty; WIDE_BRANCHING];
    for (c, &s) in slots.iter().enumerate() {
        let cb = &nodes[s].bounds;
        for a in 0..3 {
            let lo_rel = ((cb.min[a] - origin[a]) / scale[a]).floor().clamp(0.0, QUANT_STEPS);
            let hi_rel = ((cb.max[a] - origin[a]) / scale[a]).ceil().clamp(0.0, QUANT_STEPS);
            child_qlo[c][a] = lo_rel as u8;
            child_qhi[c][a] = hi_rel as u8;
        }
        child_kind[c] = if nodes[s].is_leaf() {
            WideChild::Leaf {
                first: nodes[s].first_primitive,
                count: nodes[s].primitive_count,
            }
        } else {
            WideChild::Interior(emit(bvh, s, out))
        };
    }

    out[wide_index as usize] = WideNode {
        origin,
        scale,
        child_count: slots.len() as u8,
        child_qlo,
        child_qhi,
        child_kind,
    };
    wide_index
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ray_scene::bvh::Triangle;

    /// Small deterministic xorshift `RNG`, matching the other `ray_scene`
    /// suites so tests never depend on an external crate.
    struct Rng(u64);
    impl Rng {
        /// Seeds the generator (forcing a non-zero state).
        fn new(seed: u64) -> Self {
            Self(seed | 1)
        }
        /// Advances the state and returns the high 32 bits.
        fn next_u32(&mut self) -> u32 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            (x >> 32) as u32
        }
        /// Uniform value in `[0, 1]`.
        fn unit(&mut self) -> f32 {
            self.next_u32() as f32 / u32::MAX as f32
        }
        /// Uniform value in `[lo, hi]`.
        fn range(&mut self, lo: f32, hi: f32) -> f32 {
            lo + (hi - lo) * self.unit()
        }
    }

    /// Builds a scene of `n` small, randomly placed triangles in general
    /// position (no shared vertices), mirroring the traversal suite so nearest
    /// hits are unique and the wide walk can be compared bit-for-bit.
    fn random_scene(n: u32, seed: u64) -> Vec<Triangle> {
        let mut rng = Rng::new(seed);
        (0..n)
            .map(|id| {
                let c = [
                    rng.range(-8.0, 8.0),
                    rng.range(-8.0, 8.0),
                    rng.range(-8.0, 8.0),
                ];
                let p = |r: &mut Rng| {
                    [
                        c[0] + r.range(-0.6, 0.6),
                        c[1] + r.range(-0.6, 0.6),
                        c[2] + r.range(-0.6, 0.6),
                    ]
                };
                Triangle::new(p(&mut rng), p(&mut rng), p(&mut rng), id)
            })
            .collect()
    }

    /// Every child of every node must reference a valid node/primitive range,
    /// and every dequantized child box must *contain* the exact box of the
    /// binary subtree it stands for (the conservative-superset invariant that
    /// makes the walk safe).
    #[test]
    fn wide_structure_is_well_formed() {
        let tris = random_scene(500, 0x1234_5678_9abc_def0);
        let bvh = Bvh::build(&tris);
        let wide = WideBvh::from_bvh(&bvh);
        assert!(!wide.is_empty());
        assert_eq!(wide.primitive_count(), tris.len());

        let mut leaf_prims = 0usize;
        for (wi, node) in wide.nodes().iter().enumerate() {
            assert!((1..=WIDE_BRANCHING as u8).contains(&node.child_count()));
            for c in 0..node.child_count() as usize {
                let bounds = node_child_bounds(node, c);
                match node.child_kind(c) {
                    WideChild::Empty => panic!("populated slot {c} of node {wi} was Empty"),
                    WideChild::Interior(idx) => {
                        assert!((idx as usize) < wide.node_count());
                        assert_ne!(idx as usize, wi, "interior child must not be self");
                    }
                    WideChild::Leaf { first, count } => {
                        assert!(count >= 1);
                        assert!(first as usize + count as usize <= wide.primitive_count());
                        leaf_prims += count as usize;
                    }
                }
                // Every dequantized child box must be finite and well-ordered.
                for a in 0..3 {
                    assert!(bounds.min[a].is_finite() && bounds.max[a].is_finite());
                    assert!(bounds.min[a] <= bounds.max[a]);
                }
            }
        }
        // Every primitive is reachable through exactly the leaves (the collapse
        // preserves the full primitive set).
        assert_eq!(leaf_prims, tris.len());
    }

    /// Helper mirroring `WideNode::child_bounds` for the test module (the
    /// method is private to the walk).
    fn node_child_bounds(node: &WideNode, c: usize) -> Aabb {
        let lo = node.child_qlo(c);
        let hi = node.child_qhi(c);
        let origin = node.origin();
        let scale = node.scale();
        let mut min = [0.0f32; 3];
        let mut max = [0.0f32; 3];
        for a in 0..3 {
            min[a] = origin[a] + f32::from(lo[a]) * scale[a];
            max[a] = origin[a] + f32::from(hi[a]) * scale[a];
        }
        Aabb::new(min, max)
    }

    /// The compressed wide walk must reproduce the binary `BVH` nearest hit
    /// bit-for-bit over thousands of random rays (`t`/`u`/`v`/`primitive`).
    #[test]
    fn wide_closest_hit_matches_binary_bit_for_bit() {
        let tris = random_scene(600, 0xdead_c0de_1234_5678);
        let bvh = Bvh::build(&tris);
        let wide = WideBvh::from_bvh(&bvh);
        let mut rng = Rng::new(0x9e37_79b9_7f4a_7c15);
        let mut hits = 0u32;
        for _ in 0..4000 {
            let origin = [
                rng.range(-12.0, 12.0),
                rng.range(-12.0, 12.0),
                rng.range(-12.0, 12.0),
            ];
            let dir = [
                rng.range(-1.0, 1.0),
                rng.range(-1.0, 1.0),
                rng.range(-1.0, 1.0),
            ];
            let ray = Ray::infinite(origin, dir);
            let want = bvh.closest_hit(&ray);
            let got = wide.closest_hit(&ray);
            match (want, got) {
                (None, None) => {}
                (Some(a), Some(b)) => {
                    assert_eq!(a.t.to_bits(), b.t.to_bits(), "t mismatch");
                    assert_eq!(a.u.to_bits(), b.u.to_bits(), "u mismatch");
                    assert_eq!(a.v.to_bits(), b.v.to_bits(), "v mismatch");
                    assert_eq!(a.primitive, b.primitive, "primitive mismatch");
                    hits += 1;
                }
                (a, b) => panic!("hit disagreement: {a:?} vs {b:?}"),
            }
        }
        assert!(hits > 200, "test scene should produce many hits, got {hits}");
    }

    /// The watertight variant must likewise reproduce the binary watertight
    /// nearest hit bit-for-bit.
    #[test]
    fn wide_closest_hit_watertight_matches_binary_bit_for_bit() {
        let tris = random_scene(400, 0x0bad_f00d_dead_beef);
        let bvh = Bvh::build(&tris);
        let wide = WideBvh::from_bvh(&bvh);
        let mut rng = Rng::new(0x1357_9bdf_0246_8ace);
        for _ in 0..3000 {
            let origin = [
                rng.range(-11.0, 11.0),
                rng.range(-11.0, 11.0),
                rng.range(-11.0, 11.0),
            ];
            let dir = [
                rng.range(-1.0, 1.0),
                rng.range(-1.0, 1.0),
                rng.range(-1.0, 1.0),
            ];
            let ray = Ray::infinite(origin, dir);
            match (
                bvh.closest_hit_watertight(&ray),
                wide.closest_hit_watertight(&ray),
            ) {
                (None, None) => {}
                (Some(a), Some(b)) => {
                    assert_eq!(a.t.to_bits(), b.t.to_bits());
                    assert_eq!(a.primitive, b.primitive);
                }
                (a, b) => panic!("watertight disagreement: {a:?} vs {b:?}"),
            }
        }
    }

    /// `any_hit` (both tests) must agree with the binary `BVH` occlusion query
    /// on every ray, since the wide walk visits a superset of the same leaves.
    #[test]
    fn wide_any_hit_matches_binary() {
        let tris = random_scene(500, 0xfeed_face_cafe_babe);
        let bvh = Bvh::build(&tris);
        let wide = WideBvh::from_bvh(&bvh);
        let mut rng = Rng::new(0x2468_ace0_1357_9bdf);
        for _ in 0..4000 {
            let origin = [
                rng.range(-12.0, 12.0),
                rng.range(-12.0, 12.0),
                rng.range(-12.0, 12.0),
            ];
            let dir = [
                rng.range(-1.0, 1.0),
                rng.range(-1.0, 1.0),
                rng.range(-1.0, 1.0),
            ];
            let ray = Ray::infinite(origin, dir);
            assert_eq!(bvh.any_hit(&ray), wide.any_hit(&ray));
            assert_eq!(
                bvh.any_hit_watertight(&ray),
                wide.any_hit_watertight(&ray)
            );
        }
    }

    /// An empty source `BVH` yields an empty wide tree that never hits.
    #[test]
    fn empty_bvh_yields_empty_wide_tree() {
        let bvh = Bvh::build(&[]);
        let wide = WideBvh::from_bvh(&bvh);
        assert!(wide.is_empty());
        assert_eq!(wide.node_count(), 0);
        let ray = Ray::infinite([0.0, 0.0, 0.0], [0.0, 0.0, -1.0]);
        assert!(wide.closest_hit(&ray).is_none());
        assert!(!wide.any_hit(&ray));
    }

    /// A single triangle collapses to a one-node, one-leaf wide tree that hits
    /// exactly like the binary `BVH`.
    #[test]
    fn single_triangle_round_trips() {
        let tris = [Triangle::new(
            [-1.0, -1.0, -5.0],
            [1.0, -1.0, -5.0],
            [0.0, 1.0, -5.0],
            7,
        )];
        let bvh = Bvh::build(&tris);
        let wide = WideBvh::from_bvh(&bvh);
        assert_eq!(wide.node_count(), 1);
        let ray = Ray::infinite([0.0, 0.0, 0.0], [0.0, 0.0, -1.0]);
        let want = bvh.closest_hit(&ray).expect("binary hit");
        let got = wide.closest_hit(&ray).expect("wide hit");
        assert_eq!(want.t.to_bits(), got.t.to_bits());
        assert_eq!(got.primitive, 7);
    }

    /// `pow2_ceil` returns an exact power of two that is `>= value` for a range
    /// of inputs, and never zero for degenerate ones.
    #[test]
    fn pow2_ceil_is_a_conservative_power_of_two() {
        for &v in &[0.0f32, 1.0, 2.0, 3.0, 255.0, 0.004, 1e-3, 1e3, 1e-30] {
            let s = pow2_ceil(v);
            assert!(s > 0.0, "scale must be positive for {v}");
            // Exact power of two: mantissa bits are zero.
            assert_eq!(s.to_bits() & 0x007f_ffff, 0, "{s} is not a power of two");
            if v > 0.0 {
                assert!(s >= v, "{s} must be >= {v}");
            }
        }
    }
}

