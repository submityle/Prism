//! Particle bounding-volume-hierarchy (`BVH`) acceleration-structure contract
//! (design §12, §13, §22).
//!
//! Collision and ray-tracing broadphase both need a spatial index over the
//! live-particle `AABB` set so a query touches `O(log n)` boxes instead of the
//! whole Structure-of-Arrays (`SoA`) pool. Production engines build this the
//! same way: quantize each particle centroid to a `Morton` code, sort, and
//! partition; then lay the node array out as a flat `std430` buffer a `GPU`
//! traversal kernel walks with an explicit stack. This module is the
//! `CPU`-verifiable reference for that structure — it owns the box algebra, the
//! `Morton` encoding, the node-buffer sizing arithmetic, the surface-area
//! heuristic (`SAH`) cost term, and a small median-split builder a future `GPU`
//! kernel must agree with.
//!
//! It is deliberately self-contained: it hand-rolls its own minimal [`Vec3`]
//! and [`Aabb`] rather than importing the sibling bounds module, so the only
//! dependency is the shared `std430` stride constants. Everything is pure
//! integer bit arithmetic and multiply-add: the only floating-point helpers are
//! `sqrt` (vector length), `floor` (grid quantization), and integer `div_ceil`
//! / bit-shift `log2` (tree sizing). No transcendental function is ever called
//! and `f32` equality is never tested with `==`.

use crate::particle::gpu_layout::{U32_STRIDE, VEC4_STRIDE};
use alloc::vec::Vec;

/// Absolute tolerance for `f32` equality comparisons in this module.
///
/// Floating-point `==` / `!=` are never used; call sites compare
/// `(a - b).abs() < CMP_EPS` instead so the contract stays robust to rounding.
#[cfg(test)]
const CMP_EPS: f32 = 1e-6;

/// Sentinel child index marking "no child" on a leaf node.
pub const NO_CHILD: u32 = u32::MAX;

/// Sentinel primitive index marking "not a leaf" on an internal node.
pub const NO_PRIMITIVE: u32 = u32::MAX;

/// A hand-rolled 3-component `f32` vector.
///
/// Holds `f32` fields, so it derives [`PartialEq`] (for tests) but not [`Eq`].
/// All algebra is written out by hand; no external math crate is used.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vec3 {
    /// The x component.
    pub x: f32,
    /// The y component.
    pub y: f32,
    /// The z component.
    pub z: f32,
}

impl Vec3 {
    /// Builds a vector from its three components.
    #[must_use]
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// Builds a vector with all three components equal to `v`.
    #[must_use]
    pub fn splat(v: f32) -> Self {
        Self { x: v, y: v, z: v }
    }

    /// Component-wise addition.
    #[must_use]
    #[expect(
        clippy::should_implement_trait,
        reason = "hand-rolled math keeps this contract free of operator-trait imports"
    )]
    pub fn add(self, other: Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }

    /// Component-wise subtraction (`self - other`).
    #[must_use]
    #[expect(
        clippy::should_implement_trait,
        reason = "hand-rolled math keeps this contract free of operator-trait imports"
    )]
    pub fn sub(self, other: Self) -> Self {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }

    /// Uniform scale by a scalar.
    #[must_use]
    pub fn scale(self, s: f32) -> Self {
        Self {
            x: self.x * s,
            y: self.y * s,
            z: self.z * s,
        }
    }

    /// Component-wise minimum.
    #[must_use]
    pub fn min(self, other: Self) -> Self {
        Self {
            x: self.x.min(other.x),
            y: self.y.min(other.y),
            z: self.z.min(other.z),
        }
    }

    /// Component-wise maximum.
    #[must_use]
    pub fn max(self, other: Self) -> Self {
        Self {
            x: self.x.max(other.x),
            y: self.y.max(other.y),
            z: self.z.max(other.z),
        }
    }

    /// Squared length (no `sqrt`, pure multiply-add).
    #[must_use]
    pub fn length_squared(self) -> f32 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    /// Euclidean length via `sqrt` (the only transcendental-adjacent call, and
    /// `sqrt` is explicitly permitted).
    #[must_use]
    pub fn length(self) -> f32 {
        self.length_squared().sqrt()
    }

    /// Returns the component selected by `axis` (`0` = x, `1` = y, anything
    /// else = z), used when partitioning along a chosen split axis.
    #[must_use]
    pub fn component(self, axis: u8) -> f32 {
        match axis {
            0 => self.x,
            1 => self.y,
            _ => self.z,
        }
    }
}

/// A minimal axis-aligned bounding box (`AABB`) built on [`Vec3`].
///
/// An *empty* box seeds `min` with `f32::MAX` and `max` with `f32::MIN`, so the
/// first [`Aabb::expand_point`] on any axis always wins and the empty box is
/// the identity element of [`Aabb::union`]. Holds `f32`, so it derives
/// [`PartialEq`] but not [`Eq`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Aabb {
    /// Minimum corner.
    pub min: Vec3,
    /// Maximum corner.
    pub max: Vec3,
}

impl Aabb {
    /// Builds a box directly from its two corners (no validation; callers pass
    /// a well-ordered pair or rely on [`Aabb::union`] to repair it).
    #[must_use]
    pub fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }

    /// Returns the empty box, the identity element for [`Aabb::union`].
    #[must_use]
    pub fn empty() -> Self {
        Self {
            min: Vec3::splat(f32::MAX),
            max: Vec3::splat(f32::MIN),
        }
    }

    /// Returns `true` when the box holds no points, i.e. any axis has
    /// `min > max` (never uses `==`).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.min.x > self.max.x || self.min.y > self.max.y || self.min.z > self.max.z
    }

    /// Grows the box in place so it contains the point `p`.
    pub fn expand_point(&mut self, p: Vec3) {
        self.min = self.min.min(p);
        self.max = self.max.max(p);
    }

    /// Returns the smallest box containing both `self` and `other`.
    #[must_use]
    pub fn union(&self, other: &Aabb) -> Aabb {
        Aabb {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }

    /// Returns the box center (midpoint of `min` and `max`).
    #[must_use]
    pub fn center(&self) -> Vec3 {
        self.min.add(self.max).scale(0.5)
    }

    /// Returns the per-axis half-extent (half the box size).
    #[must_use]
    pub fn half_extent(&self) -> Vec3 {
        self.max.sub(self.min).scale(0.5)
    }

    /// Returns the total surface area `2 * (dx*dy + dy*dz + dz*dx)`, using only
    /// multiply-add. Handy as the area term of a [`sah_cost`] evaluation.
    #[must_use]
    pub fn surface_area(&self) -> f32 {
        let d = self.max.sub(self.min);
        2.0 * (d.x * d.y + d.y * d.z + d.z * d.x)
    }

    /// Returns the index (`0` = x, `1` = y, `2` = z) of the longest axis. Ties
    /// resolve toward the lower index; comparisons use `>=`, never `==`.
    #[must_use]
    pub fn longest_axis(&self) -> u8 {
        let d = self.max.sub(self.min);
        if d.x >= d.y && d.x >= d.z {
            0
        } else if d.y >= d.z {
            1
        } else {
            2
        }
    }

    /// Returns `true` when `p` lies within the closed box on every axis.
    #[must_use]
    pub fn contains(&self, p: Vec3) -> bool {
        p.x >= self.min.x
            && p.x <= self.max.x
            && p.y >= self.min.y
            && p.y <= self.max.y
            && p.z >= self.min.z
            && p.z <= self.max.z
    }
}

/// Spreads the low 10 bits of `v` so two zero bits sit between each, the core
/// step of a 30-bit `Morton` code.
///
/// Pure integer bit arithmetic (the classic "magic number" bit-spread); no
/// loop, no arithmetic beyond shifts and masks. Bits above bit 9 are discarded
/// up front so the result is always a valid interleave lane.
#[must_use]
pub fn expand_bits(v: u32) -> u64 {
    let mut x = u64::from(v & 0x3FF);
    x = (x | (x << 16)) & 0x0300_00FF;
    x = (x | (x << 8)) & 0x0300_F00F;
    x = (x | (x << 4)) & 0x030C_30C3;
    x = (x | (x << 2)) & 0x0924_9249;
    x
}

/// Interleaves three 10-bit grid coordinates into a 30-bit `Morton` code.
///
/// Each axis is spread by [`expand_bits`] and shifted into its lane, so codes
/// sort into a space-filling Z-order curve. The result fits in the low 30 bits
/// of the `u64`.
#[must_use]
pub fn morton_code_3d(x: u32, y: u32, z: u32) -> u64 {
    expand_bits(x) | (expand_bits(y) << 1) | (expand_bits(z) << 2)
}

/// Quantizes a single scalar coordinate to a grid cell in `[0, resolution)`.
///
/// The value is normalized against `[lo, hi]`, clamped to `[0, 1]`, scaled by
/// `resolution`, floored, and clamped to the last cell. A degenerate domain
/// (`hi <= lo`) or zero `resolution` collapses to cell `0`, so the caller never
/// divides by zero and never indexes out of range.
#[must_use]
fn quantize_axis(v: f32, lo: f32, hi: f32, resolution: u32) -> u32 {
    let res = resolution.max(1);
    let extent = hi - lo;
    if extent <= 0.0 {
        return 0;
    }
    #[expect(
        clippy::cast_precision_loss,
        reason = "resolution is a small grid dimension, exactly representable in f32"
    )]
    let res_f = res as f32;
    let t = ((v - lo) / extent).clamp(0.0, 1.0);
    let cell_f = (t * res_f).floor();
    #[expect(
        clippy::cast_possible_truncation,
        reason = "cell_f is floored and clamped strictly below resolution"
    )]
    #[expect(
        clippy::cast_sign_loss,
        reason = "t is clamped to [0, 1] so cell_f is non-negative"
    )]
    let cell = cell_f as u32;
    cell.min(res - 1)
}

/// Quantizes a world-space point to integer grid coordinates in
/// `[0, resolution)` per axis, the input to [`morton_code_3d`].
///
/// Each axis is normalized against the matching `domain_min` / `domain_max`
/// component and clamped, so points on or outside the domain boundary map to
/// the first or last cell instead of overflowing.
#[must_use]
pub fn quantize_to_grid(
    p: Vec3,
    domain_min: Vec3,
    domain_max: Vec3,
    resolution: u32,
) -> (u32, u32, u32) {
    (
        quantize_axis(p.x, domain_min.x, domain_max.x, resolution),
        quantize_axis(p.y, domain_min.y, domain_max.y, resolution),
        quantize_axis(p.z, domain_min.z, domain_max.z, resolution),
    )
}

/// Number of `vec4` slots each `BVH` node reserves (`min.xyz` and `max.xyz`,
/// each padded to a `vec4` boundary in `std430`).
const NODE_VEC4_COUNT: usize = 2;

/// Number of scalar `u32` slots each `BVH` node reserves: the left child index,
/// the right child index, and the leaf primitive index / flag word.
const NODE_U32_COUNT: usize = 3;

/// Returns the `std430` byte stride of one flat `BVH` node.
///
/// Two `vec4`s hold the bounding-box corners and `NODE_U32_COUNT` scalar words
/// hold the child / primitive indices, matching the layout a `GPU` traversal
/// kernel binds.
#[must_use]
fn node_stride_bytes() -> usize {
    NODE_VEC4_COUNT * VEC4_STRIDE + NODE_U32_COUNT * U32_STRIDE
}

/// Returns `ceil(log2(n))` using an integer bit-shift loop (no `log`, no
/// `powf`). `n <= 1` returns `0`.
#[must_use]
fn ceil_log2(n: u32) -> u32 {
    if n <= 1 {
        return 0;
    }
    let mut v = n - 1;
    let mut bits = 0;
    while v > 0 {
        v >>= 1;
        bits += 1;
    }
    bits
}

/// Sizing plan for a flat `BVH` over `leaf_count` particle `AABB`s.
///
/// A full binary `BVH` with `n` leaves has exactly `n - 1` internal nodes and
/// `2n - 1` total nodes. This reports those counts, the `std430` byte size of
/// the node buffer, and the traversal-stack depth a `GPU` kernel must reserve —
/// all with integer arithmetic so the render graph can size `VRAM` before the
/// backend exists.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BvhTopology {
    /// Number of leaf nodes (one per input primitive).
    pub leaf_count: u32,
}

impl BvhTopology {
    /// Builds a topology plan for `leaf_count` leaves.
    #[must_use]
    pub fn new(leaf_count: u32) -> Self {
        Self { leaf_count }
    }

    /// Returns the internal-node count (`leaf_count - 1`), saturating to `0`
    /// for the empty tree so it never underflows.
    #[must_use]
    pub fn internal_node_count(&self) -> u32 {
        self.leaf_count.saturating_sub(1)
    }

    /// Returns the total node count (`2 * leaf_count - 1`); the empty tree has
    /// no nodes.
    #[must_use]
    pub fn total_node_count(&self) -> u32 {
        if self.leaf_count == 0 {
            0
        } else {
            2 * self.leaf_count - 1
        }
    }

    /// Returns the `std430` byte size of the flat node buffer.
    #[must_use]
    pub fn node_buffer_bytes(&self) -> u64 {
        let stride = u64::try_from(node_stride_bytes()).unwrap_or(0);
        u64::from(self.total_node_count()) * stride
    }

    /// Returns an upper bound on the traversal-stack depth: the balanced-tree
    /// height `ceil(log2(leaf_count))` plus a small safety margin covering
    /// median-split imbalance and the sentinel push.
    #[must_use]
    pub fn max_stack_depth(&self) -> u32 {
        const SAFETY_MARGIN: u32 = 2;
        ceil_log2(self.leaf_count) + SAFETY_MARGIN
    }
}

/// One node of the flat `BVH` node array.
///
/// Internal nodes carry `left` / `right` child indices and `is_leaf == false`;
/// leaf nodes carry a `primitive_index` with both children set to [`NO_CHILD`].
/// Holds an [`Aabb`], so it derives [`PartialEq`] but not [`Eq`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BvhNode {
    /// The node's bounding box.
    pub bounds: Aabb,
    /// Left child index, or [`NO_CHILD`] on a leaf.
    pub left: u32,
    /// Right child index, or [`NO_CHILD`] on a leaf.
    pub right: u32,
    /// Whether this node is a leaf.
    pub is_leaf: bool,
    /// Referenced primitive index on a leaf, or [`NO_PRIMITIVE`] on an internal
    /// node.
    pub primitive_index: u32,
}

/// Fixed traversal cost charged per `BVH` internal node visited.
const SAH_TRAVERSAL_COST: f32 = 1.0;

/// Cost charged per primitive intersection test.
const SAH_INTERSECT_COST: f32 = 2.0;

/// Evaluates the surface-area-heuristic (`SAH`) cost of a candidate split.
///
/// `cost = C_trav + C_isect * (A_left / A_total * n_left + A_right / A_total *
/// n_right)`, using only multiply, divide, and add. A non-positive
/// `total_area` (a degenerate parent box) collapses to the bare traversal cost
/// so the caller never divides by zero.
#[must_use]
pub fn sah_cost(
    left_area: f32,
    right_area: f32,
    total_area: f32,
    left_n: u32,
    right_n: u32,
) -> f32 {
    if total_area <= 0.0 {
        return SAH_TRAVERSAL_COST;
    }
    let inv_total = 1.0 / total_area;
    #[expect(
        clippy::cast_precision_loss,
        reason = "primitive counts are small relative to the f32 mantissa"
    )]
    let left_f = left_n as f32;
    #[expect(
        clippy::cast_precision_loss,
        reason = "primitive counts are small relative to the f32 mantissa"
    )]
    let right_f = right_n as f32;
    SAH_TRAVERSAL_COST
        + SAH_INTERSECT_COST * (left_area * inv_total * left_f + right_area * inv_total * right_f)
}

/// Recursively builds a median-split subtree over `indices`, pushing nodes into
/// `nodes` and returning the index of the subtree root.
fn build_recursive(prims: &[Aabb], indices: &mut [usize], nodes: &mut Vec<BvhNode>) -> u32 {
    let mut bounds = Aabb::empty();
    for &i in indices.iter() {
        bounds = bounds.union(&prims[i]);
    }

    if indices.len() == 1 {
        let node_index = nodes.len();
        nodes.push(BvhNode {
            bounds,
            left: NO_CHILD,
            right: NO_CHILD,
            is_leaf: true,
            primitive_index: u32::try_from(indices[0]).unwrap_or(0),
        });
        return u32::try_from(node_index).unwrap_or(0);
    }

    let mut centroid_bounds = Aabb::empty();
    for &i in indices.iter() {
        centroid_bounds.expand_point(prims[i].center());
    }
    let axis = centroid_bounds.longest_axis();
    indices.sort_by(|&a, &b| {
        let ca = prims[a].center().component(axis);
        let cb = prims[b].center().component(axis);
        ca.total_cmp(&cb)
    });

    let node_index = nodes.len();
    nodes.push(BvhNode {
        bounds,
        left: NO_CHILD,
        right: NO_CHILD,
        is_leaf: false,
        primitive_index: NO_PRIMITIVE,
    });

    let mid = indices.len() / 2;
    let (left_part, right_part) = indices.split_at_mut(mid);
    let left = build_recursive(prims, left_part, nodes);
    let right = build_recursive(prims, right_part, nodes);
    nodes[node_index].left = left;
    nodes[node_index].right = right;
    u32::try_from(node_index).unwrap_or(0)
}

/// Builds a flat `BVH` over `primitives` by recursive median split on the
/// longest centroid axis.
///
/// The returned array places the root at index `0`. An empty input yields an
/// empty array; a single primitive yields a lone leaf. Every internal node has
/// two non-empty children, so the array always holds `2n - 1` nodes with `n`
/// leaves, each leaf referencing exactly one primitive.
#[must_use]
pub fn build_median_split(primitives: &[Aabb]) -> Vec<BvhNode> {
    let mut nodes = Vec::new();
    if primitives.is_empty() {
        return nodes;
    }
    let mut indices: Vec<usize> = (0..primitives.len()).collect();
    build_recursive(primitives, &mut indices, &mut nodes);
    nodes
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < CMP_EPS
    }

    #[test]
    fn vec3_algebra_and_length() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        assert_eq!(a.add(b), Vec3::new(5.0, 7.0, 9.0));
        assert_eq!(b.sub(a), Vec3::new(3.0, 3.0, 3.0));
        assert_eq!(a.scale(2.0), Vec3::new(2.0, 4.0, 6.0));
        assert_eq!(a.min(b), Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(a.max(b), Vec3::new(4.0, 5.0, 6.0));
        assert!(approx(Vec3::new(3.0, 4.0, 0.0).length(), 5.0));
        assert!(approx(Vec3::new(0.0, 0.0, 2.0).component(2), 2.0));
    }

    #[test]
    fn aabb_algebra_is_exact() {
        let b = Aabb::new(Vec3::splat(0.0), Vec3::splat(2.0));
        assert!(approx(b.surface_area(), 24.0));
        assert_eq!(b.center(), Vec3::splat(1.0));
        assert_eq!(b.half_extent(), Vec3::splat(1.0));
        assert!(b.contains(Vec3::splat(1.0)));
        assert!(!b.contains(Vec3::new(3.0, 0.0, 0.0)));
        assert!(Aabb::empty().is_empty());
    }

    #[test]
    fn aabb_longest_axis_and_union() {
        let b = Aabb::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 5.0, 2.0));
        assert_eq!(b.longest_axis(), 1);
        let u = Aabb::empty()
            .union(&Aabb::new(Vec3::splat(-1.0), Vec3::splat(0.0)))
            .union(&Aabb::new(Vec3::splat(0.0), Vec3::splat(1.0)));
        assert_eq!(u.min, Vec3::splat(-1.0));
        assert_eq!(u.max, Vec3::splat(1.0));
    }

    #[test]
    fn expand_bits_spreads_and_masks() {
        assert_eq!(expand_bits(0), 0);
        assert_eq!(expand_bits(1), 1);
        assert_eq!(expand_bits(3), 0b1001);
        // All 10 low bits set -> the canonical every-third-bit pattern.
        assert_eq!(expand_bits(0x3FF), 0x0924_9249);
        // Bits above bit 9 are discarded before spreading.
        assert_eq!(expand_bits(0xFFFF_FFFF), 0x0924_9249);
    }

    #[test]
    fn morton_code_lanes_and_monotonic() {
        assert_eq!(morton_code_3d(1, 0, 0), 1);
        assert_eq!(morton_code_3d(0, 1, 0), 2);
        assert_eq!(morton_code_3d(0, 0, 1), 4);
        assert_eq!(morton_code_3d(1, 1, 1), 7);

        // Increasing a single axis (others fixed) yields increasing codes.
        let mut prev = morton_code_3d(0, 5, 9);
        for x in 1..16u32 {
            let code = morton_code_3d(x, 5, 9);
            assert!(code > prev);
            prev = code;
        }

        // Increasing all three axes together is also monotone.
        let mut prev_diag = morton_code_3d(0, 0, 0);
        for i in 1..16u32 {
            let code = morton_code_3d(i, i, i);
            assert!(code > prev_diag);
            prev_diag = code;
        }
    }

    #[test]
    fn quantize_clamps_boundaries() {
        let lo = Vec3::splat(0.0);
        let hi = Vec3::splat(10.0);
        assert_eq!(quantize_to_grid(Vec3::splat(0.0), lo, hi, 10), (0, 0, 0));
        assert_eq!(quantize_to_grid(Vec3::splat(5.0), lo, hi, 10), (5, 5, 5));
        // A point on the far boundary clamps into the last cell, not out.
        assert_eq!(quantize_to_grid(Vec3::splat(10.0), lo, hi, 10), (9, 9, 9));
        // Points outside the domain clamp to the first / last cell.
        assert_eq!(quantize_to_grid(Vec3::splat(-5.0), lo, hi, 10), (0, 0, 0));
        assert_eq!(quantize_to_grid(Vec3::splat(99.0), lo, hi, 10), (9, 9, 9));
        // Degenerate domain and zero resolution both collapse to cell 0.
        assert_eq!(quantize_to_grid(Vec3::splat(5.0), hi, lo, 10), (0, 0, 0));
        assert_eq!(quantize_to_grid(Vec3::splat(5.0), lo, hi, 0), (0, 0, 0));
    }

    #[test]
    fn topology_counts_and_sizing() {
        let empty = BvhTopology::new(0);
        assert_eq!(empty.internal_node_count(), 0);
        assert_eq!(empty.total_node_count(), 0);
        assert_eq!(empty.node_buffer_bytes(), 0);

        let one = BvhTopology::new(1);
        assert_eq!(one.internal_node_count(), 0);
        assert_eq!(one.total_node_count(), 1);

        let eight = BvhTopology::new(8);
        assert_eq!(eight.internal_node_count(), 7);
        assert_eq!(eight.total_node_count(), 15);
        // 2 vec4 (32) + 3 u32 (12) = 44 bytes per node, times 15 nodes.
        assert_eq!(eight.node_buffer_bytes(), 15 * 44);
    }

    #[test]
    fn max_stack_depth_tracks_height() {
        // ceil(log2) + margin(2): 1 -> 0+2, 8 -> 3+2, 1024 -> 10+2.
        assert_eq!(BvhTopology::new(1).max_stack_depth(), 2);
        assert_eq!(BvhTopology::new(8).max_stack_depth(), 5);
        assert_eq!(BvhTopology::new(1024).max_stack_depth(), 12);
        assert!(BvhTopology::new(1024).max_stack_depth() > BvhTopology::new(8).max_stack_depth());
    }

    #[test]
    fn sah_cost_matches_formula() {
        // Balanced split: 1 + 2 * (1*0.5*1 + 1*0.5*1) = 3.
        assert!(approx(sah_cost(1.0, 1.0, 2.0, 1, 1), 3.0));
        // Degenerate parent area collapses to the traversal cost.
        assert!(approx(sah_cost(1.0, 1.0, 0.0, 4, 4), SAH_TRAVERSAL_COST));
    }

    #[test]
    fn build_empty_and_single() {
        assert!(build_median_split(&[]).is_empty());
        let single = Aabb::new(Vec3::splat(0.0), Vec3::splat(1.0));
        let nodes = build_median_split(&[single]);
        assert_eq!(nodes.len(), 1);
        assert!(nodes[0].is_leaf);
        assert_eq!(nodes[0].primitive_index, 0);
        assert_eq!(nodes[0].left, NO_CHILD);
        assert_eq!(nodes[0].right, NO_CHILD);
    }

    #[test]
    fn build_small_bvh_is_well_formed() {
        let prims = [
            Aabb::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0)),
            Aabb::new(Vec3::new(5.0, 0.0, 0.0), Vec3::new(6.0, 1.0, 1.0)),
            Aabb::new(Vec3::new(0.0, 5.0, 0.0), Vec3::new(1.0, 6.0, 1.0)),
            Aabb::new(Vec3::new(5.0, 5.0, 0.0), Vec3::new(6.0, 6.0, 1.0)),
            Aabb::new(Vec3::new(2.5, 2.5, 4.0), Vec3::new(3.5, 3.5, 5.0)),
        ];
        let nodes = build_median_split(&prims);

        // node_count == 2n - 1.
        assert_eq!(nodes.len(), 2 * prims.len() - 1);
        assert_eq!(
            u32::try_from(nodes.len()).unwrap_or(0),
            BvhTopology::new(u32::try_from(prims.len()).unwrap_or(0)).total_node_count()
        );

        // Root bounds contain every input corner.
        let root = nodes[0].bounds;
        for prim in &prims {
            assert!(root.contains(prim.min));
            assert!(root.contains(prim.max));
        }

        // Every leaf references exactly one distinct primitive, covering all.
        let mut seen = [false; 5];
        let mut leaf_count = 0;
        for node in &nodes {
            if node.is_leaf {
                leaf_count += 1;
                let idx = node.primitive_index as usize;
                assert!(!seen[idx], "primitive {idx} referenced twice");
                seen[idx] = true;
                assert_eq!(node.left, NO_CHILD);
                assert_eq!(node.right, NO_CHILD);
            } else {
                assert!(node.left != NO_CHILD && node.right != NO_CHILD);
                assert_eq!(node.primitive_index, NO_PRIMITIVE);
            }
        }
        assert_eq!(leaf_count, prims.len());
        assert!(seen.iter().all(|&s| s));
    }
}
