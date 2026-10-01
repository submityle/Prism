//! Hierarchical light-BVH importance sampling — CPU golden reference.
//!
//! A scene with thousands of emitters cannot afford to evaluate every light at
//! every shading point, so production renderers organise the emitters into a
//! binary *light tree* and importance-sample a single leaf per lookup.  This
//! module is the backend-neutral numerical reference for that machinery,
//! following Conty-Estevez & Kulla 2018 (*Importance Sampling of Many Lights
//! with Adaptive Tree Splitting*, the Cycles / Arnold light tree):
//!
//! * [`LightCone`] is a bounding cone of emission directions — an orientation
//!   axis with an orientation half-angle `theta_o` and an emission half-angle
//!   `theta_e` — and [`LightCone::union`] merges two cones into the tightest
//!   enclosing cone exactly as the paper's `cone_union` does.
//! * [`LightBounds`] is the lightweight cluster representation shared by leaves
//!   and internal nodes: an axis-aligned box, a [`LightCone`], and the total
//!   emitted `power`.  [`LightBounds::point`], [`LightBounds::spot`] and
//!   [`LightBounds::area`] build the three canonical emitter kinds, and
//!   [`LightBounds::union`] aggregates two clusters.
//! * [`importance`] is the scalar cluster importance a shading point sees:
//!   `power * orientation_term / dist^2`, with a minimum-distance clamp so a
//!   point grazing a cluster centroid cannot blow the metric up to infinity.
//! * [`LightTree`] is the binary tree itself.  [`LightTree::build`] constructs
//!   it deterministically by recursive median splitting of cluster centroids,
//!   [`LightTree::sample`] performs a stochastic top-down walk driven by a
//!   single supplied uniform `u in [0, 1)` and returns the chosen leaf with the
//!   product of per-level selection probabilities as its `pdf`, and
//!   [`LightTree::leaf_pmf`] returns the full analytic probability mass over
//!   leaves (which must sum to one).
//!
//! # Conventions
//! * `no_std`: containers via `alloc`; vector math via `bevy_math`; every
//!   transcendental (`acos`, `cos`, `sin`) is routed through
//!   [`bevy_math::ops`] for cross-platform bit-stability — never `f32::cos`.
//! * All cluster data is stored as `f32` / `u32` to mirror the GPU light-tree
//!   buffer twin: a node packs an AABB (`6 f32`), a cone (`4 f32` plus one more
//!   for `theta_e`), a `power` scalar, and two `u32` child/leaf links.
//! * The stochastic walk rescales the uniform at every branch
//!   (`u' = (u - p_left) / p_right`) so one `u` drives the entire descent and
//!   the returned `pdf` is the exact product of branch probabilities — the
//!   same value [`leaf_pmf`](LightTree::leaf_pmf) assigns that leaf.
//! * A cluster with zero (or non-finite) aggregate importance makes its parent
//!   fall back to a uniform `1/2` branch split, so a fully unlit subtree is
//!   sampled uniformly instead of producing `NaN`.
//! * Every function is a deterministic pure function: no RNG, no I/O, no GPU,
//!   no global state, and no `unsafe`.

use alloc::vec::Vec;
use core::cmp::Ordering;
use core::f32::consts::{FRAC_PI_2, PI};

use bevy_math::{Vec3, ops};

/// Sentinel stored in [`LightTreeNode::right`] to mark a leaf node.
///
/// For a leaf the [`LightTreeNode::left_or_light`] field holds the emitter
/// index instead of a child-node index, and `right == LEAF`.
pub const LEAF: u32 = u32::MAX;

/// Smallest squared distance the importance metric is allowed to divide by.
///
/// This is combined with the caller-supplied `min_dist` so that even a request
/// with `min_dist == 0` cannot divide by zero when a shading point coincides
/// with a cluster centroid.
const MIN_DIST2_FLOOR: f32 = f32::MIN_POSITIVE;

/// A bounding cone of emission directions for a light cluster.
///
/// The cone is centred on the unit `axis` and spans two nested half-angles,
/// matching Conty-Estevez: `theta_o` bounds how far the emitters' *orientation*
/// normals spread from the axis, and `theta_e` bounds the additional
/// *emission* falloff beyond those normals.  A cluster emits non-negligibly
/// towards a point only when the point lies within `theta_o + theta_e` of the
/// axis.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LightCone {
    /// Unit central axis of the cone (the aggregate emission direction).
    pub axis: Vec3,
    /// Orientation half-angle in radians, clamped to `[0, pi]`.
    pub theta_o: f32,
    /// Emission half-angle in radians, clamped to `[0, pi/2]`.
    pub theta_e: f32,
}

impl LightCone {
    /// A cone that covers the whole sphere of directions (`theta_o = pi`).
    ///
    /// Used for isotropic point lights, which emit equally in all directions
    /// and therefore impose no orientation constraint.
    pub const FULL_SPHERE: Self = Self {
        axis: Vec3::Z,
        theta_o: PI,
        theta_e: FRAC_PI_2,
    };

    /// Builds a cone from an axis and the two half-angles, normalising the axis
    /// and clamping the angles into their valid ranges.
    ///
    /// A degenerate (zero-length) axis falls back to `+z`; `theta_o` is clamped
    /// to `[0, pi]` and `theta_e` to `[0, pi/2]`.
    #[inline]
    pub fn new(axis: Vec3, theta_o: f32, theta_e: f32) -> Self {
        Self {
            axis: normalize_or_z(axis),
            theta_o: clamp_finite(theta_o, 0.0, PI),
            theta_e: clamp_finite(theta_e, 0.0, FRAC_PI_2),
        }
    }

    /// Returns the tightest cone enclosing both `self` and `other`.
    ///
    /// This is the Conty-Estevez `cone_union`: the wider-orientation cone is
    /// taken as the base, and if it does not already contain the other cone the
    /// axis is rotated towards the other axis by half the uncovered spread and
    /// `theta_o` is grown to span both.  The emission half-angle is the maximum
    /// of the two.  Nearly-opposite axes that would require a `>= pi`
    /// orientation collapse to a full-sphere cone.
    #[inline]
    pub fn union(self, other: Self) -> Self {
        // Order so `a` has the wider orientation half-angle.
        let (a, b) = if self.theta_o >= other.theta_o {
            (self, other)
        } else {
            (other, self)
        };

        let theta_e = a.theta_e.max(b.theta_e);
        let theta_d = angle_between(a.axis, b.axis);

        // `a` already encloses `b`'s orientation cone.
        if (theta_d + b.theta_o).min(PI) <= a.theta_o {
            return Self {
                axis: a.axis,
                theta_o: a.theta_o,
                theta_e,
            };
        }

        // Grow `a` to span both orientation cones.
        let theta_o = (a.theta_o + theta_d + b.theta_o) * 0.5;
        if theta_o >= PI {
            return Self {
                axis: a.axis,
                theta_o: PI,
                theta_e,
            };
        }

        let theta_r = theta_o - a.theta_o;
        let axis = rotate_towards(a.axis, b.axis, theta_r);
        Self {
            axis,
            theta_o,
            theta_e,
        }
    }
}

/// Lightweight bounds of a single emitter or an aggregated cluster of emitters.
///
/// This is the only payload stored per tree node: an axis-aligned box, a
/// bounding emission [`LightCone`], and the total radiant `power` of everything
/// in the cluster.  Point, spot and area emitters all collapse into this common
/// representation via the dedicated constructors.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LightBounds {
    /// Minimum corner of the axis-aligned bounding box.
    pub aabb_min: Vec3,
    /// Maximum corner of the axis-aligned bounding box.
    pub aabb_max: Vec3,
    /// Bounding cone of emission directions for the cluster.
    pub cone: LightCone,
    /// Total emitted radiant power of the cluster (non-negative).
    pub power: f32,
}

impl LightBounds {
    /// Builds bounds for an isotropic point light at `position`.
    ///
    /// The box is the degenerate point `position`, the cone is the full sphere
    /// (a point light emits in every direction), and `power` is clamped
    /// non-negative.
    #[inline]
    pub fn point(position: Vec3, power: f32) -> Self {
        Self {
            aabb_min: position,
            aabb_max: position,
            cone: LightCone::FULL_SPHERE,
            power: power.max(0.0),
        }
    }

    /// Builds bounds for a spot light at `position` aiming along `direction`.
    ///
    /// `cone_half_angle` is the spot's outer falloff half-angle (its emission
    /// half-angle `theta_e`); the orientation half-angle `theta_o` is `0`
    /// because a single spot has one well-defined normal.
    #[inline]
    pub fn spot(position: Vec3, direction: Vec3, cone_half_angle: f32, power: f32) -> Self {
        Self {
            aabb_min: position,
            aabb_max: position,
            cone: LightCone::new(direction, 0.0, cone_half_angle),
            power: power.max(0.0),
        }
    }

    /// Builds bounds for a planar area light spanning `aabb_min..aabb_max` with
    /// surface `normal`.
    ///
    /// A flat emitter has a single orientation (`theta_o = 0`) and radiates over
    /// the hemisphere around its normal (`theta_e = pi/2`).  The corners are
    /// sorted component-wise so the box is well formed regardless of argument
    /// order.
    #[inline]
    pub fn area(aabb_min: Vec3, aabb_max: Vec3, normal: Vec3, power: f32) -> Self {
        Self {
            aabb_min: aabb_min.min(aabb_max),
            aabb_max: aabb_min.max(aabb_max),
            cone: LightCone::new(normal, 0.0, FRAC_PI_2),
            power: power.max(0.0),
        }
    }

    /// The geometric centre of the bounding box.
    #[inline]
    pub fn centroid(&self) -> Vec3 {
        (self.aabb_min + self.aabb_max) * 0.5
    }

    /// Returns the tightest bounds enclosing both `self` and `other`.
    ///
    /// The boxes are merged component-wise, the cones via [`LightCone::union`],
    /// and the powers are summed — exactly the aggregation an internal node
    /// stores for its two children.
    #[inline]
    pub fn union(self, other: Self) -> Self {
        Self {
            aabb_min: self.aabb_min.min(other.aabb_min),
            aabb_max: self.aabb_max.max(other.aabb_max),
            cone: self.cone.union(other.cone),
            power: self.power + other.power,
        }
    }
}

/// Scalar importance a shading point assigns to a light cluster.
///
/// Returns `power * orientation_term / dist^2`, where:
///
/// * `dist^2` is the squared distance from `point` to the cluster centroid,
///   floored by `max(min_dist, 0)^2` (and an absolute tiny floor) so a point at
///   the centroid yields a large-but-finite value instead of infinity.
/// * `orientation_term` is `cos(theta')` with
///   `theta' = max(theta - theta_o - theta_e, 0)`, `theta` being the angle
///   between the cone axis and the direction from the cluster to the point.  It
///   is `1` when the point lies inside the emission cone and falls to `0` as the
///   point rotates a further `pi/2` outside it.
///
/// The result is always finite and non-negative; a zero-power or
/// outside-the-cone cluster returns `0`.
#[inline]
pub fn importance(bounds: &LightBounds, point: Vec3, min_dist: f32) -> f32 {
    let power = bounds.power.max(0.0);
    if power <= 0.0 {
        return 0.0;
    }

    let centroid = bounds.centroid();
    let to_point = point - centroid;
    let dist2_raw = to_point.length_squared();

    let md = min_dist.max(0.0);
    let dist2 = dist2_raw.max(md * md).max(MIN_DIST2_FLOOR);

    // Direction from the cluster towards the shading point.  When the point
    // coincides with the centroid the orientation is unconstrained, so treat
    // the cluster as fully facing the point.
    let orientation = if dist2_raw > MIN_DIST2_FLOOR {
        let dir = to_point * dist2_raw.sqrt().recip();
        let cos_a = bounds.cone.axis.dot(dir).clamp(-1.0, 1.0);
        let theta = ops::acos(cos_a);
        let spread = bounds.cone.theta_o + bounds.cone.theta_e;
        let theta_eff = (theta - spread).max(0.0);
        if theta_eff < FRAC_PI_2 {
            ops::cos(theta_eff)
        } else {
            0.0
        }
    } else {
        1.0
    };

    let imp = power * orientation / dist2;
    if imp.is_finite() && imp > 0.0 {
        imp
    } else {
        0.0
    }
}

/// A single node of the binary [`LightTree`].
///
/// Internal and leaf nodes share the same `f32`/`u32` layout to mirror the GPU
/// buffer twin.  For an internal node `left_or_light` and `right` are child node
/// indices; for a leaf `right == LEAF` and `left_or_light` is the emitter index
/// into the slice passed to [`LightTree::build`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LightTreeNode {
    /// Aggregated bounds of everything beneath (or, for a leaf, at) this node.
    pub bounds: LightBounds,
    /// Left child node index for an internal node, or the emitter index for a
    /// leaf (`right == LEAF`).
    pub left_or_light: u32,
    /// Right child node index for an internal node, or [`LEAF`] for a leaf.
    pub right: u32,
}

impl LightTreeNode {
    /// Whether this node is a leaf (holds an emitter index rather than
    /// children).
    #[inline]
    pub fn is_leaf(&self) -> bool {
        self.right == LEAF
    }
}

/// A chosen emitter and the probability with which the walk selected it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LightSample {
    /// Index of the selected emitter in the slice passed to
    /// [`LightTree::build`].
    pub light_index: u32,
    /// Probability mass of selecting this emitter — the product of the
    /// per-level branch probabilities along the descent.
    pub pdf: f32,
}

/// A binary light BVH over a set of emitter [`LightBounds`].
///
/// Nodes are stored flat in [`LightTree::nodes`]; children are appended before
/// their parent during construction, so the last node is always the root
/// ([`LightTree::root`]).
#[derive(Clone, Debug, PartialEq)]
pub struct LightTree {
    nodes: Vec<LightTreeNode>,
    root: u32,
    leaf_count: u32,
}

impl LightTree {
    /// Builds a light tree over `lights` by recursive median splitting.
    ///
    /// At each level the cluster centroids are split along their widest axis at
    /// the median, giving a balanced, fully deterministic tree.  Emitter
    /// indices refer to positions in `lights`.  Returns `None` for an empty
    /// input (there is nothing to sample).
    pub fn build(lights: &[LightBounds]) -> Option<Self> {
        if lights.is_empty() {
            return None;
        }
        let mut nodes = Vec::with_capacity(lights.len() * 2 - 1);
        let mut indices: Vec<u32> = (0..lights.len() as u32).collect();
        let root = build_recursive(lights, &mut indices, &mut nodes);
        Some(Self {
            nodes,
            root,
            leaf_count: lights.len() as u32,
        })
    }

    /// All nodes in construction order (children before parents, root last).
    #[inline]
    pub fn nodes(&self) -> &[LightTreeNode] {
        &self.nodes
    }

    /// Index of the root node in [`nodes`](Self::nodes).
    #[inline]
    pub fn root(&self) -> u32 {
        self.root
    }

    /// Number of emitter leaves in the tree.
    #[inline]
    pub fn leaf_count(&self) -> u32 {
        self.leaf_count
    }

    /// Probability of descending into the *left* child of `node` as seen from
    /// `point`.
    ///
    /// This is the shared decision used by both [`sample`](Self::sample) and
    /// [`leaf_pmf`](Self::leaf_pmf), so the stochastic walk and the analytic
    /// mass always agree.  When both children have zero / non-finite importance
    /// it falls back to an unbiased `0.5` split.
    #[inline]
    fn left_probability(&self, node: &LightTreeNode, point: Vec3, min_dist: f32) -> f32 {
        let left = &self.nodes[node.left_or_light as usize];
        let right = &self.nodes[node.right as usize];
        let wl = importance(&left.bounds, point, min_dist);
        let wr = importance(&right.bounds, point, min_dist);
        let total = wl + wr;
        if total > 0.0 && total.is_finite() {
            (wl / total).clamp(0.0, 1.0)
        } else {
            0.5
        }
    }

    /// Stochastically walks from the root to a leaf using one uniform `u`.
    ///
    /// At each internal node the walk descends left with probability
    /// `p_left` (see [`left_probability`](Self::left_probability)); `u` is
    /// rescaled into `[0, 1)` for the next level so a single supplied uniform
    /// drives the whole descent.  The returned [`LightSample::pdf`] is the
    /// product of the branch probabilities taken, which equals the leaf's entry
    /// in [`leaf_pmf`](Self::leaf_pmf).  Returns `None` only for an empty tree.
    pub fn sample(&self, point: Vec3, min_dist: f32, u: f32) -> Option<LightSample> {
        if self.nodes.is_empty() {
            return None;
        }
        let mut node_idx = self.root;
        let mut pdf = 1.0f32;
        let mut u = u.clamp(0.0, 1.0);

        loop {
            let node = self.nodes[node_idx as usize];
            if node.is_leaf() {
                return Some(LightSample {
                    light_index: node.left_or_light,
                    pdf,
                });
            }

            let p_left = self.left_probability(&node, point, min_dist);
            let go_left = if p_left <= 0.0 {
                false
            } else if p_left >= 1.0 {
                true
            } else {
                u < p_left
            };

            if go_left {
                pdf *= p_left;
                // Rescale `u` into the left sub-interval `[0, p_left)`.
                u = if p_left > 0.0 {
                    (u / p_left).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                node_idx = node.left_or_light;
            } else {
                let p_right = 1.0 - p_left;
                pdf *= p_right;
                u = if p_right > 0.0 {
                    ((u - p_left) / p_right).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                node_idx = node.right;
            }
        }
    }

    /// Returns the analytic probability mass over all leaves for `point`.
    ///
    /// Each entry is `(emitter_index, probability)`; the probabilities sum to
    /// `1` (every internal split contributes `p_left + p_right = 1`).  This is
    /// the exact distribution [`sample`](Self::sample) draws from and is the
    /// basis for the partition-of-unity and monotonicity tests.
    pub fn leaf_pmf(&self, point: Vec3, min_dist: f32) -> Vec<(u32, f32)> {
        let mut out = Vec::with_capacity(self.leaf_count as usize);
        if !self.nodes.is_empty() {
            self.accumulate_pmf(self.root, point, min_dist, 1.0, &mut out);
        }
        out
    }

    /// Recursively distributes `prob` down the tree into per-leaf masses.
    fn accumulate_pmf(
        &self,
        node_idx: u32,
        point: Vec3,
        min_dist: f32,
        prob: f32,
        out: &mut Vec<(u32, f32)>,
    ) {
        let node = self.nodes[node_idx as usize];
        if node.is_leaf() {
            out.push((node.left_or_light, prob));
            return;
        }
        let p_left = self.left_probability(&node, point, min_dist);
        self.accumulate_pmf(node.left_or_light, point, min_dist, prob * p_left, out);
        self.accumulate_pmf(node.right, point, min_dist, prob * (1.0 - p_left), out);
    }
}

/// Recursively builds a subtree over `indices`, appending nodes to `nodes` and
/// returning the index of the subtree root.
fn build_recursive(
    lights: &[LightBounds],
    indices: &mut [u32],
    nodes: &mut Vec<LightTreeNode>,
) -> u32 {
    debug_assert!(!indices.is_empty());

    if indices.len() == 1 {
        let light = indices[0];
        nodes.push(LightTreeNode {
            bounds: lights[light as usize],
            left_or_light: light,
            right: LEAF,
        });
        return (nodes.len() - 1) as u32;
    }

    // Choose the split axis as the widest extent of the centroid bounds.
    let mut cmin = lights[indices[0] as usize].centroid();
    let mut cmax = cmin;
    for &i in indices.iter() {
        let c = lights[i as usize].centroid();
        cmin = cmin.min(c);
        cmax = cmax.max(c);
    }
    let extent = cmax - cmin;
    let axis = if extent.x >= extent.y && extent.x >= extent.z {
        0
    } else if extent.y >= extent.z {
        1
    } else {
        2
    };

    // Deterministic median split along the chosen axis; ties break by emitter
    // index so the ordering is total and stable.
    indices.sort_by(|&a, &b| {
        let ca = axis_component(lights[a as usize].centroid(), axis);
        let cb = axis_component(lights[b as usize].centroid(), axis);
        ca.partial_cmp(&cb).unwrap_or(Ordering::Equal).then(a.cmp(&b))
    });

    let mid = indices.len() / 2;
    let (left_idx, right_idx) = indices.split_at_mut(mid);
    let left = build_recursive(lights, left_idx, nodes);
    let right = build_recursive(lights, right_idx, nodes);

    let bounds = nodes[left as usize]
        .bounds
        .union(nodes[right as usize].bounds);
    nodes.push(LightTreeNode {
        bounds,
        left_or_light: left,
        right,
    });
    (nodes.len() - 1) as u32
}

/// Returns the requested component (`0 = x`, `1 = y`, else `z`) of `v`.
#[inline]
fn axis_component(v: Vec3, axis: u8) -> f32 {
    match axis {
        0 => v.x,
        1 => v.y,
        _ => v.z,
    }
}

/// Normalises `v`, falling back to `+z` for a degenerate (zero) input.
#[inline]
fn normalize_or_z(v: Vec3) -> Vec3 {
    let len_sq = v.length_squared();
    if len_sq > f32::MIN_POSITIVE {
        v * len_sq.sqrt().recip()
    } else {
        Vec3::Z
    }
}

/// Clamps `x` into `[lo, hi]`, mapping a non-finite input to `lo`.
#[inline]
fn clamp_finite(x: f32, lo: f32, hi: f32) -> f32 {
    if x.is_finite() {
        x.clamp(lo, hi)
    } else {
        lo
    }
}

/// Unsigned angle in `[0, pi]` between two (not necessarily unit) vectors.
#[inline]
fn angle_between(a: Vec3, b: Vec3) -> f32 {
    let a = normalize_or_z(a);
    let b = normalize_or_z(b);
    ops::acos(a.dot(b).clamp(-1.0, 1.0))
}

/// Rotates unit vector `from` towards unit vector `to` by `angle` radians.
///
/// Uses Rodrigues' rotation about the normalised cross product of the two
/// vectors.  When the vectors are (anti)parallel the cross product vanishes, so
/// an arbitrary perpendicular axis is chosen; this keeps the cone-union axis
/// finite even for exactly opposing child cones.
#[inline]
fn rotate_towards(from: Vec3, to: Vec3, angle: f32) -> Vec3 {
    let from = normalize_or_z(from);
    let to = normalize_or_z(to);
    let cross = from.cross(to);
    let axis = if cross.length_squared() > f32::MIN_POSITIVE {
        cross.normalize()
    } else {
        any_perpendicular(from)
    };
    let (s, c) = ops::sin_cos(angle);
    // Rodrigues' formula: v*cos + (k x v)*sin + k*(k.v)*(1 - cos).
    from * c + axis.cross(from) * s + axis * (axis.dot(from) * (1.0 - c))
}

/// Returns a unit vector perpendicular to `v`.
#[inline]
fn any_perpendicular(v: Vec3) -> Vec3 {
    // Cross with whichever cardinal axis is least aligned with `v`.
    let reference = if v.x.abs() <= v.y.abs() && v.x.abs() <= v.z.abs() {
        Vec3::X
    } else if v.y.abs() <= v.z.abs() {
        Vec3::Y
    } else {
        Vec3::Z
    };
    normalize_or_z(v.cross(reference))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds four isotropic point lights laid out along `+x`.
    fn four_point_lights(powers: [f32; 4]) -> [LightBounds; 4] {
        [
            LightBounds::point(Vec3::new(0.0, 0.0, 0.0), powers[0]),
            LightBounds::point(Vec3::new(1.0, 0.0, 0.0), powers[1]),
            LightBounds::point(Vec3::new(2.0, 0.0, 0.0), powers[2]),
            LightBounds::point(Vec3::new(3.0, 0.0, 0.0), powers[3]),
        ]
    }

    #[test]
    fn build_single_light_is_a_leaf() {
        let lights = [LightBounds::point(Vec3::ZERO, 5.0)];
        let tree = LightTree::build(&lights).unwrap();
        assert_eq!(tree.leaf_count(), 1);
        assert_eq!(tree.nodes().len(), 1);
        assert!(tree.nodes()[tree.root() as usize].is_leaf());

        let s = tree.sample(Vec3::new(0.0, 0.0, 10.0), 0.1, 0.42).unwrap();
        assert_eq!(s.light_index, 0);
        assert!((s.pdf - 1.0).abs() < 1e-6);
    }

    #[test]
    fn build_has_expected_node_count() {
        // A full binary tree over n leaves has 2n - 1 nodes.
        let lights = four_point_lights([1.0, 1.0, 1.0, 1.0]);
        let tree = LightTree::build(&lights).unwrap();
        assert_eq!(tree.nodes().len(), 2 * 4 - 1);
        assert_eq!(tree.leaf_count(), 4);
    }

    #[test]
    fn empty_input_builds_no_tree() {
        assert!(LightTree::build(&[]).is_none());
    }

    #[test]
    fn leaf_pmf_is_a_partition_of_unity() {
        let lights = four_point_lights([1.0, 2.0, 3.0, 4.0]);
        let tree = LightTree::build(&lights).unwrap();
        let point = Vec3::new(0.5, 2.0, 0.0);
        let pmf = tree.leaf_pmf(point, 0.1);
        assert_eq!(pmf.len(), 4);
        let sum: f32 = pmf.iter().map(|&(_, p)| p).sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum={sum}");
        // Every leaf appears exactly once with a non-negative mass.
        for &(_, p) in &pmf {
            assert!(p >= 0.0 && p.is_finite());
        }
    }

    #[test]
    fn sample_pdf_matches_leaf_pmf() {
        let lights = four_point_lights([1.0, 2.0, 3.0, 4.0]);
        let tree = LightTree::build(&lights).unwrap();
        let point = Vec3::new(1.5, 3.0, 1.0);
        let min_dist = 0.1;
        let pmf = tree.leaf_pmf(point, min_dist);
        // Sweep the unit interval and check each drawn pdf equals the leaf mass.
        for k in 0..64 {
            let u = (k as f32 + 0.5) / 64.0;
            let s = tree.sample(point, min_dist, u).unwrap();
            let want = pmf
                .iter()
                .find(|&&(idx, _)| idx == s.light_index)
                .map(|&(_, p)| p)
                .unwrap();
            assert!((s.pdf - want).abs() < 1e-5, "pdf={} want={want}", s.pdf);
        }
    }

    #[test]
    fn sample_covers_every_leaf_over_the_unit_interval() {
        let lights = four_point_lights([1.0, 1.0, 1.0, 1.0]);
        let tree = LightTree::build(&lights).unwrap();
        let point = Vec3::new(1.5, 5.0, 0.0);
        let mut seen = [false; 4];
        for k in 0..256 {
            let u = k as f32 / 256.0;
            let s = tree.sample(point, 0.1, u).unwrap();
            seen[s.light_index as usize] = true;
        }
        assert!(seen.iter().all(|&b| b), "not every leaf was reached");
    }

    #[test]
    fn importance_increases_with_power() {
        let point = Vec3::new(0.0, 0.0, 5.0);
        let weak = LightBounds::point(Vec3::ZERO, 1.0);
        let strong = LightBounds::point(Vec3::ZERO, 10.0);
        let iw = importance(&weak, point, 0.1);
        let is = importance(&strong, point, 0.1);
        assert!(is > iw);
        // Importance is exactly linear in power at fixed geometry.
        assert!((is - 10.0 * iw).abs() < 1e-4, "iw={iw} is={is}");
    }

    #[test]
    fn importance_falls_off_with_inverse_square_distance() {
        let light = LightBounds::point(Vec3::ZERO, 1.0);
        let near = importance(&light, Vec3::new(0.0, 0.0, 1.0), 0.01);
        let far = importance(&light, Vec3::new(0.0, 0.0, 2.0), 0.01);
        // Doubling the distance quarters the importance (point light: cone = 1).
        assert!((near - 4.0 * far).abs() < 1e-4, "near={near} far={far}");
    }

    #[test]
    fn importance_min_distance_clamp_prevents_blowup() {
        let light = LightBounds::point(Vec3::ZERO, 1.0);
        // Shading point exactly at the centroid: clamp keeps it finite.
        let at_centre = importance(&light, Vec3::ZERO, 0.5);
        assert!(at_centre.is_finite());
        // With min_dist = 0.5 the floor distance^2 is 0.25 -> importance = 4.
        assert!((at_centre - 4.0).abs() < 1e-3, "at_centre={at_centre}");
    }

    #[test]
    fn importance_respects_spot_orientation() {
        // Spot aiming along +z with a narrow cone.
        let spot = LightBounds::spot(Vec3::ZERO, Vec3::Z, 0.1, 1.0);
        let front = importance(&spot, Vec3::new(0.0, 0.0, 2.0), 0.1);
        let behind = importance(&spot, Vec3::new(0.0, 0.0, -2.0), 0.1);
        assert!(front > 0.0);
        // A point directly behind the spot is well outside the emission cone.
        assert_eq!(behind, 0.0);
    }

    #[test]
    fn importance_zero_power_is_zero() {
        let dark = LightBounds::point(Vec3::ZERO, 0.0);
        assert_eq!(importance(&dark, Vec3::new(0.0, 0.0, 3.0), 0.1), 0.0);
    }

    #[test]
    fn all_zero_power_falls_back_to_uniform() {
        // A balanced tree of four zero-power lights must sample uniformly.
        let lights = four_point_lights([0.0, 0.0, 0.0, 0.0]);
        let tree = LightTree::build(&lights).unwrap();
        let pmf = tree.leaf_pmf(Vec3::new(0.5, 1.0, 0.0), 0.1);
        for &(_, p) in &pmf {
            assert!((p - 0.25).abs() < 1e-6, "p={p}");
        }
    }

    #[test]
    fn results_are_deterministic() {
        let lights = four_point_lights([1.0, 2.0, 3.0, 4.0]);
        let a = LightTree::build(&lights).unwrap();
        let b = LightTree::build(&lights).unwrap();
        assert_eq!(a, b);
        let point = Vec3::new(0.3, 4.0, -1.0);
        assert_eq!(a.sample(point, 0.1, 0.37), b.sample(point, 0.1, 0.37));
        assert_eq!(a.leaf_pmf(point, 0.1), b.leaf_pmf(point, 0.1));
    }

    #[test]
    fn cone_union_covers_both_axes() {
        let a = LightCone::new(Vec3::X, 0.2, 0.1);
        let b = LightCone::new(Vec3::Y, 0.2, 0.1);
        let u = a.union(b);
        // The merged cone must contain both child axes within theta_o.
        let da = angle_between(u.axis, Vec3::X);
        let db = angle_between(u.axis, Vec3::Y);
        assert!(da <= u.theta_o + 1e-5, "da={da} theta_o={}", u.theta_o);
        assert!(db <= u.theta_o + 1e-5, "db={db} theta_o={}", u.theta_o);
    }

    #[test]
    fn cone_union_contained_child_is_noop_axis() {
        let big = LightCone::new(Vec3::Z, 1.0, 0.2);
        let small = LightCone::new(Vec3::new(0.05, 0.0, 1.0), 0.1, 0.1);
        let u = big.union(small);
        // `big` already encloses `small`, so the axis stays and theta_o is big's.
        assert_eq!(u.axis, big.axis);
        assert!((u.theta_o - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cone_union_opposite_axes_is_finite() {
        let a = LightCone::new(Vec3::Z, 0.3, 0.1);
        let b = LightCone::new(Vec3::NEG_Z, 0.3, 0.1);
        let u = a.union(b);
        assert!(u.axis.is_finite());
        assert!(u.theta_o.is_finite() && u.theta_o <= PI);
    }

    #[test]
    fn light_bounds_union_sums_power_and_grows_box() {
        let a = LightBounds::point(Vec3::new(-1.0, 0.0, 0.0), 2.0);
        let b = LightBounds::point(Vec3::new(1.0, 2.0, 0.0), 3.0);
        let u = a.union(b);
        assert!((u.power - 5.0).abs() < 1e-6);
        assert_eq!(u.aabb_min, Vec3::new(-1.0, 0.0, 0.0));
        assert_eq!(u.aabb_max, Vec3::new(1.0, 2.0, 0.0));
    }
}
