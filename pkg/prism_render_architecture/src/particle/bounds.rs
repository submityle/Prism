//! Parallel particle `AABB` bounds reduction contract (design §12, §13).
//!
//! Frame culling and indirect-draw setup both need a single tight axis-aligned
//! bounding box (`AABB`) over every live particle. Computing it naively is a
//! serial `min` / `max` scan; on the `GPU` it is a two-level tree reduction,
//! matching Unreal `Niagara`'s bounds pass and `Frostbite`'s classic
//! workgroup-then-global reduction. Each workgroup reduces its slice of the
//! Structure-of-Arrays (`SoA`) particle pool into one *partial* `AABB` written
//! to a scratch buffer; a second pass folds the partials down to the final box
//! that culling and indirect draw consume.
//!
//! This is the `CPU`-verifiable reference for that pipeline: it owns the box
//! algebra, the workgroup / partial-buffer sizing arithmetic (in `std430`
//! bytes, so the render graph can reserve `VRAM` before the `GPU` backend
//! exists), the reduction-step counter, and the serial fold that a future
//! `GPU` kernel must agree with bit for bit. Everything is pure `min` / `max`
//! and multiply-add: no transcendental function is ever called, and integer
//! sizing uses integer `div_ceil` rather than a floating-point `log`.

use crate::particle::gpu_layout::VEC4_STRIDE;

/// Absolute tolerance for `f32` equality comparisons in this module.
///
/// Floating-point `==` / `!=` are never used; call sites compare
/// `(a - b).abs() < CMP_EPS` instead so reductions stay robust to rounding.
#[cfg(test)]
const CMP_EPS: f32 = 1e-6;

/// A lightweight axis-aligned bounding box (`AABB`) in world space.
///
/// Holds `f32` fields, so it derives [`PartialEq`] but not [`Eq`]. An *empty*
/// box seeds `min` with `f32::MAX` and `max` with `f32::MIN`, so the first
/// [`Aabb::expand_point`] on any axis always wins and the empty box is the
/// identity element of [`Aabb::union`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Aabb {
    /// Per-axis minimum corner.
    pub min: [f32; 3],
    /// Per-axis maximum corner.
    pub max: [f32; 3],
}

impl Aabb {
    /// Returns the empty box: `min` at `+INF` semantics (`f32::MAX`) and `max`
    /// at `-INF` semantics (`f32::MIN`), the identity for [`Aabb::union`].
    #[must_use]
    pub fn empty() -> Self {
        Self {
            min: [f32::MAX; 3],
            max: [f32::MIN; 3],
        }
    }

    /// Returns `true` when the box holds no points, i.e. any axis has
    /// `min > max` (never uses `==`).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.min[0] > self.max[0] || self.min[1] > self.max[1] || self.min[2] > self.max[2]
    }

    /// Grows the box in place so it contains the point `p`.
    pub fn expand_point(&mut self, p: [f32; 3]) {
        self.min[0] = self.min[0].min(p[0]);
        self.min[1] = self.min[1].min(p[1]);
        self.min[2] = self.min[2].min(p[2]);
        self.max[0] = self.max[0].max(p[0]);
        self.max[1] = self.max[1].max(p[1]);
        self.max[2] = self.max[2].max(p[2]);
    }

    /// Returns the smallest box containing both `self` and `other`.
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

    /// Returns the box center (midpoint of `min` and `max`).
    #[must_use]
    pub fn center(&self) -> [f32; 3] {
        [
            (self.min[0] + self.max[0]) * 0.5,
            (self.min[1] + self.max[1]) * 0.5,
            (self.min[2] + self.max[2]) * 0.5,
        ]
    }

    /// Returns the per-axis half-extent (half the box size).
    #[must_use]
    pub fn half_extent(&self) -> [f32; 3] {
        [
            (self.max[0] - self.min[0]) * 0.5,
            (self.max[1] - self.min[1]) * 0.5,
            (self.max[2] - self.min[2]) * 0.5,
        ]
    }

    /// Returns the total surface area `2 * (dx*dy + dy*dz + dz*dx)`, using only
    /// multiply-add (no transcendental function). Handy as a `SAH` cost term.
    #[must_use]
    pub fn surface_area(&self) -> f32 {
        let dx = self.max[0] - self.min[0];
        let dy = self.max[1] - self.min[1];
        let dz = self.max[2] - self.min[2];
        2.0 * (dx * dy + dy * dz + dz * dx)
    }

    /// Returns the index (`0` = x, `1` = y, `2` = z) of the longest axis. Ties
    /// resolve toward the lower index; comparisons use `>=`, never `==`.
    #[must_use]
    pub fn longest_axis(&self) -> u8 {
        let dx = self.max[0] - self.min[0];
        let dy = self.max[1] - self.min[1];
        let dz = self.max[2] - self.min[2];
        if dx >= dy && dx >= dz {
            0
        } else if dy >= dz {
            1
        } else {
            2
        }
    }
}

/// Sizing plan for the two-level parallel `AABB` reduction (design §13).
///
/// Given a particle count and a workgroup size, it reports how many workgroups
/// (and therefore partial `AABB`s) the first pass produces, how much `std430`
/// scratch the partials need, whether a second pass is required, and how many
/// reduction dispatch steps the whole tree takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BoundsReduction {
    particle_count: u32,
    workgroup_size: u32,
}

impl BoundsReduction {
    /// Builds a plan, clamping `workgroup_size` to the valid `[1, 1024]` range
    /// (a `WebGPU` compute workgroup may hold at most 1024 invocations and must
    /// hold at least one).
    #[must_use]
    pub fn new(particle_count: u32, workgroup_size: u32) -> Self {
        Self {
            particle_count,
            workgroup_size: workgroup_size.clamp(1, 1024),
        }
    }

    /// Returns the number of workgroups the first pass dispatches
    /// (`ceil(particle_count / workgroup_size)`).
    #[must_use]
    pub fn workgroup_count(&self) -> u32 {
        self.particle_count.div_ceil(self.workgroup_size)
    }

    /// Returns the number of partial `AABB`s the first pass writes; each
    /// workgroup emits exactly one, so this equals [`Self::workgroup_count`].
    #[must_use]
    pub fn partial_count(&self) -> u32 {
        self.workgroup_count()
    }

    /// Returns the `std430` byte size of the partial-`AABB` scratch buffer.
    ///
    /// A tight `AABB` is six `f32`s (24 bytes), but `std430` aligns each stored
    /// box to a `vec4` boundary; two `vec4`s (2 * [`VEC4_STRIDE`] = 32 bytes)
    /// hold `min.xyz` and `max.xyz` with conservative padding, matching how the
    /// `GPU` kernel lays a partial box out in `VRAM`.
    #[must_use]
    pub fn partial_buffer_bytes(&self) -> u64 {
        let per_aabb = u64::try_from(2 * VEC4_STRIDE).unwrap_or(0);
        u64::from(self.partial_count()) * per_aabb
    }

    /// Returns `true` when more than one partial exists, so a second reduction
    /// pass is required to fold them into the final box.
    #[must_use]
    pub fn second_pass_needed(&self) -> bool {
        self.partial_count() > 1
    }

    /// Returns the total number of reduction dispatch steps: the first
    /// workgroup pass plus every second-level pass, each shrinking the partial
    /// count by `div_ceil(count, workgroup_size)` until a single box remains.
    ///
    /// Zero particles need no reduction and return `0`. The loop uses integer
    /// `div_ceil` and never a floating-point `log`.
    #[must_use]
    pub fn total_reduction_steps(&self) -> u32 {
        let mut count = self.workgroup_count();
        if count == 0 {
            return 0;
        }
        let mut steps = 1;
        while count > 1 {
            count = count.div_ceil(self.workgroup_size);
            steps += 1;
        }
        steps
    }
}

/// Folds a slice of partial `AABB`s into the final box via [`Aabb::union`].
///
/// An empty slice yields [`Aabb::empty`]; otherwise the fold seeds from the
/// empty box (the union identity), so the result is independent of order.
#[must_use]
pub fn reduce_partials(partials: &[Aabb]) -> Aabb {
    let mut out = Aabb::empty();
    for partial in partials {
        out = out.union(partial);
    }
    out
}

/// Reduces a slice of points into their bounding `AABB`.
///
/// An empty slice yields [`Aabb::empty`]. This is the serial reference the
/// parallel two-level reduction must agree with.
#[must_use]
pub fn reduce_points(points: &[[f32; 3]]) -> Aabb {
    let mut out = Aabb::empty();
    for &p in points {
        out.expand_point(p);
    }
    out
}

/// Conservatively expands a box by the distance a particle could travel this
/// frame (`max_speed * dt` on every axis), for motion-blur / swept culling.
///
/// Negative `max_speed` or `dt` clamp to `0.0`, so the box never shrinks. An
/// empty box stays empty (there is nothing to sweep).
#[must_use]
pub fn expand_by_velocity(bounds: &Aabb, max_speed: f32, dt: f32) -> Aabb {
    if bounds.is_empty() {
        return Aabb::empty();
    }
    let delta = max_speed.max(0.0) * dt.max(0.0);
    Aabb {
        min: [
            bounds.min[0] - delta,
            bounds.min[1] - delta,
            bounds.min[2] - delta,
        ],
        max: [
            bounds.max[0] + delta,
            bounds.max[1] + delta,
            bounds.max[2] + delta,
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < CMP_EPS
    }

    fn close3(a: [f32; 3], b: [f32; 3]) -> bool {
        close(a[0], b[0]) && close(a[1], b[1]) && close(a[2], b[2])
    }

    #[test]
    fn empty_box_reports_empty() {
        assert!(Aabb::empty().is_empty());
    }

    #[test]
    fn single_point_box_is_not_empty_and_degenerate() {
        let mut b = Aabb::empty();
        b.expand_point([1.0, 2.0, 3.0]);
        assert!(!b.is_empty());
        assert!(close3(b.min, [1.0, 2.0, 3.0]));
        assert!(close3(b.max, [1.0, 2.0, 3.0]));
    }

    #[test]
    fn multi_point_box_spans_extremes() {
        let mut b = Aabb::empty();
        b.expand_point([1.0, -2.0, 3.0]);
        b.expand_point([-4.0, 5.0, -6.0]);
        b.expand_point([0.0, 0.0, 0.0]);
        assert!(close3(b.min, [-4.0, -2.0, -6.0]));
        assert!(close3(b.max, [1.0, 5.0, 3.0]));
    }

    #[test]
    fn union_is_commutative() {
        let a = Aabb {
            min: [-1.0, -1.0, -1.0],
            max: [1.0, 2.0, 3.0],
        };
        let b = Aabb {
            min: [0.0, -5.0, 2.0],
            max: [4.0, 0.0, 8.0],
        };
        let ab = a.union(&b);
        let ba = b.union(&a);
        assert!(close3(ab.min, ba.min));
        assert!(close3(ab.max, ba.max));
        assert!(close3(ab.min, [-1.0, -5.0, -1.0]));
        assert!(close3(ab.max, [4.0, 2.0, 8.0]));
    }

    #[test]
    fn empty_is_union_identity() {
        let a = Aabb {
            min: [-1.0, -1.0, -1.0],
            max: [1.0, 2.0, 3.0],
        };
        let u = Aabb::empty().union(&a);
        assert!(close3(u.min, a.min));
        assert!(close3(u.max, a.max));
    }

    #[test]
    fn workgroup_size_clamped_low() {
        let r = BoundsReduction::new(100, 0);
        assert_eq!(r.workgroup_count(), 100);
    }

    #[test]
    fn workgroup_size_clamped_high() {
        let r = BoundsReduction::new(4096, 100_000);
        // Clamped to 1024, so 4096 / 1024 == 4 workgroups.
        assert_eq!(r.workgroup_count(), 4);
    }

    #[test]
    fn workgroup_count_zero_particles() {
        let r = BoundsReduction::new(0, 256);
        assert_eq!(r.workgroup_count(), 0);
        assert_eq!(r.partial_count(), 0);
    }

    #[test]
    fn workgroup_count_one_particle() {
        let r = BoundsReduction::new(1, 256);
        assert_eq!(r.workgroup_count(), 1);
    }

    #[test]
    fn workgroup_count_exact_divisor() {
        let r = BoundsReduction::new(1024, 256);
        assert_eq!(r.workgroup_count(), 4);
    }

    #[test]
    fn workgroup_count_remainder_rounds_up() {
        let r = BoundsReduction::new(1025, 256);
        assert_eq!(r.workgroup_count(), 5);
    }

    #[test]
    fn single_partial_needs_no_second_pass() {
        let r = BoundsReduction::new(200, 256);
        assert_eq!(r.partial_count(), 1);
        assert!(!r.second_pass_needed());
    }

    #[test]
    fn many_partials_need_second_pass() {
        let r = BoundsReduction::new(10_000, 256);
        assert!(r.partial_count() > 1);
        assert!(r.second_pass_needed());
    }

    #[test]
    fn partial_buffer_bytes_uses_two_vec4_stride() {
        let r = BoundsReduction::new(1025, 256);
        // 5 partials * 2 * 16 bytes == 160.
        assert_eq!(r.partial_buffer_bytes(), 160);
    }

    #[test]
    fn total_reduction_steps_single_level() {
        // 10000 / 256 = 40 partials, one second pass folds 40 -> 1.
        let r = BoundsReduction::new(10_000, 256);
        assert_eq!(r.total_reduction_steps(), 2);
    }

    #[test]
    fn total_reduction_steps_multi_level() {
        // wg=4: 1000 -> 250 -> 63 -> 16 -> 4 -> 1 (first pass + 4 second passes).
        let r = BoundsReduction::new(1000, 4);
        assert_eq!(r.total_reduction_steps(), 5);
    }

    #[test]
    fn total_reduction_steps_single_pass_when_one_workgroup() {
        let r = BoundsReduction::new(200, 256);
        assert_eq!(r.total_reduction_steps(), 1);
    }

    #[test]
    fn total_reduction_steps_zero_particles() {
        let r = BoundsReduction::new(0, 256);
        assert_eq!(r.total_reduction_steps(), 0);
    }

    #[test]
    fn reduce_partials_empty_is_empty() {
        let out = reduce_partials(&[]);
        assert!(out.is_empty());
    }

    #[test]
    fn reduce_partials_single_returns_it() {
        let a = Aabb {
            min: [-1.0, -2.0, -3.0],
            max: [1.0, 2.0, 3.0],
        };
        let out = reduce_partials(&[a]);
        assert!(close3(out.min, a.min));
        assert!(close3(out.max, a.max));
    }

    #[test]
    fn reduce_partials_multi_spans_all() {
        let parts = [
            Aabb {
                min: [-1.0, 0.0, 0.0],
                max: [0.0, 1.0, 1.0],
            },
            Aabb {
                min: [2.0, -3.0, 0.5],
                max: [4.0, 0.0, 6.0],
            },
        ];
        let out = reduce_partials(&parts);
        assert!(close3(out.min, [-1.0, -3.0, 0.0]));
        assert!(close3(out.max, [4.0, 1.0, 6.0]));
    }

    #[test]
    fn reduce_points_matches_manual_bounds() {
        let pts = [[1.0, 1.0, 1.0], [-2.0, 3.0, -1.0], [0.0, -4.0, 5.0]];
        let out = reduce_points(&pts);
        assert!(close3(out.min, [-2.0, -4.0, -1.0]));
        assert!(close3(out.max, [1.0, 3.0, 5.0]));
    }

    #[test]
    fn center_and_half_extent() {
        let b = Aabb {
            min: [-2.0, 0.0, 4.0],
            max: [2.0, 6.0, 10.0],
        };
        assert!(close3(b.center(), [0.0, 3.0, 7.0]));
        assert!(close3(b.half_extent(), [2.0, 3.0, 3.0]));
    }

    #[test]
    fn surface_area_of_unit_cube() {
        let b = Aabb {
            min: [0.0, 0.0, 0.0],
            max: [1.0, 1.0, 1.0],
        };
        assert!(close(b.surface_area(), 6.0));
    }

    #[test]
    fn longest_axis_picks_widest() {
        let b = Aabb {
            min: [0.0, 0.0, 0.0],
            max: [1.0, 5.0, 2.0],
        };
        assert_eq!(b.longest_axis(), 1);
        let c = Aabb {
            min: [0.0, 0.0, 0.0],
            max: [3.0, 1.0, 1.0],
        };
        assert_eq!(c.longest_axis(), 0);
        let d = Aabb {
            min: [0.0, 0.0, 0.0],
            max: [1.0, 1.0, 9.0],
        };
        assert_eq!(d.longest_axis(), 2);
    }

    #[test]
    fn expand_by_velocity_grows_symmetrically() {
        let b = Aabb {
            min: [0.0, 0.0, 0.0],
            max: [2.0, 2.0, 2.0],
        };
        let out = expand_by_velocity(&b, 3.0, 0.5);
        // delta = 3 * 0.5 = 1.5.
        assert!(close3(out.min, [-1.5, -1.5, -1.5]));
        assert!(close3(out.max, [3.5, 3.5, 3.5]));
    }

    #[test]
    fn expand_by_velocity_clamps_negative_inputs() {
        let b = Aabb {
            min: [0.0, 0.0, 0.0],
            max: [2.0, 2.0, 2.0],
        };
        let neg_speed = expand_by_velocity(&b, -10.0, 0.5);
        assert!(close3(neg_speed.min, b.min));
        assert!(close3(neg_speed.max, b.max));
        let neg_dt = expand_by_velocity(&b, 10.0, -0.5);
        assert!(close3(neg_dt.min, b.min));
        assert!(close3(neg_dt.max, b.max));
    }

    #[test]
    fn expand_by_velocity_keeps_empty_empty() {
        let out = expand_by_velocity(&Aabb::empty(), 5.0, 1.0);
        assert!(out.is_empty());
    }
}
