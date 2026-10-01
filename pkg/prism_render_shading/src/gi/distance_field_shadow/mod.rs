//! Distance-field soft shadow CPU golden reference.
//!
//! This module is the backend-neutral, GPU-free reference for *distance-field
//! soft shadows*: Inigo Quilez's technique of estimating a light's penumbra
//! directly from a signed-distance field by marching a shadow ray and tracking
//! how closely it grazes the scene.  It is deliberately distinct from its two
//! GI siblings:
//!
//! * [`crate::gi::global_sdf`] merges many objects into one world-space brick
//!   grid and cone/sphere-traces it for ambient occlusion and GI gather;
//! * [`crate::gi::scene`] bakes per-mesh distance fields and derives
//!   distance-field ambient occlusion.
//!
//! Here the focus is purely the *shadow ray march*: given a scene signed
//! distance function `d(p)`, march from a receiver towards a light and reduce
//! `res = min(res, k·h/t)` (the ratio of the nearest surface distance `h` to
//! the travelled distance `t`, scaled by the light sharpness `k`) to a soft
//! visibility factor in `[0, 1]`.
//!
//! # Conventions
//! * [`primitives`] — exact analytic signed distances for sphere, box, rounded
//!   box, capsule, plane and torus (Quilez's public collection), plus the CSG
//!   operators [`op_union`]/[`op_subtract`]/[`op_intersect`] and the polynomial
//!   smooth union [`op_smooth_union`].
//! * [`march`] — the soft-shadow estimators: the original
//!   [`soft_shadow`] (`min(res, k·h/t)`) and the banding-free
//!   [`soft_shadow_improved`] (Quilez's 2010 refinement).
//! * [`trace`] — the hard-shadow / visibility companions
//!   ([`hard_shadow`], [`nearest_hit`]) and the light-size-to-sharpness
//!   conversion [`soft_shadow_k`].
//! * [`evaluate_soft_shadow`] stitches these together: it builds a scene SDF
//!   closure over a slice of placed [`Primitive`]s and marches a shadow ray
//!   from a receiver towards a light, using [`SdfShadowParams`] for the march
//!   bounds and estimator choice.
//! * Everything is a deterministic pure function — no RNG, I/O, GPU, or
//!   `unsafe` — with defensive clamps so no path yields `NaN`/`inf`.
//!
//! # References
//! * Inigo Quilez, "soft shadows in raymarched SDFs" (2010),
//!   <https://iquilezles.org/articles/rmshadows/>.
//! * Inigo Quilez, "distance functions",
//!   <https://iquilezles.org/articles/distfunctions/>.

pub mod march;
pub mod primitives;
pub mod trace;

pub use march::{soft_shadow, soft_shadow_improved};
pub use primitives::{
    box_exact, capsule, op_intersect, op_smooth_union, op_subtract, op_union, plane, rounded_box,
    sphere, torus,
};
pub use trace::{hard_shadow, nearest_hit, soft_shadow_k};

use bevy_math::Vec3;

/// A placed scene primitive: one of the analytic [`primitives`] shapes carrying
/// its own placement parameters, evaluated in world space.
///
/// Each variant's distance is computed by translating the query point into the
/// shape's local frame and calling the matching [`primitives`] function.  All
/// radii/extents inherit those functions' non-negative clamps, so a degenerate
/// variant collapses rather than inverting its sign.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Primitive {
    /// A sphere of `radius` centred at `center`.
    Sphere {
        /// World-space centre.
        center: Vec3,
        /// Sphere radius; clamped non-negative.
        radius: f32,
    },
    /// An axis-aligned box of `half_extents` centred at `center`.
    Box {
        /// World-space centre.
        center: Vec3,
        /// Per-axis half extents; clamped non-negative.
        half_extents: Vec3,
    },
    /// A rounded box: a box of `half_extents` with edges rounded by `radius`.
    RoundedBox {
        /// World-space centre.
        center: Vec3,
        /// Per-axis half extents; clamped non-negative.
        half_extents: Vec3,
        /// Edge-round radius; clamped non-negative.
        radius: f32,
    },
    /// A capsule (segment `a`–`b` inflated by `radius`), in world space.
    Capsule {
        /// First segment endpoint.
        a: Vec3,
        /// Second segment endpoint.
        b: Vec3,
        /// Capsule radius; clamped non-negative.
        radius: f32,
    },
    /// An infinite plane with unit `normal` and signed `height` offset.
    Plane {
        /// Plane normal (renormalised; degenerate falls back to `+Y`).
        normal: Vec3,
        /// Signed offset from the origin along `normal`.
        height: f32,
    },
    /// An XZ-plane torus of `major`/`minor` radii centred at `center`.
    Torus {
        /// World-space centre.
        center: Vec3,
        /// Major (ring) radius; clamped non-negative.
        major: f32,
        /// Minor (tube) radius; clamped non-negative.
        minor: f32,
    },
}

impl Primitive {
    /// Signed distance from `point` (world space) to this primitive's surface.
    #[inline]
    #[must_use]
    pub fn distance(&self, point: Vec3) -> f32 {
        match *self {
            Primitive::Sphere { center, radius } => sphere(point - center, radius),
            Primitive::Box {
                center,
                half_extents,
            } => box_exact(point - center, half_extents),
            Primitive::RoundedBox {
                center,
                half_extents,
                radius,
            } => rounded_box(point - center, half_extents, radius),
            Primitive::Capsule { a, b, radius } => capsule(point, a, b, radius),
            Primitive::Plane { normal, height } => plane(point, normal, height),
            Primitive::Torus {
                center,
                major,
                minor,
            } => torus(point - center, major, minor),
        }
    }
}

/// Union signed distance of a slice of placed [`Primitive`]s at `point`.
///
/// The scene is the CSG union of its primitives, so the field is the per-point
/// minimum of their distances.  An empty slice reports [`EMPTY_FAR`] (empty
/// space everywhere) so a marcher simply clears to `t_max`.
#[inline]
#[must_use]
pub fn scene_distance(primitives: &[Primitive], point: Vec3) -> f32 {
    let mut d = EMPTY_FAR;
    for prim in primitives {
        d = d.min(prim.distance(point));
    }
    d
}

/// Distance reported for an empty scene: a large finite value standing in for
/// "no surface anywhere".
pub const EMPTY_FAR: f32 = 1.0e9;

/// Parameters controlling the high-level [`evaluate_soft_shadow`] march.
///
/// Defaults are tuned for a unit-scale scene: a small start bias to escape the
/// receiver surface, a generous ray length, and the banding-free estimator.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdfShadowParams {
    /// Start offset along the shadow ray, biasing past the receiver surface to
    /// avoid self-shadowing acne.  Sanitised to a tiny positive value.
    pub t_min: f32,
    /// Maximum shadow-ray length (e.g. the distance to the light).  Forced to
    /// exceed `t_min`.
    pub t_max: f32,
    /// Use the banding-free [`soft_shadow_improved`] estimator when `true`,
    /// otherwise the original [`soft_shadow`].
    pub improved: bool,
}

impl Default for SdfShadowParams {
    #[inline]
    fn default() -> Self {
        Self {
            t_min: 0.02,
            t_max: 100.0,
            improved: true,
        }
    }
}

/// Evaluate the soft-shadow visibility of `receiver` towards a light.
///
/// Builds a scene SDF closure over the placed `scene` primitives and marches a
/// shadow ray along `light_dir` (the direction *towards* the light), using the
/// penumbra sharpness derived from `light_angular_radius` via
/// [`soft_shadow_k`].  Returns a visibility factor in `[0, 1]`: `1` fully lit,
/// `0` fully shadowed, intermediate values in the penumbra.  The estimator and
/// march bounds come from `params`.
#[inline]
#[must_use]
pub fn evaluate_soft_shadow(
    receiver: Vec3,
    light_dir: Vec3,
    light_angular_radius: f32,
    scene: &[Primitive],
    params: SdfShadowParams,
) -> f32 {
    let k = soft_shadow_k(light_angular_radius);
    let sdf = |p: Vec3| scene_distance(scene, p);
    if params.improved {
        soft_shadow_improved(receiver, light_dir, params.t_min, params.t_max, k, sdf)
    } else {
        soft_shadow(receiver, light_dir, params.t_min, params.t_max, k, sdf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn params_default_is_sensible() {
        let p = SdfShadowParams::default();
        assert!(p.t_min > 0.0 && p.t_max > p.t_min);
        assert!(p.improved);
    }

    #[test]
    fn empty_scene_is_fully_lit() {
        let s = evaluate_soft_shadow(
            Vec3::ZERO,
            Vec3::Y,
            0.05,
            &[],
            SdfShadowParams::default(),
        );
        assert!((s - 1.0).abs() < 1.0e-4, "empty scene should be lit, got {s}");
    }

    #[test]
    fn primitive_distance_matches_free_functions() {
        let p = Vec3::new(2.0, 0.0, 0.0);
        let prim = Primitive::Sphere {
            center: Vec3::ZERO,
            radius: 1.0,
        };
        assert!((prim.distance(p) - sphere(p, 1.0)).abs() < 1.0e-6);
    }

    #[test]
    fn scene_distance_is_union_minimum() {
        let scene = [
            Primitive::Sphere {
                center: Vec3::new(-2.0, 0.0, 0.0),
                radius: 1.0,
            },
            Primitive::Sphere {
                center: Vec3::new(2.0, 0.0, 0.0),
                radius: 1.0,
            },
        ];
        // Near the right sphere the union equals the right sphere's distance.
        let p = Vec3::new(2.0, 0.0, 0.0);
        let truth = scene[1].distance(p);
        assert!((scene_distance(&scene, p) - truth).abs() < 1.0e-6);
    }

    #[test]
    fn overhead_blocker_shadows_receiver() {
        let scene = [Primitive::Sphere {
            center: Vec3::new(0.0, 5.0, 0.0),
            radius: 1.0,
        }];
        let s = evaluate_soft_shadow(
            Vec3::ZERO,
            Vec3::Y,
            0.02,
            &scene,
            SdfShadowParams::default(),
        );
        assert_eq!(s, 0.0);
    }

    #[test]
    fn offset_blocker_gives_penumbra() {
        let scene = [Primitive::Sphere {
            center: Vec3::new(1.2, 5.0, 0.0),
            radius: 1.0,
        }];
        let s = evaluate_soft_shadow(
            Vec3::ZERO,
            Vec3::Y,
            0.08,
            &scene,
            SdfShadowParams::default(),
        );
        assert!(s > 0.0 && s < 1.0, "expected penumbra, got {s}");
    }

    #[test]
    fn plain_and_improved_both_valid() {
        let scene = [Primitive::Sphere {
            center: Vec3::new(1.2, 5.0, 0.0),
            radius: 1.0,
        }];
        let mut params = SdfShadowParams::default();
        params.improved = false;
        let plain = evaluate_soft_shadow(Vec3::ZERO, Vec3::Y, 0.08, &scene, params);
        params.improved = true;
        let improved = evaluate_soft_shadow(Vec3::ZERO, Vec3::Y, 0.08, &scene, params);
        assert!((0.0..=1.0).contains(&plain));
        assert!((0.0..=1.0).contains(&improved));
    }

    #[test]
    fn mixed_scene_distance_is_finite() {
        let scene = [
            Primitive::Box {
                center: Vec3::new(0.0, 2.0, 0.0),
                half_extents: Vec3::splat(0.5),
            },
            Primitive::Capsule {
                a: Vec3::new(-1.0, 1.0, 0.0),
                b: Vec3::new(1.0, 1.0, 0.0),
                radius: 0.3,
            },
            Primitive::Plane {
                normal: Vec3::Y,
                height: 0.0,
            },
            Primitive::Torus {
                center: Vec3::new(0.0, 3.0, 0.0),
                major: 1.0,
                minor: 0.25,
            },
            Primitive::RoundedBox {
                center: Vec3::new(2.0, 2.0, 0.0),
                half_extents: Vec3::splat(0.4),
                radius: 0.1,
            },
        ];
        for p in [Vec3::ZERO, Vec3::new(0.0, 2.0, 0.0), Vec3::new(5.0, 5.0, 5.0)] {
            assert!(scene_distance(&scene, p).is_finite());
        }
    }
}
