//! Projection-volume assembly for the `Decal` renderer (design §15): the
//! deterministic `CPU` reference that turns a particle-driven oriented box into
//! a surface projection, mapping arbitrary world points to decal `UV`s and a
//! combined edge fade the way `Unreal`'s deferred decals and `Niagara`'s Decal
//! renderer do at the algorithm level, without reusing any of their code.
//!
//! This module is deliberately *orthogonal* to the primitives in
//! [`super::renderers`] (`Sprite` / `Mesh` / `Ribbon` / `Beam`) and to the
//! per-instance transforms in [`super::mesh_renderer`]. Those build geometry
//! that *emits* fragments: a `Sprite` billboard quad, a `Ribbon`/`Beam` swept
//! strip, or a `Mesh` instance transform. A decal does the opposite — it does
//! not add geometry at all. Instead it defines an oriented bounding-box (`OBB`)
//! *projector* and asks, for a surface point already produced by the scene,
//! "where inside this box does the point land, and how strongly should the
//! decal paint it?". That is a receiver-space query, not a transform-assembly,
//! so it owns its own projection math ([`DecalProjector`]) rather than reusing
//! the sprite billboard basis or the mesh affine transform. To keep the
//! dependency graph a tree this module never imports [`super::renderers`]; the
//! link to `RendererKind::Decal` lives only in this doc comment.
//!
//! Determinism matches the sibling modules: the only non-arithmetic operation
//! is `sqrt` (reached through [`Vec3`]), and no transcendental function is ever
//! called. Every angle enters as a numeric cosine (obtained by the caller from
//! a `dot` product), never as radians, so the `CPU` reference stays bit-
//! reproducible against a future `GPU` decal-projection kernel.

use super::sort_cull::Aabb;
use super::Vec3;

/// Absolute tolerance for `f32` comparisons in this module.
///
/// `f32` equality is never tested with `==`/`!=`; code and tests compare
/// against this tolerance instead. It also guards the per-axis half-extent and
/// fade-range divisions so a degenerate (zero-thickness) projector or fade band
/// falls back to a defined value rather than producing `NaN`/infinity.
pub const EPS: f32 = 1e-6;

/// Deterministic fallback projection direction used when the requested forward
/// axis is degenerate (near zero length). Choosing `+Z` keeps the constructed
/// basis reproducible instead of leaving it undefined.
const FALLBACK_FORWARD: Vec3 = Vec3::new(0.0, 0.0, 1.0);

/// An oriented bounding-box (`OBB`) decal projector.
///
/// The box is described by a world-space `center`, an orthonormal orientation
/// basis (`right`, `up`, `forward`, all unit vectors), and per-axis
/// `half_extents`. `forward` is the projection direction; a receiver point is
/// projected onto the three basis axes and normalized by the half-extents into
/// the local cube `[-1, 1]^3`, exactly as a deferred-decal projection matrix
/// does. Construct one with [`DecalProjector::from_forward_up`] so the basis is
/// always orthonormal and degenerate-safe.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DecalProjector {
    /// World-space box center.
    pub center: Vec3,
    /// Unit basis axis mapped to local `x` (decal `UV` `u`).
    pub right: Vec3,
    /// Unit basis axis mapped to local `y` (decal `UV` `v`).
    pub up: Vec3,
    /// Unit projection direction, mapped to local `z` (projection depth).
    pub forward: Vec3,
    /// Positive half-sizes along `right`, `up`, and `forward`.
    pub half_extents: Vec3,
}

/// Edge-fade parameters for a decal projection.
///
/// The angle band is expressed as two cosines (never angles): `cos_full` is the
/// alignment at which the decal is fully opaque and `cos_threshold` the
/// alignment at which it has faded out. The depth band fades the decal along
/// the projection axis: `depth_fade_start` (in local `z`, `[-1, 1]`) is where
/// full opacity ends and `depth_fade_end` is where it reaches zero.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DecalFadeParams {
    /// Local `z` at which the depth fade begins (full opacity at and before).
    pub depth_fade_start: f32,
    /// Local `z` at which the depth fade completes (zero opacity at and after).
    pub depth_fade_end: f32,
    /// Alignment cosine at which the angle fade has fully faded out.
    pub cos_threshold: f32,
    /// Alignment cosine at which the angle fade reaches full opacity.
    pub cos_full: f32,
}

/// The result of projecting a receiver point through a [`DecalProjector`].
///
/// `uv` is the sampled decal texture coordinate in `[0, 1]^2` and `fade` is the
/// combined (angle times depth) opacity multiplier in `[0, 1]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DecalSample {
    /// Decal texture coordinate in `[0, 1]^2`.
    pub uv: [f32; 2],
    /// Combined edge-fade opacity multiplier in `[0, 1]`.
    pub fade: f32,
}

impl DecalProjector {
    /// Builds a projector from a `center`, a projection `forward` direction, an
    /// `up_reference` hint, and per-axis `half_extents`.
    ///
    /// The orthonormal basis is derived with cross products
    /// (`right = up_reference x forward`, `up = forward x right`) so
    /// `right x up = forward` (right-handed). The construction is degenerate-
    /// safe: a near-zero `forward` falls back to [`FALLBACK_FORWARD`], and an
    /// `up_reference` parallel to `forward` falls back to a deterministic world
    /// axis chosen to be the least aligned with `forward`, guaranteeing a valid
    /// orthonormal basis without ever dividing by zero.
    #[must_use]
    pub fn from_forward_up(
        center: Vec3,
        forward: Vec3,
        up_reference: Vec3,
        half_extents: Vec3,
    ) -> Self {
        let forward_unit = {
            let n = forward.normalize_or_zero();
            // A normalized non-zero vector has unit length; a degenerate input
            // normalizes to zero, which we detect by its (near-zero) length.
            if n.length_squared() <= EPS {
                FALLBACK_FORWARD
            } else {
                n
            }
        };

        let right = {
            let candidate = up_reference.cross(forward_unit).normalize_or_zero();
            if candidate.length_squared() <= EPS {
                // `up_reference` is parallel to `forward`: pick the world axis
                // least aligned with `forward` so the cross product is stable.
                fallback_perpendicular(forward_unit)
                    .cross(forward_unit)
                    .normalize_or_zero()
            } else {
                candidate
            }
        };

        let up = forward_unit.cross(right).normalize_or_zero();

        Self {
            center,
            right,
            up,
            forward: forward_unit,
            half_extents,
        }
    }
}

/// Picks a world axis that is not parallel to `forward`, choosing the axis with
/// the smallest absolute component so the follow-up cross product is well
/// conditioned. Used only when the caller's `up_reference` is unusable.
#[must_use]
fn fallback_perpendicular(forward: Vec3) -> Vec3 {
    let ax = forward.x.abs();
    let ay = forward.y.abs();
    let az = forward.z.abs();
    if ax <= ay && ax <= az {
        Vec3::new(1.0, 0.0, 0.0)
    } else if ay <= az {
        Vec3::new(0.0, 1.0, 0.0)
    } else {
        Vec3::new(0.0, 0.0, 1.0)
    }
}

/// Divides `numerator` by `denominator`, falling back to `0.0` when the
/// denominator is within [`EPS`] of zero.
///
/// [`Vec3`] has no component-wise division, so the per-axis normalization is
/// done in scalar `f32` here; this guard keeps a zero-thickness projector or
/// fade band from producing `NaN`/infinity.
#[must_use]
fn safe_ratio(numerator: f32, denominator: f32) -> f32 {
    if denominator.abs() <= EPS {
        0.0
    } else {
        numerator / denominator
    }
}

/// Returns `true` when `value` lies within the local unit interval `[-1, 1]`.
#[must_use]
fn within_unit(value: f32) -> bool {
    (-1.0..=1.0).contains(&value)
}

/// Transforms a world point into the projector's normalized local coordinates.
///
/// The point is projected onto the `right`/`up`/`forward` axes (via `dot`) and
/// each component is normalized by the corresponding half-extent into `[-1, 1]`
/// for a point inside the box. A half-extent within [`EPS`] of zero yields a
/// local `0.0` on that axis (see [`safe_ratio`]) instead of dividing by zero.
#[must_use]
pub fn world_to_decal_local(projector: &DecalProjector, world: Vec3) -> Vec3 {
    let delta = world.sub(projector.center);
    Vec3::new(
        safe_ratio(delta.dot(projector.right), projector.half_extents.x),
        safe_ratio(delta.dot(projector.up), projector.half_extents.y),
        safe_ratio(delta.dot(projector.forward), projector.half_extents.z),
    )
}

/// Returns `true` when `world` lies inside the projection box.
///
/// A point is inside when all three normalized local coordinates fall within
/// the unit cube `[-1, 1]^3`.
#[must_use]
pub fn contains(projector: &DecalProjector, world: Vec3) -> bool {
    let local = world_to_decal_local(projector, world);
    within_unit(local.x) && within_unit(local.y) && within_unit(local.z)
}

/// Maps `world` to a decal texture coordinate in `[0, 1]^2`.
///
/// The local `x`/`y` are remapped from `[-1, 1]` to `[0, 1]` via
/// `(local + 1) * 0.5`, so the box center is `UV` `(0.5, 0.5)`. Returns `None`
/// when the point is outside the box on *any* axis (the projection clips it).
#[must_use]
pub fn decal_uv(projector: &DecalProjector, world: Vec3) -> Option<[f32; 2]> {
    let local = world_to_decal_local(projector, world);
    if !within_unit(local.x) || !within_unit(local.y) || !within_unit(local.z) {
        return None;
    }
    Some([(local.x + 1.0) * 0.5, (local.y + 1.0) * 0.5])
}

/// Angle-based edge fade for a decal on a surface with normal `surface_normal`.
///
/// The alignment metric is `-(surface_normal . projector_forward)`, the cosine
/// of the angle between the surface normal and the direction *toward* the
/// projector: a surface squarely facing the projector scores `1.0`. Both inputs
/// are normalized first. A back-facing surface (alignment at or below zero)
/// fades to `0.0`. Otherwise the alignment is remapped linearly from
/// `cos_threshold` (fade out, `0.0`) to `cos_full` (full opacity, `1.0`) and
/// clamped. A degenerate band (`cos_full` within [`EPS`] of `cos_threshold`)
/// becomes a hard step at `cos_full`. Angles enter only as cosines; no
/// trigonometry is performed.
#[must_use]
pub fn angle_fade(
    surface_normal: Vec3,
    projector_forward: Vec3,
    cos_threshold: f32,
    cos_full: f32,
) -> f32 {
    let normal = surface_normal.normalize_or_zero();
    let forward = projector_forward.normalize_or_zero();
    let alignment = -normal.dot(forward);
    if alignment <= 0.0 {
        return 0.0;
    }
    let span = cos_full - cos_threshold;
    if span.abs() <= EPS {
        return if alignment >= cos_full { 1.0 } else { 0.0 };
    }
    ((alignment - cos_threshold) / span).clamp(0.0, 1.0)
}

/// Depth-based (along-projection) edge fade.
///
/// `local_z` is the projector-local depth coordinate in `[-1, 1]` (see
/// [`world_to_decal_local`]). Opacity is full at and before `fade_start` and
/// zero at and after `fade_end`, interpolating linearly in between so the decal
/// softens as it approaches the far face of the projection volume. A degenerate
/// band (`fade_end` within [`EPS`] of `fade_start`) becomes a hard step at
/// `fade_end`.
#[must_use]
pub fn depth_fade(local_z: f32, fade_start: f32, fade_end: f32) -> f32 {
    let span = fade_end - fade_start;
    if span.abs() <= EPS {
        return if local_z >= fade_end { 0.0 } else { 1.0 };
    }
    ((fade_end - local_z) / span).clamp(0.0, 1.0)
}

/// Projects a receiver point through the decal, returning its `UV` and combined
/// fade, or `None` when the point is clipped by the projection box.
///
/// This is the one-call composition used by the renderer: it computes the local
/// coordinates once, clips against the unit cube, then multiplies the
/// [`angle_fade`] (using `surface_normal` and the projector's `forward`) by the
/// [`depth_fade`] (using the local `z`) into the final [`DecalSample::fade`].
#[must_use]
pub fn projected_sample(
    projector: &DecalProjector,
    world: Vec3,
    surface_normal: Vec3,
    params: DecalFadeParams,
) -> Option<DecalSample> {
    let local = world_to_decal_local(projector, world);
    if !within_unit(local.x) || !within_unit(local.y) || !within_unit(local.z) {
        return None;
    }
    let uv = [(local.x + 1.0) * 0.5, (local.y + 1.0) * 0.5];
    let angle = angle_fade(
        surface_normal,
        projector.forward,
        params.cos_threshold,
        params.cos_full,
    );
    let depth = depth_fade(local.z, params.depth_fade_start, params.depth_fade_end);
    Some(DecalSample {
        uv,
        fade: angle * depth,
    })
}

/// Computes the world-space axis-aligned bounds ([`Aabb`]) of the projection
/// box for broad-phase culling.
///
/// The eight `OBB` corners (`center +/- right*hx +/- up*hy +/- forward*hz`) are
/// transformed and reduced to their min/max, giving the tight `Aabb` that
/// encloses the oriented box. This feeds the same culling path the particle
/// bounds reduction in [`super::sort_cull`] produces.
#[must_use]
pub fn world_aabb(projector: &DecalProjector) -> Aabb {
    let rx = projector.right.scale(projector.half_extents.x);
    let uy = projector.up.scale(projector.half_extents.y);
    let fz = projector.forward.scale(projector.half_extents.z);
    let mut bounds = Aabb::empty();
    // Enumerate the eight sign combinations of the three half-extent vectors.
    let signs = [-1.0f32, 1.0f32];
    for &sx in &signs {
        for &sy in &signs {
            for &sz in &signs {
                let corner = projector
                    .center
                    .add(rx.scale(sx))
                    .add(uy.scale(sy))
                    .add(fz.scale(sz));
                bounds = bounds.expand(corner);
            }
        }
    }
    bounds
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Absolute tolerance for the test assertions.
    const TOL: f32 = 1e-5;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() <= TOL
    }

    fn axis_aligned() -> DecalProjector {
        DecalProjector::from_forward_up(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(1.0, 2.0, 3.0),
        )
    }

    fn default_fade() -> DecalFadeParams {
        DecalFadeParams {
            depth_fade_start: -1.0,
            depth_fade_end: 1.0,
            cos_threshold: 0.0,
            cos_full: 1.0,
        }
    }

    #[test]
    fn constructor_builds_orthonormal_right_handed_basis() {
        let p = axis_aligned();
        assert!(approx(p.right.length(), 1.0));
        assert!(approx(p.up.length(), 1.0));
        assert!(approx(p.forward.length(), 1.0));
        assert!(approx(p.right.dot(p.up), 0.0));
        assert!(approx(p.right.dot(p.forward), 0.0));
        assert!(approx(p.up.dot(p.forward), 0.0));
        // right x up == forward (right-handed).
        let cross = p.right.cross(p.up);
        assert!(approx(cross.x, p.forward.x));
        assert!(approx(cross.y, p.forward.y));
        assert!(approx(cross.z, p.forward.z));
    }

    #[test]
    fn contains_flags_inside_and_outside() {
        let p = axis_aligned();
        assert!(contains(&p, Vec3::new(0.0, 0.0, 0.0)));
        assert!(contains(&p, Vec3::new(1.0, 2.0, 3.0)));
        assert!(contains(&p, Vec3::new(-1.0, -2.0, -3.0)));
        // Just past a face on each axis.
        assert!(!contains(&p, Vec3::new(1.5, 0.0, 0.0)));
        assert!(!contains(&p, Vec3::new(0.0, 2.5, 0.0)));
        assert!(!contains(&p, Vec3::new(0.0, 0.0, 3.5)));
    }

    #[test]
    fn uv_center_is_half_and_corners_map_to_unit_square() {
        let p = axis_aligned();
        let center = decal_uv(&p, Vec3::new(0.0, 0.0, 0.0)).expect("center inside");
        assert!(approx(center[0], 0.5));
        assert!(approx(center[1], 0.5));
        // +right, +up corner (on the face) maps to UV (1, 1).
        let hi = decal_uv(&p, Vec3::new(1.0, 2.0, 0.0)).expect("corner inside");
        assert!(approx(hi[0], 1.0));
        assert!(approx(hi[1], 1.0));
        // -right, -up corner maps to UV (0, 0).
        let lo = decal_uv(&p, Vec3::new(-1.0, -2.0, 0.0)).expect("corner inside");
        assert!(approx(lo[0], 0.0));
        assert!(approx(lo[1], 0.0));
    }

    #[test]
    fn uv_outside_box_is_clipped_to_none() {
        let p = axis_aligned();
        assert!(decal_uv(&p, Vec3::new(2.0, 0.0, 0.0)).is_none());
        assert!(decal_uv(&p, Vec3::new(0.0, 0.0, 10.0)).is_none());
    }

    #[test]
    fn angle_fade_front_full_side_and_back_zero() {
        let forward = Vec3::new(0.0, 0.0, 1.0);
        // Surface squarely facing the projector: normal opposite forward.
        let front = angle_fade(Vec3::new(0.0, 0.0, -1.0), forward, 0.0, 1.0);
        assert!(approx(front, 1.0));
        // Grazing surface: normal perpendicular to forward -> zero.
        let side = angle_fade(Vec3::new(1.0, 0.0, 0.0), forward, 0.0, 1.0);
        assert!(approx(side, 0.0));
        // Back-facing surface: normal along forward -> zero.
        let back = angle_fade(Vec3::new(0.0, 0.0, 1.0), forward, 0.0, 1.0);
        assert!(approx(back, 0.0));
    }

    #[test]
    fn angle_fade_interpolates_within_band() {
        let forward = Vec3::new(0.0, 0.0, 1.0);
        // Normal giving alignment cos = 0.5 (already unit length).
        let normal = Vec3::new(0.0, 0.8660254, -0.5);
        let mid = angle_fade(normal, forward, 0.0, 1.0);
        assert!(approx(mid, 0.5));
        // Degenerate band collapses to a hard step at cos_full.
        let step_on = angle_fade(Vec3::new(0.0, 0.0, -1.0), forward, 0.5, 0.5);
        assert!(approx(step_on, 1.0));
        let step_off = angle_fade(normal, forward, 0.9, 0.9);
        assert!(approx(step_off, 0.0));
    }

    #[test]
    fn depth_fade_is_full_near_zero_far_and_linear_between() {
        assert!(approx(depth_fade(-1.0, -1.0, 1.0), 1.0));
        assert!(approx(depth_fade(1.0, -1.0, 1.0), 0.0));
        assert!(approx(depth_fade(0.0, -1.0, 1.0), 0.5));
        // Clamps outside the band.
        assert!(approx(depth_fade(-2.0, -1.0, 1.0), 1.0));
        assert!(approx(depth_fade(2.0, -1.0, 1.0), 0.0));
        // Degenerate band collapses to a hard step at fade_end.
        assert!(approx(depth_fade(0.4, 0.5, 0.5), 1.0));
        assert!(approx(depth_fade(0.6, 0.5, 0.5), 0.0));
    }

    #[test]
    fn projected_sample_combines_uv_and_fade() {
        let p = axis_aligned();
        let params = default_fade();
        // Center point, facing surface: uv 0.5,0.5; depth fade at z=0 is 0.5,
        // angle fade full -> combined 0.5.
        let sample = projected_sample(
            &p,
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, -1.0),
            params,
        )
        .expect("center inside");
        assert!(approx(sample.uv[0], 0.5));
        assert!(approx(sample.uv[1], 0.5));
        assert!(approx(sample.fade, 0.5));
        // Outside the box clips to None.
        assert!(projected_sample(
            &p,
            Vec3::new(5.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, -1.0),
            params
        )
        .is_none());
    }

    #[test]
    fn world_aabb_encloses_axis_aligned_box() {
        let p = axis_aligned();
        let aabb = world_aabb(&p);
        assert!(approx(aabb.min.x, -1.0));
        assert!(approx(aabb.min.y, -2.0));
        assert!(approx(aabb.min.z, -3.0));
        assert!(approx(aabb.max.x, 1.0));
        assert!(approx(aabb.max.y, 2.0));
        assert!(approx(aabb.max.z, 3.0));
    }

    #[test]
    fn degenerate_half_extent_yields_zero_local_not_nan() {
        // Zero thickness along forward (z half-extent 0).
        let p = DecalProjector::from_forward_up(
            Vec3::ZERO,
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
        );
        let local = world_to_decal_local(&p, Vec3::new(0.0, 0.0, 5.0));
        assert!(local.z.abs() <= TOL);
        assert!(!local.z.is_nan());
    }

    #[test]
    fn zero_forward_falls_back_to_valid_basis() {
        let p = DecalProjector::from_forward_up(
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(1.0, 1.0, 1.0),
        );
        assert!(approx(p.forward.length(), 1.0));
        assert!(approx(p.right.length(), 1.0));
        assert!(approx(p.up.length(), 1.0));
        assert!(approx(p.right.dot(p.up), 0.0));
        assert!(approx(p.up.dot(p.forward), 0.0));
        assert!(approx(p.right.dot(p.forward), 0.0));
    }

    #[test]
    fn parallel_up_reference_still_builds_orthonormal_basis() {
        // up_reference parallel to forward triggers the perpendicular fallback.
        let p = DecalProjector::from_forward_up(
            Vec3::ZERO,
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(1.0, 1.0, 1.0),
        );
        assert!(approx(p.right.length(), 1.0));
        assert!(approx(p.up.length(), 1.0));
        assert!(approx(p.right.dot(p.up), 0.0));
        assert!(approx(p.right.dot(p.forward), 0.0));
        assert!(approx(p.up.dot(p.forward), 0.0));
    }
}
