//! Artist-painted per-vertex simulation constraints (design §6.6).
//!
//! Following UE5 `Chaos` Cloth, an artist paints scalar weight maps over the
//! garment that locally steer the solve without editing the mesh. Each painted
//! vertex is anchored to a *skinned* reference pose (the animated character
//! surface) and its authored weights ([`super::asset::PaintedConstraint`])
//! decide how tightly it tracks that pose:
//!
//! * `anim_drive` — a pre-solve soft pull toward the animated target so the
//!   cloth *follows* animation while still simulating;
//! * `max_distance` — a post-solve hard cap on how far the simulated vertex may
//!   drift from the skinned pose (`0` pins it to the skin);
//! * `backstop` — a post-solve push-out off a cushion sphere sunk behind the
//!   vertex along its normal, keeping cloth from sinking into the body;
//! * `blend_weight` — a final blend of the simulated position toward the
//!   skinned pose (`1` fully simulated, `0` fully skinned).
//!
//! These are the调参 knobs behind "紧身处贴合稳、飘逸处自由" (design §11): a
//! high-`max_distance`, full-`blend_weight` region flows freely, while a
//! zero-`max_distance` region is welded to the character. Every pass is a
//! stateless array-in / array-out projection (design §9), so it maps onto a GPU
//! dispatch and is CPU golden-testable, and every out-of-range index is skipped
//! rather than panicking.

use super::asset::PaintedConstraint;
use super::{ClothParticle, Vec3, EPS_LEN_SQ};

/// The skinned reference pose one painted vertex tracks.
///
/// `position` is the animated character-surface point the weights steer toward
/// and `normal` is its outward surface normal, used to place the backstop
/// cushion. A zero-length normal disables only the backstop for that vertex.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SkinnedAnchor {
    /// Animated target position (the skinned character surface point).
    pub position: Vec3,
    /// Outward surface normal at the anchor; may be zero to skip the backstop.
    pub normal: Vec3,
}

impl SkinnedAnchor {
    /// Builds an anchor from a target position and outward normal.
    #[must_use]
    pub const fn new(position: Vec3, normal: Vec3) -> Self {
        Self { position, normal }
    }
}

/// Global tuning for the [`drive_toward_anim`] pre-solve pull.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AnimDriveParams {
    /// Master gain in `[0, 1]` multiplying every vertex's painted `anim_drive`;
    /// `0` disables the pull, `1` lets a fully painted vertex snap the whole
    /// remaining gap to its anchor in one frame.
    pub gain: f32,
    /// Master switch; when `false`, [`drive_toward_anim`] is a no-op.
    pub enabled: bool,
}

impl Default for AnimDriveParams {
    /// Disabled by default so a garment only follows animation once an artist
    /// opts in; a moderate gain when enabled.
    fn default() -> Self {
        Self {
            gain: 0.5,
            enabled: false,
        }
    }
}

impl AnimDriveParams {
    /// Returns a copy with `gain` clamped to `[0, 1]` and any `NaN` replaced by
    /// `0`, so a mis-authored value can never inject a `NaN` or an over-unity
    /// pull that would overshoot the anchor.
    #[must_use]
    pub fn sanitized(self) -> Self {
        let gain = if self.gain.is_nan() {
            0.0
        } else {
            self.gain.clamp(0.0, 1.0)
        };
        Self {
            gain,
            enabled: self.enabled,
        }
    }
}

/// Pulls each free vertex a fraction of the way toward its animated anchor
/// *before* the distance solve, so the solver then relaxes the stretch this
/// introduces and the cloth tracks animation without going rigid.
///
/// The per-vertex fraction is `anim_drive · gain`, clamped to `[0, 1]`, so the
/// move is bounded by the current gap and can never overshoot. Velocity is
/// advanced by the imposed displacement over `dt` (the standard position-based
/// velocity update) so momentum stays consistent with the drive. Pinned
/// vertices, a disabled pass, a non-positive `dt`, and out-of-range indices are
/// all skipped.
pub fn drive_toward_anim(
    particles: &mut [ClothParticle],
    anchors: &[SkinnedAnchor],
    painted: &[PaintedConstraint],
    params: AnimDriveParams,
    dt: f32,
) {
    let params = params.sanitized();
    if !params.enabled || params.gain <= 0.0 || dt <= 0.0 {
        return;
    }
    let inv_dt = 1.0 / dt;
    for (particle, (anchor, paint)) in particles.iter_mut().zip(anchors.iter().zip(painted.iter()))
    {
        if particle.is_pinned() {
            continue;
        }
        let paint = paint.clamped();
        let follow = (paint.anim_drive * params.gain).clamp(0.0, 1.0);
        if follow <= 0.0 {
            continue;
        }
        let delta = anchor.position.sub(particle.position).scale(follow);
        particle.position = particle.position.add(delta);
        particle.velocity = particle.velocity.add(delta.scale(inv_dt));
    }
}

/// Caps how far each vertex may drift from its skinned anchor, projecting any
/// over-limit vertex back onto the max-distance sphere.
///
/// A finite `max_distance` bounds the drift; `0` welds the vertex to the anchor
/// (a painted pin), and the default `+inf` leaves it free. This is a post-solve
/// positional clamp in the same style as self-collision and backstop passes, so
/// velocity is left untouched. Pinned vertices and out-of-range indices are
/// skipped.
pub fn clamp_max_distance(
    particles: &mut [ClothParticle],
    anchors: &[SkinnedAnchor],
    painted: &[PaintedConstraint],
) {
    for (particle, (anchor, paint)) in particles.iter_mut().zip(anchors.iter().zip(painted.iter()))
    {
        if particle.is_pinned() {
            continue;
        }
        let paint = paint.clamped();
        if !paint.max_distance.is_finite() {
            continue;
        }
        let drift = particle.position.sub(anchor.position);
        let dist_sq = drift.length_squared();
        let max_sq = paint.max_distance * paint.max_distance;
        if dist_sq <= max_sq {
            continue;
        }
        if dist_sq > EPS_LEN_SQ {
            let dist = dist_sq.sqrt();
            particle.position = anchor.position.add(drift.scale(paint.max_distance / dist));
        } else {
            particle.position = anchor.position;
        }
    }
}

/// Pushes each vertex out of the backstop cushion — a sphere of radius
/// `backstop` centred one radius behind the anchor along its normal, tangent to
/// the character surface — so cloth cannot sink into the body.
///
/// A vertex inside the sphere is projected radially onto its surface. A
/// non-positive `backstop` or a (near-)zero anchor normal disables the pass for
/// that vertex. This is a post-solve positional projection; velocity is left
/// untouched, and pinned vertices and out-of-range indices are skipped.
pub fn apply_painted_backstop(
    particles: &mut [ClothParticle],
    anchors: &[SkinnedAnchor],
    painted: &[PaintedConstraint],
) {
    for (particle, (anchor, paint)) in particles.iter_mut().zip(anchors.iter().zip(painted.iter()))
    {
        if particle.is_pinned() {
            continue;
        }
        let paint = paint.clamped();
        if paint.backstop <= 0.0 {
            continue;
        }
        let normal = anchor.normal.normalize_or_zero();
        if normal.length_squared() <= EPS_LEN_SQ {
            continue;
        }
        let center = anchor.position.sub(normal.scale(paint.backstop));
        let to = particle.position.sub(center);
        let dist_sq = to.length_squared();
        let radius = paint.backstop;
        if dist_sq >= radius * radius {
            continue;
        }
        if dist_sq > EPS_LEN_SQ {
            let dist = dist_sq.sqrt();
            particle.position = center.add(to.scale(radius / dist));
        } else {
            // Degenerate: the vertex sits at the cushion centre. Push it out
            // along the anchor normal to the nearest surface point (the anchor).
            particle.position = center.add(normal.scale(radius));
        }
    }
}

/// Blends each vertex's simulated position toward its skinned anchor by
/// `1 - blend_weight`, producing the final output pose.
///
/// `blend_weight == 1` keeps the fully simulated result and `0` snaps to the
/// skinned pose; intermediate values mix the two. Run this last, after every
/// solve and projection pass. Pinned vertices and out-of-range indices are
/// skipped; velocity is left untouched.
pub fn blend_to_skin(
    particles: &mut [ClothParticle],
    anchors: &[SkinnedAnchor],
    painted: &[PaintedConstraint],
) {
    for (particle, (anchor, paint)) in particles.iter_mut().zip(anchors.iter().zip(painted.iter()))
    {
        if particle.is_pinned() {
            continue;
        }
        let paint = paint.clamped();
        let offset = particle.position.sub(anchor.position);
        particle.position = anchor.position.add(offset.scale(paint.blend_weight));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cloth::ClothParticle;

    /// Builds an anchor grid line and matching free particles offset by `off`.
    fn anchor_and_particle(
        anchor_pos: Vec3,
        normal: Vec3,
        particle_pos: Vec3,
    ) -> (SkinnedAnchor, ClothParticle) {
        (
            SkinnedAnchor::new(anchor_pos, normal),
            ClothParticle {
                position: particle_pos,
                velocity: Vec3::ZERO,
                inverse_mass: 1.0,
            },
        )
    }

    #[test]
    fn sanitized_clamps_gain_and_scrubs_nan() {
        let dirty = AnimDriveParams {
            gain: f32::NAN,
            enabled: true,
        }
        .sanitized();
        assert!((dirty.gain - 0.0).abs() < 1e-9);
        let over = AnimDriveParams {
            gain: 5.0,
            enabled: true,
        }
        .sanitized();
        assert!((over.gain - 1.0).abs() < 1e-9);
    }

    #[test]
    fn anim_drive_pulls_toward_anchor_and_updates_velocity() {
        let (anchor, particle) = anchor_and_particle(
            Vec3::ZERO,
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
        );
        let mut particles = [particle];
        let anchors = [anchor];
        let painted = [PaintedConstraint::new(f32::INFINITY, 0.0, 1.0, 1.0)];
        let dt = 1.0 / 60.0;
        drive_toward_anim(
            &mut particles,
            &anchors,
            &painted,
            AnimDriveParams {
                gain: 0.5,
                enabled: true,
            },
            dt,
        );
        // follow = 1.0 * 0.5 = 0.5, so the vertex moves halfway to the anchor.
        assert!((particles[0].position.y - 1.0).abs() < 1e-6);
        // Velocity advanced by the imposed displacement over dt (delta.y = -1).
        assert!((particles[0].velocity.y - (-1.0 / dt)).abs() < 1e-3);
    }

    #[test]
    fn anim_drive_disabled_is_a_no_op() {
        let (anchor, particle) = anchor_and_particle(
            Vec3::ZERO,
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
        );
        let mut particles = [particle];
        let anchors = [anchor];
        let painted = [PaintedConstraint::new(f32::INFINITY, 0.0, 1.0, 1.0)];
        drive_toward_anim(
            &mut particles,
            &anchors,
            &painted,
            AnimDriveParams::default(),
            1.0 / 60.0,
        );
        assert!((particles[0].position.y - 2.0).abs() < 1e-9);
    }

    #[test]
    fn max_distance_projects_onto_sphere() {
        let (anchor, particle) =
            anchor_and_particle(Vec3::ZERO, Vec3::ZERO, Vec3::new(3.0, 0.0, 0.0));
        let mut particles = [particle];
        let anchors = [anchor];
        let painted = [PaintedConstraint::new(1.0, 0.0, 1.0, 0.0)];
        clamp_max_distance(&mut particles, &anchors, &painted);
        assert!((particles[0].position.x - 1.0).abs() < 1e-6);
    }

    #[test]
    fn zero_max_distance_welds_to_anchor() {
        let (anchor, particle) = anchor_and_particle(
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::ZERO,
            Vec3::new(4.0, 5.0, 6.0),
        );
        let mut particles = [particle];
        let anchors = [anchor];
        let painted = [PaintedConstraint::new(0.0, 0.0, 1.0, 0.0)];
        clamp_max_distance(&mut particles, &anchors, &painted);
        assert!(particles[0].position.distance(Vec3::new(1.0, 2.0, 3.0)) < 1e-6);
    }

    #[test]
    fn within_max_distance_is_untouched() {
        let (anchor, particle) =
            anchor_and_particle(Vec3::ZERO, Vec3::ZERO, Vec3::new(0.5, 0.0, 0.0));
        let mut particles = [particle];
        let anchors = [anchor];
        let painted = [PaintedConstraint::new(1.0, 0.0, 1.0, 0.0)];
        clamp_max_distance(&mut particles, &anchors, &painted);
        assert!((particles[0].position.x - 0.5).abs() < 1e-9);
    }

    #[test]
    fn backstop_pushes_vertex_out_of_cushion() {
        // Anchor at origin, outward normal +y, backstop 1 -> cushion sphere
        // centred at (0,-1,0) radius 1. A vertex just below the surface at
        // (0,-0.5,0) is inside and must be projected out.
        let (anchor, particle) = anchor_and_particle(
            Vec3::ZERO,
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, -0.5, 0.0),
        );
        let mut particles = [particle];
        let anchors = [anchor];
        let painted = [PaintedConstraint::new(f32::INFINITY, 1.0, 1.0, 0.0)];
        apply_painted_backstop(&mut particles, &anchors, &painted);
        // Projected radially from centre (0,-1,0): the nearest surface point is
        // straight up at the anchor (0,0,0).
        assert!(particles[0].position.distance(Vec3::ZERO) < 1e-6);
    }

    #[test]
    fn backstop_outside_cushion_is_untouched() {
        let (anchor, particle) = anchor_and_particle(
            Vec3::ZERO,
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.5, 0.0),
        );
        let mut particles = [particle];
        let anchors = [anchor];
        let painted = [PaintedConstraint::new(f32::INFINITY, 1.0, 1.0, 0.0)];
        apply_painted_backstop(&mut particles, &anchors, &painted);
        assert!((particles[0].position.y - 0.5).abs() < 1e-9);
    }

    #[test]
    fn blend_mixes_sim_and_skin() {
        let (anchor, particle) =
            anchor_and_particle(Vec3::ZERO, Vec3::ZERO, Vec3::new(0.0, 4.0, 0.0));
        let mut particles = [particle];
        let anchors = [anchor];
        let painted = [PaintedConstraint::new(f32::INFINITY, 0.0, 0.25, 0.0)];
        blend_to_skin(&mut particles, &anchors, &painted);
        // 0 + 0.25 * (4 - 0) = 1.0
        assert!((particles[0].position.y - 1.0).abs() < 1e-6);
    }

    #[test]
    fn pinned_vertices_are_never_moved() {
        let anchor = SkinnedAnchor::new(Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0));
        let pinned = ClothParticle::pinned(Vec3::new(9.0, 9.0, 9.0));
        let mut particles = [pinned];
        let anchors = [anchor];
        let painted = [PaintedConstraint::new(0.0, 1.0, 0.0, 1.0)];
        drive_toward_anim(
            &mut particles,
            &anchors,
            &painted,
            AnimDriveParams {
                gain: 1.0,
                enabled: true,
            },
            1.0 / 60.0,
        );
        clamp_max_distance(&mut particles, &anchors, &painted);
        apply_painted_backstop(&mut particles, &anchors, &painted);
        blend_to_skin(&mut particles, &anchors, &painted);
        assert!(particles[0].position.distance(Vec3::new(9.0, 9.0, 9.0)) < 1e-9);
    }

    #[test]
    fn out_of_range_indices_are_skipped() {
        // Two particles, one anchor, one painted weight: only index 0 is touched.
        let mut particles = [
            ClothParticle {
                position: Vec3::new(5.0, 0.0, 0.0),
                velocity: Vec3::ZERO,
                inverse_mass: 1.0,
            },
            ClothParticle {
                position: Vec3::new(9.0, 0.0, 0.0),
                velocity: Vec3::ZERO,
                inverse_mass: 1.0,
            },
        ];
        let anchors = [SkinnedAnchor::new(Vec3::ZERO, Vec3::ZERO)];
        let painted = [PaintedConstraint::new(1.0, 0.0, 1.0, 0.0)];
        clamp_max_distance(&mut particles, &anchors, &painted);
        assert!((particles[0].position.x - 1.0).abs() < 1e-6);
        assert!((particles[1].position.x - 9.0).abs() < 1e-9);
    }
}
