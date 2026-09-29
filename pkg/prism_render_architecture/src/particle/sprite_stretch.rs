//! Motion-stretch billboard *sizing* for `Sprite` renderers (design §15).
//!
//! This module is the orthogonal complement to the billboard *orientation*
//! logic in [`super::renderers`]: that module answers "which way does the quad
//! face" (the [`super::renderers::BillboardBasis`] for camera-facing,
//! velocity-aligned, and fixed-axis sprites), while this module answers "how
//! big is the quad, and how is it stretched along motion". The two concerns are
//! kept deliberately decoupled — a `VelocityAligned` sprite gets its axes from
//! `renderers` and its *anisotropic extents* from here — so we intentionally do
//! **not** import `renderers` and only borrow the shared [`Vec3`] math from the
//! parent module.
//!
//! Two flavours of motion stretch are provided, matching production `VFX`
//! stacks:
//!
//! 1. **Velocity stretch** — the quad is scaled along the (normalized) velocity
//!    direction as a function of speed, giving spark streaks / speed lines. The
//!    stretched length is `base_half + speed * stretch_scale`, clamped to a
//!    `[min_length, max_length]` window, with the perpendicular width scaled by
//!    `width_scale`.
//! 2. **Trail stretch** — the quad spans from the particle's previous-frame
//!    position to its current position, so its length equals the frame's travel
//!    distance. This is the common "connect the last two positions" ribbon-lite
//!    trail.
//!
//! All paths are degenerate-safe: a (numerically) zero velocity falls back to
//! the isotropic base size and a deterministic axis, so no result is ever
//! `NaN`. Only `sqrt` (through [`Vec3::length`] / [`Vec3::normalize_or_zero`])
//! and ordinary arithmetic are used — no transcendental functions — so this
//! stays a bit-reproducible `CPU` reference for a future `GPU` build pass.

use super::Vec3;

/// Absolute tolerance for `f32` magnitude/near-zero comparisons in this module.
///
/// Speeds and lengths are compared against this rather than a bare `==`/`!=`
/// so near-zero motion deterministically selects the isotropic fallback.
pub const EPS: f32 = 1e-6;

/// Parameters controlling velocity-driven motion stretch (design §15).
///
/// The stretched half-length along the velocity direction is
/// `base_half + speed * stretch_scale`, clamped to `[min_length, max_length]`;
/// the perpendicular half-width is `base_half * width_scale`. `min_length` must
/// not exceed `max_length`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StretchParams {
    /// Metres of extra half-length added per unit of speed.
    pub stretch_scale: f32,
    /// Lower clamp on the stretched half-length (metres).
    pub min_length: f32,
    /// Upper clamp on the stretched half-length (metres).
    pub max_length: f32,
    /// Multiplier applied to the base half-extent to obtain the half-width.
    pub width_scale: f32,
}

impl StretchParams {
    /// Computes the anisotropic half-extents for a particle moving at
    /// `velocity` with isotropic base half-extent `base_half` (design §15).
    ///
    /// `speed` is `velocity.length()` (a single `sqrt` through [`Vec3`]). When
    /// the velocity is (numerically) zero — `length_squared` below [`EPS`] —
    /// the quad is left isotropic at `base_half` in both axes and no stretch or
    /// clamp is applied. Otherwise the half-length is
    /// `(base_half + speed * stretch_scale).clamp(min_length, max_length)` and
    /// the half-width is `base_half * width_scale`.
    #[must_use]
    pub fn stretched_size(self, velocity: Vec3, base_half: f32) -> StretchedQuad {
        if velocity.length_squared() < EPS {
            return StretchedQuad {
                half_length: base_half,
                half_width: base_half,
            };
        }
        let speed = velocity.length();
        let raw_length = base_half + speed * self.stretch_scale;
        let half_length = raw_length.clamp(self.min_length, self.max_length);
        let half_width = base_half * self.width_scale;
        StretchedQuad {
            half_length,
            half_width,
        }
    }
}

/// The anisotropic half-extents of a motion-stretched sprite quad (design §15).
///
/// `half_length` runs along the (normalized) motion direction, `half_width`
/// along the perpendicular in-plane axis.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StretchedQuad {
    /// Half-extent along the velocity / trail direction (metres).
    pub half_length: f32,
    /// Half-extent perpendicular to the motion direction (metres).
    pub half_width: f32,
}

impl StretchedQuad {
    /// The anisotropic stretch ratio `half_length / half_width`, useful for
    /// compensating `UV` tiling so the sprite texture is not visibly squashed.
    ///
    /// Returns `1.0` (isotropic) when the half-width is below [`EPS`], avoiding
    /// a division by (near-)zero.
    #[must_use]
    pub fn aspect(self) -> f32 {
        if self.half_width.abs() < EPS {
            return 1.0;
        }
        self.half_length / self.half_width
    }
}

/// A trail quad connecting a particle's previous and current positions
/// (design §15).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrailQuad {
    /// The four world-space corners: near edge (`[0]`, `[1]`) at the current
    /// position and far edge (`[2]`, `[3]`) at the previous position, each edge
    /// offset by `±half_width` along the binormal.
    pub corners: [Vec3; 4],
    /// The trail length, equal to the distance travelled since last frame.
    pub length: f32,
}

/// Returns a deterministic unit vector perpendicular to `v`, or `+X` when `v`
/// is (numerically) zero. Used to synthesise a stable width axis when the
/// caller's binormal collapses.
#[must_use]
fn perpendicular_to(v: Vec3) -> Vec3 {
    // Cross with whichever world axis is least aligned with `v` (smallest
    // squared component) to avoid a near-zero cross product.
    let ax = v.x * v.x;
    let ay = v.y * v.y;
    let az = v.z * v.z;
    let reference = if ax <= ay && ax <= az {
        Vec3::new(1.0, 0.0, 0.0)
    } else if ay <= az {
        Vec3::new(0.0, 1.0, 0.0)
    } else {
        Vec3::new(0.0, 0.0, 1.0)
    };
    let perp = v.cross(reference).normalize_or_zero();
    if perp.length_squared() > EPS {
        perp
    } else {
        Vec3::new(1.0, 0.0, 0.0)
    }
}

/// Builds the four world-space corners of a stretched sprite quad (design §15).
///
/// `forward_axis` is the motion direction (typically the raw velocity) and is
/// normalized internally; a (numerically) zero axis falls back to `+Y`.
/// `camera_facing_binormal` is the in-plane width direction the caller derives
/// (typically `velocity × view`, i.e. the camera-facing side vector); it is
/// normalized internally and, when it collapses, replaced by a deterministic
/// perpendicular of the forward axis. Corners are laid out at
/// `center ± forward * half_length ± binormal * half_width`, so their centroid
/// is exactly `center` and the result is never `NaN`.
#[must_use]
pub fn stretched_corners(
    center: Vec3,
    forward_axis: Vec3,
    camera_facing_binormal: Vec3,
    half_length: f32,
    half_width: f32,
) -> [Vec3; 4] {
    let mut forward = forward_axis.normalize_or_zero();
    if forward.length_squared() <= EPS {
        forward = Vec3::new(0.0, 1.0, 0.0);
    }
    let mut binormal = camera_facing_binormal.normalize_or_zero();
    if binormal.length_squared() <= EPS {
        binormal = perpendicular_to(forward);
    }
    let fl = forward.scale(half_length);
    let fw = binormal.scale(half_width);
    [
        center.add(fl).add(fw),
        center.add(fl).sub(fw),
        center.sub(fl).add(fw),
        center.sub(fl).sub(fw),
    ]
}

/// Convenience helper that combines [`StretchParams::stretched_size`] and
/// [`stretched_corners`] for a velocity-stretched sprite (design §15).
///
/// The half-extents are derived from `velocity`/`base_half`/`params` and the
/// quad is oriented with `velocity` as its forward axis and
/// `camera_facing_binormal` as its width axis (both degenerate-safe).
#[must_use]
pub fn velocity_stretched_corners(
    center: Vec3,
    velocity: Vec3,
    camera_facing_binormal: Vec3,
    base_half: f32,
    params: StretchParams,
) -> [Vec3; 4] {
    let quad = params.stretched_size(velocity, base_half);
    stretched_corners(
        center,
        velocity,
        camera_facing_binormal,
        quad.half_length,
        quad.half_width,
    )
}

/// Builds a trail quad from the particle's previous position to its current
/// position (design §15).
///
/// The quad's length equals `center.distance(prev_position)` (the frame's
/// travel), its near edge sits at `center` and its far edge at
/// `prev_position`, each spread by `±half_width` along the binormal.
/// `camera_facing_binormal` is normalized internally and, when it collapses,
/// replaced by a deterministic perpendicular of the travel direction (or `+X`
/// when the particle did not move), so the corners are never `NaN`.
#[must_use]
pub fn stretched_from_prev(
    center: Vec3,
    prev_position: Vec3,
    camera_facing_binormal: Vec3,
    half_width: f32,
) -> TrailQuad {
    let travel = center.sub(prev_position);
    let length = travel.length();
    let mut binormal = camera_facing_binormal.normalize_or_zero();
    if binormal.length_squared() <= EPS {
        binormal = perpendicular_to(travel);
    }
    let bw = binormal.scale(half_width);
    let corners = [
        center.add(bw),
        center.sub(bw),
        prev_position.add(bw),
        prev_position.sub(bw),
    ];
    TrailQuad { corners, length }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f32 = 1e-5;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() <= TOL
    }

    fn vapprox(a: Vec3, b: Vec3) -> bool {
        approx(a.x, b.x) && approx(a.y, b.y) && approx(a.z, b.z)
    }

    fn params() -> StretchParams {
        StretchParams {
            stretch_scale: 1.0,
            min_length: 0.0,
            max_length: 100.0,
            width_scale: 1.0,
        }
    }

    #[test]
    fn zero_velocity_is_isotropic() {
        let p = params();
        let quad = p.stretched_size(Vec3::ZERO, 2.0);
        assert!(approx(quad.half_length, 2.0));
        assert!(approx(quad.half_width, 2.0));
        // A sub-EPS velocity also takes the isotropic branch.
        let tiny = Vec3::new(1e-4, 0.0, 0.0);
        let quad_tiny = p.stretched_size(tiny, 2.0);
        assert!(approx(quad_tiny.half_length, 2.0));
        assert!(approx(quad_tiny.half_width, 2.0));
    }

    #[test]
    fn speed_increases_half_length_then_clamps_to_max() {
        let p = StretchParams {
            stretch_scale: 1.0,
            min_length: 0.0,
            max_length: 5.0,
            width_scale: 1.0,
        };
        let slow = p.stretched_size(Vec3::new(1.0, 0.0, 0.0), 0.5);
        let fast = p.stretched_size(Vec3::new(3.0, 0.0, 0.0), 0.5);
        assert!(fast.half_length > slow.half_length);
        assert!(approx(slow.half_length, 1.5));
        assert!(approx(fast.half_length, 3.5));
        // Beyond the window the length saturates at max_length.
        let maxed = p.stretched_size(Vec3::new(100.0, 0.0, 0.0), 0.5);
        assert!(approx(maxed.half_length, 5.0));
    }

    #[test]
    fn min_length_clamp_applies() {
        let p = StretchParams {
            stretch_scale: 1.0,
            min_length: 1.0,
            max_length: 100.0,
            width_scale: 1.0,
        };
        // raw = 0.05 + 0.1 * 1.0 = 0.15, below min_length, so clamps up.
        let quad = p.stretched_size(Vec3::new(0.1, 0.0, 0.0), 0.05);
        assert!(approx(quad.half_length, 1.0));
    }

    #[test]
    fn width_scale_takes_effect() {
        let p = StretchParams {
            stretch_scale: 1.0,
            min_length: 0.0,
            max_length: 100.0,
            width_scale: 0.25,
        };
        let quad = p.stretched_size(Vec3::new(2.0, 0.0, 0.0), 2.0);
        assert!(approx(quad.half_width, 0.5));
    }

    #[test]
    fn corners_are_symmetric_about_center() {
        let center = Vec3::new(1.0, 2.0, 3.0);
        let corners = stretched_corners(
            center,
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            4.0,
            0.5,
        );
        let sum = corners[0].add(corners[1]).add(corners[2]).add(corners[3]);
        assert!(vapprox(sum, center.scale(4.0)));
        // Opposite corners mirror through the center.
        let mid_a = corners[0].add(corners[3]).scale(0.5);
        let mid_b = corners[1].add(corners[2]).scale(0.5);
        assert!(vapprox(mid_a, center));
        assert!(vapprox(mid_b, center));
    }

    #[test]
    fn aspect_ratio_is_length_over_width() {
        let quad = StretchedQuad {
            half_length: 4.0,
            half_width: 2.0,
        };
        assert!(approx(quad.aspect(), 2.0));
        // Zero width falls back to isotropic without dividing by zero.
        let degenerate = StretchedQuad {
            half_length: 4.0,
            half_width: 0.0,
        };
        assert!(approx(degenerate.aspect(), 1.0));
    }

    #[test]
    fn trail_length_equals_distance() {
        let center = Vec3::new(3.0, 4.0, 0.0);
        let prev = Vec3::ZERO;
        let trail = stretched_from_prev(center, prev, Vec3::new(0.0, 0.0, 1.0), 0.5);
        assert!(approx(trail.length, 5.0));
        // Near edge midpoint sits on the current position, far edge on prev.
        let near_mid = trail.corners[0].add(trail.corners[1]).scale(0.5);
        let far_mid = trail.corners[2].add(trail.corners[3]).scale(0.5);
        assert!(vapprox(near_mid, center));
        assert!(vapprox(far_mid, prev));
    }

    #[test]
    fn degenerate_directions_do_not_nan() {
        // Zero forward + zero binormal must still yield finite corners.
        let corners = stretched_corners(Vec3::ZERO, Vec3::ZERO, Vec3::ZERO, 2.0, 0.5);
        for c in corners {
            assert!(c.x.is_finite() && c.y.is_finite() && c.z.is_finite());
        }
        // Zero-travel trail with zero binormal: length 0, finite corners.
        let trail = stretched_from_prev(Vec3::ZERO, Vec3::ZERO, Vec3::ZERO, 0.5);
        assert!(approx(trail.length, 0.0));
        for c in trail.corners {
            assert!(c.x.is_finite() && c.y.is_finite() && c.z.is_finite());
        }
        // Velocity convenience path with zero velocity stays isotropic/finite.
        let vc = velocity_stretched_corners(Vec3::ZERO, Vec3::ZERO, Vec3::ZERO, 1.0, params());
        for c in vc {
            assert!(c.x.is_finite() && c.y.is_finite() && c.z.is_finite());
        }
    }
}
