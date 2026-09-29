//! Baked point-cache sampling, playback, and emission — the `CPU` reference for
//! the §8.3 "Field / Point Cache" `DataInterface` (design §8.3).
//!
//! A *point cache* is a discrete, time-sampled recording of a moving point
//! cloud: `F` frames, each holding the same `N` points with per-point channels
//! (position, velocity, and an optional scalar such as size or age). It is the
//! read-only, pre-baked counterpart to a live simulation — a `Houdini`-exported
//! point cache, an `Alembic`/`bgeo` sequence, or a `Niagara` "simulation cache"
//! played back to drive particles without re-simulating.
//!
//! This module sits orthogonally beside its two sibling samplers:
//!
//! * [`super::mesh_emission`] samples the *surface of a static mesh* (triangle
//!   picking + barycentric interpolation); it has no time axis.
//! * [`super::vector_field`] samples a *discrete 3D vector field* (a spatial
//!   `Texture3d` grid of directions); it is indexed by world position, not by
//!   time or point identity.
//! * This module samples a *discrete time-series of tracked points*: indexed by
//!   a playback time (interpolated between the two bracketing frames) and by
//!   point identity, replaying baked motion and optionally spawning particles
//!   at cached points (the `SpawnPointCache` module).
//!
//! Playback maps a wall-clock time to a fractional frame at the cache's `FPS`
//! and blends the two bracketing frames under a [`PlaybackMode`] boundary
//! policy (`Clamp` / `Loop` / `PingPong`). Every operation is a linear blend of
//! stored samples: the only floating-point primitives beyond ordinary
//! arithmetic are `f32::floor` (frame location) and `sqrt` (through [`Vec3`]);
//! there are no transcendental calls, so the `CPU` reference is bit-reproducible
//! against a future `GPU` kernel that reads the same baked buffers.

use alloc::vec::Vec;

use super::emitter::UnitCursor;
use super::sort_cull::Aabb;
use super::Vec3;

/// Absolute tolerance for the `f32` equality guards in this module (positive
/// `FPS` validation and degenerate-frame detection). Comparisons use
/// `(a - b).abs() < EPS` rather than a bare `==`.
pub const EPS: f32 = 1e-6;

/// How a playback time outside the cache's recorded `[0, F)` frame range is
/// resolved (design §8.3).
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum PlaybackMode {
    /// Hold the first/last frame: times before frame `0` read frame `0` and
    /// times past the final frame read the final frame.
    Clamp,
    /// Wrap around modulo the frame count, so the recording repeats seamlessly
    /// (frame `F` maps back to frame `0`).
    Loop,
    /// Reflect at both ends, so playback bounces `0 -> F-1 -> 0` without a jump
    /// at the turn-around.
    PingPong,
}

/// The interpolated sample a point-cache emission yields (design §8.3).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EmittedPoint {
    /// Interpolated world position of the chosen cached point.
    pub position: Vec3,
    /// Interpolated velocity of the chosen cached point.
    pub velocity: Vec3,
    /// Interpolated scalar channel, or `None` when the cache stores no scalar.
    pub scalar: Option<f32>,
}

/// A baked point cache: `F` frames of `N` points, stored frame-major with the
/// points contiguous within each frame (design §8.3).
///
/// Channels are parallel flat buffers of length `F * N`: `positions` and
/// `velocities` are always present, while the scalar channel is optional (an
/// empty buffer means "no scalar", not a zero-filled one). The linear layout
/// matches the row-major buffer a `GPU` kernel binds, so the `CPU` reference and
/// the shader address the same element for a given `(frame, point)`.
#[derive(Clone, Debug, PartialEq)]
pub struct PointCache {
    frame_count: u32,
    point_count: u32,
    fps: f32,
    positions: Vec<Vec3>,
    velocities: Vec<Vec3>,
    scalars: Vec<f32>,
}

impl PointCache {
    /// Builds a cache from parallel frame-major channel buffers.
    ///
    /// Returns `None` unless every invariant holds: `frame_count >= 1`,
    /// `point_count >= 1`, `fps > 0`, the `F * N` element count does not
    /// overflow a `usize`, `positions.len() == velocities.len() == F * N`, and
    /// the `scalars` buffer is either empty (no scalar channel) or exactly
    /// `F * N` long. This is the storage contract every accessor relies on.
    #[must_use]
    pub fn from_frames(
        frame_count: u32,
        point_count: u32,
        fps: f32,
        positions: Vec<Vec3>,
        velocities: Vec<Vec3>,
        scalars: Vec<f32>,
    ) -> Option<Self> {
        if frame_count == 0 || point_count == 0 || fps <= EPS {
            return None;
        }
        let count = (frame_count as usize).checked_mul(point_count as usize)?;
        if positions.len() != count || velocities.len() != count {
            return None;
        }
        if !scalars.is_empty() && scalars.len() != count {
            return None;
        }
        Some(Self {
            frame_count,
            point_count,
            fps,
            positions,
            velocities,
            scalars,
        })
    }

    /// The number of recorded frames.
    #[must_use]
    pub fn frame_count(&self) -> u32 {
        self.frame_count
    }

    /// The number of points recorded in every frame.
    #[must_use]
    pub fn point_count(&self) -> u32 {
        self.point_count
    }

    /// The recording rate in frames per second (`FPS`).
    #[must_use]
    pub fn fps(&self) -> f32 {
        self.fps
    }

    /// Whether the cache stores an optional scalar channel.
    #[must_use]
    pub fn has_scalar_channel(&self) -> bool {
        !self.scalars.is_empty()
    }

    /// Frame-major linear index of point `point` in frame `frame`, with both
    /// coordinates clamped into range so the accessors never panic.
    fn linear_index(&self, frame: u32, point: u32) -> usize {
        let f = frame.min(self.frame_count - 1);
        let p = point.min(self.point_count - 1);
        (f as usize) * (self.point_count as usize) + (p as usize)
    }

    /// The stored position of `point` at integer `frame` (coordinates clamped).
    #[must_use]
    pub fn position(&self, frame: u32, point: u32) -> Vec3 {
        self.positions[self.linear_index(frame, point)]
    }

    /// The stored velocity of `point` at integer `frame` (coordinates clamped).
    #[must_use]
    pub fn velocity(&self, frame: u32, point: u32) -> Vec3 {
        self.velocities[self.linear_index(frame, point)]
    }

    /// The stored scalar of `point` at integer `frame`, or `None` when the
    /// cache has no scalar channel (coordinates clamped).
    #[must_use]
    pub fn scalar(&self, frame: u32, point: u32) -> Option<f32> {
        if self.scalars.is_empty() {
            None
        } else {
            Some(self.scalars[self.linear_index(frame, point)])
        }
    }

    /// Locates a playback `time_seconds` between two frames under `mode`.
    ///
    /// Returns `(f0, f1, frac)`: the bracketing frame indices and the blend
    /// factor `frac` in `[0, 1)` where `0` is exactly `f0`. Frames are located
    /// by `floor(time * fps)` and resolved through the boundary policy, so an
    /// integer frame time returns that frame with `frac = 0`. Uses only
    /// `floor` and integer modulo/reflection arithmetic.
    #[must_use]
    pub fn frame_pair(&self, time_seconds: f32, mode: PlaybackMode) -> (u32, u32, f32) {
        let t = time_seconds * self.fps;
        let base = t.floor();
        let frac = t - base;
        let i0 = base as i32;
        let f0 = resolve_frame(i0, self.frame_count, mode);
        let f1 = resolve_frame(i0 + 1, self.frame_count, mode);
        (f0, f1, frac)
    }

    /// Interpolated position of `point` at playback `time` under `mode`.
    ///
    /// Linearly blends the two bracketing frames from [`PointCache::frame_pair`].
    #[must_use]
    pub fn position_at(&self, point: u32, time: f32, mode: PlaybackMode) -> Vec3 {
        let (f0, f1, frac) = self.frame_pair(time, mode);
        let a = self.position(f0, point);
        let b = self.position(f1, point);
        lerp_vec3(a, b, frac)
    }

    /// Interpolated velocity of `point` at playback `time` under `mode`.
    #[must_use]
    pub fn velocity_at(&self, point: u32, time: f32, mode: PlaybackMode) -> Vec3 {
        let (f0, f1, frac) = self.frame_pair(time, mode);
        let a = self.velocity(f0, point);
        let b = self.velocity(f1, point);
        lerp_vec3(a, b, frac)
    }

    /// Interpolated scalar of `point` at playback `time` under `mode`, or `None`
    /// when the cache has no scalar channel.
    #[must_use]
    pub fn scalar_at(&self, point: u32, time: f32, mode: PlaybackMode) -> Option<f32> {
        if self.scalars.is_empty() {
            return None;
        }
        let (f0, f1, frac) = self.frame_pair(time, mode);
        let a = self.scalars[self.linear_index(f0, point)];
        let b = self.scalars[self.linear_index(f1, point)];
        Some(lerp_scalar(a, b, frac))
    }

    /// Samples every point's interpolated position at playback `time` into
    /// `out` (design §8.3).
    ///
    /// Clears `out` and pushes one [`Vec3`] per cached point in point order, so
    /// callers can reuse a scratch buffer across frames without reallocating.
    pub fn sample_frame(&self, time: f32, mode: PlaybackMode, out: &mut Vec<Vec3>) {
        out.clear();
        let (f0, f1, frac) = self.frame_pair(time, mode);
        for point in 0..self.point_count {
            let a = self.position(f0, point);
            let b = self.position(f1, point);
            out.push(lerp_vec3(a, b, frac));
        }
    }

    /// Axis-aligned bounds of a single integer `frame` (design §8.3).
    ///
    /// Grows an [`Aabb`] over every point's stored position in that frame; the
    /// `frame` index is clamped into range.
    #[must_use]
    pub fn frame_bounds(&self, frame: u32) -> Aabb {
        let f = frame.min(self.frame_count - 1);
        let mut bounds = Aabb::empty();
        for point in 0..self.point_count {
            bounds = bounds.expand(self.position(f, point));
        }
        bounds
    }

    /// Axis-aligned bounds over every point across every frame (design §8.3).
    ///
    /// This is the whole-cache `AABB` a renderer uses to size the effect; it is
    /// the union of all per-frame bounds.
    #[must_use]
    pub fn cache_bounds(&self) -> Aabb {
        let mut bounds = Aabb::empty();
        for &p in &self.positions {
            bounds = bounds.expand(p);
        }
        bounds
    }
}

/// Emits one particle from a point cache at playback `time` (the
/// `SpawnPointCache` module, design §8.3).
///
/// Draws a single unit sample from `cursor` to pick a cached point index, then
/// returns that point's interpolated [`EmittedPoint`] at `time` under `mode`.
/// Determinism follows the shared [`UnitCursor`] contract: the same sample
/// sequence selects the same points, so the `CPU` and `GPU` spawn paths agree.
#[must_use]
pub fn emit_point(
    cache: &PointCache,
    time: f32,
    mode: PlaybackMode,
    cursor: &mut UnitCursor<'_>,
) -> EmittedPoint {
    let n = cache.point_count();
    let raw = (cursor.next_unit() * n as f32).floor();
    let idx = clamp_index(raw, n);
    EmittedPoint {
        position: cache.position_at(idx, time, mode),
        velocity: cache.velocity_at(idx, time, mode),
        scalar: cache.scalar_at(idx, time, mode),
    }
}

/// Clamps a floored unit-scaled draw into a valid `[0, n)` point index. `n` is
/// assumed non-zero (guaranteed by [`PointCache::from_frames`]).
fn clamp_index(raw: f32, n: u32) -> u32 {
    if raw < 0.0 {
        0
    } else {
        (raw as u32).min(n - 1)
    }
}

/// Resolves a signed frame index into a valid `[0, frame_count)` frame under a
/// boundary policy. `frame_count` is assumed non-zero.
fn resolve_frame(i: i32, frame_count: u32, mode: PlaybackMode) -> u32 {
    let f = frame_count as i32;
    match mode {
        PlaybackMode::Clamp => i.clamp(0, f - 1) as u32,
        PlaybackMode::Loop => (((i % f) + f) % f) as u32,
        PlaybackMode::PingPong => {
            if frame_count <= 1 {
                return 0;
            }
            let period = 2 * (f - 1);
            let m = (((i % period) + period) % period).abs();
            (if m < f { m } else { period - m }) as u32
        }
    }
}

/// Linear blend `a + (b - a) * frac` between two vectors (multiply-add).
fn lerp_vec3(a: Vec3, b: Vec3, frac: f32) -> Vec3 {
    a.add(b.sub(a).scale(frac))
}

/// Linear blend `a + (b - a) * frac` between two scalars (multiply-add).
fn lerp_scalar(a: f32, b: f32, frac: f32) -> f32 {
    a + (b - a) * frac
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    /// Absolute tolerance for the assertions below.
    const T: f32 = 1e-5;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < T
    }

    fn approx_vec(a: Vec3, b: Vec3) -> bool {
        approx(a.x, b.x) && approx(a.y, b.y) && approx(a.z, b.z)
    }

    /// A 3-frame, 2-point cache: point 0 walks +X per frame, point 1 walks +Y;
    /// velocities mirror the step; scalar counts up.
    fn walk_cache() -> PointCache {
        // frame 0: p0=(0,0,0) p1=(0,0,0)
        // frame 1: p0=(1,0,0) p1=(0,1,0)
        // frame 2: p0=(2,0,0) p1=(0,2,0)
        let positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
        ];
        let velocities = vec![
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let scalars = vec![0.0, 10.0, 1.0, 11.0, 2.0, 12.0];
        PointCache::from_frames(3, 2, 24.0, positions, velocities, scalars).expect("valid cache")
    }

    #[test]
    fn from_frames_validates_lengths() {
        let good = PointCache::from_frames(
            2,
            2,
            30.0,
            vec![Vec3::ZERO; 4],
            vec![Vec3::ZERO; 4],
            vec![0.0; 4],
        );
        assert!(good.is_some());
        // Mismatched position length.
        assert!(PointCache::from_frames(
            2,
            2,
            30.0,
            vec![Vec3::ZERO; 3],
            vec![Vec3::ZERO; 4],
            Vec::new()
        )
        .is_none());
        // Mismatched scalar length (non-empty but wrong).
        assert!(PointCache::from_frames(
            2,
            2,
            30.0,
            vec![Vec3::ZERO; 4],
            vec![Vec3::ZERO; 4],
            vec![0.0; 3]
        )
        .is_none());
    }

    #[test]
    fn from_frames_rejects_bad_fps_and_empty() {
        assert!(
            PointCache::from_frames(1, 1, 0.0, vec![Vec3::ZERO], vec![Vec3::ZERO], Vec::new())
                .is_none()
        );
        assert!(PointCache::from_frames(
            1,
            1,
            -5.0,
            vec![Vec3::ZERO],
            vec![Vec3::ZERO],
            Vec::new()
        )
        .is_none());
        assert!(PointCache::from_frames(0, 1, 30.0, Vec::new(), Vec::new(), Vec::new()).is_none());
        assert!(PointCache::from_frames(1, 0, 30.0, Vec::new(), Vec::new(), Vec::new()).is_none());
    }

    #[test]
    fn empty_scalar_channel_is_optional() {
        let c = PointCache::from_frames(
            2,
            2,
            30.0,
            vec![Vec3::ZERO; 4],
            vec![Vec3::ZERO; 4],
            Vec::new(),
        )
        .expect("valid");
        assert!(!c.has_scalar_channel());
        assert_eq!(c.scalar(0, 0), None);
        assert_eq!(c.scalar_at(0, 0.0, PlaybackMode::Clamp), None);
    }

    #[test]
    fn linear_index_is_frame_major() {
        let c = walk_cache();
        // frame 1, point 0 -> stored (1,0,0); frame 1 point 1 -> (0,1,0).
        assert!(approx_vec(c.position(1, 0), Vec3::new(1.0, 0.0, 0.0)));
        assert!(approx_vec(c.position(1, 1), Vec3::new(0.0, 1.0, 0.0)));
        assert!(approx_vec(c.position(2, 0), Vec3::new(2.0, 0.0, 0.0)));
        assert_eq!(c.scalar(2, 1), Some(12.0));
    }

    #[test]
    fn accessors_clamp_out_of_range() {
        let c = walk_cache();
        // frame 9 clamps to last frame 2; point 9 clamps to last point 1.
        assert!(approx_vec(c.position(9, 9), Vec3::new(0.0, 2.0, 0.0)));
    }

    #[test]
    fn frame_pair_hits_integer_frame_with_zero_frac() {
        let c = walk_cache(); // fps = 24
                              // time = 1/24 s -> exactly frame 1.
        let (f0, f1, frac) = c.frame_pair(1.0 / 24.0, PlaybackMode::Clamp);
        assert_eq!(f0, 1);
        assert_eq!(f1, 2);
        assert!(approx(frac, 0.0));
    }

    #[test]
    fn frame_pair_midpoint_frac_is_half() {
        let c = walk_cache();
        // time = 1.5/24 s -> between frame 1 and 2 at frac 0.5.
        let (f0, f1, frac) = c.frame_pair(1.5 / 24.0, PlaybackMode::Clamp);
        assert_eq!(f0, 1);
        assert_eq!(f1, 2);
        assert!(approx(frac, 0.5));
    }

    #[test]
    fn clamp_holds_boundaries() {
        let c = walk_cache();
        // Negative time clamps both frames to 0.
        let (f0, f1, _) = c.frame_pair(-5.0, PlaybackMode::Clamp);
        assert_eq!((f0, f1), (0, 0));
        // Far-future time clamps both to the last frame 2.
        let (g0, g1, _) = c.frame_pair(100.0, PlaybackMode::Clamp);
        assert_eq!((g0, g1), (2, 2));
    }

    #[test]
    fn integer_frame_returns_stored_value() {
        let c = walk_cache();
        assert!(approx_vec(
            c.position_at(0, 2.0 / 24.0, PlaybackMode::Clamp),
            Vec3::new(2.0, 0.0, 0.0)
        ));
        assert!(approx_vec(
            c.velocity_at(1, 1.0 / 24.0, PlaybackMode::Clamp),
            Vec3::new(0.0, 1.0, 0.0)
        ));
        assert_eq!(c.scalar_at(1, 2.0 / 24.0, PlaybackMode::Clamp), Some(12.0));
    }

    #[test]
    fn half_frame_is_mean_of_neighbors() {
        let c = walk_cache();
        // Point 0 between frame 0 (0,0,0) and frame 1 (1,0,0) at frac 0.5.
        let pos = c.position_at(0, 0.5 / 24.0, PlaybackMode::Clamp);
        assert!(approx_vec(pos, Vec3::new(0.5, 0.0, 0.0)));
        // Scalar for point 1: frame 0 = 10, frame 1 = 11 -> 10.5.
        let s = c
            .scalar_at(1, 0.5 / 24.0, PlaybackMode::Clamp)
            .expect("has scalar");
        assert!(approx(s, 10.5));
    }

    #[test]
    fn loop_wraps_continuously() {
        let c = walk_cache(); // 3 frames
                              // time between frame 2 and frame 3: Loop maps frame 3 -> frame 0.
        let (f0, f1, frac) = c.frame_pair(2.5 / 24.0, PlaybackMode::Loop);
        assert_eq!(f0, 2);
        assert_eq!(f1, 0);
        assert!(approx(frac, 0.5));
        // Interpolated point 0: frame 2 (2,0,0) blended toward frame 0 (0,0,0).
        let pos = c.position_at(0, 2.5 / 24.0, PlaybackMode::Loop);
        assert!(approx_vec(pos, Vec3::new(1.0, 0.0, 0.0)));
    }

    #[test]
    fn pingpong_reflects_symmetrically() {
        let c = walk_cache(); // 3 frames -> triangle-wave period = 2*(3-1) = 4.
                              // The bounce visits frames 0,1,2,1,0,1,2,1,... with no repeated endpoint.
        let expected = [0u32, 1, 2, 1, 0, 1, 2, 1, 0];
        for (idx, &want) in expected.iter().enumerate() {
            let (f0, _, _) = c.frame_pair(idx as f32 / 24.0, PlaybackMode::PingPong);
            assert_eq!(f0, want, "ping-pong frame at global index {idx}");
        }
        // Index 5 lands on frame 1, symmetric with index 1 across the bounce.
        let (h0, _, _) = c.frame_pair(5.0 / 24.0, PlaybackMode::PingPong);
        assert_eq!(h0, 1);
    }

    #[test]
    fn pingpong_single_frame_is_stable() {
        let c = PointCache::from_frames(
            1,
            1,
            30.0,
            vec![Vec3::new(7.0, 0.0, 0.0)],
            vec![Vec3::ZERO],
            Vec::new(),
        )
        .expect("valid");
        let (f0, f1, _) = c.frame_pair(123.0, PlaybackMode::PingPong);
        assert_eq!((f0, f1), (0, 0));
    }

    #[test]
    fn sample_frame_fills_all_points() {
        let c = walk_cache();
        let mut out = Vec::new();
        c.sample_frame(0.5 / 24.0, PlaybackMode::Clamp, &mut out);
        assert_eq!(out.len(), 2);
        assert!(approx_vec(out[0], Vec3::new(0.5, 0.0, 0.0)));
        assert!(approx_vec(out[1], Vec3::new(0.0, 0.5, 0.0)));
        // Reusing the buffer clears it first.
        c.sample_frame(0.0, PlaybackMode::Clamp, &mut out);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn emit_point_is_deterministic() {
        let c = walk_cache();
        // Two draws select point 0 (0.1*2 -> 0) then point 1 (0.9*2 -> 1).
        let samples = [0.1_f32, 0.9_f32];
        let mut a = UnitCursor::new(&samples);
        let mut b = UnitCursor::new(&samples);
        let time = 2.0 / 24.0; // frame 2 exactly
        let ea0 = emit_point(&c, time, PlaybackMode::Clamp, &mut a);
        let eb0 = emit_point(&c, time, PlaybackMode::Clamp, &mut b);
        assert_eq!(ea0, eb0);
        assert!(approx_vec(ea0.position, Vec3::new(2.0, 0.0, 0.0)));
        assert_eq!(ea0.scalar, Some(2.0));
        let ea1 = emit_point(&c, time, PlaybackMode::Clamp, &mut a);
        let eb1 = emit_point(&c, time, PlaybackMode::Clamp, &mut b);
        assert_eq!(ea1, eb1);
        assert!(approx_vec(ea1.position, Vec3::new(0.0, 2.0, 0.0)));
        assert_eq!(ea1.scalar, Some(12.0));
    }

    #[test]
    fn emit_point_clamps_top_of_unit_range() {
        let c = walk_cache();
        // A draw of exactly 1.0 would index N; it must clamp to the last point.
        let samples = [1.0_f32];
        let mut cursor = UnitCursor::new(&samples);
        let e = emit_point(&c, 0.0, PlaybackMode::Clamp, &mut cursor);
        assert!(approx_vec(e.position, Vec3::new(0.0, 0.0, 0.0)));
        // Point 1 at frame 0 has scalar 10.
        assert_eq!(e.scalar, Some(10.0));
    }

    #[test]
    fn frame_bounds_are_exact() {
        let c = walk_cache();
        // Frame 2: p0=(2,0,0), p1=(0,2,0) -> min (0,0,0) max (2,2,0).
        let b = c.frame_bounds(2);
        assert!(b.is_valid());
        assert!(approx_vec(b.min, Vec3::new(0.0, 0.0, 0.0)));
        assert!(approx_vec(b.max, Vec3::new(2.0, 2.0, 0.0)));
    }

    #[test]
    fn cache_bounds_span_all_frames() {
        let c = walk_cache();
        let b = c.cache_bounds();
        assert!(approx_vec(b.min, Vec3::new(0.0, 0.0, 0.0)));
        assert!(approx_vec(b.max, Vec3::new(2.0, 2.0, 0.0)));
    }
}
