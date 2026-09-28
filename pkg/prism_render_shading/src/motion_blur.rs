//! Plausible motion-blur reconstruction golden (CPU reference).
//!
//! This is the CPU twin of `shaders/motion_blur.wesl`, following Morgan
//! `McGuire` et al., *A Reconstruction Filter for Plausible Motion Blur* (I3D
//! 2012) as productised in later AAA engines. Motion blur is a shared
//! post-processing base, not a peer of the PBR/NPR shading fronts: every
//! illumination model resolves the same velocity buffer, and a stylized front
//! can dial the shutter down for a crisp, illustrative look while a
//! photographic front opens it for cinematic streaking.
//!
//! The pipeline has three stages, each a pure function here so the WESL twin
//! can mirror it arm-for-arm:
//!
//! * **Tile velocity.** [`tile_max`] reduces a screen tile to its
//!   longest velocity vector; [`neighbor_max`] dilates the tile field by one
//!   tile so a fast pixel bleeds into the neighbours it will smear across.
//! * **Depth classification.** [`soft_depth_compare`] softly decides whether a
//!   tap is in front of or behind the pixel being reconstructed, so a blurry
//!   foreground can streak over a sharp background without haloing.
//! * **Reconstruction weight.** [`sample_weight`] combines [`cone`] (a moving
//!   tap covering a static pixel) and [`cylinder`] (two moving samples
//!   overlapping) into the per-tap contribution the gather accumulates.
//!
//! Depths are linear view distances (larger = farther). All maths is plain
//! `f32` (no transcendentals), so the module needs no `bevy_math::ops` import
//! and stays bit-reproducible against the shader twin.

/// Euclidean length (in pixels) of a 2D screen-space velocity vector.
#[must_use]
pub fn velocity_length(v: [f32; 2]) -> f32 {
    (v[0] * v[0] + v[1] * v[1]).sqrt()
}

/// Scale a per-frame `clip_velocity` by the fraction of the frame the shutter
/// is open (`exposure_fraction`, typically `0.5`). Longer exposures streak
/// further; direction is preserved.
#[must_use]
pub fn shutter_velocity(clip_velocity: [f32; 2], exposure_fraction: f32) -> [f32; 2] {
    [
        clip_velocity[0] * exposure_fraction,
        clip_velocity[1] * exposure_fraction,
    ]
}

/// Clamp a velocity vector so its magnitude never exceeds `max_len` (the
/// per-frame blur budget, usually half a tile). Direction is preserved; a
/// non-positive budget collapses motion to zero, and vectors already within
/// budget pass through untouched.
#[must_use]
pub fn clamp_velocity(v: [f32; 2], max_len: f32) -> [f32; 2] {
    let len = velocity_length(v);
    if len <= max_len {
        return v;
    }
    // len > max_len here, so len > 0; a negative budget yields a zero vector.
    let scale = max_len.max(0.0) / len;
    [v[0] * scale, v[1] * scale]
}

/// Reduce a screen tile to the single velocity vector with the greatest
/// magnitude (the classic `TileMax` pass). An empty tile is motionless.
#[must_use]
pub fn tile_max(velocities: &[[f32; 2]]) -> [f32; 2] {
    let mut best = [0.0_f32, 0.0_f32];
    let mut best_len = 0.0_f32;
    for &v in velocities {
        let len = velocity_length(v);
        if len > best_len {
            best_len = len;
            best = v;
        }
    }
    best
}

/// Dilate the tile-velocity field by one tile: the longest velocity among a
/// tile and its eight neighbours (the `NeighborMax` pass), row-major with index
/// `4` the centre tile.
#[must_use]
pub fn neighbor_max(tiles: &[[f32; 2]; 9]) -> [f32; 2] {
    tile_max(tiles)
}

/// Own `smoothstep` (Hermite `3t^2 - 2t^3`) so the CPU golden and the WESL twin
/// agree bit-for-bit rather than trusting each platform's builtin. A degenerate
/// edge interval becomes a hard step at `edge0`.
#[must_use]
fn smooth_step(edge0: f32, edge1: f32, x: f32) -> f32 {
    if edge0 == edge1 {
        return if x < edge0 { 0.0 } else { 1.0 };
    }
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Soft depth ordering: `~1` when `za` is clearly in front of `zb` (nearer the
/// camera, i.e. smaller depth), `~0` when clearly behind, and a linear ramp
/// within `extent` so foreground/background classification does not pop. Equal
/// depths return `1`. `extent` is floored to a small epsilon to avoid a divide
/// by zero.
#[must_use]
pub fn soft_depth_compare(za: f32, zb: f32, extent: f32) -> f32 {
    let e = extent.max(1.0e-6);
    (1.0 - (za - zb) / e).clamp(0.0, 1.0)
}

/// Cone falloff for a moving tap covering a pixel: full weight at zero distance,
/// decreasing linearly to zero at the tap's motion `speed`. A motionless tap
/// (`speed <= 0`) covers nothing.
#[must_use]
pub fn cone(dist: f32, speed: f32) -> f32 {
    if speed <= 1.0e-6 {
        return 0.0;
    }
    (1.0 - dist / speed).clamp(0.0, 1.0)
}

/// Cylinder falloff for two overlapping moving samples: `~1` while `dist` is
/// within the sample's motion `speed`, rolling off through a narrow band around
/// the extent. A motionless sample contributes nothing.
#[must_use]
pub fn cylinder(dist: f32, speed: f32) -> f32 {
    1.0 - smooth_step(0.95 * speed, 1.05 * speed, dist)
}

/// `McGuire` reconstruction weight for one gather tap. Combines three regimes:
/// a blurry foreground sample streaking over the (sharper) centre, the centre
/// streaking over a background sample, and two moving samples overlapping
/// (the cylinder term, doubled as in the reference). `dist` is the pixel gap
/// between the centre and the tap.
#[must_use]
pub fn sample_weight(
    center_depth: f32,
    sample_depth: f32,
    dist: f32,
    center_velocity_len: f32,
    sample_velocity_len: f32,
    soft_z_extent: f32,
) -> f32 {
    // `foreground`: sample is in front of the centre -> its blur covers us.
    let foreground = soft_depth_compare(sample_depth, center_depth, soft_z_extent);
    // `background`: sample is behind the centre -> our blur covers it.
    let background = soft_depth_compare(center_depth, sample_depth, soft_z_extent);

    let mut weight = foreground * cone(dist, sample_velocity_len);
    weight += background * cone(dist, center_velocity_len);
    weight += cylinder(dist, sample_velocity_len) * cylinder(dist, center_velocity_len) * 2.0;
    weight
}

/// Artist / camera controls for the motion-blur pass.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MotionBlurParams {
    /// Master enable; when `false` the pass is skipped entirely.
    pub enabled: bool,
    /// Blur budget in pixels; velocities are clamped to this by
    /// [`clamp_velocity`] to bound the gather radius.
    pub max_velocity_px: f32,
    /// Depth softness (view-space units) fed to [`soft_depth_compare`].
    pub soft_z_extent: f32,
    /// Reconstruction gather taps per pixel.
    pub sample_count: u32,
    /// Fraction of the frame the shutter is open, fed to [`shutter_velocity`].
    pub exposure_fraction: f32,
}

impl Default for MotionBlurParams {
    fn default() -> Self {
        Self {
            enabled: false,
            max_velocity_px: 64.0,
            soft_z_extent: 1.0,
            sample_count: 16,
            exposure_fraction: 0.5,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) {
        assert!((a - b).abs() < 1.0e-5, "{a} !~= {b}");
    }

    #[test]
    fn velocity_length_is_pythagorean() {
        approx(velocity_length([3.0, 4.0]), 5.0);
    }

    #[test]
    fn velocity_length_zero_is_zero() {
        approx(velocity_length([0.0, 0.0]), 0.0);
    }

    #[test]
    fn shutter_scales_by_exposure_fraction() {
        let v = shutter_velocity([10.0, -6.0], 0.5);
        approx(v[0], 5.0);
        approx(v[1], -3.0);
    }

    #[test]
    fn clamp_velocity_below_budget_is_unchanged() {
        let v = clamp_velocity([3.0, 4.0], 10.0);
        approx(v[0], 3.0);
        approx(v[1], 4.0);
    }

    #[test]
    fn clamp_velocity_above_budget_scales_to_budget_preserving_direction() {
        let v = clamp_velocity([3.0, 4.0], 2.5);
        // Original length 5 -> scaled to 2.5, halving each component.
        approx(v[0], 1.5);
        approx(v[1], 2.0);
        approx(velocity_length(v), 2.5);
    }

    #[test]
    fn clamp_velocity_nonpositive_budget_is_zero() {
        let v = clamp_velocity([3.0, 4.0], 0.0);
        approx(v[0], 0.0);
        approx(v[1], 0.0);
        let n = clamp_velocity([3.0, 4.0], -1.0);
        approx(n[0], 0.0);
        approx(n[1], 0.0);
    }

    #[test]
    fn tile_max_picks_longest_vector() {
        let vs = [[1.0, 0.0], [0.0, 3.0], [2.0, 2.0]];
        let m = tile_max(&vs);
        approx(m[0], 0.0);
        approx(m[1], 3.0);
    }

    #[test]
    fn tile_max_empty_is_zero() {
        let m = tile_max(&[]);
        approx(m[0], 0.0);
        approx(m[1], 0.0);
    }

    #[test]
    fn neighbor_max_picks_longest_of_nine() {
        let mut tiles = [[0.0_f32, 0.0_f32]; 9];
        tiles[2] = [1.0, 0.0];
        tiles[7] = [0.0, 5.0];
        let m = neighbor_max(&tiles);
        approx(m[0], 0.0);
        approx(m[1], 5.0);
    }

    #[test]
    fn soft_depth_compare_front_is_one() {
        // za well in front (smaller depth) of zb.
        approx(soft_depth_compare(1.0, 5.0, 1.0), 1.0);
    }

    #[test]
    fn soft_depth_compare_behind_is_zero() {
        approx(soft_depth_compare(5.0, 1.0, 1.0), 0.0);
    }

    #[test]
    fn soft_depth_compare_equal_is_one() {
        approx(soft_depth_compare(3.0, 3.0, 1.0), 1.0);
    }

    #[test]
    fn soft_depth_compare_within_extent_is_partial() {
        // za = zb + 0.5, extent 1 -> 1 - 0.5 = 0.5.
        let w = soft_depth_compare(3.5, 3.0, 1.0);
        approx(w, 0.5);
    }

    #[test]
    fn cone_full_at_zero_distance() {
        approx(cone(0.0, 4.0), 1.0);
    }

    #[test]
    fn cone_zero_at_or_beyond_speed() {
        approx(cone(4.0, 4.0), 0.0);
        approx(cone(6.0, 4.0), 0.0);
    }

    #[test]
    fn cone_zero_speed_is_zero() {
        approx(cone(2.0, 0.0), 0.0);
    }

    #[test]
    fn cone_half_distance_is_half() {
        approx(cone(2.0, 4.0), 0.5);
    }

    #[test]
    fn smooth_step_endpoints() {
        approx(smooth_step(0.0, 1.0, 0.0), 0.0);
        approx(smooth_step(0.0, 1.0, 1.0), 1.0);
        approx(smooth_step(0.0, 1.0, 0.5), 0.5);
    }

    #[test]
    fn cylinder_full_inside_speed() {
        // dist well below 0.95 * speed -> smoothstep 0 -> cylinder 1.
        approx(cylinder(1.0, 10.0), 1.0);
    }

    #[test]
    fn cylinder_zero_beyond_speed() {
        // dist above 1.05 * speed -> smoothstep 1 -> cylinder 0.
        approx(cylinder(12.0, 10.0), 0.0);
    }

    #[test]
    fn cylinder_zero_speed_is_zero() {
        approx(cylinder(2.0, 0.0), 0.0);
    }

    #[test]
    fn sample_weight_static_pair_is_zero() {
        // No motion anywhere -> cones and cylinders vanish.
        let w = sample_weight(3.0, 3.0, 2.0, 0.0, 0.0, 1.0);
        approx(w, 0.0);
    }

    #[test]
    fn sample_weight_moving_foreground_covers_center() {
        // Sample in front (smaller depth) and moving fast; centre static.
        let w = sample_weight(5.0, 1.0, 2.0, 0.0, 8.0, 1.0);
        // foreground = 1, cone(2,8) = 0.75; background term 0; cylinders:
        // cylinder(2,8)=1, cylinder(2,0)=0 -> 0. So weight = 0.75.
        approx(w, 0.75);
    }

    #[test]
    fn sample_weight_both_moving_adds_cylinder_term() {
        // Equal depth, both fast, small distance -> cones + doubled cylinder.
        let w = sample_weight(3.0, 3.0, 1.0, 10.0, 10.0, 1.0);
        // foreground = background = 1; cone(1,10) = 0.9 each -> 1.8;
        // cylinder(1,10)=1 each -> 1*1*2 = 2. Total 3.8.
        approx(w, 3.8);
    }

    #[test]
    fn params_default_is_disabled() {
        let p = MotionBlurParams::default();
        assert!(!p.enabled);
        approx(p.max_velocity_px, 64.0);
        approx(p.exposure_fraction, 0.5);
        assert_eq!(p.sample_count, 16);
    }
}
