//! Backend-neutral CPU golden for industry-standard lateral chromatic
//! aberration (lens colour fringing) post-processing.
//!
//! Lateral chromatic aberration is a shared post-processing effect that runs on
//! the resolved image: a real lens focuses different wavelengths at slightly
//! different magnifications, so the red, green and blue channels of a point
//! away from the optical centre land at slightly different radii. The film-game
//! convention (Unity's Post Processing Stack v2, the Unreal Engine "Scene
//! Fringe" control) reproduces this by sampling each colour channel from a
//! *radially* offset texture coordinate: further from the centre means a larger
//! split, and the split direction is the radial direction from the centre.
//!
//! Because a CPU golden has no real texture to sample, this module exposes the
//! effect as its testable arithmetic pieces — the parts that must stay
//! bit-identical with the GPU twin — rather than a full gather:
//!
//! * **Radial frame.** [`radial_offset`] returns the unit radial direction and
//!   the distance of a `uv` from the aberration `center`. The length uses
//!   `f32::sqrt` (allowed; it is not a disallowed transcendental) and the
//!   direction degrades to the zero vector exactly at the centre, so the centre
//!   is a fixed point.
//! * **Per-channel split.** [`sample_offsets`] gives the signed scalar offset
//!   each channel travels along the radial direction: red rides outward
//!   (`+intensity * dist`), green stays put (`0`), blue rides inward
//!   (`-intensity * dist`). This is the classic three-tap split; the neutral
//!   `intensity = 0` collapses all three to zero.
//! * **Sample coordinate.** [`channel_uv`] turns a channel's scalar offset into
//!   the `uv` that channel should sample, and [`apply_chromatic_aberration`]
//!   returns the three per-channel sample coordinates for a pixel at once.
//! * **Recombine.** [`combine_channels`] takes the three sampled colours and
//!   keeps each channel from its own tap (`[r[0], g[1], b[2]]`), the recombine
//!   step a real gather performs after the three texture reads.
//! * **Spectral variant.** [`spectral_lut`] and [`spectral_offset`] provide the
//!   optional multi-tap spectral form: a smooth polynomial weight ramp over a
//!   sample parameter `t` in `[0, 1]` (blue at `0`, red at `1`) plus the linear
//!   offset that tap should use.
//!
//! Every function is pure polynomial arithmetic plus `f32::sqrt`, so no
//! `bevy_math::ops` transcendental is needed. The whole module is mirrored
//! arm-for-arm by `shaders/chromatic_aberration.wesl` (same operation order,
//! same constants) so the CPU golden and the GPU twin agree.

/// Distance below which the radial direction is treated as undefined and
/// returned as the zero vector, keeping the aberration centre a fixed point.
pub const CHROMATIC_ABERRATION_EPSILON: f32 = 1.0e-8;

/// Number of colour channels the three-tap split addresses (red, green, blue).
pub const CHROMATIC_ABERRATION_CHANNELS: usize = 3;

/// Radial direction and distance of `uv` from the aberration `center`.
///
/// Returns `(dir, dist)` where `dist = length(uv - center)` (via `f32::sqrt`)
/// and `dir` is that difference normalised. At (or within
/// [`CHROMATIC_ABERRATION_EPSILON`] of) the centre the direction is undefined,
/// so `dir` is the zero vector and the centre stays a fixed point of the whole
/// effect.
#[must_use]
pub fn radial_offset(uv: [f32; 2], center: [f32; 2]) -> ([f32; 2], f32) {
    let dx = uv[0] - center[0];
    let dy = uv[1] - center[1];
    let dist = (dx * dx + dy * dy).sqrt();
    if dist > CHROMATIC_ABERRATION_EPSILON {
        ([dx / dist, dy / dist], dist)
    } else {
        ([0.0, 0.0], dist)
    }
}

/// Signed per-channel scalar offset along the radial direction.
///
/// The three-tap convention: red rides outward (`+intensity * dist`), green
/// stays put (`0`), blue rides inward (`-intensity * dist`). Indexed by channel
/// as `[red, green, blue]`. The neutral `intensity = 0` returns all zeros.
#[must_use]
pub fn sample_offsets(dist: f32, intensity: f32) -> [f32; 3] {
    let split = intensity * dist;
    [split, 0.0, -split]
}

/// Sample coordinate for one colour channel.
///
/// Advances `uv` along the radial `dir` by the channel's scalar offset (from
/// [`sample_offsets`]) for `channel_index` in `0..3` (red, green, blue). Green
/// (index `1`) always returns `uv` unchanged.
#[must_use]
pub fn channel_uv(
    uv: [f32; 2],
    dir: [f32; 2],
    dist: f32,
    intensity: f32,
    channel_index: usize,
) -> [f32; 2] {
    let offsets = sample_offsets(dist, intensity);
    let o = offsets[channel_index];
    [uv[0] + dir[0] * o, uv[1] + dir[1] * o]
}

/// Recombine three per-channel samples into one colour.
///
/// A real gather reads the scene three times (once per channel offset) and
/// keeps each channel from its own read: `[r[0], g[1], b[2]]`. When the three
/// samples are equal this is the identity, so a zero split reproduces the
/// input.
#[must_use]
pub fn combine_channels(r_sample: [f32; 3], g_sample: [f32; 3], b_sample: [f32; 3]) -> [f32; 3] {
    [r_sample[0], g_sample[1], b_sample[2]]
}

/// Smooth polynomial spectral weight ramp for the optional multi-tap form.
///
/// Maps a sample parameter `t` in `[0, 1]` to `[red, green, blue]` weights with
/// blue peaking at `t = 0`, red at `t = 1` and green in the middle. Pure
/// quadratics (`t * t`, `4t(1 - t)`, `(1 - t)^2`) so the GPU twin mirrors it
/// exactly with no transcendental.
#[must_use]
pub fn spectral_lut(t: f32) -> [f32; 3] {
    let red = t * t;
    let green = 4.0 * t * (1.0 - t);
    let one_minus = 1.0 - t;
    let blue = one_minus * one_minus;
    [red, green, blue]
}

/// Signed scalar offset for a spectral tap at parameter `t` in `[0, 1]`.
///
/// Linearly ramps from inward (`-intensity * dist` at `t = 0`, blue) to outward
/// (`+intensity * dist` at `t = 1`, red), matching the three-tap endpoints from
/// [`sample_offsets`]. The mid tap `t = 0.5` sits at zero offset.
#[must_use]
pub fn spectral_offset(dist: f32, intensity: f32, t: f32) -> f32 {
    intensity * dist * (2.0 * t - 1.0)
}

/// Artist controls for the aberration (the neutral defaults are identity).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChromaticAberrationParams {
    /// Split strength; `0` disables the effect (all channel offsets vanish).
    pub intensity: f32,
    /// Optical centre in `uv` space; the radial split fans out from here.
    pub center: [f32; 2],
    /// Spectral tap count for the multi-tap form (three-tap uses the endpoints).
    pub samples: u32,
}

impl Default for ChromaticAberrationParams {
    /// Neutral controls: `intensity = 0` (disabled), centre at the image middle,
    /// three spectral taps. With `intensity = 0` every channel offset is zero,
    /// so [`apply_chromatic_aberration`] returns three copies of the input `uv`.
    fn default() -> Self {
        Self {
            intensity: 0.0,
            center: [0.5, 0.5],
            samples: 3,
        }
    }
}

/// Per-channel sample coordinates for a pixel at `uv`.
///
/// Returns `[red_uv, green_uv, blue_uv]`: the three texture coordinates a
/// gather would read to build the fringed pixel. Exposes the offset->uv
/// conversion so it is directly testable. With [`ChromaticAberrationParams::default`]
/// (`intensity = 0`) all three equal `uv`, i.e. the identity.
#[must_use]
pub fn apply_chromatic_aberration(uv: [f32; 2], params: &ChromaticAberrationParams) -> [[f32; 2]; 3] {
    let (dir, dist) = radial_offset(uv, params.center);
    [
        channel_uv(uv, dir, dist, params.intensity, 0),
        channel_uv(uv, dir, dist, params.intensity, 1),
        channel_uv(uv, dir, dist, params.intensity, 2),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1.0e-6;

    fn approx(a: f32, b: f32) {
        assert!((a - b).abs() <= EPS, "expected {b}, got {a}");
    }

    fn approx2(a: [f32; 2], b: [f32; 2]) {
        approx(a[0], b[0]);
        approx(a[1], b[1]);
    }

    fn approx3(a: [f32; 3], b: [f32; 3]) {
        approx(a[0], b[0]);
        approx(a[1], b[1]);
        approx(a[2], b[2]);
    }

    // --- radial_offset ---

    #[test]
    fn radial_offset_at_center_is_zero_direction() {
        let (dir, dist) = radial_offset([0.5, 0.5], [0.5, 0.5]);
        approx2(dir, [0.0, 0.0]);
        approx(dist, 0.0);
    }

    #[test]
    fn radial_offset_within_epsilon_is_zero_direction() {
        let (dir, _) = radial_offset([0.5 + 1.0e-9, 0.5], [0.5, 0.5]);
        approx2(dir, [0.0, 0.0]);
    }

    #[test]
    fn radial_offset_direction_is_unit_length() {
        let (dir, _) = radial_offset([0.9, 0.7], [0.5, 0.5]);
        let len = (dir[0] * dir[0] + dir[1] * dir[1]).sqrt();
        approx(len, 1.0);
    }

    #[test]
    fn radial_offset_distance_matches_length() {
        let (_, dist) = radial_offset([0.5 + 0.3, 0.5 + 0.4], [0.5, 0.5]);
        approx(dist, 0.5); // 3-4-5 triangle
    }

    #[test]
    fn radial_offset_axis_aligned_direction() {
        let (dir, dist) = radial_offset([1.0, 0.5], [0.5, 0.5]);
        approx2(dir, [1.0, 0.0]);
        approx(dist, 0.5);
    }

    // --- sample_offsets ---

    #[test]
    fn sample_offsets_neutral_intensity_is_zero() {
        approx3(sample_offsets(0.42, 0.0), [0.0, 0.0, 0.0]);
    }

    #[test]
    fn sample_offsets_red_positive_blue_negative() {
        let o = sample_offsets(2.0, 0.1);
        assert!(o[0] > 0.0, "red should ride outward");
        assert!(o[2] < 0.0, "blue should ride inward");
        approx(o[0], -o[2]);
    }

    #[test]
    fn sample_offsets_green_is_always_zero() {
        for &dist in &[0.0, 0.5, 3.0] {
            for &intensity in &[0.0, 0.2, 1.5] {
                approx(sample_offsets(dist, intensity)[1], 0.0);
            }
        }
    }

    #[test]
    fn sample_offsets_scales_with_distance() {
        let near = sample_offsets(1.0, 0.1);
        let far = sample_offsets(2.0, 0.1);
        approx(far[0], 2.0 * near[0]);
    }

    // --- channel_uv ---

    #[test]
    fn channel_uv_green_is_identity() {
        let uv = [0.7, 0.3];
        let (dir, dist) = radial_offset(uv, [0.5, 0.5]);
        approx2(channel_uv(uv, dir, dist, 0.5, 1), uv);
    }

    #[test]
    fn channel_uv_red_shifts_along_direction() {
        let uv = [1.0, 0.5];
        let (dir, dist) = radial_offset(uv, [0.5, 0.5]); // dir = (1, 0), dist = 0.5
        let red = channel_uv(uv, dir, dist, 0.2, 0);
        approx2(red, [1.0 + 0.2 * 0.5, 0.5]);
    }

    #[test]
    fn channel_uv_blue_shifts_opposite_to_red() {
        let uv = [1.0, 0.5];
        let (dir, dist) = radial_offset(uv, [0.5, 0.5]);
        let red = channel_uv(uv, dir, dist, 0.2, 0);
        let blue = channel_uv(uv, dir, dist, 0.2, 2);
        // Red and blue are mirror images about the source uv.
        approx2([(red[0] + blue[0]) * 0.5, (red[1] + blue[1]) * 0.5], uv);
    }

    // --- combine_channels ---

    #[test]
    fn combine_channels_picks_per_channel() {
        let out = combine_channels([0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]);
        approx3(out, [0.1, 0.5, 0.9]);
    }

    #[test]
    fn combine_channels_identity_when_equal() {
        let c = [0.25, 0.5, 0.75];
        approx3(combine_channels(c, c, c), c);
    }

    // --- spectral_lut ---

    #[test]
    fn spectral_lut_endpoints() {
        approx3(spectral_lut(0.0), [0.0, 0.0, 1.0]); // pure blue
        approx3(spectral_lut(1.0), [1.0, 0.0, 0.0]); // pure red
    }

    #[test]
    fn spectral_lut_midpoint_peaks_green() {
        let mid = spectral_lut(0.5);
        approx3(mid, [0.25, 1.0, 0.25]);
        assert!(mid[1] > mid[0] && mid[1] > mid[2], "green should peak mid-band");
    }

    // --- spectral_offset ---

    #[test]
    fn spectral_offset_endpoints_match_three_tap() {
        let three = sample_offsets(2.0, 0.1);
        approx(spectral_offset(2.0, 0.1, 0.0), three[2]); // blue end
        approx(spectral_offset(2.0, 0.1, 1.0), three[0]); // red end
    }

    #[test]
    fn spectral_offset_center_is_zero() {
        approx(spectral_offset(3.0, 0.5, 0.5), 0.0);
    }

    // --- params + apply ---

    #[test]
    fn params_default_is_neutral() {
        let p = ChromaticAberrationParams::default();
        approx(p.intensity, 0.0);
        approx2(p.center, [0.5, 0.5]);
        assert_eq!(p.samples, 3);
    }

    #[test]
    fn apply_default_is_identity() {
        let params = ChromaticAberrationParams::default();
        let uv = [0.8, 0.2];
        let samples = apply_chromatic_aberration(uv, &params);
        approx2(samples[0], uv);
        approx2(samples[1], uv);
        approx2(samples[2], uv);
    }

    #[test]
    fn apply_at_center_is_identity() {
        let params = ChromaticAberrationParams {
            intensity: 0.5,
            center: [0.5, 0.5],
            samples: 3,
        };
        let samples = apply_chromatic_aberration([0.5, 0.5], &params);
        approx2(samples[0], [0.5, 0.5]);
        approx2(samples[1], [0.5, 0.5]);
        approx2(samples[2], [0.5, 0.5]);
    }

    #[test]
    fn apply_splits_red_and_blue_opposite() {
        let params = ChromaticAberrationParams {
            intensity: 0.2,
            center: [0.5, 0.5],
            samples: 3,
        };
        let uv = [1.0, 0.5];
        let samples = apply_chromatic_aberration(uv, &params);
        // Green untouched, red/blue mirror about the source uv.
        approx2(samples[1], uv);
        approx2([(samples[0][0] + samples[2][0]) * 0.5, (samples[0][1] + samples[2][1]) * 0.5], uv);
        assert!(samples[0][0] > uv[0], "red should push outward from centre");
        assert!(samples[2][0] < uv[0], "blue should pull inward toward centre");
    }

    #[test]
    fn apply_matches_hand_composed_pipeline() {
        let params = ChromaticAberrationParams {
            intensity: 0.15,
            center: [0.4, 0.6],
            samples: 3,
        };
        let uv = [0.9, 0.1];
        let (dir, dist) = radial_offset(uv, params.center);
        let expected = [
            channel_uv(uv, dir, dist, params.intensity, 0),
            channel_uv(uv, dir, dist, params.intensity, 1),
            channel_uv(uv, dir, dist, params.intensity, 2),
        ];
        let got = apply_chromatic_aberration(uv, &params);
        approx2(got[0], expected[0]);
        approx2(got[1], expected[1]);
        approx2(got[2], expected[2]);
    }
}
