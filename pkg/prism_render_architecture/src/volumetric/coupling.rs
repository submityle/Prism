#![forbid(unsafe_code)]
//! Cloud-cavity carving and two-way scene coupling (design section 9d).
//!
//! In a next-generation cloudscape the volume is no longer a read-only
//! backdrop: aircraft and projectiles punch holes and trailing wakes into the
//! density field, mountains and buildings occlude the cloud where terrain
//! intersects it, and the cloud shadow feeds back into the ground / aerial
//! perspective. This module owns the deterministic, allocation-free planning
//! primitives for that coupling; the `GPU` `WESL` kernels mirror the same
//! curves with native intrinsics.
//!
//! Two invariants dominate the design. First, the carving brush is a *negative*
//! density source, but it can only ever *remove* density: [`density_delta`]
//! returns a value in `-1..=0` and [`apply_carve`] saturates the sum, so the
//! final density is always `>= 0` and never yields a negative optical depth or
//! `transmittance`. Second, every coupling weight is clamped to `0..=1`, so
//! adversarial positions, radii, or albedos can never panic and never escape
//! the physical range. The only permitted float intrinsic is `f32::sqrt`, which
//! this module reaches only through the shared [`super::Vec3`] distance helper;
//! all smoothing routes through the hand-rolled [`super::math`].

use super::math::{saturate, smoothstep, EPS};
use super::Vec3;

/// Half-height, in world units, of the soft band over which [`terrain_occlusion`]
/// transitions from fully occluded to fully clear above the terrain surface.
///
/// A finite band keeps the terrain-versus-cloud intersection test smooth (no
/// hard step aliasing) while remaining monotonic in the cloud altitude.
const TERRAIN_OCCLUSION_BAND: f32 = 50.0;

/// A spherical negative-density brush that carves cavities and wakes into the
/// cloud density field (design section 9d, "穿云挖洞 / `density carving`").
///
/// The brush is authored deterministically (fixed seed + fixed step upstream)
/// so networked / replay scenarios reproduce identical carving. It only ever
/// subtracts density; see [`density_delta`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CarveBrush {
    /// World-space centre of the spherical brush.
    pub center: Vec3,
    /// Brush radius in world units; values below [`EPS`] are floored so the
    /// falloff never divides by (near) zero.
    pub radius: f32,
    /// Carve strength in `0..=1`; the peak density removed at the centre.
    pub strength: f32,
}

/// Signed density delta the brush applies at world position `pos`.
///
/// The result lies in `-1..=0`: it is `-strength` at the brush centre and rises
/// smoothly to `0` at (and beyond) the brush radius via a [`smoothstep`]
/// falloff. Because the delta is bounded below by `-1` and is always
/// non-positive, applying it through [`apply_carve`] can only remove density —
/// it never manufactures a negative final density or a negative `transmittance`.
/// Out-of-range radii (`<= 0`) and strengths (outside `0..=1`) are guarded, so
/// this function never panics.
#[must_use]
pub fn density_delta(pos: Vec3, brush: CarveBrush) -> f32 {
    let radius = brush.radius.max(EPS);
    let strength = saturate(brush.strength);
    let dist = pos.distance(brush.center);
    // 1 at the centre, easing to 0 at the radius; clamped by smoothstep itself.
    let falloff = 1.0 - smoothstep(0.0, radius, dist);
    -strength * falloff
}

/// Applies a carve `delta` to a base density, clamped into `0..=1`.
///
/// This is the single choke point that guarantees the carving pipeline never
/// produces a negative density (and therefore never a negative optical depth or
/// `transmittance`): the sum is saturated, so a large negative `delta` merely
/// drives the density to `0`.
#[must_use]
pub fn apply_carve(base_density: f32, delta: f32) -> f32 {
    saturate(base_density + delta)
}

/// Terrain occlusion factor for a cloud sample at altitude `cloud_pos_y`.
///
/// Models a mountain or building punching into the cloud layer: samples at or
/// below `terrain_height` are fully occluded (`1`), samples a full
/// [`TERRAIN_OCCLUSION_BAND`] above the surface are fully clear (`0`), and the
/// transition between them is a monotonic non-increasing [`smoothstep`]. The
/// result is always in `0..=1`, deterministic, and panic-free for any input.
#[must_use]
pub fn terrain_occlusion(cloud_pos_y: f32, terrain_height: f32) -> f32 {
    let clear = smoothstep(
        terrain_height,
        terrain_height + TERRAIN_OCCLUSION_BAND,
        cloud_pos_y,
    );
    saturate(1.0 - clear)
}

/// Ground brightness modulation from the cloud shadow (design section 9d,
/// "云影写大气 / `GI`").
///
/// The two-way coupling closes here: the cloud's `transmittance` controls how
/// much sunlight reaches the ground, and the ground albedo controls how much of
/// that light is reflected back to modulate the aerial perspective / `GI`
/// sky-light. The lit-ground factor is the product of the two saturated inputs
/// and therefore stays in `0..=1`; a fully opaque cloud (`transmittance == 0`)
/// drives the ground contribution to `0`.
#[must_use]
pub fn cloud_shadow_modulation(cloud_transmittance: f32, ground_albedo: f32) -> f32 {
    saturate(saturate(ground_albedo) * saturate(cloud_transmittance))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn density_delta_is_bounded_and_non_positive() {
        let brush = CarveBrush {
            center: Vec3::new(1.0, 2.0, 3.0),
            radius: 4.0,
            strength: 0.75,
        };
        // Sweep positions from the centre out past the radius.
        let mut t = 0.0;
        while t <= 8.0 {
            let pos = Vec3::new(1.0 + t, 2.0, 3.0);
            let d = density_delta(pos, brush);
            assert!((-1.0..=0.0).contains(&d), "delta escaped [-1, 0]: {d}");
            t += 0.25;
        }
        // Deepest carve is at the centre, tapering to zero outside the radius.
        let center_delta = density_delta(brush.center, brush);
        let outside_delta = density_delta(Vec3::new(100.0, 2.0, 3.0), brush);
        assert!(
            center_delta <= outside_delta + EPS,
            "centre must carve most"
        );
        assert!(
            outside_delta.abs() < EPS,
            "carve should vanish outside radius"
        );
    }

    #[test]
    fn apply_carve_never_produces_negative_density() {
        // A very strong negative brush must not push density below zero.
        for &base in &[0.0_f32, 0.25, 0.5, 1.0] {
            for &delta in &[-1.0_f32, -0.5, -0.1, 0.0] {
                let d = apply_carve(base, delta);
                assert!(
                    (0.0..=1.0).contains(&d),
                    "carved density escaped range: {d}"
                );
            }
        }
        assert!(
            apply_carve(0.1, -5.0).abs() < EPS,
            "over-carve clamps to zero"
        );
        assert!(
            (apply_carve(2.0, 0.0) - 1.0).abs() < EPS,
            "over-dense clamps to one"
        );
    }

    #[test]
    fn terrain_occlusion_is_deterministic_bounded_and_monotonic() {
        let terrain = 120.0;
        let mut prev = terrain_occlusion(terrain - 100.0, terrain);
        let mut y = terrain - 100.0;
        while y <= terrain + 200.0 {
            let o = terrain_occlusion(y, terrain);
            assert!((0.0..=1.0).contains(&o), "occlusion out of range: {o}");
            // Rising above the terrain can only reduce occlusion.
            assert!(o <= prev + EPS, "occlusion increased with altitude at {y}");
            prev = o;
            y += 5.0;
        }
        // Below terrain is fully occluded; well above is fully clear.
        assert!((terrain_occlusion(terrain - 10.0, terrain) - 1.0).abs() < EPS);
        assert!(terrain_occlusion(terrain + 500.0, terrain).abs() < EPS);
        // Determinism.
        assert_eq!(
            terrain_occlusion(terrain + 10.0, terrain),
            terrain_occlusion(terrain + 10.0, terrain)
        );
    }

    #[test]
    fn cloud_shadow_modulation_stays_in_unit_range() {
        for &tr in &[-0.5_f32, 0.0, 0.3, 1.0, 2.0] {
            for &albedo in &[-1.0_f32, 0.0, 0.4, 1.0, 3.0] {
                let m = cloud_shadow_modulation(tr, albedo);
                assert!((0.0..=1.0).contains(&m), "modulation out of range: {m}");
            }
        }
        // Opaque cloud kills the ground contribution regardless of albedo.
        assert!(cloud_shadow_modulation(0.0, 1.0).abs() < EPS);
        // Fully open sky over bright ground returns the albedo.
        assert!((cloud_shadow_modulation(1.0, 0.8) - 0.8).abs() < EPS);
    }

    #[test]
    fn density_delta_out_of_range_inputs_do_not_panic() {
        let degenerate = CarveBrush {
            center: Vec3::ZERO,
            radius: 0.0,
            strength: 5.0,
        };
        let d = density_delta(Vec3::new(0.0, 0.0, 0.0), degenerate);
        assert!(
            (-1.0..=0.0).contains(&d),
            "degenerate brush escaped range: {d}"
        );
        let far = density_delta(Vec3::new(1000.0, 0.0, 0.0), degenerate);
        assert!(
            far.abs() < EPS,
            "degenerate brush should not reach far points"
        );
    }
}
