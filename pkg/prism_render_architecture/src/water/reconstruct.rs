//! Fluid surface reconstruction path selection and per-domain binning.
//!
//! `PBF` and `FLIP`/`APIC` produce a cloud of particles, not a surface. To
//! shade the liquid we must reconstruct an interface from that cloud every
//! frame, and no single method is best everywhere: the right choice trades
//! screen coverage against particle count and quality budget. This module owns
//! that decision as a pure, deterministic classification plus the deterministic
//! fan-out of many domains into per-method buckets, mirroring the raster-bin
//! pattern used elsewhere in the crate.
//!
//! Three reconstruction routes are supported:
//!
//! - *Screen-space* (`van der Laan` style): particle spheres are splatted to a
//!   depth buffer, the depth is bilaterally smoothed, and normals are taken
//!   from the smoothed depth. It is view-dependent and cheap, so it is the
//!   real-time default and the far-field fallback.
//! - *Anisotropic marching cubes* (`Yu-Turk` style): particle positions feed an
//!   anisotropic smoothing kernel whose stretched covariance flattens thin
//!   sheets, then a marching-cubes (`MC`) polygonization extracts a world-space
//!   mesh. It is the highest quality and reserved for the near field.
//!   `MC` is the marching-cubes acronym.
//! - *Narrow-band signed distance field* (`SDF` + `MC`): a signed distance
//!   field is built only in a thin band around the particles and polygonized,
//!   trading some quality for a bounded cost — the mid-field compromise.
//!
//! Everything here is pure and deterministic and uses only `sqrt`; there are no
//! `f32` equality tests and no AI/ML.

use alloc::vec::Vec;

use super::EPS;

/// The surface reconstruction route a fluid domain is drawn with this frame.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReconstructionMethod {
    /// `Yu-Turk` anisotropic-kernel marching cubes: the highest-quality mesh,
    /// reserved for the near field where the surface fills the screen.
    AnisotropicMarchingCubes,
    /// Narrow-band signed distance field polygonized with marching cubes: a
    /// bounded-cost mid-field compromise.
    NarrowBandSdf,
    /// `van der Laan` screen-space depth splat, bilateral smooth, and normal
    /// reconstruction: the cheap, view-dependent real-time default and
    /// far-field fallback.
    ScreenSpace,
}

impl ReconstructionMethod {
    /// A cost rank used to assert monotonic selection: cheaper routes rank
    /// lower. `ScreenSpace` is `0`, `NarrowBandSdf` is `1`, and
    /// `AnisotropicMarchingCubes` is `2`.
    #[must_use]
    pub const fn cost_rank(self) -> u32 {
        match self {
            ReconstructionMethod::ScreenSpace => 0,
            ReconstructionMethod::NarrowBandSdf => 1,
            ReconstructionMethod::AnisotropicMarchingCubes => 2,
        }
    }
}

/// Distance and particle-budget thresholds selecting a reconstruction route.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReconstructionThresholds {
    /// At or below this camera distance the domain uses anisotropic marching
    /// cubes (near field).
    pub near_max_distance: f32,
    /// Above `near_max_distance` and at or below this distance the domain uses
    /// the narrow-band `SDF`; farther domains fall back to screen space.
    pub mid_max_distance: f32,
    /// Domains whose particle count exceeds this are always reconstructed in
    /// screen space regardless of distance, because meshing that many
    /// particles would blow the frame budget.
    pub max_volumetric_particles: u32,
}

/// The per-domain inputs that drive route selection.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReconstructionContext {
    /// Distance from the camera to the domain, in world units (clamped at zero).
    pub camera_distance: f32,
    /// Number of live particles in the domain.
    pub particle_count: u32,
    /// Quality preference in `0..=1`: higher values push the expensive
    /// mesh-based routes out to farther distances.
    pub quality_bias: f32,
}

/// Chooses a reconstruction route for one domain.
///
/// Selection is monotonic in `camera_distance` at fixed particle count and
/// quality bias: as a domain recedes it is never routed to a *more* expensive
/// method, so there is no pop as it moves away. A higher `quality_bias`
/// expands the near/mid distance bands (up to 50% farther at full bias) so the
/// mesh routes hold on longer. A domain with more than
/// `max_volumetric_particles` particles is forced to screen space because
/// meshing that cloud would exceed the budget.
#[must_use]
pub fn select_reconstruction(
    ctx: ReconstructionContext,
    thresholds: ReconstructionThresholds,
) -> ReconstructionMethod {
    if ctx.particle_count > thresholds.max_volumetric_particles {
        return ReconstructionMethod::ScreenSpace;
    }
    let distance = ctx.camera_distance.max(0.0);
    let bias = ctx.quality_bias.clamp(0.0, 1.0);
    // Up to a 50% distance-band expansion at full quality bias.
    let expand = 1.0 + 0.5 * bias;
    let near_max = thresholds.near_max_distance.max(0.0) * expand;
    let mid_max = thresholds.mid_max_distance.max(0.0) * expand;
    if distance <= near_max {
        ReconstructionMethod::AnisotropicMarchingCubes
    } else if distance <= mid_max.max(near_max) {
        ReconstructionMethod::NarrowBandSdf
    } else {
        ReconstructionMethod::ScreenSpace
    }
}

/// Bilateral smoothing iteration count for the screen-space depth pass.
///
/// Screen-space reconstruction needs more smoothing when the surface is close
/// and particles cover many pixels, and less when it is far and sub-pixel. The
/// count fades linearly from `max_iterations` at `near_distance` down to a
/// single pass at or beyond `far_distance`, so it is non-increasing in
/// `camera_distance` and always at least one.
#[must_use]
pub fn screen_space_smoothing_iterations(
    camera_distance: f32,
    max_iterations: u32,
    near_distance: f32,
    far_distance: f32,
) -> u32 {
    let max_iters = max_iterations.max(1);
    let near = near_distance.max(0.0);
    let far = far_distance.max(near + EPS);
    let distance = camera_distance.max(0.0);
    if distance <= near {
        return max_iters;
    }
    if distance >= far {
        return 1;
    }
    let t = (distance - near) / (far - near);
    // Fade from max_iters down toward 1 as t goes 0 -> 1.
    let span = (max_iters - 1) as f32;
    let remaining = span * (1.0 - t);
    1 + (remaining + 0.5) as u32
}

/// Iso-surface level for the mesh-based routes, as a fraction of rest density.
///
/// Marching cubes and the narrow-band `SDF` extract the surface where the
/// smoothed density field crosses this level. Returning `rest_density *
/// iso_fraction` places the interface at a fixed fraction of the fluid's rest
/// density; the fraction is clamped to `0..=1` and the result rises
/// monotonically with it. A lower fraction inflates the surface outward, a
/// higher one pulls it inward toward the dense core.
#[must_use]
pub fn surface_isovalue(rest_density: f32, iso_fraction: f32) -> f32 {
    rest_density.max(0.0) * iso_fraction.clamp(0.0, 1.0)
}

/// Narrow-band half-width, in grid cells, for the `SDF` route.
///
/// The signed distance field is only evaluated within this many cells of the
/// particle surface; outside the band the field is left untouched, which is
/// what bounds the cost. The width scales with the particle influence radius
/// relative to the cell size `dx` and is at least one cell so the band always
/// straddles the interface. It rises monotonically with `particle_radius`.
#[must_use]
pub fn narrow_band_width_cells(particle_radius: f32, dx: f32, band_factor: f32) -> u32 {
    let cell = dx.max(EPS);
    let radius = particle_radius.max(0.0);
    let factor = band_factor.max(0.0);
    let cells = factor * radius / cell;
    // Round up so the band fully covers the influence radius, floor of one.
    let rounded = (cells + 1.0 - EPS) as u32;
    rounded.max(1)
}

/// A batch of fluid domains partitioned by reconstruction route.
///
/// Each bucket holds the domain handles (indices into the caller's parallel
/// context slice) routed to one method, preserving input order so downstream
/// submission is deterministic. The backend consumes one bucket at a time,
/// building a single reconstruction pass per route.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ReconstructionBins {
    /// Domains reconstructed with `Yu-Turk` anisotropic marching cubes.
    pub anisotropic_mc: Vec<u32>,
    /// Domains reconstructed with the narrow-band signed distance field.
    pub narrow_band_sdf: Vec<u32>,
    /// Domains reconstructed in screen space.
    pub screen_space: Vec<u32>,
}

impl ReconstructionBins {
    /// Total number of domains across every bucket.
    #[must_use]
    pub fn total(&self) -> usize {
        self.anisotropic_mc.len() + self.narrow_band_sdf.len() + self.screen_space.len()
    }

    /// Returns `true` when no domain landed in any bucket.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.anisotropic_mc.is_empty()
            && self.narrow_band_sdf.is_empty()
            && self.screen_space.is_empty()
    }

    /// Mutable handle to the bucket backing a reconstruction route.
    fn bucket_mut(&mut self, method: ReconstructionMethod) -> &mut Vec<u32> {
        match method {
            ReconstructionMethod::AnisotropicMarchingCubes => &mut self.anisotropic_mc,
            ReconstructionMethod::NarrowBandSdf => &mut self.narrow_band_sdf,
            ReconstructionMethod::ScreenSpace => &mut self.screen_space,
        }
    }
}

/// Partitions a list of domain handles into per-route reconstruction buckets.
///
/// Each handle in `handles` indexes the parallel `contexts` slice; a handle
/// whose value falls outside `contexts` is skipped rather than panicking, so a
/// stale handle list cannot crash reconstruction. Every in-range handle is
/// classified via [`select_reconstruction`] and pushed onto its bucket in input
/// order, giving a deterministic partition.
#[must_use]
pub fn bin_reconstruction(
    handles: &[u32],
    contexts: &[ReconstructionContext],
    thresholds: ReconstructionThresholds,
) -> ReconstructionBins {
    let mut bins = ReconstructionBins::default();
    for &handle in handles {
        let Some(&ctx) = contexts.get(handle as usize) else {
            continue;
        };
        let method = select_reconstruction(ctx, thresholds);
        bins.bucket_mut(method).push(handle);
    }
    bins
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    const THRESHOLDS: ReconstructionThresholds = ReconstructionThresholds {
        near_max_distance: 10.0,
        mid_max_distance: 50.0,
        max_volumetric_particles: 1_000_000,
    };

    fn ctx(distance: f32, particles: u32, bias: f32) -> ReconstructionContext {
        ReconstructionContext {
            camera_distance: distance,
            particle_count: particles,
            quality_bias: bias,
        }
    }

    #[test]
    fn selection_classifies_near_mid_far() {
        assert_eq!(
            select_reconstruction(ctx(2.0, 1000, 0.0), THRESHOLDS),
            ReconstructionMethod::AnisotropicMarchingCubes
        );
        assert_eq!(
            select_reconstruction(ctx(30.0, 1000, 0.0), THRESHOLDS),
            ReconstructionMethod::NarrowBandSdf
        );
        assert_eq!(
            select_reconstruction(ctx(500.0, 1000, 0.0), THRESHOLDS),
            ReconstructionMethod::ScreenSpace
        );
    }

    #[test]
    fn selection_is_monotonic_in_distance() {
        // As distance grows the route never gets more expensive.
        let mut prev = select_reconstruction(ctx(0.0, 1000, 0.3), THRESHOLDS).cost_rank();
        let mut d = 0.0;
        while d <= 200.0 {
            let rank = select_reconstruction(ctx(d, 1000, 0.3), THRESHOLDS).cost_rank();
            assert!(rank <= prev, "route must not get pricier as it recedes");
            prev = rank;
            d += 1.0;
        }
    }

    #[test]
    fn higher_quality_bias_extends_mesh_bands() {
        // A domain just past the base near band is screen-mid at zero bias but
        // stays on the anisotropic mesh once bias expands the band.
        let just_past_near = ctx(12.0, 1000, 0.0);
        assert_eq!(
            select_reconstruction(just_past_near, THRESHOLDS),
            ReconstructionMethod::NarrowBandSdf
        );
        let biased = ctx(12.0, 1000, 1.0);
        assert_eq!(
            select_reconstruction(biased, THRESHOLDS),
            ReconstructionMethod::AnisotropicMarchingCubes
        );
    }

    #[test]
    fn huge_particle_clouds_force_screen_space() {
        // Even point-blank, an over-budget cloud cannot be meshed.
        let heavy = ctx(0.0, 5_000_000, 1.0);
        assert_eq!(
            select_reconstruction(heavy, THRESHOLDS),
            ReconstructionMethod::ScreenSpace
        );
    }

    #[test]
    fn smoothing_iterations_are_non_increasing_and_bounded() {
        assert_eq!(screen_space_smoothing_iterations(0.0, 8, 2.0, 40.0), 8);
        assert_eq!(screen_space_smoothing_iterations(1.0, 8, 2.0, 40.0), 8);
        assert_eq!(screen_space_smoothing_iterations(1000.0, 8, 2.0, 40.0), 1);
        // Never returns zero even with a zero request.
        assert_eq!(screen_space_smoothing_iterations(0.0, 0, 2.0, 40.0), 1);
        // Monotonic non-increasing across the range.
        let mut prev = screen_space_smoothing_iterations(0.0, 12, 2.0, 40.0);
        let mut d = 0.0;
        while d <= 60.0 {
            let iters = screen_space_smoothing_iterations(d, 12, 2.0, 40.0);
            assert!(iters <= prev, "iterations must not rise with distance");
            assert!(iters >= 1, "at least one smoothing pass");
            prev = iters;
            d += 0.5;
        }
    }

    #[test]
    fn isovalue_is_monotonic_and_clamped() {
        assert!((surface_isovalue(1000.0, 0.5) - 500.0).abs() < EPS);
        // Rises monotonically with the fraction.
        let lo = surface_isovalue(1000.0, 0.2);
        let hi = surface_isovalue(1000.0, 0.8);
        assert!(hi > lo);
        // Out-of-range fractions clamp.
        assert!((surface_isovalue(1000.0, 2.0) - 1000.0).abs() < EPS);
        assert!(surface_isovalue(1000.0, -1.0).abs() < EPS);
        // Negative density floored to zero.
        assert!(surface_isovalue(-5.0, 0.5).abs() < EPS);
    }

    #[test]
    fn narrow_band_width_is_monotonic_and_at_least_one() {
        // A larger particle radius yields a wider band.
        let thin = narrow_band_width_cells(0.5, 1.0, 2.0);
        let thick = narrow_band_width_cells(4.0, 1.0, 2.0);
        assert!(thick >= thin);
        // Always at least one cell.
        assert_eq!(narrow_band_width_cells(0.0, 1.0, 2.0), 1);
        // Degenerate cell size does not divide by zero.
        assert!(narrow_band_width_cells(1.0, 0.0, 2.0) >= 1);
    }

    #[test]
    fn binning_routes_and_skips_out_of_range() {
        let contexts = vec![
            ctx(2.0, 1000, 0.0),   // 0 -> anisotropic
            ctx(30.0, 1000, 0.0),  // 1 -> narrow band
            ctx(500.0, 1000, 0.0), // 2 -> screen space
        ];
        // Handle 9 is out of range and must be skipped, not panic.
        let handles = [0u32, 1, 2, 9];
        let bins = bin_reconstruction(&handles, &contexts, THRESHOLDS);
        assert_eq!(bins.total(), 3);
        assert!(!bins.is_empty());
        assert_eq!(bins.anisotropic_mc, vec![0]);
        assert_eq!(bins.narrow_band_sdf, vec![1]);
        assert_eq!(bins.screen_space, vec![2]);
    }

    #[test]
    fn binning_preserves_input_order_and_is_deterministic() {
        let contexts = vec![ctx(500.0, 1000, 0.0); 4];
        let handles = [3u32, 1, 2, 0];
        let bins = bin_reconstruction(&handles, &contexts, THRESHOLDS);
        assert_eq!(bins.screen_space, vec![3, 1, 2, 0]);
        // Recomputing is identical.
        assert_eq!(bin_reconstruction(&handles, &contexts, THRESHOLDS), bins);
    }

    #[test]
    fn empty_bins_report_empty() {
        let bins = ReconstructionBins::default();
        assert!(bins.is_empty());
        assert_eq!(bins.total(), 0);
    }
}
