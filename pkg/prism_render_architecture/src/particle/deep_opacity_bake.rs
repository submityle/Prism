//! §20 `deep opacity map` **recording / baking** stage — the light-view
//! producer that [`super::volumetrics::sample_deep_transmittance`] consumes.
//!
//! [`super::volumetrics`] owns the *consumption* half of the `deep opacity map`
//! tier: it declares [`super::volumetrics::DeepOpacityLayer`] and samples the
//! piecewise-linear transmittance curve with
//! [`super::volumetrics::sample_deep_transmittance`]. Its doc notes that "the
//! layers themselves are the `Beer-Lambert` integral the recorder already
//! evaluated" — but no recorder existed in this crate. [`super::volume_march`]
//! marches a *view* ray to composite final radiance; it does not emit a
//! reusable light-view transmittance curve. This module fills exactly that gap
//! and nothing else: it walks a ray **from the light's viewpoint**, front-to-
//! back, accumulates transmittance through the density / `froxel` field, and
//! discretizes the result into `N` [`super::volumetrics::DeepOpacityLayer`]
//! samples that the sampler reads back unchanged (design §20, "Deep shadow /
//! deep opacity maps：从光源视角记录沿光线的透过率函数").
//!
//! **Algorithm.** Deep shadow maps were introduced by Lokovic & Veach,
//! "Deep Shadow Maps" (`SIGGRAPH` 2000): per light-ray texel, store a compressed
//! visibility-versus-depth function instead of a single depth. The real-time
//! *deep opacity map* variant (Yuksel & Keyser, "Deep Opacity Maps", `EG` 2008)
//! and the shipping volumetric self-shadow rigs in Guerrilla's `Decima`, Unreal
//! Engine's volumetric fog / cloud shadows, and `EmberGen` all record the same
//! object: a monotonically non-increasing transmittance curve `T(depth)` along
//! the light ray, then sample it cheaply for every in-volume point. This module
//! reproduces the recording end on the `CPU`. It references published algorithms
//! only; it does not vendor any engine source.
//!
//! **No transcendental math — algebraic step opacity replaces `exp`.** The
//! textbook `Beer-Lambert` law `T = exp(-∫σ)` is forbidden by the workspace
//! determinism contract (the only permitted non-arithmetic op is `sqrt`, so the
//! `CPU` reference stays bit-reproducible against a future `GPU` kernel). Instead
//! this module composites transmittance incrementally, exactly matching the
//! per-step recurrence [`super::volume_march`] already uses. Per sub-step of
//! length `ds`:
//!
//! - optical thickness `tau = sigma * ds` (extinction × step length),
//! - step opacity `alpha = tau.clamp(0.0, 1.0)` (a first-order absorber),
//! - transmittance product `transmittance *= 1.0 - alpha`.
//!
//! Over a uniform medium the product telescopes toward the analytic
//! `exp(-σ·d)` as `step_size → 0`, so the recorded curve converges to the
//! physical answer while using only `+ − × ÷`. Layer depths are evenly spaced
//! between the near and far planes and computed by integer multiplication of a
//! fixed span (not repeated addition) so successive bakes are bit-identical.

use alloc::vec::Vec;

use super::volumetrics::{DeepOpacityLayer, FroxelDensityField, EPS};
use super::Vec3;

/// Light-view recorder that bakes a layered `deep opacity map` transmittance
/// curve (design §20).
///
/// Configured with a layer count `N`, a near and far plane bracketing the
/// recorded span along the light ray, and a march `step_size`. Baking marches
/// front-to-back from the near plane to the far plane in sub-steps no longer
/// than `step_size`, accumulating transmittance algebraically, and emits `N`
/// [`DeepOpacityLayer`] samples at evenly spaced depths. Layer `0` always sits
/// at the near plane with transmittance `1.0` (the fully-lit near side), so an
/// empty medium bakes a flat unit curve.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DeepOpacityRecorder {
    /// Number of recorded layers `N` (at least `2`).
    layer_count: u32,
    /// Depth of the first recorded layer along the light ray.
    near_plane: f32,
    /// Depth of the last recorded layer along the light ray.
    far_plane: f32,
    /// Maximum march sub-step length used between adjacent layers.
    step_size: f32,
}

impl DeepOpacityRecorder {
    /// Builds a recorder, validating its configuration.
    ///
    /// Returns `None` when the configuration cannot produce a well-formed,
    /// ascending-depth layer set:
    ///
    /// - `layer_count` must be at least `2` (a near layer plus one deeper
    ///   layer); fewer layers cannot describe a curve.
    /// - the span `far_plane − near_plane` must exceed [`EPS`], so layers never
    ///   collapse to a zero-width range.
    /// - `step_size` must exceed [`EPS`], so the sub-step count stays finite.
    ///
    /// All three inputs must be finite.
    #[must_use]
    pub fn new(layer_count: u32, near_plane: f32, far_plane: f32, step_size: f32) -> Option<Self> {
        if layer_count < 2 {
            return None;
        }
        if !near_plane.is_finite() || !far_plane.is_finite() || !step_size.is_finite() {
            return None;
        }
        if far_plane - near_plane <= EPS || step_size <= EPS {
            return None;
        }
        Some(Self {
            layer_count,
            near_plane,
            far_plane,
            step_size,
        })
    }

    /// The configured number of recorded layers `N`.
    #[must_use]
    pub const fn layer_count(&self) -> u32 {
        self.layer_count
    }

    /// The near-plane depth (depth of layer `0`).
    #[must_use]
    pub const fn near_plane(&self) -> f32 {
        self.near_plane
    }

    /// The far-plane depth (depth of the last layer).
    #[must_use]
    pub const fn far_plane(&self) -> f32 {
        self.far_plane
    }

    /// The configured maximum march sub-step length.
    #[must_use]
    pub const fn step_size(&self) -> f32 {
        self.step_size
    }

    /// The fixed depth span between two adjacent recorded layers.
    ///
    /// Equal to `(far_plane − near_plane) / (layer_count − 1)`. The denominator
    /// is at least `1` because [`DeepOpacityRecorder::new`] rejects a
    /// `layer_count` below `2`.
    #[must_use]
    pub fn layer_span(&self) -> f32 {
        let intervals = (self.layer_count - 1) as f32;
        (self.far_plane - self.near_plane) / intervals
    }

    /// Depth of recorded layer `index`, by integer multiplication of the span.
    ///
    /// Using `near_plane + index · span` (rather than repeated addition) keeps
    /// the depth of a given layer identical across bakes, which the determinism
    /// contract requires.
    #[must_use]
    pub fn layer_depth(&self, index: u32) -> f32 {
        self.near_plane + (index as f32) * self.layer_span()
    }

    /// Bakes the transmittance curve from an extinction-versus-depth profile.
    ///
    /// `extinction_at` maps a depth along the light ray to the local extinction
    /// coefficient `sigma` (a non-negative absorber + out-scatter density); any
    /// negative sample is floored to `0.0`. The returned layers are the core
    /// product the ray and `froxel` bakers both build on.
    ///
    /// Marching is piecewise: each `[layer_depth(i−1), layer_depth(i)]` interval
    /// is split into equal sub-steps no longer than `step_size` and sampled at
    /// each sub-step center (midpoint rule). The accumulated transmittance after
    /// the interval becomes layer `i`. Layer `0` is seeded at the near plane
    /// with transmittance `1.0`, so the result is ordered by ascending depth and
    /// starts fully lit.
    #[must_use]
    pub fn bake_depth_profile<F>(&self, mut extinction_at: F) -> Vec<DeepOpacityLayer>
    where
        F: FnMut(f32) -> f32,
    {
        let count = self.layer_count as usize;
        let mut layers = Vec::with_capacity(count);
        let span = self.layer_span();

        // Layer 0: the fully-lit near side; nothing has absorbed yet.
        let mut transmittance = 1.0_f32;
        layers.push(DeepOpacityLayer::new(self.near_plane, transmittance));

        for layer in 1..self.layer_count {
            let depth_lo = self.near_plane + ((layer - 1) as f32) * span;
            let depth_hi = self.near_plane + (layer as f32) * span;
            let interval = depth_hi - depth_lo;

            // At least one sub-step; cap sub-step length at `step_size`. The
            // `as u32` cast saturates for pathological inputs, staying finite.
            let sub_steps = (interval / self.step_size).ceil().max(1.0) as u32;
            let sub_length = interval / (sub_steps as f32);

            for sub in 0..sub_steps {
                // Midpoint rule: sample at the center of each sub-step. The
                // literal `0.5` is the half-step offset, not a tunable magic
                // number.
                let center = depth_lo + ((sub as f32) + 0.5) * sub_length;
                let sigma = extinction_at(center).max(0.0);
                let tau = sigma * sub_length;
                let alpha = tau.clamp(0.0, 1.0);
                transmittance *= 1.0 - alpha;
            }

            layers.push(DeepOpacityLayer::new(depth_hi, transmittance));
        }

        layers
    }

    /// Bakes the curve by sampling a world-space density field along a light ray.
    ///
    /// `origin` is the ray start (typically the volume's near intersection from
    /// the light) and `direction` points away from the light into the volume; it
    /// is normalized internally (a degenerate zero direction samples a single
    /// point, yielding a flat curve). `density_at_point` returns the extinction
    /// `sigma` at a world position. Depth is measured as arc length, so a
    /// recorded layer depth maps directly to the distance a shadow query uses.
    #[must_use]
    pub fn bake_light_ray<F>(
        &self,
        origin: Vec3,
        direction: Vec3,
        mut density_at_point: F,
    ) -> Vec<DeepOpacityLayer>
    where
        F: FnMut(Vec3) -> f32,
    {
        let unit = direction.normalize_or_zero();
        self.bake_depth_profile(|depth| {
            let point = origin.add(unit.scale(depth));
            density_at_point(point)
        })
    }

    /// Bakes the curve by marching a light ray through a [`FroxelDensityField`].
    ///
    /// Convenience wrapper over [`DeepOpacityRecorder::bake_light_ray`] that
    /// reads the extinction at each sample from the shared `froxel` density grid
    /// the unified volumetric pass already fills (design §20). A sample that
    /// falls outside the grid contributes zero density (fully transmissive),
    /// matching the grid's own out-of-bounds behavior.
    #[must_use]
    pub fn bake_through_field(
        &self,
        field: &FroxelDensityField,
        origin: Vec3,
        direction: Vec3,
    ) -> Vec<DeepOpacityLayer> {
        self.bake_light_ray(origin, direction, |point| {
            let grid = field.grid();
            match grid.cell_index(point) {
                Some(cell) => field.density_at(cell),
                None => 0.0,
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::volumetrics::{sample_deep_transmittance, FroxelGrid};
    use super::*;

    /// Shared absolute tolerance for approximate `f32` comparisons.
    const APPROX_EPS: f32 = 1e-5;

    /// Approximate equality guard; the determinism contract forbids bare `f32`
    /// `==` on computed values.
    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() <= APPROX_EPS
    }

    fn recorder() -> DeepOpacityRecorder {
        // 5 layers across depth 0..=4, sub-step 0.25.
        DeepOpacityRecorder::new(5, 0.0, 4.0, 0.25).expect("valid configuration")
    }

    #[test]
    fn rejects_degenerate_configuration() {
        assert!(DeepOpacityRecorder::new(1, 0.0, 4.0, 0.25).is_none());
        assert!(DeepOpacityRecorder::new(5, 4.0, 4.0, 0.25).is_none());
        assert!(DeepOpacityRecorder::new(5, 0.0, 4.0, 0.0).is_none());
        assert!(DeepOpacityRecorder::new(5, f32::NAN, 4.0, 0.25).is_none());
    }

    #[test]
    fn empty_medium_is_fully_lit() {
        let layers = recorder().bake_depth_profile(|_depth| 0.0);
        assert_eq!(layers.len(), 5);
        for layer in &layers {
            // Zero extinction never attenuates: transmittance stays exactly 1.
            assert!(approx(layer.transmittance, 1.0));
        }
    }

    #[test]
    fn constant_density_decreases_monotonically() {
        // Uniform absorber; each layer must survive strictly less than the last.
        let layers = recorder().bake_depth_profile(|_depth| 0.5);
        for pair in layers.windows(2) {
            assert!(pair[0].transmittance > pair[1].transmittance);
        }
        // The near layer is still the fully-lit side.
        assert!(approx(layers[0].transmittance, 1.0));
        // Everything stays physically bounded in `0..=1`.
        for layer in &layers {
            assert!(layer.transmittance >= 0.0 && layer.transmittance <= 1.0);
        }
    }

    #[test]
    fn layers_are_depth_ordered() {
        let layers = recorder().bake_depth_profile(|_depth| 0.3);
        for pair in layers.windows(2) {
            assert!(pair[1].depth > pair[0].depth);
        }
        // First and last depths match the configured planes.
        assert!(approx(layers[0].depth, 0.0));
        assert!(approx(layers[layers.len() - 1].depth, 4.0));
    }

    #[test]
    fn round_trips_through_sampler() {
        // Baked layers must be consumed verbatim by the existing sampler: a
        // query exactly at a stored depth returns that layer's transmittance.
        let layers = recorder().bake_depth_profile(|_depth| 0.4);
        for layer in &layers {
            let sampled = sample_deep_transmittance(&layers, layer.depth);
            assert!(approx(sampled, layer.transmittance));
        }
        // A midpoint query lands between the two bracketing layers.
        let mid = sample_deep_transmittance(&layers, 1.0);
        assert!(mid <= layers[1].transmittance && mid >= layers[2].transmittance);
    }

    #[test]
    fn bake_is_bit_reproducible() {
        // Two bakes with identical inputs must agree bit-for-bit, not merely
        // approximately: the determinism contract demands reproducibility.
        let first = recorder().bake_depth_profile(|depth| 0.2 + 0.1 * depth);
        let second = recorder().bake_depth_profile(|depth| 0.2 + 0.1 * depth);
        assert_eq!(first.len(), second.len());
        for (lhs, rhs) in first.iter().zip(second.iter()) {
            assert_eq!(lhs.depth.to_bits(), rhs.depth.to_bits());
            assert_eq!(lhs.transmittance.to_bits(), rhs.transmittance.to_bits());
        }
    }

    #[test]
    fn uniform_density_matches_discrete_product() {
        // With a single sub-step per layer the recurrence is exactly
        // `(1 - sigma·span)^i`; verify layer 2 against the hand-computed value.
        let rec = DeepOpacityRecorder::new(5, 0.0, 4.0, 4.0).expect("valid configuration");
        let sigma = 0.1_f32;
        let layers = rec.bake_depth_profile(|_depth| sigma);
        let span = rec.layer_span();
        let step = 1.0 - sigma * span;
        // Layer 2 has absorbed across two spans.
        let expected = step * step;
        assert!(approx(layers[2].transmittance, expected));
    }

    #[test]
    fn ray_bake_tracks_a_slab() {
        // A density slab centered at depth 2 along +Z should attenuate the
        // deeper layers but leave the near layer fully lit.
        let rec = recorder();
        let layers = rec.bake_light_ray(Vec3::ZERO, Vec3::new(0.0, 0.0, 2.0), |point| {
            if point.z >= 1.0 && point.z <= 3.0 {
                0.8
            } else {
                0.0
            }
        });
        assert!(approx(layers[0].transmittance, 1.0));
        assert!(layers[layers.len() - 1].transmittance < 1.0);
    }

    #[test]
    fn froxel_bake_reads_injected_density() {
        let grid = FroxelGrid::new([4, 4, 4], Vec3::ZERO, Vec3::splat(1.0));
        let mut field = FroxelDensityField::new(grid);
        // Fill a column along +Z the light ray will traverse.
        field.inject(Vec3::new(0.5, 0.5, 1.5), 2.0);
        field.inject(Vec3::new(0.5, 0.5, 2.5), 2.0);
        let rec = DeepOpacityRecorder::new(5, 0.0, 4.0, 0.25).expect("valid configuration");
        let origin = Vec3::new(0.5, 0.5, 0.0);
        let layers = rec.bake_through_field(&field, origin, Vec3::new(0.0, 0.0, 1.0));
        // Near side lit, far side attenuated by the injected density.
        assert!(approx(layers[0].transmittance, 1.0));
        assert!(layers[layers.len() - 1].transmittance < 1.0);
        // Still a valid, ordered, bounded curve.
        for pair in layers.windows(2) {
            assert!(pair[1].depth > pair[0].depth);
            assert!(pair[1].transmittance <= pair[0].transmittance);
        }
    }
}
