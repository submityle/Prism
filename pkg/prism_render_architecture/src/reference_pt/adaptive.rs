//! Variance-driven adaptive sampling for the reference tracer.
//!
//! A fixed samples-per-pixel budget wastes work: smooth, well-converged pixels
//! keep drawing samples they no longer need while noisy pixels (glossy
//! highlights, caustic-like paths, small bright emitters) stay grainy. Adaptive
//! sampling instead measures each pixel's own error and keeps sampling only
//! while that error is above a target, spending the budget where it reduces
//! visible noise the most. This mirrors the adaptive samplers in production
//! offline renderers.
//!
//! The error metric is the standard error of the mean luminance, tracked online
//! with Welford's algorithm (one pass, numerically stable, using only `+`, `-`,
//! `*`, and `/`). Luminance is a linear combination of the channels, so the
//! luminance of the running `RGB` mean equals the running mean of the per-sample
//! luminances; a single `RGB` accumulator therefore drives the scalar variance
//! recurrence without a second pass or a separate luminance mean.
//!
//! Sampling stays unbiased: the per-pixel estimate is always the plain average
//! of the radiance samples drawn. Only *how many* samples each pixel receives is
//! data dependent, and that decision depends solely on the accumulated
//! statistics, so the whole render remains deterministic and bit-identical for
//! identical arguments (each pixel owns an independent, index-keyed `RNG` stream
//! and Halton sequence exactly as in [`super::film`]).

use alloc::vec;
use alloc::vec::Vec;

use super::camera::PinholeCamera;
use super::film::Film;
use super::filter::PixelFilter;
use super::firefly::FireflyClamp;
use super::halton::HaltonPixelSampler;
use super::integrator::{PathIntegrator, Scene};
use super::sampler::Rng;
use super::Vec3;

/// Luma weight for the red channel (standard luma coefficients, summing to one
/// so a unit-white sample has unit luminance).
const LUMA_R: f32 = 0.2126;
/// Luma weight for the green channel (see [`LUMA_R`]).
const LUMA_G: f32 = 0.7152;
/// Luma weight for the blue channel (see [`LUMA_R`]).
const LUMA_B: f32 = 0.0722;

/// Brightness floor used in the relative convergence target so that near-black
/// pixels are judged against a small reference luminance instead of their own
/// vanishing mean (which would make the relative target unreachable and force
/// every dark pixel to the maximum sample count).
const CONVERGENCE_FLOOR: f32 = 1.0e-2;

/// Relative luminance of a linear `RGB` radiance value.
fn luminance(value: Vec3) -> f32 {
    LUMA_R * value.x + LUMA_G * value.y + LUMA_B * value.z
}

/// Online per-pixel radiance statistics (Welford's algorithm).
///
/// Accumulates the running mean radiance per channel and the running sum of
/// squared luminance deviations, which together yield the sample variance and
/// the standard error of the mean luminance without storing individual samples.
#[derive(Clone, Copy, Debug, Default)]
pub struct VarianceEstimator {
    /// Running mean radiance per channel.
    mean: Vec3,
    /// Running sum of squared luminance deviations (Welford's `M2`).
    m2_luminance: f32,
    /// Number of samples accumulated so far.
    count: u32,
}

impl VarianceEstimator {
    /// Creates an empty estimator with no samples.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            mean: Vec3::ZERO,
            m2_luminance: 0.0,
            count: 0,
        }
    }

    /// Folds one radiance sample into the running mean and variance.
    pub fn add_sample(&mut self, value: Vec3) {
        self.count += 1;
        let inv_count = 1.0 / (self.count as f32);
        // Luminance of the mean before and after the update; because luminance
        // is linear these equal the running mean luminance pre/post update, so
        // the scalar Welford recurrence stays consistent with the RGB mean.
        let luma_before = luminance(self.mean);
        let delta = value.sub(self.mean);
        self.mean = self.mean.add(delta.scale(inv_count));
        let luma_after = luminance(self.mean);
        let luma_sample = luminance(value);
        self.m2_luminance += (luma_sample - luma_before) * (luma_sample - luma_after);
    }

    /// Number of samples accumulated so far.
    #[must_use]
    pub const fn count(&self) -> u32 {
        self.count
    }

    /// The current mean radiance estimate (the plain sample average).
    #[must_use]
    pub fn mean(&self) -> Vec3 {
        self.mean
    }

    /// Unbiased sample variance of the per-sample luminance.
    ///
    /// Returns zero with fewer than two samples (variance is undefined for a
    /// single observation).
    #[must_use]
    pub fn luminance_variance(&self) -> f32 {
        if self.count < 2 {
            return 0.0;
        }
        self.m2_luminance / ((self.count - 1) as f32)
    }

    /// Standard error of the estimated mean luminance (`sqrt(variance / n)`).
    ///
    /// Returns zero when there are no samples.
    #[must_use]
    pub fn luminance_std_error(&self) -> f32 {
        if self.count == 0 {
            return 0.0;
        }
        (self.luminance_variance() / (self.count as f32)).sqrt()
    }

    /// Whether the mean luminance estimate has reached the relative error
    /// target.
    ///
    /// The pixel is considered converged once the standard error of its mean
    /// luminance falls to or below `relative_tolerance` times its mean
    /// luminance, with a small absolute [`CONVERGENCE_FLOOR`] so near-black
    /// pixels terminate on absolute rather than relative error. Always returns
    /// `false` with fewer than two samples, since the error estimate needs at
    /// least two observations.
    #[must_use]
    pub fn has_converged(&self, relative_tolerance: f32) -> bool {
        if self.count < 2 {
            return false;
        }
        let reference = luminance(self.mean).max(CONVERGENCE_FLOOR);
        self.luminance_std_error() <= relative_tolerance * reference
    }
}

/// Sampling budget and stopping policy for [`render_adaptive`].
#[derive(Clone, Copy, Debug)]
pub struct AdaptiveConfig {
    /// Minimum samples every pixel receives before convergence is tested. A
    /// floor is needed because the variance estimate is unreliable from only a
    /// handful of samples.
    pub min_samples: u32,
    /// Hard cap on samples per pixel, bounding the worst-case cost of a pixel
    /// that never meets the error target.
    pub max_samples: u32,
    /// Target relative standard error of the mean luminance; smaller values ask
    /// for a cleaner image at higher cost.
    pub relative_tolerance: f32,
    /// How many samples to draw between convergence checks. Batching amortises
    /// the test and lets the variance estimate settle; it never changes the
    /// result beyond the sample count rounding to a batch boundary.
    pub batch_size: u32,
    /// Per-sample firefly clamp applied to each radiance sample before it is
    /// accumulated. The default [`FireflyClamp::Off`] keeps the estimate
    /// exactly unbiased and bit-identical to the fixed-budget renderer; an
    /// opt-in luminance clamp trades a small bounded bias for far fewer
    /// isolated bright outlier pixels.
    pub firefly: FireflyClamp,
}

impl Default for AdaptiveConfig {
    /// A balanced default: at least 16 samples, at most 256, a 5% relative error
    /// target, checked every 16 samples.
    fn default() -> Self {
        Self {
            min_samples: 16,
            max_samples: 256,
            relative_tolerance: 0.05,
            batch_size: 16,
            firefly: FireflyClamp::Off,
        }
    }
}

/// The result of an adaptive render: the image plus the per-pixel sample count
/// that produced it.
#[derive(Clone, Debug)]
pub struct AdaptiveRender {
    /// The rendered image.
    pub film: Film,
    /// Samples spent on each pixel, row-major (`index = y * width + x`). Useful
    /// as a diagnostic heat map of where the sampler concentrated effort.
    pub sample_counts: Vec<u32>,
}

/// Renders `scene` through `camera` with `integrator`, drawing between
/// `config.min_samples` and `config.max_samples` jittered primary rays per
/// pixel and stopping each pixel early once its mean luminance meets the
/// configured relative error target.
///
/// Behaves exactly like [`super::film::render_filtered`] except for the
/// per-pixel sample count: the sub-pixel jitter is the same filter-warped
/// Halton sequence, the camera engages the thin-lens model when its aperture is
/// positive, and each pixel owns an independent index-keyed `RNG` stream. The
/// returned [`AdaptiveRender::sample_counts`] records how many samples each
/// pixel actually used. A zero-sized film or a `max_samples` of zero yields a
/// cleared image.
#[must_use]
#[expect(
    clippy::too_many_arguments,
    reason = "the renderer needs the scene, camera, integrator, image size, sampling budget, seed, and reconstruction filter"
)]
pub fn render_adaptive(
    scene: &Scene,
    camera: &PinholeCamera,
    integrator: &PathIntegrator,
    width: u32,
    height: u32,
    config: AdaptiveConfig,
    seed: u64,
    filter: PixelFilter,
) -> AdaptiveRender {
    let count = (width as usize) * (height as usize);
    let mut pixels = vec![Vec3::ZERO; count];
    let mut sample_counts = vec![0u32; count];
    if width == 0 || height == 0 || config.max_samples == 0 {
        return AdaptiveRender {
            film: Film::from_pixels(width, height, pixels),
            sample_counts,
        };
    }
    // Clamp the floor into the budget and keep at least one sample per batch.
    let min_samples = config.min_samples.max(1).min(config.max_samples);
    let batch = config.batch_size.max(1);
    for y in 0..height {
        for x in 0..width {
            let index = (y as usize) * (width as usize) + (x as usize);
            // Per-pixel streams keyed by the flat index keep the render
            // deterministic and order-independent, matching the fixed-budget
            // renderer so a pixel's samples are identical for a given index.
            let mut rng = Rng::with_stream(seed, index as u64 + 1);
            let jitter_sampler = HaltonPixelSampler::new(seed, index as u64);
            let mut estimator = VarianceEstimator::new();
            let mut s: u32 = 0;
            while s < config.max_samples {
                let jitter = filter.warp(jitter_sampler.sample(u64::from(s)));
                let ray = if camera.aperture_radius() > 0.0 {
                    camera.primary_ray_lens(x, y, width, height, jitter, &mut rng)
                } else {
                    camera.primary_ray(x, y, width, height, jitter)
                };
                let radiance = integrator.radiance(scene, ray, &mut rng);
                estimator.add_sample(config.firefly.apply(radiance));
                s += 1;
                // Test convergence only on batch boundaries and never before the
                // minimum-sample floor, so a brief run of similar early samples
                // cannot stop a pixel prematurely.
                if s >= min_samples
                    && s.is_multiple_of(batch)
                    && estimator.has_converged(config.relative_tolerance)
                {
                    break;
                }
            }
            pixels[index] = estimator.mean();
            sample_counts[index] = estimator.count();
        }
    }
    AdaptiveRender {
        film: Film::from_pixels(width, height, pixels),
        sample_counts,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ray_scene::triangle_mesh::{TriangleMesh, TriangleMeshBvh};
    use crate::reference_pt::bsdf::Bsdf;
    use crate::reference_pt::integrator::Material;

    /// A large upward-facing floor triangle in the `y = 0` plane.
    fn ground_plane() -> TriangleMeshBvh {
        let positions = alloc::vec![
            [-10.0_f32, 0.0, -10.0],
            [10.0, 0.0, -10.0],
            [0.0, 0.0, 10.0],
        ];
        let indices = alloc::vec![[0u32, 1, 2]];
        let mesh = TriangleMesh::new(positions, alloc::vec![], alloc::vec![], indices)
            .expect("valid ground triangle");
        TriangleMeshBvh::build(mesh)
    }

    /// A white-furnace scene: an albedo-one floor under unit environment.
    fn white_furnace_scene() -> Scene {
        let materials = alloc::vec![Material::new(Bsdf::Lambert { albedo: Vec3::ONE })];
        Scene::new(ground_plane(), materials, alloc::vec![], Vec3::ONE)
            .expect("white furnace scene")
    }

    /// A floor plus a small overhead downward-facing emissive triangle, built as
    /// one mesh so the glowing triangle lives in the `BVH` with the floor.
    fn floor_with_overhead_emitter() -> Scene {
        let positions = alloc::vec![
            [-10.0_f32, 0.0, -10.0],
            [10.0, 0.0, -10.0],
            [0.0, 0.0, 10.0],
            [-0.5, 4.0, -0.5],
            [0.5, 4.0, -0.5],
            [-0.5, 4.0, 0.5],
        ];
        let indices = alloc::vec![[0u32, 1, 2], [3, 4, 5]];
        let mesh = TriangleMesh::new(positions, alloc::vec![], alloc::vec![], indices)
            .expect("valid floor + emitter mesh");
        let materials = alloc::vec![
            Material::new(Bsdf::Lambert {
                albedo: Vec3::splat(0.8),
            }),
            Material::emissive(Bsdf::Lambert { albedo: Vec3::ZERO }, Vec3::splat(8.0)),
        ];
        Scene::new(
            TriangleMeshBvh::build(mesh),
            materials,
            alloc::vec![],
            Vec3::ZERO,
        )
        .expect("emitter scene")
    }

    /// A camera above the origin looking straight down at the floor.
    fn overhead_camera() -> PinholeCamera {
        PinholeCamera::look_at(
            Vec3::new(0.0, 3.0, 0.0),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, -1.0),
            0.5,
            1.0,
        )
        .expect("valid overhead camera")
    }

    #[test]
    fn welford_matches_two_pass_variance() {
        // Feed grey samples (luminance equals the channel value, since the luma
        // weights sum to one) and compare the online variance to an independent
        // two-pass computation.
        let mut rng = Rng::with_stream(7, 1);
        let mut estimator = VarianceEstimator::new();
        let mut values = Vec::new();
        for _ in 0..512 {
            let v = rng.next_f32();
            values.push(v);
            estimator.add_sample(Vec3::splat(v));
        }
        let n = values.len() as f32;
        let mean = values.iter().copied().sum::<f32>() / n;
        let two_pass = values.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / (n - 1.0);
        assert!(
            (estimator.luminance_variance() - two_pass).abs() < 1.0e-4,
            "online variance {} vs two-pass {two_pass}",
            estimator.luminance_variance()
        );
        assert!(
            (estimator.mean().x - mean).abs() < 1.0e-5,
            "online mean {} vs arithmetic mean {mean}",
            estimator.mean().x
        );
    }

    #[test]
    fn constant_input_has_zero_variance_and_converges() {
        let mut estimator = VarianceEstimator::new();
        for _ in 0..32 {
            estimator.add_sample(Vec3::splat(0.5));
        }
        assert!(
            estimator.luminance_variance() < 1.0e-6,
            "constant samples must have ~zero variance, got {}",
            estimator.luminance_variance()
        );
        assert!(
            estimator.has_converged(0.05),
            "a zero-variance pixel must be reported converged"
        );
    }

    #[test]
    fn high_variance_input_does_not_converge() {
        // Alternating black and unit-white samples: mean luminance 0.5 with large
        // variance, so the relative error stays well above a 5% target.
        let mut estimator = VarianceEstimator::new();
        for i in 0..32 {
            let v = if i % 2 == 0 { 0.0 } else { 1.0 };
            estimator.add_sample(Vec3::splat(v));
        }
        assert!(
            !estimator.has_converged(0.05),
            "a high-variance pixel must not be reported converged"
        );
    }

    #[test]
    fn render_adaptive_is_deterministic() {
        let scene = floor_with_overhead_emitter();
        let camera = overhead_camera();
        let integrator = PathIntegrator::new(4, 3);
        let config = AdaptiveConfig {
            min_samples: 8,
            max_samples: 64,
            relative_tolerance: 0.05,
            batch_size: 8,
            firefly: FireflyClamp::Off,
        };
        let a = render_adaptive(
            &scene,
            &camera,
            &integrator,
            6,
            6,
            config,
            11,
            PixelFilter::Box,
        );
        let b = render_adaptive(
            &scene,
            &camera,
            &integrator,
            6,
            6,
            config,
            11,
            PixelFilter::Box,
        );
        assert_eq!(a.sample_counts, b.sample_counts, "sample counts must match");
        assert_eq!(
            a.film.pixels(),
            b.film.pixels(),
            "adaptive render must be bit-identical for identical arguments"
        );
    }

    #[test]
    fn adaptive_spends_more_samples_on_a_noisy_scene() {
        // A small bright emitter produces genuinely noisy pixels, so a tight
        // tolerance must push some pixel past the minimum sample floor while all
        // pixels stay within the configured budget.
        let scene = floor_with_overhead_emitter();
        let camera = overhead_camera();
        let integrator = PathIntegrator::new(5, 3);
        let config = AdaptiveConfig {
            min_samples: 8,
            max_samples: 1024,
            relative_tolerance: 0.002,
            batch_size: 8,
            firefly: FireflyClamp::Off,
        };
        let render = render_adaptive(
            &scene,
            &camera,
            &integrator,
            8,
            8,
            config,
            5,
            PixelFilter::Box,
        );
        let max_count = render.sample_counts.iter().copied().max().expect("pixels");
        let min_count = render.sample_counts.iter().copied().min().expect("pixels");
        assert!(
            max_count > config.min_samples,
            "noise should drive at least one pixel past the minimum, got max {max_count}"
        );
        assert!(
            min_count >= config.min_samples,
            "every pixel must receive at least the minimum, got min {min_count}"
        );
        assert!(
            max_count <= config.max_samples,
            "no pixel may exceed the budget, got max {max_count}"
        );
    }

    #[test]
    fn adaptive_white_furnace_mean_is_unit_and_stops_early() {
        // The white-furnace image must still converge to unit radiance, and with
        // a generous budget most pixels should stop well before the cap, proving
        // the adaptive stopping both preserves the mean and saves samples.
        let scene = white_furnace_scene();
        let camera = overhead_camera();
        let integrator = PathIntegrator::new(6, 4);
        let config = AdaptiveConfig {
            min_samples: 16,
            max_samples: 4096,
            relative_tolerance: 0.05,
            batch_size: 16,
            firefly: FireflyClamp::Off,
        };
        let render = render_adaptive(
            &scene,
            &camera,
            &integrator,
            4,
            4,
            config,
            1,
            PixelFilter::Box,
        );
        let mut sum = 0.0f64;
        for pixel in render.film.pixels() {
            sum += f64::from(luminance(*pixel));
        }
        let mean = sum / (render.film.pixels().len() as f64);
        assert!(
            (mean - 1.0).abs() < 3.0e-2,
            "adaptive white-furnace mean luminance {mean} must stay near unit"
        );
        let max_count = render.sample_counts.iter().copied().max().expect("pixels");
        assert!(
            max_count < config.max_samples,
            "a converging image should stop before the cap, got max {max_count}"
        );
    }

    #[test]
    fn firefly_clamp_caps_per_pixel_luminance_through_the_render() {
        // Render the emitter scene once unclamped and once with a luminance cap
        // set below the brightest unclamped pixel. Because the clamp caps every
        // sample's luminance before accumulation, each clamped pixel's mean
        // luminance must stay within the cap, and the brightest pixel must drop
        // below the unclamped peak -- proving the clamp flows through the whole
        // adaptive render path and is not a vacuous no-op.
        let scene = floor_with_overhead_emitter();
        let camera = overhead_camera();
        let integrator = PathIntegrator::new(5, 3);
        let base = AdaptiveConfig {
            min_samples: 16,
            max_samples: 128,
            relative_tolerance: 0.01,
            batch_size: 16,
            firefly: FireflyClamp::Off,
        };
        let unclamped = render_adaptive(
            &scene,
            &camera,
            &integrator,
            8,
            8,
            base,
            3,
            PixelFilter::Box,
        );
        let max_unclamped = unclamped
            .film
            .pixels()
            .iter()
            .map(|p| luminance(*p))
            .fold(0.0f32, f32::max);
        // Cap strictly below the brightest unclamped pixel so the clamp engages.
        let cap = max_unclamped * 0.5;
        assert!(
            cap > 0.0,
            "the scene must produce some radiance for the clamp test, got {max_unclamped}"
        );
        let clamped_cfg = AdaptiveConfig {
            firefly: FireflyClamp::MaxLuminance(cap),
            ..base
        };
        let clamped = render_adaptive(
            &scene,
            &camera,
            &integrator,
            8,
            8,
            clamped_cfg,
            3,
            PixelFilter::Box,
        );
        let max_clamped = clamped
            .film
            .pixels()
            .iter()
            .map(|p| luminance(*p))
            .fold(0.0f32, f32::max);
        assert!(
            max_clamped <= cap + 1.0e-4,
            "every clamped pixel mean luminance must stay within the cap {cap}, got {max_clamped}"
        );
        assert!(
            max_clamped < max_unclamped,
            "the clamp must pull the brightest pixel below the unclamped peak {max_unclamped}, got {max_clamped}"
        );
    }
}
