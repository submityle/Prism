//! Image-based environment lighting with radiance-proportional importance
//! sampling.
//!
//! A real scene is rarely lit by a single analytic light. The reference path
//! tracer therefore supports an *environment map*: an infinitely distant dome
//! of incoming radiance stored as a texture, the ground-truth counterpart of
//! the real-time image-based-lighting (`IBL`) probes. To keep variance low the
//! integrator must draw escape directions in proportion to how bright the dome
//! is in each direction, so this module couples three pieces that already live
//! in the lane:
//!
//! - the octahedral map ([`super::octahedral`]), which stores the whole sphere
//!   of directions on the square `[-1, 1]^2` with no seams and no
//!   trigonometry;
//! - the piecewise-constant sampler ([`super::distribution::Distribution2D`]),
//!   which inverts a tabulated cumulative distribution to draw texels in
//!   proportion to a weight; and
//! - the solid-angle Jacobian of the octahedral map, which converts between the
//!   square's area measure and the sphere's solid-angle measure.
//!
//! The per-texel sampling weight is `luminance * jacobian`, which makes the
//! resulting solid-angle density proportional to radiance (the optimal choice
//! for a single-sample estimator), and the matching [`EnvironmentMap::pdf`]
//! query feeds multiple-importance weighting (`MIS`) against `BSDF` sampling so
//! the two strategies combine without double counting.

use alloc::vec::Vec;

use super::distribution::Distribution2D;
use super::octahedral::{direction_to_square, solid_angle_jacobian, square_to_direction};
use super::sampler::Rng;
use super::Vec3;

/// `Rec. 709` luminance weights used to collapse a radiance triple to the
/// scalar brightness that drives importance sampling.
const LUMINANCE_WEIGHTS: Vec3 = Vec3::new(0.2126, 0.7152, 0.0722);

/// The reciprocal of the octahedral-square area; a density expressed per unit
/// area on `[-1, 1]^2` is converted to a density per unit area on the unit
/// square `[0, 1)^2` by dividing by this value (equivalently multiplying the
/// unit-square density by it).
const INV_SQUARE_AREA: f32 = 0.25;

/// One importance-sampled direction drawn from an [`EnvironmentMap`].
#[derive(Clone, Copy, Debug)]
pub struct EnvironmentSample {
    /// The sampled unit direction pointing away from the shading point toward
    /// the dome.
    pub direction: Vec3,
    /// The dome radiance arriving along `direction`.
    pub radiance: Vec3,
    /// The solid-angle probability density of `direction`; zero for a
    /// degenerate draw that callers must discard.
    pub pdf: f32,
}

/// An infinitely distant dome of incoming radiance stored in octahedral layout.
///
/// Radiance texels are laid out row-major over the octahedral square
/// `[-1, 1]^2`, which the constructor also uses to build a
/// [`Distribution2D`] whose density is proportional to the texels' radiance in
/// the solid-angle measure.
#[derive(Clone, Debug, Default)]
pub struct EnvironmentMap {
    /// The number of texel columns across the octahedral square.
    width: usize,
    /// The number of texel rows down the octahedral square.
    height: usize,
    /// The row-major radiance texels, length `width * height`.
    texels: Vec<Vec3>,
    /// The radiance-proportional sampling distribution over `[0, 1)^2`.
    dist: Distribution2D,
}

impl EnvironmentMap {
    /// Builds an environment map from row-major octahedral radiance texels.
    ///
    /// `texels` must contain exactly `width * height` entries; any other length
    /// (including an empty map) yields an inert map whose [`EnvironmentMap::pdf`]
    /// is zero and whose [`EnvironmentMap::radiance`] is black, so callers can
    /// treat "no environment" uniformly.
    ///
    /// The sampling weight of each texel is its luminance scaled by the
    /// octahedral solid-angle Jacobian evaluated at the texel center. Because
    /// the sampler's area-measure density is proportional to that weight and
    /// the Jacobian cancels when the area density is converted to a solid-angle
    /// density, the resulting solid-angle density is proportional to radiance.
    #[must_use]
    pub fn new(width: usize, height: usize, texels: Vec<Vec3>) -> Self {
        if width == 0 || height == 0 || texels.len() != width * height {
            return Self::default();
        }

        let mut weights = Vec::with_capacity(width * height);
        for row in 0..height {
            for col in 0..width {
                let (u01, v01) = texel_center(col, row, width, height);
                let dir = direction_for_unit_square(u01, v01);
                let jacobian = solid_angle_jacobian(dir);
                let luminance = texels[row * width + col].dot(LUMINANCE_WEIGHTS).max(0.0);
                weights.push(luminance * jacobian);
            }
        }

        let dist = Distribution2D::new(&weights, width, height);
        Self {
            width,
            height,
            texels,
            dist,
        }
    }

    /// Returns `true` when the map carries no texels and acts as a black,
    /// zero-probability dome.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.texels.is_empty() || self.dist.is_empty()
    }

    /// The texel grid width.
    #[must_use]
    pub fn width(&self) -> usize {
        self.width
    }

    /// The texel grid height.
    #[must_use]
    pub fn height(&self) -> usize {
        self.height
    }

    /// Looks up the dome radiance along a (not necessarily normalized)
    /// direction using nearest-neighbor filtering, matching the piecewise-constant
    /// basis the sampler assumes.
    #[must_use]
    pub fn radiance(&self, dir: Vec3) -> Vec3 {
        if self.is_empty() {
            return Vec3::ZERO;
        }
        let (u01, v01) = unit_square_for_direction(dir);
        self.texel_at(u01, v01)
    }

    /// The solid-angle probability density of drawing `dir` from
    /// [`EnvironmentMap::sample`].
    ///
    /// `dir` is assumed to be a unit vector. The density is zero for an inert
    /// map or a degenerate direction, matching the discarded samples the
    /// sampler itself produces.
    #[must_use]
    pub fn pdf(&self, dir: Vec3) -> f32 {
        if self.is_empty() {
            return 0.0;
        }
        let (u01, v01) = unit_square_for_direction(dir);
        let area_pdf = self.dist.pdf(u01, v01);
        let jacobian = solid_angle_jacobian(dir);
        if jacobian > 0.0 {
            (area_pdf * INV_SQUARE_AREA) / jacobian
        } else {
            0.0
        }
    }

    /// Draws a direction from the dome in proportion to its radiance.
    ///
    /// Returns the sampled direction, the radiance arriving along it, and the
    /// solid-angle density of the draw. A zero density signals a degenerate
    /// sample (an inert map or a direction on a seam) that the caller must skip.
    #[must_use]
    pub fn sample(&self, rng: &mut Rng) -> EnvironmentSample {
        if self.is_empty() {
            return EnvironmentSample {
                direction: Vec3::ZERO,
                radiance: Vec3::ZERO,
                pdf: 0.0,
            };
        }

        let u0 = rng.next_f32();
        let u1 = rng.next_f32();
        let sample = self.dist.sample_continuous(u0, u1);
        let direction = direction_for_unit_square(sample.u, sample.v);
        let jacobian = solid_angle_jacobian(direction);
        let pdf = if jacobian > 0.0 {
            (sample.pdf * INV_SQUARE_AREA) / jacobian
        } else {
            0.0
        };
        let radiance = self.texel_at(sample.u, sample.v);
        EnvironmentSample {
            direction,
            radiance,
            pdf,
        }
    }

    /// Fetches the nearest texel for a point on the unit square `[0, 1)^2`.
    fn texel_at(&self, u01: f32, v01: f32) -> Vec3 {
        let col = grid_index(u01, self.width);
        let row = grid_index(v01, self.height);
        self.texels[row * self.width + col]
    }
}

/// Returns the unit-square center `[0, 1)^2` coordinates of texel `(col, row)`.
fn texel_center(col: usize, row: usize, width: usize, height: usize) -> (f32, f32) {
    let u = (col as f32 + 0.5) / width as f32;
    let v = (row as f32 + 0.5) / height as f32;
    (u, v)
}

/// Maps a unit-square point `[0, 1)^2` onto the octahedral square `[-1, 1]^2`
/// and reconstructs its unit direction.
fn direction_for_unit_square(u01: f32, v01: f32) -> Vec3 {
    square_to_direction(2.0 * u01 - 1.0, 2.0 * v01 - 1.0)
}

/// Projects a direction onto the octahedral square and remaps it to the unit
/// square `[0, 1)^2`, the inverse of [`direction_for_unit_square`].
fn unit_square_for_direction(dir: Vec3) -> (f32, f32) {
    let (ou, ov) = direction_to_square(dir);
    ((ou + 1.0) * 0.5, (ov + 1.0) * 0.5)
}

/// Converts a unit-square coordinate to a clamped integer grid index.
fn grid_index(coord: f32, extent: usize) -> usize {
    let scaled = (coord * extent as f32).floor();
    let clamped = scaled.clamp(0.0, extent as f32 - 1.0);
    clamped as usize
}

#[cfg(test)]
mod tests {
    use super::super::PI;
    use super::*;

    /// Rejection-samples a uniformly distributed unit direction from the cube,
    /// avoiding the trigonometry a direct spherical draw would need.
    fn uniform_sphere(rng: &mut Rng) -> Vec3 {
        loop {
            let x = 2.0 * rng.next_f32() - 1.0;
            let y = 2.0 * rng.next_f32() - 1.0;
            let z = 2.0 * rng.next_f32() - 1.0;
            let len_sq = x * x + y * y + z * z;
            if len_sq > 1e-6 && len_sq <= 1.0 {
                let inv = 1.0 / len_sq.sqrt();
                return Vec3::new(x * inv, y * inv, z * inv);
            }
        }
    }

    /// Builds a smoothly varying test dome whose brightness depends on the
    /// octahedral texel position, exercising the non-uniform sampling path.
    fn gradient_map(width: usize, height: usize) -> EnvironmentMap {
        let mut texels = Vec::with_capacity(width * height);
        for row in 0..height {
            for col in 0..width {
                let u = (col as f32 + 0.5) / width as f32;
                let v = (row as f32 + 0.5) / height as f32;
                let brightness = 0.1 + 4.0 * u * u + 2.0 * v;
                texels.push(Vec3::new(brightness, 0.5 * brightness, 0.25 * brightness));
            }
        }
        EnvironmentMap::new(width, height, texels)
    }

    #[test]
    fn empty_map_is_inert() {
        let map = EnvironmentMap::new(0, 0, Vec::new());
        assert!(map.is_empty());
        let dir = Vec3::new(0.0, 1.0, 0.0);
        assert_eq!(map.pdf(dir), 0.0);
        assert_eq!(map.radiance(dir).max_component(), 0.0);
        let mut rng = Rng::seed(1);
        assert_eq!(map.sample(&mut rng).pdf, 0.0);
    }

    #[test]
    fn mismatched_length_falls_back_to_inert() {
        let map = EnvironmentMap::new(2, 2, alloc::vec![Vec3::ONE; 3]);
        assert!(map.is_empty());
    }

    #[test]
    fn pdf_integrates_to_one_over_the_sphere() {
        let map = gradient_map(16, 16);
        let mut rng = Rng::with_stream(7, 1);
        let samples = 400_000;
        let mut sum = 0.0f64;
        for _ in 0..samples {
            let dir = uniform_sphere(&mut rng);
            sum += f64::from(map.pdf(dir));
        }
        // Uniform directions have density 1 / (4 pi); the Monte Carlo estimate
        // of the integral of pdf over the sphere is therefore 4 pi times the
        // sample mean, which must be one.
        let integral = 4.0 * f64::from(PI) * (sum / f64::from(samples));
        assert!((integral - 1.0).abs() < 5e-3, "integral = {integral}");
    }

    #[test]
    fn importance_sampling_matches_uniform_estimate() {
        let map = gradient_map(16, 16);

        // Reference: integral of radiance luminance over the sphere via a plain
        // uniform estimator.
        let mut uniform_rng = Rng::with_stream(11, 2);
        let samples = 400_000;
        let mut uniform_sum = 0.0f64;
        for _ in 0..samples {
            let dir = uniform_sphere(&mut uniform_rng);
            uniform_sum += f64::from(map.radiance(dir).dot(LUMINANCE_WEIGHTS));
        }
        let reference = 4.0 * f64::from(PI) * (uniform_sum / f64::from(samples));

        // Importance estimator: each draw contributes radiance / pdf.
        let mut rng = Rng::with_stream(13, 3);
        let mut importance_sum = 0.0f64;
        let mut drawn = 0u64;
        for _ in 0..samples {
            let sample = map.sample(&mut rng);
            if sample.pdf > 0.0 {
                importance_sum +=
                    f64::from(sample.radiance.dot(LUMINANCE_WEIGHTS)) / f64::from(sample.pdf);
                drawn += 1;
            }
        }
        let importance = importance_sum / drawn as f64;

        let rel = (importance - reference).abs() / reference;
        assert!(
            rel < 1e-2,
            "importance = {importance}, reference = {reference}"
        );
    }

    #[test]
    fn sampled_pdf_matches_pdf_query() {
        let map = gradient_map(8, 8);
        let mut rng = Rng::with_stream(23, 5);
        for _ in 0..2_000 {
            let sample = map.sample(&mut rng);
            if sample.pdf > 0.0 {
                let queried = map.pdf(sample.direction);
                let rel = (queried - sample.pdf).abs() / sample.pdf;
                assert!(rel < 2e-2, "queried = {queried}, sampled = {}", sample.pdf);
            }
        }
    }

    #[test]
    fn constant_dome_is_uniform_in_the_mean() {
        // A constant dome weights each texel by the octahedral Jacobian at its
        // center, so the resulting solid-angle density is only exactly uniform
        // at texel centers; it integrates to one and averages to the uniform
        // density `1 / (4 pi)` over the sphere, which is what a constant dome
        // must satisfy. The radiance lookup is exact everywhere.
        let value = Vec3::new(0.8, 0.6, 0.4);
        let map = EnvironmentMap::new(32, 32, alloc::vec![value; 32 * 32]);
        let uniform_pdf = 1.0 / (4.0 * PI);

        let mut rng = Rng::with_stream(31, 7);
        let samples = 400_000;
        let mut pdf_sum = 0.0f64;
        for _ in 0..samples {
            let dir = uniform_sphere(&mut rng);
            pdf_sum += f64::from(map.pdf(dir));
            assert!(map.radiance(dir).sub(value).length() < 1e-6);
        }
        let mean_pdf = pdf_sum / f64::from(samples);
        let rel = (mean_pdf - f64::from(uniform_pdf)).abs() / f64::from(uniform_pdf);
        assert!(
            rel < 5e-3,
            "mean pdf = {mean_pdf}, expected = {uniform_pdf}"
        );
    }
}
