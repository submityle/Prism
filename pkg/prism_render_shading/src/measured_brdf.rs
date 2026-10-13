//! Data-driven measured / tabulated BRDF: Rusinkiewicz-parameterised table
//! evaluation plus BRDF-weighted importance sampling.
//!
//! Backend-neutral golden reference for the `Measured` ("open lobe") shading
//! class and the numerical twin of `measured_brdf.wesl`'s coordinate transform,
//! table lookup and evaluation path. There are no bindings, no `unsafe` and no
//! I/O: a table is either supplied by the caller or synthesised analytically so
//! the reference is deterministic and fully exercisable in isolation.
//!
//! # Parameterisation
//!
//! Dense `(theta_in, phi_in, theta_out, phi_out)` BRDF slices are enormous, so
//! acquired-material pipelines (MERL, Dupuy & Jakob 2018) store an isotropic
//! BRDF in the half/difference basis of Rusinkiewicz 1998:
//!
//! * `theta_h` - elevation of the half vector `h = normalize(wi + wo)` from the
//!   surface normal. Specular lobes concentrate near `theta_h = 0`, so this
//!   axis is warped by a square root (`u = sqrt(theta_h / (pi/2))`), the MERL
//!   convention, giving many more samples to the glossy peak.
//! * `theta_d` - the "difference" elevation, i.e. the angle between `wi` and
//!   `h` once `wi` is rotated into the frame whose `+Z` is `h`. This axis
//!   carries the Fresnel ramp towards grazing.
//! * `phi_d` - the difference azimuth, folded to `[0, pi)` by Helmholtz
//!   reciprocity (`f(wi, wo) == f(wo, wi)` maps `phi_d -> phi_d + pi`).
//!
//! Isotropy drops `phi_h`, collapsing the 4D BRDF to this compact 3D table.
//!
//! # Importance sampling
//!
//! Following the "warp the parameterisation so the lobe is easy to sample"
//! idea of Dupuy & Jakob's adaptive acquisition, [`MeasuredBrdfSampler`]
//! precomputes, per incident-elevation bin, a 2D distribution of the outgoing
//! direction in `(mu = cos theta_o, phi_o)` space whose density is proportional
//! to `f(wi, wo) * cos theta_o`. Because `d(omega) = d(mu) d(phi)` has unit
//! Jacobian in this basis, the piecewise-constant density is *exactly*
//! normalised over the hemisphere and the solid-angle pdf is simply the cell
//! density. Sampling is a marginal-then-conditional inverse-CDF draw, cacheable
//! and allocation-free at sample time. A small uniform floor keeps the pdf
//! strictly positive over the whole hemisphere so no BRDF-supported direction
//! is ever assigned zero probability.
//!
//! All transcendentals route through [`bevy_math::ops`] for cross-platform
//! determinism; all inputs are clamped and all table indices are bounds-safe.

use alloc::vec;
use alloc::vec::Vec;

use bevy_math::ops;

/// `core::f32::consts::PI` rounded to f32, byte-identical to the GPU twin.
const MB_PI: f32 = core::f32::consts::PI;
/// `pi / 2`, the upper bound of both tabulated elevation axes.
const MB_FRAC_PI_2: f32 = core::f32::consts::FRAC_PI_2;
/// `2 * pi`, the full outgoing azimuth span used by the sampler.
const MB_TAU: f32 = core::f32::consts::TAU;
/// Smallest cosine / denominator evaluated; keeps grazing angles off the
/// microfacet and normalisation singularities.
const MB_MIN_COS: f32 = 1.0e-4;
/// Smallest squared length treated as a non-degenerate direction.
const MB_MIN_LEN_SQ: f32 = 1.0e-12;
/// Microfacet roughness floor shared with the analytic synthesiser.
const MB_MIN_ROUGHNESS: f32 = 0.045;

/// Default half-elevation resolution (matches the MERL `theta_h` axis).
pub const MB_DEFAULT_THETA_H: usize = 90;
/// Default difference-elevation resolution (matches the MERL `theta_d` axis).
pub const MB_DEFAULT_THETA_D: usize = 90;
/// Default difference-azimuth resolution over the reciprocity-folded `[0, pi)`.
pub const MB_DEFAULT_PHI_D: usize = 180;

/// Resolution of a tabulated BRDF along its three Rusinkiewicz axes.
///
/// Each dimension must be at least two so that linear interpolation always has
/// a well-defined bracketing pair of samples.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MeasuredBrdfDims {
    /// Samples along the square-root-warped half elevation `theta_h`.
    pub theta_h: usize,
    /// Samples along the linear difference elevation `theta_d`.
    pub theta_d: usize,
    /// Samples along the linear, reciprocity-folded difference azimuth `phi_d`.
    pub phi_d: usize,
}

impl MeasuredBrdfDims {
    /// The MERL-compatible `90 x 90 x 180` default resolution.
    pub const MERL: Self = Self {
        theta_h: MB_DEFAULT_THETA_H,
        theta_d: MB_DEFAULT_THETA_D,
        phi_d: MB_DEFAULT_PHI_D,
    };

    /// Total number of RGB cells in a table of these dimensions.
    #[must_use]
    pub const fn cell_count(self) -> usize {
        self.theta_h * self.theta_d * self.phi_d
    }

    /// Returns whether every axis has at least two samples.
    #[must_use]
    pub const fn is_valid(self) -> bool {
        self.theta_h >= 2 && self.theta_d >= 2 && self.phi_d >= 2
    }
}

/// Errors returned when constructing a [`MeasuredBrdf`] from raw cells.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MeasuredBrdfError {
    /// A dimension had fewer than two samples.
    DegenerateDims,
    /// The supplied cell slice length did not equal [`MeasuredBrdfDims::cell_count`].
    CellCountMismatch {
        /// Number of RGB cells expected from the dimensions.
        expected: usize,
        /// Number of RGB cells actually supplied.
        found: usize,
    },
}

/// A point in the Rusinkiewicz half/difference basis.
///
/// All angles are in radians: `theta_h, theta_d` lie in `[0, pi/2]` and
/// `phi_d` is folded to `[0, pi)` by reciprocity.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RusinkiewiczCoords {
    /// Half-vector elevation from the normal.
    pub theta_h: f32,
    /// Difference elevation (angle between `wi` and the half vector).
    pub theta_d: f32,
    /// Reciprocity-folded difference azimuth.
    pub phi_d: f32,
}

/// Parameters of the analytic microfacet BRDF used to synthesise a table.
///
/// The synthesiser blends a Lambertian base with a Cook-Torrance GGX specular
/// lobe (Trowbridge-Reitz NDF, height-correlated Smith visibility, Schlick
/// Fresnel), giving a smooth, reciprocal, strictly non-negative reference whose
/// energy and shape are easy to reason about in tests.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SyntheticBrdfParams {
    /// Linear base colour (albedo for the diffuse lobe, tint for metals).
    pub base_color: [f32; 3],
    /// Perceptual roughness in `[0, 1]`; squared to the GGX `alpha`.
    pub roughness: f32,
    /// Metalness in `[0, 1]` interpolating dielectric `f0` towards `base_color`.
    pub metallic: f32,
    /// Dielectric reflectance in `[0, 1]` controlling the `0.16 * r^2` `f0`.
    pub reflectance: f32,
}

impl Default for SyntheticBrdfParams {
    fn default() -> Self {
        Self {
            base_color: [0.82, 0.67, 0.16],
            roughness: 0.35,
            metallic: 1.0,
            reflectance: 0.5,
        }
    }
}

/// A measured / tabulated isotropic BRDF stored in the Rusinkiewicz basis.
///
/// Cells are laid out contiguously with `phi_d` varying fastest, then
/// `theta_d`, then `theta_h` (the slowest axis), so a full `phi_d` row is a
/// cache-friendly contiguous span - the order both the trilinear lookup and a
/// GPU upload consume.
#[derive(Clone, Debug, PartialEq)]
pub struct MeasuredBrdf {
    dims: MeasuredBrdfDims,
    cells: Vec<[f32; 3]>,
}

impl MeasuredBrdf {
    /// Wraps a pre-populated cell buffer, validating dimensions and length.
    ///
    /// # Errors
    ///
    /// Returns [`MeasuredBrdfError`] if a dimension is degenerate or the cell
    /// count does not match the dimensions.
    pub fn new(dims: MeasuredBrdfDims, cells: Vec<[f32; 3]>) -> Result<Self, MeasuredBrdfError> {
        if !dims.is_valid() {
            return Err(MeasuredBrdfError::DegenerateDims);
        }
        let expected = dims.cell_count();
        if cells.len() != expected {
            return Err(MeasuredBrdfError::CellCountMismatch {
                expected,
                found: cells.len(),
            });
        }
        Ok(Self { dims, cells })
    }

    /// Dimensions of this table.
    #[must_use]
    pub const fn dims(&self) -> MeasuredBrdfDims {
        self.dims
    }

    /// Immutable view of the raw RGB cells (for upload / inspection).
    #[must_use]
    pub fn cells(&self) -> &[[f32; 3]] {
        &self.cells
    }

    /// Synthesises a table by evaluating an analytic microfacet BRDF at every
    /// cell centre.
    ///
    /// Each cell `(i, j, k)` is converted back to a representative `(wi, wo)`
    /// pair via [`rusinkiewicz_to_directions`] and the analytic closure is
    /// sampled there, producing a self-consistent, deterministic measured
    /// table whose round-tripped evaluation reproduces the closure to within
    /// interpolation error.
    #[must_use]
    pub fn from_analytic(dims: MeasuredBrdfDims, params: SyntheticBrdfParams) -> Self {
        let dims = MeasuredBrdfDims {
            theta_h: dims.theta_h.max(2),
            theta_d: dims.theta_d.max(2),
            phi_d: dims.phi_d.max(2),
        };
        let mut cells = vec![[0.0_f32; 3]; dims.cell_count()];
        for i in 0..dims.theta_h {
            let theta_h = unit_to_theta_h(axis_center(i, dims.theta_h));
            for j in 0..dims.theta_d {
                let theta_d = axis_center(j, dims.theta_d) * MB_FRAC_PI_2;
                for k in 0..dims.phi_d {
                    let phi_d = axis_center(k, dims.phi_d) * MB_PI;
                    let coords = RusinkiewiczCoords {
                        theta_h,
                        theta_d,
                        phi_d,
                    };
                    let (wi, wo) = rusinkiewicz_to_directions(coords);
                    let value = analytic_brdf(wi, wo, params);
                    cells[cell_index(dims, i, j, k)] = value;
                }
            }
        }
        Self { dims, cells }
    }

    /// Reads a single cell, saturating out-of-range indices to the border.
    #[must_use]
    fn cell(&self, i: usize, j: usize, k: usize) -> [f32; 3] {
        let i = i.min(self.dims.theta_h - 1);
        let j = j.min(self.dims.theta_d - 1);
        let k = k.min(self.dims.phi_d - 1);
        self.cells[cell_index(self.dims, i, j, k)]
    }

    /// Maps Rusinkiewicz coordinates to continuous, clamped grid coordinates
    /// (in cell-index units) along each warped axis.
    #[must_use]
    pub fn grid_coords(&self, coords: RusinkiewiczCoords) -> [f32; 3] {
        let u_h = theta_h_to_unit(coords.theta_h);
        let u_d = clamp01(coords.theta_d / MB_FRAC_PI_2);
        let u_p = clamp01(fold_phi_d(coords.phi_d) / MB_PI);
        [
            unit_to_grid(u_h, self.dims.theta_h),
            unit_to_grid(u_d, self.dims.theta_d),
            unit_to_grid(u_p, self.dims.phi_d),
        ]
    }

    /// Trilinearly interpolates the table at continuous grid coordinates.
    #[must_use]
    pub fn sample_grid(&self, grid: [f32; 3]) -> [f32; 3] {
        let (i0, fi) = split_grid(grid[0], self.dims.theta_h);
        let (j0, fj) = split_grid(grid[1], self.dims.theta_d);
        let (k0, fk) = split_grid(grid[2], self.dims.phi_d);
        let i1 = (i0 + 1).min(self.dims.theta_h - 1);
        let j1 = (j0 + 1).min(self.dims.theta_d - 1);
        let k1 = (k0 + 1).min(self.dims.phi_d - 1);

        let c000 = self.cell(i0, j0, k0);
        let c001 = self.cell(i0, j0, k1);
        let c010 = self.cell(i0, j1, k0);
        let c011 = self.cell(i0, j1, k1);
        let c100 = self.cell(i1, j0, k0);
        let c101 = self.cell(i1, j0, k1);
        let c110 = self.cell(i1, j1, k0);
        let c111 = self.cell(i1, j1, k1);

        let c00 = lerp3(c000, c001, fk);
        let c01 = lerp3(c010, c011, fk);
        let c10 = lerp3(c100, c101, fk);
        let c11 = lerp3(c110, c111, fk);
        let c0 = lerp3(c00, c01, fj);
        let c1 = lerp3(c10, c11, fj);
        lerp3(c0, c1, fi)
    }

    /// Evaluates the tabulated BRDF for a local-frame incident/outgoing pair.
    ///
    /// `wi` and `wo` are expressed in the shading frame whose `+Z` is the
    /// surface normal and `+X` the tangent. Directions below the horizon (or
    /// degenerate inputs) yield zero. The result is clamped non-negative, as a
    /// physical BRDF must be.
    #[must_use]
    pub fn evaluate(&self, wi: [f32; 3], wo: [f32; 3]) -> [f32; 3] {
        let wi = normalize_or_zero(wi);
        let wo = normalize_or_zero(wo);
        if wi[2] <= 0.0 || wo[2] <= 0.0 {
            return [0.0; 3];
        }
        let coords = directions_to_rusinkiewicz(wi, wo);
        let value = self.sample_grid(self.grid_coords(coords));
        [value[0].max(0.0), value[1].max(0.0), value[2].max(0.0)]
    }
}

/// A precomputed BRDF-weighted importance sampler over a [`MeasuredBrdf`].
///
/// Holds a marginal CDF over `mu = cos theta_o` and a conditional CDF over the
/// relative azimuth `phi_o` for each incident-elevation bin. Both sampling and
/// pdf evaluation read the *same* discretised density, so the pair is a
/// mutually consistent (unbiased) estimator pair for Monte Carlo integration.
#[derive(Clone, Debug, PartialEq)]
pub struct MeasuredBrdfSampler {
    theta_i: usize,
    mu: usize,
    phi: usize,
    /// Per-incident-bin cell densities (`mu`-major), length `theta_i*mu*phi`.
    density: Vec<f32>,
    /// Per-incident-bin total measure `sum(density) * d_mu * d_phi`.
    total: Vec<f32>,
    /// Per-incident-bin marginal CDF over `mu`, length `theta_i*mu`.
    marginal_cdf: Vec<f32>,
    /// Per-`(incident, mu)`-row conditional CDF over `phi`, length `theta_i*mu*phi`.
    conditional_cdf: Vec<f32>,
}

impl MeasuredBrdfSampler {
    /// Builds a sampler for `brdf` using `(theta_i, mu, phi)` discretisation.
    ///
    /// Each dimension is floored at two bins. A small uniform floor is added to
    /// every cell so the pdf is strictly positive across the whole hemisphere.
    #[must_use]
    pub fn from_brdf(brdf: &MeasuredBrdf, theta_i: usize, mu: usize, phi: usize) -> Self {
        let theta_i = theta_i.max(2);
        let mu = mu.max(2);
        let phi = phi.max(2);
        let d_mu = 1.0 / mu as f32;
        let d_phi = MB_TAU / phi as f32;

        let mut density = vec![0.0_f32; theta_i * mu * phi];
        let mut total = vec![0.0_f32; theta_i];
        let mut marginal_cdf = vec![0.0_f32; theta_i * mu];
        let mut conditional_cdf = vec![0.0_f32; theta_i * mu * phi];

        for ii in 0..theta_i {
            // Isotropy lets us pin the incident azimuth to zero.
            let theta_in = axis_center(ii, theta_i) * MB_FRAC_PI_2;
            let (sin_i, cos_i) = ops::sin_cos(theta_in);
            let wi = [sin_i, 0.0, cos_i];

            // Uniform floor density scaled to the mean BRDF-weighted density so
            // it never dominates a well-defined lobe but still covers the tails.
            let mut sum = 0.0_f32;
            for jj in 0..mu {
                let mu_o = axis_center(jj, mu);
                let sin_o = ops::sqrt((1.0 - mu_o * mu_o).max(0.0));
                for kk in 0..phi {
                    let phi_o = axis_center(kk, phi) * MB_TAU;
                    let (sin_p, cos_p) = ops::sin_cos(phi_o);
                    let wo = [sin_o * cos_p, sin_o * sin_p, mu_o];
                    let f = scalar_brdf(brdf.evaluate(wi, wo));
                    let weight = (f * mu_o).max(0.0);
                    density[sampler_index(mu, phi, ii, jj, kk)] = weight;
                    sum += weight;
                }
            }

            // Blend in a uniform floor carrying ~1% of the lobe's mass so the
            // tails stay sampleable without biasing the peak.
            let cells = (mu * phi) as f32;
            let floor = 0.01 * (sum / cells).max(f32::MIN_POSITIVE);
            let mut acc_mu = 0.0_f32;
            for jj in 0..mu {
                let mut row_sum = 0.0_f32;
                for kk in 0..phi {
                    let idx = sampler_index(mu, phi, ii, jj, kk);
                    let value = density[idx] + floor;
                    density[idx] = value;
                    row_sum += value;
                }
                // Conditional CDF over phi within this mu row.
                let mut acc_phi = 0.0_f32;
                for kk in 0..phi {
                    let idx = sampler_index(mu, phi, ii, jj, kk);
                    acc_phi += density[idx];
                    conditional_cdf[idx] = acc_phi / row_sum.max(f32::MIN_POSITIVE);
                }
                acc_mu += row_sum;
                marginal_cdf[marginal_index(mu, ii, jj)] = acc_mu;
            }
            // Normalise the marginal and record the total measure.
            let marginal_total = acc_mu.max(f32::MIN_POSITIVE);
            for jj in 0..mu {
                let idx = marginal_index(mu, ii, jj);
                marginal_cdf[idx] /= marginal_total;
            }
            total[ii] = acc_mu * d_mu * d_phi;
        }

        Self {
            theta_i,
            mu,
            phi,
            density,
            total,
            marginal_cdf,
            conditional_cdf,
        }
    }

    /// Convenience constructor using 90 incident bins and a `64 x 128`
    /// outgoing grid - a solid default for offline reference integration.
    #[must_use]
    pub fn with_default_resolution(brdf: &MeasuredBrdf) -> Self {
        Self::from_brdf(brdf, 90, 64, 128)
    }

    /// Incident-elevation bin index for a (clamped) incident direction.
    #[must_use]
    fn incident_bin(&self, cos_i: f32) -> usize {
        let theta_in = ops::acos(cos_i.clamp(0.0, 1.0));
        let u = clamp01(theta_in / MB_FRAC_PI_2);
        ((u * self.theta_i as f32) as usize).min(self.theta_i - 1)
    }

    /// Draws an outgoing direction and its solid-angle pdf for incident `wi`.
    ///
    /// `u1, u2` are stratified / random samples in `[0, 1)`. The returned
    /// direction is in the same local frame as `wi`; its azimuth is rotated to
    /// match `wi` so the isotropic table is oriented correctly. The pdf is the
    /// exact piecewise-constant solid-angle density that produced the sample.
    #[must_use]
    pub fn sample(&self, wi: [f32; 3], u1: f32, u2: f32) -> MeasuredBrdfSample {
        let wi = normalize_or_zero(wi);
        if wi[2] <= 0.0 {
            return MeasuredBrdfSample {
                direction: [0.0, 0.0, 1.0],
                pdf: 0.0,
            };
        }
        let u1 = clamp01(u1);
        let u2 = clamp01(u2);
        let ii = self.incident_bin(wi[2]);

        // Marginal draw over mu.
        let (jj, f_mu) = sample_cdf_row(&self.marginal_cdf, ii * self.mu, self.mu, u1);
        let mu_lo = jj as f32 / self.mu as f32;
        let mu_o = clamp01(mu_lo + f_mu / self.mu as f32);
        // Avoid an exactly-horizon sample so cos theta_o stays positive.
        let mu_o = mu_o.max(MB_MIN_COS);

        // Conditional draw over phi within the chosen mu row.
        let row = sampler_index(self.mu, self.phi, ii, jj, 0);
        let (kk, f_phi) = sample_cdf_row(&self.conditional_cdf, row, self.phi, u2);
        let phi_rel = (kk as f32 + f_phi) / self.phi as f32 * MB_TAU;

        // Orient the azimuth relative to the incident direction (isotropy).
        let phi_i = ops::atan2(wi[1], wi[0]);
        let phi_o = phi_rel + phi_i;
        let sin_o = ops::sqrt((1.0 - mu_o * mu_o).max(0.0));
        let (sin_p, cos_p) = ops::sin_cos(phi_o);
        let direction = normalize_or_zero([sin_o * cos_p, sin_o * sin_p, mu_o]);

        let pdf = self.pdf_in_bin(ii, mu_o, phi_rel);
        MeasuredBrdfSample { direction, pdf }
    }

    /// Solid-angle pdf of drawing `wo` given incident `wi`.
    ///
    /// Mirrors [`MeasuredBrdfSampler::sample`] exactly, so the two form a valid
    /// (pdf, sample) pair. Returns zero for below-horizon or degenerate inputs.
    #[must_use]
    pub fn pdf(&self, wi: [f32; 3], wo: [f32; 3]) -> f32 {
        let wi = normalize_or_zero(wi);
        let wo = normalize_or_zero(wo);
        if wi[2] <= 0.0 || wo[2] <= 0.0 {
            return 0.0;
        }
        let ii = self.incident_bin(wi[2]);
        let phi_i = ops::atan2(wi[1], wi[0]);
        let phi_o = ops::atan2(wo[1], wo[0]);
        let phi_rel = wrap_tau(phi_o - phi_i);
        self.pdf_in_bin(ii, wo[2], phi_rel)
    }

    /// Piecewise-constant solid-angle density of a `(mu, phi_rel)` sample in a
    /// given incident bin. The `(mu, phi)` basis has unit Jacobian, so the
    /// cell density *is* the solid-angle pdf.
    #[must_use]
    fn pdf_in_bin(&self, ii: usize, mu_o: f32, phi_rel: f32) -> f32 {
        let total = self.total[ii];
        if total <= 0.0 {
            return 0.0;
        }
        let jj = ((clamp01(mu_o) * self.mu as f32) as usize).min(self.mu - 1);
        let kk = ((wrap_tau(phi_rel) / MB_TAU * self.phi as f32) as usize).min(self.phi - 1);
        let density = self.density[sampler_index(self.mu, self.phi, ii, jj, kk)];
        density / total
    }
}

/// Result of a [`MeasuredBrdfSampler::sample`] draw.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeasuredBrdfSample {
    /// Sampled outgoing direction in the incident direction's local frame.
    pub direction: [f32; 3],
    /// Solid-angle pdf of this draw (zero if the sample is invalid).
    pub pdf: f32,
}

// --- Coordinate transforms -------------------------------------------------

/// Half vector of an incident/outgoing pair, falling back to the normal when
/// the sum is degenerate (back-to-back directions).
#[must_use]
pub fn half_vector(wi: [f32; 3], wo: [f32; 3]) -> [f32; 3] {
    normalize_or(
        [wi[0] + wo[0], wi[1] + wo[1], wi[2] + wo[2]],
        [0.0, 0.0, 1.0],
    )
}

/// Converts a local-frame `(wi, wo)` pair to Rusinkiewicz `(theta_h, theta_d,
/// phi_d)` coordinates, folding `phi_d` to `[0, pi)` by reciprocity.
///
/// The transform rotates `wi` into the frame whose `+Z` is the half vector by
/// undoing the half vector's azimuth then elevation, matching the canonical
/// MERL `std_coords_to_half_diff_coords`.
#[must_use]
pub fn directions_to_rusinkiewicz(wi: [f32; 3], wo: [f32; 3]) -> RusinkiewiczCoords {
    let wi = normalize_or_zero(wi);
    let wo = normalize_or_zero(wo);
    let h = half_vector(wi, wo);

    let theta_h = ops::acos(h[2].clamp(-1.0, 1.0));
    let phi_h = ops::atan2(h[1], h[0]);

    // Rotate wi into the half-vector frame: undo phi_h about +Z, then theta_h
    // about +Y so that h maps to +Z and wi becomes the difference vector.
    let tmp = rotate_about_z(wi, -phi_h);
    let diff = rotate_about_y(tmp, -theta_h);

    let theta_d = ops::acos(diff[2].clamp(-1.0, 1.0));
    let phi_d = fold_phi_d(ops::atan2(diff[1], diff[0]));

    RusinkiewiczCoords {
        theta_h,
        theta_d,
        phi_d,
    }
}

/// Reconstructs a representative local-frame `(wi, wo)` pair from Rusinkiewicz
/// coordinates, pinning the half-vector azimuth to zero (isotropy).
///
/// This is the exact inverse of [`directions_to_rusinkiewicz`] for
/// `phi_h == 0`: the difference vector is lifted by `theta_d, phi_d`, rotated
/// up to the half vector by `theta_h`, and `wo` is the mirror of `wi` about the
/// half vector.
#[must_use]
pub fn rusinkiewicz_to_directions(coords: RusinkiewiczCoords) -> ([f32; 3], [f32; 3]) {
    let (sin_d, cos_d) = ops::sin_cos(coords.theta_d);
    let (sin_pd, cos_pd) = ops::sin_cos(coords.phi_d);
    let diff = [sin_d * cos_pd, sin_d * sin_pd, cos_d];

    // Rotate the difference vector up to the half vector (phi_h == 0).
    let wi = rotate_about_y(diff, coords.theta_h);

    let (sin_h, cos_h) = ops::sin_cos(coords.theta_h);
    let h = [sin_h, 0.0, cos_h];
    // wo is wi mirrored about the half vector.
    let d = 2.0 * dot(wi, h);
    let wo = [d * h[0] - wi[0], d * h[1] - wi[1], d * h[2] - wi[2]];
    (normalize_or_zero(wi), normalize_or_zero(wo))
}

/// Square-root warp of the half elevation into `[0, 1]` (MERL convention):
/// `u = sqrt(theta_h / (pi/2))`, concentrating resolution at the specular peak.
#[must_use]
pub fn theta_h_to_unit(theta_h: f32) -> f32 {
    clamp01(ops::sqrt(clamp01(theta_h / MB_FRAC_PI_2)))
}

/// Inverse of [`theta_h_to_unit`]: `theta_h = u^2 * (pi/2)`.
#[must_use]
pub fn unit_to_theta_h(u: f32) -> f32 {
    let u = clamp01(u);
    u * u * MB_FRAC_PI_2
}

// --- Diagnostics -----------------------------------------------------------

/// Returns whether every cell of the table is finite and non-negative.
#[must_use]
pub fn is_non_negative(brdf: &MeasuredBrdf) -> bool {
    brdf.cells()
        .iter()
        .all(|c| c.iter().all(|&v| v.is_finite() && v >= 0.0))
}

/// Maximum reciprocity error `|f(wi, wo) - f(wo, wi)|` over a direction set.
///
/// A physical BRDF satisfies Helmholtz reciprocity exactly; the Rusinkiewicz
/// table is reciprocal by construction (swapping `wi, wo` leaves `theta_h,
/// theta_d` fixed and maps `phi_d -> phi_d + pi`, which the fold collapses), so
/// this should be at the floating-point floor.
#[must_use]
pub fn reciprocity_error(brdf: &MeasuredBrdf, directions: &[([f32; 3], [f32; 3])]) -> f32 {
    let mut max_err = 0.0_f32;
    for &(wi, wo) in directions {
        let a = brdf.evaluate(wi, wo);
        let b = brdf.evaluate(wo, wi);
        for c in 0..3 {
            max_err = max_err.max((a[c] - b[c]).abs());
        }
    }
    max_err
}

// --- Internal maths --------------------------------------------------------

/// Rotation of `v` by `angle` about `+Z` (right-handed).
#[must_use]
fn rotate_about_z(v: [f32; 3], angle: f32) -> [f32; 3] {
    let (s, c) = ops::sin_cos(angle);
    [v[0] * c - v[1] * s, v[0] * s + v[1] * c, v[2]]
}

/// Rotation of `v` by `angle` about `+Y` (right-handed).
#[must_use]
fn rotate_about_y(v: [f32; 3], angle: f32) -> [f32; 3] {
    let (s, c) = ops::sin_cos(angle);
    [v[0] * c + v[2] * s, v[1], -v[0] * s + v[2] * c]
}

/// Folds a signed difference azimuth into `[0, pi)` by reciprocity.
#[must_use]
fn fold_phi_d(phi_d: f32) -> f32 {
    let mut p = wrap_tau(phi_d);
    if p >= MB_PI {
        p -= MB_PI;
    }
    p.clamp(0.0, MB_PI)
}

/// Wraps an angle into `[0, 2*pi)`.
#[must_use]
fn wrap_tau(angle: f32) -> f32 {
    let mut a = ops::rem_euclid(angle, MB_TAU);
    if a < 0.0 {
        a += MB_TAU;
    }
    a
}

/// Flat index into a table cell buffer (`phi_d` fastest, `theta_h` slowest).
#[must_use]
const fn cell_index(dims: MeasuredBrdfDims, i: usize, j: usize, k: usize) -> usize {
    (i * dims.theta_d + j) * dims.phi_d + k
}

/// Flat index into a per-bin sampler buffer (`phi` fastest, `theta_i` slowest).
#[must_use]
const fn sampler_index(mu: usize, phi: usize, ii: usize, jj: usize, kk: usize) -> usize {
    (ii * mu + jj) * phi + kk
}

/// Flat index into a per-bin marginal buffer.
#[must_use]
const fn marginal_index(mu: usize, ii: usize, jj: usize) -> usize {
    ii * mu + jj
}

/// Centre of bin `i` of `n` uniform bins spanning `[0, 1]`.
#[must_use]
fn axis_center(i: usize, n: usize) -> f32 {
    (i as f32 + 0.5) / n as f32
}

/// Maps a `[0, 1]` unit coordinate to a continuous grid coordinate whose cell
/// centres sit at integer indices, clamped to the valid sampling range.
#[must_use]
fn unit_to_grid(u: f32, n: usize) -> f32 {
    let g = clamp01(u) * n as f32 - 0.5;
    g.clamp(0.0, (n - 1) as f32)
}

/// Splits a continuous grid coordinate into its lower integer cell and the
/// `[0, 1]` fraction towards the next cell.
#[must_use]
fn split_grid(grid: f32, n: usize) -> (usize, f32) {
    let g = grid.clamp(0.0, (n - 1) as f32);
    let base = ops::floor(g);
    let i = (base as usize).min(n - 1);
    (i, (g - base).clamp(0.0, 1.0))
}

/// Inverse-CDF lookup over a monotone `[row, row + n)` CDF normalised to end at
/// one, returning the chosen bin and the `[0, 1]` fraction within it.
#[must_use]
fn sample_cdf_row(cdf: &[f32], row: usize, n: usize, u: f32) -> (usize, f32) {
    let u = clamp01(u);
    let mut prev = 0.0_f32;
    for k in 0..n {
        let c = cdf[row + k];
        if u < c || k == n - 1 {
            let span = (c - prev).max(f32::MIN_POSITIVE);
            return (k, ((u - prev) / span).clamp(0.0, 1.0));
        }
        prev = c;
    }
    (n - 1, 0.0)
}

/// Scalar reduction of an RGB BRDF value used to drive importance sampling.
#[must_use]
fn scalar_brdf(rgb: [f32; 3]) -> f32 {
    ((rgb[0] + rgb[1] + rgb[2]) / 3.0).max(0.0)
}

/// Analytic Cook-Torrance (GGX) + Lambert BRDF used to synthesise a table.
#[must_use]
fn analytic_brdf(wi: [f32; 3], wo: [f32; 3], params: SyntheticBrdfParams) -> [f32; 3] {
    let n = [0.0, 0.0, 1.0];
    let n_dot_l = dot(n, wi);
    let n_dot_v = dot(n, wo);
    if n_dot_l <= 0.0 || n_dot_v <= 0.0 {
        return [0.0; 3];
    }
    let h = half_vector(wi, wo);
    let n_dot_h = dot(n, h).max(0.0);
    let v_dot_h = dot(wo, h).max(0.0);
    let n_dot_l = n_dot_l.max(MB_MIN_COS);
    let n_dot_v = n_dot_v.max(MB_MIN_COS);

    let metallic = params.metallic.clamp(0.0, 1.0);
    let reflectance = params.reflectance.clamp(0.0, 1.0);
    let roughness = params.roughness.clamp(MB_MIN_ROUGHNESS, 1.0);
    let alpha = roughness * roughness;
    let base = [
        params.base_color[0].max(0.0),
        params.base_color[1].max(0.0),
        params.base_color[2].max(0.0),
    ];

    let f0_dielectric = 0.16 * reflectance * reflectance;
    let f0 = [
        f0_dielectric * (1.0 - metallic) + base[0] * metallic,
        f0_dielectric * (1.0 - metallic) + base[1] * metallic,
        f0_dielectric * (1.0 - metallic) + base[2] * metallic,
    ];

    // GGX NDF.
    let d_denom = n_dot_h * n_dot_h * (alpha * alpha - 1.0) + 1.0;
    let d = alpha * alpha / (MB_PI * d_denom * d_denom).max(f32::MIN_POSITIVE);

    // Height-correlated Smith visibility (includes the 1/(4 nl nv) term).
    let a2 = alpha * alpha;
    let lambda_v = n_dot_l * ops::sqrt(n_dot_v * n_dot_v * (1.0 - a2) + a2);
    let lambda_l = n_dot_v * ops::sqrt(n_dot_l * n_dot_l * (1.0 - a2) + a2);
    let vis = 0.5 / (lambda_v + lambda_l).max(f32::MIN_POSITIVE);

    // Schlick Fresnel.
    let fc = pow5(1.0 - v_dot_h);
    let fresnel = [
        f0[0] + (1.0 - f0[0]) * fc,
        f0[1] + (1.0 - f0[1]) * fc,
        f0[2] + (1.0 - f0[2]) * fc,
    ];

    let diffuse_scale = (1.0 - metallic) / MB_PI;
    let mut out = [0.0_f32; 3];
    for c in 0..3 {
        let specular = d * vis * fresnel[c];
        out[c] = (base[c] * diffuse_scale + specular).max(0.0);
    }
    out
}

/// `x^5` via three multiplies.
#[must_use]
fn pow5(x: f32) -> f32 {
    let x2 = x * x;
    x2 * x2 * x
}

/// Dot product of two 3-vectors.
#[must_use]
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Clamps to `[0, 1]`.
#[must_use]
fn clamp01(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

/// Linear interpolation of two RGB triples.
#[must_use]
fn lerp3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}

/// Normalises `v`, returning `fallback` for degenerate / non-finite inputs.
#[must_use]
fn normalize_or(v: [f32; 3], fallback: [f32; 3]) -> [f32; 3] {
    let len_sq = dot(v, v);
    if len_sq > MB_MIN_LEN_SQ && len_sq.is_finite() {
        let inv = ops::sqrt(len_sq).recip();
        [v[0] * inv, v[1] * inv, v[2] * inv]
    } else {
        fallback
    }
}

/// Normalises `v`, returning the zero vector for degenerate inputs (used where
/// a below-horizon zero result is the correct downstream behaviour).
#[must_use]
fn normalize_or_zero(v: [f32; 3]) -> [f32; 3] {
    normalize_or(v, [0.0, 0.0, 0.0])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic Van der Corput radical inverse in base 2.
    fn vdc2(mut bits: u32) -> f32 {
        bits = bits.rotate_left(16);
        bits = ((bits & 0x5555_5555) << 1) | ((bits & 0xAAAA_AAAA) >> 1);
        bits = ((bits & 0x3333_3333) << 2) | ((bits & 0xCCCC_CCCC) >> 2);
        bits = ((bits & 0x0F0F_0F0F) << 4) | ((bits & 0xF0F0_F0F0) >> 4);
        bits = ((bits & 0x00FF_00FF) << 8) | ((bits & 0xFF00_FF00) >> 8);
        (bits as f32) * 2.328_306_4e-10
    }

    /// `i`-th 2D Hammersley point of `n` (deterministic low-discrepancy).
    fn hammersley(i: u32, n: u32) -> (f32, f32) {
        (i as f32 / n as f32, vdc2(i))
    }

    fn dir(theta: f32, phi: f32) -> [f32; 3] {
        let (st, ct) = ops::sin_cos(theta);
        let (sp, cp) = ops::sin_cos(phi);
        [st * cp, st * sp, ct]
    }

    fn vlen(v: [f32; 3]) -> f32 {
        dot(v, v).sqrt()
    }

    fn test_brdf() -> MeasuredBrdf {
        MeasuredBrdf::from_analytic(MeasuredBrdfDims::MERL, SyntheticBrdfParams::default())
    }

    #[test]
    fn dims_validation_and_errors() {
        assert!(MeasuredBrdfDims::MERL.is_valid());
        assert_eq!(MeasuredBrdfDims::MERL.cell_count(), 90 * 90 * 180);
        let degenerate = MeasuredBrdfDims {
            theta_h: 1,
            theta_d: 4,
            phi_d: 4,
        };
        assert!(!degenerate.is_valid());
        assert_eq!(
            MeasuredBrdf::new(degenerate, vec![[0.0; 3]; 16]),
            Err(MeasuredBrdfError::DegenerateDims)
        );
        let dims = MeasuredBrdfDims {
            theta_h: 2,
            theta_d: 2,
            phi_d: 2,
        };
        assert_eq!(
            MeasuredBrdf::new(dims, vec![[0.0; 3]; 7]),
            Err(MeasuredBrdfError::CellCountMismatch {
                expected: 8,
                found: 7,
            })
        );
        assert!(MeasuredBrdf::new(dims, vec![[0.0; 3]; 8]).is_ok());
    }

    #[test]
    fn theta_h_warp_is_invertible() {
        for i in 0..=20 {
            let theta = i as f32 / 20.0 * MB_FRAC_PI_2;
            let round = unit_to_theta_h(theta_h_to_unit(theta));
            assert!((round - theta).abs() < 1.0e-4, "theta_h {theta} -> {round}");
        }
        // The warp spends more unit range on small angles than large ones.
        assert!(theta_h_to_unit(0.1 * MB_FRAC_PI_2) > 0.1);
    }

    #[test]
    fn half_vector_falls_back_to_normal() {
        let h = half_vector([0.3, 0.0, 0.95], [-0.3, 0.0, -0.95]);
        assert!((h[2] - 1.0).abs() < 1.0e-5);
        assert!((vlen(h) - 1.0).abs() < 1.0e-5);
    }

    #[test]
    fn coordinate_roundtrip_recovers_directions() {
        let mut tested = 0;
        for ih in 0..8 {
            let theta_h = (ih as f32 + 0.5) / 8.0 * (MB_FRAC_PI_2 * 0.95);
            for id in 0..8 {
                let theta_d = (id as f32 + 0.5) / 8.0 * (MB_FRAC_PI_2 * 0.95);
                for ip in 0..6 {
                    let phi_d = (ip as f32 + 0.5) / 6.0 * MB_PI;
                    let coords = RusinkiewiczCoords {
                        theta_h,
                        theta_d,
                        phi_d,
                    };
                    let (wi, wo) = rusinkiewicz_to_directions(coords);
                    if wi[2] <= 1.0e-3 || wo[2] <= 1.0e-3 {
                        continue;
                    }
                    assert!((vlen(wi) - 1.0).abs() < 1.0e-4);
                    assert!((vlen(wo) - 1.0).abs() < 1.0e-4);
                    let back = directions_to_rusinkiewicz(wi, wo);
                    assert!(
                        (back.theta_h - theta_h).abs() < 2.0e-3,
                        "theta_h {theta_h} -> {}",
                        back.theta_h
                    );
                    assert!(
                        (back.theta_d - theta_d).abs() < 2.0e-3,
                        "theta_d {theta_d} -> {}",
                        back.theta_d
                    );
                    assert!(
                        (back.phi_d - phi_d).abs() < 3.0e-3,
                        "phi_d {phi_d} -> {}",
                        back.phi_d
                    );
                    tested += 1;
                }
            }
        }
        assert!(
            tested > 100,
            "expected a broad roundtrip sweep, got {tested}"
        );
    }

    #[test]
    fn reflection_pair_shares_half_vector() {
        let coords = RusinkiewiczCoords {
            theta_h: 0.4,
            theta_d: 0.5,
            phi_d: 1.1,
        };
        let (wi, wo) = rusinkiewicz_to_directions(coords);
        let h = half_vector(wi, wo);
        // wi and wo must mirror about h: their dots with h are equal.
        assert!((dot(wi, h) - dot(wo, h)).abs() < 1.0e-4);
    }

    #[test]
    fn grid_coords_stay_in_bounds() {
        let brdf = MeasuredBrdf::from_analytic(
            MeasuredBrdfDims {
                theta_h: 16,
                theta_d: 16,
                phi_d: 32,
            },
            SyntheticBrdfParams::default(),
        );
        for coords in [
            RusinkiewiczCoords {
                theta_h: -1.0,
                theta_d: -1.0,
                phi_d: -10.0,
            },
            RusinkiewiczCoords {
                theta_h: 100.0,
                theta_d: 100.0,
                phi_d: 100.0,
            },
            RusinkiewiczCoords {
                theta_h: 0.3,
                theta_d: 0.3,
                phi_d: 0.3,
            },
        ] {
            let g = brdf.grid_coords(coords);
            assert!((0.0..=15.0).contains(&g[0]), "theta_h grid {}", g[0]);
            assert!((0.0..=15.0).contains(&g[1]), "theta_d grid {}", g[1]);
            assert!((0.0..=31.0).contains(&g[2]), "phi_d grid {}", g[2]);
        }
    }

    #[test]
    fn trilinear_hits_cell_centers_and_midpoints() {
        let dims = MeasuredBrdfDims {
            theta_h: 2,
            theta_d: 2,
            phi_d: 2,
        };
        // Distinct per-cell values so interpolation is observable.
        let mut cells = vec![[0.0_f32; 3]; dims.cell_count()];
        for (idx, cell) in cells.iter_mut().enumerate() {
            *cell = [idx as f32, 0.0, 0.0];
        }
        let brdf = MeasuredBrdf::new(dims, cells).unwrap();
        // Integer grid coordinates reproduce cells exactly.
        assert_eq!(brdf.sample_grid([0.0, 0.0, 0.0])[0], 0.0);
        assert_eq!(brdf.sample_grid([1.0, 1.0, 1.0])[0], 7.0);
        assert_eq!(brdf.sample_grid([0.0, 0.0, 1.0])[0], 1.0);
        // Midpoint along phi averages the bracketing cells (0 and 1).
        assert!((brdf.sample_grid([0.0, 0.0, 0.5])[0] - 0.5).abs() < 1.0e-6);
        // Centre of the cube averages all eight corners (mean of 0..7 = 3.5).
        assert!((brdf.sample_grid([0.5, 0.5, 0.5])[0] - 3.5).abs() < 1.0e-6);
    }

    #[test]
    fn synthetic_table_is_finite_and_non_negative() {
        let brdf = test_brdf();
        assert!(is_non_negative(&brdf));
    }

    #[test]
    fn evaluate_zero_below_horizon() {
        let brdf = test_brdf();
        assert_eq!(brdf.evaluate([0.0, 0.0, -1.0], [0.0, 0.0, 1.0]), [0.0; 3]);
        assert_eq!(brdf.evaluate([0.0, 0.0, 1.0], [0.0, 0.0, -1.0]), [0.0; 3]);
        assert_eq!(brdf.evaluate([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]), [0.0; 3]);
    }

    #[test]
    fn tabulated_evaluation_tracks_the_analytic_closure() {
        let params = SyntheticBrdfParams::default();
        let brdf = MeasuredBrdf::from_analytic(MeasuredBrdfDims::MERL, params);
        let mut max_rel = 0.0_f32;
        for it in 1..9 {
            let theta_i = it as f32 / 10.0 * MB_FRAC_PI_2;
            let wi = dir(theta_i, 0.3);
            for ot in 1..9 {
                let theta_o = ot as f32 / 10.0 * MB_FRAC_PI_2;
                for op in 0..6 {
                    let phi_o = op as f32 / 6.0 * MB_TAU;
                    let wo = dir(theta_o, phi_o);
                    let reference = analytic_brdf(wi, wo, params);
                    let tabulated = brdf.evaluate(wi, wo);
                    for c in 0..3 {
                        let denom = reference[c].abs().max(0.05);
                        max_rel = max_rel.max((tabulated[c] - reference[c]).abs() / denom);
                    }
                }
            }
        }
        assert!(max_rel < 0.25, "tabulated vs analytic rel error {max_rel}");
    }

    #[test]
    fn evaluation_is_deterministic_and_reciprocal() {
        let brdf = test_brdf();
        let wi = dir(0.6, 0.2);
        let wo = dir(0.9, 2.1);
        assert_eq!(brdf.evaluate(wi, wo), brdf.evaluate(wi, wo));

        let mut pairs = Vec::new();
        for it in 1..6 {
            for ot in 1..6 {
                for op in 0..4 {
                    let wi = dir(it as f32 / 7.0 * MB_FRAC_PI_2, 0.0);
                    let wo = dir(ot as f32 / 7.0 * MB_FRAC_PI_2, op as f32 / 4.0 * MB_TAU);
                    pairs.push((wi, wo));
                }
            }
        }
        // Reciprocal to the floating-point floor (table is reciprocal by design).
        assert!(reciprocity_error(&brdf, &pairs) < 1.0e-5);
    }

    #[test]
    fn table_build_is_reproducible() {
        let a = MeasuredBrdf::from_analytic(
            MeasuredBrdfDims {
                theta_h: 12,
                theta_d: 12,
                phi_d: 24,
            },
            SyntheticBrdfParams::default(),
        );
        let b = MeasuredBrdf::from_analytic(
            MeasuredBrdfDims {
                theta_h: 12,
                theta_d: 12,
                phi_d: 24,
            },
            SyntheticBrdfParams::default(),
        );
        assert_eq!(a, b);
    }

    #[test]
    fn sampler_draws_valid_upper_hemisphere_directions() {
        let brdf = test_brdf();
        let sampler = MeasuredBrdfSampler::from_brdf(&brdf, 32, 48, 96);
        let wi = dir(0.5, 0.7);
        for i in 0..512 {
            let (u1, u2) = hammersley(i, 512);
            let s = sampler.sample(wi, u1, u2);
            assert!(
                s.direction[2] > 0.0,
                "sample below horizon: {:?}",
                s.direction
            );
            assert!((vlen(s.direction) - 1.0).abs() < 1.0e-3);
            assert!(s.pdf > 0.0, "non-positive pdf {}", s.pdf);
            assert!(s.pdf.is_finite());
        }
    }

    #[test]
    fn sampler_sample_pdf_matches_pdf_query() {
        let brdf = test_brdf();
        let sampler = MeasuredBrdfSampler::from_brdf(&brdf, 32, 48, 96);
        let wi = dir(0.4, 1.3);
        for i in 0..256 {
            let (u1, u2) = hammersley(i, 256);
            let s = sampler.sample(wi, u1, u2);
            let queried = sampler.pdf(wi, s.direction);
            assert!(
                (queried - s.pdf).abs() <= 1.0e-3 * s.pdf.max(1.0),
                "sample pdf {} vs query {}",
                s.pdf,
                queried
            );
        }
    }

    #[test]
    fn sampler_pdf_zero_below_horizon() {
        let brdf = test_brdf();
        let sampler = MeasuredBrdfSampler::from_brdf(&brdf, 16, 24, 48);
        assert_eq!(sampler.pdf([0.0, 0.0, -1.0], dir(0.3, 0.0)), 0.0);
        assert_eq!(sampler.pdf(dir(0.3, 0.0), [0.0, 0.0, -1.0]), 0.0);
        let below = sampler.sample([0.0, 0.0, -1.0], 0.5, 0.5);
        assert_eq!(below.pdf, 0.0);
    }

    #[test]
    fn sampler_pdf_integrates_to_one() {
        let brdf = test_brdf();
        let sampler = MeasuredBrdfSampler::from_brdf(&brdf, 32, 64, 128);
        for &theta_i in &[0.2_f32, 0.7, 1.2] {
            let wi = dir(theta_i, 0.9);
            // Uniform-hemisphere Monte Carlo of integral of pdf dOmega == 1.
            let n = 20000_u32;
            let mut acc = 0.0_f64;
            for i in 0..n {
                let (u1, u2) = hammersley(i, n);
                // Uniform direction over the hemisphere (density 1 / 2pi).
                let cos_o = u1;
                let sin_o = (1.0 - cos_o * cos_o).max(0.0).sqrt();
                let phi = MB_TAU * u2;
                let (sp, cp) = ops::sin_cos(phi);
                let wo = [sin_o * cp, sin_o * sp, cos_o];
                acc += (sampler.pdf(wi, wo) as f64) * (MB_TAU as f64);
            }
            let integral = acc / n as f64;
            assert!(
                (integral - 1.0).abs() < 0.05,
                "theta_i {theta_i}: integral of pdf = {integral}"
            );
        }
    }

    #[test]
    fn importance_sampling_matches_reference_albedo() {
        let brdf = test_brdf();
        let sampler = MeasuredBrdfSampler::from_brdf(&brdf, 48, 96, 192);
        for &theta_i in &[0.25_f32, 0.6, 1.0] {
            let wi = dir(theta_i, 0.4);

            // Deterministic grid reference of the cosine-weighted directional
            // albedo (scalar) in (mu, phi) space.
            let (mm, pp) = (128_u32, 256_u32);
            let mut reference = 0.0_f64;
            for a in 0..mm {
                let mu = (a as f32 + 0.5) / mm as f32;
                let sin_o = (1.0 - mu * mu).max(0.0).sqrt();
                for b in 0..pp {
                    let phi = (b as f32 + 0.5) / pp as f32 * MB_TAU;
                    let (sp, cp) = ops::sin_cos(phi);
                    let wo = [sin_o * cp, sin_o * sp, mu];
                    reference += (scalar_brdf(brdf.evaluate(wi, wo)) * mu) as f64;
                }
            }
            reference *= (1.0 / mm as f64) * (MB_TAU as f64 / pp as f64);

            // Importance-sampled estimate of the same integral.
            let n = 20000_u32;
            let mut estimate = 0.0_f64;
            for i in 0..n {
                let (u1, u2) = hammersley(i, n);
                let s = sampler.sample(wi, u1, u2);
                if s.pdf > 0.0 {
                    let f = scalar_brdf(brdf.evaluate(wi, s.direction));
                    estimate += (f * s.direction[2] / s.pdf) as f64;
                }
            }
            estimate /= n as f64;

            let rel = (estimate - reference).abs() / reference.max(1.0e-4);
            assert!(
                rel < 0.08,
                "theta_i {theta_i}: IS estimate {estimate} vs reference {reference} (rel {rel})"
            );
        }
    }

    #[test]
    fn sampler_is_deterministic() {
        let brdf = test_brdf();
        let sampler = MeasuredBrdfSampler::from_brdf(&brdf, 16, 32, 64);
        let wi = dir(0.5, 0.5);
        let a = sampler.sample(wi, 0.37, 0.82);
        let b = sampler.sample(wi, 0.37, 0.82);
        assert_eq!(a, b);
    }
}
