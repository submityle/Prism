//! Stable CPU density coverage and screen-space ordered dithering.

use std::fmt;

use crate::{ProxyInstance, VisibilityInput};

/// Continuous density and optional distance-fade policy.
///
/// Sampling depends only on `(proxy_id, cluster_id, seed)`, never on a dense
/// storage slot or iteration order. Repeated identity tuples across sectors
/// therefore have correlated ranks; sector identity is not part of the hash.
/// `density` controls a smooth coverage curve, not an exact survivor fraction.
/// Floating-point parameters must be finite and normal or zero; transition
/// must be positive and normal. Nonzero subnormal parameters are rejected.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DensityPolicy {
    density: f32,
    transition: f32,
    seed: u32,
    distance_fade: f32,
}

/// A policy parameter is non-finite or outside its supported range.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DensityPolicyError {
    /// Density must be finite, normal or zero, and in `[0, 1]`.
    InvalidDensity,
    /// Transition width must be positive, normal, and in `(0, 1]`.
    InvalidTransition,
    /// Distance-fade width must be finite, normal or zero, and nonnegative.
    InvalidDistanceFade,
}

impl fmt::Display for DensityPolicyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::InvalidDensity => "density must be finite, normal or zero, and in [0, 1]",
            Self::InvalidTransition => "transition must be positive, normal, and in (0, 1]",
            Self::InvalidDistanceFade => {
                "distance fade must be finite, normal or zero, and in [0, f32::MAX]"
            }
        })
    }
}

impl std::error::Error for DensityPolicyError {}

impl DensityPolicy {
    /// Validates all floating-point parameters without silently clamping them.
    ///
    /// Density and distance fade must be normal or zero; transition must be
    /// positive and normal. Nonzero subnormals are not portable GPU inputs.
    pub fn new(
        density: f32,
        transition: f32,
        seed: u32,
        distance_fade: f32,
    ) -> Result<Self, DensityPolicyError> {
        if !density.is_finite() || density.is_subnormal() || !(0.0..=1.0).contains(&density) {
            return Err(DensityPolicyError::InvalidDensity);
        }
        if !transition.is_normal() || transition <= 0.0 || transition > 1.0 {
            return Err(DensityPolicyError::InvalidTransition);
        }
        if !distance_fade.is_finite() || distance_fade.is_subnormal() || distance_fade < 0.0 {
            return Err(DensityPolicyError::InvalidDistanceFade);
        }
        Ok(Self {
            density,
            transition,
            seed,
            distance_fade,
        })
    }

    /// Returns the density curve's control value.
    #[must_use]
    pub const fn density(self) -> f32 {
        self.density
    }

    /// Returns the density transition width.
    #[must_use]
    pub const fn transition(self) -> f32 {
        self.transition
    }

    /// Returns the stable sampling seed.
    #[must_use]
    pub const fn seed(self) -> u32 {
        self.seed
    }

    /// Returns the distance-fade width; zero disables distance fading.
    #[must_use]
    pub const fn distance_fade(self) -> f32 {
        self.distance_fade
    }

    /// Returns a deterministic rank in `[0, 1)` from the hash's top 24 bits.
    ///
    /// Only the identity tuple and seed contribute, including when the same
    /// tuple appears in different sectors or moves to another dense slot.
    #[must_use]
    pub fn sample_rank(self, proxy_id: u32, cluster_id: u32) -> f32 {
        let hash = pcg_hash(pcg_hash(proxy_id ^ self.seed) ^ cluster_id);
        (hash >> 8) as f32 / 16_777_216.0
    }

    /// Returns smooth density coverage multiplied by optional distance coverage.
    ///
    /// The caller must perform visibility rejection first (for example, with
    /// `VisibilityInput::evaluate`). This method does not test the frustum or
    /// bounding sphere, and with distance fading disabled it does not test range.
    /// Proxy and camera coordinates and nonnegative range limits are expected to
    /// satisfy the visibility contract. With distance fading enabled, effective
    /// widths at or below `f32::MIN_POSITIVE` are degenerate and return zero.
    ///
    /// Distance uses only f32 arithmetic. For limits below one, coordinates
    /// and limits are boosted by 2^64 before taking small differences;
    /// subnormal coordinates are bit-decoded into normal boosted values.
    /// Large coordinates subtract before boosting to avoid infinity minus
    /// infinity. Otherwise subtract each axis directly. In these units take
    /// `scale = max(abs(dx), abs(dy), abs(dz), f32::MIN_POSITIVE)`, reject
    /// `scale > limit`,
    /// then evaluate `sqrt((dx/scale)^2 + (dy/scale)^2 + (dz/scale)^2) * scale`
    /// in that order. Thus subtraction overflow rejects, while squaring large
    /// finite deltas does not overflow. The normal scale floor avoids GPU
    /// subnormal flushing without rejecting normal tiny ranges at the origin.
    /// Smoothstep results and final products below `f32::MIN_POSITIVE` are
    /// zeroed. CPU/GPU coverage is compared
    /// with absolute tolerance `2e-5` in regression fixtures, not a universal
    /// error bound or bitwise-equality guarantee. Extremely narrow fades can
    /// amplify ordinary floating-point distance error.
    #[must_use]
    pub fn coverage(self, proxy: &ProxyInstance, camera: &VisibilityInput) -> f32 {
        let rank = self.sample_rank(proxy.proxy_id, proxy.cluster_id);
        // Algebraically equivalent to (density-rank*(1-transition))/transition,
        // without subtracting an independently rounded product near density.
        // The exact identity case also avoids cancellation for tiny density.
        let quality = if self.transition == 1.0 {
            self.density
        } else {
            (rank + (self.density - rank) / self.transition).clamp(0.0, 1.0)
        };
        let quality = smoothstep01(quality);
        if self.distance_fade == 0.0 {
            return quality;
        }

        let limit = proxy.max_distance.min(camera.max_distance);
        let width = self.distance_fade.min(limit);
        if width <= f32::MIN_POSITIVE {
            return 0.0;
        }

        // Below one metre, work in units boosted by 2^64. Even a difference
        // between adjacent tiny normals then remains normal on FTZ hardware.
        // Bit-decode subnormal coordinates before any float arithmetic.
        let small = limit < 1.0;
        let boost = if small {
            18_446_744_073_709_551_616.0
        } else {
            1.0
        };
        let delta = |axis: usize| {
            let a = proxy.position[axis];
            let b = camera.camera_position[axis];
            if small {
                boosted_delta(a, b)
            } else {
                a - b
            }
        };
        let [dx, dy, dz] = [delta(0), delta(1), delta(2)];
        let limit = limit * boost;
        let width = width * boost;
        let scale = dx
            .abs()
            .max(dy.abs())
            .max(dz.abs())
            .max(f32::MIN_POSITIVE * boost);
        if scale > limit {
            return 0.0;
        }
        let nx = dx / scale;
        let ny = dy / scale;
        let nz = dz / scale;
        let distance = (nx * nx + ny * ny + nz * nz).sqrt() * scale;
        let distance_coverage = ((limit - distance) / width).clamp(0.0, 1.0);
        flush_subnormal_coverage(quality * smoothstep01(distance_coverage))
    }
}

// All callers have finite coordinates. Large equal coordinates must subtract
// before boosting (to avoid inf-inf); their ULP is already safely normal.
fn boosted_delta(a: f32, b: f32) -> f32 {
    const BOOST: f32 = 18_446_744_073_709_551_616.0;
    if a.abs().max(b.abs()) > 1.0 {
        (a - b) * BOOST
    } else {
        boost_small(a) - boost_small(b)
    }
}

fn boost_small(value: f32) -> f32 {
    let bits = value.to_bits();
    if bits & 0x7f80_0000 != 0 {
        return value * 18_446_744_073_709_551_616.0;
    }
    // 2^-149 (subnormal unit) * 2^64 = 2^-85, a normal value.
    let magnitude = (bits & 0x007f_ffff) as f32 * f32::from_bits(42 << 23);
    if bits >> 31 != 0 {
        -magnitude
    } else {
        magnitude
    }
}

impl Default for DensityPolicy {
    fn default() -> Self {
        Self::new(1.0, 0.1, 0, 0.0).expect("default density policy is valid")
    }
}

/// Tests coverage against a phase-shifted 4x4 Bayer threshold.
///
/// The low two phase bits shift x and the next two shift y, wrapping pixels.
/// Coverage is compared strictly to cell-centred thresholds: zero never passes,
/// one always passes, and NaN never passes. Values are not otherwise clamped.
#[must_use]
pub fn density_dither_visible(coverage: f32, pixel: [u32; 2], phase: u32) -> bool {
    const BAYER: [u8; 16] = [0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5];
    let x = pixel[0].wrapping_add(phase & 3) & 3;
    let y = pixel[1].wrapping_add((phase >> 2) & 3) & 3;
    let threshold = (f32::from(BAYER[(y * 4 + x) as usize]) + 0.5) / 16.0;
    coverage > threshold
}

fn pcg_hash(x: u32) -> u32 {
    let state = x.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
    let word = ((state >> ((state >> 28) + 4)) ^ state).wrapping_mul(277_803_737);
    (word >> 22) ^ word
}

fn smoothstep01(c: f32) -> f32 {
    // Prevent fast-math reassociation from restoring an underflowing c*c.
    // For small positive c, shift its exponent by 32 through integer bits;
    // all polynomial intermediates are then normal until the final rescale.
    if c < f32::from_bits(95 << 23) {
        if c < f32::from_bits(63 << 23) {
            return 0.0; // Even the exact polynomial is below MIN_POSITIVE.
        }
        let scaled = f32::from_bits(c.to_bits() + (32 << 23));
        return flush_subnormal_coverage(
            scaled * scaled * (3.0 - 2.0 * c) * f32::from_bits(63 << 23),
        );
    }
    flush_subnormal_coverage(c * c * (3.0 - 2.0 * c))
}

fn flush_subnormal_coverage(coverage: f32) -> f32 {
    if coverage < f32::MIN_POSITIVE {
        0.0
    } else {
        coverage
    }
}

#[cfg(test)]
mod density_tests {
    include!("density_tests.rs");
}
