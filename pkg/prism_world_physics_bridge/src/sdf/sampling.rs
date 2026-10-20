//! Continuous distance and analytic gradient from the same resident SDF samples.

use prism_world_residency::{BrickCoord, SdfError};

use super::{SdfBridgeError, SdfPhysicsSnapshot, SdfQueryResult};

/// Trilinear distance and its derivative in physical world coordinates.
///
/// The gradient is dimensionless (millimetres of distance per millimetre of
/// motion), points toward increasing signed distance, and is not normalized.
/// Quantization and interpolation do not guarantee a unit-length gradient.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdfDistanceGradient {
    /// Signed interpolated distance, in millimetres.
    pub distance_mm: f64,
    /// World X/Y/Z derivatives of the interpolated distance.
    pub gradient: [f64; 3],
}

impl SdfDistanceGradient {
    /// Returns a unit direction toward increasing distance, or `None` for a
    /// flat field or a nonfinite value constructed outside this crate.
    #[must_use]
    pub fn unit_normal(self) -> Option<[f64; 3]> {
        // Scaling first also handles very small or large finite user values
        // without underflowing or overflowing the squared length.
        if !self.gradient.iter().all(|value| value.is_finite()) {
            return None;
        }
        let scale = self
            .gradient
            .iter()
            .fold(0.0_f64, |max, value| max.max(value.abs()));
        if scale == 0.0 {
            return None;
        }
        let scaled = self.gradient.map(|value| value / scale);
        let length = scaled.iter().map(|value| value * value).sum::<f64>().sqrt();
        Some(scaled.map(|value| value / length))
    }
}

impl SdfPhysicsSnapshot {
    /// Interpolates distance and analytically differentiates the same eight
    /// samples, without allocation, locking, or mutating the snapshot.
    ///
    /// Input positions retain the world's integer millimetre precision.
    /// Subtraction, floor and remainder use `i128` before conversion to local
    /// floating-point weights, even at an `i64` world origin. Mip spacing and
    /// distance scale both contribute to the physical gradient.
    ///
    /// All eight corners must be resident, including across brick boundaries;
    /// otherwise the result is `Unknown`, never a fabricated empty distance
    /// or normal. At lattice knots the gradient is the derivative from the
    /// positive-side cell; derivatives need not be continuous across cells.
    /// The upper corner must fit the signed-i32 voxel ABI even at a knot.
    /// This is interpolation of stored values, not an exact geometry distance.
    pub fn query_world_distance_gradient_mm(
        &self,
        world_position_mm: [i64; 3],
        mip: u8,
    ) -> Result<SdfQueryResult<SdfDistanceGradient>, SdfBridgeError> {
        let transform = self.transform();
        let spacing = transform.mip_voxel_size_mm(mip)?;
        if mip >= self.grid().mip_levels() {
            return Err(SdfBridgeError::GridQuery(SdfError::InvalidMip));
        }
        let mut corners = [[0_i32; 2]; 3];
        let mut fractions = [0.0; 3];
        for axis in 0..3 {
            let delta =
                i128::from(world_position_mm[axis]) - i128::from(transform.origin_mm()[axis]);
            let lower = i32::try_from(delta.div_euclid(spacing))
                .map_err(|_| SdfBridgeError::VoxelCoordinateOutOfRange)?;
            let upper = lower
                .checked_add(1)
                .ok_or(SdfBridgeError::VoxelCoordinateOutOfRange)?;
            let fraction = delta.rem_euclid(spacing) as f64 / spacing as f64;
            corners[axis] = [lower, upper];
            fractions[axis] = fraction;
        }

        let edge = i32::from(self.grid().brick_size());
        let mut samples = [0.0; 8];
        for (index, sample) in samples.iter_mut().enumerate() {
            let voxel = [
                corners[0][index & 1],
                corners[1][(index >> 1) & 1],
                corners[2][index >> 2],
            ];
            let brick = voxel.map(|value| value.div_euclid(edge));
            let local = voxel.map(|value| value.rem_euclid(edge) as usize);
            // Euclidean remainders avoid reconstructing a brick origin below
            // i32::MIN for non-power-of-two brick sizes.
            let Some(values) = self
                .grid()
                .brick_samples(BrickCoord::new(brick[0], brick[1], brick[2], mip))
                .map_err(SdfBridgeError::GridQuery)?
            else {
                return Ok(SdfQueryResult::Unknown);
            };
            let offset = (local[2] * edge as usize + local[1]) * edge as usize + local[0];
            *sample = f64::from(values[offset]);
        }

        let scale = f64::from(transform.distance_scale_mm());
        let derivative_scale = scale / spacing as f64;
        let [x, y, z] = fractions;
        let [a, b, c, d, e, f, g, h] = samples;
        let lower = lerp(lerp(a, b, x), lerp(c, d, x), y);
        let upper = lerp(lerp(e, f, x), lerp(g, h, x), y);
        // Difference corner pairs before interpolation, so a constant field
        // has exactly zero gradient rather than cancellation noise.
        let gradient = [
            lerp(lerp(b - a, d - c, y), lerp(f - e, h - g, y), z),
            lerp(lerp(c - a, d - b, x), lerp(g - e, h - f, x), z),
            lerp(lerp(e - a, f - b, x), lerp(g - c, h - d, x), y),
        ];
        Ok(SdfQueryResult::Value(SdfDistanceGradient {
            distance_mm: lerp(lower, upper, z) * scale,
            gradient: gradient.map(|value| value * derivative_scale),
        }))
    }
}

fn lerp(lower: f64, upper: f64, fraction: f64) -> f64 {
    lower + (upper - lower) * fraction
}
