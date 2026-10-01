//! Per-triangle texel/world area ratio, the constant term `Delta` shared by
//! every ray-cone LOD query against a given triangle + texture pairing.
//!
//! Intuitively `Delta` answers: "how many texels does one unit of world area
//! cover on this triangle?" Encoding it in `log2` space lets the per-hit LOD
//! become a single addition.
//!
//! # Conventions
//! * `tex_width` / `tex_height` are the mip-0 dimensions of the sampled texture
//!   in texels.
//! * World positions are in consistent world units (meters). UVs are in the
//!   unit square before wrapping.
//! * A degenerate triangle (near-zero world or texture area) is clamped to a
//!   large-but-finite `Delta` so downstream LOD stays finite and biases toward
//!   a coarse (safe) mip rather than producing `NaN`.
//!
//! # References
//! Ray Tracing Gems 2019, ch. 20, eq. "Delta_i" (triangle LOD constant).

use super::math::{cross3, length3, sub3, uv_double_area};

/// A triangle's texel-density constant `Delta = 0.5 * log2(T_a / W_a)`.
///
/// `T_a` is the UV-triangle area measured in texels squared and `W_a` is the
/// world-space triangle area; both use the twice-area cross products, whose
/// factor of two cancels in the ratio.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TriangleLodConstant {
    delta: f32,
}

/// Smallest area (either space) treated as non-degenerate, in squared units.
const MIN_AREA: f32 = 1.0e-12;

/// Clamp bound for `Delta` so a pathological triangle cannot push the final
/// LOD past any plausible mip pyramid (2^24 texels is far beyond 16K).
const DELTA_CLAMP: f32 = 24.0;

impl TriangleLodConstant {
    /// Build the constant from the triangle's three world positions, its three
    /// UVs, and the sampled texture's mip-0 dimensions in texels.
    #[must_use]
    pub fn new(
        world: [[f32; 3]; 3],
        uv: [[f32; 2]; 3],
        tex_width: u32,
        tex_height: u32,
    ) -> Self {
        // Texture-space twice-area in texels^2.
        let uv_area2 = uv_double_area(uv[0], uv[1], uv[2]).abs();
        let texels = (tex_width as f32) * (tex_height as f32);
        let t_a = (texels * uv_area2).max(MIN_AREA);

        // World-space twice-area.
        let e1 = sub3(world[1], world[0]);
        let e2 = sub3(world[2], world[0]);
        let w_a = length3(cross3(e1, e2)).max(MIN_AREA);

        let delta = 0.5 * (t_a / w_a).log2();
        Self {
            delta: delta.clamp(-DELTA_CLAMP, DELTA_CLAMP),
        }
    }

    /// The raw `Delta` value in `log2` texels-per-world-unit space.
    #[inline]
    #[must_use]
    pub fn delta(self) -> f32 {
        self.delta
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A unit-world triangle mapped to a unit-UV region on a 1x1 texture has
    /// `T_a == W_a`, so `Delta == 0`.
    #[test]
    fn unit_mapping_has_zero_delta() {
        let world = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let uv = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let c = TriangleLodConstant::new(world, uv, 1, 1);
        assert!(c.delta().abs() < 1.0e-6, "delta = {}", c.delta());
    }

    /// Doubling the texture resolution in each axis quadruples texel density,
    /// so Delta increases by exactly 0.5 * log2(4) = 1.0.
    #[test]
    fn resolution_scales_delta_in_log2() {
        let world = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let uv = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let lo = TriangleLodConstant::new(world, uv, 256, 256).delta();
        let hi = TriangleLodConstant::new(world, uv, 512, 512).delta();
        assert!((hi - lo - 1.0).abs() < 1.0e-5, "lo={lo} hi={hi}");
    }

    /// A degenerate (zero-area) triangle must not produce NaN/inf.
    #[test]
    fn degenerate_triangle_is_finite() {
        let world = [[0.0; 3]; 3];
        let uv = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]];
        let c = TriangleLodConstant::new(world, uv, 1024, 1024);
        assert!(c.delta().is_finite());
    }
}
