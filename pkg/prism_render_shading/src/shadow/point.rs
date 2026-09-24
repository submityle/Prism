//! Omnidirectional (point) light shadow evaluation via a distance cube map.
//!
//! A point light shadows in all directions, so its depth is stored in a cube
//! map: six square faces indexed `+X, -X, +Y, -Y, +Z, -Z` (layers `0..6`,
//! matching the standard cube-map face order).  Rather than a projected NDC
//! depth, each texel stores the **linear distance** from the light to the
//! nearest occluder, normalized by the light's far range, so the comparison is
//! a straightforward distance test that is uniform across faces and free of the
//! perspective depth precision cliff.
//!
//! This is the CPU golden twin of `shadow.wesl`'s point-light path.

use crate::shadow::bias::slope_scaled_depth_bias;
use crate::shadow::filter::{pcf_visibility, ShadowDepthSampler};
use crate::shadow::math::length3;
use crate::vecmath::sub;

/// Standard cube-map face layer indices.
const FACE_POS_X: usize = 0;
const FACE_NEG_X: usize = 1;
const FACE_POS_Y: usize = 2;
const FACE_NEG_Y: usize = 3;
const FACE_POS_Z: usize = 4;
const FACE_NEG_Z: usize = 5;

/// Selects the cube-map face (layer `0..6`) and in-face UV in `[0, 1]^2` for a
/// world-space direction from the light to the fragment.
///
/// Uses the OpenGL cube-map convention: the dominant (largest magnitude) axis
/// picks the face, and the remaining two components, divided by the dominant
/// magnitude, give the face-local `(s, t)` in `[-1, 1]` which map to UV.
pub fn cube_face_and_uv(direction: [f32; 3]) -> (usize, [f32; 2]) {
    let [x, y, z] = direction;
    let ax = x.abs();
    let ay = y.abs();
    let az = z.abs();

    let (face, sc, tc, ma) = if ax >= ay && ax >= az {
        if x >= 0.0 {
            (FACE_POS_X, -z, -y, ax)
        } else {
            (FACE_NEG_X, z, -y, ax)
        }
    } else if ay >= az {
        if y >= 0.0 {
            (FACE_POS_Y, x, z, ay)
        } else {
            (FACE_NEG_Y, x, -z, ay)
        }
    } else if z >= 0.0 {
        (FACE_POS_Z, x, -y, az)
    } else {
        (FACE_NEG_Z, -x, -y, az)
    };

    let inv_ma = ma.max(1.0e-8).recip();
    let u = 0.5 * (sc * inv_ma + 1.0);
    let v = 0.5 * (tc * inv_ma + 1.0);
    (face, [u.clamp(0.0, 1.0), v.clamp(0.0, 1.0)])
}

/// Bias/filter tunables for point-light shadows.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PointShadowConfig {
    /// Constant distance bias in normalized (`distance / range`) units.
    pub const_bias: f32,
    /// Slope-scaled distance bias coefficient (multiplied by `tan(theta)`).
    pub slope_bias: f32,
    /// Maximum total distance bias.
    pub max_bias: f32,
    /// Box PCF half-extent in cube-face texels.
    pub pcf_radius: i32,
    /// UV size of one cube-face texel (`1 / resolution`).
    pub texel_uv_size: [f32; 2],
}

/// Per-fragment inputs for a point-light shadow lookup.
#[derive(Clone, Copy, Debug)]
pub struct PointShadowInput {
    /// World-space position of the shaded surface point.
    pub world_position: [f32; 3],
    /// World-space position of the point light.
    pub light_position: [f32; 3],
    /// Far range of the light's shadow cube map (normalizes stored distance).
    pub range: f32,
    /// Clamped, non-negative `n·l` cosine (drives the slope bias).
    pub n_dot_l: f32,
}

/// Evaluates point-light visibility in `[0, 1]`, sampling the distance cube map
/// through `sampler` (layer = cube face).  `1.0` is fully lit; a fragment
/// beyond the light's range reads off-map as far and stays lit.
pub fn evaluate_point_shadow<S: ShadowDepthSampler>(
    sampler: &S,
    input: &PointShadowInput,
    config: &PointShadowConfig,
) -> f32 {
    let light_to_frag = sub(input.world_position, input.light_position);
    let distance = length3(light_to_frag);
    let range = input.range.max(1.0e-4);
    let reference = (distance / range).clamp(0.0, 1.0);

    let (face, uv) = cube_face_and_uv(light_to_frag);
    let bias = slope_scaled_depth_bias(
        input.n_dot_l,
        config.const_bias,
        config.slope_bias,
        config.max_bias,
    );

    pcf_visibility(
        sampler,
        face,
        uv,
        reference - bias,
        config.texel_uv_size,
        config.pcf_radius,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The +X, -Y and -Z directions each land on their expected face with a
    /// centred UV.
    #[test]
    fn face_selection_matches_dominant_axis() {
        let (f, uv) = cube_face_and_uv([1.0, 0.0, 0.0]);
        assert_eq!(f, FACE_POS_X);
        assert!((uv[0] - 0.5).abs() < 1.0e-6 && (uv[1] - 0.5).abs() < 1.0e-6);

        assert_eq!(cube_face_and_uv([0.0, -2.0, 0.0]).0, FACE_NEG_Y);
        assert_eq!(cube_face_and_uv([0.0, 0.0, -3.0]).0, FACE_NEG_Z);
        assert_eq!(cube_face_and_uv([-5.0, 1.0, 2.0]).0, FACE_NEG_X);
    }

    /// UV stays within the unit square across arbitrary directions.
    #[test]
    fn face_uv_in_unit_square() {
        for dir in [
            [0.3, 0.9, -0.2],
            [-0.7, -0.1, 0.6],
            [0.2, -0.2, 0.95],
            [1.0, 1.0, 1.0],
        ] {
            let (_f, uv) = cube_face_and_uv(dir);
            assert!((0.0..=1.0).contains(&uv[0]), "u out of range: {}", uv[0]);
            assert!((0.0..=1.0).contains(&uv[1]), "v out of range: {}", uv[1]);
        }
    }

    /// A per-face distance field storing an occluder at normalized distance
    /// `occluder` on every face.
    struct DistanceCube {
        occluder: f32,
    }

    impl ShadowDepthSampler for DistanceCube {
        fn sample_depth(&self, _face: usize, uv: [f32; 2]) -> f32 {
            if uv[0] < 0.0 || uv[0] > 1.0 || uv[1] < 0.0 || uv[1] > 1.0 {
                1.0
            } else {
                self.occluder
            }
        }
    }

    /// A fragment farther from the light than the stored occluder is shadowed;
    /// one nearer than the occluder is lit.
    #[test]
    fn distance_compare_shadows_far_fragments() {
        // Occluder at normalized distance 0.5 -> world distance 50 (range 100).
        let cube = DistanceCube { occluder: 0.5 };
        let cfg = PointShadowConfig {
            const_bias: 0.0,
            slope_bias: 0.0,
            max_bias: 0.01,
            pcf_radius: 1,
            texel_uv_size: [1.0 / 64.0, 1.0 / 64.0],
        };

        // Fragment at distance 80 along +X -> normalized 0.8 > 0.5 -> shadow.
        let far = PointShadowInput {
            world_position: [80.0, 0.0, 0.0],
            light_position: [0.0, 0.0, 0.0],
            range: 100.0,
            n_dot_l: 1.0,
        };
        assert!((evaluate_point_shadow(&cube, &far, &cfg) - 0.0).abs() < 1.0e-6);

        // Fragment at distance 20 along +X -> normalized 0.2 < 0.5 -> lit.
        let near = PointShadowInput {
            world_position: [20.0, 0.0, 0.0],
            ..far
        };
        assert!((evaluate_point_shadow(&cube, &near, &cfg) - 1.0).abs() < 1.0e-6);
    }

    /// A fragment beyond the light's range reads the far background (off-map)
    /// and stays lit.
    #[test]
    fn beyond_range_is_lit() {
        let cube = DistanceCube { occluder: 1.0 };
        let cfg = PointShadowConfig {
            const_bias: 0.0,
            slope_bias: 0.0,
            max_bias: 0.01,
            pcf_radius: 0,
            texel_uv_size: [1.0 / 64.0, 1.0 / 64.0],
        };
        let frag = PointShadowInput {
            world_position: [0.0, 200.0, 0.0],
            light_position: [0.0, 0.0, 0.0],
            range: 100.0,
            n_dot_l: 1.0,
        };
        // Normalized distance clamps to 1.0 and stored far is 1.0 -> lit (<=).
        assert!((evaluate_point_shadow(&cube, &frag, &cfg) - 1.0).abs() < 1.0e-6);
    }
}
