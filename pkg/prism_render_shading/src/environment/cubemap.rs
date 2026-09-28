//! Backend-neutral cube-map -> spherical-harmonic projection.
//!
//! An environment probe stored as six cube-map faces is convolved into the
//! order-two SH radiance basis consumed by [SphericalHarmonicsL2]. The
//! projection mirrors AMD's CubeMapGen / Unreal's environment-capture baking:
//! every texel is turned into a world-space direction, weighted by the exact
//! solid angle it subtends, and accumulated into the SH radiance vector.
//!
//! Keeping the math here (plain [f32; 3] faces, no engine types) lets the CPU
//! golden and any future GPU baker share one reference and lets the projection
//! be unit-tested without a device.

use bevy_math::ops;

use super::SphericalHarmonicsL2;

/// Six linear-RGB cube-map faces sharing a common edge `size`.
///
/// Faces follow the wgpu / D3D cube layout and array-layer order:
/// `+X, -X, +Y, -Y, +Z, -Z`.  Each face stores `size * size` texels in
/// row-major order (`index = y * size + x`) holding *linear* radiance; sRGB
/// or other encodings must be linearized before construction.
#[derive(Clone, Debug, PartialEq)]
pub struct CubemapFaces {
    /// Edge length in texels shared by every face.
    pub size: u32,
    /// Row-major linear-RGB texels for each of the six faces.
    pub faces: [Vec<[f32; 3]>; 6],
}

impl CubemapFaces {
    /// Builds faces from six equally sized row-major buffers.
    ///
    /// Returns `None` when `size` is zero or any face does not hold exactly
    /// `size * size` texels, so callers can fall back to a constant ambient
    /// term instead of projecting garbage.
    pub fn new(size: u32, faces: [Vec<[f32; 3]>; 6]) -> Option<Self> {
        if size == 0 {
            return None;
        }
        let expected = (size as usize) * (size as usize);
        if faces.iter().any(|face| face.len() != expected) {
            return None;
        }
        Some(Self { size, faces })
    }
}

/// The six cube faces in array-layer order.
const FACE_COUNT: usize = 6;

/// Maps a face index plus in-face coordinates `(u, v)` in `[-1, 1]` to the
/// world-space direction sampled by that texel.  The mapping matches the wgpu
/// cube convention so a baked probe lines up with the GPU sampler.
fn face_direction(face: usize, u: f32, v: f32) -> [f32; 3] {
    match face {
        0 => [1.0, -v, -u],  // +X
        1 => [-1.0, -v, u],  // -X
        2 => [u, 1.0, v],    // +Y
        3 => [u, -1.0, -v],  // -Y
        4 => [u, -v, 1.0],   // +Z
        _ => [-u, -v, -1.0], // -Z
    }
}

/// Signed area of the spherical rectangle spanning `[0, s] x [0, t]` on the
/// projected cube face (the analytic primitive AMD CubeMapGen differences to
/// recover per-texel solid angles).
fn area_element(s: f32, t: f32) -> f32 {
    ops::atan2(s * t, ops::sqrt(s * s + t * t + 1.0))
}

/// Exact solid angle subtended by the texel whose `[-1, 1]` extent is
/// `[u0, u1] x [v0, v1]`.
fn texel_solid_angle(u0: f32, u1: f32, v0: f32, v1: f32) -> f32 {
    area_element(u1, v1) - area_element(u0, v1) - area_element(u1, v0) + area_element(u0, v0)
}

/// Projects six linear-RGB cube-map faces into an order-two SH radiance probe.
///
/// Each texel contributes its radiance weighted by the exact solid angle it
/// covers, so the accumulated probe integrates the full sphere: a constant
/// radiance cube collapses to [SphericalHarmonicsL2::from_constant], and the
/// summed weights equal `4*pi`.
pub fn project_cubemap_to_sh(faces: &CubemapFaces) -> SphericalHarmonicsL2 {
    let mut probe = SphericalHarmonicsL2::ZERO;
    let size = faces.size;
    if size == 0 {
        return probe;
    }
    let inv_size = (size as f32).recip();
    for face in 0..FACE_COUNT {
        let texels = &faces.faces[face];
        for y in 0..size {
            // Texel edges in [-1, 1]; centres drive the direction, edges the
            // solid angle.
            let v0 = 2.0 * (y as f32) * inv_size - 1.0;
            let v1 = 2.0 * ((y + 1) as f32) * inv_size - 1.0;
            let v = 0.5 * (v0 + v1);
            for x in 0..size {
                let u0 = 2.0 * (x as f32) * inv_size - 1.0;
                let u1 = 2.0 * ((x + 1) as f32) * inv_size - 1.0;
                let u = 0.5 * (u0 + u1);
                let solid_angle = texel_solid_angle(u0, u1, v0, v1);
                let radiance = texels[(y * size + x) as usize];
                let direction = face_direction(face, u, v);
                probe.add_directional_radiance(direction, radiance, solid_angle);
            }
        }
    }
    probe
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f32::consts::PI;

    fn constant_faces(size: u32, color: [f32; 3]) -> CubemapFaces {
        let texels = vec![color; (size as usize) * (size as usize)];
        CubemapFaces::new(size, core::array::from_fn(|_| texels.clone())).expect("valid faces")
    }

    fn total_solid_angle(size: u32) -> f32 {
        let inv_size = (size as f32).recip();
        let mut total = 0.0;
        for _ in 0..FACE_COUNT {
            for y in 0..size {
                let v0 = 2.0 * (y as f32) * inv_size - 1.0;
                let v1 = 2.0 * ((y + 1) as f32) * inv_size - 1.0;
                for x in 0..size {
                    let u0 = 2.0 * (x as f32) * inv_size - 1.0;
                    let u1 = 2.0 * ((x + 1) as f32) * inv_size - 1.0;
                    total += texel_solid_angle(u0, u1, v0, v1);
                }
            }
        }
        total
    }

    #[test]
    fn texel_solid_angles_cover_the_full_sphere() {
        // Summed exact solid angles over the cube must equal 4*pi.
        assert!((total_solid_angle(16) - 4.0 * PI).abs() < 1.0e-3);
    }

    #[test]
    fn constant_cube_matches_from_constant() {
        let color = [0.3, 0.6, 0.9];
        let probe = project_cubemap_to_sh(&constant_faces(32, color));
        let reference = SphericalHarmonicsL2::from_constant(color);
        // Band 0 carries the DC radiance; higher bands cancel for a constant.
        for channel in 0..3 {
            assert!(
                (probe.coefficients[0][channel] - reference.coefficients[0][channel]).abs() < 1.0e-2,
                "band0 channel {channel}: {} vs {}",
                probe.coefficients[0][channel],
                reference.coefficients[0][channel]
            );
        }
        for band in 1..9 {
            for channel in 0..3 {
                assert!(
                    probe.coefficients[band][channel].abs() < 1.0e-2,
                    "band {band} channel {channel} should vanish: {}",
                    probe.coefficients[band][channel]
                );
            }
        }
    }

    #[test]
    fn constant_cube_reconstructs_uniform_radiance() {
        let probe = project_cubemap_to_sh(&constant_faces(24, [1.0, 1.0, 1.0]));
        for direction in [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
        ] {
            let radiance = probe.radiance(direction);
            for channel in 0..3 {
                assert!(
                    (radiance[channel] - 1.0).abs() < 5.0e-2,
                    "dir {direction:?} channel {channel}: {}",
                    radiance[channel]
                );
            }
        }
    }

    #[test]
    fn bright_top_face_biases_irradiance_upward() {
        // +Y bright, everything else dark: irradiance should peak looking up.
        let size = 16u32;
        let bright = vec![[1.0; 3]; (size as usize) * (size as usize)];
        let dark = vec![[0.0; 3]; (size as usize) * (size as usize)];
        let faces = CubemapFaces::new(
            size,
            [
                dark.clone(),
                dark.clone(),
                bright, // +Y
                dark.clone(),
                dark.clone(),
                dark,
            ],
        )
        .expect("valid faces");
        let probe = project_cubemap_to_sh(&faces);
        let up = probe.irradiance([0.0, 1.0, 0.0]);
        let down = probe.irradiance([0.0, -1.0, 0.0]);
        assert!(up[0] > down[0], "up {up:?} vs down {down:?}");
        assert!(up.iter().all(|c| c.is_finite() && *c >= 0.0));
    }

    #[test]
    fn rejects_mismatched_face_lengths() {
        let good = vec![[0.0; 3]; 4];
        let bad = vec![[0.0; 3]; 3];
        let faces = [
            good.clone(),
            good.clone(),
            good.clone(),
            good.clone(),
            good,
            bad,
        ];
        assert!(CubemapFaces::new(2, faces).is_none());
        assert!(CubemapFaces::new(0, core::array::from_fn(|_| Vec::new())).is_none());
    }
}
