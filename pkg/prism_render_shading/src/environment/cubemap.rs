//! Backend-neutral cube-map -> spherical-harmonic projection.
//!
//! An environment probe stored as six cube-map faces is convolved into the
//! order-two SH radiance basis consumed by [`SphericalHarmonicsL2`]. The
//! projection mirrors AMD's `CubeMapGen` / Unreal's environment-capture baking:
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

    /// Bilinearly samples the linear radiance along a world-space `direction`.
    ///
    /// The direction's major axis selects the cube face; the remaining two
    /// components give the in-face `[-1, 1]` coordinate, and the four
    /// surrounding texels are blended with clamp-to-edge behaviour (per-face,
    /// matching a GPU cube sampler within a face).  A zero/degenerate direction
    /// falls back to `+Y`.
    pub fn sample(&self, direction: [f32; 3]) -> [f32; 3] {
        let (face, u, v) = direction_to_face_uv(direction);
        let size = self.size;
        if size == 0 {
            return [0.0; 3];
        }
        let max_index = (size - 1) as i32;
        // Map [-1, 1] to texel-centre space [0, size - 1].
        let fx = ((u * 0.5 + 0.5) * (size as f32) - 0.5).clamp(0.0, max_index as f32);
        let fy = ((v * 0.5 + 0.5) * (size as f32) - 0.5).clamp(0.0, max_index as f32);
        let x0 = fx.floor() as i32;
        let y0 = fy.floor() as i32;
        let x1 = (x0 + 1).min(max_index);
        let y1 = (y0 + 1).min(max_index);
        let tx = fx - (x0 as f32);
        let ty = fy - (y0 as f32);
        let texels = &self.faces[face];
        let at = |x: i32, y: i32| texels[(y as u32 * size + x as u32) as usize];
        let c00 = at(x0, y0);
        let c10 = at(x1, y0);
        let c01 = at(x0, y1);
        let c11 = at(x1, y1);
        let lerp = |a: [f32; 3], b: [f32; 3], t: f32| {
            [
                a[0] + (b[0] - a[0]) * t,
                a[1] + (b[1] - a[1]) * t,
                a[2] + (b[2] - a[2]) * t,
            ]
        };
        lerp(lerp(c00, c10, tx), lerp(c01, c11, tx), ty)
    }
}

/// Inverse of [`face_direction`]: maps a world-space `direction` to the cube
/// face index and in-face `[-1, 1]` coordinate that sampled it.
fn direction_to_face_uv(direction: [f32; 3]) -> (usize, f32, f32) {
    let [x, y, z] = direction;
    let ax = x.abs();
    let ay = y.abs();
    let az = z.abs();
    if ax >= ay && ax >= az && ax > 0.0 {
        if x > 0.0 {
            (0, -z / ax, -y / ax) // +X: [1, -v, -u]
        } else {
            (1, z / ax, -y / ax) // -X: [-1, -v, u]
        }
    } else if ay >= ax && ay >= az && ay > 0.0 {
        if y > 0.0 {
            (2, x / ay, z / ay) // +Y: [u, 1, v]
        } else {
            (3, x / ay, -z / ay) // -Y: [u, -1, -v]
        }
    } else if az > 0.0 {
        if z > 0.0 {
            (4, x / az, -y / az) // +Z: [u, -v, 1]
        } else {
            (5, -x / az, -y / az) // -Z: [-u, -v, -1]
        }
    } else {
        (2, 0.0, 0.0) // Degenerate: look up (+Y centre).
    }
}

/// The six cube faces in array-layer order.
pub(super) const FACE_COUNT: usize = 6;

/// Maps a face index plus in-face coordinates `(u, v)` in `[-1, 1]` to the
/// world-space direction sampled by that texel.  The mapping matches the wgpu
/// cube convention so a baked probe lines up with the GPU sampler.
pub(super) fn face_direction(face: usize, u: f32, v: f32) -> [f32; 3] {
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
/// projected cube face (the analytic primitive AMD `CubeMapGen` differences to
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
/// radiance cube collapses to [`SphericalHarmonicsL2::from_constant`], and the
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
            for (channel, &value) in radiance.iter().enumerate().take(3) {
                assert!(
                    (value - 1.0).abs() < 5.0e-2,
                    "dir {direction:?} channel {channel}: {value}"
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

    #[test]
    fn direction_to_face_uv_inverts_face_direction() {
        // For each face and a grid of interior (u, v), the direction produced
        // by `face_direction` must resolve back to the same face and (u, v).
        for face in 0..FACE_COUNT {
            for &u in &[-0.7f32, -0.2, 0.0, 0.3, 0.8] {
                for &v in &[-0.6f32, -0.1, 0.0, 0.4, 0.9] {
                    let dir = face_direction(face, u, v);
                    let (rf, ru, rv) = direction_to_face_uv(dir);
                    assert_eq!(rf, face, "face {face} u {u} v {v} -> {rf}");
                    assert!((ru - u).abs() < 1.0e-5, "u {u} -> {ru}");
                    assert!((rv - v).abs() < 1.0e-5, "v {v} -> {rv}");
                }
            }
        }
    }

    #[test]
    fn degenerate_direction_falls_back_to_plus_y() {
        let (face, u, v) = direction_to_face_uv([0.0, 0.0, 0.0]);
        assert_eq!(face, 2);
        assert_eq!((u, v), (0.0, 0.0));
    }

    #[test]
    fn sample_of_constant_cube_is_constant() {
        let cube = constant_faces(8, [0.25, 0.5, 0.75]);
        for dir in [
            [1.0, 0.2, -0.1],
            [-0.3, 1.0, 0.4],
            [0.1, -0.2, 1.0],
            [-1.0, 0.0, 0.0],
        ] {
            let c = cube.sample(dir);
            assert!((c[0] - 0.25).abs() < 1.0e-6);
            assert!((c[1] - 0.5).abs() < 1.0e-6);
            assert!((c[2] - 0.75).abs() < 1.0e-6);
        }
    }

    #[test]
    fn sample_selects_the_major_axis_face() {
        // Distinct flat colour per face; sampling straight down each axis must
        // return that face's colour.
        let colors = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
        ];
        let faces = core::array::from_fn(|i| vec![colors[i]; 4 * 4]);
        let cube = CubemapFaces::new(4, faces).unwrap();
        let axes = [
            ([1.0, 0.0, 0.0], 0),
            ([-1.0, 0.0, 0.0], 1),
            ([0.0, 1.0, 0.0], 2),
            ([0.0, -1.0, 0.0], 3),
            ([0.0, 0.0, 1.0], 4),
            ([0.0, 0.0, -1.0], 5),
        ];
        for (dir, face) in axes {
            assert_eq!(cube.sample(dir), colors[face], "dir {dir:?}");
        }
    }
}
