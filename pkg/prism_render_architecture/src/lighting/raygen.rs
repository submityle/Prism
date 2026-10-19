//! Deterministic cosine-weighted GI ray generation for independent surfaces.
//!
//! Each ray's random dimensions are derived from the surface stable ID,
//! frame/sample indices, and ray index. Generation is therefore independent
//! of surface ordering, worker scheduling, and other surfaces' ray counts.

use alloc::vec::Vec;
use core::f32::consts::PI;

use glam::Vec3;

/// Inputs for generating one surface's diffuse GI rays.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfaceRayGenInput {
    /// World-space surface position.
    pub position: Vec3,
    /// Geometric or shading normal; normalized internally.
    pub normal: Vec3,
    /// Stable identity of this surface, independent of buffer ordering.
    pub surface_stable_id: u64,
    /// Frame identity used to decorrelate successive frames.
    pub frame_index: u64,
    /// Sample sequence index within the frame.
    pub sample_index: u64,
    /// Number of rays to generate; must be non-zero.
    pub ray_count: u32,
    /// Positive origin offset along the normalized surface normal.
    pub epsilon: f32,
    /// Positive maximum ray distance; must exceed `epsilon`.
    pub tmax: f32,
}

/// A normalized cosine-weighted hemisphere ray and its solid-angle PDF.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GiRay {
    /// Offset world-space origin.
    pub origin: Vec3,
    /// Unit direction in the surface's positive-normal hemisphere.
    pub direction: Vec3,
    /// Minimum trace distance. The origin is already offset, so this is zero.
    pub tmin: f32,
    /// Maximum trace distance from `origin`.
    pub tmax: f32,
    /// Cosine-weighted PDF in inverse steradians: `max(dot(n,d),0) / PI`.
    pub pdf: f32,
}

/// Invalid surface ray-generation input.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RayGenError {
    /// Position contains NaN or infinity.
    NonFinitePosition,
    /// Normal contains NaN or infinity.
    NonFiniteNormal,
    /// Normal has zero or numerically unusable length.
    DegenerateNormal,
    /// Applying the normal offset overflowed the world-space origin.
    OriginOverflow,
    /// A ray batch must contain at least one ray.
    ZeroRayCount,
    /// Epsilon must be finite and strictly positive.
    InvalidEpsilon,
    /// `tmax` must be finite and greater than epsilon.
    InvalidTmax,
    /// The ray vector could not reserve the requested capacity.
    AllocationFailed,
}

/// Generates reproducible cosine-weighted GI rays for one surface.
///
/// Directions use a concentric-independent cosine map (`r=sqrt(u)`,
/// `phi=2*pi*v`) in a robust orthonormal frame. The hash-to-float conversion
/// uses the upper 24 bits, guaranteeing samples in `[0,1)` and avoiding the
/// singular endpoint. Results are bitwise repeatable for identical inputs on
/// the same target/FP implementation.
pub fn generate_surface_gi_rays(input: SurfaceRayGenInput) -> Result<Vec<GiRay>, RayGenError> {
    if !input.position.is_finite() {
        return Err(RayGenError::NonFinitePosition);
    }
    if !input.normal.is_finite() {
        return Err(RayGenError::NonFiniteNormal);
    }
    if !input.epsilon.is_finite() || input.epsilon <= 0.0 {
        return Err(RayGenError::InvalidEpsilon);
    }
    if !input.tmax.is_finite() || input.tmax <= input.epsilon {
        return Err(RayGenError::InvalidTmax);
    }
    if input.ray_count == 0 {
        return Err(RayGenError::ZeroRayCount);
    }

    let normal_len_sq = input.normal.length_squared();
    if !normal_len_sq.is_finite() || normal_len_sq <= 1.0e-20 {
        return Err(RayGenError::DegenerateNormal);
    }
    let normal = input.normal / normal_len_sq.sqrt();
    let origin = input.position + normal * input.epsilon;
    if !origin.is_finite() {
        return Err(RayGenError::OriginOverflow);
    }
    let (tangent, bitangent) = orthonormal_basis(normal);

    let mut rays = Vec::new();
    rays.try_reserve_exact(input.ray_count as usize)
        .map_err(|_| RayGenError::AllocationFailed)?;

    for ray_index in 0..input.ray_count {
        let seed = mix64(input.surface_stable_id)
            ^ mix64(input.frame_index.wrapping_add(0x9e37_79b9_7f4a_7c15))
            ^ mix64(input.sample_index.wrapping_add(0xd1b5_4a32_d192_ed03))
            ^ mix64(u64::from(ray_index).wrapping_add(0x94d0_49bb_1331_11eb));
        let u = unit_float(mix64(seed ^ 0xa076_1d64_78bd_642f));
        let v = unit_float(mix64(seed ^ 0xe703_7ed1_a0b4_28db));

        let radius = u.sqrt();
        let phi = (2.0 * PI) * v;
        let local_x = radius * phi.cos();
        let local_y = radius * phi.sin();
        let local_z = (1.0 - u).max(0.0).sqrt();
        let direction = (tangent * local_x + bitangent * local_y + normal * local_z).normalize();
        let pdf = direction.dot(normal).max(0.0) / PI;

        rays.push(GiRay {
            origin,
            direction,
            tmin: 0.0,
            tmax: input.tmax,
            pdf,
        });
    }
    Ok(rays)
}

fn orthonormal_basis(normal: Vec3) -> (Vec3, Vec3) {
    // Frisvad's branch-free construction is less robust near -Z; choosing the
    // least-aligned cardinal axis keeps the cross product well-conditioned.
    let abs = normal.abs();
    let helper = if abs.x <= abs.y && abs.x <= abs.z {
        Vec3::X
    } else if abs.y <= abs.z {
        Vec3::Y
    } else {
        Vec3::Z
    };
    let tangent = helper.cross(normal).normalize();
    let bitangent = normal.cross(tangent).normalize();
    (tangent, bitangent)
}

fn unit_float(bits: u64) -> f32 {
    ((bits >> 40) as u32) as f32 * (1.0 / 16_777_216.0)
}

fn mix64(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input() -> SurfaceRayGenInput {
        SurfaceRayGenInput {
            position: Vec3::new(1.0, -2.0, 3.5),
            normal: Vec3::new(0.2, 0.7, -0.4),
            surface_stable_id: 0x1234_5678_9abc_def0,
            frame_index: 41,
            sample_index: 7,
            ray_count: 64,
            epsilon: 0.001,
            tmax: 1000.0,
        }
    }

    #[test]
    fn different_surfaces_have_independent_directions() {
        let a = generate_surface_gi_rays(input()).unwrap();
        let mut other = input();
        other.surface_stable_id += 1;
        let b = generate_surface_gi_rays(other).unwrap();
        assert!(a.iter().zip(b).any(|(a, b)| a.direction != b.direction));
    }

    #[test]
    fn directions_are_unit_hemisphere_samples_and_normals_are_normalized() {
        let mut value = input();
        value.normal *= 9.0;
        let rays = generate_surface_gi_rays(value).unwrap();
        let n = value.normal.normalize();
        for ray in rays {
            assert!((ray.direction.length() - 1.0).abs() < 2.0e-6);
            assert!(ray.direction.dot(n) >= 0.0);
            assert_eq!(ray.origin, value.position + n * value.epsilon);
        }
    }

    #[test]
    fn pdf_is_cosine_weighted_and_rays_preserve_bounds() {
        let value = input();
        for ray in generate_surface_gi_rays(value).unwrap() {
            let expected = ray.direction.dot(value.normal.normalize()).max(0.0) / PI;
            assert!((ray.pdf - expected).abs() <= f32::EPSILON);
            assert!((0.0..=1.0 / PI).contains(&ray.pdf));
            assert_eq!(ray.tmin, 0.0);
            assert_eq!(ray.tmax, value.tmax);
        }
    }

    #[test]
    fn generation_is_bitwise_deterministic() {
        let a = generate_surface_gi_rays(input()).unwrap();
        let b = generate_surface_gi_rays(input()).unwrap();
        assert_eq!(a.len(), b.len());
        for (a, b) in a.iter().zip(b) {
            assert_eq!(
                a.origin.to_array().map(f32::to_bits),
                b.origin.to_array().map(f32::to_bits)
            );
            assert_eq!(
                a.direction.to_array().map(f32::to_bits),
                b.direction.to_array().map(f32::to_bits)
            );
            assert_eq!(a.pdf.to_bits(), b.pdf.to_bits());
        }
    }

    #[test]
    fn rejects_invalid_inputs_and_accepts_smallest_valid_batch() {
        let mut value = input();
        value.position.x = f32::NAN;
        assert_eq!(
            generate_surface_gi_rays(value),
            Err(RayGenError::NonFinitePosition)
        );
        value = input();
        value.normal.z = f32::INFINITY;
        assert_eq!(
            generate_surface_gi_rays(value),
            Err(RayGenError::NonFiniteNormal)
        );
        value = input();
        value.normal = Vec3::ZERO;
        assert_eq!(
            generate_surface_gi_rays(value),
            Err(RayGenError::DegenerateNormal)
        );
        value = input();
        value.normal = Vec3::splat(1.0e30);
        assert_eq!(
            generate_surface_gi_rays(value),
            Err(RayGenError::DegenerateNormal)
        );
        value = input();
        value.position.x = f32::MAX;
        value.normal = Vec3::X;
        value.epsilon = f32::MAX * 0.5;
        value.tmax = f32::MAX;
        assert_eq!(
            generate_surface_gi_rays(value),
            Err(RayGenError::OriginOverflow)
        );
        value = input();
        value.ray_count = 0;
        assert_eq!(
            generate_surface_gi_rays(value),
            Err(RayGenError::ZeroRayCount)
        );
        value = input();
        value.epsilon = 0.0;
        assert_eq!(
            generate_surface_gi_rays(value),
            Err(RayGenError::InvalidEpsilon)
        );
        value = input();
        value.epsilon = f32::NAN;
        assert_eq!(
            generate_surface_gi_rays(value),
            Err(RayGenError::InvalidEpsilon)
        );
        value = input();
        value.tmax = value.epsilon;
        assert_eq!(
            generate_surface_gi_rays(value),
            Err(RayGenError::InvalidTmax)
        );
        value = input();
        value.tmax = f32::INFINITY;
        assert_eq!(
            generate_surface_gi_rays(value),
            Err(RayGenError::InvalidTmax)
        );
        value = input();
        value.ray_count = 1;
        assert_eq!(generate_surface_gi_rays(value).unwrap().len(), 1);
    }
}
