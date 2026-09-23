//! GPU-side light ABI.
//!
//! These `#[repr(C)]` records are the byte-for-byte contract the resolve
//! compute shader consumes.  They are deliberately flat, `Pod`, and padded to a
//! 16-byte grid so the same layout is valid whether it is bound as a `std430`
//! storage buffer today or promoted to a `std140` uniform later.
//!
//! The CPU golden reference in [`prism_render_shading`] owns the lighting math;
//! this module only mirrors its light *inputs* onto the GPU.  Conversions from
//! the reference [`DirectionalLight`]/[`PunctualLight`] types are provided (and
//! round-trip tested) so extraction can reuse the reference's well-tested cone
//! and attenuation encoding rather than re-deriving it here.

use bytemuck::{Pod, Zeroable};
use prism_render_shading::{DirectionalLight, PunctualLight, SphericalHarmonicsL2};

/// Flag bit: the environment carries a valid image-based (SH) probe and the
/// resolve pass must use it for the indirect term instead of the constant
/// ambient approximation.
pub const LIGHT_ENVIRONMENT_FLAG_IMAGE_BASED: u32 = 1 << 0;

/// A directional (infinitely distant) light in world space.
///
/// `direction_to_light` points **from the surface toward the emitter**, exactly
/// like [`DirectionalLight::direction`], so the shader can dot it with the
/// surface normal without a sign flip.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct GpuDirectionalLight {
    /// Unit vector pointing from the surface toward the light.
    pub direction_to_light: [f32; 3],
    /// Analytic shadow visibility term in `[0, 1]`.
    pub visibility: f32,
    /// Linear, pre-exposed illuminance contributed at the surface.
    pub illuminance: [f32; 3],
    /// Padding to a 16-byte boundary; always zero.
    pub _padding: f32,
}

impl Default for GpuDirectionalLight {
    fn default() -> Self {
        Self {
            direction_to_light: [0.0, 1.0, 0.0],
            visibility: 1.0,
            illuminance: [0.0; 3],
            _padding: 0.0,
        }
    }
}

impl From<DirectionalLight> for GpuDirectionalLight {
    fn from(light: DirectionalLight) -> Self {
        Self {
            direction_to_light: light.direction,
            visibility: light.visibility,
            illuminance: light.illuminance,
            _padding: 0.0,
        }
    }
}

impl From<GpuDirectionalLight> for DirectionalLight {
    fn from(light: GpuDirectionalLight) -> Self {
        Self {
            direction: light.direction_to_light,
            illuminance: light.illuminance,
            visibility: light.visibility,
        }
    }
}

/// A point or spot light in world space.
///
/// The spot cone is encoded with the same precomputed `scale`/`offset` pair as
/// [`PunctualLight`]; a `spot_scale` of `0` marks an omnidirectional point
/// light whose angular term evaluates to `1` everywhere.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct GpuPunctualLight {
    /// World-space position of the emitter.
    pub position: [f32; 3],
    /// Influence radius in world units; `0` disables the range window.
    pub range: f32,
    /// Linear, pre-exposed radiant intensity (candela) radiated toward the
    /// surface.
    pub intensity: [f32; 3],
    /// Precomputed `1 / (cos(inner) - cos(outer))`; `0` marks a point light.
    pub spot_scale: f32,
    /// Unit cone axis pointing from the light into the scene.
    pub direction: [f32; 3],
    /// Precomputed `-cos(outer) * spot_scale`.
    pub spot_offset: f32,
    /// Analytic shadow visibility term in `[0, 1]`.
    pub visibility: f32,
    /// Padding to a 16-byte boundary; always zero.
    pub _padding: [f32; 3],
}

impl Default for GpuPunctualLight {
    fn default() -> Self {
        Self {
            position: [0.0; 3],
            range: 0.0,
            intensity: [0.0; 3],
            spot_scale: 0.0,
            direction: [0.0, 0.0, -1.0],
            spot_offset: 1.0,
            visibility: 1.0,
            _padding: [0.0; 3],
        }
    }
}

impl From<PunctualLight> for GpuPunctualLight {
    fn from(light: PunctualLight) -> Self {
        Self {
            position: light.position,
            range: light.range,
            intensity: light.intensity,
            spot_scale: light.spot_scale,
            direction: light.direction,
            spot_offset: light.spot_offset,
            visibility: light.visibility,
            _padding: [0.0; 3],
        }
    }
}

impl From<GpuPunctualLight> for PunctualLight {
    fn from(light: GpuPunctualLight) -> Self {
        Self {
            position: light.position,
            intensity: light.intensity,
            range: light.range,
            direction: light.direction,
            spot_scale: light.spot_scale,
            spot_offset: light.spot_offset,
            visibility: light.visibility,
        }
    }
}

/// Frame-constant lighting environment shared by every shaded pixel.
///
/// The nine `sh` rows store one L2 spherical-harmonic radiance coefficient each
/// (`xyz` used, `w` padding), matching [`SphericalHarmonicsL2`].  `ambient` is
/// the constant irradiance fallback used when [`LIGHT_ENVIRONMENT_FLAG_IMAGE_BASED`]
/// is clear.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct GpuLightEnvironment {
    /// Constant ambient irradiance approximating unresolved indirect light.
    pub ambient: [f32; 3],
    /// Number of valid entries in the directional light buffer.
    pub directional_count: u32,
    /// L2 SH radiance probe; `xyz` per row, `w` padding.
    pub sh: [[f32; 4]; 9],
    /// Number of valid entries in the punctual light buffer.
    pub punctual_count: u32,
    /// Quantization band count used by the non-photoreal toon path.
    pub toon_bands: u32,
    /// Reserved exposure scale (`1.0` = neutral) for a future exposure stage.
    pub exposure: f32,
    /// Bit flags; see `LIGHT_ENVIRONMENT_FLAG_*`.
    pub flags: u32,
}

impl Default for GpuLightEnvironment {
    fn default() -> Self {
        Self {
            ambient: [0.0; 3],
            directional_count: 0,
            sh: [[0.0; 4]; 9],
            punctual_count: 0,
            toon_bands: 4,
            exposure: 1.0,
            flags: 0,
        }
    }
}

impl GpuLightEnvironment {
    /// Writes the nine L2 radiance coefficients from a CPU probe into the
    /// padded `vec4` rows and marks the environment as image-based.
    pub fn set_spherical_harmonics(&mut self, probe: &SphericalHarmonicsL2) {
        for (row, coefficient) in self.sh.iter_mut().zip(probe.coefficients) {
            *row = [coefficient[0], coefficient[1], coefficient[2], 0.0];
        }
        self.flags |= LIGHT_ENVIRONMENT_FLAG_IMAGE_BASED;
    }

    /// Reconstructs the CPU probe from the padded rows.
    pub fn spherical_harmonics(&self) -> SphericalHarmonicsL2 {
        let mut probe = SphericalHarmonicsL2::ZERO;
        for (coefficient, row) in probe.coefficients.iter_mut().zip(self.sh) {
            *coefficient = [row[0], row[1], row[2]];
        }
        probe
    }

    /// Whether the resolve pass should use the SH probe for indirect light.
    pub fn has_image_based(&self) -> bool {
        self.flags & LIGHT_ENVIRONMENT_FLAG_IMAGE_BASED != 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abi_sizes_and_alignments_are_std430_safe() {
        assert_eq!(size_of::<GpuDirectionalLight>(), 32);
        assert_eq!(size_of::<GpuPunctualLight>(), 64);
        assert_eq!(size_of::<GpuLightEnvironment>(), 176);
        assert_eq!(align_of::<GpuDirectionalLight>(), 4);
        assert_eq!(align_of::<GpuPunctualLight>(), 4);
        assert_eq!(align_of::<GpuLightEnvironment>(), 4);
        // Every record is a whole number of 16-byte GPU rows.
        assert_eq!(size_of::<GpuDirectionalLight>() % 16, 0);
        assert_eq!(size_of::<GpuPunctualLight>() % 16, 0);
        assert_eq!(size_of::<GpuLightEnvironment>() % 16, 0);
    }

    #[test]
    fn directional_round_trips_through_the_reference_type() {
        let cpu = DirectionalLight {
            direction: [0.0, 0.707, 0.707],
            illuminance: [3.0, 4.0, 5.0],
            visibility: 0.5,
        };
        let gpu = GpuDirectionalLight::from(cpu);
        assert_eq!(gpu.direction_to_light, cpu.direction);
        assert_eq!(gpu.illuminance, cpu.illuminance);
        assert_eq!(gpu.visibility, cpu.visibility);
        assert_eq!(gpu._padding, 0.0);
        assert_eq!(DirectionalLight::from(gpu), cpu);
    }

    #[test]
    fn punctual_round_trips_through_the_reference_type() {
        let cpu = PunctualLight::spot(
            [1.0, 2.0, 3.0],
            [10.0; 3],
            25.0,
            [0.0, 0.0, -1.0],
            0.9,
            0.5,
        )
        .with_visibility(0.75);
        let gpu = GpuPunctualLight::from(cpu);
        assert_eq!(gpu.position, cpu.position);
        assert_eq!(gpu.range, cpu.range);
        assert_eq!(gpu.intensity, cpu.intensity);
        assert_eq!(gpu.spot_scale, cpu.spot_scale);
        assert_eq!(gpu.spot_offset, cpu.spot_offset);
        assert_eq!(gpu.direction, cpu.direction);
        assert_eq!(gpu.visibility, cpu.visibility);
        assert_eq!(gpu._padding, [0.0; 3]);
        assert_eq!(PunctualLight::from(gpu), cpu);
    }

    #[test]
    fn environment_carries_counts_and_probe_round_trip() {
        let probe = SphericalHarmonicsL2::from_constant([0.5, 0.25, 0.125]);
        let mut environment = GpuLightEnvironment {
            ambient: [0.1, 0.2, 0.3],
            directional_count: 2,
            punctual_count: 7,
            toon_bands: 5,
            ..Default::default()
        };
        assert!(!environment.has_image_based());
        environment.set_spherical_harmonics(&probe);
        assert!(environment.has_image_based());
        assert_eq!(environment.spherical_harmonics(), probe);
        assert_eq!(environment.directional_count, 2);
        assert_eq!(environment.punctual_count, 7);
        assert_eq!(environment.toon_bands, 5);
    }

    #[test]
    fn defaults_are_neutral() {
        let directional = GpuDirectionalLight::default();
        assert_eq!(directional.illuminance, [0.0; 3]);
        assert_eq!(directional.visibility, 1.0);
        let punctual = GpuPunctualLight::default();
        assert_eq!(punctual.spot_scale, 0.0);
        assert_eq!(punctual.intensity, [0.0; 3]);
        let environment = GpuLightEnvironment::default();
        assert_eq!(environment.exposure, 1.0);
        assert_eq!(environment.toon_bands, 4);
        assert_eq!(environment.flags, 0);
    }
}
