//! Stable Custom response evaluator and execution-identity adapter.
//!
//! Custom programs share the same surface and lighting narrow waist as the
//! built-in responses.  The execution identity is data, not a permutation
//! axis: the program id and permutation select one deterministic evaluator
//! variant while the material instance remains ordinary surface data.

use crate::response_output::{CompositionError, LightingContext, ResponseOutput};
use crate::SurfaceSample;

/// View-side inputs exposed to a Custom response evaluator.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CustomViewContext {
    /// Unit or approximately unit shading normal in world space.
    pub normal: [f32; 3],
    /// View direction from the surface toward the camera.
    pub view: [f32; 3],
    /// Shared surface alpha computed by the raster/material stage.
    pub alpha: f32,
}

/// Cold-path execution data for a registered response program.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CustomData {
    /// Stable registered program identity. `0` remains a valid identity.
    pub program_id: u32,
    /// Checked permutation coordinate belonging to the registered package.
    pub permutation_index: u64,
    /// Per-program radiance tint selected from the execution identity.
    pub tint: [f32; 3],
    /// Per-program radiance gain selected from the execution identity.
    pub gain: f32,
    /// Multiplier for the shared alpha channel.
    pub alpha_scale: f32,
}

impl CustomData {
    /// Derives the deterministic ABI payload used by the CPU and GPU twins.
    #[must_use]
    pub fn from_execution(program_id: u32, permutation_index: u64) -> Self {
        let seed = program_id
            .wrapping_mul(1_664_525)
            .wrapping_add(permutation_index as u32)
            .wrapping_add(((permutation_index >> 32) as u32).wrapping_mul(1_013_904_223));
        let unit = |shift: u32| {
            let mixed = seed.wrapping_add(shift.wrapping_mul(2_654_435_761));
            (mixed & 0xff) as f32 / 255.0
        };
        Self {
            program_id,
            permutation_index,
            tint: [
                0.85 + unit(3) * 0.30,
                0.85 + unit(11) * 0.30,
                0.85 + unit(19) * 0.30,
            ],
            gain: 0.75 + unit(27) * 0.50,
            alpha_scale: 0.75 + unit(7) * 0.25,
        }
    }

    fn validate(self) -> Result<(), CustomResponseError> {
        if !self.tint.iter().all(|value| value.is_finite())
            || !self.gain.is_finite()
            || !self.alpha_scale.is_finite()
            || self.gain < 0.0
            || self.alpha_scale < 0.0
        {
            return Err(CustomResponseError::InvalidData);
        }
        Ok(())
    }
}

/// Errors raised by the stable Custom response ABI.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CustomResponseError {
    /// Shared lighting inputs failed the response composition validation.
    Composition(CompositionError),
    /// The view context contains a non-finite or degenerate value.
    InvalidView,
    /// The execution-derived Custom payload is invalid.
    InvalidData,
}

impl From<CompositionError> for CustomResponseError {
    fn from(error: CompositionError) -> Self {
        Self::Composition(error)
    }
}

/// Evaluates one registered Custom response into the shared structured output.
///
/// The function intentionally owns only response interpretation.  Geometry,
/// texture sampling, visibility and shadow sampling remain upstream shared
/// inputs.  This keeps the Custom path on the same ABI as Lit/NPR and makes
/// the evaluator independently testable on CPU.
pub fn respond_custom<T: ?Sized>(
    surface: &SurfaceSample,
    lighting: &LightingContext<'_, T>,
    view: &CustomViewContext,
    custom_data: &CustomData,
) -> Result<ResponseOutput, CustomResponseError> {
    lighting.validate()?;
    custom_data.validate()?;
    if !view.normal.iter().all(|value| value.is_finite())
        || !view.view.iter().all(|value| value.is_finite())
        || !view.alpha.is_finite()
        || !(0.0..=1.0).contains(&view.alpha)
    {
        return Err(CustomResponseError::InvalidView);
    }

    let normal = normalize_or(view.normal, [0.0, 1.0, 0.0]);
    let base = mul3(surface.base_color, [surface.base_weight; 3]);
    let mut direct = [0.0; 3];
    for light in lighting.direct {
        let cosine = dot3(normal, normalize_or(light.direction, [0.0, 1.0, 0.0])).max(0.0);
        let scale = cosine * light.visibility * custom_data.gain;
        direct = add3(direct, mul3(mul3(base, light.illuminance), [scale; 3]));
    }
    let indirect_diffuse = mul3(
        mul3(
            base,
            lighting.indirect_diffuse_irradiance.unwrap_or([0.0; 3]),
        ),
        [custom_data.gain; 3],
    );
    let reflection = mul3(
        lighting.reflection_radiance.unwrap_or([0.0; 3]),
        [surface.specular_weight * custom_data.gain; 3],
    );
    let transmission = mul3(
        lighting.transmission_radiance.unwrap_or([0.0; 3]),
        [surface.transmission * custom_data.gain; 3],
    );
    let emission = mul3(surface.emissive, [custom_data.gain; 3]);
    let output = ResponseOutput {
        direct: mul3(direct, custom_data.tint),
        indirect_diffuse: mul3(indirect_diffuse, custom_data.tint),
        reflection: mul3(reflection, custom_data.tint),
        transmission: mul3(transmission, custom_data.tint),
        emission: mul3(emission, custom_data.tint),
        alpha: (view.alpha * surface.geometry_opacity * custom_data.alpha_scale).clamp(0.0, 1.0),
        ..ResponseOutput::default()
    };
    output.validate()?;
    Ok(output)
}

/// Applies the same execution-derived variant to an already-integrated
/// aggregate.  The resolve CPU golden uses this helper for the registered
/// response class; the WESL twin derives the identical `CustomData` words.
#[must_use]
pub fn apply_custom_variant(
    aggregate: [f32; 3],
    program_id: u32,
    permutation_index: u64,
) -> [f32; 3] {
    let data = CustomData::from_execution(program_id, permutation_index);
    mul3(mul3(aggregate, [data.gain; 3]), data.tint)
}

fn add3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn mul3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] * b[0], a[1] * b[1], a[2] * b[2]]
}

fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn normalize_or(value: [f32; 3], fallback: [f32; 3]) -> [f32; 3] {
    let length_squared = dot3(value, value);
    if length_squared > 1.0e-12 && length_squared.is_finite() {
        let inverse = length_squared.sqrt().recip();
        mul3(value, [inverse; 3])
    } else {
        fallback
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn execution_identity_produces_stable_non_identity_variants() {
        let first = CustomData::from_execution(7, 3);
        let second = CustomData::from_execution(7, 3);
        assert_eq!(first, second);
        assert_ne!(first.tint, [1.0; 3]);
        assert_ne!(first.gain, 1.0);
        assert_ne!(first.alpha_scale, 1.0);
        assert_ne!(first, CustomData::from_execution(8, 3));
    }

    #[test]
    fn aggregate_variant_is_finite_and_program_specific() {
        let base = [1.0, 2.0, 3.0];
        let left = apply_custom_variant(base, 1, 0);
        let right = apply_custom_variant(base, 2, 0);
        assert!(left.iter().all(|value| value.is_finite()));
        assert_ne!(left, right);
    }

    #[test]
    fn invalid_custom_payload_is_rejected() {
        let mut data = CustomData::from_execution(0, 0);
        data.gain = f32::NAN;
        assert_eq!(data.validate(), Err(CustomResponseError::InvalidData));
    }

    #[test]
    fn cpu_variant_words_match_the_wesl_identity_formula() {
        for (program_id, permutation_index) in [(0, 0), (7, 3), (u32::MAX, u64::MAX)] {
            let data = CustomData::from_execution(program_id, permutation_index);
            let seed = program_id
                .wrapping_mul(1_664_525)
                .wrapping_add(permutation_index as u32)
                .wrapping_add(((permutation_index >> 32) as u32).wrapping_mul(1_013_904_223));
            let wesl_unit = |salt: u32| {
                let mixed = seed.wrapping_add(salt.wrapping_mul(2_654_435_761));
                (mixed & 255) as f32 / 255.0
            };
            let expected_tint = [
                0.85 + wesl_unit(3) * 0.30,
                0.85 + wesl_unit(11) * 0.30,
                0.85 + wesl_unit(19) * 0.30,
            ];
            let expected_gain = 0.75 + wesl_unit(27) * 0.50;
            let expected_alpha = 0.75 + wesl_unit(7) * 0.25;
            assert_eq!(data.tint, expected_tint);
            assert_eq!(data.gain, expected_gain);
            assert_eq!(data.alpha_scale, expected_alpha);
        }
    }
}
