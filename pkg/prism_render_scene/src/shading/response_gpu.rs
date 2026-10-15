//! Device-independent response ABI and production composition kernels.
//!
//! Split arms share lighting but retain independent executable packages. The
//! compositor overwrites the left result, requiring two rather than three
//! 64-byte radiance planes. Auxiliary semantics occupy separately typed jobs.

use bevy_render::{
    render_resource::{
        ComputePipeline, PipelineCompilationOptions, RawComputePipelineDescriptor,
        ShaderModuleDescriptor, ShaderSource,
    },
    renderer::RenderDevice,
};
use bytemuck::{Pod, Zeroable};
use prism_render_material::{BlendDomain, ChannelMask};
use prism_render_shading::response_output::{
    validate_response_composition_inputs, AlphaMergeRule, Auxiliary, AuxiliaryMergeRule,
    AuxiliaryValue, CompositionError, CompositionOptions, ResponseOutput,
};

pub const RESPONSE_COMPOSITION_SHADER: &str = include_str!("../shaders/response_composition.wgsl");
pub const RESPONSE_COMPOSITION_WORKGROUP_SIZE: u32 = 64;

/// Device-owned pipelines reused by resident composition passes. Radiance uses
/// bindings 0..3; auxiliary uses binding 4 in its independently derived layout.
pub struct ResponseCompositionPipelines {
    pub radiance: ComputePipeline,
    pub auxiliary: ComputePipeline,
}

impl ResponseCompositionPipelines {
    pub fn new(device: &RenderDevice) -> Self {
        let module = device
            .wgpu_device()
            .create_shader_module(ShaderModuleDescriptor {
                label: Some("prism response composition"),
                source: ShaderSource::Wgsl(RESPONSE_COMPOSITION_SHADER.into()),
            });
        let pipeline = |entry| {
            device.create_compute_pipeline(&RawComputePipelineDescriptor {
                label: Some(entry),
                layout: None,
                module: &module,
                entry_point: Some(entry),
                compilation_options: PipelineCompilationOptions::default(),
                cache: None,
            })
        };
        Self {
            radiance: pipeline("compose_responses"),
            auxiliary: pipeline("compose_auxiliary"),
        }
    }
}

/// Storage stride is 64 bytes; channels are consecutive RGB, followed by alpha.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct GpuResponseOutput {
    pub words: [f32; 16],
}

impl GpuResponseOutput {
    fn pack(value: &ResponseOutput) -> Self {
        let mut words = [0.0; 16];
        for (index, channel) in [
            value.direct,
            value.indirect_diffuse,
            value.reflection,
            value.transmission,
            value.emission,
        ]
        .into_iter()
        .enumerate()
        {
            words[3 * index..3 * index + 3].copy_from_slice(&channel);
        }
        words[15] = value.alpha;
        Self { words }
    }

    /// Restores the typed auxiliary plane supplied by the batch's decoder.
    pub fn unpack(self, auxiliary: Auxiliary) -> ResponseOutput {
        let channel = |i| [self.words[i], self.words[i + 1], self.words[i + 2]];
        ResponseOutput {
            direct: channel(0),
            indirect_diffuse: channel(3),
            reflection: channel(6),
            transmission: channel(9),
            emission: channel(12),
            alpha: self.words[15],
            auxiliary,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct GpuCompositionControl {
    pub mask: f32,
    pub channels: u32,
    pub shared_alpha: f32,
    pub alpha_rule: u32,
}

/// One per-pixel water coverage value in the shared response allocation.
///
/// The current response compositor reads this sidecar to preserve the
/// neutral single-arm path while allowing a water producer to publish
/// per-pixel coverage through
/// [`crate::shading::composition::ViewResponseComposition::plane`]. Keeping
/// the wrapper here makes the four-byte layout an explicit GPU contract
/// instead of re-deriving it from an unrelated scalar at each call site.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct GpuWaterCoverage(pub u32);

impl GpuCompositionControl {
    pub fn new(options: CompositionOptions<'_>) -> Result<Self, CompositionError> {
        if !options.mask.is_finite() || !(0.0..=1.0).contains(&options.mask) {
            return Err(CompositionError::InvalidMask);
        }
        let channels = match options.domain {
            BlendDomain::FullRadiance => ChannelMask::radiance().0,
            BlendDomain::DirectOnly => 1,
            BlendDomain::CustomChannelMask(mask)
                if mask.0 != 0 && ChannelMask::radiance().contains(mask) =>
            {
                mask.0
            }
            _ => return Err(CompositionError::InvalidChannelMask),
        };
        let alpha_rule = match options.alpha_rule {
            AlphaMergeRule::Shared => 0,
            AlphaMergeRule::Left => 1,
            AlphaMergeRule::Right => 2,
            AlphaMergeRule::Min => 3,
            AlphaMergeRule::Max => 4,
        };
        Ok(Self {
            mask: options.mask,
            channels,
            shared_alpha: options.shared_alpha.alpha(),
            alpha_rule,
        })
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct GpuAuxiliaryMerge {
    pub left: [u32; 4],
    pub right: [u32; 4],
    pub shared: [u32; 4],
    pub kind: u32,
    pub rule: u32,
    pub mask: f32,
    pub reserved: u32,
}

fn pack_aux(value: &AuxiliaryValue) -> (u32, [u32; 4]) {
    match value {
        AuxiliaryValue::Scalar(x) => (0, [x.to_bits(), 0, 0, 0]),
        AuxiliaryValue::Vector3(x) => (1, [x[0].to_bits(), x[1].to_bits(), x[2].to_bits(), 0]),
        AuxiliaryValue::Unsigned(x) => (2, [*x, 0, 0, 0]),
        AuxiliaryValue::Flags(x) => (3, [*x, 0, 0, 0]),
    }
}

/// Upload-ready single-pixel job. Construction enforces the CPU contract once;
/// names stay CPU-side, with only compact typed payloads uploaded to the GPU.
#[derive(Clone, Debug)]
pub struct GpuCompositionJob {
    pub left: GpuResponseOutput,
    pub right: GpuResponseOutput,
    pub control: GpuCompositionControl,
    pub auxiliary: Vec<GpuAuxiliaryMerge>,
    auxiliary_names: Vec<String>,
}

impl GpuCompositionJob {
    pub fn new(
        left: &ResponseOutput,
        right: &ResponseOutput,
        options: CompositionOptions<'_>,
    ) -> Result<Self, CompositionError> {
        let control = GpuCompositionControl::new(options)?;
        // Reuse the authoritative auxiliary validation, including one-sided
        // values, unused rules, type equality and integer operation restrictions.
        let validated = validate_response_composition_inputs(left, right, options)?;
        let mut auxiliary = Vec::with_capacity(validated.fields.len());
        let mut auxiliary_names = Vec::with_capacity(auxiliary.capacity());
        for (name, value) in &validated.fields {
            let (kind, _) = pack_aux(value);
            let source = |aux: &Auxiliary| {
                aux.fields
                    .get(name)
                    .map(pack_aux)
                    .map_or([0; 4], |(_, words)| words)
            };
            let rule = match options.auxiliary_rules.fields[name] {
                AuxiliaryMergeRule::Shared => 0,
                AuxiliaryMergeRule::Left => 1,
                AuxiliaryMergeRule::Right => 2,
                AuxiliaryMergeRule::RequireEqual => 3,
                AuxiliaryMergeRule::Linear => 4,
                AuxiliaryMergeRule::Min => 5,
                AuxiliaryMergeRule::Max => 6,
                AuxiliaryMergeRule::BitOr => 7,
            };
            auxiliary.push(GpuAuxiliaryMerge {
                left: source(&left.auxiliary),
                right: source(&right.auxiliary),
                shared: source(&options.auxiliary_rules.shared),
                kind,
                rule,
                mask: options.mask,
                reserved: 0,
            });
            auxiliary_names.push(name.clone());
        }
        Ok(Self {
            left: GpuResponseOutput::pack(left),
            right: GpuResponseOutput::pack(right),
            control,
            auxiliary,
            auxiliary_names,
        })
    }

    pub fn decode_auxiliary(
        &self,
        output: &[GpuAuxiliaryMerge],
    ) -> Result<Auxiliary, CompositionError> {
        if output.len() != self.auxiliary_names.len() {
            return Err(CompositionError::InvalidAuxiliary(
                "GPU auxiliary count".into(),
            ));
        }
        let mut auxiliary = Auxiliary::default();
        for ((name, expected), result) in
            self.auxiliary_names.iter().zip(&self.auxiliary).zip(output)
        {
            if result.kind != expected.kind {
                return Err(CompositionError::AuxiliaryTypeMismatch(name.clone()));
            }
            let value = match result.kind {
                0 => AuxiliaryValue::Scalar(f32::from_bits(result.left[0])),
                1 => AuxiliaryValue::Vector3(core::array::from_fn(|i| {
                    f32::from_bits(result.left[i])
                })),
                2 => AuxiliaryValue::Unsigned(result.left[0]),
                3 => AuxiliaryValue::Flags(result.left[0]),
                _ => return Err(CompositionError::AuxiliaryTypeMismatch(name.clone())),
            };
            auxiliary.fields.insert(name.clone(), value);
        }
        Ok(auxiliary)
    }
}

const _: () = assert!(size_of::<GpuResponseOutput>() == 64);
const _: () = assert!(size_of::<GpuCompositionControl>() == 16);
const _: () = assert!(size_of::<GpuWaterCoverage>() == 4);
const _: () = assert!(size_of::<GpuAuxiliaryMerge>() == 64);

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::{align_of, size_of};

    #[test]
    fn water_coverage_is_a_four_byte_storage_element() {
        assert_eq!(size_of::<GpuWaterCoverage>(), size_of::<u32>());
        assert_eq!(align_of::<GpuWaterCoverage>(), align_of::<u32>());
        assert_eq!(GpuWaterCoverage::default(), GpuWaterCoverage(0));
    }
}
