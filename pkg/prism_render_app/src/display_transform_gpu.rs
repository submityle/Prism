//! Standalone RHI compute execution for Prism's supported OCIO-style display
//! transform. This component intentionally does not register itself with the
//! render app: callers own its lifetime and choose where to submit the pass.

use std::sync::Arc;

use prism_render_architecture::lighting::ocio::{OcioConfig, OcioConfigError};
use prism_render_driver as rhi;
use rhi::{RenderDevice, RenderQueue};

const WORKGROUP_SIZE: u32 = 64;
const PARAM_BYTES: u64 = 48;
const STORAGE_PIXEL_BYTES: u64 = 16;

/// Errors detected before recording display-transform GPU work.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DisplayTransformGpuError {
    /// The configuration failed the CPU reference's validation contract.
    InvalidConfig(OcioConfigError),
    /// Empty inputs are not dispatched; this avoids invalid zero-sized bindings.
    EmptyImage,
    /// The storage range exceeds the device's declared binding limit.
    StorageBindingLimit { requested: u64, limit: u64 },
    /// The one-dimensional dispatch exceeds the device's workgroup limit.
    DispatchLimit { requested: u32, limit: u32 },
}

/// A device-local WGSL implementation of [`OcioConfig::reference_transform`].
///
/// Input and output buffers are arrays of 16-byte `[r, g, b, a]` `f32` pixels.
/// RGB is transformed from linear scene sRGB/D65 into the configured display
/// gamut and transfer encoding; alpha is copied unchanged. The component owns
/// its pipeline and parameter buffer and releases them when dropped.
pub struct DisplayTransformGpu {
    device: Arc<dyn RenderDevice + Send + Sync>,
    bind_group_layout: rhi::BindGroupLayoutId,
    pipeline_layout: rhi::PipelineLayoutId,
    shader: rhi::ShaderModuleId,
    pipeline: rhi::ComputePipelineId,
    params: rhi::BufferId,
}

impl DisplayTransformGpu {
    /// Creates the standalone display-transform pipeline on `device`.
    #[must_use]
    pub fn new(device: Arc<dyn RenderDevice + Send + Sync>) -> Self {
        let bind_group_layout = device.create_bind_group_layout(&rhi::BindGroupLayoutDescriptor {
            label: Some("ocio-display-transform-bindings".into()),
            entries: vec![
                storage_layout(0, true),
                storage_layout(1, false),
                rhi::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: rhi::ShaderStages::COMPUTE,
                    ty: rhi::BindingType::Buffer {
                        ty: rhi::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(PARAM_BYTES),
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&rhi::PipelineLayoutDescriptor {
            label: Some("ocio-display-transform-pipeline-layout".into()),
            bind_group_layouts: vec![bind_group_layout],
            push_constant_ranges: vec![],
        });
        let shader =
            device.create_shader_module(&rhi::ShaderModuleDescriptor::wgsl(DISPLAY_TRANSFORM_WGSL));
        let pipeline = device.create_compute_pipeline(&rhi::ComputePipelineDescriptor {
            label: Some("ocio-display-transform".into()),
            layout: Some(pipeline_layout),
            module: shader,
            entry_point: "main".into(),
        });
        let params = device.create_buffer(&rhi::BufferDescriptor::new(
            PARAM_BYTES,
            rhi::BufferUsages::UNIFORM | rhi::BufferUsages::COPY_DST,
        ));
        Self {
            device,
            bind_group_layout,
            pipeline_layout,
            shader,
            pipeline,
            params,
        }
    }

    /// Encodes and submits one transform over `pixel_count` pixels.
    ///
    /// `source` must have `STORAGE | COPY_DST` (or otherwise be initialized)
    /// usage and `destination` must have `STORAGE | COPY_SRC` usage. Both
    /// buffers must contain at least `pixel_count * 16` bytes. The queue must
    /// belong to this component's device. Submission is synchronous with
    /// respect to ownership of the temporary bind group, not GPU completion;
    /// the RHI backend retains native resources needed by submitted work.
    pub fn execute(
        &mut self,
        queue: &dyn RenderQueue,
        config: &OcioConfig,
        source: rhi::BufferId,
        destination: rhi::BufferId,
        pixel_count: u32,
    ) -> Result<(), DisplayTransformGpuError> {
        config
            .validate()
            .map_err(DisplayTransformGpuError::InvalidConfig)?;
        if pixel_count == 0 {
            return Err(DisplayTransformGpuError::EmptyImage);
        }

        let range_bytes = u64::from(pixel_count) * STORAGE_PIXEL_BYTES;
        let storage_limit = self
            .device
            .capabilities()
            .limits
            .max_storage_buffer_binding_size;
        if range_bytes > storage_limit {
            return Err(DisplayTransformGpuError::StorageBindingLimit {
                requested: range_bytes,
                limit: storage_limit,
            });
        }
        let workgroups = pixel_count.div_ceil(WORKGROUP_SIZE);
        let dispatch_limit = self
            .device
            .capabilities()
            .limits
            .max_compute_workgroups_per_dimension;
        if workgroups > dispatch_limit {
            return Err(DisplayTransformGpuError::DispatchLimit {
                requested: workgroups,
                limit: dispatch_limit,
            });
        }

        let parameters = encode_parameters(config, pixel_count);
        queue.write_buffer(self.params, 0, bytemuck::cast_slice(&parameters));
        let group = self.device.create_bind_group(&rhi::BindGroupDescriptor {
            label: Some("ocio-display-transform-resources".into()),
            layout: Some(self.bind_group_layout),
            entries: vec![
                buffer_entry(0, source, range_bytes),
                buffer_entry(1, destination, range_bytes),
                buffer_entry(2, self.params, PARAM_BYTES),
            ],
        });
        let mut encoder = rhi::CommandEncoder::labeled("ocio-display-transform");
        encoder.push_compute_pass(
            Some("ocio-display-transform".into()),
            vec![
                rhi::ComputeCommand::SetPipeline(self.pipeline),
                rhi::ComputeCommand::SetBindGroup {
                    index: 0,
                    bind_group: group,
                    dynamic_offsets: vec![],
                },
                rhi::ComputeCommand::Dispatch {
                    x: workgroups,
                    y: 1,
                    z: 1,
                },
            ],
        );
        queue.submit(&[encoder.finish()]);
        self.device.destroy_bind_group(group);
        Ok(())
    }
}

impl Drop for DisplayTransformGpu {
    fn drop(&mut self) {
        self.device.destroy_compute_pipeline(self.pipeline);
        self.device.destroy_shader_module(self.shader);
        self.device.destroy_pipeline_layout(self.pipeline_layout);
        self.device
            .destroy_bind_group_layout(self.bind_group_layout);
        self.device.destroy_buffer(self.params);
    }
}

fn storage_layout(binding: u32, read_only: bool) -> rhi::BindGroupLayoutEntry {
    rhi::BindGroupLayoutEntry {
        binding,
        visibility: rhi::ShaderStages::COMPUTE,
        ty: rhi::BindingType::Buffer {
            ty: rhi::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: Some(STORAGE_PIXEL_BYTES),
        },
        count: None,
    }
}

fn buffer_entry(binding: u32, buffer: rhi::BufferId, size: u64) -> rhi::BindGroupEntry {
    rhi::BindGroupEntry {
        binding,
        resource: rhi::BindingResource::Buffer {
            buffer,
            offset: 0,
            size: Some(size),
        },
    }
}

fn encode_parameters(config: &OcioConfig, pixel_count: u32) -> [u32; 12] {
    [
        config.tone_mapping as u32,
        config.display_gamut as u32,
        config.transfer_function as u32,
        config.exposure_stops.to_bits(),
        config.white_balance.red_gain.to_bits(),
        config.white_balance.green_gain.to_bits(),
        config.white_balance.blue_gain.to_bits(),
        0,
        pixel_count,
        0,
        0,
        0,
    ]
}

/// WGSL source kept in this file so the component has no shader-file or module
/// registration dependency.
pub const DISPLAY_TRANSFORM_WGSL: &str = r#"
struct Params {
    modes: vec4<u32>,
    gains: vec4<f32>,
    limits: vec4<u32>,
};

@group(0) @binding(0) var<storage, read> source_pixels: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> destination_pixels: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

fn tone_map(value: f32, mode: u32) -> f32 {
    if (mode == 1u) {
        let numerator = value * (2.51 * value + 0.03);
        let denominator = value * (2.43 * value + 0.59) + 0.14;
        return clamp(numerator / denominator, 0.0, 1.0);
    }
    if (mode == 2u) {
        let numerator = value * (value + 0.0245786) - 0.000090537;
        let denominator = value * (0.983729 * value + 0.432951) + 0.238081;
        return clamp(numerator / denominator, 0.0, 1.0);
    }
    return value;
}

fn gamut_convert(rgb: vec3<f32>, gamut: u32) -> vec3<f32> {
    if (gamut == 1u) {
        return vec3<f32>(
            dot(vec3<f32>(0.8225927, 0.1775330, 0.0), rgb),
            dot(vec3<f32>(0.0331995, 0.9667835, 0.0), rgb),
            dot(vec3<f32>(0.0170853, 0.0723957, 0.9103015), rgb)
        );
    }
    if (gamut == 2u) {
        return vec3<f32>(
            dot(vec3<f32>(0.6274040, 0.3292820, 0.0433136), rgb),
            dot(vec3<f32>(0.0690970, 0.9195400, 0.0113612), rgb),
            dot(vec3<f32>(0.0163916, 0.0880132, 0.8955952), rgb)
        );
    }
    return rgb;
}

fn transfer(value: f32, mode: u32) -> f32 {
    let x = clamp(value, 0.0, 1.0);
    if (mode == 1u) {
        if (x <= 0.0031308) {
            return 12.92 * x;
        }
        return 1.055 * pow(x, 1.0 / 2.4) - 0.055;
    }
    if (mode == 2u) {
        return pow(x, 1.0 / 2.2);
    }
    if (mode == 3u) {
        return pow(x, 1.0 / 2.4);
    }
    return x;
}

fn transform(rgb_in: vec3<f32>) -> vec3<f32> {
    let exposure = exp2(bitcast<f32>(params.modes.w));
    let gains = params.gains.xyz;
    let exposed = max(rgb_in, vec3<f32>(0.0)) * gains * exposure;
    let bounded = min(exposed, vec3<f32>(1.0e10));
    let mapped = vec3<f32>(
        tone_map(bounded.x, params.modes.x),
        tone_map(bounded.y, params.modes.x),
        tone_map(bounded.z, params.modes.x)
    );
    let converted = gamut_convert(mapped, params.modes.y);
    return vec3<f32>(
        transfer(max(converted.x, 0.0), params.modes.z),
        transfer(max(converted.y, 0.0), params.modes.z),
        transfer(max(converted.z, 0.0), params.modes.z)
    );
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.limits.x) {
        return;
    }
    let source = source_pixels[id.x];
    destination_pixels[id.x] = vec4<f32>(transform(source.xyz), source.w);
}
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use prism_render_architecture::lighting::ocio::{
        OcioDisplayGamut, OcioToneMapping, OcioTransferFunction,
    };
    use prism_render_driver::{BufferDescriptor, BufferUsages};
    use prism_render_driver_wgpu::{require_hardware_gpu, HardwareDeviceError, WgpuInstance};

    #[test]
    fn physical_gpu_transform_matches_cpu_reference_for_supported_pipeline_matrix() {
        let (device, queue) = match WgpuInstance::new().request_default_hardware_device() {
            Ok(device_and_queue) => device_and_queue,
            Err(HardwareDeviceError::NoAdapter) => {
                assert!(
                    !require_hardware_gpu(),
                    "PRISM_REQUIRE_GPU=1 but no physical GPU adapter was found"
                );
                eprintln!("SKIPPED OCIO GPU parity: no adapter");
                return;
            }
            Err(HardwareDeviceError::NonHardware(evidence)) => {
                assert!(
                    !require_hardware_gpu(),
                    "PRISM_REQUIRE_GPU=1 requires a physical GPU; observed {evidence}"
                );
                eprintln!("SKIPPED OCIO GPU parity: non-hardware adapter: {evidence}");
                return;
            }
            Err(HardwareDeviceError::Device(error)) => {
                panic!("physical GPU was found but device creation failed: {error}");
            }
        };
        let device: Arc<dyn RenderDevice + Send + Sync> = Arc::new(device);
        let pixels: [[f32; 4]; 6] = [
            [0.0, 0.0, 0.0, 0.0],
            [0.0031308, 0.018, 0.25, 0.2],
            [1.0, 0.5, 0.25, 0.4],
            [4.0, 2.0, 0.1, 0.6],
            [20.0, 0.01, 8.0, 0.8],
            [-2.0, 0.4, 1.5, 1.0],
        ];
        let bytes = bytemuck::cast_slice(&pixels);
        let input = device.create_buffer(&BufferDescriptor::new(
            bytes.len() as u64,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
        ));
        let output = device.create_buffer(&BufferDescriptor::new(
            bytes.len() as u64,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        ));
        queue.write_buffer(input, 0, bytes);
        let mut transform = DisplayTransformGpu::new(Arc::clone(&device));
        for tone_mapping in [
            OcioToneMapping::Linear,
            OcioToneMapping::Aces,
            OcioToneMapping::Agx,
        ] {
            for display_gamut in [
                OcioDisplayGamut::Srgb,
                OcioDisplayGamut::DisplayP3,
                OcioDisplayGamut::Rec2020,
            ] {
                for transfer_function in [
                    OcioTransferFunction::Linear,
                    OcioTransferFunction::Srgb,
                    OcioTransferFunction::Gamma22,
                    OcioTransferFunction::Gamma24,
                ] {
                    let config = OcioConfig {
                        tone_mapping,
                        exposure_stops: 0.75,
                        white_balance:
                            prism_render_architecture::lighting::ocio::OcioWhiteBalance::new(
                                1.1, 0.9, 1.2,
                            ),
                        display_gamut,
                        transfer_function,
                        ..OcioConfig::srgb()
                    };
                    transform
                        .execute(&queue, &config, input, output, pixels.len() as u32)
                        .expect("valid transform dispatch");
                    let actual_bytes = queue.read_buffer(output, bytes.len() as u64);
                    let actual: Vec<[f32; 4]> = actual_bytes
                        .chunks_exact(16)
                        .map(|pixel| {
                            core::array::from_fn(|channel| {
                                let offset = channel * 4;
                                f32::from_le_bytes(
                                    pixel[offset..offset + 4]
                                        .try_into()
                                        .expect("four-byte channel"),
                                )
                            })
                        })
                        .collect();
                    for (index, (source, result)) in pixels.iter().zip(actual).enumerate() {
                        let expected = config
                            .reference_transform([source[0], source[1], source[2]])
                            .expect("finite CPU reference input");
                        for channel in 0..3 {
                            assert!(
                                (result[channel] - expected[channel]).abs() <= 2.5e-4,
                                "tone={tone_mapping:?} gamut={display_gamut:?} transfer={transfer_function:?} pixel={index} channel={channel}: GPU={} CPU={}",
                                result[channel],
                                expected[channel]
                            );
                        }
                        assert_eq!(result[3], source[3], "alpha must pass through");
                    }
                }
            }
        }
        device.destroy_buffer(input);
        device.destroy_buffer(output);
    }
}
