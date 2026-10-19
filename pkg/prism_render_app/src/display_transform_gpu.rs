//! Render-graph compute execution for Prism's supported OCIO-style display transform.

use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc,
};

use prism_render_architecture::lighting::ocio::{OcioConfig, OcioConfigError};
use prism_render_driver as rhi;
use prism_render_graph::{ImportedTexture, PassFlags, RenderGraph, TextureDesc, TextureHandle};
use rhi::{Extent3d, RenderDevice, RenderQueue, TextureViewId};

const WORKGROUP_EDGE: u32 = 8;
const PARAM_BYTES: u64 = 48;

/// Errors detected before recording display-transform GPU work.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DisplayTransformGpuError {
    /// The configuration failed the CPU reference's validation contract.
    InvalidConfig(OcioConfigError),
    /// The image extent is empty or exceeds a device texture limit.
    InvalidExtent { width: u32, height: u32 },
    /// The two-dimensional dispatch exceeds a device workgroup limit.
    DispatchLimit { x: u32, y: u32, limit: u32 },
}

/// A device-local WGSL implementation of [`OcioConfig::reference_transform`].
///
/// Samples the frame's linear `Rgba8Unorm` color texture and writes a persistent
/// `Rgba8Unorm` texture containing display-transfer-encoded RGB. The final
/// present pass decodes sRGB for an sRGB attachment (whose hardware store then
/// performs exactly one encode); non-sRGB attachments consume encoded values
/// directly.
pub struct DisplayTransformGpu {
    device: Arc<dyn RenderDevice + Send + Sync>,
    bindings: rhi::BindGroupLayoutId,
    pipeline_layout: rhi::PipelineLayoutId,
    shader: rhi::ShaderModuleId,
    pipeline: rhi::ComputePipelineId,
    params: rhi::BufferId,
    bind_group: rhi::BindGroupId,
    output: rhi::TextureId,
    output_view: TextureViewId,
    extent: Extent3d,
    in_flight: Arc<AtomicUsize>,
    destroy_allowed: Arc<AtomicBool>,
}

impl DisplayTransformGpu {
    /// Creates the transform pipeline and persistent output image.
    #[must_use]
    pub fn new(
        device: Arc<dyn RenderDevice + Send + Sync>,
        source_view: TextureViewId,
        extent: Extent3d,
        in_flight: Arc<AtomicUsize>,
        destroy_allowed: Arc<AtomicBool>,
    ) -> Result<Self, DisplayTransformGpuError> {
        validate_extent(&*device, extent)?;
        let bindings = device.create_bind_group_layout(&rhi::BindGroupLayoutDescriptor {
            label: Some("ocio-display-transform-bindings".into()),
            entries: vec![
                rhi::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: rhi::ShaderStages::COMPUTE,
                    ty: rhi::BindingType::Texture {
                        sample_type: rhi::TextureSampleType::Float,
                        view_dimension: rhi::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                rhi::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: rhi::ShaderStages::COMPUTE,
                    ty: rhi::BindingType::StorageTexture {
                        access: rhi::StorageTextureAccess::WriteOnly,
                        format: rhi::TextureFormat::Rgba8Unorm,
                        view_dimension: rhi::TextureViewDimension::D2,
                    },
                    count: None,
                },
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
            bind_group_layouts: vec![bindings],
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
        let output = device.create_texture(&rhi::TextureDescriptor::new_2d(
            extent,
            rhi::TextureFormat::Rgba8Unorm,
            rhi::TextureUsages::RENDER_ATTACHMENT
                .union(rhi::TextureUsages::STORAGE_BINDING)
                .union(rhi::TextureUsages::TEXTURE_BINDING)
                .union(rhi::TextureUsages::COPY_SRC),
        ));
        let output_view =
            device.create_texture_view(output, &rhi::TextureViewDescriptor::default());
        let bind_group = device.create_bind_group(&rhi::BindGroupDescriptor {
            label: Some("ocio-display-transform-resources".into()),
            layout: Some(bindings),
            entries: vec![
                rhi::BindGroupEntry {
                    binding: 0,
                    resource: rhi::BindingResource::TextureView(source_view),
                },
                rhi::BindGroupEntry {
                    binding: 1,
                    resource: rhi::BindingResource::TextureView(output_view),
                },
                buffer_entry(2, params, PARAM_BYTES),
            ],
        });
        Ok(Self {
            device,
            bindings,
            pipeline_layout,
            shader,
            pipeline,
            params,
            bind_group,
            output,
            output_view,
            extent,
            in_flight,
            destroy_allowed,
        })
    }

    /// Uploads frame parameters and appends the transform to the frame graph.
    pub fn append(
        &self,
        graph: &mut RenderGraph,
        queue: &dyn RenderQueue,
        config: &OcioConfig,
        source: TextureHandle,
    ) -> Result<TextureHandle, DisplayTransformGpuError> {
        config
            .validate()
            .map_err(DisplayTransformGpuError::InvalidConfig)?;
        validate_extent(&*self.device, self.extent)?;
        let parameters = encode_parameters(config, self.extent);
        queue.write_buffer(self.params, 0, bytemuck::cast_slice(&parameters));
        let destination = graph.import_texture(
            "ocio-display-output",
            TextureDesc::color(rhi::TextureFormat::Rgba8Unorm).with_usage(
                rhi::TextureUsages::STORAGE_BINDING
                    .union(rhi::TextureUsages::TEXTURE_BINDING)
                    .union(rhi::TextureUsages::COPY_SRC),
            ),
            ImportedTexture {
                texture: self.output,
                default_view: self.output_view,
                entry_state: rhi::TextureState::initial(),
                exit_state: rhi::TextureState::shader_read(rhi::PipelineStages::FRAGMENT_SHADER),
                mip_levels: 1,
                array_layers: 1,
            },
        );
        let pipeline = self.pipeline;
        let bind_group = self.bind_group;
        let x = self.extent.width.div_ceil(WORKGROUP_EDGE);
        let y = self.extent.height.div_ceil(WORKGROUP_EDGE);
        graph.add_compute_pass(
            "ocio-display-transform",
            PassFlags::SIDE_EFFECT,
            move |builder| {
                builder.sample(source);
                builder.write_storage_texture(destination);
                move |ctx| {
                    ctx.set_compute_pipeline(pipeline)
                        .set_compute_bind_group(0, bind_group, Vec::new())
                        .dispatch(x, y, 1);
                }
            },
        );
        Ok(destination)
    }

    /// View of the persistent display output, bound by the final raster pass.
    pub fn output_view(&self) -> TextureViewId {
        self.output_view
    }

    /// Texture containing the latest encoded output, suitable for graph readback.
    pub fn output_texture(&self) -> rhi::TextureId {
        self.output
    }
}

impl Drop for DisplayTransformGpu {
    fn drop(&mut self) {
        if !self.destroy_allowed.load(Ordering::Acquire)
            || self.in_flight.load(Ordering::Acquire) != 0
        {
            return;
        }
        self.device.destroy_bind_group(self.bind_group);
        self.device.destroy_texture_view(self.output_view);
        self.device.destroy_texture(self.output);
        self.device.destroy_compute_pipeline(self.pipeline);
        self.device.destroy_shader_module(self.shader);
        self.device.destroy_pipeline_layout(self.pipeline_layout);
        self.device.destroy_bind_group_layout(self.bindings);
        self.device.destroy_buffer(self.params);
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

fn validate_extent(
    device: &dyn RenderDevice,
    extent: Extent3d,
) -> Result<(), DisplayTransformGpuError> {
    if extent.width == 0 || extent.height == 0 {
        return Err(DisplayTransformGpuError::InvalidExtent {
            width: extent.width,
            height: extent.height,
        });
    }
    let limits = device.capabilities().limits;
    if extent.width > limits.max_texture_dimension_2d
        || extent.height > limits.max_texture_dimension_2d
    {
        return Err(DisplayTransformGpuError::InvalidExtent {
            width: extent.width,
            height: extent.height,
        });
    }
    let x = extent.width.div_ceil(WORKGROUP_EDGE);
    let y = extent.height.div_ceil(WORKGROUP_EDGE);
    let limit = limits.max_compute_workgroups_per_dimension;
    if x > limit || y > limit {
        return Err(DisplayTransformGpuError::DispatchLimit { x, y, limit });
    }
    Ok(())
}

fn encode_parameters(config: &OcioConfig, extent: Extent3d) -> [u32; 12] {
    [
        config.tone_mapping as u32,
        config.display_gamut as u32,
        config.transfer_function as u32,
        config.exposure_stops.to_bits(),
        config.white_balance.red_gain.to_bits(),
        config.white_balance.green_gain.to_bits(),
        config.white_balance.blue_gain.to_bits(),
        0,
        extent.width,
        extent.height,
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

@group(0) @binding(0) var source_image: texture_2d<f32>;
@group(0) @binding(1) var destination_image: texture_storage_2d<rgba8unorm, write>;
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

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.limits.x || id.y >= params.limits.y) {
        return;
    }
    let pixel = vec2<i32>(id.xy);
    let source = textureLoad(source_image, pixel, 0);
    textureStore(destination_image, pixel, vec4<f32>(transform(source.rgb), source.a));
}
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use prism_render_architecture::lighting::ocio::{
        OcioDisplayGamut, OcioToneMapping, OcioTransferFunction,
    };
    use prism_render_driver::{
        Extent3d, ImageDataLayout, TextureDescriptor, TextureUsages, TextureViewDescriptor,
        TextureWrite,
    };
    use prism_render_driver_wgpu::{require_hardware_gpu, HardwareDeviceError, WgpuInstance};
    use prism_render_graph::{ImportedTexture, RenderGraph, TextureDesc};

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
        let device_handle = Arc::new(device);
        let device: Arc<dyn RenderDevice + Send + Sync> = device_handle.clone();
        let extent = Extent3d::new_2d(3, 2);
        let pixels: [[u8; 4]; 6] = [
            [0, 0, 0, 0],
            [1, 5, 64, 51],
            [64, 128, 255, 102],
            [255, 128, 64, 153],
            [240, 32, 128, 204],
            [17, 220, 80, 255],
        ];
        let input = device.create_texture(&TextureDescriptor::new_2d(
            extent,
            rhi::TextureFormat::Rgba8Unorm,
            TextureUsages::TEXTURE_BINDING.union(TextureUsages::COPY_DST),
        ));
        let input_view = device.create_texture_view(input, &TextureViewDescriptor::default());
        queue.write_texture(
            TextureWrite {
                texture: input,
                mip_level: 0,
                origin: [0, 0, 0],
                size: extent,
            },
            bytemuck::cast_slice(&pixels),
            ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(12),
                rows_per_image: Some(2),
            },
        );
        let lifetime_count = Arc::new(AtomicUsize::new(0));
        let destroy_allowed = Arc::new(AtomicBool::new(true));
        let mut transform = DisplayTransformGpu::new(
            Arc::clone(&device),
            input_view,
            extent,
            lifetime_count,
            destroy_allowed,
        )
        .expect("valid extent");
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
                    let mut graph = RenderGraph::new(extent);
                    let source = graph.import_texture(
                        "ocio-test-source",
                        TextureDesc::color(rhi::TextureFormat::Rgba8Unorm)
                            .with_usage(TextureUsages::TEXTURE_BINDING),
                        ImportedTexture {
                            texture: input,
                            default_view: input_view,
                            entry_state: rhi::TextureState::initial(),
                            exit_state: rhi::TextureState::shader_read(
                                rhi::PipelineStages::COMPUTE_SHADER,
                            ),
                            mip_levels: 1,
                            array_layers: 1,
                        },
                    );
                    transform
                        .append(&mut graph, &queue, &config, source)
                        .expect("valid transform graph");
                    graph
                        .execute(&*device_handle, &queue)
                        .expect("valid transform graph execution");
                    let actual = queue.read_texture_rgba8(
                        transform.output_texture(),
                        extent.width,
                        extent.height,
                    );
                    for (index, (source, result)) in
                        pixels.iter().zip(actual.chunks_exact(4)).enumerate()
                    {
                        let source = source.map(|value| f32::from(value) / 255.0);
                        let expected = config
                            .reference_transform([source[0], source[1], source[2]])
                            .expect("finite CPU reference input");
                        for channel in 0..3 {
                            let actual = f32::from(result[channel]) / 255.0;
                            assert!(
                                (actual - expected[channel]).abs() <= 0.012,
                                "tone={tone_mapping:?} gamut={display_gamut:?} transfer={transfer_function:?} pixel={index} channel={channel}: GPU={} CPU={}",
                                actual,
                                expected[channel]
                            );
                        }
                        assert_eq!(result[3], pixels[index][3], "alpha must pass through");
                    }
                }
            }
        }
        device.destroy_texture_view(input_view);
        device.destroy_texture(input);
    }
}
