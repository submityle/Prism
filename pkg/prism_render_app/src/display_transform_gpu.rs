//! Render-graph compute execution for Prism's supported OCIO-style display transform.

use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc,
};

use prism_ecs::prelude::Resource;
use prism_render_architecture::lighting::ocio::{OcioConfig, OcioConfigError};
use prism_render_driver as rhi;
use prism_render_graph::{ImportedTexture, PassFlags, RenderGraph, TextureDesc, TextureHandle};
use rhi::{Extent3d, RenderDevice, RenderQueue, TextureId, TextureViewId};

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
    /// An Ember graph could not be declared in the host frame graph.
    EmberGraph,
    /// Particle camera/owner admission rejected the host frame graph.
    ParticleGraph,
}

/// A device-local WGSL implementation of [`OcioConfig::reference_transform`].
///
/// Samples the frame's linear HDR `Rgba16Float` color texture (also accepting
/// `Rgba8Unorm` inputs) and writes a persistent `Rgba8Unorm` texture containing
/// display-transfer-encoded RGB. The final present pass decodes sRGB for an
/// sRGB attachment (whose hardware store then
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
    output: TextureId,
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
                // The headless host exposes this terminal image for readback;
                // window presentation adds a graph read that transitions it to
                // fragment access before the graph's final copy-source state.
                exit_state: rhi::TextureState::copy_src(),
                mip_levels: 1,
                array_layers: 1,
            },
        );
        let pipeline = self.pipeline;
        let bind_group = self.bind_group;
        let x = self.extent.width.div_ceil(WORKGROUP_EDGE);
        let y = self.extent.height.div_ceil(WORKGROUP_EDGE);
        let mut output = destination;
        graph.add_compute_pass(
            "ocio-display-transform",
            PassFlags::SIDE_EFFECT,
            |builder| {
                builder.sample(source);
                output = builder.write_storage_texture(destination);
                move |ctx| {
                    ctx.set_compute_pipeline(pipeline)
                        .set_compute_bind_group(0, bind_group, Vec::new())
                        .dispatch(x, y, 1);
                }
            },
        );
        Ok(output)
    }

    /// View of the persistent display output, bound by the final raster pass.
    pub fn output_view(&self) -> TextureViewId {
        self.output_view
    }

    /// Texture containing the latest encoded output, suitable for graph readback.
    pub fn output_texture(&self) -> TextureId {
        self.output
    }

    fn abandon_device_generation(&self) {
        self.destroy_allowed.store(false, Ordering::Release);
    }
}

/// Render-world owner for the headless frame's display transform.
///
/// The offscreen product uses the same canonical sRGB configuration as window
/// presentation. The transform is tied to the current frame target and device
/// generation; the host rebuilds it after a resize or device recovery.
#[derive(Resource)]
pub(crate) struct OffscreenDisplayTransform {
    config: OcioConfig,
    transform: Option<DisplayTransformGpu>,
    extent: Option<Extent3d>,
    device_generation: Option<u64>,
}

/// Latest submitted headless/offscreen display-space frame.
///
/// The texture is owned by the render device and remains valid until the next
/// target resize, device-generation replacement, or render-app shutdown. A
/// consumer can use the published submission id to coordinate readback with
/// its host; the render queue itself preserves ordering for subsequent work.
#[derive(Resource, Default)]
pub struct OffscreenFrameOutput {
    latest: Option<OffscreenOutputFrame>,
}

/// Stable metadata for one submitted offscreen frame.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OffscreenOutputFrame {
    texture: TextureId,
    extent: Extent3d,
    frame_index: u64,
    submission: u64,
    device_generation: u64,
}

impl OffscreenOutputFrame {
    pub(crate) const fn new(
        texture: TextureId,
        extent: Extent3d,
        frame_index: u64,
        submission: u64,
        device_generation: u64,
    ) -> Self {
        Self {
            texture,
            extent,
            frame_index,
            submission,
            device_generation,
        }
    }

    /// Display-encoded output texture for this frame.
    #[must_use]
    pub const fn texture(self) -> TextureId {
        self.texture
    }

    /// Physical pixel extent of the output texture.
    #[must_use]
    pub const fn extent(self) -> Extent3d {
        self.extent
    }

    /// Extracted frame index that produced this output.
    #[must_use]
    pub const fn frame_index(self) -> u64 {
        self.frame_index
    }

    /// Runtime submission sequence for ordering downstream work/readback.
    #[must_use]
    pub const fn submission(self) -> u64 {
        self.submission
    }

    /// Device generation owning [`Self::texture`].
    #[must_use]
    pub const fn device_generation(self) -> u64 {
        self.device_generation
    }
}

impl OffscreenFrameOutput {
    /// Most recently submitted transformed frame, if one has been produced.
    #[must_use]
    pub const fn latest(&self) -> Option<OffscreenOutputFrame> {
        self.latest
    }

    pub(crate) fn publish(&mut self, frame: OffscreenOutputFrame) {
        self.latest = Some(frame);
    }

    pub(crate) fn invalidate(&mut self) {
        self.latest = None;
    }
}

impl Default for OffscreenDisplayTransform {
    fn default() -> Self {
        Self::new(OcioConfig::srgb())
    }
}

impl OffscreenDisplayTransform {
    pub(crate) fn new(config: OcioConfig) -> Self {
        Self {
            config,
            transform: None,
            extent: None,
            device_generation: None,
        }
    }

    pub(crate) fn ensure(
        &mut self,
        device: Arc<dyn RenderDevice + Send + Sync>,
        source_view: TextureViewId,
        extent: Extent3d,
        device_generation: u64,
    ) -> Result<(), DisplayTransformGpuError> {
        let current = self.transform.is_some()
            && self.extent == Some(extent)
            && self.device_generation == Some(device_generation);
        if current {
            return Ok(());
        }
        if self
            .device_generation
            .is_some_and(|generation| generation != device_generation)
        {
            if let Some(previous) = &self.transform {
                previous.abandon_device_generation();
            }
        }
        self.transform = None;
        let in_flight = Arc::new(AtomicUsize::new(0));
        let destroy_allowed = Arc::new(AtomicBool::new(true));
        self.transform = Some(DisplayTransformGpu::new(
            device,
            source_view,
            extent,
            in_flight,
            destroy_allowed,
        )?);
        self.extent = Some(extent);
        self.device_generation = Some(device_generation);
        Ok(())
    }

    pub(crate) fn append(
        &self,
        graph: &mut RenderGraph,
        queue: &dyn RenderQueue,
        source: TextureHandle,
    ) -> Result<TextureHandle, DisplayTransformGpuError> {
        self.transform
            .as_ref()
            .expect("offscreen display transform is prepared before graph construction")
            .append(graph, queue, &self.config, source)
    }

    pub(crate) fn output_texture(&self) -> Option<TextureId> {
        self.transform
            .as_ref()
            .map(DisplayTransformGpu::output_texture)
    }

    pub(crate) fn allow_destruction(&self, allowed: bool) {
        if let Some(transform) = &self.transform {
            transform.destroy_allowed.store(allowed, Ordering::Release);
        }
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
    use prism_render_architecture::particle::half_float_f16::{f16_bits_to_f32, f32_to_f16_bits};
    use prism_render_driver::{
        Extent3d, ImageDataLayout, RenderDevice, TextureDescriptor, TextureUsages,
        TextureViewDescriptor, TextureWrite,
    };
    use prism_render_driver_null::NullInstance;
    use prism_render_driver_wgpu::{require_hardware_gpu, HardwareDeviceError, WgpuInstance};
    use prism_render_graph::{ImportedTexture, RenderGraph, TextureDesc};
    use prism_render_runtime::RenderRuntime;

    #[test]
    fn null_transform_returns_the_post_write_handle_for_present() {
        let extent = Extent3d::new_2d(4, 4);
        let runtime = RenderRuntime::<NullInstance>::headless(NullInstance::new(), extent)
            .expect("null runtime");
        let device = runtime.device_handle();
        let source = device.create_texture(&TextureDescriptor::new_2d(
            extent,
            rhi::TextureFormat::Rgba16Float,
            TextureUsages::TEXTURE_BINDING,
        ));
        let source_view = device.create_texture_view(source, &TextureViewDescriptor::default());
        let device_dyn: Arc<dyn RenderDevice + Send + Sync> = device.clone();
        let transform = DisplayTransformGpu::new(
            device_dyn,
            source_view,
            extent,
            Arc::new(AtomicUsize::new(0)),
            Arc::new(AtomicBool::new(true)),
        )
        .expect("null display transform");
        let mut graph = RenderGraph::new(extent);
        let source_handle = graph.import_texture(
            "null-display-source",
            TextureDesc::color(rhi::TextureFormat::Rgba16Float)
                .with_usage(TextureUsages::TEXTURE_BINDING),
            ImportedTexture {
                texture: source,
                default_view: source_view,
                entry_state: rhi::TextureState::initial(),
                exit_state: rhi::TextureState::shader_read(rhi::PipelineStages::COMPUTE_SHADER),
                mip_levels: 1,
                array_layers: 1,
            },
        );
        let transformed = transform
            .append(
                &mut graph,
                runtime.queue(),
                &OcioConfig::srgb(),
                source_handle,
            )
            .expect("null display-transform graph");
        assert_eq!(transformed.version(), 1);
        graph.compile().expect("post-write present source compiles");
        drop(transform);
        device.destroy_texture_view(source_view);
        device.destroy_texture(source);
    }

    #[test]
    fn physical_gpu_transform_matches_cpu_reference_for_supported_pipeline_matrix() {
        let pixels: [[u8; 4]; 6] = [
            [0, 0, 0, 0],
            [1, 5, 64, 51],
            [64, 128, 255, 102],
            [255, 128, 64, 153],
            [240, 32, 128, 204],
            [17, 220, 80, 255],
        ];
        assert_transform_pipeline_matrix(
            rhi::TextureFormat::Rgba8Unorm,
            &pixels.map(|pixel| pixel.map(|value| f32::from(value) / 255.0)),
            bytemuck::cast_slice(&pixels),
            0.75,
        );
    }

    #[test]
    fn physical_gpu_hdr_transform_preserves_highlights_before_negative_exposure() {
        // Direct texture fixtures test the display transform, not scene lighting.
        // All values are exactly representable in binary16. Negative exposure
        // brings highlights into SDR range, exposing any clamp before exposure.
        let pixels: [[f32; 4]; 6] = [
            [0.0, 0.0, 0.0, 0.0],
            [2.0, 4.0, 8.0, 0.25],
            [4.0, 2.0, 1.0, 0.5],
            [8.0, 16.0, 32.0, 0.75],
            [0.125, 0.5, 1.5, 1.0],
            [-2.0, 3.0, 0.25, 0.5],
        ];
        let encoded = pixels.map(|pixel| pixel.map(f32_to_f16_bits));
        let uploaded = encoded.map(|pixel| pixel.map(f16_bits_to_f32));
        assert_eq!(uploaded, pixels, "HDR fixture must be exact in binary16");
        assert_transform_pipeline_matrix(
            rhi::TextureFormat::Rgba16Float,
            &uploaded,
            bytemuck::cast_slice(&encoded),
            -4.0,
        );
    }

    // Reuse the same physical-device gate, graph, compute pipeline, output
    // readback and 3 x 3 x 4 configuration matrix for SDR and HDR sources.
    fn assert_transform_pipeline_matrix(
        source_format: rhi::TextureFormat,
        pixels: &[[f32; 4]; 6],
        upload: &[u8],
        exposure_stops: f32,
    ) {
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
        let input = device.create_texture(&TextureDescriptor::new_2d(
            extent,
            source_format,
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
            upload,
            ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(upload.len() as u32 / extent.height),
                rows_per_image: Some(extent.height),
            },
        );
        let lifetime_count = Arc::new(AtomicUsize::new(0));
        let destroy_allowed = Arc::new(AtomicBool::new(true));
        let transform = DisplayTransformGpu::new(
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
                        exposure_stops,
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
                        TextureDesc::color(source_format)
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
                    let recorded = graph
                        .execute(&*device_handle, &queue)
                        .expect("valid transform graph execution");
                    let actual = queue
                        .read_texture_rgba8(transform.output_texture(), extent.width, extent.height)
                        .expect("display-transform SDR GPU readback must complete");
                    // Same-queue synchronous readback completes the graph's GPU use.
                    recorded.release(&*device_handle);
                    assert_eq!(actual.len(), pixels.len() * 4, "SDR output is RGBA8");
                    let mut distinguishable_hdr_channels = 0;
                    for (index, (source, result)) in
                        pixels.iter().zip(actual.chunks_exact(4)).enumerate()
                    {
                        let linear = [source[0], source[1], source[2]];
                        let expected = config
                            .reference_transform(linear)
                            .expect("finite CPU reference input");
                        let early_clamped = config
                            .reference_transform(linear.map(|value| value.clamp(0.0, 1.0)))
                            .expect("finite early-clamp counterexample");
                        for channel in 0..3 {
                            let actual = f32::from(result[channel]) / 255.0;
                            assert!(
                                (actual - expected[channel]).abs() <= 2.0 / 255.0,
                                "source={source_format:?} exposure={exposure_stops} tone={tone_mapping:?} gamut={display_gamut:?} transfer={transfer_function:?} pixel={index} channel={channel}: GPU={} CPU={}",
                                actual,
                                expected[channel]
                            );
                            if (expected[channel] - early_clamped[channel]).abs() > 0.1 {
                                distinguishable_hdr_channels += 1;
                                assert!(
                                    (actual - early_clamped[channel]).abs() > 0.08,
                                    "HDR highlight was clamped before exposure: tone={tone_mapping:?} gamut={display_gamut:?} transfer={transfer_function:?} pixel={index} channel={channel}"
                                );
                            }
                        }
                        let expected_alpha = (source[3] * 255.0).round() as u8;
                        if source_format == rhi::TextureFormat::Rgba8Unorm {
                            assert_eq!(result[3], expected_alpha, "alpha must pass through");
                        } else {
                            assert!(
                                result[3].abs_diff(expected_alpha) <= 1,
                                "alpha must pass through: pixel={index} GPU={} CPU={expected_alpha}",
                                result[3]
                            );
                        }
                    }
                    if source_format == rhi::TextureFormat::Rgba16Float {
                        assert!(
                            distinguishable_hdr_channels > 0,
                            "every HDR configuration must distinguish early clamping"
                        );
                    }
                }
            }
        }
        drop(transform);
        device.destroy_texture_view(input_view);
        device.destroy_texture(input);
    }

    #[test]
    fn physical_gpu_offscreen_host_graph_transforms_and_reads_back_frame() {
        let (device, queue) = match WgpuInstance::new().request_default_hardware_device() {
            Ok(device_and_queue) => device_and_queue,
            Err(HardwareDeviceError::NoAdapter) => {
                assert!(
                    !require_hardware_gpu(),
                    "PRISM_REQUIRE_GPU=1 but no physical GPU adapter was found"
                );
                eprintln!("SKIPPED offscreen OCIO graph: no adapter");
                return;
            }
            Err(HardwareDeviceError::NonHardware(evidence)) => {
                assert!(
                    !require_hardware_gpu(),
                    "PRISM_REQUIRE_GPU=1 requires a physical GPU; observed {evidence}"
                );
                eprintln!("SKIPPED offscreen OCIO graph: non-hardware adapter: {evidence}");
                return;
            }
            Err(HardwareDeviceError::Device(error)) => {
                panic!("physical GPU was found but device creation failed: {error}");
            }
        };
        let device = Arc::new(device);
        let device_dyn: Arc<dyn RenderDevice + Send + Sync> = device.clone();
        let extent = Extent3d::new_2d(5, 3);
        let target = crate::target::RenderTarget::new(&*device, extent);
        assert_eq!(target.format(), rhi::TextureFormat::Rgba16Float);
        let config = OcioConfig {
            exposure_stops: -3.0,
            ..OcioConfig::aces()
        };
        let mut display = OffscreenDisplayTransform::new(config);
        display
            .ensure(device_dyn, target.view(), extent, 0)
            .expect("valid offscreen display transform");
        let queues = crate::queue::RenderQueues::new();
        let functions = crate::draw::canonical::registry();
        let graph = crate::frame::target_graph_with_display_transform(
            &target, &queues, &functions, None, &display, &queue,
        )
        .expect("offscreen transform pass is appended to the host frame graph");
        let stages = *graph
            .blackboard()
            .get::<crate::frame::FrameGraphStages>()
            .expect("offscreen graph publishes its stage receipt");
        assert_eq!(
            stages.present_passes, 0,
            "headless graph has no window present"
        );
        // HDR and GBuffer attachments are cleared separately because their
        // combined bytes-per-sample exceed the WebGPU color-attachment limit.
        // The transform therefore follows two host clear passes.
        assert_eq!(
            graph.pass_count(),
            stages.scene_passes as usize + 1,
            "host clears then display transform"
        );
        let host_frame = graph
            .execute(&*device, &queue)
            .expect("offscreen host graph executes");

        // The host graph's clear pass is part of the frame. Seed the persistent
        // target after that submission, then exercise the same transform over
        // HDR values. This is a transform/capture fixture, not lighting evidence.
        // The odd extent also exercises padded rows in both readback formats.
        let pixels: [[f32; 4]; 15] = [
            [0.0, 0.0, 0.0, 0.0],
            [2.0, 4.0, 8.0, 0.125],
            [4.0, 2.0, 1.0, 0.25],
            [8.0, 16.0, 32.0, 0.375],
            [0.125, 0.5, 1.5, 0.5],
            [-2.0, 3.0, 0.25, 0.625],
            [0.001, 0.003, 0.01, 0.75],
            [0.18, 0.36, 0.72, 0.875],
            [0.5, 1.0, 2.0, 1.0],
            [32.0, 8.0, 2.0, 0.0625],
            [16.0, 4.0, 1.0, 0.1875],
            [8.0, 2.0, 0.5, 0.3125],
            [4.0, 1.0, 0.25, 0.4375],
            [2.0, 0.5, 0.125, 0.6875],
            [1.0, 0.25, 0.0625, 0.9375],
        ];
        let encoded = pixels.map(|pixel| pixel.map(f32_to_f16_bits));
        let uploaded = encoded.map(|pixel| pixel.map(f16_bits_to_f32));
        queue.write_texture(
            TextureWrite {
                texture: target.texture(),
                mip_level: 0,
                origin: [0, 0, 0],
                size: extent,
            },
            bytemuck::cast_slice(&encoded),
            ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(extent.width * 8),
                rows_per_image: Some(extent.height),
            },
        );
        let linear_capture = queue
            .read_texture_rgba16f(target.texture(), extent.width, extent.height)
            .expect("display-transform HDR seed readback must complete");
        // The synchronous readback also completes the preceding host submission.
        host_frame.release(&*device);
        assert_eq!(linear_capture.as_slice(), uploaded.as_slice());
        assert!(linear_capture.iter().any(|pixel| pixel[2] > 1.0));
        assert!(linear_capture.iter().any(|pixel| pixel[0] < 0.0));
        let mut parity_graph = RenderGraph::new(extent);
        let source_handle = parity_graph.import_texture(
            "offscreen-parity-source",
            TextureDesc::color(target.format()).with_usage(TextureUsages::TEXTURE_BINDING),
            ImportedTexture {
                texture: target.texture(),
                default_view: target.view(),
                entry_state: rhi::TextureState::initial(),
                exit_state: rhi::TextureState::shader_read(rhi::PipelineStages::COMPUTE_SHADER),
                mip_levels: 1,
                array_layers: 1,
            },
        );
        display
            .append(&mut parity_graph, &queue, source_handle)
            .expect("the offscreen transform accepts seeded linear pixels");
        let parity_frame = parity_graph
            .execute(&*device, &queue)
            .expect("the seeded offscreen transform graph executes");
        assert_eq!(
            queue
                .read_texture_rgba16f(target.texture(), extent.width, extent.height)
                .expect("display-transform parity readback must complete"),
            linear_capture,
            "linear capture must remain unexposed and unmodified by the display pass"
        );
        parity_frame.release(&*device);

        let mut frame_output = OffscreenFrameOutput::default();
        frame_output.publish(OffscreenOutputFrame::new(
            display.output_texture().expect("prepared output texture"),
            extent,
            17,
            1,
            0,
        ));
        let frame = frame_output.latest().expect("public frame receipt exists");
        assert_eq!(frame.extent(), extent);
        assert_eq!(frame.frame_index(), 17);
        assert_eq!(frame.device_generation(), 0);
        let output = queue
            .read_texture_rgba8(frame.texture(), extent.width, extent.height)
            .expect("published display-transform output readback must complete");
        assert_eq!(output.len(), pixels.len() * 4, "SDR output is RGBA8");
        // The backend's RGBA8 readback helper is only valid for the display
        // output, not the HDR scene target. Compare to the uploaded half values.
        let mut distinguished = [0_usize; 5];
        for (index, (source, actual)) in uploaded.iter().zip(output.chunks_exact(4)).enumerate() {
            let linear = [source[0], source[1], source[2]];
            let expected = config
                .reference_transform(linear)
                .expect("finite frame target input");
            let counterexamples = [
                (
                    "early clamp",
                    config.reference_transform(linear.map(|v| v.clamp(0.0, 1.0))),
                ),
                (
                    "missing exposure",
                    OcioConfig {
                        exposure_stops: 0.0,
                        ..config
                    }
                    .reference_transform(linear),
                ),
                (
                    "double exposure",
                    OcioConfig {
                        exposure_stops: config.exposure_stops * 2.0,
                        ..config
                    }
                    .reference_transform(linear),
                ),
                (
                    "double display transform",
                    config.reference_transform(expected),
                ),
                (
                    "double transfer",
                    OcioConfig::srgb().reference_transform(expected),
                ),
            ];
            for channel in 0..3 {
                let actual = f32::from(actual[channel]) / 255.0;
                assert!(
                    (actual - expected[channel]).abs() <= 2.0 / 255.0,
                    "pixel={index} channel={channel}: GPU={actual} CPU={}",
                    expected[channel]
                );
                for (case, (label, result)) in counterexamples.iter().enumerate() {
                    let wrong = result.as_ref().expect("finite counterexample input")[channel];
                    if (expected[channel] - wrong).abs() > 0.05 {
                        distinguished[case] += 1;
                        assert!(
                            (actual - wrong).abs() > 0.03,
                            "{label}: pixel={index} channel={channel} GPU={actual} wrong={wrong}"
                        );
                    }
                }
            }
            let expected_alpha = (source[3] * 255.0).round() as u8;
            assert!(
                actual[3].abs_diff(expected_alpha) <= 1,
                "alpha must pass through"
            );
        }
        assert!(
            distinguished.iter().all(|count| *count > 0),
            "the HDR fixture must distinguish each ordering/duplication bug: {distinguished:?}"
        );
        drop(display);
        target.destroy(&*device);
    }
}
