//! Real-device water-surface rasterisation.
//!
//! [`render`] drives a complete `wgpu` graphics pipeline on a headless device:
//! it builds the `@vertex`/`@fragment` program from
//! `shaders/water_surface.wgsl`, synthesises a Gerstner-displaced surface grid
//! procedurally from the vertex index, rasterises it into an offscreen
//! `Rgba8Unorm` colour target with a `Depth32Float` depth buffer, and copies the
//! frame back to a tightly packed [`RenderedFrame`]. That frame can be written
//! to a viewable image with [`RenderedFrame::save_png`].
//!
//! Unlike the compute twins elsewhere in the workspace, this module calls
//! [`wgpu::Device::create_render_pipeline`] and actually draws, so a passing
//! headless run is direct evidence the water subsystem can turn solved surface
//! parameters into visible pixels — not merely compute scalar fields.

use alloc::vec::Vec;
use std::io;
use std::path::Path;

use bytemuck::{Pod, Zeroable};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
    BindingType, BlendState, BufferBindingType, BufferDescriptor, BufferUsages, Color,
    ColorTargetState, ColorWrites, CommandEncoderDescriptor, CompareFunction, DepthBiasState,
    DepthStencilState, Extent3d, FragmentState, FrontFace, LoadOp, MapMode, MultisampleState,
    Operations, PipelineCompilationOptions, PipelineLayoutDescriptor, PolygonMode, PrimitiveState,
    PrimitiveTopology, RenderPassColorAttachment, RenderPassDepthStencilAttachment,
    RenderPassDescriptor, RenderPipelineDescriptor, ShaderModuleDescriptor, ShaderSource,
    ShaderStages, StencilState, StoreOp, TexelCopyBufferInfo, TexelCopyBufferLayout,
    TexelCopyTextureInfo, TextureAspect, TextureDescriptor, TextureDimension, TextureFormat,
    TextureUsages, TextureViewDescriptor, VertexState,
};

use prism_render_architecture::water::spectrum::dispersion;
use prism_render_architecture::water::WATER_ARCHITECTURE_VERSION;

use crate::camera::Camera;
use crate::context::GpuContext;
use crate::png;

/// Row alignment, in bytes, required for `copy_texture_to_buffer` destinations.
const ROW_ALIGNMENT: u32 = 256;

/// The colour format of the offscreen render target and the read-back image.
const COLOR_FORMAT: TextureFormat = TextureFormat::Rgba8Unorm;

/// The depth format used for hidden-surface removal.
const DEPTH_FORMAT: TextureFormat = TextureFormat::Depth32Float;

/// A described water-surface scene ready to rasterise.
#[derive(Clone, Copy, Debug)]
pub struct WaterSurfaceScene {
    /// Output image width in pixels.
    pub width: u32,
    /// Output image height in pixels.
    pub height: u32,
    /// The camera the surface is viewed through.
    pub camera: Camera,
    /// Animation time in seconds (advances the Gerstner phases).
    pub time: f32,
    /// Base wave amplitude in world units.
    pub amplitude: f32,
    /// Side length of the square surface patch in world units.
    pub extent: f32,
    /// Number of grid cells along each axis of the surface patch.
    pub grid_resolution: u32,
    /// Base Gerstner wavelength in world units.
    pub wavelength: f32,
    /// Multiplier on the deep-water phase speed `sqrt(g * k)`.
    pub phase_speed: f32,
    /// Gerstner horizontal pinch (`0` = round swell, `1` = sharp crests).
    pub choppiness: f32,
    /// World-space direction toward the sun (need not be normalised).
    pub sun_dir: [f32; 3],
    /// Linear deep-water colour (`RGB`, each in `[0, 1]`).
    pub deep_color: [f32; 3],
    /// Linear sky/horizon colour reflected at grazing angles (`RGB`).
    pub sky_color: [f32; 3],
    /// Background clear colour behind the surface (`RGBA`, each in `[0, 1]`).
    pub background: [f64; 4],
}

impl WaterSurfaceScene {
    /// Returns a pleasant default ocean scene at the requested resolution.
    ///
    /// The camera sits above the water looking across it so the Gerstner swell,
    /// Fresnel rim and sun glint are all visible in one frame.
    #[must_use]
    pub fn preset(width: u32, height: u32) -> WaterSurfaceScene {
        let aspect = if height == 0 {
            1.0
        } else {
            width as f32 / height as f32
        };
        WaterSurfaceScene {
            width,
            height,
            camera: Camera {
                eye: [0.0, 7.0, 16.0],
                target: [0.0, 0.0, -2.0],
                up: [0.0, 1.0, 0.0],
                fov_y: 0.9,
                aspect,
                near: 0.1,
                far: 400.0,
            },
            time: 1.7,
            amplitude: 0.55,
            extent: 60.0,
            grid_resolution: 256,
            wavelength: 11.0,
            phase_speed: 1.0,
            choppiness: 0.75,
            sun_dir: [0.35, 0.55, 0.75],
            deep_color: [0.012, 0.11, 0.20],
            sky_color: [0.52, 0.72, 0.92],
            background: [0.46, 0.66, 0.90, 1.0],
        }
    }
}

/// A rendered frame read back to host memory as tightly packed `RGBA8`.
#[derive(Clone, Debug)]
pub struct RenderedFrame {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Row-major `RGBA8` pixels, top row first (`width * height * 4` bytes).
    pub rgba: Vec<u8>,
}

impl RenderedFrame {
    /// Encodes the frame as a `PNG` and writes it to `path`.
    ///
    /// # Errors
    ///
    /// Returns any `io` error raised while creating or writing the file.
    pub fn save_png(&self, path: &Path) -> io::Result<()> {
        let bytes = png::encode_rgba8(self.width, self.height, &self.rgba);
        std::fs::write(path, bytes)
    }

    /// Counts the number of distinct `RGBA` pixel values in the frame.
    ///
    /// A flat clear produces `1`; any rasterised, shaded geometry produces many.
    /// Tests use this as device-independent evidence that the pipeline actually
    /// drew and shaded the surface rather than clearing to a single colour.
    #[must_use]
    pub fn distinct_colors(&self) -> usize {
        let mut seen = Vec::new();
        for px in self.rgba.chunks_exact(4) {
            let key = u32::from_le_bytes([px[0], px[1], px[2], px[3]]);
            if let Err(idx) = seen.binary_search(&key) {
                seen.insert(idx, key);
            }
        }
        seen.len()
    }
}

/// Rasterises `scene` on `ctx` and reads the frame back to host memory.
///
/// # Panics
///
/// Panics if the device fails to map the read-back buffer, which indicates a
/// lost or broken adapter rather than a recoverable per-frame condition.
#[must_use]
pub fn render(ctx: &GpuContext, scene: &WaterSurfaceScene) -> RenderedFrame {
    let device = ctx.device();
    let queue = ctx.queue();

    let width = scene.width.max(1);
    let height = scene.height.max(1);

    let uniforms = Uniforms::from_scene(scene, width, height);
    let uniform_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("water-surface-uniforms"),
        contents: bytemuck::bytes_of(&uniforms),
        usage: BufferUsages::UNIFORM,
    });

    let module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("water-surface-shader"),
        source: ShaderSource::Wgsl(include_str!("../shaders/water_surface.wgsl").into()),
    });

    let bind_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("water-surface-bind-layout"),
        entries: &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::VERTEX_FRAGMENT,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });

    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("water-surface-pipeline-layout"),
        bind_group_layouts: &[Some(&bind_layout)],
        immediate_size: 0,
    });

    let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
        label: Some("water-surface-pipeline"),
        layout: Some(&pipeline_layout),
        vertex: VertexState {
            module: &module,
            entry_point: Some("vs_main"),
            compilation_options: PipelineCompilationOptions::default(),
            buffers: &[],
        },
        primitive: PrimitiveState {
            topology: PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: FrontFace::Ccw,
            cull_mode: None,
            unclipped_depth: false,
            polygon_mode: PolygonMode::Fill,
            conservative: false,
        },
        depth_stencil: Some(DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: Some(true),
            depth_compare: Some(CompareFunction::Less),
            stencil: StencilState::default(),
            bias: DepthBiasState::default(),
        }),
        multisample: MultisampleState {
            count: 1,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        fragment: Some(FragmentState {
            module: &module,
            entry_point: Some("fs_main"),
            compilation_options: PipelineCompilationOptions::default(),
            targets: &[Some(ColorTargetState {
                format: COLOR_FORMAT,
                blend: Some(BlendState::REPLACE),
                write_mask: ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    });

    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("water-surface-bind-group"),
        layout: &bind_layout,
        entries: &[BindGroupEntry {
            binding: 0,
            resource: uniform_buf.as_entire_binding(),
        }],
    });

    let extent = Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };
    let color_tex = device.create_texture(&TextureDescriptor {
        label: Some("water-surface-color"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: COLOR_FORMAT,
        usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let color_view = color_tex.create_view(&TextureViewDescriptor::default());

    let depth_tex = device.create_texture(&TextureDescriptor {
        label: Some("water-surface-depth"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let depth_view = depth_tex.create_view(&TextureViewDescriptor::default());

    let bytes_per_row = align_up(width * 4, ROW_ALIGNMENT);
    let readback = device.create_buffer(&BufferDescriptor {
        label: Some("water-surface-readback"),
        size: u64::from(bytes_per_row) * u64::from(height),
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let vertex_count = 6 * scene.grid_resolution.max(1) * scene.grid_resolution.max(1);

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("water-surface-encoder"),
    });
    {
        let clear = scene.background;
        let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("water-surface-pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: &color_view,
                depth_slice: None,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Clear(Color {
                        r: clear[0],
                        g: clear[1],
                        b: clear[2],
                        a: clear[3],
                    }),
                    store: StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                view: &depth_view,
                depth_ops: Some(Operations {
                    load: LoadOp::Clear(1.0),
                    store: StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            ..Default::default()
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.draw(0..vertex_count, 0..1);
    }

    encoder.copy_texture_to_buffer(
        TexelCopyTextureInfo {
            texture: &color_tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: TextureAspect::All,
        },
        TexelCopyBufferInfo {
            buffer: &readback,
            layout: TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(height),
            },
        },
        extent,
    );

    queue.submit([encoder.finish()]);

    readback.slice(..).map_async(MapMode::Read, |_| {});
    ctx.wait();

    let rgba = unpack_rows(&readback, width, height, bytes_per_row);
    readback.unmap();

    RenderedFrame {
        width,
        height,
        rgba,
    }
}

/// Copies a mapped, row-padded staging buffer into a tight `RGBA8` vector.
fn unpack_rows(readback: &wgpu::Buffer, width: u32, height: u32, bytes_per_row: u32) -> Vec<u8> {
    let view = readback
        .slice(..)
        .get_mapped_range()
        .expect("read-back buffer should map after the device completes the copy");
    let row_bytes = (width * 4) as usize;
    let padded = bytes_per_row as usize;
    let mut out = Vec::with_capacity(row_bytes * height as usize);
    for row in 0..height as usize {
        let start = row * padded;
        out.extend_from_slice(&view[start..start + row_bytes]);
    }
    drop(view);
    out
}

/// Rounds `value` up to the next multiple of `align` (a power of two).
fn align_up(value: u32, align: u32) -> u32 {
    value.div_ceil(align) * align
}

/// The `std140`-compatible uniform block the shader reads from `@binding(0)`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
    camera_pos: [f32; 4],
    sun_dir: [f32; 4],
    deep_color: [f32; 4],
    sky_color: [f32; 4],
    params: [f32; 4],
    wave: [f32; 4],
    /// Per-wave angular frequency from the shared dispersion relation.
    ///
    /// `xyz` carry the three Gerstner waves' `omega = sqrt(g*k) * phase_speed`
    /// evaluated on the host by `prism_render_architecture`, so the surface
    /// animates with the same deep-water physics the solver uses rather than a
    /// re-derived shader approximation. `w` carries the water-architecture
    /// version the frame was rendered against.
    wave_omega: [f32; 4],
}

impl Uniforms {
    fn from_scene(scene: &WaterSurfaceScene, width: u32, height: u32) -> Uniforms {
        let mut camera = scene.camera;
        camera.aspect = if height == 0 {
            1.0
        } else {
            width as f32 / height as f32
        };
        Uniforms {
            view_proj: camera.view_proj(),
            camera_pos: [camera.eye[0], camera.eye[1], camera.eye[2], 1.0],
            sun_dir: [scene.sun_dir[0], scene.sun_dir[1], scene.sun_dir[2], 0.0],
            deep_color: [
                scene.deep_color[0],
                scene.deep_color[1],
                scene.deep_color[2],
                1.0,
            ],
            sky_color: [
                scene.sky_color[0],
                scene.sky_color[1],
                scene.sky_color[2],
                1.0,
            ],
            params: [
                scene.time,
                scene.amplitude,
                scene.extent,
                scene.grid_resolution.max(1) as f32,
            ],
            wave: [scene.wavelength, scene.phase_speed, scene.choppiness, 0.0],
            wave_omega: wave_omega(scene.wavelength, scene.phase_speed),
        }
    }
}

/// Per-wave angular frequency for the three fixed Gerstner waves.
///
/// Mirrors the shader's wavelength ladder (full, `0.55`, `0.3` of the base) and
/// evaluates each wave's deep-water frequency through the shared
/// [`dispersion`] relation, so the host and `GPU` agree on the physics. The
/// fourth lane records [`WATER_ARCHITECTURE_VERSION`] as evidence of which
/// water-engine contract the frame targets.
fn wave_omega(base_wavelength: f32, phase_speed: f32) -> [f32; 4] {
    const SCALES: [f32; 3] = [1.0, 0.55, 0.3];
    let mut out = [0.0_f32; 4];
    for (lane, scale) in SCALES.iter().enumerate() {
        let wavelength = (base_wavelength * scale).max(1.0e-3);
        let k = core::f32::consts::TAU / wavelength;
        out[lane] = dispersion(k) * phase_speed;
    }
    out[3] = WATER_ARCHITECTURE_VERSION as f32;
    out
}
