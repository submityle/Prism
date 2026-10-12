//! Real-device `Tessendorf` `FFT` ocean rasterisation.
//!
//! Where [`crate::surface`] rasterises a procedural three-wave Gerstner grid,
//! this module renders a true spectral ocean: it drives the already-verified
//! `prism_render_architecture` spectral chain on the host
//! ([`build_initial_spectrum`] then [`synthesize_surface`]) to synthesise an
//! inverse-`FFT` height-and-choppiness field, lifts that field into a real
//! world-space vertex buffer with analytic normals and `Jacobian`-fold foam
//! weights, and draws it through a `wgpu` graphics pipeline into an offscreen
//! frame. This is the same statistical ocean model shipping products such as
//! `Tessendorf`-based `WaveWorks`, `Crest` and `UE5` Water build on, now turned
//! into visible pixels rather than a compute-only parity twin.
//!
//! Scope stays honest: this is a single offline frame of an ocean *surface*. It
//! does not yet drive cascades, the Bevy render world, `Lumen`-style global
//! illumination or the ray-traced reflections and caustics that
//! `prism_render_scene` wires at the descriptor level.

use alloc::vec::Vec;

use bytemuck::{Pod, Zeroable};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    vertex_attr_array, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BlendState, BufferBindingType, BufferUsages, Color,
    ColorTargetState, ColorWrites, CompareFunction, DepthBiasState, DepthStencilState, Face,
    FragmentState, FrontFace, IndexFormat, LoadOp, MultisampleState, Operations,
    PipelineCompilationOptions, PipelineLayoutDescriptor, PolygonMode, PrimitiveState,
    PrimitiveTopology, RenderPassColorAttachment, RenderPassDepthStencilAttachment,
    RenderPassDescriptor, RenderPipelineDescriptor, ShaderModuleDescriptor, ShaderSource,
    ShaderStages, StencilState, StoreOp, VertexBufferLayout, VertexState, VertexStepMode,
};

use prism_render_architecture::water::initial_spectrum::build_initial_spectrum;
use prism_render_architecture::water::spectrum::{SpectrumKind, SpectrumParams};
use prism_render_architecture::water::synthesis::{synthesize_surface, OceanSurface};
use prism_render_architecture::water::Vec2;
use prism_render_architecture::water::WATER_ARCHITECTURE_VERSION;

use crate::camera::Camera;
use crate::context::GpuContext;
use crate::frame::{self, RenderedFrame, COLOR_FORMAT, DEPTH_FORMAT};

/// Smallest squared length treated as non-zero when normalising a vector.
const EPS_LEN_SQ: f32 = 1.0e-12;

/// Which statistical ocean spectrum the scene is synthesised from.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OceanSpectrum {
    /// `Phillips` spectrum: the classic directional wind-driven sea.
    Phillips,
    /// `JONSWAP` spectrum: a fetch-limited peak-enhanced sea.
    Jonswap,
    /// `Pierson-Moskowitz` spectrum: a fully developed open-ocean sea.
    PiersonMoskowitz,
}

impl OceanSpectrum {
    /// Maps to the architecture's [`SpectrumKind`].
    #[must_use]
    fn kind(self) -> SpectrumKind {
        match self {
            OceanSpectrum::Phillips => SpectrumKind::Phillips,
            OceanSpectrum::Jonswap => SpectrumKind::Jonswap,
            OceanSpectrum::PiersonMoskowitz => SpectrumKind::PiersonMoskowitz,
        }
    }
}

/// A described spectral-ocean scene ready to rasterise.
#[derive(Clone, Copy, Debug)]
pub struct OceanScene {
    /// Output image width in pixels.
    pub width: u32,
    /// Output image height in pixels.
    pub height: u32,
    /// The camera the ocean is viewed through.
    pub camera: Camera,
    /// Animation time in seconds (advances the spectral phases).
    pub time: f32,
    /// Grid resolution `N`; must be a power of two or the surface is empty.
    pub resolution: u32,
    /// Tiled spectral patch size `L` in metres.
    pub patch_size: f32,
    /// Statistical spectrum shape to synthesise.
    pub spectrum: OceanSpectrum,
    /// Downwind direction (need not be normalised; zero falls back to `+x`).
    pub wind_dir: [f32; 2],
    /// Wind speed `V` in metres per second.
    pub wind_speed: f32,
    /// Overall spectral energy scale.
    pub amplitude: f32,
    /// `JONSWAP` peak-enhancement factor `gamma` (`1.0` reduces to `PM`).
    pub peak_enhancement: f32,
    /// Shortest wavelength (m) kept before high-frequency suppression.
    pub min_wavelength: f32,
    /// Directional-spread exponent; higher aligns energy tighter to the wind.
    pub directional_exponent: u32,
    /// Random seed selecting the deterministic spectral draw.
    pub seed: u32,
    /// `Tessendorf` horizontal choppiness blend (`0` = height only, `1` = full).
    pub choppiness: f32,
    /// Multiplier on the synthesised vertical displacement.
    pub height_scale: f32,
    /// Uniform world-space scale applied to the whole patch.
    pub world_scale: f32,
    /// Foam strength multiplier applied to the `Jacobian`-fold weight.
    pub foam_strength: f32,
    /// World-space direction toward the sun (need not be normalised).
    pub sun_dir: [f32; 3],
    /// Linear deep-water colour (`RGB`, each in `[0, 1]`).
    pub deep_color: [f32; 3],
    /// Linear sky/horizon colour reflected at grazing angles (`RGB`).
    pub sky_color: [f32; 3],
    /// Linear foam/whitecap colour (`RGB`, each in `[0, 1]`).
    pub foam_color: [f32; 3],
    /// Background clear colour behind the ocean (`RGBA`, each in `[0, 1]`).
    pub background: [f64; 4],
}

impl OceanScene {
    /// Returns a pleasant default open-ocean scene at the requested resolution.
    ///
    /// The camera sits above the water looking across it so the spectral swell,
    /// `Fresnel` rim, sun glint and `Jacobian` whitecaps are all visible.
    #[must_use]
    pub fn preset(width: u32, height: u32) -> OceanScene {
        let aspect = if height == 0 {
            1.0
        } else {
            width as f32 / height as f32
        };
        OceanScene {
            width,
            height,
            camera: Camera {
                eye: [0.0, 9.0, 22.0],
                target: [0.0, 0.0, -4.0],
                up: [0.0, 1.0, 0.0],
                fov_y: 0.9,
                aspect,
                near: 0.1,
                far: 600.0,
            },
            time: 3.0,
            resolution: 128,
            patch_size: 64.0,
            spectrum: OceanSpectrum::Phillips,
            wind_dir: [1.0, 0.35],
            wind_speed: 11.0,
            amplitude: 2.6,
            peak_enhancement: 3.3,
            min_wavelength: 0.2,
            directional_exponent: 2,
            seed: 0x5eed_1234,
            choppiness: 0.85,
            height_scale: 1.0,
            world_scale: 0.9,
            foam_strength: 1.6,
            sun_dir: [0.35, 0.55, 0.75],
            deep_color: [0.010, 0.09, 0.17],
            sky_color: [0.52, 0.72, 0.92],
            foam_color: [0.92, 0.95, 0.98],
            background: [0.46, 0.66, 0.90, 1.0],
        }
    }

    /// Builds the architecture spectrum parameters this scene describes.
    #[must_use]
    fn spectrum_params(&self) -> SpectrumParams {
        let dir = Vec2::new(self.wind_dir[0], self.wind_dir[1]);
        let unit = dir.normalize_or_zero();
        let heading = if unit.length_squared() < EPS_LEN_SQ {
            Vec2::new(1.0, 0.0)
        } else {
            unit
        };
        SpectrumParams {
            kind: self.spectrum.kind(),
            wind: heading.scale(self.wind_speed),
            amplitude: self.amplitude,
            peak_enhancement: self.peak_enhancement,
            min_wavelength: self.min_wavelength,
            directional_exponent: self.directional_exponent,
        }
    }
}

/// A single ocean-surface vertex: world position, normal and foam weight.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
    foam: f32,
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
    foam_color: [f32; 4],
}

/// Rasterises `scene` on `ctx` and reads the frame back to host memory.
///
/// Synthesises the spectral ocean on the host, builds a world-space mesh with
/// analytic normals and `Jacobian`-fold foam, and draws it. A degenerate scene
/// (non-power-of-two `resolution`, non-positive `patch_size`, or a calm sea)
/// yields an honest cleared frame rather than a fabricated surface.
///
/// # Panics
///
/// Panics if the device fails to map the read-back buffer, which indicates a
/// lost or broken adapter rather than a recoverable per-frame condition.
#[must_use]
pub fn render_ocean(ctx: &GpuContext, scene: &OceanScene) -> RenderedFrame {
    let width = scene.width.max(1);
    let height = scene.height.max(1);

    let field = build_initial_spectrum(
        scene.resolution,
        scene.patch_size,
        scene.spectrum_params(),
        scene.seed,
    );
    let surface = synthesize_surface(&field, scene.time, scene.choppiness);

    let (vertices, indices) = build_mesh(&surface, scene);

    let device = ctx.device();

    let uniforms = Uniforms::from_scene(scene, width, height);
    let uniform_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("ocean-uniforms"),
        contents: bytemuck::bytes_of(&uniforms),
        usage: BufferUsages::UNIFORM,
    });

    let bind_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("ocean-bind-layout"),
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
    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("ocean-bind-group"),
        layout: &bind_layout,
        entries: &[BindGroupEntry {
            binding: 0,
            resource: uniform_buf.as_entire_binding(),
        }],
    });

    let module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("ocean-shader"),
        source: ShaderSource::Wgsl(include_str!("../shaders/ocean_surface.wgsl").into()),
    });

    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("ocean-pipeline-layout"),
        bind_group_layouts: &[Some(&bind_layout)],
        immediate_size: 0,
    });

    let vertex_attrs = vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32];
    let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
        label: Some("ocean-pipeline"),
        layout: Some(&pipeline_layout),
        vertex: VertexState {
            module: &module,
            entry_point: Some("vs_main"),
            compilation_options: PipelineCompilationOptions::default(),
            buffers: &[Some(VertexBufferLayout {
                array_stride: size_of::<Vertex>() as u64,
                step_mode: VertexStepMode::Vertex,
                attributes: &vertex_attrs,
            })],
        },
        primitive: PrimitiveState {
            topology: PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: FrontFace::Ccw,
            cull_mode: Some(Face::Back),
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

    let vertex_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("ocean-vertices"),
        contents: bytemuck::cast_slice(&vertices),
        usage: BufferUsages::VERTEX,
    });
    let index_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("ocean-indices"),
        contents: bytemuck::cast_slice(&indices),
        usage: BufferUsages::INDEX,
    });

    let index_count = indices.len() as u32;
    let clear = scene.background;

    frame::render_offscreen(ctx, width, height, |encoder, color_view, depth_view| {
        let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("ocean-pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: color_view,
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
                view: depth_view,
                depth_ops: Some(Operations {
                    load: LoadOp::Clear(1.0),
                    store: StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            ..Default::default()
        });
        if index_count > 0 {
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.set_vertex_buffer(0, vertex_buf.slice(..));
            pass.set_index_buffer(index_buf.slice(..), IndexFormat::Uint32);
            pass.draw_indexed(0..index_count, 0, 0..1);
        }
    })
}

impl Uniforms {
    fn from_scene(scene: &OceanScene, width: u32, height: u32) -> Uniforms {
        let mut camera = scene.camera;
        camera.aspect = if height == 0 {
            1.0
        } else {
            width as f32 / height as f32
        };
        Uniforms {
            view_proj: camera.view_proj(),
            camera_pos: [
                camera.eye[0],
                camera.eye[1],
                camera.eye[2],
                WATER_ARCHITECTURE_VERSION as f32,
            ],
            sun_dir: [
                scene.sun_dir[0],
                scene.sun_dir[1],
                scene.sun_dir[2],
                scene.foam_strength,
            ],
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
            foam_color: [
                scene.foam_color[0],
                scene.foam_color[1],
                scene.foam_color[2],
                1.0,
            ],
        }
    }
}

/// Lifts a synthesised [`OceanSurface`] into a world-space triangle mesh.
///
/// Returns `(vertices, indices)`. Positions are the displaced grid points,
/// normals are central-difference surface normals, and the foam weight is the
/// classic `Tessendorf` folding criterion (where the horizontal-displacement
/// `Jacobian` compresses below one, whitecaps form). An empty or
/// non-power-of-two surface yields empty buffers.
fn build_mesh(surface: &OceanSurface, scene: &OceanScene) -> (Vec<Vertex>, Vec<u32>) {
    let n = surface.resolution as usize;
    if n < 2 || surface.height.len() < n * n {
        return (Vec::new(), Vec::new());
    }

    let patch = surface.patch_size;
    let inv_n = 1.0 / n as f32;
    let world = scene.world_scale;
    let height_scale = scene.height_scale;

    // Displaced grid points in patch space (pre world-scale), used for both the
    // emitted positions and the central-difference normals and foam.
    let mut disp = Vec::with_capacity(n * n);
    for i in 0..n {
        let cz = (i as f32 * inv_n - 0.5) * patch;
        for j in 0..n {
            let idx = i * n + j;
            let cx = (j as f32 * inv_n - 0.5) * patch;
            let px = cx + surface.displacement_x[idx];
            let pz = cz + surface.displacement_z[idx];
            let py = surface.height[idx] * height_scale;
            disp.push([px, py, pz]);
        }
    }

    let cell = patch * inv_n;
    let two_cell = 2.0 * cell;
    let mut vertices = Vec::with_capacity(n * n);
    for i in 0..n {
        let ip = (i + 1) % n;
        let im = (i + n - 1) % n;
        for j in 0..n {
            let idx = i * n + j;
            let jp = (j + 1) % n;
            let jm = (j + n - 1) % n;

            // Central-difference tangents over the displaced surface; the +y
            // normal is their cross product. The patch tiles, so neighbours
            // wrap seamlessly.
            let right = disp[i * n + jp];
            let left = disp[i * n + jm];
            let front = disp[ip * n + j];
            let back = disp[im * n + j];
            let tangent_x = sub3(right, left);
            let tangent_z = sub3(front, back);
            let normal = normalize3(cross3(tangent_z, tangent_x));

            // Tessendorf whitecap criterion: the horizontal-displacement
            // Jacobian folds (drops below one) where crests pinch sharply.
            let ddx_dx = (surface.displacement_x[i * n + jp] - surface.displacement_x[i * n + jm])
                / two_cell;
            let ddx_dz = (surface.displacement_x[ip * n + j] - surface.displacement_x[im * n + j])
                / two_cell;
            let ddz_dx = (surface.displacement_z[i * n + jp] - surface.displacement_z[i * n + jm])
                / two_cell;
            let ddz_dz = (surface.displacement_z[ip * n + j] - surface.displacement_z[im * n + j])
                / two_cell;
            let jacobian = (1.0 + ddx_dx) * (1.0 + ddz_dz) - ddx_dz * ddz_dx;
            let foam = (1.0 - jacobian).clamp(0.0, 1.0);

            let p = disp[idx];
            vertices.push(Vertex {
                position: [p[0] * world, p[1] * world, p[2] * world],
                normal,
                foam,
            });
        }
    }

    let mut indices = Vec::with_capacity((n - 1) * (n - 1) * 6);
    for i in 0..n - 1 {
        for j in 0..n - 1 {
            let a = (i * n + j) as u32;
            let b = (i * n + j + 1) as u32;
            let c = ((i + 1) * n + j) as u32;
            let d = ((i + 1) * n + j + 1) as u32;
            // Counter-clockwise seen from above (+y), matching back-face cull.
            indices.extend_from_slice(&[a, c, b, b, c, d]);
        }
    }

    (vertices, indices)
}

/// Component-wise difference `a - b`.
fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Right-handed cross product `a x b`.
fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Normalises `v`, falling back to a flat `+y` normal when degenerate.
fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let len_sq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    if len_sq < EPS_LEN_SQ {
        return [0.0, 1.0, 0.0];
    }
    let inv = 1.0 / len_sq.sqrt();
    [v[0] * inv, v[1] * inv, v[2] * inv]
}
