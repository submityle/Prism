//! Real offscreen output tests of the shipped water WESL entry points.
//!
//! These tests draw the production vertex/fragment stages, dispatch the
//! production render-FX and resident composition/export stages, and map their
//! actual outputs. The eight legacy cases additionally execute a byte-frozen
//! 8400fe972^ shader graph in an isolated cache, so shared PBR/NPR regressions
//! cannot supply their own expected pixels. This is a traceable executable
//! reference, not a previously recorded GPU capture. The shallow
//! and shore weight tests exercise the current ABI (authored shallow bias and
//! `normal_foam.w`), not an absent shoreline-distance/depth-texture binding.
//! Reactive is observable through the production overlay control, not a third
//! MRT or a claim about the complete Scene/TAA frame.
//!
//! No suitable hardware is an explicit skip by default. Setting
//! `PRISM_REQUIRE_GPU=1` makes every adapter/capability/device skip a failure.

use bevy_asset::{uuid::Uuid, AssetId};
use bevy_platform::future::block_on;
use bevy_shader::{Shader, ShaderCache, ShaderCacheError, ShaderCacheSource, ValidateShader};
use std::sync::OnceLock;
use wgpu::util::{BufferInitDescriptor, DeviceExt, TextureDataOrder};

#[path = "fixtures/water_legacy_41a31cee/mod.rs"]
mod legacy;

const EDGE: u32 = 16;
const PIXELS: usize = (EDGE * EDGE) as usize;
const ROW: u32 = 256;
const VOLUME_DEPTH: u32 = 16;
const IDENTITY: [[f32; 4]; 4] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

fn keep_wgsl(
    _: &(),
    source: ShaderCacheSource,
    _: &ValidateShader,
) -> Result<String, ShaderCacheError> {
    match source {
        ShaderCacheSource::Wgsl(source) => Ok(source),
        ShaderCacheSource::SpirV(_) => unreachable!("production water uses WESL"),
    }
}

fn production_wgsl(raster: bool) -> String {
    let mut cache = ShaderCache::new((), keep_wgsl);
    let sources = [
        ("lighting", include_str!("../src/shaders/lighting.wesl")),
        ("tangent", include_str!("../src/shaders/tangent.wesl")),
        (
            "specular_aa",
            include_str!("../src/shaders/specular_aa.wesl"),
        ),
        ("thin_film", include_str!("../src/shaders/thin_film.wesl")),
        ("glint", include_str!("../src/shaders/glint.wesl")),
        ("brdf", include_str!("../src/shaders/brdf.wesl")),
        ("water", include_str!("../src/shaders/water.wesl")),
        (
            "water_surface_raster",
            include_str!("../src/shaders/water_surface_raster.wesl"),
        ),
        (
            "water_render_fx",
            include_str!("../src/shaders/water_render_fx.wesl"),
        ),
    ];
    let mut ids = Vec::new();
    for (i, (name, source)) in sources.into_iter().enumerate() {
        let id = AssetId::Uuid {
            uuid: Uuid::from_u128(0x5052_4953_4d5f_5741_5445_5247_5000_0000 + i as u128),
        };
        cache.set_shader(
            id,
            Shader::from_wesl(
                source,
                format!("embedded://prism_render_scene/shaders/{name}.wesl"),
            ),
        );
        ids.push(id);
    }
    (*cache
        .get(0, ids[if raster { 7 } else { 8 }], &[])
        .expect("unmodified production water WESL must compile"))
    .clone()
}

fn unavailable(reason: impl std::fmt::Display) -> Option<Gpu> {
    assert!(
        std::env::var_os("PRISM_REQUIRE_GPU").is_none(),
        "PRISM_REQUIRE_GPU: water output GPU gate failed: {reason}"
    );
    eprintln!("water output GPU SKIP: {reason}");
    None
}

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    layouts: Vec<wgpu::BindGroupLayout>,
    programs: [Programs; 2],
}

struct Programs {
    raster: [wgpu::RenderPipeline; 3],
    compose: wgpu::ComputePipeline,
    export: wgpu::ComputePipeline,
    caustics: wgpu::ComputePipeline,
    underwater: wgpu::ComputePipeline,
}

fn gpu() -> Option<&'static Gpu> {
    static GPU: OnceLock<Option<Gpu>> = OnceLock::new();
    GPU.get_or_init(Gpu::new).as_ref()
}

fn buffer_type(storage: Option<bool>) -> wgpu::BindingType {
    wgpu::BindingType::Buffer {
        ty: storage.map_or(wgpu::BufferBindingType::Uniform, |read_only| {
            wgpu::BufferBindingType::Storage { read_only }
        }),
        has_dynamic_offset: false,
        min_binding_size: None,
    }
}

fn texture_type(dimension: wgpu::TextureViewDimension, filterable: bool) -> wgpu::BindingType {
    wgpu::BindingType::Texture {
        sample_type: wgpu::TextureSampleType::Float { filterable },
        view_dimension: dimension,
        multisampled: false,
    }
}

fn layout(
    device: &wgpu::Device,
    entries: &[(wgpu::ShaderStages, wgpu::BindingType)],
) -> wgpu::BindGroupLayout {
    let entries: Vec<_> = entries
        .iter()
        .enumerate()
        .map(|(binding, (visibility, ty))| wgpu::BindGroupLayoutEntry {
            binding: binding as u32,
            visibility: *visibility,
            ty: *ty,
            count: None,
        })
        .collect();
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("production water binding ABI"),
        entries: &entries,
    })
}

fn group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    resources: &[wgpu::BindingResource<'_>],
) -> wgpu::BindGroup {
    let entries: Vec<_> = resources
        .iter()
        .enumerate()
        .map(|(binding, resource)| wgpu::BindGroupEntry {
            binding: binding as u32,
            resource: resource.clone(),
        })
        .collect();
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("water output fixture"),
        layout,
        entries: &entries,
    })
}

impl Gpu {
    fn new() -> Option<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::METAL | wgpu::Backends::VULKAN | wgpu::Backends::DX12,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = match block_on(instance.request_adapter(&Default::default())) {
            Ok(adapter) => adapter,
            Err(error) => return unavailable(error),
        };
        let info = adapter.get_info();
        if !matches!(
            info.device_type,
            wgpu::DeviceType::IntegratedGpu | wgpu::DeviceType::DiscreteGpu
        ) {
            return unavailable(format!(
                "adapter is not classified as physical GPU hardware: {info:?}"
            ));
        }
        // Four writable response arrays + three lighting arrays + VSM + WSGI.
        let caps = adapter.limits();
        if caps.max_bind_groups < 7 || caps.max_storage_buffers_per_shader_stage < 9 {
            return unavailable(format!(
                "water needs 7 groups / 9 fragment storage buffers: {caps:?}"
            ));
        }
        let (device, queue) = match block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("water real output GPU gate"),
            required_limits: wgpu::Limits {
                max_bind_groups: 7,
                max_storage_buffers_per_shader_stage: 9,
                ..Default::default()
            },
            ..Default::default()
        })) {
            Ok(pair) => pair,
            Err(error) => return unavailable(error),
        };
        eprintln!(
            "water output adapter: {} / {:?} / {:?}",
            info.name, info.backend, info.device_type
        );

        const F: wgpu::ShaderStages = wgpu::ShaderStages::FRAGMENT;
        const V: wgpu::ShaderStages = wgpu::ShaderStages::VERTEX;
        const VF: wgpu::ShaderStages = wgpu::ShaderStages::VERTEX_FRAGMENT;
        let uniform = buffer_type(None);
        let read = buffer_type(Some(true));
        let write = buffer_type(Some(false));
        let tex = texture_type(wgpu::TextureViewDimension::D2, true);
        let unfiltered = texture_type(wgpu::TextureViewDimension::D2, false);
        let sampler = wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering);
        let layouts = vec![
            layout(
                &device,
                &[
                    (VF, uniform),
                    (V, read),
                    (V, read),
                    (V, read),
                    (V, read),
                    (F, tex),
                    (F, sampler),
                    (V, read),
                    (F, write),
                    (F, write),
                    (F, write),
                    (F, write),
                ],
            ),
            layout(&device, &[(F, read), (F, read), (F, read)]),
            layout(
                &device,
                &[
                    (F, read),
                    (F, unfiltered),
                    (
                        F,
                        wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    ),
                    (F, uniform),
                ],
            ),
            layout(&device, &[(F, unfiltered), (F, uniform)]),
            layout(&device, &[(F, uniform)]),
            layout(
                &device,
                &[
                    (F, texture_type(wgpu::TextureViewDimension::D3, true)),
                    (F, sampler),
                    (F, uniform),
                ],
            ),
            layout(
                &device,
                &[
                    (F, uniform),
                    (F, uniform),
                    (F, read),
                    (F, uniform),
                    (F, unfiltered),
                    (F, uniform),
                ],
            ),
        ];
        let refs: Vec<_> = layouts.iter().map(Some).collect();
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("water production groups 0..6"),
            bind_group_layouts: &refs,
            immediate_size: 0,
        });
        // Distinct modules and pipelines, including FX, composition and HDR:
        // no part of the frozen oracle resolves to an under-test shader.
        let programs = [false, true].map(|historical| {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(if historical {
                    legacy::COMMIT
                } else {
                    "production water raster"
                }),
                source: wgpu::ShaderSource::Wgsl(
                    if historical {
                        legacy::compile("water_surface_raster.wesl")
                    } else {
                        production_wgsl(true)
                    }
                    .into(),
                ),
            });
            let raster = [
                "water_surface_fs_pbr",
                "water_surface_fs_npr",
                if historical {
                    "water_surface_fs_hybrid"
                } else {
                    "water_surface_fs_mask_blend"
                },
            ]
            .map(|entry| {
                device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(entry),
                    layout: Some(&pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &module,
                        entry_point: Some("water_surface_vs"),
                        compilation_options: Default::default(),
                        buffers: &[],
                    },
                    primitive: wgpu::PrimitiveState {
                        cull_mode: None,
                        ..Default::default()
                    },
                    depth_stencil: Some(wgpu::DepthStencilState {
                        format: wgpu::TextureFormat::Depth32Float,
                        depth_write_enabled: Some(false),
                        depth_compare: Some(wgpu::CompareFunction::GreaterEqual),
                        stencil: Default::default(),
                        bias: Default::default(),
                    }),
                    multisample: Default::default(),
                    fragment: Some(wgpu::FragmentState {
                        module: &module,
                        entry_point: Some(entry),
                        compilation_options: Default::default(),
                        targets: &[
                            Some(wgpu::ColorTargetState {
                                format: wgpu::TextureFormat::Rgba16Float,
                                blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                                write_mask: wgpu::ColorWrites::ALL,
                            }),
                            Some(wgpu::ColorTargetState {
                                format: wgpu::TextureFormat::Rg16Float,
                                blend: None,
                                write_mask: wgpu::ColorWrites::RED | wgpu::ColorWrites::GREEN,
                            }),
                            Some(wgpu::ColorTargetState {
                                format: wgpu::TextureFormat::R8Unorm,
                                blend: None,
                                write_mask: wgpu::ColorWrites::RED,
                            }),
                        ],
                    }),
                    multiview_mask: None,
                    cache: None,
                })
            });
            let fx = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(if historical {
                    "frozen historical water render-FX"
                } else {
                    "production water render-FX"
                }),
                source: wgpu::ShaderSource::Wgsl(
                    if historical {
                        legacy::compile("water_render_fx.wesl")
                    } else {
                        production_wgsl(false)
                    }
                    .into(),
                ),
            });
            let caustics = compute(&device, &fx, "water_caustics_project");
            let underwater = compute(&device, &fx, "water_underwater_volume");
            let composition = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(if historical {
                    "frozen historical resident composition"
                } else {
                    "production resident composition"
                }),
                source: wgpu::ShaderSource::Wgsl(
                    if historical {
                        legacy::source("response_composition.wgsl")
                    } else {
                        include_str!("../src/shaders/response_composition.wgsl")
                    }
                    .into(),
                ),
            });
            let hdr = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(if historical {
                    "frozen historical HDR export"
                } else {
                    "production HDR export"
                }),
                source: wgpu::ShaderSource::Wgsl(
                    if historical {
                        legacy::source("response_to_hdr.wgsl")
                    } else {
                        include_str!("../src/shaders/response_to_hdr.wgsl")
                    }
                    .into(),
                ),
            });
            let compose = compute(&device, &composition, "compose_responses");
            let export = compute(&device, &hdr, "response_to_hdr");
            Programs {
                raster,
                compose,
                export,
                caustics,
                underwater,
            }
        });
        Some(Self {
            device,
            queue,
            layouts,
            programs,
        })
    }
}

fn compute(
    device: &wgpu::Device,
    module: &wgpu::ShaderModule,
    entry: &str,
) -> wgpu::ComputePipeline {
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(entry),
        layout: None,
        module,
        entry_point: Some(entry),
        compilation_options: Default::default(),
        cache: None,
    })
}

fn upload<T: bytemuck::Pod>(device: &wgpu::Device, values: &[T], uniform: bool) -> wgpu::Buffer {
    device.create_buffer_init(&BufferInitDescriptor {
        label: Some("water fixture ABI data"),
        contents: bytemuck::cast_slice(values),
        usage: if uniform {
            wgpu::BufferUsages::UNIFORM
        } else {
            wgpu::BufferUsages::STORAGE
        } | wgpu::BufferUsages::COPY_SRC,
    })
}

fn readback(device: &wgpu::Device, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("water actual output readback"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn read<T: bytemuck::Pod>(device: &wgpu::Device, buffer: &wgpu::Buffer) -> Vec<T> {
    let (send, recv) = std::sync::mpsc::channel();
    buffer
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            send.send(result).expect("readback receiver alive");
        });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("GPU completion");
    recv.recv()
        .expect("map callback delivered")
        .expect("actual GPU buffer mapped");
    let mapped = buffer.slice(..).get_mapped_range().expect("mapped range");
    let values = bytemuck::cast_slice(&mapped).to_vec();
    drop(mapped);
    buffer.unmap();
    values
}

fn texture(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    depth: u32,
    usage: wgpu::TextureUsages,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("water output fixture texture"),
        size: wgpu::Extent3d {
            width: EDGE,
            height: EDGE,
            depth_or_array_layers: depth,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: if depth == 1 {
            wgpu::TextureDimension::D2
        } else {
            wgpu::TextureDimension::D3
        },
        format,
        usage,
        view_formats: &[],
    })
}

fn input_texture<T: bytemuck::Pod>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    format: wgpu::TextureFormat,
    values: &[T],
) -> wgpu::Texture {
    device.create_texture_with_data(
        queue,
        &wgpu::TextureDescriptor {
            label: Some("water controlled texture input"),
            size: wgpu::Extent3d {
                width: EDGE,
                height: EDGE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        },
        TextureDataOrder::LayerMajor,
        bytemuck::cast_slice(values),
    )
}

fn copy_texture(
    encoder: &mut wgpu::CommandEncoder,
    texture: &wgpu::Texture,
    buffer: &wgpu::Buffer,
) {
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(ROW),
                rows_per_image: Some(EDGE),
            },
        },
        texture.size(),
    );
}

#[derive(Clone)]
struct Case {
    foam: f32,
    vertex_foam: Option<[f32; 3]>,
    surface_uvs: Option<[[f32; 4]; 3]>,
    shallow: f32,
    overlay: f32,
    alpha: f32,
    whiten: f32,
    foam_threshold: f32,
    camera_z: f32,
    normal: [f32; 3],
    previous_displacement: [f32; 4],
    previous_vertex_displacement: Option<[[f32; 4]; 3]>,
    previous_clip: [[f32; 4]; 4],
    background: [f32; 4],
    caustic_incident: f32,
    caustic_enable: bool,
    caustic_offsets: Option<[[f32; 2]; PIXELS]>,
    volume_enable: bool,
    surface_light: Option<[f32; PIXELS]>,
    ssr_enable: bool,
    hzb_depth: f32,
}

impl Default for Case {
    fn default() -> Self {
        Self {
            foam: 0.0,
            vertex_foam: None,
            surface_uvs: None,
            shallow: 0.0,
            overlay: 0.0,
            alpha: 0.4,
            whiten: 1.0,
            foam_threshold: 0.6,
            camera_z: 2.0,
            normal: [0.0, 0.0, 1.0],
            previous_displacement: [0.0; 4],
            previous_vertex_displacement: None,
            previous_clip: IDENTITY,
            background: [0.0; 4],
            caustic_incident: 0.0,
            caustic_enable: false,
            caustic_offsets: None,
            volume_enable: false,
            surface_light: None,
            ssr_enable: false,
            hzb_depth: 0.45,
        }
    }
}

struct Output {
    left: Vec<[f32; 16]>,
    right: Vec<[f32; 16]>,
    composed: Vec<[f32; 16]>,
    controls: Vec<[u32; 4]>,
    coverage: Vec<u32>,
    color: Vec<u16>,
    motion: Vec<u16>,
    hdr: Vec<u16>,
    caustics: Vec<f32>,
    volume: Vec<u16>,
}

fn half(value: u16) -> f32 {
    let sign = if value & 0x8000 == 0 { 1.0 } else { -1.0 };
    let exponent = u32::from((value >> 10) & 31);
    let mantissa = u32::from(value & 1023);
    sign * match exponent {
        0 => mantissa as f32 / 16_777_216.0,
        31 => {
            if mantissa == 0 {
                f32::INFINITY
            } else {
                f32::NAN
            }
        }
        _ => f32::from_bits(((exponent + 112) << 23) | (mantissa << 13)),
    }
}

fn near(actual: f32, expected: f32, epsilon: f32) {
    assert!(
        actual.is_finite() && (actual - expected).abs() <= epsilon,
        "actual {actual}, expected {expected}, tolerance {epsilon}"
    );
}

impl Output {
    const CENTER: usize = (EDGE / 2 * EDGE + EDGE / 2) as usize;

    fn response(&self) -> [f32; 4] {
        let r = self.composed[Self::CENTER];
        [r[0], r[1], r[2], r[15]]
    }

    fn color(&self) -> [f32; 4] {
        let i = (EDGE / 2 * ROW / 2 + EDGE / 2 * 4) as usize;
        core::array::from_fn(|c| half(self.color[i + c]))
    }

    fn motion(&self) -> [f32; 2] {
        let i = (EDGE / 2 * ROW / 2 + EDGE / 2 * 2) as usize;
        [half(self.motion[i]), half(self.motion[i + 1])]
    }

    fn mask(&self) -> f32 {
        f32::from_bits(self.controls[Self::CENTER][0])
    }

    fn assert_resident_chain(&self, background: [f32; 4], historical: bool) {
        for i in 0..PIXELS {
            assert_eq!(
                self.coverage[i], 1,
                "production fragment coverage pixel {i}"
            );
            assert_eq!(
                self.controls[i][1],
                if historical { 1 } else { 31 },
                "response channel selection"
            );
            assert_eq!(self.controls[i][3], 0, "shared alpha rule");
            let mask = f32::from_bits(self.controls[i][0]);
            assert!(mask.is_finite() && (0.0..=1.0).contains(&mask));
            for word in self.left[i]
                .iter()
                .chain(&self.right[i])
                .chain(&self.composed[i])
            {
                assert!(word.is_finite(), "finite response pixel {i}");
            }
            for c in 3..15 {
                near(self.composed[i][c], 0.0, 0.0);
            }
            // Validate the resident dispatch against its actual raster-written
            // arms at every pixel, independently of the MRT/HDR comparison.
            for c in 0..15 {
                near(
                    self.composed[i][c],
                    self.left[i][c] * (1.0 - mask) + self.right[i][c] * mask,
                    0.00002,
                );
            }
            let xy = (i / EDGE as usize) * (ROW / 2) as usize + (i % EDGE as usize) * 4;
            for c in 0..4 {
                let word = if c == 3 { 15 } else { c };
                near(half(self.hdr[xy + c]), self.composed[i][word], 0.0015);
                near(
                    half(self.color[xy + c]),
                    self.composed[i][word] + background[c] * (1.0 - self.composed[i][15]),
                    0.0015,
                );
            }
            near(
                f32::from_bits(self.controls[i][2]),
                self.composed[i][15],
                0.00001,
            );
        }
    }
}

impl Gpu {
    // 0=PBR, 1=NPR, 2=MaskBlend. All have identical geometry/resources.
    fn draw(&self, case: &Case, entry: usize) -> Output {
        self.draw_program(case, entry, false)
    }

    // Historical entry 2 is the removed Hybrid, kept ONLY in this test fixture.
    fn draw_legacy(&self, case: &Case, entry: usize) -> Output {
        self.draw_program(case, entry, true)
    }

    fn draw_program(&self, case: &Case, entry: usize, historical: bool) -> Output {
        use wgpu::BindingResource::{Sampler, TextureView};
        use wgpu::TextureUsages as TU;
        let d = &self.device;
        let q = &self.queue;
        let programs = &self.programs[usize::from(historical)];
        let mut view = Vec::<[f32; 4]>::from(IDENTITY);
        view.extend([
            [0.0, 0.0, case.camera_z, 0.0],
            [0.08, 0.5, 0.2, case.whiten],
            [0.1, 0.3, 0.5, case.alpha],
            [4.0, case.foam_threshold, 0.0, case.shallow],
            [
                EDGE as f32,
                EDGE as f32,
                if historical { 0.0 } else { case.overlay },
                0.0,
            ],
        ]);
        let view = upload(d, &view, true);
        let base = upload(
            d,
            &[
                [-1.0_f32, -1.0, 0.5, 1.0],
                [3.0, -1.0, 0.5, 1.0],
                [-1.0, 3.0, 0.5, 1.0],
            ],
            false,
        );
        let uvs = upload(
            d,
            &case.surface_uvs.unwrap_or([[0.5_f32, 0.5, 0.0, 0.0]; 3]),
            false,
        );
        let displacement = upload(d, &[[0.0_f32; 4]; 3], false);
        let previous = upload(
            d,
            &case
                .previous_vertex_displacement
                .unwrap_or([case.previous_displacement; 3]),
            false,
        );
        let normals = upload(
            d,
            &case
                .vertex_foam
                .unwrap_or([case.foam; 3])
                .map(|foam| [case.normal[0], case.normal[1], case.normal[2], foam]),
            false,
        );
        let scene = input_texture(
            d,
            q,
            wgpu::TextureFormat::Rgba8Unorm,
            &[[43_u8, 79, 109, 255]; PIXELS],
        );
        let scene_view = scene.create_view(&Default::default());
        let sampler = d.create_sampler(&wgpu::SamplerDescriptor::default());
        let left = upload(d, &[[0.0_f32; 16]; PIXELS], false);
        let right = upload(d, &[[0.0_f32; 16]; PIXELS], false);
        let controls = upload(d, &[[0_u32; 4]; PIXELS], false);
        let coverage = upload(d, &[0_u32; PIXELS], false);

        // Lighting ABI: 16-byte ambient/count, 9 SH rows, counts/flags row,
        // then four StylizedParams rows. No CPU water/lighting evaluation.
        let directional = upload(d, &[[0.0_f32; 4]; 2], false);
        let punctual = upload(d, &[[0.0_f32; 4]; 4], false);
        let mut env = [[0.0_f32; 4]; 15];
        env[0] = [0.12, 0.18, 0.24, f32::from_bits(0)];
        env[10][2] = 1.0;
        let env = upload(d, &env, false);
        let vsm_pages = upload(d, &[u32::MAX; 4], false);
        let vsm_params = upload(d, &[[0_u32; 4]; 6], true);
        let hzb = input_texture(
            d,
            q,
            wgpu::TextureFormat::R32Float,
            &[case.hzb_depth; PIXELS],
        );
        let hzb_view = hzb.create_view(&Default::default());
        let mut ssr = [0_u32; 64];
        let mut clip = IDENTITY;
        clip[2][2] = 1.0 / 3.0;
        clip[3][2] = 1.0;
        let mut inverse = IDENTITY;
        inverse[2][2] = 3.0;
        inverse[3][2] = -3.0;
        let mut view_from_world = IDENTITY;
        view_from_world[3][2] = -2.0;
        for (i, matrix) in [clip, inverse, view_from_world].iter().enumerate() {
            ssr[i * 16..i * 16 + 16].copy_from_slice(bytemuck::cast_slice(matrix));
        }
        ssr[48..60].copy_from_slice(&[
            0.1_f32.to_bits(),
            1.0_f32.to_bits(),
            0.02_f32.to_bits(),
            64,
            0.05_f32.to_bits(),
            0.5_f32.to_bits(),
            0.9_f32.to_bits(),
            0.8_f32.to_bits(),
            0,
            u32::from(case.ssr_enable),
            0,
            0,
        ]);
        let ssr = upload(d, &ssr, true);
        let motion_cfg = upload(d, &[IDENTITY, case.previous_clip], true);

        let caustics = texture(
            d,
            wgpu::TextureFormat::R32Float,
            1,
            TU::STORAGE_BINDING | TU::TEXTURE_BINDING | TU::COPY_SRC,
        );
        let caustics_view = caustics.create_view(&Default::default());
        let offsets = input_texture(
            d,
            q,
            wgpu::TextureFormat::Rg32Float,
            &case.caustic_offsets.unwrap_or([[0.0_f32; 2]; PIXELS]),
        );
        let offsets_view = offsets.create_view(&Default::default());
        let photons = upload(d, &[0_u32; PIXELS], false);
        let caustic_params = upload(
            d,
            &[
                case.caustic_incident.to_bits(),
                4.0_f32.to_bits(),
                0,
                1.0_f32.to_bits(),
                1.0_f32.to_bits(),
                EDGE,
                EDGE,
                0,
            ],
            true,
        );
        let caustic_group = group(
            d,
            &programs.caustics.get_bind_group_layout(0),
            &[
                photons.as_entire_binding(),
                caustic_params.as_entire_binding(),
                TextureView(&caustics_view),
                TextureView(&offsets_view),
            ],
        );

        let volume = texture(
            d,
            wgpu::TextureFormat::Rgba16Float,
            VOLUME_DEPTH,
            TU::STORAGE_BINDING | TU::TEXTURE_BINDING | TU::COPY_SRC,
        );
        let volume_view = volume.create_view(&Default::default());
        let surface_light = input_texture(
            d,
            q,
            wgpu::TextureFormat::R32Float,
            &case.surface_light.unwrap_or([0.0_f32; PIXELS]),
        );
        let surface_light_view = surface_light.create_view(&Default::default());
        let underwater_params = upload(
            d,
            &[
                0.8_f32.to_bits(),
                0.4_f32.to_bits(),
                0.2_f32.to_bits(),
                0.5_f32.to_bits(),
                0.1_f32.to_bits(),
                0.3_f32.to_bits(),
                0.5_f32.to_bits(),
                0, // isotropic phase
                0,
                if case.surface_light.is_some() {
                    0.5_f32.to_bits()
                } else {
                    0
                },
                EDGE,
                EDGE,
                VOLUME_DEPTH,
                0,
                0,
                0,
            ],
            true,
        );
        let underwater_group = group(
            d,
            &programs.underwater.get_bind_group_layout(2),
            &[
                underwater_params.as_entire_binding(),
                TextureView(&volume_view),
                TextureView(&surface_light_view),
            ],
        );
        let empty_groups: Vec<_> = (0..2)
            .map(|i| group(d, &programs.underwater.get_bind_group_layout(i), &[]))
            .collect();
        let froxel_params = upload(
            d,
            &[0.125_f32.to_bits(), u32::from(case.volume_enable), 0, 0],
            true,
        );
        let gtao = upload(d, &[[0_u32; 4]; 2], true);
        let ssgi = upload(d, &[[0_u32; 4]; 2], true);
        let probes = upload(d, &[[0_f32; 4]; 6], false);
        let wsgi = upload(d, &[[0_u32; 4]; 2], true);
        let caustic_cfg = upload(
            d,
            &[1.0_f32.to_bits(), u32::from(case.caustic_enable), 0, 0],
            true,
        );

        let groups = [
            group(
                d,
                &self.layouts[0],
                &[
                    view.as_entire_binding(),
                    base.as_entire_binding(),
                    uvs.as_entire_binding(),
                    displacement.as_entire_binding(),
                    normals.as_entire_binding(),
                    TextureView(&scene_view),
                    Sampler(&sampler),
                    previous.as_entire_binding(),
                    left.as_entire_binding(),
                    right.as_entire_binding(),
                    controls.as_entire_binding(),
                    coverage.as_entire_binding(),
                ],
            ),
            group(
                d,
                &self.layouts[1],
                &[
                    directional.as_entire_binding(),
                    punctual.as_entire_binding(),
                    env.as_entire_binding(),
                ],
            ),
            group(
                d,
                &self.layouts[2],
                &[
                    vsm_pages.as_entire_binding(),
                    TextureView(&hzb_view),
                    Sampler(&sampler),
                    vsm_params.as_entire_binding(),
                ],
            ),
            group(
                d,
                &self.layouts[3],
                &[TextureView(&hzb_view), ssr.as_entire_binding()],
            ),
            group(d, &self.layouts[4], &[motion_cfg.as_entire_binding()]),
            group(
                d,
                &self.layouts[5],
                &[
                    TextureView(&volume_view),
                    Sampler(&sampler),
                    froxel_params.as_entire_binding(),
                ],
            ),
            group(
                d,
                &self.layouts[6],
                &[
                    gtao.as_entire_binding(),
                    ssgi.as_entire_binding(),
                    probes.as_entire_binding(),
                    wsgi.as_entire_binding(),
                    TextureView(&caustics_view),
                    caustic_cfg.as_entire_binding(),
                ],
            ),
        ];
        let color = texture(
            d,
            wgpu::TextureFormat::Rgba16Float,
            1,
            TU::RENDER_ATTACHMENT | TU::COPY_SRC,
        );
        let color_view = color.create_view(&Default::default());
        let motion = texture(
            d,
            wgpu::TextureFormat::Rg16Float,
            1,
            TU::RENDER_ATTACHMENT | TU::COPY_SRC,
        );
        let motion_view = motion.create_view(&Default::default());
        let reactive = texture(
            d,
            wgpu::TextureFormat::R8Unorm,
            1,
            TU::RENDER_ATTACHMENT | TU::COPY_SRC,
        );
        let reactive_view = reactive.create_view(&Default::default());
        let depth = texture(
            d,
            wgpu::TextureFormat::Depth32Float,
            1,
            TU::RENDER_ATTACHMENT,
        );
        let depth_view = depth.create_view(&Default::default());
        let hdr = texture(
            d,
            wgpu::TextureFormat::Rgba16Float,
            1,
            TU::STORAGE_BINDING | TU::COPY_SRC,
        );
        let hdr_view = hdr.create_view(&Default::default());
        let visibility = input_texture(d, q, wgpu::TextureFormat::R32Uint, &[0_u32; PIXELS]);
        let visibility_view = visibility.create_view(&Default::default());
        let composition_group = group(
            d,
            &programs.compose.get_bind_group_layout(0),
            &[
                left.as_entire_binding(),
                right.as_entire_binding(),
                controls.as_entire_binding(),
                coverage.as_entire_binding(),
            ],
        );
        let export_group = group(
            d,
            &programs.export.get_bind_group_layout(0),
            &[
                left.as_entire_binding(),
                TextureView(&hdr_view),
                TextureView(&visibility_view),
                TextureView(&visibility_view),
            ],
        );
        let left_read = readback(d, left.size());
        let right_read = readback(d, right.size());
        let composed_read = readback(d, left.size());
        let controls_read = readback(d, controls.size());
        let coverage_read = readback(d, coverage.size());
        let color_read = readback(d, u64::from(ROW * EDGE));
        let motion_read = readback(d, u64::from(ROW * EDGE));
        let hdr_read = readback(d, u64::from(ROW * EDGE));
        let caustics_read = readback(d, u64::from(ROW * EDGE));
        let volume_read = readback(d, u64::from(ROW * EDGE * VOLUME_DEPTH));
        let mut encoder = d.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&programs.caustics);
            pass.set_bind_group(0, &caustic_group, &[]);
            pass.dispatch_workgroups(EDGE.div_ceil(8), EDGE.div_ceil(8), 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&programs.underwater);
            for (i, group) in empty_groups.iter().enumerate() {
                pass.set_bind_group(i as u32, group, &[]);
            }
            pass.set_bind_group(2, &underwater_group, &[]);
            pass.dispatch_workgroups(EDGE.div_ceil(4), EDGE.div_ceil(4), VOLUME_DEPTH.div_ceil(4));
        }
        {
            let [r, g, b, a] = case.background.map(f64::from);
            let attachments = [
                Some(wgpu::RenderPassColorAttachment {
                    view: &color_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r, g, b, a }),
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: &motion_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: &reactive_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                }),
            ];
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("draw actual production water surface"),
                color_attachments: &attachments,
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            pass.set_pipeline(&programs.raster[entry]);
            for (i, group) in groups.iter().enumerate() {
                pass.set_bind_group(i as u32, group, &[]);
            }
            pass.draw(0..3, 0..1);
        }
        encoder.copy_buffer_to_buffer(&left, 0, &left_read, 0, left.size());
        encoder.copy_buffer_to_buffer(&right, 0, &right_read, 0, right.size());
        encoder.copy_buffer_to_buffer(&controls, 0, &controls_read, 0, controls.size());
        encoder.copy_buffer_to_buffer(&coverage, 0, &coverage_read, 0, coverage.size());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&programs.compose);
            pass.set_bind_group(0, &composition_group, &[]);
            pass.dispatch_workgroups((PIXELS as u32).div_ceil(64), 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&programs.export);
            pass.set_bind_group(0, &export_group, &[]);
            pass.dispatch_workgroups(EDGE.div_ceil(8), EDGE.div_ceil(8), 1);
        }
        encoder.copy_buffer_to_buffer(&left, 0, &composed_read, 0, left.size());
        copy_texture(&mut encoder, &color, &color_read);
        copy_texture(&mut encoder, &motion, &motion_read);
        copy_texture(&mut encoder, &hdr, &hdr_read);
        copy_texture(&mut encoder, &caustics, &caustics_read);
        copy_texture(&mut encoder, &volume, &volume_read);
        q.submit([encoder.finish()]);
        let output = Output {
            left: read(d, &left_read),
            right: read(d, &right_read),
            composed: read(d, &composed_read),
            controls: read(d, &controls_read),
            coverage: read(d, &coverage_read),
            color: read(d, &color_read),
            motion: read(d, &motion_read),
            hdr: read(d, &hdr_read),
            caustics: read(d, &caustics_read),
            volume: read(d, &volume_read),
        };
        output.assert_resident_chain(case.background, historical);
        output
    }
}

fn assert_mix(actual: [f32; 4], left: [f32; 4], right: [f32; 4], weight: f32) {
    // An output relation between independently drawn production endpoints, not
    // a CPU transcription supplying the pixels under test.
    for c in 0..4 {
        near(
            actual[c],
            left[c] * (1.0 - weight) + right[c] * weight,
            0.00002,
        );
    }
}

fn assert_region_output(
    hybrid: &Output,
    pbr: &Output,
    npr: &Output,
    weight: impl Fn(usize) -> f32,
) {
    // Endpoints are independent draws of production fragment entries, not CPU
    // shading goldens. Check the pre-overlay region and the stylized arm, too:
    // equal final colors alone could conceal a broken region producer.
    for i in 0..PIXELS {
        for c in 0..16 {
            near(
                hybrid.left[i][c],
                pbr.composed[i][c] * (1.0 - weight(i)) + npr.composed[i][c] * weight(i),
                0.00002,
            );
            near(hybrid.right[i][c], npr.composed[i][c], 0.00002);
        }
    }
}

fn different(a: [f32; 4], b: [f32; 4]) {
    assert!(
        (0..3).map(|c| (a[c] - b[c]).abs()).sum::<f32>() > 0.0001,
        "controlled production input must change actual RGB: {a:?} vs {b:?}"
    );
}

/// Independent-overlay extension of the old two-stage equation, fed ONLY by
/// the frozen historical PBR/NPR draws. Unlike a direct historical Hybrid
/// comparison this permits overlay != shallow; do not call it an old capture.
/// The direct old-ABI Hybrid comparison lives alongside this check below.
fn assert_legacy_frame(
    hybrid: &Output,
    pbr: &Output,
    npr: &Output,
    case: &Case,
    foam: impl Fn(usize) -> f32,
    shallow: impl Fn(usize) -> f32,
) {
    for i in 0..PIXELS {
        let foam = foam(i).clamp(0.0, 1.0);
        let shallow = shallow(i).clamp(0.0, 1.0);
        let region = foam.max(shallow);
        let pbr_arm = pbr.composed[i];
        let npr_arm = npr.composed[i];
        let mut expected = [0.0_f32; 16];
        let blended_alpha = pbr_arm[15] * (1.0 - region) + npr_arm[15] * region;
        let motion_index = (i / EDGE as usize) * (ROW / 2) as usize + (i % EDGE as usize) * 2;
        // Keep the expected reactive/overlay independent of the Hybrid output.
        let motion = [
            half(pbr.motion[motion_index]),
            half(pbr.motion[motion_index + 1]),
        ];
        let reactive = foam
            .max(((motion[0] * motion[0] + motion[1] * motion[1]).sqrt() * 32.0).clamp(0.0, 1.0))
            .max(1.0 - blended_alpha.clamp(0.0, 1.0));
        let overlay = (case.overlay * reactive.max(0.25)).clamp(0.0, 1.0);
        for c in 0..16 {
            let blended = pbr_arm[c] * (1.0 - region) + npr_arm[c] * region;
            if c == 15 {
                expected[c] = blended;
            } else {
                expected[c] = blended * (1.0 - overlay) + npr_arm[c] * overlay;
            }
        }
        for c in 0..16 {
            near(hybrid.composed[i][c], expected[c], 0.00003);
        }
        near(
            half(hybrid.motion[motion_index]),
            half(pbr.motion[motion_index]),
            0.00001,
        );
        near(
            half(hybrid.motion[motion_index + 1]),
            half(pbr.motion[motion_index + 1]),
            0.00001,
        );
        near(f32::from_bits(hybrid.controls[i][0]), overlay, 0.00003);
        assert_eq!(
            hybrid.controls[i][1], 31,
            "legacy frame channel mask at pixel {i}"
        );
        near(f32::from_bits(hybrid.controls[i][2]), expected[15], 0.00003);
        assert_eq!(
            hybrid.controls[i][3], 0,
            "legacy frame alpha rule at pixel {i}"
        );
        assert_eq!(hybrid.coverage[i], 1, "legacy frame coverage at pixel {i}");

        let texel = (i / EDGE as usize) * (ROW / 2) as usize + (i % EDGE as usize) * 4;
        for c in 0..4 {
            let hdr_expected = if c == 3 { expected[15] } else { expected[c] };
            near(half(hybrid.hdr[texel + c]), hdr_expected, 0.0015);
            near(
                half(hybrid.color[texel + c]),
                hdr_expected + case.background[c] * (1.0 - expected[15]),
                0.0015,
            );
        }
    }
}

fn reference_close(actual: f32, expected: f32, absolute: f32, relative: f32) -> bool {
    actual.is_finite()
        && expected.is_finite()
        && (actual - expected).abs() <= absolute + relative * expected.abs()
}

fn frozen_value(actual: f32, expected: f32, half_precision: bool, label: &str) {
    // f32 tolerates small compiler/FMA differences, not a shading change.
    // Half outputs allow about two ULP at normal values plus a small floor at
    // zero. Padding bytes are never compared. No frame-averaged error metric
    // can conceal one corrupt pixel, motion component, or response word.
    let (absolute, relative) = if half_precision {
        (2.0e-6, 2.0e-3)
    } else {
        (3.0e-5, 3.0e-5)
    };
    assert!(
        reference_close(actual, expected, absolute, relative),
        "{label}: current={actual}, frozen={expected}, tolerance={absolute}+{relative}*abs(reference), reference={}",
        legacy::COMMIT,
    );
}

/// Compare all active pixels, including pre-composition arms and independent
/// render-FX readbacks. The one intentional record difference is old channels
/// 1 -> current 31: this historical producer populated only RGB channel zero.
fn assert_frozen_output(actual: &Output, reference: &Output, label: &str) {
    for i in 0..PIXELS {
        assert_eq!(
            actual.coverage[i], reference.coverage[i],
            "{label}: coverage {i}"
        );
        assert_eq!(actual.controls[i][1], 31, "{label}: current channels {i}");
        assert_eq!(
            reference.controls[i][1], 1,
            "{label}: historical channels {i}"
        );
        assert_eq!(
            actual.controls[i][3], reference.controls[i][3],
            "{label}: alpha rule {i}"
        );
        for c in [0, 2] {
            frozen_value(
                f32::from_bits(actual.controls[i][c]),
                f32::from_bits(reference.controls[i][c]),
                false,
                &format!("{label}: control pixel={i} word={c}"),
            );
        }
        for (name, a, b) in [
            ("left", &actual.left, &reference.left),
            ("right", &actual.right, &reference.right),
            ("composed", &actual.composed, &reference.composed),
        ] {
            for c in 0..16 {
                frozen_value(
                    a[i][c],
                    b[i][c],
                    false,
                    &format!("{label}: {name} pixel={i} word={c}"),
                );
            }
        }
        let y = i / EDGE as usize;
        let x = i % EDGE as usize;
        for (name, a, b, channels) in [
            ("color", &actual.color, &reference.color, 4),
            ("HDR", &actual.hdr, &reference.hdr, 4),
            ("motion", &actual.motion, &reference.motion, 2),
        ] {
            let texel = y * (ROW / 2) as usize + x * channels;
            for c in 0..channels {
                frozen_value(
                    half(a[texel + c]),
                    half(b[texel + c]),
                    true,
                    &format!("{label}: {name} pixel={i} component={c}"),
                );
            }
        }
        let caustic = y * (ROW / 4) as usize + x;
        frozen_value(
            actual.caustics[caustic],
            reference.caustics[caustic],
            false,
            &format!("{label}: caustics pixel={i}"),
        );
        for z in 0..VOLUME_DEPTH as usize {
            let texel = (z * EDGE as usize + y) * (ROW / 2) as usize + x * 4;
            for c in 0..4 {
                frozen_value(
                    half(actual.volume[texel + c]),
                    half(reference.volume[texel + c]),
                    true,
                    &format!("{label}: volume pixel={i} z={z} component={c}"),
                );
            }
        }
    }
}

fn run_legacy_frame_case(case: Case, foam: impl Fn(usize) -> f32, shallow: impl Fn(usize) -> f32) {
    let Some(gpu) = gpu() else { return };
    // A live endpoint is NEVER the expected side. Pin both separately even if
    // a region weight of 0 or 1 would hide one endpoint in the final Hybrid.
    let pbr = gpu.draw_legacy(&case, 0);
    let npr = gpu.draw_legacy(&case, 1);
    assert_frozen_output(&gpu.draw(&case, 0), &pbr, "PBR endpoint");
    assert_frozen_output(&gpu.draw(&case, 1), &npr, "NPR endpoint");
    let hybrid = gpu.draw(&case, 2);
    assert_legacy_frame(&hybrid, &pbr, &npr, &case, foam, shallow);

    // Old Hybrid aliases its overlay and shallow controls. Do not patch that
    // shader to accept today's independent overlay or pretend unequal inputs
    // were historical data. Execute its genuine old contract as a second draw.
    let old_abi_case = Case {
        overlay: case.shallow,
        ..case
    };
    let old_hybrid = gpu.draw_legacy(&old_abi_case, 2);
    let current = gpu.draw(&old_abi_case, 2);
    assert_frozen_output(&current, &old_hybrid, "direct historical Hybrid");
}

#[test]
fn frozen_reference_tolerance_is_finite_and_detects_shared_endpoint_drift() {
    // Comparator sensitivity, not a claim of an executed GPU mutation test.
    for reference in [0.0_f32, 0.12, 0.4, 1.0, 4.0] {
        assert!(reference_close(reference, reference, 3.0e-5, 3.0e-5));
        assert!(!reference_close(
            reference + 0.001,
            reference,
            3.0e-5,
            3.0e-5
        ));
        assert!(!reference_close(
            reference + 0.02,
            reference,
            2.0e-6,
            2.0e-3
        ));
    }
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(!reference_close(bad, bad, 3.0e-5, 3.0e-5));
        assert!(!reference_close(0.0, bad, 3.0e-5, 3.0e-5));
    }
}

#[test]
fn water_legacy_gpu_parity_depth_full_frame() {
    run_legacy_frame_case(
        Case {
            shallow: 0.75,
            ..Default::default()
        },
        |_| 0.0,
        |_| 0.75,
    );
    // Also freeze the real depth-dependent volume consumer, not just the
    // authored shallow scalar. Two eye depths sample the historical compute
    // volume with a nonuniform surface-light input.
    for camera_z in [2.0, 6.0] {
        run_legacy_frame_case(
            Case {
                camera_z,
                volume_enable: true,
                surface_light: Some(core::array::from_fn(|i| {
                    if i % EDGE as usize >= EDGE as usize / 2 {
                        1.0
                    } else {
                        0.0
                    }
                })),
                ..Default::default()
            },
            |_| 0.0,
            |_| 0.0,
        );
    }
}

#[test]
fn water_legacy_gpu_parity_shore_full_frame() {
    run_legacy_frame_case(
        Case {
            foam: 0.65,
            overlay: 0.0,
            whiten: 0.0,
            foam_threshold: 2.0,
            ..Default::default()
        },
        |_| 0.65,
        |_| 0.0,
    );
    run_legacy_frame_case(
        Case {
            vertex_foam: Some([0.0, 1.0, 0.0]),
            shallow: 0.2,
            whiten: 0.0,
            foam_threshold: 2.0,
            ..Default::default()
        },
        |i| ((i % EDGE as usize) as f32 + 0.5) / (2 * EDGE) as f32,
        |_| 0.2,
    );
}

#[test]
fn water_legacy_gpu_parity_foam_full_frame() {
    // Straddle the toon threshold and include full foam/alpha saturation.
    for foam in [0.59, 0.8, 1.0] {
        run_legacy_frame_case(
            Case {
                foam,
                ..Default::default()
            },
            |_| foam,
            |_| 0.0,
        );
    }
}

#[test]
fn water_legacy_gpu_parity_caustics_full_frame() {
    run_legacy_frame_case(
        Case {
            caustic_incident: 0.25,
            caustic_enable: true,
            ..Default::default()
        },
        |_| 0.0,
        |_| 0.0,
    );
    run_legacy_frame_case(
        Case {
            caustic_incident: 0.25,
            caustic_enable: true,
            surface_uvs: Some([
                [0.0, 0.5, 0.0, 0.0],
                [2.0, 0.5, 0.0, 0.0],
                [0.0, 0.5, 0.0, 0.0],
            ]),
            caustic_offsets: Some(core::array::from_fn(|i| {
                let x = i % EDGE as usize;
                [-0.5 * x.min(EDGE as usize - 1 - x) as f32, 0.0]
            })),
            ..Default::default()
        },
        |_| 0.0,
        |_| 0.0,
    );
}

#[test]
fn water_legacy_gpu_parity_reflection_full_frame() {
    for (ssr_enable, hzb_depth) in [(true, 0.45), (true, 0.0), (false, 0.45)] {
        run_legacy_frame_case(
            Case {
                normal: [0.95, 0.0, 0.3122499],
                ssr_enable,
                hzb_depth,
                ..Default::default()
            },
            |_| 0.0,
            |_| 0.0,
        );
    }
}

#[test]
fn water_legacy_gpu_parity_transparency_overlay_full_frame() {
    for alpha in [0.65, 1.0] {
        run_legacy_frame_case(
            Case {
                alpha,
                shallow: 0.2,
                overlay: 0.7,
                background: [0.125, 0.25, 0.5, 1.0],
                ..Default::default()
            },
            |_| 0.0,
            |_| 0.2,
        );
    }
}

#[test]
fn water_legacy_gpu_parity_motion_vector_full_frame() {
    run_legacy_frame_case(
        Case {
            previous_displacement: [-0.125, 0.25, 0.0, 0.0],
            ..Default::default()
        },
        |_| 0.0,
        |_| 0.0,
    );
    let mut camera = IDENTITY;
    camera[3][0] = 0.25;
    let mut behind = IDENTITY;
    behind[3][3] = -1.0;
    for previous_clip in [camera, behind] {
        run_legacy_frame_case(
            Case {
                previous_clip,
                ..Default::default()
            },
            |_| 0.0,
            |_| 0.0,
        );
    }
}

#[test]
fn water_legacy_gpu_parity_reactive_mask_full_frame() {
    let base = Case {
        alpha: 1.0,
        shallow: 0.2,
        overlay: 1.0,
        ..Default::default()
    };
    // Floor, foam, transparency, and a spatial unsaturated motion field are
    // independent branches. Motion values here are exactly representable in
    // FP16, so the endpoint MRT can feed the CPU extension without extra
    // quantization tolerance; direct old-Hybrid parity uses no CPU equation.
    for case in [
        base.clone(),
        Case {
            foam: 0.5,
            ..base.clone()
        },
        Case {
            alpha: 0.4,
            ..base.clone()
        },
        Case {
            previous_vertex_displacement: Some([[0.0; 4], [-0.125, 0.0, 0.0, 0.0], [0.0; 4]]),
            ..base
        },
    ] {
        let foam = case.foam;
        run_legacy_frame_case(case, |_| foam, |_| 0.2);
    }
}

#[test]
fn water_gpu_authored_shallow_mix_and_underwater_depth_output() {
    let Some(gpu) = gpu() else {
        return;
    };
    let base = Case::default();
    let pbr = gpu.draw(&base, 0);
    let npr = gpu.draw(&base, 1);
    different(pbr.response(), npr.response());
    // This authored scalar is not a bathymetry texture. Include clamped
    // endpoints so a stale ABI lane or absent normalization cannot pass.
    for shallow in [-1.0_f32, 0.0, 0.25, 0.5, 1.0, 2.0] {
        let out = gpu.draw(
            &Case {
                shallow,
                ..base.clone()
            },
            2,
        );
        assert_region_output(&out, &pbr, &npr, |_| shallow.clamp(0.0, 1.0));
    }
    // Separately exercise actual eye-depth consumption of a production-written
    // 3D volume. This is not a claim of bathymetric depth-to-weight wiring.
    let near_case = Case {
        volume_enable: true,
        ..base.clone()
    };
    let far_case = Case {
        camera_z: 6.0,
        ..near_case.clone()
    };
    let near_volume = gpu.draw(&near_case, 2);
    let far_volume = gpu.draw(&far_case, 2);
    let near_disabled = gpu.draw(
        &Case {
            volume_enable: false,
            ..near_case.clone()
        },
        2,
    );
    let far_disabled = gpu.draw(
        &Case {
            volume_enable: false,
            ..far_case.clone()
        },
        2,
    );
    for c in 0..3 {
        // Paired on/off draws hold the eye, Fresnel and authored shallow bias
        // fixed; changing the camera alone is not texture-consumption proof.
        assert!(near_volume.response()[c] < near_disabled.response()[c]);
        assert!(far_volume.response()[c] < far_disabled.response()[c]);
        assert!(far_volume.response()[c] < near_volume.response()[c]);
    }
    near(near_volume.mask(), 0.0, 0.0);
    near(far_volume.mask(), 0.0, 0.0);
    assert_region_output(
        &near_volume,
        &gpu.draw(&near_case, 0),
        &gpu.draw(&near_case, 1),
        |_| 0.0,
    );
    assert_region_output(
        &far_volume,
        &gpu.draw(&far_case, 0),
        &gpu.draw(&far_case, 1),
        |_| 0.0,
    );
    for z in 1..VOLUME_DEPTH as usize {
        let row = (ROW * EDGE / 2) as usize;
        assert!(
            half(near_volume.volume[z * row + 3]) < half(near_volume.volume[(z - 1) * row + 3])
        );
    }
    // Unlike the authored shallow scalar, this input is a real spatial
    // texture: production compute writes its lighting into the 3D volume,
    // then production raster samples that volume. A paired dark texture keeps
    // every scalar, camera, region weight and transmittance unchanged.
    let patterned_case = Case {
        volume_enable: true,
        surface_light: Some(core::array::from_fn(|i| {
            if i % EDGE as usize >= EDGE as usize / 2 {
                1.0
            } else {
                0.0
            }
        })),
        ..base
    };
    let patterned = gpu.draw(&patterned_case, 2);
    let dark = gpu.draw(
        &Case {
            surface_light: Some([0.0; PIXELS]),
            ..patterned_case.clone()
        },
        2,
    );
    assert_region_output(
        &patterned,
        &gpu.draw(&patterned_case, 0),
        &gpu.draw(&patterned_case, 1),
        |_| 0.0,
    );
    for i in 0..PIXELS {
        let lit = i % EDGE as usize >= EDGE as usize / 2;
        for c in 0..3 {
            if lit {
                assert!(patterned.composed[i][c] > dark.composed[i][c]);
            } else {
                near(patterned.composed[i][c], dark.composed[i][c], 0.00002);
            }
        }
        let volume_pixel = (i / EDGE as usize) * (ROW / 2) as usize + (i % EDGE as usize) * 4;
        near(
            half(patterned.volume[volume_pixel + 3]),
            half(dark.volume[volume_pixel + 3]),
            0.0,
        );
        for c in 0..3 {
            if lit {
                assert!(half(patterned.volume[volume_pixel + c]) > 0.0);
            } else {
                near(half(patterned.volume[volume_pixel + c]), 0.0, 0.0);
            }
        }
    }
}

#[test]
fn water_gpu_shore_signal_continuously_mixes_production_responses() {
    let Some(gpu) = gpu() else {
        return;
    };
    // Current production shore ABI is the solver's foam lane, not a distance
    // texture. Disable only its whitening/ink threshold to isolate weighting.
    for foam in [0.0, 0.2, 0.4, 1.0] {
        let case = Case {
            foam,
            whiten: 0.0,
            foam_threshold: 2.0,
            ..Default::default()
        };
        let pbr = gpu.draw(&case, 0);
        let npr = gpu.draw(&case, 1);
        let hybrid = gpu.draw(&case, 2);
        assert_region_output(&hybrid, &pbr, &npr, |_| foam);
    }
    let case = Case {
        foam: 0.2,
        shallow: 0.6,
        whiten: 0.0,
        foam_threshold: 2.0,
        ..Default::default()
    };
    assert_region_output(
        &gpu.draw(&case, 2),
        &gpu.draw(&case, 0),
        &gpu.draw(&case, 1),
        |_| 0.6,
    );
    // A spatial signal travels through the actual storage-buffer vertex stage
    // and interpolation. For this fullscreen triangle its foam is (x+.5)/32.
    // This is deliberately not labeled a shoreline-distance texture test.
    let spatial = Case {
        vertex_foam: Some([0.0, 1.0, 0.0]),
        shallow: 0.2,
        whiten: 0.0,
        foam_threshold: 2.0,
        ..Default::default()
    };
    let hybrid = gpu.draw(&spatial, 2);
    assert_region_output(
        &hybrid,
        &gpu.draw(&spatial, 0),
        &gpu.draw(&spatial, 1),
        |i| (((i % EDGE as usize) as f32 + 0.5) / (2 * EDGE) as f32).max(0.2),
    );
    different(
        [
            hybrid.left[0][0],
            hybrid.left[0][1],
            hybrid.left[0][2],
            hybrid.left[0][15],
        ],
        [
            hybrid.left[15][0],
            hybrid.left[15][1],
            hybrid.left[15][2],
            hybrid.left[15][15],
        ],
    );
}

#[test]
fn water_gpu_foam_changes_color_alpha_and_toon_edge() {
    let Some(gpu) = gpu() else {
        return;
    };
    let clear = gpu.draw(&Case::default(), 2);
    let foam = gpu.draw(
        &Case {
            foam: 1.0,
            ..Default::default()
        },
        2,
    );
    different(clear.response(), foam.response());
    for c in 0..4 {
        near(foam.response()[c], 1.0, 0.00002);
        near(foam.color()[c], 1.0, 0.0015);
    }
    assert!(clear.response()[3] < foam.response()[3]);
    let no_ink = gpu.draw(
        &Case {
            foam: 1.0,
            whiten: 0.0,
            foam_threshold: 2.0,
            ..Default::default()
        },
        2,
    );
    different(foam.response(), no_ink.response());
    let ink_only = gpu.draw(
        &Case {
            foam: 1.0,
            whiten: 0.0,
            ..Default::default()
        },
        2,
    );
    different(ink_only.response(), no_ink.response());
    for c in 0..4 {
        near(ink_only.response()[c], 1.0, 0.00002);
    }
}

#[test]
fn water_gpu_caustics_compute_texture_reaches_surface_and_hdr() {
    let Some(gpu) = gpu() else {
        return;
    };
    let case = Case {
        caustic_incident: 0.25,
        caustic_enable: true,
        ..Default::default()
    };
    let focused = gpu.draw(&case, 2);
    let disabled = gpu.draw(
        &Case {
            caustic_enable: false,
            ..case.clone()
        },
        2,
    );
    let dark = gpu.draw(
        &Case {
            caustic_incident: 0.0,
            ..case
        },
        2,
    );
    for y in 0..EDGE as usize {
        for x in 0..EDGE as usize {
            near(focused.caustics[y * (ROW / 4) as usize + x], 0.25, 0.00001);
            near(dark.caustics[y * (ROW / 4) as usize + x], 0.0, 0.0);
        }
    }
    for c in 0..3 {
        assert!(focused.response()[c] > disabled.response()[c]);
    }
    assert_mix(disabled.response(), dark.response(), dark.response(), 0.0);
    different(focused.color(), disabled.color());
}

#[test]
fn water_gpu_spatial_caustics_offsets_reach_surface_and_resident_hdr() {
    let Some(gpu) = gpu() else {
        return;
    };
    let case = Case {
        caustic_incident: 0.25,
        caustic_enable: true,
        // Production VS interpolates u across the framebuffer. A constant v
        // isolates the horizontal texture pattern without CPU water shading.
        surface_uvs: Some([
            [0.0, 0.5, 0.0, 0.0],
            [2.0, 0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0, 0.0],
        ]),
        caustic_offsets: Some(core::array::from_fn(|i| {
            let x = i % EDGE as usize;
            [-0.5 * x.min(EDGE as usize - 1 - x) as f32, 0.0]
        })),
        ..Default::default()
    };
    let patterned = gpu.draw(&case, 2);
    let flat_case = Case {
        caustic_offsets: None,
        ..case.clone()
    };
    let flat = gpu.draw(&flat_case, 2);
    let disabled = gpu.draw(
        &Case {
            caustic_enable: false,
            ..case
        },
        2,
    );
    let flat_disabled = gpu.draw(
        &Case {
            caustic_enable: false,
            ..flat_case
        },
        2,
    );
    for i in 0..PIXELS {
        let x = i % EDGE as usize;
        let texel = (i / EDGE as usize) * (ROW / 4) as usize + x;
        let irradiance = patterned.caustics[texel];
        assert!(irradiance.is_finite() && irradiance > 0.0);
        // The production finite-difference kernel must focus the decreasing
        // offset field and spread the increasing one. The output itself is
        // GPU-computed: no host Jacobian or caustics implementation is used.
        if x < EDGE as usize / 2 {
            assert!(irradiance > flat.caustics[texel], "focused texel {i}");
        } else {
            assert!(irradiance < flat.caustics[texel], "spread texel {i}");
        }
        near(disabled.caustics[texel], irradiance, 0.0);
        for c in 0..3 {
            if x < EDGE as usize / 2 {
                assert!(patterned.composed[i][c] > flat.composed[i][c]);
            } else {
                assert!(patterned.composed[i][c] < flat.composed[i][c]);
            }
        }
        // Disabling only the raster consumer leaves the computed texture
        // intact but removes its spatial effect from every resident response.
        for c in 0..16 {
            near(
                disabled.composed[i][c],
                flat_disabled.composed[i][c],
                0.00002,
            );
        }
    }
    // draw() also checks every pixel of the real blended MRT and HDR export
    // against these resident results, so texture generation alone cannot pass.
}

#[test]
fn water_gpu_reflection_ssr_hit_miss_and_disable_affect_output() {
    let Some(gpu) = gpu() else {
        return;
    };
    // Tilted mirror ray travels away from the eye, crosses real reverse-Z HZB
    // depth 0.45 inside the viewport, and samples the bound scene-color texture.
    let case = Case {
        normal: [0.95, 0.0, 0.3122499],
        ssr_enable: true,
        ..Default::default()
    };
    let hit = gpu.draw(&case, 2);
    let disabled = gpu.draw(
        &Case {
            ssr_enable: false,
            ..case.clone()
        },
        2,
    );
    let miss = gpu.draw(
        &Case {
            hzb_depth: 0.0,
            ..case
        },
        2,
    );
    assert_mix(
        miss.response(),
        disabled.response(),
        disabled.response(),
        0.0,
    );
    different(hit.response(), miss.response());
    for c in 0..3 {
        assert!(hit.response()[c] > miss.response()[c]);
    }
}

#[test]
fn water_gpu_transparency_overlay_and_premultiplied_blend_output() {
    let Some(gpu) = gpu() else {
        return;
    };
    let case = Case {
        shallow: 0.2,
        ..Default::default()
    };
    let plain = gpu.draw(&case, 2);
    let overlay = gpu.draw(
        &Case {
            overlay: 0.7,
            ..case.clone()
        },
        2,
    );
    different(plain.response(), overlay.response());
    near(plain.mask(), 0.0, 0.0);
    assert!(overlay.mask() > 0.0 && overlay.mask() < 0.7);
    let bg = [0.125, 0.25, 0.5, 1.0];
    let blended = gpu.draw(
        &Case {
            overlay: 0.7,
            background: bg,
            ..case
        },
        2,
    );
    for (c, background) in bg.into_iter().enumerate() {
        near(
            blended.color()[c],
            overlay.response()[c] + background * (1.0 - overlay.response()[3]),
            0.0015,
        );
    }
    // Resident responses are independent of the already-present HDR backdrop.
    assert_mix(
        blended.response(),
        overlay.response(),
        overlay.response(),
        0.0,
    );
}

#[test]
fn water_gpu_motion_mrt_tracks_wave_camera_and_invalid_previous_projection() {
    let Some(gpu) = gpu() else {
        return;
    };
    let still = gpu.draw(&Case::default(), 2);
    for component in still.motion() {
        near(component, 0.0, 0.00001);
    }
    let wave = gpu.draw(
        &Case {
            previous_displacement: [-0.125, 0.25, 0.0, 0.0],
            ..Default::default()
        },
        2,
    );
    near(wave.motion()[0], 0.0625, 0.0001);
    near(wave.motion()[1], 0.125, 0.0001);
    let mut camera = IDENTITY;
    camera[3][0] = 0.25;
    let camera_move = gpu.draw(
        &Case {
            previous_clip: camera,
            ..Default::default()
        },
        2,
    );
    near(camera_move.motion()[0], -0.125, 0.0001);
    near(camera_move.motion()[1], 0.0, 0.0001);
    let mut behind = IDENTITY;
    behind[3][3] = -1.0;
    let invalid = gpu.draw(
        &Case {
            previous_clip: behind,
            previous_displacement: [-0.125, 0.25, 0.0, 0.0],
            ..Default::default()
        },
        2,
    );
    for component in invalid.motion() {
        near(component, 0.0, 0.00001);
    }
}

#[test]
fn water_gpu_reactive_foam_motion_transparency_drive_resident_overlay() {
    let Some(gpu) = gpu() else {
        return;
    };
    let case = Case {
        alpha: 1.0,
        shallow: 0.2,
        overlay: 1.0,
        ..Default::default()
    };
    let verify = |case: &Case, expected_mask: f32, region: f32| {
        let hybrid = gpu.draw(case, 2);
        assert_region_output(&hybrid, &gpu.draw(case, 0), &gpu.draw(case, 1), |_| region);
        for control in &hybrid.controls {
            near(f32::from_bits(control[0]), expected_mask, 0.00002);
        }
        hybrid
    };
    let still = verify(&case, 0.25, 0.2);
    near(still.mask(), 0.25, 0.00002); // overlay floor, not a reactive MRT
    let moving = verify(
        &Case {
            previous_displacement: [-0.03125, 0.0, 0.0, 0.0],
            ..case.clone()
        },
        0.5,
        0.2,
    );
    near(moving.mask(), 0.5, 0.00002);
    let fast = verify(
        &Case {
            previous_displacement: [-0.125, 0.0, 0.0, 0.0],
            ..case.clone()
        },
        1.0,
        0.2,
    );
    near(fast.mask(), 1.0, 0.00002);
    let foam = verify(
        &Case {
            foam: 0.5,
            ..case.clone()
        },
        0.5,
        0.5,
    );
    near(foam.mask(), 0.5, 0.00002);
    let transparent = verify(
        &Case {
            alpha: 0.4,
            ..case.clone()
        },
        0.6,
        0.2,
    );
    near(transparent.mask(), 0.6, 0.00002);
    different(still.response(), moving.response());
    let opposite = verify(
        &Case {
            previous_displacement: [0.03125, 0.0, 0.0, 0.0],
            ..case.clone()
        },
        0.5,
        0.2,
    );
    for i in 0..PIXELS {
        for c in 0..16 {
            near(moving.composed[i][c], opposite.composed[i][c], 0.00002);
        }
    }
    near(opposite.motion()[0], -moving.motion()[0], 0.00001);
    // History behind the camera produces zero motion, so it must not elevate
    // the overlay. This verifies the motion validity branch at its consumer.
    let mut behind = IDENTITY;
    behind[3][3] = -1.0;
    let invalid = verify(
        &Case {
            previous_clip: behind,
            previous_displacement: [-0.125, 0.0, 0.0, 0.0],
            ..case.clone()
        },
        0.25,
        0.2,
    );
    for i in 0..PIXELS {
        for c in 0..16 {
            near(invalid.composed[i][c], still.composed[i][c], 0.00002);
        }
    }
    // A real motion signal cannot alter radiance through a disabled overlay.
    // Still require the motion MRT itself to change in this negative control.
    let disabled = Case {
        overlay: 0.0,
        ..case
    };
    let disabled_still = verify(&disabled, 0.0, 0.2);
    let disabled_moving = verify(
        &Case {
            previous_displacement: [-0.125, 0.0, 0.0, 0.0],
            ..disabled
        },
        0.0,
        0.2,
    );
    assert!(disabled_moving.motion()[0] > disabled_still.motion()[0]);
    for i in 0..PIXELS {
        for c in 0..16 {
            near(
                disabled_still.composed[i][c],
                disabled_moving.composed[i][c],
                0.00002,
            );
        }
    }
    // At saturated motion, the real resident compositor must choose the
    // actually rasterized right arm, not merely store a plausible control.
    let right = fast.right[Output::CENTER];
    assert_mix(
        fast.response(),
        [right[0], right[1], right[2], right[15]],
        [right[0], right[1], right[2], right[15]],
        1.0,
    );
}

#[test]
fn water_gpu_spatial_motion_drives_per_pixel_reactive_overlay_output() {
    let Some(gpu) = gpu() else {
        return;
    };
    let case = Case {
        alpha: 1.0,
        shallow: 0.2,
        overlay: 1.0,
        // Only the right vertex moved in the previous frame. The production
        // VS and interpolator turn this into a spatial motion field, not a
        // uniform authored control or a separately evaluated test shader.
        previous_vertex_displacement: Some([[0.0; 4], [-0.125, 0.0, 0.0, 0.0], [0.0; 4]]),
        ..Default::default()
    };
    let hybrid = gpu.draw(&case, 2);
    let pbr = gpu.draw(&case, 0);
    let npr = gpu.draw(&case, 1);
    let disabled = gpu.draw(
        &Case {
            overlay: 0.0,
            ..case
        },
        2,
    );
    assert_region_output(&hybrid, &pbr, &npr, |_| 0.2);
    assert_region_output(&disabled, &pbr, &npr, |_| 0.2);
    let mut radiance_delta = 0.0;
    for i in 0..PIXELS {
        let x = i % EDGE as usize;
        let motion_index = (i / EDGE as usize) * (ROW / 2) as usize + x * 2;
        let expected_motion = (x as f32 + 0.5) / (EDGE * 32) as f32;
        let expected_overlay = ((x as f32 + 0.5) / EDGE as f32).max(0.25);
        near(half(hybrid.motion[motion_index]), expected_motion, 0.00001);
        near(half(hybrid.motion[motion_index + 1]), 0.0, 0.00001);
        near(
            f32::from_bits(hybrid.controls[i][0]),
            expected_overlay,
            0.00002,
        );
        near(f32::from_bits(disabled.controls[i][0]), 0.0, 0.0);
        for c in 0..2 {
            near(
                half(disabled.motion[motion_index + c]),
                half(hybrid.motion[motion_index + c]),
                0.0,
            );
        }
        for c in 0..16 {
            // Independent production PBR/NPR draws are the endpoints. Test
            // the final output as well as the recorded mask, so a plausible
            // ABI signal without a real composition effect cannot pass.
            let stylized_weight = 0.2 + 0.8 * expected_overlay;
            near(
                hybrid.composed[i][c],
                pbr.composed[i][c] * (1.0 - stylized_weight) + npr.composed[i][c] * stylized_weight,
                0.00002,
            );
            if c < 3 {
                radiance_delta += (hybrid.composed[i][c] - disabled.composed[i][c]).abs();
            }
        }
    }
    assert!(
        radiance_delta > 0.0001,
        "reactive overlay changes actual RGB"
    );
}
