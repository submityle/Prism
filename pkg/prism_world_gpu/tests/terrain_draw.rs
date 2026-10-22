//! Physical-GPU regression for baked terrain -> Prism Mesh -> indexed raster.
//! This is deliberately an offscreen adapter test, not a renderer-graph claim.

#![cfg(feature = "gpu")]

use prism_world_gpu::gpu::GpuContext;
use prism_world_terrain::{
    BakeLimits, CubeFace, DirtyRegion, HeightSample, HeightSampleValue, HeightSampler,
    HeightfieldLodSeamPlan, HeightfieldMesh, HeightfieldSpec, HeightfieldTile, LodSeamCoverage,
    LodSeamLimits, LodSeamNeighbor, TileEdge, TileKey,
};
use wgpu::util::DeviceExt;

const SIDE: u32 = 64;
const STRIDE: usize = 32;
const LIMITS: BakeLimits = BakeLimits {
    max_samples: 81,
    max_bytes: 4_128,
};

struct ConstantHeight(i64);

impl HeightSampler for ConstantHeight {
    type Error = core::convert::Infallible;

    fn sample_height(&self, _: HeightSample) -> Result<HeightSampleValue, Self::Error> {
        Ok(HeightSampleValue::from_millimetres(self.0))
    }
}

const SHADER: &str = r#"
struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) height: f32,
    @location(1) normal_y: f32,
    @location(2) uv: vec2<f32>,
}
@vertex fn vs(
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
) -> VertexOut {
    var out: VertexOut;
    out.position = vec4<f32>(position.x, -position.z, 0.5, 1.0);
    out.height = position.y;
    out.normal_y = normal.y;
    out.uv = uv;
    return out;
}
@fragment fn fs(in: VertexOut) -> @location(0) vec4<f32> {
    return vec4<f32>(0.25 + in.height * 0.5,
                     0.25 + in.normal_y * 0.25,
                     (in.uv.x + in.uv.y) * 0.125, 1.0);
}
"#;

struct TerrainDraw {
    context: GpuContext,
    pipeline: wgpu::RenderPipeline,
    vertex: wgpu::Buffer,
    index: wgpu::Buffer,
    target: wgpu::Texture,
    readback: wgpu::Buffer,
    index_count: u32,
}

impl TerrainDraw {
    fn new(vertices: &[u8], indices: &[u8], index_count: u32) -> Self {
        let context = GpuContext::try_headless()
            .expect("terrain raster GPU acquisition failed")
            .expect("strict terrain raster requires a physical GPU adapter");
        let evidence = context.evidence();
        assert_ne!(evidence.device_type, "Cpu", "physical GPU required");
        if let Some(requested) = wgpu::Backends::from_env() {
            let actual = match evidence.backend.as_str() {
                "Metal" => wgpu::Backends::METAL,
                "Vulkan" => wgpu::Backends::VULKAN,
                "Dx12" => wgpu::Backends::DX12,
                other => panic!("unexpected native backend {other}"),
            };
            assert!(requested.contains(actual), "WGPU_BACKEND was not honored");
        }
        eprintln!("[terrain_draw] adapter={evidence:?}");
        let device = context.device();
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("terrain mesh raster regression"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let attributes = wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x2];
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("terrain mesh raster regression"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: STRIDE as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &attributes,
                })],
            },
            primitive: wgpu::PrimitiveState {
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        let vertex = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("terrain native mesh vertex bytes"),
            contents: vertices,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });
        let index = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("terrain native mesh indices"),
            contents: indices,
            usage: wgpu::BufferUsages::INDEX,
        });
        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("terrain offscreen target"),
            size: wgpu::Extent3d {
                width: SIDE,
                height: SIDE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("terrain RGBA readback"),
            size: u64::from(SIDE * SIDE * 4),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            context,
            pipeline,
            vertex,
            index,
            target,
            readback,
            index_count,
        }
    }

    fn draw(&self) -> Vec<u8> {
        let device = self.context.device();
        let view = self.target.create_view(&Default::default());
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("terrain indexed draw"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_vertex_buffer(0, self.vertex.slice(..));
            pass.set_index_buffer(self.index.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..self.index_count, 0, 0..1);
        }
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(SIDE * 4),
                    rows_per_image: Some(SIDE),
                },
            },
            wgpu::Extent3d {
                width: SIDE,
                height: SIDE,
                depth_or_array_layers: 1,
            },
        );
        self.context.queue().submit([encoder.finish()]);
        let (sender, receiver) = std::sync::mpsc::channel();
        self.readback
            .map_async(wgpu::MapMode::Read, .., move |result| {
                sender.send(result).expect("readback receiver");
            });
        self.context.wait();
        receiver
            .recv()
            .expect("readback callback")
            .expect("map terrain output");
        let bytes = self
            .readback
            .get_mapped_range(..)
            .expect("mapped terrain range")
            .to_vec();
        self.readback.unmap();
        bytes
    }
}

fn pixel(bytes: &[u8], x: usize, y: usize) -> &[u8] {
    let offset = (y * SIDE as usize + x) * 4;
    &bytes[offset..offset + 4]
}

#[test]
fn native_terrain_draw_and_local_vertex_upload_change_only_the_patch() {
    let tile = TileKey::root(CubeFace::PositiveY);
    let mut field = HeightfieldTile::bake(
        tile,
        HeightfieldSpec::new(9, 1).unwrap(),
        &ConstantHeight(0),
        LIMITS,
    )
    .unwrap();
    let mut geometry = HeightfieldMesh::build(&field, [2.0, 2.0], LIMITS).unwrap();
    let native = geometry.clone().into_mesh().unwrap();
    let vertices = native.vertex_buffer_data();
    let indices = native.indices().expect("indexed terrain");
    assert_eq!(vertices.len(), 81 * STRIDE);
    assert_eq!(indices.len(), 384);
    let draw = TerrainDraw::new(&vertices, &indices.to_bytes(), indices.len() as u32);
    let flat = draw.draw();
    // All triangles must survive backface culling and cover the whole target.
    for rgba in flat.chunks_exact(4) {
        assert_eq!(rgba[0], 64);
        assert_eq!(rgba[1], 128);
        assert_eq!(rgba[3], 255);
    }
    assert!(pixel(&flat, 60, 60)[2] > pixel(&flat, 3, 3)[2]);

    let change = field
        .rebake(
            DirtyRegion::new(tile, 24_576, 24_576, 40_959, 40_959).unwrap(),
            &ConstantHeight(1_000),
            LIMITS,
        )
        .unwrap();
    let bounds = geometry.update(&field, change).unwrap();
    assert_eq!(change.bounds().min(), [3, 3]);
    assert_eq!(change.bounds().max(), [5, 5]);
    assert_eq!(bounds.min(), [2, 2]);
    assert_eq!(bounds.max(), [6, 6]);
    let changed = geometry.into_mesh().unwrap().vertex_buffer_data();
    let mut uploaded = 0;
    // Only five affected row spans are copied into the existing GPU allocation.
    for row in bounds.min()[1]..=bounds.max()[1] {
        let start = (row as usize * 9 + bounds.min()[0] as usize) * STRIDE;
        let len = (bounds.max()[0] - bounds.min()[0] + 1) as usize * STRIDE;
        draw.context
            .queue()
            .write_buffer(&draw.vertex, start as u64, &changed[start..start + len]);
        uploaded += len;
    }
    assert_eq!(uploaded, 800);
    assert!(uploaded < vertices.len());
    let raised = draw.draw();
    assert_eq!(pixel(&raised, 32, 32)[0], 191);
    assert!(pixel(&raised, 32, 32)[1] > 100);
    assert_eq!(pixel(&flat, 3, 3), pixel(&raised, 3, 3));
    assert_eq!(pixel(&flat, 60, 60), pixel(&raised, 60, 60));
    assert_ne!(flat, raised);
    // Reference a full replacement into the same buffer; all pixels must match
    // the incremental upload, including normal-dependent shading boundaries.
    draw.context.queue().write_buffer(&draw.vertex, 0, &changed);
    assert_eq!(raised, draw.draw());
    // Host encode, submit, GPU execution, synchronization, mapping and copy;
    // these are end-to-end wall times, not hardware GPU timestamps.
    let mut wall_samples = Vec::with_capacity(64);
    for iteration in 0..72 {
        let start = std::time::Instant::now();
        let pixels = draw.draw();
        let elapsed = start.elapsed();
        assert_eq!(pixels, raised);
        if iteration >= 8 {
            wall_samples.push(elapsed);
        }
    }
    wall_samples.sort_unstable();
    let quantile =
        |percent: usize| wall_samples[(wall_samples.len() * percent).div_ceil(100) - 1].as_micros();
    eprintln!(
        "[terrain_draw_wall] samples=64 warmup=8 p50_us={} p95_us={} p99_us={}",
        quantile(50),
        quantile(95),
        quantile(99)
    );
    eprintln!("[terrain_draw] vertices=81 indices=384 full_upload={} patch_upload={uploaded} rgba_readback={}", vertices.len(), raised.len());
}

#[test]
fn adjacent_coarse_lod_corner_fan_rasterizes_with_backface_culling() {
    let tile = TileKey::new(CubeFace::PositiveZ, 3, 3, 2).unwrap();
    let field = HeightfieldTile::bake(
        tile,
        HeightfieldSpec::new(9, 1).unwrap(),
        &ConstantHeight(0),
        LIMITS,
    )
    .unwrap();
    let limits = LodSeamLimits {
        max_segments: 8,
        max_vertices: 128,
        max_indices: 1024,
        max_bytes: 16_384,
    };
    let neighbors = [TileEdge::East, TileEdge::South].map(|edge| {
        LodSeamNeighbor::with_boundary(
            edge,
            tile.neighbor(edge).parent().unwrap(),
            8,
            LodSeamCoverage::full(8),
        )
    });
    let plan = HeightfieldLodSeamPlan::build_strict(tile, 8, &neighbors, limits).unwrap();
    let mut geometry = HeightfieldMesh::build(&field, [2.0, 2.0], LIMITS).unwrap();
    geometry.apply_lod_seam_plan(&plan, 16_384).unwrap();
    assert_eq!(geometry.indices().len(), 360);
    let native = geometry.into_mesh().unwrap();
    let draw = TerrainDraw::new(
        &native.vertex_buffer_data(),
        &native.indices().unwrap().to_bytes(),
        native.indices().unwrap().len() as u32,
    );
    let rgba = draw.draw();
    for (pixel_index, pixel) in rgba.chunks_exact(4).enumerate() {
        assert_eq!(
            pixel[0], 64,
            "unculled terrain coverage at pixel {pixel_index}"
        );
        assert_eq!(pixel[1], 128, "terrain normal at pixel {pixel_index}");
        assert_eq!(pixel[3], 255, "terrain alpha at pixel {pixel_index}");
    }
    eprintln!(
        "[terrain_lod_corner_gpu] adapter={:?} vertices={} indices={} rgba_readback={}",
        draw.context.evidence(),
        native.vertex_count(),
        native.indices().unwrap().len(),
        rgba.len()
    );
}
