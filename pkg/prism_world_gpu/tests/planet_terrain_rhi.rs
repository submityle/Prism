//! Strict physical-GPU integration coverage for the bounded planet mesh cache.
#![cfg(feature = "planet_terrain_rhi")]

use std::{convert::Infallible, sync::Arc};

use prism_world_gpu::{
    planet_terrain_rhi::{PlanetTerrainRhiCache, PlanetTerrainRhiError, PlanetTerrainRhiLimits},
    GpuContext,
};
use prism_world_terrain::{
    BakeLimits, CubeFace, HeightSample, HeightSampleValue, HeightSampler, HeightfieldSpec,
    PlanetMeshConfig, PlanetMeshPublication, PlanetMeshPublicationLimits, TileEdge, TileKey,
};

struct Flat(i64);

impl HeightSampler for Flat {
    type Error = Infallible;

    fn sample_height(&self, _: HeightSample) -> Result<HeightSampleValue, Self::Error> {
        Ok(HeightSampleValue::from_millimetres(self.0))
    }
}

fn context() -> GpuContext {
    let context = GpuContext::try_headless()
        .expect("planet terrain adapter acquisition failed")
        .expect("this integration test requires a physical GPU; it never silently skips");
    assert!(context.evidence().is_hardware(), "physical GPU required");
    eprintln!("[planet_terrain_rhi] {:?}", context.evidence());
    context
}

fn source(tile: TileKey, height: i64) -> Arc<prism_world_terrain::HeightfieldTile> {
    Arc::new(
        prism_world_terrain::HeightfieldTile::bake(
            tile,
            HeightfieldSpec::new(3, 1).unwrap(),
            &Flat(height),
            BakeLimits {
                max_samples: 9,
                max_bytes: 18,
            },
        )
        .unwrap(),
    )
}

fn limits(max_tiles: usize, max_buffer_bytes: u64) -> PlanetTerrainRhiLimits {
    PlanetTerrainRhiLimits {
        max_tiles,
        max_buffers: max_tiles * 2,
        max_buffer_bytes,
    }
}

#[test]
fn full_snapshot_delta_budget_failure_and_device_identity_are_atomic() {
    let gpu = context();
    let tile_a = TileKey::new(CubeFace::PositiveZ, 1, 0, 0).unwrap();
    let tile_b = tile_a.neighbor(TileEdge::East);
    let mut publication = PlanetMeshPublication::new(PlanetMeshPublicationLimits {
        max_tiles: 4,
        max_mesh_bytes: 16_384,
        max_neighbor_comparisons: 128,
    });
    let mut initial = publication.begin();
    initial
        .stage_build(
            source(tile_a, 0),
            PlanetMeshConfig::new(6_371_000_000, [0; 3], 10_000.0).unwrap(),
            BakeLimits {
                max_samples: 9,
                max_bytes: 4096,
            },
            prism_world_terrain::PlanetHeightfieldNormalNeighborhood::empty(),
        )
        .unwrap();
    let (_, first_delta) = publication.commit_with_delta(initial).unwrap();
    let mut cache = PlanetTerrainRhiCache::new(&gpu, limits(1, 4096));
    let first = cache.apply_snapshot(&gpu, &publication).unwrap();
    assert_eq!((first.generation, first.tiles), (1, 1));
    assert!(matches!(
        cache.apply_delta(&gpu, &publication, &first_delta),
        Err(PlanetTerrainRhiError::Generation {
            expected: 2,
            actual: 1
        })
    ));

    let mut add = publication.begin();
    add.stage_build(
        source(tile_b, 0),
        PlanetMeshConfig::new(6_371_000_000, [0; 3], 10_000.0).unwrap(),
        BakeLimits {
            max_samples: 9,
            max_bytes: 4096,
        },
        prism_world_terrain::PlanetHeightfieldNormalNeighborhood::empty(),
    )
    .unwrap();
    let (_, add_delta) = publication.commit_with_delta(add).unwrap();
    let before = cache.stats();
    assert!(matches!(
        cache.apply_delta(&gpu, &publication, &add_delta),
        Err(PlanetTerrainRhiError::BudgetExceeded { tiles: 2, .. })
    ));
    assert_eq!(
        cache.stats(),
        before,
        "budget failure must retain old GPU cache snapshot"
    );

    // Full snapshot is the explicit recovery/resync path after a rejected delta.
    let resynced = cache.apply_snapshot(&gpu, &publication).unwrap_err();
    assert!(matches!(
        resynced,
        PlanetTerrainRhiError::BudgetExceeded { .. }
    ));
    assert_eq!(
        cache.stats(),
        before,
        "failed snapshot must retain old GPU cache"
    );
    let mut roomy = PlanetTerrainRhiCache::new(&gpu, limits(2, 8192));
    assert_eq!(roomy.apply_snapshot(&gpu, &publication).unwrap().tiles, 2);

    let mut remove = publication.begin();
    remove.stage_remove(tile_b).unwrap();
    let (_, remove_delta) = publication.commit_with_delta(remove).unwrap();
    let removed = roomy
        .apply_delta(&gpu, &publication, &remove_delta)
        .unwrap();
    assert_eq!((removed.generation, removed.tiles), (3, 1));

    // Build two distinct generation-4 states from the exact same generation-3
    // base. A branch delta must not be accepted alongside the main publication.
    let mut fork = publication.clone();
    let mut fork_remove = fork.begin();
    fork_remove.stage_remove(tile_a).unwrap();
    let (_, fork_delta) = fork.commit_with_delta(fork_remove).unwrap();
    publication.commit(publication.begin()).unwrap();
    assert_eq!(publication.generation(), fork.generation());
    assert!(matches!(
        roomy.apply_delta(&gpu, &publication, &fork_delta),
        Err(PlanetTerrainRhiError::TargetStateMismatch)
    ));
    assert_eq!(roomy.stats(), removed, "forked-state delta must be atomic");
    assert!(matches!(
        roomy.apply_delta(&gpu, &publication, &add_delta),
        Err(PlanetTerrainRhiError::Generation {
            expected: 4,
            actual: 2
        })
    ));
    assert_eq!(
        roomy.stats(),
        removed,
        "rejected stale delta must be atomic"
    );

    let foreign = context();
    assert_eq!(
        cache.apply_snapshot(&foreign, &publication),
        Err(PlanetTerrainRhiError::ForeignDevice)
    );
}

#[test]
fn cached_planet_mesh_encodes_an_indexed_draw_to_a_real_target() {
    let gpu = context();
    let tile = TileKey::new(CubeFace::PositiveZ, 0, 0, 0).unwrap();
    let mut publication = PlanetMeshPublication::new(PlanetMeshPublicationLimits {
        max_tiles: 1,
        max_mesh_bytes: 16_384,
        max_neighbor_comparisons: 0,
    });
    let mut transaction = publication.begin();
    transaction
        .stage_build(
            source(tile, 0),
            PlanetMeshConfig::new(6_371_000_000, [0; 3], 10_000.0).unwrap(),
            BakeLimits {
                max_samples: 9,
                max_bytes: 4096,
            },
            prism_world_terrain::PlanetHeightfieldNormalNeighborhood::empty(),
        )
        .unwrap();
    publication.commit(transaction).unwrap();
    let mut cache = PlanetTerrainRhiCache::new(&gpu, limits(1, 4096));
    cache.apply_snapshot(&gpu, &publication).unwrap();

    const SIDE: u32 = 64;
    const SHADER: &str = r#"
        @vertex fn vs(@location(2) uv: vec2<f32>) -> @builtin(position) vec4<f32> {
            return vec4<f32>(uv * 2.0 - vec2<f32>(1.0), 0.5, 1.0);
        }
        @fragment fn fs() -> @location(0) vec4<f32> { return vec4<f32>(0.1, 0.8, 0.2, 1.0); }
    "#;
    let device = gpu.device();
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("planet terrain indexed raster regression"),
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });
    let attributes = wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x2];
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("planet terrain indexed raster regression"),
        layout: None,
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs"),
            compilation_options: Default::default(),
            buffers: &[Some(wgpu::VertexBufferLayout {
                array_stride: 32,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &attributes,
            })],
        },
        primitive: wgpu::PrimitiveState {
            cull_mode: None,
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
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("planet terrain offscreen target"),
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
        label: Some("planet terrain raster readback"),
        size: u64::from(SIDE * SIDE * 4),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let view = target.create_view(&Default::default());
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("planet terrain indexed draw"),
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
        pass.set_pipeline(&pipeline);
        cache.encode_draw(tile, &mut pass).unwrap();
    }
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &target,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
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
    gpu.queue().submit([encoder.finish()]);
    let (sender, receiver) = std::sync::mpsc::channel();
    readback.map_async(wgpu::MapMode::Read, .., move |result| {
        sender.send(result).unwrap();
    });
    gpu.wait();
    receiver.recv().unwrap().unwrap();
    let pixels = readback.get_mapped_range(..).unwrap();
    assert!(pixels.chunks_exact(4).any(|pixel| pixel[1] > 100));
    drop(pixels);
    readback.unmap();
}
