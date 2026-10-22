//! Strict hardware integration: planetary producer -> indirect graphics draw -> pixel/count readback.
#![cfg(feature = "placement_rhi")]

use std::{convert::Infallible, num::NonZeroUsize, sync::Arc, time::Duration};

use prism_world_gpu::{
    planet_placement_draw::{
        PlanetPlacementColorTarget, PlanetPlacementDraw, PlanetPlacementDrawParams,
    },
    GpuContext, GpuPlanetPlacement, GpuPlanetPlacementWorkBudget,
};
use prism_world_terrain::{
    BakeLimits, CubeFace, HeightSample, HeightSampleValue, HeightSampler, HeightfieldSpec,
    PlanetPlacementConfig, PlanetPlacementRegion, TileKey,
};

struct Flat;

impl HeightSampler for Flat {
    type Error = Infallible;

    fn sample_height(&self, _: HeightSample) -> Result<HeightSampleValue, Self::Error> {
        Ok(HeightSampleValue::from_millimetres(0))
    }
}

fn context() -> GpuContext {
    let context = GpuContext::try_headless_with_limits(wgpu::Limits {
        max_storage_buffers_per_shader_stage: 7,
        max_bindings_per_bind_group: 8,
        ..wgpu::Limits::downlevel_defaults()
    })
    .expect("planet placement draw adapter acquisition failed")
    .expect("physical GPU required; this integration test never skips");
    assert!(context.evidence().is_hardware());
    eprintln!("[planet_placement_draw] {:?}", context.evidence());
    context
}

fn map(context: &GpuContext, buffer: &wgpu::Buffer) -> Vec<u8> {
    let (sender, receiver) = std::sync::mpsc::channel();
    buffer.map_async(wgpu::MapMode::Read, .., move |result| {
        sender.send(result).unwrap();
    });
    context
        .device()
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
    receiver
        .recv_timeout(Duration::from_secs(30))
        .unwrap()
        .unwrap();
    let bytes = buffer.get_mapped_range(..).unwrap().to_vec();
    buffer.unmap();
    bytes
}

#[test]
fn planet_record_compact_indirect_draw_reaches_real_color_target() {
    let context = context();
    let tile = Arc::new(
        prism_world_terrain::HeightfieldTile::bake(
            TileKey::root(CubeFace::PositiveZ),
            HeightfieldSpec::new(4, 1).unwrap(),
            &Flat,
            BakeLimits {
                max_samples: 16,
                max_bytes: 32,
            },
        )
        .unwrap(),
    );
    let mut producer = GpuPlanetPlacement::new(&context, 4, 16, 3, u64::MAX).unwrap();
    producer
        .write_request_in_frame(
            PlanetPlacementConfig {
                seed: 7,
                version: 3,
                salt: 11,
                radius_mm: 6_371_000_000,
                origin_mm: [0; 3],
                vertical_offset_mm: 0,
                acceptance_threshold: u16::MAX,
            },
            PlanetPlacementRegion::new(tile.tile(), [0, 0], [2, 2]).unwrap(),
            &tile,
            [0; 3],
            GpuPlanetPlacementWorkBudget {
                max_candidate_evaluations: 4,
                max_sample_reads: 16,
            },
        )
        .unwrap();
    let snapshot = producer.registration_snapshot().unwrap();
    assert_eq!(snapshot.records.bytes(), 4 * 116);
    assert_eq!(snapshot.compact.bytes(), 16);
    let indirect = snapshot.indirect.buffer().clone();

    // Producer compute precedes draw on one native queue; no CPU readback is
    // inserted between the compute and graphics consumers.
    let mut compute = context
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("planet placement producer before draw"),
        });
    snapshot.encode(&mut compute);
    context.queue().submit([compute.finish()]);

    let width = 64;
    let height = 64;
    let texture = context.device().create_texture(&wgpu::TextureDescriptor {
        label: Some("planet placement draw integration target"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let mut draw = PlanetPlacementDraw::new(
        &context,
        wgpu::TextureFormat::Rgba8Unorm,
        NonZeroUsize::new(1).unwrap(),
    );
    let frame_view = [
        [1.0e-7, 0.0, 0.0, 0.0],
        [0.0, 1.0e-7, 0.0, 0.0],
        [0.0, 0.0, 1.0e-7, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let receipt = draw
        .submit(
            snapshot,
            PlanetPlacementColorTarget {
                view: &view,
                format: wgpu::TextureFormat::Rgba8Unorm,
                width,
                height,
                sample_count: 1,
            },
            PlanetPlacementDrawParams {
                frame_view,
                instance_size_pixels: 10.0,
            },
        )
        .unwrap();
    assert_eq!(draw.pending_count(), 1);

    let row_bytes = width * 4;
    let padded_row_bytes = 256;
    let image_readback = context.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("planet placement draw pixel readback"),
        size: u64::from(padded_row_bytes * height),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let indirect_readback = context.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("planet placement draw count readback"),
        size: 16,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut copy = context
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("planet placement draw output readback"),
        });
    copy.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &image_readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_row_bytes),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    copy.copy_buffer_to_buffer(&indirect, 0, &indirect_readback, 0, 16);
    context.queue().submit([copy.finish()]);

    let pixels = map(&context, &image_readback);
    let counts = map(&context, &indirect_readback);
    let live_instances = u32::from_le_bytes(counts[4..8].try_into().unwrap());
    assert_eq!(
        live_instances, 4,
        "GPU compact count drives indirect instances"
    );
    let changed_pixels = pixels
        .chunks_exact(padded_row_bytes as usize)
        .flat_map(|row| row[..row_bytes as usize].chunks_exact(4))
        .filter(|rgba| rgba[0] != 0 || rgba[1] != 0 || rgba[2] != 0)
        .count();
    assert!(changed_pixels >= 12, "indirect draw must rasterize pixels");
    assert!(receipt.is_complete());
    assert_eq!(draw.poll(), 1);
    assert_eq!(draw.pending_count(), 0);
}
