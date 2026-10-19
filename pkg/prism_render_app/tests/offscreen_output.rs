//! Public offscreen output contract and physical-GPU display-transform tests.

use prism_app::App;
use prism_ecs::world::World;
use prism_render_app::prelude::{
    DisplayTransformGpu, OffscreenFrameOutput, PrismRenderPlugin, RenderApp, RenderAppConfig,
};
use prism_render_architecture::lighting::ocio::{
    OcioConfig, OcioDisplayGamut, OcioToneMapping, OcioTransferFunction,
};
use prism_render_driver::{
    Extent3d, ImageDataLayout, RenderDevice, RenderQueue, TextureDescriptor, TextureState,
    TextureUsages, TextureViewDescriptor, TextureWrite,
};
use prism_render_driver_null::NullInstance;
use prism_render_graph::{ImportedTexture, RenderGraph, TextureDesc};
use std::sync::{
    atomic::{AtomicBool, AtomicUsize},
    Arc,
};

#[test]
fn render_app_publishes_latest_offscreen_output_through_public_contract() {
    let extent = Extent3d::new_2d(11, 7);
    let config = RenderAppConfig::new(extent).with_display_transform(OcioConfig::srgb());
    let mut app = App::new();
    app.add_plugins(PrismRenderPlugin::with_config(NullInstance::new(), config));
    app.update();

    let render_world: &World = &app
        .get_sub_app(RenderApp)
        .expect("render sub-app is installed")
        .world;
    let frame = render_world
        .resource::<OffscreenFrameOutput>()
        .latest()
        .expect("the offscreen host publishes a submitted frame");
    assert_eq!(frame.extent(), extent);
    assert_eq!(frame.frame_index(), 1);
    assert_eq!(frame.device_generation(), 0);
    assert_ne!(frame.submission(), 0);
}

#[test]
fn physical_gpu_offscreen_transform_matches_cpu_for_non_black_pixels() {
    use prism_render_driver_wgpu::{require_hardware_gpu, HardwareDeviceError, WgpuInstance};

    let (device, queue) = match WgpuInstance::new().request_default_hardware_device() {
        Ok(pair) => pair,
        Err(HardwareDeviceError::NoAdapter) => {
            assert!(
                !require_hardware_gpu(),
                "PRISM_REQUIRE_GPU=1 but no physical GPU adapter was found"
            );
            eprintln!("SKIPPED offscreen OCIO parity: no adapter");
            return;
        }
        Err(HardwareDeviceError::NonHardware(evidence)) => {
            assert!(
                !require_hardware_gpu(),
                "PRISM_REQUIRE_GPU=1 requires a physical GPU; observed {evidence}"
            );
            eprintln!("SKIPPED offscreen OCIO parity: non-hardware adapter: {evidence}");
            return;
        }
        Err(HardwareDeviceError::Device(error)) => {
            panic!("physical GPU was found but device creation failed: {error}");
        }
    };
    let device = Arc::new(device);
    let device_dyn: Arc<dyn RenderDevice + Send + Sync> = device.clone();
    let extent = Extent3d::new_2d(4, 2);
    let pixels: [[u8; 4]; 8] = [
        [8, 24, 48, 16],
        [32, 64, 128, 64],
        [64, 128, 192, 112],
        [128, 192, 255, 160],
        [16, 80, 160, 208],
        [48, 112, 224, 240],
        [200, 100, 50, 80],
        [255, 128, 32, 255],
    ];
    let format = prism_render_driver::TextureFormat::Rgba8Unorm;
    let source_texture = device.create_texture(&TextureDescriptor::new_2d(
        extent,
        format,
        TextureUsages::TEXTURE_BINDING.union(TextureUsages::COPY_DST),
    ));
    let source_view = device.create_texture_view(source_texture, &TextureViewDescriptor::default());
    queue.write_texture(
        TextureWrite {
            texture: source_texture,
            mip_level: 0,
            origin: [0, 0, 0],
            size: extent,
        },
        bytemuck::cast_slice(&pixels),
        ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(extent.width * 4),
            rows_per_image: Some(extent.height),
        },
    );
    let config = OcioConfig {
        tone_mapping: OcioToneMapping::Aces,
        exposure_stops: 0.4,
        display_gamut: OcioDisplayGamut::DisplayP3,
        transfer_function: OcioTransferFunction::Srgb,
        ..OcioConfig::srgb()
    };
    let transform = DisplayTransformGpu::new(
        device_dyn,
        source_view,
        extent,
        Arc::new(AtomicUsize::new(0)),
        Arc::new(AtomicBool::new(true)),
    )
    .expect("valid display transform resources");
    let mut graph = RenderGraph::new(extent);
    let source = graph.import_texture(
        "offscreen-non-black-source",
        TextureDesc::color(format).with_usage(TextureUsages::TEXTURE_BINDING),
        ImportedTexture {
            texture: source_texture,
            default_view: source_view,
            entry_state: TextureState::initial(),
            exit_state: TextureState::shader_read(
                prism_render_driver::PipelineStages::COMPUTE_SHADER,
            ),
            mip_levels: 1,
            array_layers: 1,
        },
    );
    transform
        .append(&mut graph, &queue, &config, source)
        .expect("valid configured offscreen graph");
    let _recorded_graph = graph
        .execute(&*device, &queue)
        .expect("offscreen display transform executes on the physical GPU");
    let actual = queue.read_texture_rgba8(transform.output_texture(), extent.width, extent.height);
    for (index, (pixel, encoded)) in pixels.iter().zip(actual.chunks_exact(4)).enumerate() {
        let linear = [
            f32::from(pixel[0]) / 255.0,
            f32::from(pixel[1]) / 255.0,
            f32::from(pixel[2]) / 255.0,
        ];
        let expected = config.reference_transform(linear).expect("finite pixel");
        for channel in 0..3 {
            let actual = f32::from(encoded[channel]) / 255.0;
            assert!(
                (actual - expected[channel]).abs() <= 0.012,
                "pixel={index} channel={channel}: GPU={actual} CPU={}",
                expected[channel]
            );
        }
        assert_eq!(encoded[3], pixel[3], "alpha must pass through");
    }
    device.destroy_texture_view(source_view);
    device.destroy_texture(source_texture);
}
