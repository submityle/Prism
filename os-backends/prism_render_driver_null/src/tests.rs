//! Unit tests for the null backend.
//!
//! These run on any host: the backend needs no GPU, and the fake window reports
//! its handle ABI as unavailable (the null surface never inspects it). The
//! tests cover the full bring-up chain, the resource create/destroy lifecycle
//! including generational slot recycling, and one configure → acquire → present
//! cycle.

use alloc::sync::Arc;

use prism_render_driver as rhi;
use rhi::{
    Adapter, Backend, BufferDescriptor, BufferUsages, DeviceDescriptor, DeviceType, Extent3d,
    ImageDataLayout, Instance, PresentMode, RenderDevice, RenderQueue, RequestAdapterOptions,
    Surface, SurfaceConfig, SurfaceError, TextureDescriptor, TextureFormat, TextureUsages,
    TextureWrite,
};

use crate::{NullDevice, NullInstance, NullQueue};

/// A window handle stub that reports the handle ABI as unavailable.
///
/// The null surface accepts any handle and never reads it, so an unavailable
/// handle is a valid stand-in for a real window in a headless test.
struct FakeWindow;

impl raw_window_handle::HasWindowHandle for FakeWindow {
    fn window_handle(
        &self,
    ) -> Result<raw_window_handle::WindowHandle<'_>, raw_window_handle::HandleError> {
        Err(raw_window_handle::HandleError::Unavailable)
    }
}

impl raw_window_handle::HasDisplayHandle for FakeWindow {
    fn display_handle(
        &self,
    ) -> Result<raw_window_handle::DisplayHandle<'_>, raw_window_handle::HandleError> {
        Err(raw_window_handle::HandleError::Unavailable)
    }
}

#[test]
fn instance_brings_up_null_device_and_recycles_slots() {
    let instance = NullInstance::new();

    let adapters = instance.enumerate_adapters();
    assert_eq!(
        adapters.len(),
        1,
        "the null instance exposes exactly one adapter"
    );

    let adapter = instance
        .request_adapter(&RequestAdapterOptions::default())
        .expect("the null instance always has an adapter");
    assert_eq!(adapter.info().backend, Backend::Noop);
    assert_eq!(adapter.info().device_type, DeviceType::Cpu);

    let (device, queue): (NullDevice, NullQueue) = instance
        .create_device(&adapter, &DeviceDescriptor::default())
        .expect("the null adapter accepts the default device request");

    assert_eq!(device.capabilities().backend, Backend::Noop);
    assert_eq!(device.live_resource_count(), 0);

    let buffer = device.create_buffer(&BufferDescriptor::new(
        256,
        BufferUsages::UNIFORM.union(BufferUsages::COPY_DST),
    ));
    let texture = device.create_texture(&TextureDescriptor::new_2d(
        Extent3d::new_2d(64, 64),
        TextureFormat::Rgba8Unorm,
        TextureUsages::TEXTURE_BINDING.union(TextureUsages::COPY_DST),
    ));
    assert_eq!(device.live_resource_count(), 2);

    // The null queue accepts uploads and submissions as no-ops.
    queue.write_buffer(buffer, 0, &[0u8; 16]);
    queue.write_texture(
        TextureWrite {
            texture,
            mip_level: 0,
            origin: [0, 0, 0],
            size: Extent3d::new_2d(64, 64),
        },
        &[0u8; 64 * 64 * 4],
        ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(64 * 4),
            rows_per_image: Some(64),
        },
    );
    queue.submit(&[]);

    device.destroy_buffer(buffer);
    assert_eq!(device.live_resource_count(), 1);

    // Re-creating a buffer reuses the freed slot with a bumped generation, so
    // the destroyed id can never alias the new resource.
    let recycled = device.create_buffer(&BufferDescriptor::new(16, BufferUsages::COPY_SRC));
    assert_eq!(
        recycled.index(),
        buffer.index(),
        "the freed slot is reused"
    );
    assert_ne!(
        recycled.generation(),
        buffer.generation(),
        "the generation bumps on reuse"
    );
    assert_eq!(device.live_resource_count(), 2);

    device.destroy_texture(texture);
    assert_eq!(device.live_resource_count(), 1);

    // Destroying an already-freed id is a safe no-op.
    device.destroy_buffer(buffer);
    assert_eq!(device.live_resource_count(), 1);
}

#[test]
fn surface_configure_acquire_present_cycle() {
    let instance = NullInstance::new();
    let window = Arc::new(FakeWindow);
    let adapter = instance
        .request_adapter(&RequestAdapterOptions::default())
        .expect("the null instance always has an adapter");
    let (device, queue) = instance
        .create_device(&adapter, &DeviceDescriptor::default())
        .expect("the null adapter accepts the default device request");
    let mut surface = instance
        .create_surface(Arc::clone(&window), &device, &queue)
        .expect("the null surface is always creatable");

    let caps = surface.capabilities();
    assert!(caps.formats.contains(&TextureFormat::Rgba8Unorm));
    assert!(caps.present_modes.contains(&PresentMode::Fifo));

    // Acquiring before configuration has no swapchain to hand out.
    assert!(matches!(surface.acquire(), Err(SurfaceError::Outdated)));

    let config = SurfaceConfig::new(TextureFormat::Rgba8Unorm, 1280, 720);
    surface.configure(&config).expect("configure");
    assert_eq!(surface.size(), Extent3d::new_2d(1280, 720));

    let frame = surface.acquire().expect("frame");
    let image = surface.image(&frame);
    assert_eq!(image.format, TextureFormat::Rgba8Unorm);
    assert_eq!(image.size, Extent3d::new_2d(1280, 720));
    assert_eq!(
        device.live_resource_count(),
        2,
        "the frame mints a texture and a view on the device"
    );

    surface.present(frame).expect("present");
    assert_eq!(
        device.live_resource_count(),
        0,
        "present releases the swapchain image"
    );

    // A configuration the surface cannot satisfy is rejected.
    let unsupported = SurfaceConfig {
        format: TextureFormat::R32Float,
        ..config
    };
    assert!(matches!(
        surface.configure(&unsupported),
        Err(SurfaceError::Unsupported)
    ));
}
