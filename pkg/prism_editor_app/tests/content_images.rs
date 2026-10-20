//! Real PNG -> asynchronous UI page -> keyed Loom geometry -> GPU attachment.
#![cfg(feature = "desktop")]
extern crate alloc;
use alloc::sync::Arc;
use prism_editor_app::{
    ContentThumbnailSource, DesktopContentImages, DesktopGpu, EditorController, EditorDesktopUi,
};
use prism_editor_content::{
    thumbnail::{ThumbnailKey, ThumbnailLimits, ThumbnailService},
    AssetId, AssetKind, AssetRecord, ContentCatalog,
};
use prism_editor_ui::{ContentBrowserAction, ContentLayout};
use prism_ui_layout::{Point, Rect, Size};
use std::{
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
    time::{Duration, Instant},
};

struct Source {
    root: PathBuf,
    revision: Arc<AtomicU64>,
}
impl ContentThumbnailSource for Source {
    fn key(&self, asset: &AssetRecord, size: [u32; 2]) -> Option<ThumbnailKey> {
        Some(ThumbnailKey {
            asset: asset.id,
            source: self.root.join(&asset.path),
            revision: u128::from(self.revision.load(Ordering::Acquire)),
            width: size[0],
            height: size[1],
        })
    }
}

static NEXT: AtomicU64 = AtomicU64::new(0);
struct Fixture(PathBuf);
impl Fixture {
    fn new() -> Self {
        let root = std::env::temp_dir().join(format!(
            "prism-app-image-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&root).unwrap();
        Self(root)
    }
    fn png(&self, color: [u8; 4]) {
        let mut encoder = png::Encoder::new(
            std::fs::File::create(self.0.join("real.png")).unwrap(),
            4,
            2,
        );
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        encoder
            .write_header()
            .unwrap()
            .write_image_data(&color.repeat(8))
            .unwrap();
    }
    fn ui(&self, revision: Arc<AtomicU64>) -> EditorDesktopUi {
        let mut catalog = ContentCatalog::new();
        catalog
            .insert(
                AssetRecord::new(
                    AssetId::from_raw(7),
                    "real.png",
                    AssetKind::new("image").unwrap(),
                )
                .unwrap(),
            )
            .unwrap();
        let mut ui = EditorDesktopUi::new(EditorController::new()).unwrap();
        ui.mount_content_browser(catalog).unwrap();
        ui.content_action(ContentBrowserAction::Layout(ContentLayout::Grid {
            columns: 4,
        }))
        .unwrap();
        ui.mount_content_thumbnails(
            ThumbnailService::new(ThumbnailLimits::default()).unwrap(),
            Box::new(Source {
                root: self.0.clone(),
                revision,
            }),
        )
        .unwrap();
        ui
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).unwrap();
    }
}

fn settle(ui: &mut EditorDesktopUi) {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        ui.refresh().unwrap();
        if !ui.content_thumbnails_pending() {
            break;
        }
        assert!(Instant::now() < deadline, "nonblocking thumbnail deadline");
        std::thread::yield_now();
    }
}

#[expect(
    clippy::print_stderr,
    reason = "physical adapter evidence and explicit optional skip"
)]
fn gpu() -> Option<DesktopGpu> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let required = std::env::var("PRISM_REQUIRE_GPU").as_deref() == Ok("1");
    let adapter =
        match futures_lite::future::block_on(instance.request_adapter(&Default::default())) {
            Ok(adapter) => adapter,
            Err(error) => {
                assert!(!required, "required content GPU: {error}");
                eprintln!("SKIP content GPU: {error}");
                return None;
            }
        };
    eprintln!("Native content GPU: {:?}", adapter.get_info());
    let (device, queue) =
        futures_lite::future::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_limits: adapter.limits(),
            ..Default::default()
        }))
        .unwrap();
    Some(DesktopGpu {
        adapter,
        device,
        queue,
        format: wgpu::TextureFormat::Rgba8Unorm,
    })
}

fn target(gpu: &DesktopGpu, extent: [u32; 2]) -> wgpu::Texture {
    gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("native thumbnail pixels"),
        size: wgpu::Extent3d {
            width: extent[0],
            height: extent[1],
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: gpu.format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

fn read(gpu: &DesktopGpu, texture: &wgpu::Texture) -> Vec<u8> {
    let stride = (texture.width() * 4).div_ceil(256) * 256;
    let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("thumbnail readback"),
        size: u64::from(stride) * u64::from(texture.height()),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(stride),
                rows_per_image: Some(texture.height()),
            },
        },
        texture.size(),
    );
    gpu.queue.submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    buffer
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
    rx.recv_timeout(Duration::from_secs(10)).unwrap().unwrap();
    let mapped = buffer.slice(..).get_mapped_range().unwrap();
    mapped
        .chunks(stride as usize)
        .flat_map(|row| row[..texture.width() as usize * 4].iter().copied())
        .collect()
}

fn paint(
    gpu: &DesktopGpu,
    images: &mut DesktopContentImages,
    ui: &EditorDesktopUi,
    texture: &wgpu::Texture,
    scale: f32,
    loom: bool,
) -> Vec<u8> {
    images
        .prepare(
            ui.content_images(),
            scale,
            [texture.width(), texture.height()],
        )
        .unwrap();
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    let view = texture.create_view(&Default::default());
    if loom {
        let mut presenter = prism_ui_render_backend::GpuPresenter::new(
            &gpu.device,
            &gpu.queue,
            gpu.format,
            131_072,
        )
        .unwrap();
        presenter
            .prepare(
                &gpu.device,
                &gpu.queue,
                ui.draw_list(),
                texture.width(),
                texture.height(),
            )
            .unwrap();
        presenter
            .encode(&mut encoder, &view, Some(wgpu::Color::BLACK))
            .unwrap();
    } else {
        let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
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
    }
    images.encode(&mut encoder, texture).unwrap();
    gpu.queue.submit([encoder.finish()]);
    read(gpu, texture)
}

fn pixel(data: &[u8], width: u32, x: f32, y: f32) -> &[u8] {
    let index = (y as usize * width as usize + x as usize) * 4;
    &data[index..index + 4]
}

#[test]
fn actual_png_native_grid_pixels_dpi_clip_revision_idle_and_renderer_reconstruction() {
    let Some(gpu) = gpu() else {
        return;
    };
    let fixture = Fixture::new();
    fixture.png([255, 0, 0, 255]);
    let revision = Arc::new(AtomicU64::new(1));
    let mut ui = fixture.ui(revision.clone());
    settle(&mut ui);
    assert_eq!(
        ui.content_images().len(),
        1,
        "decoded image must have a real keyed layout anchor"
    );
    assert_eq!(
        (
            ui.content_images()[0].image.width(),
            ui.content_images()[0].image.height()
        ),
        (4, 2)
    );
    assert_eq!(ui.content_images()[0].rect.size, Size::new(128.0, 64.0));
    // Scroll the complete tree, including collections/header/footer, inside the dock.
    ui.scroll_at(10.0, 600.0, i32::MAX).unwrap();
    let image = &ui.content_images()[0];
    assert!(
        image.rect.top() >= image.clip.top(),
        "image reachable through native scrolling: {image:?}"
    );
    assert!(image.rect.bottom() <= image.clip.bottom());
    let center = [image.rect.left() + 64.0, image.rect.top() + 32.0];
    let mut images = DesktopContentImages::new(&gpu).unwrap();
    let texture = target(&gpu, [1280, 800]);
    let pixels = paint(&gpu, &mut images, &ui, &texture, 1.0, true);
    assert_eq!(pixel(&pixels, 1280, center[0], center[1]), [255, 0, 0, 255]);
    assert_ne!(
        pixel(&pixels, 1280, 700.0, 40.0),
        [255, 0, 0, 255],
        "image must not overwrite viewport/toolbar"
    );
    let counts = (images.texture_uploads(), images.geometry_uploads());
    assert!(!ui.refresh().unwrap());
    assert_eq!(paint(&gpu, &mut images, &ui, &texture, 1.0, true), pixels);
    assert_eq!(
        (images.texture_uploads(), images.geometry_uploads()),
        counts
    );

    // Same catalog generation, explicitly new source revision: a real new PNG.
    let generation = ui.content_catalog().unwrap().generation();
    fixture.png([0, 255, 0, 255]);
    revision.store(2, Ordering::Release);
    ui.invalidate_content_thumbnails();
    assert!(
        ui.content_images().is_empty(),
        "publication revokes old CPU visuals immediately"
    );
    settle(&mut ui);
    assert_eq!(ui.content_catalog().unwrap().generation(), generation);
    assert_eq!(ui.content_images()[0].request.revision, 2);
    let pixels = paint(&gpu, &mut images, &ui, &texture, 1.0, true);
    assert_eq!(pixel(&pixels, 1280, center[0], center[1]), [0, 255, 0, 255]);
    assert_eq!(images.texture_uploads(), counts.0 + 1);
    assert_eq!(images.usage(), (1, 32));

    let scaled = target(&gpu, [2560, 1600]);
    let pixels = paint(&gpu, &mut images, &ui, &scaled, 2.0, false);
    assert_eq!(
        pixel(&pixels, 2560, center[0] * 2.0, center[1] * 2.0),
        [0, 255, 0, 255]
    );
    // Partial dock clipping uses reconciled coordinates, without stretching UVs.
    let mut clipped = ui.content_images().to_vec();
    clipped[0].clip = Rect::new(
        Point::new(center[0], clipped[0].clip.top()),
        clipped[0].clip.size,
    );
    images.prepare(&clipped, 2.0, [2560, 1600]).unwrap();
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    {
        let view = scaled.create_view(&Default::default());
        let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
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
    }
    images.encode(&mut encoder, &scaled).unwrap();
    gpu.queue.submit([encoder.finish()]);
    let pixels = read(&gpu, &scaled);
    assert_eq!(
        pixel(&pixels, 2560, center[0] * 2.0 - 20.0, center[1] * 2.0),
        [0, 0, 0, 255]
    );
    assert_eq!(
        pixel(&pixels, 2560, center[0] * 2.0 + 20.0, center[1] * 2.0),
        [0, 255, 0, 255]
    );
    // Renderer reconstruction starts a new residency domain on the same device.
    let mut recreated = DesktopContentImages::new(&gpu).unwrap();
    let pixels = paint(&gpu, &mut recreated, &ui, &texture, 1.0, false);
    assert_eq!(pixel(&pixels, 1280, center[0], center[1]), [0, 255, 0, 255]);
    assert_eq!(recreated.texture_uploads(), 1);
    ui.content_catalog_mut()
        .unwrap()
        .remove(AssetId::from_raw(7))
        .unwrap();
    ui.refresh().unwrap();
    assert!(ui.content_images().is_empty());
    let pixels = paint(&gpu, &mut recreated, &ui, &texture, 1.0, false);
    assert_eq!(recreated.usage(), (0, 0));
    assert!(pixels.chunks_exact(4).all(|p| p == [0, 0, 0, 255]));
    assert_eq!(ui.controller().document_history_lengths(), (0, 0));
    assert!(!ui.controller().is_dirty());
}

#[test]
fn failed_source_and_unmount_never_produce_image_visuals() {
    let fixture = Fixture::new();
    let mut ui = fixture.ui(Arc::new(AtomicU64::new(1)));
    settle(&mut ui);
    assert!(ui.content_images().is_empty());
    assert!(!ui.content_thumbnails_pending());
    ui.unmount_content_thumbnails().unwrap();
    assert!(ui.content_images().is_empty());
    assert!(!ui.refresh().unwrap());
}
