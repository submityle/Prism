//! Pixel acceptance for the same compositor used by the native editor.

use crate::image::{ImageError, ImageHandle, ImageLimits, ImageQuad, ImageRenderer};
use crate::test_support;

const LINEAR: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
}
impl Gpu {
    fn new() -> Option<Self> {
        let (_, device, queue) = test_support::device()?;
        Some(Self { device, queue })
    }
    fn renderer(&self, format: wgpu::TextureFormat, limits: ImageLimits) -> ImageRenderer {
        ImageRenderer::new(&self.device, &self.queue, format, limits).expect("image renderer")
    }
    fn target(&self, size: [u32; 2], format: wgpu::TextureFormat) -> wgpu::Texture {
        self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("image test output"),
            size: wgpu::Extent3d {
                width: size[0],
                height: size[1],
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        })
    }
    fn encoder(&self) -> wgpu::CommandEncoder {
        self.device.create_command_encoder(&Default::default())
    }
    fn pixels(
        &self,
        renderer: &ImageRenderer,
        target: &wgpu::Texture,
        clear: wgpu::Color,
    ) -> Vec<[u8; 4]> {
        assert!(matches!(
            target.format(),
            LINEAR | wgpu::TextureFormat::Rgba8UnormSrgb
        ));
        let mut encoder = self.encoder();
        let view = target.create_view(&Default::default());
        drop(encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(clear),
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        }));
        renderer
            .encode(&mut encoder, target)
            .expect("encode images");
        let stride = (target.width() * 4).div_ceil(256) * 256;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("image test pixel readback"),
            size: u64::from(stride) * u64::from(target.height()),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(stride),
                    rows_per_image: Some(target.height()),
                },
            },
            target.size(),
        );
        self.queue.submit([encoder.finish()]);
        let (sender, receiver) = std::sync::mpsc::sync_channel(1);
        buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                sender.send(result).expect("callback receiver");
            });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("GPU completion");
        receiver.recv().expect("callback").expect("mapping");
        let mapped = buffer.slice(..).get_mapped_range().expect("pixels");
        let mut result = Vec::new();
        for row in mapped.chunks_exact(stride as usize) {
            for pixel in row[..target.width() as usize * 4].chunks_exact(4) {
                result.push(pixel.try_into().expect("RGBA"));
            }
        }
        drop(mapped);
        buffer.unmap();
        result
    }
}
fn quad(image: &ImageHandle, rect: [f32; 4]) -> ImageQuad {
    ImageQuad {
        image: image.clone(),
        rect,
        clip: [-100.0, -100.0, 200.0, 200.0],
        opacity: 1.0,
    }
}
fn near(actual: [u8; 4], expected: [u8; 4]) {
    for (a, e) in actual.into_iter().zip(expected) {
        assert!(
            a.abs_diff(e) <= 2,
            "pixel {actual:?}, expected {expected:?}"
        );
    }
}

#[test]
fn image_gpu_orientation_srgb_transfer_and_unchanged_geometry() {
    let Some(gpu) = Gpu::new() else { return };
    let mut renderer = gpu.renderer(LINEAR, ImageLimits::default());
    let image = renderer
        .upload(
            2,
            2,
            &[
                255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 128, 128, 128, 255,
            ],
        )
        .expect("upload");
    let quads = [quad(&image, [1.0, 1.0, 2.0, 2.0])];
    renderer.prepare([4, 4], &quads).expect("prepare");
    assert_eq!(renderer.geometry_uploads(), 1);
    let target = gpu.target([4, 4], LINEAR);
    let pixels = gpu.pixels(&renderer, &target, wgpu::Color::TRANSPARENT);
    near(pixels[5], [255, 0, 0, 255]);
    near(pixels[6], [0, 255, 0, 255]);
    near(pixels[9], [0, 0, 255, 255]);
    near(pixels[10], [55, 55, 55, 255]);
    assert_eq!(pixels[0], [0; 4]);
    renderer.prepare([4, 4], &quads).expect("same frame");
    assert_eq!(renderer.geometry_uploads(), 1);
    assert_eq!(
        gpu.pixels(&renderer, &target, wgpu::Color::TRANSPARENT),
        pixels
    );
    renderer.prepare([8, 8], &quads).expect("resize");
    assert_eq!(renderer.geometry_uploads(), 2);
    let large = gpu.target([8, 8], LINEAR);
    near(
        gpu.pixels(&renderer, &large, wgpu::Color::TRANSPARENT)[18],
        [55, 55, 55, 255],
    );

    let mut srgb = gpu.renderer(wgpu::TextureFormat::Rgba8UnormSrgb, ImageLimits::default());
    let gray = srgb
        .upload(1, 1, &[128, 128, 128, 255])
        .expect("sRGB source");
    srgb.prepare([1, 1], &[quad(&gray, [0.0, 0.0, 1.0, 1.0])])
        .expect("sRGB prepare");
    near(
        gpu.pixels(
            &srgb,
            &gpu.target([1, 1], wgpu::TextureFormat::Rgba8UnormSrgb),
            wgpu::Color::TRANSPARENT,
        )[0],
        [128, 128, 128, 255],
    );
}

#[test]
fn image_gpu_linear_premultiplied_filter_has_no_transparent_color_fringe() {
    let Some(gpu) = Gpu::new() else { return };
    let mut renderer = gpu.renderer(LINEAR, ImageLimits::default());
    let image = renderer
        .upload(2, 1, &[255, 0, 0, 0, 0, 255, 0, 255])
        .expect("transparent red and opaque green");
    renderer
        .prepare([4, 1], &[quad(&image, [0.0, 0.0, 4.0, 1.0])])
        .expect("prepare");
    let pixels = gpu.pixels(
        &renderer,
        &gpu.target([4, 1], LINEAR),
        wgpu::Color::TRANSPARENT,
    );
    for (actual, expected) in
        pixels
            .into_iter()
            .zip([[0; 4], [0, 64, 0, 64], [0, 191, 0, 191], [0, 255, 0, 255]])
    {
        near(actual, expected);
    }
    let ramp = renderer
        .upload(2, 1, &[0, 0, 0, 255, 255, 255, 255, 255])
        .expect("ramp");
    renderer
        .prepare([3, 1], &[quad(&ramp, [0.0, 0.0, 3.0, 1.0])])
        .expect("linear midpoint");
    near(
        gpu.pixels(
            &renderer,
            &gpu.target([3, 1], LINEAR),
            wgpu::Color::TRANSPARENT,
        )[1],
        [128, 128, 128, 255],
    );
}

#[test]
fn image_gpu_paint_order_opacity_fractional_clipping_and_load_preservation() {
    let Some(gpu) = Gpu::new() else { return };
    let mut renderer = gpu.renderer(LINEAR, ImageLimits::default());
    let red = renderer.upload(1, 1, &[255, 0, 0, 255]).expect("red");
    let green = renderer.upload(1, 1, &[0, 255, 0, 255]).expect("green");
    let base = quad(&red, [0.0, 0.0, 4.0, 4.0]);
    let mut overlay = quad(&green, [0.0, 0.0, 4.0, 4.0]);
    overlay.opacity = 0.5;
    overlay.clip = [1.6, 0.6, 1.9, 2.9];
    renderer
        .prepare([4, 4], &[base.clone(), overlay.clone()])
        .expect("layers");
    let target = gpu.target([4, 4], LINEAR);
    let pixels = gpu.pixels(&renderer, &target, wgpu::Color::BLUE);
    for y in 0..4 {
        for x in 0..4 {
            near(
                pixels[y * 4 + x],
                if x == 2 && (1..3).contains(&y) {
                    [128, 128, 0, 255]
                } else {
                    [255, 0, 0, 255]
                },
            );
        }
    }
    renderer
        .prepare([4, 4], &[overlay.clone(), base])
        .expect("reverse paint order");
    assert!(gpu
        .pixels(&renderer, &target, wgpu::Color::BLUE)
        .iter()
        .all(|p| *p == [255, 0, 0, 255]));
    renderer
        .prepare([4, 4], &[overlay])
        .expect("load existing background");
    let pixels = gpu.pixels(&renderer, &target, wgpu::Color::BLUE);
    near(pixels[6], [0, 128, 128, 255]);
    assert_eq!(pixels[0], [0, 0, 255, 255]);
    renderer.prepare([4, 4], &[]).expect("empty frame");
    assert!(gpu
        .pixels(&renderer, &target, wgpu::Color::BLUE)
        .iter()
        .all(|p| *p == [0, 0, 255, 255]));
}

#[test]
fn image_gpu_budgets_invalid_inputs_and_stale_handles_preserve_valid_frame() {
    let Some(gpu) = Gpu::new() else { return };
    let limits = ImageLimits {
        max_images: 2,
        max_texture_bytes: 8,
        max_image_edge: 2,
        max_quads: 2,
    };
    let mut renderer = gpu.renderer(LINEAR, limits);
    assert!(matches!(
        renderer.upload(u32::MAX, u32::MAX, &[]),
        Err(ImageError::InvalidImage)
    ));
    assert!(matches!(
        renderer.upload(0, 1, &[]),
        Err(ImageError::InvalidImage)
    ));
    assert!(matches!(
        renderer.upload(1, 1, &[0; 3]),
        Err(ImageError::InvalidImage)
    ));
    let red = renderer.upload(1, 1, &[255, 0, 0, 255]).expect("red");
    let other = renderer.upload(1, 1, &[0, 255, 0, 255]).expect("green");
    assert!(matches!(
        renderer.upload(1, 1, &[0; 4]),
        Err(ImageError::Capacity)
    ));
    assert_eq!(renderer.usage(), (2, 8));
    let good = quad(&red, [0.0, 0.0, 2.0, 2.0]);
    renderer
        .prepare([2, 2], core::slice::from_ref(&good))
        .expect("valid frame");
    for invalid in [f32::NAN, f32::INFINITY, -1.0] {
        let mut bad = good.clone();
        bad.rect[2] = invalid;
        assert!(matches!(
            renderer.prepare([2, 2], &[bad]),
            Err(ImageError::InvalidGeometry)
        ));
    }
    assert!(matches!(
        renderer.prepare([0, 2], &[]),
        Err(ImageError::InvalidGeometry)
    ));
    assert!(matches!(
        renderer.prepare([2, 2], &[good.clone(), good.clone(), good.clone()]),
        Err(ImageError::Capacity)
    ));
    let mut foreign = gpu.renderer(LINEAR, limits);
    let foreign_handle = foreign.upload(1, 1, &[0; 4]).expect("foreign handle");
    assert!(matches!(
        renderer.prepare([2, 2], &[quad(&foreign_handle, [0.0, 0.0, 1.0, 1.0])]),
        Err(ImageError::StaleHandle)
    ));
    assert_eq!(renderer.geometry_uploads(), 1);
    renderer.remove(&other).expect("unreferenced remove");
    let target = gpu.target([2, 2], LINEAR);
    assert!(gpu
        .pixels(&renderer, &target, wgpu::Color::TRANSPARENT)
        .iter()
        .all(|p| *p == [255, 0, 0, 255]));
    renderer.remove(&red).expect("referenced remove");
    assert!(matches!(
        renderer.encode(&mut gpu.encoder(), &target),
        Err(ImageError::NotPrepared)
    ));
    assert!(matches!(
        renderer.prepare([2, 2], &[good]),
        Err(ImageError::StaleHandle)
    ));
    let replacement = renderer.upload(1, 1, &[0; 4]).expect("capacity reclaimed");
    renderer.clear();
    assert_eq!(renderer.usage(), (0, 0));
    assert!(matches!(
        renderer.remove(&replacement),
        Err(ImageError::StaleHandle)
    ));
    assert!(matches!(
        renderer.remove(&foreign_handle),
        Err(ImageError::StaleHandle)
    ));
}

#[test]
fn image_gpu_target_contract_and_supported_attachment_formats() {
    let Some(gpu) = Gpu::new() else { return };
    let mut renderer = gpu.renderer(LINEAR, ImageLimits::default());
    let target = gpu.target([2, 2], LINEAR);
    assert!(matches!(
        renderer.encode(&mut gpu.encoder(), &target),
        Err(ImageError::NotPrepared)
    ));
    renderer.prepare([2, 2], &[]).expect("prepare empty");
    for wrong in [
        gpu.target([1, 2], LINEAR),
        gpu.target([2, 2], wgpu::TextureFormat::Rgba8UnormSrgb),
    ] {
        assert!(matches!(
            renderer.encode(&mut gpu.encoder(), &wrong),
            Err(ImageError::TargetMismatch)
        ));
    }
    let no_attachment = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: target.size(),
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: LINEAR,
        usage: wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    assert!(matches!(
        renderer.encode(&mut gpu.encoder(), &no_attachment),
        Err(ImageError::TargetMismatch)
    ));
    assert!(matches!(
        ImageRenderer::new(
            &gpu.device,
            &gpu.queue,
            wgpu::TextureFormat::R32Uint,
            ImageLimits::default()
        ),
        Err(ImageError::UnsupportedFormat)
    ));
    assert!(matches!(
        ImageRenderer::new(
            &gpu.device,
            &gpu.queue,
            LINEAR,
            ImageLimits {
                max_quads: usize::MAX,
                ..ImageLimits::default()
            }
        ),
        Err(ImageError::InvalidLimits)
    ));
    for format in [
        LINEAR,
        wgpu::TextureFormat::Rgba8UnormSrgb,
        wgpu::TextureFormat::Bgra8Unorm,
        wgpu::TextureFormat::Bgra8UnormSrgb,
        wgpu::TextureFormat::Rgba16Float,
        wgpu::TextureFormat::Rgb10a2Unorm,
    ] {
        let mut renderer = gpu.renderer(format, ImageLimits::default());
        let image = renderer
            .upload(1, 1, &[64, 128, 255, 192])
            .expect("color image");
        renderer
            .prepare([2, 2], &[quad(&image, [0.0, 0.0, 2.0, 2.0])])
            .expect("prepare format");
        let output = gpu.target([2, 2], format);
        let mut encoder = gpu.encoder();
        renderer
            .encode(&mut encoder, &output)
            .expect("supported target");
        gpu.queue.submit([encoder.finish()]);
    }
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("all formats completed");
}
