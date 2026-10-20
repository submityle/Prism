//! Retained sRGB images composited in paint order into a host render target.
//! Geometry is in physical pixels; hosts obtain layout from Loom and apply DPI.

use alloc::{collections::BTreeMap, sync::Arc, vec::Vec};
use core::fmt;

const FLOATS_PER_VERTEX: usize = 9;
const BYTES_PER_QUAD: u64 = (6 * FLOATS_PER_VERTEX * size_of::<f32>()) as u64;

/// Texture, pixel and visible-quad budgets shared by one renderer.
#[derive(Clone, Copy, Debug)]
pub struct ImageLimits {
    /// Maximum retained uploaded images.
    pub max_images: usize,
    /// Maximum retained RGBA8 source texture bytes, excluding GPU alignment.
    pub max_texture_bytes: u64,
    /// Maximum source image width/height, also checked against the device.
    pub max_image_edge: u32,
    /// Maximum quads per prepared frame, including clipped/transparent inputs.
    pub max_quads: usize,
}
impl Default for ImageLimits {
    fn default() -> Self {
        Self {
            max_images: 512,
            max_texture_bytes: 64 * 1024 * 1024,
            max_image_edge: 512,
            max_quads: 512,
        }
    }
}

/// Opaque non-reused texture identity, fenced to its owning renderer.
#[derive(Clone, Debug)]
pub struct ImageHandle {
    owner: Arc<()>,
    id: u64,
}

/// One physical rectangle, clipping rectangle and opacity in paint order.
#[derive(Clone, Debug)]
pub struct ImageQuad {
    /// Handle returned by this renderer's `upload`.
    pub image: ImageHandle,
    /// Left, top, width and height in physical target pixels.
    pub rect: [f32; 4],
    /// Left, top, width and height; clipping never changes the source UV mapping.
    pub clip: [f32; 4],
    /// Finite opacity in 0..=1.
    pub opacity: f32,
}

/// Invalid inputs/resources reject the complete operation.
#[derive(Debug)]
pub enum ImageError {
    /// Zero or unrepresentable budgets.
    InvalidLimits,
    /// Source size, data length or image edge exceeds the configured/device bound.
    InvalidImage,
    /// Image count, retained pixel bytes or visible quads exceed a budget.
    Capacity,
    /// The handle was removed, cleared or belongs to another renderer.
    StaleHandle,
    /// Nonfinite/negative geometry, opacity outside 0..=1, or invalid target size.
    InvalidGeometry,
    /// Target extent, format, usage, samples or dimension differs from preparation.
    TargetMismatch,
    /// This renderer has no successfully prepared frame.
    NotPrepared,
    /// Format lacks a supported color-rendering/blending contract.
    UnsupportedFormat,
    /// The allocation or pipeline failed validation.
    Gpu(wgpu::Error),
}
impl fmt::Display for ImageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "image rendering failed: {self:?}")
    }
}
impl core::error::Error for ImageError {}

struct Image {
    bind_group: wgpu::BindGroup,
    bytes: u64,
}
struct Prepared {
    extent: [u32; 2],
    images: Vec<u64>,
    geometry: Vec<u8>,
}

/// Bounded native GPU image compositor. Upload once per source revision, prepare
/// only after layout/visibility changes, then encode over an existing attachment.
/// No readback or full-resolution intermediate attachment is allocated.
///
/// Submit encoders before changing preparation: retained queue writes are not
/// snapshots for unsubmitted frames. Pixel budgets exclude driver overhead.
pub struct ImageRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    format: wgpu::TextureFormat,
    limits: ImageLimits,
    owner: Arc<()>,
    next: u64,
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::RenderPipeline,
    images: BTreeMap<u64, Image>,
    bytes: u64,
    vertices: wgpu::Buffer,
    vertex_capacity: u64,
    prepared: Option<Prepared>,
    geometry_uploads: u64,
}
impl ImageRenderer {
    /// Create for a matching device/queue and single-sample color format.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        format: wgpu::TextureFormat,
        limits: ImageLimits,
    ) -> Result<Self, ImageError> {
        let max_buffer = (limits.max_quads as u64)
            .checked_mul(BYTES_PER_QUAD)
            .ok_or(ImageError::InvalidLimits)?;
        if limits.max_images == 0
            || limits.max_texture_bytes == 0
            || limits.max_image_edge == 0
            || limits.max_quads == 0
            || limits.max_quads > u32::MAX as usize / 6
            || max_buffer > device.limits().max_buffer_size
        {
            return Err(ImageError::InvalidLimits);
        }
        if !matches!(
            format,
            wgpu::TextureFormat::Rgba8Unorm
                | wgpu::TextureFormat::Rgba8UnormSrgb
                | wgpu::TextureFormat::Bgra8Unorm
                | wgpu::TextureFormat::Bgra8UnormSrgb
                | wgpu::TextureFormat::Rgba16Float
                | wgpu::TextureFormat::Rgb10a2Unorm
        ) {
            return Err(ImageError::UnsupportedFormat);
        }
        resources(device, || {
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("editor image layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                }],
            });
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("editor images"),
                source: wgpu::ShaderSource::Wgsl(include_str!("image.wgsl").into()),
            });
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("editor images pipeline layout"),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("editor images pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs"),
                    compilation_options: Default::default(),
                    buffers: &[Some(wgpu::VertexBufferLayout {
                        array_stride: (FLOATS_PER_VERTEX * 4) as u64,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x2,
                                offset: 0,
                                shader_location: 0,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x2,
                                offset: 8,
                                shader_location: 1,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x4,
                                offset: 16,
                                shader_location: 2,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32,
                                offset: 32,
                                shader_location: 3,
                            },
                        ],
                    })],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            });
            Self {
                device: device.clone(),
                queue: queue.clone(),
                format,
                limits,
                owner: Arc::new(()),
                next: 1,
                layout,
                pipeline,
                images: BTreeMap::new(),
                bytes: 0,
                vertices: vertex_buffer(device, BYTES_PER_QUAD),
                vertex_capacity: BYTES_PER_QUAD,
                prepared: None,
                geometry_uploads: 0,
            }
        })
    }

    /// Upload tightly packed straight-alpha sRGB RGBA8. Source bytes are copied
    /// to the GPU; host pixels can be released after this call. Failed uploads
    /// preserve all handles and prepared geometry.
    pub fn upload(
        &mut self,
        width: u32,
        height: u32,
        rgba8: &[u8],
    ) -> Result<ImageHandle, ImageError> {
        let bytes = u64::from(width)
            .checked_mul(u64::from(height))
            .and_then(|pixels| pixels.checked_mul(4))
            .ok_or(ImageError::InvalidImage)?;
        if width == 0
            || height == 0
            || width > self.limits.max_image_edge
            || height > self.limits.max_image_edge
            || width > self.device.limits().max_texture_dimension_2d
            || height > self.device.limits().max_texture_dimension_2d
            || usize::try_from(bytes).ok() != Some(rgba8.len())
        {
            return Err(ImageError::InvalidImage);
        }
        if self.images.len() >= self.limits.max_images
            || bytes > self.limits.max_texture_bytes.saturating_sub(self.bytes)
        {
            return Err(ImageError::Capacity);
        }
        let next = self.next.checked_add(1).ok_or(ImageError::Capacity)?;
        let bind_group = resources(&self.device, || {
            let texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("editor sRGB image"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            self.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                rgba8,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(width * 4),
                    rows_per_image: Some(height),
                },
                texture.size(),
            );
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("editor image"),
                layout: &self.layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(
                        &texture.create_view(&Default::default()),
                    ),
                }],
            })
        })?;
        let id = self.next;
        self.next = next;
        self.bytes += bytes;
        self.images.insert(id, Image { bind_group, bytes });
        Ok(ImageHandle {
            owner: self.owner.clone(),
            id,
        })
    }

    /// Remove one texture. A prepared frame referencing it is revoked.
    pub fn remove(&mut self, image: &ImageHandle) -> Result<(), ImageError> {
        self.resolve(image)?;
        let image_value = self
            .images
            .remove(&image.id)
            .ok_or(ImageError::StaleHandle)?;
        self.bytes -= image_value.bytes;
        if self
            .prepared
            .as_ref()
            .is_some_and(|p| p.images.contains(&image.id))
        {
            self.prepared = None;
        }
        Ok(())
    }
    /// Release textures and revoke every handle and frame. In-flight submitted
    /// commands retain their own GPU resources according to wgpu ownership.
    pub fn clear(&mut self) {
        self.images.clear();
        self.bytes = 0;
        self.prepared = None;
        self.owner = Arc::new(());
    }
    /// Retained texture count and logical RGBA8 bytes.
    #[must_use]
    pub fn usage(&self) -> (usize, u64) {
        (self.images.len(), self.bytes)
    }
    /// Number of successful geometry uploads, useful for idle-frame assertions.
    #[must_use]
    pub const fn geometry_uploads(&self) -> u64 {
        self.geometry_uploads
    }

    /// Validate and prepare all visible image quads atomically. Identical paint
    /// geometry performs no queue upload. Rejected input keeps the old frame.
    pub fn prepare(&mut self, extent: [u32; 2], quads: &[ImageQuad]) -> Result<(), ImageError> {
        if extent.contains(&0)
            || extent
                .iter()
                .any(|e| *e > self.device.limits().max_texture_dimension_2d)
        {
            return Err(ImageError::InvalidGeometry);
        }
        if quads.len() > self.limits.max_quads {
            return Err(ImageError::Capacity);
        }
        let mut images = Vec::with_capacity(quads.len());
        let mut geometry = Vec::with_capacity(quads.len() * BYTES_PER_QUAD as usize);
        for quad in quads {
            self.resolve(&quad.image)?;
            validate_quad(quad)?;
            if quad.rect[2] == 0.0
                || quad.rect[3] == 0.0
                || quad.clip[2] == 0.0
                || quad.clip[3] == 0.0
                || quad.opacity == 0.0
            {
                continue;
            }
            images.push(quad.image.id);
            let clip = [
                quad.clip[0],
                quad.clip[1],
                quad.clip[0] + quad.clip[2],
                quad.clip[1] + quad.clip[3],
            ];
            for [u, v] in [
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ] {
                let x = (quad.rect[0] + u * quad.rect[2]) / extent[0] as f32 * 2.0 - 1.0;
                let y = 1.0 - (quad.rect[1] + v * quad.rect[3]) / extent[1] as f32 * 2.0;
                for value in [x, y, u, v, clip[0], clip[1], clip[2], clip[3], quad.opacity] {
                    geometry.extend_from_slice(&value.to_le_bytes());
                }
            }
        }
        let unchanged = self
            .prepared
            .as_ref()
            .is_some_and(|p| p.geometry == geometry);
        if !unchanged && !geometry.is_empty() {
            let required = geometry.len() as u64;
            let new_buffer = if required > self.vertex_capacity {
                let capacity = required
                    .next_power_of_two()
                    .min(self.limits.max_quads as u64 * BYTES_PER_QUAD)
                    .min(self.device.limits().max_buffer_size);
                Some((
                    resources(&self.device, || vertex_buffer(&self.device, capacity))?,
                    capacity,
                ))
            } else {
                None
            };
            let target = new_buffer
                .as_ref()
                .map_or(&self.vertices, |(buffer, _)| buffer);
            resources(&self.device, || {
                self.queue.write_buffer(target, 0, &geometry);
            })?;
            if let Some((buffer, capacity)) = new_buffer {
                self.vertices = buffer;
                self.vertex_capacity = capacity;
            }
            self.geometry_uploads = self.geometry_uploads.saturating_add(1);
        }
        self.prepared = Some(Prepared {
            extent,
            images,
            geometry,
        });
        Ok(())
    }

    /// Paint over mip 0 of a same-device target, preserving paint order.
    /// A single-layer view is constructed here so hosts cannot accidentally
    /// supply a different mip or reinterpret its format. No submission is done.
    pub fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        texture: &wgpu::Texture,
    ) -> Result<(), ImageError> {
        let p = self.prepared.as_ref().ok_or(ImageError::NotPrepared)?;
        if texture.width() != p.extent[0]
            || texture.height() != p.extent[1]
            || texture.format() != self.format
            || texture.dimension() != wgpu::TextureDimension::D2
            || texture.depth_or_array_layers() != 1
            || texture.sample_count() != 1
            || !texture
                .usage()
                .contains(wgpu::TextureUsages::RENDER_ATTACHMENT)
        {
            return Err(ImageError::TargetMismatch);
        }
        for id in &p.images {
            if !self.images.contains_key(id) {
                return Err(ImageError::StaleHandle);
            }
        }
        if p.images.is_empty() {
            return Ok(());
        }
        let target = texture.create_view(&wgpu::TextureViewDescriptor {
            mip_level_count: Some(1),
            array_layer_count: Some(1),
            ..Default::default()
        });
        let attachments = [Some(wgpu::RenderPassColorAttachment {
            view: &target,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            },
        })];
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("editor images"),
            color_attachments: &attachments,
            ..Default::default()
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_vertex_buffer(0, self.vertices.slice(..));
        for (i, id) in p.images.iter().enumerate() {
            pass.set_bind_group(0, &self.images[id].bind_group, &[]);
            pass.draw(i as u32 * 6..i as u32 * 6 + 6, 0..1);
        }
        Ok(())
    }
    fn resolve(&self, handle: &ImageHandle) -> Result<&Image, ImageError> {
        if !Arc::ptr_eq(&self.owner, &handle.owner) {
            return Err(ImageError::StaleHandle);
        }
        self.images.get(&handle.id).ok_or(ImageError::StaleHandle)
    }
}
fn validate_quad(quad: &ImageQuad) -> Result<(), ImageError> {
    if !quad.opacity.is_finite()
        || !(0.0..=1.0).contains(&quad.opacity)
        || quad
            .rect
            .iter()
            .chain(quad.clip.iter())
            .any(|v| !v.is_finite() || v.abs() > 1.0e7)
        || quad.rect[2..]
            .iter()
            .chain(quad.clip[2..].iter())
            .any(|v| *v < 0.0)
    {
        return Err(ImageError::InvalidGeometry);
    }
    Ok(())
}
fn vertex_buffer(device: &wgpu::Device, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("editor image geometry"),
        size,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}
fn resources<T>(device: &wgpu::Device, create: impl FnOnce() -> T) -> Result<T, ImageError> {
    let oom = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let internal = device.push_error_scope(wgpu::ErrorFilter::Internal);
    let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let value = create();
    let validation = futures_lite::future::block_on(validation.pop());
    let internal = futures_lite::future::block_on(internal.pop());
    let oom = futures_lite::future::block_on(oom.pop());
    if let Some(error) = validation.or(internal).or(oom) {
        Err(ImageError::Gpu(error))
    } else {
        Ok(value)
    }
}
