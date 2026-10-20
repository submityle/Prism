//! Executable wgpu ID-buffer picking for an editor viewport.
//!
//! The backend owns ID/depth targets and readback resources. A renderer owns
//! the object-ID pipeline and draws into the pass returned by
//! [`GpuPickingFrame::begin_render_pass`]. This keeps the backend independent
//! of scene, mesh, material, and camera implementations while still making the
//! GPU path concrete and testable. [`crate::SceneIdGeometryRenderer`] supplies
//! an indexed-geometry implementation without a scene-engine dependency.

use core::fmt;
use std::{mem::size_of, sync::mpsc};

use wgpu::{
    Adapter, Buffer, BufferUsages, CommandEncoder, Device, Extent3d, LoadOp, Operations, PollType,
    Queue, RenderPass, RenderPassColorAttachment, RenderPassDescriptor, StoreOp, Texture,
    TextureFormat, TextureUsages, TextureView, TextureViewDescriptor,
};

use crate::{
    PickRequest, PickResult, PixelPoint, RenderTargetHandle, RenderTargetState, ViewportId,
};

mod box_selection;
pub use box_selection::{
    GpuBoxSelectionError, GpuBoxSelectionLimits, InFlightBoxSelectionReadback,
    PendingBoxSelectionReadback,
};

/// The integer format used by the ID target.
pub const ID_BUFFER_FORMAT: TextureFormat = TextureFormat::Rgba32Uint;

/// Bytes in one ID texel.
pub const ID_BUFFER_BYTES_PER_TEXEL: u32 = 16;

/// wgpu's required row alignment for texture-to-buffer copies.
pub const READBACK_ROW_ALIGNMENT: u32 = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;

const READBACK_ROW_BYTES: u32 = READBACK_ROW_ALIGNMENT;
/// A four-channel integer ID written by an object-ID fragment shader.
///
/// All-zero is reserved for a miss. `SceneIdEncoder` uses the low two channels
/// for its persistent scene-object token; the entity helpers remain available
/// for adapters that own their own one-to-one entity encoding.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EncodedPickId([u32; 4]);

impl EncodedPickId {
    /// Returns the reserved miss value.
    #[must_use]
    pub const fn none() -> Self {
        Self([0; 4])
    }

    /// Encodes an editor entity ID. The maximum `u128` value is rejected so
    /// the reserved all-zero miss value remains unambiguous.
    pub fn from_entity(entity: u128) -> Result<Self, GpuPickingError> {
        let encoded = entity
            .checked_add(1)
            .ok_or(GpuPickingError::EntityIdCapacityExceeded)?;
        Ok(Self([
            encoded as u32,
            (encoded >> 32) as u32,
            (encoded >> 64) as u32,
            (encoded >> 96) as u32,
        ]))
    }

    /// Decodes a shader result into an optional editor entity ID.
    #[must_use]
    pub fn decode_entity(self) -> Option<u128> {
        let encoded = self.0[0] as u128
            | ((self.0[1] as u128) << 32)
            | ((self.0[2] as u128) << 64)
            | ((self.0[3] as u128) << 96);
        if encoded == 0 {
            None
        } else {
            Some(encoded - 1)
        }
    }

    /// Returns the four WGSL `u32` channels in low-word-first order.
    #[must_use]
    pub const fn channels(self) -> [u32; 4] {
        self.0
    }

    /// Constructs an encoded value from four GPU-read words.
    #[must_use]
    pub const fn from_channels(channels: [u32; 4]) -> Self {
        Self(channels)
    }

    pub(crate) fn from_token(token: u64) -> Result<Self, GpuPickingError> {
        if token == 0 {
            return Err(GpuPickingError::EntityIdCapacityExceeded);
        }
        Ok(Self([token as u32, (token >> 32) as u32, 0, 0]))
    }

    pub(crate) fn decode_token(self) -> Option<u64> {
        let channels = self.0;
        let token = u64::from(channels[0]) | (u64::from(channels[1]) << 32);
        if token == 0 || channels[2] != 0 || channels[3] != 0 {
            None
        } else {
            Some(token)
        }
    }
}

/// Configuration for an ID target.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuPickingConfig {
    /// Target width in pixels.
    pub width: u32,
    /// Target height in pixels.
    pub height: u32,
}

impl GpuPickingConfig {
    /// Creates a target configuration.
    #[must_use]
    pub const fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }
}

/// Failure from GPU target creation, encoding, or synchronous readback.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GpuPickingError {
    /// A zero-sized target is not a valid render attachment.
    InvalidExtent {
        /// Requested target width.
        width: u32,
        /// Requested target height.
        height: u32,
    },
    /// The target exceeds the device's 2D texture dimension limit.
    DimensionCapacityExceeded {
        /// Requested target width.
        width: u32,
        /// Requested target height.
        height: u32,
        /// Device maximum width or height.
        max_dimension: u32,
    },
    /// The selected device cannot render, copy and sample the integer target.
    UnsupportedTargetUsage,
    /// The entity ID cannot be represented while reserving zero for a miss.
    EntityIdCapacityExceeded,
    /// A requested pixel lies outside the target.
    PixelOutOfBounds {
        /// Requested pixel coordinate.
        pixel: PixelPoint,
        /// Target width.
        width: u32,
        /// Target height.
        height: u32,
    },
    /// The request belongs to another viewport.
    ViewportMismatch {
        /// Viewport associated with the frame.
        expected: ViewportId,
        /// Viewport carried by the request.
        actual: ViewportId,
    },
    /// The request was recorded for a different viewport/camera generation.
    GenerationMismatch {
        /// Generation associated with the frame.
        expected: u64,
        /// Generation carried by the request.
        actual: u64,
    },
    /// The request was recorded for different scene content.
    SceneGenerationMismatch {
        /// Scene generation associated with the frame.
        expected: u64,
        /// Scene generation carried by the request.
        actual: u64,
    },
    /// The request was recorded against a different target allocation.
    TargetGenerationMismatch {
        /// Generation associated with the frame.
        expected: u64,
        /// Generation carried by the request.
        actual: u64,
    },
    /// The request was recorded against another target allocation.
    TargetHandleMismatch {
        /// Handle associated with the frame.
        expected: RenderTargetHandle,
        /// Handle carried by the request.
        actual: RenderTargetHandle,
    },
    /// The device could not complete the blocking poll.
    DevicePollFailed(String),
    /// The asynchronous map callback reported a failure.
    ReadbackMapFailed(String),
    /// The callback channel closed before a map result arrived.
    ReadbackCallbackClosed,
    /// The mapped range was shorter than the fixed readback record.
    ReadbackBufferTooSmall {
        /// Actual mapped byte count.
        actual: usize,
        /// Minimum required byte count.
        required: usize,
    },
}

impl fmt::Display for GpuPickingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidExtent { width, height } => {
                write!(
                    f,
                    "GPU picking target extent must be non-zero: {width}x{height}"
                )
            }
            Self::DimensionCapacityExceeded {
                width,
                height,
                max_dimension,
            } => write!(
                f,
                "GPU picking target {width}x{height} exceeds device dimension {max_dimension}"
            ),
            Self::UnsupportedTargetUsage => {
                f.write_str("Rgba32Uint does not support render attachment, copy-source and texture-binding usage")
            }
            Self::EntityIdCapacityExceeded => {
                f.write_str("entity ID is reserved by the all-zero miss encoding")
            }
            Self::PixelOutOfBounds {
                pixel,
                width,
                height,
            } => write!(
                f,
                "pick pixel ({}, {}) is outside target {}x{}",
                pixel.x, pixel.y, width, height
            ),
            Self::ViewportMismatch { expected, actual } => write!(
                f,
                "pick request viewport {} does not match frame viewport {}",
                actual.raw(),
                expected.raw()
            ),
            Self::GenerationMismatch { expected, actual } => write!(
                f,
                "pick request generation {actual} does not match frame generation {expected}"
            ),
            Self::TargetGenerationMismatch { expected, actual } => write!(
                f,
                "pick request target generation {actual} does not match frame target generation {expected}"
            ),
            Self::SceneGenerationMismatch { expected, actual } => write!(
                f,
                "pick request scene generation {actual} does not match frame scene generation {expected}"
            ),
            Self::TargetHandleMismatch { expected, actual } => write!(
                f,
                "pick request target handle {} does not match frame target handle {}",
                actual.raw(),
                expected.raw()
            ),
            Self::DevicePollFailed(error) => write!(f, "GPU picking device poll failed: {error}"),
            Self::ReadbackMapFailed(error) => write!(f, "GPU picking readback map failed: {error}"),
            Self::ReadbackCallbackClosed => f.write_str("GPU picking readback callback closed"),
            Self::ReadbackBufferTooSmall { actual, required } => write!(
                f,
                "GPU picking readback buffer has {actual} bytes, requires {required}"
            ),
        }
    }
}

impl std::error::Error for GpuPickingError {}

/// A pending one-pixel readback encoded into a command encoder.
pub struct PendingPickReadback {
    request: PickRequest,
    buffer: Buffer,
}

impl fmt::Debug for PendingPickReadback {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PendingPickReadback")
            .field("request", &self.request)
            .finish_non_exhaustive()
    }
}

/// A submitted readback that may be polled without blocking the render loop.
pub struct InFlightPickReadback {
    request: PickRequest,
    device: Device,
    buffer: Buffer,
    receiver: mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
    completed: bool,
}

impl fmt::Debug for InFlightPickReadback {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("InFlightPickReadback")
            .field("request", &self.request)
            .field("completed", &self.completed)
            .finish_non_exhaustive()
    }
}

impl InFlightPickReadback {
    /// Returns immutable correlation metadata, including after a viewport closes.
    #[must_use]
    pub const fn request(&self) -> PickRequest {
        self.request
    }

    /// Polls without blocking, even after the originating target is closed.
    /// A result or terminal error is delivered at most once.
    pub fn poll(&mut self) -> Result<Option<PickResult>, GpuPickingError> {
        if self.completed {
            return Ok(None);
        }
        if let Err(error) = self.device.poll(PollType::Poll) {
            self.completed = true;
            self.buffer.destroy();
            return Err(GpuPickingError::DevicePollFailed(error.to_string()));
        }
        match self.receiver.try_recv() {
            Ok(Ok(())) => {
                self.completed = true;
                read_mapped_result(self).map(Some)
            }
            Ok(Err(error)) => {
                self.completed = true;
                self.buffer.destroy();
                Err(GpuPickingError::ReadbackMapFailed(error.to_string()))
            }
            Err(mpsc::TryRecvError::Empty) => Ok(None),
            Err(mpsc::TryRecvError::Disconnected) => {
                self.completed = true;
                self.buffer.destroy();
                Err(GpuPickingError::ReadbackCallbackClosed)
            }
        }
    }
}

impl Drop for InFlightPickReadback {
    fn drop(&mut self) {
        // Destroy cancels an outstanding map and also accepts an already-unmapped
        // buffer. Calling unmap again after successful readback is invalid in wgpu.
        self.buffer.destroy();
    }
}

/// A real wgpu ID-buffer target with asynchronous and explicit blocking readback.
pub struct GpuPickingBackend {
    device: Device,
    queue: Queue,
    target: Texture,
    target_view: TextureView,
    depth_view: TextureView,
    config: GpuPickingConfig,
    max_dimension: u32,
}

impl fmt::Debug for GpuPickingBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GpuPickingBackend")
            .field("config", &self.config)
            .field("max_dimension", &self.max_dimension)
            .finish_non_exhaustive()
    }
}

impl GpuPickingBackend {
    /// Allocates an ID target using the device's native texture limits.
    pub fn new(
        device: &Device,
        adapter: &Adapter,
        queue: &Queue,
        config: GpuPickingConfig,
    ) -> Result<Self, GpuPickingError> {
        let max_dimension = device.limits().max_texture_dimension_2d;
        validate_extent(config, max_dimension)?;
        let allowed = adapter
            .get_texture_format_features(ID_BUFFER_FORMAT)
            .allowed_usages;
        let required_usage = TextureUsages::RENDER_ATTACHMENT
            | TextureUsages::COPY_SRC
            | TextureUsages::TEXTURE_BINDING;
        if !allowed.contains(required_usage) {
            return Err(GpuPickingError::UnsupportedTargetUsage);
        }

        let (target, target_view) = create_target(device, config);
        let depth_view = create_depth_target(device, config);
        Ok(Self {
            device: device.clone(),
            queue: queue.clone(),
            target,
            target_view,
            depth_view,
            config,
            max_dimension,
        })
    }

    /// Returns the target extent.
    #[must_use]
    pub const fn config(&self) -> GpuPickingConfig {
        self.config
    }

    /// Returns the format expected by the object-ID fragment shader.
    #[must_use]
    pub const fn format(&self) -> TextureFormat {
        ID_BUFFER_FORMAT
    }

    /// Returns the native device for renderer-owned pipeline creation.
    #[must_use]
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Returns the native queue for renderer-owned submissions.
    #[must_use]
    pub fn queue(&self) -> &Queue {
        &self.queue
    }

    /// Returns the target's native `Rgba32Uint` view. The texture has
    /// `TEXTURE_BINDING` usage: bind as `TextureSampleType::Uint`, use WGSL
    /// `texture_2d<u32>`/`textureLoad` without a sampler for GPU-only ID preview.
    /// Rebuild host bind groups after resize/close/reopen; do not sample this
    /// view while it is attached to the same render pass.
    #[must_use]
    pub fn target_view(&self) -> &TextureView {
        &self.target_view
    }

    /// Returns the matching single-sample `Depth32Float` attachment.
    #[must_use]
    pub fn depth_view(&self) -> &TextureView {
        &self.depth_view
    }

    /// Recreates the target at a new size. Existing command encoders must be
    /// submitted or dropped before calling this method.
    pub fn resize(&mut self, config: GpuPickingConfig) -> Result<(), GpuPickingError> {
        validate_extent(config, self.max_dimension)?;
        let (target, target_view) = create_target(&self.device, config);
        let depth_view = create_depth_target(&self.device, config);
        self.target = target;
        self.target_view = target_view;
        self.depth_view = depth_view;
        self.config = config;
        Ok(())
    }

    /// Starts a frame correlation record for one viewport generation.
    #[must_use]
    pub(crate) const fn begin_frame(
        &self,
        state: RenderTargetState,
        scene_generation: u64,
    ) -> GpuPickingFrame<'_> {
        GpuPickingFrame {
            backend: self,
            state,
            scene_generation,
        }
    }

    fn create_readback_buffer(&self) -> Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("prism-editor-picking-readback"),
            size: u64::from(READBACK_ROW_BYTES),
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Submits encoded render/copy commands and starts an asynchronous ID readback.
    pub fn submit_readback(
        &self,
        encoder: CommandEncoder,
        pending: PendingPickReadback,
    ) -> InFlightPickReadback {
        self.queue.submit([encoder.finish()]);
        self.start_readback(pending)
    }

    /// Starts mapping after the host has submitted the encoder containing this
    /// copy to the shared queue. Allows one host submission to contain multiple
    /// viewport/box copies and color passes. Never call before submitting.
    pub fn start_readback(&self, pending: PendingPickReadback) -> InFlightPickReadback {
        let (sender, receiver) = mpsc::sync_channel(1);
        pending
            .buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = sender.send(result);
            });
        InFlightPickReadback {
            request: pending.request,
            device: self.device.clone(),
            buffer: pending.buffer,
            receiver,
            completed: false,
        }
    }

    /// Polls an asynchronous readback without blocking for GPU completion.
    pub fn poll_readback(
        &self,
        pending: &mut InFlightPickReadback,
    ) -> Result<Option<PickResult>, GpuPickingError> {
        pending.poll()
    }

    /// Submits the encoded render/copy commands and blocks until one ID texel
    /// is mapped. This is an explicit synchronous policy intended for editor
    /// pointer picks, not a per-frame bulk readback path.
    pub fn submit_and_read_sync(
        &self,
        encoder: CommandEncoder,
        pending: PendingPickReadback,
    ) -> Result<PickResult, GpuPickingError> {
        let mut pending = self.submit_readback(encoder, pending);
        self.device
            .poll(PollType::wait_indefinitely())
            .map_err(|error| GpuPickingError::DevicePollFailed(error.to_string()))?;
        self.poll_readback(&mut pending)?
            .ok_or(GpuPickingError::ReadbackCallbackClosed)
    }
}

fn read_mapped_result(pending: &mut InFlightPickReadback) -> Result<PickResult, GpuPickingError> {
    let slice = pending.buffer.slice(..);
    let mapped = match slice.get_mapped_range() {
        Ok(mapped) => mapped,
        Err(error) => {
            pending.buffer.unmap();
            return Err(GpuPickingError::ReadbackMapFailed(error.to_string()));
        }
    };
    if mapped.len() < size_of::<[u32; 4]>() {
        let actual = mapped.len();
        drop(mapped);
        pending.buffer.unmap();
        return Err(GpuPickingError::ReadbackBufferTooSmall {
            actual,
            required: size_of::<[u32; 4]>(),
        });
    }
    let mut channels = [0_u32; 4];
    for (index, channel) in channels.iter_mut().enumerate() {
        let offset = index * 4;
        let mut bytes = [0_u8; 4];
        bytes.copy_from_slice(&mapped[offset..offset + 4]);
        *channel = u32::from_le_bytes(bytes);
    }
    drop(mapped);
    pending.buffer.unmap();
    Ok(PickResult {
        id: pending.request.id,
        viewport: pending.request.viewport,
        generation: pending.request.generation,
        scene_generation: pending.request.scene_generation,
        target_generation: pending.request.target_generation,
        target_handle: pending.request.target_handle,
        encoded: EncodedPickId::from_channels(channels),
    })
}

/// A frame-scoped viewport/generation correlation used by the renderer pass.
pub struct GpuPickingFrame<'a> {
    backend: &'a GpuPickingBackend,
    state: RenderTargetState,
    scene_generation: u64,
}

impl fmt::Debug for GpuPickingFrame<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GpuPickingFrame")
            .field("state", &self.state)
            .field("scene_generation", &self.scene_generation)
            .field("config", &self.backend.config)
            .finish()
    }
}

impl<'a> GpuPickingFrame<'a> {
    /// Returns the complete frame allocation/camera correlation record.
    #[must_use]
    pub const fn state(&self) -> RenderTargetState {
        self.state
    }

    /// Returns the frame viewport identity.
    #[must_use]
    pub const fn viewport(&self) -> ViewportId {
        self.state.viewport
    }

    /// Returns the frame viewport/camera generation.
    #[must_use]
    pub const fn generation(&self) -> u64 {
        self.state.viewport_generation
    }

    /// Returns the independent scene-content generation.
    #[must_use]
    pub const fn scene_generation(&self) -> u64 {
        self.scene_generation
    }

    /// Returns the render-target allocation generation.
    #[must_use]
    pub const fn target_generation(&self) -> u64 {
        self.state.target_generation
    }

    /// Returns the stable allocation handle for the ID target.
    #[must_use]
    pub const fn target_handle(&self) -> RenderTargetHandle {
        self.state.handle
    }

    /// Begins the renderer-owned ID pass, clearing the target to a miss.
    pub fn begin_render_pass<'pass>(
        &'pass self,
        encoder: &'pass mut CommandEncoder,
        label: Option<&'pass str>,
    ) -> RenderPass<'pass> {
        let color_attachments = [Some(RenderPassColorAttachment {
            view: &self.backend.target_view,
            depth_slice: None,
            resolve_target: None,
            ops: Operations {
                load: LoadOp::Clear(wgpu::Color::TRANSPARENT),
                store: StoreOp::Store,
            },
        })];
        encoder.begin_render_pass(&RenderPassDescriptor {
            label,
            color_attachments: &color_attachments,
            ..Default::default()
        })
    }

    /// Begins an ID/depth pass, clearing IDs to miss and depth to the selected
    /// convention's far plane. Geometry uses WebGPU clip depth `[0, 1]`.
    pub fn begin_geometry_render_pass<'pass>(
        &'pass self,
        encoder: &'pass mut CommandEncoder,
        depth: crate::SceneIdDepth,
        label: Option<&'pass str>,
    ) -> RenderPass<'pass> {
        let color_attachments = [Some(RenderPassColorAttachment {
            view: &self.backend.target_view,
            depth_slice: None,
            resolve_target: None,
            ops: Operations {
                load: LoadOp::Clear(wgpu::Color::TRANSPARENT),
                store: StoreOp::Store,
            },
        })];
        encoder.begin_render_pass(&RenderPassDescriptor {
            label,
            color_attachments: &color_attachments,
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.backend.depth_view,
                depth_ops: Some(Operations {
                    load: LoadOp::Clear(depth.clear_value()),
                    store: StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            ..Default::default()
        })
    }

    /// Encodes a one-pixel texture-to-buffer copy correlated to `request`.
    pub fn encode_readback(
        &self,
        encoder: &mut CommandEncoder,
        request: PickRequest,
    ) -> Result<PendingPickReadback, GpuPickingError> {
        validate_request(
            self.state,
            self.scene_generation,
            request,
            self.backend.config,
        )?;

        let buffer = self.backend.create_readback_buffer();
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.backend.target,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: request.pixel.x,
                    y: request.pixel.y,
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(READBACK_ROW_BYTES),
                    rows_per_image: Some(1),
                },
            },
            Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        Ok(PendingPickReadback { request, buffer })
    }
}

fn validate_extent(config: GpuPickingConfig, max_dimension: u32) -> Result<(), GpuPickingError> {
    if config.width == 0 || config.height == 0 {
        return Err(GpuPickingError::InvalidExtent {
            width: config.width,
            height: config.height,
        });
    }
    if config.width > max_dimension || config.height > max_dimension {
        return Err(GpuPickingError::DimensionCapacityExceeded {
            width: config.width,
            height: config.height,
            max_dimension,
        });
    }
    Ok(())
}

fn validate_pixel(pixel: PixelPoint, config: GpuPickingConfig) -> Result<(), GpuPickingError> {
    if pixel.x >= config.width || pixel.y >= config.height {
        return Err(GpuPickingError::PixelOutOfBounds {
            pixel,
            width: config.width,
            height: config.height,
        });
    }
    Ok(())
}

fn validate_request(
    state: RenderTargetState,
    scene_generation: u64,
    request: PickRequest,
    config: GpuPickingConfig,
) -> Result<(), GpuPickingError> {
    validate_correlation(
        state,
        scene_generation,
        request.viewport,
        request.generation,
        request.scene_generation,
        request.target_generation,
        request.target_handle,
    )?;
    validate_pixel(request.pixel, config)
}

fn validate_correlation(
    state: RenderTargetState,
    scene_generation: u64,
    viewport: ViewportId,
    generation: u64,
    requested_scene: u64,
    target_generation: u64,
    target_handle: RenderTargetHandle,
) -> Result<(), GpuPickingError> {
    if viewport != state.viewport {
        return Err(GpuPickingError::ViewportMismatch {
            expected: state.viewport,
            actual: viewport,
        });
    }
    if generation != state.viewport_generation {
        return Err(GpuPickingError::GenerationMismatch {
            expected: state.viewport_generation,
            actual: generation,
        });
    }
    if requested_scene != scene_generation {
        return Err(GpuPickingError::SceneGenerationMismatch {
            expected: scene_generation,
            actual: requested_scene,
        });
    }
    if target_generation != state.target_generation {
        return Err(GpuPickingError::TargetGenerationMismatch {
            expected: state.target_generation,
            actual: target_generation,
        });
    }
    if target_handle != state.handle {
        return Err(GpuPickingError::TargetHandleMismatch {
            expected: state.handle,
            actual: target_handle,
        });
    }
    Ok(())
}

fn create_target(device: &Device, config: GpuPickingConfig) -> (Texture, TextureView) {
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("prism-editor-picking-id-buffer"),
        size: Extent3d {
            width: config.width,
            height: config.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: ID_BUFFER_FORMAT,
        usage: TextureUsages::RENDER_ATTACHMENT
            | TextureUsages::COPY_SRC
            | TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = target.create_view(&TextureViewDescriptor::default());
    (target, view)
}

fn create_depth_target(device: &Device, config: GpuPickingConfig) -> TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("prism-editor-picking-depth"),
            size: Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: TextureFormat::Depth32Float,
            usage: TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
        .create_view(&TextureViewDescriptor::default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PickRequestId, ViewportId};

    #[test]
    fn entity_encoding_reserves_miss_and_preserves_all_supported_bits() {
        let values = [0, 1, u128::from(u32::MAX), u128::MAX - 1];
        for value in values {
            let encoded = EncodedPickId::from_entity(value).expect("supported entity");
            assert_eq!(encoded.decode_entity(), Some(value));
        }
        assert_eq!(EncodedPickId::none().decode_entity(), None);
        assert_eq!(
            EncodedPickId::from_entity(u128::MAX),
            Err(GpuPickingError::EntityIdCapacityExceeded)
        );
    }

    #[test]
    fn boundary_validation_rejects_zero_extent_and_out_of_bounds_pixels() {
        assert_eq!(
            validate_extent(GpuPickingConfig::new(0, 4), 4096),
            Err(GpuPickingError::InvalidExtent {
                width: 0,
                height: 4
            })
        );
        assert_eq!(
            validate_extent(GpuPickingConfig::new(4097, 4), 4096),
            Err(GpuPickingError::DimensionCapacityExceeded {
                width: 4097,
                height: 4,
                max_dimension: 4096
            })
        );
        assert_eq!(
            validate_pixel(PixelPoint { x: 4, y: 0 }, GpuPickingConfig::new(4, 4)),
            Err(GpuPickingError::PixelOutOfBounds {
                pixel: PixelPoint { x: 4, y: 0 },
                width: 4,
                height: 4
            })
        );
    }

    #[test]
    fn frame_rejects_wrong_target_identity_before_gpu_encoding() {
        let state = RenderTargetState {
            viewport: ViewportId::from_raw(1),
            handle: RenderTargetHandle::from_raw(1),
            viewport_generation: 7,
            target_generation: 3,
            config: GpuPickingConfig::new(4, 4),
        };
        let request = PickRequest {
            id: PickRequestId(9),
            viewport: ViewportId::from_raw(2),
            generation: 8,
            scene_generation: 0,
            target_generation: 3,
            target_handle: RenderTargetHandle::from_raw(1),
            pixel: PixelPoint { x: 0, y: 0 },
        };
        assert_eq!(
            validate_request(state, 0, request, GpuPickingConfig::new(4, 4)),
            Err(GpuPickingError::ViewportMismatch {
                expected: ViewportId::from_raw(1),
                actual: ViewportId::from_raw(2)
            })
        );
        let generation_request = PickRequest {
            viewport: ViewportId::from_raw(1),
            ..request
        };
        assert_eq!(
            validate_request(state, 0, generation_request, GpuPickingConfig::new(4, 4)),
            Err(GpuPickingError::GenerationMismatch {
                expected: 7,
                actual: 8
            })
        );
        let target_generation_request = PickRequest {
            target_generation: 4,
            ..generation_request
        };
        assert_eq!(
            validate_request(
                RenderTargetState {
                    viewport_generation: 8,
                    ..state
                },
                0,
                target_generation_request,
                GpuPickingConfig::new(4, 4)
            ),
            Err(GpuPickingError::TargetGenerationMismatch {
                expected: 3,
                actual: 4
            })
        );
        let target_handle_request = PickRequest {
            target_handle: RenderTargetHandle::from_raw(2),
            ..generation_request
        };
        assert_eq!(
            validate_request(
                RenderTargetState {
                    viewport_generation: 8,
                    ..state
                },
                0,
                target_handle_request,
                GpuPickingConfig::new(4, 4)
            ),
            Err(GpuPickingError::TargetHandleMismatch {
                expected: RenderTargetHandle::from_raw(1),
                actual: RenderTargetHandle::from_raw(2)
            })
        );
    }

    #[cfg(feature = "gpu-tests")]
    #[test]
    fn real_gpu_draw_readback_round_trips_encoded_id() {
        let Some((adapter, device, queue)) = crate::test_support::device() else {
            return;
        };
        let backend =
            GpuPickingBackend::new(&device, &adapter, &queue, GpuPickingConfig::new(4, 4))
                .expect("Rgba32Uint render target");
        let mut viewports = crate::ViewportRegistry::new();
        let viewport = viewports.open("GPU test").expect("viewport");
        let mut targets = crate::RenderTargetRegistry::new();
        let target = targets
            .open(viewport.id(), viewport.generation(), backend)
            .expect("open target");
        let object = crate::SceneObjectId::new(
            crate::EntityId::from_raw(u128::MAX),
            crate::MeshId::from_raw(u128::MAX - 1),
            crate::SubobjectId::from_raw(u64::MAX),
        );
        let mut scene = crate::SceneIdPass::new();
        scene
            .submit([crate::SceneIdPassItem::new(object, 1, 2, true, 0)])
            .expect("submit scene");
        let encoded = scene.draw_entries()[0].encoded_id();
        let channels = encoded.channels();
        let shader_source = format!(
            "
                @vertex
                fn vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {{
                    var positions = array<vec2<f32>, 3>(
                        vec2<f32>(-1.0, -1.0),
                        vec2<f32>(3.0, -1.0),
                        vec2<f32>(-1.0, 3.0),
                    );
                    return vec4<f32>(positions[index], 0.0, 1.0);
                }}

                @fragment
                fn fs() -> @location(0) vec4<u32> {{
                    return vec4<u32>({}u, {}u, {}u, {}u);
                }}
            ",
            channels[0], channels[1], channels[2], channels[3],
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("editor-picking-test-shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("editor-picking-test-pipeline"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: ID_BUFFER_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        let mut picks = crate::PickCoordinator::new();
        for (draw, close) in [(false, false), (true, false), (true, true)] {
            let request = picks
                .submit(&viewport, target, &scene, PixelPoint { x: 2, y: 2 })
                .expect("pick request");
            let mut in_flight = {
                let render_target = targets.get(viewport.id()).expect("target");
                let frame = render_target.begin_frame(&scene);
                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("editor-picking-test-encoder"),
                });
                {
                    let mut pass =
                        frame.begin_render_pass(&mut encoder, Some("editor-picking-test-pass"));
                    if draw {
                        pass.set_pipeline(&pipeline);
                        pass.draw(0..3, 0..1);
                    }
                }
                let pending = frame
                    .encode_readback(&mut encoder, request)
                    .expect("encode pick readback");
                render_target.backend().submit_readback(encoder, pending)
            };
            assert_eq!(in_flight.request(), request);
            if close {
                targets.close(viewport.id()).expect("close before polling");
            }
            device
                .poll(PollType::wait_indefinitely())
                .expect("wait for async readback");
            let result = in_flight
                .poll()
                .expect("poll result")
                .expect("completed result");
            assert_eq!(
                result.encoded,
                if draw { encoded } else { EncodedPickId::none() }
            );
            assert_eq!(result.viewport, request.viewport);
            assert_eq!(result.generation, request.generation);
            assert_eq!(result.scene_generation, request.scene_generation);
            assert_eq!(result.target_generation, request.target_generation);
            assert_eq!(result.target_handle, request.target_handle);
            assert_eq!(scene.resolve(result.encoded), draw.then_some(object));
            assert_eq!(in_flight.poll().expect("second poll"), None);
            let expected = if close {
                crate::PickOutcome::Rejected(crate::PickRejectReason::StaleRenderTarget)
            } else {
                crate::PickOutcome::Accepted {
                    object: draw.then_some(object),
                }
            };
            assert_eq!(
                picks.resolve(&viewport, targets.state(viewport.id()), &scene, result),
                expected
            );
        }
        crate::test_support::diagnostic(format_args!(
            "EXERCISED viewport GPU draw/miss/readback/close correlation"
        ));
    }
}
