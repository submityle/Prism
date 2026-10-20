//! Actual texture snapshots taken inside the application's GPU command stream.
//!
//! Record the selected pass with [`TexturePreviewProducer::encoder`], call
//! [`TexturePreviewProducer::copy_texture`] before subsequent aliasing writes,
//! then continue recording or submit. Descriptor metadata never synthesizes pixels.

use std::sync::{
    mpsc::{self, Receiver, TryRecvError},
    Arc,
};

use crate::{capture::CaptureHandle, CapturedResourceLayout, GraphCaptureReport};

/// Independent caps for GPU staging allocation and CPU publication.
#[derive(Clone, Copy, Debug)]
pub struct TexturePreviewLimits {
    /// Slots, including cancelled submissions whose callbacks have not drained.
    pub max_in_flight: usize,
    /// Maximum padded staging allocation for one texture.
    pub max_bytes_per_texture: u64,
    /// Maximum simultaneous padded staging bytes across all occupied slots.
    pub max_total_bytes: u64,
    /// Maximum pixels in a single selected mip/layer.
    pub max_pixels: u64,
}
impl Default for TexturePreviewLimits {
    fn default() -> Self {
        Self {
            max_in_flight: 3,
            max_bytes_per_texture: 64 * 1024 * 1024,
            max_total_bytes: 128 * 1024 * 1024,
            max_pixels: 4 * 1024 * 1024,
        }
    }
}

/// Validation, lifecycle and backend failures; no fallback texture is returned.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TexturePreviewError {
    /// Invalid configured limits or display parameters.
    InvalidConfig,
    /// Pass/resource pair is not an access in the exact capture.
    MissingAccess,
    /// Texture layout differs from the selected authoritative resource.
    LayoutMismatch,
    /// Format/dimension/sample count cannot be copied with this recorder.
    UnsupportedTexture,
    /// Requested mip or layer does not exist.
    InvalidSubresource,
    /// Slot, byte or pixel limit would be exceeded.
    AtCapacity,
    /// Handle is cancelled, consumed or from a different producer.
    StaleHandle,
    /// Capture must record exactly one copy before submission.
    InvalidState,
    /// Device polling failed; pending slots remain occupied.
    GpuPoll,
    /// GPU mapping or mapped range failed.
    GpuMap,
    /// Process capture identity space was exhausted.
    Overflow,
}
impl std::fmt::Display for TexturePreviewError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "texture preview: {self:?}")
    }
}
impl std::error::Error for TexturePreviewError {}

/// Exact capture identity and original pass/resource vector positions.
#[derive(Clone, Debug)]
pub struct TexturePreviewBinding {
    report: Arc<GraphCaptureReport>,
    pass: u32,
    resource: u32,
}
impl TexturePreviewBinding {
    /// Validate the selected resource is actually used by this pass.
    pub fn new(
        report: Arc<GraphCaptureReport>,
        pass: u32,
        resource: u32,
    ) -> Result<Self, TexturePreviewError> {
        let p = report
            .passes()
            .get(pass as usize)
            .ok_or(TexturePreviewError::MissingAccess)?;
        if report.resources().get(resource as usize).is_none()
            || !p.accesses.iter().any(|a| a.resource == resource)
        {
            return Err(TexturePreviewError::MissingAccess);
        }
        Ok(Self {
            report,
            pass,
            resource,
        })
    }
    /// Immutable report retaining its exact publication identity.
    pub fn report(&self) -> &Arc<GraphCaptureReport> {
        &self.report
    }
    /// Selected pass vector index (not its semantic key).
    pub const fn pass(&self) -> u32 {
        self.pass
    }
    /// Selected resource vector index.
    pub const fn resource(&self) -> u32 {
        self.resource
    }
    /// Equal numeric frame IDs from independent captures are never interchangeable.
    pub fn matches(&self, report: &Arc<GraphCaptureReport>, pass: u32, resource: u32) -> bool {
        Arc::ptr_eq(&self.report, report) && self.pass == pass && self.resource == resource
    }
}

struct Readback {
    buffer: wgpu::Buffer,
    width: u32,
    height: u32,
    pitch: u32,
    row_bytes: u32,
    format: wgpu::TextureFormat,
    mip: u32,
    layer: u32,
}
impl Readback {
    fn bytes(&self) -> u64 {
        u64::from(self.pitch) * u64::from(self.height)
    }
}
struct Recording {
    handle: CaptureHandle,
    binding: TexturePreviewBinding,
    encoder: wgpu::CommandEncoder,
    copy: Option<Readback>,
}
enum Slot {
    Idle,
    Recording(Recording),
    Pending {
        handle: CaptureHandle,
        binding: TexturePreviewBinding,
        copy: Readback,
        receiver: Receiver<Result<(), wgpu::BufferAsyncError>>,
        cancelled: bool,
    },
}
impl Slot {
    fn handle(&self) -> Option<CaptureHandle> {
        match self {
            Self::Recording(r) => Some(r.handle),
            Self::Pending {
                handle,
                cancelled: false,
                ..
            } => Some(*handle),
            _ => None,
        }
    }
    fn bytes(&self) -> u64 {
        match self {
            Self::Recording(r) => r.copy.as_ref().map_or(0, Readback::bytes),
            Self::Pending { copy, .. } => copy.bytes(),
            Self::Idle => 0,
        }
    }
}

/// One real, tightly packed, unmodified GPU texture subresource.
#[derive(Debug)]
pub struct TexturePreview {
    binding: TexturePreviewBinding,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
    mip: u32,
    layer: u32,
    bytes: Vec<u8>,
}
impl TexturePreview {
    /// Exact immutable report/pass/resource association.
    pub fn binding(&self) -> &TexturePreviewBinding {
        &self.binding
    }
    /// Copied mip width.
    pub const fn width(&self) -> u32 {
        self.width
    }
    /// Copied mip height.
    pub const fn height(&self) -> u32 {
        self.height
    }
    /// Original GPU format; raw bytes are not display transformed.
    pub const fn format(&self) -> wgpu::TextureFormat {
        self.format
    }
    /// Selected mip.
    pub const fn mip(&self) -> u32 {
        self.mip
    }
    /// Selected array layer.
    pub const fn layer(&self) -> u32 {
        self.layer
    }
    /// Raw little endian texels, top row first, with staging padding removed.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
    /// Actual texel values. Color sRGB formats are converted to linear light;
    /// integer and depth channels retain their numeric values. The f64 result
    /// represents every u32 ID and every source f32 exactly.
    pub fn pixel(&self, x: u32, y: u32) -> Option<[f64; 4]> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let stride = bytes_per_pixel(self.format)? as usize;
        let at = (u64::from(y) * u64::from(self.width) + u64::from(x)) as usize * stride;
        let b = &self.bytes[at..at + stride];
        use wgpu::TextureFormat as F;
        let mut rgba = match self.format {
            F::Rgba8Unorm | F::Rgba8UnormSrgb => std::array::from_fn(|i| f64::from(b[i]) / 255.0),
            F::Bgra8Unorm | F::Bgra8UnormSrgb => [
                f64::from(b[2]) / 255.0,
                f64::from(b[1]) / 255.0,
                f64::from(b[0]) / 255.0,
                f64::from(b[3]) / 255.0,
            ],
            F::Rgba16Float => std::array::from_fn(|i| {
                f64::from(half(u16::from_le_bytes([b[i * 2], b[i * 2 + 1]])))
            }),
            F::Rgba32Float => std::array::from_fn(|i| {
                f64::from(f32::from_le_bytes(
                    b[i * 4..i * 4 + 4].try_into().expect("four bytes"),
                ))
            }),
            F::R32Float | F::Depth32Float => [
                f64::from(f32::from_le_bytes(b.try_into().expect("four bytes"))),
                0.0,
                0.0,
                1.0,
            ],
            F::R32Uint => [
                f64::from(u32::from_le_bytes(b.try_into().expect("four bytes"))),
                0.0,
                0.0,
                1.0,
            ],
            _ => return None,
        };
        if matches!(self.format, F::Rgba8UnormSrgb | F::Bgra8UnormSrgb) {
            for channel in &mut rgba[..3] {
                *channel = if *channel <= 0.04045 {
                    *channel / 12.92
                } else {
                    ((*channel + 0.055) / 1.055).powf(2.4)
                };
            }
        }
        Some(rgba)
    }
    /// Display transform into sRGB straight-alpha RGBA8. Invalid GPU numbers
    /// appear magenta. Scalar mode uses the explicit linear diagnostic range.
    pub fn rgba8(&self, display: TexturePreviewDisplay) -> Result<Vec<u8>, TexturePreviewError> {
        if !display.exposure_ev.is_finite()
            || display.exposure_ev.abs() > 24.0
            || !display.range[0].is_finite()
            || !display.range[1].is_finite()
            || !(display.range[1] - display.range[0]).is_finite()
            || display.range[0] >= display.range[1]
            || display.channel.is_some_and(|c| c > 3)
        {
            return Err(TexturePreviewError::InvalidConfig);
        }
        let mut out = Vec::with_capacity(self.width as usize * self.height as usize * 4);
        let exposure = f64::from(display.exposure_ev).exp2();
        let range = display.range[1] - display.range[0];
        for y in 0..self.height {
            for x in 0..self.width {
                let mut p = self.pixel(x, y).expect("validated supported texture");
                if p.iter().any(|n| !n.is_finite()) {
                    out.extend_from_slice(&[255, 0, 255, 255]);
                    continue;
                }
                if let Some(channel) = display.channel {
                    let t = ((p[channel] - display.range[0]) / range).clamp(0.0, 1.0);
                    let v = (t * 255.0).round() as u8;
                    out.extend_from_slice(&[v, v, v, 255]);
                } else {
                    for v in &mut p[..3] {
                        let mut value = (*v * exposure).max(0.0);
                        if display.reinhard {
                            value /= 1.0 + value;
                        }
                        value = value.clamp(0.0, 1.0);
                        let srgb = if value <= 0.0031308 {
                            value * 12.92
                        } else {
                            1.055 * value.powf(1.0 / 2.4) - 0.055
                        };
                        out.push((srgb * 255.0).round() as u8);
                    }
                    out.push((p[3].clamp(0.0, 1.0) * 255.0).round() as u8);
                }
            }
        }
        Ok(out)
    }
}
/// Explicit color/depth/channel inspection policy; no automatic HDR guessing.
#[derive(Clone, Copy, Debug)]
pub struct TexturePreviewDisplay {
    /// None for color; Some(0..=3) for a grayscale channel.
    pub channel: Option<usize>,
    /// Scalar diagnostic range, useful for depth and integer ID textures.
    pub range: [f64; 2],
    /// Linear color exposure in stops.
    pub exposure_ev: f32,
    /// Apply x/(1+x) before the sRGB display transfer.
    pub reinhard: bool,
}
impl Default for TexturePreviewDisplay {
    fn default() -> Self {
        Self {
            channel: None,
            range: [0.0, 1.0],
            exposure_ev: 0.0,
            reinhard: false,
        }
    }
}
/// One noncancelled completion. A mapping failure still retains its identity.
pub struct TexturePreviewResult {
    /// Process unique capture generation.
    pub handle: CaptureHandle,
    /// Actual pixels or explicit GPU failure.
    pub preview: Result<TexturePreview, TexturePreviewError>,
}
/// Bounded owned encoders and asynchronous readbacks on the application's device.
pub struct TexturePreviewProducer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    limits: TexturePreviewLimits,
    slots: Vec<Slot>,
}
impl TexturePreviewProducer {
    /// `queue` must belong to `device`. Limit validation allocates no GPU buffers.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        limits: TexturePreviewLimits,
    ) -> Result<Self, TexturePreviewError> {
        if limits.max_in_flight == 0
            || limits.max_in_flight > 8
            || limits.max_pixels == 0
            || limits.max_pixels > 16 * 1024 * 1024
            || limits.max_bytes_per_texture == 0
            || limits.max_bytes_per_texture > 256 * 1024 * 1024
            || limits.max_total_bytes < limits.max_bytes_per_texture
            || limits.max_total_bytes > 512 * 1024 * 1024
        {
            return Err(TexturePreviewError::InvalidConfig);
        }
        Ok(Self {
            device: device.clone(),
            queue: queue.clone(),
            limits,
            slots: (0..limits.max_in_flight).map(|_| Slot::Idle).collect(),
        })
    }
    /// Start an exact selected capture. Real passes and uploads can be encoded
    /// before the copy and later passes after it in the same command stream.
    pub fn begin(
        &mut self,
        binding: TexturePreviewBinding,
    ) -> Result<CaptureHandle, TexturePreviewError> {
        let slot = self
            .slots
            .iter_mut()
            .find(|s| matches!(s, Slot::Idle))
            .ok_or(TexturePreviewError::AtCapacity)?;
        let handle = CaptureHandle::next().map_err(|_| TexturePreviewError::Overflow)?;
        *slot = Slot::Recording(Recording {
            handle,
            binding,
            encoder: self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("editor texture preview stream"),
                }),
            copy: None,
        });
        Ok(handle)
    }
    /// Borrow the command stream; all render/compute passes must end before copy.
    pub fn encoder(
        &mut self,
        handle: CaptureHandle,
    ) -> Result<&mut wgpu::CommandEncoder, TexturePreviewError> {
        match self.slot_mut(handle)? {
            Slot::Recording(r) => Ok(&mut r.encoder),
            _ => Err(TexturePreviewError::InvalidState),
        }
    }
    /// Copy one actual D2 mip/layer at this precise point in the command stream.
    /// Requires COPY_SRC, a supported uncompressed format and one sample.
    /// Rejections leave the encoder and byte budget unchanged.
    pub fn copy_texture(
        &mut self,
        handle: CaptureHandle,
        texture: &wgpu::Texture,
        mip: u32,
        layer: u32,
    ) -> Result<(), TexturePreviewError> {
        let occupied = self.allocated_bytes();
        let limits = self.limits;
        let device = self.device.clone();
        let Slot::Recording(r) = self.slot_mut(handle)? else {
            return Err(TexturePreviewError::InvalidState);
        };
        if r.copy.is_some() {
            return Err(TexturePreviewError::InvalidState);
        }
        let expected = &r.binding.report.resources()[r.binding.resource as usize].layout;
        let CapturedResourceLayout::Texture {
            extent,
            dimension,
            format,
            mip_levels,
            samples,
            ..
        } = expected
        else {
            return Err(TexturePreviewError::LayoutMismatch);
        };
        let size = texture.size();
        if *extent != [size.width, size.height, size.depth_or_array_layers]
            || *dimension != 2
            || texture.dimension() != wgpu::TextureDimension::D2
            || *format != format!("{:?}", texture.format())
            || *mip_levels != texture.mip_level_count()
            || *samples != texture.sample_count()
        {
            return Err(TexturePreviewError::LayoutMismatch);
        }
        let bpp =
            bytes_per_pixel(texture.format()).ok_or(TexturePreviewError::UnsupportedTexture)?;
        if texture.sample_count() != 1 || !texture.usage().contains(wgpu::TextureUsages::COPY_SRC) {
            return Err(TexturePreviewError::UnsupportedTexture);
        }
        if mip >= texture.mip_level_count() || layer >= size.depth_or_array_layers {
            return Err(TexturePreviewError::InvalidSubresource);
        }
        let width = (size.width >> mip).max(1);
        let height = (size.height >> mip).max(1);
        let row_bytes = width
            .checked_mul(bpp)
            .ok_or(TexturePreviewError::AtCapacity)?;
        let pitch = row_bytes
            .checked_add(255)
            .map(|n| n & !255)
            .ok_or(TexturePreviewError::AtCapacity)?;
        let bytes = u64::from(pitch) * u64::from(height);
        if u64::from(width) * u64::from(height) > limits.max_pixels
            || bytes > limits.max_bytes_per_texture
            || bytes > device.limits().max_buffer_size
            || occupied
                .checked_add(bytes)
                .is_none_or(|n| n > limits.max_total_bytes)
        {
            return Err(TexturePreviewError::AtCapacity);
        }
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("editor actual texture staging"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        r.encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: mip,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: 0,
                    z: layer,
                },
                aspect: if texture.format() == wgpu::TextureFormat::Depth32Float {
                    wgpu::TextureAspect::DepthOnly
                } else {
                    wgpu::TextureAspect::All
                },
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(pitch),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        r.copy = Some(Readback {
            buffer,
            width,
            height,
            pitch,
            row_bytes,
            format: texture.format(),
            mip,
            layer,
        });
        Ok(())
    }
    /// Submit once, then map asynchronously. Call poll even after cancellation.
    pub fn submit(
        &mut self,
        handle: CaptureHandle,
    ) -> Result<wgpu::SubmissionIndex, TexturePreviewError> {
        self.submit_ordered(handle, [], [])
    }
    /// Submit native work before and after the exact texture copy in one ordered
    /// queue submission. All command buffers must belong to this device.
    /// No command buffer is cloned and no completion wait occurs.
    pub fn submit_ordered(
        &mut self,
        handle: CaptureHandle,
        before: impl IntoIterator<Item = wgpu::CommandBuffer>,
        after: impl IntoIterator<Item = wgpu::CommandBuffer>,
    ) -> Result<wgpu::SubmissionIndex, TexturePreviewError> {
        let queue = self.queue.clone();
        let slot = self.slot_mut(handle)?;
        if !matches!(slot, Slot::Recording(r) if r.copy.is_some()) {
            return Err(TexturePreviewError::InvalidState);
        }
        let Slot::Recording(r) = std::mem::replace(slot, Slot::Idle) else {
            unreachable!();
        };
        let copy = r.copy.expect("validated copy");
        let submission = queue.submit(before.into_iter().chain([r.encoder.finish()]).chain(after));
        let (sender, receiver) = mpsc::sync_channel(1);
        copy.buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = sender.try_send(result);
            });
        *slot = Slot::Pending {
            handle,
            binding: r.binding,
            copy,
            receiver,
            cancelled: false,
        };
        Ok(submission)
    }
    /// Nonblocking poll. Submitted cancellations retain both slot and byte
    /// charges until the mapping callback is observed and the buffer unmapped.
    pub fn poll(&mut self) -> Result<Vec<TexturePreviewResult>, TexturePreviewError> {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.device.poll(wgpu::PollType::Poll)
        }))
        .map_err(|_| TexturePreviewError::GpuPoll)?
        .map_err(|_| TexturePreviewError::GpuPoll)?;
        let mut results = Vec::new();
        for slot in &mut self.slots {
            let Slot::Pending { receiver, .. } = slot else {
                continue;
            };
            let ready = match receiver.try_recv() {
                Ok(result) => result.map_err(|_| TexturePreviewError::GpuMap),
                Err(TryRecvError::Empty) => continue,
                Err(TryRecvError::Disconnected) => Err(TexturePreviewError::GpuMap),
            };
            let Slot::Pending {
                handle,
                binding,
                copy,
                cancelled,
                ..
            } = std::mem::replace(slot, Slot::Idle)
            else {
                unreachable!();
            };
            if !cancelled {
                let preview = ready.and_then(|()| {
                    let mapped = copy
                        .buffer
                        .slice(..)
                        .get_mapped_range()
                        .map_err(|_| TexturePreviewError::GpuMap)?;
                    let mut bytes =
                        Vec::with_capacity(copy.row_bytes as usize * copy.height as usize);
                    for row in mapped.chunks_exact(copy.pitch as usize) {
                        bytes.extend_from_slice(&row[..copy.row_bytes as usize]);
                    }
                    Ok(TexturePreview {
                        binding,
                        width: copy.width,
                        height: copy.height,
                        format: copy.format,
                        mip: copy.mip,
                        layer: copy.layer,
                        bytes,
                    })
                });
                results.push(TexturePreviewResult { handle, preview });
            }
            copy.buffer.unmap();
        }
        Ok(results)
    }
    /// Cancel recording immediately or quarantine a submitted map.
    pub fn cancel(&mut self, handle: CaptureHandle) -> Result<(), TexturePreviewError> {
        let slot = self.slot_mut(handle)?;
        match slot {
            Slot::Recording(_) => *slot = Slot::Idle,
            Slot::Pending { cancelled, .. } => *cancelled = true,
            Slot::Idle => unreachable!(),
        }
        Ok(())
    }
    /// Occupied slots including cancelled pending maps.
    pub fn occupied_slots(&self) -> usize {
        self.slots
            .iter()
            .filter(|s| !matches!(s, Slot::Idle))
            .count()
    }
    /// Current padded GPU staging allocation, including cancelled pending maps.
    pub fn allocated_bytes(&self) -> u64 {
        self.slots.iter().map(Slot::bytes).sum()
    }
    fn slot_mut(&mut self, handle: CaptureHandle) -> Result<&mut Slot, TexturePreviewError> {
        self.slots
            .iter_mut()
            .find(|s| s.handle() == Some(handle))
            .ok_or(TexturePreviewError::StaleHandle)
    }
}
fn bytes_per_pixel(format: wgpu::TextureFormat) -> Option<u32> {
    use wgpu::TextureFormat as F;
    match format {
        F::Rgba8Unorm
        | F::Rgba8UnormSrgb
        | F::Bgra8Unorm
        | F::Bgra8UnormSrgb
        | F::R32Float
        | F::R32Uint
        | F::Depth32Float => Some(4),
        F::Rgba16Float => Some(8),
        F::Rgba32Float => Some(16),
        _ => None,
    }
}
fn half(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exponent = u32::from((bits >> 10) & 31);
    let mantissa = u32::from(bits & 1023);
    // All binary16 values fit exactly in binary32. Normalize subnormals using
    // integer bits; large HDR images do not need per-channel powers.
    let value = match (exponent, mantissa) {
        (0, 0) => 0,
        (0, _) => {
            let shift = mantissa.leading_zeros() - 21;
            ((113 - shift) << 23) | (((mantissa << shift) & 1023) << 13)
        }
        (31, _) => 0x7f80_0000 | (mantissa << 13),
        _ => ((exponent + 112) << 23) | (mantissa << 13),
    };
    f32::from_bits(sign | value)
}
