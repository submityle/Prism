//! Bounded real PNG thumbnails with revision-keyed caching and asynchronous I/O.
//! Host revisions are authoritative: change the revision when source content
//! changes. Filesystem metadata checks detect ordinary concurrent writes, not
//! malicious replacements preserving metadata. Drop cancels without blocking;
//! the worker retains its slot until any blocking OS call and cleanup finish.

use crate::AssetId;
use alloc::{
    collections::BTreeMap,
    string::{String, ToString},
    sync::Arc,
    vec,
    vec::Vec,
};
use core::{
    fmt,
    sync::atomic::{AtomicBool, AtomicUsize, Ordering},
    task::Poll,
    time::Duration,
};
use std::{
    fs,
    io::{Cursor, Read},
    path::PathBuf,
    sync::Mutex,
    thread::{self, JoinHandle},
    time::Instant,
};

/// Cache identity for one source revision and requested aspect-fit dimensions.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct ThumbnailKey {
    /// Stable asset identity from the host manifest.
    pub asset: AssetId,
    /// Host-owned source revision; must change after any effective source edit.
    pub revision: u128,
    /// Absolute source file path, included in cache identity.
    pub source: PathBuf,
    /// Maximum output width; images are never upscaled.
    pub width: u32,
    /// Maximum output height; original aspect ratio is preserved within rounding.
    pub height: u32,
}

/// Bounds each decoder/worker and the cache retained by one shared service.
#[derive(Clone, Copy, Debug)]
pub struct ThumbnailLimits {
    /// Running and completed-but-unconsumed requests, including cached hits.
    pub max_in_flight: usize,
    /// Maximum bytes read from a source file.
    pub max_encoded_bytes: usize,
    /// Maximum decoded source pixels, checked before frame allocation.
    pub max_source_pixels: usize,
    /// PNG decoder internal allocation budget, separate from source/frame/output.
    pub max_decoder_bytes: usize,
    /// Maximum width/height accepted for an output request.
    pub max_output_edge: u32,
    /// Maximum RGBA bytes allocated for one requested bounding rectangle.
    pub max_output_bytes: usize,
    /// Cache entry ceiling. Zero disables retained caching.
    pub max_cache_entries: usize,
    /// Retained RGBA pixel bytes. Caller-owned Arc references are not counted.
    pub max_cache_bytes: usize,
    /// Bounds native source path bytes in requests and cache keys.
    pub max_path_bytes: usize,
    /// Cooperative worker deadline, including time before thread scheduling.
    pub timeout: Duration,
}
impl Default for ThumbnailLimits {
    fn default() -> Self {
        Self {
            max_in_flight: 4,
            max_encoded_bytes: 32 * 1024 * 1024,
            max_source_pixels: 16 * 1024 * 1024,
            max_decoder_bytes: 128 * 1024 * 1024,
            max_output_edge: 512,
            max_output_bytes: 1024 * 1024,
            max_cache_entries: 512,
            max_cache_bytes: 64 * 1024 * 1024,
            max_path_bytes: 4096,
            timeout: Duration::from_secs(10),
        }
    }
}

/// Immutable straight-alpha sRGB RGBA8 pixels, ready for host GPU upload.
#[derive(Debug)]
pub struct Thumbnail {
    width: u32,
    height: u32,
    rgba8: Vec<u8>,
}
impl Thumbnail {
    /// Aspect-fit output width.
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }
    /// Aspect-fit output height.
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }
    /// Immutable tightly packed sRGB RGBA8 data.
    #[must_use]
    pub fn rgba8(&self) -> &[u8] {
        &self.rgba8
    }
}

/// Failures produce no thumbnail and publish no cache entry.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ThumbnailError {
    /// Invalid service or request configuration.
    InvalidConfiguration,
    /// Named resource exceeded its configured ceiling.
    Limit(&'static str),
    /// All shared request slots are occupied.
    Busy,
    /// Native file or worker creation failure.
    Io(String),
    /// PNG validation/decode failure.
    Decode(String),
    /// Animated PNG, unsupported color profile, or unhandled decoded format.
    UnsupportedFormat,
    /// Source metadata changed during the read/decode.
    SourceChanged,
    /// A directory, symlink, or special file was supplied.
    NotRegularFile,
    /// Request was cancelled, including a ready result not yet consumed.
    Cancelled,
    /// Cooperative deadline exceeded.
    TimedOut,
    /// Cache was cleared after the request started.
    Stale,
    /// Worker panic contained at the join boundary.
    WorkerPanicked,
    /// Result was already consumed.
    AlreadyConsumed,
    /// Cache mutex poisoned by another host panic.
    Poisoned,
}
impl fmt::Display for ThumbnailError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "thumbnail failed: {self:?}")
    }
}
impl core::error::Error for ThumbnailError {}

struct Entry {
    image: Arc<Thumbnail>,
    used: u64,
}
struct Cache {
    entries: BTreeMap<ThumbnailKey, Entry>,
    bytes: usize,
    clock: u64,
    epoch: Arc<()>,
}
struct Shared {
    limits: ThumbnailLimits,
    active: AtomicUsize,
    cache: Mutex<Cache>,
}

/// Shared bounded service; clones share request and cache budgets.
#[derive(Clone)]
pub struct ThumbnailService {
    shared: Arc<Shared>,
}
impl ThumbnailService {
    /// Construct without filesystem access. Zero cache budgets disable caching.
    pub fn new(limits: ThumbnailLimits) -> Result<Self, ThumbnailError> {
        if limits.max_in_flight == 0
            || limits.max_encoded_bytes == 0
            || limits.max_source_pixels == 0
            || limits.max_decoder_bytes == 0
            || limits.max_output_edge == 0
            || limits.max_output_bytes == 0
            || limits.max_path_bytes == 0
            || limits.max_encoded_bytes == usize::MAX
            || Instant::now().checked_add(limits.timeout).is_none()
        {
            return Err(ThumbnailError::InvalidConfiguration);
        }
        Ok(Self {
            shared: Arc::new(Shared {
                limits,
                active: AtomicUsize::new(0),
                cache: Mutex::new(Cache {
                    entries: BTreeMap::new(),
                    bytes: 0,
                    clock: 0,
                    epoch: Arc::new(()),
                }),
            }),
        })
    }

    /// Current reserved requests, including ready results and worker cleanup.
    #[must_use]
    pub fn in_flight(&self) -> usize {
        self.shared.active.load(Ordering::Acquire)
    }

    /// Retained cache entries and pixel bytes (excluding caller Arc ownership).
    pub fn cache_usage(&self) -> Result<(usize, usize), ThumbnailError> {
        let cache = self
            .shared
            .cache
            .lock()
            .map_err(|_| ThumbnailError::Poisoned)?;
        Ok((cache.entries.len(), cache.bytes))
    }

    /// Clear retained pixels and revoke every outstanding request epoch. This
    /// prevents late completions from repopulating a closed/replaced project.
    pub fn clear(&self) -> Result<(), ThumbnailError> {
        let mut cache = self
            .shared
            .cache
            .lock()
            .map_err(|_| ThumbnailError::Poisoned)?;
        cache.entries.clear();
        cache.bytes = 0;
        cache.epoch = Arc::new(());
        Ok(())
    }

    /// Start a real decode, or return a cached immutable image. No file I/O runs
    /// on the caller thread. Completed results reserve capacity until consumed.
    pub fn request(&self, key: ThumbnailKey) -> Result<ThumbnailRequest, ThumbnailError> {
        let limits = self.shared.limits;
        if key.width == 0
            || key.height == 0
            || key.width > limits.max_output_edge
            || key.height > limits.max_output_edge
            || !key.source.is_absolute()
        {
            return Err(ThumbnailError::InvalidConfiguration);
        }
        if key.source.as_os_str().len() > limits.max_path_bytes {
            return Err(ThumbnailError::Limit("source path bytes"));
        }
        if (key.width as usize)
            .checked_mul(key.height as usize)
            .and_then(|n| n.checked_mul(4))
            .is_none_or(|bytes| bytes > limits.max_output_bytes)
        {
            return Err(ThumbnailError::Limit("output bytes"));
        }
        self.shared
            .active
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |count| {
                (count < limits.max_in_flight).then(|| count + 1)
            })
            .map_err(|_| ThumbnailError::Busy)?;
        let permit = Arc::new(Permit(self.shared.clone()));
        let cancelled = Arc::new(AtomicBool::new(false));
        let mut cache = self
            .shared
            .cache
            .lock()
            .map_err(|_| ThumbnailError::Poisoned)?;
        let epoch = cache.epoch.clone();
        let used = cache.tick();
        let cached = cache.entries.get_mut(&key).map(|entry| {
            entry.used = used;
            entry.image.clone()
        });
        drop(cache);
        let worker = if cached.is_none() {
            let worker_key = key.clone();
            let worker_cancelled = cancelled.clone();
            let worker_permit = permit.clone();
            let deadline = Instant::now()
                .checked_add(limits.timeout)
                .ok_or(ThumbnailError::InvalidConfiguration)?;
            Some(
                thread::Builder::new()
                    .name("prism-thumbnail".into())
                    .spawn(move || {
                        let _permit = worker_permit;
                        produce(&worker_key, limits, &worker_cancelled, deadline).map(Arc::new)
                    })
                    .map_err(|error| ThumbnailError::Io(error.to_string()))?,
            )
        } else {
            None
        };
        Ok(ThumbnailRequest {
            key,
            shared: self.shared.clone(),
            epoch,
            cancelled,
            worker,
            cached,
            permit: Some(permit),
            consumed: false,
        })
    }
}
impl Cache {
    fn tick(&mut self) -> u64 {
        if self.clock == u64::MAX {
            // Preserve LRU order across clock rollover without per-frame work.
            let mut order: Vec<_> = self
                .entries
                .iter()
                .map(|(key, entry)| (entry.used, key.clone()))
                .collect();
            order.sort();
            for (index, (_, key)) in order.into_iter().enumerate() {
                self.entries.get_mut(&key).expect("cache key retained").used = index as u64;
            }
            self.clock = self.entries.len() as u64;
        }
        self.clock += 1;
        self.clock
    }
    fn insert(&mut self, key: ThumbnailKey, image: Arc<Thumbnail>, limits: ThumbnailLimits) {
        let bytes = image.rgba8.len();
        if limits.max_cache_entries == 0 || bytes > limits.max_cache_bytes {
            return;
        }
        if let Some(old) = self.entries.remove(&key) {
            self.bytes -= old.image.rgba8.len();
        }
        while self.entries.len() >= limits.max_cache_entries
            || bytes > limits.max_cache_bytes - self.bytes
        {
            let victim = self
                .entries
                .iter()
                .min_by_key(|(_, entry)| entry.used)
                .map(|(key, _)| key.clone());
            let Some(victim) = victim else { break };
            if let Some(old) = self.entries.remove(&victim) {
                self.bytes -= old.image.rgba8.len();
            }
        }
        let used = self.tick();
        self.bytes += bytes;
        self.entries.insert(key, Entry { image, used });
    }
}
struct Permit(Arc<Shared>);
impl Drop for Permit {
    fn drop(&mut self) {
        self.0.active.fetch_sub(1, Ordering::AcqRel);
    }
}

/// One asynchronous thumbnail. Drop cancels; a worker retains its slot through
/// cleanup. Poll after `is_finished` to observe terminal errors without waiting.
pub struct ThumbnailRequest {
    key: ThumbnailKey,
    shared: Arc<Shared>,
    epoch: Arc<()>,
    cancelled: Arc<AtomicBool>,
    worker: Option<JoinHandle<Result<Arc<Thumbnail>, ThumbnailError>>>,
    cached: Option<Arc<Thumbnail>>,
    permit: Option<Arc<Permit>>,
    consumed: bool,
}
impl ThumbnailRequest {
    /// Exact host key used by the request.
    #[must_use]
    pub fn key(&self) -> &ThumbnailKey {
        &self.key
    }
    /// Signal cancellation; even an unconsumed completed image is discarded.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }
    /// Whether polling can return without waiting on an active thread.
    #[must_use]
    pub fn is_finished(&self) -> bool {
        self.worker.as_ref().is_none_or(JoinHandle::is_finished)
    }
    /// Nonblocking completion and cache publication. Cancelled/stale/failed
    /// work is never cached; clearing the service also rejects cached hits.
    pub fn poll(&mut self) -> Poll<Result<Arc<Thumbnail>, ThumbnailError>> {
        if self.consumed {
            return Poll::Ready(Err(ThumbnailError::AlreadyConsumed));
        }
        if !self.is_finished() {
            return Poll::Pending;
        }
        self.consumed = true;
        let result = if let Some(worker) = self.worker.take() {
            worker.join().unwrap_or(Err(ThumbnailError::WorkerPanicked))
        } else {
            self.cached.take().ok_or(ThumbnailError::AlreadyConsumed)
        };
        self.permit.take();
        if self.cancelled.load(Ordering::Acquire) {
            return Poll::Ready(Err(ThumbnailError::Cancelled));
        }
        let result = result.and_then(|image| {
            let mut cache = self
                .shared
                .cache
                .lock()
                .map_err(|_| ThumbnailError::Poisoned)?;
            if !Arc::ptr_eq(&self.epoch, &cache.epoch) {
                return Err(ThumbnailError::Stale);
            }
            cache.insert(self.key.clone(), image.clone(), self.shared.limits);
            Ok(image)
        });
        Poll::Ready(result)
    }
}
impl Drop for ThumbnailRequest {
    fn drop(&mut self) {
        self.cancel();
    }
}

fn checkpoint(cancelled: &AtomicBool, deadline: Instant) -> Result<(), ThumbnailError> {
    if cancelled.load(Ordering::Acquire) {
        Err(ThumbnailError::Cancelled)
    } else if Instant::now() >= deadline {
        Err(ThumbnailError::TimedOut)
    } else {
        Ok(())
    }
}
fn same_source(left: &fs::Metadata, right: &fs::Metadata) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if left.dev() != right.dev() || left.ino() != right.ino() {
            return false;
        }
    }
    left.is_file()
        && right.is_file()
        && left.len() == right.len()
        && left.modified().ok() == right.modified().ok()
}
fn produce(
    key: &ThumbnailKey,
    limits: ThumbnailLimits,
    cancelled: &AtomicBool,
    deadline: Instant,
) -> Result<Thumbnail, ThumbnailError> {
    checkpoint(cancelled, deadline)?;
    let before =
        fs::symlink_metadata(&key.source).map_err(|e| ThumbnailError::Io(e.to_string()))?;
    if !before.is_file() {
        return Err(ThumbnailError::NotRegularFile);
    }
    if before.len() > limits.max_encoded_bytes as u64 {
        return Err(ThumbnailError::Limit("encoded bytes"));
    }
    let mut file = fs::File::open(&key.source).map_err(|e| ThumbnailError::Io(e.to_string()))?;
    let opened = file
        .metadata()
        .map_err(|e| ThumbnailError::Io(e.to_string()))?;
    if !opened.is_file() {
        return Err(ThumbnailError::NotRegularFile);
    }
    if !same_source(&before, &opened) {
        return Err(ThumbnailError::SourceChanged);
    }
    let mut bytes = Vec::new();
    // Bounded chunks allow cancellation between ordinary file reads.
    let mut chunk = [0u8; 16 * 1024];
    loop {
        checkpoint(cancelled, deadline)?;
        let available = (limits.max_encoded_bytes + 1 - bytes.len()).min(chunk.len());
        let read = file
            .read(&mut chunk[..available])
            .map_err(|e| ThumbnailError::Io(e.to_string()))?;
        bytes.extend_from_slice(&chunk[..read]);
        if bytes.len() > limits.max_encoded_bytes {
            return Err(ThumbnailError::Limit("encoded bytes"));
        }
        if read == 0 {
            break;
        }
    }
    let mut decoder = png::Decoder::new_with_limits(
        Cursor::new(bytes),
        png::Limits {
            bytes: limits.max_decoder_bytes,
        },
    );
    decoder.set_ignore_text_chunk(true);
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = decoder
        .read_info()
        .map_err(|e| ThumbnailError::Decode(e.to_string()))?;
    let info = reader.info();
    if info.animation_control.is_some()
        || info.icc_profile.is_some()
        || info.coding_independent_code_points.is_some()
        || info.mastering_display_color_volume.is_some()
        || info.content_light_level.is_some()
        || (info.srgb.is_none() && info.chromaticities().is_some())
    {
        return Err(ThumbnailError::UnsupportedFormat);
    }
    // Untagged pixels use the sRGB convention. gAMA-only images use their
    // declared transfer curve with assumed sRGB primaries; explicit alternate
    // primaries/profiles require a color-management adapter.
    let gamma = if info.srgb.is_none() {
        info.gamma().map(|g| f64::from(g.into_value()))
    } else {
        None
    };
    if gamma.is_some_and(|g| !g.is_finite() || g <= 0.0) {
        return Err(ThumbnailError::UnsupportedFormat);
    }
    let pixels = (info.width as usize)
        .checked_mul(info.height as usize)
        .filter(|pixels| *pixels <= limits.max_source_pixels)
        .ok_or(ThumbnailError::Limit("source pixels"))?;
    let len = reader
        .output_buffer_size()
        .filter(|len| *len <= pixels.saturating_mul(4))
        .ok_or(ThumbnailError::Limit("decoded frame bytes"))?;
    let mut decoded = vec![0; len];
    checkpoint(cancelled, deadline)?;
    let frame = reader
        .next_frame(&mut decoded)
        .map_err(|e| ThumbnailError::Decode(e.to_string()))?;
    reader
        .finish()
        .map_err(|e| ThumbnailError::Decode(e.to_string()))?;
    checkpoint(cancelled, deadline)?;
    let after = fs::symlink_metadata(&key.source).map_err(|e| ThumbnailError::Io(e.to_string()))?;
    if !same_source(&before, &after) || !same_source(&opened, &after) {
        return Err(ThumbnailError::SourceChanged);
    }
    if frame.bit_depth != png::BitDepth::Eight {
        return Err(ThumbnailError::UnsupportedFormat);
    }
    let channels = match frame.color_type {
        png::ColorType::Rgba => 4,
        png::ColorType::Rgb => 3,
        png::ColorType::Grayscale => 1,
        png::ColorType::GrayscaleAlpha => 2,
        _ => return Err(ThumbnailError::UnsupportedFormat),
    };
    resample(
        &decoded[..frame.buffer_size()],
        &frame,
        channels,
        gamma,
        key,
        cancelled,
        deadline,
    )
}

// Area filter in linear light with premultiplied alpha prevents dark fringes
// and undersampling of high-frequency detail. Each source pixel participates
// in at most four output footprints during downsampling; no integral-image copy.
fn resample(
    source: &[u8],
    frame: &png::OutputInfo,
    channels: usize,
    gamma: Option<f64>,
    key: &ThumbnailKey,
    cancelled: &AtomicBool,
    deadline: Instant,
) -> Result<Thumbnail, ThumbnailError> {
    let (width, height) = (frame.width, frame.height);
    let scale = (f64::from(key.width) / f64::from(width))
        .min(f64::from(key.height) / f64::from(height))
        .min(1.0);
    let output_width = (f64::from(width) * scale).round().max(1.0) as u32;
    let output_height = (f64::from(height) * scale).round().max(1.0) as u32;
    let output_len = (output_width as usize)
        .checked_mul(output_height as usize)
        .and_then(|n| n.checked_mul(4))
        .ok_or(ThumbnailError::Limit("output bytes"))?;
    let mut rgba8 = vec![0; output_len];
    let linear: [f64; 256] = core::array::from_fn(|value| {
        let v = value as f64 / 255.0;
        if let Some(gamma) = gamma {
            v.powf(1.0 / gamma)
        } else if v <= 0.04045 {
            v / 12.92
        } else {
            ((v + 0.055) / 1.055).powf(2.4)
        }
    });
    for y in 0..output_height {
        checkpoint(cancelled, deadline)?;
        let y0 = f64::from(y) * f64::from(height) / f64::from(output_height);
        let y1 = f64::from(y + 1) * f64::from(height) / f64::from(output_height);
        for x in 0..output_width {
            let x0 = f64::from(x) * f64::from(width) / f64::from(output_width);
            let x1 = f64::from(x + 1) * f64::from(width) / f64::from(output_width);
            let mut color = [0.0; 3];
            let mut alpha = 0.0;
            for sy in (y0.floor() as u32)..(y1.ceil() as u32).min(height) {
                checkpoint(cancelled, deadline)?;
                let wy = (y1.min(f64::from(sy + 1)) - y0.max(f64::from(sy))).max(0.0);
                for sx in (x0.floor() as u32)..(x1.ceil() as u32).min(width) {
                    if sx % 4096 == 0 {
                        checkpoint(cancelled, deadline)?;
                    }
                    let weight = wy * (x1.min(f64::from(sx + 1)) - x0.max(f64::from(sx))).max(0.0);
                    let offset = (sy as usize * width as usize + sx as usize) * channels;
                    let p = &source[offset..offset + channels];
                    let (rgb, a) = match channels {
                        1 => ([p[0]; 3], 255),
                        2 => ([p[0]; 3], p[1]),
                        3 => ([p[0], p[1], p[2]], 255),
                        _ => ([p[0], p[1], p[2]], p[3]),
                    };
                    let aw = f64::from(a) / 255.0 * weight;
                    alpha += aw;
                    for i in 0..3 {
                        color[i] += linear[usize::from(rgb[i])] * aw;
                    }
                }
            }
            let offset = (y as usize * output_width as usize + x as usize) * 4;
            if alpha > 0.0 {
                for i in 0..3 {
                    let value = (color[i] / alpha).clamp(0.0, 1.0);
                    let srgb = if value <= 0.0031308 {
                        value * 12.92
                    } else {
                        1.055 * value.powf(1.0 / 2.4) - 0.055
                    };
                    rgba8[offset + i] = (srgb * 255.0).round().clamp(0.0, 255.0) as u8;
                }
            }
            rgba8[offset + 3] = (alpha / ((x1 - x0) * (y1 - y0)) * 255.0)
                .round()
                .clamp(0.0, 255.0) as u8;
        }
    }
    checkpoint(cancelled, deadline)?;
    Ok(Thumbnail {
        width: output_width,
        height: output_height,
        rgba8,
    })
}
