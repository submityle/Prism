//! Engine-neutral host and presentation seams for the editor application.

use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

/// Stable identity assigned to an editor-owned host window.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub struct WindowId(u64);

impl WindowId {
    /// Creates an identity from persisted data.
    #[must_use]
    pub const fn from_raw(value: u64) -> Self {
        Self(value)
    }

    /// Returns the numeric identity.
    #[must_use]
    pub const fn raw(self) -> u64 {
        self.0
    }
}

/// Stable identity returned by a GPU surface backend.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub struct SurfaceId(u64);

impl SurfaceId {
    /// Creates an identity from backend-owned data.
    #[must_use]
    pub const fn from_raw(value: u64) -> Self {
        Self(value)
    }

    /// Returns the numeric identity.
    #[must_use]
    pub const fn raw(self) -> u64 {
        self.0
    }
}

/// Physical drawable size shared by window and surface backends.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PixelSize {
    /// Width in physical pixels.
    pub width: u32,
    /// Height in physical pixels.
    pub height: u32,
}

impl PixelSize {
    /// Creates a size, rejecting zero-sized drawables.
    pub const fn new(width: u32, height: u32) -> Result<Self, PixelSizeError> {
        if width == 0 || height == 0 {
            return Err(PixelSizeError::Zero);
        }
        Ok(Self { width, height })
    }
}

/// Invalid physical size supplied to a backend boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PixelSizeError {
    /// Either dimension was zero.
    Zero,
}

impl fmt::Display for PixelSizeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Zero => formatter.write_str("window and surface sizes must be non-zero"),
        }
    }
}

impl core::error::Error for PixelSizeError {}

/// Description used when opening an editor window.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WindowDescriptor {
    /// Stable identity selected by the application.
    pub id: WindowId,
    /// User-visible title.
    pub title: String,
    /// Initial drawable size.
    pub size: PixelSize,
    /// Whether the backend should make the window visible immediately.
    pub visible: bool,
}

impl WindowDescriptor {
    /// Creates a visible window descriptor.
    pub fn new(id: WindowId, title: impl Into<String>, size: PixelSize) -> Self {
        Self {
            id,
            title: title.into(),
            size,
            visible: true,
        }
    }
}

/// Events emitted by a window backend in deterministic polling order.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WindowEvent {
    /// The host requested that this window close.
    CloseRequested,
    /// The drawable size changed.
    Resized(PixelSize),
    /// The window gained or lost keyboard focus.
    Focused(bool),
    /// The host suspended the window's presentation resources.
    Suspended,
    /// The host resumed the window's presentation resources.
    Resumed,
    /// The host asked the application to render another frame.
    RedrawRequested,
}

/// A window event associated with a stable window identity.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WindowEventEnvelope {
    /// Target window.
    pub window: WindowId,
    /// Event payload.
    pub event: WindowEvent,
}

/// Broad class of window backend failures.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WindowBackendErrorKind {
    /// The backend cannot satisfy the request on this host.
    Unsupported,
    /// The host rejected a descriptor or operation.
    InvalidRequest,
    /// A temporary host failure may be retried.
    Recoverable,
    /// The backend cannot continue without replacement.
    Fatal,
}

/// Error returned by a [`WindowBackend`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WindowBackendError {
    /// Failure classification used by recovery policy.
    pub kind: WindowBackendErrorKind,
    /// Stable human-readable diagnostic.
    pub message: String,
}

impl WindowBackendError {
    /// Creates a backend error.
    pub fn new(kind: WindowBackendErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }

    /// Whether retrying the same backend is meaningful.
    #[must_use]
    pub const fn is_recoverable(&self) -> bool {
        matches!(self.kind, WindowBackendErrorKind::Recoverable)
    }
}

impl fmt::Display for WindowBackendError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl core::error::Error for WindowBackendError {}

/// Window host contract consumed by [`crate::EditorApplication`].
///
/// The contract contains no OS, event-loop, or rendering types. A platform
/// adapter owns those details and exposes only bounded polling and presentation
/// operations to the application orchestration layer.
pub trait WindowBackend: Send + Sync {
    /// Opens or attaches the requested windows.
    fn start(&mut self, windows: &[WindowDescriptor]) -> Result<(), WindowBackendError>;

    /// Drains at most `limit` queued events, retaining the rest for later polls.
    ///
    /// Implementations must bound queue storage independently and report
    /// overflow explicitly instead of silently discarding input.
    fn poll_events(&mut self, limit: usize)
        -> Result<Vec<WindowEventEnvelope>, WindowBackendError>;

    /// Presents the completed frame for one window.
    fn present(&mut self, window: WindowId) -> Result<(), WindowBackendError>;

    /// Releases all host resources. The application calls this even after a
    /// partial startup, so implementations must make it idempotent.
    fn stop(&mut self) -> Result<(), WindowBackendError>;
}

/// Broad class of GPU surface lifecycle failures.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuSurfaceErrorKind {
    /// The device or context was lost and can be recreated.
    DeviceLost,
    /// The requested surface configuration is unsupported.
    Unsupported,
    /// The request violates the surface contract.
    InvalidRequest,
    /// A temporary presentation failure may be retried.
    Recoverable,
    /// The GPU backend cannot continue.
    Fatal,
}

/// Error returned by a [`GpuSurfaceLifecycle`] implementation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GpuSurfaceError {
    /// Failure classification used by recovery policy.
    pub kind: GpuSurfaceErrorKind,
    /// Stable human-readable diagnostic.
    pub message: String,
}

impl GpuSurfaceError {
    /// Creates a GPU lifecycle error.
    pub fn new(kind: GpuSurfaceErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }

    /// Whether the backend can plausibly be restarted.
    #[must_use]
    pub const fn is_recoverable(&self) -> bool {
        matches!(
            self.kind,
            GpuSurfaceErrorKind::DeviceLost | GpuSurfaceErrorKind::Recoverable
        )
    }
}

impl fmt::Display for GpuSurfaceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl core::error::Error for GpuSurfaceError {}

/// Surface creation/configuration request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SurfaceDescriptor {
    /// Owning window.
    pub window: WindowId,
    /// Initial drawable size.
    pub size: PixelSize,
}

/// Backend-independent surface lifecycle state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SurfaceState {
    /// Surface can receive frames.
    Active,
    /// Presentation is paused and transient resources may be released.
    Suspended,
}

/// GPU surface lifecycle contract consumed by the editor application.
///
/// Device creation, swapchain implementation, command encoding, and resource
/// replay stay in the adapter. This trait only makes ownership and ordering
/// explicit so headless, software, and native GPU hosts share one application
/// state machine.
pub trait GpuSurfaceLifecycle: Send + Sync {
    /// Creates or acquires the device/context used by subsequent surfaces.
    fn start(&mut self) -> Result<(), GpuSurfaceError>;

    /// Creates one surface for a window.
    fn create_surface(
        &mut self,
        descriptor: SurfaceDescriptor,
    ) -> Result<SurfaceId, GpuSurfaceError>;

    /// Resizes one existing surface.
    fn resize_surface(
        &mut self,
        surface: SurfaceId,
        size: PixelSize,
    ) -> Result<(), GpuSurfaceError>;

    /// Releases one surface while retaining the device when possible.
    fn destroy_surface(&mut self, surface: SurfaceId) -> Result<(), GpuSurfaceError>;

    /// Pauses presentation for one surface.
    fn suspend_surface(&mut self, surface: SurfaceId) -> Result<(), GpuSurfaceError>;

    /// Resumes presentation for one surface.
    fn resume_surface(&mut self, surface: SurfaceId) -> Result<(), GpuSurfaceError>;

    /// Presents one completed surface frame.
    fn present_surface(&mut self, surface: SurfaceId) -> Result<(), GpuSurfaceError>;

    /// Releases device/context resources. The call is idempotent from the
    /// orchestration layer's perspective and may follow partial startup.
    fn stop(&mut self) -> Result<(), GpuSurfaceError>;
}
