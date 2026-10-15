use prism_render_architecture::view_family::ViewHandle;

/// Stable error categories returned by camera validation and derivation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CameraErrorCode {
    InvalidProjection,
    InvalidViewport,
    InvalidTargetSize,
    NonFinitePose,
    NonFiniteDerivedView,
    SingularView,
    StaleViewHandle,
}

/// A compact, allocation-free camera error.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CameraError {
    pub code: CameraErrorCode,
    pub field: &'static str,
    pub value: f32,
    pub handle: Option<ViewHandle>,
}

impl CameraError {
    pub const fn new(code: CameraErrorCode, field: &'static str, value: f32) -> Self {
        Self {
            code,
            field,
            value,
            handle: None,
        }
    }

    pub const fn with_handle(mut self, handle: ViewHandle) -> Self {
        self.handle = Some(handle);
        self
    }
}
