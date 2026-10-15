//! Engine-neutral camera intent and per-frame view derivation.
//!
//! The crate owns camera mathematics and immutable frame snapshots. It does
//! not own ECS storage, GPU resources, window creation, or physics queries.

#![cfg_attr(not(test), no_std)]
#![forbid(unsafe_code)]

extern crate alloc;

mod error;
mod frustum;
mod gpu;
mod jitter;
mod lens;
mod projection;
mod target;
mod view;

pub use error::{CameraError, CameraErrorCode};
pub use frustum::{extract_frustum, Frustum, HalfSpace};
pub use gpu::{GpuViewRecord, VIEW_SCHEMA_VERSION};
pub use jitter::{JitterSequence, JitterState};
pub use lens::{Exposure, Filmback, Lens};
pub use projection::{
    FovAxis, OrthographicProjection, PerspectiveProjection, Projection, ProjectionError,
};
pub use target::{ClearMode, RenderTarget, Viewport, ViewportRect};
pub use prism_render_architecture::view_family::{ViewHandle, ViewImportance, ViewKind};
pub use view::{
    derive_view, publish_view, CameraIntent, CameraPose, ViewData, ViewHistory, ViewStatus,
};

/// Public types commonly needed by camera authors.
pub mod prelude {
    pub use crate::{
        derive_view, publish_view, CameraError, CameraErrorCode, CameraIntent, CameraPose,
        ClearMode, Exposure, Filmback, Frustum, GpuViewRecord, JitterSequence, JitterState,
        Lens, OrthographicProjection, PerspectiveProjection, Projection, RenderTarget, ViewData,
        ViewHandle, ViewHistory, ViewImportance, ViewKind, Viewport, ViewportRect, ViewStatus,
    };
}
