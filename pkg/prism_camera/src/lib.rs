//! Engine-neutral camera intent and per-frame view derivation.
//!
//! The crate owns camera mathematics and immutable frame snapshots. It does
//! not own ECS storage, GPU resources, window creation, or physics queries.

#![cfg_attr(not(any(test, feature = "std")), no_std)]
#![forbid(unsafe_code)]

extern crate alloc;

mod advanced;
mod error;
mod frustum;
mod gpu;
mod jitter;
mod lens;
mod pick;
mod post;
mod projection;
mod stereo;
mod target;
mod view;

pub use advanced::{
    CameraDeterminismDigest, CameraKeyframe, CameraOcclusionIntent, CameraTrack, CameraTrackSample,
    CompositionGuide, CutsceneEndPolicy, ExternalPoseError, ExternalPoseSample,
    ExternalPoseValidator, FocusSource, FocusTracker, FoveationHint, HistoryRejectReason,
    HistoryStabilizer, LookaheadPredictor, OcclusionFadeState, OcclusionSample, Shutter,
    ViewBatchKey, ViewLod,
};
pub use error::{CameraError, CameraErrorCode};
pub use frustum::{extract_frustum, Frustum, HalfSpace};
pub use gpu::{GpuViewRecord, VIEW_SCHEMA_VERSION};
pub use jitter::{JitterSequence, JitterState};
pub use lens::{Exposure, Filmback, Lens};
pub use pick::PickError;
pub use post::{
    BloomSettings, ColorGradeSettings, DofSettings, ExposureSettings, MotionBlurSettings,
    PassDomain, PostNode, PostStack, PostStackError, PostStage, Tonemapper,
};
pub use prism_render_architecture::view_family::{ViewHandle, ViewImportance, ViewKind};
pub use projection::{
    FovAxis, ObliquePlane, OrthographicProjection, PerspectiveProjection, Projection,
    ProjectionError, SubView,
};
pub use stereo::{Eye, LateLatchPose, StereoConfig, StereoPair};
pub use target::{ClearMode, RenderTarget, Viewport, ViewportRect};
pub use view::{
    derive_view, publish_view, CameraIntent, CameraPose, ViewData, ViewHistory, ViewStatus,
};

/// Public types commonly needed by camera authors.
pub mod prelude {
    pub use crate::{
        derive_view, publish_view, BloomSettings, CameraError, CameraErrorCode, CameraIntent,
        CameraPose, ClearMode, ColorGradeSettings, DofSettings, Exposure, ExposureSettings,
        Filmback, Frustum, GpuViewRecord, JitterSequence, JitterState, Lens, MotionBlurSettings,
        ObliquePlane, OrthographicProjection, PassDomain, PerspectiveProjection, PickError,
        PostNode, PostStack, PostStage, Projection, RenderTarget, StereoConfig, StereoPair,
        SubView, ViewData, ViewHandle, ViewHistory, ViewImportance, ViewKind, ViewStatus, Viewport,
        ViewportRect,
    };
}
