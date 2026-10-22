//! Engine-neutral camera intent and per-frame view derivation.
//!
//! The crate owns camera mathematics and immutable frame snapshots. It does
//! not own ECS storage, GPU resources, window creation, or physics queries.

#![cfg_attr(not(any(test, feature = "std")), no_std)]
#![forbid(unsafe_code)]

extern crate alloc;

mod advanced;
pub mod advanced_runtime;
mod error;
mod frustum;
mod gpu;
mod jitter;
mod lens;
mod pick;
#[cfg(feature = "post")]
mod post;
mod projection;
#[cfg(feature = "stereo")]
mod stereo;
mod target;
mod view;
#[cfg(feature = "view_batch")]
mod view_batch;

#[cfg(feature = "determinism")]
pub use advanced::CameraDeterminismDigest;
#[cfg(feature = "occlusion_fade")]
pub use advanced::OcclusionFadeState;
#[cfg(feature = "shutter")]
pub use advanced::Shutter;
#[cfg(feature = "cutscene")]
pub use advanced::{
    CameraKeyframe, CameraTrack, CameraTrackSample, CameraTrackValidationCode,
    CameraTrackValidationError, CutsceneEndPolicy,
};
#[cfg(feature = "occlusion_fade")]
pub use advanced::{CameraOcclusionIntent, OcclusionSample};
#[cfg(feature = "composition_aid")]
pub use advanced::{
    CompositionAidResult, CompositionGizmo, CompositionGizmoKind, CompositionGuide,
};
#[cfg(feature = "virtual_production")]
pub use advanced::{
    ExternalPoseError, ExternalPoseResult, ExternalPoseSample, ExternalPoseValidator, FoveationHint,
};
#[cfg(feature = "autofocus")]
pub use advanced::{FocusSource, FocusTracker};
#[cfg(feature = "edge_stabilize")]
pub use advanced::{HistoryRejectReason, HistoryStabilizer};
#[cfg(feature = "lookahead_stream")]
pub use advanced::{LookaheadPredictor, StreamingContribution};
#[cfg(feature = "view_lod")]
pub use advanced::{ViewLod, ViewLodCandidate, ViewLodDecision, ViewLodScheduler};
pub use advanced_runtime::{
    AdvancedRuntime, AdvancedRuntimeConfig, AdvancedRuntimeConfigError, AdvancedRuntimeError,
    AdvancedRuntimeInput, AdvancedRuntimeLimits, AdvancedRuntimeOutput, CapacityField, FixedTick,
    RuntimeDegradation,
};
pub use error::{CameraError, CameraErrorCode};
pub use frustum::{extract_frustum, Frustum, HalfSpace};
pub use gpu::{
    GpuViewRecord, GPU_VIEW_OFFSET_CLIP_FROM_WORLD, GPU_VIEW_OFFSET_CLIP_FROM_WORLD_UNJITTERED,
    GPU_VIEW_OFFSET_EXPOSURE, GPU_VIEW_OFFSET_FLAGS, GPU_VIEW_OFFSET_FRUSTUM,
    GPU_VIEW_OFFSET_JITTER, GPU_VIEW_OFFSET_PREV_CLIP_FROM_WORLD, GPU_VIEW_OFFSET_SCHEMA_VERSION,
    GPU_VIEW_OFFSET_VIEWPORT, GPU_VIEW_OFFSET_VIEW_FROM_WORLD, GPU_VIEW_OFFSET_VIEW_INDEX,
    GPU_VIEW_OFFSET_WORLD_FROM_VIEW, GPU_VIEW_OFFSET_WORLD_ORIGIN_HI,
    GPU_VIEW_OFFSET_WORLD_ORIGIN_LO, GPU_VIEW_RECORD_ALIGNMENT, GPU_VIEW_RECORD_SIZE,
    VIEW_SCHEMA_VERSION,
};
pub use jitter::{JitterSequence, JitterState};
pub use lens::Exposure;
#[cfg(feature = "physical_lens")]
pub use lens::{Filmback, Lens};
pub use pick::PickError;
#[cfg(feature = "post")]
pub use post::{
    BloomSettings, ColorGradeSettings, DofSettings, ExposureSettings, MotionBlurSettings,
    PassDomain, PostNode, PostStack, PostStackError, PostStage, Tonemapper,
};
pub use prism_render_architecture::view_family::{ViewHandle, ViewImportance, ViewKind};
pub use projection::{
    FovAxis, ObliquePlane, OrthographicProjection, PerspectiveProjection, Projection,
    ProjectionError, SubView,
};
#[cfg(feature = "stereo")]
pub use stereo::{Eye, LateLatchPose, StereoConfig, StereoPair};
pub use target::{ClearMode, RenderTarget, Viewport, ViewportRect};
pub use view::{
    derive_view, publish_view, CameraIntent, CameraPose, ViewData, ViewHistory, ViewStatus,
};
#[cfg(feature = "view_batch")]
pub use view_batch::{
    CoarseFrustumStatus, ViewBatch, ViewBatchCache, ViewBatchCacheStamp, ViewBatchCacheStatus,
    ViewBatchCapacityField, ViewBatchCollection, ViewBatchCollectionStatus, ViewBatchCollector,
    ViewBatchConfigError, ViewBatchError, ViewBatchKey, ViewBatchLimits, ViewBatchMember,
    MAX_VIEW_BATCH_GROUPS, MAX_VIEW_BATCH_MEMBERS_PER_GROUP, MAX_VIEW_BATCH_VIEWS,
};

/// Public types commonly needed by camera authors.
pub mod prelude {
    pub use crate::{
        derive_view, publish_view, CameraError, CameraErrorCode, CameraIntent, CameraPose,
        ClearMode, Exposure, Frustum, GpuViewRecord, JitterSequence, JitterState, ObliquePlane,
        OrthographicProjection, PerspectiveProjection, PickError, Projection, RenderTarget,
        SubView, ViewData, ViewHandle, ViewHistory, ViewImportance, ViewKind, ViewStatus, Viewport,
        ViewportRect,
    };
    #[cfg(feature = "post")]
    pub use crate::{
        BloomSettings, ColorGradeSettings, DofSettings, ExposureSettings, PostNode, PostStack,
        PostStage,
    };
    #[cfg(feature = "stereo")]
    pub use crate::{Eye, LateLatchPose, StereoConfig, StereoPair};
    #[cfg(feature = "physical_lens")]
    pub use crate::{Filmback, Lens};
}
