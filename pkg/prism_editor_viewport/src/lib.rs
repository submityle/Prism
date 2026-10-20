//! Deterministic viewport, picking, and gizmo interaction contracts.
//!
//! The crate also includes an executable wgpu ID-buffer backend. It owns only
//! target/readback resources and leaves object-ID drawing to the renderer.

#![forbid(unsafe_code)]

extern crate alloc;

mod gpu_picking;
mod render_targets;
mod scene;
#[cfg(all(test, feature = "gpu-tests"))]
mod test_support;

pub use gpu_picking::{
    EncodedPickId, GpuPickingBackend, GpuPickingConfig, GpuPickingError, GpuPickingFrame,
    InFlightPickReadback, PendingPickReadback, ID_BUFFER_BYTES_PER_TEXEL, ID_BUFFER_FORMAT,
    READBACK_ROW_ALIGNMENT,
};
pub use render_targets::{
    RenderTargetError, RenderTargetHandle, RenderTargetRegistry, RenderTargetState,
    ViewportRenderTarget,
};
pub use scene::{
    EntityId, MeshId, SceneIdEncoder, SceneIdPass, SceneIdPassCommit, SceneIdPassDelta,
    SceneIdPassEntry, SceneIdPassError, SceneIdPassItem, SceneIdPassOperation, SceneIdPassSink,
    SceneObjectId, SubobjectId,
};

use alloc::{collections::BTreeMap, string::String, vec::Vec};
use core::fmt;

/// Monotonic identity for an editor viewport.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub struct ViewportId(u64);

impl ViewportId {
    /// Creates an identity from a persisted value.
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

/// Camera projection used by a viewport.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Projection {
    /// Perspective projection with a vertical field of view in millidegrees.
    Perspective {
        /// Vertical field of view in millidegrees.
        fov_millidegrees: u32,
    },
    /// Orthographic projection with a scale in milli-units.
    Orthographic {
        /// Visible height scale in milli-units.
        scale_milliunits: u32,
    },
}

impl Default for Projection {
    fn default() -> Self {
        Self::Perspective {
            fov_millidegrees: 60_000,
        }
    }
}

/// Integer camera position and orientation representation.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CameraState {
    /// Position in world milli-units.
    pub position_milliunits: [i64; 3],
    /// Yaw, pitch, and roll in millidegrees.
    pub rotation_millidegrees: [i32; 3],
    /// Projection parameters.
    pub projection: Projection,
}

/// Stable render-side options exposed by the viewport model.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ViewportSettings {
    /// Whether the grid overlay is requested.
    pub grid: bool,
    /// Whether the selection outline is requested.
    pub selection_outline: bool,
    /// Integer render scale in percent.
    pub render_scale_percent: u16,
}

impl Default for ViewportSettings {
    fn default() -> Self {
        Self {
            grid: true,
            selection_outline: true,
            render_scale_percent: 100,
        }
    }
}

/// One materialized viewport and its editor-visible state.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Viewport {
    id: ViewportId,
    title: String,
    camera: CameraState,
    settings: ViewportSettings,
    generation: u64,
}

impl Viewport {
    /// Returns the stable viewport identity.
    #[must_use]
    pub const fn id(&self) -> ViewportId {
        self.id
    }

    /// Returns the user-facing title.
    #[must_use]
    pub fn title(&self) -> &str {
        &self.title
    }

    /// Returns the camera state.
    #[must_use]
    pub const fn camera(&self) -> CameraState {
        self.camera
    }

    /// Returns the render settings.
    #[must_use]
    pub const fn settings(&self) -> ViewportSettings {
        self.settings
    }

    /// Returns the viewport/camera generation used to reject old async results.
    #[must_use]
    pub const fn generation(&self) -> u64 {
        self.generation
    }
}

/// Failure from viewport lifecycle or state updates.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ViewportError {
    /// The requested viewport does not exist.
    Unknown(ViewportId),
    /// The monotonic identity space is exhausted.
    IdExhausted,
    /// A title cannot be empty or whitespace-only.
    EmptyTitle,
    /// A render scale is outside the supported range.
    InvalidRenderScale(u16),
    /// Too many asynchronous pointer picks are waiting for renderer results.
    PendingPickCapacityExceeded,
    /// The target does not belong to this viewport or its current generation.
    InvalidRenderTarget,
    /// Requested pixels are empty or outside the target extent.
    InvalidPickRegion,
    /// A viewport generation cannot advance without wrapping.
    GenerationExhausted,
}

impl fmt::Display for ViewportError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unknown(id) => write!(formatter, "unknown viewport {}", id.raw()),
            Self::IdExhausted => formatter.write_str("viewport ID space exhausted"),
            Self::EmptyTitle => formatter.write_str("viewport title must not be empty"),
            Self::InvalidRenderScale(value) => {
                write!(formatter, "invalid viewport render scale: {value}%")
            }
            Self::PendingPickCapacityExceeded => {
                formatter.write_str("pending viewport pick capacity exceeded")
            }
            Self::InvalidRenderTarget => {
                formatter.write_str("pick target does not match the viewport")
            }
            Self::InvalidPickRegion => formatter.write_str("pick region is outside the target"),
            Self::GenerationExhausted => formatter.write_str("viewport generation exhausted"),
        }
    }
}

impl core::error::Error for ViewportError {}

/// Deterministic registry of open viewports.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ViewportRegistry {
    viewports: BTreeMap<ViewportId, Viewport>,
    next_id: u64,
}

impl Default for ViewportRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ViewportRegistry {
    /// Creates an empty registry whose first identity is one.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            viewports: BTreeMap::new(),
            next_id: 1,
        }
    }

    /// Opens a viewport with default camera and overlay settings.
    pub fn open(&mut self, title: impl Into<String>) -> Result<Viewport, ViewportError> {
        let title = title.into();
        if title.trim().is_empty() {
            return Err(ViewportError::EmptyTitle);
        }
        let id = ViewportId(self.next_id);
        self.next_id = self
            .next_id
            .checked_add(1)
            .ok_or(ViewportError::IdExhausted)?;
        let viewport = Viewport {
            id,
            title,
            camera: CameraState::default(),
            settings: ViewportSettings::default(),
            generation: 0,
        };
        self.viewports.insert(id, viewport.clone());
        Ok(viewport)
    }

    /// Closes a viewport and returns its final state.
    pub fn close(&mut self, id: ViewportId) -> Result<Viewport, ViewportError> {
        self.viewports.remove(&id).ok_or(ViewportError::Unknown(id))
    }

    /// Returns a viewport by identity.
    #[must_use]
    pub fn get(&self, id: ViewportId) -> Option<&Viewport> {
        self.viewports.get(&id)
    }

    /// Returns all viewports in stable identity order.
    pub fn iter(&self) -> impl Iterator<Item = &Viewport> {
        self.viewports.values()
    }

    /// Updates the camera and advances the viewport generation.
    pub fn set_camera(&mut self, id: ViewportId, camera: CameraState) -> Result<(), ViewportError> {
        let viewport = self
            .viewports
            .get_mut(&id)
            .ok_or(ViewportError::Unknown(id))?;
        let generation = viewport
            .generation
            .checked_add(1)
            .ok_or(ViewportError::GenerationExhausted)?;
        viewport.camera = camera;
        viewport.generation = generation;
        Ok(())
    }

    /// Updates render settings and advances the viewport generation.
    pub fn set_settings(
        &mut self,
        id: ViewportId,
        settings: ViewportSettings,
    ) -> Result<(), ViewportError> {
        if !(25..=200).contains(&settings.render_scale_percent) {
            return Err(ViewportError::InvalidRenderScale(
                settings.render_scale_percent,
            ));
        }
        let viewport = self
            .viewports
            .get_mut(&id)
            .ok_or(ViewportError::Unknown(id))?;
        let generation = viewport
            .generation
            .checked_add(1)
            .ok_or(ViewportError::GenerationExhausted)?;
        viewport.settings = settings;
        viewport.generation = generation;
        Ok(())
    }
}

/// Pixel coordinate supplied to an asynchronous pick request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PixelPoint {
    /// Horizontal pixel coordinate.
    pub x: u32,
    /// Vertical pixel coordinate.
    pub y: u32,
}

/// Correlation identity for a pending pick request.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub struct PickRequestId(u64);

impl PickRequestId {
    /// Returns the numeric request identity.
    #[must_use]
    pub const fn raw(self) -> u64 {
        self.0
    }
}

/// Renderer-neutral request for an ID-buffer or ray-pick operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PickRequest {
    /// Request correlation identity.
    pub id: PickRequestId,
    /// Viewport that produced the request.
    pub viewport: ViewportId,
    /// Viewport/camera generation captured by the request.
    pub generation: u64,
    /// Independent scene-content generation captured by the request.
    pub scene_generation: u64,
    /// Render-target allocation generation captured by the request.
    pub target_generation: u64,
    /// Stable render-target allocation handle captured by the request.
    pub target_handle: RenderTargetHandle,
    /// Pixel to resolve.
    pub pixel: PixelPoint,
}

/// Result returned by a picking adapter.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PickResult {
    /// Correlation identity copied from the request.
    pub id: PickRequestId,
    /// Viewport that produced the result.
    pub viewport: ViewportId,
    /// Viewport/camera generation used by the renderer.
    pub generation: u64,
    /// Independent scene-content generation used by the renderer.
    pub scene_generation: u64,
    /// Render-target allocation generation used by the renderer.
    pub target_generation: u64,
    /// Stable render-target allocation handle used by the renderer.
    pub target_handle: RenderTargetHandle,
    /// Raw ID value copied from the integer target. `EncodedPickId::none()` is a miss.
    pub encoded: EncodedPickId,
}

/// Why a pick result was not accepted.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PickRejectReason {
    /// The request was not submitted by this coordinator.
    UnknownRequest,
    /// The result refers to a viewport generation that is no longer current.
    StaleGeneration,
    /// Scene content changed after this request was recorded.
    StaleScene,
    /// A nonzero GPU ID is absent or invisible in the current scene.
    UnknownObject,
    /// The result belongs to another viewport.
    WrongViewport,
    /// The result refers to a target allocation that has been resized or closed.
    StaleRenderTarget,
}

/// Accepted or rejected outcome of resolving a pick result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PickOutcome {
    /// The result is current and may update editor selection.
    Accepted {
        /// Full scene object identity, if the current scene still contains the hit.
        object: Option<SceneObjectId>,
    },
    /// The result is safe to discard.
    Rejected(PickRejectReason),
}

/// Bounded correlation state for asynchronous picks.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PickCoordinator {
    next_id: u64,
    pending: BTreeMap<PickRequestId, PickRequest>,
}

const MAX_PENDING_PICK_REQUESTS: usize = 1024;

impl Default for PickCoordinator {
    fn default() -> Self {
        Self::new()
    }
}

impl PickCoordinator {
    /// Creates an empty coordinator.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            next_id: 1,
            pending: BTreeMap::new(),
        }
    }

    /// Submits a request and returns its stable correlation record.
    pub fn submit(
        &mut self,
        viewport: &Viewport,
        target: RenderTargetState,
        scene: &SceneIdPass,
        pixel: PixelPoint,
    ) -> Result<PickRequest, ViewportError> {
        validate_pick_target(viewport, target)?;
        if pixel.x >= target.config.width || pixel.y >= target.config.height {
            return Err(ViewportError::InvalidPickRegion);
        }
        if self.pending.len() >= MAX_PENDING_PICK_REQUESTS {
            return Err(ViewportError::PendingPickCapacityExceeded);
        }
        let id = PickRequestId(self.next_id);
        self.next_id = self
            .next_id
            .checked_add(1)
            .ok_or(ViewportError::IdExhausted)?;
        let request = PickRequest {
            id,
            viewport: viewport.id,
            generation: viewport.generation,
            scene_generation: scene.generation(),
            target_generation: target.target_generation,
            target_handle: target.handle,
            pixel,
        };
        self.pending.insert(id, request);
        Ok(request)
    }

    /// Resolves a renderer result against the current viewport state.
    pub fn resolve(
        &mut self,
        current: &Viewport,
        target: Option<RenderTargetState>,
        scene: &SceneIdPass,
        result: PickResult,
    ) -> PickOutcome {
        let Some(request) = self.pending.remove(&result.id) else {
            return PickOutcome::Rejected(PickRejectReason::UnknownRequest);
        };
        if request.viewport != current.id || result.viewport != current.id {
            return PickOutcome::Rejected(PickRejectReason::WrongViewport);
        }
        if request.generation != current.generation || result.generation != current.generation {
            return PickOutcome::Rejected(PickRejectReason::StaleGeneration);
        }
        if request.scene_generation != scene.generation()
            || result.scene_generation != scene.generation()
        {
            return PickOutcome::Rejected(PickRejectReason::StaleScene);
        }
        let Some(target) = target else {
            return PickOutcome::Rejected(PickRejectReason::StaleRenderTarget);
        };
        if target.viewport != current.id
            || target.viewport_generation != current.generation
            || request.target_generation != target.target_generation
            || result.target_generation != target.target_generation
            || request.target_handle != target.handle
            || result.target_handle != target.handle
        {
            return PickOutcome::Rejected(PickRejectReason::StaleRenderTarget);
        }
        let object = scene.resolve(result.encoded);
        if result.encoded != EncodedPickId::none() && object.is_none() {
            return PickOutcome::Rejected(PickRejectReason::UnknownObject);
        }
        PickOutcome::Accepted { object }
    }

    /// Returns the number of requests awaiting a renderer result.
    #[must_use]
    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }

    /// Drops all pending correlations for a resized or closed viewport.
    pub fn cancel_viewport(&mut self, viewport: ViewportId) -> usize {
        let previous = self.pending.len();
        self.pending
            .retain(|_, request| request.viewport != viewport);
        previous - self.pending.len()
    }
}

fn validate_pick_target(
    viewport: &Viewport,
    target: RenderTargetState,
) -> Result<(), ViewportError> {
    if target.viewport != viewport.id || target.viewport_generation != viewport.generation {
        return Err(ViewportError::InvalidRenderTarget);
    }
    Ok(())
}

/// Failure while constructing a box-selection request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SelectionError {
    /// The rectangle has no covered pixels.
    EmptyRectangle,
}

impl fmt::Display for SelectionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyRectangle => {
                formatter.write_str("box-selection rectangle must be non-empty")
            }
        }
    }
}

impl core::error::Error for SelectionError {}

/// An inclusive-exclusive pixel rectangle used for box selection.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PixelRect {
    /// Left edge in target pixels.
    min_x: u32,
    /// Top edge in target pixels.
    min_y: u32,
    /// Exclusive right edge in target pixels.
    max_x: u32,
    /// Exclusive bottom edge in target pixels.
    max_y: u32,
}

impl PixelRect {
    /// Creates a non-empty rectangle from inclusive minimum and exclusive maximum edges.
    pub fn new(min_x: u32, min_y: u32, max_x: u32, max_y: u32) -> Result<Self, SelectionError> {
        if min_x >= max_x || min_y >= max_y {
            return Err(SelectionError::EmptyRectangle);
        }
        Ok(Self {
            min_x,
            min_y,
            max_x,
            max_y,
        })
    }

    /// Returns the rectangle width.
    #[must_use]
    pub const fn width(self) -> u32 {
        self.max_x - self.min_x
    }

    /// Returns the rectangle height.
    #[must_use]
    pub const fn height(self) -> u32 {
        self.max_y - self.min_y
    }

    /// Returns the inclusive minimum corner in target pixels.
    #[must_use]
    pub const fn min(self) -> PixelPoint {
        PixelPoint {
            x: self.min_x,
            y: self.min_y,
        }
    }

    /// Returns the exclusive maximum corner in target pixels.
    #[must_use]
    pub const fn max(self) -> PixelPoint {
        PixelPoint {
            x: self.max_x,
            y: self.max_y,
        }
    }
}

/// Correlation identity for an asynchronous box-selection request.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub struct SelectionRequestId(u64);

impl SelectionRequestId {
    /// Returns the numeric request identity.
    #[must_use]
    pub const fn raw(self) -> u64 {
        self.0
    }
}

/// Renderer-neutral request for a box-selection readback.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BoxSelectionRequest {
    /// Request correlation identity.
    pub id: SelectionRequestId,
    /// Viewport that produced the request.
    pub viewport: ViewportId,
    /// Viewport/camera generation captured by the request.
    pub generation: u64,
    /// Independent scene-content generation captured by the request.
    pub scene_generation: u64,
    /// Render-target allocation generation captured by the request.
    pub target_generation: u64,
    /// Stable render-target allocation handle captured by the request.
    pub target_handle: RenderTargetHandle,
    /// Pixel rectangle to resolve.
    pub rectangle: PixelRect,
}

/// Result of a box-selection readback before scene identity resolution.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BoxSelectionResult {
    /// Request correlation identity.
    pub id: SelectionRequestId,
    /// Viewport that produced the result.
    pub viewport: ViewportId,
    /// Viewport/camera generation used by the renderer.
    pub generation: u64,
    /// Independent scene-content generation used by the renderer.
    pub scene_generation: u64,
    /// Render-target allocation generation used by the renderer.
    pub target_generation: u64,
    /// Stable render-target allocation handle used by the renderer.
    pub target_handle: RenderTargetHandle,
    /// Encoded, potentially duplicated IDs sampled from the rectangle.
    pub ids: Vec<EncodedPickId>,
}

/// Why a box-selection result was not accepted.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SelectionRejectReason {
    /// The result was not submitted by this coordinator.
    UnknownRequest,
    /// The result refers to an old viewport/camera generation.
    StaleGeneration,
    /// Scene content changed after the request was recorded.
    StaleScene,
    /// A nonzero sampled ID is absent or invisible in the current scene.
    UnknownObject,
    /// The result refers to an old or closed target allocation.
    StaleRenderTarget,
    /// The result belongs to another viewport.
    WrongViewport,
    /// The renderer exceeded the coordinator's result bound.
    TooManyHits,
}

/// Accepted or rejected box-selection result.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SelectionOutcome {
    /// The result is current and contains deduplicated full scene identities.
    Accepted(Vec<SceneObjectId>),
    /// The result is safe to discard.
    Rejected(SelectionRejectReason),
}

/// Bounded correlation state for asynchronous box selection.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BoxSelectionCoordinator {
    next_id: u64,
    max_hits: usize,
    pending: BTreeMap<SelectionRequestId, BoxSelectionRequest>,
}

/// Identifier for the editor subsystem currently receiving a viewport's input.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub struct InputOwnerId(u64);

impl InputOwnerId {
    /// Creates an engine-neutral input owner identity.
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

/// Input-focus state for one viewport.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ViewportInputFocus {
    viewport: ViewportId,
    focused: Option<InputOwnerId>,
    captured: Option<InputOwnerId>,
}

impl ViewportInputFocus {
    /// Creates unfocused input state for one viewport.
    #[must_use]
    pub const fn new(viewport: ViewportId) -> Self {
        Self {
            viewport,
            focused: None,
            captured: None,
        }
    }

    /// Returns the viewport controlled by this state.
    #[must_use]
    pub const fn viewport(&self) -> ViewportId {
        self.viewport
    }

    /// Returns the focused owner, if any.
    #[must_use]
    pub const fn focused(&self) -> Option<InputOwnerId> {
        self.focused
    }

    /// Returns the pointer-capturing owner, if any.
    #[must_use]
    pub const fn captured(&self) -> Option<InputOwnerId> {
        self.captured
    }

    /// Gives keyboard or pointer focus to an owner and clears stale capture.
    pub fn focus(&mut self, owner: InputOwnerId) {
        self.focused = Some(owner);
        if self.captured != Some(owner) {
            self.captured = None;
        }
    }

    /// Begins pointer capture for the current focused owner.
    pub fn capture(&mut self, owner: InputOwnerId) -> bool {
        if self.focused != Some(owner) {
            return false;
        }
        self.captured = Some(owner);
        true
    }

    /// Releases pointer capture only for its current owner.
    pub fn release(&mut self, owner: InputOwnerId) -> bool {
        if self.captured != Some(owner) {
            return false;
        }
        self.captured = None;
        true
    }

    /// Clears focus and capture when the platform deactivates the viewport.
    pub fn clear(&mut self) {
        self.focused = None;
        self.captured = None;
    }
}

impl BoxSelectionCoordinator {
    /// Creates a coordinator with explicit per-result and pending-request bounds.
    #[must_use]
    pub const fn new(max_hits: usize) -> Self {
        Self {
            next_id: 1,
            max_hits,
            pending: BTreeMap::new(),
        }
    }

    /// Submits a box-selection request.
    pub fn submit(
        &mut self,
        viewport: &Viewport,
        target: RenderTargetState,
        scene: &SceneIdPass,
        rectangle: PixelRect,
    ) -> Result<BoxSelectionRequest, ViewportError> {
        validate_pick_target(viewport, target)?;
        if rectangle.max_x > target.config.width || rectangle.max_y > target.config.height {
            return Err(ViewportError::InvalidPickRegion);
        }
        if self.pending.len() >= 64 {
            return Err(ViewportError::PendingPickCapacityExceeded);
        }
        let id = SelectionRequestId(self.next_id);
        self.next_id = self
            .next_id
            .checked_add(1)
            .ok_or(ViewportError::IdExhausted)?;
        let request = BoxSelectionRequest {
            id,
            viewport: viewport.id,
            generation: viewport.generation,
            scene_generation: scene.generation(),
            target_generation: target.target_generation,
            target_handle: target.handle,
            rectangle,
        };
        self.pending.insert(id, request);
        Ok(request)
    }

    /// Resolves a result against the current viewport and target generations.
    pub fn resolve(
        &mut self,
        current: &Viewport,
        target: Option<RenderTargetState>,
        scene: &SceneIdPass,
        result: BoxSelectionResult,
    ) -> SelectionOutcome {
        let Some(request) = self.pending.remove(&result.id) else {
            return SelectionOutcome::Rejected(SelectionRejectReason::UnknownRequest);
        };
        if request.viewport != current.id || result.viewport != current.id {
            return SelectionOutcome::Rejected(SelectionRejectReason::WrongViewport);
        }
        if request.generation != current.generation || result.generation != current.generation {
            return SelectionOutcome::Rejected(SelectionRejectReason::StaleGeneration);
        }
        if request.scene_generation != scene.generation()
            || result.scene_generation != scene.generation()
        {
            return SelectionOutcome::Rejected(SelectionRejectReason::StaleScene);
        }
        let Some(target) = target else {
            return SelectionOutcome::Rejected(SelectionRejectReason::StaleRenderTarget);
        };
        if target.viewport != current.id
            || target.viewport_generation != current.generation
            || request.target_generation != target.target_generation
            || result.target_generation != target.target_generation
            || request.target_handle != target.handle
            || result.target_handle != target.handle
        {
            return SelectionOutcome::Rejected(SelectionRejectReason::StaleRenderTarget);
        }
        if result.ids.len() > self.max_hits {
            return SelectionOutcome::Rejected(SelectionRejectReason::TooManyHits);
        }
        if result
            .ids
            .iter()
            .any(|id| *id != EncodedPickId::none() && scene.resolve(*id).is_none())
        {
            return SelectionOutcome::Rejected(SelectionRejectReason::UnknownObject);
        }
        SelectionOutcome::Accepted(scene.resolve_selection(result.ids))
    }

    /// Returns the number of requests awaiting renderer results.
    #[must_use]
    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }

    /// Drops all pending correlations for a resized or closed viewport.
    pub fn cancel_viewport(&mut self, viewport: ViewportId) -> usize {
        let previous = self.pending.len();
        self.pending
            .retain(|_, request| request.viewport != viewport);
        previous - self.pending.len()
    }
}

/// Transform operation controlled by a gizmo.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GizmoMode {
    /// Translate the selected objects.
    Translate,
    /// Rotate the selected objects.
    Rotate,
    /// Scale the selected objects.
    Scale,
}

/// Deterministic snapping options for a gizmo interaction.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SnapSettings {
    /// Translation increment in milli-units; zero disables translation snap.
    pub translation_milliunits: u32,
    /// Rotation increment in millidegrees; zero disables rotation snap.
    pub rotation_millidegrees: u32,
    /// Scale increment in milli-units; zero disables scale snap.
    pub scale_milliunits: u32,
}

impl Default for SnapSettings {
    fn default() -> Self {
        Self {
            translation_milliunits: 100,
            rotation_millidegrees: 15_000,
            scale_milliunits: 100,
        }
    }
}

/// Cumulative transform delta emitted by a gizmo drag.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TransformDelta {
    /// Translation delta in world milli-units.
    pub translation_milliunits: [i64; 3],
    /// Rotation delta in millidegrees.
    pub rotation_millidegrees: [i32; 3],
    /// Multiplicative scale delta in milli-units, where 1000 is identity.
    pub scale_milliunits: [i64; 3],
}

impl TransformDelta {
    fn snap(self, mode: GizmoMode, settings: SnapSettings) -> Self {
        let mut delta = self;
        match mode {
            GizmoMode::Translate => {
                for value in &mut delta.translation_milliunits {
                    *value = snap_i64(*value, settings.translation_milliunits);
                }
            }
            GizmoMode::Rotate => {
                for value in &mut delta.rotation_millidegrees {
                    *value = snap_i32(*value, settings.rotation_millidegrees);
                }
            }
            GizmoMode::Scale => {
                for value in &mut delta.scale_milliunits {
                    *value = snap_i64(*value, settings.scale_milliunits);
                }
            }
        }
        delta
    }
}

fn snap_i64(value: i64, step: u32) -> i64 {
    let step = i64::from(step);
    if step == 0 {
        return value;
    }
    let quotient = value.div_euclid(step);
    let remainder = value.rem_euclid(step);
    if remainder.saturating_mul(2) >= step {
        (quotient + 1).saturating_mul(step)
    } else {
        quotient.saturating_mul(step)
    }
}

fn snap_i32(value: i32, step: u32) -> i32 {
    snap_i64(i64::from(value), step).clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
}

/// A command-oriented gizmo interaction. Intermediate updates are coalesced.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GizmoSession {
    mode: GizmoMode,
    snap: SnapSettings,
    accumulated: TransformDelta,
    active: bool,
}

/// Declarative debug primitive emitted by a gizmo adapter.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GizmoDrawCommand {
    /// Draw one oriented interaction axis from an application-defined origin.
    Axis {
        /// Gizmo transform mode represented by the axis.
        mode: GizmoMode,
        /// Axis index in application-defined local space.
        axis: u8,
        /// Whether the axis is actively manipulated.
        active: bool,
    },
}

impl GizmoSession {
    /// Starts a new drag.
    #[must_use]
    pub fn begin(mode: GizmoMode, snap: SnapSettings) -> Self {
        Self {
            mode,
            snap,
            accumulated: TransformDelta {
                scale_milliunits: [1000; 3],
                ..TransformDelta::default()
            },
            active: true,
        }
    }

    /// Adds an input delta and returns the snapped cumulative delta.
    pub fn update(&mut self, delta: TransformDelta) -> Option<TransformDelta> {
        if !self.active {
            return None;
        }
        self.accumulated.translation_milliunits = add_i64(
            self.accumulated.translation_milliunits,
            delta.translation_milliunits,
        );
        self.accumulated.rotation_millidegrees = add_i32(
            self.accumulated.rotation_millidegrees,
            delta.rotation_millidegrees,
        );
        self.accumulated.scale_milliunits = add_i64(
            self.accumulated.scale_milliunits,
            delta
                .scale_milliunits
                .map(|value| value.saturating_sub(1000)),
        );
        Some(self.accumulated.snap(self.mode, self.snap))
    }

    /// Finishes the drag and returns the one undoable cumulative delta.
    pub fn finish(&mut self) -> Option<TransformDelta> {
        if !self.active {
            return None;
        }
        self.active = false;
        Some(self.accumulated.snap(self.mode, self.snap))
    }

    /// Cancels the drag without producing an edit.
    pub fn cancel(&mut self) -> bool {
        if !self.active {
            return false;
        }
        self.active = false;
        true
    }

    /// Returns whether the drag is still active.
    #[must_use]
    pub const fn is_active(&self) -> bool {
        self.active
    }

    /// Emits renderer-neutral gizmo primitives for the current interaction state.
    #[must_use]
    pub fn draw_commands(&self, active_axis: Option<u8>) -> [GizmoDrawCommand; 3] {
        [0, 1, 2].map(|axis| GizmoDrawCommand::Axis {
            mode: self.mode,
            axis,
            active: self.active && active_axis == Some(axis),
        })
    }
}

fn add_i64(left: [i64; 3], right: [i64; 3]) -> [i64; 3] {
    [
        left[0].saturating_add(right[0]),
        left[1].saturating_add(right[1]),
        left[2].saturating_add(right[2]),
    ]
}

fn add_i32(left: [i32; 3], right: [i32; 3]) -> [i32; 3] {
    [
        left[0].saturating_add(right[0]),
        left[1].saturating_add(right[1]),
        left[2].saturating_add(right[2]),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn target(viewport: ViewportId, generation: u64, handle: u64) -> RenderTargetState {
        RenderTargetState {
            viewport,
            handle: RenderTargetHandle::from_raw(handle),
            viewport_generation: 0,
            target_generation: generation,
            config: GpuPickingConfig::new(64, 64),
        }
    }

    fn pick_result(request: PickRequest, encoded: EncodedPickId) -> PickResult {
        PickResult {
            id: request.id,
            viewport: request.viewport,
            generation: request.generation,
            scene_generation: request.scene_generation,
            target_generation: request.target_generation,
            target_handle: request.target_handle,
            encoded,
        }
    }

    fn box_result(request: BoxSelectionRequest, ids: Vec<EncodedPickId>) -> BoxSelectionResult {
        BoxSelectionResult {
            id: request.id,
            viewport: request.viewport,
            generation: request.generation,
            scene_generation: request.scene_generation,
            target_generation: request.target_generation,
            target_handle: request.target_handle,
            ids,
        }
    }

    #[test]
    fn scene_only_change_invalidates_pick_and_box_without_camera_change() {
        let mut viewports = ViewportRegistry::new();
        let viewport = viewports.open("Scene").expect("viewport");
        let target = target(viewport.id(), 0, 1);
        let mut scene = SceneIdPass::new();
        let mut picks = PickCoordinator::new();
        let pick = picks
            .submit(&viewport, target, &scene, PixelPoint { x: 0, y: 0 })
            .expect("pick");
        let mut boxes = BoxSelectionCoordinator::new(4);
        let rectangle = PixelRect::new(0, 0, 2, 2).expect("rectangle");
        let selection = boxes
            .submit(&viewport, target, &scene, rectangle)
            .expect("box");
        scene
            .submit([SceneIdPassItem::new(
                SceneObjectId::default(),
                1,
                2,
                true,
                0,
            )])
            .expect("scene change");
        assert_eq!(viewport.generation(), 0);
        assert_eq!(
            picks.resolve(
                &viewport,
                Some(target),
                &scene,
                pick_result(pick, EncodedPickId::none())
            ),
            PickOutcome::Rejected(PickRejectReason::StaleScene)
        );
        assert_eq!(
            boxes.resolve(
                &viewport,
                Some(target),
                &scene,
                box_result(selection, vec![])
            ),
            SelectionOutcome::Rejected(SelectionRejectReason::StaleScene)
        );
    }

    #[test]
    fn results_are_correlated_out_of_order_and_consumed_once() {
        let mut viewports = ViewportRegistry::new();
        let viewport = viewports.open("Scene").expect("viewport");
        let target = target(viewport.id(), 0, 1);
        let scene = SceneIdPass::new();
        let mut picks = PickCoordinator::new();
        let first = picks
            .submit(&viewport, target, &scene, PixelPoint { x: 0, y: 0 })
            .expect("first");
        let second = picks
            .submit(&viewport, target, &scene, PixelPoint { x: 1, y: 1 })
            .expect("second");
        for request in [second, first] {
            let result = pick_result(request, EncodedPickId::none());
            assert_eq!(
                picks.resolve(&viewport, Some(target), &scene, result),
                PickOutcome::Accepted { object: None }
            );
            assert_eq!(
                picks.resolve(&viewport, Some(target), &scene, result),
                PickOutcome::Rejected(PickRejectReason::UnknownRequest)
            );
        }
        let request = picks
            .submit(&viewport, target, &scene, PixelPoint { x: 0, y: 0 })
            .expect("unknown object");
        assert_eq!(
            picks.resolve(
                &viewport,
                Some(target),
                &scene,
                pick_result(request, EncodedPickId::from_channels([999, 0, 0, 0]))
            ),
            PickOutcome::Rejected(PickRejectReason::UnknownObject)
        );
    }

    #[test]
    fn reopened_target_rejects_pick_and_box_with_matching_generations() {
        let mut viewports = ViewportRegistry::new();
        let viewport = viewports.open("Scene").expect("viewport");
        let original = target(viewport.id(), 0, 1);
        let reopened = target(viewport.id(), 0, 2);
        let scene = SceneIdPass::new();
        let mut picks = PickCoordinator::new();
        let request = picks
            .submit(&viewport, original, &scene, PixelPoint { x: 0, y: 0 })
            .expect("pick");
        assert_eq!(
            picks.resolve(
                &viewport,
                Some(reopened),
                &scene,
                pick_result(request, EncodedPickId::none())
            ),
            PickOutcome::Rejected(PickRejectReason::StaleRenderTarget)
        );
        let mut boxes = BoxSelectionCoordinator::new(4);
        let request = boxes
            .submit(
                &viewport,
                original,
                &scene,
                PixelRect::new(0, 0, 1, 1).expect("rectangle"),
            )
            .expect("box");
        assert_eq!(
            boxes.resolve(
                &viewport,
                Some(reopened),
                &scene,
                box_result(request, vec![])
            ),
            SelectionOutcome::Rejected(SelectionRejectReason::StaleRenderTarget)
        );
    }

    #[test]
    fn submission_bounds_and_cancellation_are_per_viewport() {
        let mut viewports = ViewportRegistry::new();
        let viewport = viewports.open("Scene").expect("viewport");
        let other = viewports.open("Other").expect("other");
        let target = target(viewport.id(), 0, 1);
        let scene = SceneIdPass::new();
        let mut picks = PickCoordinator::new();
        let pixel = PixelPoint { x: 0, y: 0 };
        assert_eq!(
            picks.submit(&other, target, &scene, pixel),
            Err(ViewportError::InvalidRenderTarget)
        );
        assert_eq!(
            picks.submit(&viewport, target, &scene, PixelPoint { x: 64, y: 0 }),
            Err(ViewportError::InvalidPickRegion)
        );
        for _ in 0..MAX_PENDING_PICK_REQUESTS {
            picks
                .submit(&viewport, target, &scene, pixel)
                .expect("within bound");
        }
        assert_eq!(
            picks.submit(&viewport, target, &scene, pixel),
            Err(ViewportError::PendingPickCapacityExceeded)
        );
        assert_eq!(picks.cancel_viewport(other.id()), 0);
        assert_eq!(
            picks.cancel_viewport(viewport.id()),
            MAX_PENDING_PICK_REQUESTS
        );
        picks
            .submit(&viewport, target, &scene, pixel)
            .expect("capacity recovered");

        let mut boxes = BoxSelectionCoordinator::new(4);
        assert_eq!(
            boxes.submit(
                &viewport,
                target,
                &scene,
                PixelRect::new(0, 0, 65, 1).expect("rectangle")
            ),
            Err(ViewportError::InvalidPickRegion)
        );
        let rectangle = PixelRect::new(0, 0, 1, 1).expect("rectangle");
        for _ in 0..64 {
            boxes
                .submit(&viewport, target, &scene, rectangle)
                .expect("within bound");
        }
        assert_eq!(
            boxes.submit(&viewport, target, &scene, rectangle),
            Err(ViewportError::PendingPickCapacityExceeded)
        );
        assert_eq!(boxes.cancel_viewport(other.id()), 0);
        assert_eq!(boxes.cancel_viewport(viewport.id()), 64);
        assert_eq!(boxes.pending_len(), 0);
    }

    #[test]
    fn viewport_ids_are_stable_and_generation_changes_are_monotonic() {
        let mut registry = ViewportRegistry::new();
        let first = registry.open("Scene").expect("first");
        let second = registry.open("Game").expect("second");
        assert_eq!(first.id().raw(), 1);
        assert_eq!(second.id().raw(), 2);
        registry
            .set_camera(first.id(), CameraState::default())
            .expect("camera");
        assert_eq!(registry.get(first.id()).expect("viewport").generation(), 1);
        registry.close(first.id()).expect("close");
        let reopened = registry.open("Scene").expect("reopen");
        assert_eq!(reopened.id().raw(), 3);
    }

    #[test]
    fn pick_results_reject_unknown_viewport_and_stale_generation() {
        let mut registry = ViewportRegistry::new();
        let viewport = registry.open("Scene").expect("viewport");
        let mut coordinator = PickCoordinator::new();
        let target = target(viewport.id(), 0, 1);
        let scene = SceneIdPass::new();
        let request = coordinator
            .submit(&viewport, target, &scene, PixelPoint { x: 10, y: 20 })
            .expect("submit");
        registry
            .set_camera(viewport.id(), CameraState::default())
            .expect("camera");
        let current = registry.get(viewport.id()).expect("current");
        assert_eq!(
            coordinator.resolve(
                current,
                Some(target),
                &scene,
                PickResult {
                    id: request.id,
                    viewport: current.id(),
                    generation: request.generation,
                    scene_generation: request.scene_generation,
                    target_generation: request.target_generation,
                    target_handle: request.target_handle,
                    encoded: EncodedPickId::none(),
                },
            ),
            PickOutcome::Rejected(PickRejectReason::StaleGeneration)
        );
        assert_eq!(coordinator.pending_len(), 0);
    }

    #[test]
    fn box_selection_is_correlated_and_bounded() {
        let mut registry = ViewportRegistry::new();
        let viewport = registry.open("Scene").expect("viewport");
        let rectangle = PixelRect::new(1, 2, 4, 6).expect("rectangle");
        assert_eq!(rectangle.width(), 3);
        assert_eq!(rectangle.height(), 4);
        let mut coordinator = BoxSelectionCoordinator::new(1);
        let target = target(viewport.id(), 4, 1);
        let scene = SceneIdPass::new();
        let request = coordinator
            .submit(&viewport, target, &scene, rectangle)
            .expect("submit");
        let result = BoxSelectionResult {
            id: request.id,
            viewport: request.viewport,
            generation: request.generation,
            scene_generation: request.scene_generation,
            target_generation: request.target_generation,
            target_handle: request.target_handle,
            ids: vec![EncodedPickId::none(), EncodedPickId::none()],
        };
        assert_eq!(
            coordinator.resolve(&viewport, Some(target), &scene, result),
            SelectionOutcome::Rejected(SelectionRejectReason::TooManyHits)
        );
        assert_eq!(coordinator.pending_len(), 0);
        assert_eq!(
            PixelRect::new(2, 2, 2, 3),
            Err(SelectionError::EmptyRectangle)
        );
    }

    #[test]
    fn pick_result_rejects_resized_target_generation() {
        let mut registry = ViewportRegistry::new();
        let viewport = registry.open("Scene").expect("viewport");
        let mut coordinator = PickCoordinator::new();
        let initial_target = target(viewport.id(), 5, 1);
        let scene = SceneIdPass::new();
        let request = coordinator
            .submit(&viewport, initial_target, &scene, PixelPoint { x: 0, y: 0 })
            .expect("submit");
        assert_eq!(
            coordinator.resolve(
                &viewport,
                Some(target(viewport.id(), 6, 1)),
                &scene,
                PickResult {
                    id: request.id,
                    viewport: request.viewport,
                    generation: request.generation,
                    scene_generation: request.scene_generation,
                    target_generation: request.target_generation,
                    target_handle: request.target_handle,
                    encoded: EncodedPickId::none(),
                },
            ),
            PickOutcome::Rejected(PickRejectReason::StaleRenderTarget)
        );
    }

    #[test]
    fn closed_target_rejects_pending_pick_even_when_generations_match() {
        let mut registry = ViewportRegistry::new();
        let viewport = registry.open("Scene").expect("viewport");
        let target = target(viewport.id(), 0, 9);
        let scene = SceneIdPass::new();
        let mut coordinator = PickCoordinator::new();
        let request = coordinator
            .submit(&viewport, target, &scene, PixelPoint { x: 1, y: 1 })
            .expect("submit");
        assert_eq!(
            coordinator.resolve(
                &viewport,
                None,
                &scene,
                PickResult {
                    id: request.id,
                    viewport: request.viewport,
                    generation: request.generation,
                    scene_generation: request.scene_generation,
                    target_generation: request.target_generation,
                    target_handle: request.target_handle,
                    encoded: EncodedPickId::none(),
                },
            ),
            PickOutcome::Rejected(PickRejectReason::StaleRenderTarget)
        );
    }

    #[test]
    fn current_pick_and_box_results_resolve_full_scene_objects() {
        let mut registry = ViewportRegistry::new();
        let viewport = registry.open("Scene").expect("viewport");
        let target = target(viewport.id(), 0, 1);
        let object = SceneObjectId::new(
            EntityId::from_raw(9),
            MeshId::from_raw(4),
            SubobjectId::from_raw(2),
        );
        let mut scene = SceneIdPass::new();
        scene
            .submit([SceneIdPassItem::new(object, 10, 20, true, 30)])
            .expect("scene");
        let encoded = scene.entries().next().expect("entry").encoded_id();

        let mut picks = PickCoordinator::new();
        let request = picks
            .submit(&viewport, target, &scene, PixelPoint { x: 0, y: 0 })
            .expect("pick request");
        assert_eq!(
            picks.resolve(
                &viewport,
                Some(target),
                &scene,
                PickResult {
                    id: request.id,
                    viewport: request.viewport,
                    generation: request.generation,
                    scene_generation: request.scene_generation,
                    target_generation: request.target_generation,
                    target_handle: request.target_handle,
                    encoded,
                },
            ),
            PickOutcome::Accepted {
                object: Some(object)
            }
        );

        let mut boxes = BoxSelectionCoordinator::new(4);
        let request = boxes
            .submit(
                &viewport,
                target,
                &scene,
                PixelRect::new(0, 0, 2, 2).expect("rectangle"),
            )
            .expect("box request");
        assert_eq!(
            boxes.resolve(
                &viewport,
                Some(target),
                &scene,
                BoxSelectionResult {
                    id: request.id,
                    viewport: request.viewport,
                    generation: request.generation,
                    scene_generation: request.scene_generation,
                    target_generation: request.target_generation,
                    target_handle: request.target_handle,
                    ids: vec![encoded, EncodedPickId::none(), encoded],
                },
            ),
            SelectionOutcome::Accepted(vec![object])
        );
    }

    #[test]
    fn input_focus_only_captures_the_active_owner() {
        let viewport = ViewportId::from_raw(5);
        let tool = InputOwnerId::from_raw(1);
        let other = InputOwnerId::from_raw(2);
        let mut focus = ViewportInputFocus::new(viewport);
        assert!(!focus.capture(tool));
        focus.focus(tool);
        assert!(focus.capture(tool));
        assert!(!focus.release(other));
        focus.focus(other);
        assert_eq!(focus.focused(), Some(other));
        assert_eq!(focus.captured(), None);
        assert!(focus.capture(other));
        focus.clear();
        assert_eq!(focus.focused(), None);
        assert_eq!(focus.captured(), None);
    }

    #[test]
    fn gizmo_updates_are_snapped_and_finish_once() {
        let mut session = GizmoSession::begin(
            GizmoMode::Translate,
            SnapSettings {
                translation_milliunits: 100,
                ..SnapSettings::default()
            },
        );
        let update = session
            .update(TransformDelta {
                translation_milliunits: [149, -151, 0],
                scale_milliunits: [1000; 3],
                ..TransformDelta::default()
            })
            .expect("update");
        assert_eq!(update.translation_milliunits, [100, -200, 0]);
        assert_eq!(
            session.finish().expect("finish").translation_milliunits,
            [100, -200, 0]
        );
        assert!(session.finish().is_none());
    }

    #[test]
    fn gizmo_draw_commands_expose_mode_and_active_axis_without_renderer_types() {
        let session = GizmoSession::begin(GizmoMode::Rotate, SnapSettings::default());
        assert_eq!(
            session.draw_commands(Some(1)),
            [
                GizmoDrawCommand::Axis {
                    mode: GizmoMode::Rotate,
                    axis: 0,
                    active: false,
                },
                GizmoDrawCommand::Axis {
                    mode: GizmoMode::Rotate,
                    axis: 1,
                    active: true,
                },
                GizmoDrawCommand::Axis {
                    mode: GizmoMode::Rotate,
                    axis: 2,
                    active: false,
                },
            ]
        );
    }
}
