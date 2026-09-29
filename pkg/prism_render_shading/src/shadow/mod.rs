//! Backend-neutral shadow evaluation core: the CPU golden references that
//! compute the per-light `visibility` term consumed by the resolve pass.
//!
//! The module is deliberately split by concern so each piece stays a small,
//! independently testable numerical twin of the future `shadow.wesl`:
//!
//! * [`math`] - column-major matrix / vector helpers for light-space projection.
//! * [`cascade`] - PSSM split scheme and cascade selection for directional CSM.
//! * [`csm`] - stabilized cascade light-matrix construction (frustum fit + snap).
//! * [`atlas`] - shadow-atlas layer budgeting and contiguous-range allocation.
//! * [`bias`] - normal-offset and slope-scaled depth bias (acne / peter-panning).
//! * [`filter`] - the [`ShadowDepthSampler`] abstraction plus PCF and PCSS.
//! * [`directional`] - the directional (cascaded) shadow orchestrator.
//! * [`point`] - the omnidirectional (cube distance map) shadow orchestrator.
//! * [`spot`] - the cone-restricted (single perspective layer) shadow orchestrator.
//! * [`depth_view`] - per-layer draw plan for the shadow-map depth pass.
//! * [`virtual_sm`] - virtual shadow map (UE5-style) demand-paging golden: page
//!   table, clipmap levels, LRU physical pool, request generation and caster
//!   invalidation, plus a one-frame orchestrator.
//!
//! The GPU shadow-map render passes (depth rasterization into the atlas,
//! directional CSM stabilization / texel snapping, cube-face rendering, and
//! atlas allocation) and the resolve-side wiring that feeds these matrices and
//! samplers are built in later slices and require on-device validation for
//! numerical parity.

pub mod atlas;
pub mod bias;
pub mod cascade;
pub mod csm;
pub mod depth_view;
pub mod directional;
pub mod filter;
pub mod math;
pub mod point;
pub mod spot;
pub mod virtual_sm;

pub use atlas::{
    allocate_shadow_atlas, AtlasAllocation, AtlasConfig, AtlasSlot, ShadowKind, ShadowRequest,
    POINT_LAYER_COUNT,
};
pub use bias::{apply_normal_offset, slope_scaled_depth_bias};
pub use cascade::{
    cascade_blend_weight, compute_cascade_splits, select_cascade, CascadeSplits, MAX_CASCADE_COUNT,
};
pub use csm::{compute_cascade_matrices, CascadeMatrix};
pub use depth_view::{
    plan_shadow_depth_draws, ShadowDepthDraw, ShadowDepthMode, ShadowDepthView, ShadowViewGeometry,
};
pub use directional::{
    evaluate_directional_shadow, DirectionalShadowConfig, DirectionalShadowInput, ShadowFilter,
};
pub use filter::{
    blocker_search, pcf_visibility, pcss_visibility, BlockerSearch, PcssConfig, ShadowDepthSampler,
};
pub use math::{invert, transform_direction, transform_point, Mat4};
pub use point::{
    cube_face_and_uv, cube_face_view_projections, evaluate_point_shadow, PointShadowConfig,
    PointShadowInput,
};
pub use spot::{evaluate_spot_shadow, spot_view_projection, SpotShadowConfig, SpotShadowInput};
pub use virtual_sm::{
    camera_move_invalidates_pages, decode_window_slot, filter_page_radius, generate_page_requests,
    generate_receiver, invalidate_casters, reconstruct_world_position, slot_to_page_key,
    window_slot, window_slot_count, window_slots_per_level, Allocation, AllocatorStats, BudgetStats,
    CasterMovement, ClipmapConfig, ClipmapLevel, FrameInput, FrameResult, Invalidation,
    PageRequestSet, PageTableStats, PhysicalPageAllocator, Receiver, ReceiverProjection, Residency,
    ShadowPageKey, VirtualPageTable, VirtualShadowMap, VirtualShadowSettings,
};
