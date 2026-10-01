//! Shadowing: virtual shadow maps (page/clipmap addressing), ray-traced
//! contact shadows, and PCSS-style penumbra estimation.
//!
//! # Conventions
//! * Virtual shadow maps use a clipmap of virtual pages; only the addressing
//!   and level-selection math lives here (no GPU residency management).
//! * Penumbra width follows the similar-triangles blocker model
//!   `w = (d_receiver - d_blocker) / d_blocker * light_size`.
//! * All helpers are deterministic CPU golden pure functions (no RNG/IO/GPU/unsafe).

pub mod contact;
pub mod penumbra;
pub mod vsm;

pub use contact::{
    analytic_sphere_occlusion, clamp_self_shadow_bias, project_march_end, screen_edge_fade,
    screen_space_contact, sdf_soft_contact, softness_to_hardness, ContactShadowParams,
};
pub use penumbra::{
    blocker_search, centered_jitter, pcf_visibility, pcss_visibility, penumbra_filter_radius,
    penumbra_width, BlockerResult, PcssParams,
};
pub use vsm::{VsmClipmap, VsmPageTable, VSM_PAGE_UNMAPPED};
