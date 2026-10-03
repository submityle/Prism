//! Bevy ECS integration for Prism's next-generation renderer.
//!
//! This crate is the connective tissue between Bevy's entity/component world
//! and Prism's engine-neutral, GPU-driven render core. Prism's `pkg/prism_*`
//! crates intentionally never depend on Bevy's render stack, so a dedicated
//! integration crate lives *downstream* of both: it depends on the Bevy ECS
//! crates and on the Prism render crates, and wires them together with plain
//! systems and resources.
//!
//! The first slice implemented here is **GPU-driven visibility**:
//!
//! * [`PrismRenderScene`] is a [`bevy_ecs`] resource holding Prism's stable
//!   `CpuRenderScene` plus the geometry/material registries the culler needs.
//! * [`PrismRenderable`] tags entities that should be mirrored into that scene;
//!   their [`bevy_camera::primitives::Aabb`] + [`bevy_transform::components::GlobalTransform`]
//!   are lowered into Prism instance records through the tested
//!   `prism_render_visibility::bevy_bridge` conversions.
//! * [`PrismCamera`] describes a culling view; each frame the plugin builds a
//!   `GpuViewRecord` from the camera's [`bevy_camera::primitives::Frustum`] and
//!   runs the real `cull_view`, writing the result back to each entity's
//!   [`PrismViewVisibility`].
//!
//! Everything here is CPU-side and deterministic, so it is fully exercised by
//! the integration tests without a GPU device. Later slices (two-phase HZB
//! occlusion, virtual-geometry raster, etc.) extend the same resource.
//!
//! ```no_run
//! use bevy_app::prelude::*;
//! use prism_bevy::PrismVisibilityPlugin;
//!
//! App::new().add_plugins(PrismVisibilityPlugin);
//! ```

extern crate alloc;

mod components;
mod plugin;
mod scene;

pub use components::{PrismCamera, PrismRenderable, PrismViewVisibility};
pub use plugin::{PrismVisibilityPlugin, PrismVisibilitySystems};
pub use scene::PrismRenderScene;

// Re-export the handle/record vocabulary callers need to register backend
// geometry and materials, so downstream crates do not have to name the exact
// Prism crate each type lives in.
pub use prism_render_architecture::gpu_scene::{GeometryHandle, SceneMaterialHandle};
pub use prism_render_material::MaterialRecord;
pub use prism_render_visibility::{GeometryLod, GeometryLodChain, ViewFlags, ViewHandle};
