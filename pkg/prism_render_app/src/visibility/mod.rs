//! Native main-world visibility, independent of a render device or Bevy.
//!
//! Add [`PrismVisibilityPlugin`] to a `prism_app::App`, register geometry and
//! materials on [`PrismRenderScene`], and attach [`PrismRenderable`],
//! [`LocalBounds`] and a native transform to instances. Cameras carry
//! [`PrismCamera`], [`PrismFrustum`] and a native transform. A local
//! `prism_transform::Transform` is propagated; standalone
//! `prism_transform::GlobalTransform` components are also supported.
//!
//! Local/hierarchy writers must finish before [`PrismVisibilitySystems::Sync`]
//! in `PostUpdate`; visibility consumers run after [`PrismVisibilitySystems::Cull`].
//! Culling and boolean writeback stay in the authoritative main world. Render
//! extraction may read the resulting scene/work but never writes these results
//! back from a render world. This plugin does not install a GPU draw pipeline or
//! assign the separate `crate::RenderTransformBinding` component.

mod components;
mod lower;
mod scene;
mod systems;

pub use components::{PrismCamera, PrismFrustum, PrismRenderable, PrismViewVisibility};
pub use lower::{scene_bounds, scene_transform};
pub use prism_render_architecture::gpu_scene::{GeometryHandle, SceneHandle, SceneMaterialHandle};
pub use prism_render_material::MaterialRecord;
pub use prism_render_visibility::{GeometryLod, GeometryLodChain, ViewFlags, ViewHandle};
pub use prism_transform::ecs::LocalBounds;
pub use scene::{PrismRenderScene, PrismViewResult};
pub use systems::{cull_views, sync_scene, PrismVisibilityPlugin, PrismVisibilitySystems};

#[cfg(test)]
mod native_tests;
