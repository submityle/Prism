//! The render sub-app's GPU-resource startup: the [`RenderStartup`] schedule
//! and the `init_gpu_resource` seam.
//!
//! A renderer's GPU resources cannot be plain `Default` values — a vertex buffer
//! has to be created against the device, a bind group against a layout that
//! device compiled — so Bevy runs a `RenderStartup` schedule every time a render
//! device is acquired, and `init_gpu_resource::<R>` is how a subsystem registers
//! one. Prism keeps that shape:
//!
//! ```no_run
//! use prism_app::App;
//! use prism_ecs::prelude::Resource;
//! use prism_ecs::resource::FromWorld;
//! use prism_render_app::prelude::*;
//! use prism_render_app::{GpuResourceAppExt, RenderApp};
//! use prism_render_driver_wgpu::WgpuInstance;
//!
//! #[derive(Resource)]
//! struct MyGpuBuffers;
//!
//! impl FromWorld for MyGpuBuffers {
//!     fn from_world(_: &mut prism_ecs::world::World) -> Self {
//!         Self
//!     }
//! }
//!
//! let mut app = App::new();
//! app.add_plugins(PrismRenderPlugin::new(WgpuInstance::new()));
//! app.sub_app_mut(RenderApp)
//!     .expect("the render sub-app")
//!     .init_gpu_resource::<MyGpuBuffers>();
//! ```
//!
//! The schedule runs once, lazily, before the first frame's
//! [`Render`](crate::Render) schedule — so a plugin may register a resource from
//! *any* plugin, in any order, and it is still built before anything reads it.
//! [`RenderStartupState`] is the guard: clearing it (device recovery, a rebuilt
//! runtime) makes the next frame rebuild every registered resource.

use prism_app::SubApp;
use prism_ecs::prelude::{FromWorld, Resource};
use prism_ecs::schedule::{Schedule, Schedules};
use prism_ecs::world::World;

use crate::label::RenderStartup;

/// Constructs `R` from the render world and inserts it.
///
/// `R` sees the whole world, so it can read the runtime (device and queue) it
/// must be created against, and any other resource it depends on. This function
/// is itself the startup *system*: registering it (directly, or through
/// [`GpuResourceAppExt::init_gpu_resource`]) is the whole contract.
pub fn init_gpu_resource<R: Resource + FromWorld>(world: &mut World) {
    let resource = R::from_world(world);
    world.insert_resource(resource);
}

/// Whether [`RenderStartup`] has run against the current device.
///
/// The guard that makes startup idempotent (one run per device, however many
/// frames follow) and re-runnable: a device-recovery path calls
/// [`invalidate`](Self::invalidate) and the next frame rebuilds every registered
/// GPU resource.
#[derive(Resource, Default, Debug)]
pub struct RenderStartupState {
    initialized: bool,
}

impl RenderStartupState {
    /// Whether startup has already run for the current device.
    #[must_use]
    pub const fn initialized(&self) -> bool {
        self.initialized
    }

    /// Records that startup has run for the current device.
    pub fn mark_initialized(&mut self) {
        self.initialized = true;
    }

    /// Marks the current startup stale, so the next frame re-runs it.
    pub fn invalidate(&mut self) {
        self.initialized = false;
    }
}

/// Registers a GPU resource to be built during [`RenderStartup`].
///
/// The Prism counterpart of Bevy's `GpuResourceAppExt`, for the same reason: a
/// subsystem should not have to know when the device is acquired, only that its
/// resource exists before the first frame that reads it.
pub trait GpuResourceAppExt {
    /// Registers `R` to be constructed during [`RenderStartup`].
    fn init_gpu_resource<R: Resource + FromWorld>(&mut self) -> &mut Self;
}

impl GpuResourceAppExt for SubApp {
    fn init_gpu_resource<R: Resource + FromWorld>(&mut self) -> &mut Self {
        if !self
            .world
            .resource_mut::<Schedules>()
            .contains(RenderStartup)
        {
            self.world
                .resource_mut::<Schedules>()
                .insert(RenderStartup, Schedule::new());
        }
        self.world
            .resource_mut::<Schedules>()
            .get_mut(RenderStartup)
            .expect("the startup schedule was just registered")
            .add_systems(init_gpu_resource::<R>);
        self
    }
}

/// Runs [`RenderStartup`] once for the current device, before a frame needs it.
///
/// Idempotent: after the first call the state marks startup done and later calls
/// return immediately, so this is cheap enough to sit on the per-frame path. A
/// caller that replaced the device calls
/// [`RenderStartupState::invalidate`] first, and the next call rebuilds every
/// registered resource.
pub(crate) fn ensure_render_startup(world: &mut World) {
    world.init_resource::<RenderStartupState>();
    if world.resource::<RenderStartupState>().initialized() {
        return;
    }
    world.run_schedule(RenderStartup);
    world
        .resource_mut::<RenderStartupState>()
        .mark_initialized();
}
