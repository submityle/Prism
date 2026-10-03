//! The [`App`]: the top-level engine assembly and driver.
//!
//! An `App` owns a main [`SubApp`] (world + schedules), an ordered set of
//! [`Plugin`]s, a [`PluginsState`] assembly state machine, and a swappable
//! [runner](crate::runner). The public surface mirrors `bevy_app`'s so the
//! eventual migration is near "change-the-import":
//!
//! ```no_run
//! use prism_app::prelude::*;
//!
//! App::new()
//!     .add_systems(Startup, || { /* setup */ })
//!     .add_systems(Update, || { /* per-frame */ })
//!     .run();
//! ```

use std::any::TypeId;
use std::collections::HashSet;

use prism_ecs::resource::Resource;
use prism_ecs::schedule::{IntoSystemConfigs, Schedule, ScheduleLabel, Schedules};
use prism_ecs::world::World;

use crate::exit::{AppExit, AppExitRequest};
use crate::plugin::Plugin;
use crate::plugin_group::PluginGroup;
use crate::schedule::{
    First, Last, PostStartup, PostUpdate, PreStartup, PreUpdate, StateTransition, Startup, Update,
};
use crate::sub_app::SubApp;

/// The monotonic plugin-assembly state machine (design §21).
///
/// It only ever advances; there is no path back to an earlier state. Hot
/// reloading (which conceptually "re-adds" plugins) is a later milestone that
/// goes through an explicit rebuild, not a backward transition.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PluginsState {
    /// Plugins may still be added; their [`build`](Plugin::build) runs on add.
    Adding,
    /// Every plugin reported [`ready`](Plugin::ready); eligible to finish.
    Ready,
    /// Every plugin's [`finish`](Plugin::finish) has run.
    Finished,
    /// Every plugin's [`cleanup`](Plugin::cleanup) has run.
    Cleaned,
}

/// A boxed runner: consumes the fully-assembled [`App`] and drives it to an
/// [`AppExit`].
type RunnerFn = Box<dyn FnOnce(App) -> AppExit>;

/// The top-level engine instance.
pub struct App {
    main: SubApp,
    runner: Option<RunnerFn>,
    plugins: Vec<Box<dyn Plugin>>,
    plugin_names: HashSet<String>,
    plugins_state: PluginsState,
    /// State types already wired into the [`StateTransition`](crate::schedule::StateTransition)
    /// schedule, so a repeated `insert_state`/`init_state` only re-queues the
    /// initial value instead of registering a second transition system.
    pub(crate) initialized_states: HashSet<TypeId>,
    /// Event types already registered via [`App::add_event`](crate::App::add_event),
    /// so a repeated `add_event` neither reinserts the `Events` resource (which
    /// would discard buffered events) nor schedules a second rotation system.
    pub(crate) added_events: HashSet<TypeId>,
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}

impl App {
    /// Create an app with a main sub-app, the built-in core phase schedules
    /// installed (empty), and the [`AppExitRequest`] resource ready for systems
    /// to signal shutdown.
    pub fn new() -> Self {
        let mut app = Self::empty();
        app.init_core_schedules();
        app.main.world.insert_resource(AppExitRequest::default());
        app
    }

    /// Install an empty [`Schedule`] for every core phase so the phase labels
    /// always resolve, even before a user adds systems to them.
    ///
    /// The schedules live in the main world's
    /// [`Schedules`](prism_ecs::schedule::Schedules) resource (design §5), run
    /// by label through [`World::run_schedule`](prism_ecs::world::World::run_schedule).
    fn init_core_schedules(&mut self) {
        let schedules = self.main.world.resource_mut::<Schedules>();
        schedules.insert(PreStartup, Schedule::new());
        schedules.insert(Startup, Schedule::new());
        schedules.insert(PostStartup, Schedule::new());
        schedules.insert(First, Schedule::new());
        schedules.insert(PreUpdate, Schedule::new());
        schedules.insert(StateTransition, Schedule::new());
        schedules.insert(Update, Schedule::new());
        schedules.insert(PostUpdate, Schedule::new());
        schedules.insert(Last, Schedule::new());
    }

    /// Create a bare app: a main sub-app with no schedules, no plugins, and no
    /// resources installed. Used as the `mem::replace` sentinel in
    /// [`run`](App::run); prefer [`new`](App::new) for real use.
    pub fn empty() -> Self {
        Self {
            main: SubApp::new(),
            runner: None,
            plugins: Vec::new(),
            plugin_names: HashSet::new(),
            plugins_state: PluginsState::Adding,
            initialized_states: HashSet::new(),
            added_events: HashSet::new(),
        }
    }

    // ---- assembly ---------------------------------------------------------

    /// Add one or more plugins. Accepts a single [`Plugin`], a tuple of
    /// plugins, or a [`PluginGroup`]. Each plugin's [`build`](Plugin::build)
    /// runs immediately.
    ///
    /// # Panics
    ///
    /// Panics if a unique plugin (default) is added twice, or if called after
    /// plugin assembly has been finalized.
    pub fn add_plugins<M>(&mut self, plugins: impl Plugins<M>) -> &mut Self {
        plugins.add_to_app(self);
        self
    }

    /// Add a single already-boxed plugin, honoring uniqueness and running its
    /// `build` immediately. Shared by every [`add_plugins`](App::add_plugins)
    /// entry point.
    pub(crate) fn add_boxed_plugin(&mut self, plugin: Box<dyn Plugin>) {
        assert_eq!(
            self.plugins_state,
            PluginsState::Adding,
            "cannot add plugin {:?} after plugin assembly has been finalized",
            plugin.name()
        );
        if plugin.is_unique() && !self.plugin_names.insert(plugin.name().to_string()) {
            panic!(
                "plugin {:?} was added more than once; override Plugin::is_unique to allow duplicates",
                plugin.name()
            );
        }
        plugin.build(self);
        self.plugins.push(plugin);
    }

    /// Register systems under a schedule label (creating the schedule if it
    /// does not yet exist).
    pub fn add_systems<M>(
        &mut self,
        label: impl ScheduleLabel + Clone,
        systems: impl IntoSystemConfigs<M>,
    ) -> &mut Self {
        let schedules = self.main.world.resource_mut::<Schedules>();
        if !schedules.contains(label.clone()) {
            schedules.insert(label.clone(), Schedule::new());
        }
        schedules
            .get_mut(label)
            .expect("schedule was just ensured to exist")
            .add_systems(systems);
        self
    }

    /// Insert a resource into the main world, returning `&mut self` for
    /// chaining.
    pub fn insert_resource<R: Resource>(&mut self, value: R) -> &mut Self {
        self.main.world.insert_resource(value);
        self
    }

    /// Insert a resource via [`Default`] if absent, returning `&mut self`.
    pub fn init_resource<R: Resource + Default>(&mut self) -> &mut Self {
        self.main.world.init_resource::<R>();
        self
    }

    /// Set the runner that drives this app in [`run`](App::run).
    pub fn set_runner(&mut self, runner: impl FnOnce(App) -> AppExit + 'static) -> &mut Self {
        self.runner = Some(Box::new(runner));
        self
    }

    // ---- plugin lifecycle -------------------------------------------------

    /// The current plugin-assembly state.
    pub fn plugins_state(&self) -> PluginsState {
        self.plugins_state
    }

    /// Whether every plugin currently reports [`ready`](Plugin::ready).
    pub fn all_plugins_ready(&self) -> bool {
        self.plugins.iter().all(|p| p.ready(self))
    }

    /// Advance assembly to [`Finished`](PluginsState::Finished): verify every
    /// plugin is ready, then run each plugin's [`finish`](Plugin::finish).
    ///
    /// Idempotent once past [`Adding`](PluginsState::Adding).
    ///
    /// # Panics
    ///
    /// Panics if a plugin is not yet ready. M0 has no async runner loop to wait
    /// on readiness; a plugin that defers readiness is an M1 concern, and
    /// failing loudly here is more honest than silently finishing early.
    pub fn finish(&mut self) -> &mut Self {
        if self.plugins_state != PluginsState::Adding {
            return self;
        }
        assert!(
            self.all_plugins_ready(),
            "App::finish called while a plugin is not ready; async readiness waiting is an M1 runner feature"
        );
        self.plugins_state = PluginsState::Ready;
        let plugins = core::mem::take(&mut self.plugins);
        for plugin in &plugins {
            plugin.finish(self);
        }
        self.plugins = plugins;
        self.plugins_state = PluginsState::Finished;
        self
    }

    /// Advance assembly to [`Cleaned`](PluginsState::Cleaned) by running each
    /// plugin's [`cleanup`](Plugin::cleanup). Requires [`finish`](App::finish)
    /// to have run; idempotent once cleaned.
    pub fn cleanup(&mut self) -> &mut Self {
        if self.plugins_state != PluginsState::Finished {
            return self;
        }
        let plugins = core::mem::take(&mut self.plugins);
        for plugin in &plugins {
            plugin.cleanup(self);
        }
        self.plugins = plugins;
        self.plugins_state = PluginsState::Cleaned;
        self
    }

    // ---- running ----------------------------------------------------------

    /// Run the startup schedules exactly once, in order
    /// (`PreStartup → Startup → PostStartup`).
    fn run_startup(&mut self) {
        self.main.run_startup();
    }

    /// Run one variable-step frame of the main sub-app (design §7 M0 subset).
    pub fn update(&mut self) {
        self.main.update();
    }

    /// Finalize plugins, run startup once, then hand the app to its runner.
    ///
    /// If no runner was set, defaults to [`run_once`](crate::runner::run_once),
    /// which drives exactly one frame. Returns the runner's [`AppExit`].
    pub fn run(&mut self) -> AppExit {
        self.finish();
        self.cleanup();
        self.run_startup();

        let mut app = core::mem::replace(self, App::empty());
        let runner = app
            .runner
            .take()
            .unwrap_or_else(|| Box::new(crate::runner::run_once));
        runner(app)
    }

    /// The pending exit request, if a system has signalled shutdown via
    /// [`AppExitRequest`].
    pub fn should_exit(&self) -> Option<AppExit> {
        self.main
            .world
            .get_resource::<AppExitRequest>()
            .and_then(AppExitRequest::get)
    }

    // ---- accessors --------------------------------------------------------

    /// Shared access to the main sub-app.
    pub fn main(&self) -> &SubApp {
        &self.main
    }

    /// Mutable access to the main sub-app.
    pub fn main_mut(&mut self) -> &mut SubApp {
        &mut self.main
    }

    /// Shared access to the main world.
    pub fn world(&self) -> &World {
        &self.main.world
    }

    /// Mutable access to the main world.
    pub fn world_mut(&mut self) -> &mut World {
        &mut self.main.world
    }
}

/// Types that can be added via [`App::add_plugins`]: a single [`Plugin`], a
/// tuple of pluginish values, or a [`PluginGroup`].
///
/// `Marker` disambiguates the blanket impls (the standard trait-marker trick);
/// callers never name it.
pub trait Plugins<Marker> {
    /// Add every plugin this value represents to `app`.
    fn add_to_app(self, app: &mut App);
}

/// Marker for the single-[`Plugin`] impl.
#[doc(hidden)]
pub struct IsPlugin;

/// Marker for the [`PluginGroup`] impl.
#[doc(hidden)]
pub struct IsPluginGroup;

/// Marker for the tuple impl.
#[doc(hidden)]
pub struct IsPluginTuple;

impl<P: Plugin> Plugins<IsPlugin> for P {
    fn add_to_app(self, app: &mut App) {
        app.add_boxed_plugin(Box::new(self));
    }
}

impl<G: PluginGroup> Plugins<IsPluginGroup> for G {
    fn add_to_app(self, app: &mut App) {
        for plugin in self.build().into_plugins() {
            app.add_boxed_plugin(plugin);
        }
    }
}

macro_rules! impl_plugins_for_tuple {
    ($(($P:ident, $M:ident)),+) => {
        impl<$($P, $M),+> Plugins<(IsPluginTuple, $($M,)+)> for ($($P,)+)
        where
            $($P: Plugins<$M>,)+
        {
            #[allow(non_snake_case, unused_variables)]
            fn add_to_app(self, app: &mut App) {
                let ($($P,)+) = self;
                $($P.add_to_app(app);)+
            }
        }
    };
}

impl_plugins_for_tuple!((P0, M0));
impl_plugins_for_tuple!((P0, M0), (P1, M1));
impl_plugins_for_tuple!((P0, M0), (P1, M1), (P2, M2));
impl_plugins_for_tuple!((P0, M0), (P1, M1), (P2, M2), (P3, M3));
impl_plugins_for_tuple!((P0, M0), (P1, M1), (P2, M2), (P3, M3), (P4, M4));
impl_plugins_for_tuple!((P0, M0), (P1, M1), (P2, M2), (P3, M3), (P4, M4), (P5, M5));
impl_plugins_for_tuple!(
    (P0, M0), (P1, M1), (P2, M2), (P3, M3), (P4, M4), (P5, M5), (P6, M6)
);
impl_plugins_for_tuple!(
    (P0, M0), (P1, M1), (P2, M2), (P3, M3), (P4, M4), (P5, M5), (P6, M6), (P7, M7)
);
