//! Labels that identify the render sub-app and its frame schedule.

/// The label of the dedicated render [`SubApp`](prism_app::SubApp).
///
/// A real [`SubAppLabel`](prism_app::SubAppLabel) (the blanket impl applies to
/// any `Clone + Eq + Hash + Debug + Send + Sync + 'static` type), this is the
/// key the render sub-app is registered under in the app's sub-app collection.
/// It is the `RenderApp` of the Prism engine: the world that owns the GPU
/// device, the frame target, and the render schedule.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct RenderApp;

/// The render sub-app's per-frame [`ScheduleLabel`](prism_ecs::schedule::ScheduleLabel).
///
/// Systems that build and record GPU work are registered here. Each frame the
/// render sub-app's `Last` phase runs this schedule, so a single
/// `App::update` extracts the main world, then records and submits one frame.
///
/// The label is separate from the core phases on purpose: render systems are
/// ordered relative to each other (prepare → queue → record → submit) without
/// reordering the main-frame phase sequence.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Render;
