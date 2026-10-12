//! Shared task-pool wiring for the `multi_thread` feature (design §8.2).

use prism_ecs::resource::Resource;
use prism_tasks::TaskPool;

/// The `prism_tasks` pool a sub-app's schedules dispatch onto when the
/// `multi_thread` feature is on (design §8.2 conflict-graph executor).
///
/// Stored as a world resource so
/// [`SubApp::run_schedule`](crate::sub_app::SubApp::run_schedule) can fetch a
/// cheap clone and hand it to
/// [`World::run_schedule_parallel`](prism_ecs::world::World::run_schedule_parallel);
/// a world without it runs its schedules single-threaded. The pool is wrapped in
/// a newtype because the orphan rule forbids implementing the resource marker on
/// `TaskPool` here.
#[derive(Clone)]
pub struct TaskPoolResource(pub TaskPool);

impl Resource for TaskPoolResource {}
