//! The unit of interruptible work the scheduler drives.
//!
//! Reconciliation is expressed as a [`Work`] value that advances one small,
//! self-contained step per [`step`](Work::step) call and reports whether it is
//! [`Done`](StepOutcome::Done) or has [`More`](StepOutcome::More) to do. The
//! retained, immutable Loom element tree makes this safe: a half-finished unit
//! leaves no torn mutable state, so the scheduler can pause between steps and
//! resume on a later frame — or let a higher lane preempt — without corruption.

use alloc::boxed::Box;

/// The result of advancing a [`Work`] by one step.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StepOutcome {
    /// More steps remain; the scheduler should resume this unit later.
    More,
    /// The unit finished and must not be stepped again.
    Done,
}

impl StepOutcome {
    /// Whether the unit has finished.
    #[must_use]
    pub const fn is_done(self) -> bool {
        matches!(self, StepOutcome::Done)
    }
}

/// A resumable unit of reconciliation.
///
/// Each [`step`](Work::step) must perform a bounded slice of work and return
/// promptly so the scheduler can re-check its deadline between steps. Returning
/// [`StepOutcome::Done`] retires the unit; the scheduler never calls `step`
/// again afterwards.
pub trait Work {
    /// Advances the unit by one bounded step.
    fn step(&mut self) -> StepOutcome;
}

/// A boxed, type-erased [`Work`] as stored in the scheduler's lane queues.
pub type BoxWork = Box<dyn Work>;

/// Adapts a closure into a [`Work`] unit.
///
/// Handy for one-off budgeted jobs whose state lives in the captured
/// environment rather than a bespoke struct.
pub struct FnWork<F> {
    step: F,
}

impl<F> FnWork<F>
where
    F: FnMut() -> StepOutcome,
{
    /// Wraps `step`, which is invoked once per scheduler step.
    #[must_use]
    pub const fn new(step: F) -> FnWork<F> {
        FnWork { step }
    }
}

impl<F> Work for FnWork<F>
where
    F: FnMut() -> StepOutcome,
{
    fn step(&mut self) -> StepOutcome {
        (self.step)()
    }
}

/// A [`Work`] that runs exactly one closure then reports [`StepOutcome::Done`].
///
/// Fuses a plain side-effecting job into the lane model when no incremental
/// slicing is needed; the whole closure runs within a single step.
pub struct OnceWork<F> {
    job: Option<F>,
}

impl<F> OnceWork<F>
where
    F: FnOnce(),
{
    /// Wraps `job` to run on the first (and only) step.
    #[must_use]
    pub const fn new(job: F) -> OnceWork<F> {
        OnceWork { job: Some(job) }
    }
}

impl<F> Work for OnceWork<F>
where
    F: FnOnce(),
{
    fn step(&mut self) -> StepOutcome {
        if let Some(job) = self.job.take() {
            job();
        }
        StepOutcome::Done
    }
}
