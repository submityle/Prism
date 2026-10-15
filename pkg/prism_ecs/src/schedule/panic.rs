//! Opt-in isolation of unwinding system panics.
//!
//! Insert [`SystemPanicPolicy::DisableAndContinue`] into a world to retire a
//! failing system and continue its schedule. Direct writes and commands already
//! applied are not rolled back. Unapplied commands and local state are dropped
//! with the failed system. Initialization and invalid schedule graphs remain
//! fail-fast; `panic = "abort"` cannot be intercepted.

use alloc::boxed::Box;
use alloc::collections::VecDeque;
use alloc::string::{String, ToString};
use core::any::Any;
use core::marker::PhantomData;
use std::panic::{self, AssertUnwindSafe};

use crate::query::Access;
use crate::resource::Resource;
use crate::system::{System, UnsafeWorldCell};
use crate::world::World;

/// How executors react to a system or run-condition panic in this world.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum SystemPanicPolicy {
    /// Propagate the panic. Also the behavior when no policy is present.
    #[default]
    FailFast,
    /// Retire the failing system (or close the failing shared condition), record
    /// the failure, and run the remaining systems. This is not a transaction.
    DisableAndContinue,
}
impl Resource for SystemPanicPolicy {}

impl SystemPanicPolicy {
    pub(crate) fn read(world: &World) -> Self {
        world.get_resource::<Self>().copied().unwrap_or_default()
    }
    pub(crate) fn invoke<T>(self, f: impl FnOnce() -> T) -> Result<T, PanicPayload> {
        match self {
            Self::FailFast => Ok(f()),
            Self::DisableAndContinue => panic::catch_unwind(AssertUnwindSafe(f)),
        }
    }
}

/// The executor operation that failed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SystemFailureStage {
    /// Serial run (including deferred application), or a parallel body.
    Run,
    /// Deferred application at a parallel wave's sync point.
    Deferred,
    /// A node's private run condition.
    Condition,
    /// A shared set condition. Its gate stays closed for every set member.
    SetCondition,
}

/// A captured failure. Node indices are local to the schedule that ran it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SystemFailure {
    /// System/condition name, truncated to at most 1024 UTF-8 bytes.
    pub name: String,
    /// Node index, or `None` for a shared set condition.
    pub node: Option<usize>,
    /// Operation that panicked.
    pub stage: SystemFailureStage,
    /// Panic text, truncated to at most 4096 UTF-8 bytes.
    pub message: String,
}

/// Bounded failure history, inserted lazily on the first isolated panic.
/// Parallel failures are recorded in schedule order after every worker joins.
#[derive(Debug)]
pub struct SystemFailures {
    entries: VecDeque<SystemFailure>,
    capacity: usize,
    total: u64,
}
impl Resource for SystemFailures {}
impl Default for SystemFailures {
    fn default() -> Self {
        Self::with_capacity(128)
    }
}
impl SystemFailures {
    /// Retain at most `capacity` failures. Zero keeps only the total counter.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            entries: VecDeque::new(),
            capacity,
            total: 0,
        }
    }
    /// Retained failures, oldest first.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = &SystemFailure> {
        self.entries.iter()
    }
    /// Total isolated failures, including entries evicted or cleared.
    #[must_use]
    pub fn total(&self) -> u64 {
        self.total
    }
    /// Remove retained entries while preserving the total failure count.
    pub fn clear(&mut self) {
        self.entries.clear();
    }
    pub(crate) fn record(world: &mut World, failure: SystemFailure) {
        if !world.contains_resource::<Self>() {
            world.insert_resource(Self::default());
        }
        let history = world.resource_mut::<Self>();
        history.total = history.total.saturating_add(1);
        if history.capacity != 0 {
            if history.entries.len() == history.capacity {
                history.entries.pop_front();
            }
            history.entries.push_back(failure);
        }
    }
}

pub(crate) type PanicPayload = Box<dyn Any + Send + 'static>;

pub(crate) fn failure(
    name: &str,
    node: Option<usize>,
    stage: SystemFailureStage,
    payload: &PanicPayload,
) -> SystemFailure {
    let message = if let Some(s) = payload.downcast_ref::<String>() {
        s.as_str()
    } else if let Some(s) = payload.downcast_ref::<&str>() {
        s
    } else {
        "<non-string panic payload>"
    };
    SystemFailure {
        name: bounded(name, 1024),
        node,
        stage,
        message: bounded(message, 4096),
    }
}

fn bounded(text: &str, max: usize) -> String {
    let mut end = text.len().min(max);
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    text[..end].to_string()
}

/// A retired system retains its name and ordering position, but owns no old
/// parameters, commands, or accesses. `bool::default()` permanently closes gates.
pub(crate) struct RetiredSystem<T> {
    name: String,
    access: Access,
    output: PhantomData<T>,
}
impl<T> RetiredSystem<T> {
    pub(crate) fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            access: Access::default(),
            output: PhantomData,
        }
    }
}
impl<T: Default + Send + Sync + 'static> System for RetiredSystem<T> {
    type Out = T;
    fn name(&self) -> &str {
        &self.name
    }
    fn initialize(&mut self, _: &mut World) {}
    fn access(&self) -> &Access {
        &self.access
    }
    unsafe fn run_unsafe(&mut self, _: UnsafeWorldCell<'_>) -> T {
        T::default()
    }
    fn apply_deferred(&mut self, _: &mut World) {}
}
