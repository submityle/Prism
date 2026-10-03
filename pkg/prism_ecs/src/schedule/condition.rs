//! Run conditions (design §8.2): read-only systems returning `bool` that gate
//! whether a system (or a whole [`SystemSet`](crate::schedule::SystemSet)) runs
//! this tick.
//!
//! A condition is any [`IntoSystem`] whose output is `bool`. The schedule boxes
//! them into [`BoxedCondition`], initialises them alongside normal systems, and
//! evaluates each at most once per run (set conditions are cached the first time
//! a member system is reached).
//!
//! A small library of combinators ([`not`], [`and`], [`or`]) and built-ins
//! ([`run_once`], [`resource_exists`], [`resource_equals`]) is provided; all are
//! real systems, not markers.

use alloc::boxed::Box;

use crate::resource::Resource;
use crate::system::function::{IntoSystem, System};
use crate::system::world_cell::UnsafeWorldCell;
use crate::query::Access;
use crate::world::World;

/// A type-erased, owned run condition.
pub type BoxedCondition = Box<dyn System<Out = bool>>;

/// Anything convertible into a `bool`-producing [`System`]. Blanket-implemented
/// for every [`IntoSystem<bool, _>`](IntoSystem).
pub trait Condition<Marker>: IntoSystem<bool, Marker> {
    /// Convert into a [`BoxedCondition`].
    fn into_boxed_condition(self) -> BoxedCondition {
        Box::new(IntoSystem::into_system(self))
    }
}

impl<Marker, T> Condition<Marker> for T where T: IntoSystem<bool, Marker> {}

/// Returns a condition that is `true` only on its very first evaluation and
/// `false` on every subsequent tick. Backed by a private
/// [`Local`](crate::system::Local) flag.
#[inline]
pub fn run_once() -> impl FnMut(crate::system::Local<bool>) -> bool {
    |mut has_run: crate::system::Local<bool>| {
        if *has_run {
            false
        } else {
            *has_run = true;
            true
        }
    }
}

/// Returns a condition that is `true` while resource `R` exists in the world.
#[inline]
pub fn resource_exists<R: Resource>()
-> impl FnMut(Option<crate::system::Res<R>>) -> bool {
    |res: Option<crate::system::Res<R>>| res.is_some()
}

/// Returns a condition that is `true` while resource `R` exists and equals
/// `value`.
#[inline]
pub fn resource_equals<R>(value: R) -> impl FnMut(Option<crate::system::Res<R>>) -> bool
where
    R: Resource + PartialEq,
{
    move |res: Option<crate::system::Res<R>>| {
        res.map(|r| *r == value).unwrap_or(false)
    }
}

/// A condition that inverts another condition.
pub struct Not {
    inner: BoxedCondition,
}

impl Not {
    /// Wrap `inner`, negating its output.
    #[inline]
    #[must_use]
    pub fn new<M>(inner: impl Condition<M>) -> Self {
        Self {
            inner: inner.into_boxed_condition(),
        }
    }
}

impl System for Not {
    type Out = bool;

    #[inline]
    fn name(&self) -> &str {
        "Not"
    }

    #[inline]
    fn initialize(&mut self, world: &mut World) {
        self.inner.initialize(world);
    }

    #[inline]
    fn access(&self) -> &Access {
        self.inner.access()
    }

    #[inline]
    fn is_exclusive(&self) -> bool {
        self.inner.is_exclusive()
    }

    #[inline]
    unsafe fn run_unsafe(&mut self, world: UnsafeWorldCell<'_>) -> bool {
        // SAFETY: forwarded unchanged; the caller's non-aliasing guarantee for
        // `Not`'s access (which is exactly `inner`'s access) covers `inner`.
        !unsafe { self.inner.run_unsafe(world) }
    }

    #[inline]
    fn apply_deferred(&mut self, world: &mut World) {
        self.inner.apply_deferred(world);
    }

    #[inline]
    fn run(&mut self, world: &mut World) -> bool {
        !self.inner.run(world)
    }
}

/// Negate a condition. `not(c)` runs the system when `c` is `false`.
#[inline]
#[must_use]
pub fn not<M>(condition: impl Condition<M>) -> Not {
    Not::new(condition)
}

/// A binary combinator over two conditions.
enum Combine {
    And,
    Or,
}

/// A condition combining two inner conditions with `&&` or `||`.
///
/// Both inner conditions are always evaluated (no short-circuit) so that any
/// internal state (e.g. a [`Local`](crate::system::Local)) advances
/// deterministically regardless of the other branch.
pub struct Combined {
    a: BoxedCondition,
    b: BoxedCondition,
    op: Combine,
    access: Access,
}

impl Combined {
    fn new<Ma, Mb>(a: impl Condition<Ma>, b: impl Condition<Mb>, op: Combine) -> Self {
        Self {
            a: a.into_boxed_condition(),
            b: b.into_boxed_condition(),
            op,
            access: Access::new(),
        }
    }
}

impl System for Combined {
    type Out = bool;

    #[inline]
    fn name(&self) -> &str {
        match self.op {
            Combine::And => "And",
            Combine::Or => "Or",
        }
    }

    #[inline]
    fn initialize(&mut self, world: &mut World) {
        self.a.initialize(world);
        self.b.initialize(world);
        let mut access = Access::new();
        access.extend(self.a.access());
        access.extend(self.b.access());
        self.access = access;
    }

    #[inline]
    fn access(&self) -> &Access {
        &self.access
    }

    #[inline]
    fn is_exclusive(&self) -> bool {
        self.a.is_exclusive() || self.b.is_exclusive()
    }

    #[inline]
    unsafe fn run_unsafe(&mut self, world: UnsafeWorldCell<'_>) -> bool {
        // SAFETY: `self.access` is the union of both inner accesses, so the
        // caller's non-aliasing guarantee covers evaluating both in turn.
        let a = unsafe { self.a.run_unsafe(world) };
        // SAFETY: same guarantee as above — `self.access` unions both inner
        // accesses, so the caller's non-aliasing contract covers `b` too.
        let b = unsafe { self.b.run_unsafe(world) };
        match self.op {
            Combine::And => a && b,
            Combine::Or => a || b,
        }
    }

    #[inline]
    fn apply_deferred(&mut self, world: &mut World) {
        self.a.apply_deferred(world);
        self.b.apply_deferred(world);
    }

    #[inline]
    fn run(&mut self, world: &mut World) -> bool {
        let a = self.a.run(world);
        let b = self.b.run(world);
        match self.op {
            Combine::And => a && b,
            Combine::Or => a || b,
        }
    }
}

/// Combine two conditions with logical AND (both evaluated).
#[inline]
#[must_use]
pub fn and<Ma, Mb>(a: impl Condition<Ma>, b: impl Condition<Mb>) -> Combined {
    Combined::new(a, b, Combine::And)
}

/// Combine two conditions with logical OR (both evaluated).
#[inline]
#[must_use]
pub fn or<Ma, Mb>(a: impl Condition<Ma>, b: impl Condition<Mb>) -> Combined {
    Combined::new(a, b, Combine::Or)
}
