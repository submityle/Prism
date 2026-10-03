//! Schedule labels: type-erased keys identifying a stored
//! [`Schedule`](crate::schedule::Schedule) in the
//! [`Schedules`](crate::schedule::Schedules) resource (design §8.2).
//!
//! A label is any `Clone + Eq + Hash + Debug + Send + Sync + 'static` type —
//! typically a tiny unit struct or an enum variant. The
//! [`States`](crate::schedule::State) machine uses labels heavily:
//! [`OnEnter`](crate::schedule::OnEnter) and [`OnExit`](crate::schedule::OnExit)
//! are labels carrying a state value, so a world can hold one transition
//! schedule per `(state, edge)` pair.
//!
//! Because labels of *different* concrete types must coexist as keys in one map,
//! the trait exposes manual `dyn`-dispatched clone/eq/hash so a
//! [`BoxedScheduleLabel`] can act as a `HashMap` key without the caller knowing
//! the concrete type. The hash deliberately mixes in the concrete
//! [`TypeId`](core::any::TypeId) so two different label types can never collide
//! even if their inner values hash identically.

use alloc::boxed::Box;
use core::any::{Any, TypeId};
use core::fmt::Debug;
use core::hash::{Hash, Hasher};

/// A type-erasable key identifying a [`Schedule`](crate::schedule::Schedule).
///
/// Blanket-implemented for every `Clone + Eq + Hash + Debug + Send + Sync +
/// 'static` type, so user code never implements it by hand — it just derives
/// the standard traits on a label type.
pub trait ScheduleLabel: Debug + Send + Sync + 'static {
    /// Clone `self` into a fresh boxed trait object.
    fn dyn_clone(&self) -> Box<dyn ScheduleLabel>;

    /// Upcast to [`Any`] for downcasting during equality checks.
    fn as_any(&self) -> &dyn Any;

    /// Structural equality against another type-erased label. Returns `false`
    /// when the concrete types differ.
    fn dyn_eq(&self, other: &dyn ScheduleLabel) -> bool;

    /// Hash `self`, first mixing in the concrete [`TypeId`] so distinct label
    /// types with equal inner hashes never collide.
    fn dyn_hash(&self, state: &mut dyn Hasher);
}

impl<T> ScheduleLabel for T
where
    T: Clone + Eq + Hash + Debug + Send + Sync + 'static,
{
    #[inline]
    fn dyn_clone(&self) -> Box<dyn ScheduleLabel> {
        Box::new(self.clone())
    }

    #[inline]
    fn as_any(&self) -> &dyn Any {
        self
    }

    #[inline]
    fn dyn_eq(&self, other: &dyn ScheduleLabel) -> bool {
        other
            .as_any()
            .downcast_ref::<T>()
            .is_some_and(|other| self == other)
    }

    #[inline]
    fn dyn_hash(&self, mut state: &mut dyn Hasher) {
        TypeId::of::<T>().hash(&mut state);
        Hash::hash(self, &mut state);
    }
}

/// An owned, type-erased [`ScheduleLabel`] usable as a hash-map key.
///
/// Wraps a `Box<dyn ScheduleLabel>` and forwards [`Clone`], [`PartialEq`],
/// [`Eq`], and [`Hash`] through the trait's `dyn_*` methods, so labels of
/// different concrete types can share one
/// [`Schedules`](crate::schedule::Schedules) map.
pub struct BoxedScheduleLabel(Box<dyn ScheduleLabel>);

impl BoxedScheduleLabel {
    /// Box a concrete label.
    #[inline]
    #[must_use]
    pub fn new(label: impl ScheduleLabel) -> Self {
        Self(Box::new(label))
    }

    /// Borrow the inner label as a trait object.
    #[inline]
    #[must_use]
    pub fn as_dyn(&self) -> &dyn ScheduleLabel {
        &*self.0
    }
}

impl Clone for BoxedScheduleLabel {
    #[inline]
    fn clone(&self) -> Self {
        Self(self.0.dyn_clone())
    }
}

impl PartialEq for BoxedScheduleLabel {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.0.dyn_eq(&*other.0)
    }
}

impl Eq for BoxedScheduleLabel {}

impl Hash for BoxedScheduleLabel {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.dyn_hash(state);
    }
}

impl Debug for BoxedScheduleLabel {
    #[inline]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        Debug::fmt(&self.0, f)
    }
}
