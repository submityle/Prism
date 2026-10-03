//! Sub-app labels: type-erased keys identifying a secondary
//! [`SubApp`](crate::sub_app::SubApp) stored in the
//! [`SubApps`](crate::sub_app::SubApps) collection (design §5, §21).
//!
//! A label is any `Clone + Eq + Hash + Debug + Send + Sync + 'static` type —
//! typically a tiny unit struct such as a `RenderApp` marker. The design calls
//! out `SubAppLabel` as a **versioned contract** (§21), so this mirrors the
//! [`ScheduleLabel`](prism_ecs::schedule::ScheduleLabel) shape exactly: the
//! trait exposes manual `dyn`-dispatched clone/eq/hash so a
//! [`BoxedSubAppLabel`] can act as a map key without the caller knowing the
//! concrete type, and the hash mixes in the concrete
//! [`TypeId`] so two different label types can never collide
//! even if their inner values hash identically.

use std::any::{Any, TypeId};
use std::fmt::Debug;
use std::hash::{Hash, Hasher};

/// A type-erasable key identifying a secondary [`SubApp`](crate::sub_app::SubApp).
///
/// Blanket-implemented for every `Clone + Eq + Hash + Debug + Send + Sync +
/// 'static` type, so user code never implements it by hand — it just derives
/// the standard traits on a marker type.
pub trait SubAppLabel: Debug + Send + Sync + 'static {
    /// Clone `self` into a fresh boxed trait object.
    fn dyn_clone(&self) -> Box<dyn SubAppLabel>;

    /// Upcast to [`Any`] for downcasting during equality checks.
    fn as_any(&self) -> &dyn Any;

    /// Structural equality against another type-erased label. Returns `false`
    /// when the concrete types differ.
    fn dyn_eq(&self, other: &dyn SubAppLabel) -> bool;

    /// Hash `self`, first mixing in the concrete [`TypeId`] so distinct label
    /// types with equal inner hashes never collide.
    fn dyn_hash(&self, state: &mut dyn Hasher);
}

impl<T> SubAppLabel for T
where
    T: Clone + Eq + Hash + Debug + Send + Sync + 'static,
{
    #[inline]
    fn dyn_clone(&self) -> Box<dyn SubAppLabel> {
        Box::new(self.clone())
    }

    #[inline]
    fn as_any(&self) -> &dyn Any {
        self
    }

    #[inline]
    fn dyn_eq(&self, other: &dyn SubAppLabel) -> bool {
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

/// An owned, type-erased [`SubAppLabel`] usable as a map key.
///
/// Wraps a `Box<dyn SubAppLabel>` and forwards [`Clone`], [`PartialEq`],
/// [`Eq`], and [`Hash`] through the trait's `dyn_*` methods, so labels of
/// different concrete types can key one [`SubApps`](crate::sub_app::SubApps)
/// collection.
pub struct BoxedSubAppLabel(Box<dyn SubAppLabel>);

impl BoxedSubAppLabel {
    /// Box a concrete label.
    #[inline]
    #[must_use]
    pub fn new(label: impl SubAppLabel) -> Self {
        Self(Box::new(label))
    }

    /// Borrow the inner label as a trait object.
    #[inline]
    #[must_use]
    pub fn as_dyn(&self) -> &dyn SubAppLabel {
        &*self.0
    }
}

impl Clone for BoxedSubAppLabel {
    #[inline]
    fn clone(&self) -> Self {
        Self(self.0.dyn_clone())
    }
}

impl PartialEq for BoxedSubAppLabel {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.0.dyn_eq(&*other.0)
    }
}

impl Eq for BoxedSubAppLabel {}

impl Hash for BoxedSubAppLabel {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.dyn_hash(state);
    }
}

impl Debug for BoxedSubAppLabel {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(&self.0, f)
    }
}
