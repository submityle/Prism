//! The generic continuous-axis store shared by analog inputs.
//!
//! [`Axis<T>`] maps each axis identifier of type `T` to a scalar value, used
//! for gamepad sticks/triggers and any other analog source. Values are stored
//! raw; [`get`](Axis::get) clamps to the conventional `[-1.0, 1.0]` range while
//! [`get_unclamped`](Axis::get_unclamped) exposes the raw reading for callers
//! that apply their own response curve.

use alloc::collections::BTreeMap;
use alloc::collections::btree_map;

/// Stores the current value of a set of analog axes of type `T`.
#[derive(Clone, PartialEq, Debug)]
pub struct Axis<T: Copy + Ord> {
    values: BTreeMap<T, f32>,
}

impl<T: Copy + Ord> Default for Axis<T> {
    fn default() -> Self {
        Self {
            values: BTreeMap::new(),
        }
    }
}

impl<T: Copy + Ord> Axis<T> {
    /// The smallest value [`get`](Self::get) returns.
    pub const MIN: f32 = -1.0;
    /// The largest value [`get`](Self::get) returns.
    pub const MAX: f32 = 1.0;

    /// Creates an empty axis store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the raw value of `axis`, returning the previous raw value if any.
    pub fn set(&mut self, axis: T, value: f32) -> Option<f32> {
        self.values.insert(axis, value)
    }

    /// Returns the value of `axis` clamped to `[MIN, MAX]`, or `None` if the
    /// axis has never been set.
    #[must_use]
    pub fn get(&self, axis: T) -> Option<f32> {
        self.values
            .get(&axis)
            .copied()
            .map(|v| v.clamp(Self::MIN, Self::MAX))
    }

    /// Returns the raw stored value of `axis`, without clamping.
    #[must_use]
    pub fn get_unclamped(&self, axis: T) -> Option<f32> {
        self.values.get(&axis).copied()
    }

    /// Removes `axis`, returning its last raw value if it was present.
    pub fn remove(&mut self, axis: T) -> Option<f32> {
        self.values.remove(&axis)
    }

    /// Whether `axis` has a stored value.
    #[must_use]
    pub fn contains(&self, axis: T) -> bool {
        self.values.contains_key(&axis)
    }

    /// Iterates the axes that currently have a value, in deterministic order.
    pub fn all_axes(&self) -> btree_map::Keys<'_, T, f32> {
        self.values.keys()
    }
}
