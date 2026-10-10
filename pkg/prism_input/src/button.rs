//! The generic press/release button tracker shared by every digital input.
//!
//! [`ButtonInput<T>`] is the keystone of the input layer: keyboard keys, mouse
//! buttons, and gamepad buttons are all just different `T`s over the same
//! edge-tracking logic. It records three sets — currently held, pressed this
//! frame, and released this frame — so systems can query level state
//! (`pressed`) and edge state (`just_pressed`/`just_released`) uniformly.
//!
//! The sets are ordered ([`BTreeSet`]) so iteration is deterministic across
//! runs and platforms, which matters for replay and lockstep networking.

use alloc::collections::btree_set;
use alloc::collections::BTreeSet;

/// Tracks the pressed/released state of a set of buttons of type `T`.
///
/// Call [`press`](Self::press) / [`release`](Self::release) as events arrive,
/// then call [`clear`](Self::clear) once per frame to retire the
/// `just_pressed` / `just_released` edges while keeping the held set intact.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ButtonInput<T: Copy + Ord> {
    pressed: BTreeSet<T>,
    just_pressed: BTreeSet<T>,
    just_released: BTreeSet<T>,
}

impl<T: Copy + Ord> Default for ButtonInput<T> {
    fn default() -> Self {
        Self {
            pressed: BTreeSet::new(),
            just_pressed: BTreeSet::new(),
            just_released: BTreeSet::new(),
        }
    }
}

impl<T: Copy + Ord> ButtonInput<T> {
    /// Creates an empty tracker with no buttons held.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a press. Marks `input` as held and as pressed this frame; a
    /// press that repeats while already held does not re-fire the edge.
    pub fn press(&mut self, input: T) {
        if self.pressed.insert(input) {
            self.just_pressed.insert(input);
        }
    }

    /// Registers a release. Clears the held state and marks `input` as released
    /// this frame; releasing a button that was not held does not fire an edge.
    pub fn release(&mut self, input: T) {
        if self.pressed.remove(&input) {
            self.just_released.insert(input);
        }
    }

    /// Releases every held button, firing a `just_released` edge for each.
    pub fn release_all(&mut self) {
        let drained: BTreeSet<T> = core::mem::take(&mut self.pressed);
        self.just_released.extend(drained);
    }

    /// Whether `input` is currently held.
    #[must_use]
    pub fn pressed(&self, input: T) -> bool {
        self.pressed.contains(&input)
    }

    /// Whether any of `inputs` is currently held.
    pub fn any_pressed(&self, inputs: impl IntoIterator<Item = T>) -> bool {
        inputs.into_iter().any(|it| self.pressed(it))
    }

    /// Whether every one of `inputs` is currently held.
    pub fn all_pressed(&self, inputs: impl IntoIterator<Item = T>) -> bool {
        inputs.into_iter().all(|it| self.pressed(it))
    }

    /// Whether `input` was pressed during this frame (a rising edge).
    #[must_use]
    pub fn just_pressed(&self, input: T) -> bool {
        self.just_pressed.contains(&input)
    }

    /// Whether any of `inputs` was pressed this frame.
    pub fn any_just_pressed(&self, inputs: impl IntoIterator<Item = T>) -> bool {
        inputs.into_iter().any(|it| self.just_pressed(it))
    }

    /// Whether `input` was released during this frame (a falling edge).
    #[must_use]
    pub fn just_released(&self, input: T) -> bool {
        self.just_released.contains(&input)
    }

    /// Whether any of `inputs` was released this frame.
    pub fn any_just_released(&self, inputs: impl IntoIterator<Item = T>) -> bool {
        inputs.into_iter().any(|it| self.just_released(it))
    }

    /// Consumes the `just_pressed` edge for `input`, returning whether it was
    /// set. Use this to claim an edge so later systems in the same frame do not
    /// also react to it.
    pub fn clear_just_pressed(&mut self, input: T) -> bool {
        self.just_pressed.remove(&input)
    }

    /// Consumes the `just_released` edge for `input`, returning whether it was
    /// set.
    pub fn clear_just_released(&mut self, input: T) -> bool {
        self.just_released.remove(&input)
    }

    /// Forgets all state for `input`: it is no longer held and both of its
    /// edges are cleared.
    pub fn reset(&mut self, input: T) {
        self.pressed.remove(&input);
        self.just_pressed.remove(&input);
        self.just_released.remove(&input);
    }

    /// Forgets all state for every button.
    pub fn reset_all(&mut self) {
        self.pressed.clear();
        self.just_pressed.clear();
        self.just_released.clear();
    }

    /// Retires this frame's edges while leaving the held set intact. Call once
    /// per frame after input has been consumed.
    pub fn clear(&mut self) {
        self.just_pressed.clear();
        self.just_released.clear();
    }

    /// Iterates the currently held buttons in deterministic order.
    pub fn get_pressed(&self) -> btree_set::Iter<'_, T> {
        self.pressed.iter()
    }

    /// Iterates the buttons pressed this frame in deterministic order.
    pub fn get_just_pressed(&self) -> btree_set::Iter<'_, T> {
        self.just_pressed.iter()
    }

    /// Iterates the buttons released this frame in deterministic order.
    pub fn get_just_released(&self) -> btree_set::Iter<'_, T> {
        self.just_released.iter()
    }
}
