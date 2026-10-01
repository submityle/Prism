#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

//! Loom automatic motion: the implicit animation layer for Prism's `prism_ui`.
//!
//! This crate turns *declarative state changes* into motion automatically, in
//! the spirit of `Framer` Motion, Flutter's `Hero` widget and `SwiftUI`'s
//! `matchedGeometryEffect`. It builds on the deterministic primitives in
//! [`prism_ui_anim`] and the pure-data style values in [`prism_ui_style`], and
//! like them it is `no_std + alloc` friendly and free of `unsafe` code.
//!
//! It provides three layers:
//!
//! * [`TransitionTracker`] — **implicit style transitions**: remember the last
//!   value of each [`StyleProp`] and tween to new values automatically.
//! * [`FlipAnimation`] / [`FlipState`] — **`FLIP` layout animation**: move an
//!   element smoothly from an old layout rectangle to a new one.
//! * [`SharedElementTransition`] — **shared-element (Hero) transitions**: fly a
//!   keyed element from a source rectangle to a destination rectangle.
//!
//! The shared geometry types [`Rect`] and [`Transform`] implement
//! [`prism_ui_anim::Lerp`], so they drop straight into a
//! [`prism_ui_anim::Tween`].
//!
//! # Example
//!
//! ```
//! use prism_ui_motion::{FlipAnimation, Rect, TransitionSpec, TransitionTracker};
//! use prism_ui_anim::Easing;
//! use prism_ui_style::{StyleProp, StyleValue};
//!
//! // Implicit transition: width animates whenever its value changes.
//! let mut tracker = TransitionTracker::new()
//!     .with_transition(StyleProp::Width, TransitionSpec::linear(1.0));
//! tracker.observe(StyleProp::Width, StyleValue::px(0.0)); // first value: no motion
//! tracker.observe(StyleProp::Width, StyleValue::px(100.0)); // change: starts a tween
//! tracker.step(0.5); // advance half a second
//! assert_eq!(tracker.value(StyleProp::Width), Some(StyleValue::px(50.0)));
//!
//! // FLIP: an element that jumped 200px to the right glides back into place.
//! let prev = Rect::new(0.0, 0.0, 100.0, 100.0);
//! let current = Rect::new(200.0, 0.0, 100.0, 100.0);
//! let flip = FlipAnimation::new(prev, current, 1.0, Easing::Linear);
//! assert_eq!(flip.sample(0.0).tx, -200.0); // starts visually at the old spot
//! assert!(flip.sample(1.0).is_identity()); // ends at the new spot
//! ```

extern crate alloc;

pub mod flip;
pub mod geometry;
pub mod shared_element;
pub mod transition;
pub mod value_anim;

pub use flip::{FlipAnimation, FlipState};
pub use geometry::{Rect, Transform};
pub use shared_element::{FallbackRole, SharedElementTransition, SharedPair};
pub use transition::{PropertyTransition, TransitionSpec, TransitionTracker};
pub use value_anim::{interpolate, interpolate_with, is_continuous, AnimatableValue};

// Re-export the style property type used throughout the public API so callers
// do not need an explicit `prism_ui_style` dependency for common usage.
pub use prism_ui_style::StyleProp;
