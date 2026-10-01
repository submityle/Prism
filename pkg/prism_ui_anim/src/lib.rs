//! `prism_ui_anim` — easing, spring physics and timeline animation for
//! Prism's Loom UI/scene notation.
//!
//! This crate provides small, deterministic, engine-agnostic animation
//! primitives:
//!
//! - [`Lerp`]: linear interpolation for scalars, arrays and tuples.
//! - [`Easing`]: CSS-style easing curves, including a general
//!   [`Easing::CubicBezier`] solved with Newton/bisection and
//!   [`Easing::Steps`].
//! - [`Spring`] / [`SpringState`]: an exact analytic damped-spring integrator
//!   that converges for under-, critically- and over-damped regimes.
//! - [`Timeline`] / [`Keyframe`]: a sorted keyframe timeline with per-segment
//!   easing.
//! - [`Tween`] / [`Transition`]: a time-driven tween and an enter/exit
//!   transition helper.
//!
//! The crate is `no_std`-capable (it only relies on `core` and `alloc`) and
//! contains no `unsafe` code. Transcendental math (`sin`, `cos`, `exp`,
//! `sqrt`) uses the standard library when the `std` feature is enabled and
//! falls back to internal range-reduced polynomial approximations otherwise,
//! so results are available with and without default features.
#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

extern crate alloc;

pub mod driver;
pub mod easing;
pub mod lerp;
pub(crate) mod math;
pub mod spring;
pub mod timeline;

pub use driver::{Transition, TransitionPhase, Tween};
pub use easing::{Easing, StepPosition};
pub use lerp::Lerp;
pub use spring::{Spring, SpringState};
pub use timeline::{Keyframe, Timeline};
