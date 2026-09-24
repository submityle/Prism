//! Deferred game-logic commands and their queue.
//!
//! In a pipelined simulation the game logic and the physics step run at
//! different rates, so game logic must not mutate solver state mid-step. M2.5
//! routes every game-logic write (apply an impulse, set a velocity, teleport a
//! body) through a [`queue::CommandQueue`]. The pipeline drains and applies the
//! whole queue at a single, well-defined step boundary, which makes the effect
//! of a given set of inputs independent of *when* within the frame they were
//! issued — the property determinism and rollback rely on.
//!
//! - [`kind::PhysicsCommand`] is the closed set of deferred mutations.
//! - [`queue::CommandQueue`] buffers commands in issue order and applies them
//!   FIFO against a [`PhysicsWorld`].
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. The
//! deferred-command / double-buffered-input pattern is a standard, publicly
//! documented simulation technique implemented from scratch.

pub mod kind;
pub mod queue;

pub use kind::PhysicsCommand;
pub use queue::CommandQueue;
