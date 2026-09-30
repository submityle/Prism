//! Lock-free real-time runtime bridge for Prism's next-generation audio engine.
//!
//! This crate is the L2/L4 seam of the Resonance architecture: it moves control
//! and data across the boundary between the ECS / task threads and the single
//! audio-callback thread without ever allocating, locking, or blocking on the
//! audio side. It sits on top of the device-agnostic DSP kernel in
//! [`prism_audio_core`] and below the device backends and ECS front-end.
//!
//! # Architecture
//!
//! - [`ring`] — bounded, lock-free single-payload rings. One instance is the
//!   *command ring* (task threads -> audio thread); another is the *telemetry
//!   ring* (audio thread -> observer thread).
//! - [`command`] — the `Copy`, pointer-free [`command::AudioCommand`] enum that
//!   travels over the command ring.
//! - [`telemetry`] — the `Copy` [`telemetry::TelemetryFrame`] snapshot emitted
//!   once per rendered block.
//! - [`epoch`] — deferred reclamation ([`epoch::RetireQueue`]) and the
//!   capacity-one graph hand-off ([`epoch::GraphHandoff`]) that together let the
//!   graph evolve at runtime while guaranteeing the audio thread never runs a
//!   destructor that frees memory.
//! - [`runtime`] — the [`runtime::AudioRuntime`] that drives it all on the audio
//!   thread, plus the [`runtime::AudioRuntimeClient`] used by task threads.
//!
//! # Real-time contract
//!
//! Everything reachable from [`runtime::AudioRuntime::process_block`] is
//! **allocation free, lock free, and panic free**. Rings hand payloads back to
//! the caller when full instead of growing; retired resources are dropped by a
//! [`epoch::Collector`] on a task thread; graphs are compiled off-thread and
//! published through the hand-off. The audio thread only ever *moves* owned
//! values, never allocates or frees them.
//!
//! # Concurrency model
//!
//! ```text
//!   task threads                         audio thread
//!  ┌───────────────┐   command ring    ┌────────────────────┐
//!  │ AudioRuntime  │ ────────────────▶ │   AudioRuntime      │
//!  │   Client      │   graph hand-off  │  - drain commands   │
//!  │  - send()     │ ────────────────▶ │  - swap graph       │
//!  │  - publish()  │                   │  - render block     │
//!  │  - recv()     │ ◀──────────────── │  - master gain      │
//!  └───────────────┘   telemetry ring  │  - emit telemetry   │
//!         ▲            retire queue     │  - retire old graph │
//!         │          ◀──────────────────┘                    │
//!  ┌──────┴────────┐                    └────────────────────┘
//!  │  Collector    │  drops retired resources off the audio thread
//!  └───────────────┘
//! ```
//!
//! # Provenance
//!
//! This crate is engine-agnostic and contains **no Unreal Engine, Unity,
//! Godot, Wwise, or FMOD source or derived code**. The lock-free rings are thin
//! wrappers around the safe, bounded [`crossbeam_queue::ArrayQueue`]; this crate
//! itself contains no `unsafe` code. The command-ring / telemetry-ring /
//! deferred-reclamation design follows standard, publicly documented real-time
//! audio engineering practice.
//!
//! [`prism_audio_core`]: https://docs.rs/prism_audio_core

extern crate alloc;

pub mod command;
pub mod epoch;
pub mod ring;
pub mod runtime;
pub mod telemetry;

pub use command::AudioCommand;
pub use epoch::{Collector, GraphConsumer, GraphHandoff, GraphProducer, RetireQueue, Retirer};
pub use ring::{RingConsumer, RingProducer, ring};
pub use runtime::{AudioRuntime, AudioRuntimeClient, AudioRuntimeConfig, runtime};
pub use telemetry::TelemetryFrame;
