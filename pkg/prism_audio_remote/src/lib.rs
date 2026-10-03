//! Live authoring and remote-tool API for the next-generation audio engine.
//!
//! This crate lets an editor or external tool connect to a running engine,
//! observe read-only telemetry, and issue a *whitelisted* set of write commands
//! (change a parameter, trigger an event, switch a state, swap a snapshot). All
//! writes are serialized onto the engine command ring; nothing here touches
//! real-time memory directly, so remote edits and in-game code edits share one
//! command path and therefore one deterministic, replayable semantics. The
//! remote surface is gated to development builds and is compiled out of release
//! builds to shrink the attack surface. There is no AI/ML.
//!
//! # Module map
//!
//! * [`transport`] -- the [`transport::AuthoringTransport`] trait plus an
//!   in-process channel implementation used for tests and same-process tools.
//! * [`command`] -- the whitelisted [`command::AuthoringCommand`] set and its
//!   validation against a capability allow-list.
//! * [`telemetry`] -- a read-only [`telemetry::TelemetryMirror`] that snapshots
//!   the engine telemetry ring for remote panels.
//! * [`session`] -- the [`session::RemoteSession`] state machine and capability
//!   negotiation that decides bandwidth and sampling.
//! * [`tuning`] -- [`tuning::LiveTuner`], smoothed live RTPC / bus-gain /
//!   attenuation adjustment with write-back to the authoring asset.
//! * [`auth`] -- the [`auth::Authenticator`] and command-whitelist security
//!   boundary.
//! * [`hotreload`] -- [`hotreload::HotReloadEvent`] descriptors for authoring
//!   data (event / container / bank / patch) reload requests.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements design section 38 (live authoring and remote tool API), aligned
//! with the Wwise Authoring API and FMOD Studio Live Update concepts. Commands
//! are routed through the command ring of design section 21; live tuning uses
//! the smoothed-parameter path of design section 7; telemetry mirrors the ring
//! of design section 26; hot reload drives the recompiled `ExecPlan` swap of
//! design sections 29 and 30. This crate carries only serializable contracts
//! and reuses `prism_audio_core` parameter types rather than re-implementing
//! them; it intentionally does not depend on the RT transport crate.
#![forbid(unsafe_code)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod auth;
pub mod command;
pub mod hotreload;
pub mod session;
pub mod telemetry;
pub mod transport;
pub mod tuning;
