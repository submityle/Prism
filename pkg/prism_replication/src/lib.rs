//! Prism's authority / replication kernel (design
//! `docs/prism_network_design_zh.md`).
//!
//! This crate is the **cross-cutting multiplayer layer**: it sits above
//! `prism_scene` (content truth), the persist override-delta log, and
//! `prism_ecs`/`prism_time`, and carries authoritative change between peers. The
//! full design spans transport (reliable UDP/QUIC), replication strategy, area-
//! of-interest, prediction/rollback, and server meshing — each a replaceable
//! provider. This crate grows from the **pure, transport-independent core** the
//! rest is built on, because that core is exactly the part that can be verified
//! on one machine without a network:
//!
//! - [`sequence`] — deterministic command ordering: a Lamport logical clock plus
//!   a per-source in-order delivery window, so a reliable-ordered or unreliable
//!   transport delivering commands out of order / twice still yields one agreed,
//!   gap-free order at every peer (design §9 / §13).
//! - [`bitpack`] / [`quantize`] — the delta layer's wire primitives: a
//!   bit-width-addressable stream and a uniform scalar quantizer, so a replicated
//!   field costs the bits it needs (design §15).
//! - [`delta`] — baseline-relative field deltas: a schema, a changed-field mask,
//!   and only the changed values, with the baseline revision checked on apply —
//!   plus full-record snapshots, the seed a late joiner or a resyncing peer
//!   applies before any delta.
//! - [`snapshot`] — the bounded baseline ring that applies those deltas forward
//!   into new baselines and serves any retained revision for a late-join or a
//!   resync.
//! - [`aoi`] — interest management: a deterministic uniform spatial grid and the
//!   enter/leave edges between two interest views (design §8).
//! - [`prediction`] — client-side prediction and reconciliation: apply a local
//!   input at once, then roll back to authority and replay the un-acknowledged
//!   inputs when the correction arrives (design §10).
//! - [`transport`] — the transport seam and a deterministic in-memory
//!   [`LoopbackLink`](transport::LoopbackLink) with latency, jitter and loss, so
//!   the sequencing layer can be exercised against a real reordering/lossy channel
//!   without a socket (design §6).
//! - [`frame`] — LEB128 length-prefixed message framing with an incremental
//!   decoder, the boundary a stream transport needs (design §6).
//! - [`session`] — the **wiring**: [`ServerSession`](session::ServerSession)
//!   (authority: seed snapshots, step deltas, lagging-client catch-up) and
//!   [`ClientSession`](session::ClientSession) (prediction fed by inbound frames),
//!   exchanging framed bytes over any transport.
//!
//! With the `app` feature, [`plugin`] wires a client session into a `prism_app`
//! `App` as a plugin (design §25.2).
//!
//! Later blocks (state-snapshot delta encoding, interest management, prediction
//! and reconciliation, the transport providers) build on this and are added as
//! they land; nothing here is a stub.
//!
//! The core is `no_std` + `alloc`.
#![no_std]

extern crate alloc;

#[cfg(feature = "std")]
extern crate std;

pub mod sequence;

pub mod aoi;

pub mod bitpack;

pub mod delta;

pub mod quantize;

pub mod snapshot;

pub mod prediction;

pub mod transport;

pub mod frame;

pub mod session;

#[cfg(feature = "app")]
pub mod plugin;

#[cfg(feature = "std")]
pub mod udp;
