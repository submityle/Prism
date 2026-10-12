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
//!   and only the changed values, with the baseline revision checked on apply.
//!
//! Later blocks (state-snapshot delta encoding, interest management, prediction
//! and reconciliation, the transport providers) build on this and are added as
//! they land; nothing here is a stub.
//!
//! The core is `no_std` + `alloc`.
#![no_std]

extern crate alloc;

pub mod sequence;

pub mod bitpack;

pub mod delta;

pub mod quantize;
