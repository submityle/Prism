//! `prism_ui_sdui` — Loom's server-driven UI (SDUI) sandbox.
//!
//! A server may ship a `.loom`-shaped description to drive A/B tests, live
//! operations slots or UI hot-fixes. Because that description is fully
//! untrusted, this crate interposes a strict security model between the wire
//! and the reconciler, mirroring the capability-whitelist approach popularized
//! by Airbnb's server-driven UI work and described in
//! `docs/prism_ui_loom_design_zh.md` §9.10.
//!
//! The pipeline has three stages:
//!
//! * **Version negotiation** — a [`VersionSet`] the client supports is matched
//!   against the document's declared [`Version`] by [`negotiate`], downgrading
//!   gracefully when the server is ahead and refusing incompatible families.
//! * **Capability sandbox** — a [`Sandbox`] built from a [`CapabilitySet`]
//!   rewrites the untrusted [`RemoteNode`] tree so it references only
//!   whitelisted kinds, style tokens and events. Rejected content is replaced
//!   by safe placeholders and never silently dropped: every rewrite yields a
//!   [`Diagnostic`].
//! * **Safe decoding** — [`decode`] turns the sanitized tree into a
//!   [`prism_ui::Element`], routing anything unrenderable to an
//!   error-boundary-style [`fallback_element`].
//!
//! [`render`] runs all three stages end to end.
//!
//! The crate is `no_std` (allocating via `alloc`), forbids `unsafe` code, and
//! performs no I/O: decoding bytes into a [`RemoteDocument`] is left to a future
//! layer, keeping this core pure, deterministic and panic-free on hostile
//! input.
//!
//! # Example
//!
//! ```
//! use prism_ui::ElementKind;
//! use prism_ui_sdui::{
//!     render, CapabilitySet, RemoteDocument, RemoteNode, Sandbox, Version, VersionSet,
//! };
//!
//! // The client speaks schema 2.0.
//! let mut client = VersionSet::new();
//! client.insert(Version::new(2, 0));
//!
//! // Only a small surface of capabilities is trusted.
//! let sandbox = Sandbox::new(
//!     CapabilitySet::new()
//!         .allow_kinds(["box", "text"])
//!         .allow_style("card"),
//! );
//!
//! // An untrusted document mixing allowed and disallowed content.
//! let doc = RemoteDocument::new(
//!     Version::new(2, 0),
//!     RemoteNode::new("box")
//!         .style("card")
//!         .child(RemoteNode::text("hello"))
//!         .child(RemoteNode::new("script")), // not whitelisted
//! );
//!
//! let outcome = render(&client, &sandbox, &doc);
//!
//! // The `script` node was replaced by a safe placeholder and recorded.
//! assert_eq!(outcome.diagnostics.len(), 1);
//! assert_eq!(outcome.element.kind(), &ElementKind::Box);
//! assert_eq!(outcome.element.child_elements()[0].text_content(), Some("hello"));
//! ```
#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

extern crate alloc;

pub mod capability;
pub mod decode;
pub mod pipeline;
pub mod sandbox;
pub mod schema;
pub mod version;

pub use capability::CapabilitySet;
pub use decode::{decode, fallback_element, DEFAULT_FALLBACK_MESSAGE, FALLBACK_CLASS};
pub use pipeline::{render, RenderOutcome};
pub use sandbox::{Diagnostic, DiagnosticKind, Sandbox, SanitizeResult, DEFAULT_MAX_DEPTH};
pub use schema::{RemoteDocument, RemoteNode, KIND_BOX, KIND_FALLBACK, KIND_TEXT};
pub use version::{negotiate, Resolution, Version, VersionSet};
