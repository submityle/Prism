//! `prism_ui_snapshot` — snapshot regression testing for the Loom UI system.
//!
//! This crate is to Loom what Jest or Vitest snapshots and React Testing
//! Library are to the web: it turns a rendered view into deterministic,
//! human-readable text that can be committed as a golden file, then diffed on
//! every later run. It builds on [`prism_ui_devtools::snapshot`], which walks a
//! live [`prism_ui::Element`] tree into an owned [`prism_ui_devtools::TreeSnapshot`].
//!
//! # Capabilities
//!
//! * [`serialize_tree`] / [`parse_tree`] — a stable, reversible text encoding
//!   of a [`prism_ui_devtools::TreeSnapshot`].
//! * [`Snapshot`] / [`Comparison`] — golden comparison with a structured
//!   match or mismatch result.
//! * [`diff`] — a line-level, path-annotated diff for mismatches.
//! * [`capture_layout`] / [`serialize_layout`] / [`parse_layout`] — the same
//!   workflow for computed [`prism_ui_layout`] geometry.
//!
//! Everything is deterministic and `no_std`-friendly (requiring only `alloc`),
//! so snapshots are reproducible across platforms and runs.
//!
//! # Example
//!
//! ```
//! use prism_ui::Element;
//! use prism_ui_devtools::snapshot;
//! use prism_ui_snapshot::{parse_tree, serialize_tree, Snapshot};
//!
//! // Build a view and capture an owned snapshot of its element tree.
//! let view = Element::box_()
//!     .class("card")
//!     .child(Element::text("hello"));
//! let captured = snapshot(&view);
//!
//! // Serialize to deterministic golden text.
//! let golden = serialize_tree(&captured);
//! assert_eq!(
//!     golden,
//!     "kind=Box text=- classes=,card\n  kind=Text text=+hello classes=\n",
//! );
//!
//! // The text round-trips back to an equal snapshot.
//! let restored = parse_tree(&golden).expect("valid snapshot text");
//! assert_eq!(restored, captured);
//!
//! // Golden comparison succeeds against the stored text, and a changed view
//! // produces a mismatch with a readable diff.
//! let comparison = Snapshot::from_tree(&captured).assert_matches(&golden);
//! assert!(comparison.is_match());
//!
//! let changed = snapshot(&Element::box_().class("card").child(Element::text("world")));
//! let mismatch = Snapshot::from_tree(&changed).assert_matches(&golden);
//! assert!(mismatch.diff().unwrap().contains("+hello"));
//! ```

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

extern crate alloc;

mod escape;

pub mod compare;
pub mod diff;
pub mod layout_snapshot;
pub mod serialize;

pub use compare::{Comparison, Snapshot};
pub use diff::diff;
pub use layout_snapshot::{
    capture_layout, format_fixed, parse_layout, serialize_layout, LayoutParseError, LayoutQuery,
    LayoutSnapshotNode,
};
pub use serialize::{parse_tree, serialize_tree, ParseError};
