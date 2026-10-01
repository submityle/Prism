//! `prism_ui_inspector` — element-tree query and op-trace perf inspection for Loom.
//!
//! The inspector is the read-only, analysis half of Loom's tooling. Where
//! [`prism_ui_devtools`] turns a live [`prism_ui::Element`] tree into an owned
//! [`TreeSnapshot`], this crate lets you interrogate that snapshot and the
//! backend-op stream a flush produced:
//!
//! * [`path`] addresses nodes positionally. A [`NodePath`] renders as `/0/2/1`,
//!   parses back from the same syntax, and [`paths_of`] enumerates every node in
//!   deterministic depth-first order.
//! * [`query`] filters the tree. A [`Query`] composes kind, class, and
//!   substring-text conditions and returns matches paired with their paths.
//! * [`perf`] aggregates measurements. [`PerfReport`] tallies an [`OpTrace`]
//!   into per-category counts plus integer *churn* metrics, and [`TreeMetrics`]
//!   summarises a snapshot's shape.
//!
//! Everything is `no_std`-friendly (uses `alloc`), avoids floating point in its
//! metrics, and produces deterministic output suited to golden tests.
//!
//! # Example
//!
//! ```
//! use prism_ui::layout::{AvailableSpace, Size};
//! use prism_ui::{Element, RecordingBackend, Ui};
//! use prism_ui_devtools::{snapshot, OpTrace};
//! use prism_ui_inspector::{NodePath, PerfReport, Query};
//!
//! // Build a small view with a classed card wrapping some text.
//! let view = Element::box_()
//!     .class("app")
//!     .child(Element::box_().class("card").child(Element::text("hello")))
//!     .child(Element::text("world"));
//!
//! // Snapshot it and query for the single `.card` box.
//! let snap = snapshot(&view);
//! let hits = Query::new().kind("Box").with_class("card").find_all(&snap);
//! assert_eq!(hits.len(), 1);
//! assert_eq!(hits[0].0, "/0".parse::<NodePath>().unwrap());
//!
//! // Mount it through the recording backend and aggregate the op trace.
//! let mut ui = Ui::new(RecordingBackend::new());
//! ui.mount(&view);
//! ui.compute_layout(Size::new(
//!     AvailableSpace::Definite(800.0),
//!     AvailableSpace::Definite(600.0),
//! ));
//! let report = PerfReport::from_trace(&OpTrace::from_ops(ui.backend().ops()));
//! assert_eq!(report.creates, snap.node_count());
//! assert_eq!(report.set_texts, 2);
//! assert_eq!(report.total, report.churn() + report.set_texts + report.set_layouts + report.set_paints);
//! ```
#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

extern crate alloc;

pub mod path;
pub mod perf;
pub mod query;

pub use path::{paths_of, resolve, resolve_in_node, NodePath, ParsePathError};
pub use perf::{PerfReport, TreeMetrics};
pub use query::{find_by_class, find_by_kind, Query};
