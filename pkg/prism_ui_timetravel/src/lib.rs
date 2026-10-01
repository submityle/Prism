//! `prism_ui_timetravel` — time-travel debugging for the Loom UI system.
//!
//! This crate records a chronological timeline of UI *frames* and lets you
//! walk through it like an editor's undo history. Each [`Frame`] captures an
//! owned [`prism_ui_devtools::TreeSnapshot`] of the view at a moment in time,
//! plus an optional [`prism_ui_devtools::OpTrace`] describing the backend
//! mutations that produced it. A [`Timeline`] stores those frames and keeps a
//! movable cursor so you can step backward and forward, jump to any point, or
//! fork a new branch by recording after rewinding.
//!
//! On top of the timeline, [`Replay`] offers read-only inspection — textual
//! diffs between any two frames, a per-step change summary, and the ordered
//! labels — while [`Replayer`] is a forward cursor for driving a backend
//! through recorded history one frame at a time.
//!
//! All navigation is deterministic and free of floating-point math, so the
//! behavior is fully reproducible and simple to test.
//!
//! # Example
//!
//! ```
//! use prism_ui::Element;
//! use prism_ui_timetravel::{Frame, Timeline};
//!
//! // Two different views of the same widget.
//! let first = Element::box_().child(Element::text("hello"));
//! let second = Element::box_().child(Element::text("world"));
//!
//! // Record both as frames on a timeline.
//! let mut timeline = Timeline::new();
//! timeline.record(Frame::capture("v1", &first));
//! timeline.record(Frame::capture("v2", &second));
//!
//! assert_eq!(timeline.len(), 2);
//! assert_eq!(timeline.current().unwrap().label(), "v2");
//!
//! // Navigate backward and forward through history.
//! assert_eq!(timeline.back().unwrap().label(), "v1");
//! assert_eq!(timeline.forward().unwrap().label(), "v2");
//!
//! // Diff two frames; the report mentions both texts.
//! let report = timeline.replay().diff_between(0, 1).unwrap();
//! assert!(report.contains("hello"));
//! assert!(report.contains("world"));
//! ```
#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

extern crate alloc;

pub mod frame;
pub mod replay;
pub mod timeline;

pub use frame::Frame;
pub use replay::{Replay, ReplayStep, Replayer};
pub use timeline::Timeline;
