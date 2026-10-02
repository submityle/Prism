#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

//! Loom input system: hit testing, event dispatch, gestures, and focus.
//!
//! This crate turns raw pointer and keyboard events into high-level,
//! declarative interactions over `prism_ui`'s retained
//! [`Element`](prism_ui::Element) tree. It is engine-agnostic and `no_std`
//! friendly (allocation only), so it can run anywhere the retained tree does.
//!
//! The pipeline has four stages, one per module:
//!
//! * [`hit_test`] — walk a tree of layout rectangles to find the
//!   root-to-target path for a point, honouring `z-index` and
//!   `pointer-events`.
//! * [`dispatch`] — deliver an event along that path in capture, target, and
//!   bubble phases, with `stop_propagation` and `prevent_default`.
//! * [`gesture`] — recognize taps, long presses, drags, and pinches, resolving
//!   conflicts with a Flutter-style [`GestureArena`].
//! * [`focus`] — maintain a wrapping tab order with [`FocusRing`].
//!
//! Geometry types are re-used from `prism_ui`'s layout crate via [`geometry`].
//!
//! # Example
//!
//! Build a small hit-test tree, resolve the path for a point, then dispatch a
//! pointer event along it:
//!
//! ```
//! use prism_ui_input::{Dispatcher, HitNode, NodeId, Phase, PointerEvent, PointerId, PointerKind, hit_test};
//! use prism_ui_input::geometry::{Point, Rect, Size};
//! use std::rc::Rc;
//! use std::cell::RefCell;
//!
//! let root = HitNode::new(NodeId::new(1), Rect::new(Point::new(0.0, 0.0), Size::new(100.0, 100.0)))
//!     .child(HitNode::new(NodeId::new(2), Rect::new(Point::new(10.0, 10.0), Size::new(30.0, 30.0))));
//!
//! let path = hit_test(&root, Point::new(15.0, 15.0));
//! assert_eq!(path, vec![NodeId::new(1), NodeId::new(2)]);
//!
//! let hits = Rc::new(RefCell::new(0));
//! let mut dispatcher = Dispatcher::new();
//! let counter = Rc::clone(&hits);
//! dispatcher.on(NodeId::new(2), Phase::Target, move |_ctx, _event| {
//!     *counter.borrow_mut() += 1;
//! });
//!
//! let event = PointerEvent::new(PointerId::new(1), PointerKind::Down, Point::new(15.0, 15.0), 0);
//! dispatcher.dispatch(&path, &event);
//! assert_eq!(*hits.borrow(), 1);
//! ```

extern crate alloc;

pub mod dispatch;
pub mod event;
pub mod focus;
pub mod geometry;
pub mod gesture;
pub mod hit_test;

pub use dispatch::{DispatchOutcome, Dispatcher, EventContext};
pub use event::{
    KeyCode, KeyEvent, Modifiers, NodeId, Phase, PointerButton, PointerEvent, PointerId,
    PointerKind,
};
pub use focus::FocusRing;
pub use gesture::{
    DragAxis, DragRecognizer, GestureArena, GestureRecognizer, GestureState, LongPressRecognizer,
    PinchRecognizer, TapRecognizer,
};
pub use hit_test::{hit_test, HitNode, PointerEvents};
