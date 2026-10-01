//! `prism_ui_overlay` — an overlay/portal manager for Prism's **Loom** UI.
//!
//! Modern UIs grow a second plane above the document: tooltips, popovers,
//! modal dialogs and toasts. They must stack in a predictable order, trap
//! keyboard focus, draw a backdrop when they are modal, and dismiss on escape
//! or scrim click. This crate provides that plane as plain data on top of
//! [`prism_ui::Element`] trees, so it stays engine-agnostic and reconciler
//! friendly.
//!
//! # Pieces
//!
//! * [`OverlayKind`] — the four overlay kinds and their fixed z-priority.
//! * [`OverlayId`] — an opaque, `Copy` handle to a pushed overlay.
//! * [`OverlayManager`] — the stack: push, dismiss, iterate in z-order, and
//!   [`render`](OverlayManager::render) a portal layer over a base view.
//! * [`FocusTrap`] — cyclic keyboard focus over an ordered set of keys.
//!
//! # Z-order
//!
//! Overlays are composited in ascending [`OverlayKind::z_priority`], ties
//! broken by insertion order. Modals sit at the base of the overlay plane
//! behind their backdrop; tooltips and toasts float on top so they are never
//! obscured.
//!
//! # Example
//!
//! ```
//! use prism_ui::Element;
//! use prism_ui_overlay::{OverlayKind, OverlayManager};
//!
//! let mut overlays = OverlayManager::new();
//! let dialog = overlays.push(OverlayKind::Modal, Element::text("Save changes?"));
//! assert_eq!(overlays.len(), 1);
//!
//! // Rendering composes the base view with a portal layer holding the modal
//! // and the backdrop it paints behind itself: base + portal.
//! let view = overlays.render(Element::box_());
//! assert_eq!(view.child_elements().len(), 2);
//! let portal = &view.child_elements()[1];
//! assert_eq!(portal.child_elements().len(), 2); // backdrop + modal layer
//!
//! // Escape dismisses the top-most dismissible overlay.
//! assert_eq!(overlays.on_escape(), Some(dialog));
//! assert!(overlays.is_empty());
//! ```
#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

extern crate alloc;

mod focus;
mod id;
mod kind;
mod manager;

pub use focus::FocusTrap;
pub use id::OverlayId;
pub use kind::OverlayKind;
pub use manager::{OverlayEntry, OverlayIter, OverlayManager};
