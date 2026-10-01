//! `prism_ui_router` — a small reactive, client-side router for Prism's Loom
//! UI/scene notation.
//!
//! The router is layered on [`prism_ui_reactive`]: the current location lives in
//! a reactive signal, so route resolution, rendering, and side effects all
//! update automatically as the user navigates. It is `no_std`-friendly (it only
//! requires `alloc`) and does no floating-point work — routing is pure
//! integer/string matching.
//!
//! # Pieces
//!
//! * [`Location`] — a parsed path with segments, query parameters, and
//!   fragment.
//! * [`RoutePattern`] / [`RouteMatch`] — a compiled pattern (static segments,
//!   `:name` params, trailing `*name` wildcard) and the captures it produces.
//! * [`RouteTable`] / [`RouteId`] — an ordered, first-match-wins set of routes.
//! * [`Router`] — ties a [`RouteTable`] to a reactive `Runtime`, exposing the
//!   current location, a resolved-route [`Memo`](prism_ui_reactive::Memo), and a
//!   back/forward history stack.
//!
//! # Example
//!
//! ```
//! use prism_ui_reactive::Runtime;
//! use prism_ui_router::{RouteId, RouteTable, Router};
//!
//! let rt = Runtime::new();
//! let table = RouteTable::new()
//!     .route("/", RouteId::new(0))
//!     .route("/users/:id", RouteId::new(1));
//!
//! let router = Router::new(&rt, "/", table);
//! let current = router.current_match();
//!
//! // The initial location resolves to the root route.
//! assert_eq!(current.get().map(|(id, _)| id), Some(RouteId::new(0)));
//!
//! // Navigating updates the location and the resolved-route memo.
//! router.navigate("/users/42");
//! let (id, matched) = current.get().expect("user route matches");
//! assert_eq!(id, RouteId::new(1));
//! assert_eq!(matched.param("id"), Some("42"));
//!
//! // History supports back/forward navigation.
//! assert!(router.back());
//! assert_eq!(router.location().path(), "/");
//! ```
#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

extern crate alloc;

mod matcher;
mod path;
mod route;
mod router;

pub use matcher::{RouteId, RouteTable};
pub use path::Location;
pub use route::{RouteMatch, RoutePattern};
pub use router::Router;
