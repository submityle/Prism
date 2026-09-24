//! Rigid-body state storage.
//!
//! The [`handle`] module defines the generational [`handle::BodyHandle`], the
//! [`body`] module defines the body description and mass property types, and
//! the [`storage`] module defines the Structure-of-Arrays [`storage::BodyStorage`],
//! and the [`view`] module exposes a borrowing solver view over its columns.

pub mod body;
pub mod handle;
pub mod storage;
pub mod view;
