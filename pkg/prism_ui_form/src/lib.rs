//! `prism_ui_form` — reactive form state and declarative validation for Loom.
//!
//! This crate layers a small, framework-agnostic form model on top of
//! [`prism_ui_reactive`]. Each field is backed by a reactive text signal, so
//! two-way binding is just reading and writing that signal, and validation is
//! expressed declaratively as an ordered list of [`Validator`]s per field.
//!
//! The pieces are:
//!
//! * [`Form`] — registers fields, reads/writes values, tracks `touched`/`dirty`
//!   state, and aggregates [`ValidationError`]s. [`Form::errors_memo`] exposes a
//!   reactive view that recomputes whenever any field changes.
//! * [`Validator`] — the rule trait, with ready-made rules [`required`],
//!   [`min_len`], [`max_len`], [`int_range`], [`pattern`], and [`custom`].
//! * [`FieldId`] — an opaque, cheaply clonable field key.
//!
//! # Example
//!
//! ```
//! use prism_ui_form::{Form, min_len, required};
//! use prism_ui_reactive::Runtime;
//!
//! let form = Form::new(Runtime::new());
//! form.register("name", "", vec![required(), min_len(3)]);
//!
//! // The field starts invalid: it is empty and too short.
//! assert!(!form.is_valid());
//! assert_eq!(form.first_error("name").unwrap().message, "This field is required.");
//!
//! // Writing a value marks the field dirty and revalidates it.
//! form.set("name", "Ada");
//! assert!(form.is_valid());
//! assert!(form.is_dirty("name"));
//! ```
#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

extern crate alloc;

mod error;
mod field;
mod form;
mod validator;

pub use error::ValidationError;
pub use field::{parse_i64, FieldId};
pub use form::Form;
pub use validator::{
    custom, int_range, max_len, min_len, pattern, required, BoxedValidator, Validator,
};
