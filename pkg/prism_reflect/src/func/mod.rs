//! Function reflection (design §12, §22 — milestone **M5**).
//!
//! This module lets free functions and methods be registered into a
//! [`FunctionRegistry`] and called **by name** with a list of reflected
//! arguments, returning a reflected result. It is the script-bridge and
//! editor-action substrate: a caller that only knows a name and some
//! `&dyn Reflect` values can invoke native Rust code without per-function FFI
//! glue.
//!
//! # Safety of type erasure
//! The crate is `#![forbid(unsafe_code)]`, and function reflection keeps that
//! promise. Arguments are not transmuted: each one is rebuilt with
//! [`FromReflect`](crate::FromReflect), and the call's arity is validated
//! before any argument is read. A wrong count or a wrong argument type yields a
//! typed [`FunctionError`] rather than undefined behaviour (design §23's
//! "函数反射安全" risk).
//!
//! ```
//! use prism_reflect::{ArgList, FunctionRegistry};
//!
//! fn add(a: i32, b: i32) -> i32 {
//!     a + b
//! }
//!
//! let mut functions = FunctionRegistry::new();
//! functions.register("add", add);
//!
//! let result = functions
//!     .call("add", ArgList::new().push(2_i32).push(3_i32))
//!     .expect("call add");
//! assert_eq!(result.downcast_ref::<i32>(), Some(&5));
//! ```

mod args;
mod call;
mod registry;

pub use args::ArgList;
pub use call::{DynamicFunction, FunctionError, FunctionInfo, IntoFunction};
pub use registry::FunctionRegistry;
