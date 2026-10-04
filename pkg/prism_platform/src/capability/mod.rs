//! Platform capability database and declarative degradation (design doc §24.6).
//!
//! This module turns the coarse [`crate::platform::PlatformCaps`] bits into a
//! structured, queryable **capability → support-level → fallback-path** matrix
//! so upper layers get a correct degradation plan from a single query instead
//! of re-deriving `cfg!(...)` logic at every call site:
//!
//! - [`catalog`]: the closed, ordered set of [`Capability`] values plus static
//!   metadata ([`Category`], stable keys, human summaries).
//! - [`support`]: the three-way [`SupportLevel`] (native / degraded /
//!   unsupported) and the [`Support`] status carrying a fallback/`reason`.
//! - [`database`]: [`CapabilityDatabase`], the resolved matrix for one platform,
//!   derived from the OS family and the capability flags; `const`-constructible
//!   and fully testable for every platform, not just the test host.
//! - [`require`]: the declarative `require(cap).select(native, fallback)` API —
//!   "use huge pages or fall back to normal allocation" — evaluating only the
//!   branch actually taken.
//! - [`report`]: a deterministic, byte-stable startup report of which AAA paths
//!   run native / degraded / unavailable (feeds `prism_diagnostic`).
//!
//! The whole module is pure `core` + `alloc` (the report builds a `String`): it
//! performs no I/O and probes nothing at runtime, so it is available in every
//! build configuration, including `no_std`.

pub mod catalog;
pub mod database;
pub mod report;
pub mod require;
pub mod support;

pub use catalog::{Capability, Category};
pub use database::CapabilityDatabase;
pub use require::{Requirement, Selection};
pub use support::{Support, SupportLevel};
