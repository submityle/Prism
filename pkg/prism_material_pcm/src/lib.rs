//! PCM — the Prism Capability Model: one registry for every material-pipeline
//! extension point (design doc §5).
//!
//! # Why a single registry
//!
//! Before PCM the pipeline had **nine** separate, mostly-closed extension
//! mechanisms — add a lobe (edit a bitmask enum), add a parameter channel
//! (hand-edit the ABI header), add a shading response mode (edit a closed
//! four-value enum), a transfer remap, a cache aux, a data provider, a
//! downgrade policy, an author node, a composite pass — each with its own
//! hard-coded ceiling and its own special-case test. That is the same disease
//! the material refactor criticises elsewhere: two unrelated mechanisms, one
//! per "fixed line with a hard cap" and one per "unfixed line that quietly
//! folds in". PCM replaces all nine with **one declarative capability
//! contract**; a lobe (`kind = Lobe`) becomes just one kind among nine.
//!
//! # The performance red line (§5.2)
//!
//! Every governance machine here — namespace resolution, capability
//! negotiation, `used`/`required` sets, bit/region allocation — runs at
//! **bake / load time and then evaporates**. The runtime gets only the already
//! monomorphised WESL plus a fixed-width `u64` digest in the material header.
//! There is no dispatch, no interpretation and no registry lookup on the hot
//! path; this crate is deliberately *not* a plugin VM.
//!
//! # What lives here
//!
//! * [`Namespace`] / [`Region`] / [`Stability`] / [`Kind`] / [`Capability`] —
//!   the vocabulary.
//! * [`ExtensionManifest`] — one extension point's declaration, splitting its
//!   **immutable identity** (`uid`, `registry_slot`, `origin`) from its
//!   **mutable attribution** (`namespace`, `vendor`, `stability`) plus the
//!   derived display `id`.
//! * [`ExtensionRegistry`] — the table, its namespace-partitioned slot
//!   allocator, the promotion state machine and the §4.5 field predicates made
//!   machine-checkable.
//! * [`CapabilitySet`] / [`negotiate`] — the load-time capability handshake
//!   (§5.3): a missing `required` capability is a hard error, a missing `used`
//!   one is a graceful projection back to the variant without it.
//! * [`ExtensionSet`] — the fixed-width `used`/`required` digest carried by the
//!   material header ABI.

#![forbid(unsafe_code)]

extern crate alloc;

pub mod builtin;
pub mod capability;
pub mod extension_set;
pub mod manifest;
pub mod namespace;
pub mod negotiate;
pub mod registry;

#[cfg(test)]
mod tests;

pub use capability::{Capability, CapabilitySet};
pub use extension_set::{ExtensionSet, ExtensionSetError};
pub use manifest::{ExtensionId, ExtensionManifest, ManifestError};
pub use namespace::{Kind, Namespace, Region, Stability};
pub use negotiate::{Negotiation, NegotiationError, UnsupportedExtension};
pub use registry::{ExtensionRegistry, RegistryError, RegistryPoolExhausted};

/// Number of registry slots representable in one material header digest.
///
/// This is the width of the hot-path address space, **not** a cap on the
/// content model: the authored graph may add any number of `Layer`/`Mix`
/// nodes, lobes and channels, and the manifest/payload side is unbounded. Only
/// the *fixed-width header digest* is bounded, and exhausting a namespace
/// region is an explicit [`RegistryPoolExhausted`] error rather than a silent
/// fold-in — see [`ExtensionRegistry::register`].
pub const HEADER_DIGEST_SLOTS: u32 = 64;

/// Words in the material header reserved for the extension digest
/// (`extensions_used_low/high`, `extensions_required_low/high`).
pub const HEADER_DIGEST_WORDS: u32 = 4;
