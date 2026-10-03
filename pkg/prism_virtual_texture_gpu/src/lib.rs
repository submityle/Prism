//! Optional `wgpu` compute backend for Prism's sparse/runtime virtual texturing.
//!
//! Runtime virtual texturing (RVT/SVT) streams fixed-size tiles (pages) of a
//! very large virtual texture into a bounded physical pool and resolves a
//! sampled page coordinate to its physical slot through an *indirection page
//! table*. The deterministic contract for that pipeline — residency, feedback
//! prioritisation, scheduling, physical-slot allocation, and the flat,
//! binary-searchable page table — lives device-free in
//! [`prism_render_architecture::texture_streaming`].
//!
//! This crate is the real-device half of the final, hottest step: a compute
//! kernel that binary-searches the page table on the `GPU` exactly as the CPU
//! golden
//! [`GpuPageTable::lookup`](prism_render_architecture::texture_streaming::GpuPageTable::lookup)
//! does. Device acquisition is best-effort via [`GpuContext::try_headless`], so
//! the parity tests exercise a real Apple `M`-series (or other native) `GPU`
//! when present and skip cleanly otherwise.
//!
//! # Exact parity
//!
//! The lookup is **integer-only** — the compare words are packed so an unsigned
//! lexicographic compare of `(w0, w1, w2)` orders identically to the golden
//! `TexturePageKey`, and the binary search uses only `u32` comparisons and
//! integer midpoint arithmetic. There is no floating point anywhere, so the
//! device result equals the golden result bit-for-bit and the parity tests
//! assert exact equality rather than a tolerance.
//!
//! Provenance: id Tech `MegaTexture` → runtime virtual texturing (RVT/SVT) with a
//! DX12 Sampler-Feedback-style demand signal. Classical, data-oblivious integer
//! search; no neural, learned, or data-driven components. No Unreal Engine
//! source or derived code.

extern crate alloc;

pub mod buffer;
pub mod context;
pub mod feedback;
pub mod lookup;

pub use context::GpuContext;
pub use feedback::{CellMap, GpuFeedbackDecode, REQ_NONE};
pub use lookup::{GpuPageLookup, MISS};
