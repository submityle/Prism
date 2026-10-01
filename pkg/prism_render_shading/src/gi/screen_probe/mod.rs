//! Screen-probe GI sampling and reuse (CPU golden).
//!
//! Screen-space probes gather indirect light at a sparse set of screen
//! locations and reuse those samples aggressively across space and time to hit
//! a real-time ray budget.  This module root collects the backend-neutral,
//! GPU-free numerical references for that pipeline:
//!
//! * [`restir`] — reservoir-based spatiotemporal resampling (`ReSTIR` GI /
//!   `GRIS`): the [`Reservoir`] container, streaming RIS updates, confidence-
//!   weighted spatiotemporal merging, and unbiased contribution weights, plus
//!   the [`GiSample`] payload and its scalar resampling [`target_function`].
//! * [`guided_sampling`] — deterministic path-guided importance sampling: a
//!   piecewise-constant, equal-solid-angle [`GuidingDistribution`] with
//!   consistent `sample`/`pdf` and a cosine-lobe MIS mixture.
//!
//! Every item is a deterministic pure function with unit tests; its output is
//! the numerical reference the WESL/GPU twin passes must reproduce.

pub mod guided_sampling;
pub mod restir;

pub use guided_sampling::GuidingDistribution;
pub use restir::{balance_heuristic, geometric_term, luminance, target_function, GiSample, Reservoir};
