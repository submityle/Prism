#![cfg_attr(not(feature = "std"), no_std)]

//! Loom GPU-driven retained draw-stream backend (scaffold).
//!
//! Deepened design lives in `docs/prism_ui_loom_design_zh.md` §9.8. This crate
//! will provide a draw-command buffer, batching/instancing, SDF rounded-rect/
//! shadow/border, layer compositing and a headless reference backend, with an
//! optional `wgpu` backend validated against the headless twin.
//!
//! `unsafe` is confined to the optional `gpu` feature (wgpu buffer mapping); the
//! default build is safe.
#![cfg_attr(not(feature = "gpu"), forbid(unsafe_code))]
