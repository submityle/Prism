//! Single-backend `GPU` contracts.
//!
//! Prism runs on exactly one backend: the portable **wgpu** runtime, which
//! itself covers Metal, `Vulkan`, `D3D12`, and `WebGPU`. There is no second,
//! "native" backend to escalate to. Everything the renderer needs is expressed
//! as wgpu capabilities: the guaranteed `WebGPU` baseline plus opt-in wgpu
//! extension features (bindless, mesh shading, ray query, multi-draw indirect
//! count, ray tracing). wgpu maps those extensions onto the underlying
//! `Vulkan`/`Metal`/`D3D12` extensions; capabilities an adapter cannot provide
//! are simply unavailable and drive a fallback path, not a backend switch.
//!
//! This module owns the deterministic, device-free contracts around that single
//! backend:
//!
//! * [`capability`] — the [`Capability`] bitset, the [`CapabilitySet`] set math,
//!   the always-present [`CapabilitySet::WEBGPU_BASELINE`], and
//!   [`FeatureRequirement`] validation that reports exactly which capabilities a
//!   feature is missing.
//! * [`negotiation`] — [`negotiate_features`], which intersects the capabilities
//!   an adapter advertises with the extension features the app requests and
//!   yields the [`EnabledFeatures`] the device will run with, plus the requested
//!   extensions the adapter could not provide.
//!
//! Device ownership, resource creation, command recording, and submission live
//! in the concrete `prism_render_driver` runtime. This module intentionally
//! exposes only the device-independent capability contracts used by render
//! planning and feature negotiation.

pub mod capability;
pub mod negotiation;

pub use capability::{Capability, CapabilitySet, FeatureRequirement, MissingCapabilities};
pub use negotiation::{negotiate_features, EnabledFeatures};
