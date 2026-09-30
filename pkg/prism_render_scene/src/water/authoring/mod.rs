//! High-level water authoring presets.
//!
//! [`WaterBody`](crate::water::body::WaterBody) is the low-level `#[repr(C)]`
//! mirror the shaders read directly: filling it by hand means drawing an entire
//! `Tessendorf` initial spectrum, seeding a `FLIP`/`APIC` particle lattice,
//! sizing a spatial hash, or solving a `CFL` bound before a single cell steps.
//! That is the right contract for the solver, but far too much ceremony for a
//! game that just wants "a stormy ocean", "a splashing pool" or "a rippling
//! pond" in the world. This is the single authoring layer that turns a small,
//! art-directable preset into a fully live body, mirroring the one-line
//! ocean / lake / river actors of `UE5` Water and `Crest`.
//!
//! Each preset lives in its own single-responsibility file so no one module
//! grows unwieldy, and none of them is a stub: every one consumes the
//! deterministic, float-audited helpers of the dependency-free
//! [`prism_render_architecture::water`] core to derive physically bounded state
//! (a real initial spectrum, a `CFL`-bounded timestep, a `Q <= 1` `Gerstner`
//! fan, a valid spatial-hash extent) and lights exactly the passes the golden
//! schedule expects. A degenerate preset (calm sea, empty pool) yields an
//! honest no-op body rather than a fabricated one.
//!
//! * [`coastline`] - a terrain-flooded coastline: a `Landscape` heightfield
//!   flooded under a sea level into a Shallow-Water basin with wind drift and
//!   coastal foam, mirroring the `UE5` Water `Landscape` blend.
//! * [`ocean`] - wind-driven spectral `IFFT` + analytic `Gerstner` swell.
//! * [`flip`] - a `FLIP`/`APIC` liquid volume seeded in a box, with screen-space
//!   surface reconstruction.
//! * [`lake`] - a polygon-bounded lake: a closed `Catmull-Rom` shoreline
//!   rasterized into a Shallow-Water basin with wind-driven surface drift and
//!   optional stream-mouth inflows, mirroring `UE5` Water lakes.
//! * [`pbf`] - a `Position-Based-Fluids` particle pool with its spatial hash.
//! * [`river`] - a spline-driven river: a `Catmull-Rom` centerline rasterized
//!   into a Shallow-Water flow field, mirroring `UE5` Water river splines.
//! * [`swe`] - a Shallow-Water height field stepped under a `CFL` bound.

mod coastline;
mod flip;
mod lake;
mod ocean;
mod pbf;
mod river;
mod swe;

pub use coastline::CoastlinePreset;
pub use flip::FlipPoolPreset;
pub use lake::{LakeInflow, LakePreset};
pub use ocean::OceanPreset;
pub use pbf::PbfPoolPreset;
pub use river::{RiverControlPoint, RiverPreset};
pub use swe::ShallowWaterPreset;
