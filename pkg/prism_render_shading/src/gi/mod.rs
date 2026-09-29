//! Global-illumination CPU golden reference implementations.
//!
//! Houses backend-neutral, GPU-free numerical references for the render
//! engine's GI passes.  The [`world_space`] submodule implements a Lumen-style
//! screen-probe + world-space radiance-cache pipeline.

pub mod world_space;
