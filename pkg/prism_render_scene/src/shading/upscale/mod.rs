//! GPU temporal-upscale (FSR2 / UE-TSR-class) subsystem.
//!
//! The CPU golden twin lives in [`prism_render_shading::upscale`] and the
//! shaders in `shaders/upscale_reconstruct.wesl` and `shaders/upscale_rcas.wesl`;
//! this module is the render-world plumbing that runs them. Where TAA
//! ([`super::taa`]) accumulates a *same-resolution* jittered history to
//! anti-alias, temporal upscaling reconstructs a **higher**-resolution image
//! from a stream of *lower*-resolution render targets: the camera is jittered
//! over a long Halton sequence, each frame is drawn at `render_scale` of the
//! display resolution, and the accumulation resolves those samples onto the
//! full display grid. It is the modern alternative to brute-force
//! supersampling — the renderer shades far fewer pixels yet, integrated over
//! time, resolves detail close to native.
//!
//! Two compute passes, mirroring the golden pipeline:
//!
//! * [`shaders/upscale_reconstruct.wesl`] — the reconstruction + accumulation
//!   pass (golden `upscale::upscale_pixel` plus the Lanczos-2 reconstruction
//!   and the reproject / disocclusion tests): Lanczos-2 upsample, motion
//!   reproject, `YCoCg` neighbourhood clip, thin-feature lock and temporal
//!   accumulation blend.
//! * [`shaders/upscale_rcas.wesl`] — the RCAS finishing sharpen (golden
//!   `upscale::sharpen`) run over the resolved display image.
//!
//! Landing as cohesive files, following the rest of the shading pipeline:
//!
//! * [`abi`] — the two immediate blocks shared with the shaders, kept
//!   byte-for-byte in sync with the golden tunables and the `InvalidationMask`
//!   history state.
//! * [`settings`] — the [`settings::UpscaleSettings`] resource, the render-world
//!   mirror of the architecture [`prism_render_architecture::temporal_upscale`]
//!   contract that drives both passes.
//!
//! The remaining slices (the persistent display-resolution history, the two
//! compute pipelines and their bind groups, and the `Core3d` graph nodes) land
//! with their first live consumer, so the committed ABI is never dead. This
//! module is the interconnect point the later graph wiring reads from.

// The ABI and settings are the committed contract the later graph wiring reads
// from; until that slice lands their only consumers are the layout / mapping
// unit tests, so the definitions are `dead_code`-allowed rather than deleted.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "committed GPU ABI whose only non-test consumer is the graph wiring landing in a later slice"
    )
)]
mod abi;
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "committed settings resource whose only non-test consumer is the graph wiring landing in a later slice"
    )
)]
mod settings;

#[cfg(test)]
mod shader_tests;

#[expect(
    unused_imports,
    reason = "re-exported for the graph wiring landing in a later slice"
)]
pub(crate) use abi::{
    GpuUpscaleRcasParams, GpuUpscaleReconstructParams, UPSCALE_WORKGROUP_SIZE,
};
#[expect(
    unused_imports,
    reason = "re-exported for the graph wiring landing in a later slice"
)]
pub(crate) use settings::UpscaleSettings;
