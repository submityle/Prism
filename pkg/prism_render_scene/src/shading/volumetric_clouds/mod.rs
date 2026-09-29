//! Volumetric cloud / atmospheric-volume subsystem — shader-twin plumbing.
//!
//! The CPU golden for this subsystem lives in
//! [`prism_render_architecture::volumetric`] (a zero-third-party, `no_std`
//! contract crate: the whole §14 file set — noise / modeling / weather /
//! raymarch / scatter / multiscatter / avsm / shadow / atmosphere / spectral /
//! storm / coupling / fog / cloud_lod / temporal / reference / math / budget —
//! plus the `gpu` compute-graph scaffold that names each on-device kernel and
//! its dispatch domain). Because that crate forbids third-party dependencies it
//! cannot host a `bevy_shader` compilation harness, so the shader twin and its
//! GPU-less compile verification land here in `prism_render_scene`, alongside
//! the froxel [`super::volumetrics`] fog subsystem it shares the atmosphere LUT
//! layer with.
//!
//! `shaders/volumetric_clouds.wesl` is the on-device mirror of the eight
//! compute kernels the architecture crate's `gpu::kernels` contract names:
//!
//! * `volumetric_weather_advect` — semi-Lagrangian weather-map advection.
//! * `volumetric_noise_bake` — Perlin-Worley + curl detail volume bake.
//! * `volumetric_modeling` — coverage/type/height gradient + erosion compose.
//! * `volumetric_multiscatter_lut_bake` — multiple-scattering energy LUT bake.
//! * `volumetric_raymarch` — adaptive-step, empty-space-skipping march.
//! * `volumetric_scatter_resolve` — HG double-lobe + powder + octave resolve.
//! * `volumetric_shadow_march` — light-space cloud-shadow / AVSM march.
//! * `volumetric_upsample` — temporal reprojection + history-clamp upsample.
//!
//! The kernel is self-contained (no intra-crate `import`s, matching
//! `volumetrics.wesl` / `ssgi.wesl`), so the compile test below also guards the
//! ported cloud math — Perlin-Worley noise, the coverage/type/height modeling
//! remaps, Henyey-Greenstein + Draine phase, Beer-Lambert transmittance, the
//! multiple-scattering octave sum and the temporal reprojection clamp — against
//! drift from its CPU golden twin. There is no `#[repr(C)]` ABI to pin here:
//! each entry carries its own `var<immediate>` param block and the compile test
//! does not lock immediate sizes, so this module is test-only plumbing.

#[cfg(test)]
mod shader_tests;
