//! Volumetric cloud / atmospheric-volume subsystem — shader-twin plumbing.
//!
//! The CPU golden for this subsystem lives in
//! [`prism_render_architecture::volumetric`] (a zero-third-party, `no_std`
//! contract crate: the whole §14 file set — noise / modeling / weather /
//! raymarch / scatter / multiscatter / avsm / shadow / atmosphere / spectral /
//! storm / coupling / fog / `cloud_lod` / temporal / reference / math / budget —
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
//! `volumetrics.wesl` / `ssgi.wesl`), so the compile test also guards the
//! ported cloud math — Perlin-Worley noise, the coverage/type/height modeling
//! remaps, Henyey-Greenstein + Draine phase, Beer-Lambert transmittance, the
//! multiple-scattering octave sum and the temporal reprojection clamp — against
//! drift from its CPU golden twin.
//!
//! Mirroring the sibling cloth / water compute subsystems, the module is split
//! into cohesive files rather than one large module:
//!
//! * [`abi`] — the `#[repr(C)]` host mirrors of the eight per-dispatch
//!   `var<immediate>` push-constant blocks and of every resident buffer
//!   element (density cache / weather map / ray-march target / reprojection
//!   history / `AVSM` cloud-shadow map / multiple-scatter `LUT`), plus the
//!   `size_of` contract tests that pin each record to the golden buffer strides
//!   exported by [`prism_render_architecture::volumetric::gpu::buffers`] and the
//!   workgroup tiles the `gpu::kernels` descriptors launch, so a layout drift
//!   fails the build.
//!
//! The remaining slices add the pipeline table (keyed by the architecture
//! crate's `VolumetricKernel`), the per-view resident bind groups, the `Core3d`
//! graph node that records the eight dispatches in kernel order, and the plugin
//! that embeds the shader and installs the pipelines and graph node.

mod abi;
mod pipeline;
mod resources;
mod settings;

#[cfg(test)]
mod shader_tests;
