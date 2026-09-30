//! `GPU` compute-kernel contract scaffold for the water subsystem.
//!
//! The sibling modules in this crate resolve every per-frame decision on the
//! `CPU` (solver routing, budget arbitration, ocean binning, reconstruction and
//! caustics selection — see [`super::pipeline`]). The heavy numerical passes
//! themselves — the spectral inverse-`FFT`, the `Gerstner` displacement, the
//! Shallow-Water step, the `PBF` density solve, the `FLIP`/`APIC` particle-grid
//! transfer and pressure projection, the surface reconstruction, the caustics
//! projection, and the foam advection — run as `GPU` compute dispatches driven
//! by `WESL`-authored shaders.
//!
//! This module owns only the *contract* for those dispatches: which
//! bind-group resources each kernel binds, the workgroup tiling it is launched
//! with, the dispatch domain shape that sizes the launch, and the stable
//! `WESL` entry-point name the shader codegen emits. The `WESL` shader source,
//! its codegen, and the actual `GPU` execution are out of scope for this
//! `CPU`-verifiable contract layer and are documented as "pending the GPU
//! backend" where the descriptors anticipate them.
//!
//! Any frame-budget, bandwidth, or occupancy figure implied by a workgroup
//! size here starts life as a **design target** encoding the intended tiling
//! (a 8x8 screen tile, a 4x4x4 voxel brick, a 64-lane linear particle group).
//! The ocean spectrum `evolve`/`assemble` passes are no longer unmeasured: the
//! `prism_render_scene::water::gpu_bench` harness now times them on real
//! hardware via `TIMESTAMP_QUERY` (median of repeated dispatches), so those
//! tiles can be re-tuned against captured microseconds rather than estimates.
//! The remaining tiles stay design targets until an equivalent measured pass
//! lands for each and should still be validated against real Metal captures.
//!
//! Everything exposed here is a pure, deterministic function of a
//! [`WaterKernel`] tag — no `GPU` state, no floats, no allocation — so the
//! descriptor table can be asserted in `CPU` tests and diffed across builds.

/// The bind-group resource counts a kernel declares, grouped by binding class.
///
/// The renderer builds the concrete `GPU` bind-group layout from these counts;
/// the contract layer only needs the shape (how many of each class) to validate
/// that a kernel binds at least one resource and to size descriptor pools. The
/// exact buffers/textures and their formats are chosen by the `WESL` codegen and
/// are out of scope here.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct BindGroupLayout {
    /// Read/write `storage` buffers (particle pools, grid velocity/pressure,
    /// spectrum amplitude buffers, foam fields).
    pub storage_buffers: u32,
    /// Read-only `uniform` buffers (per-frame constants, solver parameters).
    pub uniform_buffers: u32,
    /// Writable `storage` textures (displacement/normal maps, reconstructed
    /// height, caustics render targets).
    pub storage_textures: u32,
    /// Sampled (read-only) textures bound with a sampler (source height fields,
    /// environment inputs).
    pub sampled_textures: u32,
}

impl BindGroupLayout {
    /// Total number of bindings across every class.
    ///
    /// Saturating so a pathological descriptor can never overflow the count the
    /// descriptor-pool sizing reads. A valid kernel always returns a positive
    /// total (it reads or writes at least one resource).
    #[must_use]
    pub fn total(self) -> u32 {
        self.storage_buffers
            .saturating_add(self.uniform_buffers)
            .saturating_add(self.storage_textures)
            .saturating_add(self.sampled_textures)
    }
}

/// The workgroup (thread-group) tiling a compute kernel is launched with.
///
/// This is the `@workgroup_size` the `WESL` shader declares; the dispatch
/// divides the [`DispatchDomain`] extent by this tile to get the group count.
/// The specific tile shapes are design targets (see the module doc), not
/// measured optima.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct WorkgroupSize {
    /// Threads along X.
    pub x: u32,
    /// Threads along Y (1 for 1D linear kernels).
    pub y: u32,
    /// Threads along Z (1 for 1D/2D kernels).
    pub z: u32,
}

impl WorkgroupSize {
    /// Number of invocations one workgroup contains (`x * y * z`).
    ///
    /// Saturating so an over-large declared tile can never overflow the linear
    /// invocation count [`linear_group_count`] divides against.
    #[must_use]
    pub fn invocations_per_group(self) -> u32 {
        self.x.saturating_mul(self.y).saturating_mul(self.z)
    }

    /// Returns `true` when every axis is non-zero, i.e. the tile can actually
    /// dispatch work. A zero axis would make [`invocations_per_group`] zero and
    /// the dispatch a no-op, which is always a contract bug.
    ///
    /// [`invocations_per_group`]: WorkgroupSize::invocations_per_group
    #[must_use]
    pub fn is_nonzero(self) -> bool {
        self.x != 0 && self.y != 0 && self.z != 0
    }
}

/// The shape of the index space a kernel is dispatched over.
///
/// The renderer reads the live extent for the matching shape from the frame
/// plan ([`super::pipeline`]) and turns it into a group count; the contract
/// layer only records which shape a kernel expects so the launch is sized
/// against the right magnitude (grid cells vs. particles vs. screen pixels).
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum DispatchDomain {
    /// A 2D grid: height-field / displacement texels (`SWE`, `Gerstner`,
    /// spectral cascades, screen-independent reconstruction tiles).
    Grid2d,
    /// A 3D grid: the `MAC` velocity/pressure voxels of a `FLIP`/`APIC` domain.
    Grid3d,
    /// A flat list of particles (`PBF` / `FLIP` particle passes).
    Particle,
    /// A full-screen pass: caustics projection and screen-space reconstruction
    /// iterate the framebuffer.
    Screen,
}

/// A fully described `GPU` compute dispatch for one water pass.
///
/// This is the immutable contract the renderer consumes to build and record a
/// dispatch: the resource layout, the workgroup tiling, and the domain shape.
/// It carries no `GPU` handles — those are resolved when the backend lands.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct KernelDescriptor {
    /// The kernel this descriptor belongs to.
    pub kernel: WaterKernel,
    /// The bind-group resource counts the kernel declares.
    pub layout: BindGroupLayout,
    /// The workgroup tiling the kernel is launched with.
    pub workgroup: WorkgroupSize,
    /// The index-space shape the dispatch is sized against.
    pub domain: DispatchDomain,
}

/// Every `GPU` compute kernel the water subsystem dispatches.
///
/// The passes map one-to-one onto the solver and rendering stages the `CPU`
/// contract layer schedules: the spectral and `Gerstner` displacement writers,
/// the `SWE` height-field step, the `PBF` density solve, the three `FLIP`/`APIC`
/// passes (`P2G` scatter, pressure projection, `G2P` gather), the surface
/// reconstruction, the caustics projection, and the foam advection.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum WaterKernel {
    /// Spectral inverse-`FFT` (`Tessendorf`): evolve and transform the wave
    /// spectrum into a displacement/normal field per cascade.
    SpectrumIfft,
    /// Sum the analytic `Gerstner` wave trains into a displacement field.
    GerstnerDisplace,
    /// One Shallow-Water Equations step over the height/velocity grid.
    SweStep,
    /// The `PBF` density-constraint (`XPBD`) projection iteration.
    PbfDensitySolve,
    /// `FLIP`/`APIC` particle-to-grid scatter (`P2G`).
    FlipP2G,
    /// `FLIP`/`APIC` grid pressure projection (the incompressibility solve).
    FlipPressureSolve,
    /// `FLIP`/`APIC` grid-to-particle gather (`G2P`).
    FlipG2P,
    /// Reconstruct a renderable surface from the particle set (screen-space,
    /// anisotropic marching cubes, or narrow-band `SDF`).
    SurfaceReconstruct,
    /// Project caustic light intensity onto the receiving geometry.
    CausticsProject,
    /// Advect and decay the foam coverage field (semi-Lagrangian).
    FoamAdvect,
    /// Append crest-spray particles into the `Ember` particle pool.
    SprayEmit,
    /// Advance the per-cell surface wetness/moisture field one step.
    WetnessStep,
    /// Rasterize the soft waterline transition mask (above/below water).
    WaterlineMask,
    /// Screen-space spectral refraction with per-channel (`RGB`) `IOR` offsets.
    DispersionRefract,
    /// Accumulate underwater volumetric single-scatter into the froxel volume.
    UnderwaterVolume,
    /// Read back bounded two-way coupling field queries from the `GPU`.
    CouplingReadback,
    /// Evolve the `Tessendorf` wave spectrum to time `t` and write the four
    /// packed cascade grids the separable `FFT` consumes.
    SpectrumEvolve,
    /// Bit-reverse reorder one `FFT` axis: the radix-2 butterfly prologue.
    FftBitReverse,
    /// One radix-2 butterfly stage over one `FFT` axis.
    FftStage,
    /// Normalize the inverse-`FFT` output by `1/(N*N)`.
    FftNormalize,
    /// Assemble the four cascade grids into displacement and normal textures.
    SpectrumAssemble,
}

impl WaterKernel {
    /// Every kernel, in a stable order, for descriptor-table iteration and
    /// exhaustiveness tests.
    pub const ALL: [WaterKernel; 21] = [
        WaterKernel::SpectrumIfft,
        WaterKernel::GerstnerDisplace,
        WaterKernel::SweStep,
        WaterKernel::PbfDensitySolve,
        WaterKernel::FlipP2G,
        WaterKernel::FlipPressureSolve,
        WaterKernel::FlipG2P,
        WaterKernel::SurfaceReconstruct,
        WaterKernel::CausticsProject,
        WaterKernel::FoamAdvect,
        WaterKernel::SprayEmit,
        WaterKernel::WetnessStep,
        WaterKernel::WaterlineMask,
        WaterKernel::DispersionRefract,
        WaterKernel::UnderwaterVolume,
        WaterKernel::CouplingReadback,
        WaterKernel::SpectrumEvolve,
        WaterKernel::FftBitReverse,
        WaterKernel::FftStage,
        WaterKernel::FftNormalize,
        WaterKernel::SpectrumAssemble,
    ];

    /// The stable `WESL` entry-point name the shader codegen emits for this
    /// kernel. Unique across kernels so the pipeline cache keys on it.
    ///
    /// The `WESL` source behind each name is pending the `GPU` backend; only the
    /// name is contracted here.
    #[must_use]
    pub fn wesl_entry_point(self) -> &'static str {
        match self {
            WaterKernel::SpectrumIfft => "water_spectrum_ifft",
            WaterKernel::GerstnerDisplace => "water_gerstner_displace",
            WaterKernel::SweStep => "water_swe_step",
            WaterKernel::PbfDensitySolve => "water_pbf_density_solve",
            WaterKernel::FlipP2G => "water_flip_p2g",
            WaterKernel::FlipPressureSolve => "water_flip_pressure_solve",
            WaterKernel::FlipG2P => "water_flip_g2p",
            WaterKernel::SurfaceReconstruct => "water_surface_reconstruct",
            WaterKernel::CausticsProject => "water_caustics_project",
            WaterKernel::FoamAdvect => "water_foam_advect",
            WaterKernel::SprayEmit => "water_spray_emit",
            WaterKernel::WetnessStep => "water_wetness_step",
            WaterKernel::WaterlineMask => "water_waterline_mask",
            WaterKernel::DispersionRefract => "water_dispersion_refract",
            WaterKernel::UnderwaterVolume => "water_underwater_volume",
            WaterKernel::CouplingReadback => "water_coupling_readback",
            WaterKernel::SpectrumEvolve => "water_spectrum_evolve",
            WaterKernel::FftBitReverse => "water_fft_bitrev",
            WaterKernel::FftStage => "water_fft_stage",
            WaterKernel::FftNormalize => "water_fft_normalize",
            WaterKernel::SpectrumAssemble => "water_spectrum_assemble",
        }
    }

    /// The full dispatch contract for this kernel.
    ///
    /// The workgroup tiles are design targets (module doc): 2D writers use an
    /// 8x8 texel tile, the 3D pressure/transfer voxel passes a 4x4x4 brick,
    /// the linear particle passes a 64-lane group, and the full-screen passes an
    /// 8x8 pixel tile. Every descriptor binds at least one resource and has a
    /// non-zero tile, which the `CPU` tests enforce.
    #[must_use]
    pub fn descriptor(self) -> KernelDescriptor {
        let (layout, workgroup, domain) = match self {
            WaterKernel::SpectrumIfft => (
                BindGroupLayout {
                    storage_buffers: 2,
                    uniform_buffers: 1,
                    storage_textures: 2,
                    sampled_textures: 0,
                },
                WorkgroupSize { x: 8, y: 8, z: 1 },
                DispatchDomain::Grid2d,
            ),
            WaterKernel::GerstnerDisplace => (
                BindGroupLayout {
                    storage_buffers: 1,
                    uniform_buffers: 1,
                    storage_textures: 2,
                    sampled_textures: 0,
                },
                WorkgroupSize { x: 8, y: 8, z: 1 },
                DispatchDomain::Grid2d,
            ),
            WaterKernel::SweStep => (
                BindGroupLayout {
                    storage_buffers: 2,
                    uniform_buffers: 1,
                    storage_textures: 1,
                    sampled_textures: 1,
                },
                WorkgroupSize { x: 8, y: 8, z: 1 },
                DispatchDomain::Grid2d,
            ),
            WaterKernel::PbfDensitySolve | WaterKernel::FlipP2G | WaterKernel::FlipG2P => (
                BindGroupLayout {
                    storage_buffers: 3,
                    uniform_buffers: 1,
                    storage_textures: 0,
                    sampled_textures: 0,
                },
                WorkgroupSize { x: 64, y: 1, z: 1 },
                DispatchDomain::Particle,
            ),
            WaterKernel::FlipPressureSolve => (
                BindGroupLayout {
                    storage_buffers: 3,
                    uniform_buffers: 1,
                    storage_textures: 0,
                    sampled_textures: 0,
                },
                WorkgroupSize { x: 4, y: 4, z: 4 },
                DispatchDomain::Grid3d,
            ),
            WaterKernel::SurfaceReconstruct => (
                BindGroupLayout {
                    storage_buffers: 2,
                    uniform_buffers: 1,
                    storage_textures: 1,
                    sampled_textures: 0,
                },
                WorkgroupSize { x: 8, y: 8, z: 1 },
                DispatchDomain::Screen,
            ),
            WaterKernel::CausticsProject => (
                BindGroupLayout {
                    storage_buffers: 1,
                    uniform_buffers: 1,
                    storage_textures: 1,
                    sampled_textures: 1,
                },
                WorkgroupSize { x: 8, y: 8, z: 1 },
                DispatchDomain::Screen,
            ),
            WaterKernel::FoamAdvect => (
                BindGroupLayout {
                    storage_buffers: 1,
                    uniform_buffers: 1,
                    storage_textures: 1,
                    sampled_textures: 1,
                },
                WorkgroupSize { x: 8, y: 8, z: 1 },
                DispatchDomain::Grid2d,
            ),
            WaterKernel::SprayEmit | WaterKernel::CouplingReadback => (
                BindGroupLayout {
                    storage_buffers: 2,
                    uniform_buffers: 1,
                    storage_textures: 0,
                    sampled_textures: 0,
                },
                WorkgroupSize { x: 64, y: 1, z: 1 },
                DispatchDomain::Particle,
            ),
            WaterKernel::WetnessStep => (
                BindGroupLayout {
                    storage_buffers: 1,
                    uniform_buffers: 1,
                    storage_textures: 1,
                    sampled_textures: 0,
                },
                WorkgroupSize { x: 8, y: 8, z: 1 },
                DispatchDomain::Grid2d,
            ),
            WaterKernel::WaterlineMask => (
                BindGroupLayout {
                    storage_buffers: 0,
                    uniform_buffers: 1,
                    storage_textures: 1,
                    sampled_textures: 1,
                },
                WorkgroupSize { x: 8, y: 8, z: 1 },
                DispatchDomain::Screen,
            ),
            WaterKernel::DispersionRefract => (
                BindGroupLayout {
                    storage_buffers: 0,
                    uniform_buffers: 1,
                    storage_textures: 1,
                    sampled_textures: 2,
                },
                WorkgroupSize { x: 8, y: 8, z: 1 },
                DispatchDomain::Screen,
            ),
            WaterKernel::UnderwaterVolume => (
                BindGroupLayout {
                    storage_buffers: 0,
                    uniform_buffers: 1,
                    storage_textures: 1,
                    sampled_textures: 1,
                },
                WorkgroupSize { x: 4, y: 4, z: 4 },
                DispatchDomain::Grid3d,
            ),
            WaterKernel::SpectrumEvolve => (
                BindGroupLayout {
                    storage_buffers: 6,
                    uniform_buffers: 1,
                    storage_textures: 0,
                    sampled_textures: 0,
                },
                WorkgroupSize { x: 8, y: 8, z: 1 },
                DispatchDomain::Grid2d,
            ),
            WaterKernel::SpectrumAssemble => (
                BindGroupLayout {
                    storage_buffers: 4,
                    uniform_buffers: 1,
                    storage_textures: 2,
                    sampled_textures: 0,
                },
                WorkgroupSize { x: 8, y: 8, z: 1 },
                DispatchDomain::Grid2d,
            ),
            WaterKernel::FftBitReverse | WaterKernel::FftStage | WaterKernel::FftNormalize => (
                BindGroupLayout {
                    storage_buffers: 2,
                    uniform_buffers: 1,
                    storage_textures: 0,
                    sampled_textures: 0,
                },
                WorkgroupSize { x: 8, y: 8, z: 1 },
                DispatchDomain::Grid2d,
            ),
        };
        KernelDescriptor {
            kernel: self,
            layout,
            workgroup,
            domain,
        }
    }
}

/// Number of workgroups needed to cover `invocations` at `group` invocations
/// per group.
///
/// A saturating ceiling division mirroring the particle scheduler's
/// `workgroup_count`: a zero group size or zero invocation count yields zero
/// groups (nothing to dispatch) rather than dividing by zero, and the count
/// never overflows.
#[must_use]
pub fn linear_group_count(invocations: u32, group: u32) -> u32 {
    if group == 0 || invocations == 0 {
        return 0;
    }
    let last = invocations - 1;
    (last / group) + 1
}

#[cfg(test)]
mod tests {
    use super::{linear_group_count, BindGroupLayout, DispatchDomain, WaterKernel, WorkgroupSize};
    use alloc::collections::BTreeSet;

    #[test]
    fn every_kernel_has_a_nonzero_workgroup() {
        for kernel in WaterKernel::ALL {
            let d = kernel.descriptor();
            assert!(
                d.workgroup.is_nonzero(),
                "{kernel:?} declared a zero workgroup axis"
            );
            assert!(d.workgroup.invocations_per_group() > 0);
        }
    }

    #[test]
    fn every_kernel_binds_at_least_one_resource() {
        for kernel in WaterKernel::ALL {
            assert!(
                kernel.descriptor().layout.total() > 0,
                "{kernel:?} bound no resources"
            );
        }
    }

    #[test]
    fn descriptor_kernel_tag_matches_its_key() {
        for kernel in WaterKernel::ALL {
            assert_eq!(kernel.descriptor().kernel, kernel);
        }
    }

    #[test]
    fn entry_points_are_unique() {
        let mut names = BTreeSet::new();
        for kernel in WaterKernel::ALL {
            assert!(
                names.insert(kernel.wesl_entry_point()),
                "duplicate entry point for {kernel:?}"
            );
        }
        assert_eq!(names.len(), WaterKernel::ALL.len());
    }

    #[test]
    fn particle_kernels_are_linear_and_grid_kernels_are_tiled() {
        for kernel in WaterKernel::ALL {
            let d = kernel.descriptor();
            match d.domain {
                DispatchDomain::Particle => {
                    assert_eq!(d.workgroup.y, 1);
                    assert_eq!(d.workgroup.z, 1);
                }
                DispatchDomain::Grid3d => assert!(d.workgroup.z > 1),
                DispatchDomain::Grid2d | DispatchDomain::Screen => {
                    assert!(d.workgroup.x > 1 && d.workgroup.y > 1);
                    assert_eq!(d.workgroup.z, 1);
                }
            }
        }
    }

    #[test]
    fn descriptor_is_deterministic() {
        for kernel in WaterKernel::ALL {
            assert_eq!(kernel.descriptor(), kernel.descriptor());
        }
    }

    #[test]
    fn invocations_per_group_saturates_and_multiplies() {
        assert_eq!(
            WorkgroupSize { x: 8, y: 8, z: 1 }.invocations_per_group(),
            64
        );
        assert_eq!(
            WorkgroupSize { x: 4, y: 4, z: 4 }.invocations_per_group(),
            64
        );
        assert_eq!(
            WorkgroupSize {
                x: u32::MAX,
                y: 2,
                z: 2
            }
            .invocations_per_group(),
            u32::MAX
        );
    }

    #[test]
    fn is_nonzero_rejects_any_zero_axis() {
        assert!(WorkgroupSize { x: 1, y: 1, z: 1 }.is_nonzero());
        assert!(!WorkgroupSize { x: 0, y: 1, z: 1 }.is_nonzero());
        assert!(!WorkgroupSize { x: 1, y: 0, z: 1 }.is_nonzero());
        assert!(!WorkgroupSize { x: 1, y: 1, z: 0 }.is_nonzero());
    }

    #[test]
    fn bind_group_total_sums_every_class_and_saturates() {
        let layout = BindGroupLayout {
            storage_buffers: 3,
            uniform_buffers: 1,
            storage_textures: 2,
            sampled_textures: 1,
        };
        assert_eq!(layout.total(), 7);
        assert_eq!(
            BindGroupLayout {
                storage_buffers: u32::MAX,
                uniform_buffers: 1,
                storage_textures: 0,
                sampled_textures: 0,
            }
            .total(),
            u32::MAX
        );
    }

    #[test]
    fn linear_group_count_is_saturating_ceiling_division() {
        assert_eq!(linear_group_count(0, 64), 0);
        assert_eq!(linear_group_count(100, 0), 0);
        assert_eq!(linear_group_count(64, 64), 1);
        assert_eq!(linear_group_count(65, 64), 2);
        assert_eq!(linear_group_count(1, 64), 1);
        assert_eq!(linear_group_count(128, 64), 2);
        assert_eq!(linear_group_count(u32::MAX, 1), u32::MAX);
    }
}
