//! `GPU` compute-kernel dispatch contract for the volumetric subsystem
//! (design section 17, milestone M8).
//!
//! The sibling `volumetric` `CPU` modules own the golden reference and every
//! per-frame decision: [`super::super::noise`] and [`super::super::modeling`]
//! define the density field, [`super::super::weather`] the advection,
//! [`super::super::raymarch`] the step/skip/early-out policy,
//! [`super::super::scatter`] and [`super::super::multiscatter`] the phase and
//! energy, [`super::super::shadow`] the light-space march, and
//! [`super::super::temporal`] the reprojection plan. This module owns only the
//! *contract* for running those same numerical passes as `GPU` compute
//! dispatches authored in `WESL`: which bind-group resources each kernel binds,
//! the workgroup tiling it launches with, the dispatch-domain shape that sizes
//! the launch, and the stable `WESL` entry-point name the shader codegen emits.
//!
//! It mirrors [`crate::cloth::gpu::kernels`] so both read identically in `CPU`
//! tests, and — like it — carries no `GPU` handles, no floats and no
//! allocation, so the descriptor table is a pure deterministic function of a
//! [`VolumetricKernel`] tag.
//!
//! **Not machine-verified.** The sandbox has no `GPU`; the `WESL` kernels these
//! descriptors anticipate are not compiled here. Every workgroup tile below is
//! a **design target, not a measured value** (a 4x4x4 voxel brick for the 3D
//! density/`LUT` bakes, an 8x8 tile for the 2D screen/shadow passes) and must
//! be re-tuned against real captures once the backend lands.

/// The bind-group resource counts a kernel declares, grouped by binding class.
///
/// The renderer builds the concrete `GPU` bind-group layout from these counts;
/// the contract layer only needs the shape (how many of each class) to validate
/// that a kernel binds at least one resource and to size descriptor pools.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct BindGroupLayout {
    /// Read/write `storage` buffers (weather cells, multi-scatter `LUT` axes,
    /// the compacted temporal history metadata).
    pub storage_buffers: u32,
    /// Read-only `uniform` buffers (per-frame camera/sun/wind constants, the
    /// ray-march step policy, the storm-stage curve block).
    pub uniform_buffers: u32,
    /// Writable `storage` textures (the baked 3D density cache, the low-res
    /// scattering/transmittance target, the light-space cloud-shadow map).
    pub storage_textures: u32,
    /// Sampled (read-only) textures bound with a sampler (the weather map, the
    /// Perlin-Worley noise atlases, the previous-frame history colour, the
    /// shared atmosphere `LUT`).
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
    /// dispatch work. A zero axis would make [`Self::invocations_per_group`]
    /// zero and the dispatch a no-op, which is always a contract bug.
    #[must_use]
    pub fn is_nonzero(self) -> bool {
        self.x != 0 && self.y != 0 && self.z != 0
    }

    /// Returns `true` when the tile is a 3D brick (every axis `> 1`), used by
    /// the volumetric bakes over a 3D index space.
    #[must_use]
    pub fn is_brick(self) -> bool {
        self.x > 1 && self.y > 1 && self.z > 1
    }
}

/// The shape of the index space a kernel is dispatched over.
///
/// The renderer reads the live extent for the matching shape from the frame
/// plan ([`super::pipeline`]) and turns it into a group count; the contract
/// layer only records which shape a kernel expects so the launch is sized
/// against the right magnitude.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum DispatchDomain {
    /// A 2D grid of weather-map texels (the semi-Lagrangian advection pass).
    WeatherTexel,
    /// A 3D voxel grid of the cloud-domain density cache (noise bake and
    /// density-field modelling composition).
    DensityVoxel,
    /// The small 3D multiple-scatter `LUT` (cosine x optical-depth x albedo).
    MultiscatterLut,
    /// A 2D grid of low-resolution ray-march tiles (checkerboard / quarter-res
    /// view integration and the octave-scatter resolve).
    ScreenTile,
    /// A 2D grid of light-space texels (the cloud-shadow / `AVSM` march that
    /// feeds the shared virtual shadow map).
    ShadowTexel,
    /// A 2D grid of full-resolution screen pixels (the temporal reprojection /
    /// upsample resolve that reconstructs the full-res cloud buffer).
    ScreenPixel,
}

impl DispatchDomain {
    /// Returns `true` when the domain is a 3D index space (voxel grid or the
    /// multi-scatter `LUT`), which the brick-tiled kernels launch over.
    #[must_use]
    pub fn is_volumetric(self) -> bool {
        matches!(
            self,
            DispatchDomain::DensityVoxel | DispatchDomain::MultiscatterLut
        )
    }
}

/// A fully described `GPU` compute dispatch for one volumetric pass.
///
/// This is the immutable contract the renderer consumes to build and record a
/// dispatch: the resource layout, the workgroup tiling, and the domain shape.
/// It carries no `GPU` handles — those are resolved when the backend lands.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct KernelDescriptor {
    /// The kernel this descriptor belongs to.
    pub kernel: VolumetricKernel,
    /// The bind-group resource counts the kernel declares.
    pub layout: BindGroupLayout,
    /// The workgroup tiling the kernel is launched with.
    pub workgroup: WorkgroupSize,
    /// The index-space shape the dispatch is sized against.
    pub domain: DispatchDomain,
}

/// Every `GPU` compute kernel the volumetric subsystem dispatches, in frame
/// order (design section 17, milestone M8: noise / modelling / ray-march /
/// scatter / cloud-shadow / upsample, plus the weather advection that feeds
/// modelling and the multi-scatter `LUT` bake that feeds scatter).
///
/// The passes map onto the `CPU` golden modules one-to-one: weather advection
/// ([`super::super::weather`]), the Perlin-Worley bake ([`super::super::noise`]),
/// density-field composition ([`super::super::modeling`]), the multi-scatter
/// `LUT` ([`super::super::multiscatter`]), the view ray-march
/// ([`super::super::raymarch`]), the octave-scatter resolve
/// ([`super::super::scatter`]), the light-space shadow march
/// ([`super::super::shadow`] / [`super::super::avsm`]), and the temporal
/// upsample ([`super::super::temporal`]).
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum VolumetricKernel {
    /// Advect the weather map one step with the semi-Lagrangian back-trace, so
    /// coverage/type/precip evolve under the wind field.
    WeatherAdvect,
    /// Bake the Perlin-Worley base + detail noise into the 3D density cache the
    /// modelling and ray-march passes sample.
    NoiseBake,
    /// Compose the density field: coverage remap, cloud-type shape, height
    /// gradient and detail erosion into the cache the ray-march reads.
    Modeling,
    /// Pre-integrate the environment multiple-scatter `LUT` (cosine x depth x
    /// albedo) the scatter resolve gathers energy from.
    MultiscatterLutBake,
    /// Low-resolution view ray-march: adaptive step, empty-space skip, early
    /// out, accumulating single scattering and transmittance per tile.
    Raymarch,
    /// Octave-scatter resolve: fold the multi-scatter `LUT` and dual-lobe phase
    /// into the ray-march result to approximate multiple scattering.
    ScatterResolve,
    /// Light-space cloud-shadow / `AVSM` march producing the deep-shadow curve
    /// the shared virtual shadow map and god-ray injection consume.
    ShadowMarch,
    /// Temporal reprojection + upsample: reconstruct the full-resolution cloud
    /// buffer from the low-res ray-march and the reprojected history.
    Upsample,
}

impl VolumetricKernel {
    /// Every kernel, in a stable frame order, for descriptor-table iteration
    /// and exhaustiveness tests.
    pub const ALL: [VolumetricKernel; 8] = [
        VolumetricKernel::WeatherAdvect,
        VolumetricKernel::NoiseBake,
        VolumetricKernel::Modeling,
        VolumetricKernel::MultiscatterLutBake,
        VolumetricKernel::Raymarch,
        VolumetricKernel::ScatterResolve,
        VolumetricKernel::ShadowMarch,
        VolumetricKernel::Upsample,
    ];

    /// The stable `WESL` entry-point name the shader codegen emits for this
    /// kernel. Unique across kernels so the pipeline cache keys on it.
    #[must_use]
    pub fn wesl_entry_point(self) -> &'static str {
        match self {
            VolumetricKernel::WeatherAdvect => "volumetric_weather_advect",
            VolumetricKernel::NoiseBake => "volumetric_noise_bake",
            VolumetricKernel::Modeling => "volumetric_modeling",
            VolumetricKernel::MultiscatterLutBake => "volumetric_multiscatter_lut_bake",
            VolumetricKernel::Raymarch => "volumetric_raymarch",
            VolumetricKernel::ScatterResolve => "volumetric_scatter_resolve",
            VolumetricKernel::ShadowMarch => "volumetric_shadow_march",
            VolumetricKernel::Upsample => "volumetric_upsample",
        }
    }

    /// Returns `true` when the pass writes a resource a later pass in the same
    /// frame reads (the bake/advection/march producers), so the frame plan must
    /// insert a barrier before its consumer. The two terminal resolves
    /// (`ScatterResolve` into the view buffer, `Upsample` into the full-res
    /// target) are consumed by downstream compositing, not by another
    /// volumetric pass, so they return `false`.
    #[must_use]
    pub fn produces_for_later_pass(self) -> bool {
        matches!(
            self,
            VolumetricKernel::WeatherAdvect
                | VolumetricKernel::NoiseBake
                | VolumetricKernel::Modeling
                | VolumetricKernel::MultiscatterLutBake
                | VolumetricKernel::Raymarch
                | VolumetricKernel::ShadowMarch
        )
    }

    /// The full dispatch contract for this kernel.
    ///
    /// The workgroup tiles are design targets (module doc): the 3D density and
    /// `LUT` bakes use a 4x4x4 voxel brick, and the 2D screen / shadow passes
    /// use an 8x8 tile. Every descriptor binds at least one resource and has a
    /// non-zero tile, which the `CPU` tests enforce.
    #[must_use]
    pub fn descriptor(self) -> KernelDescriptor {
        let brick = WorkgroupSize { x: 4, y: 4, z: 4 };
        let tile = WorkgroupSize { x: 8, y: 8, z: 1 };
        let (layout, workgroup, domain) = match self {
            VolumetricKernel::WeatherAdvect => (
                // Weather cell storage + wind uniform; samples the previous
                // weather map to back-trace.
                BindGroupLayout {
                    storage_buffers: 1,
                    uniform_buffers: 1,
                    storage_textures: 1,
                    sampled_textures: 1,
                },
                tile,
                DispatchDomain::WeatherTexel,
            ),
            VolumetricKernel::NoiseBake | VolumetricKernel::Modeling => (
                // NoiseBake writes the density cache from the tiling
                // Perlin-Worley atlases; Modeling reads that noise plus the
                // weather map and writes the composed density cache. Both bind
                // the same shape: two sampled textures, one param uniform and
                // the density-cache storage texture, over the voxel grid.
                BindGroupLayout {
                    storage_buffers: 0,
                    uniform_buffers: 1,
                    storage_textures: 1,
                    sampled_textures: 2,
                },
                brick,
                DispatchDomain::DensityVoxel,
            ),
            VolumetricKernel::MultiscatterLutBake => (
                // Writes the small 3D multi-scatter LUT; a uniform holds the
                // octave attenuation/contribution params.
                BindGroupLayout {
                    storage_buffers: 0,
                    uniform_buffers: 1,
                    storage_textures: 1,
                    sampled_textures: 0,
                },
                brick,
                DispatchDomain::MultiscatterLut,
            ),
            VolumetricKernel::Raymarch => (
                // Samples the density cache + shadow map + atmosphere LUT,
                // writes the low-res scattering/transmittance target; two
                // uniforms carry camera and the ray-march step policy.
                BindGroupLayout {
                    storage_buffers: 0,
                    uniform_buffers: 2,
                    storage_textures: 1,
                    sampled_textures: 3,
                },
                tile,
                DispatchDomain::ScreenTile,
            ),
            VolumetricKernel::ScatterResolve => (
                // Reads the low-res ray-march target + multi-scatter LUT, writes
                // the resolved scattering; a uniform carries the phase params.
                BindGroupLayout {
                    storage_buffers: 0,
                    uniform_buffers: 1,
                    storage_textures: 1,
                    sampled_textures: 2,
                },
                tile,
                DispatchDomain::ScreenTile,
            ),
            VolumetricKernel::ShadowMarch => (
                // Samples the density cache, writes the light-space cloud-shadow
                // map (AVSM curve); a uniform carries the sun transform.
                BindGroupLayout {
                    storage_buffers: 1,
                    uniform_buffers: 1,
                    storage_textures: 1,
                    sampled_textures: 1,
                },
                tile,
                DispatchDomain::ShadowTexel,
            ),
            VolumetricKernel::Upsample => (
                // Reads the low-res resolved target + previous-frame history,
                // writes the full-res cloud buffer; a uniform carries the
                // reprojection matrices and the history-clamp params.
                BindGroupLayout {
                    storage_buffers: 0,
                    uniform_buffers: 1,
                    storage_textures: 1,
                    sampled_textures: 2,
                },
                tile,
                DispatchDomain::ScreenPixel,
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
/// A saturating ceiling division: a zero group size or zero invocation count
/// yields zero groups (nothing to dispatch) rather than dividing by zero, and
/// the count never overflows.
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
    use super::{
        linear_group_count, BindGroupLayout, DispatchDomain, VolumetricKernel, WorkgroupSize,
    };
    use alloc::collections::BTreeSet;

    #[test]
    fn every_kernel_has_a_nonzero_workgroup() {
        for kernel in VolumetricKernel::ALL {
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
        for kernel in VolumetricKernel::ALL {
            assert!(
                kernel.descriptor().layout.total() > 0,
                "{kernel:?} bound no resources"
            );
        }
    }

    #[test]
    fn every_kernel_writes_at_least_one_target() {
        // Every volumetric pass produces a texture (density cache, LUT, low-res
        // target, shadow map, or full-res buffer); none is read-only.
        for kernel in VolumetricKernel::ALL {
            assert!(
                kernel.descriptor().layout.storage_textures > 0,
                "{kernel:?} wrote no storage texture"
            );
        }
    }

    #[test]
    fn descriptor_kernel_tag_matches_its_key() {
        for kernel in VolumetricKernel::ALL {
            assert_eq!(kernel.descriptor().kernel, kernel);
        }
    }

    #[test]
    fn entry_points_are_unique_and_prefixed() {
        let mut names = BTreeSet::new();
        for kernel in VolumetricKernel::ALL {
            let name = kernel.wesl_entry_point();
            assert!(name.starts_with("volumetric_"), "{kernel:?} bad prefix");
            assert!(names.insert(name), "duplicate entry point for {kernel:?}");
        }
        assert_eq!(names.len(), VolumetricKernel::ALL.len());
    }

    #[test]
    fn volumetric_bakes_are_bricks_and_screen_passes_are_tiles() {
        for kernel in VolumetricKernel::ALL {
            let d = kernel.descriptor();
            if d.domain.is_volumetric() {
                assert!(d.workgroup.is_brick(), "{kernel:?} 3D domain wants a brick");
            } else {
                assert_eq!(d.workgroup.z, 1, "{kernel:?} 2D domain wants z==1");
                assert!(d.workgroup.x > 1 && d.workgroup.y > 1);
            }
        }
    }

    #[test]
    fn only_terminal_resolves_are_not_producers() {
        for kernel in VolumetricKernel::ALL {
            let produces = kernel.produces_for_later_pass();
            let terminal = matches!(
                kernel,
                VolumetricKernel::ScatterResolve | VolumetricKernel::Upsample
            );
            assert_eq!(produces, !terminal, "{kernel:?} producer flag mismatch");
        }
    }

    #[test]
    fn descriptor_is_deterministic() {
        for kernel in VolumetricKernel::ALL {
            assert_eq!(kernel.descriptor(), kernel.descriptor());
        }
    }

    #[test]
    fn layout_total_saturates() {
        let layout = BindGroupLayout {
            storage_buffers: u32::MAX,
            uniform_buffers: 1,
            storage_textures: 1,
            sampled_textures: 1,
        };
        assert_eq!(layout.total(), u32::MAX);
    }

    #[test]
    fn invocations_per_group_saturates() {
        let tile = WorkgroupSize {
            x: u32::MAX,
            y: 2,
            z: 2,
        };
        assert_eq!(tile.invocations_per_group(), u32::MAX);
    }

    #[test]
    fn linear_group_count_is_saturating_ceil_div() {
        assert_eq!(linear_group_count(0, 64), 0);
        assert_eq!(linear_group_count(100, 0), 0);
        assert_eq!(linear_group_count(64, 64), 1);
        assert_eq!(linear_group_count(65, 64), 2);
        assert_eq!(linear_group_count(128, 64), 2);
        assert_eq!(linear_group_count(u32::MAX, 1), u32::MAX);
    }

    #[test]
    fn domain_volumetric_classification() {
        assert!(DispatchDomain::DensityVoxel.is_volumetric());
        assert!(DispatchDomain::MultiscatterLut.is_volumetric());
        assert!(!DispatchDomain::ScreenTile.is_volumetric());
        assert!(!DispatchDomain::WeatherTexel.is_volumetric());
        assert!(!DispatchDomain::ShadowTexel.is_volumetric());
        assert!(!DispatchDomain::ScreenPixel.is_volumetric());
    }
}
