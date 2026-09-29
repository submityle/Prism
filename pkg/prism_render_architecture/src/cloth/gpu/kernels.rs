//! `GPU` compute-kernel dispatch contract for the cloth subsystem.
//!
//! The sibling `cloth` modules resolve every per-frame decision on the `CPU`
//! (constraint coloring, `LOD` selection, budget arbitration, collision proxy
//! setup — see [`super::super::pipeline`] and [`super::super::dynamics`]). The
//! heavy per-vertex and per-constraint numerical passes — the predict/integrate
//! step, the graph-colored `XPBD` distance/bending/`LRA` projection, the strain
//! limiter, the velocity recovery, the self-collision spatial-hash build and
//! resolve, the body-proxy collision, and the render-mesh skin embedding — run
//! as `GPU` compute dispatches driven by `WESL`-authored shaders.
//!
//! This module owns only the *contract* for those dispatches: which bind-group
//! resources each kernel binds, the workgroup tiling it launches with, the
//! dispatch-domain shape that sizes the launch, and the stable `WESL`
//! entry-point name the shader codegen emits. It mirrors the water subsystem's
//! [`crate::water::kernels`] contract layer so both read identically in `CPU`
//! tests, and — like it — carries no `GPU` handles, no floats, and no
//! allocation, so the descriptor table is a pure deterministic function of a
//! [`ClothKernel`] tag.
//!
//! Any workgroup size here is a **design target, not a measured value**: a
//! 64-lane linear group for the per-vertex and per-constraint passes and a
//! 4x4x4 brick for the spatial-hash grid encode the intended tiling and must be
//! re-tuned against real Metal captures once the backend lands.

/// The bind-group resource counts a kernel declares, grouped by binding class.
///
/// The renderer builds the concrete `GPU` bind-group layout from these counts;
/// the contract layer only needs the shape (how many of each class) to validate
/// that a kernel binds at least one resource and to size descriptor pools.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct BindGroupLayout {
    /// Read/write `storage` buffers (particle position/velocity pools, the
    /// colored constraint arrays, the spatial-hash cell/entry buffers, the
    /// render-mesh embed weights).
    pub storage_buffers: u32,
    /// Read-only `uniform` buffers (per-substep solver constants, gravity,
    /// compliance, collision-proxy transforms).
    pub uniform_buffers: u32,
    /// Writable `storage` textures (none for the current cloth passes, kept for
    /// contract parity with the water layer).
    pub storage_textures: u32,
    /// Sampled (read-only) textures bound with a sampler (painted-constraint
    /// weight maps, wrinkle tension maps).
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
/// against the right magnitude.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum DispatchDomain {
    /// A flat list of sim-mesh particles (predict, velocity update, strain
    /// limit, body collision, self-collision resolve).
    Particle,
    /// A flat list of the constraints in one graph color (the `XPBD` distance,
    /// bending and `LRA` projection passes; the launch is sized per color).
    ConstraintBatch,
    /// A 3D grid of spatial-hash cells (the self-collision hash build).
    HashGrid,
    /// A flat list of render-mesh vertices (the barycentric skin embedding that
    /// makes the high-resolution render mesh follow the coarse sim mesh).
    RenderVertex,
}

/// A fully described `GPU` compute dispatch for one cloth pass.
///
/// This is the immutable contract the renderer consumes to build and record a
/// dispatch: the resource layout, the workgroup tiling, and the domain shape.
/// It carries no `GPU` handles — those are resolved when the backend lands.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct KernelDescriptor {
    /// The kernel this descriptor belongs to.
    pub kernel: ClothKernel,
    /// The bind-group resource counts the kernel declares.
    pub layout: BindGroupLayout,
    /// The workgroup tiling the kernel is launched with.
    pub workgroup: WorkgroupSize,
    /// The index-space shape the dispatch is sized against.
    pub domain: DispatchDomain,
}

/// Every `GPU` compute kernel the cloth subsystem dispatches, in solver order.
///
/// The passes map one-to-one onto the `CPU` golden solver stages
/// ([`super::super::dynamics::solve_cloth_with_collision`]): predict, project
/// the colored constraint graph (distance, then bending, then long-range
/// attachment), strain-limit, collide, recover velocity — plus the
/// `GPU`-driven self-collision spatial hash and the render-mesh skin embedding
/// that a persistent-pipeline cloth engine folds into the same frame.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ClothKernel {
    /// Predict positions: damp velocity, apply gravity and wind, integrate.
    /// Pinned particles are skipped by the shader on its inverse-mass test.
    Predict,
    /// Accumulate the per-triangle aerodynamic (drag + lift) force as a
    /// race-free per-vertex *gather*: one thread per vertex sums a third of the
    /// wind force of each incident triangle from a frozen velocity snapshot, so
    /// every velocity is written by exactly one thread with no scatter atomics.
    /// Mirrors the `CPU` golden [`super::super::aero_gather::accumulate_aero_gather`];
    /// runs as an aero pre-pass after predict and before constraint projection.
    Aerodynamics,
    /// Project one graph color's distance constraints (`XPBD` compliance),
    /// parallel within the color and serial across colors. Sized per color.
    ProjectDistanceBatch,
    /// Project one graph color's dihedral bending constraints. Sized per color.
    ProjectBendingBatch,
    /// Project the long-range attachment / tether constraints that cap a
    /// particle's geodesic distance to its attachment root.
    ProjectLongRangeBatch,
    /// Hard-clamp structural-edge overstretch to the strain limit.
    StrainLimit,
    /// Resolve body-proxy collision (sphere / capsule / plane) per particle.
    BodyCollision,
    /// Push each particle onto the front side of its painted backstop plane,
    /// so a garment cannot sink more than an authored distance behind the
    /// skinned body. Per particle; a piece with no painted backstops skips it.
    Backstop,
    /// Build the self-collision spatial hash: bin each particle into its grid
    /// cell and write the cell/entry buffers the resolve pass reads.
    SelfCollisionHashBuild,
    /// Resolve self-collision by testing each particle against the occupants of
    /// its own and neighbouring hash cells.
    SelfCollisionResolve,
    /// Recover velocity from the position delta accumulated across the substep.
    VelocityUpdate,
    /// Embed the render mesh: each render vertex follows the coarse sim mesh
    /// through its barycentric binding to a sim triangle.
    SkinEmbed,
}

impl ClothKernel {
    /// Every kernel, in a stable solver order, for descriptor-table iteration
    /// and exhaustiveness tests.
    pub const ALL: [ClothKernel; 12] = [
        ClothKernel::Predict,
        ClothKernel::Aerodynamics,
        ClothKernel::ProjectDistanceBatch,
        ClothKernel::ProjectBendingBatch,
        ClothKernel::ProjectLongRangeBatch,
        ClothKernel::StrainLimit,
        ClothKernel::BodyCollision,
        ClothKernel::Backstop,
        ClothKernel::SelfCollisionHashBuild,
        ClothKernel::SelfCollisionResolve,
        ClothKernel::VelocityUpdate,
        ClothKernel::SkinEmbed,
    ];

    /// The stable `WESL` entry-point name the shader codegen emits for this
    /// kernel. Unique across kernels so the pipeline cache keys on it.
    #[must_use]
    pub fn wesl_entry_point(self) -> &'static str {
        match self {
            ClothKernel::Predict => "cloth_predict",
            ClothKernel::Aerodynamics => "cloth_aerodynamics",
            ClothKernel::ProjectDistanceBatch => "cloth_project_distance_batch",
            ClothKernel::ProjectBendingBatch => "cloth_project_bending_batch",
            ClothKernel::ProjectLongRangeBatch => "cloth_project_long_range_batch",
            ClothKernel::StrainLimit => "cloth_strain_limit",
            ClothKernel::BodyCollision => "cloth_body_collision",
            ClothKernel::Backstop => "cloth_backstop",
            ClothKernel::SelfCollisionHashBuild => "cloth_self_collision_hash_build",
            ClothKernel::SelfCollisionResolve => "cloth_self_collision_resolve",
            ClothKernel::VelocityUpdate => "cloth_velocity_update",
            ClothKernel::SkinEmbed => "cloth_skin_embed",
        }
    }

    /// Returns `true` when the pass mutates constraint endpoints in place and
    /// therefore must run serially across graph colors (parallel only within a
    /// color) to preserve the Gauss-Seidel ordering the solver depends on.
    ///
    /// The predict, strain-limit, collision, velocity and embed passes are
    /// per-particle and race-free, so they return `false`.
    #[must_use]
    pub fn is_color_serial(self) -> bool {
        matches!(
            self,
            ClothKernel::ProjectDistanceBatch
                | ClothKernel::ProjectBendingBatch
                | ClothKernel::ProjectLongRangeBatch
        )
    }

    /// The full dispatch contract for this kernel.
    ///
    /// The workgroup tiles are design targets (module doc): the per-particle
    /// and per-constraint linear passes use a 64-lane group, and the
    /// self-collision hash-grid build uses a 4x4x4 voxel brick. Every
    /// descriptor binds at least one resource and has a non-zero tile, which
    /// the `CPU` tests enforce.
    #[must_use]
    pub fn descriptor(self) -> KernelDescriptor {
        let (layout, workgroup, domain) = match self {
            ClothKernel::Aerodynamics => (
                // Read-write position and velocity pools (the inverse mass is
                // packed into `positions.w`, exactly like every other cloth
                // pass, so it needs no separate buffer), the read-only triangle
                // topology buffer and the two read-only `CSR` vertex->triangle
                // adjacency buffers (offsets + entries), plus the wind/aero/dt
                // uniform block. One thread per vertex gathers its incident
                // faces, so no writable texture and no scatter buffer are
                // needed.
                BindGroupLayout {
                    storage_buffers: 5,
                    uniform_buffers: 1,
                    storage_textures: 0,
                    sampled_textures: 0,
                },
                WorkgroupSize { x: 64, y: 1, z: 1 },
                DispatchDomain::Particle,
            ),
            ClothKernel::Predict => (
                // Position, velocity and inverse-mass storage plus the
                // per-substep uniform block; the sampled texture is the
                // painted wind/tension map the predict step reads to modulate
                // the aerodynamic force per vertex.
                BindGroupLayout {
                    storage_buffers: 3,
                    uniform_buffers: 1,
                    storage_textures: 0,
                    sampled_textures: 1,
                },
                WorkgroupSize { x: 64, y: 1, z: 1 },
                DispatchDomain::Particle,
            ),
            ClothKernel::ProjectDistanceBatch
            | ClothKernel::ProjectBendingBatch
            | ClothKernel::ProjectLongRangeBatch => (
                BindGroupLayout {
                    storage_buffers: 2,
                    uniform_buffers: 1,
                    storage_textures: 0,
                    sampled_textures: 0,
                },
                WorkgroupSize { x: 64, y: 1, z: 1 },
                DispatchDomain::ConstraintBatch,
            ),
            ClothKernel::StrainLimit
            | ClothKernel::VelocityUpdate
            | ClothKernel::BodyCollision
            | ClothKernel::Backstop => (
                BindGroupLayout {
                    storage_buffers: 2,
                    uniform_buffers: 1,
                    storage_textures: 0,
                    sampled_textures: 0,
                },
                WorkgroupSize { x: 64, y: 1, z: 1 },
                DispatchDomain::Particle,
            ),
            ClothKernel::SelfCollisionHashBuild => (
                BindGroupLayout {
                    storage_buffers: 3,
                    uniform_buffers: 1,
                    storage_textures: 0,
                    sampled_textures: 0,
                },
                WorkgroupSize { x: 4, y: 4, z: 4 },
                DispatchDomain::HashGrid,
            ),
            ClothKernel::SelfCollisionResolve => (
                BindGroupLayout {
                    storage_buffers: 3,
                    uniform_buffers: 1,
                    storage_textures: 0,
                    sampled_textures: 0,
                },
                WorkgroupSize { x: 64, y: 1, z: 1 },
                DispatchDomain::Particle,
            ),
            ClothKernel::SkinEmbed => (
                BindGroupLayout {
                    storage_buffers: 3,
                    uniform_buffers: 1,
                    storage_textures: 0,
                    sampled_textures: 0,
                },
                WorkgroupSize { x: 64, y: 1, z: 1 },
                DispatchDomain::RenderVertex,
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
    use super::{linear_group_count, BindGroupLayout, ClothKernel, DispatchDomain, WorkgroupSize};
    use alloc::collections::BTreeSet;

    #[test]
    fn every_kernel_has_a_nonzero_workgroup() {
        for kernel in ClothKernel::ALL {
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
        for kernel in ClothKernel::ALL {
            assert!(
                kernel.descriptor().layout.total() > 0,
                "{kernel:?} bound no resources"
            );
        }
    }

    #[test]
    fn descriptor_kernel_tag_matches_its_key() {
        for kernel in ClothKernel::ALL {
            assert_eq!(kernel.descriptor().kernel, kernel);
        }
    }

    #[test]
    fn entry_points_are_unique() {
        let mut names = BTreeSet::new();
        for kernel in ClothKernel::ALL {
            assert!(
                names.insert(kernel.wesl_entry_point()),
                "duplicate entry point for {kernel:?}"
            );
        }
        assert_eq!(names.len(), ClothKernel::ALL.len());
    }

    #[test]
    fn particle_and_constraint_kernels_are_linear_and_hash_is_a_brick() {
        for kernel in ClothKernel::ALL {
            let d = kernel.descriptor();
            match d.domain {
                DispatchDomain::Particle
                | DispatchDomain::ConstraintBatch
                | DispatchDomain::RenderVertex => {
                    assert_eq!(d.workgroup.y, 1);
                    assert_eq!(d.workgroup.z, 1);
                    assert!(d.workgroup.x > 1);
                }
                DispatchDomain::HashGrid => {
                    assert!(d.workgroup.x > 1 && d.workgroup.y > 1 && d.workgroup.z > 1);
                }
            }
        }
    }

    #[test]
    fn only_projection_passes_are_color_serial() {
        for kernel in ClothKernel::ALL {
            let serial = kernel.is_color_serial();
            let is_projection = matches!(
                kernel,
                ClothKernel::ProjectDistanceBatch
                    | ClothKernel::ProjectBendingBatch
                    | ClothKernel::ProjectLongRangeBatch
            );
            assert_eq!(serial, is_projection, "{kernel:?} color-serial mismatch");
        }
    }

    #[test]
    fn descriptor_is_deterministic() {
        for kernel in ClothKernel::ALL {
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
}
