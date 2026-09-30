//! ABI shared between the cloth compute passes and the sibling `WESL` shaders
//! `shaders/cloth_sim.wesl`, `shaders/cloth_collision.wesl` and
//! `shaders/cloth_embed.wesl`.
//!
//! Every record here is a `#[repr(C)]` host mirror of a `WESL` `struct`, laid
//! out byte-for-byte so a plain `bytemuck` cast can upload the CPU-golden
//! solver state the `prism_render_architecture::cloth::gpu` scheduler sizes.
//! The `size_of` contract tests below pin each record to the resident stride
//! constants exported by
//! [`prism_render_architecture::cloth::gpu::buffers`], so a drift between the
//! host allocation, the shader `struct` and the golden buffer sizing fails the
//! build rather than corrupting a dispatch at run time.
//!
//! `WESL`/`WGSL` layout rules mirrored here:
//!
//! * A `var<storage>` array element uses its `std430` size, which for these
//!   all-scalar records equals the packed `#[repr(C)]` size.
//! * A `var<uniform>` block is rounded up to a 16-byte multiple, so the
//!   uniform mirrors carry explicit trailing pad words and the host upload
//!   covers the full binding.
//! * A `vec3<f32>` has 16-byte alignment, so a trailing scalar packs into the
//!   same 16-byte row (the classic `vec3 + f32` slot).

use bytemuck::{Pod, Zeroable};

/// Workgroup size shared by every per-particle / per-element cloth compute
/// entry point. Must match every `@workgroup_size(...)` in the cloth shaders.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "shader-mirror oracle: the WESL @workgroup_size / size_of contract tests assert against it; no non-test consumer because the architecture-layer KernelDescriptor owns the live dispatch tiling"
    )
)]
pub(crate) const CLOTH_WORKGROUP_SIZE: u32 = 64;

/// Collider variant discriminant: analytic sphere. Mirrors
/// `CLOTH_COLLIDER_SPHERE` in `cloth_collision.wesl`.
pub(crate) const CLOTH_COLLIDER_SPHERE: u32 = 0;

/// Collider variant discriminant: capsule (segment + radius). Mirrors
/// `CLOTH_COLLIDER_CAPSULE` in `cloth_collision.wesl`.
pub(crate) const CLOTH_COLLIDER_CAPSULE: u32 = 1;

/// Collider variant discriminant: half-space (plane). Mirrors
/// `CLOTH_COLLIDER_HALF_SPACE` in `cloth_collision.wesl`.
pub(crate) const CLOTH_COLLIDER_HALF_SPACE: u32 = 2;

/// Empty-cell sentinel for the self-collision spatial hash linked list.
/// Mirrors `CLOTH_COL_SENTINEL` in `cloth_collision.wesl`.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "shader-mirror oracle: the abi contract test asserts it equals u32::MAX to match CLOTH_COL_SENTINEL in cloth_collision.wesl; the live empty-cell sentinel is written by the shader, not host code"
    )
)]
pub(crate) const CLOTH_HASH_SENTINEL: u32 = 0xffff_ffff;

/// Constraint-kind tags stamped into [`GpuClothConstraint::kind`]. These mirror
/// the `CLOTH_CONSTRAINT_*` constants in `cloth_sim.wesl` and encode
/// `prism_render_architecture::cloth::ConstraintKind` in its declaration order.
/// Only [`CLOTH_CONSTRAINT_STRETCH`] is currently consulted on the GPU (by the
/// strain limiter), but the full ladder is encoded so future per-kind kernels
/// need no ABI change.
///
/// Structural warp/weft edge — the only kind the strain limiter clamps.
pub(crate) const CLOTH_CONSTRAINT_STRETCH: u32 = 0;
/// Cross-diagonal bending-resistance distance edge.
pub(crate) const CLOTH_CONSTRAINT_BEND: u32 = 1;
/// Quad-diagonal shear-resistance edge.
pub(crate) const CLOTH_CONSTRAINT_SHEAR: u32 = 2;
/// Long-range attachment leash to a pinned anchor.
pub(crate) const CLOTH_CONSTRAINT_LRA: u32 = 3;
/// One-sided tether leash to an anchor.
pub(crate) const CLOTH_CONSTRAINT_TETHER: u32 = 4;

/// One distance constraint. Byte-compatible with `ClothConstraint` in
/// `cloth_sim.wesl` and the golden `CONSTRAINT_STRIDE` (20 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuClothConstraint {
    /// First endpoint particle index.
    pub a: u32,
    /// Second endpoint particle index.
    pub b: u32,
    /// Rest length the projection drives the edge toward.
    pub rest_length: f32,
    /// XPBD compliance (inverse stiffness); `<= 0` = rigid.
    pub compliance: f32,
    /// Constraint-kind tag (`CLOTH_CONSTRAINT_*`). The projection kernels treat
    /// every two-sided distance edge alike, but the strain limiter must clamp
    /// only structural (stretch) edges to mirror the CPU golden
    /// `apply_strain_limit`, so the kind travels with each record.
    pub kind: u32,
}

/// One dihedral bending hinge. Byte-compatible with `ClothBendingConstraint`
/// in `cloth_sim.wesl`: four stencil indices, four bend-Laplacian weights, an
/// area-derived energy scale and the XPBD compliance (ten 4-byte words).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuClothBendingConstraint {
    /// Stencil vertex index 0 (`edge0`).
    pub v0: u32,
    /// Stencil vertex index 1 (`edge1`).
    pub v1: u32,
    /// Stencil vertex index 2 (`apex_a`).
    pub v2: u32,
    /// Stencil vertex index 3 (`apex_b`).
    pub v3: u32,
    /// Bend-Laplacian weight for `v0`.
    pub w0: f32,
    /// Bend-Laplacian weight for `v1`.
    pub w1: f32,
    /// Bend-Laplacian weight for `v2`.
    pub w2: f32,
    /// Bend-Laplacian weight for `v3`.
    pub w3: f32,
    /// Area-derived energy scale `3 / (area_a + area_b)`.
    pub scale: f32,
    /// XPBD compliance (inverse bending stiffness); `<= 0` = rigid.
    pub compliance: f32,
}

/// Per-substep solver uniform. Byte-compatible with `ClothSimParams` in
/// `cloth_sim.wesl`: `gravity` packs with `dt_sub` into the first 16-byte row,
/// then the scalar tail is padded to the 48-byte uniform stride.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuClothSimParams {
    /// Constant external acceleration (gravity), world units per second².
    pub gravity: [f32; 3],
    /// Substep duration `dt / substeps`, seconds (packs into the gravity row).
    pub dt_sub: f32,
    /// Velocity retention `1 - damping` applied each substep.
    pub retain: f32,
    /// Strain limiter maximum stretch scale `1 + strain_limit`.
    pub strain_max_scale: f32,
    /// Number of particles bounding the per-particle dispatches.
    pub particle_count: u32,
    /// Number of distance constraints bounding the per-constraint dispatches.
    pub constraint_count: u32,
    /// Number of bending hinges bounding the per-hinge dispatch.
    pub bending_count: u32,
    /// Trailing pad to the 48-byte 16-byte-rounded uniform stride (the `vec3`
    /// gravity forces 16-byte struct alignment); never read.
    pub _pad: [u32; 3],
}

/// One collision proxy. Byte-compatible with `ClothCollider` in
/// `cloth_collision.wesl`: a variant tag plus two points and a radius/offset
/// (eight 4-byte words).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuClothCollider {
    /// Selects the collider variant (`CLOTH_COLLIDER_*`).
    pub kind: u32,
    /// First point `x`: sphere centre / capsule `p0` / half-space normal `x`.
    pub ax: f32,
    /// First point `y`.
    pub ay: f32,
    /// First point `z`.
    pub az: f32,
    /// Second point `x` (capsule `p1` only, else ignored).
    pub bx: f32,
    /// Second point `y`.
    pub by: f32,
    /// Second point `z`.
    pub bz: f32,
    /// Radius (sphere / capsule) or signed offset (half-space).
    pub radius: f32,
}

/// One spatial-hash cell header. Byte-compatible with `ClothHashCell` in
/// `cloth_collision.wesl` and the golden `HASH_CELL_STRIDE` (8 bytes). The
/// shader declares the two words `atomic<u32>`; the host mirror keeps them as
/// plain `u32` because the memory layout is identical and the host never
/// touches them atomically.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuClothHashCell {
    /// Index of the first particle chained into this cell, or the sentinel.
    pub head: u32,
    /// Number of particles chained into this cell.
    pub count: u32,
}

/// Body-collision dispatch uniform. Byte-compatible with `ClothBodyParams` in
/// `cloth_collision.wesl`, padded to the 16-byte uniform stride.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuClothBodyParams {
    /// Number of particles bounding the per-particle dispatch.
    pub particle_count: u32,
    /// Number of colliders every free particle is projected against, in order.
    pub collider_count: u32,
    /// Cloth-side Coulomb friction coefficient (`mu`), clamped to `0..=1` on the
    /// host before upload and re-clamped in the kernel. Sourced from
    /// `FabricMaterial::friction`; a garment that leaves it `0` gets the exact
    /// frictionless body-collision path.
    pub friction: f32,
    /// Trailing pad to the 16-byte-rounded uniform stride; never read.
    pub _pad: [u32; 1],
}

/// Self-collision dispatch uniform. Byte-compatible with `ClothSelfParams` in
/// `cloth_collision.wesl` (already a full 16-byte row).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuClothSelfParams {
    /// Number of particles bounding the per-particle dispatches.
    pub particle_count: u32,
    /// Number of hash buckets in the cell table (the modulus for the hash).
    pub table_size: u32,
    /// Uniform grid cell edge, world units (assumed `> 0`).
    pub cell_size: f32,
    /// Separation distance below which a particle pair is pushed apart.
    pub thickness: f32,
}

/// One painted backstop plane. Byte-compatible with `ClothBackstop` in
/// `cloth_collision.wesl` and the golden `BACKSTOP_STRIDE` (32 bytes): the
/// anchor `origin.xyz` plus the scalar `distance`, then the plane `normal.xyz`
/// plus one pad word, packed as two 16-byte `vec4` slots.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuClothBackstop {
    /// Anchor point on the skinned surface, `x`.
    pub ox: f32,
    /// Anchor `y`.
    pub oy: f32,
    /// Anchor `z`.
    pub oz: f32,
    /// How far behind `origin` (along `-normal`) the particle may travel.
    pub distance: f32,
    /// Outward plane normal (need not be unit), `x`.
    pub nx: f32,
    /// Normal `y`.
    pub ny: f32,
    /// Normal `z`.
    pub nz: f32,
    /// Trailing pad word so the record occupies exactly two `vec4` slots.
    pub pad: f32,
}

/// Backstop dispatch uniform. Byte-compatible with `ClothBackstopParams` in
/// `cloth_collision.wesl`, padded to the 16-byte uniform stride.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuClothBackstopParams {
    /// Number of particles (and paired backstop records) the pass runs over.
    pub particle_count: u32,
    /// Trailing pad to the 16-byte-rounded uniform stride; never read.
    pub _pad: [u32; 3],
}

/// One render-vertex embedding. Byte-compatible with `ClothEmbedBinding` in
/// `cloth_embed.wesl` and the golden `EMBED_STRIDE` (32 bytes): a host-triangle
/// index triple, three barycentric weights, a signed normal offset and one pad
/// word (eight 4-byte words).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuClothEmbedBinding {
    /// Index of the first host-triangle sim particle.
    pub tri0: u32,
    /// Index of the second host-triangle sim particle.
    pub tri1: u32,
    /// Index of the third host-triangle sim particle.
    pub tri2: u32,
    /// Barycentric weight of the first host vertex.
    pub w0: f32,
    /// Barycentric weight of the second host vertex.
    pub w1: f32,
    /// Barycentric weight of the third host vertex.
    pub w2: f32,
    /// Signed distance along the host triangle's face normal (thickness).
    pub normal_offset: f32,
    /// Padding to the 32-byte `std430` stride; never read.
    pub pad: u32,
}

/// Skin-embed dispatch uniform. Byte-compatible with `ClothEmbedParams` in
/// `cloth_embed.wesl`, padded to the 16-byte uniform stride.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuClothEmbedParams {
    /// Number of render-mesh vertices, bounding the linear dispatch.
    pub render_vertex_count: u32,
    /// Trailing pad to the 16-byte-rounded uniform stride; never read.
    pub _pad: [u32; 3],
}

/// Aerodynamic (wind drag + lift) dispatch uniform. Byte-compatible with
/// `ClothAeroParams` in `cloth_aerodynamics.wesl`: the steady wind velocity
/// packs with the turbulence strength into the first 16-byte row, then the two
/// aerodynamic coefficients, the substep timestep and the particle bound fill
/// the second row, so the whole block is one 32-byte uniform stride. The host
/// sanitizes `wind` / `turbulence` / `drag` / `lift` (matching the golden
/// `WindField::sanitized` / `AeroParams::sanitized`) before upload, so the
/// shader reads finite, range-clamped values directly.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuClothAeroParams {
    /// Steady world-space wind velocity, world units per second.
    pub wind: [f32; 3],
    /// Turbulence strength clamped to `0..=1` (packs into the wind row).
    pub turbulence: f32,
    /// Normal-direction (drag) coefficient; sanitized non-negative on the host.
    pub drag: f32,
    /// In-plane (lift) coefficient; sanitized non-negative on the host.
    pub lift: f32,
    /// Full-frame timestep `dt`, seconds. Aerodynamics is a single
    /// pre-solve impulse applied once per frame (before the substep loop),
    /// mirroring the `CPU` golden `apply_aero_forces`; the per-vertex velocity
    /// increment is `force * inverse_mass * dt`.
    pub dt: f32,
    /// Number of particles (= gather vertices) bounding the per-vertex dispatch.
    pub particle_count: u32,
}

/// One virtual-particle-tier self-collision sample, byte-compatible with
/// `ClothVpSample` in `cloth_self_collision_virtual.wesl`. A real vertex `i`
/// is encoded as `verts = (i, i, i)`, `weights = (1, 0, 0)`; a virtual
/// particle carries its triangle's three corner indices and barycentric
/// weights. The two trailing pads keep the flat 32-byte `std430` record stride
/// so the reals-then-virtuals sample array packs without gaps.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuClothVpSample {
    /// First active vertex index.
    pub v0: u32,
    /// Second active vertex index.
    pub v1: u32,
    /// Third active vertex index.
    pub v2: u32,
    /// Barycentric weight for `v0`.
    pub w0: f32,
    /// Barycentric weight for `v1`.
    pub w1: f32,
    /// Barycentric weight for `v2`.
    pub w2: f32,
    /// Trailing pad word; never read.
    pub _pad0: f32,
    /// Trailing pad word; never read.
    pub _pad1: f32,
}

/// Virtual-particle self-collision dispatch uniform. Byte-compatible with
/// `ClothVpParams` in `cloth_self_collision_virtual.wesl`: four counts fill the
/// first 16-byte row, then the cell size, the thickness, the augment-mode flag
/// and a trailing pad fill the second, so the whole block is one 32-byte
/// uniform stride mirroring the `CPU` golden `virtual_particles_jacobi` inputs.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuClothVpParams {
    /// Total sample count (reals + in-range virtuals), bounding the
    /// hash/resolve dispatches.
    pub sample_count: u32,
    /// Number of real particles; samples `< real_count` are real vertices.
    pub real_count: u32,
    /// Number of real vertices, bounding the scatter dispatch.
    pub vertex_count: u32,
    /// Number of hash buckets in the cell table (the cell-hash modulus).
    pub table_size: u32,
    /// Uniform grid cell edge, world units.
    pub cell_size: f32,
    /// Separation distance below which a sample pair is pushed apart.
    pub thickness: f32,
    /// When non-zero, real-vs-real pairs are skipped (augment mode); zero
    /// resolves every pair (self-contained tier). Mirrors the `CPU` `PairScope`.
    pub virtual_only: u32,
    /// Trailing pad so the block is a flat 32-byte uniform; never read.
    pub _pad: u32,
}

/// `ClothCcdParams` in `cloth_self_ccd.wesl`: the particle count, hash
/// modulus, cell size and thickness fill the first two 16-byte rows, then the
/// restitution, the inverse frame time and two trailing pads fill the rest, so
/// the whole block is one 32-byte uniform stride mirroring the sanitized `CPU`
/// golden `self_ccd::SelfCcdParams` inputs plus the `inv_dt` the resolver
/// derives once per frame.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuClothCcdParams {
    /// Number of particles, bounding every self-CCD dispatch.
    pub particle_count: u32,
    /// Number of hash buckets in the cell table (the cell-hash modulus).
    pub table_size: u32,
    /// Uniform grid cell edge, world units.
    pub cell_size: f32,
    /// Minimum enforced separation; the TOI target distance.
    pub thickness: f32,
    /// Normal restitution in `[0, 1]`.
    pub restitution: f32,
    /// Inverse frame time `1/dt`, or `0` when `|dt|` is negligible.
    pub inv_dt: f32,
    /// Trailing pad so the block is a flat 32-byte uniform; never read.
    pub _pad0: u32,
    /// Trailing pad so the block is a flat 32-byte uniform; never read.
    pub _pad1: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_render_architecture::cloth::gpu::buffers::{
        BACKSTOP_STRIDE, CONSTRAINT_STRIDE, EMBED_STRIDE, HASH_CELL_STRIDE, PARTICLE_VEC_STRIDE,
    };

    /// The position / velocity / previous-position buffers are plain
    /// `array<vec4<f32>>`, so their element matches the golden particle stride.
    #[test]
    fn particle_vec_matches_golden_stride() {
        assert_eq!(size_of::<[f32; 4]>() as u32, PARTICLE_VEC_STRIDE);
    }

    /// Distance-constraint record equals the golden constraint stride.
    #[test]
    fn constraint_matches_golden_stride() {
        assert_eq!(size_of::<GpuClothConstraint>() as u32, CONSTRAINT_STRIDE);
    }

    /// The bending hinge is ten 4-byte words; no golden stride constant exists
    /// for it, so pin the raw size the shader `struct` implies.
    #[test]
    fn bending_constraint_is_forty_bytes() {
        assert_eq!(size_of::<GpuClothBendingConstraint>(), 40);
    }

    /// The solver uniform rounds up to the 48-byte `WGSL` uniform stride.
    #[test]
    fn sim_params_is_uniform_stride() {
        assert_eq!(size_of::<GpuClothSimParams>(), 48);
        assert_eq!(size_of::<GpuClothSimParams>() % 16, 0);
    }

    /// A collider proxy is eight 4-byte words (32 bytes).
    #[test]
    fn collider_is_thirty_two_bytes() {
        assert_eq!(size_of::<GpuClothCollider>(), 32);
    }

    /// The hash-cell header equals the golden hash-cell stride.
    #[test]
    fn hash_cell_matches_golden_stride() {
        assert_eq!(size_of::<GpuClothHashCell>() as u32, HASH_CELL_STRIDE);
    }

    /// The body-collision uniform rounds up to a 16-byte stride.
    #[test]
    fn body_params_is_uniform_stride() {
        assert_eq!(size_of::<GpuClothBodyParams>(), 16);
    }

    /// The friction word sits between `collider_count` and the trailing pad, so
    /// a populated coefficient survives a byte round-trip and `Default` leaves
    /// it frictionless.
    #[test]
    fn body_params_carries_friction() {
        assert_eq!(GpuClothBodyParams::default().friction, 0.0);
        let params = GpuClothBodyParams {
            particle_count: 7,
            collider_count: 3,
            friction: 0.42,
            _pad: [0],
        };
        let bytes = bytemuck::bytes_of(&params);
        let round: GpuClothBodyParams = *bytemuck::from_bytes(bytes);
        assert_eq!(round, params);
        // The friction f32 occupies the third 4-byte word (offset 8).
        assert_eq!(f32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]), 0.42);
    }

    /// The self-collision uniform is already one full 16-byte row.
    #[test]
    fn self_params_is_uniform_stride() {
        assert_eq!(size_of::<GpuClothSelfParams>(), 16);
    }

    /// The backstop record equals the golden backstop stride.
    #[test]
    fn backstop_matches_golden_stride() {
        assert_eq!(size_of::<GpuClothBackstop>() as u32, BACKSTOP_STRIDE);
    }

    /// The backstop uniform rounds up to a 16-byte stride.
    #[test]
    fn backstop_params_is_uniform_stride() {
        assert_eq!(size_of::<GpuClothBackstopParams>(), 16);
    }

    /// The embedding record equals the golden embed stride.
    #[test]
    fn embed_binding_matches_golden_stride() {
        assert_eq!(size_of::<GpuClothEmbedBinding>() as u32, EMBED_STRIDE);
    }

    /// The embed uniform rounds up to a 16-byte stride.
    #[test]
    fn embed_params_is_uniform_stride() {
        assert_eq!(size_of::<GpuClothEmbedParams>(), 16);
    }

    /// The aerodynamic uniform is exactly two 16-byte `WGSL` uniform rows: the
    /// wind vector packs with the turbulence strength into the first row, and
    /// the two coefficients, the timestep and the particle bound fill the
    /// second. Pin the 32-byte stride so a field drift fails the build rather
    /// than silently misaligning the shader read.
    #[test]
    fn aero_params_is_uniform_stride() {
        assert_eq!(size_of::<GpuClothAeroParams>(), 32);
        assert_eq!(size_of::<GpuClothAeroParams>() % 16, 0);
    }

    /// The virtual-particle sample record is a flat 32-byte `std430` stride:
    /// three vertex indices, three barycentric weights and two trailing pads,
    /// matching `ClothVpSample` in `cloth_self_collision_virtual.wesl`. Pin the
    /// size so a field drift fails the build rather than silently misaligning
    /// the reals-then-virtuals sample array.
    #[test]
    fn vp_sample_is_std430_stride() {
        assert_eq!(size_of::<GpuClothVpSample>(), 32);
        assert_eq!(align_of::<GpuClothVpSample>(), 4);
    }

    /// The virtual-particle dispatch uniform is exactly two 16-byte `WGSL`
    /// uniform rows: four counts in the first, then the cell size, the
    /// thickness, the augment flag and a pad in the second. Pin the 32-byte
    /// stride against drift from `ClothVpParams`.
    #[test]
    fn vp_params_is_uniform_stride() {
        assert_eq!(size_of::<GpuClothVpParams>(), 32);
        assert_eq!(size_of::<GpuClothVpParams>() % 16, 0);
        assert_eq!(align_of::<GpuClothVpParams>(), 4);
    }

    /// The collider discriminants match the shader's `CLOTH_COLLIDER_*` order.
    #[test]
    fn collider_discriminants_are_contiguous() {
        assert_eq!(CLOTH_COLLIDER_SPHERE, 0);
        assert_eq!(CLOTH_COLLIDER_CAPSULE, 1);
        assert_eq!(CLOTH_COLLIDER_HALF_SPACE, 2);
    }

    /// The hash sentinel is the all-ones `u32` the shader tests against, and
    /// the shared workgroup size matches the cloth `@workgroup_size(64)`.
    #[test]
    fn scalar_contracts_match_shader() {
        assert_eq!(CLOTH_HASH_SENTINEL, u32::MAX);
        assert_eq!(CLOTH_WORKGROUP_SIZE, 64);
    }
}
