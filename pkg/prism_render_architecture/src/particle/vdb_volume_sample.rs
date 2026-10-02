//! Sparse `VDB` volume sampling — the `CPU` reference behind
//! [`super::modules::BuiltinDataInterface::SampleVdb`], its `vdb_tree` storage
//! binding, and the `sample_vdb_density` / `sample_vdb_gradient` functions the
//! §8.3 data-interface manifest names (design
//! `docs/prism_particle_engine_design_zh.md` §8.3, "场: ... `SDF`/`VDB`
//! (碰撞与吸附, 法线=梯度)").
//!
//! # Why a separate module from [`super::sdf`]
//!
//! [`super::sdf`] samples a *dense* scalar grid: one contiguous `f32` per
//! texel, every texel allocated. That is the right storage for a tightly
//! bounded baked signed-distance field, but it is hopeless for the mostly empty
//! volumes a next-generation smoke / fire / explosion effect produces (design
//! §10): a dense `512^3` density grid is `512` MiB even when `99%` of it is
//! vacuum. This module adds the complementary *sparse* representation — a
//! `VDB`-style tree whose empty regions cost nothing — so the two modules are
//! strictly orthogonal: dense distances in `sdf.rs`, sparse density here.
//!
//! # Tree topology (`root` → `internal` → `leaf`)
//!
//! The layout follows the public `OpenVDB` / `NanoVDB` design (Museth 2013,
//! "`VDB`: High-Resolution Sparse Volumes with Dynamic Topology") and the
//! `Niagara` / `EmberGen` sparse-voxel convention, reproduced from the public
//! algorithm only — no vendor source is copied:
//!
//! * A **`leaf`** is a *dense* block of `LEAF_DIM^3` voxels (`LEAF_DIM = 8`, so
//!   `512` voxels). Each voxel carries a value plus an *active* flag; an
//!   inactive voxel reads the tree `background` value.
//! * An **`internal`** node is a dense `INTERNAL_DIM^3` grid of child slots
//!   (`INTERNAL_DIM = 4`, so `64` slots). A slot either points at a `leaf` or
//!   is an inactive tile that reads `background`.
//! * The **`root`** is a dense `root_dims` grid of child slots over the whole
//!   domain; each slot points at an `internal` node or is an inactive tile.
//!
//! Every level's branching factor is a power of two, so decomposing a voxel
//! coordinate into `(root slot, internal child, leaf voxel)` is a chain of
//! shifts and masks — exactly how `VDB` achieves `O(1)` random access. Nodes
//! are stored in flat compact [`alloc::vec::Vec`] arrays and referenced by
//! index (`-1` meaning "inactive / unallocated"), which keeps the structure
//! cache-friendly and trivially serialisable for the future `GPU` twin.
//!
//! # Sampling
//!
//! * [`sample_vdb_density`] reconstructs a continuous density by **trilinear**
//!   interpolation of the eight voxels surrounding a continuous voxel-space
//!   position. Each corner is fetched through the sparse descent, so corners in
//!   unallocated regions contribute `background` with no special-casing — the
//!   same math as [`super::sdf::SdfField::sample_distance`], differing only in
//!   how a corner value is looked up.
//! * [`sample_vdb_gradient`] takes the **central-difference** gradient of that
//!   trilinear field (∇density, pointing toward *increasing* density) and
//!   returns it as a unit vector, matching the "法线=梯度" rule of §8.3. The
//!   normalisation multiplies by `1/sqrt(len^2)`; a degenerate (flat) region
//!   yields [`super::Vec3::ZERO`] rather than a `NaN`.
//!
//! # Determinism
//!
//! Everything is pure. The only floating-point primitives beyond multiply-add
//! are `f32::floor` (integer voxel location) and `f32::sqrt` (the one allowed
//! transcendental, used only to normalise the gradient). There are no
//! `sin`/`cos`/`exp`/`ln`/`pow` calls, so a `GPU` kernel that descends the same
//! `vdb_tree` in the same order is bit-reproducible against this reference
//! (design §5, §9 require the dual-backend parity).

use alloc::vec;
use alloc::vec::Vec;

use super::gpu_layout::U32_STRIDE;
use super::Vec3;

/// Base-two logarithm of a leaf's per-axis voxel count. A leaf spans
/// `LEAF_DIM = 2^LEAF_LOG2 = 8` voxels per axis, the `OpenVDB` default leaf
/// size.
const LEAF_LOG2: u32 = 3;

/// Per-axis voxel count of a leaf (`8`).
const LEAF_DIM: u32 = 1 << LEAF_LOG2;

/// Low-bit mask extracting a voxel's leaf-local coordinate (`LEAF_DIM - 1`).
const LEAF_MASK: u32 = LEAF_DIM - 1;

/// Total voxels in one leaf block, `LEAF_DIM^3 = 512`. Written as a shift by
/// `3 * LEAF_LOG2` so the cube is exact integer arithmetic.
const LEAF_SIZE: usize = 1 << (3 * LEAF_LOG2);

/// Base-two logarithm of an internal node's per-axis child count. An internal
/// node spans `INTERNAL_DIM = 2^INTERNAL_LOG2 = 4` leaves per axis.
const INTERNAL_LOG2: u32 = 2;

/// Per-axis child count of an internal node (`4`).
const INTERNAL_DIM: u32 = 1 << INTERNAL_LOG2;

/// Low-bit mask extracting an internal node's child coordinate
/// (`INTERNAL_DIM - 1`).
const INTERNAL_MASK: u32 = INTERNAL_DIM - 1;

/// Total child slots in one internal node, `INTERNAL_DIM^3 = 64`.
const INTERNAL_SIZE: usize = 1 << (3 * INTERNAL_LOG2);

/// Number of low bits a voxel coordinate dedicates to the sub-tree below the
/// root (internal child bits plus leaf voxel bits). Each root slot therefore
/// covers `2^SLOT_LOG2 = 32` voxels per axis.
const SLOT_LOG2: u32 = LEAF_LOG2 + INTERNAL_LOG2;

/// Sentinel child index meaning "inactive tile / unallocated child"; such a
/// slot reads the tree `background` value.
const NO_CHILD: i32 = -1;

/// Squared-length floor below which the central-difference gradient is treated
/// as degenerate (a flat region) and reported as [`Vec3::ZERO`] rather than
/// normalised, guarding the `1/sqrt(len^2)` step against division blow-up.
/// Mirrors the `EPS_LEN_SQ` guard [`super::Vec3::normalize_or_zero`] uses.
const GRAD_EPS_SQ: f32 = 1e-12;

/// Half-voxel finite-difference step for [`sample_vdb_gradient`]. Sampling the
/// trilinear field at the midpoints on either side of a coordinate yields the
/// reconstructed field's gradient without landing exactly on the (piecewise)
/// integer voxel planes — the same choice as [`super::sdf`]'s `GRAD_STEP`.
const GRAD_STEP: f32 = 0.5;

/// A dense `LEAF_DIM^3` voxel block: the finest, fully populated level of the
/// sparse tree. Stored flat, `X`-fastest.
#[derive(Clone, Debug, PartialEq)]
struct LeafNode {
    /// `LEAF_SIZE` voxel values, `X`-fastest then `Y` then `Z`. Positions whose
    /// `active` flag is clear are ignored by queries (they read `background`).
    values: Vec<f32>,
    /// Per-voxel active flags, parallel to `values`. A freshly allocated leaf is
    /// entirely inactive until [`VdbTree::set_voxel`] writes a voxel.
    active: Vec<bool>,
}

impl LeafNode {
    /// Allocates an all-inactive leaf (every voxel reads `background`).
    fn inactive() -> Self {
        Self {
            values: vec![0.0; LEAF_SIZE],
            active: vec![false; LEAF_SIZE],
        }
    }
}

/// A dense `INTERNAL_DIM^3` grid of child slots pointing at [`LeafNode`]s (or
/// [`NO_CHILD`] for an inactive tile). The single internal level between
/// `root` and `leaf` keeps the reference tree a faithful three-level `VDB`
/// without the full configurable depth of production `OpenVDB`.
#[derive(Clone, Debug, PartialEq, Eq)]
struct InternalNode {
    /// `INTERNAL_SIZE` leaf indices into [`VdbTree::leaves`], `X`-fastest; a
    /// slot holding [`NO_CHILD`] is an inactive tile.
    children: Vec<i32>,
}

impl InternalNode {
    /// Allocates an internal node with every child slot inactive.
    fn empty() -> Self {
        Self {
            children: vec![NO_CHILD; INTERNAL_SIZE],
        }
    }
}

/// A sparse `VDB` density volume: a three-level (`root` → `internal` → `leaf`)
/// tree over an integer voxel domain, with a uniform `background` value for
/// every unallocated or inactive region (design §8.3).
///
/// Coordinates passed to the samplers live in *voxel space*: a voxel centre
/// sits on its integer coordinate, matching the dense convention of
/// [`super::sdf`] so an effect can swap a dense field for a sparse one without
/// changing call sites. The domain is `root_dims[a] * 2^SLOT_LOG2` voxels wide
/// on axis `a`; anything outside reads `background`.
#[derive(Clone, Debug, PartialEq)]
pub struct VdbTree {
    /// Value returned for every inactive tile and inactive voxel, and for
    /// samples outside the domain — the `VDB` "background" (typically `0.0` for
    /// a density field, i.e. vacuum).
    background: f32,
    /// Root dimensions in *internal-node slots* along `[X, Y, Z]`; each slot
    /// covers `2^SLOT_LOG2` voxels per axis. Every axis is at least `1`.
    root_dims: [u32; 3],
    /// Dense `root_dims` grid of child slots (`X`-fastest), each an index into
    /// [`VdbTree::internals`] or [`NO_CHILD`] for an inactive tile.
    root: Vec<i32>,
    /// Compact pool of allocated internal nodes, referenced by index from
    /// [`VdbTree::root`].
    internals: Vec<InternalNode>,
    /// Compact pool of allocated leaf blocks, referenced by index from an
    /// [`InternalNode`]'s child slots.
    leaves: Vec<LeafNode>,
}

/// `std430` byte-size report for the flattened `vdb_tree` storage buffers a
/// future `GPU` descent kernel binds (design §5, §9). Pure integer sizing, like
/// [`super::bvh`]'s topology plan — it describes the `ABI`, it does not encode
/// data.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct VdbStd430Layout {
    /// Byte size of the root slot buffer (one `u32` index per root slot).
    pub root_bytes: u64,
    /// Byte size of the internal child-index buffer across all internal nodes
    /// (`INTERNAL_SIZE` `u32`s each).
    pub internal_bytes: u64,
    /// Byte size of the leaf value buffer across all leaves (`LEAF_SIZE` `f32`s
    /// each).
    pub leaf_value_bytes: u64,
    /// Byte size of the leaf active-mask buffer: one `u32` bitmask word per `32`
    /// voxels, across all leaves.
    pub leaf_mask_bytes: u64,
}

impl VdbStd430Layout {
    /// Total `std430` byte size of every `vdb_tree` sub-buffer combined.
    #[must_use]
    pub fn total_bytes(&self) -> u64 {
        self.root_bytes
            .saturating_add(self.internal_bytes)
            .saturating_add(self.leaf_value_bytes)
            .saturating_add(self.leaf_mask_bytes)
    }
}

impl VdbTree {
    /// Builds an empty tree over a `root_dims` grid of root slots with the given
    /// `background` value.
    ///
    /// Returns `None` when any axis is `0` or the root slot count would overflow
    /// a `usize`, since a zero-extent volume cannot be addressed. The tree
    /// starts fully sparse: every slot is inactive, so every query returns
    /// `background` until [`VdbTree::set_voxel`] populates voxels.
    #[must_use]
    pub fn new(root_dims: [u32; 3], background: f32) -> Option<Self> {
        let count = Self::checked_root_count(root_dims)?;
        Some(Self {
            background,
            root_dims,
            root: vec![NO_CHILD; count],
            internals: Vec::new(),
            leaves: Vec::new(),
        })
    }

    /// Root slot count for `root_dims`, or `None` on a zero axis / `usize`
    /// overflow.
    #[must_use]
    fn checked_root_count(root_dims: [u32; 3]) -> Option<usize> {
        let [nx, ny, nz] = root_dims;
        if nx == 0 || ny == 0 || nz == 0 {
            return None;
        }
        (nx as usize)
            .checked_mul(ny as usize)
            .and_then(|xy| xy.checked_mul(nz as usize))
    }

    /// The uniform background value read by every inactive region.
    #[must_use]
    pub fn background(&self) -> f32 {
        self.background
    }

    /// Root dimensions in internal-node slots along `[X, Y, Z]`.
    #[must_use]
    pub fn root_dims(&self) -> [u32; 3] {
        self.root_dims
    }

    /// Voxel domain size along `[X, Y, Z]`: `root_dims[a] << SLOT_LOG2`. Samples
    /// at or beyond these bounds (or at negative coordinates) read `background`.
    #[must_use]
    pub fn domain_dims(&self) -> [u32; 3] {
        [
            self.root_dims[0] << SLOT_LOG2,
            self.root_dims[1] << SLOT_LOG2,
            self.root_dims[2] << SLOT_LOG2,
        ]
    }

    /// Number of allocated internal nodes (sparse occupancy, for diagnostics /
    /// sizing).
    #[must_use]
    pub fn internal_node_count(&self) -> usize {
        self.internals.len()
    }

    /// Number of allocated leaf blocks (sparse occupancy, for diagnostics /
    /// sizing).
    #[must_use]
    pub fn leaf_count(&self) -> usize {
        self.leaves.len()
    }

    /// Row-major (`X`-fastest) linear index of root slot `(sx, sy, sz)`. The
    /// caller supplies in-range slot coordinates.
    #[must_use]
    fn root_index(&self, sx: u32, sy: u32, sz: u32) -> usize {
        let [nx, ny, _] = self.root_dims;
        (((sz * ny) + sy) * nx + sx) as usize
    }

    /// Whether a voxel coordinate lies inside the addressable domain. Negative
    /// components are outside by definition.
    #[must_use]
    fn in_domain(&self, coord: [i32; 3]) -> bool {
        let [dx, dy, dz] = self.domain_dims();
        coord[0] >= 0
            && coord[1] >= 0
            && coord[2] >= 0
            && (coord[0] as u32) < dx
            && (coord[1] as u32) < dy
            && (coord[2] as u32) < dz
    }

    /// Reads the density stored at integer voxel `coord`, descending
    /// `root` → `internal` → `leaf`.
    ///
    /// Returns `background` whenever any level along the path is an inactive
    /// tile, the voxel's active flag is clear, or `coord` is outside the domain.
    /// This is the sparse analogue of [`super::sdf::SdfField::sample_texel`].
    #[must_use]
    pub fn voxel_value(&self, coord: [i32; 3]) -> f32 {
        if !self.in_domain(coord) {
            return self.background;
        }
        // Safe to treat as unsigned after the in-domain guard.
        let cx = coord[0] as u32;
        let cy = coord[1] as u32;
        let cz = coord[2] as u32;

        // Level 1: root slot (top bits of each coordinate).
        let root_idx = self.root_index(cx >> SLOT_LOG2, cy >> SLOT_LOG2, cz >> SLOT_LOG2);
        let internal_idx = self.root[root_idx];
        if internal_idx == NO_CHILD {
            return self.background;
        }

        // Level 2: internal child (middle bits).
        let child_slot = internal_child_index(cx, cy, cz);
        let leaf_idx = self.internals[internal_idx as usize].children[child_slot];
        if leaf_idx == NO_CHILD {
            return self.background;
        }

        // Level 3: dense leaf voxel (low bits).
        let voxel_slot = leaf_voxel_index(cx, cy, cz);
        let leaf = &self.leaves[leaf_idx as usize];
        if leaf.active[voxel_slot] {
            leaf.values[voxel_slot]
        } else {
            self.background
        }
    }

    /// Whether the voxel at integer `coord` is active (allocated and written).
    /// Inactive voxels and out-of-domain coordinates return `false`.
    #[must_use]
    pub fn is_voxel_active(&self, coord: [i32; 3]) -> bool {
        if !self.in_domain(coord) {
            return false;
        }
        let cx = coord[0] as u32;
        let cy = coord[1] as u32;
        let cz = coord[2] as u32;
        let root_idx = self.root_index(cx >> SLOT_LOG2, cy >> SLOT_LOG2, cz >> SLOT_LOG2);
        let internal_idx = self.root[root_idx];
        if internal_idx == NO_CHILD {
            return false;
        }
        let leaf_idx =
            self.internals[internal_idx as usize].children[internal_child_index(cx, cy, cz)];
        if leaf_idx == NO_CHILD {
            return false;
        }
        self.leaves[leaf_idx as usize].active[leaf_voxel_index(cx, cy, cz)]
    }

    /// Writes `value` at integer voxel `coord`, allocating the `internal` and
    /// `leaf` nodes along the path on demand and marking the voxel active.
    ///
    /// Returns `true` on success, or `false` (and writes nothing) when `coord`
    /// is outside the domain. This is the construction primitive the authoring
    /// / baking path drives to populate the sparse volume.
    pub fn set_voxel(&mut self, coord: [i32; 3], value: f32) -> bool {
        if !self.in_domain(coord) {
            return false;
        }
        let cx = coord[0] as u32;
        let cy = coord[1] as u32;
        let cz = coord[2] as u32;

        // Ensure the root slot points at an internal node.
        let root_idx = self.root_index(cx >> SLOT_LOG2, cy >> SLOT_LOG2, cz >> SLOT_LOG2);
        let internal_slot = if self.root[root_idx] == NO_CHILD {
            let new_idx = self.internals.len() as i32;
            self.internals.push(InternalNode::empty());
            self.root[root_idx] = new_idx;
            new_idx
        } else {
            self.root[root_idx]
        };
        let internal_idx = internal_slot as usize;

        // Ensure the internal child slot points at a leaf.
        let child_slot = internal_child_index(cx, cy, cz);
        let leaf_slot = if self.internals[internal_idx].children[child_slot] == NO_CHILD {
            let new_idx = self.leaves.len() as i32;
            self.leaves.push(LeafNode::inactive());
            self.internals[internal_idx].children[child_slot] = new_idx;
            new_idx
        } else {
            self.internals[internal_idx].children[child_slot]
        };
        let leaf_idx = leaf_slot as usize;

        // Write the dense leaf voxel.
        let voxel_slot = leaf_voxel_index(cx, cy, cz);
        let leaf = &mut self.leaves[leaf_idx];
        leaf.values[voxel_slot] = value;
        leaf.active[voxel_slot] = true;
        true
    }

    /// Trilinearly reconstructs the density at a continuous voxel-space
    /// position (design §8.3).
    ///
    /// A position exactly on an integer voxel returns that voxel's value (or
    /// `background` if inactive); the midpoint between two voxels returns their
    /// average. The eight surrounding corners are fetched through
    /// [`VdbTree::voxel_value`], so corners in unallocated regions contribute
    /// `background` with no special case. Pure multiply-add plus one `floor`.
    #[must_use]
    pub fn sample_density(&self, pos: Vec3) -> f32 {
        let i0f = pos.x.floor();
        let j0f = pos.y.floor();
        let k0f = pos.z.floor();
        let fx = pos.x - i0f;
        let fy = pos.y - j0f;
        let fz = pos.z - k0f;
        let i0 = i0f as i32;
        let j0 = j0f as i32;
        let k0 = k0f as i32;
        let i1 = i0 + 1;
        let j1 = j0 + 1;
        let k1 = k0 + 1;

        let c000 = self.voxel_value([i0, j0, k0]);
        let c100 = self.voxel_value([i1, j0, k0]);
        let c010 = self.voxel_value([i0, j1, k0]);
        let c110 = self.voxel_value([i1, j1, k0]);
        let c001 = self.voxel_value([i0, j0, k1]);
        let c101 = self.voxel_value([i1, j0, k1]);
        let c011 = self.voxel_value([i0, j1, k1]);
        let c111 = self.voxel_value([i1, j1, k1]);

        let gx = 1.0 - fx;
        let gy = 1.0 - fy;
        let gz = 1.0 - fz;

        let c00 = c000 * gx + c100 * fx;
        let c10 = c010 * gx + c110 * fx;
        let c01 = c001 * gx + c101 * fx;
        let c11 = c011 * gx + c111 * fx;

        let c0 = c00 * gy + c10 * fy;
        let c1 = c01 * gy + c11 * fy;

        c0 * gz + c1 * fz
    }

    /// Raw (unnormalised) central-difference gradient of the trilinear density
    /// field at `pos`, in density-per-voxel units. Points toward increasing
    /// density.
    #[must_use]
    fn raw_gradient(&self, pos: Vec3) -> Vec3 {
        let dx = self.sample_density(pos.add(Vec3::new(GRAD_STEP, 0.0, 0.0)))
            - self.sample_density(pos.sub(Vec3::new(GRAD_STEP, 0.0, 0.0)));
        let dy = self.sample_density(pos.add(Vec3::new(0.0, GRAD_STEP, 0.0)))
            - self.sample_density(pos.sub(Vec3::new(0.0, GRAD_STEP, 0.0)));
        let dz = self.sample_density(pos.add(Vec3::new(0.0, 0.0, GRAD_STEP)))
            - self.sample_density(pos.sub(Vec3::new(0.0, 0.0, GRAD_STEP)));
        // Divide the central difference by the full `2 * GRAD_STEP` span.
        Vec3::new(dx, dy, dz).scale(1.0 / (2.0 * GRAD_STEP))
    }

    /// Unit gradient (surface normal) of the density field at `pos`, defined as
    /// the normalised central-difference gradient (design §8.3, "法线=梯度").
    ///
    /// The gradient points toward increasing density, so a particle shading or
    /// collision path can use it directly as an outward normal. Normalisation
    /// multiplies by `1/sqrt(len^2)`; a flat region (gradient below
    /// [`GRAD_EPS_SQ`]) returns [`Vec3::ZERO`] instead of a `NaN`.
    #[must_use]
    pub fn sample_gradient(&self, pos: Vec3) -> Vec3 {
        let g = self.raw_gradient(pos);
        let len2 = g.length_squared();
        if len2 > GRAD_EPS_SQ {
            // Unit normal via one `sqrt` — the only allowed transcendental.
            g.scale(1.0 / len2.sqrt())
        } else {
            Vec3::ZERO
        }
    }

    /// Reports the `std430` byte sizes of the flattened `vdb_tree` buffers for
    /// the `GPU` twin (design §5, §9). Pure integer sizing; it never allocates
    /// or encodes voxel data.
    #[must_use]
    pub fn std430_layout(&self) -> VdbStd430Layout {
        let u32_stride = U32_STRIDE as u64;
        let root_slots = self.root.len() as u64;
        let internals = self.internals.len() as u64;
        let leaves = self.leaves.len() as u64;
        // One `u32` bitmask word covers 32 voxels; round the leaf voxel count up.
        const MASK_WORDS_PER_LEAF: u64 = (LEAF_SIZE as u64).div_ceil(32);
        VdbStd430Layout {
            root_bytes: root_slots.saturating_mul(u32_stride),
            internal_bytes: internals
                .saturating_mul(INTERNAL_SIZE as u64)
                .saturating_mul(u32_stride),
            leaf_value_bytes: leaves
                .saturating_mul(LEAF_SIZE as u64)
                .saturating_mul(u32_stride),
            leaf_mask_bytes: leaves
                .saturating_mul(MASK_WORDS_PER_LEAF)
                .saturating_mul(u32_stride),
        }
    }
}

/// Flat `X`-fastest child index of a voxel within its internal node, from the
/// middle bits of the voxel coordinate.
#[must_use]
fn internal_child_index(cx: u32, cy: u32, cz: u32) -> usize {
    let ix = (cx >> LEAF_LOG2) & INTERNAL_MASK;
    let iy = (cy >> LEAF_LOG2) & INTERNAL_MASK;
    let iz = (cz >> LEAF_LOG2) & INTERNAL_MASK;
    (ix | (iy << INTERNAL_LOG2) | (iz << (2 * INTERNAL_LOG2))) as usize
}

/// Flat `X`-fastest voxel index within a leaf block, from the low bits of the
/// voxel coordinate.
#[must_use]
fn leaf_voxel_index(cx: u32, cy: u32, cz: u32) -> usize {
    let lx = cx & LEAF_MASK;
    let ly = cy & LEAF_MASK;
    let lz = cz & LEAF_MASK;
    (lx | (ly << LEAF_LOG2) | (lz << (2 * LEAF_LOG2))) as usize
}

/// Trilinear density sample of a sparse `VDB` volume at voxel-space `pos`
/// (design §8.3). Free-function form of [`VdbTree::sample_density`] matching the
/// `sample_vdb_density` name in the §8.3 data-interface manifest; `tree` is the
/// `vdb_tree` binding.
#[must_use]
pub fn sample_vdb_density(tree: &VdbTree, pos: Vec3) -> f32 {
    tree.sample_density(pos)
}

/// Unit density gradient (surface normal) of a sparse `VDB` volume at
/// voxel-space `pos` (design §8.3). Free-function form of
/// [`VdbTree::sample_gradient`] matching the `sample_vdb_gradient` name in the
/// §8.3 data-interface manifest; `tree` is the `vdb_tree` binding.
#[must_use]
pub fn sample_vdb_gradient(tree: &VdbTree, pos: Vec3) -> Vec3 {
    tree.sample_gradient(pos)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Absolute tolerance for the trilinear / gradient `f32` comparisons.
    const EPS: f32 = 1e-5;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() <= EPS
    }

    fn approx_vec(a: Vec3, b: Vec3) -> bool {
        approx(a.x, b.x) && approx(a.y, b.y) && approx(a.z, b.z)
    }

    #[test]
    fn new_rejects_zero_extent() {
        assert!(VdbTree::new([0, 1, 1], 0.0).is_none());
        assert!(VdbTree::new([1, 0, 1], 0.0).is_none());
        assert!(VdbTree::new([1, 1, 0], 0.0).is_none());
        assert!(VdbTree::new([1, 1, 1], 0.0).is_some());
    }

    #[test]
    fn empty_tree_reads_background_everywhere() {
        let bg = 0.25;
        let tree = VdbTree::new([2, 2, 2], bg).unwrap();
        // No node allocated: both voxel reads and the trilinear sampler see bg.
        assert!(approx(tree.voxel_value([0, 0, 0]), bg));
        assert!(approx(tree.voxel_value([40, 10, 5]), bg));
        assert!(approx(tree.sample_density(Vec3::new(3.5, 7.2, 1.0)), bg));
        assert_eq!(tree.internal_node_count(), 0);
        assert_eq!(tree.leaf_count(), 0);
    }

    #[test]
    fn hit_returns_voxel_miss_returns_background() {
        let bg = -1.0;
        let mut tree = VdbTree::new([1, 1, 1], bg).unwrap();
        assert!(tree.set_voxel([3, 4, 5], 2.0));
        assert!(approx(tree.voxel_value([3, 4, 5]), 2.0));
        // A neighbour in the same (now allocated) leaf is still inactive -> bg.
        assert!(approx(tree.voxel_value([3, 4, 6]), bg));
        assert!(tree.is_voxel_active([3, 4, 5]));
        assert!(!tree.is_voxel_active([3, 4, 6]));
    }

    #[test]
    fn out_of_domain_reads_background() {
        let bg = 7.0;
        let mut tree = VdbTree::new([1, 1, 1], bg).unwrap();
        // Domain is 32^3 for a single root slot.
        assert_eq!(tree.domain_dims(), [32, 32, 32]);
        assert!(approx(tree.voxel_value([-1, 0, 0]), bg));
        assert!(approx(tree.voxel_value([0, -5, 0]), bg));
        assert!(approx(tree.voxel_value([32, 0, 0]), bg));
        assert!(approx(tree.voxel_value([0, 0, 999]), bg));
        assert!(!tree.is_voxel_active([-1, 0, 0]));
        assert!(!tree.set_voxel([-1, 0, 0], 1.0));
        assert!(!tree.set_voxel([32, 0, 0], 1.0));
    }

    #[test]
    fn trilinear_is_exact_on_grid_points() {
        let mut tree = VdbTree::new([1, 1, 1], 0.0).unwrap();
        tree.set_voxel([2, 2, 2], 5.0);
        tree.set_voxel([3, 2, 2], 9.0);
        // Exactly on a voxel centre returns that voxel.
        assert!(approx(tree.sample_density(Vec3::new(2.0, 2.0, 2.0)), 5.0));
        assert!(approx(tree.sample_density(Vec3::new(3.0, 2.0, 2.0)), 9.0));
        // Midpoint along x averages the two.
        assert!(approx(tree.sample_density(Vec3::new(2.5, 2.0, 2.0)), 7.0));
    }

    #[test]
    fn trilinear_crosses_leaf_boundary_continuously() {
        // Voxel 7 is in leaf 0; voxel 8 is in the next leaf (leaf-local 0 of the
        // adjacent internal child). The sampler must still interpolate across
        // the leaf boundary.
        let mut tree = VdbTree::new([1, 1, 1], 0.0).unwrap();
        tree.set_voxel([7, 1, 1], 4.0);
        tree.set_voxel([8, 1, 1], 8.0);
        assert!(tree.leaf_count() >= 2);
        assert!(approx(tree.sample_density(Vec3::new(7.5, 1.0, 1.0)), 6.0));
    }

    #[test]
    fn gradient_points_toward_increasing_density() {
        // Build a density ramp increasing along +x: value == x.
        let mut tree = VdbTree::new([1, 1, 1], 0.0).unwrap();
        for x in 0..12 {
            tree.set_voxel([x, 5, 5], x as f32);
        }
        let g = tree.sample_gradient(Vec3::new(5.0, 5.0, 5.0));
        // Normal must face +x and be a unit vector.
        assert!(g.x > 0.0);
        assert!(approx(g.x, 1.0));
        assert!(approx(g.y, 0.0));
        assert!(approx(g.z, 0.0));
        assert!(approx(g.length(), 1.0));
    }

    #[test]
    fn gradient_is_unit_length_for_diagonal_ramp() {
        // Density == x + y + z over a filled block; gradient is the (1,1,1) dir.
        let mut tree = VdbTree::new([1, 1, 1], 0.0).unwrap();
        for z in 0..10 {
            for y in 0..10 {
                for x in 0..10 {
                    tree.set_voxel([x, y, z], (x + y + z) as f32);
                }
            }
        }
        let g = tree.sample_gradient(Vec3::new(5.0, 5.0, 5.0));
        assert!(approx(g.length(), 1.0));
        let inv = 1.0 / 3.0_f32.sqrt();
        assert!(approx_vec(g, Vec3::new(inv, inv, inv)));
    }

    #[test]
    fn gradient_degenerates_to_zero_in_flat_region() {
        // A single filled plateau far from the sample point -> locally flat.
        let tree = VdbTree::new([1, 1, 1], 3.0).unwrap();
        // Constant background everywhere: gradient must vanish, not NaN.
        let g = tree.sample_gradient(Vec3::new(10.0, 10.0, 10.0));
        assert!(approx_vec(g, Vec3::ZERO));
    }

    #[test]
    fn sampling_is_bit_for_bit_deterministic() {
        let mut tree = VdbTree::new([2, 2, 2], 0.0).unwrap();
        tree.set_voxel([10, 20, 30], 1.5);
        tree.set_voxel([11, 20, 30], 2.5);
        let p = Vec3::new(10.3, 20.0, 30.0);
        let a = tree.sample_density(p);
        let b = tree.sample_density(p);
        // Identical inputs -> identical bits (no reordering, no transcendental).
        assert_eq!(a.to_bits(), b.to_bits());
        let ga = tree.sample_gradient(p);
        let gb = tree.sample_gradient(p);
        assert_eq!(ga.x.to_bits(), gb.x.to_bits());
        assert_eq!(ga.y.to_bits(), gb.y.to_bits());
        assert_eq!(ga.z.to_bits(), gb.z.to_bits());
    }

    #[test]
    fn free_functions_match_methods() {
        let mut tree = VdbTree::new([1, 1, 1], 0.0).unwrap();
        tree.set_voxel([4, 4, 4], 3.0);
        tree.set_voxel([5, 4, 4], 6.0);
        let p = Vec3::new(4.5, 4.0, 4.0);
        assert_eq!(
            sample_vdb_density(&tree, p).to_bits(),
            tree.sample_density(p).to_bits()
        );
        let gf = sample_vdb_gradient(&tree, p);
        let gm = tree.sample_gradient(p);
        assert!(approx_vec(gf, gm));
    }

    #[test]
    fn std430_layout_tracks_sparse_occupancy() {
        let mut tree = VdbTree::new([2, 1, 1], 0.0).unwrap();
        let empty = tree.std430_layout();
        // Two root slots, no nodes allocated yet.
        assert_eq!(empty.root_bytes, 2 * U32_STRIDE as u64);
        assert_eq!(empty.internal_bytes, 0);
        assert_eq!(empty.leaf_value_bytes, 0);

        tree.set_voxel([0, 0, 0], 1.0);
        let one = tree.std430_layout();
        assert_eq!(one.internal_bytes, (INTERNAL_SIZE * U32_STRIDE) as u64);
        assert_eq!(one.leaf_value_bytes, (LEAF_SIZE * U32_STRIDE) as u64);
        assert!(one.total_bytes() > empty.total_bytes());
    }

    #[test]
    fn epsilon_guard_rejects_sub_tolerance_difference() {
        // The abs-diff comparison the tests rely on behaves as a tolerance band.
        assert!(approx(1.0, 1.0 + EPS * 0.5));
        assert!(!approx(1.0, 1.0 + EPS * 10.0));
    }
}
