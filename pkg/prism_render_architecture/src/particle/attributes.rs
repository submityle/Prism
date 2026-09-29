//! Particle attribute system and Structure-of-Arrays (`SoA`) GPU layout planner
//! (design §5).
//!
//! An attribute is a named, typed, semantic per-particle channel (position,
//! velocity, colour, and so on). Following production VFX engines (Unreal
//! `Niagara`, Unity `VFX Graph`, `PopcornFX`), Ember stores attributes in a
//! Structure-of-Arrays layout: each attribute is a contiguous buffer indexed by
//! particle slot, which keeps every kernel's memory access coalesced and
//! bandwidth-friendly.
//!
//! This module owns the *CPU-verifiable allocation contract* that the node-graph
//! compiler (design §6) drives: given the set of attributes a compiled system
//! actually reads and writes, it decides
//!
//! * which buffers to allocate at all (unused semantics cost zero bytes),
//! * how wide each element is (packed vs. `std430`-aligned strides),
//! * which attributes need a ping-pong (double) buffer because a stage reads the
//!   previous-frame value while another writes the new one, and
//! * the total device-memory footprint and the per-frame streaming bandwidth.
//!
//! The `GPU` buffers themselves are created by the backend; this layer produces
//! the deterministic plan the backend consumes and that tests can assert on.

use alloc::vec::Vec;

use super::{EmberShadingModel, ShadingBasis};

/// The built-in per-particle attribute semantics plus an escape hatch for
/// user-authored channels (design §5.1).
///
/// The discriminant order is stable and is the canonical ordering the layout
/// planner emits buffers in, so a given attribute set always produces the same
/// plan regardless of insertion order.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum AttributeSemantic {
    /// World/local position (design §5.1).
    Position,
    /// Linear velocity.
    Velocity,
    /// Seconds lived so far.
    Age,
    /// Total lifetime in seconds; death when `age >= lifetime`.
    Lifetime,
    /// Linear colour, premultiplied or straight per renderer.
    Color,
    /// Uniform or billboard size.
    Size,
    /// Scalar roll (radians) for billboards.
    Rotation,
    /// Non-uniform scale for mesh particles.
    Scale,
    /// Liveness flag; `0` means the slot is free.
    Alive,
    /// Stable per-particle identifier (deterministic `RNG` streams, design §29).
    ParticleId,
    /// Ribbon/trail chain identifier used to link segments (design §15).
    RibbonId,
    /// Quantized sort key (design §12).
    SortKey,
    /// Shading normal (only allocated when the shading model is lit).
    Normal,
    /// Shading tangent (anisotropy; only when lit and requested).
    Tangent,
    /// Emissive radiance for additive/energy looks.
    Emissive,
    /// `PBR` roughness scalar.
    Roughness,
    /// `PBR` metallic scalar.
    Metallic,
    /// Material identifier mapping into the shared material system (design §16).
    MaterialId,
    /// Packed extra shading parameters consumed by the shading closure.
    ShadingParams,
    /// A user-defined attribute keyed by a stable identifier.
    Custom(u32),
}

impl AttributeSemantic {
    /// A stable sort ordinal so plans are deterministic across insertion orders.
    ///
    /// `Custom` attributes always sort after every built-in and among themselves
    /// by their identifier.
    #[must_use]
    pub const fn ordinal(self) -> u32 {
        match self {
            AttributeSemantic::Position => 0,
            AttributeSemantic::Velocity => 1,
            AttributeSemantic::Age => 2,
            AttributeSemantic::Lifetime => 3,
            AttributeSemantic::Color => 4,
            AttributeSemantic::Size => 5,
            AttributeSemantic::Rotation => 6,
            AttributeSemantic::Scale => 7,
            AttributeSemantic::Alive => 8,
            AttributeSemantic::ParticleId => 9,
            AttributeSemantic::RibbonId => 10,
            AttributeSemantic::SortKey => 11,
            AttributeSemantic::Normal => 12,
            AttributeSemantic::Tangent => 13,
            AttributeSemantic::Emissive => 14,
            AttributeSemantic::Roughness => 15,
            AttributeSemantic::Metallic => 16,
            AttributeSemantic::MaterialId => 17,
            AttributeSemantic::ShadingParams => 18,
            // Bias custom ids above the built-in block; saturating add keeps the
            // ordinal monotone even at `u32::MAX` without overflowing.
            AttributeSemantic::Custom(id) => 19u32.saturating_add(id),
        }
    }

    /// The natural storage format each built-in semantic uses (design §5.1).
    ///
    /// `Custom` has no intrinsic format; callers must supply one explicitly, so
    /// this returns `None` for it.
    #[must_use]
    pub const fn default_format(self) -> Option<AttributeFormat> {
        let format = match self {
            AttributeSemantic::Position
            | AttributeSemantic::Velocity
            | AttributeSemantic::Scale
            | AttributeSemantic::Normal
            | AttributeSemantic::Tangent
            | AttributeSemantic::Emissive => AttributeFormat::Vec3,
            AttributeSemantic::Color | AttributeSemantic::ShadingParams => AttributeFormat::Vec4,
            AttributeSemantic::Age
            | AttributeSemantic::Lifetime
            | AttributeSemantic::Size
            | AttributeSemantic::Rotation
            | AttributeSemantic::Roughness
            | AttributeSemantic::Metallic => AttributeFormat::F32,
            AttributeSemantic::Alive
            | AttributeSemantic::ParticleId
            | AttributeSemantic::RibbonId
            | AttributeSemantic::SortKey
            | AttributeSemantic::MaterialId => AttributeFormat::U32,
            AttributeSemantic::Custom(_) => return None,
        };
        Some(format)
    }

    /// Whether this semantic is a shading input that should only be allocated
    /// when the emitter's shading model actually consumes it (design §5.1,
    /// §16). Non-shading semantics (position, velocity, ...) return `false`.
    #[must_use]
    pub const fn is_shading_input(self) -> bool {
        matches!(
            self,
            AttributeSemantic::Normal
                | AttributeSemantic::Tangent
                | AttributeSemantic::Roughness
                | AttributeSemantic::Metallic
                | AttributeSemantic::MaterialId
                | AttributeSemantic::ShadingParams
        )
    }
}

/// The device storage format of an attribute element (design §5.1).
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum AttributeFormat {
    /// A single 32-bit float.
    F32,
    /// A single 32-bit unsigned integer.
    U32,
    /// Two 32-bit floats.
    Vec2,
    /// Three 32-bit floats.
    Vec3,
    /// Four 32-bit floats.
    Vec4,
}

impl AttributeFormat {
    /// The tightly packed byte size of one element.
    #[must_use]
    pub const fn packed_size(self) -> u32 {
        match self {
            AttributeFormat::F32 | AttributeFormat::U32 => 4,
            AttributeFormat::Vec2 => 8,
            AttributeFormat::Vec3 => 12,
            AttributeFormat::Vec4 => 16,
        }
    }

    /// The array-element stride under `std430`-style rules, where a three-float
    /// vector is padded to a 16-byte boundary. This is the stride the backend
    /// uses when a buffer is bound as a shader storage array.
    #[must_use]
    pub const fn std430_stride(self) -> u32 {
        match self {
            AttributeFormat::F32 | AttributeFormat::U32 => 4,
            AttributeFormat::Vec2 => 8,
            // A three-float vector's array stride rounds up to `vec4`.
            AttributeFormat::Vec3 | AttributeFormat::Vec4 => 16,
        }
    }
}

/// Bit flags describing how a compiled system touches an attribute over a frame.
///
/// The distinction the layout planner cares about is whether a stage needs the
/// *previous frame's* value while another stage writes the new one: that forces
/// a ping-pong (double) buffer (design §5.1, "read old / write new").
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct AttributeAccess(u8);

impl AttributeAccess {
    /// The attribute is read within the frame.
    pub const READ: Self = Self(1 << 0);
    /// The attribute is written within the frame.
    pub const WRITE: Self = Self(1 << 1);
    /// A stage reads the value produced by the *previous* frame (not the value
    /// written earlier this frame). Combined with `WRITE` this requires a
    /// ping-pong buffer.
    pub const READ_PREV: Self = Self(1 << 2);

    /// An empty access set (attribute untouched).
    #[must_use]
    pub const fn empty() -> Self {
        Self(0)
    }

    /// The union of two access sets.
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Whether every bit in `flags` is present.
    #[must_use]
    pub const fn contains(self, flags: Self) -> bool {
        (self.0 & flags.0) == flags.0
    }

    /// Whether the set is empty (attribute is neither read nor written).
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Whether the attribute must be double-buffered: a previous-frame read
    /// coexists with a write this frame.
    #[must_use]
    pub const fn needs_ping_pong(self) -> bool {
        self.contains(Self::READ_PREV) && self.contains(Self::WRITE)
    }
}

/// A declared usage of one attribute by a compiled system.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct AttributeUsage {
    /// The semantic channel.
    pub semantic: AttributeSemantic,
    /// The storage format (for a built-in, typically its `default_format`).
    pub format: AttributeFormat,
    /// How the system touches it over the frame.
    pub access: AttributeAccess,
}

impl AttributeUsage {
    /// Declares a usage with an explicit format.
    #[must_use]
    pub const fn new(
        semantic: AttributeSemantic,
        format: AttributeFormat,
        access: AttributeAccess,
    ) -> Self {
        Self {
            semantic,
            format,
            access,
        }
    }
}

/// A single planned attribute buffer in the final layout.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PlannedAttribute {
    /// Which semantic this buffer backs.
    pub semantic: AttributeSemantic,
    /// The element format.
    pub format: AttributeFormat,
    /// The per-element array stride actually allocated (`std430`).
    pub stride: u32,
    /// The number of device copies: `2` for ping-pong, otherwise `1`.
    pub copies: u32,
    /// The byte offset of this attribute's (first copy) region within the pool's
    /// total allocation. Regions are laid out in deterministic ordinal order and
    /// never overlap.
    pub byte_offset: u64,
    /// The total bytes this attribute occupies across all copies.
    pub byte_size: u64,
}

/// The complete `SoA` layout plan for a pool of `capacity` particles.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AttributeLayoutPlan {
    /// The capacity (element count) every attribute buffer is sized for.
    pub capacity: u32,
    /// The planned buffers in deterministic ordinal order.
    pub attributes: Vec<PlannedAttribute>,
    /// The sum of every attribute's `byte_size` (device footprint).
    pub total_bytes: u64,
}

impl AttributeLayoutPlan {
    /// Builds a deterministic `SoA` layout from a set of attribute usages.
    ///
    /// Usages whose access set is empty are dropped (unused semantics cost zero,
    /// design §5.1). When the same semantic appears more than once its accesses
    /// are unioned and the widest format wins, so callers may accumulate usages
    /// per stage without pre-merging. Output is sorted by
    /// [`AttributeSemantic::ordinal`], giving a stable, offset-monotone layout.
    #[must_use]
    pub fn build(capacity: u32, usages: &[AttributeUsage]) -> Self {
        // Merge duplicate semantics: union access, keep the widest stride.
        let mut merged: Vec<AttributeUsage> = Vec::new();
        for usage in usages {
            if usage.access.is_empty() {
                continue;
            }
            if let Some(existing) = merged.iter_mut().find(|u| u.semantic == usage.semantic) {
                existing.access = existing.access.union(usage.access);
                if usage.format.std430_stride() > existing.format.std430_stride() {
                    existing.format = usage.format;
                }
            } else {
                merged.push(*usage);
            }
        }
        merged.sort_by_key(|u| u.semantic.ordinal());

        let mut attributes = Vec::with_capacity(merged.len());
        let mut offset: u64 = 0;
        for usage in &merged {
            let stride = usage.format.std430_stride();
            let copies = if usage.access.needs_ping_pong() { 2 } else { 1 };
            // `u64` math keeps the footprint exact even at the compile-time
            // capacity ceiling, so a large pool never overflows the accumulator.
            let per_copy = u64::from(stride) * u64::from(capacity);
            let byte_size = per_copy * u64::from(copies);
            attributes.push(PlannedAttribute {
                semantic: usage.semantic,
                format: usage.format,
                stride,
                copies,
                byte_offset: offset,
                byte_size,
            });
            offset += byte_size;
        }

        Self {
            capacity,
            attributes,
            total_bytes: offset,
        }
    }

    /// Looks up a planned attribute by semantic, if present.
    #[must_use]
    pub fn get(&self, semantic: AttributeSemantic) -> Option<&PlannedAttribute> {
        self.attributes.iter().find(|a| a.semantic == semantic)
    }

    /// Whether a semantic was allocated at all.
    #[must_use]
    pub fn contains(&self, semantic: AttributeSemantic) -> bool {
        self.get(semantic).is_some()
    }

    /// The number of attribute buffers that are double-buffered.
    #[must_use]
    pub fn ping_pong_count(&self) -> usize {
        self.attributes.iter().filter(|a| a.copies == 2).count()
    }

    /// The bytes streamed for one full-pool update pass: every read source plus
    /// every write destination, one copy each (ping-pong reads the *other*
    /// copy, so it is still one read + one write). This is the per-frame
    /// bandwidth estimate the budget planner uses (design §27).
    #[must_use]
    pub fn frame_bandwidth_bytes(&self, usages: &[AttributeUsage]) -> u64 {
        let mut total: u64 = 0;
        for planned in &self.attributes {
            let Some(usage) = usages.iter().find(|u| u.semantic == planned.semantic) else {
                continue;
            };
            let per_copy = u64::from(planned.stride) * u64::from(self.capacity);
            let reads = usage.access.contains(AttributeAccess::READ) || planned.copies == 2;
            let writes = usage.access.contains(AttributeAccess::WRITE);
            if reads {
                total += per_copy;
            }
            if writes {
                total += per_copy;
            }
        }
        total
    }
}

/// The set of shading-input attributes a shading model consumes, so the compiler
/// only allocates the shading buffers a lit model actually needs (design §5.1,
/// §16). `Unlit` needs none; `PBR` needs the full physical set; `NPR` needs the
/// stylized subset; `Custom`/`Hybrid` conservatively request the union so a
/// user closure never reads an unallocated buffer.
#[must_use]
pub fn shading_input_attributes(model: EmberShadingModel) -> Vec<AttributeSemantic> {
    match model {
        EmberShadingModel::Unlit => Vec::new(),
        EmberShadingModel::Pbr => alloc::vec![
            AttributeSemantic::Normal,
            AttributeSemantic::Roughness,
            AttributeSemantic::Metallic,
            AttributeSemantic::MaterialId,
        ],
        EmberShadingModel::Npr => alloc::vec![
            AttributeSemantic::Normal,
            AttributeSemantic::MaterialId,
            AttributeSemantic::ShadingParams,
        ],
        EmberShadingModel::Custom(_) => full_shading_set(),
        EmberShadingModel::Hybrid { base, overlay, .. } => {
            let mut set = basis_shading_attributes(base);
            for semantic in basis_shading_attributes(overlay) {
                if !set.contains(&semantic) {
                    set.push(semantic);
                }
            }
            set.sort_by_key(|s| s.ordinal());
            set
        }
    }
}

/// The shading attributes a single [`ShadingBasis`] lobe consumes.
#[must_use]
fn basis_shading_attributes(basis: ShadingBasis) -> Vec<AttributeSemantic> {
    match basis {
        ShadingBasis::Unlit => Vec::new(),
        ShadingBasis::Pbr => alloc::vec![
            AttributeSemantic::Normal,
            AttributeSemantic::Roughness,
            AttributeSemantic::Metallic,
            AttributeSemantic::MaterialId,
        ],
        ShadingBasis::Npr => alloc::vec![
            AttributeSemantic::Normal,
            AttributeSemantic::MaterialId,
            AttributeSemantic::ShadingParams,
        ],
        ShadingBasis::Custom(_) => full_shading_set(),
    }
}

/// The conservative union of every shading-input attribute.
#[must_use]
fn full_shading_set() -> Vec<AttributeSemantic> {
    alloc::vec![
        AttributeSemantic::Normal,
        AttributeSemantic::Tangent,
        AttributeSemantic::Roughness,
        AttributeSemantic::Metallic,
        AttributeSemantic::MaterialId,
        AttributeSemantic::ShadingParams,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    const R: AttributeAccess = AttributeAccess::READ;
    const W: AttributeAccess = AttributeAccess::WRITE;

    fn usage(s: AttributeSemantic, access: AttributeAccess) -> AttributeUsage {
        let format = s.default_format().unwrap_or(AttributeFormat::F32);
        AttributeUsage::new(s, format, access)
    }

    #[test]
    fn format_sizes_and_strides_are_correct() {
        assert_eq!(AttributeFormat::F32.packed_size(), 4);
        assert_eq!(AttributeFormat::U32.packed_size(), 4);
        assert_eq!(AttributeFormat::Vec2.packed_size(), 8);
        assert_eq!(AttributeFormat::Vec3.packed_size(), 12);
        assert_eq!(AttributeFormat::Vec4.packed_size(), 16);
        // vec3 array element rounds up to 16 under std430.
        assert_eq!(AttributeFormat::Vec3.std430_stride(), 16);
        assert_eq!(AttributeFormat::Vec4.std430_stride(), 16);
        assert_eq!(AttributeFormat::F32.std430_stride(), 4);
    }

    #[test]
    fn empty_usage_set_allocates_nothing() {
        let plan = AttributeLayoutPlan::build(1024, &[]);
        assert!(plan.attributes.is_empty());
        assert_eq!(plan.total_bytes, 0);
        assert_eq!(plan.ping_pong_count(), 0);
    }

    #[test]
    fn untouched_attributes_cost_zero() {
        let usages = [
            usage(AttributeSemantic::Position, R.union(W)),
            usage(AttributeSemantic::Normal, AttributeAccess::empty()),
        ];
        let plan = AttributeLayoutPlan::build(64, &usages);
        assert!(plan.contains(AttributeSemantic::Position));
        assert!(!plan.contains(AttributeSemantic::Normal));
        assert_eq!(plan.attributes.len(), 1);
    }

    #[test]
    fn layout_is_ordinal_sorted_and_offsets_are_monotone_non_overlapping() {
        // Deliberately out of order on input.
        let usages = [
            usage(AttributeSemantic::Color, W),
            usage(AttributeSemantic::Position, W),
            usage(AttributeSemantic::Age, W),
        ];
        let plan = AttributeLayoutPlan::build(100, &usages);
        let order: Vec<_> = plan.attributes.iter().map(|a| a.semantic).collect();
        assert_eq!(
            order,
            alloc::vec![
                AttributeSemantic::Position,
                AttributeSemantic::Age,
                AttributeSemantic::Color,
            ]
        );
        // Offsets are monotone and each region ends exactly where the next
        // begins (no overlap, no gap).
        let mut cursor = 0u64;
        for attr in &plan.attributes {
            assert_eq!(attr.byte_offset, cursor);
            cursor += attr.byte_size;
        }
        assert_eq!(plan.total_bytes, cursor);
    }

    #[test]
    fn ping_pong_only_for_prev_read_plus_write() {
        let usages = [
            // read-prev + write => ping-pong.
            usage(
                AttributeSemantic::Position,
                AttributeAccess::READ_PREV.union(W),
            ),
            // in-place read + write => single copy.
            usage(AttributeSemantic::Velocity, R.union(W)),
            // read-prev only, no write => single copy.
            usage(AttributeSemantic::Age, AttributeAccess::READ_PREV),
        ];
        let plan = AttributeLayoutPlan::build(10, &usages);
        assert_eq!(plan.ping_pong_count(), 1);
        assert_eq!(plan.get(AttributeSemantic::Position).unwrap().copies, 2);
        assert_eq!(plan.get(AttributeSemantic::Velocity).unwrap().copies, 1);
        assert_eq!(plan.get(AttributeSemantic::Age).unwrap().copies, 1);
    }

    #[test]
    fn duplicate_semantics_union_access_and_widen_format() {
        let usages = [
            AttributeUsage::new(AttributeSemantic::Custom(7), AttributeFormat::F32, R),
            AttributeUsage::new(AttributeSemantic::Custom(7), AttributeFormat::Vec4, W),
        ];
        let plan = AttributeLayoutPlan::build(8, &usages);
        assert_eq!(plan.attributes.len(), 1);
        let custom = plan.get(AttributeSemantic::Custom(7)).unwrap();
        // Widest format wins (vec4 stride 16) and access is the union.
        assert_eq!(custom.stride, 16);
        assert_eq!(custom.byte_size, 16 * 8);
    }

    #[test]
    fn custom_attributes_sort_after_builtins_by_id() {
        assert!(
            AttributeSemantic::Custom(0).ordinal() > AttributeSemantic::ShadingParams.ordinal()
        );
        assert!(AttributeSemantic::Custom(1).ordinal() > AttributeSemantic::Custom(0).ordinal());
        // Saturating add keeps the ordinal well-defined at the ceiling.
        assert!(
            AttributeSemantic::Custom(u32::MAX).ordinal() >= AttributeSemantic::Custom(0).ordinal()
        );
    }

    #[test]
    fn byte_size_uses_std430_stride_and_capacity() {
        let usages = [usage(AttributeSemantic::Position, W)]; // vec3 -> stride 16
        let plan = AttributeLayoutPlan::build(1000, &usages);
        assert_eq!(plan.total_bytes, 16 * 1000);
    }

    #[test]
    fn large_capacity_does_not_overflow_the_footprint() {
        // vec4 * u32::MAX must fit in u64 without wrapping.
        let usages = [usage(AttributeSemantic::Color, W)];
        let plan = AttributeLayoutPlan::build(u32::MAX, &usages);
        assert_eq!(plan.total_bytes, 16 * u64::from(u32::MAX));
    }

    #[test]
    fn unlit_needs_no_shading_attributes() {
        assert!(shading_input_attributes(EmberShadingModel::Unlit).is_empty());
    }

    #[test]
    fn pbr_requests_the_physical_shading_set() {
        let set = shading_input_attributes(EmberShadingModel::Pbr);
        assert!(set.contains(&AttributeSemantic::Normal));
        assert!(set.contains(&AttributeSemantic::Roughness));
        assert!(set.contains(&AttributeSemantic::Metallic));
        assert!(!set.contains(&AttributeSemantic::Tangent));
    }

    #[test]
    fn hybrid_unions_both_bases_and_is_sorted() {
        let model = EmberShadingModel::Hybrid {
            base: ShadingBasis::Pbr,
            overlay: ShadingBasis::Npr,
            weight: 0.5,
        };
        let set = shading_input_attributes(model);
        // Union contains PBR-only (roughness) and NPR-only (shading params).
        assert!(set.contains(&AttributeSemantic::Roughness));
        assert!(set.contains(&AttributeSemantic::ShadingParams));
        // Deduplicated and sorted by ordinal.
        let mut sorted = set.clone();
        sorted.sort_by_key(|s| s.ordinal());
        assert_eq!(set, sorted);
        let unique: Vec<_> = {
            let mut v = set.clone();
            v.dedup();
            v
        };
        assert_eq!(unique.len(), set.len());
    }

    #[test]
    fn all_builtins_have_a_default_format() {
        let builtins = [
            AttributeSemantic::Position,
            AttributeSemantic::Velocity,
            AttributeSemantic::Age,
            AttributeSemantic::Lifetime,
            AttributeSemantic::Color,
            AttributeSemantic::Size,
            AttributeSemantic::Rotation,
            AttributeSemantic::Scale,
            AttributeSemantic::Alive,
            AttributeSemantic::ParticleId,
            AttributeSemantic::RibbonId,
            AttributeSemantic::SortKey,
            AttributeSemantic::Normal,
            AttributeSemantic::Tangent,
            AttributeSemantic::Emissive,
            AttributeSemantic::Roughness,
            AttributeSemantic::Metallic,
            AttributeSemantic::MaterialId,
            AttributeSemantic::ShadingParams,
        ];
        for b in builtins {
            assert!(b.default_format().is_some(), "{b:?} lacks a default format");
        }
        assert!(AttributeSemantic::Custom(0).default_format().is_none());
    }

    #[test]
    fn bandwidth_counts_reads_and_writes() {
        // One vec3 position read+write (in place) at capacity 100, stride 16.
        let usages = [usage(AttributeSemantic::Position, R.union(W))];
        let plan = AttributeLayoutPlan::build(100, &usages);
        // read (1600) + write (1600).
        assert_eq!(plan.frame_bandwidth_bytes(&usages), 16 * 100 * 2);
    }

    #[test]
    fn write_only_attribute_counts_bandwidth_once() {
        let usages = [usage(AttributeSemantic::SortKey, W)]; // u32 stride 4
        let plan = AttributeLayoutPlan::build(50, &usages);
        assert_eq!(plan.frame_bandwidth_bytes(&usages), 4 * 50);
    }

    #[test]
    fn access_flag_algebra() {
        assert!(AttributeAccess::empty().is_empty());
        let rw = R.union(W);
        assert!(rw.contains(R));
        assert!(rw.contains(W));
        assert!(!rw.needs_ping_pong());
        assert!(AttributeAccess::READ_PREV.union(W).needs_ping_pong());
    }
}
