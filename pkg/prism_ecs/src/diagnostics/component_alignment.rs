//! Per-component column alignment and SIMD-packing readiness census
//! (design §17 列对齐 + SIMD / §16.6).
//!
//! The design's performance thesis (§17) rests on *chunked SoA* storage: hot
//! [`Table`](crate::component::StorageType::Table) components live in columnar
//! [`BlobVec`](crate::storage) runs that are iterated — and ideally
//! *vectorised* — a chunk at a time. Vectorisation (`core::simd`, design §7
//! `simd_iter`) is only natural when a column's elements sit on wide-enough
//! boundaries: a 128-bit lane wants 16-byte alignment, a 256-bit lane 32, a
//! 512-bit lane 64. A hot component declared with a scalar-aligned layout
//! silently falls back to the scalar path.
//!
//! Two sibling reports already touch layout, but neither answers the
//! per-component SIMD-readiness question:
//!
//! * [`storage_distribution`](super::storage_distribution) folds alignment into
//!   a single [`max_align`](super::storage_distribution::StorageBucketEntry::max_align)
//!   *per storage bucket* — it cannot say which individual component is
//!   mis-aligned, nor how the registry's alignments are distributed.
//! * [`component_memory`](super::component_memory) sums each component's
//!   [`Layout::size`](core::alloc::Layout::size) *with no padding* — it is a
//!   deliberately padding-free resident-byte model and so is blind to the
//!   column stride a non-size-multiple layout actually costs.
//!
//! This report takes the per-component, layout-first view: for every registered
//! component it records element [`size`](ComponentAlignmentEntry::size),
//! [`align`](ComponentAlignmentEntry::align), the packed column
//! [`stride`](ComponentAlignmentEntry::stride)
//! ([`Layout::pad_to_align`](core::alloc::Layout::pad_to_align)) and the
//! per-element [`padding`](ComponentAlignmentEntry::padding) it wastes, flags
//! whether it is [`simd_aligned`](ComponentAlignmentEntry::simd_aligned), and
//! rolls the registry up into an alignment histogram plus a SIMD-ready share.
//! An editor layout panel or a CI gate reads it to spot the hot component one
//! `#[repr(align(…))]` away from a vectorised column, or the dynamically
//! registered layout (design §16.2) whose odd size bleeds padding into every
//! row.
//!
//! # Honest scope
//! Figures describe each component type's *intrinsic* [`Layout`], the floor the
//! storage layer builds on. A real chunk column may be over-aligned beyond the
//! type's own alignment (an orthogonal storage decision); this census measures
//! the type-level readiness that over-alignment cannot manufacture — a column
//! of a 4-byte-aligned element is not safely loadable as a 16-byte vector
//! however the column base is aligned. It needs only the component
//! [`registry`](Components): it describes *types*, not live instances, so it is
//! well-defined on an entity-less world.
//!
//! Everything is deterministic (design §14): entries are sorted by ascending
//! component id and histogram buckets by ascending alignment, independent of
//! registration or hashing order.

use alloc::string::String;
use alloc::vec::Vec;

use crate::collections::HashMap;
use crate::component::{ComponentId, Components};

/// Minimum element alignment (bytes) for a column to be loadable as a
/// 128-bit SIMD lane — the narrowest `core::simd` width the design targets
/// (§7 / §17). Components aligned to at least this are counted
/// [`simd_aligned`](ComponentAlignmentEntry::simd_aligned).
pub const SIMD_LANE_BYTES: usize = 16;

/// Integer permille (`parts per thousand`) of `num / den`, returning `0` when
/// `den` is zero.
#[inline]
fn permille(num: u64, den: u64) -> u64 {
    (num * 1000).checked_div(den).unwrap_or(0)
}

/// One component type's column layout: element size, alignment, packed stride,
/// per-element padding and SIMD-readiness verdict (design §17).
#[derive(Clone, Debug)]
pub struct ComponentAlignmentEntry {
    /// The component this entry describes.
    pub component: ComponentId,
    /// Registered name of the component (from
    /// [`ComponentInfo::name`](crate::component::ComponentInfo::name)).
    pub name: String,
    /// Element size in bytes ([`Layout::size`](core::alloc::Layout::size)).
    /// `0` marks a zero-sized tag component.
    pub size: usize,
    /// Element alignment in bytes ([`Layout::align`](core::alloc::Layout::align)).
    pub align: usize,
    /// Packed column stride in bytes
    /// ([`Layout::pad_to_align`](core::alloc::Layout::pad_to_align)): the span
    /// one element actually occupies in a tightly packed column, `size` rounded
    /// up to `align`.
    pub stride: usize,
    /// Per-element padding wasted in the column (`stride - size`). Always `0`
    /// for a native Rust layout (whose size is a multiple of its alignment);
    /// non-zero only for a dynamically registered layout (design §16.2) whose
    /// size is not a multiple of its alignment.
    pub padding: usize,
    /// Whether this is a zero-sized (marker / tag) component with no column
    /// payload.
    pub zero_sized: bool,
    /// Whether the element is aligned to at least [`SIMD_LANE_BYTES`], i.e. its
    /// column can be loaded as a 128-bit (or wider) SIMD lane. Always `false`
    /// for a zero-sized component.
    pub simd_aligned: bool,
}

/// One alignment class and how many registered components share it.
#[derive(Clone, Copy, Debug)]
pub struct ComponentAlignmentBucket {
    /// Element alignment in bytes shared by this bucket's components.
    pub align: usize,
    /// Number of registered components with exactly this alignment.
    pub component_count: usize,
}

/// Read-only census of every registered component's column alignment and
/// SIMD-packing readiness, plus an alignment histogram and SIMD-ready share
/// (design §17 / §16.6).
#[derive(Clone, Debug)]
pub struct ComponentAlignmentReport {
    component_count: usize,
    zero_sized_count: usize,
    max_align: usize,
    min_sized_align: usize,
    simd_aligned_count: usize,
    total_padding_bytes: usize,
    buckets: Vec<ComponentAlignmentBucket>,
    entries: Vec<ComponentAlignmentEntry>,
}

impl ComponentAlignmentReport {
    /// Build the census from a world via
    /// [`World::components`](crate::world::World::components).
    #[inline]
    pub fn capture(world: &crate::world::World) -> Self {
        Self::from_components(world.components())
    }

    /// Build the census directly from a component [`registry`](Components).
    ///
    /// Each registered component contributes one [`ComponentAlignmentEntry`];
    /// entries are returned sorted by ascending component id and the alignment
    /// histogram by ascending alignment (design §14).
    pub fn from_components(components: &Components) -> Self {
        let mut entries: Vec<ComponentAlignmentEntry> = Vec::new();
        let mut histogram: HashMap<usize, usize> = HashMap::default();
        let mut zero_sized_count = 0usize;
        let mut max_align = 0usize;
        let mut min_sized_align = usize::MAX;
        let mut simd_aligned_count = 0usize;
        let mut total_padding_bytes = 0usize;

        let len = components.len();
        for i in 0..len {
            let id = ComponentId::new(i as u32);
            let Some(info) = components.info(id) else {
                continue;
            };
            let layout = info.layout();
            let size = layout.size();
            let align = layout.align();
            let stride = layout.pad_to_align().size();
            let padding = stride - size;
            let zero_sized = size == 0;
            let simd_aligned = !zero_sized && align >= SIMD_LANE_BYTES;

            total_padding_bytes += padding;
            if align > max_align {
                max_align = align;
            }
            if !zero_sized {
                if align < min_sized_align {
                    min_sized_align = align;
                }
                if simd_aligned {
                    simd_aligned_count += 1;
                }
            } else {
                zero_sized_count += 1;
            }
            *histogram.entry(align).or_insert(0) += 1;

            entries.push(ComponentAlignmentEntry {
                component: id,
                name: String::from(info.name()),
                size,
                align,
                stride,
                padding,
                zero_sized,
                simd_aligned,
            });
        }

        entries.sort_unstable_by_key(|entry| entry.component.index());

        let mut buckets: Vec<ComponentAlignmentBucket> = histogram
            .into_iter()
            .map(|(align, component_count)| ComponentAlignmentBucket {
                align,
                component_count,
            })
            .collect();
        buckets.sort_unstable_by_key(|bucket| bucket.align);

        let component_count = entries.len();
        let min_sized_align = if min_sized_align == usize::MAX {
            0
        } else {
            min_sized_align
        };

        Self {
            component_count,
            zero_sized_count,
            max_align,
            min_sized_align,
            simd_aligned_count,
            total_padding_bytes,
            buckets,
            entries,
        }
    }

    /// Total number of registered components described.
    #[inline]
    pub fn component_count(&self) -> usize {
        self.component_count
    }

    /// Whether the registry held no components.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.component_count == 0
    }

    /// Number of zero-sized (marker / tag) components, which carry no column
    /// payload and are never SIMD-aligned.
    #[inline]
    pub fn zero_sized_count(&self) -> usize {
        self.zero_sized_count
    }

    /// Number of sized (non-zero-sized) components — those with an actual
    /// column payload.
    #[inline]
    pub fn sized_count(&self) -> usize {
        self.component_count - self.zero_sized_count
    }

    /// Largest element alignment (bytes) across all components, `0` on an
    /// empty registry.
    #[inline]
    pub fn max_align(&self) -> usize {
        self.max_align
    }

    /// Smallest alignment (bytes) among *sized* components, `0` when there are
    /// none. The narrowest-aligned payload column — a scalar-fallback suspect.
    #[inline]
    pub fn min_sized_align(&self) -> usize {
        self.min_sized_align
    }

    /// Number of sized components aligned to at least [`SIMD_LANE_BYTES`]
    /// (vectorisable columns).
    #[inline]
    pub fn simd_aligned_count(&self) -> usize {
        self.simd_aligned_count
    }

    /// Permille (parts per thousand) of *sized* components that are
    /// SIMD-aligned, `0` when there are no sized components.
    #[inline]
    pub fn simd_aligned_permille(&self) -> u64 {
        permille(self.simd_aligned_count as u64, self.sized_count() as u64)
    }

    /// Whether every sized component is SIMD-aligned (and at least one exists).
    #[inline]
    pub fn all_simd_aligned(&self) -> bool {
        self.sized_count() > 0 && self.simd_aligned_count == self.sized_count()
    }

    /// Total per-element column padding wasted across all components. Non-zero
    /// only when a dynamically registered layout (design §16.2) has a size that
    /// is not a multiple of its alignment.
    #[inline]
    pub fn total_padding_bytes(&self) -> usize {
        self.total_padding_bytes
    }

    /// Per-component entries, sorted by ascending component id.
    #[inline]
    pub fn entries(&self) -> &[ComponentAlignmentEntry] {
        &self.entries
    }

    /// Alignment histogram buckets, sorted by ascending alignment.
    #[inline]
    pub fn buckets(&self) -> &[ComponentAlignmentBucket] {
        &self.buckets
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::component::{Component, StorageType};
    use alloc::vec;
    use core::alloc::Layout;

    #[derive(Debug)]
    struct Tag;
    impl Component for Tag {}

    #[derive(Debug)]
    struct Small {
        _a: u32,
    }
    impl Component for Small {}

    #[derive(Debug)]
    struct Wide {
        _a: u64,
    }
    impl Component for Wide {}

    #[repr(align(16))]
    #[derive(Debug)]
    struct Simd16 {
        _a: [u8; 16],
    }
    impl Component for Simd16 {}

    #[test]
    fn empty_registry_is_empty() {
        let components = Components::new();
        let report = ComponentAlignmentReport::from_components(&components);
        assert!(report.is_empty());
        assert_eq!(report.component_count(), 0);
        assert_eq!(report.sized_count(), 0);
        assert_eq!(report.max_align(), 0);
        assert_eq!(report.min_sized_align(), 0);
        assert_eq!(report.simd_aligned_count(), 0);
        assert_eq!(report.simd_aligned_permille(), 0);
        assert!(!report.all_simd_aligned());
        assert!(report.entries().is_empty());
        assert!(report.buckets().is_empty());
    }

    #[test]
    fn zero_sized_tag_classified() {
        let mut components = Components::new();
        let id = components.register::<Tag>();
        let report = ComponentAlignmentReport::from_components(&components);
        assert_eq!(report.component_count(), 1);
        assert_eq!(report.zero_sized_count(), 1);
        assert_eq!(report.sized_count(), 0);
        let entry = &report.entries()[0];
        assert_eq!(entry.component, id);
        assert_eq!(entry.size, 0);
        assert!(entry.zero_sized);
        assert!(!entry.simd_aligned);
        assert_eq!(entry.stride, 0);
        assert_eq!(entry.padding, 0);
        // A zero-sized tag never counts toward the SIMD-ready share.
        assert_eq!(report.simd_aligned_permille(), 0);
    }

    #[test]
    fn size_and_align_recorded() {
        let mut components = Components::new();
        components.register::<Small>();
        components.register::<Wide>();
        let report = ComponentAlignmentReport::from_components(&components);
        assert_eq!(report.component_count(), 2);
        assert_eq!(report.sized_count(), 2);
        assert_eq!(report.max_align(), 8);
        assert_eq!(report.min_sized_align(), 4);

        let small = report.entries().iter().find(|e| e.align == 4).unwrap();
        assert_eq!(small.size, 4);
        assert_eq!(small.stride, 4);
        assert_eq!(small.padding, 0);
        assert!(!small.simd_aligned);

        let wide = report.entries().iter().find(|e| e.align == 8).unwrap();
        assert_eq!(wide.size, 8);
        assert!(!wide.simd_aligned);
    }

    #[test]
    fn simd_aligned_component_detected() {
        let mut components = Components::new();
        components.register::<Small>(); // align 4, not SIMD.
        components.register::<Simd16>(); // align 16, SIMD-ready.
        let report = ComponentAlignmentReport::from_components(&components);
        assert_eq!(report.sized_count(), 2);
        assert_eq!(report.simd_aligned_count(), 1);
        assert_eq!(report.max_align(), 16);
        assert_eq!(report.min_sized_align(), 4);
        assert_eq!(report.simd_aligned_permille(), 500);
        assert!(!report.all_simd_aligned());

        let simd = report
            .entries()
            .iter()
            .find(|e| e.align >= SIMD_LANE_BYTES)
            .unwrap();
        assert!(simd.simd_aligned);
        assert_eq!(simd.size, 16);
        assert_eq!(simd.align, 16);
        assert_eq!(simd.stride, 16);
    }

    #[test]
    fn all_simd_aligned_when_only_wide() {
        let mut components = Components::new();
        components.register::<Simd16>();
        let report = ComponentAlignmentReport::from_components(&components);
        assert!(report.all_simd_aligned());
        assert_eq!(report.simd_aligned_permille(), 1000);
    }

    #[test]
    fn histogram_counts_alignment_classes() {
        let mut components = Components::new();
        components.register::<Small>(); // align 4
        components.register::<Wide>(); // align 8
        components.register::<Simd16>(); // align 16
        components.register::<Tag>(); // align 1 (zero-sized)
        let report = ComponentAlignmentReport::from_components(&components);

        let aligns: Vec<usize> = report.buckets().iter().map(|b| b.align).collect();
        // Sorted ascending, one bucket per distinct alignment.
        assert_eq!(aligns, vec![1, 4, 8, 16]);
        for bucket in report.buckets() {
            assert_eq!(bucket.component_count, 1);
        }
        // Buckets partition the whole registry.
        let total: usize = report.buckets().iter().map(|b| b.component_count).sum();
        assert_eq!(total, report.component_count());
    }

    #[test]
    fn dynamic_layout_padding_counted() {
        // A dynamically registered layout whose size (12) is not a multiple of
        // its alignment (8) packs on a 16-byte stride, wasting 4 bytes/row —
        // waste component_memory's size-only model is blind to.
        let mut components = Components::new();
        let layout = Layout::from_size_align(12, 8).unwrap();
        let id = components.register_dynamic("Padded", layout, StorageType::Table, None);
        let report = ComponentAlignmentReport::from_components(&components);
        let entry = report
            .entries()
            .iter()
            .find(|e| e.component == id)
            .unwrap();
        assert_eq!(entry.size, 12);
        assert_eq!(entry.align, 8);
        assert_eq!(entry.stride, 16);
        assert_eq!(entry.padding, 4);
        assert_eq!(report.total_padding_bytes(), 4);
    }

    #[test]
    fn entries_sorted_by_component_id() {
        let mut components = Components::new();
        components.register::<Wide>();
        components.register::<Small>();
        components.register::<Simd16>();
        let report = ComponentAlignmentReport::from_components(&components);
        let ids: Vec<u32> = report.entries().iter().map(|e| e.component.index()).collect();
        let mut sorted = ids.clone();
        sorted.sort_unstable();
        assert_eq!(ids, sorted);
    }

    #[test]
    fn capture_from_world_matches_registry() {
        let mut world = crate::world::World::new();
        world.register_component::<Small>();
        world.register_component::<Simd16>();
        let report = ComponentAlignmentReport::capture(&world);
        assert_eq!(report.sized_count(), 2);
        assert_eq!(report.simd_aligned_count(), 1);
    }
}
