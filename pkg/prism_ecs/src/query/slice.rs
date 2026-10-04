//! Typed-column-slice parallel path (design §8.3 `jobs.par_chunks`).
//!
//! Where [`par`](crate::query::par) fans a query's rows out **one row at a
//! time** (`Fn(D::Item<'_>)`), this module hands each sub-job an aligned,
//! contiguous **typed column slice** (`&[T]` / `&mut [T]`) covering a whole run
//! of rows, so a DOTS-style `IJobChunk` body can run a SIMD kernel over the
//! slice in one pass (design §8.3's headline
//! `jobs.par_chunks(&mut q, |chunk| { /* SIMD 蒙皮 */ })`).
//!
//! # Why a separate `QueryData`-style trait
//!
//! The per-row [`QueryData`] yields a *scalar* item ([`Mut<T>`](crate::change::Mut)
//! with its own change ticks). A slice body wants the raw backing storage, so
//! this path uses two narrower traits that are deliberately *more* restrictive
//! than the row path, because handing out a whole `&mut [T]` loses the per-row
//! structure the row path relies on for soundness:
//!
//! * [`ColumnSliceData`] — only terms whose column is one contiguous typed run:
//!   `&T` / `&mut T` over [`Table`](crate::component::StorageType::Table)
//!   storage, and [`Entity`]. A sparse/shared component has no such run, so it
//!   is rejected (`assert_sliceable`). Change detection cannot be per-row once a
//!   bare slice is exposed, so a `&mut T` term conservatively stamps the
//!   *entire* handed-out range as changed up front.
//! * [`ArchetypalFilter`] — only filters whose verdict is uniform across an
//!   archetype (`()`, [`With`]/[`Without`] over table/shared storage, their
//!   [`Or`]/tuple combinations). The per-row change filters
//!   [`Added`](crate::query::Added)/[`Changed`](crate::query::Changed) and
//!   sparse membership are *not* archetypal, so they cannot be a `par_chunks`
//!   filter — enforced at compile time by the missing trait impl.
//!
//! # Soundness model
//!
//! The driver ([`par_chunks_raw`](crate::query::par::par_chunks_raw)) partitions
//! each archetype into **chunk-aligned** batches: every batch start is a
//! multiple of the table's `rows_per_chunk`, so each task owns *whole chunks*
//! (only the final batch may end inside the last, partial chunk — of which it is
//! still the sole owner). This is stronger than the row path's arbitrary
//! `batch_size` split and is exactly what makes the coarse per-chunk change
//! version a task-exclusive write: no two tasks ever touch rows in the same
//! chunk, so stamping the chunk version from several tasks never races.
//!
//! Within a batch, `&mut T` forms `&mut [T]` over `[start, end)` built directly
//! from the offset element base (see
//! [`Column::as_mut_slice_range`](crate::storage::Column::as_mut_slice_range)),
//! never by subslicing a whole-column `&mut [T]`, so disjoint ranges of one
//! column lent to different tasks never overlap. Distinct components in a tuple
//! cannot alias because [`QueryState::new`](crate::query::QueryState) already
//! rejects a term set that writes the same component twice (or `&mut A` with
//! `&A`).
//!
//! Enabled by the `multi_thread` feature.

use crate::archetype::Archetype;
use crate::change::Tick;
use crate::component::{Component, StorageType};
use crate::entity::Entity;
use crate::query::fetch::QueryData;
use crate::query::filter::{Or, QueryFilter, With, Without};

/// A [`QueryData`] term that can be exposed as a contiguous typed column slice
/// for the chunk-parallel slice path (design §8.3).
///
/// # Safety
/// Implementors must uphold:
///
/// * [`assert_sliceable`](ColumnSliceData::assert_sliceable) must panic unless
///   every component term is stored as one contiguous typed run — i.e.
///   [`StorageType::Table`]. A sparse/shared term has no such run and must be
///   rejected before any slice is formed.
/// * [`column_slice`](ColumnSliceData::column_slice) must build its slice only
///   from rows `[start, end)` of `archetype` and borrow only the components the
///   term's [`QueryData::update_access`] registered. For a `&mut T` term it must
///   form a unique `&mut [T]` over that range (the driver guarantees the range
///   is a task-exclusive run of rows) and must stamp the change ticks of every
///   row it hands out so change detection stays sound once the per-row `Mut`
///   wrapper is gone.
pub unsafe trait ColumnSliceData: QueryData {
    /// The typed slice yielded for one chunk-aligned batch, borrowing the world
    /// for `'w`.
    type Slice<'w>;

    /// Panic unless every term of this data is sliceable (table-backed, or
    /// [`Entity`]). Called once at the `par_chunks` entry point.
    fn assert_sliceable();

    /// Build the typed slice for rows `[start, end)` of `archetype`.
    ///
    /// # Safety
    /// * `start <= end <= archetype.len()`.
    /// * The caller holds the access `self`'s terms require over rows
    ///   `[start, end)` for the slice's lifetime, and no other task touches
    ///   those rows (the driver's chunk-aligned, disjoint partition).
    /// * [`assert_sliceable`](ColumnSliceData::assert_sliceable) has succeeded,
    ///   so every term is table-backed.
    unsafe fn column_slice<'w>(
        state: &Self::State,
        archetype: &'w Archetype,
        start: usize,
        end: usize,
        this_run: Tick,
    ) -> Self::Slice<'w>;
}

// SAFETY: `&T` reads one table component immutably; `assert_sliceable` rejects
// non-table storage, and `column_slice` forms a shared `&[T]` over the exact
// range from the matched column (guaranteed to exist by `QueryData::matches`).
unsafe impl<T: Component> ColumnSliceData for &T {
    type Slice<'w> = &'w [T];

    fn assert_sliceable() {
        assert!(
            matches!(T::STORAGE, StorageType::Table),
            "par_chunks requires table-backed components; `&{}` over sparse/shared storage (design §6) has no contiguous column slice",
            core::any::type_name::<T>(),
        );
    }

    unsafe fn column_slice<'w>(
        state: &Self::State,
        archetype: &'w Archetype,
        start: usize,
        end: usize,
        _this_run: Tick,
    ) -> Self::Slice<'w> {
        let col = archetype
            .table()
            .column(*state)
            .expect("matched archetype must contain the queried table column");
        // SAFETY: `T` is this column's type; `start <= end <= len`; the caller
        // holds shared access to the range for the slice's lifetime.
        unsafe { col.as_slice_range::<T>(start, end) }
    }
}

// SAFETY: `&mut T` writes one table component; `assert_sliceable` rejects
// non-table storage. `column_slice` stamps every row's changed + chunk-version
// tick with `this_run` *before* forming a unique `&mut [T]` over the exact
// range — the whole range is marked changed because the bare slice loses the
// per-row write information. The driver's chunk-aligned partition makes the
// range task-exclusive, so both the slice and the chunk-version stamps are
// race-free.
unsafe impl<T: Component> ColumnSliceData for &mut T {
    type Slice<'w> = &'w mut [T];

    fn assert_sliceable() {
        assert!(
            matches!(T::STORAGE, StorageType::Table),
            "par_chunks requires table-backed components; `&mut {}` over sparse/shared storage (design §6) has no contiguous column slice",
            core::any::type_name::<T>(),
        );
    }

    unsafe fn column_slice<'w>(
        state: &Self::State,
        archetype: &'w Archetype,
        start: usize,
        end: usize,
        this_run: Tick,
    ) -> Self::Slice<'w> {
        let col = archetype
            .table()
            .column(*state)
            .expect("matched archetype must contain the queried table column");
        // Conservatively mark the whole handed-out range changed: once a bare
        // `&mut [T]` is exposed we cannot know which rows the kernel writes, so
        // the fine per-row tick and the coarse per-chunk version are both
        // stamped with `this_run`.
        for row in start..end {
            // SAFETY: `row < end <= len`; the range is task-exclusive, so this
            // unique access to the row's changed-tick cell does not alias. The
            // chunk-version cell is only written with `this_run`, upholding its
            // monotone upper-bound invariant, and no sibling task writes any row
            // of this chunk.
            unsafe {
                *col.changed_tick_ptr(row) = this_run;
                *col.chunk_changed_ptr(row) = this_run;
            }
        }
        // SAFETY: `T` is this column's type; `start <= end <= len`; the caller
        // holds unique access to the range for the slice's lifetime.
        unsafe { col.as_mut_slice_range::<T>(start, end) }
    }
}

// SAFETY: `Entity` touches no component storage; the entity id column of a
// table is one contiguous `&[Entity]` run, always sliceable.
unsafe impl ColumnSliceData for Entity {
    type Slice<'w> = &'w [Entity];

    fn assert_sliceable() {}

    unsafe fn column_slice<'w>(
        _state: &Self::State,
        archetype: &'w Archetype,
        start: usize,
        end: usize,
        _this_run: Tick,
    ) -> Self::Slice<'w> {
        // `start <= end <= len` (caller contract) keeps this index in-bounds.
        &archetype.table().entities()[start..end]
    }
}

macro_rules! impl_column_slice_data_tuple {
    ($($T:ident),+) => {
        // SAFETY: each element is a `ColumnSliceData` upholding the trait
        // contract; the tuple slices each element over the same `[start, end)`
        // range of the same archetype. Distinct components cannot alias because
        // `QueryState::new` rejects a duplicate-`&mut` term set, so the per-
        // element soundness arguments compose.
        #[allow(non_snake_case)]
        unsafe impl<$($T: ColumnSliceData),+> ColumnSliceData for ($($T,)+) {
            type Slice<'w> = ($($T::Slice<'w>,)+);

            fn assert_sliceable() {
                $($T::assert_sliceable();)+
            }

            unsafe fn column_slice<'w>(
                state: &Self::State,
                archetype: &'w Archetype,
                start: usize,
                end: usize,
                this_run: Tick,
            ) -> Self::Slice<'w> {
                let ($($T,)+) = state;
                // SAFETY: forwarded — same range/archetype for every element;
                // distinct components so the `&mut` slices do not alias.
                unsafe { ($($T::column_slice($T, archetype, start, end, this_run),)+) }
            }
        }
    };
}

impl_column_slice_data_tuple!(A);
impl_column_slice_data_tuple!(A, B);
impl_column_slice_data_tuple!(A, B, C);
impl_column_slice_data_tuple!(A, B, C, D);
impl_column_slice_data_tuple!(A, B, C, D, E);
impl_column_slice_data_tuple!(A, B, C, D, E, F);
impl_column_slice_data_tuple!(A, B, C, D, E, F, G);
impl_column_slice_data_tuple!(A, B, C, D, E, F, G, H);
impl_column_slice_data_tuple!(A, B, C, D, E, F, G, H, I);
impl_column_slice_data_tuple!(A, B, C, D, E, F, G, H, I, J);
impl_column_slice_data_tuple!(A, B, C, D, E, F, G, H, I, J, K);
impl_column_slice_data_tuple!(A, B, C, D, E, F, G, H, I, J, K, L);

/// A [`QueryFilter`] whose match verdict is uniform across a whole archetype,
/// so it can gate the chunk-parallel slice path without a per-row test.
///
/// The slice path does **not** run [`QueryFilter::filter_fetch`] per row — it
/// trusts the archetype-level [`QueryFilter::matches`] decision already applied
/// by [`QueryState::matched_archetypes`](crate::query::QueryState). That is only
/// correct for filters whose per-row verdict equals their per-archetype verdict:
/// `()`, and [`With`]/[`Without`] over table/shared storage (where membership is
/// an archetype property), plus their [`Or`] and tuple combinations.
///
/// The per-row change filters [`Added`](crate::query::Added) /
/// [`Changed`](crate::query::Changed) and sparse-set membership are *not*
/// archetypal, so they intentionally do not implement this trait and cannot be
/// used as a `par_chunks` filter (a compile error, not a silent miss).
///
/// # Safety
/// Implementors must guarantee that for every archetype their
/// [`QueryFilter::matches`] admits, every row of that archetype also satisfies
/// the filter — i.e. skipping the per-row [`QueryFilter::filter_fetch`] never
/// admits a row the serial iterator would have skipped.
pub unsafe trait ArchetypalFilter: QueryFilter {
    /// Panic unless the filter is archetype-uniform (rejecting a sparse-backed
    /// [`With`]/[`Without`], whose membership is per entity). Called once at the
    /// `par_chunks` entry point.
    fn assert_archetypal();
}

// SAFETY: the empty filter admits every archetype and every row.
unsafe impl ArchetypalFilter for () {
    fn assert_archetypal() {}
}

// SAFETY: for table/shared storage `With<T>` membership is an archetype
// property — `matches` and the per-row gate agree — so skipping the per-row
// test is exact. `assert_archetypal` rejects sparse storage, whose membership
// is per entity.
unsafe impl<T: Component> ArchetypalFilter for With<T> {
    fn assert_archetypal() {
        assert!(
            !matches!(T::STORAGE, StorageType::SparseSet),
            "par_chunks filters must be archetype-uniform; `With<{}>` over sparse storage (design §6) is a per-entity test",
            core::any::type_name::<T>(),
        );
    }
}

// SAFETY: as `With<T>` — table/shared absence is an archetype property; sparse
// storage is rejected.
unsafe impl<T: Component> ArchetypalFilter for Without<T> {
    fn assert_archetypal() {
        assert!(
            !matches!(T::STORAGE, StorageType::SparseSet),
            "par_chunks filters must be archetype-uniform; `Without<{}>` over sparse storage (design §6) is a per-entity test",
            core::any::type_name::<T>(),
        );
    }
}

macro_rules! impl_archetypal_filter_tuple {
    ($($F:ident),+) => {
        // SAFETY: an AND of archetype-uniform filters is archetype-uniform: a
        // row of an admitted archetype satisfies every conjunct, so it satisfies
        // the conjunction.
        #[allow(non_snake_case)]
        unsafe impl<$($F: ArchetypalFilter),+> ArchetypalFilter for ($($F,)+) {
            fn assert_archetypal() {
                $($F::assert_archetypal();)+
            }
        }

        // SAFETY: an OR of archetype-uniform filters is archetype-uniform. Each
        // branch's per-row verdict equals its per-archetype verdict, so the
        // per-row OR equals the per-archetype OR that `matches` already applied
        // when selecting the archetype.
        #[allow(non_snake_case)]
        unsafe impl<$($F: ArchetypalFilter),+> ArchetypalFilter for Or<($($F,)+)> {
            fn assert_archetypal() {
                $($F::assert_archetypal();)+
            }
        }
    };
}

impl_archetypal_filter_tuple!(A);
impl_archetypal_filter_tuple!(A, B);
impl_archetypal_filter_tuple!(A, B, C);
impl_archetypal_filter_tuple!(A, B, C, D);
impl_archetypal_filter_tuple!(A, B, C, D, E);
impl_archetypal_filter_tuple!(A, B, C, D, E, F);
impl_archetypal_filter_tuple!(A, B, C, D, E, F, G);
impl_archetypal_filter_tuple!(A, B, C, D, E, F, G, H);
