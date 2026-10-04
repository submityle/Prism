//! Field-level dirty tracking (design §24.5).
//!
//! A [`DirtyMask`] is a compact, growable bitset indexed by a struct's
//! positional field index (the same index space used by
//! [`Struct::field_at`](crate::Struct::field_at) and
//! [`AccessPlan`](crate::AccessPlan)). The replication layer sets a bit when a
//! field is written and later asks the mask which fields changed, so only those
//! fields are encoded into a delta — the "每实体每帧只序列化变化字段" story of
//! the design.
//!
//! Two routes populate a mask:
//!
//! 1. **Explicit marking.** A system that mutates a field calls
//!    [`DirtyMask::mark`] with the field index (ECS change detection, an editor
//!    edit, a gameplay tick).
//! 2. **Diff-derived marking.** [`DirtyMask::changed`] compares a previous
//!    snapshot against the current value and marks exactly the fields that
//!    moved, reusing the reflection [`diff`](crate::diff) so nested structural
//!    change is detected without a bespoke comparator.

use crate::diff::diff;
use crate::integration::ReplicationPlan;
use crate::net_delta::DeltaError;
use crate::reflect::{Reflect, ReflectRef, Struct};
use alloc::vec::Vec;

/// A growable bitset of dirty struct-field indices (design §24.5).
///
/// Bit `i` corresponds to the field at positional index `i`. The set stores one
/// `u64` word per 64 indices and grows on demand, so an entity with a handful
/// of fields costs a single word.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DirtyMask {
    words: Vec<u64>,
}

impl DirtyMask {
    /// Create an empty mask.
    #[must_use]
    pub const fn new() -> Self {
        Self { words: Vec::new() }
    }

    /// Create an empty mask pre-sized to hold `field_count` indices without a
    /// later reallocation.
    #[must_use]
    pub fn with_field_capacity(field_count: usize) -> Self {
        Self {
            words: Vec::with_capacity(field_count.div_ceil(64)),
        }
    }

    /// Mark the field at `index` dirty.
    pub fn mark(&mut self, index: usize) {
        let word = index / 64;
        let bit = index % 64;
        if word >= self.words.len() {
            self.words.resize(word + 1, 0);
        }
        self.words[word] |= 1u64 << bit;
    }

    /// Clear the dirty bit for the field at `index`.
    pub fn unmark(&mut self, index: usize) {
        let word = index / 64;
        let bit = index % 64;
        if let Some(slot) = self.words.get_mut(word) {
            *slot &= !(1u64 << bit);
        }
    }

    /// Whether the field at `index` is marked dirty.
    #[must_use]
    pub fn is_marked(&self, index: usize) -> bool {
        let word = index / 64;
        let bit = index % 64;
        self.words
            .get(word)
            .is_some_and(|w| w & (1u64 << bit) != 0)
    }

    /// Clear every dirty bit, keeping the allocated capacity for reuse.
    pub fn clear(&mut self) {
        for word in &mut self.words {
            *word = 0;
        }
    }

    /// Whether no field is marked dirty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.words.iter().all(|w| *w == 0)
    }

    /// The number of fields currently marked dirty.
    #[must_use]
    pub fn count(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// Iterate the dirty field indices in ascending order.
    pub fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.words.iter().enumerate().flat_map(|(word, bits)| {
            let base = word * 64;
            (0..64).filter_map(move |bit| (bits & (1u64 << bit) != 0).then_some(base + bit))
        })
    }

    /// Union `other`'s dirty bits into `self`.
    pub fn union(&mut self, other: &DirtyMask) {
        if other.words.len() > self.words.len() {
            self.words.resize(other.words.len(), 0);
        }
        for (slot, bits) in self.words.iter_mut().zip(&other.words) {
            *slot |= bits;
        }
    }

    /// Build a mask marking every field that differs between `old` and `new`.
    ///
    /// Both values must be the same named-field struct. Each field is compared
    /// with the reflection [`diff`](crate::diff); a field is marked only when
    /// its patch is not [`Unchanged`](crate::Patch::is_unchanged), so nested
    /// structural change is captured without a per-type comparator.
    ///
    /// # Errors
    /// Returns [`DeltaError::NotAStruct`] if either value is not a named-field
    /// struct, or [`DeltaError::FieldCountMismatch`] if the two structs expose a
    /// different number of fields.
    pub fn changed(old: &dyn Reflect, new: &dyn Reflect) -> Result<Self, DeltaError> {
        let (old_s, new_s) = struct_pair(old, new)?;
        let mut mask = Self::with_field_capacity(new_s.field_count());
        for (index, _) in changed_fields(old_s, new_s) {
            mask.mark(index);
        }
        Ok(mask)
    }

    /// Build a mask marking fields that both differ between `old` and `new`
    /// **and** participate in `plan` (honouring the design's
    /// `#[reflect(replicate)]`/`#[reflect(no_replicate)]` intent via
    /// [`ReplicationPlan`]).
    ///
    /// # Errors
    /// Returns [`DeltaError::NotAStruct`] or [`DeltaError::FieldCountMismatch`]
    /// under the same conditions as [`DirtyMask::changed`].
    pub fn changed_in_plan(
        old: &dyn Reflect,
        new: &dyn Reflect,
        plan: &ReplicationPlan,
    ) -> Result<Self, DeltaError> {
        let (old_s, new_s) = struct_pair(old, new)?;
        let mut mask = Self::with_field_capacity(new_s.field_count());
        for (index, name) in changed_fields(old_s, new_s) {
            if plan.fields().iter().any(|f| f == name) {
                mask.mark(index);
            }
        }
        Ok(mask)
    }
}

/// Resolve two reflected values to their [`Struct`] views, checking shape.
pub(crate) fn struct_pair<'a>(
    old: &'a dyn Reflect,
    new: &'a dyn Reflect,
) -> Result<(&'a dyn Struct, &'a dyn Struct), DeltaError> {
    let (ReflectRef::Struct(old_s), ReflectRef::Struct(new_s)) =
        (old.reflect_ref(), new.reflect_ref())
    else {
        return Err(DeltaError::NotAStruct);
    };
    if old_s.field_count() != new_s.field_count() {
        return Err(DeltaError::FieldCountMismatch {
            old: old_s.field_count(),
            new: new_s.field_count(),
        });
    }
    Ok((old_s, new_s))
}

/// Yield `(index, field_name)` for every field that differs between two equally
/// shaped structs.
fn changed_fields<'a>(
    old_s: &'a dyn Struct,
    new_s: &'a dyn Struct,
) -> impl Iterator<Item = (usize, &'a str)> {
    (0..new_s.field_count()).filter_map(move |index| {
        let name = new_s.name_at(index)?;
        let old_field = old_s.field_at(index)?;
        let new_field = new_s.field_at(index)?;
        (!diff(old_field, new_field).is_unchanged()).then_some((index, name))
    })
}
