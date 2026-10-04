//! `GPU` indirection page table: the resident set packed for shader lookup.
//!
//! The [`PhysicalPagePool`](super::pool::PhysicalPagePool) owns the mapping from
//! virtual [`TexturePageKey`] to physical slot; this module serialises that
//! mapping into the flat `u32` buffer a shader binary-searches to translate a
//! sampled page coordinate into its physical slot. Each resident page is one
//! fixed-width entry of [`PAGE_TABLE_ENTRY_WORDS`] words:
//!
//! * `w0 = texture`
//! * `w1 = (mip << 24) | (layer << 8)` (low 8 bits reserved/zero)
//! * `w2 = (x << 16) | y`
//! * `w3 = slot`
//!
//! The first three words are the page's *compare key*. They are laid out so that
//! an unsigned lexicographic compare of `(w0, w1, w2)` is identical to the
//! derived [`Ord`] on [`TexturePageKey`] (`texture`, then `mip`, `layer`, `x`,
//! `y`). Because [`PhysicalPagePool::iter`](super::pool::PhysicalPagePool::iter)
//! already yields pages in key order, [`GpuPageTable::from_pool`] emits entries
//! already sorted ascending, so both the golden [`GpuPageTable::lookup`] here and
//! a shader-side binary search over the same bytes resolve a key with an
//! identical comparison sequence. No device handle is held: the buffer is raw
//! words for the backend to upload.

use super::pool::PhysicalPagePool;
use super::TexturePageKey;
use alloc::vec::Vec;
use core::cmp::Ordering;

/// Number of `u32` words per page-table entry: three compare-key words plus the
/// physical slot.
pub const PAGE_TABLE_ENTRY_WORDS: usize = 4;

/// Flat, sorted, binary-searchable page table over a resident page set.
///
/// Holds `len()` entries of [`PAGE_TABLE_ENTRY_WORDS`] words each, ascending by
/// [`TexturePageKey`]. The backing [`words`](Self::words) slice is uploaded
/// verbatim to a `GPU` storage buffer; a shader resolves a page coordinate with
/// the same binary search [`lookup`](Self::lookup) implements here.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct GpuPageTable {
    /// Entry words, `PAGE_TABLE_ENTRY_WORDS` per resident page, key-ascending.
    words: Vec<u32>,
}

impl GpuPageTable {
    /// Creates an empty page table.
    #[must_use]
    pub fn new() -> Self {
        Self { words: Vec::new() }
    }

    /// Builds the table from a physical pool's resident `(page, slot)` bindings.
    ///
    /// [`PhysicalPagePool::iter`](super::pool::PhysicalPagePool::iter) is already
    /// key-ordered, so entries are emitted sorted and no post-sort is needed.
    #[must_use]
    pub fn from_pool(pool: &PhysicalPagePool) -> Self {
        let mut words = Vec::with_capacity(pool.resident_count() as usize * PAGE_TABLE_ENTRY_WORDS);
        for (key, slot) in pool.iter() {
            let [w0, w1, w2] = Self::compare_words(key);
            words.push(w0);
            words.push(w1);
            words.push(w2);
            words.push(slot);
        }
        Self { words }
    }

    /// Builds the table from an explicit `(page, slot)` binding list.
    ///
    /// Unlike [`from_pool`](Self::from_pool), the caller supplies the exact set
    /// of bindings to publish, so a streamer can withhold pages that are seated
    /// in the pool but whose upload has not yet completed. The bindings are
    /// sorted ascending by [`TexturePageKey`] here, so the caller need not
    /// pre-sort; the resulting buffer still matches the shader compare order.
    #[must_use]
    pub fn from_bindings(bindings: &[(TexturePageKey, u32)]) -> Self {
        let mut sorted: Vec<(TexturePageKey, u32)> = bindings.to_vec();
        sorted.sort_unstable_by_key(|entry| entry.0);
        let mut words = Vec::with_capacity(sorted.len() * PAGE_TABLE_ENTRY_WORDS);
        for (key, slot) in sorted {
            let [w0, w1, w2] = Self::compare_words(key);
            words.push(w0);
            words.push(w1);
            words.push(w2);
            words.push(slot);
        }
        Self { words }
    }

    /// Packs a key into its three compare words `(w0, w1, w2)`.
    ///
    /// Lexicographic order over the returned words equals [`Ord`] on
    /// [`TexturePageKey`].
    #[must_use]
    pub fn compare_words(key: TexturePageKey) -> [u32; 3] {
        let w0 = key.texture;
        let w1 = (u32::from(key.mip) << 24) | (u32::from(key.layer) << 8);
        let w2 = (u32::from(key.x) << 16) | u32::from(key.y);
        [w0, w1, w2]
    }

    /// Reconstructs the key encoded by an entry's three compare words.
    ///
    /// Inverse of [`compare_words`](Self::compare_words); the reserved low 8 bits
    /// of `w1` are ignored.
    #[must_use]
    pub fn unpack_key(w0: u32, w1: u32, w2: u32) -> TexturePageKey {
        TexturePageKey {
            texture: w0,
            mip: (w1 >> 24) as u8,
            layer: ((w1 >> 8) & 0xffff) as u16,
            x: (w2 >> 16) as u16,
            y: (w2 & 0xffff) as u16,
        }
    }

    /// Number of resident entries in the table.
    #[must_use]
    pub fn len(&self) -> usize {
        self.words.len() / PAGE_TABLE_ENTRY_WORDS
    }

    /// Whether the table holds no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.words.is_empty()
    }

    /// Raw entry words for upload to a `GPU` storage buffer.
    #[must_use]
    pub fn words(&self) -> &[u32] {
        &self.words
    }

    /// The `(key, slot)` binding stored at entry `index`, if in range.
    #[must_use]
    pub fn entry(&self, index: usize) -> Option<(TexturePageKey, u32)> {
        let base = index.checked_mul(PAGE_TABLE_ENTRY_WORDS)?;
        let slice = self.words.get(base..base + PAGE_TABLE_ENTRY_WORDS)?;
        Some((Self::unpack_key(slice[0], slice[1], slice[2]), slice[3]))
    }

    /// Resolves `key` to its physical slot via binary search.
    ///
    /// This is the golden contract a shader reproduces: it compares the packed
    /// `(w0, w1, w2)` words, which order identically to [`TexturePageKey`].
    #[must_use]
    pub fn lookup(&self, key: TexturePageKey) -> Option<u32> {
        let target = Self::compare_words(key);
        let mut lo = 0usize;
        let mut hi = self.len();
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let base = mid * PAGE_TABLE_ENTRY_WORDS;
            let entry = [self.words[base], self.words[base + 1], self.words[base + 2]];
            match entry.cmp(&target) {
                Ordering::Less => lo = mid + 1,
                Ordering::Greater => hi = mid,
                Ordering::Equal => return Some(self.words[base + 3]),
            }
        }
        None
    }

    /// Resolves `key` to the best resident tile, falling back to a coarser mip
    /// when the exact page has not streamed in yet.
    ///
    /// A virtual-texture sampler must never return a hole: when the page at the
    /// requested `mip` is absent, it reads the nearest resident coarser mip of
    /// the same `(texture, layer)` that still covers the sampled texels. Because
    /// every page holds a fixed texel count, mip `m + d` packs the footprint of
    /// `key` into page coordinate `(x >> d, y >> d)`, so this walks `d` upward
    /// from `0` (an exact hit) until a resident covering page is found or
    /// `coarsest_mip` is exceeded. `coarsest_mip` is clamped up to `key.mip`, so
    /// a value below the request still probes the exact page.
    ///
    /// The returned [`PageResolution::mip_bias`] is that `d`: the sampler scales
    /// its intra-page `UV` by `1 / (1 << mip_bias)` and offsets into the
    /// sub-tile the fine page occupies to read the correct texels from the
    /// coarser tile. Returns [`None`] only when neither the exact page nor any
    /// coarser covering page up to `coarsest_mip` is resident.
    #[must_use]
    pub fn resolve(&self, key: TexturePageKey, coarsest_mip: u8) -> Option<PageResolution> {
        let last = coarsest_mip.max(key.mip);
        for mip in key.mip..=last {
            let d = mip - key.mip;
            let shift = u32::from(d);
            let candidate = TexturePageKey {
                texture: key.texture,
                mip,
                layer: key.layer,
                // A `d >= 16` delta shifts every bit of the `u16` coordinate out,
                // collapsing onto the single top-mip page `(0, 0)`.
                x: key.x.checked_shr(shift).unwrap_or(0),
                y: key.y.checked_shr(shift).unwrap_or(0),
            };
            if let Some(slot) = self.lookup(candidate) {
                return Some(PageResolution {
                    key: candidate,
                    slot,
                    mip_bias: d,
                });
            }
        }
        None
    }
}

/// Outcome of resolving a sampled page against the resident set, including any
/// fallback to a coarser resident mip (see [`GpuPageTable::resolve`]).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PageResolution {
    /// The resident page that satisfied the request: the exact page when
    /// [`mip_bias`](Self::mip_bias) is `0`, otherwise the coarser covering page.
    pub key: TexturePageKey,
    /// Physical slot bound to [`key`](Self::key).
    pub slot: u32,
    /// Mip levels coarser than the request: `0` for an exact hit, `n > 0` when
    /// an `n`-levels-coarser page was substituted.
    pub mip_bias: u8,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::texture_streaming::scheduler::StreamingPlan;

    fn key(texture: u32, mip: u8, layer: u16, x: u16, y: u16) -> TexturePageKey {
        TexturePageKey {
            texture,
            mip,
            layer,
            x,
            y,
        }
    }

    #[test]
    fn pack_unpack_round_trips_every_field() {
        let k = key(0x1234_5678, 7, 0xABCD, 0x4321, 0xFEDC);
        let [w0, w1, w2] = GpuPageTable::compare_words(k);
        assert_eq!(GpuPageTable::unpack_key(w0, w1, w2), k);
    }

    #[test]
    fn compare_word_order_matches_key_ord() {
        // A spread of keys differing in each field, including boundary values.
        let keys = [
            key(0, 0, 0, 0, 0),
            key(0, 0, 0, 0, 1),
            key(0, 0, 0, 1, 0),
            key(0, 0, 1, 0, 0),
            key(0, 1, 0, 0, 0),
            key(0, 1, 0xFFFF, 0xFFFF, 0xFFFF),
            key(1, 0, 0, 0, 0),
            key(0xFFFF_FFFF, 0xFF, 0xFFFF, 0xFFFF, 0xFFFF),
        ];
        for a in keys {
            for b in keys {
                let wa = GpuPageTable::compare_words(a);
                let wb = GpuPageTable::compare_words(b);
                assert_eq!(wa.cmp(&wb), a.cmp(&b), "mismatch for {a:?} vs {b:?}");
            }
        }
    }

    #[test]
    fn low_eight_bits_of_w1_are_reserved_zero() {
        let [_, w1, _] = GpuPageTable::compare_words(key(0, 0xFF, 0xFFFF, 0, 0));
        assert_eq!(w1 & 0xFF, 0);
    }

    #[test]
    fn empty_table_has_no_entries() {
        let table = GpuPageTable::new();
        assert!(table.is_empty());
        assert_eq!(table.len(), 0);
        assert_eq!(table.words(), &[] as &[u32]);
        assert_eq!(table.lookup(key(0, 0, 0, 0, 0)), None);
    }

    #[test]
    fn from_pool_emits_sorted_entries_with_slots() {
        let mut pool = PhysicalPagePool::new(4);
        // Admit out of key order; the pool stores them key-ordered.
        let s_c = pool.admit(key(2, 0, 0, 0, 0)).expect("admit c");
        let s_a = pool.admit(key(0, 1, 0, 0, 0)).expect("admit a");
        let s_b = pool.admit(key(0, 1, 0, 0, 5)).expect("admit b");

        let table = GpuPageTable::from_pool(&pool);
        assert_eq!(table.len(), 3);
        assert_eq!(table.words().len(), 3 * PAGE_TABLE_ENTRY_WORDS);

        // Entries are ascending by key regardless of admission order.
        let ordered: Vec<TexturePageKey> = (0..table.len())
            .map(|i| table.entry(i).unwrap().0)
            .collect();
        let mut sorted = ordered.clone();
        sorted.sort();
        assert_eq!(ordered, sorted);

        // Slots survive the pack: lookup returns what the pool assigned.
        assert_eq!(table.lookup(key(0, 1, 0, 0, 0)), Some(s_a));
        assert_eq!(table.lookup(key(0, 1, 0, 0, 5)), Some(s_b));
        assert_eq!(table.lookup(key(2, 0, 0, 0, 0)), Some(s_c));
        assert_eq!(table.lookup(key(9, 9, 9, 9, 9)), None);
    }

    #[test]
    fn lookup_agrees_with_pool_slot_of_for_every_resident_page() {
        let mut pool = PhysicalPagePool::new(16);
        let plan = StreamingPlan {
            loads: alloc::vec![
                key(3, 2, 1, 7, 7),
                key(3, 2, 1, 7, 8),
                key(3, 0, 0, 0, 0),
                key(1, 5, 9, 2, 3),
                key(0, 0, 0, 0, 0),
            ],
            evicts: alloc::vec![],
            resident_bytes: 0,
        };
        pool.apply_plan(&plan);

        let table = GpuPageTable::from_pool(&pool);
        for (k, slot) in pool.iter() {
            assert_eq!(table.lookup(k), Some(slot));
        }
    }

    #[test]
    fn entry_out_of_range_is_none() {
        let table = GpuPageTable::new();
        assert_eq!(table.entry(0), None);
    }

    #[test]
    fn from_bindings_sorts_and_matches_lookup() {
        // Deliberately unsorted bindings; a withheld page is simply absent.
        let bindings = [
            (key(2, 0, 0, 1, 0), 7u32),
            (key(1, 0, 0, 0, 0), 3u32),
            (key(1, 1, 0, 0, 0), 5u32),
        ];
        let table = GpuPageTable::from_bindings(&bindings);
        // Entries come out key-ascending regardless of input order.
        let mut prev: Option<TexturePageKey> = None;
        for i in 0..table.len() {
            let (k, _slot) = table.entry(i).expect("entry in range");
            if let Some(p) = prev {
                assert!(p <= k, "entries must be key-ascending");
            }
            prev = Some(k);
        }
        assert_eq!(table.lookup(key(1, 0, 0, 0, 0)), Some(3));
        assert_eq!(table.lookup(key(1, 1, 0, 0, 0)), Some(5));
        assert_eq!(table.lookup(key(2, 0, 0, 1, 0)), Some(7));
        // A page not in the bindings resolves to nothing.
        assert_eq!(table.lookup(key(9, 0, 0, 0, 0)), None);
    }

    #[test]
    fn resolve_returns_exact_hit_with_zero_bias() {
        let table = GpuPageTable::from_bindings(&[(key(3, 2, 1, 7, 9), 42)]);
        let hit = table.resolve(key(3, 2, 1, 7, 9), 10).expect("exact page resident");
        assert_eq!(hit.key, key(3, 2, 1, 7, 9));
        assert_eq!(hit.slot, 42);
        assert_eq!(hit.mip_bias, 0);
    }

    #[test]
    fn resolve_falls_back_to_coarser_covering_page() {
        // Fine page (mip 0, x=6, y=10) is absent; its mip-2 cover is (6>>2, 10>>2)
        // = (1, 2), which IS resident.
        let table = GpuPageTable::from_bindings(&[(key(1, 2, 0, 1, 2), 5)]);
        let res = table.resolve(key(1, 0, 0, 6, 10), 4).expect("coarser cover resident");
        assert_eq!(res.key, key(1, 2, 0, 1, 2));
        assert_eq!(res.slot, 5);
        assert_eq!(res.mip_bias, 2);
    }

    #[test]
    fn resolve_prefers_finest_resident_mip() {
        // Both mip 1 and mip 2 covers are resident; the finer (smaller bias) wins.
        let table = GpuPageTable::from_bindings(&[
            (key(1, 1, 0, 3, 3), 11), // cover of (mip0, 6,6) at d=1 -> (3,3)
            (key(1, 2, 0, 1, 1), 22), // cover at d=2 -> (1,1)
        ]);
        let res = table.resolve(key(1, 0, 0, 6, 6), 5).expect("a cover resident");
        assert_eq!(res.key, key(1, 1, 0, 3, 3));
        assert_eq!(res.slot, 11);
        assert_eq!(res.mip_bias, 1);
    }

    #[test]
    fn resolve_respects_coarsest_mip_cap() {
        // The only cover sits at mip 3, but the cap stops the walk at mip 2.
        let table = GpuPageTable::from_bindings(&[(key(1, 3, 0, 0, 0), 7)]);
        assert_eq!(table.resolve(key(1, 0, 0, 4, 4), 2), None);
        // Raising the cap to 3 lets the same request resolve.
        let res = table.resolve(key(1, 0, 0, 4, 4), 3).expect("cover within raised cap");
        assert_eq!(res.key, key(1, 3, 0, 0, 0));
        assert_eq!(res.mip_bias, 3);
    }

    #[test]
    fn resolve_returns_none_when_nothing_covers() {
        let table = GpuPageTable::from_bindings(&[(key(2, 0, 0, 0, 0), 1)]);
        // Different texture: no cover at any mip.
        assert_eq!(table.resolve(key(1, 0, 0, 0, 0), 8), None);
    }

    #[test]
    fn resolve_does_not_cross_layers() {
        // A resident cover on layer 0 must not satisfy a request on layer 1.
        let table = GpuPageTable::from_bindings(&[(key(1, 2, 0, 0, 0), 9)]);
        assert_eq!(table.resolve(key(1, 0, 1, 0, 0), 4), None);
    }

    #[test]
    fn resolve_cap_below_request_still_probes_exact_page() {
        // coarsest_mip < key.mip is clamped up so the exact page is still tried.
        let table = GpuPageTable::from_bindings(&[(key(1, 5, 0, 2, 2), 8)]);
        let res = table.resolve(key(1, 5, 0, 2, 2), 0).expect("exact page probed");
        assert_eq!(res.key, key(1, 5, 0, 2, 2));
        assert_eq!(res.mip_bias, 0);
    }

    #[test]
    fn resolve_large_mip_delta_collapses_to_top_page_without_panic() {
        // A >=16-level delta shifts every coordinate bit out, collapsing to (0,0).
        let table = GpuPageTable::from_bindings(&[(key(1, 20, 0, 0, 0), 3)]);
        let res = table
            .resolve(key(1, 0, 0, 0xFFFF, 0xFFFF), 20)
            .expect("top-mip page covers everything");
        assert_eq!(res.key, key(1, 20, 0, 0, 0));
        assert_eq!(res.mip_bias, 20);
    }
}
