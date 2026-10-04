//! Mip-tail residency floor: the stateless primitive that guarantees a
//! virtual-texture sampler never reads a hole.
//!
//! [`GpuPageTable::resolve`](super::indirection::GpuPageTable::resolve) falls
//! back from an absent fine page to the nearest resident *coarser* page, but
//! that fallback only succeeds if some coarser covering page is actually
//! resident. Under heavy streaming churn even the coarse pages can be evicted,
//! and then `resolve` returns [`None`] — a visible texture hole / pop-in.
//!
//! A production virtual-texture system removes that failure mode by pinning a
//! *mip tail*: for a chosen `floor_mip`, every page at or below the floor that
//! any demand touches is forced resident, so `resolve(key, floor_mip)` is
//! guaranteed to find at least the floor-level covering page. The tail is tiny
//! (a coarse mip has few pages) yet it bounds worst-case quality: the sampler
//! degrades to the floor mip instead of to nothing.
//!
//! This module computes, for a set of demanded page keys, the deduplicated set
//! of floor-level pages that cover them. It is pure integer arithmetic with no
//! cross-frame state, mirroring the per-page footprint math in
//! [`GpuPageTable::resolve`](super::indirection::GpuPageTable::resolve): because
//! every page holds a fixed texel count, the page at mip `floor` covering a page
//! at mip `m <= floor` is at coordinate `(x >> (floor - m), y >> (floor - m))`.

use alloc::vec::Vec;

use super::TexturePageKey;

/// Computes the deduplicated, ascending set of `floor_mip`-level pages that
/// cover `keys`.
///
/// Only keys at or finer than `floor_mip` (`key.mip <= floor_mip`) contribute: a
/// page already coarser than the floor cannot be covered by a floor-level page
/// (the mip pyramid only grows coarser upward), so the floor makes no guarantee
/// for it and it is skipped. For a contributing key the covering page shares the
/// key's `(texture, layer)`, sits at `floor_mip`, and lies at page coordinate
/// `(x >> d, y >> d)` where `d = floor_mip - key.mip`.
///
/// A `d >= 16` delta shifts every bit of the `u16` coordinate out, collapsing
/// onto the single top page `(0, 0)` of that mip; this uses
/// [`u16::checked_shr`] with a `0` fallback to stay well-defined there.
///
/// The result is sorted ascending by [`TexturePageKey`] order and deduplicated,
/// so many fine demands sharing one coarse ancestor yield exactly one cover.
#[must_use]
pub fn mip_tail_covers<I>(keys: I, floor_mip: u8) -> Vec<TexturePageKey>
where
    I: IntoIterator<Item = TexturePageKey>,
{
    let mut covers: Vec<TexturePageKey> = keys
        .into_iter()
        .filter(|key| key.mip <= floor_mip)
        .map(|key| {
            let shift = u32::from(floor_mip - key.mip);
            TexturePageKey {
                texture: key.texture,
                mip: floor_mip,
                layer: key.layer,
                x: key.x.checked_shr(shift).unwrap_or(0),
                y: key.y.checked_shr(shift).unwrap_or(0),
            }
        })
        .collect();
    covers.sort_unstable();
    covers.dedup();
    covers
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

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
    fn empty_input_yields_no_covers() {
        assert!(mip_tail_covers(core::iter::empty(), 4).is_empty());
    }

    #[test]
    fn fine_page_maps_to_shifted_floor_coordinate() {
        // mip 0 page (13, 7) under a floor of mip 3 -> (13 >> 3, 7 >> 3) = (1, 0).
        let covers = mip_tail_covers(vec![key(2, 0, 1, 13, 7)], 3);
        assert_eq!(covers, vec![key(2, 3, 1, 1, 0)]);
    }

    #[test]
    fn page_at_floor_is_its_own_cover() {
        let covers = mip_tail_covers(vec![key(5, 3, 0, 9, 4)], 3);
        assert_eq!(covers, vec![key(5, 3, 0, 9, 4)]);
    }

    #[test]
    fn page_coarser_than_floor_is_skipped() {
        // mip 5 is coarser than a floor of mip 3; the floor cannot cover it.
        let covers = mip_tail_covers(vec![key(1, 5, 0, 0, 0)], 3);
        assert!(covers.is_empty());
    }

    #[test]
    fn many_fine_pages_dedup_onto_one_cover() {
        // Four adjacent mip-0 pages all collapse onto floor-mip-1 page (0, 0).
        let keys = vec![
            key(1, 0, 0, 0, 0),
            key(1, 0, 0, 1, 0),
            key(1, 0, 0, 0, 1),
            key(1, 0, 0, 1, 1),
        ];
        let covers = mip_tail_covers(keys, 1);
        assert_eq!(covers, vec![key(1, 1, 0, 0, 0)]);
    }

    #[test]
    fn covers_are_sorted_ascending() {
        let keys = vec![
            key(3, 0, 0, 40, 0),
            key(1, 0, 0, 0, 0),
            key(2, 0, 0, 8, 0),
        ];
        let covers = mip_tail_covers(keys, 2);
        let mut sorted = covers.clone();
        sorted.sort_unstable();
        assert_eq!(covers, sorted);
        assert_eq!(covers.len(), 3);
    }

    #[test]
    fn large_delta_collapses_to_origin_page() {
        // A floor 20 levels above the request shifts the whole u16 coord out.
        let covers = mip_tail_covers(vec![key(7, 0, 2, 65535, 65535)], 20);
        assert_eq!(covers, vec![key(7, 20, 2, 0, 0)]);
    }

    #[test]
    fn distinct_textures_and_layers_stay_separate() {
        let keys = vec![
            key(1, 0, 0, 0, 0),
            key(1, 0, 1, 0, 0),
            key(2, 0, 0, 0, 0),
        ];
        let covers = mip_tail_covers(keys, 2);
        assert_eq!(
            covers,
            vec![key(1, 2, 0, 0, 0), key(1, 2, 1, 0, 0), key(2, 2, 0, 0, 0)]
        );
    }
}
