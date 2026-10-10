//! [`RelocMap`]: an immutable, sorted, relocatable key/value map.
//!
//! The map is baked once from a set of pairs and then read in place. Its blob
//! is a fixed 28-byte header followed by a sorted key region and a parallel
//! value region:
//!
//! ```text
//! +-------+----------+----------+------------------+------------------+--------+--------+
//! | magic | key_size | val_size | keys:OffsetSlice | vals:OffsetSlice | keys   | vals   |
//! | u32@0 | u32@4    | u32@8    | i32@12 u32@16    | i32@20 u32@24    | @28    | ...    |
//! +-------+----------+----------+------------------+------------------+--------+--------+
//! ```
//!
//! The two regions are addressed through self-relative [`OffsetSlice`]s, so the
//! blob is fully position-independent: lookup is an `O(log n)` binary search
//! over the sorted key region, decoding candidates straight out of the borrowed
//! bytes. This is the worked example of offset pointers composing inside a
//! container.

extern crate alloc;

use alloc::vec::Vec;
use core::marker::PhantomData;

use super::offset::OffsetSlice;
use super::{read_u32, Reloc, RelocError};

/// Magic tag at the start of a [`RelocMap`] blob (`b"RMP1"`, little-endian).
const MAGIC: u32 = 0x3150_4D52;
/// Fixed header length in bytes.
const HEADER_LEN: usize = 28;
/// Byte position of the keys [`OffsetSlice`] field.
const KEYS_FIELD_POS: usize = 12;
/// Byte position of the values [`OffsetSlice`] field.
const VALS_FIELD_POS: usize = 20;

/// Decode and validate the header, returning the pair count and the key/value
/// [`OffsetSlice`]s. Confirms magic, both element sizes, that the key and value
/// counts agree, and that both regions lie within `blob`.
fn decode_header<K: Reloc, V: Reloc>(
    blob: &[u8],
) -> Result<(usize, OffsetSlice<K>, OffsetSlice<V>), RelocError> {
    if blob.len() < HEADER_LEN {
        return Err(RelocError::OutOfBounds);
    }
    if read_u32(blob, 0)? != MAGIC {
        return Err(RelocError::BadMagic);
    }
    if read_u32(blob, 4)? as usize != K::SIZE || read_u32(blob, 8)? as usize != V::SIZE {
        return Err(RelocError::SizeMismatch);
    }
    let keys = OffsetSlice::<K>::decode(&blob[KEYS_FIELD_POS..KEYS_FIELD_POS + 8]);
    let vals = OffsetSlice::<V>::decode(&blob[VALS_FIELD_POS..VALS_FIELD_POS + 8]);
    if keys.len() != vals.len() {
        return Err(RelocError::SizeMismatch);
    }
    keys.validate(blob, KEYS_FIELD_POS)?;
    vals.validate(blob, VALS_FIELD_POS)?;
    Ok((keys.len(), keys, vals))
}

/// An owned, immutable, sorted, relocatable map.
///
/// Build it with [`from_pairs`](RelocMap::from_pairs), hand
/// [`as_bytes`](RelocMap::as_bytes) to a serializer or `mmap` writer, and re-open
/// a copied blob with [`from_bytes`](RelocMap::from_bytes) (owning) or
/// [`RelocMapView::new`] (borrowing, zero-copy).
#[derive(Clone)]
pub struct RelocMap<K, V> {
    blob: Vec<u8>,
    marker: PhantomData<fn() -> (K, V)>,
}

impl<K: Reloc + Ord, V: Reloc> RelocMap<K, V> {
    /// Build a map from key/value pairs.
    ///
    /// The pairs are sorted by key; on duplicate keys the **last** pair in the
    /// input wins (so this behaves like inserting left-to-right into a map).
    ///
    /// # Panics
    /// Panics only if the resulting blob would exceed the `i32` self-relative
    /// offset range (about 2 GiB of values), which no realistic baked asset
    /// reaches.
    #[must_use]
    pub fn from_pairs(pairs: &[(K, V)]) -> Self {
        let mut sorted: Vec<(K, V)> = pairs.to_vec();
        // Stable sort by key so that, among equal keys, the original (input)
        // order is preserved and the last occurrence is kept below.
        sorted.sort_by_key(|p| p.0);

        let mut deduped: Vec<(K, V)> = Vec::with_capacity(sorted.len());
        for (k, v) in sorted {
            if let Some(last) = deduped.last_mut()
                && last.0 == k
            {
                last.1 = v;
                continue;
            }
            deduped.push((k, v));
        }

        let len = deduped.len();
        let keys_region = HEADER_LEN;
        let vals_region = keys_region + len * K::SIZE;
        let total = vals_region + len * V::SIZE;

        let keys_off = i32::try_from(keys_region - KEYS_FIELD_POS)
            .expect("RelocMap blob exceeds the 2 GiB self-relative offset range");
        let vals_off = i32::try_from(vals_region - VALS_FIELD_POS)
            .expect("RelocMap blob exceeds the 2 GiB self-relative offset range");
        let count = u32::try_from(len).expect("RelocMap length exceeds u32");

        let mut blob = Vec::with_capacity(total);
        blob.extend_from_slice(&MAGIC.to_le_bytes());
        blob.extend_from_slice(&(K::SIZE as u32).to_le_bytes());
        blob.extend_from_slice(&(V::SIZE as u32).to_le_bytes());
        let mut field = [0u8; 8];
        OffsetSlice::<K>::from_raw(keys_off, count).encode(&mut field);
        blob.extend_from_slice(&field);
        OffsetSlice::<V>::from_raw(vals_off, count).encode(&mut field);
        blob.extend_from_slice(&field);

        // Keys region, then the parallel values region.
        for (k, _) in &deduped {
            let start = blob.len();
            blob.resize(start + K::SIZE, 0);
            k.encode(&mut blob[start..start + K::SIZE]);
        }
        for (_, v) in &deduped {
            let start = blob.len();
            blob.resize(start + V::SIZE, 0);
            v.encode(&mut blob[start..start + V::SIZE]);
        }

        Self {
            blob,
            marker: PhantomData,
        }
    }

    /// The number of entries.
    #[must_use]
    pub fn len(&self) -> usize {
        let b = &self.blob[KEYS_FIELD_POS + 4..KEYS_FIELD_POS + 8];
        u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize
    }

    /// Whether the map is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Look up `key`, returning its decoded value if present.
    #[must_use]
    pub fn get(&self, key: &K) -> Option<V> {
        self.view().ok()?.get(key)
    }

    /// Whether `key` is present.
    #[must_use]
    pub fn contains_key(&self, key: &K) -> bool {
        self.get(key).is_some()
    }

    /// Iterate over the entries in ascending key order.
    #[must_use]
    pub fn iter(&self) -> RelocMapIter<'_, K, V> {
        // Reconstruct the slices from the owned (always valid) header.
        let keys = OffsetSlice::<K>::decode(&self.blob[KEYS_FIELD_POS..KEYS_FIELD_POS + 8]);
        let vals = OffsetSlice::<V>::decode(&self.blob[VALS_FIELD_POS..VALS_FIELD_POS + 8]);
        RelocMapIter {
            blob: &self.blob,
            keys,
            vals,
            index: 0,
        }
    }

    /// The relocatable blob backing this map.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.blob
    }

    /// Consume the map and return the owned blob.
    #[must_use]
    pub fn into_bytes(self) -> Vec<u8> {
        self.blob
    }

    /// Re-open an owned map from a (possibly copied) blob, validating it.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, RelocError> {
        decode_header::<K, V>(bytes)?;
        Ok(Self {
            blob: bytes.to_vec(),
            marker: PhantomData,
        })
    }

    /// Borrow a zero-copy [`view`](RelocMapView) over this map's blob.
    pub fn view(&self) -> Result<RelocMapView<'_, K, V>, RelocError> {
        RelocMapView::new(&self.blob)
    }
}

/// A borrowing, zero-copy view over a (possibly `memcpy`'d / `mmap`'d) map blob.
///
/// Construction validates the header once; afterwards [`get`](RelocMapView::get)
/// is an `O(log n)` binary search decoding candidates in place.
#[derive(Clone, Copy)]
pub struct RelocMapView<'a, K, V> {
    blob: &'a [u8],
    len: usize,
    keys: OffsetSlice<K>,
    vals: OffsetSlice<V>,
}

impl<'a, K: Reloc + Ord, V: Reloc> RelocMapView<'a, K, V> {
    /// Validate `blob` as a [`RelocMap`] blob and borrow a view over it.
    pub fn new(blob: &'a [u8]) -> Result<Self, RelocError> {
        let (len, keys, vals) = decode_header::<K, V>(blob)?;
        Ok(Self {
            blob,
            len,
            keys,
            vals,
        })
    }

    /// The number of entries.
    #[inline]
    #[must_use]
    pub fn len(self) -> usize {
        self.len
    }

    /// Whether the map is empty.
    #[inline]
    #[must_use]
    pub fn is_empty(self) -> bool {
        self.len == 0
    }

    /// The key at sorted position `index` (debug/iteration helper).
    #[inline]
    fn key_at(self, index: usize) -> Option<K> {
        self.keys
            .get(self.blob, KEYS_FIELD_POS, index)
            .ok()
            .flatten()
    }

    /// The value at sorted position `index`.
    #[inline]
    fn val_at(self, index: usize) -> Option<V> {
        self.vals
            .get(self.blob, VALS_FIELD_POS, index)
            .ok()
            .flatten()
    }

    /// Look up `key`, returning its decoded value if present.
    #[must_use]
    pub fn get(self, key: &K) -> Option<V> {
        let mut lo = 0usize;
        let mut hi = self.len;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let probe = self.key_at(mid)?;
            match probe.cmp(key) {
                core::cmp::Ordering::Less => lo = mid + 1,
                core::cmp::Ordering::Greater => hi = mid,
                core::cmp::Ordering::Equal => return self.val_at(mid),
            }
        }
        None
    }

    /// Whether `key` is present.
    #[must_use]
    pub fn contains_key(self, key: &K) -> bool {
        self.get(key).is_some()
    }

    /// Iterate over the entries in ascending key order.
    #[must_use]
    pub fn iter(self) -> RelocMapIter<'a, K, V> {
        RelocMapIter {
            blob: self.blob,
            keys: self.keys,
            vals: self.vals,
            index: 0,
        }
    }
}

/// Iterator over the entries of a [`RelocMap`] / [`RelocMapView`], in ascending
/// key order.
pub struct RelocMapIter<'a, K, V> {
    blob: &'a [u8],
    keys: OffsetSlice<K>,
    vals: OffsetSlice<V>,
    index: usize,
}

impl<K: Reloc, V: Reloc> Iterator for RelocMapIter<'_, K, V> {
    type Item = (K, V);

    fn next(&mut self) -> Option<(K, V)> {
        let k = self
            .keys
            .get(self.blob, KEYS_FIELD_POS, self.index)
            .ok()??;
        let v = self
            .vals
            .get(self.blob, VALS_FIELD_POS, self.index)
            .ok()??;
        self.index += 1;
        Some((k, v))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.keys.len().saturating_sub(self.index);
        (remaining, Some(remaining))
    }
}

impl<K: Reloc, V: Reloc> ExactSizeIterator for RelocMapIter<'_, K, V> {}
