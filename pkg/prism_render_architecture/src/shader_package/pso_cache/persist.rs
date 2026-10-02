//! Deterministic serialization of a persisted pipeline-state-object (PSO) cache.
//!
//! Eliminating shader-compilation hitching across *sessions* (not just within
//! one run) requires persisting the driver's compiled pipeline binaries to disk
//! and reloading them on the next launch, so a cold start skips recompilation
//! entirely. This mirrors `VkPipelineCache` blobs and D3D12
//! `ID3D12PipelineLibrary` serialization, and the engine wraps the same idea
//! around its own [`PsoCacheKey`]-addressed entries.
//!
//! A persisted blob is only safe to reuse when the device, driver, backend, and
//! render `ABI` are unchanged, so every blob embeds the
//! [`DeviceFingerprint`](super::DeviceFingerprint) it was produced under and is
//! rejected on mismatch (see [`PersistedPsoCache::load_for`]). The blob also
//! carries a trailing integrity checksum so a truncated or corrupted file is
//! rejected rather than fed to the driver.
//!
//! This module models the **deterministic, backend-agnostic wire format** only:
//! it turns an in-memory [`PersistedPsoCache`] into a byte vector and back,
//! byte-for-byte reproducibly, leaving the actual file read/write to the
//! engine-side consumer (consistent with this crate's "contracts, not backend"
//! charter). The format is little-endian throughout and framed so a golden test
//! can pin the exact bytes.
//!
//! # Layout
//!
//! ```text
//! magic         : [u8; 4]  = b"PPSO"
//! format_version: u32
//! backend_tag   : u8        (+ u32 discriminant when tag == OTHER)
//! vendor_id     : u32
//! device_id     : u32
//! driver_version: u64
//! abi_hash      : [u8; 32]
//! entry_count   : u32
//! entries[]     : { pkg_len: u32, pkg_utf8: [u8; pkg_len],
//!                   permutation_index: u64, state_hash: u64,
//!                   blob_len: u64, blob: [u8; blob_len] }
//! checksum      : u64        (FNV-1a over every preceding byte)
//! ```
//!
//! Entries are always written in [`PsoCacheKey`] order, so the encoding of a
//! given logical cache is unique regardless of insertion order.

use alloc::string::String;
use alloc::vec::Vec;

use super::fingerprint::{DeviceFingerprint, FingerprintMismatch, GraphicsBackend};
use super::{LruPsoCache, PipelineStateHash, PsoCacheKey};
use crate::abi::AbiHash;
use crate::shader_package::ShaderPackageId;

/// Four-byte magic identifying a Prism PSO cache blob (`b"PPSO"`).
const MAGIC: [u8; 4] = *b"PPSO";

/// Wire format version. Bumped on any incompatible layout change so an older
/// engine refuses a newer blob (and vice versa) instead of misparsing it.
const FORMAT_VERSION: u32 = 1;

// Backend discriminants on the wire. Kept explicit (not `as u8` of the enum) so
// reordering [`GraphicsBackend`] variants can never silently repurpose a tag.
const BACKEND_VULKAN: u8 = 0;
const BACKEND_METAL: u8 = 1;
const BACKEND_DX12: u8 = 2;
const BACKEND_WEBGPU: u8 = 3;
const BACKEND_OTHER: u8 = 4;

/// FNV-1a 64-bit offset basis.
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
/// FNV-1a 64-bit prime.
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// One compiled pipeline persisted in the cache: its [`PsoCacheKey`] identity
/// plus the opaque driver binary bytes to hand back to the backend on reload.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PersistedPipeline {
    /// Fully-qualified pipeline identity.
    pub key: PsoCacheKey,
    /// Opaque driver pipeline binary (e.g. a `VkPipelineCache` slice). Treated
    /// as bytes here; its meaning is the backend's.
    pub blob: Vec<u8>,
}

/// An in-memory image of a persisted PSO cache: the [`DeviceFingerprint`] it was
/// produced under plus its pipeline entries.
///
/// Build one with [`PersistedPsoCache::new`], append entries, [`encode`] to
/// bytes for the engine to write out, and [`load_for`] to decode + fingerprint
/// check a blob read back on the next launch.
///
/// [`encode`]: PersistedPsoCache::encode
/// [`load_for`]: PersistedPsoCache::load_for
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PersistedPsoCache {
    fingerprint: DeviceFingerprint,
    entries: Vec<PersistedPipeline>,
}

/// Why decoding a persisted PSO cache blob failed.
///
/// Every variant means the blob must be discarded and the cache rebuilt; none
/// is recoverable in place.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PersistError {
    /// The blob is shorter than the field currently being read.
    UnexpectedEof,
    /// The leading magic bytes are not `b"PPSO"`.
    BadMagic,
    /// The format version is one this build does not understand.
    UnsupportedVersion(u32),
    /// The backend tag byte is outside the known set.
    BadBackendTag(u8),
    /// A package identifier was not valid UTF-8.
    BadPackageId,
    /// The trailing checksum did not match the recomputed one (corruption).
    ChecksumMismatch,
    /// Bytes remained after the declared structure was fully read.
    TrailingBytes,
}

/// Why reusing a persisted blob on the current device was refused.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PersistLoadError {
    /// The blob could not be decoded (corrupt / wrong version / truncated).
    Decode(PersistError),
    /// The blob decoded, but its fingerprint is incompatible with this device,
    /// so the compiled pipelines cannot be reused.
    Fingerprint(FingerprintMismatch),
}

impl PersistedPsoCache {
    /// Creates an empty cache image tagged with the producing device fingerprint.
    #[must_use]
    pub fn new(fingerprint: DeviceFingerprint) -> Self {
        Self {
            fingerprint,
            entries: Vec::new(),
        }
    }

    /// The device fingerprint this cache was produced under.
    #[must_use]
    pub fn fingerprint(&self) -> &DeviceFingerprint {
        &self.fingerprint
    }

    /// The persisted pipeline entries, in [`PsoCacheKey`] order after [`encode`]
    /// (insertion order before encoding).
    ///
    /// [`encode`]: PersistedPsoCache::encode
    #[must_use]
    pub fn entries(&self) -> &[PersistedPipeline] {
        &self.entries
    }

    /// Number of persisted pipelines.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache holds no pipelines.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Inserts (or replaces) the entry for `key` with `blob`.
    ///
    /// Re-inserting an existing key overwrites its blob so the newest compiled
    /// binary wins, keeping one entry per pipeline identity.
    pub fn insert(&mut self, key: PsoCacheKey, blob: Vec<u8>) {
        if let Some(existing) = self.entries.iter_mut().find(|e| e.key == key) {
            existing.blob = blob;
        } else {
            self.entries.push(PersistedPipeline { key, blob });
        }
    }

    /// Serializes the cache to a self-describing, little-endian byte vector with
    /// a trailing integrity checksum.
    ///
    /// Entries are emitted in [`PsoCacheKey`] order, so the output is a
    /// canonical function of the logical cache contents (insertion order does
    /// not affect the bytes).
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut sorted: Vec<&PersistedPipeline> = self.entries.iter().collect();
        sorted.sort_by(|a, b| a.key.cmp(&b.key));

        let mut out = Vec::new();
        out.extend_from_slice(&MAGIC);
        out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        encode_fingerprint(&self.fingerprint, &mut out);

        // `entry_count` is bounded by `u32`; a cache with >4G distinct
        // pipelines is not a real scenario and would exhaust memory first.
        out.extend_from_slice(&(sorted.len() as u32).to_le_bytes());
        for entry in sorted {
            let id = entry.key.package.as_str().as_bytes();
            out.extend_from_slice(&(id.len() as u32).to_le_bytes());
            out.extend_from_slice(id);
            out.extend_from_slice(&entry.key.permutation_index.to_le_bytes());
            out.extend_from_slice(&entry.key.state.0.to_le_bytes());
            out.extend_from_slice(&(entry.blob.len() as u64).to_le_bytes());
            out.extend_from_slice(&entry.blob);
        }

        let checksum = fnv1a64(&out);
        out.extend_from_slice(&checksum.to_le_bytes());
        out
    }

    /// Decodes a blob without checking the device fingerprint.
    ///
    /// Use [`load_for`](PersistedPsoCache::load_for) at startup to also reject a
    /// fingerprint mismatch; this bare decode is for inspection/tests.
    ///
    /// # Errors
    ///
    /// Returns a [`PersistError`] if the magic, version, structure, or trailing
    /// checksum is wrong.
    pub fn decode(bytes: &[u8]) -> Result<Self, PersistError> {
        // Split off and verify the trailing checksum first so a corrupt blob is
        // rejected before any field is trusted.
        if bytes.len() < 8 {
            return Err(PersistError::UnexpectedEof);
        }
        let (payload, stored_checksum) = bytes.split_at(bytes.len() - 8);
        let stored = u64::from_le_bytes(
            stored_checksum
                .try_into()
                .expect("split_at leaves exactly 8 trailing bytes"),
        );
        if fnv1a64(payload) != stored {
            return Err(PersistError::ChecksumMismatch);
        }

        let mut reader = Reader::new(payload);
        if reader.read_array4()? != MAGIC {
            return Err(PersistError::BadMagic);
        }
        let version = reader.read_u32()?;
        if version != FORMAT_VERSION {
            return Err(PersistError::UnsupportedVersion(version));
        }
        let fingerprint = decode_fingerprint(&mut reader)?;

        let entry_count = reader.read_u32()? as usize;
        let mut entries = Vec::with_capacity(entry_count);
        for _ in 0..entry_count {
            let id_len = reader.read_u32()? as usize;
            let id_bytes = reader.read_slice(id_len)?;
            let id = core::str::from_utf8(id_bytes).map_err(|_| PersistError::BadPackageId)?;
            let permutation_index = reader.read_u64()?;
            let state = PipelineStateHash(reader.read_u64()?);
            let blob_len = reader.read_u64()? as usize;
            let blob = reader.read_slice(blob_len)?.to_vec();
            entries.push(PersistedPipeline {
                key: PsoCacheKey::new(ShaderPackageId::new(id), permutation_index, state),
                blob,
            });
        }

        if !reader.is_empty() {
            return Err(PersistError::TrailingBytes);
        }
        Ok(Self {
            fingerprint,
            entries,
        })
    }

    /// Decodes a blob and verifies it may be reused on `current`.
    ///
    /// This is the startup path: a decode failure or a fingerprint mismatch both
    /// mean the on-disk cache must be dropped and rebuilt for the current
    /// device.
    ///
    /// # Errors
    ///
    /// Returns [`PersistLoadError::Decode`] if the bytes are corrupt/unknown, or
    /// [`PersistLoadError::Fingerprint`] if the compiled pipelines target a
    /// different device/driver/backend/`ABI`.
    pub fn load_for(bytes: &[u8], current: &DeviceFingerprint) -> Result<Self, PersistLoadError> {
        let cache = Self::decode(bytes).map_err(PersistLoadError::Decode)?;
        cache
            .fingerprint
            .check_against(current)
            .map_err(PersistLoadError::Fingerprint)?;
        Ok(cache)
    }

    /// Seeds an [`LruPsoCache`] residency model from the persisted entries,
    /// charging each pipeline its blob length and letting the budget evict as
    /// needed.
    ///
    /// Entries are admitted in [`PsoCacheKey`] order so eviction (if the budget
    /// is smaller than the persisted set) is deterministic. Returns the number
    /// of entries that ended up resident.
    pub fn hydrate_into(&self, cache: &mut LruPsoCache) -> usize {
        let mut sorted: Vec<&PersistedPipeline> = self.entries.iter().collect();
        sorted.sort_by(|a, b| a.key.cmp(&b.key));
        for entry in sorted {
            cache.admit(entry.key.clone(), entry.blob.len() as u64);
        }
        cache.len()
    }
}

/// Writes a [`DeviceFingerprint`] in the wire layout documented on the module.
fn encode_fingerprint(fp: &DeviceFingerprint, out: &mut Vec<u8>) {
    match fp.backend {
        GraphicsBackend::Vulkan => out.push(BACKEND_VULKAN),
        GraphicsBackend::Metal => out.push(BACKEND_METAL),
        GraphicsBackend::Dx12 => out.push(BACKEND_DX12),
        GraphicsBackend::WebGpu => out.push(BACKEND_WEBGPU),
        GraphicsBackend::Other(disc) => {
            out.push(BACKEND_OTHER);
            out.extend_from_slice(&disc.to_le_bytes());
        }
    }
    out.extend_from_slice(&fp.vendor_id.to_le_bytes());
    out.extend_from_slice(&fp.device_id.to_le_bytes());
    out.extend_from_slice(&fp.driver_version.to_le_bytes());
    out.extend_from_slice(&fp.abi_hash.0);
}

/// Reads a [`DeviceFingerprint`] written by [`encode_fingerprint`].
fn decode_fingerprint(reader: &mut Reader<'_>) -> Result<DeviceFingerprint, PersistError> {
    let backend = match reader.read_u8()? {
        BACKEND_VULKAN => GraphicsBackend::Vulkan,
        BACKEND_METAL => GraphicsBackend::Metal,
        BACKEND_DX12 => GraphicsBackend::Dx12,
        BACKEND_WEBGPU => GraphicsBackend::WebGpu,
        BACKEND_OTHER => GraphicsBackend::Other(reader.read_u32()?),
        other => return Err(PersistError::BadBackendTag(other)),
    };
    let vendor_id = reader.read_u32()?;
    let device_id = reader.read_u32()?;
    let driver_version = reader.read_u64()?;
    let abi_hash = AbiHash(reader.read_array32()?);
    Ok(DeviceFingerprint::new(
        backend,
        vendor_id,
        device_id,
        driver_version,
        abi_hash,
    ))
}

/// FNV-1a 64-bit hash of `bytes`, used as the blob's integrity checksum.
///
/// A content fingerprint for corruption detection, not a cryptographic hash;
/// deterministic and dependency-free so golden tests pin exact bytes.
fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = FNV_OFFSET;
    for &b in bytes {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Minimal little-endian cursor over a byte slice; every read is bounds-checked
/// and yields [`PersistError::UnexpectedEof`] past the end.
struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn is_empty(&self) -> bool {
        self.pos >= self.bytes.len()
    }

    fn read_slice(&mut self, len: usize) -> Result<&'a [u8], PersistError> {
        let end = self.pos.checked_add(len).ok_or(PersistError::UnexpectedEof)?;
        let slice = self.bytes.get(self.pos..end).ok_or(PersistError::UnexpectedEof)?;
        self.pos = end;
        Ok(slice)
    }

    fn read_u8(&mut self) -> Result<u8, PersistError> {
        Ok(self.read_slice(1)?[0])
    }

    fn read_u32(&mut self) -> Result<u32, PersistError> {
        let s = self.read_slice(4)?;
        Ok(u32::from_le_bytes(s.try_into().expect("read_slice(4) yields 4 bytes")))
    }

    fn read_u64(&mut self) -> Result<u64, PersistError> {
        let s = self.read_slice(8)?;
        Ok(u64::from_le_bytes(s.try_into().expect("read_slice(8) yields 8 bytes")))
    }

    fn read_array4(&mut self) -> Result<[u8; 4], PersistError> {
        let s = self.read_slice(4)?;
        Ok(s.try_into().expect("read_slice(4) yields 4 bytes"))
    }

    fn read_array32(&mut self) -> Result<[u8; 32], PersistError> {
        let s = self.read_slice(32)?;
        Ok(s.try_into().expect("read_slice(32) yields 32 bytes"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn fp() -> DeviceFingerprint {
        DeviceFingerprint::new(GraphicsBackend::Vulkan, 0x10DE, 0x2204, 42, AbiHash([7u8; 32]))
    }

    fn key(pkg: &str, perm: u64, state: u64) -> PsoCacheKey {
        PsoCacheKey::new(ShaderPackageId::new(pkg), perm, PipelineStateHash(state))
    }

    fn sample() -> PersistedPsoCache {
        let mut cache = PersistedPsoCache::new(fp());
        cache.insert(key("gbuffer", 3, 0x11), vec![1, 2, 3, 4]);
        cache.insert(key("shadow", 0, 0x22), vec![9, 9]);
        cache.insert(key("gbuffer", 1, 0x00), vec![]);
        cache
    }

    #[test]
    fn roundtrips_entries_and_fingerprint() {
        let cache = sample();
        let bytes = cache.encode();
        let decoded = PersistedPsoCache::decode(&bytes).unwrap();
        assert_eq!(decoded.fingerprint(), &fp());
        assert_eq!(decoded.len(), 3);
        // Decoded order is canonical (sorted by key).
        let keys: Vec<_> = decoded.entries().iter().map(|e| e.key.clone()).collect();
        assert_eq!(
            keys,
            vec![key("gbuffer", 1, 0x00), key("gbuffer", 3, 0x11), key("shadow", 0, 0x22)]
        );
        // Blobs survive, including the empty one.
        assert_eq!(decoded.entries()[0].blob, Vec::<u8>::new());
        assert_eq!(decoded.entries()[1].blob, vec![1, 2, 3, 4]);
        assert_eq!(decoded.entries()[2].blob, vec![9, 9]);
    }

    #[test]
    fn encoding_is_canonical_regardless_of_insertion_order() {
        let a = sample();

        let mut b = PersistedPsoCache::new(fp());
        b.insert(key("gbuffer", 1, 0x00), vec![]);
        b.insert(key("shadow", 0, 0x22), vec![9, 9]);
        b.insert(key("gbuffer", 3, 0x11), vec![1, 2, 3, 4]);

        assert_eq!(a.encode(), b.encode());
    }

    #[test]
    fn insert_replaces_blob_for_existing_key() {
        let mut cache = PersistedPsoCache::new(fp());
        cache.insert(key("a", 0, 0), vec![1]);
        cache.insert(key("a", 0, 0), vec![2, 2]);
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.entries()[0].blob, vec![2, 2]);
    }

    #[test]
    fn load_for_accepts_matching_device() {
        let bytes = sample().encode();
        let loaded = PersistedPsoCache::load_for(&bytes, &fp()).unwrap();
        assert_eq!(loaded.len(), 3);
    }

    #[test]
    fn load_for_rejects_fingerprint_mismatch() {
        let bytes = sample().encode();
        let mut other = fp();
        other.driver_version = 43;
        assert_eq!(
            PersistedPsoCache::load_for(&bytes, &other),
            Err(PersistLoadError::Fingerprint(FingerprintMismatch::Driver))
        );
    }

    #[test]
    fn decode_rejects_bad_magic() {
        let mut bytes = sample().encode();
        bytes[0] = b'X';
        // Corrupting a byte also breaks the checksum; checksum is verified first.
        assert_eq!(PersistedPsoCache::decode(&bytes), Err(PersistError::ChecksumMismatch));
    }

    #[test]
    fn decode_rejects_bad_magic_with_valid_checksum() {
        // Flip the magic and re-stamp a correct checksum so the magic check is
        // what actually fires.
        let bytes = sample().encode();
        let mut payload = bytes[..bytes.len() - 8].to_vec();
        payload[0] = b'X';
        let checksum = fnv1a64(&payload);
        payload.extend_from_slice(&checksum.to_le_bytes());
        assert_eq!(PersistedPsoCache::decode(&payload), Err(PersistError::BadMagic));
    }

    #[test]
    fn decode_rejects_unsupported_version() {
        let bytes = sample().encode();
        let mut payload = bytes[..bytes.len() - 8].to_vec();
        payload[4..8].copy_from_slice(&999u32.to_le_bytes());
        let checksum = fnv1a64(&payload);
        payload.extend_from_slice(&checksum.to_le_bytes());
        assert_eq!(
            PersistedPsoCache::decode(&payload),
            Err(PersistError::UnsupportedVersion(999))
        );
    }

    #[test]
    fn decode_rejects_checksum_mismatch() {
        let mut bytes = sample().encode();
        let last = bytes.len() - 1;
        bytes[last] ^= 0xFF;
        assert_eq!(PersistedPsoCache::decode(&bytes), Err(PersistError::ChecksumMismatch));
    }

    #[test]
    fn decode_rejects_truncation() {
        let bytes = sample().encode();
        let truncated = &bytes[..bytes.len() - 10];
        // A truncated blob fails checksum (or EOF for very short inputs).
        assert!(matches!(
            PersistedPsoCache::decode(truncated),
            Err(PersistError::ChecksumMismatch) | Err(PersistError::UnexpectedEof)
        ));
    }

    #[test]
    fn decode_rejects_short_input() {
        assert_eq!(PersistedPsoCache::decode(&[0u8; 4]), Err(PersistError::UnexpectedEof));
    }

    #[test]
    fn decode_rejects_bad_backend_tag() {
        let bytes = sample().encode();
        let mut payload = bytes[..bytes.len() - 8].to_vec();
        // Backend tag sits right after magic(4) + version(4).
        payload[8] = 200;
        let checksum = fnv1a64(&payload);
        payload.extend_from_slice(&checksum.to_le_bytes());
        assert_eq!(
            PersistedPsoCache::decode(&payload),
            Err(PersistError::BadBackendTag(200))
        );
    }

    #[test]
    fn other_backend_discriminant_roundtrips() {
        let mut cache = PersistedPsoCache::new(DeviceFingerprint::new(
            GraphicsBackend::Other(77),
            1,
            2,
            3,
            AbiHash::ZERO,
        ));
        cache.insert(key("p", 0, 0), vec![0xAB]);
        let decoded = PersistedPsoCache::decode(&cache.encode()).unwrap();
        assert_eq!(decoded.fingerprint().backend, GraphicsBackend::Other(77));
    }

    #[test]
    fn empty_cache_roundtrips() {
        let cache = PersistedPsoCache::new(fp());
        let decoded = PersistedPsoCache::decode(&cache.encode()).unwrap();
        assert!(decoded.is_empty());
        assert_eq!(decoded.fingerprint(), &fp());
    }

    #[test]
    fn hydrate_into_charges_blob_sizes_and_evicts_in_key_order() {
        let mut cache = PersistedPsoCache::new(fp());
        cache.insert(key("a", 0, 0), vec![0u8; 400]);
        cache.insert(key("b", 0, 0), vec![0u8; 400]);
        cache.insert(key("c", 0, 0), vec![0u8; 400]);

        let mut lru = LruPsoCache::with_budget(1000);
        let resident = cache.hydrate_into(&mut lru);
        // 1200 bytes into a 1000 budget: admitting in key order a,b,c evicts the
        // least-recently-used (a) once c overflows.
        assert_eq!(resident, 2);
        assert!(!lru.contains(&key("a", 0, 0)));
        assert!(lru.contains(&key("b", 0, 0)));
        assert!(lru.contains(&key("c", 0, 0)));
    }
}
