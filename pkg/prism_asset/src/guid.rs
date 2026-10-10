//! Stable, content-addressed asset identity.
//!
//! [`AssetIndex`](crate::AssetIndex) is a *runtime* identity: it is dense,
//! `Copy`, and cheap, but only meaningful within one process run (its slot and
//! generation are assigned on load). It must never be written to a save file,
//! a container directory, or the network, because the same asset will land in
//! a different slot next run.
//!
//! [`StableGuid`] is the *persistent* identity: a 128-bit value derived purely
//! from an asset's source path (or its bytes), identical across runs, machines,
//! and platforms. Containers, content catalogs, save files, and cross-asset
//! references store `StableGuid`; the runtime interns it to an
//! [`AssetIndex`](crate::AssetIndex) on load.
//!
//! ## Stability contract
//! The derivation (domain tags + canonical FNV-1a 128-bit from
//! [`crate::hash`]) and the path-normalization rules in [`normalize_path`] are
//! a **frozen, versioned contract**. Changing either reshuffles every
//! previously stored guid, invalidating existing saves and container
//! directories, so a change must go through an explicit format-version bump and
//! a remap table. See `docs/prism_asset_design_zh.md` §20.

use crate::hash::{fnv1a_128, fnv1a_128_fold};
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

/// Domain-separation tags mixed into the hash so that identities minted from
/// different sources (a path, raw bytes, or a parent+label pair) occupy
/// disjoint spaces and can never collide across kinds.
///
/// These bytes are part of the frozen stability contract; do not reorder or
/// reuse a value.
mod domain {
    /// Guid derived from a normalized source path.
    pub const PATH: u8 = 0x01;
    /// Guid derived from raw content bytes.
    pub const CONTENT: u8 = 0x02;
    /// Guid derived from a parent guid plus a sub-asset label.
    pub const SUB_ASSET: u8 = 0x03;
}

/// A 128-bit, cross-run/-platform-stable asset identity.
///
/// `StableGuid` is `Copy` and 16 bytes, so soft references
/// ([`SoftHandle`](crate::SoftHandle)) can be stored by the million at zero I/O
/// cost. It is **not** cryptographic and carries no type tag; pair it with an
/// [`AssetTypeId`](crate::AssetTypeId) when type safety is required.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StableGuid(u128);

impl StableGuid {
    /// The nil guid (all zero), reserved to mean "no asset". No real asset
    /// hashes to it in practice, and APIs may use it as a sentinel.
    pub const NIL: Self = Self(0);

    /// Derives a guid from an asset source path.
    ///
    /// The path is normalized ([`normalize_path`]) before hashing, so
    /// `"a//b/../c.png"` and `"a/c.png"` yield the same guid. Callers should
    /// split a sub-asset `#label` off first and use [`StableGuid::derive_sub`]
    /// so that, for example, `scene.gltf#Mesh0` is a stable child of
    /// `scene.gltf` rather than an unrelated top-level guid.
    #[must_use]
    pub fn from_path(path: &str) -> Self {
        let normalized = normalize_path(path);
        Self::hash_domain(domain::PATH, normalized.as_bytes())
    }

    /// Derives a guid directly from content bytes (content-addressed identity),
    /// used by the bake pipeline for immutable, deduplicated artifacts.
    #[must_use]
    pub fn from_content(bytes: &[u8]) -> Self {
        Self::hash_domain(domain::CONTENT, bytes)
    }

    /// Derives a stable child guid for a sub-asset of `parent` selected by
    /// `label` (for example the `Mesh0` sub-asset of a glTF scene).
    ///
    /// The parent's 16 bytes are folded in first, then a `#` separator, then
    /// the label, under the sub-asset domain tag, so sibling sub-assets of the
    /// same parent get distinct guids and the same `(parent, label)` always
    /// reproduces.
    #[must_use]
    pub fn derive_sub(parent: StableGuid, label: &str) -> Self {
        let mut acc = fnv1a_128(&[domain::SUB_ASSET]);
        acc = fnv1a_128_fold(acc, &parent.0.to_le_bytes());
        acc = fnv1a_128_fold(acc, b"#");
        acc = fnv1a_128_fold(acc, label.as_bytes());
        Self(acc)
    }

    /// The raw 128-bit value, for serialization into containers/catalogs.
    #[must_use]
    pub const fn to_u128(self) -> u128 {
        self.0
    }

    /// Reconstructs a guid from a raw 128-bit value read back from storage.
    #[must_use]
    pub const fn from_u128(value: u128) -> Self {
        Self(value)
    }

    /// Whether this is the reserved [`StableGuid::NIL`] sentinel.
    #[must_use]
    pub const fn is_nil(self) -> bool {
        self.0 == 0
    }

    /// Hashes `bytes` under domain tag `tag`, equivalent to hashing
    /// `[tag] ++ bytes` with canonical FNV-1a 128-bit.
    fn hash_domain(tag: u8, bytes: &[u8]) -> Self {
        Self(fnv1a_128_fold(fnv1a_128(&[tag]), bytes))
    }
}

impl fmt::Debug for StableGuid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "StableGuid({:032x})", self.0)
    }
}

impl fmt::Display for StableGuid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:032x}", self.0)
    }
}

/// Normalizes a source path into its canonical form for stable hashing.
///
/// Rules (frozen contract):
/// - Backslashes `\` become forward slashes `/` (Windows authoring parity).
/// - Repeated slashes collapse (`a//b` → `a/b`).
/// - `.` segments are dropped.
/// - `..` segments pop the previous normal segment when one exists; a leading
///   `..` that cannot be popped is preserved for a relative path (it escapes
///   the root) but discarded for an absolute path (you cannot go above root).
/// - A single leading `/` (absolute) is preserved; trailing slashes are
///   removed (except the lone root `/`).
///
/// Normalization is pure ASCII-slash structural work; it does not lower-case or
/// touch Unicode, so case-sensitive sources stay distinct.
#[must_use]
pub fn normalize_path(path: &str) -> String {
    let absolute = path.starts_with('/') || path.starts_with('\\');
    let mut segments: Vec<&str> = Vec::new();
    for segment in path.split(['/', '\\']) {
        match segment {
            "" | "." => {}
            ".." => {
                if matches!(segments.last(), Some(&prev) if prev != "..") {
                    segments.pop();
                } else if !absolute {
                    segments.push("..");
                }
                // A leading `..` under an absolute root is discarded: you
                // cannot escape above root.
            }
            normal => segments.push(normal),
        }
    }

    let mut out = String::new();
    if absolute {
        out.push('/');
    }
    for (i, segment) in segments.iter().enumerate() {
        if i > 0 {
            out.push('/');
        }
        out.push_str(segment);
    }
    out
}
