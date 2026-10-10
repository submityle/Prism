//! Loader selection policy: mapping an [`AssetPath`] to the loader that should
//! decode it (design §9.1).
//!
//! The *execution* half of a loader — the `async fn load(...)` that reads bytes
//! and decodes them — necessarily depends on `std`/async IO and lives in a
//! later milestone's execution crate. The *policy* half — "given this path (and
//! maybe a requested asset type), which registered loader claims it?" — is pure
//! data and decision logic, so it lives here in the deterministic `no_std`
//! core, fully unit-testable and shared by every backend.
//!
//! ## Rules (design §9.1)
//! Extensions are only a *hint* for picking a loader; identity is the
//! [`StableGuid`](crate::StableGuid), never the suffix. This registry therefore
//! supports the flexible mapping the design calls for:
//!
//! - **One loader, many extensions**: a loader may claim any number of suffixes.
//! - **Longest-suffix wins**: compound suffixes such as `tar.zst`, `gltf`, or
//!   `ktx2` are matched before a shorter `zst` could steal them.
//! - **Case-insensitive**: `.PNG` and `.png` are equivalent.
//! - **Type-priority disambiguation**: a typed request (`for_type`) only selects
//!   loaders that produce the requested [`AssetTypeId`]; an untyped request
//!   falls back to registration priority, where a later registration can
//!   explicitly override an earlier one.
//! - **Aliases**: a suffix can be remapped to a canonical one (`jpeg → jpg`), so
//!   a loader claiming `jpg` also handles `jpeg` without re-declaring it.
//! - **Custom suffixes are first-class**: there is no built-in whitelist;
//!   `.lvl`, `.mymesh`, `.myasset` work the moment a loader registers them.
//!
//! Content/magic-byte sniffing (for unreliable or absent suffixes) is a
//! separate fallback layered on top by the execution crate; this module models
//! the deterministic suffix policy it falls back *from*.

use crate::path::AssetPath;
use crate::type_id::AssetTypeId;
use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

/// An opaque handle to a registered loader slot.
///
/// The registry hands these out in registration order. The execution crate
/// keeps the parallel table of trait objects keyed by the same id, so this core
/// never needs to name a concrete loader type.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct LoaderId(u32);

impl LoaderId {
    /// The raw slot index, for pairing with the execution-side loader table.
    #[must_use]
    pub const fn index(self) -> u32 {
        self.0
    }
}

impl fmt::Display for LoaderId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LoaderId({})", self.0)
    }
}

/// A diagnostic emitted during registration when two loaders claim the same
/// canonical suffix (design §9.1: "conflicts can be diagnosed at registration
/// time"). It is advisory — registration still succeeds, and the conflict is
/// resolved deterministically at resolve time — but surfacing it lets tooling
/// warn authors about ambiguous suffix maps.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct SuffixConflict {
    /// The canonical suffix (no leading dot, lowercased) claimed more than once.
    pub suffix: String,
    /// The already-registered loader claiming it.
    pub existing: LoaderId,
    /// The loader just registered that also claims it.
    pub incoming: LoaderId,
}

/// Normalizes a user-supplied suffix: drops any leading dots and lowercases it.
/// `".PNG"`, `"png"`, and `"PNG"` all normalize to `"png"`.
fn normalize_suffix(ext: &str) -> String {
    ext.trim_start_matches('.').to_ascii_lowercase()
}

/// One registered loader's selection metadata.
struct LoaderEntry {
    id: LoaderId,
    produced_type: AssetTypeId,
    /// Normalized claimed suffixes (no dot, lowercased), deduplicated.
    extensions: Vec<String>,
    /// Higher wins. Ties break on registration sequence (later wins), so a
    /// later registration can explicitly override an earlier one.
    priority: i32,
    seq: u64,
}

impl LoaderEntry {
    fn claims(&self, canonical: &str) -> bool {
        self.extensions.iter().any(|e| e == canonical)
    }
}

/// The deterministic loader-selection table (design §9.1).
///
/// Registration order and the `(priority, seq)` tie-break make every
/// resolution reproducible across runs, which the networked/consistency tests
/// rely on.
#[derive(Default)]
pub struct LoaderRegistry {
    entries: Vec<LoaderEntry>,
    /// Alias suffix → canonical suffix (both normalized), e.g. `jpeg → jpg`.
    aliases: BTreeMap<String, String>,
    next_seq: u64,
}

impl LoaderRegistry {
    /// Creates an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a loader that produces `produced_type` and claims every suffix
    /// in `extensions` at the given `priority` (higher wins; ties break toward
    /// the later registration). Returns the new [`LoaderId`] and any
    /// [`SuffixConflict`]s with already-registered loaders — advisory only.
    pub fn register<I, S>(
        &mut self,
        produced_type: AssetTypeId,
        extensions: I,
        priority: i32,
    ) -> (LoaderId, Vec<SuffixConflict>)
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let id = LoaderId(self.entries.len() as u32);
        let seq = self.next_seq;
        self.next_seq += 1;

        let mut exts: Vec<String> = Vec::new();
        for ext in extensions {
            let norm = normalize_suffix(ext.as_ref());
            if !norm.is_empty() && !exts.contains(&norm) {
                exts.push(norm);
            }
        }

        // Detect conflicts against existing loaders claiming the same canonical
        // suffix (before inserting, so we don't report self-conflicts).
        let mut conflicts = Vec::new();
        for ext in &exts {
            for existing in &self.entries {
                if existing.claims(ext) {
                    conflicts.push(SuffixConflict {
                        suffix: ext.clone(),
                        existing: existing.id,
                        incoming: id,
                    });
                }
            }
        }

        self.entries.push(LoaderEntry {
            id,
            produced_type,
            extensions: exts,
            priority,
            seq,
        });
        (id, conflicts)
    }

    /// Remaps `alias` to the canonical suffix `canonical` (both normalized), so
    /// a filename ending in `alias` resolves to loaders claiming `canonical`.
    pub fn add_alias(&mut self, alias: &str, canonical: &str) {
        let alias = normalize_suffix(alias);
        let canonical = normalize_suffix(canonical);
        if !alias.is_empty() && !canonical.is_empty() && alias != canonical {
            self.aliases.insert(alias, canonical);
        }
    }

    /// The produced type of a registered loader, if the id is known.
    #[must_use]
    pub fn produced_type(&self, id: LoaderId) -> Option<AssetTypeId> {
        self.entries
            .iter()
            .find(|e| e.id == id)
            .map(|e| e.produced_type)
    }

    /// The number of registered loaders.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether no loaders are registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The lowercased final path segment (filename) of `path`, ignoring scheme
    /// and `#label` (identity components, not suffix hints).
    fn file_name(path: &AssetPath) -> String {
        let p = path.path();
        let name = match p.rfind('/') {
            Some(slash) => &p[slash + 1..],
            None => p,
        };
        name.to_ascii_lowercase()
    }

    /// Finds the longest claimed-or-aliased suffix that `file_name` ends with,
    /// and returns the *canonical* suffix (aliases resolved). Requires a
    /// non-empty stem before the suffix so a bare dotfile like `.png` does not
    /// match.
    fn match_canonical_suffix(&self, file_name: &str) -> Option<String> {
        // Candidate tokens: every claimed extension plus every alias key. The
        // longest token that the filename ends with (after a dot, with a stem)
        // wins, so compound suffixes beat their shorter tails.
        let mut best: Option<&str> = None;
        // Iterate owned sources so the matched token can be borrowed from the
        // registry's data. The longest matching token wins; on an exact length
        // tie we keep the first seen (deterministic, and resolution later
        // disambiguates by `(priority, seq)` anyway).
        for entry in &self.entries {
            for ext in &entry.extensions {
                if best.is_none_or(|b| ext.len() > b.len())
                    && Self::ends_with_suffix(file_name, ext)
                {
                    best = Some(ext);
                }
            }
        }
        for alias in self.aliases.keys() {
            if best.is_none_or(|b| alias.len() > b.len())
                && Self::ends_with_suffix(file_name, alias)
            {
                best = Some(alias);
            }
        }

        let matched = best?;
        // Resolve an alias token to its canonical suffix.
        Some(
            self.aliases
                .get(matched)
                .cloned()
                .unwrap_or_else(|| matched.into()),
        )
    }

    /// Whether `file_name` ends with `.suffix` and has a non-empty stem.
    fn ends_with_suffix(file_name: &str, suffix: &str) -> bool {
        let needed = suffix.len() + 1;
        file_name.len() > needed
            && file_name.as_bytes()[file_name.len() - needed] == b'.'
            && file_name.ends_with(suffix)
    }

    /// Resolves the loader for a *typed* request: only loaders producing
    /// `requested` are eligible. Among eligible loaders claiming the matched
    /// canonical suffix, the highest `(priority, seq)` wins. Returns `None` if
    /// no suffix matches or no matching loader produces the requested type.
    #[must_use]
    pub fn resolve_for_type(&self, path: &AssetPath, requested: AssetTypeId) -> Option<LoaderId> {
        let file_name = Self::file_name(path);
        let canonical = self.match_canonical_suffix(&file_name)?;
        self.best_claimer(&canonical, Some(requested))
    }

    /// Resolves the loader for an *untyped* request: any loader claiming the
    /// matched canonical suffix is eligible, and the highest `(priority, seq)`
    /// wins (so a later registration can explicitly override an earlier one).
    #[must_use]
    pub fn resolve_untyped(&self, path: &AssetPath) -> Option<LoaderId> {
        let file_name = Self::file_name(path);
        let canonical = self.match_canonical_suffix(&file_name)?;
        self.best_claimer(&canonical, None)
    }

    /// Picks the best loader claiming `canonical`, optionally restricted to one
    /// producing `requested`. "Best" = highest `priority`, then highest `seq`
    /// (later registration), deterministic.
    fn best_claimer(&self, canonical: &str, requested: Option<AssetTypeId>) -> Option<LoaderId> {
        self.entries
            .iter()
            .filter(|e| e.claims(canonical))
            .filter(|e| requested.is_none_or(|t| e.produced_type == t))
            .max_by_key(|e| (e.priority, e.seq))
            .map(|e| e.id)
    }
}
