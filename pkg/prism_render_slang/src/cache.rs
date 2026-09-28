//! Content-addressed variant cache.
//!
//! Specialization produces many permutations of a base module (for example
//! `uber + clearcoat + Lit` vs `uber + Stylized`). Each is keyed by a
//! deterministic hash over the source bytes, the target, the entry/stage, and
//! the sorted defines, so an identical request is only compiled once.
//!
//! The key derivation is dependency-free and fully deterministic (see
//! [`crate::hash`]); it does not depend on `slangc` being present, so it is
//! unit-testable in isolation.

use std::collections::HashMap;

use crate::compile::{CompileRequest, Define};
use crate::hash::Fnv1a;
use crate::target::{ShaderStage, Target};

/// A stable identity for one compiled variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VariantKey(pub u64);

impl VariantKey {
    /// Derive a key from a full [`CompileRequest`] plus the source bytes.
    ///
    /// The source *contents* are hashed (not the path) so moving or renaming
    /// a file does not invalidate an otherwise-identical variant, while any
    /// real source change does.
    pub fn from_request(request: &CompileRequest, source_bytes: &[u8]) -> Self {
        Self::derive(
            source_bytes,
            request.target,
            &request.entry,
            request.stage,
            &request.defines,
        )
    }

    /// Derive a key from the individual inputs.
    pub fn derive(
        source_bytes: &[u8],
        target: Target,
        entry: &str,
        stage: ShaderStage,
        defines: &[Define],
    ) -> Self {
        let mut h = Fnv1a::new();
        h.write_framed("prism.slang.variant.v1");
        h.write(source_bytes);
        h.write_framed(target.slangc_name());
        h.write_framed(entry);
        h.write_framed(stage.slangc_name());

        // Defines are order-independent: sort a normalized token list first.
        let mut tokens: Vec<String> = defines.iter().map(Define::as_arg).collect();
        tokens.sort();
        h.write_u64(tokens.len() as u64);
        for token in tokens {
            h.write_framed(&token);
        }

        VariantKey(h.finish())
    }
}

/// One cached artifact: the compiled bytes plus optional reflection JSON.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CachedVariant {
    /// Compiled artifact bytes (WGSL/SPIR-V/Metal/C++ text or blob).
    pub artifact: Vec<u8>,
    /// Reflection JSON text, if it was produced.
    pub reflection_json: Option<String>,
}

/// An in-memory variant cache.
///
/// Persisting to disk is intentionally left to callers (a build script may
/// key files by [`VariantKey`]); this type owns only the dedup logic.
#[derive(Debug, Default)]
pub struct VariantCache {
    entries: HashMap<VariantKey, CachedVariant>,
    hits: u64,
    misses: u64,
}

impl VariantCache {
    /// Create an empty cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// Fetch a cached variant, recording a hit or miss.
    pub fn get(&mut self, key: VariantKey) -> Option<&CachedVariant> {
        if self.entries.contains_key(&key) {
            self.hits += 1;
            self.entries.get(&key)
        } else {
            self.misses += 1;
            None
        }
    }

    /// Insert or replace a variant.
    pub fn insert(&mut self, key: VariantKey, variant: CachedVariant) {
        self.entries.insert(key, variant);
    }

    /// Number of stored variants.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache holds no variants.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Recorded hit count.
    pub fn hits(&self) -> u64 {
        self.hits
    }

    /// Recorded miss count.
    pub fn misses(&self) -> u64 {
        self.misses
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_is_stable_for_identical_inputs() {
        let a = VariantKey::derive(b"src", Target::Wgsl, "main", ShaderStage::Compute, &[]);
        let b = VariantKey::derive(b"src", Target::Wgsl, "main", ShaderStage::Compute, &[]);
        assert_eq!(a, b);
    }

    #[test]
    fn key_changes_with_source_target_and_defines() {
        let base = VariantKey::derive(b"src", Target::Wgsl, "main", ShaderStage::Compute, &[]);
        let diff_src =
            VariantKey::derive(b"src2", Target::Wgsl, "main", ShaderStage::Compute, &[]);
        let diff_target =
            VariantKey::derive(b"src", Target::Spirv, "main", ShaderStage::Compute, &[]);
        let diff_define = VariantKey::derive(
            b"src",
            Target::Wgsl,
            "main",
            ShaderStage::Compute,
            &[Define::flag("USE_RT")],
        );
        assert_ne!(base, diff_src);
        assert_ne!(base, diff_target);
        assert_ne!(base, diff_define);
    }

    #[test]
    fn defines_are_order_independent() {
        let a = VariantKey::derive(
            b"src",
            Target::Wgsl,
            "main",
            ShaderStage::Compute,
            &[Define::flag("A"), Define::flag("B")],
        );
        let b = VariantKey::derive(
            b"src",
            Target::Wgsl,
            "main",
            ShaderStage::Compute,
            &[Define::flag("B"), Define::flag("A")],
        );
        assert_eq!(a, b);
    }

    #[test]
    fn cache_dedups_and_tracks_hits() {
        let mut cache = VariantCache::new();
        let key = VariantKey(42);
        assert!(cache.get(key).is_none());
        assert_eq!(cache.misses(), 1);
        cache.insert(
            key,
            CachedVariant {
                artifact: b"wgsl".to_vec(),
                reflection_json: None,
            },
        );
        assert!(cache.get(key).is_some());
        assert_eq!(cache.hits(), 1);
        assert_eq!(cache.len(), 1);
    }
}
