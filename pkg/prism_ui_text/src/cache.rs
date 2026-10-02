//! A shaping-result cache keyed by `(text, style, width)`.
//!
//! Shaping is comparatively expensive, so [`ShapeCache`] memoises
//! [`ShapedRun`]s. Only text whose key is not already present is reshaped,
//! which lets callers re-render unchanged runs for free. The float components
//! of the key are stored by their raw bit pattern via [`f32::to_bits`], giving
//! an exact, order-stable key without any transcendental maths.

use crate::rich_text::TextStyle;
use crate::shaper::{ShapedRun, Shaper};
use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

/// A deterministic, hashable identity for a shaping request.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct CacheKey {
    /// Source text.
    pub text: String,
    /// Font size encoded as raw IEEE-754 bits for exact comparison.
    pub font_size_bits: u32,
    /// Font weight value.
    pub weight: u16,
    /// Whether the run is italic.
    pub italic: bool,
    /// Underline flag.
    pub underline: bool,
    /// Wrapping width encoded as raw IEEE-754 bits for exact comparison.
    pub width_bits: u32,
}

impl CacheKey {
    /// Builds a cache key from text, a style and a wrapping width.
    #[must_use]
    pub fn new(text: &str, style: &TextStyle, width: f32) -> Self {
        Self {
            text: String::from(text),
            font_size_bits: style.font_size.to_bits(),
            weight: style.weight.0,
            italic: style.italic,
            underline: style.underline,
            width_bits: width.to_bits(),
        }
    }
}

/// A memoising store of [`ShapedRun`]s keyed by [`CacheKey`].
#[derive(Clone, Debug, Default)]
pub struct ShapeCache {
    /// Backing map of cached runs.
    entries: BTreeMap<CacheKey, ShapedRun>,
}

impl ShapeCache {
    /// Builds an empty cache.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: BTreeMap::new(),
        }
    }

    /// Returns the number of cached runs.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` when the cache holds no runs.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns `true` when `key` is already cached.
    #[must_use]
    pub fn contains(&self, key: &CacheKey) -> bool {
        self.entries.contains_key(key)
    }

    /// Returns the cached run for `key`, if present.
    #[must_use]
    pub fn get(&self, key: &CacheKey) -> Option<&ShapedRun> {
        self.entries.get(key)
    }

    /// Inserts `run` under `key`, returning any previously cached run.
    pub fn insert(&mut self, key: CacheKey, run: ShapedRun) -> Option<ShapedRun> {
        self.entries.insert(key, run)
    }

    /// Removes the cached run for `key`, returning it if present.
    pub fn invalidate(&mut self, key: &CacheKey) -> Option<ShapedRun> {
        self.entries.remove(key)
    }

    /// Clears all cached runs.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Returns the cache keys in sorted order.
    #[must_use]
    pub fn keys(&self) -> Vec<CacheKey> {
        self.entries.keys().cloned().collect()
    }

    /// Returns the cached run for `(text, style, width)`, shaping and storing it
    /// with `shaper` on a miss.
    pub fn shape_cached<S: Shaper>(
        &mut self,
        shaper: &S,
        text: &str,
        style: &TextStyle,
        width: f32,
    ) -> &ShapedRun {
        let key = CacheKey::new(text, style, width);
        self.entries
            .entry(key)
            .or_insert_with(|| shaper.shape(text, style))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shaper::MetricShaper;

    #[test]
    fn miss_then_hit() {
        let mut cache = ShapeCache::new();
        assert!(cache.is_empty());
        let shaper = MetricShaper::default();
        let style = TextStyle::default();
        let width = {
            let run = cache.shape_cached(&shaper, "abc", &style, 100.0);
            run.width()
        };
        assert_eq!(cache.len(), 1);
        // Second call is a hit and returns the same geometry.
        let again = cache.shape_cached(&shaper, "abc", &style, 100.0);
        assert_eq!(again.width(), width);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn distinct_keys_distinct_entries() {
        let mut cache = ShapeCache::new();
        let shaper = MetricShaper::default();
        let style = TextStyle::default();
        cache.shape_cached(&shaper, "abc", &style, 100.0);
        cache.shape_cached(&shaper, "abc", &style, 200.0);
        cache.shape_cached(&shaper, "abc", &style.bold(), 100.0);
        assert_eq!(cache.len(), 3);
    }

    #[test]
    fn invalidate_and_clear() {
        let mut cache = ShapeCache::new();
        let shaper = MetricShaper::default();
        let style = TextStyle::default();
        let key = CacheKey::new("abc", &style, 100.0);
        cache.shape_cached(&shaper, "abc", &style, 100.0);
        assert!(cache.contains(&key));
        assert!(cache.invalidate(&key).is_some());
        assert!(!cache.contains(&key));
        cache.shape_cached(&shaper, "x", &style, 10.0);
        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn key_uses_exact_float_bits() {
        let style = TextStyle::default();
        let a = CacheKey::new("x", &style, 1.0);
        let b = CacheKey::new("x", &style, 1.0);
        let c = CacheKey::new("x", &style, 2.0);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn keys_are_sorted_and_insertable() {
        let mut cache = ShapeCache::new();
        let shaper = MetricShaper::default();
        let style = TextStyle::default();
        let run = shaper.shape("manual", &style);
        let key = CacheKey::new("manual", &style, 50.0);
        assert!(cache.insert(key.clone(), run).is_none());
        let keys = cache.keys();
        assert_eq!(keys.len(), 1);
        assert_eq!(keys[0], key);
        assert!(cache.get(&key).is_some());
    }
}
