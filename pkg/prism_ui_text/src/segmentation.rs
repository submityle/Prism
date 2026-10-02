//! Grapheme-cluster and word segmentation.
//!
//! This module is a thin, allocation-aware wrapper over
//! [`unicode_segmentation`] that returns byte-indexed [`Segment`] values and a
//! handful of grapheme-boundary helpers used by the cursor model.

use alloc::string::String;
use alloc::vec::Vec;
use unicode_segmentation::UnicodeSegmentation;

/// A contiguous slice of the source text identified by a byte range.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Segment {
    /// Byte offset of the first byte of the segment within the source string.
    pub start: usize,
    /// Byte offset one past the final byte of the segment.
    pub end: usize,
    /// Owned copy of the segment text, for convenient inspection.
    pub text: String,
}

impl Segment {
    /// Returns the length of the segment in bytes.
    #[must_use]
    pub fn len_bytes(&self) -> usize {
        self.end - self.start
    }

    /// Returns `true` when the segment spans no bytes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.start == self.end
    }
}

/// Returns the extended grapheme clusters of `text` with byte offsets.
#[must_use]
pub fn graphemes(text: &str) -> Vec<Segment> {
    text.grapheme_indices(true)
        .map(|(i, g)| Segment {
            start: i,
            end: i + g.len(),
            text: String::from(g),
        })
        .collect()
}

/// Returns the number of extended grapheme clusters in `text`.
#[must_use]
pub fn grapheme_count(text: &str) -> usize {
    text.graphemes(true).count()
}

/// Returns the Unicode words of `text`, excluding whitespace and punctuation.
#[must_use]
pub fn words(text: &str) -> Vec<Segment> {
    text.unicode_word_indices()
        .map(|(i, w)| Segment {
            start: i,
            end: i + w.len(),
            text: String::from(w),
        })
        .collect()
}

/// Returns every word-boundary segment, including whitespace and punctuation.
#[must_use]
pub fn word_bounds(text: &str) -> Vec<Segment> {
    text.split_word_bound_indices()
        .map(|(i, w)| Segment {
            start: i,
            end: i + w.len(),
            text: String::from(w),
        })
        .collect()
}

/// Collects every grapheme-cluster boundary offset, including `0` and the
/// final length. The returned vector is sorted and always contains at least
/// one element.
fn boundaries(text: &str) -> Vec<usize> {
    let mut result: Vec<usize> = text.grapheme_indices(true).map(|(i, _)| i).collect();
    result.push(text.len());
    result
}

/// Returns `true` when `offset` lies on a grapheme-cluster boundary.
#[must_use]
pub fn is_grapheme_boundary(text: &str, offset: usize) -> bool {
    boundaries(text).contains(&offset)
}

/// Returns the first grapheme boundary strictly greater than `offset`.
///
/// Offsets at or beyond the end of `text` return the text length.
#[must_use]
pub fn next_grapheme_boundary(text: &str, offset: usize) -> usize {
    for b in boundaries(text) {
        if b > offset {
            return b;
        }
    }
    text.len()
}

/// Returns the last grapheme boundary strictly less than `offset`.
///
/// Offsets at or below `0` return `0`.
#[must_use]
pub fn prev_grapheme_boundary(text: &str, offset: usize) -> usize {
    let mut best = 0;
    for b in boundaries(text) {
        if b < offset {
            best = b;
        } else {
            break;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graphemes_split_combining_marks() {
        // "e" + combining acute accent is one grapheme cluster.
        let g = graphemes("e\u{0301}z");
        assert_eq!(g.len(), 2);
        assert_eq!(g[0].text, "e\u{0301}");
        assert_eq!(g[1].text, "z");
        assert_eq!(g[0].len_bytes(), 3);
    }

    #[test]
    fn grapheme_count_counts_clusters() {
        assert_eq!(grapheme_count("abc"), 3);
        assert_eq!(grapheme_count(""), 0);
        assert_eq!(grapheme_count("e\u{0301}"), 1);
    }

    #[test]
    fn words_excludes_punctuation() {
        let w = words("hi, world!");
        assert_eq!(w.len(), 2);
        assert_eq!(w[0].text, "hi");
        assert_eq!(w[1].text, "world");
    }

    #[test]
    fn word_bounds_include_separators() {
        let w = word_bounds("a b");
        let joined: String = w.iter().map(|s| s.text.clone()).collect();
        assert_eq!(joined, "a b");
        assert!(w.len() >= 3);
    }

    #[test]
    fn boundary_navigation_is_symmetric() {
        let text = "e\u{0301}z";
        assert!(is_grapheme_boundary(text, 0));
        assert!(!is_grapheme_boundary(text, 1));
        assert_eq!(next_grapheme_boundary(text, 0), 3);
        assert_eq!(prev_grapheme_boundary(text, 3), 0);
        assert_eq!(next_grapheme_boundary(text, 3), 4);
        assert_eq!(prev_grapheme_boundary(text, 0), 0);
    }

    #[test]
    fn empty_text_has_single_boundary() {
        assert!(is_grapheme_boundary("", 0));
        assert_eq!(next_grapheme_boundary("", 0), 0);
        assert_eq!(prev_grapheme_boundary("", 5), 0);
    }
}
