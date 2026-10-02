//! UAX#14 line-break opportunities and greedy width-based wrapping.
//!
//! [`break_opportunities`] exposes the raw break points reported by
//! [`unicode_linebreak`]. [`wrap_by_width`] turns those opportunities into
//! concrete lines using display-column widths from [`unicode_width`], so the
//! result is deterministic and font-independent (one display column is treated
//! as one unit of width).

use alloc::vec::Vec;
use unicode_linebreak::{linebreaks, BreakOpportunity};
use unicode_width::UnicodeWidthStr;

/// Whether a break opportunity must be taken or merely may be taken.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BreakKind {
    /// A hard break that must be taken (for example after a line feed).
    Mandatory,
    /// A soft break that may be taken to satisfy a width constraint.
    Allowed,
}

/// A single UAX#14 break opportunity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BreakPoint {
    /// Byte offset immediately after which a break may occur.
    pub offset: usize,
    /// Whether the break is mandatory or merely allowed.
    pub kind: BreakKind,
}

/// A line produced by [`wrap_by_width`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WrappedLine {
    /// Byte offset of the first byte of the line.
    pub start: usize,
    /// Byte offset one past the final byte of the line.
    pub end: usize,
    /// Display-column width of the line content.
    pub width: usize,
    /// Whether the line ended at a mandatory (hard) break.
    pub hard_break: bool,
}

impl WrappedLine {
    /// Returns the byte length of the line.
    #[must_use]
    pub fn len_bytes(&self) -> usize {
        self.end - self.start
    }

    /// Returns `true` when the line spans no bytes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.start == self.end
    }
}

/// Returns every UAX#14 break opportunity for `text`.
///
/// The final element is always a mandatory break at `text.len()`.
#[must_use]
pub fn break_opportunities(text: &str) -> Vec<BreakPoint> {
    linebreaks(text)
        .map(|(offset, kind)| BreakPoint {
            offset,
            kind: match kind {
                BreakOpportunity::Mandatory => BreakKind::Mandatory,
                BreakOpportunity::Allowed => BreakKind::Allowed,
            },
        })
        .collect()
}

/// Greedily wraps `text` to `max_width` display columns.
///
/// Mandatory breaks always start a new line. Soft breaks are taken when the
/// next inter-break piece would overflow `max_width` and the current line is
/// not empty, so a single piece wider than `max_width` is still emitted on its
/// own line rather than being dropped. Widths are measured with
/// [`unicode_width`], counting one display column as one unit.
#[must_use]
pub fn wrap_by_width(text: &str, max_width: usize) -> Vec<WrappedLine> {
    let mut lines = Vec::new();
    let mut cur_start = 0usize;
    let mut cur_end = 0usize;
    let mut cur_width = 0usize;
    let mut prev = 0usize;

    for bp in break_opportunities(text) {
        let piece_width = text[prev..bp.offset].width();
        let piece_start = prev;
        prev = bp.offset;

        if cur_end == cur_start {
            // Current line is empty: always accept the piece.
            cur_end = bp.offset;
            cur_width = piece_width;
        } else if cur_width + piece_width > max_width {
            // The piece would overflow: break before it onto a fresh line.
            lines.push(WrappedLine {
                start: cur_start,
                end: cur_end,
                width: cur_width,
                hard_break: false,
            });
            cur_start = piece_start;
            cur_end = bp.offset;
            cur_width = piece_width;
        } else {
            cur_end = bp.offset;
            cur_width += piece_width;
        }

        if bp.kind == BreakKind::Mandatory {
            lines.push(WrappedLine {
                start: cur_start,
                end: cur_end,
                width: cur_width,
                hard_break: true,
            });
            cur_start = cur_end;
            cur_width = 0;
        }
    }

    if cur_end > cur_start || lines.is_empty() {
        lines.push(WrappedLine {
            start: cur_start,
            end: cur_end,
            width: cur_width,
            hard_break: false,
        });
    }

    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opportunities_end_with_mandatory() {
        let bps = break_opportunities("ab");
        let last = bps.last().copied().unwrap_or(BreakPoint {
            offset: 0,
            kind: BreakKind::Allowed,
        });
        assert_eq!(last.offset, 2);
        assert_eq!(last.kind, BreakKind::Mandatory);
    }

    #[test]
    fn newline_is_mandatory() {
        let bps = break_opportunities("a\nb");
        assert!(bps
            .iter()
            .any(|b| b.offset == 2 && b.kind == BreakKind::Mandatory));
    }

    #[test]
    fn wrap_splits_on_width() {
        // Pieces carry their trailing space, so "aaa bbb " is 8 columns wide.
        let lines = wrap_by_width("aaa bbb ccc", 8);
        assert_eq!(lines.len(), 2);
        assert_eq!(&"aaa bbb ccc"[lines[0].start..lines[0].end], "aaa bbb ");
        assert_eq!(&"aaa bbb ccc"[lines[1].start..lines[1].end], "ccc");
    }

    #[test]
    fn wrap_hard_break_splits_lines() {
        let lines = wrap_by_width("ab\ncd", 100);
        assert_eq!(lines.len(), 2);
        assert!(lines[0].hard_break);
        assert_eq!(&"ab\ncd"[lines[0].start..lines[0].end], "ab\n");
        assert_eq!(&"ab\ncd"[lines[1].start..lines[1].end], "cd");
    }

    #[test]
    fn wrap_oversized_piece_is_kept() {
        let lines = wrap_by_width("abcdefgh", 3);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0].width, 8);
    }

    #[test]
    fn wrap_empty_text_yields_one_line() {
        let lines = wrap_by_width("", 10);
        assert_eq!(lines.len(), 1);
        assert!(lines[0].is_empty());
    }
}
