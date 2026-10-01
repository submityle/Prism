//! Reversible escaping of field values used by the textual snapshot format.
//!
//! The snapshot grammar packs several fields onto a single line, separated by
//! spaces, and separates class names with commas. To keep that grammar
//! unambiguous for arbitrary user strings, every field value is escaped so it
//! can never contain a raw space, comma, or line break. [`escape`] performs
//! that transformation and [`unescape`] inverts it exactly, giving a lossless
//! round trip for any `&str`.

use alloc::string::String;

/// Escapes `input` so it contains no spaces, commas, tabs, or line breaks.
///
/// The backslash character is the escape introducer, so it is doubled. The
/// transformation is total and reversible by [`unescape`].
pub(crate) fn escape(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for ch in input.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            ' ' => out.push_str("\\s"),
            ',' => out.push_str("\\c"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            other => out.push(other),
        }
    }
    out
}

/// Reverses [`escape`], returning `None` on a malformed escape sequence.
///
/// A sequence is malformed when a backslash is the final character or is
/// followed by a byte that [`escape`] never emits.
pub(crate) fn unescape(input: &str) -> Option<String> {
    let mut out = String::with_capacity(input.len());
    let mut chars = input.chars();
    while let Some(ch) = chars.next() {
        if ch == '\\' {
            match chars.next() {
                Some('\\') => out.push('\\'),
                Some('s') => out.push(' '),
                Some('c') => out.push(','),
                Some('n') => out.push('\n'),
                Some('r') => out.push('\r'),
                Some('t') => out.push('\t'),
                _ => return None,
            }
        } else {
            out.push(ch);
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::{escape, unescape};
    use alloc::string::ToString;

    #[test]
    fn escapes_special_characters() {
        assert_eq!(escape("a b"), "a\\sb");
        assert_eq!(escape("a,b"), "a\\cb");
        assert_eq!(escape("a\\b"), "a\\\\b");
        assert_eq!(escape("a\nb\r\tc"), "a\\nb\\r\\tc");
    }

    #[test]
    fn plain_text_is_unchanged() {
        assert_eq!(escape("Button"), "Button");
        assert_eq!(unescape("Button"), Some("Button".to_string()));
    }

    #[test]
    fn round_trips_arbitrary_strings() {
        for s in [
            "",
            " ",
            ",",
            "\\",
            "\n",
            "a\\s b, c\td\n",
            "mixed \\\\ , \n end",
        ] {
            let escaped = escape(s);
            assert!(!escaped.contains(' '));
            assert!(!escaped.contains(','));
            assert!(!escaped.contains('\n'));
            assert_eq!(unescape(&escaped).as_deref(), Some(s));
        }
    }

    #[test]
    fn rejects_malformed_escapes() {
        assert_eq!(unescape("abc\\"), None);
        assert_eq!(unescape("a\\xb"), None);
    }
}
