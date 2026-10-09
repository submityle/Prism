//! A self-contained 5x7 bitmap font for printable ASCII (`0x20..=0x7E`).
//!
//! Loom's reference rasteriser turns text runs into *readable* pixels in two
//! tiers. The always-available base tier embeds a compact, hand-authored
//! 5-wide / 7-tall glyph table (this module) and exposes nearest-neighbour
//! sampling plus simple metrics — it needs no font files and works in
//! `no_std`. The optional [`vector`] tier (behind the `vector` feature) parses
//! a real embedded OpenType face with `ab_glyph` and rasterises Bézier
//! outlines for crisp text at any size. Together they are the `P0`
//! ("make text legible") and `B` ("CPU vector") stages of the roadmap in
//! `docs/prism_loom_text_rendering_design_zh.md`; the GPU `MSDF` atlas path
//! layered on top is future work and lives behind that design.
//!
//! Row encoding: each glyph is `[u8; GLYPH_H]`, one byte per scanline from top
//! to bottom. Only the low [`GLYPH_W`] bits are used; bit `GLYPH_W - 1` is the
//! left-most column and bit `0` the right-most. A set bit is an inked pixel.
//!
//! The glyph art is original to this crate (drawn directly in the binary
//! literals below) and contains no Unreal Engine source or derived code.

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

#[cfg(feature = "vector")]
extern crate alloc;

/// Real vector (TrueType outline) glyph rasterisation (the crisp-text tier).
#[cfg(feature = "vector")]
pub mod vector;

/// Single-channel SDF glyph atlas (any-scale sharp tier, GPU-friendly).
#[cfg(feature = "vector")]
pub mod sdf;

/// Glyph cell width in source pixels (columns).
pub const GLYPH_W: usize = 5;
/// Glyph cell height in source pixels (rows).
pub const GLYPH_H: usize = 7;
/// Recommended inter-glyph spacing in source pixels, so monospace advance is
/// [`GLYPH_W`] `+` [`GLYPH_GAP`].
pub const GLYPH_GAP: usize = 1;
/// Monospace horizontal advance in source pixels.
pub const GLYPH_ADVANCE: usize = GLYPH_W + GLYPH_GAP;

/// First printable ASCII codepoint covered by [`FONT`].
pub const FIRST: u8 = 0x20;
/// Last printable ASCII codepoint covered by [`FONT`].
pub const LAST: u8 = 0x7E;

/// Whether `(row, col)` of `glyph` is inked. `col` is measured from the left.
#[must_use]
#[inline]
pub fn pixel(glyph: &[u8; GLYPH_H], row: usize, col: usize) -> bool {
    if row >= GLYPH_H || col >= GLYPH_W {
        return false;
    }
    let bit = GLYPH_W - 1 - col;
    (glyph[row] >> bit) & 1 == 1
}

/// Returns the bitmap for `ch`, or the `?`-style fallback for anything outside
/// the printable-ASCII range. Lowercase/uppercase are both covered directly.
#[must_use]
pub fn glyph(ch: char) -> &'static [u8; GLYPH_H] {
    let code = ch as u32;
    if code < FIRST as u32 || code > LAST as u32 {
        return &FALLBACK;
    }
    &FONT[(code as u8 - FIRST) as usize]
}

/// The `.notdef` glyph: a filled box, drawn for unsupported codepoints.
static FALLBACK: [u8; GLYPH_H] = [
    0b11111, 0b11111, 0b11111, 0b11111, 0b11111, 0b11111, 0b11111,
];

/// The printable-ASCII glyph table, indexed by `codepoint - FIRST`.
#[rustfmt::skip]
pub static FONT: [[u8; GLYPH_H]; (LAST - FIRST + 1) as usize] = [
    // 0x20 ' '
    [0b00000,0b00000,0b00000,0b00000,0b00000,0b00000,0b00000],
    // 0x21 '!'
    [0b00100,0b00100,0b00100,0b00100,0b00100,0b00000,0b00100],
    // 0x22 '"'
    [0b01010,0b01010,0b01010,0b00000,0b00000,0b00000,0b00000],
    // 0x23 '#'
    [0b01010,0b01010,0b11111,0b01010,0b11111,0b01010,0b01010],
    // 0x24 '$'
    [0b00100,0b01111,0b10100,0b01110,0b00101,0b11110,0b00100],
    // 0x25 '%'
    [0b11000,0b11001,0b00010,0b00100,0b01000,0b10011,0b00011],
    // 0x26 '&'
    [0b01100,0b10010,0b10100,0b01000,0b10101,0b10010,0b01101],
    // 0x27 '\''
    [0b00100,0b00100,0b01000,0b00000,0b00000,0b00000,0b00000],
    // 0x28 '('
    [0b00010,0b00100,0b01000,0b01000,0b01000,0b00100,0b00010],
    // 0x29 ')'
    [0b01000,0b00100,0b00010,0b00010,0b00010,0b00100,0b01000],
    // 0x2A '*'
    [0b00000,0b00100,0b10101,0b01110,0b10101,0b00100,0b00000],
    // 0x2B '+'
    [0b00000,0b00100,0b00100,0b11111,0b00100,0b00100,0b00000],
    // 0x2C ','
    [0b00000,0b00000,0b00000,0b00000,0b00100,0b00100,0b01000],
    // 0x2D '-'
    [0b00000,0b00000,0b00000,0b11111,0b00000,0b00000,0b00000],
    // 0x2E '.'
    [0b00000,0b00000,0b00000,0b00000,0b00000,0b01100,0b01100],
    // 0x2F '/'
    [0b00001,0b00010,0b00100,0b00100,0b00100,0b01000,0b10000],
    // 0x30 '0'
    [0b01110,0b10001,0b10011,0b10101,0b11001,0b10001,0b01110],
    // 0x31 '1'
    [0b00100,0b01100,0b00100,0b00100,0b00100,0b00100,0b01110],
    // 0x32 '2'
    [0b01110,0b10001,0b00001,0b00010,0b00100,0b01000,0b11111],
    // 0x33 '3'
    [0b11111,0b00010,0b00100,0b00010,0b00001,0b10001,0b01110],
    // 0x34 '4'
    [0b00010,0b00110,0b01010,0b10010,0b11111,0b00010,0b00010],
    // 0x35 '5'
    [0b11111,0b10000,0b11110,0b00001,0b00001,0b10001,0b01110],
    // 0x36 '6'
    [0b00110,0b01000,0b10000,0b11110,0b10001,0b10001,0b01110],
    // 0x37 '7'
    [0b11111,0b00001,0b00010,0b00100,0b01000,0b01000,0b01000],
    // 0x38 '8'
    [0b01110,0b10001,0b10001,0b01110,0b10001,0b10001,0b01110],
    // 0x39 '9'
    [0b01110,0b10001,0b10001,0b01111,0b00001,0b00010,0b01100],
    // 0x3A ':'
    [0b00000,0b01100,0b01100,0b00000,0b01100,0b01100,0b00000],
    // 0x3B ';'
    [0b00000,0b01100,0b01100,0b00000,0b01100,0b00100,0b01000],
    // 0x3C '<'
    [0b00010,0b00100,0b01000,0b10000,0b01000,0b00100,0b00010],
    // 0x3D '='
    [0b00000,0b00000,0b11111,0b00000,0b11111,0b00000,0b00000],
    // 0x3E '>'
    [0b01000,0b00100,0b00010,0b00001,0b00010,0b00100,0b01000],
    // 0x3F '?'
    [0b01110,0b10001,0b00001,0b00010,0b00100,0b00000,0b00100],
    // 0x40 '@'
    [0b01110,0b10001,0b10111,0b10101,0b10111,0b10000,0b01110],
    // 0x41 'A'
    [0b01110,0b10001,0b10001,0b11111,0b10001,0b10001,0b10001],
    // 0x42 'B'
    [0b11110,0b10001,0b10001,0b11110,0b10001,0b10001,0b11110],
    // 0x43 'C'
    [0b01110,0b10001,0b10000,0b10000,0b10000,0b10001,0b01110],
    // 0x44 'D'
    [0b11100,0b10010,0b10001,0b10001,0b10001,0b10010,0b11100],
    // 0x45 'E'
    [0b11111,0b10000,0b10000,0b11110,0b10000,0b10000,0b11111],
    // 0x46 'F'
    [0b11111,0b10000,0b10000,0b11110,0b10000,0b10000,0b10000],
    // 0x47 'G'
    [0b01110,0b10001,0b10000,0b10111,0b10001,0b10001,0b01111],
    // 0x48 'H'
    [0b10001,0b10001,0b10001,0b11111,0b10001,0b10001,0b10001],
    // 0x49 'I'
    [0b01110,0b00100,0b00100,0b00100,0b00100,0b00100,0b01110],
    // 0x4A 'J'
    [0b00111,0b00010,0b00010,0b00010,0b00010,0b10010,0b01100],
    // 0x4B 'K'
    [0b10001,0b10010,0b10100,0b11000,0b10100,0b10010,0b10001],
    // 0x4C 'L'
    [0b10000,0b10000,0b10000,0b10000,0b10000,0b10000,0b11111],
    // 0x4D 'M'
    [0b10001,0b11011,0b10101,0b10101,0b10001,0b10001,0b10001],
    // 0x4E 'N'
    [0b10001,0b10001,0b11001,0b10101,0b10011,0b10001,0b10001],
    // 0x4F 'O'
    [0b01110,0b10001,0b10001,0b10001,0b10001,0b10001,0b01110],
    // 0x50 'P'
    [0b11110,0b10001,0b10001,0b11110,0b10000,0b10000,0b10000],
    // 0x51 'Q'
    [0b01110,0b10001,0b10001,0b10001,0b10101,0b10010,0b01101],
    // 0x52 'R'
    [0b11110,0b10001,0b10001,0b11110,0b10100,0b10010,0b10001],
    // 0x53 'S'
    [0b01111,0b10000,0b10000,0b01110,0b00001,0b00001,0b11110],
    // 0x54 'T'
    [0b11111,0b00100,0b00100,0b00100,0b00100,0b00100,0b00100],
    // 0x55 'U'
    [0b10001,0b10001,0b10001,0b10001,0b10001,0b10001,0b01110],
    // 0x56 'V'
    [0b10001,0b10001,0b10001,0b10001,0b10001,0b01010,0b00100],
    // 0x57 'W'
    [0b10001,0b10001,0b10001,0b10101,0b10101,0b10101,0b01010],
    // 0x58 'X'
    [0b10001,0b10001,0b01010,0b00100,0b01010,0b10001,0b10001],
    // 0x59 'Y'
    [0b10001,0b10001,0b01010,0b00100,0b00100,0b00100,0b00100],
    // 0x5A 'Z'
    [0b11111,0b00001,0b00010,0b00100,0b01000,0b10000,0b11111],
    // 0x5B '['
    [0b01110,0b01000,0b01000,0b01000,0b01000,0b01000,0b01110],
    // 0x5C '\\'
    [0b10000,0b01000,0b00100,0b00100,0b00100,0b00010,0b00001],
    // 0x5D ']'
    [0b01110,0b00010,0b00010,0b00010,0b00010,0b00010,0b01110],
    // 0x5E '^'
    [0b00100,0b01010,0b10001,0b00000,0b00000,0b00000,0b00000],
    // 0x5F '_'
    [0b00000,0b00000,0b00000,0b00000,0b00000,0b00000,0b11111],
    // 0x60 '`'
    [0b01000,0b00100,0b00010,0b00000,0b00000,0b00000,0b00000],
    // 0x61 'a'
    [0b00000,0b00000,0b01110,0b00001,0b01111,0b10001,0b01111],
    // 0x62 'b'
    [0b10000,0b10000,0b10110,0b11001,0b10001,0b10001,0b11110],
    // 0x63 'c'
    [0b00000,0b00000,0b01110,0b10001,0b10000,0b10001,0b01110],
    // 0x64 'd'
    [0b00001,0b00001,0b01101,0b10011,0b10001,0b10001,0b01111],
    // 0x65 'e'
    [0b00000,0b00000,0b01110,0b10001,0b11111,0b10000,0b01110],
    // 0x66 'f'
    [0b00110,0b01001,0b01000,0b11100,0b01000,0b01000,0b01000],
    // 0x67 'g'
    [0b00000,0b01111,0b10001,0b10001,0b01111,0b00001,0b01110],
    // 0x68 'h'
    [0b10000,0b10000,0b10110,0b11001,0b10001,0b10001,0b10001],
    // 0x69 'i'
    [0b00100,0b00000,0b01100,0b00100,0b00100,0b00100,0b01110],
    // 0x6A 'j'
    [0b00010,0b00000,0b00110,0b00010,0b00010,0b10010,0b01100],
    // 0x6B 'k'
    [0b10000,0b10000,0b10010,0b10100,0b11000,0b10100,0b10010],
    // 0x6C 'l'
    [0b01100,0b00100,0b00100,0b00100,0b00100,0b00100,0b01110],
    // 0x6D 'm'
    [0b00000,0b00000,0b11010,0b10101,0b10101,0b10001,0b10001],
    // 0x6E 'n'
    [0b00000,0b00000,0b10110,0b11001,0b10001,0b10001,0b10001],
    // 0x6F 'o'
    [0b00000,0b00000,0b01110,0b10001,0b10001,0b10001,0b01110],
    // 0x70 'p'
    [0b00000,0b11110,0b10001,0b10001,0b11110,0b10000,0b10000],
    // 0x71 'q'
    [0b00000,0b01111,0b10001,0b10001,0b01111,0b00001,0b00001],
    // 0x72 'r'
    [0b00000,0b00000,0b10110,0b11001,0b10000,0b10000,0b10000],
    // 0x73 's'
    [0b00000,0b00000,0b01111,0b10000,0b01110,0b00001,0b11110],
    // 0x74 't'
    [0b01000,0b01000,0b11100,0b01000,0b01000,0b01001,0b00110],
    // 0x75 'u'
    [0b00000,0b00000,0b10001,0b10001,0b10001,0b10011,0b01101],
    // 0x76 'v'
    [0b00000,0b00000,0b10001,0b10001,0b10001,0b01010,0b00100],
    // 0x77 'w'
    [0b00000,0b00000,0b10001,0b10001,0b10101,0b10101,0b01010],
    // 0x78 'x'
    [0b00000,0b00000,0b10001,0b01010,0b00100,0b01010,0b10001],
    // 0x79 'y'
    [0b00000,0b10001,0b10001,0b10001,0b01111,0b00001,0b01110],
    // 0x7A 'z'
    [0b00000,0b00000,0b11111,0b00010,0b00100,0b01000,0b11111],
    // 0x7B '{'
    [0b00010,0b00100,0b00100,0b01000,0b00100,0b00100,0b00010],
    // 0x7C '|'
    [0b00100,0b00100,0b00100,0b00100,0b00100,0b00100,0b00100],
    // 0x7D '}'
    [0b01000,0b00100,0b00100,0b00010,0b00100,0b00100,0b01000],
    // 0x7E '~'
    [0b00000,0b00000,0b01000,0b10101,0b00010,0b00000,0b00000],
];

/// Advance width, in source pixels, of a string rendered monospace.
#[must_use]
pub fn measure(text: &str) -> usize {
    let n = text.chars().count();
    if n == 0 {
        0
    } else {
        n * GLYPH_ADVANCE - GLYPH_GAP
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_is_complete() {
        assert_eq!(FONT.len(), (LAST - FIRST + 1) as usize);
    }

    #[test]
    fn known_letters_resolve_in_range() {
        assert!(core::ptr::eq(glyph('A'), &FONT[('A' as u8 - FIRST) as usize]));
        assert!(core::ptr::eq(glyph(' '), &FONT[0]));
        assert!(core::ptr::eq(glyph('~'), &FONT[FONT.len() - 1]));
    }

    #[test]
    fn out_of_range_uses_fallback() {
        assert!(core::ptr::eq(glyph('\u{1F600}'), &FALLBACK));
        assert!(core::ptr::eq(glyph('\u{1}'), &FALLBACK));
    }

    #[test]
    fn pixel_reads_leftmost_bit_first() {
        // 'T' top row is all five columns inked.
        let t = glyph('T');
        for col in 0..GLYPH_W {
            assert!(pixel(t, 0, col), "T row0 col{col}");
        }
        // and only the centre column on row 1.
        assert!(pixel(t, 1, 2));
        assert!(!pixel(t, 1, 0));
    }

    #[test]
    fn measure_is_monospace() {
        assert_eq!(measure(""), 0);
        assert_eq!(measure("A"), GLYPH_W);
        assert_eq!(measure("AB"), GLYPH_W * 2 + GLYPH_GAP);
    }
}
