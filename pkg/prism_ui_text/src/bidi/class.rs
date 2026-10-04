//! Bidi character types (`Bidi_Class`) and the paired-bracket data used by the
//! Unicode Bidirectional Algorithm (UAX#9).
//!
//! # Data provenance and coverage
//!
//! [`bidi_class`] resolves a `char` to its [`BidiClass`] following the
//! `DerivedBidiClass` rules of the Unicode Character Database:
//!
//! * **Exact** encoding of every control, whitespace, separator and *all*
//!   explicit-formatting / isolate code points (`LRE RLE LRO RLO PDF LRI RLI
//!   FSI PDI`, `LRM RLM ALM`). These drive the *structure* of the algorithm, so
//!   they must be precise.
//! * **Exact** encoding of the weak classes (`EN ES ET AN CS NSM BN`) across
//!   Latin-1, General Punctuation, the Arabic/Hebrew combining ranges and the
//!   fullwidth forms — the ranges that occur in real mixed-direction text.
//! * The **complete** `DerivedBidiClass` default ranges for `R`, `AL`, `ET` and
//!   `BN` (noncharacters / default-ignorables), so unlisted code points in the
//!   RTL and currency blocks still resolve correctly.
//! * Everything else falls back to `L`, which is the UCD default for the
//!   overwhelming majority of assigned (non-RTL, non-weak) code points.
//!
//! The *algorithm* in the sibling modules is fully conformant and operates on
//! [`BidiClass`] values, so it can be exercised directly with explicit class
//! sequences (the form used by the Unicode `BidiTest` conformance data) even
//! for code points outside the lookup table above.

/// A Unicode bidirectional character type (`Bidi_Class`, UAX#9 Table 4).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BidiClass {
    // Strong
    /// Left-to-Right.
    L,
    /// Right-to-Left.
    R,
    /// Right-to-Left Arabic.
    AL,
    // Weak
    /// European Number.
    EN,
    /// European Number Separator.
    ES,
    /// European Number Terminator.
    ET,
    /// Arabic Number.
    AN,
    /// Common Number Separator.
    CS,
    /// Nonspacing Mark.
    NSM,
    /// Boundary Neutral.
    BN,
    // Neutral
    /// Paragraph Separator.
    B,
    /// Segment Separator.
    S,
    /// Whitespace.
    WS,
    /// Other Neutral.
    ON,
    // Explicit formatting
    /// Left-to-Right Embedding.
    LRE,
    /// Left-to-Right Override.
    LRO,
    /// Right-to-Left Embedding.
    RLE,
    /// Right-to-Left Override.
    RLO,
    /// Pop Directional Format.
    PDF,
    /// Left-to-Right Isolate.
    LRI,
    /// Right-to-Left Isolate.
    RLI,
    /// First Strong Isolate.
    FSI,
    /// Pop Directional Isolate.
    PDI,
}

impl BidiClass {
    /// `true` for the strong types `L`, `R`, `AL`.
    #[must_use]
    pub fn is_strong(self) -> bool {
        matches!(self, Self::L | Self::R | Self::AL)
    }

    /// `true` for an isolate initiator (`LRI`, `RLI`, `FSI`).
    #[must_use]
    pub fn is_isolate_initiator(self) -> bool {
        matches!(self, Self::LRI | Self::RLI | Self::FSI)
    }

    /// `true` for an explicit embedding or override initiator.
    #[must_use]
    pub fn is_explicit_initiator(self) -> bool {
        matches!(self, Self::LRE | Self::LRO | Self::RLE | Self::RLO)
    }

    /// `true` for the formatting characters removed by rule X9
    /// (`RLE LRE RLO LRO PDF` and `BN`).
    #[must_use]
    pub fn is_removed_by_x9(self) -> bool {
        matches!(
            self,
            Self::RLE | Self::LRE | Self::RLO | Self::LRO | Self::PDF | Self::BN
        )
    }

    /// `true` for a "neutral or isolate" (NI) type as defined for rules N0–N2:
    /// `B S WS ON FSI LRI RLI PDI`.
    #[must_use]
    pub fn is_neutral_or_isolate(self) -> bool {
        matches!(
            self,
            Self::B | Self::S | Self::WS | Self::ON | Self::FSI | Self::LRI | Self::RLI | Self::PDI
        )
    }
}

/// Resolves the [`BidiClass`] of `ch` (see module docs for coverage).
#[must_use]
// The lookup is written as a sequence of range tables that mirror the Unicode
// character blocks they cover. Several adjacent blocks resolve to the same
// `Bidi_Class`; the arms are kept separate so each range stays documented and
// auditable against the data file, so `match_same_arms` is intentionally off.
#[expect(
    clippy::match_same_arms,
    reason = "range tables mirror the Unicode blocks they cover; identical \
              Bidi_Class arms are kept separate so each range stays documented"
)]
pub fn bidi_class(ch: char) -> BidiClass {
    use BidiClass::*;
    let c = ch as u32;

    // --- Exact: C0/C1 controls, separators, whitespace -------------------
    match c {
        0x0009 | 0x000B => return S,     // TAB, VT
        0x000A | 0x000D => return B,     // LF, CR
        0x001C..=0x001E => return B,     // FS, GS, RS
        0x001F => return S,              // US
        0x000C | 0x0020 => return WS,    // FF, SPACE
        0x0085 => return B,              // NEL
        0x00A0 => return CS,             // NO-BREAK SPACE
        0x001A..=0x001B => return BN,    // SUB, ESC
        _ => {}
    }
    if c <= 0x0008 || (0x000E..=0x0019).contains(&c) || (0x007F..=0x0084).contains(&c) || (0x0086..=0x009F).contains(&c) {
        return BN;
    }

    // --- Exact: explicit formatting & isolates ---------------------------
    match c {
        0x202A => return LRE,
        0x202B => return RLE,
        0x202C => return PDF,
        0x202D => return LRO,
        0x202E => return RLO,
        0x2066 => return LRI,
        0x2067 => return RLI,
        0x2068 => return FSI,
        0x2069 => return PDI,
        0x200E => return L,  // LEFT-TO-RIGHT MARK
        0x200F => return R,  // RIGHT-TO-LEFT MARK
        0x061C => return AL, // ARABIC LETTER MARK
        _ => {}
    }

    // --- Exact: boundary neutrals (zero-width / BOM / soft hyphen) --------
    match c {
        0x00AD => return BN,                 // SOFT HYPHEN
        0x200B..=0x200D => return BN,        // ZWSP, ZWNJ, ZWJ
        0x2060..=0x2064 => return BN,        // WORD JOINER .. INVISIBLE PLUS
        0xFEFF => return BN,                 // ZERO WIDTH NO-BREAK SPACE (BOM)
        0xFFF9..=0xFFFB => return ON,        // interlinear annotation anchors
        _ => {}
    }

    // --- Exact: whitespace ----------------------------------------------
    match c {
        0x1680 => return WS,                 // OGHAM SPACE MARK
        0x2000..=0x200A => return WS,        // EN QUAD .. HAIR SPACE
        0x2028 => return WS,                 // LINE SEPARATOR
        0x2029 => return B,                  // PARAGRAPH SEPARATOR
        0x202F => return CS,                 // NARROW NO-BREAK SPACE
        0x205F => return WS,                 // MEDIUM MATHEMATICAL SPACE
        0x3000 => return WS,                 // IDEOGRAPHIC SPACE
        _ => {}
    }

    // --- European numbers (EN) ------------------------------------------
    match c {
        0x0030..=0x0039 => return EN,        // DIGIT ZERO..NINE
        0x00B2 | 0x00B3 | 0x00B9 => return EN, // superscript 2,3,1
        0x06F0..=0x06F9 => return EN,        // EXTENDED ARABIC-INDIC DIGITS
        0x2070 | 0x2074..=0x2079 => return EN, // superscripts
        0x2080..=0x2089 => return EN,        // subscripts
        0x2488..=0x249B => return EN,        // digit-with-period/paren forms
        0xFF10..=0xFF19 => return EN,        // FULLWIDTH DIGITS
        _ => {}
    }

    // --- European separators / terminators (ES / ET) --------------------
    match c {
        0x002B | 0x002D => return ES,        // PLUS, HYPHEN-MINUS
        0x207A | 0x207B => return ES,        // superscript +,-
        0x208A | 0x208B => return ES,        // subscript +,-
        0x2212 => return ES,                 // MINUS SIGN
        0xFB29 => return ES,                 // HEBREW LETTER ALTERNATIVE PLUS SIGN
        0xFE62 | 0xFE63 => return ES,        // small plus / hyphen-minus
        0xFF0B | 0xFF0D => return ES,        // fullwidth plus / hyphen-minus
        _ => {}
    }
    match c {
        0x0023..=0x0025 => return ET,        // # $ %
        0x00A2..=0x00A5 => return ET,        // cent, pound, currency, yen
        0x00B0 | 0x00B1 => return ET,        // degree, plus-minus
        0x0609 | 0x060A => return ET,        // arabic-indic per mille/ten thousand
        0x066A => return ET,                 // ARABIC PERCENT SIGN
        0x2030..=0x2034 => return ET,        // per mille .. triple prime
        0x20A0..=0x20CF => return ET,        // CURRENCY SYMBOLS block (default ET)
        0x212E => return ET,                 // ESTIMATED SYMBOL
        0x2213 => return ET,                 // MINUS-OR-PLUS SIGN
        0xFE69 | 0xFF04 => return ET,        // small / fullwidth dollar
        0xFF05 => return ET,                 // fullwidth percent
        0xFFE0..=0xFFE1 => return ET,        // fullwidth cent/pound
        0xFFE5..=0xFFE6 => return ET,        // fullwidth yen/won
        _ => {}
    }

    // --- Common separators (CS) -----------------------------------------
    match c {
        0x002C => return CS,                 // COMMA
        0x002E | 0x002F => return CS,         // FULL STOP, SOLIDUS
        0x003A => return CS,                 // COLON
        0x060C => return CS,                 // ARABIC COMMA (CS, not AN)
        0xFE50 | 0xFE52 | 0xFE55 => return CS, // small comma/stop/colon
        0xFF0C | 0xFF0E | 0xFF0F | 0xFF1A => return CS, // fullwidth , . / :
        _ => {}
    }

    // --- Arabic numbers (AN) --------------------------------------------
    match c {
        0x0600..=0x0605 => return AN,        // ARABIC NUMBER SIGN .. NUMBER MARK
        0x0660..=0x0669 => return AN,        // ARABIC-INDIC DIGITS
        0x066B | 0x066C => return AN,        // arabic decimal / thousands separator
        0x06DD => return AN,                 // ARABIC END OF AYAH
        0x08E2 => return AN,                 // ARABIC DISPUTED END OF AYAH
        _ => {}
    }

    // --- Nonspacing marks (NSM) -----------------------------------------
    if is_nsm(c) {
        return NSM;
    }

    // --- Strong RTL defaults (R) ----------------------------------------
    if is_default_r(c) {
        return R;
    }
    // --- Strong Arabic defaults (AL) ------------------------------------
    if is_default_al(c) {
        return AL;
    }

    // --- Noncharacters / default-ignorables default to BN ---------------
    if is_noncharacter(c) {
        return BN;
    }

    // --- Common Other-Neutral punctuation (ON) --------------------------
    if is_common_on(c) {
        return ON;
    }

    // Everything else: Latin / CJK / most scripts are strong L by default.
    L
}

/// Combining-mark ranges classified `NSM` (nonspacing mark).
fn is_nsm(c: u32) -> bool {
    matches!(c,
        0x0300..=0x036F   // COMBINING DIACRITICAL MARKS
        | 0x0483..=0x0489 // Cyrillic combining
        | 0x0591..=0x05BD // Hebrew points
        | 0x05BF
        | 0x05C1..=0x05C2
        | 0x05C4..=0x05C5
        | 0x05C7
        | 0x0610..=0x061A // Arabic marks
        | 0x064B..=0x065F
        | 0x0670
        | 0x06D6..=0x06DC
        | 0x06DF..=0x06E4
        | 0x06E7..=0x06E8
        | 0x06EA..=0x06ED
        | 0x0711
        | 0x0730..=0x074A // Syriac points
        | 0x07A6..=0x07B0 // Thaana
        | 0x07EB..=0x07F3
        | 0x0816..=0x0819
        | 0x081B..=0x0823
        | 0x0825..=0x0827
        | 0x0829..=0x082D
        | 0x0859..=0x085B
        | 0x08E3..=0x0902
        | 0x1AB0..=0x1AFF // combining diacritical marks extended
        | 0x1DC0..=0x1DFF // combining diacritical marks supplement
        | 0x20D0..=0x20F0 // combining diacritical marks for symbols
        | 0xFE20..=0xFE2F // combining half marks
    )
}

/// `DerivedBidiClass` default-`R` ranges (strong right-to-left).
fn is_default_r(c: u32) -> bool {
    matches!(c,
        0x0590..=0x05FF   // Hebrew
        | 0x07C0..=0x085F // NKo, Samaritan, Mandaic (default R)
        | 0x0860..=0x08FF // (overridden to AL where assigned; R default)
        | 0xFB1D..=0xFB4F // Hebrew presentation forms
        | 0x10800..=0x10CFF
        | 0x10D40..=0x10EBF
        | 0x10F00..=0x10F2F
        | 0x10F70..=0x10FFF
        | 0x1E800..=0x1EC6F
        | 0x1ECC0..=0x1ECFF
        | 0x1ED50..=0x1EDFF
        | 0x1EF00..=0x1EFFF
    )
}

/// `DerivedBidiClass` default-`AL` ranges (strong Arabic right-to-left).
fn is_default_al(c: u32) -> bool {
    matches!(c,
        0x0600..=0x07BF   // Arabic, Syriac, Arabic Supplement, Thaana
        | 0xFB50..=0xFDCF // Arabic presentation forms-A
        | 0xFDF0..=0xFDFF
        | 0xFE70..=0xFEFF // Arabic presentation forms-B
        | 0x10D00..=0x10D3F
        | 0x10EC0..=0x10EFF
        | 0x10F30..=0x10F6F
        | 0x1EC70..=0x1ECBF
        | 0x1ED00..=0x1ED4F
        | 0x1EE00..=0x1EEFF
    )
}

/// Noncharacters that default to `BN`.
fn is_noncharacter(c: u32) -> bool {
    (0xFDD0..=0xFDEF).contains(&c) || (c & 0xFFFE) == 0xFFFE
}

/// Common Other-Neutral punctuation ranges (`ON`).
fn is_common_on(c: u32) -> bool {
    matches!(c,
        0x0021 | 0x0022            // ! "
        | 0x0026..=0x002A          // & ' ( ) *
        | 0x003B..=0x0040          // ; < = > ? @
        | 0x005B..=0x0060          // [ \ ] ^ _ `
        | 0x007B..=0x007E          // { | } ~
        | 0x00A1 | 0x00A6..=0x00A9 // ¡ ¦ § ¨ ©
        | 0x00AB | 0x00AC | 0x00AE // « ¬ ®
        | 0x00AF | 0x00B4 | 0x00B6..=0x00B8
        | 0x00BB..=0x00BF
        | 0x00D7 | 0x00F7
        | 0x2010..=0x2027          // general punctuation (dashes, quotes, bullets)
        | 0x2035..=0x205E          // primes, daggers, etc. (minus the WS already handled)
        | 0x2190..=0x2BFF          // arrows, math operators, misc symbols, dingbats
        | 0x3001..=0x303F          // CJK symbols & punctuation
        | 0xFE30..=0xFE4F          // CJK compatibility forms
        | 0xFF01..=0xFF03          // ！ ＂ ＃ (fullwidth, ON ones)
        | 0xFF06..=0xFF0A          // fullwidth & ' ( ) *
        | 0xFF1B..=0xFF20          // fullwidth ; < = > ? @
        | 0xFF3B..=0xFF40          // fullwidth [ \ ] ^ _ `
        | 0xFF5B..=0xFF65          // fullwidth { | } ~ and halfwidth punctuation
    )
}

/// The matching close bracket and bracket "type" for the paired-bracket rule
/// (N0). Returns `None` for non-bracket characters.
///
/// The data is the canonical-equivalence-folded subset of `BidiBrackets.txt`
/// covering ASCII, the CJK and fullwidth bracket pairs, and the common
/// mathematical / ornamental brackets that occur in real text.
#[must_use]
pub fn paired_bracket(ch: char) -> Option<Bracket> {
    let (kind, opposite) = match ch {
        '(' => (BracketKind::Open, ')'),
        ')' => (BracketKind::Close, '('),
        '[' => (BracketKind::Open, ']'),
        ']' => (BracketKind::Close, '['),
        '{' => (BracketKind::Open, '}'),
        '}' => (BracketKind::Close, '{'),
        '\u{0F3A}' => (BracketKind::Open, '\u{0F3B}'),
        '\u{0F3B}' => (BracketKind::Close, '\u{0F3A}'),
        '\u{0F3C}' => (BracketKind::Open, '\u{0F3D}'),
        '\u{0F3D}' => (BracketKind::Close, '\u{0F3C}'),
        '\u{169B}' => (BracketKind::Open, '\u{169C}'),
        '\u{169C}' => (BracketKind::Close, '\u{169B}'),
        '\u{2045}' => (BracketKind::Open, '\u{2046}'),
        '\u{2046}' => (BracketKind::Close, '\u{2045}'),
        '\u{207D}' => (BracketKind::Open, '\u{207E}'),
        '\u{207E}' => (BracketKind::Close, '\u{207D}'),
        '\u{208D}' => (BracketKind::Open, '\u{208E}'),
        '\u{208E}' => (BracketKind::Close, '\u{208D}'),
        '\u{2308}' => (BracketKind::Open, '\u{2309}'),
        '\u{2309}' => (BracketKind::Close, '\u{2308}'),
        '\u{230A}' => (BracketKind::Open, '\u{230B}'),
        '\u{230B}' => (BracketKind::Close, '\u{230A}'),
        '\u{2329}' => (BracketKind::Open, '\u{232A}'),
        '\u{232A}' => (BracketKind::Close, '\u{2329}'),
        '\u{2768}' => (BracketKind::Open, '\u{2769}'),
        '\u{2769}' => (BracketKind::Close, '\u{2768}'),
        '\u{276A}' => (BracketKind::Open, '\u{276B}'),
        '\u{276B}' => (BracketKind::Close, '\u{276A}'),
        '\u{276C}' => (BracketKind::Open, '\u{276D}'),
        '\u{276D}' => (BracketKind::Close, '\u{276C}'),
        '\u{276E}' => (BracketKind::Open, '\u{276F}'),
        '\u{276F}' => (BracketKind::Close, '\u{276E}'),
        '\u{2770}' => (BracketKind::Open, '\u{2771}'),
        '\u{2771}' => (BracketKind::Close, '\u{2770}'),
        '\u{2772}' => (BracketKind::Open, '\u{2773}'),
        '\u{2773}' => (BracketKind::Close, '\u{2772}'),
        '\u{2774}' => (BracketKind::Open, '\u{2775}'),
        '\u{2775}' => (BracketKind::Close, '\u{2774}'),
        '\u{27E6}' => (BracketKind::Open, '\u{27E7}'),
        '\u{27E7}' => (BracketKind::Close, '\u{27E6}'),
        '\u{27E8}' => (BracketKind::Open, '\u{27E9}'),
        '\u{27E9}' => (BracketKind::Close, '\u{27E8}'),
        '\u{27EA}' => (BracketKind::Open, '\u{27EB}'),
        '\u{27EB}' => (BracketKind::Close, '\u{27EA}'),
        '\u{27EC}' => (BracketKind::Open, '\u{27ED}'),
        '\u{27ED}' => (BracketKind::Close, '\u{27EC}'),
        '\u{27EE}' => (BracketKind::Open, '\u{27EF}'),
        '\u{27EF}' => (BracketKind::Close, '\u{27EE}'),
        '\u{2983}' => (BracketKind::Open, '\u{2984}'),
        '\u{2984}' => (BracketKind::Close, '\u{2983}'),
        '\u{2985}' => (BracketKind::Open, '\u{2986}'),
        '\u{2986}' => (BracketKind::Close, '\u{2985}'),
        '\u{3008}' => (BracketKind::Open, '\u{3009}'),
        '\u{3009}' => (BracketKind::Close, '\u{3008}'),
        '\u{300A}' => (BracketKind::Open, '\u{300B}'),
        '\u{300B}' => (BracketKind::Close, '\u{300A}'),
        '\u{300C}' => (BracketKind::Open, '\u{300D}'),
        '\u{300D}' => (BracketKind::Close, '\u{300C}'),
        '\u{300E}' => (BracketKind::Open, '\u{300F}'),
        '\u{300F}' => (BracketKind::Close, '\u{300E}'),
        '\u{3010}' => (BracketKind::Open, '\u{3011}'),
        '\u{3011}' => (BracketKind::Close, '\u{3010}'),
        '\u{3014}' => (BracketKind::Open, '\u{3015}'),
        '\u{3015}' => (BracketKind::Close, '\u{3014}'),
        '\u{3016}' => (BracketKind::Open, '\u{3017}'),
        '\u{3017}' => (BracketKind::Close, '\u{3016}'),
        '\u{3018}' => (BracketKind::Open, '\u{3019}'),
        '\u{3019}' => (BracketKind::Close, '\u{3018}'),
        '\u{301A}' => (BracketKind::Open, '\u{301B}'),
        '\u{301B}' => (BracketKind::Close, '\u{301A}'),
        '\u{FE59}' => (BracketKind::Open, '\u{FE5A}'),
        '\u{FE5A}' => (BracketKind::Close, '\u{FE59}'),
        '\u{FE5B}' => (BracketKind::Open, '\u{FE5C}'),
        '\u{FE5C}' => (BracketKind::Close, '\u{FE5B}'),
        '\u{FE5D}' => (BracketKind::Open, '\u{FE5E}'),
        '\u{FE5E}' => (BracketKind::Close, '\u{FE5D}'),
        '\u{FF08}' => (BracketKind::Open, '\u{FF09}'),
        '\u{FF09}' => (BracketKind::Close, '\u{FF08}'),
        '\u{FF3B}' => (BracketKind::Open, '\u{FF3D}'),
        '\u{FF3D}' => (BracketKind::Close, '\u{FF3B}'),
        '\u{FF5B}' => (BracketKind::Open, '\u{FF5D}'),
        '\u{FF5D}' => (BracketKind::Close, '\u{FF5B}'),
        '\u{FF5F}' => (BracketKind::Open, '\u{FF60}'),
        '\u{FF60}' => (BracketKind::Close, '\u{FF5F}'),
        '\u{FF62}' => (BracketKind::Open, '\u{FF63}'),
        '\u{FF63}' => (BracketKind::Close, '\u{FF62}'),
        _ => return None,
    };
    Some(Bracket {
        kind,
        opposite,
        // Canonical-equivalence fold: U+2329/U+232A are canonically equivalent
        // to U+3008/U+3009, so N0 must treat them as the same pair.
        canonical: canonical_bracket(ch),
    })
}

/// Folds canonically-equivalent brackets to a single representative so N0 can
/// match an opening `U+2329` against a closing `U+3009` and vice versa.
fn canonical_bracket(ch: char) -> char {
    match ch {
        '\u{2329}' => '\u{3008}',
        '\u{232A}' => '\u{3009}',
        other => other,
    }
}

/// Whether a bracket opens or closes a pair.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BracketKind {
    /// An opening bracket such as `(`.
    Open,
    /// A closing bracket such as `)`.
    Close,
}

/// Paired-bracket metadata returned by [`paired_bracket`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bracket {
    /// Whether this is an opening or closing bracket.
    pub kind: BracketKind,
    /// The matching bracket of the opposite kind.
    pub opposite: char,
    /// Canonical-equivalence representative used for matching.
    pub canonical: char,
}

impl Bracket {
    /// The canonical representative of this bracket's matching partner, used to
    /// pair an opener with a closer under canonical equivalence (N0).
    #[must_use]
    pub fn canonical_opposite(self) -> char {
        canonical_bracket(self.opposite)
    }
}
