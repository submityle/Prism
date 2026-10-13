//! Baseline-relative field deltas (design §7.2 / §15 "delta 压缩").
//!
//! A replicated record is a fixed, ordered set of typed fields — a [`Schema`].
//! Sending a whole record every tick wastes bandwidth on fields that did not
//! change, so a *delta* names a **baseline revision** and carries only the fields
//! that differ, bit-packed to their declared width ([`crate::bitpack`]) and
//! quantized for floats ([`crate::quantize`]).
//!
//! Wire form (byte-aligned header, then a bit stream):
//!
//! ```text
//! [revision: u32 LE][field_count: u8][bit stream: changed mask, then each changed value]
//! ```
//!
//! The changed mask is one bit per field (in schema order); the values follow for
//! the set bits only, in the same order. A receiver holds the baseline revision
//! and applies the delta to its copy — a revision mismatch is refused rather than
//! applied to the wrong base, and a short payload is a [`DeltaError::Truncated`]
//! rather than a silent zero-fill.

use alloc::vec::Vec;

use crate::bitpack::{BitReader, BitWriter};
use crate::quantize::Quantizer;

/// The wire type of one replicated field, fixed by the schema.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FieldKind {
    /// A boolean, packed as one bit.
    Bool,
    /// A signed whole number, zigzagged into `bits` bits (1..=64).
    ///
    /// A value outside the representable signed range
    /// `[-(2^(bits-1)), 2^(bits-1) - 1]` (all of `i64` at 64 bits) is reported as
    /// [`DeltaError::ValueOutOfRange`] by encode rather than silently truncated.
    Int {
        /// Code width in bits (1..=64); an out-of-range width is reported as
        /// [`DeltaError::InvalidKind`] by encode/decode rather than panicking.
        bits: u8,
    },
    /// A float quantized over `[min, max]` with `bits` of precision.
    ///
    /// Valid when `min`/`max` are finite with `min < max`, the span `max - min`
    /// yields a finite positive per-unit step (so the endpoints cannot collapse to
    /// `NaN`/`inf`), and `bits` is `1..=24` (the same contract as
    /// [`Quantizer::new`]); an invalid kind is reported as
    /// [`DeltaError::InvalidKind`] by encode/decode rather than panicking.
    Float {
        /// Range lower bound.
        min: f32,
        /// Range upper bound.
        max: f32,
        /// Code width in bits.
        bits: u8,
    },
}

impl FieldKind {
    /// Whether `value` has the variant this kind declares.
    #[must_use]
    pub fn matches(self, value: FieldValue) -> bool {
        matches!(
            (self, value),
            (FieldKind::Bool, FieldValue::Bool(_))
                | (FieldKind::Int { .. }, FieldValue::Int(_))
                | (FieldKind::Float { .. }, FieldValue::Float(_))
        )
    }
}

/// A fixed, ordered schema for a replicated record.
#[derive(Clone, Debug, PartialEq)]
pub struct Schema {
    kinds: Vec<FieldKind>,
}

impl Schema {
    /// A schema over `kinds`, in field order.
    #[must_use]
    pub fn new(kinds: Vec<FieldKind>) -> Self {
        Self { kinds }
    }

    /// The number of fields.
    #[must_use]
    pub fn len(&self) -> usize {
        self.kinds.len()
    }

    /// Whether the schema has no fields.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.kinds.is_empty()
    }

    /// The field kinds, in order.
    #[must_use]
    pub fn kinds(&self) -> &[FieldKind] {
        &self.kinds
    }

    /// Validate that `record` matches this schema: the same field count, and a
    /// value variant matching each field's kind.
    ///
    /// # Errors
    /// [`DeltaError::Shape`] / [`DeltaError::KindMismatch`].
    pub fn validate(&self, record: &Record) -> Result<(), DeltaError> {
        check_shape(self, record)
    }
}

/// One field's typed value; the variant must match the schema kind.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FieldValue {
    /// A boolean.
    Bool(bool),
    /// A whole number.
    Int(i64),
    /// A float (quantized on the wire when its kind is `Float`).
    Float(f32),
}

/// A record's values, in schema order.
#[derive(Clone, Debug, PartialEq)]
pub struct Record {
    values: Vec<FieldValue>,
}

impl Record {
    /// A record with `values` in schema order.
    #[must_use]
    pub fn new(values: Vec<FieldValue>) -> Self {
        Self { values }
    }

    /// The values, in schema order.
    #[must_use]
    pub fn values(&self) -> &[FieldValue] {
        &self.values
    }

    /// The value at `index`, if present.
    #[must_use]
    pub fn value(&self, index: usize) -> Option<FieldValue> {
        self.values.get(index).copied()
    }

    /// Overwrite the value at `index` (a no-op when out of range).
    pub fn set(&mut self, index: usize, value: FieldValue) {
        if let Some(slot) = self.values.get_mut(index) {
            *slot = value;
        }
    }
}

/// A delta encoding or application failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeltaError {
    /// A record's field count does not match the schema.
    Shape {
        /// Fields the schema declares.
        expected: usize,
        /// Fields the record holds.
        found: usize,
    },
    /// A record field's variant does not match its schema kind.
    KindMismatch {
        /// The offending field index.
        index: usize,
    },
    /// An integer field value is outside the signed range its `bits`-wide code
    /// can represent, so encoding it would silently truncate.
    ValueOutOfRange {
        /// The offending field index.
        index: usize,
    },
    /// A schema field kind has out-of-range parameters (an `Int`/`Float` width
    /// outside `1..=64` / `1..=24`, or a non-increasing/ non-finite `Float`
    /// range) — reported instead of panicking inside the bit packer / quantizer.
    InvalidKind {
        /// The offending field index.
        index: usize,
    },
    /// The encoded field count does not match the schema being decoded with.
    SchemaMismatch {
        /// Fields the schema declares.
        expected: usize,
        /// Fields the payload declares.
        found: usize,
    },
    /// The schema has more fields than the wire format can address (> 255).
    TooManyFields {
        /// Fields the schema declares.
        len: usize,
    },
    /// The payload ended before every field was read.
    Truncated,
    /// The delta was encoded against a different baseline revision.
    RevisionMismatch {
        /// The revision the receiver holds.
        expected: u32,
        /// The revision the delta declares.
        found: u32,
    },
}

/// A decoded delta: the baseline revision and the changed fields, in schema order.
#[derive(Clone, Debug, PartialEq)]
pub struct Delta {
    revision: u32,
    changed: Vec<(usize, FieldValue)>,
}

impl Delta {
    /// The baseline revision the delta was encoded against.
    #[must_use]
    pub const fn revision(&self) -> u32 {
        self.revision
    }

    /// The changed `(index, value)` pairs, in schema order.
    #[must_use]
    pub fn changed(&self) -> &[(usize, FieldValue)] {
        &self.changed
    }
}

/// Encode the fields of `current` that differ from `baseline`, against revision
/// `revision`, appending to `out`.
///
/// # Errors
/// [`DeltaError::Shape`] / [`DeltaError::KindMismatch`] for a record that does
/// not match `schema`, and [`DeltaError::TooManyFields`] past 255 fields.
pub fn encode_delta(
    schema: &Schema,
    baseline: &Record,
    current: &Record,
    revision: u32,
    out: &mut Vec<u8>,
) -> Result<(), DeltaError> {
    check_shape(schema, baseline)?;
    check_shape(schema, current)?;
    let count =
        u8::try_from(schema.len()).map_err(|_| DeltaError::TooManyFields { len: schema.len() })?;

    out.extend_from_slice(&revision.to_le_bytes());
    out.push(count);

    let mut bits = BitWriter::new();
    for (kind, (base, now)) in schema
        .kinds
        .iter()
        .zip(baseline.values.iter().zip(&current.values))
    {
        let changed = base != now;
        bits.write(u64::from(changed), 1);
        if changed {
            write_value(*kind, *now, &mut bits);
        }
    }
    out.extend_from_slice(&bits.into_bytes());
    Ok(())
}

/// Decode a delta encoded against `schema`.
///
/// # Errors
/// [`DeltaError::SchemaMismatch`] when the payload was encoded with a different
/// schema width, [`DeltaError::Truncated`] when it ends early.
pub fn decode_delta(schema: &Schema, bytes: &[u8]) -> Result<Delta, DeltaError> {
    check_kinds(schema)?;
    if bytes.len() < 5 {
        return Err(DeltaError::Truncated);
    }
    let revision = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    let found = bytes[4] as usize;
    if found != schema.len() {
        return Err(DeltaError::SchemaMismatch {
            expected: schema.len(),
            found,
        });
    }

    let mut reader = BitReader::new(&bytes[5..]);
    let mut changed = Vec::new();
    for (index, kind) in schema.kinds.iter().enumerate() {
        if reader.read(1).ok_or(DeltaError::Truncated)? != 0 {
            changed.push((index, read_value(*kind, &mut reader)?));
        }
    }
    Ok(Delta { revision, changed })
}

/// Apply a delta to `base`, which the receiver holds at `held_revision`.
///
/// Returns the number of fields written. On a revision mismatch the delta is
/// refused (the caller must resync from a snapshot) and `base` is untouched.
///
/// # Errors
/// Any [`decode_delta`] error, plus [`DeltaError::RevisionMismatch`].
pub fn apply_delta(
    schema: &Schema,
    base: &mut Record,
    held_revision: u32,
    bytes: &[u8],
) -> Result<usize, DeltaError> {
    let delta = decode_delta(schema, bytes)?;
    if delta.revision != held_revision {
        return Err(DeltaError::RevisionMismatch {
            expected: held_revision,
            found: delta.revision,
        });
    }
    check_shape(schema, base)?;
    let mut applied = 0;
    for &(index, value) in &delta.changed {
        base.set(index, value);
        applied += 1;
    }
    Ok(applied)
}

/// Encode a **full** record at `revision` — the seed a late joiner or a resyncing
/// peer applies before any delta arrives.
///
/// Same header as a [`encode_delta`], but every field is written in schema order
/// with no changed-field mask (the mask would always be all-ones).
///
/// # Errors
/// The same record/schema errors as [`encode_delta`].
pub fn encode_snapshot(
    schema: &Schema,
    revision: u32,
    record: &Record,
    out: &mut Vec<u8>,
) -> Result<(), DeltaError> {
    check_shape(schema, record)?;
    let count =
        u8::try_from(schema.len()).map_err(|_| DeltaError::TooManyFields { len: schema.len() })?;

    out.extend_from_slice(&revision.to_le_bytes());
    out.push(count);

    let mut bits = BitWriter::new();
    for (kind, value) in schema.kinds.iter().zip(&record.values) {
        write_value(*kind, *value, &mut bits);
    }
    out.extend_from_slice(&bits.into_bytes());
    Ok(())
}

/// Decode a full-record snapshot: its revision and values.
///
/// # Errors
/// [`DeltaError::SchemaMismatch`] on a field-count mismatch and
/// [`DeltaError::Truncated`] when the payload ends early, exactly as
/// [`decode_delta`].
pub fn decode_snapshot(schema: &Schema, bytes: &[u8]) -> Result<(u32, Record), DeltaError> {
    check_kinds(schema)?;
    if bytes.len() < 5 {
        return Err(DeltaError::Truncated);
    }
    let revision = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    let found = bytes[4] as usize;
    if found != schema.len() {
        return Err(DeltaError::SchemaMismatch {
            expected: schema.len(),
            found,
        });
    }

    let mut reader = BitReader::new(&bytes[5..]);
    let mut values = Vec::with_capacity(schema.len());
    for kind in &schema.kinds {
        values.push(read_value(*kind, &mut reader)?);
    }
    Ok((revision, Record::new(values)))
}

fn check_shape(schema: &Schema, record: &Record) -> Result<(), DeltaError> {
    check_kinds(schema)?;
    if record.values.len() != schema.len() {
        return Err(DeltaError::Shape {
            expected: schema.len(),
            found: record.values.len(),
        });
    }
    for (index, (kind, value)) in schema.kinds.iter().zip(&record.values).enumerate() {
        if !kind.matches(*value) {
            return Err(DeltaError::KindMismatch { index });
        }
        if !value_fits_kind(*kind, *value) {
            return Err(DeltaError::ValueOutOfRange { index });
        }
    }
    Ok(())
}

/// Whether `value` is representable by `kind`'s wire form. Only integers have a
/// bounded value domain; a bool or a quantized float always fits (the float's own
/// range is a schema concern, checked by [`check_kinds`]).
fn value_fits_kind(kind: FieldKind, value: FieldValue) -> bool {
    match (kind, value) {
        (FieldKind::Int { bits }, FieldValue::Int(number)) => int_fits(number, bits),
        _ => true,
    }
}

/// Whether `value` fits the signed range a `bits`-wide zigzag code can represent:
/// `[-(2^(bits-1)), 2^(bits-1) - 1]` for `bits < 64`, and all of `i64` at 64 bits
/// (where the high bit carries the sign). A zero width fits nothing.
fn int_fits(value: i64, bits: u8) -> bool {
    if bits >= 64 {
        return true;
    }
    if bits == 0 {
        return false;
    }
    let half = 1i64 << (bits - 1);
    (-half..=half - 1).contains(&value)
}

/// Validate the schema's own field-kind parameters, so the bit packer and
/// quantizer are never handed a width or range they would panic on.
fn check_kinds(schema: &Schema) -> Result<(), DeltaError> {
    for (index, kind) in schema.kinds.iter().enumerate() {
        let valid = match *kind {
            FieldKind::Bool => true,
            FieldKind::Int { bits } => (1..=64).contains(&bits),
            FieldKind::Float { min, max, bits } => {
                if (1..=24).contains(&bits) {
                    // A finite range whose span overflows `f32` (or is so small
                    // that the per-unit step overflows) makes `Quantizer` collapse
                    // endpoints to `NaN`/`inf`; require a finite, positive step.
                    let inv_step = ((1u64 << bits) - 1) as f32 / (max - min);
                    min.is_finite()
                        && max.is_finite()
                        && min < max
                        && inv_step.is_finite()
                        && inv_step > 0.0
                } else {
                    false
                }
            }
        };
        if !valid {
            return Err(DeltaError::InvalidKind { index });
        }
    }
    Ok(())
}

fn write_value(kind: FieldKind, value: FieldValue, bits: &mut BitWriter) {
    match (kind, value) {
        (FieldKind::Bool, FieldValue::Bool(flag)) => bits.write(u64::from(flag), 1),
        (FieldKind::Int { bits: width }, FieldValue::Int(number)) => {
            bits.write(zigzag(number), width);
        }
        (
            FieldKind::Float {
                min,
                max,
                bits: width,
            },
            FieldValue::Float(number),
        ) => {
            bits.write(Quantizer::new(min, max, width).quantize(number), width);
        }
        // `check_shape` already rejected a mismatched variant.
        (FieldKind::Bool, _) | (FieldKind::Int { .. }, _) | (FieldKind::Float { .. }, _) => {}
    }
}

fn read_value(kind: FieldKind, reader: &mut BitReader<'_>) -> Result<FieldValue, DeltaError> {
    match kind {
        FieldKind::Bool => Ok(FieldValue::Bool(
            reader.read(1).ok_or(DeltaError::Truncated)? != 0,
        )),
        FieldKind::Int { bits } => Ok(FieldValue::Int(unzigzag(
            reader.read(bits).ok_or(DeltaError::Truncated)?,
        ))),
        FieldKind::Float { min, max, bits } => Ok(FieldValue::Float(
            Quantizer::new(min, max, bits)
                .dequantize(reader.read(bits).ok_or(DeltaError::Truncated)?),
        )),
    }
}

/// Zigzag an `i64` so small magnitudes (either sign) get small codes.
fn zigzag(value: i64) -> u64 {
    ((value << 1) ^ (value >> 63)) as u64
}

/// Inverse of [`zigzag`].
fn unzigzag(code: u64) -> i64 {
    ((code >> 1) as i64) ^ -((code & 1) as i64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn schema() -> Schema {
        Schema::new(vec![
            FieldKind::Bool,
            FieldKind::Int { bits: 12 },
            FieldKind::Float {
                min: -10.0,
                max: 10.0,
                bits: 16,
            },
            FieldKind::Int { bits: 8 },
        ])
    }

    fn record(flag: bool, first: i64, angle: f32, second: i64) -> Record {
        Record::new(vec![
            FieldValue::Bool(flag),
            FieldValue::Int(first),
            FieldValue::Float(angle),
            FieldValue::Int(second),
        ])
    }

    #[test]
    fn round_trip_applies_only_the_changed_fields() {
        let schema = schema();
        let base = record(false, 10, 0.0, 3);
        let current = record(true, 10, 0.0, 9); // fields 0 and 3 changed

        let mut bytes = Vec::new();
        encode_delta(&schema, &base, &current, 7, &mut bytes).unwrap();

        let delta = decode_delta(&schema, &bytes).unwrap();
        assert_eq!(delta.revision(), 7);
        assert_eq!(
            delta.changed(),
            &[(0, FieldValue::Bool(true)), (3, FieldValue::Int(9))]
        );

        let mut applied = base.clone();
        assert_eq!(apply_delta(&schema, &mut applied, 7, &bytes).unwrap(), 2);
        assert_eq!(applied, current);
    }

    #[test]
    fn an_unchanged_record_encodes_to_header_only() {
        let schema = schema();
        let base = record(true, -5, 1.0, 2);
        let mut bytes = Vec::new();
        encode_delta(&schema, &base, &base, 0, &mut bytes).unwrap();
        // Five header bytes plus the four-bit mask, padded to one byte.
        assert_eq!(bytes.len(), 6, "no field body when nothing changed");
        assert!(decode_delta(&schema, &bytes).unwrap().changed().is_empty());
    }

    #[test]
    fn a_revision_mismatch_is_refused_and_leaves_the_base_untouched() {
        let schema = schema();
        let base = record(false, 1, 0.0, 1);
        let current = record(true, 1, 0.0, 1);
        let mut bytes = Vec::new();
        encode_delta(&schema, &base, &current, 4, &mut bytes).unwrap();

        let mut applied = base.clone();
        assert_eq!(
            apply_delta(&schema, &mut applied, 5, &bytes),
            Err(DeltaError::RevisionMismatch {
                expected: 5,
                found: 4
            })
        );
        assert_eq!(applied, base, "a refused delta writes nothing");
    }

    #[test]
    fn schema_and_truncation_errors_are_reported() {
        let schema = schema();
        let base = record(false, 1, 0.0, 1);
        let mut bytes = Vec::new();
        encode_delta(&schema, &base, &base, 0, &mut bytes).unwrap();

        let narrower = Schema::new(vec![FieldKind::Bool]);
        assert_eq!(
            decode_delta(&narrower, &bytes),
            Err(DeltaError::SchemaMismatch {
                expected: 1,
                found: 4
            })
        );

        // A truncated body fails rather than zero-filling.
        let current = record(true, 200, 5.0, 100);
        let mut full = Vec::new();
        encode_delta(&schema, &base, &current, 0, &mut full).unwrap();
        let truncated = &full[..full.len() - 1];
        assert!(matches!(
            decode_delta(&schema, truncated),
            Err(DeltaError::Truncated)
        ));
    }

    /// An integer past its declared width would silently truncate; encode must
    /// reject it, and the boundary values must still round-trip exactly.
    #[test]
    fn an_integer_out_of_its_declared_width_is_rejected_not_truncated() {
        // A 12-bit signed field represents `[-2048, 2047]`.
        let schema = Schema::new(vec![FieldKind::Int { bits: 12 }]);
        let base = Record::new(vec![FieldValue::Int(0)]);

        for value in [-2048i64, -1, 0, 2047] {
            let target = Record::new(vec![FieldValue::Int(value)]);
            let mut bytes = Vec::new();
            encode_snapshot(&schema, 1, &target, &mut bytes).unwrap();
            assert_eq!(decode_snapshot(&schema, &bytes).unwrap().1, target);
        }

        for value in [-2049i64, 2048, 3000] {
            let target = Record::new(vec![FieldValue::Int(value)]);
            let mut snapshot = Vec::new();
            assert_eq!(
                encode_snapshot(&schema, 1, &target, &mut snapshot),
                Err(DeltaError::ValueOutOfRange { index: 0 }),
                "value {value} must be refused, not wrapped"
            );
            assert!(snapshot.is_empty(), "nothing is written on rejection");
            let mut delta = Vec::new();
            assert_eq!(
                encode_delta(&schema, &base, &target, 0, &mut delta),
                Err(DeltaError::ValueOutOfRange { index: 0 })
            );
        }

        // A 64-bit field carries the whole `i64` range.
        let wide = Schema::new(vec![FieldKind::Int { bits: 64 }]);
        for value in [i64::MIN, -1, 0, i64::MAX] {
            let target = Record::new(vec![FieldValue::Int(value)]);
            let mut bytes = Vec::new();
            encode_snapshot(&wide, 1, &target, &mut bytes).unwrap();
            assert_eq!(decode_snapshot(&wide, &bytes).unwrap().1, target);
        }
    }

    /// A finite float range whose span overflows `f32` would make the quantizer
    /// collapse every value to `NaN`/`inf`; it is refused as an invalid kind.
    #[test]
    fn a_float_range_whose_span_overflows_is_invalid() {
        let schema = Schema::new(vec![FieldKind::Float {
            min: -3.0e38,
            max: 3.0e38,
            bits: 24,
        }]);
        let record = Record::new(vec![FieldValue::Float(0.0)]);
        let mut bytes = Vec::new();
        assert_eq!(
            encode_snapshot(&schema, 1, &record, &mut bytes),
            Err(DeltaError::InvalidKind { index: 0 })
        );
        // Decoding any payload under the same schema is refused up front too.
        assert_eq!(
            decode_snapshot(&schema, &[0, 0, 0, 0, 1, 0, 0, 0]),
            Err(DeltaError::InvalidKind { index: 0 })
        );
    }

    #[test]
    fn wrong_shaped_record_or_wrong_kind_is_rejected() {
        let schema = schema();
        let short = Record::new(vec![FieldValue::Bool(true)]);
        let mut bytes = Vec::new();
        assert_eq!(
            encode_delta(&schema, &short, &short, 0, &mut bytes),
            Err(DeltaError::Shape {
                expected: 4,
                found: 1
            })
        );

        let wrong_kind = Record::new(vec![
            FieldValue::Int(0), // should be Bool
            FieldValue::Int(1),
            FieldValue::Float(0.0),
            FieldValue::Int(2),
        ]);
        assert_eq!(
            encode_delta(&schema, &wrong_kind, &wrong_kind, 0, &mut bytes),
            Err(DeltaError::KindMismatch { index: 0 })
        );
    }

    #[test]
    fn quantized_float_fields_round_trip_within_the_error_bound() {
        let schema = schema();
        let base = record(false, 0, 0.0, 0);
        let current = record(false, 0, 3.7, 0);
        let mut bytes = Vec::new();
        encode_delta(&schema, &base, &current, 0, &mut bytes).unwrap();

        let mut applied = base.clone();
        apply_delta(&schema, &mut applied, 0, &bytes).unwrap();
        let FieldValue::Float(got) = applied.value(2).unwrap() else {
            panic!("field 2 is a float");
        };
        let bound = Quantizer::new(-10.0, 10.0, 16).max_error();
        assert!((got - 3.7).abs() <= bound, "got {got}, bound {bound}");
    }

    #[test]
    fn zigzag_round_trips_every_extreme() {
        for value in [0, 1, -1, i64::MAX, i64::MIN, 12345, -12345] {
            assert_eq!(unzigzag(zigzag(value)), value, "value {value}");
        }
    }

    /// An invalid field kind is a reported error, never a panic inside the bit
    /// packer / quantizer, and it is caught before any header is appended.
    #[test]
    fn invalid_field_kinds_are_reported_not_panicked() {
        let base = Record::new(vec![FieldValue::Int(1)]);
        for kind in [
            FieldKind::Int { bits: 0 },
            FieldKind::Int { bits: 65 },
            FieldKind::Float {
                min: 1.0,
                max: 0.0,
                bits: 8,
            },
            FieldKind::Float {
                min: 0.0,
                max: 1.0,
                bits: 25,
            },
        ] {
            let schema = Schema::new(vec![kind]);
            let mut bytes = Vec::new();
            assert_eq!(
                encode_delta(&schema, &base, &base, 0, &mut bytes),
                Err(DeltaError::InvalidKind { index: 0 }),
                "encode kind {kind:?}"
            );
            assert!(bytes.is_empty(), "no header is written for an invalid kind");
            assert_eq!(
                decode_delta(&schema, &[0, 0, 0, 0, 1, 0]),
                Err(DeltaError::InvalidKind { index: 0 }),
                "decode kind {kind:?}"
            );
        }
    }

    #[test]
    fn a_full_snapshot_round_trips_and_rejects_mismatches() {
        let schema = schema();
        let rec = record(true, -42, 2.5, 7);
        let mut bytes = Vec::new();
        encode_snapshot(&schema, 5, &rec, &mut bytes).unwrap();

        let (revision, decoded) = decode_snapshot(&schema, &bytes).unwrap();
        assert_eq!(revision, 5);
        assert_eq!(decoded.value(0), Some(FieldValue::Bool(true)));
        assert_eq!(decoded.value(1), Some(FieldValue::Int(-42)));
        assert_eq!(decoded.value(3), Some(FieldValue::Int(7)));
        let FieldValue::Float(angle) = decoded.value(2).unwrap() else {
            panic!("field 2 is a float");
        };
        let bound = Quantizer::new(-10.0, 10.0, 16).max_error();
        assert!((angle - 2.5).abs() <= bound, "angle {angle}, bound {bound}");

        let narrower = Schema::new(vec![FieldKind::Bool]);
        assert_eq!(
            decode_snapshot(&narrower, &bytes),
            Err(DeltaError::SchemaMismatch {
                expected: 1,
                found: 4
            })
        );
        let truncated = &bytes[..bytes.len() - 1];
        assert!(matches!(
            decode_snapshot(&schema, truncated),
            Err(DeltaError::Truncated)
        ));
    }
}
