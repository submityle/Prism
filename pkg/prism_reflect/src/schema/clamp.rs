//! Clamping a reflected value's numeric fields into their metadata ranges
//! (design §24.7 — the `#[reflect(clamp(min..=max))]` attribute).
//!
//! Where [`validate`](super::validate::validate) only *reports* an out-of-range
//! field, [`clamp`] actively *repairs* one: for every range-constrained field
//! in a [`TypeMetadata`] it reads the live value through reflection, pins it
//! into the inclusive `[min, max]` window, and writes the corrected value back.
//! This is the runtime half of the clamp attribute — used by editors (slider
//! drags that must stay in-bounds) and by deserialization of untrusted payloads
//! that should be sanitized rather than rejected.
//!
//! Clamping is **lossless within range** (an in-bounds value is left untouched,
//! so no field is reported as changed) and operates per leaf type: integer
//! fields saturate to the integer nearest the bound, floating-point fields pin
//! exactly. Non-numeric range-constrained fields are skipped (they are a
//! metadata authoring error that [`validate`](super::validate::validate)
//! surfaces as [`NotNumeric`](super::validate::ValidationError::NotNumeric)).
//!
//! ```
//! use prism_reflect::prelude::*;
//! use prism_reflect::schema::{clamp, FieldMetadata, TypeMetadata};
//!
//! #[derive(Reflect, Default)]
//! struct Light {
//!     intensity: f32,
//!     bounces: i32,
//! }
//!
//! let meta = TypeMetadata::new()
//!     .with_field(FieldMetadata::new("intensity").with_range(0.0, 1.0))
//!     .with_field(FieldMetadata::new("bounces").with_range(0.0, 8.0));
//!
//! let mut light = Light { intensity: 4.5, bounces: -3 };
//! let changed = clamp(&mut light, &meta);
//! assert_eq!(changed.len(), 2);
//! assert_eq!(light.intensity, 1.0);
//! assert_eq!(light.bounces, 0);
//! ```

use crate::reflect::Reflect;
use crate::schema::metadata::TypeMetadata;
use crate::ReflectMut;
use alloc::string::String;
use alloc::vec::Vec;

/// A record of one field whose value was pinned into its range by [`clamp`].
#[derive(Debug, Clone, PartialEq)]
pub struct Clamped {
    /// The clamped field's name.
    pub field: String,
    /// The value before clamping, widened to `f64`.
    pub from: f64,
    /// The value after clamping, widened to `f64`.
    pub to: f64,
}

/// Clamp every range-constrained numeric field of `value` into its metadata
/// bounds, writing corrected values back in place.
///
/// Returns one [`Clamped`] record per field that was actually changed; an empty
/// vector means every field was already in range (or the value is not a struct,
/// or no field carries a range). In-range values are never rewritten, so this
/// is idempotent: a second call on the output always returns an empty vector.
#[must_use = "the returned records report which fields were changed"]
pub fn clamp(value: &mut dyn Reflect, metadata: &TypeMetadata) -> Vec<Clamped> {
    let mut changed = Vec::new();

    let ReflectMut::Struct(source) = value.reflect_mut() else {
        return changed;
    };

    for field in metadata.fields() {
        let Some((min, max)) = field.range() else {
            continue;
        };
        // A malformed `min > max` range cannot clamp meaningfully; skip it.
        if min > max {
            continue;
        }
        let Some(slot) = source.field_mut(field.name()) else {
            continue;
        };
        if let Some((from, to)) = clamp_leaf(slot, min, max) {
            changed.push(Clamped {
                field: String::from(field.name()),
                from,
                to,
            });
        }
    }

    changed
}

/// Clamp a single numeric leaf in place.
///
/// Returns `Some((from, to))` (both widened to `f64`) when the stored value was
/// outside `[min, max]` and has been rewritten, or `None` when the value was
/// already in range or the leaf is not a supported numeric type.
fn clamp_leaf(slot: &mut dyn Reflect, min: f64, max: f64) -> Option<(f64, f64)> {
    let any = slot.as_any_mut();

    // Floating-point leaves pin exactly.
    if let Some(v) = any.downcast_mut::<f64>() {
        let clamped = v.clamp(min, max);
        return (clamped != *v).then(|| {
            let from = *v;
            *v = clamped;
            (from, clamped)
        });
    }
    if let Some(v) = any.downcast_mut::<f32>() {
        let from = f64::from(*v);
        // Clamp in f64 then narrow; the bounds themselves are f64.
        let clamped = from.clamp(min, max) as f32;
        return (clamped != *v).then(|| {
            *v = clamped;
            (from, f64::from(clamped))
        });
    }

    // Integer leaves saturate to the integer nearest the (possibly fractional)
    // bound that lies inside the window: round the clamped value toward the
    // interior so the result always satisfies `min <= result <= max`.
    macro_rules! clamp_int {
        ($ty:ty) => {
            if let Some(v) = any.downcast_mut::<$ty>() {
                let from = *v as f64;
                if from < min || from > max {
                    // Round toward the interior: ceil the lower bound, floor the
                    // upper, so a fractional bound never pushes back out of range.
                    let lo = min.ceil();
                    let hi = max.floor();
                    let target = from.clamp(lo, hi);
                    let narrowed = target as $ty;
                    *v = narrowed;
                    return Some((from, narrowed as f64));
                }
                return None;
            }
        };
    }
    clamp_int!(i8);
    clamp_int!(i16);
    clamp_int!(i32);
    clamp_int!(i64);
    clamp_int!(isize);
    clamp_int!(u8);
    clamp_int!(u16);
    clamp_int!(u32);
    clamp_int!(u64);
    clamp_int!(usize);

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;
    use crate::schema::FieldMetadata;

    #[derive(Reflect, Default, Debug)]
    struct Sample {
        f: f32,
        d: f64,
        i: i32,
        u: u8,
        name: String,
    }

    fn meta() -> TypeMetadata {
        TypeMetadata::new()
            .with_field(FieldMetadata::new("f").with_range(0.0, 1.0))
            .with_field(FieldMetadata::new("d").with_range(-10.0, 10.0))
            .with_field(FieldMetadata::new("i").with_range(0.0, 100.0))
            .with_field(FieldMetadata::new("u").with_range(5.0, 200.0))
    }

    #[test]
    fn clamps_each_leaf_type_to_bounds() {
        let mut s = Sample {
            f: 5.0,
            d: -99.0,
            i: 1000,
            u: 1,
            name: String::from("x"),
        };
        let changed = clamp(&mut s, &meta());
        assert_eq!(changed.len(), 4);
        assert_eq!(s.f, 1.0);
        assert_eq!(s.d, -10.0);
        assert_eq!(s.i, 100);
        assert_eq!(s.u, 5);
    }

    #[test]
    fn in_range_values_are_untouched_and_idempotent() {
        let mut s = Sample {
            f: 0.5,
            d: 3.0,
            i: 50,
            u: 50,
            name: String::from("y"),
        };
        let first = clamp(&mut s, &meta());
        assert!(first.is_empty(), "nothing should change");
        // Idempotence: clamping again still changes nothing.
        let second = clamp(&mut s, &meta());
        assert!(second.is_empty());
    }

    #[test]
    fn report_records_from_and_to() {
        let mut s = Sample {
            f: 2.0,
            ..Default::default()
        };
        let changed = clamp(&mut s, &meta());
        let rec = changed.iter().find(|c| c.field == "f").unwrap();
        assert_eq!(rec.from, 2.0);
        assert_eq!(rec.to, 1.0);
    }

    #[test]
    fn fractional_bounds_round_into_the_interior_for_integers() {
        // Range 0.3..=9.7 on an i32: a value of 20 must land on 9 (floor of the
        // upper bound), never 10 (which would exceed the bound).
        let mut s = Sample {
            i: 20,
            ..Default::default()
        };
        let meta = TypeMetadata::new()
            .with_field(FieldMetadata::new("i").with_range(0.3, 9.7));
        let changed = clamp(&mut s, &meta);
        assert_eq!(changed.len(), 1);
        assert_eq!(s.i, 9);
        assert!(f64::from(s.i) <= 9.7);
    }

    #[test]
    fn non_numeric_and_missing_fields_are_skipped() {
        let mut s = Sample::default();
        let meta = TypeMetadata::new()
            .with_field(FieldMetadata::new("name").with_range(0.0, 1.0))
            .with_field(FieldMetadata::new("ghost").with_range(0.0, 1.0));
        let changed = clamp(&mut s, &meta);
        assert!(changed.is_empty(), "non-numeric/missing fields are skipped");
    }

    #[test]
    fn malformed_range_is_skipped() {
        let mut s = Sample {
            i: 500,
            ..Default::default()
        };
        let meta = TypeMetadata::new()
            .with_field(FieldMetadata::new("i").with_range(100.0, 1.0));
        let changed = clamp(&mut s, &meta);
        assert!(changed.is_empty(), "min > max cannot clamp");
        assert_eq!(s.i, 500);
    }
}
