//! Structural value equality used by [`DynamicMap`](super::DynamicMap) keys.

use crate::reflect::Reflect;
use alloc::string::String;

/// Best-effort equality for two reflected leaf values.
///
/// Compares `a` and `b` by downcasting through the set of built-in leaf types
/// (`bool`, `char`, the integer types, `f32`/`f64`, and `String`). Any pair
/// whose concrete types differ, or that is not a known leaf type, compares as
/// not equal. This is sufficient for the string/scalar keys dynamic maps use in
/// M2.
pub(crate) fn reflect_values_equal(a: &dyn Reflect, b: &dyn Reflect) -> bool {
    macro_rules! try_eq {
        ($($ty:ty),* $(,)?) => {
            $(
                if let (Some(lhs), Some(rhs)) = (
                    a.as_any().downcast_ref::<$ty>(),
                    b.as_any().downcast_ref::<$ty>(),
                ) {
                    return lhs == rhs;
                }
            )*
        };
    }

    try_eq!(bool, char, i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize, String,);

    if let (Some(lhs), Some(rhs)) = (
        a.as_any().downcast_ref::<f32>(),
        b.as_any().downcast_ref::<f32>(),
    ) {
        return lhs == rhs;
    }
    if let (Some(lhs), Some(rhs)) = (
        a.as_any().downcast_ref::<f64>(),
        b.as_any().downcast_ref::<f64>(),
    ) {
        return lhs == rhs;
    }

    false
}

/// Exact (bit-pattern) equality for two reflected leaf values.
///
/// Same contract as [`reflect_values_equal`], except `f32`/`f64` compare by
/// [`f32::to_bits`] rather than `==`. [`diff`](crate::diff::diff) uses this so
/// `diff(a, a)` is always `Unchanged` — even when a leaf is `NaN` — and a
/// `+0.0`/`-0.0` change is not silently missed (CR-found). Map-key lookups keep
/// the IEEE `==` semantics of [`reflect_values_equal`].
pub(crate) fn reflect_values_identical(a: &dyn Reflect, b: &dyn Reflect) -> bool {
    if let (Some(lhs), Some(rhs)) = (
        a.as_any().downcast_ref::<f32>(),
        b.as_any().downcast_ref::<f32>(),
    ) {
        return lhs.to_bits() == rhs.to_bits();
    }
    if let (Some(lhs), Some(rhs)) = (
        a.as_any().downcast_ref::<f64>(),
        b.as_any().downcast_ref::<f64>(),
    ) {
        return lhs.to_bits() == rhs.to_bits();
    }
    reflect_values_equal(a, b)
}
