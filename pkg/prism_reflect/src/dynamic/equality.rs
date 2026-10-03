//! Structural value equality used by [`DynamicMap`](super::DynamicMap) keys.

use crate::reflect::Reflect;
use std::string::String;

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

    try_eq!(
        bool, char, i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize, String,
    );

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
