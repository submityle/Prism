//! Large-world partitioning (design §13): cell streaming, entity LOD/dormancy,
//! and 64-bit floating-origin rebasing.
//!
//! M5 build-up. The 64-bit floating-origin kernel lands first in
//! [`floating_origin`]; cell streaming and LOD/dormancy follow.

pub mod floating_origin;
