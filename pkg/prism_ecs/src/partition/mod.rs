//! Large-world partitioning (design §13): cell streaming, entity LOD/dormancy,
//! and 64-bit floating-origin rebasing.
//!
//! M5 build-up. The 64-bit floating-origin kernel lands first in
//! [`floating_origin`]; cell streaming and LOD/dormancy follow.

pub mod cell;
pub mod data_layer;
pub mod driver;
pub mod dormant;
pub mod floating_origin;
pub mod hlod;
pub mod interest;
pub mod lod;
pub mod processor;
pub mod streaming;
pub mod view;
pub mod world_partition;
