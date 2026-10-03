//! The M1 container reflection kinds: `Enum`, `List`, `Array`, `Map`, `Set`.
//!
//! Each kind is a `Reflect` subtrait exposing dynamic traversal over a shape
//! that cannot be described by the M0 `Struct`/`TupleStruct`/`Value` kinds.
//! The standard-library collection implementations live alongside their trait
//! in each submodule.

mod array;
mod enum_;
mod list;
mod map;
mod set;

pub use array::{Array, ArrayIter};
pub use enum_::{Enum, VariantType};
pub use list::{List, ListIter};
pub use map::{Map, MapIter};
pub use set::{Set, SetIter};
