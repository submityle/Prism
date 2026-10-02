//! Block-compressed texture *encoders* (the compression path).
//!
//! These are the inverses of the decoders in the parent
//! [`texture_codec`](super) module: they take uncompressed `RGBA8`/`R8`
//! tiles and emit GPU block-compressed blocks for an offline texture-bake
//! tool. Encoding is pure analytic/integer arithmetic -- no AI/ML -- so the
//! CPU result is deterministic and reproduces a GPU twin bit-for-bit given
//! the same documented tie-breaks.
//!
//! * [`bc1`] -- BC1/DXT1 colour encoder (PCA endpoint fit + least-squares
//!   refinement, opaque 4-colour and 1-bit punch-through modes).
//! * [`bc4`] -- BC4/RGTC single-channel encoder (min/max endpoints,
//!   eight-value / six-value mode selection by block error).

mod bc1;
mod bc4;

pub use bc1::encode_bc1;
pub use bc4::encode_bc4;
