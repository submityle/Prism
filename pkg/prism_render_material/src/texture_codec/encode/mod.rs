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
//! * [`bc2`] -- BC2/DXT3 encoder (opaque BC1 colour + explicit 4-bit alpha).
//! * [`bc3`] -- BC3/DXT5 encoder (opaque BC1 colour + BC4 alpha composite).
//! * [`bc4`] -- BC4/RGTC single-channel encoder (min/max endpoints,
//!   eight-value / six-value mode selection by block error).
//! * [`bc5`] -- BC5/RGTC2 two-channel encoder (two BC4 blocks, tangent
//!   normal `XY`).
//! * [`bc7`] -- BC7/BPTC single-subset encoders: mode 6 (4D `RGBA` PCA axis,
//!   7-bit endpoints + p-bits, shared index) and mode 5 (3D `RGB` PCA +
//!   separate 8-bit alpha, independent 2-bit colour/alpha indices, rotation),
//!   both with least-squares index refit.
//! * [`bc6h`] -- BC6H/BPTC unsigned and signed mode-11 HDR encoders (half-float
//!   domain, 10-bit direct endpoints, finish-chain-aware index assignment).

mod bc1;
mod bc2;
mod bc3;
mod bc4;
mod bc5;
mod bc6h;
mod bc7;

pub use bc1::encode_bc1;
pub use bc2::encode_bc2;
pub use bc3::encode_bc3;
pub use bc4::encode_bc4;
pub use bc5::encode_bc5;
pub use bc6h::{encode_bc6h_mode11_signed, encode_bc6h_mode11_unsigned};
pub use bc7::{encode_bc7_mode5, encode_bc7_mode6};
