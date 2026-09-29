//! Optional `wgpu` compute twin of Prism's virtual-geometry vis-buffer software
//! rasterizer.
//!
//! Virtualized geometry routes sub-pixel clusters through a compute *software
//! rasterizer* that writes a visibility buffer rather than shaded pixels. The
//! CPU golden standard for that math lives in
//! [`prism_render_architecture::virtual_geometry`]; this crate is the `GPU`
//! twin, validated texel-for-texel against that reference so a passing
//! real-device parity test is direct evidence the ported kernel computes the
//! same coverage and depth as the reference, not merely that its shader
//! compiles.
//!
//! # Portability
//!
//! The shipping meshlet raster packs a 64-bit `(depth << 32) | payload` key
//! into a storage-texture atomic. Two twins cover the two portability tiers:
//!
//! * [`GpuSoftwareRaster`] composites only the 32-bit reversed-Z depth key
//!   through [`atomicMax`](https://www.w3.org/TR/WGSL/#atomic-rmw) on a plain
//!   `atomic<u32>` storage buffer, the portable core-`WGSL` subset every
//!   backend implements, so it runs unmodified on Metal, Vulkan, and DX12 with
//!   no optional feature. It is a faithful test of the coverage rule, the
//!   winding swap, the top-left fill, and the reversed-Z depth encode.
//! * [`GpuPayloadRaster`] composites the full 64-bit `(depth << 32) | payload`
//!   word through a single `atomic<u64>` `atomicMax`, reproducing the shipping
//!   visibility word bit-for-bit (nearest depth wins and records the winning
//!   payload for free). It uses a `storage`-buffer 64-bit atomic rather than an
//!   `r64uint` storage-texture atomic, so it runs anywhere the `SHADER_INT64`
//!   and `SHADER_INT64_ATOMIC_MIN_MAX` features are present - including Metal on
//!   Apple silicon - and [`GpuPayloadRaster::new`] returns [`None`] where they
//!   are not, so callers skip gracefully.
//!
//! # Correctness model
//!
//! For triangles built on integer pixel coordinates the signed edge function is
//! evaluated on exactly representable operands, so its value is identical on
//! `CPU` and `GPU` regardless of fused-multiply-add contraction, making the
//! covered texel set bit-exact. When vertex depths are exact negative powers of
//! two the barycentric depth blend is likewise fma-immune (each `b * d` product
//! is exact), so the composited depth key is bit-exact too. The parity test
//! exploits this to assert texel-for-texel equality rather than a tolerance.
//!
//! # Physical page table
//!
//! Beyond the rasterizer, [`GpuPageTable`] is the on-device twin of the
//! physical page pool's key-to-slot lookup: it binary-searches the sorted
//! resident `(key, slot)` table the CPU golden
//! [`PagePool`](prism_render_architecture::paging::PagePool) exports, one
//! thread per query, and is validated slot-for-slot against
//! [`PagePool::slot_of`](prism_render_architecture::paging::PagePool::slot_of).
//! It uses only portable integer WGSL, so it needs no optional feature.
//!
//! # Physical page storage
//!
//! One layer below the table, [`GpuPageStorage`] is the on-device twin of the
//! physical page-data placement: it scatters streamed page payloads into their
//! pool slots and gathers single words back out, mirroring
//! [`PageStorage`](prism_render_architecture::paging::PageStorage) word-for-word.
//! Together the two twins cover the streaming path end to end - resolve a page
//! key to a slot, then place and read that slot's data. It too uses only
//! portable integer WGSL and needs no optional feature.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard signed-edge / top-left-rule software rasterization; no
//! Unreal Engine source or derived code.
#![forbid(unsafe_code)]

pub mod context;
pub mod page_pool;
pub mod page_storage;
pub mod payload_raster;
pub mod raster;

pub use context::{block_on, GpuContext};
pub use page_pool::{GpuPageTable, ResolveError};
pub use page_storage::GpuPageStorage;
pub use payload_raster::GpuPayloadRaster;
pub use raster::{GpuSoftwareRaster, RasterError};
