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
//! The shipping meshlet raster packs a 64-bit depth/payload key into an
//! `r64uint` storage-texture atomic (`textureAtomicMax`), which Metal does not
//! support. This twin instead composites the 32-bit reversed-Z depth key
//! through [`atomicMax`](https://www.w3.org/TR/WGSL/#atomic-rmw) on a plain
//! `atomic<u32>` storage buffer, the portable core-`WGSL` subset every backend
//! implements, so it runs unmodified on Metal, Vulkan, and DX12. Carrying the
//! full 64-bit payload key (the `r64uint` extension path) is the documented
//! follow-up. The depth-only twin is still a faithful test of the coverage
//! rule, the winding swap, the top-left fill, and the reversed-Z depth encode,
//! which are the parts the CPU reference pins.
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
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard signed-edge / top-left-rule software rasterization; no
//! Unreal Engine source or derived code.
#![forbid(unsafe_code)]

pub mod context;
pub mod raster;

pub use context::{block_on, GpuContext};
pub use raster::{GpuSoftwareRaster, RasterError};
