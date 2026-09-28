//! Slang shader toolchain for Prism.
//!
//! This crate is the shader-language layer of Prism's cross-platform strategy
//! (see `docs/prism_material_pipeline_slang_design_zh.md`). It owns three
//! responsibilities that used to be missing or hand-maintained:
//!
//! 1. [`toolchain`] — locate and describe the `slangc` compiler without
//!    hard-coding a path or committing the binary into the repository.
//! 2. [`compile`] — drive `slangc` to emit per-backend artifacts. WGSL is the
//!    primary product (consumed by `wgpu`); SPIR-V / Metal / C++ host targets
//!    are also expressible. The C++ host target is what makes
//!    "CPU golden reference = the shader itself" achievable instead of a
//!    hand-aligned second implementation.
//! 3. [`reflection`] + [`codegen`] — parse Slang reflection JSON into a stable
//!    ABI model and generate `#[repr(C)]` Rust bindings from it, so the GPU and
//!    CPU views of a material struct *cannot* drift.
//!
//! [`cache`] provides a content-addressed variant cache so specialization
//! permutations are only compiled once.
//!
//! The pure data-processing paths (reflection parsing, codegen, hashing,
//! cache keys) are fully exercisable without a GPU and without `slangc`
//! present, which is exactly what the sandboxed CI can verify. Only
//! [`compile`] and the `slangc -v` probe in [`toolchain`] need the binary,
//! and both degrade gracefully when it is absent.

pub mod cache;
pub mod codegen;
pub mod compile;
pub mod error;
pub mod hash;
pub mod reflection;
pub mod target;
pub mod toolchain;

pub use error::{SlangError, SlangResult};
pub use target::{ShaderStage, Target};

/// Semantic version of this toolchain's reflection-to-ABI contract.
///
/// Bump this when the *shape* of generated bindings changes (field ordering,
/// padding strategy, derive set). It is distinct from any per-material ABI
/// version, which is derived from reflection content via
/// [`reflection::AbiModel::abi_version`].
pub const TOOLCHAIN_ABI_CONTRACT: u32 = 1;
