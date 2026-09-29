//! Virtual-shadow-map **page-table upload bridge** -- the allocation stage that
//! closes the paging loop between the GPU page-mark request pass and the
//! resolve / raster-fill consumers.
//!
//! The GPU page-mark pass produces a resident-window request bitmap but has no
//! in-crate reader; this module is that reader. [`readback`] copies the bitmap
//! back one frame late, [`decode`] turns it into the golden driver's request set
//! and back into the flat virtual->physical page table, [`pages`] resolves the
//! this-frame render pages, and [`resources`] owns the per-view golden driver and
//! the GPU page-table buffer cache. The attached per-view components feed the
//! resolve-pass sample integration and the physical-atlas raster fill pass
//! wired in later slices.

mod decode;
mod pages;
mod readback;
mod resources;

pub(crate) use readback::{
    collect_vsm_page_readback, map_submitted_vsm_page_readback, request_vsm_page_readback,
    VsmPageRequestReadback,
};
pub(crate) use resources::{VirtualShadowMapDriver, VsmPageTableBufferCache};
