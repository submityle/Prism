//! Command-line entry point for the local Prism asset-build vertical slice.
//!
//! The CLI deliberately delegates identity, graph execution, caching, and
//! publication to `prism_asset_build_core`; it contains no second build
//! implementation or format-specific cache logic.

mod args;
mod commands;
mod error;
mod mapping;
mod patch;
mod pipeline;

fn main() {
    if let Err(error) = commands::run(argh::from_env()) {
        eprintln!("prism asset build failed: {error}");
        std::process::exit(1);
    }
}
