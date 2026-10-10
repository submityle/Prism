//! `prism-material-codegen` — regenerate every product of the surface schema
//! (design doc §17.5, stage S2).
//!
//! The schema in `prism_material_schema::schema/surface.toml` is the single
//! declarative source of truth for the über-BSDF layout. This binary emits its
//! committed, generated twins:
//!
//! * `pkg/prism_render_scene/src/shaders/material_unpack.wesl` — the GPU decoder.
//! * `pkg/prism_render_material/src/surface_layout.rs` — the CPU layout facts.
//!
//! Usage:
//!
//! ```text
//! cargo run -p prism_material_schema --bin prism-material-codegen -- write
//! cargo run -p prism_material_schema --bin prism-material-codegen -- check
//! ```
//!
//! `write` rewrites both files from the schema. `check` regenerates in memory
//! and exits non-zero (listing the stale files) if either committed file drifts
//! from the schema — the hook a CI job and the `codegen_freshness` tests use.

#![expect(
    clippy::print_stdout,
    clippy::print_stderr,
    reason = "This is a developer/CI CLI; progress and errors go to stdout/stderr."
)]

use std::path::PathBuf;
use std::process::ExitCode;

use prism_material_schema::codegen::{emit_rust_layout, emit_wesl_unpack};
use prism_material_schema::surface_schema;

/// A generated product: its workspace-relative path and freshly emitted text.
struct Product {
    relative_path: &'static str,
    contents: String,
}

/// The workspace root, derived from this crate's manifest dir (`pkg/<crate>`).
fn workspace_root() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .parent()
        .and_then(|p| p.parent())
        .map(PathBuf::from)
        .expect("crate manifest should live under <workspace>/pkg/<crate>")
}

/// Emit every product from the current committed schema.
fn products() -> Vec<Product> {
    let schema = surface_schema();
    vec![
        Product {
            relative_path: "pkg/prism_render_scene/src/shaders/material_unpack.wesl",
            contents: emit_wesl_unpack(&schema),
        },
        Product {
            relative_path: "pkg/prism_render_material/src/surface_layout.rs",
            contents: emit_rust_layout(&schema),
        },
    ]
}

fn main() -> ExitCode {
    let mode = std::env::args().nth(1).unwrap_or_default();
    let root = workspace_root();
    let products = products();

    match mode.as_str() {
        "write" => {
            for p in &products {
                let path = root.join(p.relative_path);
                if let Err(e) = std::fs::write(&path, &p.contents) {
                    eprintln!("error: failed to write {}: {e}", path.display());
                    return ExitCode::FAILURE;
                }
                println!("wrote {}", p.relative_path);
            }
            ExitCode::SUCCESS
        }
        "check" => {
            let mut stale = Vec::new();
            for p in &products {
                let path = root.join(p.relative_path);
                match std::fs::read_to_string(&path) {
                    Ok(committed) if committed == p.contents => {}
                    Ok(_) => stale.push(p.relative_path),
                    Err(e) => {
                        eprintln!("error: failed to read {}: {e}", path.display());
                        return ExitCode::FAILURE;
                    }
                }
            }
            if stale.is_empty() {
                println!("codegen is up to date ({} products)", products.len());
                ExitCode::SUCCESS
            } else {
                eprintln!("error: generated files are stale; run `-- write`:");
                for s in stale {
                    eprintln!("  {s}");
                }
                ExitCode::FAILURE
            }
        }
        other => {
            eprintln!("usage: prism-material-codegen <write|check>");
            if !other.is_empty() {
                eprintln!("unknown mode: {other}");
            }
            ExitCode::FAILURE
        }
    }
}
