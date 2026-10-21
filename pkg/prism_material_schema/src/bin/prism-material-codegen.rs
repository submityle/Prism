//! `prism-material-codegen` — regenerate every product of the surface schema
//! (design doc §17.5, stage S2).
//!
//! The schema in `prism_material_schema::schema/surface.toml` is the single
//! declarative source of truth for the über-BSDF layout. This binary emits its
//! committed, generated twins:
//!
//! * `pkg/prism_render_material/src/shaders/material_unpack.wesl` — the GPU decoder.
//! * `pkg/prism_render_material/src/surface_layout.rs` — the CPU layout facts.
//! * `pkg/prism_material_schema/src/generated_contract.rs` — S3/S4
//!   specialization packing and CPU sample/golden field metadata.
//! * `pkg/prism_render_material/src/shaders/material_classification.wesl` — the
//!   schema-owned classification metadata block.
//!
//! Usage:
//!
//! ```text
//! cargo run -p prism_material_schema --bin prism-material-codegen -- write
//! cargo run -p prism_material_schema --bin prism-material-codegen -- write-rust
//! cargo run -p prism_material_schema --bin prism-material-codegen -- write-contract
//! cargo run -p prism_material_schema --bin prism-material-codegen -- check
//! ```
//!
//! `write` updates changed products from the schema. `check` regenerates in memory
//! and exits non-zero (listing the stale files) if any committed file drifts
//! from the schema — the hook a CI job and the `codegen_freshness` tests use.
//! `write-rust` rewrites only the Rust product, which is useful when the
//! material schema changes are intentionally scoped away from the render-scene
//! shader tree.
//! `write-contract` updates only the schema crate's generated CPU contract.

#![expect(
    clippy::print_stdout,
    clippy::print_stderr,
    reason = "This is a developer/CI CLI; progress and errors go to stdout/stderr."
)]

use std::path::PathBuf;
use std::process::ExitCode;

use prism_material_schema::codegen::{
    emit_material_classification, emit_rust_layout, emit_schema_contract, emit_wesl_unpack,
};
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
fn products(root: &std::path::Path) -> std::io::Result<Vec<Product>> {
    let schema = surface_schema();
    // Preserve the current algorithm on disk, including concurrent runtime edits.
    // A compiled-in template could overwrite them when an existing binary runs.
    let classification_template = std::fs::read_to_string(
        root.join("pkg/prism_render_material/src/shaders/material_classification.wesl"),
    )?;
    Ok(vec![
        Product {
            relative_path: "pkg/prism_render_material/src/shaders/material_unpack.wesl",
            contents: emit_wesl_unpack(&schema),
        },
        Product {
            relative_path: "pkg/prism_render_material/src/surface_layout.rs",
            contents: emit_rust_layout(&schema),
        },
        Product {
            relative_path: "pkg/prism_material_schema/src/generated_contract.rs",
            contents: emit_schema_contract(&schema),
        },
        Product {
            relative_path: "pkg/prism_render_material/src/shaders/material_classification.wesl",
            contents: emit_material_classification(&schema, &classification_template),
        },
    ])
}

fn main() -> ExitCode {
    let mode = std::env::args().nth(1).unwrap_or_default();
    let root = workspace_root();
    let products = match products(&root) {
        Ok(products) => products,
        Err(error) => {
            eprintln!("error: failed to read classification template: {error}");
            return ExitCode::FAILURE;
        }
    };

    match mode.as_str() {
        "write" | "write-rust" | "write-contract" => {
            for p in &products {
                let selected = match mode.as_str() {
                    "write-rust" => {
                        p.relative_path == "pkg/prism_render_material/src/surface_layout.rs"
                    }
                    "write-contract" => {
                        p.relative_path == "pkg/prism_material_schema/src/generated_contract.rs"
                    }
                    _ => true,
                };
                if !selected {
                    continue;
                }
                let path = root.join(p.relative_path);
                if std::fs::read_to_string(&path).is_ok_and(|current| current == p.contents) {
                    continue;
                }
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
            eprintln!("usage: prism-material-codegen <write|write-rust|write-contract|check>");
            if !other.is_empty() {
                eprintln!("unknown mode: {other}");
            }
            ExitCode::FAILURE
        }
    }
}
