//! Source-level regression gates for the Render Fabric migration.
//!
//! The old water/volume frontend and service-set type names were removed in a
//! prior migration slice. This test keeps them from reappearing as a second
//! authority while the remaining compatibility entry points are retired under
//! their own behavior gates.

use std::fs;
use std::path::{Path, PathBuf};

const FORBIDDEN_SYMBOLS: &[&str] = &[
    "SharedBaseServices",
    "ShadingFrontend",
    "LightingServiceCaps",
    "LightingServices",
    "PARTICLE_LIGHTING_SERVICES",
    "bound_base",
    "shared_base",
];

fn visit_sources(path: &Path, files: &mut Vec<PathBuf>) {
    let metadata = fs::metadata(path).unwrap_or_else(|error| {
        panic!(
            "Render Fabric legacy gate cannot inspect {}: {error}",
            path.display()
        )
    });
    if metadata.is_file() {
        if matches!(
            path.extension().and_then(|extension| extension.to_str()),
            Some("rs" | "wesl")
        ) {
            files.push(path.to_path_buf());
        }
        return;
    }
    for entry in fs::read_dir(path).unwrap_or_else(|error| {
        panic!(
            "Render Fabric legacy gate cannot read {}: {error}",
            path.display()
        )
    }) {
        let entry = entry.expect("Render Fabric legacy gate directory entry");
        visit_sources(&entry.path(), files);
    }
}

fn contains_identifier(source: &str, wanted: &str) -> bool {
    source
        .split(|character: char| !(character.is_ascii_alphanumeric() || character == '_'))
        .any(|identifier| identifier == wanted)
}

#[test]
fn removed_render_fabric_authorities_stay_absent() {
    let architecture = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let roots = [
        architecture.join("src/water"),
        architecture.join("src/volumetric"),
        architecture.join("src/particle"),
        architecture.join("src/render_fabric"),
        architecture.join("tests"),
        architecture.join("../prism_render_scene/src"),
        architecture.join("../prism_render_scene/tests"),
    ];
    let mut files = Vec::new();
    for root in roots {
        visit_sources(&root, &mut files);
    }

    for file in files {
        // The gate necessarily contains the forbidden spellings in its own
        // test data.  Exclude only this file; all production and other test
        // sources remain covered by the repository audit.
        if file.ends_with("render_fabric_legacy_gate.rs") {
            continue;
        }
        let source = fs::read_to_string(&file).unwrap_or_else(|error| {
            panic!(
                "Render Fabric legacy gate cannot read {}: {error}",
                file.display()
            )
        });
        for symbol in FORBIDDEN_SYMBOLS {
            assert!(
                !contains_identifier(&source, symbol),
                "removed Render Fabric symbol {symbol:?} reappeared in {}",
                file.display()
            );
        }
    }
}
