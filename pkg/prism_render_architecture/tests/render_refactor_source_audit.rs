//! Repository-level guard for executable unfinished markers in the render refactor.
//!
//! Semantic gaps such as an unimplemented production consumer must remain
//! visible in the audit documentation.  This test only rejects unfinished
//! implementation markers that can silently ship in the scoped source tree;
//! test-only fixtures are excluded because they may intentionally mention the
//! marker vocabulary while checking shader diagnostics.

use std::fs;
use std::path::{Path, PathBuf};

const FORBIDDEN_MARKERS: &[&str] = &["TODO", "FIXME", "XXX", "todo!", "unimplemented!"];
const FORBIDDEN_REMOVED_APIS: &[&str] = &[
    "FixedPageStreamingManager",
    "release_after(",
    "WATER_SPRAY_RESOURCE",
];

fn is_test_source(path: &Path) -> bool {
    if path
        .components()
        .any(|component| component.as_os_str() == "tests")
    {
        return true;
    }
    let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
        return false;
    };
    name == "tests.rs" || name.ends_with("_tests.rs")
}

fn collect_files(root: &Path, files: &mut Vec<PathBuf>) {
    let mut entries = fs::read_dir(root)
        .unwrap_or_else(|error| panic!("cannot read render-refactor audit scope {root:?}: {error}"))
        .collect::<Result<Vec<_>, _>>()
        .unwrap_or_else(|error| {
            panic!("cannot enumerate render-refactor audit scope {root:?}: {error}")
        });
    entries.sort_by_key(|entry| entry.path());

    for entry in entries {
        let path = entry.path();
        if path.is_dir() {
            collect_files(&path, files);
            continue;
        }
        let extension = path.extension().and_then(|extension| extension.to_str());
        let is_scoped_source = matches!(extension, Some("rs")) && !is_test_source(&path);
        let is_scoped_document = matches!(extension, Some("md"));
        if is_scoped_source || is_scoped_document {
            files.push(path);
        }
    }
}

#[test]
fn render_refactor_scope_has_no_executable_unfinished_markers() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let repo_root = manifest_dir
        .parent()
        .and_then(Path::parent)
        .expect("architecture crate must live under the repository pkg directory");

    let relative_scopes = [
        "pkg/prism_render_app/src",
        "pkg/prism_render_scene/src",
        "pkg/prism_render_architecture/src",
        "pkg/prism_render_gpudriven/src",
        "docs/refactoring-dev",
    ];
    let mut files = Vec::new();
    for relative_scope in relative_scopes {
        let scope = repo_root.join(relative_scope);
        assert!(
            scope.is_dir(),
            "missing render-refactor audit scope {scope:?}"
        );
        collect_files(&scope, &mut files);
    }

    let mut violations = Vec::new();
    for path in files {
        let source = fs::read_to_string(&path).unwrap_or_else(|error| {
            panic!("cannot read render-refactor audit file {path:?}: {error}")
        });
        for marker in FORBIDDEN_MARKERS {
            let mut search_from = 0;
            while let Some(relative) = source[search_from..].find(marker) {
                let offset = search_from + relative;
                let line = source[..offset]
                    .bytes()
                    .filter(|byte| *byte == b'\n')
                    .count()
                    + 1;
                violations.push(format!("{}:{line}: {marker}", path.display()));
                search_from = offset + marker.len();
            }
        }
    }

    assert!(
        violations.is_empty(),
        "render-refactor scope contains executable unfinished markers:\n{}",
        violations.join("\n")
    );
}

#[test]
fn render_refactor_scope_has_no_removed_unpublished_api_symbols() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let repo_root = manifest_dir
        .parent()
        .and_then(Path::parent)
        .expect("architecture crate must live under the repository pkg directory");

    let relative_scopes = [
        "pkg/prism_render_app/src",
        "pkg/prism_render_scene/src",
        "pkg/prism_render_architecture/src",
        "pkg/prism_render_gpudriven/src",
        "pkg/prism_render_driver/src",
        "os-backends/prism_render_driver_wgpu/src",
    ];
    let mut files = Vec::new();
    for relative_scope in relative_scopes {
        let scope = repo_root.join(relative_scope);
        assert!(scope.is_dir(), "missing removed-API audit scope {scope:?}");
        collect_files(&scope, &mut files);
    }

    let mut violations = Vec::new();
    for path in files {
        if path.extension().and_then(|extension| extension.to_str()) != Some("rs") {
            continue;
        }
        let source = fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("cannot read removed-API audit file {path:?}: {error}"));
        for marker in FORBIDDEN_REMOVED_APIS {
            let mut search_from = 0;
            while let Some(relative) = source[search_from..].find(marker) {
                let offset = search_from + relative;
                let line = source[..offset]
                    .bytes()
                    .filter(|byte| *byte == b'\n')
                    .count()
                    + 1;
                violations.push(format!("{}:{line}: {marker}", path.display()));
                search_from = offset + marker.len();
            }
        }
    }

    assert!(
        violations.is_empty(),
        "removed unpublished API symbols remain in render sources:\n{}",
        violations.join("\n")
    );
}
