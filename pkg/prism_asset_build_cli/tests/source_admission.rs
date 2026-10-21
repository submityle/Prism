//! Source admission and recorded SDK planning at the real CLI process boundary.

use prism_asset_build_core::import::{CanonicalPngProcessor, ImageMipArtifact};
use prism_asset_build_core::{AssetId, BuildRevision, ContentHash, RevisionStore};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const LOGICAL_PATH: &str = "textures/source-admission.png";

struct Fixture {
    root: tempfile::TempDir,
    source: PathBuf,
    store: PathBuf,
}

impl Fixture {
    fn new() -> Self {
        let root = tempfile::Builder::new()
            .prefix("prism-source-admission-")
            .tempdir()
            .unwrap();
        Self {
            source: root.path().join("source.png"),
            store: root.path().join("store"),
            root,
        }
    }

    fn run(&self, command: &str, logical_path: &str) -> Output {
        let mut process = Command::new(env!("CARGO_BIN_EXE_prism_asset_build_cli"));
        process
            .current_dir(self.root.path())
            .args(["--json", command, "--source"])
            .arg(&self.source)
            .args(["--logical-path", logical_path, "--asset-id", "7"]);
        if command != "plan" {
            process.arg("--store").arg(&self.store);
        }
        process.output().unwrap()
    }

    fn success(&self, command: &str, logical_path: &str) -> Value {
        terminal(&self.run(command, logical_path), command, 0)["result"].clone()
    }

    fn failure(&self, command: &str, code: &str) -> Value {
        let event = terminal(&self.run(command, LOGICAL_PATH), command, 1);
        assert_eq!(event["error"]["code"], code, "{command}: {event}");
        event["error"].clone()
    }

    fn children(&self) -> BTreeSet<OsString> {
        fs::read_dir(self.root.path())
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect()
    }
}

fn terminal(output: &Output, command: &str, exit_code: i32) -> Value {
    assert_eq!(
        output.status.code(),
        Some(exit_code),
        "{command}: {output:?}"
    );
    assert!(output.stderr.is_empty(), "{command}: {output:?}");
    let stdout = std::str::from_utf8(&output.stdout).unwrap();
    assert!(stdout.ends_with('\n'), "{command}: {stdout}");
    let events: Vec<Value> = stdout
        .lines()
        .map(|line| serde_json::from_str(line).expect("each line must be a JSON event"))
        .collect();
    assert_eq!(events.len(), 2, "{command}: {stdout}");
    for (sequence, event) in events.iter().enumerate() {
        assert_eq!(event["schema_version"], 1);
        assert_eq!(event["sequence"], sequence);
        assert_eq!(event["command"], command);
    }
    assert_eq!(events[0]["event"], "command_started");
    assert!(events[0]["result"].is_null());
    assert!(events[0]["error"].is_null());
    assert!(events[0]["exit_code"].is_null());
    let end = &events[1];
    assert_eq!(end["exit_code"], exit_code);
    if exit_code == 0 {
        assert_eq!(end["event"], "command_succeeded");
        assert!(end["result"].is_object());
        assert!(end["error"].is_null());
    } else {
        assert_eq!(end["event"], "command_failed");
        assert!(end["result"].is_null());
        assert!(!end["error"]["message"].as_str().unwrap().is_empty());
    }
    end.clone()
}

fn png(rgb: [u8; 3]) -> Vec<u8> {
    let mut bytes = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut bytes, 1, 1);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().unwrap();
        writer.write_image_data(&rgb).unwrap();
        writer.finish().unwrap();
    }
    bytes
}

// Use an encoder-produced, CRC-valid IHDR, but provide no IDAT or pixel bytes.
// The rejection must come from declared-size admission during SDK analysis.
fn header_only(width: u32, height: u32) -> Vec<u8> {
    let mut bytes = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut bytes, width, height);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        drop(encoder.write_header().unwrap());
    }
    bytes.truncate(33);
    assert_eq!(bytes.len(), 33);
    bytes
}

// Capture durable publication evidence, allowing unreferenced CAS candidates.
// Rejected analysis is not required to undo source admission into CAS.
fn publication_files(store: &Path) -> BTreeMap<PathBuf, Vec<u8>> {
    fn visit(root: &Path, directory: &Path, files: &mut BTreeMap<PathBuf, Vec<u8>>) {
        for entry in fs::read_dir(directory).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                visit(root, &path, files);
            } else {
                files.insert(
                    path.strip_prefix(root).unwrap().to_path_buf(),
                    fs::read(&path).unwrap(),
                );
            }
        }
    }
    let mut files = BTreeMap::new();
    files.insert(
        PathBuf::from("CURRENT"),
        fs::read(store.join("CURRENT")).unwrap(),
    );
    visit(store, &store.join("revisions"), &mut files);
    visit(store, &store.join("transactions"), &mut files);
    files
}

#[test]
fn sparse_oversized_source_rejects_before_persistent_store_initialization() {
    let fixture = Fixture::new();
    let source = fs::File::create(&fixture.source).unwrap();
    let oversized = CanonicalPngProcessor::MAX_SOURCE_BYTES + 1;
    // A sparse file exercises metadata admission without allocating its contents.
    source.set_len(oversized).unwrap();
    drop(source);
    let children = fixture.children();
    for command in ["plan", "import", "build"] {
        let error = fixture.failure(command, "invalid_input");
        let message = error["message"].as_str().unwrap();
        assert!(message.contains("regular file"), "{command}: {error}");
        assert!(
            message.contains(&CanonicalPngProcessor::MAX_SOURCE_BYTES.to_string()),
            "{command}: {error}"
        );
        assert!(!fixture.store.exists(), "{command} initialized a store");
        assert_eq!(fixture.children(), children, "{command} created files");
        assert_eq!(fs::metadata(&fixture.source).unwrap().len(), oversized);
    }
}

#[test]
fn directory_source_rejects_before_persistent_store_initialization() {
    let fixture = Fixture::new();
    fs::create_dir(&fixture.source).unwrap();
    let marker = fixture.source.join("keep.txt");
    fs::write(&marker, b"source directory must remain untouched").unwrap();
    let children = fixture.children();
    for command in ["plan", "import", "build"] {
        let error = fixture.failure(command, "invalid_input");
        assert!(
            error["message"].as_str().unwrap().contains("regular file"),
            "{command}: {error}"
        );
        assert!(!fixture.store.exists(), "{command} initialized a store");
        assert_eq!(fixture.children(), children);
        assert!(fixture.source.is_dir());
        assert_eq!(
            fs::read(&marker).unwrap(),
            b"source directory must remain untouched"
        );
        assert_eq!(fs::read_dir(&fixture.source).unwrap().count(), 1);
    }
}

#[test]
fn planned_keys_match_published_manifest_and_verified_cas_outputs() {
    // Explicit image/png selection also supports logical paths without extensions.
    for logical_path in [LOGICAL_PATH, "textures/extensionless"] {
        let fixture = Fixture::new();
        let mut previous_keys: Option<(Value, Value)> = None;
        for (revision, command, rgb, expected_hits) in [
            (0, "import", [1, 2, 3], 0),
            (1, "build", [1, 2, 3], 2),
            (2, "build", [8, 5, 13], 0),
        ] {
            let bytes = png(rgb);
            fs::write(&fixture.source, &bytes).unwrap();
            let before_plan = (revision != 0).then(|| publication_files(&fixture.store));
            let first = fixture.run("plan", logical_path);
            let second = fixture.run("plan", logical_path);
            let planned = terminal(&first, "plan", 0)["result"].clone();
            terminal(&second, "plan", 0);
            assert_eq!(first.stdout, second.stdout, "plan must be byte-stable");
            assert_eq!(planned["source_path"], logical_path);
            if let Some(before) = before_plan {
                assert_eq!(publication_files(&fixture.store), before);
            } else {
                assert!(
                    !fixture.store.exists(),
                    "plan initialized a persistent store"
                );
            }
            let keys = (
                planned["source_artifact_key"].clone(),
                planned["image_artifact_key"].clone(),
            );
            if let Some(previous) = &previous_keys {
                if revision == 1 {
                    assert_eq!(&keys, previous, "warm plan changed keys");
                } else {
                    assert_ne!(keys.0, previous.0, "source edit retained source key");
                    assert_ne!(keys.1, previous.1, "source edit retained image key");
                }
            }
            let published = fixture.success(command, logical_path);
            assert_eq!(published["revision"], revision);
            assert_eq!(published["cache_hits"], expected_hits);
            assert_eq!(published["source_artifact_key"], keys.0);
            assert_eq!(published["image_artifact_key"], keys.1);
            let store = RevisionStore::open(&fixture.store).unwrap();
            assert_eq!(
                store.current_revision().unwrap(),
                Some(BuildRevision(revision))
            );
            let manifest = store.manifest(BuildRevision(revision)).unwrap();
            assert_eq!(manifest.entries.len(), 2);
            let source = &manifest.entries[&AssetId(6)];
            let image = &manifest.entries[&AssetId(7)];
            assert_eq!(json!(source.key.to_hex()), keys.0);
            assert_eq!(json!(image.key.to_hex()), keys.1);
            assert_eq!(image.dependencies, vec![source.key]);
            assert!(source.dependencies.is_empty());
            assert_eq!(source.content_hash, ContentHash::digest(&bytes));
            assert_eq!(store.cas().read(source.content_hash).unwrap(), bytes);
            assert_eq!(
                published["build_manifest_hash"],
                ContentHash::digest(&manifest.canonical_bytes()).to_hex()
            );
            assert_eq!(published["image_content_hash"], image.content_hash.to_hex());
            let decoded =
                ImageMipArtifact::decode(&store.cas().read(image.content_hash).unwrap()).unwrap();
            assert_eq!(
                (decoded.width, decoded.height, decoded.levels.len()),
                (1, 1, 1)
            );
            assert_eq!(decoded.levels[0].rgba8, [rgb[0], rgb[1], rgb[2], 255]);
            assert_eq!(fs::read(&fixture.source).unwrap(), bytes);
            previous_keys = Some(keys);
        }
    }
}

#[test]
fn forged_oversized_ihdr_rejects_without_publishing_or_replacing_a_revision() {
    for warm in [false, true] {
        let fixture = Fixture::new();
        let baseline = if warm {
            fs::write(&fixture.source, png([1, 2, 3])).unwrap();
            assert_eq!(fixture.success("build", LOGICAL_PATH)["revision"], 0);
            Some(publication_files(&fixture.store))
        } else {
            None
        };
        for (width, height, expected) in [
            (
                CanonicalPngProcessor::MAX_DIMENSION + 1,
                1,
                "dimension limit",
            ),
            (8192, 8192, "decoded byte limit"),
            (4096, 4096, "mip byte"),
            (1548, 8128, "encoded byte limit"),
        ] {
            let bytes = header_only(width, height);
            fs::write(&fixture.source, &bytes).unwrap();
            for command in ["plan", "import", "build"] {
                // Analysis errors retain the CLI's build_error protocol code.
                let error = fixture.failure(command, "build_error");
                assert!(
                    error["message"].as_str().unwrap().contains(expected),
                    "{command}, warm={warm}, {width}x{height}: {error}"
                );
                assert_eq!(fs::read(&fixture.source).unwrap(), bytes);
                if let Some(baseline) = &baseline {
                    assert_eq!(
                        &publication_files(&fixture.store),
                        baseline,
                        "{command}, {width}x{height} changed durable publication"
                    );
                } else {
                    assert!(!fixture.store.join("CURRENT").exists());
                    let revisions = fixture.store.join("revisions");
                    if revisions.exists() {
                        assert_eq!(fs::read_dir(revisions).unwrap().count(), 0);
                    }
                }
            }
        }
    }
}
