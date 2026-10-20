//! The protocol is checked at the process boundary, not by scraping human text.

use prism_asset_build_core::{
    BuildRevision, ContentHash, ReleaseSigner, ReleaseTrustStore, TrustPolicy,
};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::{SystemTime, UNIX_EPOCH};

const LOGICAL_PATH: &str = "textures/events.png";
const COMMANDS: [&str; 8] = [
    "plan", "import", "build", "package", "patch", "validate", "inspect", "why",
];

struct Fixture {
    root: PathBuf,
    source: PathBuf,
    store: PathBuf,
    mapping: PathBuf,
    package: PathBuf,
    patch: PathBuf,
}

impl Fixture {
    fn new(label: &str) -> Self {
        let root = std::env::temp_dir().join(format!(
            "prism-json-{label}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
        ));
        fs::create_dir_all(&root).unwrap();
        let fixture = Self {
            source: root.join("source.png"),
            store: root.join("store"),
            mapping: root.join("mapping.txt"),
            package: root.join("package"),
            patch: root.join("patch"),
            root,
        };
        fixture.write_png([1, 2, 3]);
        fs::write(&fixture.mapping, format!(
            "version=1\nasset=6;guid=sub:{LOGICAL_PATH}#source;container=core;priority=0;chunk_priority=0\nasset=7;guid=path:{LOGICAL_PATH};container=core;priority=0;chunk_priority=1\n"
        )).unwrap();
        fixture
    }

    fn write_png(&self, rgb: [u8; 3]) {
        let mut bytes = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut bytes, 1, 1);
            encoder.set_color(png::ColorType::Rgb);
            encoder.set_depth(png::BitDepth::Eight);
            let mut writer = encoder.write_header().unwrap();
            writer.write_image_data(&rgb).unwrap();
        }
        fs::write(&self.source, bytes).unwrap();
    }

    fn plan_args(&self) -> Vec<&str> {
        vec![
            "plan",
            "--source",
            text(&self.source),
            "--logical-path",
            LOGICAL_PATH,
            "--asset-id",
            "7",
        ]
    }

    fn publish_args<'a>(&'a self, command: &'a str) -> Vec<&'a str> {
        let mut args = self.plan_args();
        args[0] = command;
        args.extend(["--store", text(&self.store)]);
        args
    }

    fn why_args<'a>(&'a self, revision: &'a str) -> Vec<&'a str> {
        vec!["why", "--store", text(&self.store), "--revision", revision]
    }

    fn input_path(&self, revision: u64) -> PathBuf {
        let prefix = format!("{revision}.inputs-");
        let paths: Vec<_> = fs::read_dir(self.store.join("revisions"))
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter(|path| {
                path.file_name()
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .starts_with(&prefix)
            })
            .collect();
        assert_eq!(paths.len(), 1);
        paths[0].clone()
    }

    fn package_args(&self) -> Vec<&str> {
        vec![
            "package",
            "--store",
            text(&self.store),
            "--output",
            text(&self.package),
            "--mapping",
            text(&self.mapping),
        ]
    }

    fn validate_args(&self) -> Vec<&str> {
        vec![
            "validate",
            "--store",
            text(&self.store),
            "--package",
            text(&self.package),
            "--mapping",
            text(&self.mapping),
        ]
    }

    fn patch_args(&self) -> Vec<&str> {
        vec![
            "patch",
            "--store",
            text(&self.store),
            "--base-revision",
            "0",
            "--base-package",
            text(&self.package),
            "--base-mapping",
            text(&self.mapping),
            "--mapping",
            text(&self.mapping),
            "--output",
            text(&self.patch),
        ]
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

fn text(path: &Path) -> &str {
    path.to_str().unwrap()
}

fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_prism_asset_build_cli"))
        .args(args)
        .output()
        .unwrap()
}

fn run_json(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_prism_asset_build_cli"))
        .arg("--json")
        .args(args)
        .output()
        .unwrap()
}

fn terminal(output: &Output, command: Option<&str>, exit_code: i32) -> Value {
    assert_eq!(output.status.code(), Some(exit_code), "{output:?}");
    assert!(
        output.stderr.is_empty(),
        "JSON diagnostics escaped to stderr: {output:?}"
    );
    let stdout = std::str::from_utf8(&output.stdout).unwrap();
    assert!(stdout.ends_with('\n'));
    let events: Vec<Value> = stdout
        .lines()
        .map(|line| serde_json::from_str(line).expect("each stdout line is a JSON event"))
        .collect();
    assert_eq!(events.len(), 2, "{stdout}");
    let fields = BTreeSet::from([
        "schema_version",
        "sequence",
        "event",
        "command",
        "result",
        "error",
        "exit_code",
    ]);
    for (sequence, event) in events.iter().enumerate() {
        assert_eq!(
            event
                .as_object()
                .unwrap()
                .keys()
                .map(String::as_str)
                .collect::<BTreeSet<_>>(),
            fields
        );
        assert_eq!(event["schema_version"], 1);
        assert_eq!(event["sequence"], sequence);
        assert_eq!(event["command"], json!(command));
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
        assert_result_schema(command, &end["result"]);
    } else {
        assert_eq!(end["event"], "command_failed");
        assert!(end["result"].is_null());
        assert_eq!(
            end["error"]
                .as_object()
                .unwrap()
                .keys()
                .map(String::as_str)
                .collect::<BTreeSet<_>>(),
            BTreeSet::from(["code", "message"])
        );
        assert!(!end["error"]["message"].as_str().unwrap().is_empty());
    }
    end.clone()
}

fn assert_fields(value: &Value, fields: &[&str]) {
    assert_eq!(
        value
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect::<BTreeSet<_>>(),
        fields.iter().copied().collect::<BTreeSet<_>>()
    );
}

fn assert_result_schema(command: Option<&str>, result: &Value) {
    if result.get("help").is_some() {
        assert_fields(result, &["help"]);
        assert!(result["help"].is_string());
        return;
    }
    match command.unwrap() {
        "plan" => assert_fields(
            result,
            &[
                "source_path",
                "source_artifact_key",
                "image_artifact_key",
                "image_asset_id",
            ],
        ),
        "import" | "build" => assert_fields(
            result,
            &[
                "revision",
                "build_manifest_hash",
                "source_artifact_key",
                "image_artifact_key",
                "image_asset_id",
                "image_content_hash",
                "image_width",
                "image_height",
                "image_mip_levels",
                "cache_hits",
                "determinism_verified_jobs",
                "artifacts",
            ],
        ),
        "inspect" => assert_fields(
            result,
            &[
                "current_revision",
                "build_manifest_hash",
                "asset_count",
                "artifacts",
            ],
        ),
        "why" => {
            assert_fields(
                result,
                &["revision", "direct_jobs", "affected_jobs", "changes"],
            );
            for change in result["changes"].as_array().unwrap() {
                assert_fields(
                    change,
                    &[
                        "job_id",
                        "kind",
                        "before_path",
                        "before_hash",
                        "after_path",
                        "after_hash",
                        "profile_fields",
                    ],
                );
                for field in change["profile_fields"].as_array().unwrap() {
                    assert_fields(field, &["field", "before_hash", "after_hash"]);
                }
            }
        }
        "package" | "validate" => {
            assert_fields(
                result,
                &[
                    "revision",
                    "output_directory",
                    "build_manifest_hash",
                    "runtime_manifest_hash",
                    "runtime_manifest_bytes",
                    "container_count",
                    "containers",
                    "files",
                    "quality",
                    "signature",
                ],
            );
            assert_fields(
                &result["quality"],
                &[
                    "checked_assets",
                    "checked_artifacts",
                    "checked_blobs",
                    "checked_bytes",
                    "checked_dependency_edges",
                ],
            );
        }
        "patch" => assert_fields(
            result,
            &[
                "base_revision",
                "target_revision",
                "output_directory",
                "removed_routes",
                "patch_container_count",
                "patch_manifest_hash",
                "target_manifest_hash",
                "containers",
                "files",
            ],
        ),
        command => panic!("unexpected command {command}"),
    }
    if let Some(artifacts) = result.get("artifacts") {
        for artifact in artifacts.as_array().unwrap() {
            assert_fields(
                artifact,
                &[
                    "asset_id",
                    "artifact_key",
                    "kind",
                    "schema",
                    "bytes",
                    "content_hash",
                    "dependencies",
                ],
            );
        }
    }
    if let Some(files) = result.get("files") {
        for file in files.as_array().unwrap() {
            assert_fields(file, &["path", "bytes", "content_hash"]);
        }
    }
    if let Some(containers) = result.get("containers") {
        for container in containers.as_array().unwrap() {
            assert_fields(container, &["name", "bytes", "content_hash", "entries"]);
        }
    }
}

fn success(args: &[&str]) -> Value {
    terminal(&run_json(args), Some(args[0]), 0)["result"].clone()
}

fn failure(args: &[&str], code: &str) -> Value {
    let end = terminal(&run_json(args), Some(args[0]), 1);
    assert_eq!(end["error"]["code"], code);
    end
}

fn assert_hash(value: &Value) {
    let hash = value.as_str().unwrap();
    assert_eq!(hash.len(), 64);
    assert!(hash
        .bytes()
        .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)));
}

fn assert_files(result: &Value, directory: &Path) {
    assert_eq!(result["output_directory"], text(directory));
    let files = result["files"].as_array().unwrap();
    let names: Vec<_> = files
        .iter()
        .map(|file| file["path"].as_str().unwrap())
        .collect();
    assert!(!names.is_empty());
    assert!(names.windows(2).all(|pair| pair[0] < pair[1]));
    for file in files {
        let bytes = fs::read(directory.join(file["path"].as_str().unwrap())).unwrap();
        assert_eq!(file["bytes"], bytes.len());
        assert_eq!(file["content_hash"], ContentHash::digest(&bytes).to_hex());
    }
    let containers = result["containers"].as_array().unwrap();
    let names: Vec<_> = containers
        .iter()
        .map(|entry| entry["name"].as_str().unwrap())
        .collect();
    assert!(names.windows(2).all(|pair| pair[0] < pair[1]));
    for container in containers {
        let bytes = fs::read(
            directory
                .join("containers")
                .join(container["name"].as_str().unwrap()),
        )
        .unwrap();
        assert_eq!(container["bytes"], bytes.len());
        assert_eq!(
            container["content_hash"],
            ContentHash::digest(&bytes).to_hex()
        );
        assert!(container["entries"].as_u64().unwrap() > 0);
    }
}

#[test]
fn every_command_emits_structured_success_from_real_artifacts() {
    let fixture = Fixture::new("round-trip-雪");
    let first = run_json(&fixture.plan_args());
    let second = run_json(&fixture.plan_args());
    assert_eq!(
        first.stdout, second.stdout,
        "unchanged plans are byte-stable"
    );
    let planned = terminal(&first, Some("plan"), 0)["result"].clone();
    assert_eq!(planned["source_path"], LOGICAL_PATH);
    assert_eq!(planned["image_asset_id"], "7");
    assert_hash(&planned["source_artifact_key"]);
    assert_hash(&planned["image_artifact_key"]);

    let imported = success(&fixture.publish_args("import"));
    assert_eq!(imported["revision"], 0);
    assert_eq!(
        imported["image_artifact_key"],
        planned["image_artifact_key"]
    );
    assert_eq!(
        imported["source_artifact_key"],
        planned["source_artifact_key"]
    );
    assert_eq!(imported["image_asset_id"], "7");
    assert_eq!(imported["image_width"], 1);
    assert_eq!(imported["image_height"], 1);
    assert_eq!(imported["image_mip_levels"], 1);
    assert_eq!(imported["cache_hits"], 0);
    assert!(imported["determinism_verified_jobs"].is_null());
    assert_hash(&imported["build_manifest_hash"]);
    assert_hash(&imported["image_content_hash"]);
    let artifacts = imported["artifacts"].as_array().unwrap();
    assert_eq!(artifacts.len(), 2);
    assert_eq!(artifacts[0]["asset_id"], "6");
    assert_eq!(artifacts[1]["asset_id"], "7");
    for artifact in artifacts {
        assert!(artifact["kind"].as_str().unwrap().parse::<u128>().is_ok());
        assert!(artifact["schema"].is_u64());
        assert!(artifact["bytes"].as_u64().unwrap() > 0);
        assert_hash(&artifact["content_hash"]);
        assert_hash(&artifact["artifact_key"]);
    }
    assert_eq!(
        artifacts[1]["dependencies"],
        json!([planned["source_artifact_key"]])
    );

    let packaged = success(&fixture.package_args());
    assert_eq!(packaged["revision"], 0);
    assert_eq!(
        packaged["build_manifest_hash"],
        imported["build_manifest_hash"]
    );
    assert_eq!(packaged["container_count"], 1);
    assert_eq!(packaged["containers"][0]["entries"], 2);
    assert_eq!(packaged["signature"], "unsigned");
    assert_eq!(packaged["quality"]["checked_assets"], 2);
    assert_eq!(packaged["quality"]["checked_artifacts"], 2);
    assert_eq!(packaged["quality"]["checked_blobs"], 2);
    assert_eq!(packaged["quality"]["checked_dependency_edges"], 1);
    assert!(packaged["quality"]["checked_bytes"].as_u64().unwrap() > 0);
    assert_eq!(
        packaged["runtime_manifest_bytes"],
        fs::metadata(fixture.package.join("manifest.bin"))
            .unwrap()
            .len()
    );
    assert_files(&packaged, &fixture.package);
    assert_eq!(packaged, success(&fixture.validate_args()));

    let inspected = success(&["inspect", "--store", text(&fixture.store)]);
    assert_eq!(inspected["current_revision"], 0);
    assert_eq!(
        inspected["build_manifest_hash"],
        imported["build_manifest_hash"]
    );
    assert_eq!(inspected["asset_count"], 2);
    assert_eq!(inspected["artifacts"], imported["artifacts"]);

    fixture.write_png([9, 8, 7]);
    let mut args = fixture.publish_args("build");
    args.push("--verify-determinism");
    let built = success(&args);
    assert_eq!(built["revision"], 1);
    assert_eq!(built["determinism_verified_jobs"], 2);
    assert_eq!(built["cache_hits"], 0);
    assert_ne!(built["image_content_hash"], imported["image_content_hash"]);

    let first_patch = run_json(&fixture.patch_args());
    let patched = terminal(&first_patch, Some("patch"), 0)["result"].clone();
    assert_eq!(patched["base_revision"], 0);
    assert_eq!(patched["target_revision"], 1);
    assert_eq!(patched["removed_routes"], 0);
    assert_eq!(patched["patch_container_count"], 1);
    assert_hash(&patched["patch_manifest_hash"]);
    assert_eq!(
        patched["target_manifest_hash"],
        ContentHash::digest(&fs::read(fixture.patch.join("target_manifest.bin")).unwrap()).to_hex()
    );
    assert_files(&patched, &fixture.patch);
    let second_patch = run_json(&fixture.patch_args());
    terminal(&second_patch, Some("patch"), 0);
    assert_eq!(
        first_patch.stdout, second_patch.stdout,
        "idempotent patch events are byte-stable"
    );
}

#[test]
fn every_command_reports_operation_failure_and_nonzero_exit() {
    let fixture = Fixture::new("operation-errors");
    fs::remove_file(&fixture.source).unwrap();
    failure(&fixture.plan_args(), "io_error");
    fs::write(&fixture.source, b"not a PNG").unwrap();
    for command in ["import", "build"] {
        failure(&fixture.publish_args(command), "build_error");
    }
    failure(&fixture.package_args(), "invalid_input");
    failure(&fixture.validate_args(), "invalid_input");
    failure(&fixture.patch_args(), "invalid_input");
    let file = fixture.root.join("not-a-directory");
    fs::write(&file, b"file, not a revision store").unwrap();
    failure(&["inspect", "--store", text(&file)], "store_error");
}

#[test]
fn every_command_reports_argument_failure_with_exit_two() {
    for command in COMMANDS {
        let end = terminal(&run_json(&[command]), Some(command), 2);
        assert_eq!(end["error"]["code"], "invalid_arguments");
    }
    for args in [
        vec![],
        vec!["unknown-command"],
        vec!["--unknown"],
        vec!["--help", "--unknown"],
    ] {
        let end = terminal(&run_json(&args), None, 2);
        assert_eq!(end["error"]["code"], "invalid_arguments");
    }
    let fixture = Fixture::new("bad-argument");
    let mut args = fixture.plan_args();
    *args.last_mut().unwrap() = "not-a-u128";
    let end = terminal(&run_json(&args), Some("plan"), 2);
    assert_eq!(end["error"]["code"], "invalid_arguments");
    args = fixture.plan_args();
    args.push("--unknown");
    let end = terminal(&run_json(&args), Some("plan"), 2);
    assert_eq!(end["error"]["code"], "invalid_arguments");
}

#[test]
fn empty_inspection_and_cached_builds_have_fixed_nullable_fields() {
    let fixture = Fixture::new("empty-cache");
    assert_eq!(
        success(&["inspect", "--store", text(&fixture.store)]),
        json!({
            "current_revision": null, "build_manifest_hash": null, "asset_count": 0, "artifacts": []
        })
    );
    for command in ["import", "build"] {
        let mut verified = fixture.publish_args(command);
        verified.push("--verify-determinism");
        let fresh = success(&verified);
        assert_eq!(fresh["determinism_verified_jobs"], 2);
        assert_eq!(fresh["cache_hits"], 0);
        let cached = success(&fixture.publish_args(command));
        assert!(cached["determinism_verified_jobs"].is_null());
        assert_eq!(cached["cache_hits"], 2);
        assert_eq!(cached["artifacts"], fresh["artifacts"]);
    }
}

#[test]
fn tampering_is_a_terminal_failure_without_overwriting_artifacts() {
    let fixture = Fixture::new("tamper");
    success(&fixture.publish_args("build"));
    success(&fixture.package_args());
    let manifest = fixture.package.join("manifest.bin");
    fs::write(&manifest, b"tampered\n\"manifest\"").unwrap();
    failure(&fixture.validate_args(), "invalid_input");
    failure(&fixture.package_args(), "invalid_input");
    assert_eq!(fs::read(&manifest).unwrap(), b"tampered\n\"manifest\"");
}

#[test]
fn signing_reports_status_without_exposing_key_or_signature_material() {
    let fixture = Fixture::new("signing");
    success(&fixture.publish_args("build"));
    let seed = *b"DO-NOT-LOG-THIS-SECRET-SEED-1234";
    let key_path = fixture.root.join("signing.key");
    fs::write(&key_path, seed).unwrap();
    let signer = ReleaseSigner::from_seed(&seed).unwrap();
    let public_key = fixture.root.join("verify.key");
    fs::write(&public_key, signer.public_key().public_key).unwrap();
    let mut package_args = fixture.package_args();
    package_args.extend(["--signing-key-file", text(&key_path)]);
    let output = run_json(&package_args);
    let packaged = terminal(&output, Some("package"), 0)["result"].clone();
    assert_eq!(packaged["signature"], "signed");
    let stdout = std::str::from_utf8(&output.stdout).unwrap();
    assert!(!stdout.contains(std::str::from_utf8(&seed).unwrap()));
    assert!(!stdout.contains(
        &seed
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    ));
    assert!(!stdout.contains("\"public_key\""));
    assert!(!stdout.contains("\"signature_bytes\""));
    assert!(!stdout.contains("release.sig"));
    let signature_hex = fs::read(fixture.package.join("release.sig"))
        .unwrap()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    assert!(!stdout.contains(&signature_hex));
    assert_files(&packaged, &fixture.package);
    assert_eq!(
        success(&fixture.package_args())["signature"],
        "present_unverified"
    );
    failure(&fixture.validate_args(), "invalid_input");
    let mut validate_args = fixture.validate_args();
    validate_args.extend(["--verification-key-file", text(&public_key)]);
    assert_eq!(success(&validate_args)["signature"], "verified_public_key");

    let policy_path = fixture.root.join("release.trust");
    let mut policy = TrustPolicy::new(0, BuildRevision(0));
    policy.add_key(signer.public_key()).unwrap();
    ReleaseTrustStore::create(&policy_path, policy).unwrap();
    let mut trust_args = fixture.validate_args();
    trust_args.extend(["--trust-policy-file", text(&policy_path)]);
    assert_eq!(success(&trust_args)["signature"], "verified_trust_policy");
    fs::write(
        &public_key,
        ReleaseSigner::from_seed(&[8; 32])
            .unwrap()
            .public_key()
            .public_key,
    )
    .unwrap();
    failure(&validate_args, "invalid_input");
    fs::write(&key_path, &seed[..31]).unwrap();
    let failed = failure(&package_args, "invalid_input");
    assert!(!failed.to_string().contains("DO-NOT-LOG-THIS-SECRET"));
}

#[test]
fn json_escapes_strings_and_preserves_full_u128_ids() {
    let fixture = Fixture::new("escaping");
    let logical = "textures/雪\"quoted.png";
    let id = u128::MAX.to_string();
    let result = success(&[
        "plan",
        "--source",
        text(&fixture.source),
        "--logical-path",
        logical,
        "--asset-id",
        &id,
    ]);
    assert_eq!(result["source_path"], logical);
    assert_eq!(result["image_asset_id"], id);
    let end = terminal(
        &run_json(&[
            "plan",
            "--source",
            text(&fixture.source),
            "--logical-path",
            logical,
            "--asset-id",
            "not-a-number\n\"雪\\",
        ]),
        Some("plan"),
        2,
    );
    assert_eq!(end["error"]["code"], "invalid_arguments");
}

#[test]
fn help_is_structured_in_json_and_human_streams_remain_separate() {
    let help = terminal(&run_json(&["--help"]), None, 0);
    assert!(help["result"]["help"].as_str().unwrap().contains("--json"));
    for command in COMMANDS {
        let help = terminal(&run_json(&[command, "--help"]), Some(command), 0);
        assert!(help["result"]["help"].as_str().unwrap().contains(command));
    }
    let fixture = Fixture::new("human");
    let output = run(&fixture.plan_args());
    assert!(output.status.success());
    assert!(output.stderr.is_empty());
    assert!(std::str::from_utf8(&output.stdout)
        .unwrap()
        .starts_with("source_path="));
    let help = run(&["--help"]);
    assert!(help.status.success());
    assert!(help.stderr.is_empty());
    assert!(std::str::from_utf8(&help.stdout)
        .unwrap()
        .contains("Usage:"));
    fs::remove_file(&fixture.source).unwrap();
    let failed = run(&fixture.plan_args());
    assert_eq!(failed.status.code(), Some(1));
    assert!(failed.stdout.is_empty());
    assert!(std::str::from_utf8(&failed.stderr)
        .unwrap()
        .contains("prism asset build failed:"));
    let invalid = run(&["build"]);
    assert_eq!(invalid.status.code(), Some(2));
    assert!(invalid.stdout.is_empty());
    assert!(!invalid.stderr.is_empty());
}

#[test]
fn option_value_named_json_does_not_enable_json_output() {
    let fixture = Fixture::new("flag-value");
    let output = run(&[
        "plan",
        "--source",
        text(&fixture.source),
        "--logical-path",
        "--json",
        "--asset-id",
        "7",
    ]);
    assert!(output.status.success(), "{output:?}");
    assert!(output.stderr.is_empty());
    assert!(std::str::from_utf8(&output.stdout)
        .unwrap()
        .starts_with("source_path=--json\n"));
}

#[test]
fn top_level_option_terminator_keeps_the_command_identity_consistent() {
    let fixture = Fixture::new("option-terminator");
    let args = fixture.plan_args();
    let mut with_terminator = vec!["--"];
    with_terminator.extend(args);
    terminal(&run_json(&with_terminator), Some("plan"), 0);
    // After the terminator, --json is a positional token, not an output flag.
    let output = run(&["--", "--json"]);
    assert_eq!(output.status.code(), Some(2));
    assert!(output.stdout.is_empty());
    assert!(!output.stderr.is_empty());
}

#[cfg(unix)]
#[test]
fn invalid_utf8_arguments_still_emit_a_terminal_json_error() {
    use std::ffi::OsString;
    use std::os::unix::ffi::OsStringExt;

    let output = Command::new(env!("CARGO_BIN_EXE_prism_asset_build_cli"))
        .args(["--json", "inspect", "--store"])
        .arg(OsString::from_vec(vec![0xff, 0xfe]))
        .output()
        .unwrap();
    let end = terminal(&output, Some("inspect"), 2);
    assert_eq!(end["error"]["code"], "invalid_arguments");
    assert_eq!(
        end["error"]["message"],
        "command-line arguments must be valid UTF-8"
    );
}

// Include empty directories as well as every byte: why must not initialize a
// missing DDC, recover a prepared transaction, or truncate an interrupted tail.
fn store_tree(root: &Path) -> BTreeMap<PathBuf, Option<Vec<u8>>> {
    fn visit(root: &Path, path: &Path, tree: &mut BTreeMap<PathBuf, Option<Vec<u8>>>) {
        if path.is_dir() {
            tree.insert(path.strip_prefix(root).unwrap().to_owned(), None);
            for entry in fs::read_dir(path).unwrap() {
                visit(root, &entry.unwrap().path(), tree);
            }
        } else {
            tree.insert(
                path.strip_prefix(root).unwrap().to_owned(),
                Some(fs::read(path).unwrap()),
            );
        }
    }
    let mut tree = BTreeMap::new();
    if root.exists() {
        visit(root, root, &mut tree);
    }
    tree
}

#[test]
fn why_survives_process_reopen_reports_real_changes_and_preserves_failed_baseline() {
    let fixture = Fixture::new("why-lifecycle");
    let imported = success(&fixture.publish_args("import"));
    assert_eq!(imported["revision"], 0);
    assert_eq!(imported["cache_hits"], 0);
    let cold = success(&fixture.why_args("0"));
    assert_eq!(cold["direct_jobs"], json!(["1", "2"]));
    assert_eq!(cold["affected_jobs"], json!(["1", "2"]));
    assert_eq!(cold["changes"].as_array().unwrap().len(), 5);
    for change in cold["changes"].as_array().unwrap() {
        assert!(change["before_hash"].is_null());
        assert_hash(&change["after_hash"]);
        assert_eq!(change["profile_fields"], json!([]));
    }
    let source_change = &cold["changes"][0];
    assert_eq!(source_change["kind"], "source");
    assert_eq!(source_change["after_path"], LOGICAL_PATH);
    let initial_source_hash = ContentHash::digest(&fs::read(&fixture.source).unwrap()).to_hex();
    assert_eq!(source_change["after_hash"], initial_source_hash);
    let before_query = store_tree(&fixture.store);
    assert_eq!(success(&fixture.why_args("0")), cold);
    let text_result = run(&fixture.why_args("0"));
    assert!(text_result.status.success(), "{text_result:?}");
    assert!(text_result.stderr.is_empty());
    let human = String::from_utf8(text_result.stdout).unwrap();
    assert!(human.contains("revision=0\ndirect_jobs=1,2\naffected_jobs=1,2\n"));
    assert!(human.contains("change job=1 kind=source before_path=none before_hash=none"));
    assert!(human.contains(&initial_source_hash));
    assert_eq!(store_tree(&fixture.store), before_query);

    let warm = success(&fixture.publish_args("build"));
    assert_eq!(warm["revision"], 1);
    assert_eq!(warm["cache_hits"], 2);
    assert_eq!(
        success(&fixture.why_args("1")),
        json!({"revision": 1,
        "direct_jobs": [], "affected_jobs": [], "changes": []})
    );

    fixture.write_png([9, 8, 7]);
    let changed_source = fs::read(&fixture.source).unwrap();
    let changed = success(&fixture.publish_args("import"));
    assert_eq!(changed["revision"], 2);
    assert_eq!(changed["cache_hits"], 0);
    let changed_why = success(&fixture.why_args("2"));
    assert_eq!(
        changed_why,
        json!({"revision": 2, "direct_jobs": ["1"],
        "affected_jobs": ["1", "2"], "changes": [{
            "job_id": "1", "kind": "source", "before_path": LOGICAL_PATH,
            "before_hash": initial_source_hash, "after_path": LOGICAL_PATH,
            "after_hash": ContentHash::digest(&changed_source).to_hex(), "profile_fields": []
        }]})
    );
    assert_eq!(success(&fixture.why_args("0")), cold);
    // Every invocation is a fresh process. Neither a failed import nor its
    // partially populated DDC may replace revision 2's persisted baseline.
    let pointer = fs::read(fixture.store.join("CURRENT")).unwrap();
    fs::write(&fixture.source, b"invalid PNG after a valid build").unwrap();
    failure(&fixture.publish_args("build"), "build_error");
    assert_eq!(fs::read(fixture.store.join("CURRENT")).unwrap(), pointer);
    assert_eq!(success(&fixture.why_args("2")), changed_why);
    failure(&fixture.why_args("3"), "rebuild_explanation_error");
    fs::remove_file(&fixture.source).unwrap();
    assert_eq!(success(&fixture.why_args("2")), changed_why);
    fs::write(&fixture.source, changed_source).unwrap();
    let retry = success(&fixture.publish_args("build"));
    assert_eq!(retry["revision"], 3);
    assert_eq!(retry["cache_hits"], 2);
    assert_eq!(success(&fixture.why_args("3"))["changes"], json!([]));
}

#[test]
fn why_rejects_missing_corrupt_or_swapped_metadata_without_mutating_the_store() {
    let fixture = Fixture::new("why-metadata");
    success(&fixture.publish_args("build"));
    success(&fixture.publish_args("build"));
    let path = fixture.input_path(1);
    let bytes = fs::read(&path).unwrap();
    let initial = success(&fixture.why_args("1"));
    let pointer = fs::read(fixture.store.join("CURRENT")).unwrap();
    let mut corrupt = bytes.clone();
    *corrupt.last_mut().unwrap() ^= 1;
    for replacement in [
        Some(corrupt),
        Some(bytes[..bytes.len() / 2].to_vec()),
        Some(fs::read(fixture.input_path(0)).unwrap()),
        None,
    ] {
        if let Some(replacement) = replacement {
            fs::write(&path, replacement).unwrap();
        } else {
            fs::remove_file(&path).unwrap();
        }
        let before = store_tree(&fixture.store);
        failure(&fixture.why_args("1"), "rebuild_explanation_error");
        let human = run(&fixture.why_args("1"));
        assert_eq!(human.status.code(), Some(1));
        assert!(human.stdout.is_empty());
        assert!(!human.stderr.is_empty());
        assert_eq!(store_tree(&fixture.store), before);
        // Missing/corrupt evidence cannot silently become an empty baseline.
        failure(&fixture.publish_args("build"), "build_error");
        assert_eq!(fs::read(fixture.store.join("CURRENT")).unwrap(), pointer);
        assert_eq!(store_tree(&fixture.store), before);
        fs::write(&path, &bytes).unwrap();
        assert_eq!(success(&fixture.why_args("1")), initial);
    }
    let manifest = fixture.store.join("revisions/1.manifest");
    let original = fs::read(&manifest).unwrap();
    let mut altered = original.clone();
    *altered.last_mut().unwrap() ^= 1;
    fs::write(&manifest, altered).unwrap();
    let before = store_tree(&fixture.store);
    failure(&fixture.why_args("1"), "rebuild_explanation_error");
    assert_eq!(store_tree(&fixture.store), before);
    fs::remove_file(&manifest).unwrap();
    let before = store_tree(&fixture.store);
    failure(&fixture.why_args("1"), "rebuild_explanation_error");
    assert_eq!(store_tree(&fixture.store), before);
    fs::write(manifest, original).unwrap();
    assert_eq!(success(&fixture.why_args("1")), initial);
}

#[test]
fn why_requires_matching_prepare_and_commit_and_leaves_interrupted_journals_untouched() {
    let fixture = Fixture::new("why-journal");
    success(&fixture.publish_args("build"));
    let expected = success(&fixture.why_args("0"));
    let path = fixture.store.join("transactions/journal.bin");
    let journal = fs::read(&path).unwrap();
    let header = b"PRISM-TXN-JNL\0\x02".len();
    let frame = 8 + 99 + 32;
    assert_eq!(journal.len(), header + 2 * frame);
    let commit_payload = header + frame + 8;
    let mut corrupt_checksum = journal.clone();
    *corrupt_checksum.last_mut().unwrap() ^= 1;
    fs::write(&path, corrupt_checksum).unwrap();
    let before = store_tree(&fixture.store);
    failure(&fixture.why_args("0"), "rebuild_explanation_error");
    assert_eq!(store_tree(&fixture.store), before);
    // Recompute checksums so these exercise transaction binding, not merely
    // frame corruption. Test id, manifest, terminal kind and baseline.
    for (offset, value) in [
        (1, journal[commit_payload + 1] ^ 1),
        (34, journal[commit_payload + 34] ^ 1),
        (0, 3),
        (25, 1),
    ] {
        let mut altered = journal.clone();
        altered[commit_payload + offset] = value;
        let checksum = ContentHash::digest(&altered[commit_payload..commit_payload + 99]);
        altered[commit_payload + 99..commit_payload + 131].copy_from_slice(&checksum.0);
        fs::write(&path, altered).unwrap();
        let before = store_tree(&fixture.store);
        failure(&fixture.why_args("0"), "rebuild_explanation_error");
        assert_eq!(store_tree(&fixture.store), before);
    }
    // The manifest, CURRENT and metadata already exist, but Prepare alone
    // is never reported as committed and why must not perform recovery.
    fs::write(&path, &journal[..header + frame]).unwrap();
    let before = store_tree(&fixture.store);
    failure(&fixture.why_args("0"), "rebuild_explanation_error");
    assert_eq!(store_tree(&fixture.store), before);
    for end in [
        header + frame + 1,
        header + frame + 7,
        header + frame + 8,
        commit_payload + 98,
        journal.len() - 1,
    ] {
        fs::write(&path, &journal[..end]).unwrap();
        failure(&fixture.why_args("0"), "rebuild_explanation_error");
        assert_eq!(fs::read(&path).unwrap(), journal[..end]);
    }
    let mut interrupted = journal.clone();
    interrupted.extend_from_slice(b"TX");
    fs::write(&path, &interrupted).unwrap();
    assert_eq!(success(&fixture.why_args("0")), expected);
    assert_eq!(fs::read(&path).unwrap(), interrupted);
    fs::write(&path, journal).unwrap();
    assert_eq!(success(&fixture.why_args("0")), expected);
}

#[test]
fn why_is_read_only_for_missing_stores_missing_ddc_and_unbound_sidecars() {
    let fixture = Fixture::new("why-read-only");
    assert!(!fixture.store.exists());
    let before = store_tree(&fixture.root);
    failure(&fixture.why_args("0"), "rebuild_explanation_error");
    let human = run(&fixture.why_args("0"));
    assert_eq!(human.status.code(), Some(1));
    assert_eq!(store_tree(&fixture.root), before);
    assert!(!fixture.store.exists());

    // An unbound old sidecar cannot create commitment or initialize stores.
    let sidecar = fixture.store.join("ddc/rebuild-explanations/0.bin");
    fs::create_dir_all(sidecar.parent().unwrap()).unwrap();
    fs::write(&sidecar, b"unbound explanation").unwrap();
    let before = store_tree(&fixture.store);
    failure(&fixture.why_args("0"), "rebuild_explanation_error");
    assert_eq!(store_tree(&fixture.store), before);

    success(&fixture.publish_args("import"));
    let expected = success(&fixture.why_args("0"));
    fs::remove_dir_all(fixture.store.join("ddc")).unwrap();
    let before = store_tree(&fixture.store);
    assert_eq!(success(&fixture.why_args("0")), expected);
    failure(&fixture.why_args("99"), "rebuild_explanation_error");
    assert_eq!(store_tree(&fixture.store), before);
    assert!(!fixture.store.join("ddc").exists());
}
