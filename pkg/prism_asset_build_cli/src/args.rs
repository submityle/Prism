use argh::FromArgs;
use std::path::PathBuf;

/// Top-level Prism asset build command.
#[derive(FromArgs)]
pub struct PrismAssetBuild {
    /// emit versioned JSONL events instead of text (place before the command)
    #[argh(switch)]
    pub json: bool,
    /// operation to perform
    #[argh(subcommand)]
    pub command: Command,
}

/// Supported local build operations.
#[derive(FromArgs)]
#[argh(subcommand)]
pub enum Command {
    Plan(Plan),
    EditablePlan(EditablePlan),
    EditableBuild(EditableBuild),
    EditableDiff(EditableDiff),
    Import(Import),
    Build(Build),
    Package(Package),
    Patch(Patch),
    Validate(Validate),
    Inspect(Inspect),
    Why(Why),
    Graph(Graph),
    Diff(Diff),
    Cache(Cache),
}

impl Command {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Plan(_) => "plan",
            Self::EditablePlan(_) => "editable-plan",
            Self::EditableBuild(_) => "editable-build",
            Self::EditableDiff(_) => "editable-diff",
            Self::Import(_) => "import",
            Self::Build(_) => "build",
            Self::Package(_) => "package",
            Self::Patch(_) => "patch",
            Self::Validate(_) => "validate",
            Self::Inspect(_) => "inspect",
            Self::Why(_) => "why",
            Self::Graph(_) => "graph",
            Self::Diff(_) => "diff",
            Self::Cache(_) => "cache",
        }
    }
}

/// Derive and print deterministic artifact keys without publishing.
#[derive(FromArgs)]
#[argh(subcommand, name = "plan")]
pub struct Plan {
    /// source PNG path
    #[argh(option)]
    pub source: PathBuf,
    /// canonical project-relative source path used for identity
    #[argh(option)]
    pub logical_path: String,
    /// published image AssetId as a decimal u128
    #[argh(option)]
    pub asset_id: u128,
}

/// Plan the hard closure of saved .prism payloads without importing source.
#[derive(FromArgs)]
#[argh(subcommand, name = "editable-plan")]
pub struct EditablePlan {
    /// editable workspace directory containing saved .prism packages
    #[argh(option)]
    pub workspace: PathBuf,
    /// root AssetId as decimal u128; repeat for multiple roots
    #[argh(option)]
    pub root: Vec<u128>,
    /// aggregate encoded snapshot budget, positive and at most 536870912 bytes
    #[argh(option)]
    pub max_snapshot_bytes: Option<u64>,
    /// explicit 32-byte profile hash as 64 hexadecimal digits
    #[argh(option)]
    pub profile_hash: Option<String>,
    /// explicit UTF-8 toolchain fingerprint, at most 1 MiB
    #[argh(option)]
    pub toolchain_fingerprint: Option<String>,
    /// explicit UTF-8 environment fingerprint, at most 1 MiB
    #[argh(option)]
    pub environment_fingerprint: Option<String>,
}

/// Cook saved .prism payloads and atomically publish a build revision.
#[derive(FromArgs)]
#[argh(subcommand, name = "editable-build")]
pub struct EditableBuild {
    /// editable workspace directory containing saved .prism packages
    #[argh(option)]
    pub workspace: PathBuf,
    /// root AssetId as decimal u128; repeat for multiple roots
    #[argh(option)]
    pub root: Vec<u128>,
    /// local revision store directory
    #[argh(option)]
    pub store: PathBuf,
    /// aggregate encoded snapshot budget, positive and at most 536870912 bytes
    #[argh(option)]
    pub max_snapshot_bytes: Option<u64>,
    /// explicit 32-byte profile hash as 64 hexadecimal digits
    #[argh(option)]
    pub profile_hash: Option<String>,
    /// explicit UTF-8 toolchain fingerprint, at most 1 MiB
    #[argh(option)]
    pub toolchain_fingerprint: Option<String>,
    /// explicit UTF-8 environment fingerprint, at most 1 MiB
    #[argh(option)]
    pub environment_fingerprint: Option<String>,
    /// maximum simultaneous cooker jobs; must be positive
    #[argh(option, default = "1")]
    pub max_concurrency: usize,
    /// declared simultaneous CPU thread budget; must be positive
    #[argh(option, default = "1")]
    pub cpu_threads: u64,
    /// declared cooker working memory budget in bytes (default 256 MiB)
    #[argh(option, default = "268435456")]
    pub memory_bytes: u64,
    /// bypass DDC and compare two fresh executions per job before publication
    #[argh(switch)]
    pub verify_determinism: bool,
}

/// Compare two saved .prism snapshots without opening a workspace or store.
#[derive(FromArgs)]
#[argh(subcommand, name = "editable-diff")]
pub struct EditableDiff {
    /// saved .prism snapshot before the edit
    #[argh(option)]
    pub before: PathBuf,
    /// saved .prism snapshot after the edit
    #[argh(option)]
    pub after: PathBuf,
    /// per-file encoded byte budget, positive and at most 8388608 (default)
    #[argh(option)]
    pub max_input_bytes: Option<u64>,
}

/// Import a PNG and publish its canonical RGBA8 artifact.
#[derive(FromArgs)]
#[argh(subcommand, name = "import")]
pub struct Import {
    /// source PNG path
    #[argh(option)]
    pub source: PathBuf,
    /// canonical project-relative source path used for identity
    #[argh(option)]
    pub logical_path: String,
    /// published image AssetId as a decimal u128
    #[argh(option)]
    pub asset_id: u128,
    /// local revision store directory
    #[argh(option)]
    pub store: PathBuf,
    /// bypass DDC and compare two fresh executions per job before publication
    #[argh(switch)]
    pub verify_determinism: bool,
}

/// Build is the explicit publication spelling of the import vertical slice.
#[derive(FromArgs)]
#[argh(subcommand, name = "build")]
pub struct Build {
    /// source PNG path
    #[argh(option)]
    pub source: PathBuf,
    /// canonical project-relative source path used for identity
    #[argh(option)]
    pub logical_path: String,
    /// published image AssetId as a decimal u128
    #[argh(option)]
    pub asset_id: u128,
    /// local revision store directory
    #[argh(option)]
    pub store: PathBuf,
    /// bypass DDC and compare two fresh executions per job before publication
    #[argh(switch)]
    pub verify_determinism: bool,
}

/// Cook the current build revision into a deterministic runtime package.
#[derive(FromArgs)]
#[argh(subcommand, name = "package")]
pub struct Package {
    /// local revision store directory
    #[argh(option)]
    pub store: PathBuf,
    /// output directory for the runtime manifest and containers
    #[argh(option)]
    pub output: PathBuf,
    /// primary published image AssetId as a decimal u128
    #[argh(option)]
    pub asset_id: Option<u128>,
    /// canonical project-relative source path used for the primary runtime GUID
    #[argh(option)]
    pub logical_path: Option<String>,
    /// explicit versioned runtime mapping file
    #[argh(option)]
    pub mapping: Option<PathBuf>,
    /// saved .prism workspace; exclusive with mapping and legacy PNG identity
    #[argh(option)]
    pub workspace: Option<PathBuf>,
    /// editable root AssetId as decimal u128; requires --workspace; repeatable
    #[argh(option)]
    pub root: Vec<u128>,
    /// editable aggregate snapshot byte budget, positive and at most 536870912
    #[argh(option)]
    pub max_snapshot_bytes: Option<u64>,
    /// editable build profile hash as 64 hexadecimal digits
    #[argh(option)]
    pub profile_hash: Option<String>,
    /// explicit UTF-8 toolchain fingerprint used by the editable build
    #[argh(option)]
    pub toolchain_fingerprint: Option<String>,
    /// explicit UTF-8 environment fingerprint used by the editable build
    #[argh(option)]
    pub environment_fingerprint: Option<String>,
    /// raw 32-byte Ed25519 seed file; writes release.sig without logging key material
    #[argh(option)]
    pub signing_key_file: Option<PathBuf>,
}

/// Build an incremental runtime patch from a published base package to the
/// current target revision.
#[derive(FromArgs)]
#[argh(subcommand, name = "patch")]
pub struct Patch {
    /// target local revision store directory
    #[argh(option)]
    pub store: PathBuf,
    /// base committed build revision number
    #[argh(option)]
    pub base_revision: u64,
    /// previously emitted base runtime package directory
    #[argh(option)]
    pub base_package: PathBuf,
    /// explicit versioned mapping file for the base revision
    #[argh(option)]
    pub base_mapping: PathBuf,
    /// explicit versioned mapping file for the target revision
    #[argh(option)]
    pub mapping: PathBuf,
    /// output directory for the patch descriptor, manifests, and containers
    #[argh(option)]
    pub output: PathBuf,
}

/// Validate a previously emitted runtime package against the current revision.
#[derive(FromArgs)]
#[argh(subcommand, name = "validate")]
pub struct Validate {
    /// local revision store directory
    #[argh(option)]
    pub store: PathBuf,
    /// previously emitted runtime package directory
    #[argh(option)]
    pub package: PathBuf,
    /// primary published image AssetId as a decimal u128
    #[argh(option)]
    pub asset_id: Option<u128>,
    /// canonical project-relative source path used for the primary runtime GUID
    #[argh(option)]
    pub logical_path: Option<String>,
    /// explicit versioned runtime mapping file
    #[argh(option)]
    pub mapping: Option<PathBuf>,
    /// saved .prism workspace; exclusive with mapping and legacy PNG identity
    #[argh(option)]
    pub workspace: Option<PathBuf>,
    /// editable root AssetId as decimal u128; requires --workspace; repeatable
    #[argh(option)]
    pub root: Vec<u128>,
    /// editable aggregate snapshot byte budget, positive and at most 536870912
    #[argh(option)]
    pub max_snapshot_bytes: Option<u64>,
    /// editable build profile hash as 64 hexadecimal digits
    #[argh(option)]
    pub profile_hash: Option<String>,
    /// explicit UTF-8 toolchain fingerprint used by the editable build
    #[argh(option)]
    pub toolchain_fingerprint: Option<String>,
    /// explicit UTF-8 environment fingerprint used by the editable build
    #[argh(option)]
    pub environment_fingerprint: Option<String>,
    /// raw 32-byte Ed25519 public key file used to verify release.sig
    #[argh(option)]
    pub verification_key_file: Option<PathBuf>,
    /// persisted canonical ReleaseTrustStore file used to verify release.sig
    #[argh(option)]
    pub trust_policy_file: Option<PathBuf>,
}

/// Inspect the current local revision and its published artifact descriptors.
#[derive(FromArgs)]
#[argh(subcommand, name = "inspect")]
pub struct Inspect {
    /// local revision store directory
    #[argh(option)]
    pub store: PathBuf,
}

/// Explain the persisted invalidation plan for a published revision.
#[derive(FromArgs)]
#[argh(subcommand, name = "why")]
pub struct Why {
    /// local revision store directory
    #[argh(option)]
    pub store: PathBuf,
    /// committed build revision whose rebuild explanation should be read
    #[argh(option)]
    pub revision: u64,
}

/// Query a committed dependency graph (manifest <=64 MiB, journal <=256 MiB).
#[derive(FromArgs)]
#[argh(subcommand, name = "graph")]
pub struct Graph {
    /// existing local revision store directory (opened read-only)
    #[argh(option)]
    pub store: PathBuf,
    /// committed build revision
    #[argh(option)]
    pub revision: u64,
    /// select this decimal u128 AssetId and its transitive dependencies
    #[argh(option)]
    pub asset_id: Option<u128>,
    /// select transitive dependents instead; requires --asset-id
    #[argh(switch)]
    pub reverse: bool,
}

/// Compare committed descriptors (manifest <=64 MiB, journal <=256 MiB).
#[derive(FromArgs)]
#[argh(subcommand, name = "diff")]
pub struct Diff {
    /// existing local revision store directory (opened read-only)
    #[argh(option)]
    pub store: PathBuf,
    /// base committed build revision
    #[argh(option)]
    pub base_revision: u64,
    /// target committed revision; equal and older revisions are supported
    #[argh(option)]
    pub target_revision: u64,
}

/// Inspect, verify, or explicitly prune an existing local cache.
#[derive(FromArgs)]
#[argh(subcommand, name = "cache")]
pub struct Cache {
    #[argh(subcommand)]
    pub command: CacheCommand,
}

#[derive(FromArgs)]
#[argh(subcommand)]
pub enum CacheCommand {
    Status(CacheStatus),
    Verify(CacheVerify),
    Prune(CachePrune),
}

/// Report cache metadata without initialization or recovery.
#[derive(FromArgs)]
#[argh(subcommand, name = "status")]
pub struct CacheStatus {
    /// existing local revision store directory
    #[argh(option)]
    pub store: PathBuf,
    /// maximum admitted complete DDC index bytes (default 64 MiB)
    #[argh(option, default = "crate::cache::DEFAULT_MAX_INDEX_BYTES")]
    pub max_index_bytes: u64,
}

/// Verify DDC, all CAS blobs and current committed payloads without repair.
#[derive(FromArgs)]
#[argh(subcommand, name = "verify")]
pub struct CacheVerify {
    /// existing local revision store directory
    #[argh(option)]
    pub store: PathBuf,
    /// maximum admitted complete DDC index bytes (default 64 MiB)
    #[argh(option, default = "crate::cache::DEFAULT_MAX_INDEX_BYTES")]
    pub max_index_bytes: u64,
}

/// Evict DDC metadata to explicit bounds, then sweep unreachable CAS blobs.
#[derive(FromArgs)]
#[argh(subcommand, name = "prune")]
pub struct CachePrune {
    /// existing local revision store directory
    #[argh(option)]
    pub store: PathBuf,
    /// DDC entry capacity; at least one capacity bound is required
    #[argh(option)]
    pub max_entries: Option<usize>,
    /// complete encoded DDC index capacity, not CAS payload bytes
    #[argh(option)]
    pub max_encoded_bytes: Option<usize>,
    /// explicit minimum age in seconds for unreachable CAS blobs (zero allowed)
    #[argh(option)]
    pub grace_seconds: u64,
    /// maximum admitted complete DDC index bytes (default 64 MiB)
    #[argh(option, default = "crate::cache::DEFAULT_MAX_INDEX_BYTES")]
    pub max_index_bytes: u64,
}
