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
    Import(Import),
    Build(Build),
    Package(Package),
    Patch(Patch),
    Validate(Validate),
    Inspect(Inspect),
    Why(Why),
}

impl Command {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Plan(_) => "plan",
            Self::Import(_) => "import",
            Self::Build(_) => "build",
            Self::Package(_) => "package",
            Self::Patch(_) => "patch",
            Self::Validate(_) => "validate",
            Self::Inspect(_) => "inspect",
            Self::Why(_) => "why",
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
