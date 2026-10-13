use argh::FromArgs;
use std::path::PathBuf;

/// Top-level Prism asset build command.
#[derive(FromArgs)]
pub struct PrismAssetBuild {
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
    Validate(Validate),
    Inspect(Inspect),
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
}

/// Inspect the current local revision and its published artifact descriptors.
#[derive(FromArgs)]
#[argh(subcommand, name = "inspect")]
pub struct Inspect {
    /// local revision store directory
    #[argh(option)]
    pub store: PathBuf,
}
