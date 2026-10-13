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

/// Inspect the current local revision and its published artifact descriptors.
#[derive(FromArgs)]
#[argh(subcommand, name = "inspect")]
pub struct Inspect {
    /// local revision store directory
    #[argh(option)]
    pub store: PathBuf,
}
