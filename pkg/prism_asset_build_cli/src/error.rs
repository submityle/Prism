use prism_asset_build_core::CasError;
use std::fmt;

#[derive(Debug)]
pub enum CommandError {
    InvalidInput(String),
    Io(std::io::Error),
    Source(prism_asset_build_core::SourceError),
    Build(prism_asset_build_core::BuildExecutionError),
    Store(prism_asset_build_core::PublishError),
    Cas(CasError),
    Ddc(prism_asset_build_core::DdcError),
    Artifact(prism_asset_build_core::import::ImageArtifactError),
    Quality(prism_asset_build_core::ValidationError),
    Package(prism_asset_build_core::PackageError),
    Publication(&'static str),
    RebuildExplanation(prism_asset_build_core::RebuildExplanationError),
}

impl CommandError {
    /// Stable protocol codes; display strings remain diagnostic, not machine keys.
    pub fn code(&self) -> &'static str {
        match self {
            Self::InvalidInput(_) => "invalid_input",
            Self::Io(_) => "io_error",
            Self::Source(_) => "source_error",
            Self::Build(_) => "build_error",
            Self::Store(_) => "store_error",
            Self::Cas(_) => "cas_error",
            Self::Ddc(_) => "ddc_error",
            Self::Artifact(_) => "artifact_error",
            Self::Quality(_) => "quality_error",
            Self::Package(_) => "package_error",
            Self::Publication(_) => "publication_error",
            Self::RebuildExplanation(_) => "rebuild_explanation_error",
        }
    }
}

impl fmt::Display for CommandError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(message) => f.write_str(message),
            Self::Io(error) => write!(f, "I/O error: {error}"),
            Self::Source(error) => write!(f, "source error: {error}"),
            Self::Build(error) => write!(f, "{error}"),
            Self::Store(error) => write!(f, "store error: {error}"),
            Self::Cas(error) => write!(f, "CAS error: {error}"),
            Self::Ddc(error) => write!(f, "DDC error: {error}"),
            Self::Artifact(error) => write!(f, "artifact error: {error}"),
            Self::Quality(error) => write!(f, "quality gate failed: {error}"),
            Self::Package(error) => write!(f, "runtime packaging failed: {error}"),
            Self::Publication(error) => write!(f, "publication descriptor failed: {error}"),
            Self::RebuildExplanation(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for CommandError {}

impl From<std::io::Error> for CommandError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<prism_asset_build_core::SourceError> for CommandError {
    fn from(error: prism_asset_build_core::SourceError) -> Self {
        Self::Source(error)
    }
}

impl From<prism_asset_build_core::BuildExecutionError> for CommandError {
    fn from(error: prism_asset_build_core::BuildExecutionError) -> Self {
        Self::Build(error)
    }
}

impl From<prism_asset_build_core::PublishError> for CommandError {
    fn from(error: prism_asset_build_core::PublishError) -> Self {
        Self::Store(error)
    }
}

impl From<CasError> for CommandError {
    fn from(error: CasError) -> Self {
        Self::Cas(error)
    }
}

impl From<prism_asset_build_core::DdcError> for CommandError {
    fn from(error: prism_asset_build_core::DdcError) -> Self {
        Self::Ddc(error)
    }
}

impl From<prism_asset_build_core::import::ImageArtifactError> for CommandError {
    fn from(error: prism_asset_build_core::import::ImageArtifactError) -> Self {
        Self::Artifact(error)
    }
}

impl From<prism_asset_build_core::ValidationError> for CommandError {
    fn from(error: prism_asset_build_core::ValidationError) -> Self {
        Self::Quality(error)
    }
}

impl From<prism_asset_build_core::PackageError> for CommandError {
    fn from(error: prism_asset_build_core::PackageError) -> Self {
        Self::Package(error)
    }
}
