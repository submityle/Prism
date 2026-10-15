//! Source importers that emit versioned, deterministic runtime artifacts.

use crate::job::{Processor, ProcessorError};
use crate::recipe::{ImportProcessor, ProcessorIdentity};
use std::sync::Arc;

mod image;

pub use image::{CanonicalPngProcessor, ImageImportError, IMAGE_ARTIFACT_KIND};
pub use prism_image::{
    ImageArtifact, ImageArtifactError, ImageMipArtifact, IMAGE_ARTIFACT_SCHEMA,
    IMAGE_MIP_ARTIFACT_SCHEMA,
};

/// Canonical extension/MIME metadata used to select one importer plugin.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ImportSelector {
    /// Optional normalized extension without a leading dot.
    pub extension: Option<String>,
    /// Optional normalized MIME type without parameters.
    pub mime: Option<String>,
}

impl ImportSelector {
    /// Creates a selector from optional extension and MIME values.
    pub fn new(extension: Option<&str>, mime: Option<&str>) -> Self {
        Self {
            extension: extension.and_then(normalize_extension),
            mime: mime.and_then(normalize_mime),
        }
    }

    /// Creates a selector from a logical path and optional MIME type.
    pub fn for_path(path: &str, mime: Option<&str>) -> Self {
        let extension = path
            .rsplit('/')
            .next()
            .and_then(|name| name.rsplit_once('.'))
            .map(|(_, extension)| extension);
        Self::new(extension, mime)
    }
}

/// Stable importer metadata registered by a typed plugin.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ImporterDescriptor {
    identity: ProcessorIdentity,
    extensions: Vec<String>,
    mime_types: Vec<String>,
    profile_fields: Vec<String>,
}

impl ImporterDescriptor {
    fn from_plugin<P: ImporterPlugin>() -> Result<Self, ImporterRegistryError> {
        let identity = P::identity();
        let extensions = normalize_many(P::EXTENSIONS, normalize_extension)
            .map_err(|reason| ImporterRegistryError::InvalidDescriptor { identity, reason })?;
        let mime_types = normalize_many(P::MIME_TYPES, normalize_mime)
            .map_err(|reason| ImporterRegistryError::InvalidDescriptor { identity, reason })?;
        let profile_fields = P::PROFILE_FIELDS
            .iter()
            .map(|field| (*field).to_string())
            .collect::<Vec<_>>();
        if extensions.is_empty() && mime_types.is_empty() {
            return Err(ImporterRegistryError::InvalidDescriptor {
                identity,
                reason: "plugin must declare an extension or MIME type".into(),
            });
        }
        if profile_fields.iter().any(|field| !valid_name(field)) {
            return Err(ImporterRegistryError::InvalidDescriptor {
                identity,
                reason: "profile dependency names must be non-empty and whitespace-free".into(),
            });
        }
        Ok(Self {
            identity,
            extensions,
            mime_types,
            profile_fields,
        })
    }

    /// Stable processor identity.
    pub fn identity(&self) -> ProcessorIdentity {
        self.identity
    }
    /// Supported normalized extensions.
    pub fn extensions(&self) -> &[String] {
        &self.extensions
    }
    /// Supported normalized MIME types.
    pub fn mime_types(&self) -> &[String] {
        &self.mime_types
    }
    /// Profile fields consumed by this plugin.
    pub fn profile_fields(&self) -> &[String] {
        &self.profile_fields
    }

    fn matches(&self, selector: &ImportSelector) -> bool {
        selector.extension.as_ref().is_none_or(|extension| {
            self.extensions
                .iter()
                .any(|candidate| candidate == extension)
        }) && selector
            .mime
            .as_ref()
            .is_none_or(|mime| self.mime_types.iter().any(|candidate| candidate == mime))
    }
}

/// Compile-time plugin contract used to create runtime registry entries.
pub trait ImporterPlugin: ImportProcessor + 'static {
    /// File extensions accepted by this plugin.
    const EXTENSIONS: &'static [&'static str];
    /// MIME types accepted by this plugin.
    const MIME_TYPES: &'static [&'static str];
    /// Effective-profile fields consumed by this plugin.
    const PROFILE_FIELDS: &'static [&'static str] = &[];
}

/// A selected importer handle that can be attached to a build job.
#[derive(Clone)]
pub struct ImporterHandle {
    descriptor: ImporterDescriptor,
    processor: Arc<dyn Processor>,
}

impl std::fmt::Debug for ImporterHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ImporterHandle")
            .field("identity", &self.descriptor.identity)
            .finish_non_exhaustive()
    }
}

impl ImporterHandle {
    /// Stable identity of the selected plugin.
    pub fn identity(&self) -> ProcessorIdentity {
        self.descriptor.identity()
    }
    /// Metadata of the selected plugin.
    pub fn descriptor(&self) -> &ImporterDescriptor {
        &self.descriptor
    }
}

impl Processor for ImporterHandle {
    fn process(&self, inputs: &[&[u8]]) -> Result<Vec<u8>, ProcessorError> {
        self.processor.process(inputs)
    }
}

struct RegisteredImporter {
    descriptor: ImporterDescriptor,
    processor: Arc<dyn Processor>,
}

/// Deterministic in-process importer registry.
#[derive(Default)]
pub struct ImporterRegistry {
    importers: Vec<RegisteredImporter>,
}

impl std::fmt::Debug for ImporterRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list()
            .entries(self.importers.iter().map(|entry| &entry.descriptor))
            .finish()
    }
}

impl ImporterRegistry {
    /// Creates an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a registry containing the built-in PNG importer.
    pub fn with_builtin() -> Result<Self, ImporterRegistryError> {
        let mut registry = Self::new();
        registry.register(CanonicalPngProcessor)?;
        Ok(registry)
    }

    /// Registers a typed plugin and normalizes its metadata.
    pub fn register<P>(&mut self, processor: P) -> Result<(), ImporterRegistryError>
    where
        P: ImporterPlugin,
    {
        let descriptor = ImporterDescriptor::from_plugin::<P>()?;
        if self
            .importers
            .iter()
            .any(|entry| entry.descriptor.identity == descriptor.identity)
        {
            return Err(ImporterRegistryError::DuplicateProcessorIdentity(
                descriptor.identity,
            ));
        }
        self.importers.push(RegisteredImporter {
            descriptor,
            processor: Arc::new(processor),
        });
        self.importers
            .sort_by_key(|entry| entry.descriptor.identity);
        Ok(())
    }

    /// Returns registered descriptors in stable identity order.
    pub fn descriptors(&self) -> impl Iterator<Item = &ImporterDescriptor> {
        self.importers.iter().map(|entry| &entry.descriptor)
    }

    /// Selects exactly one importer using all supplied selector dimensions.
    pub fn select(
        &self,
        selector: &ImportSelector,
    ) -> Result<ImporterHandle, ImporterSelectionError> {
        if selector.extension.is_none() && selector.mime.is_none() {
            return Err(ImporterSelectionError::MissingSelector);
        }
        let matches = self
            .importers
            .iter()
            .filter(|entry| entry.descriptor.matches(selector))
            .collect::<Vec<_>>();
        match matches.as_slice() {
            [] => Err(ImporterSelectionError::NoMatch {
                extension: selector.extension.clone(),
                mime: selector.mime.clone(),
            }),
            [entry] => Ok(ImporterHandle {
                descriptor: entry.descriptor.clone(),
                processor: Arc::clone(&entry.processor),
            }),
            many => Err(ImporterSelectionError::Ambiguous {
                extension: selector.extension.clone(),
                mime: selector.mime.clone(),
                candidates: many.iter().map(|entry| entry.descriptor.identity).collect(),
            }),
        }
    }

    /// Selects exactly one importer from a logical source path and optional MIME.
    pub fn select_for_path(
        &self,
        path: &str,
        mime: Option<&str>,
    ) -> Result<ImporterHandle, ImporterSelectionError> {
        self.select(&ImportSelector::for_path(path, mime))
    }
}

/// Registry construction failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ImporterRegistryError {
    /// A plugin descriptor contains invalid metadata.
    InvalidDescriptor {
        identity: ProcessorIdentity,
        reason: String,
    },
    /// A stable processor identity was registered more than once.
    DuplicateProcessorIdentity(ProcessorIdentity),
}

impl std::fmt::Display for ImporterRegistryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidDescriptor { identity, reason } => write!(
                f,
                "invalid importer descriptor for {:032x} v{}: {reason}",
                identity.id, identity.version
            ),
            Self::DuplicateProcessorIdentity(identity) => write!(
                f,
                "duplicate importer processor identity {:032x} v{}",
                identity.id, identity.version
            ),
        }
    }
}

impl std::error::Error for ImporterRegistryError {}

/// Selection failure with stable, actionable diagnostics.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ImporterSelectionError {
    /// Neither extension nor MIME was supplied.
    MissingSelector,
    /// No registered plugin accepts the selector.
    NoMatch {
        extension: Option<String>,
        mime: Option<String>,
    },
    /// More than one registered plugin accepts the selector.
    Ambiguous {
        extension: Option<String>,
        mime: Option<String>,
        candidates: Vec<ProcessorIdentity>,
    },
}

impl std::fmt::Display for ImporterSelectionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingSelector => f.write_str("importer selection requires an extension or MIME type"),
            Self::NoMatch { extension, mime } => {
                write!(f, "no importer matches extension={extension:?}, mime={mime:?}")
            }
            Self::Ambiguous {
                extension,
                mime,
                candidates,
            } => write!(
                f,
                "ambiguous importer selection for extension={extension:?}, mime={mime:?}: {candidates:?}"
            ),
        }
    }
}

impl std::error::Error for ImporterSelectionError {}

fn normalize_many(
    values: &[&str],
    normalize: fn(&str) -> Option<String>,
) -> Result<Vec<String>, String> {
    let mut normalized = values
        .iter()
        .map(|value| normalize(value).ok_or_else(|| format!("invalid matcher {value:?}")))
        .collect::<Result<Vec<_>, _>>()?;
    normalized.sort();
    normalized.dedup();
    Ok(normalized)
}

fn normalize_extension(value: &str) -> Option<String> {
    let value = value.trim().strip_prefix('.').unwrap_or(value.trim());
    (!value.is_empty()
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-')))
    .then(|| value.to_ascii_lowercase())
}

fn normalize_mime(value: &str) -> Option<String> {
    let value = value.split(';').next()?.trim().to_ascii_lowercase();
    let (kind, subtype) = value.split_once('/')?;
    (!kind.is_empty()
        && !subtype.is_empty()
        && !kind.contains(char::is_whitespace)
        && !subtype.contains(char::is_whitespace))
    .then_some(value)
}

fn valid_name(value: &str) -> bool {
    !value.is_empty()
        && !value
            .chars()
            .any(|character| character.is_whitespace() || character.is_control())
}
