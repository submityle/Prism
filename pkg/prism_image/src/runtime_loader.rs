//! Runtime loading of canonical image artifacts from validated asset containers.
//!
//! The build-side `AssetId -> StableGuid` mapping is resolved by the caller at
//! the package boundary. Runtime storage then assigns its own generational
//! [`prism_asset::AssetId<Image>`]. This loader owns only the runtime manifest
//! and mounted container snapshot; it does not depend on build services or CAS.

use core::fmt;

use prism_asset::{AssetId, AssetLoader, Assets, Handle, LoadContext, LoadError, StableGuid};
use prism_asset_container::{ManifestReadError, OverlaySet, RuntimeManifest};

use crate::{Image, ImageArtifact, ImageArtifactError, ImageError};

/// Failure to route, read, decode, or store a packaged runtime image.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RuntimeImageLoadError {
    /// The manifest route, mount, container identity, or payload read failed.
    Read(ManifestReadError),
    /// The raw container payload was not the supported canonical image format.
    Artifact(ImageArtifactError),
    /// The artifact could not be represented as a validated runtime image.
    Image(ImageError),
}

impl fmt::Display for RuntimeImageLoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Read(error) => write!(f, "runtime image read failed: {error}"),
            Self::Artifact(error) => write!(f, "runtime image artifact failed: {error}"),
            Self::Image(error) => write!(f, "runtime image validation failed: {error}"),
        }
    }
}

impl std::error::Error for RuntimeImageLoadError {}

impl From<ManifestReadError> for RuntimeImageLoadError {
    fn from(error: ManifestReadError) -> Self {
        Self::Read(error)
    }
}

impl From<ImageArtifactError> for RuntimeImageLoadError {
    fn from(error: ImageArtifactError) -> Self {
        Self::Artifact(error)
    }
}

impl From<ImageError> for RuntimeImageLoadError {
    fn from(error: ImageError) -> Self {
        Self::Image(error)
    }
}

/// Synchronous runtime image loader backed by an immutable manifest/overlay snapshot.
///
/// `load` verifies the manifest route and mounted container identity, reads the
/// complete payload (thereby checking its BLAKE3 digest), and decodes the
/// versioned canonical RGBA8 artifact. `load_into` additionally inserts the
/// validated image into the caller's typed runtime arena and returns its strong
/// handle. This is a real CPU asset load; GPU allocation/upload remains the
/// responsibility of the renderer's existing image preparation path.
#[derive(Clone, Debug)]
pub struct RuntimeImageLoader {
    manifest: RuntimeManifest,
    overlays: OverlaySet,
}

impl RuntimeImageLoader {
    /// Creates a loader from a runtime manifest and already validated mounts.
    /// Binding is checked by the manifest reader for each operation so that a
    /// remounted/replaced container fails closed.
    #[must_use]
    pub fn new(manifest: RuntimeManifest, overlays: OverlaySet) -> Self {
        Self { manifest, overlays }
    }

    /// Reads and decodes one image addressed by its stable runtime GUID.
    pub fn load(&self, guid: StableGuid) -> Result<Image, RuntimeImageLoadError> {
        let bytes = self.manifest.bind(&self.overlays)?.read_all(guid)?;
        ImageArtifact::decode(&bytes)?
            .to_image()
            .map_err(RuntimeImageLoadError::Image)
    }

    /// Loads an image and stores it in `assets`, minting a process-local typed
    /// [`AssetId<Image>`] and strong [`Handle<Image>`] through the asset kernel.
    pub fn load_into(
        &self,
        guid: StableGuid,
        assets: &mut Assets<Image>,
    ) -> Result<Handle<Image>, RuntimeImageLoadError> {
        Ok(assets.insert(self.load(guid)?))
    }

    /// Loads an image and returns both its runtime id and keeping handle.
    pub fn load_with_id(
        &self,
        guid: StableGuid,
        assets: &mut Assets<Image>,
    ) -> Result<(AssetId<Image>, Handle<Image>), RuntimeImageLoadError> {
        let handle = self.load_into(guid, assets)?;
        Ok((handle.id(), handle))
    }
}

impl AssetLoader for RuntimeImageLoader {
    type Asset = Image;

    fn extensions(&self) -> &'static [&'static str] {
        &["prismimg"]
    }

    fn load(&self, bytes: &[u8], _ctx: &mut LoadContext) -> Result<Self::Asset, LoadError> {
        ImageArtifact::decode(bytes)
            .map_err(|error| LoadError::Malformed(error.to_string()))?
            .to_image()
            .map_err(|error| LoadError::Malformed(error.to_string()))
    }
}
