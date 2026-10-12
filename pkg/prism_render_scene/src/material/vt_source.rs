//! CPU-side material virtual-texture source lifetime.
//!
//! Bevy's image render-asset path is allowed to move `Image::data` out of the
//! main-world asset when an image uses `RenderAssetUsages::RENDER_WORLD` only.
//! A render-world `GpuImage` therefore is not a source of CPU texels.  This
//! module is the boundary for the source lifetime: callers capture an image
//! while it is still available in the main world, and the registry retains an
//! immutable [`MaterialVtSourceSnapshot`] independent of the render asset.
//!
//! The capture path is CPU-only.  It validates the image's format, dimension,
//! complete mip chain, layer count, and byte length/order; it does not create
//! a GPU object, allocate an atlas slot, or upload anything.

use std::{fmt, sync::Arc};

use bevy_asset::{AssetId, Handle};
use bevy_ecs::prelude::Resource;
use bevy_image::Image;
use bevy_pbr::StandardMaterial;
use bevy_platform::collections::HashMap;
use bevy_render::render_resource::{TextureDataOrder, TextureDimension, TextureFormat};
use prism_render_material::{
    MaterialVtError, MaterialVtFormat, MaterialVtSourceDescriptor, MaterialVtSourceSnapshot,
    NO_VIRTUAL_TEXTURE,
};

/// Default logical page edge used by the material source/producer seam.
/// This does not imply that any page is resident on the GPU.
pub const DEFAULT_MATERIAL_VT_PAGE_SIZE: u32 = 128;

/// The configuration needed to turn one main-world [`Image`] into a material
/// VT source.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaterialVtSourceConfig {
    /// Stable VT identity written into page keys and material texture rows.
    pub texture_id: u32,
    /// Edge length of one virtual page in source texels.
    pub page_size: u32,
}

impl MaterialVtSourceConfig {
    /// Creates a source configuration.
    #[must_use]
    pub const fn new(texture_id: u32, page_size: u32) -> Self {
        Self {
            texture_id,
            page_size,
        }
    }
}

/// Why a main-world image could not become a material VT source.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaterialVtSourceError {
    /// The image asset was not present when its event was processed.
    MissingAsset,
    /// No non-reserved stable VT texture id remains.
    TextureIdExhausted,
    /// The image has already been extracted to the GPU or was created without
    /// CPU texels (`Image::data == None`).
    MissingData,
    /// The image format is not one of the uncompressed formats understood by
    /// the CPU page source contract.
    UnsupportedFormat {
        /// The Bevy/wgpu format that was supplied.
        format: TextureFormat,
    },
    /// Material VT sources are 2D images or 2D arrays, not 1D or 3D images.
    UnsupportedDimension {
        /// The image texture dimension.
        dimension: TextureDimension,
    },
    /// Material VT sources currently require a single-sample texture.
    UnsupportedSampleCount {
        /// The image sample count.
        samples: u32,
    },
    /// The image extent cannot be represented by the source contract.
    InvalidExtent {
        /// Width of mip zero.
        width: u32,
        /// Height of mip zero.
        height: u32,
        /// Number of array layers.
        layers: u32,
    },
    /// The array-layer count cannot be represented in the shared page key.
    LayerCountOverflow {
        /// Number of layers found in the image.
        layers: u32,
    },
    /// The image does not contain the complete pyramid implied by its largest
    /// base dimension.
    MipCountMismatch {
        /// Required complete mip count.
        expected: u32,
        /// Image descriptor mip count.
        actual: u32,
    },
    /// The supplied image bytes do not exactly match the validated mip layout.
    PayloadLengthMismatch {
        /// Required byte length.
        expected: u64,
        /// Supplied byte length.
        actual: u64,
    },
    /// A lower-level source descriptor or snapshot check failed.
    Snapshot(MaterialVtError),
}

impl fmt::Display for MaterialVtSourceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingAsset => formatter.write_str(
                "material VT source image asset was not present while processing its asset event",
            ),
            Self::TextureIdExhausted => {
                formatter.write_str("material VT source texture-id space is exhausted")
            }
            Self::MissingData => formatter.write_str(
                "material VT source image has no CPU data; capture it from the main world before GPU extraction",
            ),
            Self::UnsupportedFormat { format } => {
                write!(formatter, "unsupported material VT image format: {format:?}")
            }
            Self::UnsupportedDimension { dimension } => {
                write!(formatter, "unsupported material VT image dimension: {dimension:?}")
            }
            Self::UnsupportedSampleCount { samples } => {
                write!(formatter, "material VT image must be single-sampled, got {samples}")
            }
            Self::InvalidExtent {
                width,
                height,
                layers,
            } => write!(
                formatter,
                "invalid material VT image extent {width}x{height} with {layers} layers"
            ),
            Self::LayerCountOverflow { layers } => write!(
                formatter,
                "material VT image layer count {layers} exceeds the page-key range"
            ),
            Self::MipCountMismatch { expected, actual } => write!(
                formatter,
                "material VT image requires {expected} complete mips, got {actual}"
            ),
            Self::PayloadLengthMismatch { expected, actual } => write!(
                formatter,
                "material VT image payload length mismatch: expected {expected}, got {actual}"
            ),
            Self::Snapshot(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for MaterialVtSourceError {}

/// Captures an immutable CPU source from a main-world image.
///
/// This function must be called against the main-world [`Image`] during the
/// extract boundary, before Bevy's `GpuImage::take_gpu_data` is allowed to
/// move the pixels.  It intentionally does not inspect `RenderAssets<GpuImage>`
/// and does not assume that a render-world image still has `data`.
pub fn capture_material_vt_source(
    image: &Image,
    config: MaterialVtSourceConfig,
) -> Result<MaterialVtSourceSnapshot, MaterialVtSourceError> {
    let Some(data) = image.data.as_deref() else {
        return Err(MaterialVtSourceError::MissingData);
    };

    let descriptor = &image.texture_descriptor;
    let format =
        material_vt_format(descriptor.format).ok_or(MaterialVtSourceError::UnsupportedFormat {
            format: descriptor.format,
        })?;
    if descriptor.dimension != TextureDimension::D2 {
        return Err(MaterialVtSourceError::UnsupportedDimension {
            dimension: descriptor.dimension,
        });
    }
    if descriptor.sample_count != 1 {
        return Err(MaterialVtSourceError::UnsupportedSampleCount {
            samples: descriptor.sample_count,
        });
    }

    let width = descriptor.size.width;
    let height = descriptor.size.height;
    let layers = descriptor.size.depth_or_array_layers;
    if width == 0 || height == 0 || layers == 0 {
        return Err(MaterialVtSourceError::InvalidExtent {
            width,
            height,
            layers,
        });
    }
    let layers_u16 =
        u16::try_from(layers).map_err(|_| MaterialVtSourceError::LayerCountOverflow { layers })?;

    let expected_mip_count = width.max(height).ilog2() + 1;
    let actual_mip_count = descriptor.mip_level_count;
    if actual_mip_count != expected_mip_count {
        return Err(MaterialVtSourceError::MipCountMismatch {
            expected: expected_mip_count,
            actual: actual_mip_count,
        });
    }

    let layout = ImageMipLayout::new(
        width,
        height,
        layers,
        actual_mip_count,
        u64::from(format.bytes_per_texel()),
    )?;
    let actual_bytes = u64::try_from(data.len()).unwrap_or(u64::MAX);
    if actual_bytes != layout.total_bytes {
        return Err(MaterialVtSourceError::PayloadLengthMismatch {
            expected: layout.total_bytes,
            actual: actual_bytes,
        });
    }

    // The shared source snapshot is mip-major. Bevy/wgpu can provide either
    // order, so normalize LayerMajor here while the bytes are still owned by
    // this capture. MipMajor can be handed to the snapshot without a second
    // reordering pass, though the snapshot still owns its own immutable copy.
    let source_bytes = match image.data_order {
        TextureDataOrder::MipMajor => data.to_vec(),
        TextureDataOrder::LayerMajor => layout.to_mip_major(data),
    };

    let descriptor = MaterialVtSourceDescriptor::new(
        config.texture_id,
        width,
        height,
        layers_u16,
        config.page_size,
        u8::try_from(actual_mip_count).expect("a u32 image mip count is at most 32 here"),
        format,
    )
    .map_err(MaterialVtSourceError::Snapshot)?;
    MaterialVtSourceSnapshot::new(descriptor, Arc::<[u8]>::from(source_bytes))
        .map_err(MaterialVtSourceError::Snapshot)
}

/// Stable identity for one registry version of one image asset.
///
/// The asset id identifies the owning main-world image. The monotonically
/// increasing revision distinguishes an image modification from its previous
/// immutable snapshot, including a modification that becomes unavailable.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MaterialVtSourceIdentity {
    asset_id: AssetId<Image>,
    revision: u64,
}

impl MaterialVtSourceIdentity {
    /// The main-world image asset id.
    #[must_use]
    pub const fn asset_id(self) -> AssetId<Image> {
        self.asset_id
    }

    /// Registry revision of this source state.
    #[must_use]
    pub const fn revision(self) -> u64 {
        self.revision
    }
}

/// The current state retained for one image asset.
#[derive(Clone, Debug)]
pub enum MaterialVtSourceState {
    /// A valid immutable CPU source snapshot.
    Available(Arc<MaterialVtSourceSnapshot>),
    /// The asset exists but its current image cannot be used as a CPU source.
    Unavailable(MaterialVtSourceError),
}

impl MaterialVtSourceState {
    /// Borrows the immutable snapshot when the source is available.
    #[must_use]
    pub fn snapshot(&self) -> Option<&MaterialVtSourceSnapshot> {
        match self {
            Self::Available(snapshot) => Some(snapshot),
            Self::Unavailable(_) => None,
        }
    }

    /// Clones the immutable snapshot handle when the source is available.
    #[must_use]
    pub fn snapshot_handle(&self) -> Option<Arc<MaterialVtSourceSnapshot>> {
        match self {
            Self::Available(snapshot) => Some(Arc::clone(snapshot)),
            Self::Unavailable(_) => None,
        }
    }

    /// Returns the capture error when the source is unavailable.
    #[must_use]
    pub const fn error(&self) -> Option<&MaterialVtSourceError> {
        match self {
            Self::Available(_) => None,
            Self::Unavailable(error) => Some(error),
        }
    }
}

/// A registry entry, including the identity of the immutable source version.
#[derive(Clone, Debug)]
pub struct MaterialVtSourceEntry {
    identity: MaterialVtSourceIdentity,
    state: MaterialVtSourceState,
}

impl MaterialVtSourceEntry {
    /// The identity of this source version.
    #[must_use]
    pub const fn identity(&self) -> MaterialVtSourceIdentity {
        self.identity
    }

    /// The current available/unavailable state.
    #[must_use]
    pub const fn state(&self) -> &MaterialVtSourceState {
        &self.state
    }
}

/// A borrowed lookup result for one source asset.
#[derive(Clone, Copy, Debug)]
pub enum MaterialVtSourceLookup<'a> {
    /// No entry exists because the image asset was removed.
    Missing,
    /// The current image has a valid immutable source.
    Available {
        /// Version identity.
        identity: MaterialVtSourceIdentity,
        /// Immutable CPU source snapshot.
        snapshot: &'a MaterialVtSourceSnapshot,
    },
    /// The current image exists but cannot be used as a CPU source.
    Unavailable {
        /// Version identity.
        identity: MaterialVtSourceIdentity,
        /// Why capture failed.
        error: &'a MaterialVtSourceError,
    },
}

/// Availability of one image reference in the source registry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaterialVtSourceAvailability {
    /// No current registry entry exists for the image.
    Missing,
    /// The image has an immutable CPU snapshot.
    Available,
    /// The image has an explicit unavailable state.
    Unavailable,
}

/// Aggregate source state for the image references of one material.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MaterialVtSourceQuery {
    /// Number of image references inspected, including repeated references.
    pub referenced: u32,
    /// References with an immutable source snapshot.
    pub available: u32,
    /// References with an explicit unavailable state.
    pub unavailable: u32,
    /// References with no registry entry.
    pub missing: u32,
}

impl MaterialVtSourceQuery {
    /// Returns the aggregate availability class for the query.
    #[must_use]
    pub const fn availability(self) -> MaterialVtSourceAvailability {
        if self.missing != 0 {
            MaterialVtSourceAvailability::Missing
        } else if self.unavailable != 0 {
            MaterialVtSourceAvailability::Unavailable
        } else {
            MaterialVtSourceAvailability::Available
        }
    }
}

/// Render-world source availability cached for each material that has been
/// published. This is deliberately a CPU-source query only: it contains no
/// atlas slot, page-table entry, or residency claim.
#[derive(Resource, Debug, Default)]
pub struct MaterialVtMaterialSourceQueries {
    entries: HashMap<AssetId<StandardMaterial>, MaterialVtSourceQuery>,
}

impl MaterialVtMaterialSourceQueries {
    /// Replaces the latest source query for one material asset.
    pub fn update(&mut self, material: AssetId<StandardMaterial>, query: MaterialVtSourceQuery) {
        self.entries.insert(material, query);
    }

    /// Removes the query when the material leaves the render registry.
    pub fn remove(&mut self, material: AssetId<StandardMaterial>) {
        self.entries.remove(&material);
    }

    /// Returns the latest source query for a material.
    #[must_use]
    pub fn get(&self, material: AssetId<StandardMaterial>) -> Option<MaterialVtSourceQuery> {
        self.entries.get(&material).copied()
    }

    /// Number of material queries currently retained.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }
}

/// Main-world image source registry for material VT.
///
/// The registry is intentionally independent of `RenderAssets<GpuImage>`.
/// Parent systems should call [`Self::upsert_from_image`] from an image add or
/// modify event while holding the main-world image, and [`Self::remove`] for
/// an asset removal. A failed update replaces the old snapshot with an
/// explicit [`MaterialVtSourceState::Unavailable`] entry, so stale source
/// bytes are never mistaken for the modified asset.
#[derive(Resource, Debug)]
pub struct MaterialVtSourceRegistry {
    entries: HashMap<AssetId<Image>, MaterialVtSourceEntry>,
    texture_ids: HashMap<AssetId<Image>, u32>,
    next_revision: u64,
    next_texture_id: u32,
    initial_scan_complete: bool,
}

impl Default for MaterialVtSourceRegistry {
    fn default() -> Self {
        Self {
            entries: HashMap::default(),
            texture_ids: HashMap::default(),
            next_revision: 0,
            next_texture_id: 1,
            initial_scan_complete: false,
        }
    }
}

impl MaterialVtSourceRegistry {
    /// Iterates the current source entries in arbitrary map order.
    ///
    /// Consumers that derive GPU state must sort by the immutable descriptor
    /// identity before assigning slots; the registry deliberately does not
    /// make hash-map traversal order part of the source contract.
    pub(crate) fn entries(&self) -> impl Iterator<Item = &MaterialVtSourceEntry> {
        self.entries.values()
    }

    /// Inserts or replaces an image source and returns its new identity.
    ///
    /// `Image::data` is copied into an immutable snapshot on success. If the
    /// pixels have already been moved by the render-asset extractor, the entry
    /// is retained as unavailable with [`MaterialVtSourceError::MissingData`].
    pub fn upsert_from_image(
        &mut self,
        asset_id: AssetId<Image>,
        image: &Image,
        config: MaterialVtSourceConfig,
    ) -> MaterialVtSourceIdentity {
        let identity = self.next_identity(asset_id);
        let state = match capture_material_vt_source(image, config) {
            Ok(snapshot) => MaterialVtSourceState::Available(Arc::new(snapshot)),
            Err(error) => MaterialVtSourceState::Unavailable(error),
        };
        self.entries
            .insert(asset_id, MaterialVtSourceEntry { identity, state });
        identity
    }

    /// Assigns a collision-free, stable VT identity to an image asset.
    pub fn config_for_asset(
        &mut self,
        asset_id: AssetId<Image>,
    ) -> Result<MaterialVtSourceConfig, MaterialVtSourceError> {
        let texture_id = if let Some(texture_id) = self.texture_ids.get(&asset_id) {
            *texture_id
        } else {
            let texture_id = self.next_texture_id;
            if texture_id == NO_VIRTUAL_TEXTURE {
                return Err(MaterialVtSourceError::TextureIdExhausted);
            }
            self.next_texture_id = texture_id.saturating_add(1);
            self.texture_ids.insert(asset_id, texture_id);
            texture_id
        };
        Ok(MaterialVtSourceConfig::new(
            texture_id,
            DEFAULT_MATERIAL_VT_PAGE_SIZE,
        ))
    }

    /// Captures an image using its registry-assigned stable VT identity.
    pub fn upsert_asset_image(
        &mut self,
        asset_id: AssetId<Image>,
        image: &Image,
    ) -> MaterialVtSourceIdentity {
        let config = match self.config_for_asset(asset_id) {
            Ok(config) => config,
            Err(error) => return self.upsert_unavailable(asset_id, error),
        };
        self.upsert_from_image(asset_id, image, config)
    }

    /// Replaces the current image source with an explicit unavailable state.
    pub fn upsert_unavailable(
        &mut self,
        asset_id: AssetId<Image>,
        error: MaterialVtSourceError,
    ) -> MaterialVtSourceIdentity {
        let identity = self.next_identity(asset_id);
        self.entries.insert(
            asset_id,
            MaterialVtSourceEntry {
                identity,
                state: MaterialVtSourceState::Unavailable(error),
            },
        );
        identity
    }

    /// Inserts an already-captured immutable snapshot as a new source version.
    pub fn insert_snapshot(
        &mut self,
        asset_id: AssetId<Image>,
        snapshot: MaterialVtSourceSnapshot,
    ) -> MaterialVtSourceIdentity {
        let identity = self.next_identity(asset_id);
        self.entries.insert(
            asset_id,
            MaterialVtSourceEntry {
                identity,
                state: MaterialVtSourceState::Available(Arc::new(snapshot)),
            },
        );
        identity
    }

    /// Removes an image source after the owning main-world asset is removed.
    pub fn remove(&mut self, asset_id: AssetId<Image>) -> Option<MaterialVtSourceEntry> {
        self.entries.remove(&asset_id)
    }

    /// Whether the initial `Assets<Image>` scan has run.
    #[must_use]
    pub(crate) const fn initial_scan_complete(&self) -> bool {
        self.initial_scan_complete
    }

    /// Marks the initial `Assets<Image>` scan complete.
    pub(crate) fn mark_initial_scan_complete(&mut self) {
        self.initial_scan_complete = true;
    }

    /// Looks up the current source state for an image asset.
    #[must_use]
    pub fn lookup(&self, asset_id: AssetId<Image>) -> MaterialVtSourceLookup<'_> {
        let Some(entry) = self.entries.get(&asset_id) else {
            return MaterialVtSourceLookup::Missing;
        };
        match &entry.state {
            MaterialVtSourceState::Available(snapshot) => MaterialVtSourceLookup::Available {
                identity: entry.identity,
                snapshot,
            },
            MaterialVtSourceState::Unavailable(error) => MaterialVtSourceLookup::Unavailable {
                identity: entry.identity,
                error,
            },
        }
    }

    /// Queries image source state without claiming residency or mutating the
    /// registry. This is the seam consumed by material publication until a
    /// producer turns snapshots into atlas/page-table writes.
    #[must_use]
    pub fn query_images(
        &self,
        image_ids: impl IntoIterator<Item = AssetId<Image>>,
    ) -> MaterialVtSourceQuery {
        let mut query = MaterialVtSourceQuery::default();
        for image_id in image_ids {
            query.referenced = query.referenced.saturating_add(1);
            match self.lookup(image_id) {
                MaterialVtSourceLookup::Missing => {
                    query.missing = query.missing.saturating_add(1);
                }
                MaterialVtSourceLookup::Available { .. } => {
                    query.available = query.available.saturating_add(1);
                }
                MaterialVtSourceLookup::Unavailable { .. } => {
                    query.unavailable = query.unavailable.saturating_add(1);
                }
            }
        }
        query
    }

    /// Queries the exact image set consumed by the StandardMaterial bridge.
    ///
    /// This is the minimum real seam for a producer: it observes the current
    /// immutable source state, but it never allocates residency or changes a
    /// page table.
    #[must_use]
    pub fn query_material(&self, material: &StandardMaterial) -> MaterialVtSourceQuery {
        self.query_images(material_texture_ids(material))
    }

    /// Returns the complete registry entry, if one exists.
    #[must_use]
    pub fn entry(&self, asset_id: AssetId<Image>) -> Option<&MaterialVtSourceEntry> {
        self.entries.get(&asset_id)
    }

    /// Number of image ids currently represented, including unavailable ones.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the registry has no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    fn next_identity(&mut self, asset_id: AssetId<Image>) -> MaterialVtSourceIdentity {
        self.next_revision = self.next_revision.saturating_add(1);
        MaterialVtSourceIdentity {
            asset_id,
            revision: self.next_revision,
        }
    }
}

/// Enumerates the image handles that `import_standard_material` currently
/// lowers into material texture rows. Keeping this list shared with the source
/// query avoids silently querying a different set of images than the material
/// lifecycle actually publishes.
pub(crate) fn material_texture_ids(
    material: &StandardMaterial,
) -> impl Iterator<Item = AssetId<Image>> + '_ {
    [
        material.base_color_texture.as_ref().map(Handle::id),
        material.emissive_texture.as_ref().map(Handle::id),
        material.metallic_roughness_texture.as_ref().map(Handle::id),
        material.normal_map_texture.as_ref().map(Handle::id),
        material.occlusion_texture.as_ref().map(Handle::id),
    ]
    .into_iter()
    .flatten()
}

/// Maps the subset of Bevy/wgpu formats that the CPU VT source can preserve
/// without decoding or synthesizing data.
fn material_vt_format(format: TextureFormat) -> Option<MaterialVtFormat> {
    Some(match format {
        TextureFormat::R8Unorm => MaterialVtFormat::R8Unorm,
        TextureFormat::Rg8Unorm => MaterialVtFormat::Rg8Unorm,
        TextureFormat::Rgba8Unorm => MaterialVtFormat::Rgba8Unorm,
        TextureFormat::Rgba8UnormSrgb => MaterialVtFormat::Rgba8Srgb,
        TextureFormat::Rgba16Float => MaterialVtFormat::Rgba16Float,
        TextureFormat::Rgba32Float => MaterialVtFormat::Rgba32Float,
        _ => return None,
    })
}

struct ImageMipLayout {
    mip_major_offsets: Vec<u64>,
    layer_major_offsets: Vec<u64>,
    mip_strides: Vec<u64>,
    layer_strides: Vec<u64>,
    layers: u32,
    per_layer_bytes: u64,
    total_bytes: u64,
}

impl ImageMipLayout {
    fn new(
        width: u32,
        height: u32,
        layers: u32,
        mip_count: u32,
        bytes_per_texel: u64,
    ) -> Result<Self, MaterialVtSourceError> {
        let mut layer_major_offsets = Vec::with_capacity(usize::try_from(mip_count).unwrap_or(0));
        let mut mip_strides = Vec::with_capacity(usize::try_from(mip_count).unwrap_or(0));
        let mut layer_strides = Vec::with_capacity(usize::try_from(mip_count).unwrap_or(0));
        let mut per_layer_bytes = 0_u64;
        for mip in 0..mip_count {
            let mip_width = width.checked_shr(mip).unwrap_or(0).max(1);
            let mip_height = height.checked_shr(mip).unwrap_or(0).max(1);
            let row_bytes = u64::from(mip_width).checked_mul(bytes_per_texel).ok_or(
                MaterialVtSourceError::PayloadLengthMismatch {
                    expected: u64::MAX,
                    actual: 0,
                },
            )?;
            let layer_stride = row_bytes.checked_mul(u64::from(mip_height)).ok_or(
                MaterialVtSourceError::PayloadLengthMismatch {
                    expected: u64::MAX,
                    actual: 0,
                },
            )?;
            let mip_stride = layer_stride.checked_mul(u64::from(layers)).ok_or(
                MaterialVtSourceError::PayloadLengthMismatch {
                    expected: u64::MAX,
                    actual: 0,
                },
            )?;
            layer_major_offsets.push(per_layer_bytes);
            layer_strides.push(layer_stride);
            per_layer_bytes = per_layer_bytes.checked_add(layer_stride).ok_or(
                MaterialVtSourceError::PayloadLengthMismatch {
                    expected: u64::MAX,
                    actual: 0,
                },
            )?;
            mip_strides.push(mip_stride);
        }
        let total_bytes = per_layer_bytes.checked_mul(u64::from(layers)).ok_or(
            MaterialVtSourceError::PayloadLengthMismatch {
                expected: u64::MAX,
                actual: 0,
            },
        )?;

        let mut mip_major_offsets = Vec::with_capacity(usize::try_from(mip_count).unwrap_or(0));
        let mut offset = 0_u64;
        for stride in &mip_strides {
            mip_major_offsets.push(offset);
            offset = offset.checked_add(*stride).ok_or(
                MaterialVtSourceError::PayloadLengthMismatch {
                    expected: u64::MAX,
                    actual: 0,
                },
            )?;
        }

        Ok(Self {
            mip_major_offsets,
            layer_major_offsets,
            mip_strides,
            layer_strides,
            layers,
            per_layer_bytes,
            total_bytes,
        })
    }

    fn to_mip_major(&self, data: &[u8]) -> Vec<u8> {
        let mut output = vec![0_u8; data.len()];
        for (mip, (&mip_major_offset, &mip_stride)) in self
            .mip_major_offsets
            .iter()
            .zip(&self.mip_strides)
            .enumerate()
        {
            let layer_stride = self.layer_strides[mip];
            for layer in 0..u64::from(self.layers) {
                let source_start = layer * self.per_layer_bytes + self.layer_major_offsets[mip];
                let destination_start = mip_major_offset + layer * layer_stride;
                let source_start = usize::try_from(source_start).expect("validated source offset");
                let destination_start =
                    usize::try_from(destination_start).expect("validated destination offset");
                let layer_stride = usize::try_from(layer_stride).expect("validated layer stride");
                let _mip_stride = usize::try_from(mip_stride).expect("validated mip stride");
                output[destination_start..destination_start + layer_stride]
                    .copy_from_slice(&data[source_start..source_start + layer_stride]);
            }
        }
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_asset::{uuid::Uuid, Handle, RenderAssetUsages};
    use prism_render_material::NO_VIRTUAL_TEXTURE;

    fn image_id(value: u128) -> AssetId<Image> {
        Handle::<Image>::from(Uuid::from_u128(value)).id()
    }

    fn image_4x4(value: u8) -> Image {
        Image::new(
            bevy_render::render_resource::Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            vec![value; 4 * 4],
            TextureFormat::R8Unorm,
            RenderAssetUsages::default(),
        )
    }

    fn image_4x4_complete(value: u8) -> Image {
        let mut image = image_4x4(value);
        image.texture_descriptor.mip_level_count = 3;
        image.data = Some(vec![value; 16 + 4 + 1]);
        image
    }

    fn config() -> MaterialVtSourceConfig {
        MaterialVtSourceConfig::new(17, 2)
    }

    #[test]
    fn source_identity_is_stable_for_a_lookup_and_changes_on_modification() {
        let id = image_id(1);
        let mut registry = MaterialVtSourceRegistry::default();
        let first = registry.upsert_from_image(id, &image_4x4_complete(1), config());
        let first_snapshot = match registry.lookup(id) {
            MaterialVtSourceLookup::Available { identity, snapshot } => {
                assert_eq!(identity, first);
                assert_eq!(snapshot.bytes()[0], 1);
                Arc::clone(match registry.entry(id).unwrap().state() {
                    MaterialVtSourceState::Available(snapshot) => snapshot,
                    MaterialVtSourceState::Unavailable(_) => unreachable!(),
                })
            }
            MaterialVtSourceLookup::Missing | MaterialVtSourceLookup::Unavailable { .. } => {
                panic!("valid source should be available")
            }
        };

        let second = registry.upsert_from_image(id, &image_4x4_complete(2), config());
        assert_eq!(first.asset_id(), second.asset_id());
        assert!(second.revision() > first.revision());
        assert_eq!(first_snapshot.bytes()[0], 1, "old snapshot stays immutable");
        match registry.lookup(id) {
            MaterialVtSourceLookup::Available { identity, snapshot } => {
                assert_eq!(identity, second);
                assert_eq!(snapshot.bytes()[0], 2);
            }
            MaterialVtSourceLookup::Missing | MaterialVtSourceLookup::Unavailable { .. } => {
                panic!("modified valid source should be available")
            }
        }
    }

    #[test]
    fn modification_replaces_dimensions_and_payload_without_stale_bytes() {
        let id = image_id(2);
        let mut registry = MaterialVtSourceRegistry::default();
        registry.upsert_from_image(id, &image_4x4_complete(3), config());

        let mut modified = Image::new(
            bevy_render::render_resource::Extent3d {
                width: 2,
                height: 2,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            vec![9; 4],
            TextureFormat::R8Unorm,
            RenderAssetUsages::default(),
        );
        modified.texture_descriptor.mip_level_count = 2;
        modified.data = Some(vec![9; 4 + 1]);
        registry.upsert_from_image(id, &modified, config());

        match registry.lookup(id) {
            MaterialVtSourceLookup::Available { snapshot, .. } => {
                assert_eq!(snapshot.descriptor().width(), 2);
                assert_eq!(snapshot.bytes(), &[9, 9, 9, 9, 9]);
            }
            MaterialVtSourceLookup::Missing | MaterialVtSourceLookup::Unavailable { .. } => {
                panic!("modified valid source should be available")
            }
        }
    }

    #[test]
    fn removed_asset_is_missing_and_does_not_leave_a_snapshot() {
        let id = image_id(3);
        let mut registry = MaterialVtSourceRegistry::default();
        registry.upsert_from_image(id, &image_4x4_complete(1), config());
        assert!(registry.remove(id).is_some());
        assert!(matches!(
            registry.lookup(id),
            MaterialVtSourceLookup::Missing
        ));
        assert!(registry.remove(id).is_none());
        assert!(registry.is_empty());
    }

    #[test]
    fn unavailable_state_is_explicit_for_missing_data_format_dimension_and_mips() {
        let mut registry = MaterialVtSourceRegistry::default();

        let missing_id = image_id(4);
        let missing = Image::new_uninit(
            bevy_render::render_resource::Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            TextureFormat::R8Unorm,
            RenderAssetUsages::RENDER_WORLD,
        );
        registry.upsert_from_image(missing_id, &missing, config());
        assert!(matches!(
            registry.lookup(missing_id),
            MaterialVtSourceLookup::Unavailable {
                error: &MaterialVtSourceError::MissingData,
                ..
            }
        ));

        let format_id = image_id(5);
        let format = Image::new(
            bevy_render::render_resource::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            vec![0; 4],
            TextureFormat::R32Float,
            RenderAssetUsages::default(),
        );
        registry.upsert_from_image(format_id, &format, config());
        assert!(matches!(
            registry.lookup(format_id),
            MaterialVtSourceLookup::Unavailable {
                error: &MaterialVtSourceError::UnsupportedFormat { .. },
                ..
            }
        ));

        let dimension_id = image_id(6);
        let dimension = Image::new(
            bevy_render::render_resource::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            TextureDimension::D3,
            vec![0; 1],
            TextureFormat::R8Unorm,
            RenderAssetUsages::default(),
        );
        registry.upsert_from_image(dimension_id, &dimension, config());
        assert!(matches!(
            registry.lookup(dimension_id),
            MaterialVtSourceLookup::Unavailable {
                error: &MaterialVtSourceError::UnsupportedDimension { .. },
                ..
            }
        ));

        let mip_id = image_id(7);
        let mip = image_4x4(0);
        registry.upsert_from_image(mip_id, &mip, config());
        assert!(matches!(
            registry.lookup(mip_id),
            MaterialVtSourceLookup::Unavailable {
                error: &MaterialVtSourceError::MipCountMismatch {
                    expected: 3,
                    actual: 1
                },
                ..
            }
        ));
    }

    #[test]
    fn layer_major_images_are_normalized_to_the_snapshot_layout() {
        let id = image_id(8);
        let mut image = image_4x4_complete(0);
        image.texture_descriptor.size.depth_or_array_layers = 2;
        image.data = Some(vec![
            // layer 0: mip 0, mip 1, mip 2
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3,
            // layer 1: mip 0, mip 1, mip 2
            4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 6,
        ]);
        image.data_order = TextureDataOrder::LayerMajor;
        let snapshot = capture_material_vt_source(&image, config()).unwrap();
        assert_eq!(snapshot.bytes()[0..16], [1; 16]);
        assert_eq!(snapshot.bytes()[16..32], [4; 16]);
        assert_eq!(snapshot.bytes()[32..36], [2; 4]);
        assert_eq!(snapshot.bytes()[36..40], [5; 4]);
        assert_eq!(snapshot.bytes()[40], 3);
        assert_eq!(snapshot.bytes()[41], 6);
        let _ = id;
    }

    #[test]
    fn reserved_texture_id_and_zero_page_size_make_a_source_unavailable() {
        let image = image_4x4_complete(1);
        assert!(matches!(
            capture_material_vt_source(&image, MaterialVtSourceConfig::new(NO_VIRTUAL_TEXTURE, 2)),
            Err(MaterialVtSourceError::Snapshot(
                MaterialVtError::ReservedTextureId
            ))
        ));
        assert!(matches!(
            capture_material_vt_source(&image, MaterialVtSourceConfig::new(1, 0)),
            Err(MaterialVtSourceError::Snapshot(
                MaterialVtError::ZeroPageSize
            ))
        ));
    }
}
