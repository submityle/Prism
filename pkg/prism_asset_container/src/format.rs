use core::fmt;

/// Eight-byte format discriminator.
pub const MAGIC: [u8; 8] = *b"PRISMCN\0";
/// Current on-disk format version. Incompatible versions are rejected.
pub const FORMAT_VERSION: u16 = 1;

pub(crate) const HEADER_SIZE: usize = 96;
pub(crate) const RECORD_SIZE: usize = 80;

/// Payload encoding identifier stored in each directory/chunk-table record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Codec {
    /// Stored bytes are the raw asset bytes (the only implemented codec).
    Raw = 0,
}

impl Codec {
    pub(crate) fn decode(value: u8) -> Result<Self, ContainerError> {
        match value {
            0 => Ok(Self::Raw),
            other => Err(ContainerError::UnsupportedCodec(other)),
        }
    }
}

/// Validated top-level container metadata.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ContainerInfo {
    /// Validated on-disk format version.
    pub version: u16,
    /// Number of entries in the table of contents.
    pub entry_count: u32,
    /// Exact container byte length.
    pub total_len: u64,
}

/// Public information for a single GUID-addressed chunk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EntryInfo {
    /// Stable identity of the asset.
    pub guid: prism_asset::StableGuid,
    /// Absolute byte offset of the stored payload.
    pub offset: u64,
    /// Number of bytes stored in the container.
    pub stored_len: u64,
    /// Number of decoded bytes; equal to `stored_len` for `Raw`.
    pub raw_len: u64,
    /// Payload encoding; only `Raw` is currently supported.
    pub codec: Codec,
    /// BLAKE3 digest of the payload bytes.
    pub hash: [u8; 32],
    /// Required absolute payload alignment in the container.
    pub alignment: u32,
}

/// Errors produced by strict container validation and reads.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ContainerError {
    /// Input ended before a complete declared section or payload was available.
    Truncated,
    /// The file does not begin with the Prism container discriminator.
    BadMagic,
    /// The container uses an unsupported format version.
    UnsupportedVersion(u16),
    /// Header fields or section layout are inconsistent.
    InvalidHeader,
    /// TOC fields are non-canonical or internally inconsistent.
    InvalidDirectory,
    /// An offset, length, or range cannot be represented safely.
    IntegerOverflow,
    /// A payload range escapes the declared payload section.
    OutOfBounds,
    /// A GUID appears more than once in one container.
    DuplicateGuid(prism_asset::StableGuid),
    /// A TOC entry names a codec not implemented by this reader.
    UnsupportedCodec(u8),
    /// A directory or complete-payload digest does not match its bytes.
    HashMismatch,
    /// No mounted container contains the requested asset.
    MissingAsset(prism_asset::StableGuid),
    /// The requested payload range exceeds the raw payload length.
    InvalidRange,
    /// A payload alignment is zero or not a power of two.
    InvalidAlignment,
    /// The caller-provided output buffer has an invalid length.
    LengthMismatch,
    /// The backing range source failed while serving a request.
    Source(String),
}

impl fmt::Display for ContainerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated => f.write_str("container is truncated"),
            Self::BadMagic => f.write_str("invalid container magic"),
            Self::UnsupportedVersion(v) => write!(f, "unsupported container version {v}"),
            Self::InvalidHeader => f.write_str("invalid container header"),
            Self::InvalidDirectory => f.write_str("invalid container directory"),
            Self::IntegerOverflow => f.write_str("container integer arithmetic overflow"),
            Self::OutOfBounds => f.write_str("container range is out of bounds"),
            Self::DuplicateGuid(g) => write!(f, "duplicate asset guid {g:?}"),
            Self::UnsupportedCodec(c) => write!(f, "unsupported payload codec {c}"),
            Self::HashMismatch => f.write_str("container or payload hash mismatch"),
            Self::MissingAsset(g) => write!(f, "asset {g:?} is not mounted"),
            Self::InvalidRange => f.write_str("requested asset range is invalid"),
            Self::InvalidAlignment => f.write_str("payload alignment is invalid"),
            Self::LengthMismatch => f.write_str("output buffer length does not match range"),
            Self::Source(error) => write!(f, "container range source failed: {error}"),
        }
    }
}

impl std::error::Error for ContainerError {}

pub(crate) fn read_u16(b: &[u8], at: usize) -> Result<u16, ContainerError> {
    let end = at.checked_add(2).ok_or(ContainerError::IntegerOverflow)?;
    Ok(u16::from_le_bytes(
        b.get(at..end)
            .ok_or(ContainerError::Truncated)?
            .try_into()
            .unwrap(),
    ))
}
pub(crate) fn read_u32(b: &[u8], at: usize) -> Result<u32, ContainerError> {
    let end = at.checked_add(4).ok_or(ContainerError::IntegerOverflow)?;
    Ok(u32::from_le_bytes(
        b.get(at..end)
            .ok_or(ContainerError::Truncated)?
            .try_into()
            .unwrap(),
    ))
}
pub(crate) fn read_u64(b: &[u8], at: usize) -> Result<u64, ContainerError> {
    let end = at.checked_add(8).ok_or(ContainerError::IntegerOverflow)?;
    Ok(u64::from_le_bytes(
        b.get(at..end)
            .ok_or(ContainerError::Truncated)?
            .try_into()
            .unwrap(),
    ))
}
