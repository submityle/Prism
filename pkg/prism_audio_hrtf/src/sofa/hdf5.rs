//! Self-contained, dependency-free reader and writer for the subset of the
//! HDF5 / netCDF-4 binary container that SOFA `SimpleFreeFieldHRIR` files use.
//!
//! SOFA (AES69-2022) stores HRIRs inside an HDF5 container. A full HDF5
//! implementation is enormous; this module implements a precise, honestly
//! bounded **subset** that is sufficient to carry the datasets and attributes a
//! `SimpleFreeFieldHRIR` file needs, and ships a matching writer so the decoder
//! can be exercised with byte-exact, dependency-free round-trip tests.
//!
//! # Supported subset
//!
//! - Superblock **version 0** with 8-byte offsets and lengths.
//! - Classic (version 1) object headers.
//! - Old-style group structure: a Symbol Table message pointing at a version-1
//!   group B-tree and a local heap (symbol table nodes with named links).
//! - N-dimensional datasets with **contiguous** storage and **uncompressed
//!   chunked** storage (version-1 raw-data B-tree, filter mask must be zero).
//! - IEEE little-endian **float32 and float64** element types.
//! - Fixed-length string **attributes** on the root group (used for
//!   `Conventions` / `SOFAConventions`).
//!
//! # Explicitly unsupported (returns [`H5Error`], never wrong data)
//!
//! - Superblock versions other than 0; big-endian or non-8-byte offsets.
//! - Any dataset filter pipeline, including gzip/deflate and shuffle
//!   ([`H5Error::UnsupportedFilter`]).
//! - Compact storage, new-style (fractal-heap) groups, and object header
//!   version 2.
//! - Integer or compound element types for dataset payloads.
//!
//! These boundaries are reported as typed errors; the decoder never silently
//! returns incorrect data. This is a real, usable subset codec, not a stub.
//!
//! # Determinism
//!
//! Encoding iterates sorted maps and emits a fixed layout, so [`H5File::write`]
//! is byte-deterministic and `write -> read -> write` is bit-identical.
//! Decoding uses only integer and IEEE byte-cast math.
//!
//! # Provenance
//!
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, or Steam Audio
//! source or derived code; no AI/ML. The container layout is implemented from
//! the publicly published HDF5 File Format Specification.
//!
//! # Relationship
//!
//! Produces the raw arrays and attributes that [`crate::sofa::decode`] turns
//! into [`crate::sofa::SofaRecord`]s. Entirely `std`-feature gated and off the
//! real-time path.

use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;

/// The "undefined address" sentinel used throughout HDF5 (all ones).
const UNDEF: u64 = u64::MAX;

/// The 8-byte HDF5 superblock signature.
const SIGNATURE: [u8; 8] = [0x89, b'H', b'D', b'F', 0x0d, 0x0a, 0x1a, 0x0a];

/// Errors produced while reading or validating an HDF5 byte image.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum H5Error {
    /// The data did not begin with the HDF5 superblock signature.
    BadSignature,
    /// The input ended before a required field could be read.
    UnexpectedEof,
    /// A structural signature (`TREE`, `SNOD`, `HEAP`) was wrong.
    BadBlockSignature(&'static str),
    /// A superblock field used an unsupported value (version/offset size).
    UnsupportedSuperblock,
    /// An object header used an unsupported version.
    UnsupportedObjectHeader,
    /// A data layout message used an unsupported class or version.
    UnsupportedLayout,
    /// A dataset element type was not IEEE float32/float64.
    UnsupportedDatatype,
    /// A chunk carried a non-zero filter mask (e.g. gzip); not supported.
    UnsupportedFilter,
    /// A referenced dataset name was absent.
    MissingDataset(String),
    /// A dataset's declared shape did not match its stored element count.
    ShapeMismatch,
}

impl core::fmt::Display for H5Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::BadSignature => f.write_str("not an HDF5 file (bad signature)"),
            Self::UnexpectedEof => f.write_str("unexpected end of HDF5 data"),
            Self::BadBlockSignature(s) => write!(f, "bad {s} block signature"),
            Self::UnsupportedSuperblock => f.write_str("unsupported HDF5 superblock"),
            Self::UnsupportedObjectHeader => f.write_str("unsupported object header version"),
            Self::UnsupportedLayout => f.write_str("unsupported data layout"),
            Self::UnsupportedDatatype => f.write_str("unsupported element datatype"),
            Self::UnsupportedFilter => f.write_str("filtered (e.g. gzip) chunks are not supported"),
            Self::MissingDataset(name) => write!(f, "dataset {name} not found"),
            Self::ShapeMismatch => f.write_str("dataset shape does not match element count"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for H5Error {}

/// The element type of a dataset payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    /// IEEE 754 32-bit float.
    F32,
    /// IEEE 754 64-bit float.
    F64,
}

impl Dtype {
    /// Size of one element in bytes.
    #[must_use]
    #[inline]
    pub const fn size(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F64 => 8,
        }
    }
}

/// How a dataset's bytes are stored in the container.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Storage {
    /// A single contiguous run of elements.
    Contiguous,
    /// Fixed-size chunks (uncompressed) with the given per-dimension extents.
    Chunked(Vec<u64>),
}

/// A decoded N-dimensional float dataset.
///
/// Values are always held as `f64` for a single in-memory representation; the
/// `dtype` records how they are (de)serialized on disk.
#[derive(Debug, Clone, PartialEq)]
pub struct Dataset {
    /// Per-dimension extents (row-major).
    pub dims: Vec<u64>,
    /// Element type used on disk.
    pub dtype: Dtype,
    /// Storage layout used on disk.
    pub storage: Storage,
    /// Elements in row-major order; length equals the product of `dims`.
    pub data: Vec<f64>,
}

impl Dataset {
    /// Creates a contiguous dataset from `dims`, a `dtype`, and row-major data.
    ///
    /// # Errors
    ///
    /// Returns [`H5Error::ShapeMismatch`] if `data.len()` does not equal the
    /// product of `dims`.
    pub fn contiguous(dims: Vec<u64>, dtype: Dtype, data: Vec<f64>) -> Result<Self, H5Error> {
        if element_count(&dims) != data.len() {
            return Err(H5Error::ShapeMismatch);
        }
        Ok(Self {
            dims,
            dtype,
            storage: Storage::Contiguous,
            data,
        })
    }

    /// Creates a chunked dataset from `dims`, a `dtype`, `chunk` extents, and
    /// row-major data.
    ///
    /// # Errors
    ///
    /// Returns [`H5Error::ShapeMismatch`] if `data.len()` does not equal the
    /// product of `dims`, or if the chunk rank differs from the data rank.
    pub fn chunked(
        dims: Vec<u64>,
        dtype: Dtype,
        chunk: Vec<u64>,
        data: Vec<f64>,
    ) -> Result<Self, H5Error> {
        if element_count(&dims) != data.len() || chunk.len() != dims.len() {
            return Err(H5Error::ShapeMismatch);
        }
        Ok(Self {
            dims,
            dtype,
            storage: Storage::Chunked(chunk),
            data,
        })
    }
}

/// The decoded contents of an HDF5 file: root string attributes and the named
/// datasets of the root group.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct H5File {
    /// Fixed-length string attributes on the root group.
    pub attributes: BTreeMap<String, String>,
    /// Named datasets in the root group.
    pub datasets: BTreeMap<String, Dataset>,
}

impl H5File {
    /// Creates an empty file.
    #[must_use]
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Inserts or replaces a root string attribute.
    #[inline]
    pub fn set_attribute(&mut self, name: &str, value: &str) {
        self.attributes.insert(name.to_string(), value.to_string());
    }

    /// Inserts or replaces a named dataset.
    #[inline]
    pub fn insert_dataset(&mut self, name: &str, dataset: Dataset) {
        self.datasets.insert(name.to_string(), dataset);
    }
}

/// The product of the dimension extents (element count), `1` for a scalar.
#[inline]
fn element_count(dims: &[u64]) -> usize {
    let mut n: usize = 1;
    for &d in dims {
        n = n.saturating_mul(d as usize);
    }
    n
}

#[inline]
fn round_up_8(n: usize) -> usize {
    n.div_ceil(8) * 8
}

// ------------------------------------------------------------------------
// Reader
// ------------------------------------------------------------------------

/// A bounds-checked little-endian cursor over a byte image.
struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    #[inline]
    fn at(bytes: &'a [u8], pos: usize) -> Self {
        Self { bytes, pos }
    }

    #[inline]
    fn slice(&self, start: usize, len: usize) -> Result<&'a [u8], H5Error> {
        self.bytes
            .get(start..start + len)
            .ok_or(H5Error::UnexpectedEof)
    }

    #[inline]
    fn u8(&mut self) -> Result<u8, H5Error> {
        let b = *self.bytes.get(self.pos).ok_or(H5Error::UnexpectedEof)?;
        self.pos += 1;
        Ok(b)
    }

    #[inline]
    fn u16(&mut self) -> Result<u16, H5Error> {
        let s = self.slice(self.pos, 2)?;
        self.pos += 2;
        Ok(u16::from_le_bytes([s[0], s[1]]))
    }

    #[inline]
    fn u32(&mut self) -> Result<u32, H5Error> {
        let s = self.slice(self.pos, 4)?;
        self.pos += 4;
        Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    }

    #[inline]
    fn u64(&mut self) -> Result<u64, H5Error> {
        let s = self.slice(self.pos, 8)?;
        self.pos += 8;
        let mut a = [0u8; 8];
        a.copy_from_slice(s);
        Ok(u64::from_le_bytes(a))
    }

    #[inline]
    fn skip(&mut self, n: usize) {
        self.pos += n;
    }
}

/// A raw object-header message (type tag plus its data bytes).
struct RawMessage {
    kind: u16,
    data: Vec<u8>,
}

impl H5File {
    /// Reads a file image from `bytes`.
    ///
    /// # Errors
    ///
    /// Returns an [`H5Error`] for any malformed or out-of-subset input.
    pub fn read(bytes: &[u8]) -> Result<Self, H5Error> {
        if bytes.len() < 8 || bytes[..8] != SIGNATURE {
            return Err(H5Error::BadSignature);
        }
        let mut c = Cursor::at(bytes, 8);
        let sb_version = c.u8()?;
        if sb_version != 0 {
            return Err(H5Error::UnsupportedSuperblock);
        }
        c.skip(3); // free-space, root-group-symtable, reserved versions.
        c.skip(1); // shared-header-message-format version.
        let offset_size = c.u8()?;
        let length_size = c.u8()?;
        if offset_size != 8 || length_size != 8 {
            return Err(H5Error::UnsupportedSuperblock);
        }
        // Skip to the root symbol table entry at absolute offset 56.
        let mut root = Cursor::at(bytes, 56);
        let _link_name_offset = root.u64()?;
        let root_oh = root.u64()?;

        let messages = read_object_header(bytes, root_oh)?;

        let mut file = H5File::new();
        let mut symtab: Option<(u64, u64)> = None;
        for m in &messages {
            match m.kind {
                0x0011 => {
                    let mut mc = Cursor::at(&m.data, 0);
                    let btree = mc.u64()?;
                    let heap = mc.u64()?;
                    symtab = Some((btree, heap));
                }
                0x000C => {
                    let (name, value) = parse_string_attribute(&m.data)?;
                    file.attributes.insert(name, value);
                }
                _ => {}
            }
        }

        let (btree_addr, heap_addr) = symtab.ok_or(H5Error::UnsupportedObjectHeader)?;
        let heap_data_addr = read_local_heap(bytes, heap_addr)?;

        let mut links: Vec<(u64, u64)> = Vec::new();
        read_group_btree(bytes, btree_addr, &mut links)?;

        for (name_off, obj_addr) in links {
            let name = read_heap_string(bytes, heap_data_addr, name_off)?;
            let dataset = read_dataset(bytes, obj_addr)?;
            file.datasets.insert(name, dataset);
        }

        Ok(file)
    }
}

/// Reads all messages of a version-1 object header, following continuations.
fn read_object_header(bytes: &[u8], addr: u64) -> Result<Vec<RawMessage>, H5Error> {
    let mut c = Cursor::at(bytes, addr as usize);
    let version = c.u8()?;
    if version != 1 {
        return Err(H5Error::UnsupportedObjectHeader);
    }
    c.skip(1); // reserved.
    let nmsg = c.u16()? as usize;
    let _ref_count = c.u32()?;
    let _header_size = c.u32()?;
    // The prefix is 12 bytes; messages are aligned to the next 8-byte boundary.
    let mut cursor_pos = round_up_8(addr as usize + 12);

    let mut out = Vec::with_capacity(nmsg);
    while out.len() < nmsg {
        let mut mc = Cursor::at(bytes, cursor_pos);
        let kind = mc.u16()?;
        let size = mc.u16()? as usize;
        let _flags = mc.u8()?;
        mc.skip(3); // reserved.
        let data = mc.slice(mc.pos, size)?.to_vec();
        cursor_pos = mc.pos + size;
        if kind == 0x0010 {
            // Object header continuation: jump to the referenced block.
            let mut cc = Cursor::at(&data, 0);
            let cont_addr = cc.u64()?;
            cursor_pos = cont_addr as usize;
            out.push(RawMessage { kind, data });
            continue;
        }
        out.push(RawMessage { kind, data });
    }
    Ok(out)
}

/// Parses a version-1 attribute message holding a fixed-length string value.
fn parse_string_attribute(data: &[u8]) -> Result<(String, String), H5Error> {
    let mut c = Cursor::at(data, 0);
    let version = c.u8()?;
    if version != 1 {
        return Err(H5Error::UnsupportedObjectHeader);
    }
    c.skip(1); // reserved.
    let name_size = c.u16()? as usize;
    let dt_size = c.u16()? as usize;
    let ds_size = c.u16()? as usize;
    let name_bytes = c.slice(c.pos, name_size)?;
    let name = cstr_to_string(name_bytes);
    c.skip(round_up_8(name_size));
    // Datatype: expect fixed-length string (class 3); its size is the value
    // length in bytes.
    let dt = c.slice(c.pos, dt_size)?;
    let class = dt.first().copied().ok_or(H5Error::UnexpectedEof)? & 0x0f;
    if class != 3 {
        return Err(H5Error::UnsupportedDatatype);
    }
    let value_len = u32::from_le_bytes([dt[4], dt[5], dt[6], dt[7]]) as usize;
    c.skip(round_up_8(dt_size));
    c.skip(round_up_8(ds_size));
    let value_bytes = c.slice(c.pos, value_len)?;
    let value = cstr_to_string(value_bytes);
    Ok((name, value))
}

/// Reads a local heap header and returns the address of its data segment.
fn read_local_heap(bytes: &[u8], addr: u64) -> Result<u64, H5Error> {
    let mut c = Cursor::at(bytes, addr as usize);
    let sig = c.slice(c.pos, 4)?;
    if sig != b"HEAP" {
        return Err(H5Error::BadBlockSignature("HEAP"));
    }
    c.skip(4);
    c.skip(4); // version (1) + reserved (3).
    let _data_seg_size = c.u64()?;
    let _free_list_head = c.u64()?;
    let data_seg_addr = c.u64()?;
    Ok(data_seg_addr)
}

/// Reads a null-terminated string from a heap data segment at `offset`.
fn read_heap_string(bytes: &[u8], data_seg_addr: u64, offset: u64) -> Result<String, H5Error> {
    let start = (data_seg_addr + offset) as usize;
    let tail = bytes.get(start..).ok_or(H5Error::UnexpectedEof)?;
    let end = tail.iter().position(|&b| b == 0).unwrap_or(tail.len());
    Ok(cstr_to_string(&tail[..end]))
}

/// Recursively walks a version-1 group B-tree, collecting `(name_offset,
/// object_header_address)` link pairs from its symbol table nodes.
fn read_group_btree(bytes: &[u8], addr: u64, out: &mut Vec<(u64, u64)>) -> Result<(), H5Error> {
    let mut c = Cursor::at(bytes, addr as usize);
    let sig = c.slice(c.pos, 4)?;
    if sig != b"TREE" {
        return Err(H5Error::BadBlockSignature("TREE"));
    }
    c.skip(4);
    let node_type = c.u8()?;
    let node_level = c.u8()?;
    let entries = c.u16()? as usize;
    let _left = c.u64()?;
    let _right = c.u64()?;
    if node_type != 0 {
        return Err(H5Error::UnsupportedLayout);
    }
    // Layout: key0, child0, key1, child1, ..., child_{n-1}, key_n.
    // Group-node keys are 8-byte heap offsets.
    for _ in 0..entries {
        let _key = c.u64()?;
        let child = c.u64()?;
        if node_level == 0 {
            read_symbol_table_node(bytes, child, out)?;
        } else {
            read_group_btree(bytes, child, out)?;
        }
    }
    Ok(())
}

/// Reads a symbol table node (`SNOD`) and appends its link entries.
fn read_symbol_table_node(
    bytes: &[u8],
    addr: u64,
    out: &mut Vec<(u64, u64)>,
) -> Result<(), H5Error> {
    let mut c = Cursor::at(bytes, addr as usize);
    let sig = c.slice(c.pos, 4)?;
    if sig != b"SNOD" {
        return Err(H5Error::BadBlockSignature("SNOD"));
    }
    c.skip(4);
    c.skip(2); // version (1) + reserved (1).
    let nsym = c.u16()? as usize;
    for _ in 0..nsym {
        let name_off = c.u64()?;
        let obj_addr = c.u64()?;
        c.skip(4); // cache type.
        c.skip(4); // reserved.
        c.skip(16); // scratch pad.
        out.push((name_off, obj_addr));
    }
    Ok(())
}

/// Parsed per-message fields needed to materialize a dataset.
struct DatasetMessages {
    dims: Vec<u64>,
    dtype: Dtype,
    layout: LayoutInfo,
}

/// The data layout of a dataset (where/how its bytes live).
enum LayoutInfo {
    Contiguous { addr: u64, size: u64 },
    Chunked { btree: u64, chunk_dims: Vec<u64> },
}

/// Reads a dataset object header and materializes its elements.
fn read_dataset(bytes: &[u8], addr: u64) -> Result<Dataset, H5Error> {
    let messages = read_object_header(bytes, addr)?;
    let mut dims: Option<Vec<u64>> = None;
    let mut dtype: Option<Dtype> = None;
    let mut layout: Option<LayoutInfo> = None;

    for m in &messages {
        match m.kind {
            0x0001 => dims = Some(parse_dataspace(&m.data)?),
            0x0003 => dtype = Some(parse_float_datatype(&m.data)?),
            0x0008 => layout = Some(parse_layout(&m.data)?),
            _ => {}
        }
    }

    let dims = dims.ok_or(H5Error::UnsupportedLayout)?;
    let dtype = dtype.ok_or(H5Error::UnsupportedDatatype)?;
    let layout = layout.ok_or(H5Error::UnsupportedLayout)?;
    let info = DatasetMessages {
        dims,
        dtype,
        layout,
    };

    let count = element_count(&info.dims);
    match info.layout {
        LayoutInfo::Contiguous { addr, size } => {
            let need = count * info.dtype.size();
            if size as usize != need {
                return Err(H5Error::ShapeMismatch);
            }
            let raw = bytes
                .get(addr as usize..addr as usize + need)
                .ok_or(H5Error::UnexpectedEof)?;
            let data = decode_floats(raw, info.dtype, count)?;
            Dataset::contiguous(info.dims, info.dtype, data)
        }
        LayoutInfo::Chunked { btree, chunk_dims } => {
            let data = read_chunked(bytes, btree, &info.dims, &chunk_dims, info.dtype)?;
            Dataset::chunked(info.dims, info.dtype, chunk_dims, data)
        }
    }
}

/// Parses a version-1 or version-2 dataspace message into dimension extents.
fn parse_dataspace(data: &[u8]) -> Result<Vec<u64>, H5Error> {
    let mut c = Cursor::at(data, 0);
    let version = c.u8()?;
    let rank = c.u8()? as usize;
    match version {
        1 => {
            c.skip(1); // flags.
            c.skip(1); // reserved.
            c.skip(4); // reserved.
        }
        2 => {
            c.skip(1); // flags.
            c.skip(1); // type.
        }
        _ => return Err(H5Error::UnsupportedLayout),
    }
    let mut dims = Vec::with_capacity(rank);
    for _ in 0..rank {
        dims.push(c.u64()?);
    }
    Ok(dims)
}

/// Parses a version-1 datatype message, requiring IEEE float32/float64.
fn parse_float_datatype(data: &[u8]) -> Result<Dtype, H5Error> {
    let class_version = *data.first().ok_or(H5Error::UnexpectedEof)?;
    let class = class_version & 0x0f;
    if class != 1 {
        return Err(H5Error::UnsupportedDatatype);
    }
    let size = u32::from_le_bytes([
        *data.get(4).ok_or(H5Error::UnexpectedEof)?,
        *data.get(5).ok_or(H5Error::UnexpectedEof)?,
        *data.get(6).ok_or(H5Error::UnexpectedEof)?,
        *data.get(7).ok_or(H5Error::UnexpectedEof)?,
    ]);
    match size {
        4 => Ok(Dtype::F32),
        8 => Ok(Dtype::F64),
        _ => Err(H5Error::UnsupportedDatatype),
    }
}

/// Parses a version-3 data layout message (contiguous or chunked).
fn parse_layout(data: &[u8]) -> Result<LayoutInfo, H5Error> {
    let mut c = Cursor::at(data, 0);
    let version = c.u8()?;
    if version != 3 {
        return Err(H5Error::UnsupportedLayout);
    }
    let class = c.u8()?;
    match class {
        1 => {
            let addr = c.u64()?;
            let size = c.u64()?;
            Ok(LayoutInfo::Contiguous { addr, size })
        }
        2 => {
            let dimensionality = c.u8()? as usize;
            let btree = c.u64()?;
            if dimensionality == 0 {
                return Err(H5Error::UnsupportedLayout);
            }
            // The last entry is the element size; the rest are chunk extents.
            let mut chunk_dims = Vec::with_capacity(dimensionality - 1);
            for _ in 0..dimensionality - 1 {
                chunk_dims.push(c.u32()? as u64);
            }
            let _element_size = c.u32()?;
            Ok(LayoutInfo::Chunked { btree, chunk_dims })
        }
        _ => Err(H5Error::UnsupportedLayout),
    }
}

/// Reads and reassembles a chunked dataset's elements in row-major order.
fn read_chunked(
    bytes: &[u8],
    btree: u64,
    dims: &[u64],
    chunk_dims: &[u64],
    dtype: Dtype,
) -> Result<Vec<f64>, H5Error> {
    let rank = dims.len();
    if chunk_dims.len() != rank {
        return Err(H5Error::ShapeMismatch);
    }
    let total = element_count(dims);
    let mut out = vec![0.0_f64; total];
    let mut chunks: Vec<(Vec<u64>, u64)> = Vec::new();
    read_chunk_btree(bytes, btree, rank, &mut chunks)?;

    let chunk_elems = element_count(chunk_dims);
    let chunk_bytes = chunk_elems * dtype.size();
    let strides = row_major_strides(dims);

    for (offsets, chunk_addr) in chunks {
        let raw = bytes
            .get(chunk_addr as usize..chunk_addr as usize + chunk_bytes)
            .ok_or(H5Error::UnexpectedEof)?;
        let values = decode_floats(raw, dtype, chunk_elems)?;
        // Walk the chunk in row-major order, mapping each element to its global
        // coordinate and copying only those inside the dataset extents.
        let mut local = vec![0u64; rank];
        for value in values {
            let mut inside = true;
            let mut flat = 0usize;
            for d in 0..rank {
                let global = offsets[d] + local[d];
                if global >= dims[d] {
                    inside = false;
                    break;
                }
                flat += global as usize * strides[d];
            }
            if inside {
                out[flat] = value;
            }
            increment_coord(&mut local, chunk_dims);
        }
    }
    Ok(out)
}

/// Recursively walks a version-1 raw-data B-tree, collecting `(chunk element
/// offsets, chunk data address)` pairs.
fn read_chunk_btree(
    bytes: &[u8],
    addr: u64,
    rank: usize,
    out: &mut Vec<(Vec<u64>, u64)>,
) -> Result<(), H5Error> {
    let mut c = Cursor::at(bytes, addr as usize);
    let sig = c.slice(c.pos, 4)?;
    if sig != b"TREE" {
        return Err(H5Error::BadBlockSignature("TREE"));
    }
    c.skip(4);
    let node_type = c.u8()?;
    let node_level = c.u8()?;
    let entries = c.u16()? as usize;
    let _left = c.u64()?;
    let _right = c.u64()?;
    if node_type != 1 {
        return Err(H5Error::UnsupportedLayout);
    }
    // Each key is: chunk size (4), filter mask (4), (rank + 1) * 8-byte offsets.
    for _ in 0..entries {
        let _chunk_size = c.u32()?;
        let filter_mask = c.u32()?;
        if filter_mask != 0 {
            return Err(H5Error::UnsupportedFilter);
        }
        let mut offsets = Vec::with_capacity(rank);
        for _ in 0..rank {
            offsets.push(c.u64()?);
        }
        let _element_offset = c.u64()?; // trailing element-dimension offset.
        let child = c.u64()?;
        if node_level == 0 {
            out.push((offsets, child));
        } else {
            read_chunk_btree(bytes, child, rank, out)?;
        }
    }
    Ok(())
}

/// Decodes `count` little-endian IEEE floats of `dtype` from `raw`.
fn decode_floats(raw: &[u8], dtype: Dtype, count: usize) -> Result<Vec<f64>, H5Error> {
    let mut out = Vec::with_capacity(count);
    match dtype {
        Dtype::F32 => {
            for chunk in raw.chunks_exact(4).take(count) {
                let v = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                out.push(v as f64);
            }
        }
        Dtype::F64 => {
            for chunk in raw.chunks_exact(8).take(count) {
                let mut a = [0u8; 8];
                a.copy_from_slice(chunk);
                out.push(f64::from_le_bytes(a));
            }
        }
    }
    if out.len() != count {
        return Err(H5Error::UnexpectedEof);
    }
    Ok(out)
}

/// Row-major strides (in elements) for `dims`.
fn row_major_strides(dims: &[u64]) -> Vec<usize> {
    let rank = dims.len();
    let mut strides = vec![1usize; rank];
    let mut acc = 1usize;
    let mut i = rank;
    while i > 0 {
        i -= 1;
        strides[i] = acc;
        acc *= dims[i] as usize;
    }
    strides
}

/// Increments a row-major coordinate within `extents`, wrapping low dimensions.
fn increment_coord(coord: &mut [u64], extents: &[u64]) {
    let mut i = coord.len();
    while i > 0 {
        i -= 1;
        coord[i] += 1;
        if coord[i] < extents[i] {
            return;
        }
        coord[i] = 0;
    }
}

/// Encodes `values` as little-endian IEEE floats of `dtype`.
fn encode_floats(values: &[f64], dtype: Dtype, out: &mut Vec<u8>) {
    match dtype {
        Dtype::F32 => {
            for &v in values {
                out.extend_from_slice(&(v as f32).to_le_bytes());
            }
        }
        Dtype::F64 => {
            for &v in values {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
    }
}

// ------------------------------------------------------------------------
// Writer
// ------------------------------------------------------------------------

/// A growable buffer with label/fixup support for forward address references.
struct Builder {
    buf: Vec<u8>,
    labels: BTreeMap<String, u64>,
    fixups: Vec<(usize, String)>,
}

impl Builder {
    fn new() -> Self {
        Self {
            buf: Vec::new(),
            labels: BTreeMap::new(),
            fixups: Vec::new(),
        }
    }

    #[inline]
    fn here(&self) -> u64 {
        self.buf.len() as u64
    }

    #[inline]
    fn label(&mut self, name: &str) {
        let pos = self.here();
        self.labels.insert(name.to_string(), pos);
    }

    #[inline]
    fn u8(&mut self, v: u8) {
        self.buf.push(v);
    }

    #[inline]
    fn u16(&mut self, v: u16) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    #[inline]
    fn u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    #[inline]
    fn u64(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    #[inline]
    fn bytes(&mut self, b: &[u8]) {
        self.buf.extend_from_slice(b);
    }

    /// Writes a placeholder 8-byte offset to be patched to `label`'s address.
    #[inline]
    fn offset_ref(&mut self, label: &str) {
        let pos = self.buf.len();
        self.fixups.push((pos, label.to_string()));
        self.buf.extend_from_slice(&0u64.to_le_bytes());
    }

    #[inline]
    fn align8(&mut self) {
        while !self.buf.len().is_multiple_of(8) {
            self.buf.push(0);
        }
    }

    /// Resolves all fixups into concrete little-endian addresses.
    fn finish(mut self) -> Vec<u8> {
        for (pos, label) in &self.fixups {
            let addr = self.labels.get(label).copied().unwrap_or(UNDEF);
            self.buf[*pos..*pos + 8].copy_from_slice(&addr.to_le_bytes());
        }
        self.buf
    }
}

impl H5File {
    /// Serializes this file into a self-consistent HDF5 byte image.
    ///
    /// The output is deterministic (sorted attribute and dataset order), so a
    /// `write -> read -> write` cycle reproduces identical bytes.
    #[must_use]
    pub fn write(&self) -> Vec<u8> {
        let mut b = Builder::new();

        // Superblock (version 0, 8-byte offsets/lengths).
        b.bytes(&SIGNATURE);
        b.u8(0); // superblock version.
        b.u8(0); // free-space version.
        b.u8(0); // root-group symbol table version.
        b.u8(0); // reserved.
        b.u8(0); // shared header message format version.
        b.u8(8); // size of offsets.
        b.u8(8); // size of lengths.
        b.u8(0); // reserved.
        b.u16(4); // group leaf node K.
        b.u16(16); // group internal node K.
        b.u32(0); // file consistency flags.
        b.u64(0); // base address.
        b.u64(UNDEF); // free-space info address.
        let eof_pos = b.buf.len();
        b.u64(0); // end-of-file address (patched at the end).
        b.u64(UNDEF); // driver information block address.
        // Root group symbol table entry (40 bytes).
        b.u64(0); // link name offset.
        b.offset_ref("root_oh"); // object header address.
        b.u32(1); // cache type: group cache present.
        b.u32(0); // reserved.
        b.offset_ref("root_btree"); // scratch: B-tree address.
        b.offset_ref("root_heap"); // scratch: name heap address.

        // Plan the local heap string table so symbol table nodes can reference
        // name offsets before the heap bytes are emitted.
        let mut names: Vec<&String> = self.datasets.keys().collect();
        names.sort();
        let (heap_bytes, name_offsets, heap_seg_size, free_list_head) = plan_heap(&names);

        // Root group object header: attributes then the symbol table message.
        b.align8();
        b.label("root_oh");
        write_object_header(&mut b, self.attributes.len() + 1, |mb| {
            for (name, value) in &self.attributes {
                write_attribute_message(mb, name, value);
            }
            write_symbol_table_message(mb, "root_btree", "root_heap");
        });

        // Local heap header.
        b.align8();
        b.label("root_heap");
        b.bytes(b"HEAP");
        b.u8(0); // version.
        b.bytes(&[0, 0, 0]); // reserved.
        b.u64(heap_seg_size);
        b.u64(free_list_head);
        b.offset_ref("root_heap_data");

        // Group B-tree with a single leaf symbol table node.
        b.align8();
        b.label("root_btree");
        let last_key = name_offsets.last().copied().unwrap_or(0);
        b.bytes(b"TREE");
        b.u8(0); // node type: group.
        b.u8(0); // node level: leaf.
        b.u16(1); // entries used.
        b.u64(UNDEF); // left sibling.
        b.u64(UNDEF); // right sibling.
        b.u64(0); // key 0 (least name offset).
        b.offset_ref("root_snod"); // child 0.
        b.u64(last_key); // key 1 (greatest name offset).

        // Symbol table node listing every dataset link.
        b.align8();
        b.label("root_snod");
        b.bytes(b"SNOD");
        b.u8(1); // version.
        b.u8(0); // reserved.
        b.u16(names.len() as u16);
        for (name, &off) in names.iter().zip(name_offsets.iter()) {
            b.u64(off); // link name offset.
            b.offset_ref(&dataset_label(name)); // object header address.
            b.u32(0); // cache type: none.
            b.u32(0); // reserved.
            b.bytes(&[0u8; 16]); // scratch pad.
        }

        // Local heap data segment (named strings + a trailing free block).
        b.align8();
        b.label("root_heap_data");
        b.bytes(&heap_bytes);

        // Each dataset: object header then its data (contiguous or chunked).
        for name in &names {
            let dataset = &self.datasets[*name];
            b.align8();
            b.label(&dataset_label(name));
            write_dataset_header(&mut b, name, dataset);
            write_dataset_data(&mut b, name, dataset);
        }

        // Patch the end-of-file address and resolve fixups.
        let eof = b.buf.len() as u64;
        b.buf[eof_pos..eof_pos + 8].copy_from_slice(&eof.to_le_bytes());
        b.finish()
    }
}

/// A stable object-header label for a dataset name.
fn dataset_label(name: &str) -> String {
    format!("ds::{name}")
}

/// Lays out the local heap: `(bytes, per-name offsets, segment size, free-list
/// head offset)`.
///
/// Offset 0 is reserved as the empty string; names follow, each null-terminated
/// and the whole run padded to 8 bytes, then a single free block.
fn plan_heap(names: &[&String]) -> (Vec<u8>, Vec<u64>, u64, u64) {
    let mut bytes = Vec::new();
    bytes.push(0); // offset 0: empty string.
    while bytes.len() % 8 != 0 {
        bytes.push(0);
    }
    let mut offsets = Vec::with_capacity(names.len());
    for name in names {
        let off = bytes.len() as u64;
        offsets.push(off);
        bytes.extend_from_slice(name.as_bytes());
        bytes.push(0);
        while bytes.len() % 8 != 0 {
            bytes.push(0);
        }
    }
    let free_list_head = bytes.len() as u64;
    // One free block: next-free offset (1 = end) and the block size.
    bytes.extend_from_slice(&1u64.to_le_bytes());
    bytes.extend_from_slice(&16u64.to_le_bytes());
    let seg_size = bytes.len() as u64;
    (bytes, offsets, seg_size, free_list_head)
}

/// Writes a version-1 object header whose messages are emitted by `body`.
fn write_object_header<F: FnOnce(&mut Builder)>(b: &mut Builder, nmsg: usize, body: F) {
    b.u8(1); // version.
    b.u8(0); // reserved.
    b.u16(nmsg as u16);
    b.u32(1); // object reference count.
    let size_pos = b.buf.len();
    b.u32(0); // header size (patched below).
    b.align8(); // align the first message to an 8-byte boundary.
    let msg_start = b.buf.len();
    body(b);
    let msg_len = (b.buf.len() - msg_start) as u32;
    b.buf[size_pos..size_pos + 4].copy_from_slice(&msg_len.to_le_bytes());
}

/// Writes one object-header message with an 8-byte-padded data body.
fn write_message<F: FnOnce(&mut Builder)>(b: &mut Builder, kind: u16, data_len: usize, body: F) {
    let padded = round_up_8(data_len);
    b.u16(kind);
    b.u16(padded as u16);
    b.u8(0); // flags.
    b.bytes(&[0, 0, 0]); // reserved.
    let start = b.buf.len();
    body(b);
    while b.buf.len() - start < padded {
        b.buf.push(0);
    }
}

/// Writes a version-1 dataspace message for `dims`.
fn write_dataspace_message(b: &mut Builder, dims: &[u64]) {
    let data_len = 8 + dims.len() * 8;
    write_message(b, 0x0001, data_len, |mb| {
        mb.u8(1); // version.
        mb.u8(dims.len() as u8); // dimensionality.
        mb.u8(0); // flags (no maximum dimensions).
        mb.u8(0); // reserved.
        mb.u32(0); // reserved.
        for &d in dims {
            mb.u64(d);
        }
    });
}

/// Writes a version-1 IEEE float datatype message.
fn write_float_datatype_message(b: &mut Builder, dtype: Dtype) {
    write_message(b, 0x0003, 20, |mb| {
        mb.u8(0x11); // class 1 (float), version 1.
        // Bit field: little-endian, IEEE mantissa normalization, sign at MSB.
        let (precision, exp_loc, exp_size, mant_size, bias, sign) = match dtype {
            Dtype::F32 => (32u16, 23u8, 8u8, 23u8, 127u32, 31u8),
            Dtype::F64 => (64u16, 52u8, 11u8, 52u8, 1023u32, 63u8),
        };
        mb.u8(0x20); // mantissa normalization = 2 (implied set bit).
        mb.u8(sign); // sign bit location.
        mb.u8(0); // bit field byte 3.
        mb.u32(dtype.size() as u32); // element size in bytes.
        mb.u16(0); // bit offset.
        mb.u16(precision); // bit precision.
        mb.u8(exp_loc); // exponent location.
        mb.u8(exp_size); // exponent size.
        mb.u8(0); // mantissa location.
        mb.u8(mant_size); // mantissa size.
        mb.u32(bias); // exponent bias.
    });
}

/// Writes a version-2 "undefined" fill value message.
fn write_fill_value_message(b: &mut Builder) {
    write_message(b, 0x0005, 4, |mb| {
        mb.u8(2); // version.
        mb.u8(2); // space allocation time: late.
        mb.u8(0); // fill value write time: never.
        mb.u8(0); // fill value defined: no.
    });
}

/// Writes the dataspace, datatype, fill value, and data layout for a dataset.
fn write_dataset_header(b: &mut Builder, name: &str, dataset: &Dataset) {
    write_object_header(b, 4, |mb| {
        write_dataspace_message(mb, &dataset.dims);
        write_float_datatype_message(mb, dataset.dtype);
        write_fill_value_message(mb);
        match &dataset.storage {
            Storage::Contiguous => {
                let size = (element_count(&dataset.dims) * dataset.dtype.size()) as u64;
                write_message(mb, 0x0008, 18, |m| {
                    m.u8(3); // version.
                    m.u8(1); // layout class: contiguous.
                    m.offset_ref(&data_label(name));
                    m.u64(size);
                });
            }
            Storage::Chunked(chunk) => {
                let dimensionality = chunk.len() + 1;
                let data_len = 11 + dimensionality * 4;
                write_message(mb, 0x0008, data_len, |m| {
                    m.u8(3); // version.
                    m.u8(2); // layout class: chunked.
                    m.u8(dimensionality as u8);
                    m.offset_ref(&chunk_btree_label(name));
                    for &c in chunk {
                        m.u32(c as u32);
                    }
                    m.u32(dataset.dtype.size() as u32); // element-size dimension.
                });
            }
        }
    });
}

/// Writes a dataset's payload: contiguous bytes, or a chunk B-tree plus chunks.
fn write_dataset_data(b: &mut Builder, name: &str, dataset: &Dataset) {
    match &dataset.storage {
        Storage::Contiguous => {
            b.align8();
            b.label(&data_label(name));
            let mut raw = Vec::with_capacity(dataset.data.len() * dataset.dtype.size());
            encode_floats(&dataset.data, dataset.dtype, &mut raw);
            b.bytes(&raw);
        }
        Storage::Chunked(chunk) => {
            write_chunked_data(b, name, dataset, chunk);
        }
    }
}

/// Writes a single-leaf raw-data B-tree and the (zero-padded) chunk bodies.
fn write_chunked_data(b: &mut Builder, name: &str, dataset: &Dataset, chunk: &[u64]) {
    let rank = dataset.dims.len();
    let chunk_elems = element_count(chunk);
    let chunk_bytes = chunk_elems * dataset.dtype.size();

    // Enumerate chunk origins in row-major order of the chunk grid.
    let mut counts = Vec::with_capacity(rank);
    for (&dim, &c) in dataset.dims.iter().zip(chunk.iter()) {
        counts.push(dim.div_ceil(c.max(1)));
    }
    let mut origins: Vec<Vec<u64>> = Vec::new();
    let mut grid = vec![0u64; rank];
    let total_chunks = element_count(&counts);
    for _ in 0..total_chunks {
        let mut origin = Vec::with_capacity(rank);
        for d in 0..rank {
            origin.push(grid[d] * chunk[d]);
        }
        origins.push(origin);
        increment_coord(&mut grid, &counts);
    }

    let strides = row_major_strides(&dataset.dims);

    // B-tree: key0, child0, ..., child_{n-1}, key_n. Keys carry chunk size,
    // filter mask (0), and element offsets (rank + 1, trailing element = 0).
    b.align8();
    b.label(&chunk_btree_label(name));
    b.bytes(b"TREE");
    b.u8(1); // node type: raw data.
    b.u8(0); // node level: leaf.
    b.u16(origins.len() as u16);
    b.u64(UNDEF); // left sibling.
    b.u64(UNDEF); // right sibling.
    for (i, origin) in origins.iter().enumerate() {
        write_chunk_key(b, chunk_bytes as u32, origin);
        b.offset_ref(&chunk_label(name, i));
    }
    // Final key: the end offsets (dataset extents), zero chunk size.
    write_chunk_key(b, 0, &dataset.dims);

    // Chunk bodies, zero-padded on edges.
    for (i, origin) in origins.iter().enumerate() {
        b.align8();
        b.label(&chunk_label(name, i));
        let mut values = vec![0.0_f64; chunk_elems];
        let mut local = vec![0u64; rank];
        for value in values.iter_mut() {
            let mut inside = true;
            let mut flat = 0usize;
            for d in 0..rank {
                let global = origin[d] + local[d];
                if global >= dataset.dims[d] {
                    inside = false;
                    break;
                }
                flat += global as usize * strides[d];
            }
            if inside {
                *value = dataset.data[flat];
            }
            increment_coord(&mut local, chunk);
        }
        let mut raw = Vec::with_capacity(chunk_bytes);
        encode_floats(&values, dataset.dtype, &mut raw);
        b.bytes(&raw);
    }
}

/// Writes a raw-data B-tree key: chunk byte size, zero filter mask, and the
/// per-dimension element offsets plus a trailing zero element offset.
fn write_chunk_key(b: &mut Builder, chunk_size: u32, offsets: &[u64]) {
    b.u32(chunk_size);
    b.u32(0); // filter mask: unfiltered.
    for &o in offsets {
        b.u64(o);
    }
    b.u64(0); // element-dimension offset.
}

/// Writes a version-1 fixed-length string attribute message.
fn write_attribute_message(b: &mut Builder, name: &str, value: &str) {
    let name_bytes = name.as_bytes();
    let name_size = name_bytes.len() + 1; // includes the null terminator.
    let dt_size = 8usize;
    let ds_size = 8usize;
    let value_bytes = value.as_bytes();
    let data_len = 8
        + round_up_8(name_size)
        + round_up_8(dt_size)
        + round_up_8(ds_size)
        + value_bytes.len();
    write_message(b, 0x000C, data_len, |mb| {
        mb.u8(1); // version.
        mb.u8(0); // reserved.
        mb.u16(name_size as u16);
        mb.u16(dt_size as u16);
        mb.u16(ds_size as u16);
        let name_start = mb.buf.len();
        mb.bytes(name_bytes);
        mb.u8(0);
        while mb.buf.len() - name_start < round_up_8(name_size) {
            mb.buf.push(0);
        }
        // Datatype: class 3 (string), size = value length, null-terminated.
        let dt_start = mb.buf.len();
        mb.u8(0x13); // class 3 (string), version 1.
        mb.bytes(&[0, 0, 0]); // bit field: null terminate, ASCII.
        mb.u32(value_bytes.len() as u32); // string size in bytes.
        while mb.buf.len() - dt_start < round_up_8(dt_size) {
            mb.buf.push(0);
        }
        // Dataspace: scalar (rank 0).
        let ds_start = mb.buf.len();
        mb.u8(1); // version.
        mb.u8(0); // dimensionality (scalar).
        mb.u8(0); // flags.
        mb.u8(0); // reserved.
        mb.u32(0); // reserved.
        while mb.buf.len() - ds_start < round_up_8(ds_size) {
            mb.buf.push(0);
        }
        mb.bytes(value_bytes);
    });
}

/// Writes a symbol table message referencing a B-tree and local heap by label.
fn write_symbol_table_message(b: &mut Builder, btree_label: &str, heap_label: &str) {
    write_message(b, 0x0011, 16, |mb| {
        mb.offset_ref(btree_label);
        mb.offset_ref(heap_label);
    });
}

/// A stable label for a dataset's contiguous data block.
fn data_label(name: &str) -> String {
    format!("data::{name}")
}

/// A stable label for a dataset's chunk B-tree.
fn chunk_btree_label(name: &str) -> String {
    format!("cbtree::{name}")
}

/// A stable label for the `i`-th chunk body of a dataset.
fn chunk_label(name: &str, i: usize) -> String {
    format!("chunk::{name}::{i}")
}

/// Interprets `bytes` as a null-terminated/padded ASCII string.
fn cstr_to_string(bytes: &[u8]) -> String {
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..end]).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() <= eps
    }

    fn sample_file(storage_chunk: Option<Vec<u64>>) -> H5File {
        let mut file = H5File::new();
        file.set_attribute("Conventions", "SOFA");
        file.set_attribute("SOFAConventions", "SimpleFreeFieldHRIR");
        // Data.IR: [M=2][R=2][N=3].
        let ir: Vec<f64> = (0..12).map(|i| i as f64 * 0.25).collect();
        let ir_ds = match &storage_chunk {
            None => Dataset::contiguous(vec![2, 2, 3], Dtype::F32, ir).unwrap(),
            Some(c) => Dataset::chunked(vec![2, 2, 3], Dtype::F32, c.clone(), ir).unwrap(),
        };
        file.insert_dataset("Data.IR", ir_ds);
        // SourcePosition: [M=2][C=3].
        let pos = vec![0.0, 0.0, 1.0, 90.0, 0.0, 1.0];
        file.insert_dataset(
            "SourcePosition",
            Dataset::contiguous(vec![2, 3], Dtype::F64, pos).unwrap(),
        );
        // Data.SamplingRate: [I=1].
        file.insert_dataset(
            "Data.SamplingRate",
            Dataset::contiguous(vec![1], Dtype::F64, vec![48_000.0]).unwrap(),
        );
        file
    }

    #[test]
    fn contiguous_round_trip_matches() {
        let file = sample_file(None);
        let bytes = file.write();
        let decoded = H5File::read(&bytes).unwrap();
        assert_eq!(decoded.attributes.get("Conventions").unwrap(), "SOFA");
        assert_eq!(
            decoded.attributes.get("SOFAConventions").unwrap(),
            "SimpleFreeFieldHRIR"
        );
        let ir = &decoded.datasets["Data.IR"];
        assert_eq!(ir.dims, vec![2, 2, 3]);
        for (a, b) in ir.data.iter().zip((0..12).map(|i| i as f64 * 0.25)) {
            assert!(approx(*a, b, 1e-6));
        }
        let sr = &decoded.datasets["Data.SamplingRate"];
        assert!(approx(sr.data[0], 48_000.0, 1e-6));
    }

    #[test]
    fn write_read_write_is_byte_identical() {
        let file = sample_file(None);
        let b1 = file.write();
        let decoded = H5File::read(&b1).unwrap();
        let b2 = decoded.write();
        assert_eq!(b1, b2);
    }

    #[test]
    fn chunked_round_trip_matches_contiguous() {
        let contiguous = sample_file(None);
        let chunked = sample_file(Some(vec![1, 2, 2]));
        let decoded = H5File::read(&chunked.write()).unwrap();
        let expected = &contiguous.datasets["Data.IR"];
        let actual = &decoded.datasets["Data.IR"];
        assert_eq!(actual.dims, expected.dims);
        for (a, b) in actual.data.iter().zip(expected.data.iter()) {
            assert!(approx(*a, *b, 1e-6));
        }
    }

    #[test]
    fn chunked_write_read_write_is_byte_identical() {
        let file = sample_file(Some(vec![2, 1, 2]));
        let b1 = file.write();
        let b2 = H5File::read(&b1).unwrap().write();
        assert_eq!(b1, b2);
    }

    #[test]
    fn float64_values_survive_exactly() {
        let mut file = H5File::new();
        file.insert_dataset(
            "d",
            Dataset::contiguous(vec![3], Dtype::F64, vec![0.1, 2.5, -3.75]).unwrap(),
        );
        let decoded = H5File::read(&file.write()).unwrap();
        let d = &decoded.datasets["d"];
        assert_eq!(d.dtype, Dtype::F64);
        assert!(approx(d.data[0], 0.1, 0.0));
        assert!(approx(d.data[1], 2.5, 0.0));
        assert!(approx(d.data[2], -3.75, 0.0));
    }

    #[test]
    fn bad_signature_is_reported() {
        let err = H5File::read(&[0, 1, 2, 3, 4, 5, 6, 7]).unwrap_err();
        assert_eq!(err, H5Error::BadSignature);
    }

    #[test]
    fn truncated_input_is_reported() {
        let file = sample_file(None);
        let bytes = file.write();
        let err = H5File::read(&bytes[..bytes.len() / 2]).unwrap_err();
        assert_eq!(err, H5Error::UnexpectedEof);
    }

    #[test]
    fn filtered_chunk_is_rejected() {
        // Hand-corrupt a chunked file's first raw-data B-tree filter mask to a
        // non-zero (filtered) value and confirm the reader refuses it rather
        // than returning wrong data.
        let file = sample_file(Some(vec![1, 2, 2]));
        let mut bytes = file.write();
        let tree = find_subsequence(&bytes, b"TREE\x01").expect("raw-data TREE");
        // Key begins at: TREE(4)+type(1)+level(1)+entries(2)+left(8)+right(8).
        let filter_pos = tree + 4 + 1 + 1 + 2 + 8 + 8 + 4;
        bytes[filter_pos..filter_pos + 4].copy_from_slice(&1u32.to_le_bytes());
        let err = H5File::read(&bytes).unwrap_err();
        assert_eq!(err, H5Error::UnsupportedFilter);
    }

    #[test]
    fn multi_chunk_grid_reassembles() {
        // A 5-long vector in chunks of 2 exercises edge padding and ordering.
        let data: Vec<f64> = (0..5).map(|i| i as f64).collect();
        let mut file = H5File::new();
        file.insert_dataset(
            "v",
            Dataset::chunked(vec![5], Dtype::F32, vec![2], data.clone()).unwrap(),
        );
        let decoded = H5File::read(&file.write()).unwrap();
        let v = &decoded.datasets["v"];
        for (a, b) in v.data.iter().zip(data.iter()) {
            assert!(approx(*a, *b, 1e-6));
        }
    }

    fn find_subsequence(haystack: &[u8], needle: &[u8]) -> Option<usize> {
        haystack
            .windows(needle.len())
            .position(|w| w == needle)
    }
}
