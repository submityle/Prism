//! Engine-neutral voxel representation and storage contracts.

use alloc::vec::Vec;

/// The supported two-byte voxel component encodings.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VoxelComponentFormat {
    /// Signed distance in Q8.8 units.
    SignedDistanceQ8_8,
    /// Unsigned density in the full `u16` range.
    DensityU16,
    /// Unsigned material identifier.
    MaterialIdU16,
}

impl VoxelComponentFormat {
    /// Returns the number of bytes per voxel component.
    pub const fn byte_width(self) -> usize {
        2
    }
}

/// A validated row-major voxel volume layout.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VoxelLayout {
    dimensions: [u32; 3],
    brick_size: u32,
    format: VoxelComponentFormat,
}

impl VoxelLayout {
    /// Creates a layout. Bricks must be powers of two and dimensions are non-zero.
    pub fn new(
        dimensions: [u32; 3],
        brick_size: u32,
        format: VoxelComponentFormat,
    ) -> Result<Self, VoxelError> {
        if dimensions.contains(&0) {
            return Err(VoxelError::InvalidDimensions);
        }
        if brick_size == 0 || !brick_size.is_power_of_two() {
            return Err(VoxelError::InvalidBrickSize);
        }
        let layout = Self {
            dimensions,
            brick_size,
            format,
        };
        layout.expected_byte_len()?;
        Ok(layout)
    }

    /// Returns the voxel dimensions in X/Y/Z order.
    pub const fn dimensions(self) -> [u32; 3] {
        self.dimensions
    }

    /// Returns the streaming brick edge length.
    pub const fn brick_size(self) -> u32 {
        self.brick_size
    }

    /// Returns the component format.
    pub const fn format(self) -> VoxelComponentFormat {
        self.format
    }

    /// Returns the number of voxels, or zero only if the count overflows.
    pub fn voxel_count(self) -> usize {
        (self.dimensions[0] as usize)
            .checked_mul(self.dimensions[1] as usize)
            .and_then(|count| count.checked_mul(self.dimensions[2] as usize))
            .unwrap_or(0)
    }

    /// Returns the exact little-endian byte length required by this layout.
    pub fn expected_byte_len(self) -> Result<usize, VoxelError> {
        self.checked_voxel_count()?
            .checked_mul(self.format.byte_width())
            .ok_or(VoxelError::SizeOverflow)
    }

    /// Returns the row-major index with X as the fastest-changing coordinate.
    pub fn linear_index(self, coordinate: [u32; 3]) -> Result<usize, VoxelError> {
        if coordinate
            .iter()
            .zip(self.dimensions)
            .any(|(coordinate, dimension)| *coordinate >= dimension)
        {
            return Err(VoxelError::OutOfBounds { coordinate });
        }
        let x = coordinate[0] as usize;
        let y = coordinate[1] as usize;
        let z = coordinate[2] as usize;
        Ok(x + self.dimensions[0] as usize * (y + self.dimensions[1] as usize * z))
    }

    fn checked_voxel_count(self) -> Result<usize, VoxelError> {
        (self.dimensions[0] as usize)
            .checked_mul(self.dimensions[1] as usize)
            .and_then(|count| count.checked_mul(self.dimensions[2] as usize))
            .ok_or(VoxelError::SizeOverflow)
    }
}

/// Errors returned by voxel layout and storage operations.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VoxelError {
    /// At least one volume dimension is zero.
    InvalidDimensions,
    /// The brick edge is not a non-zero power of two.
    InvalidBrickSize,
    /// A size calculation overflowed `usize`.
    SizeOverflow,
    /// The byte buffer does not match the layout exactly.
    InvalidByteLength {
        /// Required byte length.
        expected: usize,
        /// Supplied byte length.
        actual: usize,
    },
    /// A coordinate lies outside the volume.
    OutOfBounds {
        /// Requested voxel coordinate.
        coordinate: [u32; 3],
    },
    /// The requested accessor does not match the layout format.
    FormatMismatch,
}

/// One decoded voxel value.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VoxelSample {
    /// Signed Q8.8 distance.
    SignedDistance(i16),
    /// Unsigned density.
    Density(u16),
    /// Unsigned material identifier.
    MaterialId(u16),
}

/// A row-major voxel field backed by an exact byte buffer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VoxelBuffer {
    layout: VoxelLayout,
    bytes: Vec<u8>,
}

impl VoxelBuffer {
    /// Creates a buffer after checking its exact representation contract.
    pub fn new(layout: VoxelLayout, bytes: Vec<u8>) -> Result<Self, VoxelError> {
        let expected = layout.expected_byte_len()?;
        if bytes.len() != expected {
            return Err(VoxelError::InvalidByteLength {
                expected,
                actual: bytes.len(),
            });
        }
        Ok(Self { layout, bytes })
    }

    /// Returns the layout.
    pub const fn layout(&self) -> VoxelLayout {
        self.layout
    }

    /// Returns the exact little-endian backing bytes.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Returns the row-major index for a coordinate.
    pub fn linear_index(&self, coordinate: [u32; 3]) -> Result<usize, VoxelError> {
        self.layout.linear_index(coordinate)
    }

    /// Reads one value according to the declared format.
    pub fn sample(&self, coordinate: [u32; 3]) -> Result<VoxelSample, VoxelError> {
        let index = self.linear_index(coordinate)?;
        let offset = index * 2;
        let bytes = [self.bytes[offset], self.bytes[offset + 1]];
        Ok(match self.layout.format {
            VoxelComponentFormat::SignedDistanceQ8_8 => {
                VoxelSample::SignedDistance(i16::from_le_bytes(bytes))
            }
            VoxelComponentFormat::DensityU16 => VoxelSample::Density(u16::from_le_bytes(bytes)),
            VoxelComponentFormat::MaterialIdU16 => {
                VoxelSample::MaterialId(u16::from_le_bytes(bytes))
            }
        })
    }
}

/// A source of voxel values with an explicit representation contract.
pub trait VoxelField {
    /// Returns the field's layout.
    fn layout(&self) -> VoxelLayout;

    /// Samples a voxel coordinate.
    fn sample(&self, coordinate: [u32; 3]) -> Result<VoxelSample, VoxelError>;
}

impl VoxelField for VoxelBuffer {
    fn layout(&self) -> VoxelLayout {
        self.layout()
    }

    fn sample(&self, coordinate: [u32; 3]) -> Result<VoxelSample, VoxelError> {
        self.sample(coordinate)
    }
}
