//! Versioned CPU, HLSL, WESL, and SPIR-V data contracts.

/// Identifies a version of a generated render ABI.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct AbiVersion(pub u32);

/// A stable, generational index shared across CPU and GPU code.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct GenerationalHandle {
    pub index: u32,
    pub generation: u32,
}

impl GenerationalHandle {
    pub const INVALID: Self = Self {
        index: u32::MAX,
        generation: 0,
    };

    pub const fn is_valid(self) -> bool {
        self.index != u32::MAX
    }
}

/// Hashes all inputs that affect a generated ABI package.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct AbiHash(pub [u8; 32]);
