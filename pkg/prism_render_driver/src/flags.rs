//! Hand-rolled bitflag sets for GPU usage and stage masks.
//!
//! Prism's RHI core avoids external crates (including `bitflags`) to stay fully
//! controllable and dependency-free. The [`bitflags`] macro in this module
//! generates a `u32`-backed newtype with the usual set algebra
//! (union/intersection/difference/complement) plus `contains`, `insert`,
//! `remove`, and `toggle`, with every generated item documented.

/// Declares a `u32`-backed bitflag newtype with named flag constants and the
/// standard set operations.
///
/// Each flag constant and the type itself carry the doc comments supplied at
/// the call site, satisfying the workspace `missing_docs` lint.
macro_rules! bitflags {
    (
        $(#[$outer:meta])*
        $vis:vis struct $name:ident {
            $(
                $(#[$flag_meta:meta])*
                const $flag:ident = $value:expr;
            )*
        }
    ) => {
        $(#[$outer])*
        #[derive(Clone, Copy, PartialEq, Eq, Hash)]
        $vis struct $name(u32);

        impl $name {
            $(
                $(#[$flag_meta])*
                pub const $flag: Self = Self($value);
            )*

            /// The empty set (no flags set).
            pub const NONE: Self = Self(0);

            /// Creates a set from raw bits, discarding bits not covered by any
            /// declared flag.
            #[must_use]
            pub const fn from_bits_truncate(bits: u32) -> Self {
                Self(bits & Self::all().0)
            }

            /// The raw backing bits.
            #[must_use]
            pub const fn bits(self) -> u32 {
                self.0
            }

            /// The union of every declared flag.
            #[must_use]
            pub const fn all() -> Self {
                Self(0 $(| $value)*)
            }

            /// Whether no flags are set.
            #[must_use]
            pub const fn is_empty(self) -> bool {
                self.0 == 0
            }

            /// Whether every flag in `other` is also set in `self`.
            #[must_use]
            pub const fn contains(self, other: Self) -> bool {
                (self.0 & other.0) == other.0
            }

            /// Whether `self` and `other` share at least one flag.
            #[must_use]
            pub const fn intersects(self, other: Self) -> bool {
                (self.0 & other.0) != 0
            }

            /// Returns `self` with every flag in `other` added.
            #[must_use]
            pub const fn union(self, other: Self) -> Self {
                Self(self.0 | other.0)
            }

            /// Returns the flags common to `self` and `other`.
            #[must_use]
            pub const fn intersection(self, other: Self) -> Self {
                Self(self.0 & other.0)
            }

            /// Returns the flags in `self` that are not in `other`.
            #[must_use]
            pub const fn difference(self, other: Self) -> Self {
                Self(self.0 & !other.0)
            }

            /// Adds every flag in `other` to `self` in place.
            pub fn insert(&mut self, other: Self) {
                self.0 |= other.0;
            }

            /// Removes every flag in `other` from `self` in place.
            pub fn remove(&mut self, other: Self) {
                self.0 &= !other.0;
            }

            /// Toggles every flag in `other` on `self` in place.
            pub fn toggle(&mut self, other: Self) {
                self.0 ^= other.0;
            }
        }

        impl core::ops::BitOr for $name {
            type Output = Self;
            fn bitor(self, rhs: Self) -> Self {
                Self(self.0 | rhs.0)
            }
        }

        impl core::ops::BitOrAssign for $name {
            fn bitor_assign(&mut self, rhs: Self) {
                self.0 |= rhs.0;
            }
        }

        impl core::ops::BitAnd for $name {
            type Output = Self;
            fn bitand(self, rhs: Self) -> Self {
                Self(self.0 & rhs.0)
            }
        }

        impl core::ops::BitAndAssign for $name {
            fn bitand_assign(&mut self, rhs: Self) {
                self.0 &= rhs.0;
            }
        }

        impl core::ops::BitXor for $name {
            type Output = Self;
            fn bitxor(self, rhs: Self) -> Self {
                Self(self.0 ^ rhs.0)
            }
        }

        impl core::ops::Not for $name {
            type Output = Self;
            fn not(self) -> Self {
                Self(!self.0 & Self::all().0)
            }
        }

        impl core::fmt::Debug for $name {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                write!(f, concat!(stringify!($name), "("))?;
                let mut first = true;
                $(
                    if self.contains(Self::$flag) {
                        if !first {
                            f.write_str(" | ")?;
                        }
                        f.write_str(stringify!($flag))?;
                        first = false;
                    }
                )*
                if first {
                    f.write_str("NONE")?;
                }
                f.write_str(")")
            }
        }
    };
}

/// Re-export the local `bitflags!` macro by path so sibling modules
/// (e.g. `capabilities`) can invoke it as `crate::flags::bitflags`.
pub(crate) use bitflags;

bitflags! {
    /// How a texture may be used by the GPU. The validator rejects textures
    /// used in ways their creation flags did not permit.
    pub struct TextureUsages {
        /// Readable as the source of a copy.
        const COPY_SRC = 1 << 0;
        /// Writable as the destination of a copy.
        const COPY_DST = 1 << 1;
        /// Bindable as a sampled texture in a shader.
        const TEXTURE_BINDING = 1 << 2;
        /// Bindable as a read/write storage texture in a shader.
        const STORAGE_BINDING = 1 << 3;
        /// Usable as a color or depth/stencil render attachment.
        const RENDER_ATTACHMENT = 1 << 4;
    }
}

bitflags! {
    /// How a buffer may be used by the GPU.
    pub struct BufferUsages {
        /// Readable as the source of a copy.
        const COPY_SRC = 1 << 0;
        /// Writable as the destination of a copy.
        const COPY_DST = 1 << 1;
        /// Usable as an index buffer in indexed draws.
        const INDEX = 1 << 2;
        /// Usable as a vertex buffer.
        const VERTEX = 1 << 3;
        /// Bindable as a uniform buffer.
        const UNIFORM = 1 << 4;
        /// Bindable as a (possibly writable) storage buffer.
        const STORAGE = 1 << 5;
        /// Readable as the source of indirect draw/dispatch arguments.
        const INDIRECT = 1 << 6;
    }
}

bitflags! {
    /// Which shader stages a binding or push constant range is visible to.
    pub struct ShaderStages {
        /// The vertex stage.
        const VERTEX = 1 << 0;
        /// The fragment stage.
        const FRAGMENT = 1 << 1;
        /// The compute stage.
        const COMPUTE = 1 << 2;
    }
}

bitflags! {
    /// Which color channels a render target write mask permits.
    pub struct ColorWrites {
        /// The red channel.
        const RED = 1 << 0;
        /// The green channel.
        const GREEN = 1 << 1;
        /// The blue channel.
        const BLUE = 1 << 2;
        /// The alpha channel.
        const ALPHA = 1 << 3;
    }
}

impl ShaderStages {
    /// The vertex and fragment stages, the common graphics visibility mask.
    #[must_use]
    pub const fn vertex_fragment() -> Self {
        Self::VERTEX.union(Self::FRAGMENT)
    }
}

impl ColorWrites {
    /// All color channels (RGB) without alpha.
    #[must_use]
    pub const fn color() -> Self {
        Self::RED.union(Self::GREEN).union(Self::BLUE)
    }
}
