//! Bounded, host-neutral instance-frame layout and byte storage.

use alloc::{vec, vec::Vec};
use core::fmt;

use crate::BindingSlot;

/// A frame field's validated byte layout.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FrameField {
    /// Stable binding slot represented by this field.
    pub slot: BindingSlot,
    /// Byte offset from the beginning of the frame.
    pub offset: usize,
    /// Field size in bytes.
    pub size: usize,
    /// Required alignment in bytes.
    pub align: usize,
}

/// A compiled, deterministic layout for an instance frame.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FrameLayout {
    fields: Vec<FrameField>,
    size: usize,
    align: usize,
}

/// A checked byte-backed instance frame.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InstanceFrame {
    layout: FrameLayout,
    bytes: Vec<u8>,
}

/// Failure while compiling or accessing a frame layout.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FrameError {
    /// A field has zero size or zero/non-power-of-two alignment.
    InvalidField {
        /// The invalid field's slot.
        slot: BindingSlot,
    },
    /// Two fields use one slot.
    DuplicateSlot(BindingSlot),
    /// Arithmetic overflow occurred while calculating the layout.
    Overflow,
    /// A requested slot is absent from the layout.
    MissingSlot(BindingSlot),
    /// A byte range does not fit inside the selected field.
    OutOfBounds,
}

impl fmt::Display for FrameError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidField { slot } => write!(f, "invalid frame field at slot {}", slot.0),
            Self::DuplicateSlot(slot) => write!(f, "duplicate frame slot {}", slot.0),
            Self::Overflow => f.write_str("frame layout arithmetic overflow"),
            Self::MissingSlot(slot) => write!(f, "frame slot {} is not declared", slot.0),
            Self::OutOfBounds => f.write_str("frame access is out of bounds"),
        }
    }
}

fn align_up(value: usize, align: usize) -> Option<usize> {
    value
        .checked_add(align - 1)
        .map(|value| value & !(align - 1))
}

impl FrameLayout {
    /// Compiles a layout from `(slot, size, align)` declarations.
    pub fn compile(mut declarations: Vec<(BindingSlot, usize, usize)>) -> Result<Self, FrameError> {
        declarations.sort_by_key(|(slot, _, _)| *slot);
        let mut fields = Vec::with_capacity(declarations.len());
        let mut offset = 0;
        let mut layout_align = 1;
        for (slot, size, align) in declarations {
            if size == 0 || align == 0 || !align.is_power_of_two() {
                return Err(FrameError::InvalidField { slot });
            }
            if fields
                .last()
                .is_some_and(|field: &FrameField| field.slot == slot)
            {
                return Err(FrameError::DuplicateSlot(slot));
            }
            offset = align_up(offset, align).ok_or(FrameError::Overflow)?;
            let end = offset.checked_add(size).ok_or(FrameError::Overflow)?;
            fields.push(FrameField {
                slot,
                offset,
                size,
                align,
            });
            offset = end;
            layout_align = layout_align.max(align);
        }
        let size = align_up(offset, layout_align).ok_or(FrameError::Overflow)?;
        Ok(Self {
            fields,
            size,
            align: layout_align,
        })
    }

    /// Returns the total frame size including trailing padding.
    #[must_use]
    pub const fn size(&self) -> usize {
        self.size
    }

    /// Returns the maximum field alignment.
    #[must_use]
    pub const fn align(&self) -> usize {
        self.align
    }

    /// Returns fields in stable slot order.
    #[must_use]
    pub fn fields(&self) -> &[FrameField] {
        &self.fields
    }

    fn field(&self, slot: BindingSlot) -> Result<FrameField, FrameError> {
        self.fields
            .binary_search_by_key(&slot, |field| field.slot)
            .map(|index| self.fields[index])
            .map_err(|_| FrameError::MissingSlot(slot))
    }
}

impl InstanceFrame {
    /// Allocates a zero-initialized frame for a compiled layout.
    #[must_use]
    pub fn new(layout: FrameLayout) -> Self {
        let bytes = vec![0; layout.size];
        Self { layout, bytes }
    }

    /// Returns the frame layout.
    #[must_use]
    pub const fn layout(&self) -> &FrameLayout {
        &self.layout
    }

    /// Returns the raw frame bytes for serialization or hashing.
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Reads a checked byte range from a slot.
    pub fn read(
        &self,
        slot: BindingSlot,
        range: core::ops::Range<usize>,
    ) -> Result<&[u8], FrameError> {
        let field = self.layout.field(slot)?;
        let end = range.end;
        if range.start > end || end > field.size {
            return Err(FrameError::OutOfBounds);
        }
        Ok(&self.bytes[field.offset + range.start..field.offset + end])
    }

    /// Writes a checked byte range into a slot.
    pub fn write(
        &mut self,
        slot: BindingSlot,
        range: core::ops::Range<usize>,
        value: &[u8],
    ) -> Result<(), FrameError> {
        let field = self.layout.field(slot)?;
        let end = range.end;
        if range.start > end || end > field.size || value.len() != end - range.start {
            return Err(FrameError::OutOfBounds);
        }
        self.bytes[field.offset + range.start..field.offset + end].copy_from_slice(value);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{FrameError, FrameLayout, InstanceFrame};
    use crate::BindingSlot;
    use alloc::vec;

    #[test]
    fn compiles_aligned_layout_and_checked_access() {
        let layout =
            FrameLayout::compile(vec![(BindingSlot(2), 4, 4), (BindingSlot(0), 1, 1)]).unwrap();
        assert_eq!(layout.fields()[0].slot, BindingSlot(0));
        assert_eq!(layout.fields()[1].offset, 4);
        let mut frame = InstanceFrame::new(layout);
        frame.write(BindingSlot(2), 0..4, &[1, 2, 3, 4]).unwrap();
        assert_eq!(frame.read(BindingSlot(2), 1..3).unwrap(), &[2, 3]);
        assert_eq!(
            frame.write(BindingSlot(2), 0..3, &[1, 2]),
            Err(FrameError::OutOfBounds)
        );
    }

    #[test]
    fn rejects_invalid_and_duplicate_declarations() {
        assert!(matches!(
            FrameLayout::compile(vec![(BindingSlot(1), 0, 1)]),
            Err(FrameError::InvalidField { .. })
        ));
        assert_eq!(
            FrameLayout::compile(vec![(BindingSlot(1), 1, 1), (BindingSlot(1), 2, 2)]),
            Err(FrameError::DuplicateSlot(BindingSlot(1)))
        );
    }
}
