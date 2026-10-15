//! NUMA-aware virtual-memory allocation.
//!
//! A successful [`NumaAllocation`] means that the operating system accepted a
//! node-local policy for the mapping. It is never inferred from a requested
//! node id. Callers that can safely use ordinary memory may opt into
//! [`NumaAllocation::on_node_or_fallback`]; the returned placement records
//! whether the fallback was used.

use core::fmt;
use std::alloc::{self, Layout};
use std::ptr::NonNull;

use super::{page_size, Protection, Reservation, VmError};

/// Why a NUMA allocation request did not become node-bound.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NumaAllocationError {
    /// This target has no implemented NUMA memory-policy backend.
    Unsupported,
    /// The requested size or alignment is invalid.
    InvalidArgument,
    /// The target node or node mask was rejected by the operating system.
    InvalidNode,
    /// The operating system could not provide the requested mapping.
    OutOfMemory,
    /// The operating system rejected the policy request; contains its error
    /// code (`errno` on Unix or the corresponding platform code).
    SystemError(i32),
}

impl fmt::Display for NumaAllocationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsupported => f.write_str("NUMA memory binding is unsupported on this platform"),
            Self::InvalidArgument => f.write_str("invalid NUMA allocation argument"),
            Self::InvalidNode => f.write_str("requested NUMA node is invalid or unavailable"),
            Self::OutOfMemory => f.write_str("out of memory for NUMA allocation"),
            Self::SystemError(code) => write!(f, "OS rejected NUMA memory policy (code {code})"),
        }
    }
}

impl std::error::Error for NumaAllocationError {}

/// Where a [`NumaAllocation`] ended up.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NumaPlacement {
    /// The operating system accepted a policy binding for this node.
    Bound {
        /// Node accepted by the operating system.
        node: u16,
    },
    /// Ordinary memory was used after the requested NUMA policy was not
    /// available. This is explicit and must not be treated as node-local.
    Fallback {
        /// Node requested by the caller.
        requested_node: u16,
        /// The reason the binding was not applied.
        reason: NumaAllocationError,
    },
}

impl NumaPlacement {
    /// Returns `true` only when the OS accepted a node binding.
    #[must_use]
    pub const fn is_bound(self) -> bool {
        matches!(self, Self::Bound { .. })
    }
}

enum Storage {
    Reservation(Reservation),
    Heap { ptr: NonNull<u8>, layout: Layout },
}

/// Owned memory whose NUMA placement is explicit at runtime.
pub struct NumaAllocation {
    storage: Storage,
    len: usize,
    placement: NumaPlacement,
}

impl NumaAllocation {
    /// Returns whether this build has a real NUMA memory-policy backend.
    #[must_use]
    pub const fn supported() -> bool {
        backend::SUPPORTED
    }

    /// Allocate `size` bytes with the OS node-local policy.
    ///
    /// This method does not silently fall back. Use
    /// [`Self::on_node_or_fallback`] when ordinary unbound memory is an
    /// acceptable degradation.
    pub fn on_node(size: usize, align: usize, node: u16) -> Result<Self, NumaAllocationError> {
        let layout = Layout::from_size_align(size.max(1), align)
            .map_err(|_| NumaAllocationError::InvalidArgument)?;
        if size == 0 || !align.is_power_of_two() {
            return Err(NumaAllocationError::InvalidArgument);
        }
        if align > page_size() {
            return Err(NumaAllocationError::InvalidArgument);
        }
        let rounded = round_up(size, page_size()).ok_or(NumaAllocationError::InvalidArgument)?;
        let reservation = Reservation::reserve(rounded).map_err(map_vm_error)?;
        backend::bind(reservation.as_mut_ptr(), reservation.len(), node)?;
        let _ = layout;
        reservation
            .commit(0, reservation.len(), Protection::ReadWrite)
            .map_err(map_vm_error)?;
        Ok(Self {
            len: size,
            storage: Storage::Reservation(reservation),
            placement: NumaPlacement::Bound { node },
        })
    }

    /// Allocate on `node`, falling back to ordinary heap memory only when the
    /// NUMA backend is unsupported or the OS rejects the policy. The returned
    /// [`NumaPlacement::Fallback`] preserves the reason, so callers cannot
    /// accidentally claim that the allocation is node-local.
    pub fn on_node_or_fallback(
        size: usize,
        align: usize,
        node: u16,
    ) -> Result<Self, NumaAllocationError> {
        match Self::on_node(size, align, node) {
            Ok(allocation) => Ok(allocation),
            Err(
                reason @ (NumaAllocationError::Unsupported
                | NumaAllocationError::InvalidNode
                | NumaAllocationError::SystemError(_)),
            ) => {
                let layout = Layout::from_size_align(size.max(1), align)
                    .map_err(|_| NumaAllocationError::InvalidArgument)?;
                if size == 0 {
                    return Err(NumaAllocationError::InvalidArgument);
                }
                // SAFETY: `layout` is valid and the allocation is owned by the
                // returned value until its `Drop` implementation.
                #[expect(unsafe_code, reason = "allocate explicit fallback NUMA storage")]
                let raw = unsafe { alloc::alloc(layout) };
                let ptr = NonNull::new(raw).ok_or(NumaAllocationError::OutOfMemory)?;
                Ok(Self {
                    storage: Storage::Heap { ptr, layout },
                    len: size,
                    placement: NumaPlacement::Fallback {
                        requested_node: node,
                        reason,
                    },
                })
            }
            Err(error) => Err(error),
        }
    }

    /// Pointer to the first byte of the allocation.
    #[must_use]
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        match &mut self.storage {
            Storage::Reservation(reservation) => reservation.as_mut_ptr(),
            Storage::Heap { ptr, .. } => ptr.as_ptr(),
        }
    }

    /// Requested usable length in bytes. The backing mapping may be rounded up
    /// to a page boundary.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` when the allocation has zero requested length.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Runtime placement result.
    #[must_use]
    pub const fn placement(&self) -> NumaPlacement {
        self.placement
    }
}

impl Drop for NumaAllocation {
    fn drop(&mut self) {
        if let Storage::Heap { ptr, layout } = self.storage {
            // SAFETY: this is the matching deallocation for the sole owning
            // heap allocation created above.
            #[expect(unsafe_code, reason = "deallocate the owned fallback NUMA storage")]
            unsafe {
                alloc::dealloc(ptr.as_ptr(), layout)
            };
        }
    }
}

fn round_up(value: usize, align: usize) -> Option<usize> {
    value.checked_add(align - 1).map(|v| v & !(align - 1))
}

fn map_vm_error(error: VmError) -> NumaAllocationError {
    match error {
        VmError::Unsupported => NumaAllocationError::Unsupported,
        VmError::InvalidArgument => NumaAllocationError::InvalidArgument,
        VmError::OutOfMemory => NumaAllocationError::OutOfMemory,
        VmError::SystemError(code) => NumaAllocationError::SystemError(code),
    }
}

#[cfg(target_os = "linux")]
mod backend {
    use super::NumaAllocationError;
    use core::ffi::{c_int, c_long, c_ulong, c_void};

    // Linux's mbind(2) policy constants. These are stable UAPI values and are
    // kept local to avoid adding a libc dependency to the platform kernel.
    const MPOL_BIND: c_int = 2;
    const MPOL_F_STATIC_NODES: c_ulong = 1 << 15;
    const ENOMEM: i32 = 12;
    const EINVAL: i32 = 22;

    unsafe extern "C" {
        fn mbind(
            start: *mut c_void,
            len: c_ulong,
            mode: c_int,
            nodemask: *const c_ulong,
            maxnode: c_ulong,
            flags: c_ulong,
        ) -> c_long;
        fn __errno_location() -> *mut i32;
    }

    pub(super) const SUPPORTED: bool = true;

    pub(super) fn bind(ptr: *mut u8, len: usize, node: u16) -> Result<(), NumaAllocationError> {
        let bits = usize::BITS as usize;
        let word = usize::from(node) / bits;
        let bit = usize::from(node) % bits;
        let mut mask = vec![0 as c_ulong; word + 1];
        mask[word] = (1 as c_ulong) << bit;
        // SAFETY: `ptr`/`len` identify the live anonymous mapping owned by the
        // caller; `mask` remains alive for the complete syscall.
        #[expect(unsafe_code, reason = "call Linux mbind for the requested NUMA policy")]
        let rc = unsafe {
            mbind(
                ptr.cast::<c_void>(),
                len as c_ulong,
                MPOL_BIND,
                mask.as_ptr(),
                (usize::from(node) + 1) as c_ulong,
                MPOL_F_STATIC_NODES,
            )
        };
        if rc == 0 {
            Ok(())
        } else {
            // SAFETY: errno is a valid pointer for the calling thread.
            #[expect(unsafe_code, reason = "read Linux thread-local errno after mbind")]
            let errno = unsafe { *__errno_location() };
            Err(if errno == EINVAL {
                NumaAllocationError::InvalidNode
            } else if errno == ENOMEM {
                NumaAllocationError::OutOfMemory
            } else {
                NumaAllocationError::SystemError(errno)
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{NumaAllocation, NumaAllocationError, NumaPlacement};
    use crate::page_size;

    #[test]
    fn placement_never_claims_binding_for_fallback_memory() {
        let allocation = NumaAllocation::on_node_or_fallback(page_size(), 64, 0)
            .expect("ordinary fallback allocation should remain usable");
        match allocation.placement() {
            NumaPlacement::Bound { node } => assert_eq!(node, 0),
            NumaPlacement::Fallback {
                requested_node,
                reason,
            } => {
                assert_eq!(requested_node, 0);
                assert!(!allocation.placement().is_bound());
                if !NumaAllocation::supported() {
                    assert_eq!(reason, NumaAllocationError::Unsupported);
                }
            }
        }
    }

    #[test]
    fn invalid_arguments_are_not_converted_to_fallback() {
        assert!(matches!(
            NumaAllocation::on_node(0, 64, 0),
            Err(NumaAllocationError::InvalidArgument)
        ));
        assert!(matches!(
            NumaAllocation::on_node(page_size(), page_size() * 2, 0),
            Err(NumaAllocationError::InvalidArgument)
        ));
    }

    #[cfg(not(target_os = "linux"))]
    #[test]
    fn unsupported_backend_is_reported_explicitly() {
        assert!(!NumaAllocation::supported());
        assert!(matches!(
            NumaAllocation::on_node(page_size(), 64, 0),
            Err(NumaAllocationError::Unsupported)
        ));
    }
}

#[cfg(not(target_os = "linux"))]
mod backend {
    use super::NumaAllocationError;

    pub(super) const SUPPORTED: bool = false;

    pub(super) fn bind(_ptr: *mut u8, _len: usize, _node: u16) -> Result<(), NumaAllocationError> {
        Err(NumaAllocationError::Unsupported)
    }
}
