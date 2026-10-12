//! Atomic POD records: plain-old-data that can be updated word-by-word from
//! multiple threads.
//!
//! Some renderer structures — GPU scene instance/transform/bounds rows — are
//! written from helper threads without shipping the new value over a channel
//! first. This module provides the Prism-side machinery for that: an
//! [`AtomicPod`] record pairs with a generated *blob* of
//! [`AtomicU32`](core::sync::atomic::AtomicU32) words of the same bit pattern,
//! so a value can be read and written a word at a time.
//!
//! It is a Prism replacement for the equivalent `bevy_render` utility, so the
//! retained-scene rows can drop their `bevy_render` dependency. As with any
//! word-by-word atomic update, it is memory-safe but not data-race-free: a
//! reader may observe a partially-updated value, so use it only where measured
//! performance justifies it.

use bytemuck::Pod;

pub use bytemuck;

/// A blob: a transparent wrapper around `[AtomicU32; N]` with the same bit
/// pattern as some [`AtomicPod`] record.
///
/// # Invariant
///
/// Implementors must be `#[repr(transparent)]` wrappers around
/// `[AtomicU32; N]`. The trait is safe here (this crate forbids `unsafe`), which
/// means the invariant is on the implementor: the generated implementations from
/// [`impl_atomic_pod!`](crate::impl_atomic_pod) satisfy it, and hand-written
/// ones must too. Note that viewing a blob as raw bytes for GPU upload is *not*
/// sound through `bytemuck` and is deliberately not provided; upload the
/// [`AtomicPod`] value itself instead.
pub trait AtomicPodBlob: Default + Send + Sync + 'static {}

/// Plain-old-data that can be converted to and from an array of
/// [`AtomicU32`](core::sync::atomic::AtomicU32) words.
///
/// Implement it with [`impl_atomic_pod!`](crate::impl_atomic_pod) rather than by
/// hand.
pub trait AtomicPod: Pod + Default + Send + Sync + 'static {
    /// The blob type that allows shared mutation.
    type Blob: AtomicPodBlob;

    /// Reads this value from `blob`, word by word.
    fn read_from_blob(blob: &Self::Blob) -> Self;

    /// Writes `self` into `blob`, word by word.
    ///
    /// Takes `&blob` (not `&mut`) because the words are atomics.
    fn write_to_blob(&self, blob: &Self::Blob);
}

/// Generates a blob type plus the [`AtomicPod`] implementation for a POD
/// record, and optional per-field accessor/mutator methods on the blob.
///
/// ```ignore
/// #[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
/// #[repr(C)]
/// struct Row { a: u32, b: u32 }
/// prism_render_architecture::impl_atomic_pod!(
///     Row,
///     RowBlob,
///     field(a: u32, a, set_a),
///     field(b: u32, b, set_b),
/// );
/// ```
#[macro_export]
macro_rules! impl_atomic_pod {
    (
        $pod_ty: ty,
        $blob_ty: ident
        $(, field($field_name: ident : $field_ty: ty, $getter: ident $(, $($setter: ident)?)?))*
        $(,)?
    ) => {
        #[repr(transparent)]
        pub struct $blob_ty(
            pub [::core::sync::atomic::AtomicU32; ::core::mem::size_of::<$pod_ty>() / 4],
        );

        impl ::core::ops::Deref for $blob_ty {
            type Target = [::core::sync::atomic::AtomicU32; ::core::mem::size_of::<$pod_ty>() / 4];

            fn deref(&self) -> &Self::Target {
                &self.0
            }
        }

        impl ::core::ops::DerefMut for $blob_ty {
            fn deref_mut(&mut self) -> &mut Self::Target {
                &mut self.0
            }
        }

        // `#[derive(Default)]` cannot be used for arrays longer than 32.
        impl Default for $blob_ty {
            fn default() -> Self {
                Self(
                    [const { ::core::sync::atomic::AtomicU32::new(0) };
                        ::core::mem::size_of::<$pod_ty>() / 4],
                )
            }
        }

        impl $crate::atomic_pod::AtomicPod for $pod_ty {
            type Blob = $blob_ty;

            fn read_from_blob(blob: &Self::Blob) -> Self {
                const _ASSERT_POD_SIZE_IS_WORD_MULTIPLE: () =
                    ::core::assert!(::core::mem::size_of::<$pod_ty>() % 4 == 0);

                let words: [u32; ::core::mem::size_of::<$pod_ty>() / 4] =
                    ::core::array::from_fn(|i| {
                        blob.0[i].load(::core::sync::atomic::Ordering::Relaxed)
                    });
                $crate::atomic_pod::bytemuck::must_cast(words)
            }

            fn write_to_blob(&self, blob: &Self::Blob) {
                let words: [u32; ::core::mem::size_of::<$pod_ty>() / 4] =
                    $crate::atomic_pod::bytemuck::must_cast(*self);
                for (destination, source) in blob.0.iter().zip(words.iter()) {
                    destination.store(*source, ::core::sync::atomic::Ordering::Relaxed);
                }
            }
        }

        // The blob is a `#[repr(transparent)]` wrapper around `[AtomicU32; N]`,
        // satisfying `AtomicPodBlob`'s invariant.
        impl $crate::atomic_pod::AtomicPodBlob for $blob_ty {}

        impl<'a> ::core::convert::From<&'a $pod_ty> for $blob_ty {
            fn from(pod: &'a $pod_ty) -> Self {
                let blob = <Self as Default>::default();
                $crate::atomic_pod::AtomicPod::write_to_blob(pod, &blob);
                blob
            }
        }

        impl $blob_ty {
            $(
                $(
                    pub fn $getter(&self) -> $field_ty {
                        const _ASSERT_FIELD_IS_WORD_MULTIPLE: () =
                            ::core::assert!(::core::mem::size_of::<$field_ty>() % 4 == 0);

                        let words: [u32; ::core::mem::size_of::<$field_ty>() / 4] =
                            ::core::array::from_fn(|i| {
                                self.0[::core::mem::offset_of!($pod_ty, $field_name) / 4 + i]
                                    .load(::core::sync::atomic::Ordering::Relaxed)
                            });
                        *$crate::atomic_pod::bytemuck::must_cast_ref(&words)
                    }

                    $(
                        pub fn $setter(&self, value: $field_ty) {
                            let words: [u32; ::core::mem::size_of::<$field_ty>() / 4] =
                                $crate::atomic_pod::bytemuck::must_cast(value);
                            for i in 0..(::core::mem::size_of::<$field_ty>() / 4) {
                                self.0[::core::mem::offset_of!($pod_ty, $field_name) / 4 + i]
                                    .store(words[i], ::core::sync::atomic::Ordering::Relaxed);
                            }
                        }
                    )?
                )*
            )?
        }
    };
}

#[cfg(test)]
mod tests {
    use super::{AtomicPod, bytemuck::{Pod, Zeroable}};

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
    #[repr(C)]
    struct Row {
        a: u32,
        b: u32,
    }

    crate::impl_atomic_pod!(Row, RowBlob, field(a: u32, a, set_a), field(b: u32, b, set_b));

    #[test]
    fn round_trips_a_record_through_its_blob() {
        let row = Row { a: 7, b: 0xDEAD_BEEF };
        let blob = RowBlob::from(&row);
        assert_eq!(Row::read_from_blob(&blob), row);
    }

    #[test]
    fn field_accessors_touch_only_their_words() {
        let blob = RowBlob::default();
        blob.set_a(11);
        blob.set_b(22);
        assert_eq!(blob.a(), 11);
        assert_eq!(blob.b(), 22);
        assert_eq!(Row::read_from_blob(&blob), Row { a: 11, b: 22 });
    }

    #[test]
    fn blob_is_a_transparent_word_array() {
        let blob = RowBlob::default();
        assert_eq!(blob.len(), size_of::<Row>() / 4);
    }
}
