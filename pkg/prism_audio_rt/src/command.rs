//! Control-plane commands sent from task threads to the audio thread over the
//! [command ring](crate::ring).
//!
//! Every variant is `Copy` and holds only plain scalars and small
//! [`prism_audio_core`] value types, so the whole enum lives in the
//! pre-allocated ring without any heap ownership. Anything that needs to
//! transfer heap-allocated state (a freshly compiled graph, sample data, an
//! effect instance) is delivered out of band through the
//! [graph hand-off](crate::epoch::GraphProducer) or a dedicated resource queue,
//! never inside a command, so the audio thread never allocates or frees while
//! draining commands.
//!
//! ## Timing model
//!
//! Voice-lifecycle and configuration commands are applied at the start of the
//! block in which they are drained (block-granular control), matching how
//! mixer/bus control changes are handled by production engines. Sample-accurate
//! *musical* event timing is the job of [`prism_audio_core::scheduler`], which
//! a task thread drives directly. The one place this bridge honours a sub-block
//! sample offset is [`AudioCommand::SetMasterGain`], whose `at_frame` schedules
//! the start of a click-free master-gain ramp within the current block.

use prism_audio_core::voice::{Importance, VoiceHandle, VoiceRequest};

/// A control-plane message consumed by [`crate::runtime::AudioRuntime`].
///
/// The enum is deliberately `Copy` and free of owning pointers so it can travel
/// through the lock-free [command ring](crate::ring) without allocation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AudioCommand {
    /// Allocate a new voice from the runtime's [`VoicePool`]. Applied at the
    /// start of the draining block; the resulting handle is observable through
    /// pool queries and telemetry voice counts.
    ///
    /// [`VoicePool`]: prism_audio_core::voice::VoicePool
    SpawnVoice {
        /// The voice request forwarded verbatim to
        /// [`VoicePool::allocate`](prism_audio_core::voice::VoicePool::allocate).
        request: VoiceRequest,
    },
    /// Release a previously allocated voice. Ignored if the handle no longer
    /// resolves (already released or superseded by a newer occupant).
    StopVoice {
        /// Generation-checked handle identifying the voice to release.
        handle: VoiceHandle,
    },
    /// Update the effective importance of a live voice, influencing future
    /// stealing and virtualization decisions.
    SetVoiceImportance {
        /// Generation-checked handle identifying the voice to retune.
        handle: VoiceHandle,
        /// New effective importance value.
        importance: Importance,
    },
    /// Schedule a click-free change of the master output gain.
    SetMasterGain {
        /// Target gain as a linear multiplier (`1.0` is unity, `0.0` is
        /// silence). Sanitized to a finite, non-negative value by the runtime.
        linear: f32,
        /// Frame offset within the current block at which the ramp toward
        /// `linear` begins. Values beyond the block length are clamped to the
        /// final frame.
        at_frame: u32,
        /// Length of the linear ramp toward `linear`, in frames. `0` applies
        /// the change immediately at `at_frame`.
        ramp_frames: u32,
    },
    /// Resize the runtime's physical-voice budget, i.e. the maximum number of
    /// simultaneously audible voices before newcomers are virtualized.
    SetMaxPhysicalVoices {
        /// New physical-voice budget forwarded to
        /// [`VoicePool::set_max_physical`](prism_audio_core::voice::VoicePool::set_max_physical).
        max_physical: usize,
    },
}
