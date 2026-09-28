//! Keyframe + sparse-delta trajectory tracks for baked physics.
//!
//! A track stores the quantized position history of one contiguous set of
//! particles across all baked frames. Storing every frame in full would be
//! wasteful because most particles move only slightly between adjacent frames,
//! so the track keeps periodic **keyframes** (a full integer snapshot) and
//! encodes the frames in between as **sparse deltas**: only the particles whose
//! quantized index actually changed are recorded, as `(index, delta)` pairs.
//!
//! Reconstruction is exact: to recover frame `f`, start from the keyframe at or
//! before `f` and replay each intervening delta. Because the deltas are the
//! difference of quantized integers, the reconstructed integers are bit-equal
//! to what was pushed, which keeps the whole cache deterministic.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. Keyframe
//! plus delta compression is a standard time-series / animation encoding.

/// Minimum allowed keyframe interval. A value of 1 makes every frame a
/// keyframe (no delta compression), which is the densest legal track.
pub const MIN_KEYFRAME_INTERVAL: u32 = 1;

/// One recorded frame within a [`CacheTrack`].
///
/// The first frame and every keyframe-interval-th frame are stored as a full
/// [`FrameRecord::Key`] snapshot; all other frames store only the particles
/// that moved since the previous frame as a [`FrameRecord::Delta`].
#[derive(Clone, PartialEq, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum FrameRecord {
    /// A full snapshot of every particle's quantized position.
    Key(Vec<[i32; 3]>),
    /// A sparse list of `(particle_index, quantized_delta)` pairs describing the
    /// change from the previous frame. Particles absent from the list did not
    /// move on the quantization grid.
    Delta(Vec<(u32, [i32; 3])>),
}

/// The quantized position history of one particle group across baked frames.
///
/// Build a track incrementally with [`CacheTrack::push_frame`]; frame 0 and
/// every [`keyframe_interval`](Self::keyframe_interval)-th frame become
/// keyframes and the rest become sparse deltas. Recover any frame with
/// [`CacheTrack::reconstruct`].
#[derive(Clone, PartialEq, Debug, Default)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct CacheTrack {
    /// Number of particles in every frame of this track.
    particle_count: u32,
    /// Distance (in frames) between successive keyframes. Always at least
    /// [`MIN_KEYFRAME_INTERVAL`].
    keyframe_interval: u32,
    /// The recorded frames in capture order.
    frames: Vec<FrameRecord>,
    /// The most recently pushed quantized snapshot, used to diff the next
    /// frame. Not serialized: it is a build-time scratch value that is empty on
    /// a freshly loaded track.
    #[cfg_attr(feature = "serialize", serde(skip))]
    last_quantized: Vec<[i32; 3]>,
}

impl CacheTrack {
    /// Creates an empty track for `particle_count` particles that emits a
    /// keyframe every `keyframe_interval` frames.
    ///
    /// A `keyframe_interval` below [`MIN_KEYFRAME_INTERVAL`] is clamped up, so a
    /// value of 0 becomes 1 (every frame is a keyframe).
    #[must_use]
    pub fn new(particle_count: u32, keyframe_interval: u32) -> CacheTrack {
        CacheTrack {
            particle_count,
            keyframe_interval: keyframe_interval.max(MIN_KEYFRAME_INTERVAL),
            frames: Vec::new(),
            last_quantized: Vec::new(),
        }
    }

    /// Returns the number of particles in each frame.
    #[must_use]
    pub fn particle_count(&self) -> u32 {
        self.particle_count
    }

    /// Returns the keyframe interval in frames.
    #[must_use]
    pub fn keyframe_interval(&self) -> u32 {
        self.keyframe_interval
    }

    /// Returns the number of frames recorded so far.
    #[must_use]
    pub fn frame_count(&self) -> u32 {
        self.frames.len() as u32
    }

    /// Returns `true` when no frames have been recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Returns `true` when `frame` is a keyframe index for this track.
    #[must_use]
    fn is_keyframe_index(&self, frame: u32) -> bool {
        frame == 0 || frame.is_multiple_of(self.keyframe_interval)
    }

    /// Appends one frame of quantized positions to the track.
    ///
    /// Frame 0 and every keyframe-interval-th frame are stored in full; the
    /// rest are diffed against the previous frame and stored sparsely. The
    /// `quantized` slice must have exactly [`particle_count`](Self::particle_count)
    /// entries; a mismatched length is rejected (the frame is not recorded) and
    /// `false` is returned.
    pub fn push_frame(&mut self, quantized: &[[i32; 3]]) -> bool {
        if quantized.len() as u32 != self.particle_count {
            return false;
        }
        let index = self.frames.len() as u32;
        if self.is_keyframe_index(index) || self.last_quantized.is_empty() {
            self.frames.push(FrameRecord::Key(quantized.to_vec()));
        } else {
            let mut deltas: Vec<(u32, [i32; 3])> = Vec::new();
            for (i, (&cur, &prev)) in quantized.iter().zip(self.last_quantized.iter()).enumerate() {
                let d = [cur[0] - prev[0], cur[1] - prev[1], cur[2] - prev[2]];
                if d != [0, 0, 0] {
                    deltas.push((i as u32, d));
                }
            }
            self.frames.push(FrameRecord::Delta(deltas));
        }
        self.last_quantized.clear();
        self.last_quantized.extend_from_slice(quantized);
        true
    }

    /// Returns the keyframe index at or before `frame`.
    #[must_use]
    fn keyframe_at_or_before(&self, frame: u32) -> u32 {
        (frame / self.keyframe_interval) * self.keyframe_interval
    }

    /// Reconstructs the quantized positions of `frame`, or `None` when the
    /// frame index is out of range or the containing keyframe is missing.
    ///
    /// The method walks forward from the nearest preceding keyframe, applying
    /// each stored delta, so the returned integers exactly match what was
    /// pushed for that frame.
    #[must_use]
    pub fn reconstruct(&self, frame: u32) -> Option<Vec<[i32; 3]>> {
        if frame >= self.frame_count() {
            return None;
        }
        let key = self.keyframe_at_or_before(frame);
        let mut state = match self.frames.get(key as usize)? {
            FrameRecord::Key(positions) => positions.clone(),
            // A keyframe index must hold a Key record; a Delta here means the
            // track was constructed inconsistently and cannot be reconstructed.
            FrameRecord::Delta(_) => return None,
        };
        for f in (key + 1)..=frame {
            match self.frames.get(f as usize)? {
                FrameRecord::Key(positions) => state.clone_from(positions),
                FrameRecord::Delta(deltas) => {
                    for &(i, d) in deltas {
                        let slot = state.get_mut(i as usize)?;
                        slot[0] += d[0];
                        slot[1] += d[1];
                        slot[2] += d[2];
                    }
                }
            }
        }
        Some(state)
    }
}

/// Reports the number of raw `i32` components a naive (per-frame full) encoding
/// of `frame_count` frames of `particle_count` particles would occupy. Used by
/// callers to compute a compression ratio against the sparse track.
#[must_use]
pub fn raw_component_count(particle_count: u32, frame_count: u32) -> u64 {
    u64::from(particle_count) * u64::from(frame_count) * 3
}

/// The number of `i32` components stored per particle per frame (x, y, z).
pub const COMPONENTS_PER_PARTICLE: u32 = 3;

#[cfg(test)]
mod tests {
    use super::*;

    fn q(x: i32, y: i32, z: i32) -> [i32; 3] {
        [x, y, z]
    }

    #[test]
    fn zero_interval_clamps_to_one() {
        let t = CacheTrack::new(2, 0);
        assert_eq!(t.keyframe_interval(), MIN_KEYFRAME_INTERVAL);
    }

    #[test]
    fn wrong_length_frame_is_rejected() {
        let mut t = CacheTrack::new(2, 4);
        assert!(!t.push_frame(&[q(0, 0, 0)]));
        assert_eq!(t.frame_count(), 0);
    }

    #[test]
    fn reconstruct_matches_pushed_frames() {
        let mut t = CacheTrack::new(2, 3);
        let frames = vec![
            vec![q(0, 0, 0), q(10, 0, 0)],
            vec![q(0, 1, 0), q(10, 1, 0)],
            vec![q(0, 2, 0), q(11, 1, 0)],
            vec![q(0, 3, 0), q(11, 1, 0)],
            vec![q(0, 4, 0), q(12, 1, 0)],
        ];
        for f in &frames {
            assert!(t.push_frame(f));
        }
        for (i, expected) in frames.iter().enumerate() {
            assert_eq!(
                t.reconstruct(i as u32).as_deref(),
                Some(expected.as_slice())
            );
        }
        assert_eq!(t.reconstruct(frames.len() as u32), None);
    }

    #[test]
    fn intermediate_frames_are_sparse_deltas() {
        let mut t = CacheTrack::new(3, 4);
        t.push_frame(&[q(0, 0, 0), q(0, 0, 0), q(0, 0, 0)]);
        // Only particle 1 moves: the delta record should hold a single pair.
        t.push_frame(&[q(0, 0, 0), q(0, 5, 0), q(0, 0, 0)]);
        match &t.frames[1] {
            FrameRecord::Delta(d) => {
                assert_eq!(d.len(), 1);
                assert_eq!(d[0], (1, [0, 5, 0]));
            }
            FrameRecord::Key(_) => panic!("frame 1 should be a delta"),
        }
    }
}
