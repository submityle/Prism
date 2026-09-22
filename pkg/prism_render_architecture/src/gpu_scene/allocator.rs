use alloc::collections::BinaryHeap;
use core::cmp::Reverse;
use core::fmt;

use super::SceneHandle;

/// A monotonically increasing GPU timeline or fence completion value.
#[derive(Clone, Copy, Debug, Default, Eq, Ord, PartialEq, PartialOrd)]
pub struct GpuCompletionValue(pub u64);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SlotState {
    Reserved,
    Free,
    Live,
    Retiring(GpuCompletionValue),
    Exhausted,
}

#[derive(Clone, Copy, Debug)]
struct Slot {
    generation: u32,
    state: SlotState,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct RetiredSlot {
    completion: GpuCompletionValue,
    index: u32,
    generation: u32,
}

/// Allocation failed because the configured stable-handle space is exhausted.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SceneCapacityError {
    pub max_slots: u32,
}

impl fmt::Display for SceneCapacityError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "GPU scene capacity of {} slots is exhausted",
            self.max_slots
        )
    }
}

impl std::error::Error for SceneCapacityError {}

/// A handle operation targeted an invalid, stale, or non-live slot.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SceneHandleError {
    Invalid,
    OutOfRange(SceneHandle),
    Stale {
        handle: SceneHandle,
        current_generation: u32,
    },
    NotLive(SceneHandle),
}

impl fmt::Display for SceneHandleError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid => formatter.write_str("invalid GPU scene handle"),
            Self::OutOfRange(handle) => {
                write!(formatter, "scene handle {handle:?} is out of range")
            }
            Self::Stale {
                handle,
                current_generation,
            } => write!(
                formatter,
                "scene handle {handle:?} is stale; slot generation is {current_generation}"
            ),
            Self::NotLive(handle) => write!(formatter, "scene handle {handle:?} is not live"),
        }
    }
}

impl std::error::Error for SceneHandleError {}

/// Current allocator state for diagnostics and soak tests.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SceneHandleStats {
    pub live: u32,
    pub retiring: u32,
    pub free: u32,
    pub exhausted: u32,
    pub allocated_slots: u32,
    pub max_slots: u32,
}

/// Allocates stable generational scene handles and delays reuse until the GPU
/// has completed every submission that can still reference a retired slot.
pub struct SceneHandleAllocator {
    slots: Vec<Slot>,
    free: Vec<u32>,
    retired: BinaryHeap<Reverse<RetiredSlot>>,
    max_slots: u32,
    live_count: u32,
    exhausted_count: u32,
}

impl SceneHandleAllocator {
    /// Creates an allocator. Slot zero is permanently reserved as the null or
    /// placeholder object, so `max_slots` must contain at least two entries.
    pub fn new(max_slots: u32) -> Self {
        assert!(
            max_slots >= 2,
            "GPU scene requires a reserved slot and one live slot"
        );
        Self {
            slots: vec![Slot {
                generation: 0,
                state: SlotState::Reserved,
            }],
            free: Vec::new(),
            retired: BinaryHeap::new(),
            max_slots,
            live_count: 0,
            exhausted_count: 0,
        }
    }

    pub fn allocate(&mut self) -> Result<SceneHandle, SceneCapacityError> {
        let index = if let Some(index) = self.free.pop() {
            index
        } else if self.slots.len() < self.max_slots as usize {
            let index = self.slots.len() as u32;
            self.slots.push(Slot {
                generation: 1,
                state: SlotState::Free,
            });
            index
        } else {
            return Err(SceneCapacityError {
                max_slots: self.max_slots,
            });
        };

        let slot = &mut self.slots[index as usize];
        debug_assert_eq!(slot.state, SlotState::Free);
        slot.state = SlotState::Live;
        self.live_count += 1;
        Ok(SceneHandle {
            index,
            generation: slot.generation,
        })
    }

    pub fn allocate_batch(&mut self, count: usize) -> Result<Vec<SceneHandle>, SceneCapacityError> {
        if count > self.available_capacity() as usize {
            return Err(SceneCapacityError {
                max_slots: self.max_slots,
            });
        }
        Ok((0..count)
            .map(|_| {
                self.allocate()
                    .expect("capacity was checked before batch allocation")
            })
            .collect())
    }

    pub fn retire(
        &mut self,
        handle: SceneHandle,
        completion: GpuCompletionValue,
    ) -> Result<(), SceneHandleError> {
        self.validate_live(handle)?;
        self.slots[handle.index as usize].state = SlotState::Retiring(completion);
        self.retired.push(Reverse(RetiredSlot {
            completion,
            index: handle.index,
            generation: handle.generation,
        }));
        self.live_count -= 1;
        Ok(())
    }

    /// Immediately releases a handle that was allocated but never published
    /// to any GPU submission.
    pub fn cancel_allocation(&mut self, handle: SceneHandle) -> Result<(), SceneHandleError> {
        self.validate_live(handle)?;
        let slot = &mut self.slots[handle.index as usize];
        slot.state = SlotState::Free;
        self.free.push(handle.index);
        self.live_count -= 1;
        Ok(())
    }

    /// Makes all slots whose GPU work has completed available for reuse.
    pub fn reclaim_completed(&mut self, completed: GpuCompletionValue) -> u32 {
        let mut reclaimed = 0;
        while let Some(Reverse(retired)) = self.retired.peek().copied() {
            if retired.completion > completed {
                break;
            }
            self.retired.pop();
            let slot = &mut self.slots[retired.index as usize];
            if slot.generation != retired.generation
                || slot.state != SlotState::Retiring(retired.completion)
            {
                continue;
            }
            if let Some(next) = slot.generation.checked_add(1) {
                slot.generation = next;
                slot.state = SlotState::Free;
                self.free.push(retired.index);
                reclaimed += 1;
            } else {
                slot.state = SlotState::Exhausted;
                self.exhausted_count += 1;
            }
        }
        reclaimed
    }

    pub fn validate(&self, handle: SceneHandle) -> bool {
        self.validate_live(handle).is_ok()
    }

    pub fn generation_at(&self, index: u32) -> Option<u32> {
        self.slots.get(index as usize).map(|slot| slot.generation)
    }

    pub fn available_capacity(&self) -> u32 {
        self.free.len() as u32 + self.max_slots - self.slots.len() as u32
    }

    pub fn stats(&self) -> SceneHandleStats {
        SceneHandleStats {
            live: self.live_count,
            retiring: self.retired.len() as u32,
            free: self.free.len() as u32,
            exhausted: self.exhausted_count,
            allocated_slots: self.slots.len() as u32,
            max_slots: self.max_slots,
        }
    }

    fn validate_live(&self, handle: SceneHandle) -> Result<(), SceneHandleError> {
        if !handle.is_valid() || handle.index == 0 {
            return Err(SceneHandleError::Invalid);
        }
        let Some(slot) = self.slots.get(handle.index as usize) else {
            return Err(SceneHandleError::OutOfRange(handle));
        };
        if slot.generation != handle.generation {
            return Err(SceneHandleError::Stale {
                handle,
                current_generation: slot.generation,
            });
        }
        if slot.state != SlotState::Live {
            return Err(SceneHandleError::NotLive(handle));
        }
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn force_generation_for_test(&mut self, handle: SceneHandle, generation: u32) {
        self.slots[handle.index as usize].generation = generation;
    }
}
