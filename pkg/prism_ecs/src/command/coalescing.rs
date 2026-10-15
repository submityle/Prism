//! Ordered structural replay. Only adjacent edits to one entity are folded;
//! arbitrary code, spawns, despawns and entity changes are replay barriers.

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::mem::MaybeUninit;

use crate::bundle::Bundle;
use crate::component::{ComponentId, ComponentSet, Components, StorageType};
use crate::entity::Entity;
use crate::world::World;

pub(super) enum Command {
    Opaque(Box<dyn FnOnce(&mut World) + Send + Sync>),
    Structural(Entity, StructuralCommand),
}

pub(super) enum StructuralCommand {
    Insert(Box<dyn InsertBundle>),
    Remove {
        ids: fn(&mut Components, &mut Vec<ComponentId>),
        apply: fn(&mut World, Entity),
    },
}

pub(super) trait InsertBundle: Send + Sync {
    fn ids(&self, components: &mut Components, ids: &mut Vec<ComponentId>);
    fn apply(self: Box<Self>, world: &mut World, entity: Entity);
    /// Transfer the bundle's values in id order to `take`.
    ///
    /// # Safety
    /// The callback must consume each value exactly once.
    unsafe fn take(self: Box<Self>, take: &mut dyn FnMut(*mut u8));
}

impl<B: Bundle> InsertBundle for B {
    fn ids(&self, components: &mut Components, ids: &mut Vec<ComponentId>) {
        B::component_ids(components, ids);
    }

    fn apply(self: Box<Self>, world: &mut World, entity: Entity) {
        world.insert(entity, *self);
    }

    unsafe fn take(self: Box<Self>, take: &mut dyn FnMut(*mut u8)) {
        // SAFETY: forwarded Bundle contract; the caller consumes each value.
        unsafe { (*self).get_components(take) };
    }
}

impl Command {
    pub(super) fn opaque(f: impl FnOnce(&mut World) + Send + Sync + 'static) -> Self {
        Self::Opaque(Box::new(f))
    }

    pub(super) fn insert<B: Bundle>(entity: Entity, bundle: B) -> Self {
        Self::Structural(entity, StructuralCommand::Insert(Box::new(bundle)))
    }

    pub(super) fn remove<B: Bundle>(entity: Entity) -> Self {
        Self::Structural(
            entity,
            StructuralCommand::Remove {
                ids: B::component_ids,
                apply: |world, entity| {
                    world.remove::<B>(entity);
                },
            },
        )
    }
}

impl StructuralCommand {
    fn ids(&self, world: &mut World) -> Vec<ComponentId> {
        let mut ids = Vec::new();
        match self {
            Self::Insert(bundle) => bundle.ids(world.components_mut(), &mut ids),
            Self::Remove { ids: collect, .. } => collect(world.components_mut(), &mut ids),
        }
        ids
    }

    fn apply(self, world: &mut World, entity: Entity) {
        match self {
            Self::Insert(bundle) => bundle.apply(world, entity),
            Self::Remove { apply, .. } => apply(world, entity),
        }
    }
}

/// One component's final state. Byte buffers only contain drop-free values;
/// storage copies their bytes into an aligned column before interpreting them.
pub(crate) struct CoalescedChange {
    pub(crate) id: ComponentId,
    pub(crate) value: Option<Vec<MaybeUninit<u8>>>,
    /// A remove preceding the final insert starts a new Added<T> lifetime.
    pub(crate) removed: bool,
}

fn flush(world: &mut World, pending: &mut Option<Entity>, changes: &mut Vec<CoalescedChange>) {
    if let Some(entity) = pending.take() {
        world.commit_coalesced_structural(entity, core::mem::take(changes));
    }
}

/// Registration and bundle extraction are user code. If either unwinds, commit
/// only the successful prefix; the current command must not enter `changes`
/// before all of its user code (including bundle destruction) returns.
struct AcceptedCommandsGuard<'a> {
    world: &'a mut World,
    pending: &'a mut Option<Entity>,
    changes: &'a mut Vec<CoalescedChange>,
    armed: bool,
}

impl Drop for AcceptedCommandsGuard<'_> {
    fn drop(&mut self) {
        if self.armed {
            // Eligible changes have no drop glue or lifecycle callbacks. Commit
            // relocates untouched owned values without dropping them, so this
            // unwind path does not invoke user code that could panic again.
            // `flush` takes `pending` before migrating: never retry a commit
            // from a guard if migration itself unwinds.
            flush(self.world, self.pending, self.changes);
        }
    }
}

/// Replay in input order. Lifecycle callbacks and constructors can inspect
/// intermediate states, and Drop can have arbitrary side effects, so all such
/// operations retain the immediate path. No command is moved across a barrier.
/// If an eligible bundle panics during extraction, earlier complete commands
/// are committed, its own staged values are discarded, and replay stops.
pub(super) fn replay(world: &mut World, commands: impl IntoIterator<Item = Command>) {
    let mut pending = None;
    let mut changes: Vec<CoalescedChange> = Vec::new();
    for command in commands {
        let Command::Structural(entity, command) = command else {
            flush(world, &mut pending, &mut changes);
            if let Command::Opaque(command) = command {
                command(world);
            }
            continue;
        };
        if pending.is_some_and(|p| p != entity) {
            flush(world, &mut pending, &mut changes);
        }

        // Check before registering future bundle types: callbacks from an
        // earlier command must not see future component registrations.
        if world.components().has_hooks()
            || world.components().has_required()
            || !world.observers.is_empty()
            || world.owning_group_count() != 0
            || world
                .entities()
                .location(entity)
                .is_none_or(|l| l.is_empty())
        {
            flush(world, &mut pending, &mut changes);
            command.apply(world, entity);
            continue;
        }

        let ids = {
            let mut guard = AcceptedCommandsGuard {
                world,
                pending: &mut pending,
                changes: &mut changes,
                armed: true,
            };
            let ids = command.ids(guard.world);
            guard.armed = false;
            ids
        };
        // Component registration may itself install hooks or requirements.
        let eligible = !world.components().has_hooks()
            && !world.components().has_required()
            && (!matches!(command, StructuralCommand::Insert(_))
                || ComponentSet::from_ids(ids.iter().copied()).len() == ids.len())
            && ids.iter().all(|id| {
                let info = world.components().info(*id).expect("registered component");
                info.storage() != StorageType::Shared && info.drop_fn().is_none()
            });
        if !eligible {
            flush(world, &mut pending, &mut changes);
            command.apply(world, entity);
            continue;
        }

        match command {
            StructuralCommand::Insert(bundle) => {
                // Keep this command isolated even if it yields every value
                // before panicking. In particular it must not overwrite an
                // earlier accepted value or remove/reinsert lifetime marker.
                let mut staged = Vec::with_capacity(ids.len());
                let mut guard = AcceptedCommandsGuard {
                    world,
                    pending: &mut pending,
                    changes: &mut changes,
                    armed: true,
                };
                let mut i = 0;
                // SAFETY: Bundle yields each owned value in ids order. Only
                // drop-free components reach here; byte-copying transfers their
                // representation and discarded intermediate values need no drop.
                unsafe {
                    bundle.take(&mut |ptr| {
                        let id = ids[i];
                        i += 1;
                        let size = guard.world.components().info(id).unwrap().layout().size();
                        let mut value = Vec::with_capacity(size);
                        core::ptr::copy_nonoverlapping(
                            ptr.cast::<MaybeUninit<u8>>(),
                            value.as_mut_ptr(),
                            size,
                        );
                        value.set_len(size);
                        staged.push(CoalescedChange {
                            id,
                            value: Some(value),
                            removed: false,
                        });
                    });
                }
                guard.armed = false;
                drop(guard);
                // No bundle code or component destructor runs while merging.
                // Staging contains only drop-free byte representations, so
                // unwinding never drops a transferred component a second time.
                pending = Some(entity);
                for change in staged {
                    if let Some(accepted) = changes.iter_mut().find(|c| c.id == change.id) {
                        accepted.value = change.value;
                    } else {
                        changes.push(change);
                    }
                }
            }
            StructuralCommand::Remove { .. } => {
                pending = Some(entity);
                for id in ids {
                    if let Some(change) = changes.iter_mut().find(|c| c.id == id) {
                        change.value = None;
                        change.removed = true;
                    } else {
                        changes.push(CoalescedChange {
                            id,
                            value: None,
                            removed: true,
                        });
                    }
                }
            }
        }
    }
    flush(world, &mut pending, &mut changes);
}
