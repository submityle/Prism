//! P3-a object/texture-space shade-cache foundation.
//!
//! This is deliberately separate from the existing world-space surfel cache.
//! It owns the contracts needed by a future writable VT atlas pass: stable
//! texel-space keys, deterministic visibility-driven planning, generation and
//! epoch validation, and an active/staging page-table swap. No page is
//! sampleable until the entire staging table is published.

mod key;
mod plan;
mod residency;
mod store;

pub use key::{ShadeChannel, ShadePageKey};
pub use plan::{build_shade_plan, ShadePlan, ShadePlanItem};
pub use residency::{
    ShadePageLookup, ShadePageRequest, ShadePageState, ShadeResidencyError, ShadeResidencyTable,
    ShadeWriteTicket,
};
pub use store::{
    ShadeFrameToken, ShadePageHandle, ShadePageStore, ShadePageWrite, ShadeStoreError,
    ShadeStoreLookup,
};

#[cfg(test)]
mod tests {
    use super::*;

    fn key(x: u16, y: u16) -> ShadePageKey {
        ShadePageKey {
            object_id: 4,
            material_id: 9,
            material_epoch: 3,
            face: 1,
            mip: 0,
            channel: ShadeChannel::LitDiffuse,
            page_x: x,
            page_y: y,
        }
    }

    fn request(key: ShadePageKey, priority: u64, byte_cost: u64) -> ShadePageRequest {
        ShadePageRequest {
            key,
            priority,
            byte_cost,
            frame: 1,
        }
    }

    #[test]
    fn coarsening_preserves_identity_and_saturates_page_coordinates() {
        let fine = key(u16::MAX, 0x8001);

        let coarse = fine.coarsen(3);
        assert_eq!(coarse.object_id, fine.object_id);
        assert_eq!(coarse.material_id, fine.material_id);
        assert_eq!(coarse.material_epoch, fine.material_epoch);
        assert_eq!(coarse.face, fine.face);
        assert_eq!(coarse.channel, fine.channel);
        assert_eq!(coarse.mip, 3);
        assert_eq!(coarse.page_x, u16::MAX >> 3);
        assert_eq!(coarse.page_y, 0x8001 >> 3);

        let beyond_coordinate_width = fine.coarsen(u16::BITS as u8);
        assert_eq!(beyond_coordinate_width.page_x, 0);
        assert_eq!(beyond_coordinate_width.page_y, 0);
        assert_eq!(beyond_coordinate_width.mip, u16::BITS as u8);

        let saturated_mip = fine.coarsen(u8::MAX);
        assert_eq!(saturated_mip.mip, u8::MAX);
        assert_eq!(saturated_mip.page_x, 0);
        assert_eq!(saturated_mip.page_y, 0);
    }

    #[test]
    fn residency_requires_request_and_publishes_generation() {
        let mut table = ShadeResidencyTable::new();
        let page = key(2, 3);
        assert_eq!(
            table.begin_write(page, 3),
            Err(ShadeResidencyError::UnknownPage)
        );
        table.request(request(page, 10, 64), 3);
        let ticket = table.begin_write(page, 3).expect("requested page writes");
        assert!(matches!(table.state(page), ShadePageState::Writing { .. }));
        assert_eq!(table.lookup(page, 3), ShadePageLookup::Fallback);
        table
            .commit_write(ticket, 2)
            .expect("matching ticket commits");
        assert!(matches!(
            table.lookup(page, 3),
            ShadePageLookup::Ready { .. }
        ));
        assert_eq!(
            table.commit_write(ticket, 3),
            Err(ShadeResidencyError::InvalidWrite)
        );
    }

    #[test]
    fn stale_page_never_reports_ready_for_a_new_epoch() {
        let mut table = ShadeResidencyTable::new();
        let page = key(0, 0);
        table.request(request(page, 1, 16), 3);
        let ticket = table.begin_write(page, 3).unwrap();
        table.commit_write(ticket, 1).unwrap();
        table.invalidate(page, 4).unwrap();
        assert!(matches!(
            table.lookup(page, 4),
            ShadePageLookup::Stale { .. }
        ));
        assert!(matches!(
            table.lookup(page, 3),
            ShadePageLookup::Stale { .. }
        ));
    }

    #[test]
    fn shade_plan_is_priority_then_key_deterministic_and_budgeted() {
        let mut table = ShadeResidencyTable::new();
        let requests = [
            request(key(2, 0), 5, 32),
            request(key(0, 0), 9, 32),
            request(key(1, 0), 9, 32),
        ];
        let plan = build_shade_plan(&mut table, requests.into_iter().rev(), 3, 64);
        assert_eq!(plan.selected_bytes, 64);
        assert_eq!(
            plan.selected
                .iter()
                .map(|item| item.key.page_x)
                .collect::<Vec<_>>(),
            vec![0, 1]
        );
        assert_eq!(plan.deferred[0].key.page_x, 2);
    }

    #[test]
    fn duplicate_requests_are_order_independent_and_reserve_worst_case_cost() {
        let page = key(0, 0);
        let requests = [request(page, 3, 32), request(page, 9, 64)];
        let mut forward = ShadeResidencyTable::new();
        let mut reverse = ShadeResidencyTable::new();
        let a = build_shade_plan(&mut forward, requests, 3, 63);
        let b = build_shade_plan(&mut reverse, requests.into_iter().rev(), 3, 63);

        assert_eq!(a, b);
        assert!(a.selected.is_empty());
        assert_eq!(a.deferred[0].priority, 9);
        assert_eq!(a.deferred[0].byte_cost, 64);
    }

    #[test]
    fn request_metadata_is_refreshed_each_frame_and_old_requests_are_ignored() {
        let page = key(1, 0);
        let mut table = ShadeResidencyTable::new();
        table.request(
            ShadePageRequest {
                frame: 7,
                ..request(page, 100, 64)
            },
            3,
        );
        table.request(
            ShadePageRequest {
                frame: 8,
                ..request(page, 2, 16)
            },
            3,
        );
        table.request(
            ShadePageRequest {
                frame: 7,
                ..request(page, 1000, 128)
            },
            3,
        );
        let plan = build_shade_plan(&mut table, [], 3, 16);

        assert_eq!(plan.selected.len(), 1);
        assert_eq!(plan.selected[0].priority, 2);
        assert_eq!(plan.selected[0].byte_cost, 16);
    }

    #[test]
    fn request_with_mismatched_material_epoch_is_ignored() {
        let page = key(0, 0);
        let mut table = ShadeResidencyTable::new();
        table.request(request(page, 10, 16), 4);
        assert_eq!(table.state(page), ShadePageState::Missing);
        assert!(build_shade_plan(&mut table, [], 4, 16).selected.is_empty());
    }

    #[test]
    fn staging_writes_are_not_visible_until_publish_and_old_generation_survives() {
        let mut store = ShadePageStore::new(2);
        let page = key(0, 0);
        let first_frame = store.begin_frame(3).unwrap();
        let first = store.begin_write(first_frame, page, 1).unwrap();
        store.commit_write(first).unwrap();
        assert_eq!(store.lookup(page, 3), ShadeStoreLookup::Fallback);
        store.publish(first_frame).unwrap();
        let old = match store.lookup(page, 3) {
            ShadeStoreLookup::Ready(handle) => handle,
            ShadeStoreLookup::Fallback => panic!("published page missing"),
        };

        let second_frame = store.begin_frame(3).unwrap();
        let second = store.begin_write(second_frame, page, 2).unwrap();
        store.commit_write(second).unwrap();
        assert_eq!(store.lookup(page, 3), ShadeStoreLookup::Ready(old));
        assert_eq!(store.lookup(page, 4), ShadeStoreLookup::Fallback);
        store.publish(second_frame).unwrap();
        assert_eq!(
            store.lookup(page, 3),
            ShadeStoreLookup::Ready(second.handle)
        );
        assert_ne!(old.generation, second.handle.generation);
    }

    #[test]
    fn replacing_a_published_page_uses_a_different_slot() {
        let mut store = ShadePageStore::new(2);
        let page = key(0, 0);

        let first_frame = store.begin_frame(3).unwrap();
        let first = store.begin_write(first_frame, page, 1).unwrap();
        store.commit_write(first).unwrap();
        store.publish(first_frame).unwrap();

        let second_frame = store.begin_frame(3).unwrap();
        let second = store.begin_write(second_frame, page, 2).unwrap();
        assert_ne!(first.handle.slot, second.handle.slot);
        assert_eq!(store.lookup(page, 3), ShadeStoreLookup::Ready(first.handle));
        store.commit_write(second).unwrap();
        store.publish(second_frame).unwrap();

        assert_eq!(
            store.lookup(page, 3),
            ShadeStoreLookup::Ready(second.handle)
        );
    }

    #[test]
    fn capacity_shortage_keeps_the_active_generation_untouched() {
        let mut store = ShadePageStore::new(1);
        let page = key(0, 0);

        let first_frame = store.begin_frame(3).unwrap();
        let first = store.begin_write(first_frame, page, 1).unwrap();
        store.commit_write(first).unwrap();
        store.publish(first_frame).unwrap();

        let second_frame = store.begin_frame(3).unwrap();
        assert_eq!(
            store.begin_write(second_frame, page, 2),
            Err(ShadeStoreError::CapacityExhausted)
        );
        assert_eq!(store.lookup(page, 3), ShadeStoreLookup::Ready(first.handle));
        store.publish(second_frame).unwrap();
        assert_eq!(store.lookup(page, 3), ShadeStoreLookup::Ready(first.handle));
    }

    #[test]
    fn only_one_write_lease_per_page_can_be_open() {
        let mut store = ShadePageStore::new(3);
        let page = key(0, 0);
        let frame = store.begin_frame(3).unwrap();
        let write = store.begin_write(frame, page, 1).unwrap();

        assert_eq!(
            store.begin_write(frame, page, 2),
            Err(ShadeStoreError::WriteAlreadyOpen)
        );
        store.commit_write(write).unwrap();
        assert_eq!(
            store.begin_write(frame, page, 2),
            Err(ShadeStoreError::WriteAlreadyCommitted)
        );
        store.publish(frame).unwrap();
    }

    #[test]
    fn default_residency_table_starts_generations_at_one() {
        let mut table = ShadeResidencyTable::default();
        let page = key(2, 2);
        table.request(request(page, 1, 16), 3);
        let ticket = table.begin_write(page, 3).unwrap();
        assert_eq!(ticket.generation, 1);
    }
}
