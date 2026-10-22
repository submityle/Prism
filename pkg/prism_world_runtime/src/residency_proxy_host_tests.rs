use super::*;
use alloc::vec;
use alloc::vec::Vec;
use prism_world::{ResidencyTransition, SceneHost};
use prism_world_gpu::{ProxyInstance, ProxyRegistry, PROXY_FLAG_DISABLED};
use prism_world_residency::ResidencyState;
use prism_world_stream::SectorId;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Failure;

impl core::fmt::Display for Failure {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("injected transition failure")
    }
}

impl std::error::Error for Failure {}

#[derive(Default)]
struct Host {
    transitions: Vec<(u64, ResidencyState, ResidencyState)>,
    fail_id: Option<u64>,
    unloaded: Vec<SectorId>,
}

impl prism_world::SceneHost for Host {
    type Error = Failure;

    fn load_cell(&mut self, _: SectorId) -> Result<(), Self::Error> {
        Ok(())
    }

    fn unload_cell(&mut self, sector: SectorId) -> Result<(), Self::Error> {
        self.unloaded.push(sector);
        Ok(())
    }

    fn transition_residency(
        &mut self,
        id: u64,
        from: ResidencyState,
        to: ResidencyState,
    ) -> Result<(), Self::Error> {
        if self.fail_id == Some(id) {
            return Err(Failure);
        }
        self.transitions.push((id, from, to));
        Ok(())
    }
}

fn sector() -> SectorId {
    SectorId::new(0, 4, 0, 0)
}

fn proxy(flags: u32) -> ProxyInstance {
    let mut proxy = ProxyInstance::new(7, 1, [0.0; 3], 1.0, 1.0, 100.0);
    proxy.flags = flags;
    proxy
}

#[test]
fn residency_binding_and_transition_survive_short_lived_decorators() {
    let mut registry = ProxyRegistry::new(2).unwrap();
    registry.insert_sector(sector(), &[proxy(0)]).unwrap();
    let mut host = Host::default();
    {
        let mut decorator = ResidencyProxySceneHost::new(&mut host, &mut registry);
        decorator.bind_residency_proxy(91, sector(), 7).unwrap();
    }
    {
        let mut decorator = ResidencyProxySceneHost::new(&mut host, &mut registry);
        decorator
            .transition_residency(91, ResidencyState::TierC, ResidencyState::TierB)
            .unwrap();
        assert_ne!(
            decorator.proxy_for_residency(91).unwrap().flags & PROXY_FLAG_DISABLED,
            0
        );
    }
    {
        let mut decorator = ResidencyProxySceneHost::new(&mut host, &mut registry);
        decorator
            .transition_residency(91, ResidencyState::TierB, ResidencyState::TierC)
            .unwrap();
        assert_eq!(
            decorator.proxy_for_residency(91).unwrap().flags & PROXY_FLAG_DISABLED,
            0
        );
    }
    assert_eq!(host.transitions.len(), 2);
}

#[test]
fn transition_failure_preserves_proxy_state_and_batch_reports_prefix() {
    let mut registry = ProxyRegistry::new(2).unwrap();
    registry
        .insert_sector(
            sector(),
            &[proxy(0), {
                let mut second = proxy(0);
                second.proxy_id = 8;
                second
            }],
        )
        .unwrap();
    registry.bind_residency_proxy(1, sector(), 7).unwrap();
    registry.bind_residency_proxy(2, sector(), 8).unwrap();
    let mut host = Host {
        fail_id: Some(2),
        ..Host::default()
    };
    let mut decorator = ResidencyProxySceneHost::new(&mut host, &mut registry);
    let transitions = [
        ResidencyTransition {
            id: 1,
            from: ResidencyState::TierC,
            to: ResidencyState::TierB,
        },
        ResidencyTransition {
            id: 2,
            from: ResidencyState::TierC,
            to: ResidencyState::TierB,
        },
    ];
    let error = decorator
        .transition_residency_batch(&transitions)
        .unwrap_err();
    assert_eq!(error.completed, 1);
    assert_ne!(
        decorator.proxy_for_residency(1).unwrap().flags & PROXY_FLAG_DISABLED,
        0
    );
    assert_eq!(
        decorator.proxy_for_residency(2).unwrap().flags & PROXY_FLAG_DISABLED,
        0
    );
    assert_eq!(host.transitions.len(), 1);
}

#[test]
fn unload_clears_registry_bindings_only_after_host_success() {
    let mut registry = ProxyRegistry::new(1).unwrap();
    registry.insert_sector(sector(), &[proxy(0)]).unwrap();
    registry.bind_residency_proxy(5, sector(), 7).unwrap();
    let mut host = Host::default();
    let mut decorator = ResidencyProxySceneHost::new(&mut host, &mut registry);
    decorator.unload_cell(sector()).unwrap();
    assert!(decorator.proxy_for_residency(5).is_none());
    assert_eq!(host.unloaded, vec![sector()]);
}
