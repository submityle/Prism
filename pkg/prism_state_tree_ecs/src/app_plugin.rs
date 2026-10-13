//! `prism_app` plugin assembly for scheduled `StateTree` ticking.

use prism_app::{App, Plugin, Update};

use crate::schedule::run_scheduled_tick;

/// Installs the deterministic `StateTree` tick into the host App `Update` phase.
///
/// This is opt-in through the `app` feature. It does not create a hidden
/// runner, asset loader, or Context gather; it only registers the already
/// explicit ECS tick system at the App integration boundary.
#[derive(Clone, Copy, Debug, Default)]
pub struct StateTreePlugin;

impl Plugin for StateTreePlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, run_scheduled_tick);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::StateTreeTickReport;

    #[test]
    fn plugin_registers_one_tick_in_app_update() {
        let mut app = App::new();
        app.add_plugins(StateTreePlugin);
        app.update();
        assert_eq!(app.world().resource::<StateTreeTickReport>().run_count(), 1);
    }
}
