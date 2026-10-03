//! Assembly-time topological ordering of plugins by their declared
//! dependencies.
//!
//! A [`PluginGroup`](crate::plugin_group::PluginGroup) collects its members in
//! an explicit order (insertion plus `add_before` / `add_after` edits). Before
//! the group hands those members to the [`App`](crate::app::App), it runs the
//! topological sort here to reorder them so every
//! [`Plugin::dependencies`](crate::plugin::Plugin::dependencies) edge is
//! honored — a plugin always builds after the plugins it depends on.
//!
//! Both failure modes are detected *here, at assembly time*, never deferred to
//! run time (design §23 risk #5): a dependency on a plugin type absent from the
//! group ([`PluginGraphError::MissingDependency`]) and a dependency cycle
//! ([`PluginGraphError::Cycle`]).
//!
//! The sort is **stable with respect to the explicit order**: among plugins
//! that are all simultaneously eligible (no remaining unmet dependency), the
//! one that appeared earliest in the group's explicit order is emitted first.
//! A group with no dependency edges therefore comes out exactly in its explicit
//! order.

use core::any::TypeId;
use std::collections::HashMap;
use std::fmt;

use crate::plugin::PluginDependency;

/// A node fed to [`topological_order`]: one plugin's identity plus the plugin
/// types it must build after.
pub(crate) struct Node {
    /// The plugin type's own [`TypeId`].
    pub type_id: TypeId,
    /// The plugin type's name, used only for diagnostics.
    pub name: &'static str,
    /// The plugin types this plugin must build after.
    pub dependencies: Vec<PluginDependency>,
}

/// A dependency-resolution failure detected while assembling a plugin group.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PluginGraphError {
    /// A plugin declared a dependency on a plugin type that is not a member of
    /// the group (or was disabled out of it).
    MissingDependency {
        /// The plugin that declared the dependency.
        dependent: &'static str,
        /// The depended-on plugin type that is missing from the group.
        missing: &'static str,
    },
    /// The dependency edges contain a cycle. `members` lists the names of the
    /// plugins that remained unresolved (i.e. participate in or feed the
    /// cycle), in the group's explicit order.
    Cycle {
        /// Names of the plugins left unresolved by the cycle.
        members: Vec<&'static str>,
    },
}

impl fmt::Display for PluginGraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PluginGraphError::MissingDependency { dependent, missing } => write!(
                f,
                "plugin {dependent:?} depends on {missing:?}, which is not a member of the group"
            ),
            PluginGraphError::Cycle { members } => {
                write!(f, "plugin dependency cycle among {members:?}")
            }
        }
    }
}

impl std::error::Error for PluginGraphError {}

/// Topologically order `nodes` so every dependency builds before its dependent,
/// breaking ties by the input (explicit) order.
///
/// Returns the resolved order as indices into `nodes`. Fails with
/// [`PluginGraphError::MissingDependency`] if any declared dependency is not
/// present in `nodes`, or [`PluginGraphError::Cycle`] if the edges cannot be
/// linearized.
///
/// Uses Kahn's algorithm, always selecting the eligible node with the smallest
/// original index so the result is deterministic and preserves the explicit
/// order wherever the dependency edges leave it free.
pub(crate) fn topological_order(nodes: &[Node]) -> Result<Vec<usize>, PluginGraphError> {
    let index_of: HashMap<TypeId, usize> = nodes
        .iter()
        .enumerate()
        .map(|(i, n)| (n.type_id, i))
        .collect();

    // Validate dependencies up front and build the in-degree counts. An edge
    // runs dependency -> dependent (the dependency must come first).
    let mut in_degree = vec![0usize; nodes.len()];
    let mut dependents: Vec<Vec<usize>> = vec![Vec::new(); nodes.len()];
    for (i, node) in nodes.iter().enumerate() {
        for dep in &node.dependencies {
            let Some(&dep_idx) = index_of.get(&dep.type_id()) else {
                return Err(PluginGraphError::MissingDependency {
                    dependent: node.name,
                    missing: dep.name(),
                });
            };
            // A self-dependency is a trivial one-node cycle; count it so the
            // cycle check below reports it rather than silently resolving.
            dependents[dep_idx].push(i);
            in_degree[i] += 1;
        }
    }

    let mut resolved = Vec::with_capacity(nodes.len());
    let mut done = vec![false; nodes.len()];
    // Kahn's algorithm with a smallest-index selection rule for stability.
    loop {
        let next = (0..nodes.len()).find(|&i| !done[i] && in_degree[i] == 0);
        let Some(i) = next else { break };
        done[i] = true;
        resolved.push(i);
        for &d in &dependents[i] {
            in_degree[d] -= 1;
        }
    }

    if resolved.len() != nodes.len() {
        let members = (0..nodes.len())
            .filter(|&i| !done[i])
            .map(|i| nodes[i].name)
            .collect();
        return Err(PluginGraphError::Cycle { members });
    }

    Ok(resolved)
}
