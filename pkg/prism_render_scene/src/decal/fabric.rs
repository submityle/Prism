//! Decal SceneFabric object-scoped lowering declarations.

use prism_render_architecture::render_fabric::{ObjectLayoutPolicy, ObjectResourceAccess, PassKey};

use crate::fabric_domains::resources::{
    DECAL_CLUSTER_INDEX_RESOURCE_ROLE, DECAL_CLUSTER_RANGE_RESOURCE_ROLE,
    DECAL_INSTANCE_RESOURCE_ROLE, DECAL_MATERIAL_RESOURCE_ROLE, DECAL_RECEIVER_RESOURCE_ROLE,
    DECAL_RESOLVE_RESOURCE_ROLE,
};

pub(crate) const DECAL_PASS: PassKey = PassKey(0x5007);

/// Every Decal resource is independently scoped to its object/history stream.
/// This prevents two projectors from aliasing one shared semantic buffer while
/// keeping the pass and feature selection device-lifetime and unchanged.
pub(crate) const DECAL_OBJECT_ACCESSES: &[ObjectResourceAccess] = &[
    ObjectResourceAccess {
        pass: DECAL_PASS,
        ordinal: 0,
        role: DECAL_INSTANCE_RESOURCE_ROLE,
        layout: ObjectLayoutPolicy::Exact,
    },
    ObjectResourceAccess {
        pass: DECAL_PASS,
        ordinal: 1,
        role: DECAL_MATERIAL_RESOURCE_ROLE,
        layout: ObjectLayoutPolicy::Exact,
    },
    ObjectResourceAccess {
        pass: DECAL_PASS,
        ordinal: 2,
        role: DECAL_RECEIVER_RESOURCE_ROLE,
        layout: ObjectLayoutPolicy::Exact,
    },
    ObjectResourceAccess {
        pass: DECAL_PASS,
        ordinal: 3,
        role: DECAL_CLUSTER_RANGE_RESOURCE_ROLE,
        layout: ObjectLayoutPolicy::Exact,
    },
    ObjectResourceAccess {
        pass: DECAL_PASS,
        ordinal: 4,
        role: DECAL_CLUSTER_INDEX_RESOURCE_ROLE,
        layout: ObjectLayoutPolicy::Exact,
    },
    ObjectResourceAccess {
        pass: DECAL_PASS,
        ordinal: 5,
        role: DECAL_RESOLVE_RESOURCE_ROLE,
        layout: ObjectLayoutPolicy::Exact,
    },
];
