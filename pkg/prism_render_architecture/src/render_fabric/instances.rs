//! Explicit object-resource instantiation, independent of scene ownership.
use super::*;

/// Semantic identity, not a physical handle, content version or temporal frame.
/// The object retains its generation; its enum discriminant authenticates domain.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct ObjectResourceIdentity {
    object: RenderObjectId,
    history_key: u32,
    role: PayloadRole,
}

impl ObjectResourceIdentity {
    pub const fn new(object: RenderObjectId, history_key: u32, role: PayloadRole) -> Self {
        Self {
            object,
            history_key,
            role,
        }
    }
    pub const fn domain(self) -> RenderDomain {
        self.object.domain()
    }
    pub const fn object(self) -> RenderObjectId {
        self.object
    }
    pub const fn history_key(self) -> u32 {
        self.history_key
    }
    pub const fn role(self) -> PayloadRole {
        self.role
    }
}

/// `None` is explicitly shared, never an unspecified/first matching object.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct ResourceAddress {
    pub key: ResourceKey,
    pub frame: u64,
    pub identity: Option<ObjectResourceIdentity>,
}
impl ResourceAddress {
    pub const fn shared(key: ResourceKey, frame: u64) -> Self {
        Self {
            key,
            frame,
            identity: None,
        }
    }
    pub const fn object(key: ResourceKey, frame: u64, identity: ObjectResourceIdentity) -> Self {
        Self {
            key,
            frame,
            identity: Some(identity),
        }
    }
}

/// Opt-in layout variation. Only sizes (buffer bytes or texture extent/mips)
/// may change. Kind, usage, dimension, format and sample count remain exact.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ObjectLayoutPolicy {
    Exact,
    Sized { max_bytes: u64 },
}
impl ObjectLayoutPolicy {
    pub(super) fn accepts(self, declared: ResourceLayout, actual: ResourceLayout) -> bool {
        let Some(bytes) = actual.size() else {
            return false;
        };
        match self {
            Self::Exact => actual == declared,
            Self::Sized { max_bytes } => {
                if bytes > max_bytes {
                    return false;
                }
                match (declared, actual) {
                    (
                        ResourceLayout::Buffer { usage: a, .. },
                        ResourceLayout::Buffer { usage: b, .. },
                    ) => a == b,
                    (
                        ResourceLayout::Texture {
                            dimension: a,
                            format: b,
                            samples: c,
                            usage: d,
                            ..
                        },
                        ResourceLayout::Texture {
                            dimension: x,
                            format: y,
                            samples: z,
                            usage: w,
                            ..
                        },
                    ) => (a, b, c, d) == (x, y, z, w),
                    _ => false,
                }
            }
        }
    }
}

/// Declares exactly which original template ordinal expands per object/history
/// stream. All other accesses are shared. This is compiler input, not a second
/// pass template, scheduler or recorder registration mechanism.
/// For independent writers of the same key in different roles, first supply
/// these declarations to `FeatureCatalog::new_with_object_accesses`; lowering
/// must use the same declarations, including layout policies and ordinals.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ObjectResourceAccess {
    pub pass: PassKey,
    pub ordinal: usize,
    pub role: PayloadRole,
    pub layout: ObjectLayoutPolicy,
}

/// Full identity accompanies the unchanged external-storage description.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ObjectResourceBinding {
    pub identity: ObjectResourceIdentity,
    pub binding: FrameResourceBinding,
}
impl ObjectResourceBinding {
    pub const fn address(self) -> ResourceAddress {
        ResourceAddress::object(self.binding.key, self.binding.frame, self.identity)
    }
}
