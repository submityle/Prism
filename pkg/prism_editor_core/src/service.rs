use std::any::{Any, TypeId};
use std::collections::HashMap;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ServiceError {
    type_name: &'static str,
}

impl ServiceError {
    fn missing<T: 'static>() -> Self {
        Self {
            type_name: std::any::type_name::<T>(),
        }
    }
}

impl std::fmt::Display for ServiceError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "service {} is not registered", self.type_name)
    }
}

impl std::error::Error for ServiceError {}

#[derive(Default)]
pub struct ServiceRegistry {
    services: HashMap<TypeId, Box<dyn Any + Send + Sync>>,
}

impl ServiceRegistry {
    pub fn register<T: Any + Send + Sync>(&mut self, service: T) -> Option<T> {
        self.services
            .insert(TypeId::of::<T>(), Box::new(service))
            .and_then(|previous| previous.downcast::<T>().ok().map(|value| *value))
    }

    pub fn remove<T: Any + Send + Sync>(&mut self) -> Option<T> {
        self.services
            .remove(&TypeId::of::<T>())
            .and_then(|previous| previous.downcast::<T>().ok().map(|value| *value))
    }

    pub fn get<T: Any + Send + Sync>(&self) -> Result<&T, ServiceError> {
        self.services
            .get(&TypeId::of::<T>())
            .and_then(|service| service.downcast_ref::<T>())
            .ok_or_else(ServiceError::missing::<T>)
    }

    pub fn get_mut<T: Any + Send + Sync>(&mut self) -> Result<&mut T, ServiceError> {
        self.services
            .get_mut(&TypeId::of::<T>())
            .and_then(|service| service.downcast_mut::<T>())
            .ok_or_else(ServiceError::missing::<T>)
    }

    #[must_use]
    pub fn contains<T: Any + Send + Sync>(&self) -> bool {
        self.services.contains_key(&TypeId::of::<T>())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn services_are_replaced_and_type_safe() {
        let mut registry = ServiceRegistry::default();
        assert!(!registry.contains::<u32>());
        assert_eq!(registry.register(7_u32), None);
        assert_eq!(registry.get::<u32>().expect("service"), &7);
        assert_eq!(registry.register(9_u32), Some(7));
        assert_eq!(registry.remove::<u32>(), Some(9));
    }
}
