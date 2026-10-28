//! CPU reference coverage for clustered Decal candidate ordering.

use prism_math::Mat4;
use prism_render_architecture::motion::MotionSource;
use prism_render_architecture::render_fabric::{
    resolve_decal, resolve_decal_candidates, DecalInstance, DecalMaterial, DecalReceiver,
    DecalResolvedReceiver, DomainObjectHandle,
};
use prism_render_shading::gi::decal::DecalProjector;

use crate::fabric_domains::{owners::DecalRenderWorldOwner, DomainFrameIdentity};

fn resolve_candidates(
    receiver: &DecalReceiver,
    instances: &[DecalInstance],
    materials: &[DecalMaterial],
    indices: &[u32],
) -> DecalResolvedReceiver {
    resolve_decal_candidates(receiver, instances, materials, indices)
}

#[test]
fn clustered_cpu_reference_preserves_shader_candidate_order_and_skips_invalid_entries() {
    let receiver = DecalReceiver {
        world_position: [0.0, 0.0, 0.0, 1.0],
        world_normal: [0.0, 0.0, -1.0, 0.0],
        albedo: [0.1, 0.2, 0.3, 1.0],
        scalar: [0.6, 0.1, 0.0, 0.0],
    };
    let instances = [
        DecalInstance::new(
            Mat4::IDENTITY.to_cols_array(),
            [0.0, 0.0, 1.0],
            0.0,
            0.0,
            0.0,
            0,
        ),
        DecalInstance::new(
            Mat4::IDENTITY.to_cols_array(),
            [0.0, 0.0, 1.0],
            0.0,
            0.0,
            0.0,
            1,
        ),
    ];
    let materials = [
        DecalMaterial {
            albedo: [0.8, 0.0, 0.0, 1.0],
            ..DecalMaterial::default()
        },
        DecalMaterial {
            albedo: [0.0, 0.8, 0.0, 1.0],
            ..DecalMaterial::default()
        },
    ];

    let resolved = resolve_candidates(&receiver, &instances, &materials, &[99, 0, 100, 1]);
    let first = resolve_decal(&receiver, &instances[0], &materials[0]);
    let expected = prism_render_architecture::render_fabric::resolve_decal_over(
        &first,
        &receiver,
        &instances[1],
        &materials[1],
    );
    assert_eq!(resolved, expected);
}

#[test]
fn clustered_cpu_reference_accumulates_alpha_over_and_additive_candidates() {
    let receiver = DecalReceiver {
        world_position: [0.0, 0.0, 0.0, 1.0],
        world_normal: [0.0, 0.0, -1.0, 0.0],
        albedo: [0.1, 0.2, 0.3, 1.0],
        scalar: [0.6, 0.1, 0.0, 0.0],
    };
    let instances = [
        DecalInstance::new(
            Mat4::IDENTITY.to_cols_array(),
            [0.0, 0.0, 1.0],
            0.0,
            0.0,
            0.0,
            0,
        ),
        DecalInstance::new(
            Mat4::IDENTITY.to_cols_array(),
            [0.0, 0.0, 1.0],
            0.0,
            0.0,
            0.0,
            1,
        ),
    ];
    let materials = [
        DecalMaterial {
            albedo: [0.8, 0.0, 0.0, 1.0],
            scalar_blend: [0.5, 0.0, 0.5, 0.0],
            ..DecalMaterial::default()
        },
        DecalMaterial {
            albedo: [0.0, 0.8, 0.0, 1.0],
            scalar_blend: [0.25, 0.0, 0.5, 1.0],
            ..DecalMaterial::default()
        },
    ];

    let resolved = resolve_candidates(&receiver, &instances, &materials, &[0, 1]);
    assert!(resolved.albedo[0] > receiver.albedo[0]);
    assert!(resolved.albedo[1] > receiver.albedo[1]);
    assert!(resolved.scalar_coverage[2] > 0.0);
}

#[test]
fn decal_owner_removal_releases_identity_and_re_registration_starts_new_history_stream() {
    let object = DomainObjectHandle::new(17, 3);
    let mut owner = DecalRenderWorldOwner::default();
    let first_identity = DomainFrameIdentity {
        object,
        history_key: 41,
        motion_source: MotionSource::Rigid,
    };
    owner
        .register_with_data(
            first_identity,
            DecalProjector::new(Mat4::IDENTITY, 0.2, 0.0, 0.5),
            DecalMaterial::default(),
            DecalReceiver::default(),
        )
        .unwrap();
    assert_eq!(owner.len(), 1);
    assert_eq!(owner.get(object).unwrap().identity(), first_identity);

    let removed = owner
        .remove(object)
        .expect("registered Decal must be removable");
    assert_eq!(removed.identity(), first_identity);
    assert!(owner.is_empty());
    assert!(owner.get(object).is_none());

    let second_identity = DomainFrameIdentity {
        history_key: 42,
        ..first_identity
    };
    owner
        .register_with_data(
            second_identity,
            DecalProjector::new(Mat4::from_translation(prism_math::Vec3::X), 0.2, 0.0, 0.5),
            DecalMaterial::default(),
            DecalReceiver::default(),
        )
        .unwrap();
    assert_eq!(owner.get(object).unwrap().identity(), second_identity);
    assert_ne!(
        owner.get(object).unwrap().identity().history_key,
        removed.identity().history_key,
        "removal/re-registration must not retain the retired history stream"
    );
}
