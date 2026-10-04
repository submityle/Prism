//! Analytic collider shapes and a shared shape registry.
//!
//! [`ColliderShape`] enumerates the analytic primitives M0 understands. Each
//! shape can report a local-space axis-aligned bounding box and closed-form (or
//! documented-approximate) mass properties. [`ShapeRegistry`] lets many bodies
//! share a single shape definition by [`ColliderHandle`].
//!
//! The inertia formulas here are standard closed-form results and are not
//! derived from Unreal Engine source.

use crate::state::body::MassProperties;
use glam::Vec3;

pub mod material;

pub use material::PhysicsMaterial;
pub mod convex_mesh;

pub use convex_mesh::{ConvexMeshData, ConvexMeshHandle, ConvexProjection, ConvexRayHit};
pub mod tri_mesh;

pub use tri_mesh::{TriMeshData, TriMeshHandle};
pub mod hull;

pub use hull::convex_hull;
pub mod faces;

pub use faces::{merge_coplanar_faces, PolygonFace, DEFAULT_COPLANAR_DOT};
pub mod decompose;

pub use decompose::{convex_decompose, DecompositionParams};
pub mod simplify;

pub use simplify::{simplify_convex_hull, SimplifiedHull, MIN_HULL_VERTICES};
pub mod kdop;

pub use kdop::{DopKind, KDop};
pub mod obb;

pub use obb::{fit_obb, Obb};
pub mod bounding_sphere;

pub use bounding_sphere::{mesh_bounding_sphere, minimal_bounding_sphere, BoundingSphere};
pub mod inertia;

pub use inertia::{full_inertia_tensor, principal_axes, MeshInertia, PrincipalInertia};
pub mod heightfield;

pub use heightfield::{HeightField, HeightFieldRayHit};
pub mod sdf;

pub use sdf::{MeshSdf, SdfBuildParams};
pub mod sdf_narrowband;

pub use sdf_narrowband::{NarrowBandParams, NarrowBandSdf};

pub mod weld;

pub use weld::{weld_mesh, WeldParams, WeldedMesh};
pub mod mesh_bvh;

pub use mesh_bvh::{MeshBvh, MeshClosestPoint, MeshRayHit};
pub mod quadric;

pub use quadric::Quadric;
pub mod decimate;

pub use decimate::{decimate_mesh, DecimateParams, DecimateTarget, DecimatedMesh};

pub mod lod;

pub use lod::{
    build_lod_chain, LodChainParams, LodSchedule, MeshLod, MeshLodChain, MIN_LOD_TRIANGLES,
};

pub mod connectivity;

pub use connectivity::{split_connected_components, ConnectivityParams, MeshComponent};

pub mod cook_shells;

pub use cook_shells::{cook_collision_shells, CookReport, CookShellParams, CookedShells};

pub mod topology;

pub use topology::{analyze_topology, MeshTopology};

pub mod winding_number;

pub use winding_number::{generalized_winding_number, point_is_inside, winding_numbers};

pub mod vertex_normals;

pub use vertex_normals::{face_normals, vertex_normals};

pub mod feature_edges;

pub use feature_edges::{
    extract_feature_edges, EdgeKind, FeatureEdge, FeatureEdgeParams, FeatureEdges,
};

pub mod quality_report;

pub use quality_report::{analyze_mesh_quality, MeshQualityParams, MeshQualityReport};

pub mod self_intersection;

pub use self_intersection::{
    detect_self_intersections, IntersectingPair, SelfIntersectionParams, SelfIntersectionReport,
};

pub mod surface_sampling;

pub use surface_sampling::{
    sample_surface, total_surface_area, SurfaceSample, SurfaceSampleParams,
};

pub mod curvature;

pub use curvature::{estimate_curvature, CurvatureParams, CurvatureReport, VertexCurvature};

pub mod boundary_loops;

pub use boundary_loops::{extract_boundary_loops, BoundaryLoopParams, BoundaryLoops};
pub mod hole_fill;

pub use hole_fill::{fill_boundary_loops, FillHolesParams, HoleFill};
pub mod solidity;

pub use solidity::{measure_solidity, MeshSolidity, DEFAULT_CONVEX_TOLERANCE};
pub mod shell_thickness;

pub use shell_thickness::{
    estimate_shell_thickness, ShellThickness, ShellThicknessParams, ThicknessSample,
};
pub mod hausdorff;
pub use hausdorff::{
    measure_directed_distance, measure_mesh_distance, DirectedMeshDistance, MeshDistance,
    MeshDistanceParams,
};
pub mod diameter;
pub use diameter::{mesh_diameter, MeshDiameter};
pub mod lod_error;
pub use lod_error::{evaluate_lod_errors, LodErrorReport, LodLevelError, LodMeshRef};
pub mod symmetry;
pub use symmetry::{detect_mirror_symmetry, MeshSymmetry, SymmetryParams, SymmetryPlane};
pub mod rotational_symmetry;
pub use rotational_symmetry::{
    detect_rotational_symmetry, MeshRotationalSymmetry, RotationalAxis, RotationalSymmetryParams,
};
pub mod geodesic;
pub use geodesic::{GeodesicPath, MeshEdgeGraph};
pub mod min_area_fill;
pub use min_area_fill::{triangulate_min_area, MinAreaFill, MAX_LOOP_VERTICES};
pub mod smoothing;
pub use smoothing::{taubin_smooth, SmoothingParams};
pub mod concavity;
pub use concavity::{measure_concavity, MeshConcavity, DEFAULT_CONCAVITY_TOLERANCE};
pub mod bounding_capsule;
pub use bounding_capsule::{fit_bounding_capsule, BoundingCapsule};
pub mod component_cleanup;
pub use component_cleanup::{remove_small_components, CleanedMesh, CleanupParams};
pub mod curvature_tensor;
pub use curvature_tensor::{estimate_curvature_tensor, CurvatureTensorReport, PrincipalCurvature};
pub mod auto_collision;
pub use auto_collision::{
    cook_auto_collision, select_representation, AutoCollisionParams, AutoCollisionResult,
    CollisionRepresentation, ConvexHullPiece, RepresentationKind, ShellCollision,
};
pub mod mesh_offset;
pub use mesh_offset::{offset_mesh, MeshOffsetParams, OffsetMesh};
pub mod orientation;
pub use orientation::{orient_outward, OrientedMesh};
pub mod tetrahedralize;
pub use tetrahedralize::{tetrahedralize, TetMesh, TetMeshParams};
pub mod tet_quality;
pub use tet_quality::{
    analyze_tet_mesh_quality, tet_quality, TetMeshQualityReport, TetQuality, TetQualityParams,
};
pub mod tet_smooth;
pub use tet_smooth::{smooth_tet_mesh, TetSmoothParams, TetSmoothResult};
pub mod tet_boundary;
pub use tet_boundary::{extract_tet_boundary, TetBoundary};
pub mod tet_conform;
pub use tet_conform::{conform_tet_boundary, TetConformParams, TetConformResult};
pub mod tet_mass;
pub use tet_mass::{compute_tet_mass_properties, TetMassParams, TetMassProperties};
pub mod tet_adjacency;
pub use tet_adjacency::{build_tet_adjacency, TetAdjacency};
pub mod tet_coloring;
pub use tet_coloring::{colour_tet_adjacency, colour_tet_mesh, TetColoring};
pub mod tet_components;
pub use tet_components::{label_tet_components, tet_components, TetComponents};
pub mod tet_partition;
pub use tet_partition::{
    partition_tet_adjacency, partition_tet_mesh, TetPartition, TetPartitionParams,
};
pub mod tet_vertex_adjacency;
pub use tet_vertex_adjacency::{build_tet_vertex_adjacency, TetVertexAdjacency};
pub mod tet_vertex_coloring;
pub use tet_vertex_coloring::{colour_tet_vertex_graph, colour_tet_vertices, TetVertexColoring};
pub mod tet_vertex_reorder;
pub use tet_vertex_reorder::{
    reorder_tet_vertex_graph, reorder_tet_vertices, vertex_bandwidth, TetVertexReorder,
};
pub mod tet_fem_basis;
pub use tet_fem_basis::{build_tet_fem_basis, TetFemBasis, TetFemBasisParams, TetFemElement};
pub mod tet_embedding;
pub use tet_embedding::{
    build_tet_embedding, locate_point, TetBinding, TetEmbedding, TetEmbeddingParams,
};
pub mod tet_fem_stiffness;
pub use tet_fem_stiffness::{element_stiffness, IsotropicElasticity, TetStiffness};
pub mod tet_fem_assembly;
pub use tet_fem_assembly::{assemble_global_stiffness, GlobalStiffness};
pub mod tet_lumped_mass;
pub use tet_lumped_mass::{build_lumped_mass, build_lumped_mass_from_mesh, LumpedMass};
pub mod tet_fem_cg;
pub use tet_fem_cg::{conjugate_gradient, solve_implicit_system, CgParams, CgReport};
pub mod tet_fem_constitutive;
pub use tet_fem_constitutive::{
    linear_first_piola, linear_strain_energy_density, stable_neo_hookean_first_piola,
    stable_neo_hookean_strain_energy_density, stvk_first_piola, stvk_strain_energy_density,
    HyperelasticModel, LameParameters,
};
pub mod tet_fem_internal_force;
pub use tet_fem_internal_force::{element_energy, element_internal_force};
pub mod tet_fem_force_assembly;
pub use tet_fem_force_assembly::{assemble_internal_forces, total_elastic_energy};
pub mod tet_fem_pcg;
pub use tet_fem_pcg::{preconditioned_conjugate_gradient, solve_implicit_system_jacobi};
pub mod tet_fem_corotational;
pub use tet_fem_corotational::{
    corotational_internal_force, corotational_stiffness, element_rotation,
};
pub mod tet_fem_corotational_assembly;
pub use tet_fem_corotational_assembly::{
    assemble_corotational_forces, assemble_corotational_stiffness,
};
pub mod tet_fem_block_jacobi;
pub use tet_fem_block_jacobi::solve_implicit_system_block_jacobi;
pub mod tet_fem_implicit;
pub mod tet_fem_integrator;
pub use tet_fem_integrator::{
    step_implicit_corotational, FemPreconditioner, FemStepResult, ImplicitStepParams,
    RayleighDamping,
};
pub mod tet_fem_body;
pub use tet_fem_body::TetFemBody;
pub mod tet_fem_newmark;
pub use tet_fem_newmark::{
    initial_acceleration, step_newmark, NewmarkParams, NewmarkState, NewmarkStepResult,
};
pub mod tet_fem_dirichlet;
pub use tet_fem_dirichlet::step_newmark_prescribed;
pub mod tet_fem_substep;
pub use tet_fem_substep::{step_newmark_substeps, SubstepParams, SubstepResult};
pub mod tet_fem_diagnostics;
pub use tet_fem_diagnostics::{angular_momentum, elastic_energy, kinetic_energy, linear_momentum};
pub mod tet_fem_strain_limit;
pub use tet_fem_strain_limit::{project_strain_limits, StrainLimitParams, StrainLimitReport};
pub mod tet_fem_volume_projection;
pub use tet_fem_volume_projection::{
    project_volume, VolumeProjectionParams, VolumeProjectionReport,
};
pub mod tet_fem_adaptive_substep;
pub use tet_fem_adaptive_substep::{
    step_newmark_adaptive, AdaptiveSubstepParams, AdaptiveSubstepResult,
};
pub mod tet_fem_projection_pass;
pub use tet_fem_projection_pass::{
    run_projection_pass, ProjectionOrder, ProjectionPassParams, ProjectionPassReport,
};
pub mod tet_fem_prescribed_substep;
pub use tet_fem_prescribed_substep::step_newmark_prescribed_substeps;
pub mod tet_fem_fiber_strain_limit;
pub use tet_fem_fiber_strain_limit::{
    project_fiber_strain_limits, FiberStrainLimitParams, FiberStrainLimitReport,
};
pub mod tet_fem_orthotropic_strain_limit;
pub use tet_fem_orthotropic_strain_limit::{
    project_orthotropic_strain_limits, OrthotropicStrainLimitParams, OrthotropicStrainLimitReport,
};
pub mod tet_fem_fiber_response_limit;
pub use tet_fem_fiber_response_limit::{
    project_fiber_response_strain_limits, FiberResponseBand, FiberResponseLimitParams,
    FiberResponseLimitReport,
};
pub mod tet_fem_anisotropic_projection_pass;
pub use tet_fem_anisotropic_projection_pass::{
    run_anisotropic_projection_pass, AnisotropicProjectionParams, AnisotropicProjectionReport,
    AnisotropicStage,
};
pub mod tet_fem_anisotropic;
pub use tet_fem_anisotropic::{
    orthotropic_fiber_energy, orthotropic_fiber_first_piola, FiberDirection, FiberFamily,
    FiberResponse, TransverselyIsotropicMaterial,
};
pub mod tet_fem_plasticity;
pub use tet_fem_plasticity::{return_map, PlasticModel, PlasticState, PlasticStep};
pub mod tet_fem_elastoplastic_force;
pub use tet_fem_elastoplastic_force::{
    elastic_potential_energy, element_elastoplastic_force, ElastoplasticForce,
};
pub mod tet_fem_sand_plasticity;
pub use tet_fem_sand_plasticity::{return_map_sand, SandModel, SandState, SandStep, SandYield};
pub mod tet_fem_snow_plasticity;
pub use tet_fem_snow_plasticity::{hardened_lame, return_map_snow, SnowModel, SnowState, SnowStep};
pub mod tet_fem_snow_force;
pub use tet_fem_snow_force::{element_snow_force, snow_elastic_potential_energy, SnowForce};
pub mod tet_fem_sand_force;
pub use tet_fem_sand_force::{element_sand_force, sand_elastic_potential_energy, SandForce};
pub mod tet_fem_plastic_force_assembly;
pub use tet_fem_plastic_force_assembly::{
    assemble_elastoplastic_forces, rest_states, total_elastic_potential_energy,
    ElastoplasticAssembly,
};
pub mod tet_fem_snow_force_assembly;
pub use tet_fem_snow_force_assembly::{
    assemble_snow_forces, rest_snow_states, total_snow_elastic_potential_energy, SnowAssembly,
};
pub mod tet_fem_sand_force_assembly;
pub use tet_fem_sand_force_assembly::{
    assemble_sand_forces, rest_sand_states, total_sand_elastic_potential_energy, SandAssembly,
};
pub mod tet_fem_camclay_plasticity;
pub use tet_fem_camclay_plasticity::{return_map_camclay, CamClayModel, CamClayState, CamClayStep};
pub mod tet_fem_camclay_force;
pub use tet_fem_camclay_force::{
    camclay_elastic_potential_energy, element_camclay_force, CamClayForce,
};
pub mod tet_fem_camclay_force_assembly;
pub use tet_fem_camclay_force_assembly::{
    assemble_camclay_forces, rest_camclay_states, total_camclay_elastic_potential_energy,
    CamClayAssembly,
};
pub mod tet_fem_drucker_prager_plasticity;
pub use tet_fem_drucker_prager_plasticity::{
    return_map_drucker_prager, DruckerPragerModel, DruckerPragerState, DruckerPragerStep,
    DruckerPragerYield,
};
pub mod tet_fem_drucker_prager_force;
pub use tet_fem_drucker_prager_force::{
    drucker_prager_elastic_potential_energy, element_drucker_prager_force, DruckerPragerForce,
};
pub mod tet_fem_drucker_prager_force_assembly;
pub use tet_fem_drucker_prager_force_assembly::{
    assemble_drucker_prager_forces, rest_drucker_prager_states,
    total_drucker_prager_elastic_potential_energy, DruckerPragerAssembly,
};
pub mod tet_fem_mohr_coulomb_plasticity;
pub use tet_fem_mohr_coulomb_plasticity::{
    return_map_mohr_coulomb, MohrCoulombModel, MohrCoulombState, MohrCoulombStep, MohrCoulombYield,
};
pub mod tet_fem_mohr_coulomb_force;
pub use tet_fem_mohr_coulomb_force::{
    element_mohr_coulomb_force, mohr_coulomb_elastic_potential_energy, MohrCoulombForce,
};
pub mod tet_fem_mohr_coulomb_force_assembly;
pub use tet_fem_mohr_coulomb_force_assembly::{
    assemble_mohr_coulomb_forces, rest_mohr_coulomb_states,
    total_mohr_coulomb_elastic_potential_energy, MohrCoulombAssembly,
};
pub mod tet_fem_hoek_brown_plasticity;
pub use tet_fem_hoek_brown_plasticity::{
    return_map_hoek_brown, HoekBrownModel, HoekBrownState, HoekBrownStep, HoekBrownYield,
};
pub mod tet_fem_hoek_brown_force;
pub use tet_fem_hoek_brown_force::{
    element_hoek_brown_force, hoek_brown_elastic_potential_energy, HoekBrownForce,
};
pub mod tet_fem_hoek_brown_force_assembly;
pub use tet_fem_hoek_brown_force_assembly::{
    assemble_hoek_brown_forces, rest_hoek_brown_states, total_hoek_brown_elastic_potential_energy,
    HoekBrownAssembly,
};
pub mod tet_fem_perzyna_viscoplasticity;
pub use tet_fem_perzyna_viscoplasticity::{
    return_map_perzyna, PerzynaModel, PerzynaState, PerzynaStep,
};
pub mod tet_fem_damage;
pub use tet_fem_damage::{equivalent_strain, update_damage, DamageModel, DamageState, DamageStep};
pub mod tet_fem_damage_force;
pub use tet_fem_damage_force::{
    damaged_elastic_potential_energy, element_damaged_force, DamagedForce,
};
pub mod tet_fem_damage_force_assembly;
pub use tet_fem_damage_force_assembly::{
    assemble_damaged_forces, total_degraded_potential_energy, DamageAssembly,
};
pub mod cohesive_zone;
pub use cohesive_zone::{
    cohesive_traction, dissipated_energy, CohesiveModel, CohesiveState, CohesiveStep,
};
pub mod cohesive_zone_assembly;
pub use cohesive_zone_assembly::{
    assemble_cohesive_forces, rest_states as cohesive_rest_states, total_dissipated_energy,
    CohesiveAssembly, CohesiveFacetStep, CohesiveInterface,
};
pub mod bonded_particle;
pub use bonded_particle::{update_bond, BondModel, BondState, BondStep};
pub mod cohesive_interface_builder;
pub use cohesive_interface_builder::{insert_cohesive_interfaces, CohesiveMesh};

pub mod cohesive_zone_integrator;
pub use cohesive_zone_integrator::{CohesiveBody, CohesiveStepReport};
pub mod bonded_particle_assembly;
pub use bonded_particle_assembly::{
    assemble_bond_forces, broken_bond_count, intact_bond_count, rest_bond_states, BondAssembly,
    BondNetworkStep, ParticleBond,
};
pub mod bonded_particle_network_builder;
pub use bonded_particle_network_builder::{build_bond_network, BondNetwork};

pub mod bonded_particle_integrator;
pub use bonded_particle_integrator::{BondStepReport, BondedParticleBody};

pub mod bonded_particle_contact;
pub use bonded_particle_contact::{contact_between, evaluate_contact, ContactForce, ContactModel};

pub mod bonded_particle_contact_resolver;
pub use bonded_particle_contact_resolver::{resolve_contacts, Contact, ContactResolution};

pub mod bonded_particle_contact_integrator;
pub use bonded_particle_contact_integrator::{ContactBody, ContactStepReport};

pub mod hertz_contact;
pub use hertz_contact::{evaluate_hertz_contact, hertz_contact_between, HertzModel};

pub mod hertz_contact_resolver;
pub use hertz_contact_resolver::{resolve_hertz_contacts, HertzContact, HertzContactResolution};

pub mod hertz_contact_integrator;
pub use hertz_contact_integrator::{HertzContactBody, HertzContactStepReport};

pub mod tangential_history_contact;
pub use tangential_history_contact::{
    evaluate_tangential_history, tangential_history_between, CundallStrackModel,
};

pub mod tangential_history_resolver;
pub use tangential_history_resolver::{
    TangentialContact, TangentialHistoryResolution, TangentialHistoryResolver,
};

pub mod tangential_history_integrator;
pub use tangential_history_integrator::{TangentialHistoryBody, TangentialHistoryStepReport};

pub mod rotational_contact;
pub use rotational_contact::{
    rotational_contact_between, ContactSprings, RollingContactModel, RotationalContact,
};

pub mod rotational_contact_resolver;
pub use rotational_contact_resolver::{
    RollingContactResolution, RollingContactResolver, RollingPairContact,
};

pub mod rotational_contact_integrator;
pub use rotational_contact_integrator::{RotationalContactBody, RotationalContactStepReport};
pub mod rotational_boundary_contact;
pub use rotational_boundary_contact::{grain_boundary_contact, BoundaryContact, HalfSpace};
pub mod rotational_boundary_resolver;
pub use rotational_boundary_resolver::{
    BoundaryContactResolution, BoundaryContactResolver, BoundaryPairContact,
};
pub mod granular_pile_integrator;
pub use granular_pile_integrator::{GranularPileBody, GranularPileStepReport};
pub mod capillary_bridge;
pub use capillary_bridge::{CapillaryBridge, CapillaryBridgeModel};
pub mod capillary_bridge_resolver;
pub use capillary_bridge_resolver::{
    CapillaryBridgeResolver, CapillaryPairBridge, CapillaryResolution,
};
pub mod wet_granular_pile;
pub use wet_granular_pile::{WetGranularPileBody, WetGranularPileStepReport};
pub mod uniform_grid_broadphase;
pub use uniform_grid_broadphase::UniformGridBroadphase;
pub mod capillary_liquid_distribution;
pub use capillary_liquid_distribution::LiquidDistribution;
pub mod finite_liquid_capillary_forces;
pub use finite_liquid_capillary_forces::{
    FiniteLiquidBridge, FiniteLiquidCapillary, FiniteLiquidResolution,
};
pub mod wet_bridge_hysteresis;
pub use wet_bridge_hysteresis::{WetBridgeHysteresis, WetBridgeReport};
pub mod wet_cohesion_driver;
pub use wet_cohesion_driver::{WetCohesionDriver, WetCohesionReport};
pub mod sphere_narrow_phase;
pub use sphere_narrow_phase::{SphereContact, SphereNarrowPhase};
pub mod sphere_contact_forces;
pub use sphere_contact_forces::{resolve_sphere_contact_forces, SphereContactForceResolution};
pub mod sphere_dem_integrator;
pub use sphere_dem_integrator::{SphereDemIntegrator, SphereDemState, SphereDemStepReport};
pub mod sphere_cundall_strack_driver;
pub use sphere_cundall_strack_driver::{SphereCundallStrackDriver, SphereCundallStrackResolution};
pub mod sphere_dem_friction_integrator;
pub use sphere_dem_friction_integrator::{
    SphereDemFrictionIntegrator, SphereDemFrictionStepReport,
};
pub mod sphere_boundary_driver;
pub use sphere_boundary_driver::{SphereBoundaryDriver, SphereBoundaryResolution};
pub mod sphere_packing;
pub use sphere_packing::{pack_spheres, SpherePacking, SpherePackingParams};
pub mod boundary_container;
pub use boundary_container::{closed_box, open_top_box, wedge_hopper};
pub mod grain_size_distribution;
pub use grain_size_distribution::{GrainSizeDistribution, SieveBin};
pub mod grid_sphere_packing;
pub use grid_sphere_packing::pack_spheres_grid;
pub mod distribution_packing;
pub use distribution_packing::{pack_spheres_from_distribution, DistributionPackingParams};
pub mod packing_diagnostics;
pub use packing_diagnostics::PackingDiagnostics;
pub mod gravity_settle;
pub use gravity_settle::{GravitySettleParams, GravitySettleReport, GravitySettler};
pub mod granular_scene;
pub use granular_scene::{ContainerKind, GranularScene, GranularSceneParams};
pub mod radial_distribution;
pub use radial_distribution::RadialDistribution;
pub mod angle_of_repose;
pub use angle_of_repose::AngleOfRepose;
pub mod porosity_profile;
pub use porosity_profile::PorosityProfile;
pub mod hopper_discharge;
pub use hopper_discharge::{mass_flow_between, BeverlooSlot, DischargeCensus};
pub mod janssen_pressure;
pub use janssen_pressure::{JanssenProfile, SiloCrossSection, StressSample};
pub mod fabric_tensor;
pub use fabric_tensor::FabricTensor;
pub mod granular_temperature;
pub use granular_temperature::GranularTemperature;
pub mod size_segregation;
pub use size_segregation::SizeSegregation;
pub mod granular_rheology;
pub use granular_rheology::{DilatancyLaw, GranularRheology};
pub mod kinetic_theory;
pub use kinetic_theory::GranularKineticState;
pub mod flowability;
pub use flowability::{FlowCharacter, PowderFlowability};
pub mod haff_cooling;
pub use haff_cooling::HaffCooling;

/// A handle into a [`ShapeRegistry`].
///
/// This is a plain index handle; shapes are immutable once inserted, so no
/// generation counter is required for correctness in M0.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ColliderHandle(pub u32);

/// An analytic collision shape.
#[derive(Clone, Copy, PartialEq, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum ColliderShape {
    /// A solid sphere.
    Sphere {
        /// Sphere radius.
        radius: f32,
    },
    /// An axis-aligned box given by its half-extents in local space.
    Cuboid {
        /// Half the box size along each local axis.
        half_extents: Vec3,
    },
    /// A capsule aligned with the local Y axis: a cylinder of the given
    /// half-height capped by two hemispheres of the given radius.
    Capsule {
        /// Half the length of the central cylinder (excluding the caps).
        half_height: f32,
        /// Radius of the cylinder and the hemispherical caps.
        radius: f32,
    },
    /// An infinite half-space plane `dot(normal, x) = offset`.
    Plane {
        /// Unit outward normal of the plane.
        normal: Vec3,
        /// Signed distance of the plane from the origin along `normal`.
        offset: f32,
    },
    /// A bounded convex hull, stored in the owning [`ShapeRegistry`]'s
    /// convex-mesh arena and referenced here by [`ConvexMeshHandle`].
    ///
    /// The geometric quantities needed for the broad phase and mass setup
    /// ([`local_aabb`](ColliderShape::local_aabb),
    /// [`mass_properties`](ColliderShape::mass_properties)) are cached inline so
    /// those two methods stay registry-free and `Copy`; the full vertex/face
    /// data (needed by the narrow phase, scene queries, and CCD) is resolved
    /// from the arena via `mesh`. Build one with
    /// [`ShapeRegistry::convex_hull_shape`], which fills the cache from the
    /// stored [`ConvexMeshData`].
    ConvexHull {
        /// Handle into the registry's convex-mesh arena.
        mesh: ConvexMeshHandle,
        /// Cached local-space AABB minimum.
        local_aabb_min: Vec3,
        /// Cached local-space AABB maximum.
        local_aabb_max: Vec3,
        /// Cached enclosed volume (cubic metres).
        volume: f32,
        /// Cached unit-density diagonal inertia (scales linearly with density).
        unit_density_inertia: Vec3,
    },
    /// A static triangle mesh (concave allowed), stored in the owning
    /// [`ShapeRegistry`]'s triangle-mesh arena and referenced by
    /// [`TriMeshHandle`].
    ///
    /// Like [`ConvexHull`](ColliderShape::ConvexHull) the AABB and mass cache is
    /// inline so the two registry-free methods stay `Copy`; the triangle soup
    /// (needed by queries and the narrow phase) is resolved from the arena via
    /// `mesh`. Build one with [`ShapeRegistry::tri_mesh_shape`]. A triangle mesh
    /// is intended as immovable scene geometry: its cached mass is the
    /// closed-solid mass of the (possibly open) surface and callers typically
    /// pair it with a static body.
    TriangleMesh {
        /// Handle into the registry's triangle-mesh arena.
        mesh: TriMeshHandle,
        /// Cached local-space AABB minimum.
        local_aabb_min: Vec3,
        /// Cached local-space AABB maximum.
        local_aabb_max: Vec3,
        /// Cached unit-density mass (scales linearly with density).
        unit_density_mass: f32,
        /// Cached unit-density diagonal inertia (scales linearly with density).
        unit_density_inertia: Vec3,
    },
}

impl ColliderShape {
    /// Returns the local-space axis-aligned bounding box as `(min, max)`.
    ///
    /// A [`ColliderShape::Plane`] is unbounded; it reports a degenerate box
    /// that spans the full finite range so that broad-phase code can treat it
    /// as "covers everything" without producing NaNs.
    #[must_use]
    pub fn local_aabb(&self) -> (Vec3, Vec3) {
        match *self {
            ColliderShape::Sphere { radius } => (Vec3::splat(-radius), Vec3::splat(radius)),
            ColliderShape::Cuboid { half_extents } => (-half_extents, half_extents),
            ColliderShape::Capsule {
                half_height,
                radius,
            } => {
                let ext = Vec3::new(radius, half_height + radius, radius);
                (-ext, ext)
            }
            ColliderShape::Plane { .. } => {
                let big = Vec3::splat(f32::MAX);
                (-big, big)
            }
            ColliderShape::ConvexHull {
                local_aabb_min,
                local_aabb_max,
                ..
            }
            | ColliderShape::TriangleMesh {
                local_aabb_min,
                local_aabb_max,
                ..
            } => (local_aabb_min, local_aabb_max),
        }
    }

    /// Computes mass properties for this shape at uniform `density`.
    ///
    /// - [`ColliderShape::Sphere`] and [`ColliderShape::Cuboid`] use exact
    ///   closed-form inertia.
    /// - [`ColliderShape::Capsule`] uses a documented approximation: a solid
    ///   cylinder plus a solid sphere (the two caps), with the sphere's
    ///   perpendicular contribution offset by the half-height via the parallel
    ///   axis theorem. This is accurate for the axial moment and a good
    ///   approximation for the perpendicular moments.
    /// - [`ColliderShape::Plane`] is treated as immovable and returns
    ///   [`MassProperties::zero`].
    #[must_use]
    pub fn mass_properties(&self, density: f32) -> MassProperties {
        match *self {
            ColliderShape::Sphere { radius } => {
                let mass =
                    density * (4.0 / 3.0) * crate::math::scalar::PI * radius * radius * radius;
                let inertia = 0.4 * mass * radius * radius;
                mass_props_from_diagonal(mass, Vec3::splat(inertia))
            }
            ColliderShape::Cuboid { half_extents } => {
                let full = half_extents * 2.0;
                let mass = density * full.x * full.y * full.z;
                let ix = (1.0 / 12.0) * mass * (full.y * full.y + full.z * full.z);
                let iy = (1.0 / 12.0) * mass * (full.x * full.x + full.z * full.z);
                let iz = (1.0 / 12.0) * mass * (full.x * full.x + full.y * full.y);
                mass_props_from_diagonal(mass, Vec3::new(ix, iy, iz))
            }
            ColliderShape::Capsule {
                half_height,
                radius,
            } => {
                let pi = crate::math::scalar::PI;
                let cyl_h = 2.0 * half_height;
                let cyl_mass = density * pi * radius * radius * cyl_h;
                let sphere_mass = density * (4.0 / 3.0) * pi * radius * radius * radius;
                let total = cyl_mass + sphere_mass;

                // Axial moment (about the Y capsule axis).
                let iy = 0.5 * cyl_mass * radius * radius + 0.4 * sphere_mass * radius * radius;

                // Perpendicular moment (about X/Z), cylinder plus offset sphere.
                let cyl_perp = cyl_mass * (cyl_h * cyl_h / 12.0 + radius * radius / 4.0);
                let sphere_perp =
                    0.4 * sphere_mass * radius * radius + sphere_mass * half_height * half_height;
                let iperp = cyl_perp + sphere_perp;

                mass_props_from_diagonal(total, Vec3::new(iperp, iy, iperp))
            }
            ColliderShape::Plane { .. } => MassProperties::zero(),
            ColliderShape::ConvexHull {
                volume,
                unit_density_inertia,
                ..
            } => {
                if volume <= 0.0 || density <= 0.0 {
                    return MassProperties::zero();
                }
                mass_props_from_diagonal(density * volume, unit_density_inertia * density)
            }
            ColliderShape::TriangleMesh {
                unit_density_mass,
                unit_density_inertia,
                ..
            } => {
                if unit_density_mass <= 0.0 || density <= 0.0 {
                    return MassProperties::zero();
                }
                mass_props_from_diagonal(
                    density * unit_density_mass,
                    unit_density_inertia * density,
                )
            }
        }
    }
}

/// Builds [`MassProperties`] from a total mass and a diagonal inertia tensor,
/// inverting each nonzero component and leaving zero (infinite) components at
/// zero.
fn mass_props_from_diagonal(mass: f32, inertia: Vec3) -> MassProperties {
    let inv_mass = if mass > 0.0 { 1.0 / mass } else { 0.0 };
    let inv = |i: f32| if i > 0.0 { 1.0 / i } else { 0.0 };
    MassProperties {
        inv_mass,
        inv_inertia: Vec3::new(inv(inertia.x), inv(inertia.y), inv(inertia.z)),
    }
}

/// A registry of shared, immutable collision shapes.
#[derive(Clone, Debug, Default)]
pub struct ShapeRegistry {
    shapes: Vec<ColliderShape>,
    convex_meshes: Vec<ConvexMeshData>,
    tri_meshes: Vec<TriMeshData>,
}

impl ShapeRegistry {
    /// Creates an empty registry.
    #[must_use]
    pub fn new() -> ShapeRegistry {
        ShapeRegistry::default()
    }

    /// Inserts a shape and returns a handle referring to it.
    pub fn insert(&mut self, shape: ColliderShape) -> ColliderHandle {
        let handle = ColliderHandle(self.shapes.len() as u32);
        self.shapes.push(shape);
        handle
    }

    /// Returns a reference to the shape for `handle`, or `None` if the handle
    /// is out of range.
    #[must_use]
    pub fn get(&self, handle: ColliderHandle) -> Option<&ColliderShape> {
        self.shapes.get(handle.0 as usize)
    }

    /// Returns the number of registered shapes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.shapes.len()
    }

    /// Returns `true` if the registry holds no shapes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.shapes.is_empty()
    }

    /// Inserts a convex mesh into the convex-mesh arena and returns a handle.
    pub fn insert_convex_mesh(&mut self, mesh: ConvexMeshData) -> ConvexMeshHandle {
        let handle = ConvexMeshHandle(self.convex_meshes.len() as u32);
        self.convex_meshes.push(mesh);
        handle
    }

    /// Returns a reference to the convex mesh for `handle`, or `None` if the
    /// handle is out of range.
    #[must_use]
    pub fn convex_mesh(&self, handle: ConvexMeshHandle) -> Option<&ConvexMeshData> {
        self.convex_meshes.get(handle.0 as usize)
    }

    /// Returns the number of convex meshes stored in the arena.
    #[must_use]
    pub fn convex_mesh_count(&self) -> usize {
        self.convex_meshes.len()
    }

    /// Inserts a triangle mesh into the triangle-mesh arena and returns a handle.
    pub fn insert_tri_mesh(&mut self, mesh: TriMeshData) -> TriMeshHandle {
        let handle = TriMeshHandle(self.tri_meshes.len() as u32);
        self.tri_meshes.push(mesh);
        handle
    }

    /// Returns a reference to the triangle mesh for `handle`, or `None` if the
    /// handle is out of range.
    #[must_use]
    pub fn tri_mesh(&self, handle: TriMeshHandle) -> Option<&TriMeshData> {
        self.tri_meshes.get(handle.0 as usize)
    }

    /// Returns the number of triangle meshes stored in the arena.
    #[must_use]
    pub fn tri_mesh_count(&self) -> usize {
        self.tri_meshes.len()
    }

    /// Builds a [`ColliderShape::ConvexHull`] referring to the convex mesh at
    /// `handle`, filling the inline broad-phase / mass cache from the stored
    /// [`ConvexMeshData`].
    ///
    /// Returns [`None`] when `handle` is out of range. The cached AABB, volume,
    /// and unit-density inertia are read once here so that
    /// [`ColliderShape::local_aabb`] and [`ColliderShape::mass_properties`] stay
    /// registry-free and `Copy`.
    #[must_use]
    pub fn convex_hull_shape(&self, handle: ConvexMeshHandle) -> Option<ColliderShape> {
        let mesh = self.convex_meshes.get(handle.0 as usize)?;
        let (local_aabb_min, local_aabb_max) = mesh.local_aabb();
        Some(ColliderShape::ConvexHull {
            mesh: handle,
            local_aabb_min,
            local_aabb_max,
            volume: mesh.volume(),
            unit_density_inertia: mesh.unit_density_inertia(),
        })
    }

    /// Builds a [`ColliderShape::TriangleMesh`] referring to the triangle mesh
    /// at `handle`, filling the inline broad-phase / mass cache from the stored
    /// [`TriMeshData`].
    ///
    /// Returns [`None`] when `handle` is out of range. A triangle mesh is
    /// intended as immovable scene geometry; the cached closed-solid mass is
    /// kept so callers that make it dynamic still receive a finite inertia.
    #[must_use]
    pub fn tri_mesh_shape(&self, handle: TriMeshHandle) -> Option<ColliderShape> {
        let mesh = self.tri_meshes.get(handle.0 as usize)?;
        let (local_aabb_min, local_aabb_max) = mesh.local_aabb();
        Some(ColliderShape::TriangleMesh {
            mesh: handle,
            local_aabb_min,
            local_aabb_max,
            unit_density_mass: mesh.unit_density_mass(),
            unit_density_inertia: mesh.unit_density_inertia(),
        })
    }

    /// Cooks a concave triangle mesh into a *compound convex collider*: a set of
    /// convex hulls (via [`convex_decompose`]) that together approximate the
    /// solid, each inserted into this registry and paired with a ready
    /// [`ColliderShape::ConvexHull`].
    ///
    /// This is the content-pipeline entry point for authored concave geometry
    /// (the analogue of `PhysX`/Chaos compound-convex cooking): attach every
    /// returned shape to the same body at the same local transform to collide
    /// against the whole approximation. Returns an empty vector when the mesh is
    /// degenerate and cannot form a single solid hull.
    pub fn cook_convex_decomposition(
        &mut self,
        vertices: &[Vec3],
        triangles: &[[u32; 3]],
        params: DecompositionParams,
    ) -> Vec<(ConvexMeshHandle, ColliderShape)> {
        let hulls = convex_decompose(vertices, triangles, params);
        let mut out = Vec::with_capacity(hulls.len());
        for mesh in hulls {
            let handle = self.insert_convex_mesh(mesh);
            if let Some(shape) = self.convex_hull_shape(handle) {
                out.push((handle, shape));
            }
        }
        out
    }

    /// Cooks a point cloud into a *vertex-limited* convex collider: builds the
    /// exact convex hull, reduces it to at most `max_vertices` vertices via
    /// [`simplify_convex_hull`], inserts the simplified mesh into this registry,
    /// and returns the handle paired with a ready
    /// [`ColliderShape::ConvexHull`].
    ///
    /// This is the content-pipeline entry point for capping a cooked convex
    /// collider's complexity (the analogue of `PhysX` `PxConvexMeshDesc`'s
    /// `vertexLimit` or Jolt's convex-hull vertex budget): dense render meshes
    /// are reduced to a solver-friendly proxy. The simplified hull is an *inner*
    /// approximation of the full hull; see [`simplify_convex_hull`] for the
    /// contract and `removed_volume` accounting.
    ///
    /// Returns [`None`] when `max_vertices` is below [`MIN_HULL_VERTICES`] or the
    /// cloud is degenerate and cannot form a single solid hull.
    pub fn cook_simplified_convex(
        &mut self,
        points: &[Vec3],
        max_vertices: usize,
    ) -> Option<(ConvexMeshHandle, ColliderShape, f32)> {
        let simplified = simplify_convex_hull(points, max_vertices)?;
        let removed_volume = simplified.removed_volume;
        let handle = self.insert_convex_mesh(simplified.to_convex_mesh());
        let shape = self.convex_hull_shape(handle)?;
        Some((handle, shape, removed_volume))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sphere_inertia_matches_closed_form() {
        let density = 3.0;
        let radius = 2.0;
        let shape = ColliderShape::Sphere { radius };
        let mp = shape.mass_properties(density);

        let mass = density * (4.0 / 3.0) * crate::math::scalar::PI * radius * radius * radius;
        let inertia = 0.4 * mass * radius * radius;
        assert!((mp.mass() - mass).abs() < 1e-3);
        let expected_inv = 1.0 / inertia;
        assert!((mp.inv_inertia.x - expected_inv).abs() < 1e-6);
        assert!((mp.inv_inertia.y - expected_inv).abs() < 1e-6);
        assert!((mp.inv_inertia.z - expected_inv).abs() < 1e-6);
    }

    #[test]
    fn cuboid_inertia_matches_closed_form() {
        let density = 2.0;
        let he = Vec3::new(1.0, 2.0, 3.0);
        let shape = ColliderShape::Cuboid { half_extents: he };
        let mp = shape.mass_properties(density);

        let full = he * 2.0;
        let mass = density * full.x * full.y * full.z;
        let ix = (1.0 / 12.0) * mass * (full.y * full.y + full.z * full.z);
        let iz = (1.0 / 12.0) * mass * (full.x * full.x + full.y * full.y);
        assert!((mp.mass() - mass).abs() < 1e-3);
        assert!((mp.inv_inertia.x - 1.0 / ix).abs() < 1e-6);
        assert!((mp.inv_inertia.z - 1.0 / iz).abs() < 1e-6);
    }

    #[test]
    fn plane_is_immovable() {
        let shape = ColliderShape::Plane {
            normal: Vec3::Y,
            offset: 0.0,
        };
        let mp = shape.mass_properties(5.0);
        assert_eq!(mp.inv_mass, 0.0);
        assert_eq!(mp.inv_inertia, Vec3::ZERO);
    }

    #[test]
    fn aabb_bounds_are_correct() {
        let (min, max) = ColliderShape::Sphere { radius: 1.5 }.local_aabb();
        assert_eq!(min, Vec3::splat(-1.5));
        assert_eq!(max, Vec3::splat(1.5));
    }

    #[test]
    fn registry_shares_shapes() {
        let mut reg = ShapeRegistry::new();
        assert!(reg.is_empty());
        let h = reg.insert(ColliderShape::Sphere { radius: 1.0 });
        assert_eq!(reg.len(), 1);
        assert!(matches!(
            reg.get(h),
            Some(ColliderShape::Sphere { radius }) if (*radius - 1.0).abs() < 1e-6
        ));
        assert!(reg.get(ColliderHandle(99)).is_none());
    }

    /// A closed, watertight L-shaped prism (concave) for cooking tests.
    fn l_prism() -> (Vec<Vec3>, Vec<[u32; 3]>) {
        let xy = [
            (0.0f32, 0.0f32),
            (2.0, 0.0),
            (2.0, 1.0),
            (1.0, 1.0),
            (1.0, 2.0),
            (0.0, 2.0),
            (0.0, 1.0),
        ];
        let mut verts = Vec::with_capacity(14);
        for &(x, y) in &xy {
            verts.push(Vec3::new(x, y, 0.0));
        }
        for &(x, y) in &xy {
            verts.push(Vec3::new(x, y, 1.0));
        }
        let cap: [[u32; 3]; 5] = [[0, 1, 2], [0, 2, 3], [0, 3, 6], [6, 3, 4], [6, 4, 5]];
        let boundary: [(u32, u32); 7] = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 0)];
        let mut tris: Vec<[u32; 3]> = Vec::new();
        for t in &cap {
            tris.push([t[0] + 7, t[1] + 7, t[2] + 7]);
        }
        for t in &cap {
            tris.push([t[0], t[2], t[1]]);
        }
        for &(a, b) in &boundary {
            tris.push([a, b, b + 7]);
            tris.push([a, b + 7, a + 7]);
        }
        (verts, tris)
    }

    #[test]
    fn cook_convex_decomposition_produces_compound_hulls() {
        let mut reg = ShapeRegistry::new();
        let (v, t) = l_prism();
        let params = DecompositionParams {
            resolution: 24,
            max_convex_hulls: 8,
            ..DecompositionParams::default()
        };
        let parts = reg.cook_convex_decomposition(&v, &t, params);
        assert!(parts.len() >= 2, "concave L must cook to >= 2 hulls");
        assert_eq!(reg.convex_mesh_count(), parts.len());
        for (handle, shape) in &parts {
            assert!(matches!(shape, ColliderShape::ConvexHull { .. }));
            let mesh = reg.convex_mesh(*handle).expect("handle resolves");
            assert!(mesh.volume() > 0.0);
        }
    }

    #[test]
    fn cook_degenerate_mesh_yields_no_hulls() {
        let mut reg = ShapeRegistry::new();
        let parts = reg.cook_convex_decomposition(&[], &[], DecompositionParams::default());
        assert!(parts.is_empty());
        assert_eq!(reg.convex_mesh_count(), 0);
    }

    #[test]
    fn cook_simplified_convex_caps_vertex_count() {
        // A box plus small near-coplanar bumps: the bumps are the lowest-volume
        // vertices, so a budget of 8 recovers a near-box proxy registered as a
        // ready convex-hull shape.
        let mut reg = ShapeRegistry::new();
        let mut points = Vec::new();
        for sx in [-1.0_f32, 1.0] {
            for sy in [-1.0_f32, 1.0] {
                for sz in [-1.0_f32, 1.0] {
                    points.push(Vec3::new(sx, sy, sz));
                }
            }
        }
        points.push(Vec3::new(0.0, 0.0, 1.02));
        points.push(Vec3::new(0.0, 0.0, -1.02));

        let (handle, shape, removed) = reg
            .cook_simplified_convex(&points, 8)
            .expect("cooks a proxy");
        assert!(matches!(shape, ColliderShape::ConvexHull { .. }));
        assert_eq!(reg.convex_mesh_count(), 1);
        let mesh = reg.convex_mesh(handle).expect("handle resolves");
        assert!(mesh.vertices().len() <= 8);
        assert!(mesh.volume() > 0.0);
        assert!(removed >= 0.0);
    }

    #[test]
    fn cook_simplified_convex_rejects_bad_input() {
        let mut reg = ShapeRegistry::new();
        let cube: Vec<Vec3> = [-1.0_f32, 1.0]
            .into_iter()
            .flat_map(|x| {
                [-1.0_f32, 1.0]
                    .into_iter()
                    .flat_map(move |y| [-1.0_f32, 1.0].into_iter().map(move |z| Vec3::new(x, y, z)))
            })
            .collect();
        // Below a tetrahedron budget is rejected and nothing is registered.
        assert!(reg.cook_simplified_convex(&cube, 3).is_none());
        // Degenerate cloud is rejected too.
        assert!(reg.cook_simplified_convex(&[], 8).is_none());
        assert_eq!(reg.convex_mesh_count(), 0);
    }
}
