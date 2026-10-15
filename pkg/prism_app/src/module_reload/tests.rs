//! Integration coverage using real, independently compiled cdylib fixtures.

use super::*;
use crate::App;
use prism_ecs::resource::Resource;
use prism_ecs::schedule::{IntoSystemConfigs, Schedule, StepReport, Stepping};
use prism_reflect::DynamicStruct;
use std::fs;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

static TEST_SEQUENCE: AtomicU64 = AtomicU64::new(0);

struct TestDir(PathBuf);
impl TestDir {
    fn new() -> Self {
        let id = TEST_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("prism-module-test-{}-{id}", std::process::id()));
        fs::create_dir(&path).expect("unique fixture directory");
        Self(path)
    }

    fn build(&self, name: &str, flags: &[&str]) -> PathBuf {
        let output = self.0.join(format!(
            "{}{name}{}",
            std::env::consts::DLL_PREFIX,
            std::env::consts::DLL_SUFFIX
        ));
        let source =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("src/module_reload/fixtures/module.rs");
        let mut command = Command::new(std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into()));
        command
            .arg("--crate-name")
            .arg("prism_module_fixture")
            .arg("--crate-type=cdylib")
            .arg("--edition=2024")
            .arg("-Cpanic=abort")
            .arg(&source)
            .arg("-o")
            .arg(&output)
            .env(
                "PRISM_FIXTURE_DROP_LOG",
                self.0.join(format!("{name}.drops")),
            );
        for flag in flags {
            command.arg("--cfg").arg(flag);
        }
        let result = command.output().expect("run rustc for real cdylib fixture");
        assert!(
            result.status.success(),
            "fixture compilation failed: {}",
            String::from_utf8_lossy(&result.stderr)
        );
        output
    }

    fn publish(&self, binary: &Path) -> PathBuf {
        let watched = self
            .0
            .join(format!("watched{}", std::env::consts::DLL_SUFFIX));
        fs::copy(binary, &watched).expect("publish library bytes");
        watched
    }

    fn drops(&self, name: &str) -> String {
        fs::read_to_string(self.0.join(format!("{name}.drops"))).unwrap_or_default()
    }

    fn snapshots(&self) -> usize {
        fs::read_dir(&self.0)
            .unwrap()
            .filter_map(Result::ok)
            .filter(|entry| {
                entry
                    .file_name()
                    .to_string_lossy()
                    .starts_with(".prism-module-")
            })
            .count()
    }
}
impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[derive(Default)]
struct Counter(u64);
impl Resource for Counter {}

#[derive(prism_reflect::Reflect, Clone)]
struct LegacyState {
    count: u64,
}

#[derive(prism_reflect::Reflect, Clone)]
struct RenamedState {
    total: u64,
}

#[derive(prism_reflect::Reflect, Clone)]
struct ExpandedState {
    total: u64,
    bonus: u64,
}

fn host(world: &mut World, operation: u64, input: &[u8]) -> Result<Vec<u8>, String> {
    if operation != 1 || input.len() != 8 {
        return Err("unknown operation/payload".into());
    }
    let amount = u64::from_le_bytes(input.try_into().unwrap());
    world.resource_mut::<Counter>().0 += amount;
    Ok(Vec::new())
}

fn world() -> World {
    let mut world = World::new();
    world.insert_resource(Counter::default());
    world
}

fn watcher(path: PathBuf) -> ModuleReload {
    ModuleReload::new(path, ModuleReloadConfig::new(42, 84), host).unwrap()
}

fn reflect_watcher(path: PathBuf) -> ModuleReload {
    let mut contract = ReflectStateContract::new("module-state");
    contract
        .register::<LegacyState, _, _>(
            1,
            |bytes| {
                if bytes.len() != 8 {
                    return Err("legacy state must contain one u64".into());
                }
                Ok(LegacyState {
                    count: u64::from_le_bytes(bytes.try_into().unwrap()),
                })
            },
            |state| Ok(state.count.to_le_bytes().to_vec()),
        )
        .unwrap();
    contract
        .register::<RenamedState, _, _>(
            2,
            |bytes| {
                if bytes.len() != 16 || &bytes[..8] != b"V2STATE!" {
                    return Err("renamed state has an invalid native envelope".into());
                }
                Ok(RenamedState {
                    total: u64::from_le_bytes(bytes[8..].try_into().unwrap()),
                })
            },
            |state| {
                let mut bytes = b"V2STATE!".to_vec();
                bytes.extend_from_slice(&state.total.to_le_bytes());
                Ok(bytes)
            },
        )
        .unwrap();
    contract
        .register::<ExpandedState, _, _>(
            3,
            |bytes| {
                if bytes.len() != 24 || &bytes[..8] != b"V3STATE!" {
                    return Err("expanded state has an invalid native envelope".into());
                }
                Ok(ExpandedState {
                    total: u64::from_le_bytes(bytes[8..16].try_into().unwrap()),
                    bonus: u64::from_le_bytes(bytes[16..].try_into().unwrap()),
                })
            },
            |state| {
                let mut bytes = b"V3STATE!".to_vec();
                bytes.extend_from_slice(&state.total.to_le_bytes());
                bytes.extend_from_slice(&state.bonus.to_le_bytes());
                Ok(bytes)
            },
        )
        .unwrap();
    contract
        .migration(1, 2, |value: &mut DynamicStruct| {
            let count = value
                .remove("count")
                .ok_or_else(|| "missing count during rename".to_owned())?;
            value.insert_boxed("total", count);
            Ok(())
        })
        .unwrap();
    contract
        .migration(2, 3, |value: &mut DynamicStruct| {
            value.insert("bonus", 7_u64);
            Ok(())
        })
        .unwrap();
    ModuleReload::new(path, ModuleReloadConfig::new(42, 84), host)
        .unwrap()
        .with_reflect_state(contract)
}

fn poll(watch: &mut ModuleReload) -> Result<ModuleReloadOutcome, ModuleReloadError> {
    // SAFETY: fixtures above are trusted code implementing this exact ABI.
    unsafe { watch.poll() }
}

#[test]
fn host_reflect_migration_renames_adds_default_and_rejects_state_loss() {
    let directory = TestDir::new();
    let v1 = directory.build("v1", &[]);
    let corrupt = directory.build("corrupt", &["fixture_v3", "fixture_corrupt_state"]);
    let v3 = directory.build("v3", &["fixture_v3"]);
    let mut watch = reflect_watcher(directory.publish(&v1));
    poll(&mut watch).unwrap();
    directory.publish(&corrupt);
    assert!(matches!(
        poll(&mut watch),
        Err(ModuleReloadError::Reflect(_))
    ));
    assert_eq!(watch.active().unwrap().unwrap().module_version, 1);
    directory.publish(&v3);
    let info = match poll(&mut watch).unwrap() {
        ModuleReloadOutcome::Activated(info) => info,
        ModuleReloadOutcome::Unchanged => panic!("changed binary was not activated"),
    };
    assert_eq!((info.module_version, info.state_schema), (20, 3));
    let exported = watch.export_state().unwrap();
    assert_eq!(&exported[..8], b"V3STATE!");
    assert_eq!(&exported[8..16], &0_u64.to_le_bytes());
    assert_eq!(&exported[16..], &7_u64.to_le_bytes());

    let mut world = world();
    let entity = world.spawn(());
    watch.run(7, &mut world).unwrap();
    assert_eq!(world.resource::<Counter>().0, 20);
    assert!(world.entities().contains(entity));
    let exported = watch.export_state().unwrap();
    assert_eq!(&exported[8..16], &20_u64.to_le_bytes());
    assert_eq!(&exported[16..], &7_u64.to_le_bytes());
}

#[test]
fn app_update_polls_at_the_boundary_and_keeps_old_module_on_failure() {
    let directory = TestDir::new();
    let v1 = directory.build("v1", &[]);
    let bad = directory.build("bad", &["fixture_v2", "fixture_fail_create"]);
    let v2 = directory.build("v2", &["fixture_v2"]);
    let mut app = App::new();
    app.add_module_reload(watcher(directory.publish(&v1)))
        .unwrap();
    app.update();
    assert_eq!(
        app.world()
            .resource::<ModuleReloadRegistry>()
            .active(42)
            .unwrap()
            .unwrap()
            .module_version,
        1
    );

    directory.publish(&bad);
    app.update();
    assert_eq!(
        app.world()
            .resource::<ModuleReloadRegistry>()
            .active(42)
            .unwrap()
            .unwrap()
            .module_version,
        1
    );
    assert_eq!(
        app.world_mut()
            .resource_mut::<ModuleReloadRegistry>()
            .take_failures()
            .len(),
        1
    );

    directory.publish(&v2);
    app.update();
    assert_eq!(
        app.world()
            .resource::<ModuleReloadRegistry>()
            .active(42)
            .unwrap()
            .unwrap()
            .module_version,
        10
    );
}

#[test]
fn real_library_rebinds_existing_schedule_and_migrates_state_without_replacing_world() {
    let directory = TestDir::new();
    let v1 = directory.build("v1", &[]);
    let v2 = directory.build("v2", &["fixture_v2"]);
    let mut watch = watcher(directory.publish(&v1));
    assert!(matches!(
        poll(&mut watch),
        Ok(ModuleReloadOutcome::Activated(_))
    ));
    assert_eq!(poll(&mut watch), Ok(ModuleReloadOutcome::Unchanged));
    let mut world = world();
    let id = world.id();
    let entity = world.spawn(());
    let mut schedule = Schedule::new();
    schedule.add_systems(watch.system(7).unwrap());
    schedule.run(&mut world);
    assert_eq!(world.resource::<Counter>().0, 1);
    assert_eq!(watch.export_state().unwrap(), 1_u64.to_le_bytes());
    directory.publish(&v2);
    let ModuleReloadOutcome::Activated(info) = poll(&mut watch).unwrap() else {
        panic!("changed binary");
    };
    assert_eq!(
        (info.generation, info.module_version, info.state_schema),
        (2, 10, 2)
    );
    assert_eq!(
        directory.drops("v1"),
        "1\n",
        "old state destroyed by old library"
    );
    assert_eq!(
        directory.snapshots(),
        1,
        "old library snapshot released after state"
    );
    // Same existing Schedule node, condition/order configuration and World.
    schedule.run(&mut world);
    assert_eq!(schedule.len(), 1);
    assert_eq!(world.id(), id);
    assert!(world.entities().contains(entity));
    assert_eq!(world.resource::<Counter>().0, 11);
    let exported = watch.export_state().unwrap();
    assert_eq!(&exported[..8], b"V2STATE!");
    assert_eq!(&exported[8..], &11_u64.to_le_bytes());
    assert!(watch.take_run_failures().unwrap().is_empty());
}

#[test]
fn rejects_abi_identity_contract_and_missing_callback_without_touching_active_state() {
    let directory = TestDir::new();
    let v1 = directory.build("good", &[]);
    let mut watch = watcher(directory.publish(&v1));
    poll(&mut watch).unwrap();
    let mut world = world();
    for flag in [
        "fixture_bad_abi",
        "fixture_bad_size",
        "fixture_bad_identity",
        "fixture_bad_contract",
        "fixture_missing_callback",
    ] {
        let bad = directory.build(flag, &[flag]);
        directory.publish(&bad);
        assert!(
            matches!(poll(&mut watch), Err(ModuleReloadError::Abi(_))),
            "{flag}"
        );
        assert_eq!(watch.active().unwrap().unwrap().generation, 1);
        assert_eq!(
            directory.snapshots(),
            1,
            "rejected library must be unloaded"
        );
        watch.run(7, &mut world).unwrap();
    }
    assert_eq!(world.resource::<Counter>().0, 5);
    assert_eq!(watch.export_state().unwrap(), 5_u64.to_le_bytes());
}

#[test]
fn migration_failure_destroys_only_candidate_and_can_retry_then_recover() {
    let directory = TestDir::new();
    let v1 = directory.build("old", &[]);
    let bad = directory.build("candidate", &["fixture_v2", "fixture_fail_create"]);
    let good = directory.build("fixed", &["fixture_v2"]);
    let mut watch = watcher(directory.publish(&v1));
    poll(&mut watch).unwrap();
    let mut world = world();
    watch.run(7, &mut world).unwrap();
    directory.publish(&bad);
    for _ in 0..2 {
        assert!(matches!(
            poll(&mut watch),
            Err(ModuleReloadError::Callback {
                stage: "create/migrate",
                ..
            })
        ));
        assert_eq!(watch.export_state().unwrap(), 1_u64.to_le_bytes());
    }
    assert_eq!(directory.drops("candidate"), "10\n10\n");
    assert_eq!(directory.drops("old"), "");
    assert_eq!(directory.snapshots(), 1);
    watch.run(7, &mut world).unwrap();
    assert_eq!(world.resource::<Counter>().0, 2);
    directory.publish(&good);
    poll(&mut watch).unwrap();
    watch.run(7, &mut world).unwrap();
    assert_eq!(world.resource::<Counter>().0, 12);
}

#[test]
fn rejects_changed_bindings_and_incomplete_binary_and_keeps_old_symbols_callable() {
    let directory = TestDir::new();
    let v1 = directory.build("old", &[]);
    let changed = directory.build("changed", &["fixture_changed_system"]);
    let mut watch = watcher(directory.publish(&v1));
    poll(&mut watch).unwrap();
    directory.publish(&changed);
    assert_eq!(poll(&mut watch), Err(ModuleReloadError::SystemSetChanged));
    fs::write(watch.path(), b"linker is still writing").unwrap();
    assert!(matches!(poll(&mut watch), Err(ModuleReloadError::Load(_))));
    fs::remove_file(watch.path()).unwrap();
    assert!(matches!(poll(&mut watch), Err(ModuleReloadError::Load(_))));
    let mut world = world();
    watch.run(7, &mut world).unwrap();
    assert_eq!(world.resource::<Counter>().0, 1);
    assert_eq!(
        watch.run(123, &mut world),
        Err(ModuleReloadError::UnknownSystem(123))
    );
}

#[test]
fn binding_pins_state_and_library_after_watcher_drop() {
    let directory = TestDir::new();
    let v1 = directory.build("pinned", &[]);
    let mut watch = watcher(directory.publish(&v1));
    poll(&mut watch).unwrap();
    let mut schedule = Schedule::new();
    schedule.add_systems(watch.system(7).unwrap());
    drop(watch);
    assert_eq!(directory.snapshots(), 1);
    assert!(directory.drops("pinned").is_empty());
    let mut world = world();
    schedule.run(&mut world);
    assert_eq!(world.resource::<Counter>().0, 1);
    drop(schedule);
    assert_eq!(directory.drops("pinned"), "1\n");
    assert_eq!(directory.snapshots(), 0);
}

#[test]
fn export_error_or_size_limit_blocks_replacement_and_preserves_old_module() {
    for flag in ["fixture_fail_export", "fixture_oversized_export"] {
        let directory = TestDir::new();
        let old = directory.build("old", &[flag]);
        let new = directory.build("new", &["fixture_v2"]);
        let mut config = ModuleReloadConfig::new(42, 84);
        config.max_state_bytes = 16;
        let mut watch = ModuleReload::new(directory.publish(&old), config, host).unwrap();
        poll(&mut watch).unwrap();
        directory.publish(&new);
        assert!(poll(&mut watch).is_err());
        assert_eq!(watch.active().unwrap().unwrap().generation, 1);
        assert!(directory.drops("old").is_empty());
        assert_eq!(directory.snapshots(), 1);
        let mut world = world();
        watch.run(7, &mut world).unwrap();
        assert_eq!(world.resource::<Counter>().0, 1);
    }
}

#[test]
fn stepped_and_budgeted_schedule_nodes_resolve_reloaded_code() {
    let directory = TestDir::new();
    let v1 = directory.build("v1", &[]);
    let v2 = directory.build("v2", &["fixture_v2"]);
    let mut watch = watcher(directory.publish(&v1));
    poll(&mut watch).unwrap();
    let mut schedule = Schedule::new();
    schedule.add_systems(watch.system(7).unwrap().run_if(|| true));
    let mut world = world();
    let mut stepping = Stepping::new();
    assert_eq!(
        stepping.step_system(&mut schedule, &mut world),
        StepReport::Ran(0)
    );
    directory.publish(&v2);
    poll(&mut watch).unwrap();
    assert_eq!(
        schedule.run_budgeted(&mut world, std::time::Duration::ZERO),
        0
    );
    assert_eq!(world.resource::<Counter>().0, 11);
}

#[test]
fn host_failure_is_reported_without_retiring_binding_and_host_panic_never_unwinds_through_c() {
    let directory = TestDir::new();
    let v1 = directory.build("v1", &[]);
    let path = directory.publish(&v1);
    let mut watch = ModuleReload::new(
        path.clone(),
        ModuleReloadConfig::new(42, 84),
        |_: &mut World, _: u64, _: &[u8]| Err("host refused".into()),
    )
    .unwrap();
    poll(&mut watch).unwrap();
    let mut schedule = Schedule::new();
    schedule.add_systems(watch.system(7).unwrap());
    let mut world = world();
    schedule.run(&mut world);
    schedule.run(&mut world);
    assert_eq!(watch.take_run_failures().unwrap().len(), 2);
    let mut panicking = ModuleReload::new(
        path,
        ModuleReloadConfig::new(42, 84),
        |_: &mut World, _: u64, _: &[u8]| -> Result<Vec<u8>, String> {
            panic!("host panic");
        },
    )
    .unwrap();
    poll(&mut panicking).unwrap();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        panicking.run(7, &mut world)
    }));
    assert!(
        result.is_err(),
        "panic resumed on Rust side of callback boundary"
    );
    assert_eq!(panicking.active(), Err(ModuleReloadError::Poisoned));
}

#[test]
fn library_limit_rejects_before_mapping_and_cross_world_bindings_reject_before_run() {
    let directory = TestDir::new();
    let v1 = directory.build("v1", &[]);
    let path = directory.publish(&v1);
    let mut config = ModuleReloadConfig::new(42, 84);
    config.max_library_bytes = 1;
    let mut limited = ModuleReload::new(path.clone(), config, host).unwrap();
    assert_eq!(poll(&mut limited), Err(ModuleReloadError::Limit("library")));
    assert_eq!(directory.snapshots(), 0);
    let mut watch = watcher(path);
    poll(&mut watch).unwrap();
    let mut schedule = Schedule::new();
    schedule.add_systems(watch.system(7).unwrap());
    let mut first = world();
    schedule.run(&mut first);
    let mut second = world();
    let result =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| schedule.run(&mut second)));
    assert!(result.is_err());
    assert_eq!(second.resource::<Counter>().0, 0);
}

#[cfg(feature = "multi_thread")]
#[test]
fn parallel_schedule_rebinds_the_same_exclusive_node() {
    let directory = TestDir::new();
    let v1 = directory.build("v1", &[]);
    let v2 = directory.build("v2", &["fixture_v2"]);
    let mut watch = watcher(directory.publish(&v1));
    poll(&mut watch).unwrap();
    let mut schedule = Schedule::new();
    schedule.add_systems(watch.system(7).unwrap());
    let pool = prism_tasks::TaskPool::with_threads(2);
    let mut world = world();
    schedule.run_parallel(&pool, &mut world);
    directory.publish(&v2);
    poll(&mut watch).unwrap();
    schedule.run_parallel_budgeted(&pool, &mut world, std::time::Duration::ZERO);
    assert_eq!(world.resource::<Counter>().0, 11);
}
