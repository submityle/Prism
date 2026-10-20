//! Production graph lowering/capture/codec into real retained Loom and raster.
use prism_editor_profiler::{
    capture_runtime_graph, GraphCaptureInputs, GraphCaptureLimits, GraphCaptureReport,
    GraphPassTiming, GraphTimingSource,
};
use prism_editor_ui::{
    FrameGraphPage as Page, FrameGraphPanelAction as Action,
    FrameGraphPanelController as Controller, FrameGraphPanelError as Error, FrameGraphPanelLimits,
    FrameGraphSelection as Selection, SelectionMode, UiEvent, MAX_FRAME_GRAPH_PAGE_ROWS,
};
use prism_render_architecture::{
    frame_graph::{AccessKind, QueueClass, ResourceLifetime},
    memory::{FrameBudgetLedger, FrameWorkBudget},
    render_fabric::*,
};
use prism_render_graph::capacity::GraphLimits;
use prism_ui::{
    backend::RecordingBackend,
    layout::{AvailableSpace, Rect, Size},
    Element, Key, Ui,
};
use prism_ui_render_backend::{rasterize, RetainedScene};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
    time::Instant,
};

fn table<T>(v: Vec<T>) -> &'static [T] {
    Box::leak(v.into_boxed_slice())
}
fn usage(key: ResourceKey, access: AccessKind) -> ResourceUse {
    ResourceUse {
        key,
        access,
        scope: ResourceScope::Whole,
        frame_lag: None,
        input: ResourceInput::Automatic,
    }
}
// Real catalog declarations, production lowering and codec. No GPU submission
// or running-editor hook is claimed by these graph-construction tests.
fn report(count: usize, measured_passes: usize) -> GraphCaptureReport {
    let feature = FeatureId(872);
    let keys: Vec<_> = (0..count)
        .map(|i| ResourceKey::of(feature, i as u16))
        .collect();
    let passes = table(
        (0..count)
            .map(|i| {
                let mut accesses = Vec::new();
                if i > 0 {
                    accesses.push(usage(
                        keys[i - 1],
                        if (i - 1) % 2 == 0 {
                            AccessKind::StorageRead
                        } else {
                            AccessKind::SampledRead
                        },
                    ));
                }
                accesses.push(usage(
                    keys[i],
                    if i % 2 == 0 {
                        AccessKind::StorageWrite
                    } else {
                        AccessKind::ColorAttachment
                    },
                ));
                PassTemplate {
                    id: PassKey(100 + i as u32),
                    queue: if i % 2 == 0 {
                        QueueClass::Compute
                    } else {
                        QueueClass::Graphics
                    },
                    depends_on: if i == 0 {
                        &[]
                    } else {
                        table(vec![PassKey(99 + i as u32)])
                    },
                    accesses: table(accesses),
                }
            })
            .collect(),
    );
    let variants = table(vec![FeatureVariant {
        id: VariantId(9),
        rank: 0,
        min_quality: RenderQualityTier::Low,
        required_caps: CapabilitySet::EMPTY,
        requires_services_extra: ServiceSet::EMPTY,
        passes,
        shader_permutation: None,
    }]);
    let catalog = FeatureCatalog::new(
        table(vec![FeatureDefinition::subsystem(
            feature,
            RenderDomain::Surface,
            ServiceSet::EMPTY,
            variants,
        )]),
        table(keys.clone()),
    )
    .unwrap();
    let roots = FrozenRoots::from_build_profile(&catalog, table(vec![feature]), &[]).unwrap();
    let selection = FeatureSelection::resolve(
        &catalog,
        &roots,
        CapabilitySet::WEBGPU_BASELINE,
        RenderQualityTier::Low,
    )
    .unwrap();
    let templates: Vec<_> = keys
        .iter()
        .enumerate()
        .map(|(i, &key)| ResourceTemplate {
            key,
            lifetime: ResourceLifetime::Transient,
            alignment: 256,
            layout: if i % 2 == 0 {
                ResourceLayout::Buffer {
                    bytes: 64,
                    usage: BufferUsages::STORAGE,
                }
            } else {
                ResourceLayout::Texture {
                    extent: [8, 8, 1],
                    dimension: TextureDimension::D2,
                    format: TextureFormat::Rgba8Unorm,
                    mip_levels: 1,
                    samples: 1,
                    usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
                }
            },
        })
        .collect();
    let compiled = ResourceTemplates::new(&catalog, &templates)
        .unwrap()
        .compile(&selection, 73, &[], &[], GraphLimits::default())
        .unwrap();
    let bindings = compiled
        .passes()
        .iter()
        .map(|p| RuntimePassBinding::new(p.key(), |_| {}))
        .collect();
    let runtime = compiled
        .prepare_runtime(prism_render_driver::Extent3d::new_2d(8, 8), &[], bindings)
        .unwrap();
    let timings: Vec<_> = (0..measured_passes)
        .map(|i| {
            let start = Instant::now();
            for n in 0..1000 {
                std::hint::black_box(n * n + i);
            }
            GraphPassTiming {
                pass: i as u32,
                duration_ns: start.elapsed().as_nanos().try_into().unwrap(),
            }
        })
        .collect();
    let mut ledger = FrameBudgetLedger::new(FrameWorkBudget {
        upload_bytes: 4,
        readback_bytes: 10,
        relocation_bytes: 0,
        acceleration_structure_builds: 0,
    });
    ledger.record_upload(5);
    ledger.record_readback(5);
    ledger.record_acceleration_structure_builds(1);
    let captured = capture_runtime_graph(
        &runtime,
        GraphCaptureInputs {
            timing_source: (measured_passes > 0).then_some(GraphTimingSource::Cpu),
            timings: &timings,
            work_budget: Some(&ledger),
            alias_heap_budget: Some(0),
        },
        GraphCaptureLimits::default(),
    )
    .unwrap();
    let bytes = captured.to_bytes(GraphCaptureLimits::default()).unwrap();
    let decoded = GraphCaptureReport::from_bytes(&bytes, GraphCaptureLimits::default()).unwrap();
    assert_eq!(decoded, captured);
    assert_eq!(
        decoded.to_bytes(GraphCaptureLimits::default()).unwrap(),
        bytes
    );
    decoded
}
fn element_keys(e: &Element, out: &mut Vec<Option<String>>) {
    out.push(match e.explicit_key() {
        Some(Key::Str(s)) => Some(s.clone()),
        _ => None,
    });
    for child in e.child_elements() {
        element_keys(child, out);
    }
}
fn layout(e: &Element) -> (Ui<RetainedScene>, BTreeMap<String, Rect>) {
    let mut ui = Ui::new(RetainedScene::new());
    ui.mount(e);
    ui.compute_layout(Size::new(
        AvailableSpace::Definite(900.),
        AvailableSpace::MaxContent,
    ));
    let mut keys = Vec::new();
    element_keys(e, &mut keys);
    let mut rects = Vec::new();
    ui.backend().visit_layout(|_, r| rects.push(r));
    assert_eq!(keys.len(), rects.len());
    let keyed: Vec<_> = keys
        .into_iter()
        .zip(rects)
        .filter_map(|(k, r)| k.map(|k| (k, r)))
        .collect();
    let distinct: BTreeSet<_> = keyed.iter().map(|(k, _)| k).collect();
    assert_eq!(keyed.len(), distinct.len(), "all retained keys unique");
    (ui, keyed.into_iter().collect())
}
fn activate(c: &mut Controller, key: &str) {
    let view = c.build();
    c.handle_event(&view, key, &UiEvent::Activate).unwrap();
}
fn near(a: f32, b: f32) {
    assert!((a - b).abs() < 0.1, "{a} != {b}");
}
fn text(e: &Element) -> String {
    std::iter::once(e.text_content().unwrap_or_default().to_owned())
        .chain(e.child_elements().iter().map(text))
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn production_capture_codec_reaches_real_retained_graph_and_selection_pixels() {
    let report = Arc::new(report(4, 0));
    let mut c = Controller::default();
    c.publish(Some(report.clone())).unwrap();
    let view = c.build();
    assert!(Arc::ptr_eq(view.report().unwrap(), &report));
    assert_eq!(view.passes().len(), 4);
    assert_eq!(view.dependencies().len(), report.dependencies().len());
    assert!(view.passes().iter().all(|p| !p.critical));
    assert!(c.critical_path().is_none());
    let (ui, rects) = layout(view.element());
    for (key, _) in view.bindings() {
        assert!(
            rects[key].size.width > 0. && rects[key].size.height > 0.,
            "bound actual geometry {key}"
        );
    }
    let canvas = rects["frame-graph:canvas"];
    let pass = &view.passes()[0];
    let box_rect = rects[&pass.key];
    near(box_rect.left(), canvas.left() + pass.rect[0]);
    near(box_rect.top(), canvas.top() + pass.rect[1]);
    near(box_rect.size.width, pass.rect[2]);
    near(box_rect.size.height, pass.rect[3]);
    assert!(view.dependencies().iter().any(|e| !e.segments.is_empty()));
    let x = (box_rect.left() + 2.) as u32;
    let y = (box_rect.top() + 29.) as u32;
    let pixels = rasterize(&ui.backend().to_draw_list(), 900, 2000);
    for edge in view.dependencies() {
        for (segment, expected) in edge.segments.iter().enumerate() {
            let key = format!("frame-graph:wire:{}:{segment}", edge.dependency);
            let actual = rects[&key];
            near(actual.left(), canvas.left() + expected[0]);
            near(actual.top(), canvas.top() + expected[1]);
            near(actual.size.width, expected[2]);
            near(actual.size.height, expected[3]);
        }
    }
    let wire = view
        .dependencies()
        .iter()
        .find(|d| !d.segments.is_empty())
        .unwrap();
    let rail = rects[&format!("frame-graph:wire:{}:1", wire.dependency)];
    let wire_pixel = pixels.pixel((rail.left() + 1.) as u32, rail.top() as u32);
    near(wire_pixel[0], 0.35);
    near(wire_pixel[2], 0.9);
    let before = pixels.pixel(x, y);
    near(before[0], 0.15);
    near(before[2], 0.7);
    c.handle_event(&view, &pass.key, &UiEvent::Activate)
        .unwrap();
    assert_eq!(c.selection(), Some(Selection::Pass(pass.pass)));
    let next = c.build();
    let (ui, _) = layout(next.element());
    let pixels = rasterize(&ui.backend().to_draw_list(), 900, 2000);
    let after = pixels.pixel(x, y);
    assert!(
        after[0] > 0.99 && after[2] < 0.13,
        "selection changes real raster {after:?}"
    );
    let mut retained = Ui::new(RecordingBackend::new());
    retained.mount(next.element());
    retained.backend_mut().clear();
    retained.update(c.build().element());
    assert!(
        retained.backend().ops().is_empty(),
        "identical publication/navigation reconciles without writes"
    );
}

#[test]
fn real_ssa_resource_users_and_dependency_endpoints_are_interactive() {
    let r = Arc::new(report(4, 0));
    let mut c = Controller::default();
    c.publish(Some(r.clone())).unwrap();
    activate(&mut c, "frame-graph:pass:1");
    let view = c.build();
    assert_eq!(view.accesses().len(), 2);
    let access = &view.accesses()[0];
    assert_eq!(
        (
            access.pass,
            access.ordinal,
            access.resource,
            access.input_version,
            access.output_version
        ),
        (1, 0, 0, 1, None)
    );
    assert!(view.dependencies().iter().all(|d| d.endpoints.contains(&1)));
    let hazard = view
        .dependencies()
        .iter()
        .find(|d| {
            matches!(
                r.dependencies()[d.dependency].kind,
                prism_editor_profiler::CapturedDependencyKind::Hazard { .. }
            )
        })
        .unwrap()
        .dependency;
    c.handle_event(
        &view,
        &format!("frame-graph:dependency:{hazard}"),
        &UiEvent::Activate,
    )
    .unwrap();
    assert_eq!(c.selection(), Some(Selection::Dependency(hazard)));
    activate(&mut c, "frame-graph:edge:resource");
    let view = c.build();
    assert_eq!(c.selection(), Some(Selection::Resource(0)));
    let displayed = text(view.element());
    assert!(displayed.contains("buffer 64 B"));
    assert!(displayed.contains("storage write"));
    assert!(displayed.contains("input v1"));
    assert_eq!(
        view.accesses().iter().map(|a| a.pass).collect::<Vec<_>>(),
        vec![0, 1]
    );
    let (ui, rects) = layout(view.element());
    assert!(rects["frame-graph:details"].size.height > 0.);
    assert!(!ui.backend().to_draw_list().is_empty());
    activate(&mut c, "frame-graph:access:0:0:pass");
    assert_eq!(c.selection(), Some(Selection::Pass(0)));
    activate(&mut c, "frame-graph:resource:1");
    assert_eq!(c.selection(), Some(Selection::Resource(1)));
    assert!(matches!(
        r.resources()[1].layout,
        prism_editor_profiler::CapturedResourceLayout::Texture { .. }
    ));
    let view = c.build();
    let displayed = text(view.element());
    assert!(displayed.contains("texture [8, 8, 1]"));
    assert!(displayed.contains("Rgba8Unorm"));
    assert!(displayed.contains("mips 1"));
    assert_eq!(
        view.accesses().iter().map(|a| a.pass).collect::<Vec<_>>(),
        vec![1, 2]
    );
}

#[test]
fn independent_bounded_pages_reach_all_actual_entries_without_duplicates() {
    let r = Arc::new(report(41, 0));
    let mut c = Controller::new(FrameGraphPanelLimits {
        passes: 3,
        resources: 2,
        dependencies: 3,
        accesses: 1,
        budgets: 2,
    })
    .unwrap();
    c.publish(Some(r.clone())).unwrap();
    for (page, name, index) in [
        (Page::Passes, "passes", 0),
        (Page::Resources, "resources", 1),
        (Page::Dependencies, "dependencies", 2),
        (Page::Budgets, "budgets", 4),
    ] {
        let mut seen = BTreeSet::new();
        loop {
            let view = c.build();
            let stats = view.stats();
            for (n, cap) in stats.visible.iter().zip([3, 2, 3, 1, 2]) {
                assert!(*n <= cap);
            }
            let mut ui = Ui::new(RecordingBackend::new());
            ui.mount(view.element());
            assert!(ui.node_count() < 180, "bounded tree {}", ui.node_count());
            let ids: Vec<_> = match page {
                Page::Passes => view.passes().iter().map(|p| p.pass as usize).collect(),
                Page::Resources => view
                    .bindings()
                    .filter_map(|(key, _)| {
                        key.strip_prefix("frame-graph:resource:")
                            .map(|s| s.parse().unwrap())
                    })
                    .collect(),
                Page::Dependencies => view.dependencies().iter().map(|e| e.dependency).collect(),
                Page::Budgets => view.budgets().iter().map(|b| b.budget).collect(),
                _ => unreachable!(),
            };
            for id in ids {
                assert!(seen.insert(id), "no duplicate across pages");
            }
            let key = format!("frame-graph:{name}:next");
            if view.bindings().any(|(k, _)| k == key) {
                c.handle_event(&view, &key, &UiEvent::Activate).unwrap();
            } else {
                assert_eq!(seen.len(), stats.total[index]);
                break;
            }
        }
        while c.build().stats().offsets[index] > 0 {
            c.apply(Action::Previous(page)).unwrap();
        }
    }
    c.apply(Action::Pass(1)).unwrap();
    assert_eq!(c.build().accesses().len(), 1);
    let first = c.build().accesses()[0].ordinal;
    activate(&mut c, "frame-graph:accesses:next");
    assert_ne!(c.build().accesses()[0].ordinal, first);
    c.apply(Action::Resource(1)).unwrap();
    let first = c.build().accesses()[0].pass;
    activate(&mut c, "frame-graph:accesses:next");
    assert_ne!(c.build().accesses()[0].pass, first);
    c.apply(Action::Pass(40)).unwrap();
    assert!(c.build().passes().iter().any(|p| p.pass == 40));
    c.apply(Action::RelatedOnly(false)).unwrap();
    assert_eq!(c.build().stats().total[2], r.dependencies().len());
}

#[test]
fn maximum_pass_page_stays_in_canvas_and_cross_page_edges_reveal_endpoints() {
    let mut c = Controller::new(FrameGraphPanelLimits {
        passes: MAX_FRAME_GRAPH_PAGE_ROWS,
        ..Default::default()
    })
    .unwrap();
    c.publish(Some(Arc::new(report(41, 0)))).unwrap();
    let view = c.build();
    let (_, rects) = layout(view.element());
    let canvas = rects["frame-graph:canvas"];
    for pass in view.passes() {
        let rect = rects[&pass.key];
        assert!(rect.left() >= canvas.left());
        assert!(rect.left() + rect.size.width <= canvas.left() + canvas.size.width);
        assert!(rect.top() + rect.size.height <= canvas.top() + canvas.size.height);
    }
    c.apply(Action::Pass(32)).unwrap();
    let view = c.build();
    assert!(view.stats().off_page_dependencies > 0);
    let edge = view
        .dependencies()
        .iter()
        .find(|d| d.segments.is_empty())
        .unwrap();
    let endpoints = edge.endpoints;
    c.handle_event(
        &view,
        &format!("frame-graph:dependency:{}", edge.dependency),
        &UiEvent::Activate,
    )
    .unwrap();
    activate(&mut c, "frame-graph:edge:before");
    assert_eq!(c.selection(), Some(Selection::Pass(endpoints[0])));
    assert!(c.build().passes().iter().any(|p| p.pass == endpoints[0]));
}

#[test]
fn published_arc_controller_and_navigation_identity_fences_reject_stale_events() {
    let r = Arc::new(report(3, 0));
    let mut a = Controller::default();
    let mut b = Controller::default();
    a.publish(Some(r.clone())).unwrap();
    b.publish(Some(r.clone())).unwrap();
    let old = a.build();
    let key = old.passes()[0].key.clone();
    assert_eq!(
        b.handle_event(&old, &key, &UiEvent::Activate),
        Err(Error::StaleView)
    );
    assert_eq!(
        a.handle_event(
            &old,
            &key,
            &UiEvent::Select {
                mode: SelectionMode::Replace
            }
        ),
        Err(Error::UnboundEvent)
    );
    assert_eq!(
        a.handle_event(&old, "not-a-painted-target", &UiEvent::Activate),
        Err(Error::UnboundEvent)
    );
    assert_eq!(a.apply(Action::Pass(u32::MAX)), Err(Error::MissingTarget));
    a.handle_event(&old, &key, &UiEvent::Activate).unwrap();
    assert_eq!(
        a.handle_event(&old, &key, &UiEvent::Activate),
        Err(Error::StaleView)
    );
    let v = a.build();
    a.publish(Some(r.clone())).unwrap();
    assert_eq!(
        a.handle_event(&v, &key, &UiEvent::Activate),
        Err(Error::StaleView)
    );
    let v = a.build();
    let equal_report = Arc::new((*r).clone());
    assert!(!Arc::ptr_eq(&r, &equal_report));
    a.publish(Some(equal_report)).unwrap();
    assert_eq!(
        a.handle_event(&v, &key, &UiEvent::Activate),
        Err(Error::StaleView)
    );
    a.publish(None).unwrap();
    assert!(a.build().passes().is_empty());
    assert!(a.build().bindings().next().is_none());
    assert!(a.report().is_none());
}

#[test]
fn only_real_complete_cpu_measurements_enable_critical_path_and_budget_pixels() {
    let r = Arc::new(report(4, 4));
    let mut c = Controller::default();
    c.publish(Some(r.clone())).unwrap();
    assert!(r.passes().iter().all(|p| p.duration_ns.unwrap() > 0));
    assert_eq!(
        c.critical_path().unwrap(),
        &r.critical_path(GraphCaptureLimits::default())
            .unwrap()
            .unwrap()
    );
    let view = c.build();
    assert!(view.passes().iter().all(|p| p.critical));
    let (ui, rects) = layout(view.element());
    let pixels = rasterize(&ui.backend().to_draw_list(), 900, 2000);
    for gauge in view.budgets() {
        let source = &r.budgets()[gauge.budget];
        assert_eq!(
            (gauge.consumed, gauge.limit, gauge.exceeded),
            (source.consumed, source.limit, source.exceeded())
        );
        let fill = rects[&gauge.key];
        near(fill.size.width, 320. * gauge.fraction);
        if gauge.fraction > 0. {
            let pixel = pixels.pixel((fill.left() + 2.) as u32, (fill.top() + 4.) as u32);
            if gauge.exceeded {
                near(pixel[0], 0.95);
                near(pixel[1], 0.15);
            } else {
                near(pixel[0], 0.15);
                near(pixel[1], 0.75);
            }
        }
    }
    assert!(view
        .budgets()
        .iter()
        .any(|b| b.limit == 0 && b.consumed == 0 && !b.exceeded && b.fraction == 0.));
    assert!(view
        .budgets()
        .iter()
        .any(|b| b.limit == 0 && b.consumed > 0 && b.exceeded && b.fraction == 1.));
    c.publish(Some(Arc::new(report(4, 1)))).unwrap();
    assert!(c.critical_path().is_none());
    assert!(c.build().passes().iter().all(|p| !p.critical));
}

#[test]
fn admission_rejects_oversized_report_atomically_and_limits_and_empty_are_explicit() {
    for n in [0, MAX_FRAME_GRAPH_PAGE_ROWS + 1] {
        assert!(matches!(
            Controller::new(FrameGraphPanelLimits {
                passes: n,
                ..Default::default()
            }),
            Err(Error::InvalidLimits)
        ));
    }
    let mut c = Controller::default();
    assert!(c.build().passes().is_empty());
    let r = Arc::new(report(2, 0));
    c.publish(Some(r.clone())).unwrap();
    let view = c.build();
    // Malformed/admission regression derives from genuine encoded production
    // output. This enlarged diagnostic label is not a runtime capture claim.
    let mut wire = r.to_bytes(GraphCaptureLimits::default()).unwrap();
    let length_at = 48 + 8 + usize::from(r.resources()[0].owner.is_some()) * 8 + 8 + 8;
    let old_length =
        u64::from_le_bytes(wire[length_at..length_at + 8].try_into().unwrap()) as usize;
    let label = "x".repeat(5000);
    wire.splice(
        length_at..length_at + 8 + old_length,
        (label.len() as u64)
            .to_le_bytes()
            .into_iter()
            .chain(label.bytes()),
    );
    let big = GraphCaptureReport::from_bytes(
        &wire,
        GraphCaptureLimits {
            text_bytes: 8192,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(c.publish(Some(Arc::new(big))), Err(Error::ReportCapacity));
    assert!(Arc::ptr_eq(c.report().unwrap(), &r));
    c.handle_event(&view, &view.passes()[0].key, &UiEvent::Activate)
        .unwrap();
    for prefix in 0..32 {
        assert!(
            GraphCaptureReport::from_bytes(&wire[..prefix], GraphCaptureLimits::default()).is_err()
        );
    }
}
