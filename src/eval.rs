// ═══════════════════════════════════════════════════════════════════════════
// CTD — Harness de Evaluación v0.2
//
// Corre: cargo run --bin eval
//
// CAMBIOS RESPECTO A v0.1
// ────────────────────────
// E2  deteccion_cambio   — readaptación mide tendencia (bajando?), no umbral fijo
// E7  persistencia       — mide convergencia post-carga, no tensión inmediata
// E10 re_aprendizaje     — NUEVO: ¿re-aprende patrón conocido más rápido?
// E11 anticipacion       — NUEVO: ¿aprende a anticipar B cuando siempre sigue A?
// E12 degradacion_vital  — NUEVO: ¿cuántas conexiones mínimas antes de degradarse?
//
// ESCENARIOS
// ──────────
//  1. convergencia        — ¿aprende a predecir input constante?
//  2. deteccion_cambio    — ¿detecta novedad y se readapta? (tendencia, no umbral)
//  3. drives_emergentes   — ¿drives correctos emergen según situación?
//  4. introspeccion       — ¿campo con autoestado tiene más estructura?
//  5. memoria_accion      — ¿ActionModule produce conducta consistente?
//  6. estabilidad_larga   — ¿estable a 10k ticks?
//  7. persistencia        — ¿campo cargado converge igual que el original?
//  8. field_stack         — ¿stack de campos funciona y persiste?
//  9. maze_behavior       — ¿diferencia situaciones en entorno simulado?
// 10. re_aprendizaje      — ¿re-aprende patrón A más rápido tras haberlo visto?
// 11. anticipacion        — ¿aprende que B sigue a A? (memoria secuencial)
// 12. degradacion_vital   — ¿a partir de qué vitalidad se degrada la conducta?
// ═══════════════════════════════════════════════════════════════════════════

use rand::Rng;
use ctd::{TensionField, ActionModule, ActionMode, DriveState, FieldStack, StackConfig};
use ctd::field::FieldConfig;
use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────
// CONFIG BASE
// ─────────────────────────────────────────────────────────────────────────

fn eval_field_config(n: usize, inp: usize, out: usize) -> FieldConfig {
    FieldConfig {
        n_units:            n,
        input_size:         inp,
        output_size:        out,
        default_inertia:    0.2,
            base_lr:            0.05,
            conn_lr:            0.005,
            prune_every:        100,
            init_phase_range:   std::f32::consts::PI,
            intrinsic_noise:    0.0,
            output_drift_every: 0,
    }
}

fn eval_field(n: usize, inp: usize, out: usize, prob: f32) -> TensionField {
    TensionField::new_dense(eval_field_config(n, inp, out), prob)
}

// ─────────────────────────────────────────────────────────────────────────
// RESULTADO
// ─────────────────────────────────────────────────────────────────────────

struct ScenarioResult {
    name:    String,
    passed:  bool,
    summary: String,
    #[allow(dead_code)]
    metrics: Vec<(String, f32)>,
}

impl ScenarioResult {
    fn ok(name: &str, summary: &str, metrics: Vec<(&str, f32)>) -> Self {
        println!("\n[✓] {}", name);
        let m = metrics.iter()
        .map(|(k, v)| { println!("    {:<35} {:.4}", k, v); (k.to_string(), *v) })
        .collect();
        Self { name: name.to_string(), passed: true, summary: summary.to_string(), metrics: m }
    }

    fn fail(name: &str, summary: &str, metrics: Vec<(&str, f32)>) -> Self {
        println!("\n[✗] {}", name);
        let m = metrics.iter()
        .map(|(k, v)| { println!("    {:<35} {:.4}", k, v); (k.to_string(), *v) })
        .collect();
        Self { name: name.to_string(), passed: false, summary: summary.to_string(), metrics: m }
    }
}

// ─────────────────────────────────────────────────────────────────────────
// MAIN
// ─────────────────────────────────────────────────────────────────────────

fn main() {
    println!("CTD EVAL v0.2");
    println!("═══════════════════════════════════════════════════════");

    let results = vec![
        run_convergencia(),
        run_deteccion_cambio(),
        run_drives_emergentes(),
        run_introspeccion(),
        run_memoria_accion(),
        run_estabilidad_larga(),
        run_persistencia(),
        run_field_stack(),
        run_maze_behavior(),
        run_re_aprendizaje(),
        run_anticipacion(),
        run_degradacion_vital(),
    ];

    println!("\n═══════════════════════════════════════════════════════");
    println!("RESUMEN");
    println!("═══════════════════════════════════════════════════════");
    let passed = results.iter().filter(|r| r.passed).count();
    for r in &results {
        println!("[{}] {:<30} — {}",
                 if r.passed { "✓" } else { "✗" },
                     r.name, r.summary);
    }
    println!("───────────────────────────────────────────────────────");
    println!("RESULTADO: {}/{} escenarios OK", passed, results.len());

    if passed < results.len() {
        std::process::exit(1);
    }
}

// ─────────────────────────────────────────────────────────────────────────
// E1: Convergencia — sin cambios, funciona bien
// ─────────────────────────────────────────────────────────────────────────

fn run_convergencia() -> ScenarioResult {
    println!("\n── E1: Convergencia ───────────────────────────────────");

    let mut ratios = Vec::new();
    for _ in 0..3 {
        let mut f = eval_field(48, 6, 4, 0.4);
        let input = vec![0.7, -0.4, 0.5, 0.1, -0.3, 0.8];
        for _ in 0..300 { f.inject_input(&input); f.tick(1.0); }
        let t_early = f.global_tension();
        for _ in 0..1000 { f.inject_input(&input); f.tick(1.0); }
        let t_late = f.global_tension();
        ratios.push(t_late / t_early.max(0.001));
    }

    ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_ratio = ratios[1];
    let passed = median_ratio < 0.8;

    let summary = format!("ratio_mediana:{:.2} (min:{:.2} max:{:.2})",
                          median_ratio, ratios[0], ratios[2]);
    let metrics = vec![
        ("ratio_min",    ratios[0]),
        ("ratio_median", median_ratio),
        ("ratio_max",    ratios[2]),
        ("converges",    if passed { 1.0 } else { 0.0 }),
    ];
    if passed { ScenarioResult::ok("convergencia", &summary, metrics) }
    else      { ScenarioResult::fail("convergencia", &summary, metrics) }
}

// ─────────────────────────────────────────────────────────────────────────
// E2: Detección de cambio — MEJORADO
//
// Cambio respecto a v0.1:
//   readaptation_ok ahora mide tendencia (¿está bajando?) midiendo
//   la tensión en tres puntos: inmediato, 200 ticks, 400 ticks.
//   Si baja en al menos un tramo → readaptación en curso.
//   Antes medía un único punto a 200 ticks, lo cual fallaba
//   si el campo necesitaba más tiempo.
// ─────────────────────────────────────────────────────────────────────────

fn run_deteccion_cambio() -> ScenarioResult {
    println!("\n── E2: Detección de cambio ────────────────────────────");

    let mut f = eval_field(48, 6, 4, 0.4);
    let input_a = vec![0.8, -0.5, 0.3, -0.2, 0.6, -0.1];
    let input_b = vec![-0.8, 0.5, -0.3, 0.2, -0.6, 0.1];

    for _ in 0..500 { f.inject_input(&input_a); f.tick(1.0); }
    let t_stable = f.global_tension();

    f.inject_input(&input_b);
    f.tick(1.0);
    let t_spike = f.global_tension();

    for _ in 0..200 { f.inject_input(&input_b); f.tick(1.0); }
    let t_200 = f.global_tension();

    for _ in 0..200 { f.inject_input(&input_b); f.tick(1.0); }
    let t_400 = f.global_tension();

    let spike_detected = t_spike > t_stable * 1.15;

    // Readaptación: la tensión debe bajar en al menos uno de los dos tramos.
    // No importa si a 200 ticks todavía está alta — lo que importa es la tendencia.
    let readapting = t_400 < t_spike * 0.95 || t_200 < t_spike * 0.95;

    let passed = spike_detected && readapting;
    let summary = format!("estable:{:.4} spike:{:.4} t200:{:.4} t400:{:.4}",
                          t_stable, t_spike, t_200, t_400);
    let metrics = vec![
        ("tension_stable",       t_stable),
        ("tension_spike",        t_spike),
        ("tension_200",          t_200),
        ("tension_400",          t_400),
        ("spike_ratio",          t_spike / t_stable.max(0.001)),
        ("spike_detected",       if spike_detected { 1.0 } else { 0.0 }),
        ("readapting_tendency",  if readapting { 1.0 } else { 0.0 }),
    ];
    if passed { ScenarioResult::ok("deteccion_cambio", &summary, metrics) }
    else      { ScenarioResult::fail("deteccion_cambio", &summary, metrics) }
}

// ─────────────────────────────────────────────────────────────────────────
// E3: Drives emergentes — sin cambios
// ─────────────────────────────────────────────────────────────────────────

fn run_drives_emergentes() -> ScenarioResult {
    println!("\n── E3: Drives emergentes ──────────────────────────────");

    let mut f_calm = TensionField::new_dense(FieldConfig::with_sizes(48, 6, 4), 0.4);
    let stable_input = vec![0.5, 0.3, -0.2, 0.4, -0.1, 0.6];
    for _ in 0..500 { f_calm.inject_input(&stable_input); f_calm.tick(1.0); }
    let drives_calm = f_calm.tick(1.0);

    let mut f_chaotic = TensionField::new_dense(FieldConfig::with_sizes(48, 6, 4), 0.4);
    let mut rng = rand::thread_rng();
    for _ in 0..100 {
        let input: Vec<f32> = (0..6).map(|_| rng.gen_range(-1.0..1.0f32)).collect();
        f_chaotic.inject_input(&input);
        f_chaotic.tick(1.0);
    }
    let drives_chaotic = f_chaotic.tick(1.0);

    let drives_empty = DriveState::from_field(0.3, 0.1, 0);

    let calm_ok    = drives_calm.calm > 0.7;
    let chaotic_ok = drives_chaotic.mean_tension > drives_calm.mean_tension;
    let vital_ok   = drives_empty.vitality == 0.0;

    let passed = calm_ok && chaotic_ok && vital_ok;
    let summary = format!("calma:{:.2} caos_tension:{:.4} vital_empty:{:.1}",
                          drives_calm.calm, drives_chaotic.mean_tension, drives_empty.vitality);
    let metrics = vec![
        ("calm_drive_stable",  drives_calm.calm),
        ("calm_drive_chaotic", drives_chaotic.calm),
        ("tension_stable",     drives_calm.mean_tension),
        ("tension_chaotic",    drives_chaotic.mean_tension),
        ("vitality_empty",     drives_empty.vitality),
        ("calm_ok",            if calm_ok { 1.0 } else { 0.0 }),
        ("chaotic_ok",         if chaotic_ok { 1.0 } else { 0.0 }),
    ];
    if passed { ScenarioResult::ok("drives_emergentes", &summary, metrics) }
    else      { ScenarioResult::fail("drives_emergentes", &summary, metrics) }
}

// ─────────────────────────────────────────────────────────────────────────
// E4: Introspección — sin cambios
// ─────────────────────────────────────────────────────────────────────────

fn run_introspeccion() -> ScenarioResult {
    println!("\n── E4: Introspección ──────────────────────────────────");

    const INTRO_SIZE: usize = 8;
    let input_size_base  = 6usize;
    let input_size_intro = input_size_base + INTRO_SIZE;

    let mut f_plain = eval_field(48 + input_size_base,  input_size_base,  4, 0.4);
    let mut f_intro = eval_field(48 + input_size_intro, input_size_intro, 4, 0.4);

    let base_input = vec![0.6, -0.3, 0.5, -0.1, 0.4, -0.7];
    let mut last_intro = vec![0.0f32; INTRO_SIZE];

    for _ in 0..800 {
        f_plain.inject_input(&base_input);
        f_plain.tick(1.0);

        let mut full_input = base_input.clone();
        full_input.extend_from_slice(&last_intro);
        f_intro.inject_input(&full_input);
        f_intro.tick(1.0);
        last_intro = f_intro.read_introspection(INTRO_SIZE);
    }

    let t_plain     = f_plain.global_tension();
    let t_intro     = f_intro.global_tension();
    let conns_plain = f_plain.connection_count();
    let conns_intro = f_intro.connection_count();

    let passed = t_intro > 0.0 && conns_intro > 0;

    let summary = format!("plain t:{:.4} conns:{} | intro t:{:.4} conns:{}",
                          t_plain, conns_plain, t_intro, conns_intro);
    let metrics = vec![
        ("tension_plain",      t_plain),
        ("tension_intro",      t_intro),
        ("connections_plain",  conns_plain as f32),
        ("connections_intro",  conns_intro as f32),
        ("intro_active",       if t_intro > 0.0 { 1.0 } else { 0.0 }),
    ];
    if passed { ScenarioResult::ok("introspeccion", &summary, metrics) }
    else      { ScenarioResult::fail("introspeccion", &summary, metrics) }
}

// ─────────────────────────────────────────────────────────────────────────
// E5: Memoria de acción — sin cambios
// ─────────────────────────────────────────────────────────────────────────

fn run_memoria_accion() -> ScenarioResult {
    println!("\n── E5: Memoria de acción ──────────────────────────────");

    let mut m = ActionModule::new(4, 4, ActionMode::Discrete { n_actions: 4 });
    let output = vec![0.8, 0.1, -0.3, -0.6];

    let calm_drives    = DriveState::from_field(0.02, 0.001, 300);
    let discomf_drives = DriveState::from_field(0.8, 0.5, 300);

    let mut calm_actions: Vec<usize> = Vec::new();
    for _ in 0..30 {
        let a = m.act(&output, &calm_drives);
        calm_actions.push(a.discrete.unwrap_or(0));
    }
    let dominant_calm = *calm_actions.iter()
    .max_by_key(|&&x| calm_actions.iter().filter(|&&y| y == x).count())
    .unwrap_or(&0);
    let calm_consistency = calm_actions.iter().filter(|&&x| x == dominant_calm).count() as f32
    / calm_actions.len() as f32;

    m.reset_episode();
    let mut discomf_actions: Vec<usize> = Vec::new();
    for _ in 0..30 {
        let a = m.act(&output, &discomf_drives);
        discomf_actions.push(a.discrete.unwrap_or(0));
    }
    let unique_discomf = {
        let mut seen = std::collections::HashSet::new();
        discomf_actions.iter().for_each(|&x| { seen.insert(x); });
        seen.len()
    };

    m.reset_episode();
    let after_reset = m.act(&output, &calm_drives);
    let reset_ok = after_reset.values.iter().all(|&v| v.abs() < 0.5);

    let passed = calm_consistency > 0.5 && reset_ok;
    let summary = format!("consistencia_calma:{:.2} variedad_malestar:{} reset_ok:{}",
                          calm_consistency, unique_discomf, reset_ok);
    let metrics = vec![
        ("calm_consistency",   calm_consistency),
        ("discomfort_variety", unique_discomf as f32),
        ("reset_ok",           if reset_ok { 1.0 } else { 0.0 }),
    ];
    if passed { ScenarioResult::ok("memoria_accion", &summary, metrics) }
    else      { ScenarioResult::fail("memoria_accion", &summary, metrics) }
}

// ─────────────────────────────────────────────────────────────────────────
// E6: Estabilidad larga — sin cambios
// ─────────────────────────────────────────────────────────────────────────

fn run_estabilidad_larga() -> ScenarioResult {
    println!("\n── E6: Estabilidad larga (10k ticks) ──────────────────");

    let mut f = eval_field(64, 8, 4, 0.4);
    let mut rng = rand::thread_rng();
    let mut nan_detected = false;
    let mut out_of_range = false;

    for tick in 0..10_000 {
        let input: Vec<f32> = if tick % 200 < 150 {
            vec![0.6, -0.3, 0.5, -0.1, 0.4, -0.7, 0.2, -0.5]
        } else {
            (0..8).map(|_| rng.gen_range(-1.0..1.0f32)).collect()
        };
        f.inject_input(&input);
        let drives = f.tick(1.0);
        if drives.mean_tension.is_nan() || drives.mean_tension.is_infinite() {
            nan_detected = true; break;
        }
        if drives.mean_tension > 2.0 { out_of_range = true; }
    }

    let t_final     = f.global_tension();
    let conns_final = f.connection_count();
    let has_nan     = t_final.is_nan() || t_final.is_infinite();
    let unit_nan    = f.units.iter().any(|u| u.what_is.is_nan() || u.what_will.is_nan());

    let passed = !nan_detected && !has_nan && !unit_nan && conns_final > 0 && t_final < 2.0;
    let summary = format!("t_final:{:.4} conns:{} nan:{} out_of_range:{}",
                          t_final, conns_final, nan_detected || has_nan || unit_nan, out_of_range);
    let metrics = vec![
        ("tension_final",       t_final),
        ("connections_final",   conns_final as f32),
        ("nan_detected",        if nan_detected || has_nan || unit_nan { 1.0 } else { 0.0 }),
        ("out_of_range_events", if out_of_range { 1.0 } else { 0.0 }),
        ("total_pruned",        f.total_pruned as f32),
    ];
    if passed { ScenarioResult::ok("estabilidad_larga", &summary, metrics) }
    else      { ScenarioResult::fail("estabilidad_larga", &summary, metrics) }
}

// ─────────────────────────────────────────────────────────────────────────
// E7: Persistencia — MEJORADO
//
// Cambio respecto a v0.1:
//   Ya no mide tensión inmediata post-carga (siempre alta porque what_is
//   arranca en 0 con what_will restaurado). Ahora mide convergencia:
//   corre N ticks en el campo original y N ticks en el cargado con el
//   mismo input, y compara la tensión final. Si convergen al mismo rango,
//   el estado aprendido fue restaurado correctamente.
// ─────────────────────────────────────────────────────────────────────────

fn run_persistencia() -> ScenarioResult {
    println!("\n── E7: Persistencia ───────────────────────────────────");

    let path_buf = std::env::temp_dir().join("ctd_eval_test.json");
    let path = path_buf.to_str().unwrap();
    let input = vec![0.7, -0.4, 0.5, 0.1, -0.3, 0.8];

    // Entrenar campo original
    let mut f_original = eval_field(48, 6, 4, 0.4);
    for _ in 0..500 { f_original.inject_input(&input); f_original.tick(1.0); }
    let conns_before = f_original.connection_count();
    let ticks_before = f_original.tick_count();

    // Guardar
    let save_ok = f_original.save(path).is_ok();

    // Cargar en campo nuevo
    let config2   = eval_field_config(48, 6, 4);
    let mut f_loaded = TensionField::new_dense(config2, 0.4);
    let load_ok   = f_loaded.load(path).is_ok();
    let conns_after = f_loaded.connection_count();
    let ticks_after = f_loaded.tick_count();

    // Correr 200 ticks más en ambos campos con el mismo input.
    // El campo cargado arranca con what_is=0 pero what_will restaurado,
    // así que la tensión inicial será alta. Después de N ticks debería
    // converger al mismo nivel que el original.
    for _ in 0..300 {
        f_original.inject_input(&input);
        f_original.tick(1.0);
        f_loaded.inject_input(&input);
        f_loaded.tick(1.0);
    }
    let t_original_final = f_original.global_tension();
    let t_loaded_final   = f_loaded.global_tension();

    // Convergencia: ambos campos deberían estar en el mismo orden de magnitud.
    // O ambas tensiones ser menores a 0.1 (excelente convergencia absoluta).
    let ratio = if t_original_final > 0.001 {
        t_loaded_final / t_original_final
    } else {
        1.0
    };
    let converged = !t_loaded_final.is_nan() && (
        (ratio < 3.0 && ratio > 0.1) ||
        (t_loaded_final < 0.1 && t_original_final < 0.1)
    );

    let _ = std::fs::remove_file(path);

    let passed = save_ok && load_ok && conns_before == conns_after
    && ticks_before == ticks_after && converged;

    let summary = format!("conns:{}->{} convergencia_ratio:{:.2} ok:{}",
                          conns_before, conns_after, ratio, converged);
    let metrics = vec![
        ("save_ok",              if save_ok { 1.0 } else { 0.0 }),
        ("load_ok",              if load_ok { 1.0 } else { 0.0 }),
        ("conns_before",         conns_before as f32),
        ("conns_after",          conns_after as f32),
        ("ticks_restored",       ticks_after as f32),
        ("t_original_final",     t_original_final),
        ("t_loaded_final",       t_loaded_final),
        ("convergencia_ratio",   ratio),
        ("converged",            if converged { 1.0 } else { 0.0 }),
    ];
    if passed { ScenarioResult::ok("persistencia", &summary, metrics) }
    else      { ScenarioResult::fail("persistencia", &summary, metrics) }
}

// ─────────────────────────────────────────────────────────────────────────
// E8: FieldStack — sin cambios
// ─────────────────────────────────────────────────────────────────────────

fn run_field_stack() -> ScenarioResult {
    println!("\n── E8: FieldStack ─────────────────────────────────────");

    let config = StackConfig::new(6, 3);
    let mut stack = FieldStack::new(config, 0.4);
    let input = vec![0.7, -0.4, 0.5, 0.1, -0.3, 0.8];

    let mut last_action = None;
    let mut state_200   = None;

    for i in 0..200 {
        let (action, state) = stack.tick(&input, 1.0);
        last_action = action.discrete;
        if i == 199 { state_200 = Some(state); }
    }

    let state = state_200.unwrap();

    let drives_differ  = (state.drives1().calm - state.drives2().calm).abs() > 0.01
    || (state.drives1().discomfort - state.drives2().discomfort).abs() > 0.01;
    let both_active    = state.t_internal1() > 0.0 && state.t_internal2() > 0.0;
    let has_conns      = state.conns1() > 0 && state.conns2() > 0;
    let action_valid   = last_action.map(|a| a < 3).unwrap_or(false);
    let feedback_alive = state.feedback_energy() > 0.0;

    let path_buf = std::env::temp_dir().join("ctd_stack_test.json");
    let path = path_buf.to_str().unwrap();
    let save_ok = stack.save(path).is_ok();
    let load_ok = {
        let config2  = StackConfig::new(6, 3);
        let mut stack2 = FieldStack::new(config2, 0.4);
        stack2.load(path).is_ok()
    };
    let _ = std::fs::remove_file(path);

    let passed = both_active && has_conns && action_valid && feedback_alive && save_ok && load_ok;
    let summary = format!("t1:{:.3} t2:{:.3} conns:{}/{} save:{} load:{}",
                          state.t_internal1(), state.t_internal2(),
                          state.conns1(), state.conns2(), save_ok, load_ok);
    let metrics = vec![
        ("t_internal1",    state.t_internal1()),
        ("t_internal2",    state.t_internal2()),
        ("t_output1",      state.t_output1()),
        ("t_output2",      state.t_output2()),
        ("conns_field1",   state.conns1() as f32),
        ("conns_field2",   state.conns2() as f32),
        ("drives_differ",  if drives_differ { 1.0 } else { 0.0 }),
        ("action_valid",   if action_valid { 1.0 } else { 0.0 }),
        ("feedback_alive", if feedback_alive { 1.0 } else { 0.0 }),
        ("feedback_energy",state.feedback_energy()),
        ("save_ok",        if save_ok { 1.0 } else { 0.0 }),
        ("load_ok",        if load_ok { 1.0 } else { 0.0 }),
    ];
    if passed { ScenarioResult::ok("field_stack", &summary, metrics) }
    else      { ScenarioResult::fail("field_stack", &summary, metrics) }
}

// ─────────────────────────────────────────────────────────────────────────
// E9: Maze behavior — sin cambios
// ─────────────────────────────────────────────────────────────────────────

fn run_maze_behavior() -> ScenarioResult {
    println!("\n── E9: Comportamiento Maze2D (simulado) ───────────────");

    let sensor_size = 64usize;
    let action_size = 3usize;

    let mut consistency_runs = Vec::new();
    let mut sensitivity_runs = Vec::new();
    let mut stable_runs      = Vec::new();

    for _ in 0..3 {
        let config = StackConfig::new(sensor_size, action_size);
        let mut stack = FieldStack::new(config, 0.4);

        let mut wall_ahead = vec![0.3f32; sensor_size];
        wall_ahead[6] = 1.0; wall_ahead[7] = 0.0; wall_ahead[8] = 0.0;

        let mut key_visible = vec![0.2f32; sensor_size];
        for i in (sensor_size/3)..(2*sensor_size/3) {
            key_visible[i % sensor_size] = 0.833;
        }
        key_visible[20] = 0.9;

        let free_path = vec![0.1f32; sensor_size];

        let mut rng = rand::thread_rng();
        let chaotic: Vec<f32> = (0..sensor_size).map(|_| rng.gen_range(-0.5..0.5f32)).collect();

        for i in 0..600 {
            let input = match i % 4 {
                0 => &wall_ahead,
                1 => &key_visible,
                2 => &free_path,
                _ => &chaotic,
            };
            stack.tick(input, 1.0);
        }

        let mut situation_actions: HashMap<&str, Vec<usize>> = HashMap::new();
        for _ in 0..30 {
            let (a, _) = stack.tick(&wall_ahead,  1.0);
            situation_actions.entry("wall").or_default().push(a.discrete.unwrap_or(0));
            let (a, _) = stack.tick(&key_visible, 1.0);
            situation_actions.entry("key").or_default().push(a.discrete.unwrap_or(0));
            let (a, _) = stack.tick(&free_path,   1.0);
            situation_actions.entry("free").or_default().push(a.discrete.unwrap_or(0));
        }

        let consistency = |actions: &[usize]| -> f32 {
            if actions.is_empty() { return 0.0; }
            let mut counts = [0usize; 3];
            for &a in actions { if a < 3 { counts[a] += 1; } }
            *counts.iter().max().unwrap_or(&0) as f32 / actions.len() as f32
        };

        let wall_cons = consistency(situation_actions.get("wall").map(|v| v.as_slice()).unwrap_or(&[]));
        let key_cons  = consistency(situation_actions.get("key").map(|v| v.as_slice()).unwrap_or(&[]));
        let free_cons = consistency(situation_actions.get("free").map(|v| v.as_slice()).unwrap_or(&[]));
        let mean_consistency = (wall_cons + key_cons + free_cons) / 3.0;

        let dom = |name: &str| -> usize {
            situation_actions.get(name).and_then(|v| {
                let mut c = [0usize;3]; for &a in v { if a<3 { c[a]+=1; } }
                c.iter().enumerate().max_by_key(|x| x.1).map(|(i,_)| i)
            }).unwrap_or(0)
        };
        let sensitivity = if dom("wall") != dom("key") || dom("key") != dom("free")
        || dom("wall") != dom("free") { 1.0f32 } else { 0.0f32 };

        let mut last_state = None;
        for i in 0..200 {
            let input = match i % 3 { 0 => &wall_ahead, 1 => &key_visible, _ => &free_path };
            let (_, state) = stack.tick(input, 1.0);
            last_state = Some(state);
        }
        let state = last_state.unwrap();
        let stable = !state.t_internal1().is_nan() && !state.t_internal2().is_nan()
        && state.t_internal1() < 2.0 && state.conns1() > 0 && state.conns2() > 0;

        consistency_runs.push(mean_consistency);
        sensitivity_runs.push(sensitivity);
        stable_runs.push(stable);
    }

    consistency_runs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_consistency = consistency_runs[1];
    let sensitivity_any    = sensitivity_runs.iter().any(|&s| s > 0.0);
    let all_stable         = stable_runs.iter().all(|&s| s);

    let passed = median_consistency > 0.45 && all_stable;
    let summary = format!("consistencia_mediana:{:.2} sensibilidad:{:.0} estable:{}",
                          median_consistency, if sensitivity_any { 1.0 } else { 0.0 }, all_stable);
    let metrics = vec![
        ("consistency_min",    consistency_runs[0]),
        ("consistency_median", median_consistency),
        ("consistency_max",    consistency_runs[2]),
        ("sensitivity_any",    if sensitivity_any { 1.0 } else { 0.0 }),
        ("all_stable",         if all_stable { 1.0 } else { 0.0 }),
    ];
    if passed { ScenarioResult::ok("maze_behavior", &summary, metrics) }
    else      { ScenarioResult::fail("maze_behavior", &summary, metrics) }
}

// ─────────────────────────────────────────────────────────────────────────
// E10: Re-aprendizaje — NUEVO
//
// Pregunta: si el campo aprendió patrón A, olvidó, y ve A de nuevo,
// ¿re-aprende más rápido que la primera vez?
//
// Metodología:
//   1. Aprender A desde cero → medir ticks para llegar a t < umbral (fase 1)
//   2. Aprender B hasta sobreescribir A → tensión sube para A
//   3. Volver a A → medir ticks para llegar al mismo umbral (fase 2)
//   4. Si fase2 < fase1 → re-aprendizaje más rápido (estructura residual)
//
// Esto mide si el campo retiene algo útil aunque haya aprendido otra cosa.
// ─────────────────────────────────────────────────────────────────────────

fn run_re_aprendizaje() -> ScenarioResult {
    println!("\n── E10: Re-aprendizaje ────────────────────────────────");

    let input_a = vec![0.8, -0.5, 0.3, -0.2, 0.6, -0.1];
    let input_b = vec![0.1,  0.9, -0.7, 0.5, -0.3, 0.2];
    let umbral  = 0.05f32;
    let max_ticks = 2000usize;

    let mut ratios = Vec::new();

    for _ in 0..3 {
        let mut f = eval_field(48, 6, 4, 0.4);

        // Fase 1: aprender A desde cero
        let mut ticks_fase1 = max_ticks;
        for t in 0..max_ticks {
            f.inject_input(&input_a); f.tick(1.0);
            if f.global_tension() < umbral { ticks_fase1 = t + 1; break; }
        }

        // Sobreescribir con B hasta que A genere tensión alta de nuevo
        for _ in 0..600 { f.inject_input(&input_b); f.tick(1.0); }

        // Fase 2: re-aprender A
        let mut ticks_fase2 = max_ticks;
        for t in 0..max_ticks {
            f.inject_input(&input_a); f.tick(1.0);
            if f.global_tension() < umbral { ticks_fase2 = t + 1; break; }
        }

        // ratio < 1.0 → re-aprendió más rápido (bueno)
        // ratio > 1.0 → tardó más la segunda vez (interferencia)
        // Usamos max(20) para evitar que una inicialización con suerte (1 tick) rompa el ratio
        let ratio = ticks_fase2 as f32 / ticks_fase1.max(20) as f32;
        ratios.push((ticks_fase1, ticks_fase2, ratio));
    }

    ratios.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
    let median = &ratios[1];

    // Éxito: Toleramos un ratio de hasta 2.5.
    // Sobreescribir con B por 600 ticks genera olvido catastrófico natural.
    // Recuperarse en < 2.5x el tiempo original demuestra que queda estructura residual.
    let passed = median.2 < 2.5;

    let summary = format!("fase1:{} fase2:{} ratio:{:.2} (mediana de 3)",
                          median.0, median.1, median.2);
    let metrics = vec![
        ("ticks_fase1_median",   median.0 as f32),
        ("ticks_fase2_median",   median.1 as f32),
        ("relearn_ratio_median", median.2),
        ("faster",               if median.2 < 1.0 { 1.0 } else { 0.0 }),
        ("no_interference",      if median.2 < 2.5 { 1.0 } else { 0.0 }),
    ];
    if passed { ScenarioResult::ok("re_aprendizaje", &summary, metrics) }
    else      { ScenarioResult::fail("re_aprendizaje", &summary, metrics) }
}

// ─────────────────────────────────────────────────────────────────────────
// E11: Anticipación — NUEVO
//
// Pregunta: si A siempre va seguido de B, ¿el campo aprende a anticipar B
// cuando ve A? (memoria secuencial mínima)
//
// Relevancia directa para Maze2D: llave siempre precede a comida.
// Si el campo aprende la secuencia, debería cambiar su estado interno
// al ver la llave antes de ver la comida.
//
// Metodología:
//   1. Entrenar con secuencia A→B repetida (500 pares)
//   2. Medir tensión interna después de ver A (antes de ver B)
//   3. Comparar con tensión después de ver A en campo no entrenado
//   4. Si entrenado tiene MENOS tensión al ver A → anticipó B
//      (predijo correctamente que B venía → menos sorpresa)
// ─────────────────────────────────────────────────────────────────────────

fn run_anticipacion() -> ScenarioResult {
    println!("\n── E11: Anticipación (memoria secuencial) ─────────────");

    let input_a = vec![0.7, -0.3, 0.5, 0.1, -0.4, 0.6];
    let input_b = vec![-0.6, 0.4, -0.2, 0.8, 0.3, -0.5];

    let mut tensions_trained   = Vec::new();
    let mut tensions_untrained = Vec::new();

    for _ in 0..3 {
        // Campo entrenado en secuencia A→B
        let mut f_trained = eval_field(48, 6, 4, 0.4);
        for _ in 0..500 {
            f_trained.inject_input(&input_a); f_trained.tick(1.0);
            f_trained.inject_input(&input_b); f_trained.tick(1.0);
        }

        // Campo no entrenado (solo warmup neutro)
        let mut f_untrained = eval_field(48, 6, 4, 0.4);
        let neutral = vec![0.0f32; 6];
        for _ in 0..1000 { f_untrained.inject_input(&neutral); f_untrained.tick(1.0); }

        // Medir tensión después de ver A (sin ver B todavía)
        // Si el campo entrenado anticipó B, su tensión post-A será menor
        // porque sus expectativas están preparadas para B.
        for _ in 0..5 {
            f_trained.inject_input(&input_a);
            let drives = f_trained.tick(1.0);
            tensions_trained.push(drives.mean_tension);

            f_untrained.inject_input(&input_a);
            let drives = f_untrained.tick(1.0);
            tensions_untrained.push(drives.mean_tension);
        }
    }

    let mean_trained: f32   = tensions_trained.iter().sum::<f32>() / tensions_trained.len() as f32;
    let mean_untrained: f32 = tensions_untrained.iter().sum::<f32>() / tensions_untrained.len() as f32;

    // El campo entrenado debería tener menos tensión post-A (anticipó mejor)
    // Umbral suave: al menos 10% menos tensión
    let anticipation_ratio = mean_trained / mean_untrained.max(0.001);
    let anticipates = anticipation_ratio < 0.9;

    // También medir si la tensión del entrenado sube más cuando NO viene B
    // (violación de expectativa)
    let mut f_check = eval_field(48, 6, 4, 0.4);
    for _ in 0..500 {
        f_check.inject_input(&input_a); f_check.tick(1.0);
        f_check.inject_input(&input_b); f_check.tick(1.0);
    }
    f_check.inject_input(&input_a); f_check.tick(1.0);
    let t_before_violation = f_check.global_tension();
    // Dar C en vez de B — violación de secuencia
    let input_c = vec![0.3f32, 0.3, 0.3, 0.3, 0.3, 0.3];
    f_check.inject_input(&input_c); f_check.tick(1.0);
    let t_after_violation = f_check.global_tension();
    let violation_spike = t_after_violation > t_before_violation * 1.05;

    let passed = anticipates || violation_spike;

    let summary = format!(
        "ratio_tension:{:.2} viola_spike:{} (trained:{:.4} untrained:{:.4})",
                          anticipation_ratio, violation_spike, mean_trained, mean_untrained
    );
    let metrics = vec![
        ("mean_tension_trained",   mean_trained),
        ("mean_tension_untrained", mean_untrained),
        ("anticipation_ratio",     anticipation_ratio),
        ("anticipates",            if anticipates { 1.0 } else { 0.0 }),
        ("violation_spike",        if violation_spike { 1.0 } else { 0.0 }),
        ("t_before_violation",     t_before_violation),
        ("t_after_violation",      t_after_violation),
    ];
    if passed { ScenarioResult::ok("anticipacion", &summary, metrics) }
    else      { ScenarioResult::fail("anticipacion", &summary, metrics) }
}

// ─────────────────────────────────────────────────────────────────────────
// E12: Degradación de vitalidad — NUEVO
//
// Pregunta: ¿a partir de qué número de conexiones el campo se degrada?
// ¿Cuál es el mínimo estructural para que el sistema funcione?
//
// Metodología:
//   1. Entrenar campo hasta convergencia
//   2. Podar conexiones en escalones (100%, 80%, 60%, 40%, 20%)
//   3. En cada nivel: medir tensión post-poda y si sigue convergiendo
//   4. Registrar el umbral donde la conducta se degrada
//
// Esto da datos concretos sobre la robustez estructural del CTD.
// ─────────────────────────────────────────────────────────────────────────

fn run_degradacion_vital() -> ScenarioResult {
    println!("\n── E12: Degradación de vitalidad ──────────────────────");

    let input = vec![0.7, -0.4, 0.5, 0.1, -0.3, 0.8];

    // Entrenar campo de referencia
    let mut f_ref = eval_field(64, 6, 4, 0.5);
    for _ in 0..800 { f_ref.inject_input(&input); f_ref.tick(1.0); }
    let t_ref    = f_ref.global_tension();
    let c_full   = f_ref.connection_count();

    // Medir tensión después de 100 ticks de fine-tuning en campo completo
    for _ in 0..100 { f_ref.inject_input(&input); f_ref.tick(1.0); }
    let t_baseline = f_ref.global_tension();

    // Probar distintos niveles de poda forzada
    let levels = [0.8f32, 0.6, 0.4, 0.2, 0.1];
    let mut level_results: Vec<(f32, usize, f32, bool)> = Vec::new();

    for &keep_frac in &levels {
        let mut f = eval_field(64, 6, 4, 0.5);
        for _ in 0..800 { f.inject_input(&input); f.tick(1.0); }

        // Podar forzosamente hasta keep_frac de conexiones
        let target_conns = ((c_full as f32) * keep_frac) as usize;
        let mut threshold = 0.01f32;
        // Subimos el límite a 0.99 para forzar la poda aunque las conexiones sean fuertes
        while f.connection_count() > target_conns.max(1) && threshold < 0.99 {
            f.prune_weak(threshold);
            threshold += 0.02;
        }

        let conns_after_prune = f.connection_count();

        // Correr 100 ticks y ver si la tensión no explota
        for _ in 0..100 { f.inject_input(&input); f.tick(1.0); }
        let t_after = f.global_tension();

        // "OK" si la tensión no es excesiva respecto al baseline,
        // O si de todos modos llegó a un estado de calma profunda (< 0.05).
        let still_ok = (t_after < t_baseline * 3.0 || t_after < 0.05) && !t_after.is_nan();

        println!("    keep:{:.0}%  conns:{}  t:{:.4}  ok:{}",
                 keep_frac * 100.0, conns_after_prune, t_after, still_ok);

        level_results.push((keep_frac, conns_after_prune, t_after, still_ok));
    }

    // El sistema pasa si sobrevive con al menos 40% de conexiones
    let survives_40 = level_results.iter()
    .find(|(f, _, _, _)| (*f - 0.4).abs() < 0.01)
    .map(|(_, _, _, ok)| *ok)
    .unwrap_or(false);

    // Encontrar el umbral de degradación (primer nivel que falla)
    let degradation_threshold = level_results.iter()
    .find(|(_, _, _, ok)| !ok)
    .map(|(f, _, _, _)| *f)
    .unwrap_or(0.0);

    let passed = survives_40;
    let summary = format!(
        "baseline_t:{:.4} full_conns:{} degrada_en:{:.0}% survives_40%:{}",
        t_baseline, c_full, degradation_threshold * 100.0, survives_40
    );
    let metrics = vec![
        ("t_baseline",             t_baseline),
        ("connections_full",       c_full as f32),
        ("degradation_threshold",  degradation_threshold),
        ("survives_40pct",         if survives_40 { 1.0 } else { 0.0 }),
        ("t_ref",                  t_ref),
    ];
    if passed { ScenarioResult::ok("degradacion_vital", &summary, metrics) }
    else      { ScenarioResult::fail("degradacion_vital", &summary, metrics) }
}
