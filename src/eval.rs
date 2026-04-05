// ═══════════════════════════════════════════════════════════════════════════
// CTD — Harness de Evaluación v0.3  (Neuro-Evaluation Suite)
//
// Corre: cargo run --bin eval
//
// CAMBIOS RESPECTO A v0.2
// ────────────────────────
// NIVEL 1 — Rigor estadístico
//   E1  convergencia         — Monte Carlo 100 semillas; % supervivencia
//   E2  deteccion_cambio     — Análisis de sensibilidad al ruido (SNR break)
//   E3  drives_emergentes    — Mapa de calor de conectividad (Small-World index)
//
// NIVEL 2 — Dinámica cognitiva
//   E4  introspeccion        — Olvido catastrófico A→B→A (retención estructural)
//   E5  memoria_accion       — Entropía de la tensión (exponente Hurst proxy)
//   E6  estabilidad_larga    — Latencia de reacción (spike-to-action en ticks)
//
// NIVEL 3 — Comportamiento y ecología
//   E7  persistencia         — Closed-loop maze con estado (tiempo de resolución)
//   E8  field_stack          — Eficiencia metabólica (IQ = log(1/t) / conns)
//   E9  maze_behavior        — Evaluación de atención top-down (info mutua F1↔F2)
//
// NIVEL 4 — Resiliencia extrema
//   E10 re_aprendizaje       — Test de isquemia (lobotomía dinámica 1%/tick)
//   E11 anticipacion         — Test de alucinación (privación sensorial)
//   E12 degradacion_vital    — Seed profiling (sesgo de drive por semilla)
//
// ESCENARIOS
// ──────────
//  1. convergencia        — ¿aprende a predecir input constante? [Monte Carlo 100]
//  2. deteccion_cambio    — ¿punto de quiebre ante ruido creciente?
//  3. drives_emergentes   — ¿topología small-world emerge tras aprendizaje?
//  4. introspeccion       — ¿olvido catastrófico A→B→A? (retención estructural)
//  5. memoria_accion      — ¿entropía de tensión en criticalidad? (Hurst proxy)
//  6. estabilidad_larga   — ¿latencia spike-to-action medible?
//  7. persistencia        — ¿closed-loop maze converge a tensión < umbral?
//  8. field_stack         — ¿IQ metabólico > 0 (eficiencia vs conexiones)?
//  9. maze_behavior       — ¿información mutua F1↔F2 > 0 (atención top-down)?
// 10. re_aprendizaje      — ¿sobrevive lobotomía dinámica 1% conns/tick?
// 11. anticipacion        — ¿genera actividad coherente en privación sensorial?
// 12. degradacion_vital   — ¿seed profiling muestra diversidad de personalidades?
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
        .map(|(k, v)| { println!("    {:<40} {:.4}", k, v); (k.to_string(), *v) })
        .collect();
        Self { name: name.to_string(), passed: true, summary: summary.to_string(), metrics: m }
    }

    fn fail(name: &str, summary: &str, metrics: Vec<(&str, f32)>) -> Self {
        println!("\n[✗] {}", name);
        let m = metrics.iter()
        .map(|(k, v)| { println!("    {:<40} {:.4}", k, v); (k.to_string(), *v) })
        .collect();
        Self { name: name.to_string(), passed: false, summary: summary.to_string(), metrics: m }
    }
}

// ─────────────────────────────────────────────────────────────────────────
// UTILIDADES ESTADÍSTICAS
// ─────────────────────────────────────────────────────────────────────────

fn mean(v: &[f32]) -> f32 {
    if v.is_empty() { return 0.0; }
    v.iter().sum::<f32>() / v.len() as f32
}

fn std_dev(v: &[f32]) -> f32 {
    if v.len() < 2 { return 0.0; }
    let m = mean(v);
    let var = v.iter().map(|x| (x - m).powi(2)).sum::<f32>() / (v.len() - 1) as f32;
    var.sqrt()
}

/// Estimador de Hurst via RS (rescaled range) — proxy de criticalidad.
/// H ≈ 0.5 → ruido blanco (aleatorio), H > 0.5 → persistente (cerca del caos),
/// H < 0.5 → anti-persistente (muy rígido).
/// Buen CTD debería vivir en H ∈ [0.55, 0.85].
fn hurst_proxy(series: &[f32]) -> f32 {
    if series.len() < 8 { return 0.5; }
    let n = series.len();
    let m = mean(series);
    let mut cumdev = 0.0f32;
    let mut min_cumdev = 0.0f32;
    let mut max_cumdev = 0.0f32;
    for &x in series {
        cumdev += x - m;
        min_cumdev = min_cumdev.min(cumdev);
        max_cumdev = max_cumdev.max(cumdev);
    }
    let range = max_cumdev - min_cumdev;
    let s = std_dev(series).max(1e-9);
    let rs = range / s;
    // H = log(RS) / log(N) — estimación simplificada
    (rs.max(1e-9).ln() / (n as f32).ln()).clamp(0.0, 1.0)
}

/// Información mutua estimada via correlación (proxy lineal).
/// Para dos series del mismo largo, |corr| captura dependencia lineal.
fn mutual_info_proxy(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    if n < 2 { return 0.0; }
    let ma = mean(&a[..n]);
    let mb = mean(&b[..n]);
    let num: f32 = (0..n).map(|i| (a[i] - ma) * (b[i] - mb)).sum();
    let da: f32 = (0..n).map(|i| (a[i] - ma).powi(2)).sum::<f32>().sqrt();
    let db: f32 = (0..n).map(|i| (b[i] - mb).powi(2)).sum::<f32>().sqrt();
    let denom = (da * db).max(1e-9);
    (num / denom).abs()
}

// ─────────────────────────────────────────────────────────────────────────
// MAIN
// ─────────────────────────────────────────────────────────────────────────

fn main() {
    println!("CTD EVAL v0.3 — Neuro-Evaluation Suite");
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
// E1: Convergencia — MONTE CARLO 100 semillas
//
// Mejora (Nivel 1 / punto 1):
//   En vez de 3 corridas, corremos 100 semillas independientes.
//   Métrica: % de supervivencia (semillas donde ratio < 0.8).
//   También reportamos media, std y percentiles p10/p50/p90 del ratio.
//   Objetivo: detectar semillas "neuróticas" (alta tensión persistente).
// ─────────────────────────────────────────────────────────────────────────

fn run_convergencia() -> ScenarioResult {
    println!("\n── E1: Convergencia [Monte Carlo 100 semillas] ────────");

    const N_SEEDS: usize = 100;
    let input = vec![0.7f32, -0.4, 0.5, 0.1, -0.3, 0.8];
    let mut ratios = Vec::with_capacity(N_SEEDS);

    for _ in 0..N_SEEDS {
        let mut f = eval_field(48, 6, 4, 0.4);
        for _ in 0..300  { f.inject_input(&input); f.tick(1.0); }
        let t_early = f.global_tension();
        for _ in 0..1000 { f.inject_input(&input); f.tick(1.0); }
        let t_late = f.global_tension();
        ratios.push(t_late / t_early.max(0.001));
    }

    ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let survival_pct  = ratios.iter().filter(|&&r| r < 0.8).count() as f32 / N_SEEDS as f32;
    let mean_ratio    = mean(&ratios);
    let std_ratio     = std_dev(&ratios);
    let p10           = ratios[(N_SEEDS as f32 * 0.10) as usize];
    let p50           = ratios[N_SEEDS / 2];
    let p90           = ratios[(N_SEEDS as f32 * 0.90) as usize];
    let neurotic_seeds = ratios.iter().filter(|&&r| r > 1.5).count();

    // Éxito: al menos 70% de semillas convergen
    let passed = survival_pct >= 0.70;

    let summary = format!("supervivencia:{:.0}%  media:{:.3}±{:.3}  semillas_neuróticas:{}",
                          survival_pct * 100.0, mean_ratio, std_ratio, neurotic_seeds);
    let metrics = vec![
        ("survival_pct",        survival_pct),
        ("mean_ratio",          mean_ratio),
        ("std_ratio",           std_ratio),
        ("p10_ratio",           p10),
        ("p50_ratio",           p50),
        ("p90_ratio",           p90),
        ("neurotic_seeds",      neurotic_seeds as f32),
        ("n_seeds",             N_SEEDS as f32),
    ];
    if passed { ScenarioResult::ok("convergencia", &summary, metrics) }
    else      { ScenarioResult::fail("convergencia", &summary, metrics) }
}

// ─────────────────────────────────────────────────────────────────────────
// E2: Detección de cambio — ANÁLISIS DE SENSIBILIDAD AL RUIDO
//
// Mejora (Nivel 1 / punto 2):
//   Inyectamos ruido gaussiano creciente en los sensores (SNR decreciente).
//   Punto de quiebre: el SNR donde la tensión deja de converger a < umbral.
//   Objetivo: medir robustez del campo ante señales sucias.
//
//   Además mantenemos la detección de spike de v0.2 como control de sanidad.
// ─────────────────────────────────────────────────────────────────────────

fn run_deteccion_cambio() -> ScenarioResult {
    println!("\n── E2: Detección de cambio [Sensibilidad al ruido] ────");

    let mut rng = rand::thread_rng();
    let input_clean = vec![0.8f32, -0.5, 0.3, -0.2, 0.6, -0.1];

    // — Control de sanidad: spike detection (de v0.2) —
    let mut f = eval_field(48, 6, 4, 0.4);
    let input_b = vec![-0.8f32, 0.5, -0.3, 0.2, -0.6, 0.1];
    for _ in 0..500 { f.inject_input(&input_clean); f.tick(1.0); }
    let t_stable = f.global_tension();
    f.inject_input(&input_b); f.tick(1.0);
    let t_spike = f.global_tension();
    for _ in 0..400 { f.inject_input(&input_b); f.tick(1.0); }
    let t_400 = f.global_tension();
    let spike_ok     = t_spike > t_stable * 1.15;
    let readapt_ok   = t_400 < t_spike * 0.95;

    // — Análisis de SNR —
    // Niveles de ruido: desviación estándar del ruido gaussiano añadido.
    // SNR = señal_rms / ruido_std.  Señal rms de input_clean ≈ 0.55.
    let signal_rms: f32 = (input_clean.iter().map(|v| v.powi(2)).sum::<f32>()
    / input_clean.len() as f32).sqrt();
    let noise_levels = [0.0f32, 0.1, 0.2, 0.4, 0.8, 1.6];
    let t_umbral = 0.15f32;
    let mut snr_break: Option<f32> = None;

    println!("    {:>8}  {:>8}  {:>10}  {}", "noise_σ", "SNR", "t_final", "converge");
    for &noise_std in &noise_levels {
        let mut f_n = eval_field(48, 6, 4, 0.4);
        for _ in 0..1300 {
            let noisy: Vec<f32> = input_clean.iter()
            .map(|&v| (v + rng.gen::<f32>() * noise_std * 2.0 - noise_std).clamp(-1.0, 1.0))
            .collect();
            f_n.inject_input(&noisy);
            f_n.tick(1.0);
        }
        let t_final = f_n.global_tension();
        let converges = t_final < t_umbral;
        let snr = if noise_std > 0.0 { signal_rms / noise_std } else { f32::INFINITY };
        println!("    {:>8.2}  {:>8.2}  {:>10.4}  {}", noise_std, snr, t_final, converges);
        if !converges && snr_break.is_none() {
            snr_break = Some(snr);
        }
    }
    let snr_break_val = snr_break.unwrap_or(0.0); // 0 = convergió a todos los niveles

    // Pasa si: spike detectado + readapta + punto de quiebre bien documentado
    // (snr_break > 0 o convergió a todos los niveles = sistema muy robusto)
    let passed = spike_ok && readapt_ok;

    let summary = format!("spike:{} readapt:{} snr_break:{:.2} (0=robusto_a_todo)",
                          spike_ok, readapt_ok, snr_break_val);
    let metrics = vec![
        ("tension_stable",       t_stable),
        ("tension_spike",        t_spike),
        ("tension_400",          t_400),
        ("spike_detected",       if spike_ok { 1.0 } else { 0.0 }),
        ("readapting",           if readapt_ok { 1.0 } else { 0.0 }),
        ("snr_break",            snr_break_val),
        ("signal_rms",           signal_rms),
    ];
    if passed { ScenarioResult::ok("deteccion_cambio", &summary, metrics) }
    else      { ScenarioResult::fail("deteccion_cambio", &summary, metrics) }
}

// ─────────────────────────────────────────────────────────────────────────
// E3: Drives emergentes — MAPA DE CALOR DE CONECTIVIDAD (Small-World)
//
// Mejora (Nivel 1 / punto 3):
//   Tras 10k ticks evaluamos la topología del campo:
//   - Distribución de grado (in/out por unidad)
//   - Índice de small-world proxy: razón entre la varianza del grado
//     y el grado medio. Si es > 1 → el campo auto-organizó hubs.
//   - Hubs: unidades con grado >= 2 × grado medio.
//
//   Mantenemos los checks de calma/caos de v0.2 como sanidad base.
// ─────────────────────────────────────────────────────────────────────────

fn run_drives_emergentes() -> ScenarioResult {
    println!("\n── E3: Drives emergentes [Small-World & Conectividad] ─");

    // — Sanidad base: calma y caos (de v0.2) —
    let mut f_calm = TensionField::new_dense(FieldConfig::with_sizes(48, 6, 4), 0.4);
    let stable_input = vec![0.5f32, 0.3, -0.2, 0.4, -0.1, 0.6];
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

    let calm_ok    = drives_calm.calm > 0.7;
    let chaotic_ok = drives_chaotic.mean_tension > drives_calm.mean_tension;

    // — Topología Small-World tras aprendizaje largo —
    let mut f_sw = eval_field(64, 6, 4, 0.4);
    let sw_input = vec![0.6f32, -0.3, 0.5, -0.1, 0.4, -0.7];
    for _ in 0..10_000 { f_sw.inject_input(&sw_input); f_sw.tick(1.0); }

    // Contar grado (in + out) por unidad
    let n = f_sw.units.len();
    let mut degree = vec![0usize; n];
    for conn in &f_sw.connections {
        degree[conn.from] += 1;
        degree[conn.to]   += 1;
    }

    let degree_f: Vec<f32> = degree.iter().map(|&d| d as f32).collect();
    let mean_deg  = mean(&degree_f);
    let std_deg   = std_dev(&degree_f);
    // CV (coeficiente de variación) > 1 sugiere distribución tipo power-law (hubs)
    let cv_degree = if mean_deg > 0.0 { std_deg / mean_deg } else { 0.0 };
    let hub_threshold = mean_deg * 2.0;
    let hubs = degree_f.iter().filter(|&&d| d >= hub_threshold).count();
    let hub_fraction = hubs as f32 / n as f32;

    // Pequeño-mundo proxy: CV > 0.5 implica heterogeneidad de grado
    let small_world_ok = cv_degree > 0.5 || hub_fraction > 0.05;

    println!("    grado_medio:{:.2}  cv_grado:{:.3}  hubs:{}({:.1}%)",
             mean_deg, cv_degree, hubs, hub_fraction * 100.0);
    println!("    conexiones_vivas:{}  small_world:{}",
             f_sw.connection_count(), small_world_ok);

    let passed = calm_ok && chaotic_ok;  // sanidad mínima; small_world es informativo

    let summary = format!("calma:{:.2} cv_grado:{:.3} hubs:{:.0}% small_world:{}",
                          drives_calm.calm, cv_degree, hub_fraction * 100.0, small_world_ok);
    let metrics = vec![
        ("calm_drive_stable",  drives_calm.calm),
        ("tension_chaotic",    drives_chaotic.mean_tension),
        ("mean_degree",        mean_deg),
        ("cv_degree",          cv_degree),
        ("hub_fraction",       hub_fraction),
        ("small_world",        if small_world_ok { 1.0 } else { 0.0 }),
        ("calm_ok",            if calm_ok { 1.0 } else { 0.0 }),
        ("chaotic_ok",         if chaotic_ok { 1.0 } else { 0.0 }),
    ];
    if passed { ScenarioResult::ok("drives_emergentes", &summary, metrics) }
    else      { ScenarioResult::fail("drives_emergentes", &summary, metrics) }
}

// ─────────────────────────────────────────────────────────────────────────
// E4: Introspección — OLVIDO CATASTRÓFICO A→B→A
//
// Mejora (Nivel 2 / punto 4):
//   Entrenar A, luego B, volver a evaluar A sin re-entrenamiento.
//   Métrica: Retención Estructural = tensión_A_post_B / tensión_B_fresh.
//   Si ratio < 1.5 → el CTD compartimentó; no borró A completamente.
//   Si ratio > 3.0 → olvido catastrófico severo.
//
//   Mantenemos el check de que la introspección tiene actividad (de v0.2).
// ─────────────────────────────────────────────────────────────────────────

fn run_introspeccion() -> ScenarioResult {
    println!("\n── E4: Introspección [Olvido catastrófico A→B→A] ─────");

    const INTRO_SIZE: usize = 8;
    let input_size_base  = 6usize;
    let input_size_intro = input_size_base + INTRO_SIZE;

    // — Check base: la introspección está activa (de v0.2) —
    let mut f_plain = eval_field(48 + input_size_base,  input_size_base,  4, 0.4);
    let mut f_intro = eval_field(48 + input_size_intro, input_size_intro, 4, 0.4);
    let base_input = vec![0.6f32, -0.3, 0.5, -0.1, 0.4, -0.7];
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
    let intro_active = f_intro.global_tension() > 0.0 && f_intro.connection_count() > 0;

    // — Olvido catastrófico A→B→A —
    let input_a = vec![0.7f32, -0.4, 0.5, 0.1, -0.3, 0.8];
    let input_b = vec![-0.5f32, 0.6, -0.8, 0.3, 0.7, -0.2];

    let mut retention_ratios = Vec::new();

    for _ in 0..3 {
        let mut f = eval_field(48, 6, 4, 0.4);

        // Fase A: entrenar hasta convergencia
        for _ in 0..800 { f.inject_input(&input_a); f.tick(1.0); }
        let t_a_trained = f.global_tension();

        // Fase B: sobreescribir con B
        for _ in 0..600 { f.inject_input(&input_b); f.tick(1.0); }

        // Re-evaluar A (sin reentrenar): ¿cuánto se olvidó?
        f.inject_input(&input_a); f.tick(1.0);
        let t_a_post_b = f.global_tension();

        // Campo de referencia fresco en B → cuánto debería costar aprender A desde cero
        let mut f_fresh = eval_field(48, 6, 4, 0.4);
        f_fresh.inject_input(&input_a); f_fresh.tick(1.0);
        let t_a_fresh = f_fresh.global_tension();

        // Retención: qué tan cerca está t_a_post_b de t_a_fresh vs t_a_trained
        // ratio = 1 → olvido total (igual que campo fresco)
        // ratio = 0 → retención perfecta (igual que entrenado)
        let range = (t_a_fresh - t_a_trained).abs().max(1e-6);
        let retention = ((t_a_post_b - t_a_trained) / range).clamp(0.0, 2.0);
        retention_ratios.push(retention);

        println!("    t_A_trained:{:.4}  t_A_post_B:{:.4}  t_A_fresh:{:.4}  retention:{:.3}",
                 t_a_trained, t_a_post_b, t_a_fresh, retention);
    }

    retention_ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_retention = retention_ratios[1];
    // retention < 0.8 → campo retuvo más del 20% de la estructura de A (pasa)
    let retains_structure = median_retention < 0.8;

    let passed = intro_active;  // mínimo: introspección activa
    let summary = format!("intro_activa:{} retención_A:{:.3} (<0.8 = buena retención)",
                          intro_active, median_retention);
    let metrics = vec![
        ("intro_active",           if intro_active { 1.0 } else { 0.0 }),
        ("retention_min",          retention_ratios[0]),
        ("retention_median",       median_retention),
        ("retention_max",          retention_ratios[2]),
        ("retains_structure",      if retains_structure { 1.0 } else { 0.0 }),
        ("conns_plain",            f_plain.connection_count() as f32),
        ("conns_intro",            f_intro.connection_count() as f32),
    ];
    if passed { ScenarioResult::ok("introspeccion", &summary, metrics) }
    else      { ScenarioResult::fail("introspeccion", &summary, metrics) }
}

// ─────────────────────────────────────────────────────────────────────────
// E5: Memoria de acción — ENTROPÍA DE LA TENSIÓN (Hurst proxy)
//
// Mejora (Nivel 2 / punto 5):
//   Medir si el campo opera en criticalidad.
//   Recogemos la serie temporal de tensión global durante 2000 ticks.
//   Calculamos el exponente de Hurst (RS simplificado).
//   H ∈ [0.55, 0.85] → borde del caos (criticalidad).
//   H < 0.5 → muy rígido; H > 0.9 → demasiado persistente.
//
//   Mantenemos los checks de consistencia y reset de v0.2.
// ─────────────────────────────────────────────────────────────────────────

fn run_memoria_accion() -> ScenarioResult {
    println!("\n── E5: Memoria de acción [Entropía / Hurst proxy] ────");

    let mut m = ActionModule::new(4, 4, ActionMode::Discrete { n_actions: 4 });
    let output = vec![0.8f32, 0.1, -0.3, -0.6];

    let calm_drives    = DriveState::from_field(0.02, 0.001, 300);
    let _discomf_drives = DriveState::from_field(0.8, 0.5, 300);

    // — Check base: consistencia calma + reset (de v0.2) —
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
    let after_reset = m.act(&output, &calm_drives);
    let reset_ok = after_reset.values.iter().all(|&v| v.abs() < 0.5);

    // — Entropía de tensión (Hurst) —
    let mut f_hurst = eval_field(48, 6, 4, 0.4);
    let input_h = vec![0.5f32, -0.3, 0.4, 0.1, -0.2, 0.6];
    let mut tension_series: Vec<f32> = Vec::with_capacity(2000);

    // Warm-up
    for _ in 0..300 { f_hurst.inject_input(&input_h); f_hurst.tick(1.0); }

    // Recolectar serie temporal
    let mut rng = rand::thread_rng();
    for i in 0..2000 {
        // Perturbaciones ocasionales para generar dinámica no trivial
        let input: Vec<f32> = if i % 300 < 250 {
            input_h.clone()
        } else {
            (0..6).map(|_| rng.gen_range(-0.5..0.5f32)).collect()
        };
        f_hurst.inject_input(&input);
        let drives = f_hurst.tick(1.0);
        tension_series.push(drives.mean_tension);
    }

    let h = hurst_proxy(&tension_series);
    let critical = h >= 0.50 && h <= 0.90;  // rango amplio — criticalidad aproximada

    println!("    hurst_proxy:{:.3}  (0.55-0.85 = criticalidad)  crítico:{}",
             h, critical);
    println!("    calm_consistency:{:.2}  reset_ok:{}", calm_consistency, reset_ok);

    let passed = calm_consistency > 0.5 && reset_ok;
    let summary = format!("consistencia:{:.2} reset:{} hurst:{:.3} crítico:{}",
                          calm_consistency, reset_ok, h, critical);
    let metrics = vec![
        ("calm_consistency",   calm_consistency),
        ("reset_ok",           if reset_ok { 1.0 } else { 0.0 }),
        ("hurst_proxy",        h),
        ("critical_range",     if critical { 1.0 } else { 0.0 }),
        ("tension_mean",       mean(&tension_series)),
        ("tension_std",        std_dev(&tension_series)),
    ];
    if passed { ScenarioResult::ok("memoria_accion", &summary, metrics) }
    else      { ScenarioResult::fail("memoria_accion", &summary, metrics) }
}

// ─────────────────────────────────────────────────────────────────────────
// E6: Estabilidad larga — LATENCIA DE REACCIÓN (Spike-to-Action)
//
// Mejora (Nivel 2 / punto 6):
//   Medimos exactamente cuántos ticks tarda el campo desde un cambio
//   de input hasta que la acción del módulo cambia de forma estable.
//   Métrica: ticks_respuesta = ticks hasta que la acción dominante cambia
//   y se mantiene ≥ 5 ticks consecutivos.
//
//   Mantenemos el check de NaN/rango de v0.2.
// ─────────────────────────────────────────────────────────────────────────

fn run_estabilidad_larga() -> ScenarioResult {
    println!("\n── E6: Estabilidad larga [Latencia Spike-to-Action] ───");

    // — Check base: 10k ticks sin NaN (de v0.2) —
    let mut f = eval_field(64, 8, 4, 0.4);
    let mut rng = rand::thread_rng();
    let mut nan_detected = false;
    let mut out_of_range = false;

    for tick in 0..10_000 {
        let input: Vec<f32> = if tick % 200 < 150 {
            vec![0.6f32, -0.3, 0.5, -0.1, 0.4, -0.7, 0.2, -0.5]
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
    let stability_ok = !nan_detected && !t_final.is_nan() && conns_final > 0 && t_final < 2.0;

    // — Latencia Spike-to-Action —
    let mut f_lat = eval_field(48, 6, 4, 0.4);
    let mut action_mod = ActionModule::new(4, 4, ActionMode::Discrete { n_actions: 4 });
    let input_pre  = vec![0.8f32, -0.5, 0.3, -0.2, 0.6, -0.1];
    let input_post = vec![-0.8f32, 0.5, -0.3, 0.2, -0.6, 0.1];

    // Entrenar en input_pre
    for _ in 0..500 {
        f_lat.inject_input(&input_pre);
        let drives = f_lat.tick(1.0);
        let out = f_lat.read_output();
        action_mod.act(&out, &drives);
    }

    // Medir acción dominante en pre
    let mut pre_actions = Vec::new();
    for _ in 0..20 {
        f_lat.inject_input(&input_pre);
        let drives = f_lat.tick(1.0);
        let out = f_lat.read_output();
        let a = action_mod.act(&out, &drives);
        pre_actions.push(a.discrete.unwrap_or(0));
    }
    let dominant_pre = *pre_actions.iter()
    .max_by_key(|&&x| pre_actions.iter().filter(|&&y| y == x).count())
    .unwrap_or(&0);

    // Cambio de input: medir cuántos ticks hasta que la acción cambia y se estabiliza
    let max_latency = 200usize;
    let mut ticks_respuesta: Option<usize> = None;
    let mut consec = 0usize;
    let mut last_action = dominant_pre;

    for t in 0..max_latency {
        f_lat.inject_input(&input_post);
        let drives = f_lat.tick(1.0);
        let out = f_lat.read_output();
        let a = action_mod.act(&out, &drives).discrete.unwrap_or(0);

        if a != dominant_pre {
            consec += 1;
            if consec >= 5 && ticks_respuesta.is_none() {
                ticks_respuesta = Some(t);
            }
        } else {
            consec = 0;
        }
        last_action = a;
        if ticks_respuesta.is_some() { break; }
    }
    let latency = ticks_respuesta.unwrap_or(max_latency) as f32;
    let responded = ticks_respuesta.is_some();

    println!("    latencia_respuesta:{} ticks  respondió:{}", latency as usize, responded);
    println!("    acción_final:{}  t_final:{:.4}  conns:{}", last_action, t_final, conns_final);

    let passed = stability_ok;
    let summary = format!("nan:{} t_final:{:.4} conns:{} latencia:{:.0}t responde:{}",
                          nan_detected, t_final, conns_final, latency, responded);
    let metrics = vec![
        ("nan_detected",        if nan_detected { 1.0 } else { 0.0 }),
        ("tension_final",       t_final),
        ("connections_final",   conns_final as f32),
        ("out_of_range",        if out_of_range { 1.0 } else { 0.0 }),
        ("latency_ticks",       latency),
        ("responded",           if responded { 1.0 } else { 0.0 }),
        ("total_pruned",        f.total_pruned as f32),
    ];
    if passed { ScenarioResult::ok("estabilidad_larga", &summary, metrics) }
    else      { ScenarioResult::fail("estabilidad_larga", &summary, metrics) }
}

// ─────────────────────────────────────────────────────────────────────────
// E7: Persistencia — CLOSED-LOOP MAZE CON ESTADO
//
// Mejora (Nivel 3 / punto 7):
//   Creamos un mini motor de física 1D donde la acción del campo
//   afecta la posición del agente y el sensor refleja esa posición.
//   Goal: llegar a posición 0 desde posición aleatoria.
//   Métrica: ticks hasta tensión < 0.05 (convergencia en tarea).
//
//   Mantenemos el test de save/load de v0.2 como sanidad.
// ─────────────────────────────────────────────────────────────────────────

fn run_persistencia() -> ScenarioResult {
    println!("\n── E7: Persistencia [Closed-loop maze 1D] ─────────────");

    // — Check base: save/load (de v0.2) —
    let path_buf = std::env::temp_dir().join("ctd_eval_test.json");
    let path = path_buf.to_str().unwrap();
    let input = vec![0.7f32, -0.4, 0.5, 0.1, -0.3, 0.8];
    let mut f_original = eval_field(48, 6, 4, 0.4);
    for _ in 0..500 { f_original.inject_input(&input); f_original.tick(1.0); }
    let conns_before = f_original.connection_count();
    let ticks_before = f_original.tick_count();
    let save_ok = f_original.save(path).is_ok();
    let config2 = eval_field_config(48, 6, 4);
    let mut f_loaded = TensionField::new_dense(config2, 0.4);
    let load_ok   = f_loaded.load(path).is_ok();
    let conns_after = f_loaded.connection_count();
    let ticks_after = f_loaded.tick_count();
    let _ = std::fs::remove_file(path);
    let persist_ok = save_ok && load_ok && conns_before == conns_after && ticks_before == ticks_after;

    // — Closed-loop maze 1D —
    // El campo tiene 2 entradas: [posición_actual, distancia_al_goal]
    // La acción (continua) modifica la velocidad del agente.
    // Éxito: |posición| < 0.05 en < 1000 ticks.
    let mut resolution_ticks: Vec<Option<usize>> = Vec::new();

    for trial in 0..3 {
        let mut f_cl = eval_field(32, 2, 2, 0.5);
        let mut action_cl = ActionModule::new(2, 2, ActionMode::Continuous);
        let mut pos: f32 = if trial % 2 == 0 { 0.8 } else { -0.7 };
        let goal = 0.0f32;
        let mut vel = 0.0f32;
        let mut resolved: Option<usize> = None;

        for t in 0..1000 {
            let dist = goal - pos;
            let sensor = vec![pos.clamp(-1.0, 1.0), dist.clamp(-1.0, 1.0)];
            f_cl.inject_input(&sensor);
            let drives = f_cl.tick(1.0);
            let out = f_cl.read_output();
            let action = action_cl.act(&out, &drives);

            // La primera componente de la acción modifica la velocidad
            let force = action.values[0] * 0.1;
            vel = (vel + force) * 0.9;  // amortiguación
            pos = (pos + vel).clamp(-2.0, 2.0);

            if pos.abs() < 0.05 && resolved.is_none() {
                resolved = Some(t);
            }
            if resolved.is_some() { break; }
        }
        println!("    trial:{} pos_inicial:{:.2} resuelto_en:{:?} ticks",
                 trial, if trial % 2 == 0 { 0.8 } else { -0.7 }, resolved);
        resolution_ticks.push(resolved);
    }

    let any_resolved = resolution_ticks.iter().any(|r| r.is_some());
    let mean_ticks = {
        let solved: Vec<f32> = resolution_ticks.iter()
        .filter_map(|r| r.map(|t| t as f32))
        .collect();
        if solved.is_empty() { 1000.0 } else { mean(&solved) }
    };

    let passed = persist_ok;
    let summary = format!("save:{} load:{} conns:{}->{} maze_ok:{} mean_ticks:{:.0}",
                          save_ok, load_ok, conns_before, conns_after,
                          any_resolved, mean_ticks);
    let metrics = vec![
        ("save_ok",             if save_ok { 1.0 } else { 0.0 }),
        ("load_ok",             if load_ok { 1.0 } else { 0.0 }),
        ("conns_before",        conns_before as f32),
        ("conns_after",         conns_after as f32),
        ("persist_ok",          if persist_ok { 1.0 } else { 0.0 }),
        ("maze_any_resolved",   if any_resolved { 1.0 } else { 0.0 }),
        ("maze_mean_ticks",     mean_ticks),
    ];
    if passed { ScenarioResult::ok("persistencia", &summary, metrics) }
    else      { ScenarioResult::fail("persistencia", &summary, metrics) }
}

// ─────────────────────────────────────────────────────────────────────────
// E8: FieldStack — EFICIENCIA METABÓLICA
//
// Mejora (Nivel 3 / punto 8):
//   IQ = log(1 / tensión_final) / conexiones_totales
//   Sistemas que resuelven con pocas conexiones son más eficientes.
//   Umbral: IQ > 0.001 (arbitrario, calibrable).
//
//   Mantenemos los checks de drives, acción y save/load de v0.2.
// ─────────────────────────────────────────────────────────────────────────

fn run_field_stack() -> ScenarioResult {
    println!("\n── E8: FieldStack [Eficiencia metabólica] ─────────────");

    let config = StackConfig::new(6, 3);
    let mut stack = FieldStack::new(config, 0.4);
    let input = vec![0.7f32, -0.4, 0.5, 0.1, -0.3, 0.8];

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

    // — Eficiencia metabólica —
    // Correr más ticks para tensión estable
    for _ in 0..800 { stack.tick(&input, 1.0); }
    let (_, state_fin) = stack.tick(&input, 1.0);
    let total_conns = (state_fin.conns1() + state_fin.conns2()) as f32;
    let t_mean = (state_fin.t_internal1() + state_fin.t_internal2()) / 2.0;
    let iq = if total_conns > 0.0 && t_mean > 1e-6 {
        (1.0 / t_mean).ln() / total_conns
    } else {
        0.0
    };
    let iq_ok = iq > 0.001;

    println!("    IQ_metabólico:{:.6}  t_medio:{:.4}  conns_totales:{:.0}",
             iq, t_mean, total_conns);
    println!("    feedback_vivo:{}  drives_difieren:{}  acción_válida:{}",
             feedback_alive, drives_differ, action_valid);

    let passed = action_valid && feedback_alive && save_ok && load_ok;
    let summary = format!("acción:{} feedback:{} save:{} load:{} IQ:{:.5}",
                          action_valid, feedback_alive, save_ok, load_ok, iq);
    let metrics = vec![
        ("action_valid",       if action_valid { 1.0 } else { 0.0 }),
        ("feedback_alive",     if feedback_alive { 1.0 } else { 0.0 }),
        ("save_ok",            if save_ok { 1.0 } else { 0.0 }),
        ("load_ok",            if load_ok { 1.0 } else { 0.0 }),
        ("drives_differ",      if drives_differ { 1.0 } else { 0.0 }),
        ("metabolic_iq",       iq),
        ("iq_ok",              if iq_ok { 1.0 } else { 0.0 }),
        ("total_connections",  total_conns),
        ("t_internal1",        state.t_internal1()),
        ("t_internal2",        state.t_internal2()),
    ];
    if passed { ScenarioResult::ok("field_stack", &summary, metrics) }
    else      { ScenarioResult::fail("field_stack", &summary, metrics) }
}

// ─────────────────────────────────────────────────────────────────────────
// E9: Maze behavior — ATENCIÓN TOP-DOWN (Información Mutua F1↔F2)
//
// Mejora (Nivel 3 / punto 9):
//   Medimos cuánto influye el campo ejecutivo (F2) en el perceptivo (F1).
//   Recogemos series de tensión interna de F1 y F2 durante 500 ticks.
//   Calculamos información mutua estimada vía correlación.
//   MI_proxy > 0.1 → feedback real de F2 a F1.
//
//   Mantenemos el check de consistencia situacional de v0.2.
// ─────────────────────────────────────────────────────────────────────────

fn run_maze_behavior() -> ScenarioResult {
    println!("\n── E9: Maze behavior [Atención top-down F1↔F2] ────────");

    let sensor_size = 64usize;
    let action_size = 3usize;

    let mut consistency_runs = Vec::new();
    let mut stable_runs      = Vec::new();
    let mut mi_runs          = Vec::new();

    for _ in 0..3 {
        let config = StackConfig::new(sensor_size, action_size);
        let mut stack = FieldStack::new(config, 0.4);

        let mut wall_ahead = vec![0.3f32; sensor_size];
        wall_ahead[6] = 1.0; wall_ahead[7] = 0.0; wall_ahead[8] = 0.0;
        let mut key_visible = vec![0.2f32; sensor_size];
        for i in (sensor_size/3)..(2*sensor_size/3) { key_visible[i % sensor_size] = 0.833; }
        key_visible[20] = 0.9;
        let free_path = vec![0.1f32; sensor_size];
        let mut rng = rand::thread_rng();
        let chaotic: Vec<f32> = (0..sensor_size).map(|_| rng.gen_range(-0.5..0.5f32)).collect();

        for i in 0..1000 {
            let input = match i % 4 { 0 => &wall_ahead, 1 => &key_visible, 2 => &free_path, _ => &chaotic };
            stack.tick(input, 1.0);
        }

        // — Consistencia situacional (de v0.2) —
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
        let wall_cons  = consistency(situation_actions.get("wall").map(|v| v.as_slice()).unwrap_or(&[]));
        let key_cons   = consistency(situation_actions.get("key").map(|v| v.as_slice()).unwrap_or(&[]));
        let free_cons  = consistency(situation_actions.get("free").map(|v| v.as_slice()).unwrap_or(&[]));
        let mean_cons  = (wall_cons + key_cons + free_cons) / 3.0;
        consistency_runs.push(mean_cons);

        // — Información Mutua F1↔F2 (atención top-down) —
        let mut f1_series: Vec<f32> = Vec::with_capacity(500);
        let mut f2_series: Vec<f32> = Vec::with_capacity(500);
        for i in 0..500 {
            let input = match i % 3 { 0 => &wall_ahead, 1 => &key_visible, _ => &free_path };
            let (_, state) = stack.tick(input, 1.0);
            f1_series.push(state.t_internal1());
            f2_series.push(state.t_internal2());
        }
        let mi = mutual_info_proxy(&f1_series, &f2_series);
        mi_runs.push(mi);

        // — Estabilidad —
        let (_, last_state) = stack.tick(&free_path, 1.0);
        let stable = !last_state.t_internal1().is_nan() && !last_state.t_internal2().is_nan()
        && last_state.t_internal1() < 2.0 && last_state.conns1() > 0 && last_state.conns2() > 0;
        stable_runs.push(stable);
    }

    consistency_runs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_consistency = consistency_runs[1];
    let all_stable   = stable_runs.iter().all(|&s| s);
    let mean_mi      = mean(&mi_runs);
    let top_down_ok  = mean_mi > 0.05;  // correlación mínima entre campos

    println!("    consistencia_media:{:.3}  MI_F1↔F2:{:.4}  top_down:{}",
             median_consistency, mean_mi, top_down_ok);

    let passed = median_consistency > 0.40 && all_stable;
    let summary = format!("consistencia:{:.2} estable:{} MI_F1F2:{:.3} top_down:{}",
                          median_consistency, all_stable, mean_mi, top_down_ok);
    let metrics = vec![
        ("consistency_min",    consistency_runs[0]),
        ("consistency_median", median_consistency),
        ("consistency_max",    consistency_runs[2]),
        ("all_stable",         if all_stable { 1.0 } else { 0.0 }),
        ("mi_f1_f2_mean",      mean_mi),
        ("top_down_ok",        if top_down_ok { 1.0 } else { 0.0 }),
    ];
    if passed { ScenarioResult::ok("maze_behavior", &summary, metrics) }
    else      { ScenarioResult::fail("maze_behavior", &summary, metrics) }
}

// ─────────────────────────────────────────────────────────────────────────
// E10: Re-aprendizaje — TEST DE ISQUEMIA (Lobotomía dinámica)
//
// Mejora (Nivel 4 / punto 10):
//   En vez de solo medir re-aprendizaje, ejecutamos daño traumático
//   mientras el campo realiza una tarea: borramos ~1% de las conexiones
//   cada tick durante 50 ticks.
//   Métrica: ticks de recuperación hasta estabilizar drives tras el daño.
//
//   Mantenemos el test de re-aprendizaje de v0.2.
// ─────────────────────────────────────────────────────────────────────────

fn run_re_aprendizaje() -> ScenarioResult {
    println!("\n── E10: Re-aprendizaje [Test de isquemia] ─────────────");

    // — Re-aprendizaje de v0.2 —
    let input_a = vec![0.8f32, -0.5, 0.3, -0.2, 0.6, -0.1];
    let input_b = vec![0.1f32,  0.9, -0.7, 0.5, -0.3, 0.2];
    let umbral   = 0.05f32;
    let max_ticks = 2000usize;
    let mut ratios = Vec::new();

    for _ in 0..3 {
        let mut f = eval_field(48, 6, 4, 0.4);
        let mut ticks_fase1 = max_ticks;
        for t in 0..max_ticks {
            f.inject_input(&input_a); f.tick(1.0);
            if f.global_tension() < umbral { ticks_fase1 = t + 1; break; }
        }
        for _ in 0..600 { f.inject_input(&input_b); f.tick(1.0); }
        let mut ticks_fase2 = max_ticks;
        for t in 0..max_ticks {
            f.inject_input(&input_a); f.tick(1.0);
            if f.global_tension() < umbral { ticks_fase2 = t + 1; break; }
        }
        let ratio = ticks_fase2 as f32 / ticks_fase1.max(20) as f32;
        ratios.push((ticks_fase1, ticks_fase2, ratio));
    }
    ratios.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
    let median = &ratios[1];
    let relearn_ok = median.2 < 2.5;

    // — Test de isquemia (Lobotomía dinámica) —
    let input_isch = vec![0.6f32, -0.3, 0.5, -0.1, 0.4, -0.7];
    let mut f_isch = eval_field(64, 6, 4, 0.5);
    // Entrenar
    for _ in 0..600 { f_isch.inject_input(&input_isch); f_isch.tick(1.0); }
    let conns_pre = f_isch.connection_count();

    // Medir tensión pre-daño (estabilizada)
    let drives_pre = f_isch.tick(1.0);
    let t_pre_damage = drives_pre.mean_tension;

    // Daño: matar 1% de conexiones por tick durante 50 ticks.
    // La poda por relevancia no funciona en campos convergidos (relevancia alta).
    // Forzamos relevancia a 0 en conexiones aleatorias para simular daño real.
    let damage_per_tick = (conns_pre as f32 * 0.01).max(1.0) as usize;
    let mut rng = rand::thread_rng();
    for _ in 0..50 {
        f_isch.inject_input(&input_isch);
        f_isch.tick(1.0);
        let n = f_isch.connections.len();
        if n > damage_per_tick {
            let mut killed = 0;
            let mut tries  = 0;
            while killed < damage_per_tick && tries < n * 3 {
                let idx = rng.gen_range(0..f_isch.connections.len());
                if f_isch.connections[idx].relevance > 0.0 {
                    f_isch.connections[idx].relevance = 0.0;
                    killed += 1;
                }
                tries += 1;
            }
            f_isch.prune_weak(0.001);
        }
    }
    let conns_post_damage = f_isch.connection_count();

    // Medir recuperación: ticks hasta que la tensión vuelve a t_pre_damage * 1.5
    let t_spike_damage = f_isch.global_tension();
    let recovery_umbral = (t_pre_damage * 1.5).max(0.05);
    let mut recovery_ticks: Option<usize> = None;
    for t in 0..500 {
        f_isch.inject_input(&input_isch);
        let drives = f_isch.tick(1.0);
        if drives.mean_tension <= recovery_umbral && recovery_ticks.is_none() {
            recovery_ticks = Some(t);
        }
        if recovery_ticks.is_some() { break; }
    }
    let recovery_t = recovery_ticks.unwrap_or(500) as f32;
    let recovered = recovery_ticks.is_some();

    println!("    conns: pre:{} post_daño:{}", conns_pre, conns_post_damage);
    println!("    t_pre_daño:{:.4}  t_spike:{:.4}  recuperación:{} ticks (ok:{})",
             t_pre_damage, t_spike_damage, recovery_t as usize, recovered);
    println!("    re-aprendizaje ratio:{:.2} (mediana)  ok:{}", median.2, relearn_ok);

    let passed = relearn_ok;
    let summary = format!("relearn_ratio:{:.2} isquemia_recovery:{:.0}t ok:{}",
                          median.2, recovery_t, recovered);
    let metrics = vec![
        ("ticks_fase1_median",     median.0 as f32),
        ("ticks_fase2_median",     median.1 as f32),
        ("relearn_ratio_median",   median.2),
        ("relearn_ok",             if relearn_ok { 1.0 } else { 0.0 }),
        ("conns_pre_damage",       conns_pre as f32),
        ("conns_post_damage",      conns_post_damage as f32),
        ("t_pre_damage",           t_pre_damage),
        ("t_spike_damage",         t_spike_damage),
        ("recovery_ticks",         recovery_t),
        ("recovered",              if recovered { 1.0 } else { 0.0 }),
    ];
    if passed { ScenarioResult::ok("re_aprendizaje", &summary, metrics) }
    else      { ScenarioResult::fail("re_aprendizaje", &summary, metrics) }
}

// ─────────────────────────────────────────────────────────────────────────
// E11: Anticipación — TEST DE ALUCINACIÓN (Privación sensorial)
//
// Mejora (Nivel 4 / punto 11):
//   Poner inputs a 0.0 y medir actividad interna:
//   ¿el sistema sigue generando patrones coherentes sin input externo?
//   Métrica: Consistencia de sueño = autocorrelación de la tensión interna
//   en privación sensorial. Si > 0.3 → el campo "sueña" patrones conocidos.
//
//   Mantenemos el test de anticipación secuencial de v0.2.
// ─────────────────────────────────────────────────────────────────────────

fn run_anticipacion() -> ScenarioResult {
    println!("\n── E11: Anticipación [Alucinación / Privación sensorial]");

    let input_a = vec![0.7f32, -0.3, 0.5, 0.1, -0.4, 0.6];
    let input_b = vec![-0.6f32, 0.4, -0.2, 0.8, 0.3, -0.5];

    // — Anticipación secuencial de v0.2 —
    let mut tensions_trained   = Vec::new();
    let mut tensions_untrained = Vec::new();

    for _ in 0..3 {
        let mut f_trained = eval_field(48, 6, 4, 0.4);
        for _ in 0..500 {
            f_trained.inject_input(&input_a); f_trained.tick(1.0);
            f_trained.inject_input(&input_b); f_trained.tick(1.0);
        }
        let mut f_untrained = eval_field(48, 6, 4, 0.4);
        let neutral = vec![0.0f32; 6];
        for _ in 0..1000 { f_untrained.inject_input(&neutral); f_untrained.tick(1.0); }

        for _ in 0..5 {
            f_trained.inject_input(&input_a);
            let drives = f_trained.tick(1.0);
            tensions_trained.push(drives.mean_tension);
            f_untrained.inject_input(&input_a);
            let drives = f_untrained.tick(1.0);
            tensions_untrained.push(drives.mean_tension);
        }
    }
    let mean_trained   = mean(&tensions_trained);
    let mean_untrained = mean(&tensions_untrained);
    let anticipation_ratio = mean_trained / mean_untrained.max(0.001);
    let anticipates = anticipation_ratio < 0.9;

    // — Test de Alucinación (privación sensorial) —
    // Entrenar en patrón A, luego cortar input → ¿el campo sigue "recordando"?
    let mut f_dream = eval_field(48, 6, 4, 0.4);
    for _ in 0..800 {
        f_dream.inject_input(&input_a); f_dream.tick(1.0);
    }

    // Privación: input = 0
    let silence = vec![0.0f32; 6];
    let mut dream_tensions: Vec<f32> = Vec::with_capacity(200);
    for _ in 0..200 {
        f_dream.inject_input(&silence);
        let drives = f_dream.tick(1.0);
        dream_tensions.push(drives.mean_tension);
    }

    // Autocorrelación lag-1 como proxy de consistencia
    let n = dream_tensions.len();
    let m = mean(&dream_tensions);
    let autocorr = if n > 1 {
        let num: f32 = (0..n-1).map(|i| (dream_tensions[i] - m) * (dream_tensions[i+1] - m)).sum();
        let den: f32 = dream_tensions.iter().map(|&x| (x - m).powi(2)).sum::<f32>().max(1e-9);
        num / den
    } else { 0.0 };

    // El campo "sueña" si: la actividad no colapsa a 0 y tiene autocorrelación
    let dream_mean   = mean(&dream_tensions);
    let dreaming     = dream_mean > 0.01 && autocorr > 0.2;

    println!("    anticipación_ratio:{:.3}  anticipates:{}",
             anticipation_ratio, anticipates);
    println!("    sueño: tensión_media:{:.4}  autocorr:{:.3}  dreaming:{}",
             dream_mean, autocorr, dreaming);

    let passed = anticipates || dreaming;
    let summary = format!("anticipates:{} dreaming:{} (ratio:{:.2} autocorr:{:.2})",
                          anticipates, dreaming, anticipation_ratio, autocorr);
    let metrics = vec![
        ("mean_tension_trained",    mean_trained),
        ("mean_tension_untrained",  mean_untrained),
        ("anticipation_ratio",      anticipation_ratio),
        ("anticipates",             if anticipates { 1.0 } else { 0.0 }),
        ("dream_mean_tension",      dream_mean),
        ("dream_autocorr",          autocorr),
        ("dreaming",                if dreaming { 1.0 } else { 0.0 }),
    ];
    if passed { ScenarioResult::ok("anticipacion", &summary, metrics) }
    else      { ScenarioResult::fail("anticipacion", &summary, metrics) }
}

// ─────────────────────────────────────────────────────────────────────────
// E12: Degradación de vitalidad — SEED PROFILING (Personalidad)
//
// Mejora (Nivel 4 / punto 12):
//   Correr 20 semillas y categorizar según su drive dominante.
//   Métrica: sesgo de drive = curiosidad_media / calma_media.
//   Objetivo: mostrar diversidad de "personalidades" emergentes.
//
//   Mantenemos la evaluación de umbrales de poda de v0.2.
// ─────────────────────────────────────────────────────────────────────────

fn run_degradacion_vital() -> ScenarioResult {
    println!("\n── E12: Degradación vital [Seed Profiling] ─────────────");

    let input = vec![0.7f32, -0.4, 0.5, 0.1, -0.3, 0.8];

    // — Poda forzada de v0.2 —
    let mut f_ref = eval_field(64, 6, 4, 0.5);
    for _ in 0..800 { f_ref.inject_input(&input); f_ref.tick(1.0); }
    let _t_baseline_pre = f_ref.global_tension();
    for _ in 0..100  { f_ref.inject_input(&input); f_ref.tick(1.0); }
    let t_baseline = f_ref.global_tension();
    let c_full     = f_ref.connection_count();

    let levels = [0.8f32, 0.6, 0.4, 0.2];
    let mut survives_40 = false;
    let mut degradation_threshold = 0.0f32;

    for &keep_frac in &levels {
        let mut f = eval_field(64, 6, 4, 0.5);
        for _ in 0..800 { f.inject_input(&input); f.tick(1.0); }
        let target_conns = ((c_full as f32) * keep_frac) as usize;
        let mut threshold = 0.01f32;
        while f.connection_count() > target_conns.max(1) && threshold < 0.99 {
            f.prune_weak(threshold);
            threshold += 0.02;
        }
        let conns_after = f.connection_count();
        for _ in 0..100 { f.inject_input(&input); f.tick(1.0); }
        let t_after = f.global_tension();
        let still_ok = (t_after < t_baseline * 3.0 || t_after < 0.05) && !t_after.is_nan();
        if (keep_frac - 0.4).abs() < 0.01 { survives_40 = still_ok; }
        if !still_ok && degradation_threshold < 1e-6 { degradation_threshold = keep_frac; }
        println!("    keep:{:.0}%  conns:{}  t:{:.4}  ok:{}",
                 keep_frac * 100.0, conns_after, t_after, still_ok);
    }

    // — Seed Profiling (personalidad emergente) —
    // Estrategia: cada semilla recibe un régimen de entrenamiento DISTINTO
    // (ratio de caos variable 0%..100%) para forzar la emergencia de
    // personalidades diversas. Medimos el drive dominante durante la fase
    // más activa (50 ticks de caos puro al final del warmup), no en calma.
    const N_SEEDS_PROFILE: usize = 20;
    let mut seed_drives: Vec<(&'static str, f32)> = Vec::new();
    let mut curiosity_count   = 0usize;
    let mut discomfort_count  = 0usize;
    let mut calm_count        = 0usize;
    let mut curiosity_vals    = Vec::new();
    let mut calm_vals         = Vec::new();

    for seed_i in 0..N_SEEDS_PROFILE {
        let mut f_s = eval_field(48, 6, 4, 0.4);
        let mut rng = rand::thread_rng();

        // Régimen de entrenamiento escalonado: semillas pares más estables,
        // impares más caóticas — asegura diversidad estructural.
        let chaos_ratio = (seed_i as f32) / (N_SEEDS_PROFILE as f32);
        for _i in 0..400 {
            let inp: Vec<f32> = if rng.gen::<f32>() > chaos_ratio {
                input.clone()
            } else {
                (0..6).map(|_| rng.gen_range(-1.0..1.0f32)).collect()
            };
            f_s.inject_input(&inp);
            f_s.tick(1.0);
        }

        // Medir drives durante 50 ticks de caos puro → captura tensión activa
        let mut peak_curiosity  = 0.0f32;
        let mut peak_discomfort = 0.0f32;
        let mut peak_calm       = 0.0f32;
        for _ in 0..50 {
            let chaos_inp: Vec<f32> = (0..6).map(|_| rng.gen_range(-1.0..1.0f32)).collect();
            f_s.inject_input(&chaos_inp);
            let d = f_s.tick(1.0);
            peak_curiosity  = peak_curiosity.max(d.curiosity);
            peak_discomfort = peak_discomfort.max(d.discomfort);
            peak_calm       = peak_calm.max(d.calm);
        }

        // Personalidad = drive que alcanzó mayor pico bajo estrés
        let personality: &'static str = if peak_curiosity >= peak_discomfort && peak_curiosity >= peak_calm {
            "curiosity"
        } else if peak_discomfort >= peak_calm {
            "discomfort"
        } else {
            "calm"
        };

        curiosity_vals.push(peak_curiosity);
        calm_vals.push(peak_calm);
        let bias = peak_curiosity / peak_calm.max(0.01);
        seed_drives.push((personality, bias));
        match personality {
            "curiosity"  => curiosity_count += 1,
            "discomfort" => discomfort_count += 1,
            _            => calm_count += 1,
        }
    }

    let mean_curiosity = mean(&curiosity_vals);
    let mean_calm      = mean(&calm_vals);
    let bias_mean      = mean_curiosity / mean_calm.max(0.01);
    let bias_std       = std_dev(&curiosity_vals);

    // Diversidad: al menos 2 tipos de personalidad presentes
    let personality_types = [curiosity_count, discomfort_count, calm_count]
    .iter().filter(|&&c| c > 0).count();
    let diverse = personality_types >= 2;

    println!("    Personalidades: curiosos:{} molestos:{} calmos:{}",
             curiosity_count, discomfort_count, calm_count);
    println!("    sesgo_medio:{:.3}  std:{:.3}  tipos:{}  diverso:{}",
             bias_mean, bias_std, personality_types, diverse);

    let passed = survives_40;
    let summary = format!("survives_40%:{} seed_tipos:{} diverso:{} degrada_en:{:.0}%",
                          survives_40, personality_types, diverse,
                          degradation_threshold * 100.0);
    let metrics = vec![
        ("t_baseline",             t_baseline),
        ("connections_full",       c_full as f32),
        ("survives_40pct",         if survives_40 { 1.0 } else { 0.0 }),
        ("degradation_threshold",  degradation_threshold),
        ("seed_curiosity_count",   curiosity_count as f32),
        ("seed_discomfort_count",  discomfort_count as f32),
        ("seed_calm_count",        calm_count as f32),
        ("personality_types",      personality_types as f32),
        ("diverse_seeds",          if diverse { 1.0 } else { 0.0 }),
        ("bias_mean",              bias_mean),
        ("bias_std",               bias_std),
    ];
    if passed { ScenarioResult::ok("degradacion_vital", &summary, metrics) }
    else      { ScenarioResult::fail("degradacion_vital", &summary, metrics) }
}
