// ═══════════════════════════════════════════════════════════════════════════
// CTD — World v0.2 — Entorno homeostático 2D
//
// Corre: cargo run --bin world
//
// CAMBIOS v0.2
// ────────────
// - N_EPISODES: 20 → 50  (más tiempo para que el aprendizaje emerja)
// - MAX_TICKS:  2000 → 5000  (episodios más largos = más oportunidades)
// - prune_every: 200 → 100  (poda más frecuente, fix pruned:0)
// - rng unificado en World struct (fix warnings)
// - episode removido de run_episode (fix warning variable no usada)
// - Log detallado cada 500t (menos ruido que 200t)
// - Tendencia en cuartos además de mitades (curva más visible)
// - Snapshot del campo cada REPORT_EVERY episodios
//
// DEBUG: logging de diagnóstico en episodios que mueren sin comer
//
// DISEÑO
// ──────
// Grid 8×8. El agente tiene posición (x, y) y un drive de hambre.
// La hambre sube cada tick. Bajarla requiere encontrar y comer comida.
// El agente aprende solo — nadie le dice dónde está la comida ni
// que tiene que comerla. Lo descubre porque actuar sin comer mata.
//
// SENSORES → CTD (12 inputs)
// ───────────────────────────
//   [0]    x / 7.0              posición normalizada
//   [1]    y / 7.0
//   [2]    hambre               nivel [0, 1]
//   [3..5] dx, dy, stock        fuente 0 (dirección relativa + disponibilidad)
//   [6..8] dx, dy, stock        fuente 1
//   [9..11] dx, dy, stock       fuente 2
//
// ACCIONES (4 discretas)
// ──────────────────────
//   0 → Norte (y-1)
//   1 → Sur   (y+1)
//   2 → Oeste (x-1)
//   3 → Este  (x+1)
//
// DINÁMICA
// ────────
// hambre += HUNGER_RATE por tick
// si agente en celda con comida → hambre -= HUNGER_RELIEF, stock--
// si stock == 0 → fuente reaparece en posición aleatoria tras RESPAWN_TICKS
// si hambre >= 1.0 → episodio termina (muerte)
// si tick >= MAX_TICKS → episodio termina (supervivencia)
//
// EVALUACIÓN
// ──────────
// Métrica principal: hambre_media por episodio.
// Un agente aleatorio muere ~100t con hambre_media ~0.60.
// Un agente que aprende: hambre_media baja, ticks_vividos sube.
// La tendencia en cuartos muestra si el aprendizaje es continuo o se estanca.
// ═══════════════════════════════════════════════════════════════════════════

use ctd::{TensionField, ActionModule, ActionMode};
use ctd::field::FieldConfig;
use rand::Rng;

// ─────────────────────────────────────────────────────────────────────────
// CONSTANTES
// ─────────────────────────────────────────────────────────────────────────

const GRID_SIZE:     usize = 8;
const N_FOOD:        usize = 3;

/// Cuánto sube la hambre por tick.
const HUNGER_RATE:   f32   = 0.003;  // 0.008 → 0.003: igual que convergence, muere en 267t sin comer

/// Cuánto baja la hambre al comer.
const HUNGER_RELIEF: f32   = 0.45;

/// Stock inicial de cada fuente de comida.
const FOOD_STOCK:    u32   = 5;

/// Ticks hasta que una fuente agotada reaparece.
const RESPAWN_TICKS: u32   = 80;

/// Ticks máximos por episodio (supervivencia completa).
const MAX_TICKS:     u32   = 5_000;

/// Episodios a simular.
const N_EPISODES:    usize = 500;  // 50 → 500: el campo necesita tiempo para aprender

/// Cada cuántos episodios imprimir log detallado interno.
const REPORT_EVERY:  usize = 10;

/// Cada cuántos ticks imprimir estado dentro del log detallado.
const LOG_EVERY_T:   u32   = 500;

// ─────────────────────────────────────────────────────────────────────────
// MUNDO
// ─────────────────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
struct FoodSource {
    x:          usize,
    y:          usize,
    stock:      u32,
    /// Ticks hasta reaparición. 0 = disponible.
    respawn_in: u32,
}

impl FoodSource {
    fn new(x: usize, y: usize) -> Self {
        Self { x, y, stock: FOOD_STOCK, respawn_in: 0 }
    }

    fn is_available(&self) -> bool {
        self.respawn_in == 0 && self.stock > 0
    }

    fn tick_respawn(&mut self, rng: &mut impl Rng) {
        if self.respawn_in > 0 {
            self.respawn_in -= 1;
            if self.respawn_in == 0 {
                self.x     = rng.gen_range(0..GRID_SIZE);
                self.y     = rng.gen_range(0..GRID_SIZE);
                self.stock = FOOD_STOCK;
            }
        }
    }
}

struct World {
    foods: Vec<FoodSource>,
    rng:   rand::rngs::ThreadRng,
}

impl World {
    fn new() -> Self {
        let mut rng = rand::thread_rng();
        let foods = (0..N_FOOD).map(|_| {
            FoodSource::new(
                rng.gen_range(0..GRID_SIZE),
                            rng.gen_range(0..GRID_SIZE),
            )
        }).collect();
        Self { foods, rng }
    }

    fn reset(&mut self) {
        for f in self.foods.iter_mut() {
            f.x          = self.rng.gen_range(0..GRID_SIZE);
            f.y          = self.rng.gen_range(0..GRID_SIZE);
            f.stock      = FOOD_STOCK;
            f.respawn_in = 0;
        }
    }

    fn tick(&mut self) {
        // Borrow checker no permite usar self.rng dentro del iter_mut.
        // Creamos un rng local — el overhead es insignificante.
        let mut rng = rand::thread_rng();
        for f in self.foods.iter_mut() {
            f.tick_respawn(&mut rng);
        }
    }

    /// Intentar comer en (ax, ay). Retorna true si comió.
    fn try_eat(&mut self, ax: usize, ay: usize) -> bool {
        for f in self.foods.iter_mut() {
            if f.is_available() && f.x == ax && f.y == ay {
                f.stock -= 1;
                if f.stock == 0 {
                    f.respawn_in = RESPAWN_TICKS;
                }
                return true;
            }
        }
        false
    }

    /// Vector de sensores para CTD: 3 + N_FOOD×3 = 12 valores.
    fn sensors(&self, ax: usize, ay: usize, hunger: f32) -> Vec<f32> {
        let mut s = Vec::with_capacity(3 + N_FOOD * 3);

        s.push(ax as f32 / (GRID_SIZE - 1) as f32);
        s.push(ay as f32 / (GRID_SIZE - 1) as f32);
        s.push(hunger);

        for f in &self.foods {
            if f.respawn_in == 0 {
                // Fuente visible: dirección relativa normalizada y stock
                let dx = (f.x as f32 - ax as f32) / (GRID_SIZE - 1) as f32;
                let dy = (f.y as f32 - ay as f32) / (GRID_SIZE - 1) as f32;
                let st = f.stock as f32 / FOOD_STOCK as f32;
                s.push(dx);
                s.push(dy);
                s.push(st);
            } else {
                // Fuente en reaparición: señal neutra
                s.push(0.0);
                s.push(0.0);
                s.push(0.0);
            }
        }

        s
    }
}

// ─────────────────────────────────────────────────────────────────────────
// AGENTE
// ─────────────────────────────────────────────────────────────────────────

struct Agent {
    x:      usize,
    y:      usize,
    hunger: f32,
}

impl Agent {
    fn new() -> Self {
        let mut rng = rand::thread_rng();
        Self {
            x:      rng.gen_range(0..GRID_SIZE),
            y:      rng.gen_range(0..GRID_SIZE),
            hunger: 0.2,
        }
    }

    fn reset(&mut self) {
        let mut rng = rand::thread_rng();
        self.x      = rng.gen_range(0..GRID_SIZE);
        self.y      = rng.gen_range(0..GRID_SIZE);
        self.hunger = 0.2;
    }

    fn apply_action(&mut self, action: usize) {
        match action {
            0 => { if self.y > 0 { self.y -= 1; } }
            1 => { if self.y < GRID_SIZE - 1 { self.y += 1; } }
            2 => { if self.x > 0 { self.x -= 1; } }
            3 => { if self.x < GRID_SIZE - 1 { self.x += 1; } }
            _ => {}
        }
    }

    /// Tick: hambre sube. Retorna false si murió.
    fn tick(&mut self) -> bool {
        self.hunger = (self.hunger + HUNGER_RATE).clamp(0.0, 1.0);
        self.hunger < 1.0
    }

    fn eat(&mut self) {
        self.hunger = (self.hunger - HUNGER_RELIEF).max(0.0);
    }
}

// ─────────────────────────────────────────────────────────────────────────
// ESTADÍSTICAS
// ─────────────────────────────────────────────────────────────────────────

struct EpisodeStats {
    ticks:          u32,
    hunger_mean:    f32,
    hunger_peak:    f32,
    times_ate:      u32,
    survived:       bool,
    first_eat_tick: Option<u32>,
}

// ─────────────────────────────────────────────────────────────────────────
// LOOP DE EPISODIO
// ─────────────────────────────────────────────────────────────────────────

fn run_episode(
    field:      &mut TensionField,
    action_mod: &mut ActionModule,
    world:      &mut World,
    agent:      &mut Agent,
    verbose:    bool,
    ep_num:     usize,
) -> EpisodeStats {
    world.reset();
    agent.reset();
    action_mod.reset_episode();

    // ── DEBUG: estado inicial ─────────────────────────────────────────────
    let foods_init: Vec<(usize, usize)> = world.foods.iter()
    .map(|f| (f.x, f.y))
    .collect();
    println!("  [ep{}·init] agente:({},{})  fuentes:{:?}",
             ep_num, agent.x, agent.y, foods_init);

    let mut hunger_sum  = 0.0f32;
    let mut hunger_peak = 0.0f32;
    let mut times_ate   = 0u32;
    let mut first_eat   = None;
    let mut tick        = 0u32;

    // Buffer para debug de primeros ticks — solo se imprime si muere sin comer
    let mut early_log: Vec<String> = Vec::new();

    loop {
        tick += 1;

        // 1. Sensores → CTD
        let sensors = world.sensors(agent.x, agent.y, agent.hunger);
        field.inject_input(&sensors);
        let drives = field.tick(1.0);
        let output = field.read_output();
        let action = action_mod.act(&output, &drives);

        // 2. Acción
        let chosen = action.discrete.unwrap_or(0);

        // ── DEBUG: loggear primeros 20 ticks ─────────────────────────────
        if tick <= 20 {
            let dir = ["N", "S", "O", "E"][chosen];
            // distancia manhattan a la fuente más cercana disponible
            let dist_min = world.foods.iter()
            .filter(|f| f.is_available())
            .map(|f| {
                let dx = (f.x as i32 - agent.x as i32).abs();
                let dy = (f.y as i32 - agent.y as i32).abs();
                dx + dy
            })
            .min()
            .unwrap_or(99);
            early_log.push(format!(
                "    t:{:3} pos:({},{}) accion:{} dist_comida:{} hambre:{:.3} drives:[c:{:.2} m:{:.2} k:{:.2}]",
                                   tick, agent.x, agent.y, dir, dist_min, agent.hunger,
                                   drives.curiosity, drives.discomfort, drives.calm
            ));
        }

        agent.apply_action(chosen);

        // 3. Comer
        if world.try_eat(agent.x, agent.y) {
            agent.eat();
            times_ate += 1;
            if first_eat.is_none() {
                first_eat = Some(tick);
            }
        }

        // 4. Tick mundo y agente
        world.tick();
        let alive = agent.tick();

        // 5. Métricas
        hunger_sum  += agent.hunger;
        hunger_peak  = hunger_peak.max(agent.hunger);

        // 6. Log interno (verbose)
        if verbose && tick % LOG_EVERY_T == 0 {
            println!("    t:{:5}  pos:({},{})  hambre:{:.3}  comidas:{}  drive:{}  conns:{}",
                     tick, agent.x, agent.y, agent.hunger,
                     times_ate, drives.dominant(), field.connection_count());
        }

        // 7. Fin de episodio
        if !alive || tick >= MAX_TICKS {
            // ── DEBUG: si murió sin comer, volcar el log de primeros ticks ──
            if times_ate == 0 {
                println!("  [ep{}·muerte_sin_comer] ticks:{}", ep_num, tick);
                for line in &early_log {
                    println!("{}", line);
                }
            }

            return EpisodeStats {
                ticks: tick,
                hunger_mean: hunger_sum / tick as f32,
                hunger_peak,
                times_ate,
                survived: alive && tick >= MAX_TICKS,
                first_eat_tick: first_eat,
            };
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────
// RESUMEN POR CUARTOS
// ─────────────────────────────────────────────────────────────────────────

fn quarter_summary(label: &str, stats: &[EpisodeStats]) {
    if stats.is_empty() { return; }
    let n = stats.len() as f32;
    let hunger = stats.iter().map(|s| s.hunger_mean).sum::<f32>() / n;
    let ticks  = stats.iter().map(|s| s.ticks as f32).sum::<f32>() / n;
    let vivos  = stats.iter().filter(|s| s.survived).count();
    let eats   = stats.iter().map(|s| s.times_ate as f32).sum::<f32>() / n;
    println!("  {}  hambre:{:.4}  ticks:{:7.1}  vivos:{}/{}  comidas/ep:{:.1}",
             label, hunger, ticks, vivos, stats.len(), eats);
}

// ─────────────────────────────────────────────────────────────────────────
// MAIN
// ─────────────────────────────────────────────────────────────────────────

fn main() {
    println!("═══════════════════════════════════════════════════════");
    println!("  CTD World v0.2 — Entorno homeostático 2D (8×8)      ");
    println!("═══════════════════════════════════════════════════════\n");

    println!("Parámetros:");
    println!("  Grid:          {}×{}", GRID_SIZE, GRID_SIZE);
    println!("  Fuentes:       {} (stock={}, respawn={}t)", N_FOOD, FOOD_STOCK, RESPAWN_TICKS);
    println!("  Hunger rate:   {:.4}/tick  relief:{:.2}", HUNGER_RATE, HUNGER_RELIEF);
    println!("  Max ticks:     {}", MAX_TICKS);
    println!("  Episodios:     {}\n", N_EPISODES);

    // Campo: 12 inputs, 64 unidades internas, 4 outputs
    let config = FieldConfig {
        n_units:            150,  // 80 → 150: más capacidad, igual que convergence
        input_size:         12,
        output_size:        4,
        default_inertia:    0.15,
            base_lr:            0.06,
            conn_lr:            0.006,
            prune_every:        150,
            init_phase_range:   std::f32::consts::PI,
            intrinsic_noise:    0.03,
            output_drift_every: 1000,
    };

    let mut field      = TensionField::new_dense(config, 0.4);
    let mut action_mod = ActionModule::new(4, 4, ActionMode::Discrete { n_actions: 4 });
    let mut world      = World::new();
    let mut agent      = Agent::new();

    println!("{:<5} {:>8} {:>12} {:>10} {:>8} {:>10}  {}",
             "ep", "ticks", "hambre_med", "hambre_max", "comidas",
             "1er_comer", "resultado");
    println!("{}", "─".repeat(72));

    let mut all_stats: Vec<EpisodeStats> = Vec::new();

    for ep in 0..N_EPISODES {
        let verbose = (ep + 1) % REPORT_EVERY == 0;

        if verbose {
            println!("\n── Episodio {} (detalle) ──", ep + 1);
        }

        let stats = run_episode(
            &mut field,
            &mut action_mod,
            &mut world,
            &mut agent,
            verbose,
            ep + 1,
        );

        let resultado = if stats.survived { "VIVO ✓" } else { "muerto" };
        let primer    = stats.first_eat_tick
        .map(|t| format!("{}t", t))
        .unwrap_or_else(|| "nunca".to_string());

        println!("{:<5} {:>8} {:>12.4} {:>10.4} {:>8} {:>10}  {}",
                 ep + 1,
                 stats.ticks,
                 stats.hunger_mean,
                 stats.hunger_peak,
                 stats.times_ate,
                 primer,
                 resultado);

        // Snapshot del campo cada REPORT_EVERY episodios
        if (ep + 1) % REPORT_EVERY == 0 {
            println!("  [campo] conns:{}  tension:{:.4}  pruned:{}",
                     field.connection_count(),
                     field.global_tension(),
                     field.total_pruned);
        }

        all_stats.push(stats);
    }

    // ── Tendencia por cuartos ─────────────────────────────────────────────
    println!("\n{}", "═".repeat(72));
    println!("TENDENCIA POR CUARTOS");
    println!("{}", "─".repeat(72));

    let n = all_stats.len();
    let q = n / 4;
    if q > 0 {
        quarter_summary("Q1:", &all_stats[..q]);
        quarter_summary("Q2:", &all_stats[q..2*q]);
        quarter_summary("Q3:", &all_stats[2*q..3*q]);
        quarter_summary("Q4:", &all_stats[3*q..]);
    }

    // ── Tendencia mitad vs mitad ──────────────────────────────────────────
    println!("\n{}", "─".repeat(72));
    println!("PRIMERA MITAD vs SEGUNDA MITAD");
    println!("{}", "─".repeat(72));

    let first  = &all_stats[..n/2];
    let second = &all_stats[n/2..];

    let h1 = first.iter().map(|s| s.hunger_mean).sum::<f32>()  / first.len() as f32;
    let h2 = second.iter().map(|s| s.hunger_mean).sum::<f32>() / second.len() as f32;
    let t1 = first.iter().map(|s| s.ticks as f32).sum::<f32>() / first.len() as f32;
    let t2 = second.iter().map(|s| s.ticks as f32).sum::<f32>()/ second.len() as f32;
    let s1 = first.iter().filter(|s| s.survived).count();
    let s2 = second.iter().filter(|s| s.survived).count();
    let e1 = first.iter().map(|s| s.times_ate as f32).sum::<f32>()  / first.len() as f32;
    let e2 = second.iter().map(|s| s.times_ate as f32).sum::<f32>() / second.len() as f32;

    println!("                    primera mitad    segunda mitad    tendencia");
    println!("  hambre_media:     {:>10.4}       {:>10.4}       {}",
             h1, h2, if h2 < h1 { "↓ mejor" } else { "↑ peor" });
    println!("  ticks_vividos:    {:>10.1}       {:>10.1}       {}",
             t1, t2, if t2 > t1 { "↑ mejor" } else { "↓ peor" });
    println!("  supervivencias:   {:>10}       {:>10}       {}",
             s1, s2, if s2 >= s1 { "= o mejor" } else { "↓ peor" });
    println!("  comidas/ep:       {:>10.1}       {:>10.1}       {}",
             e1, e2, if e2 > e1 { "↑ mejor" } else { "↓ peor" });

    println!("\n  Veredicto: {}",
             if h2 < h1 && t2 > t1 { "[✓] CTD está aprendiendo a sobrevivir" }
             else if h2 < h1 || t2 > t1 { "[~] señales mixtas — más episodios recomendados" }
             else { "[✗] sin mejora observable — revisar configuración" });

    // ── Estado final del campo ────────────────────────────────────────────
    println!("\n── Estado final del campo ──");
    println!("{}", field.summary());
    println!("  pruned total: {}", field.total_pruned);
}
