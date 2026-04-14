// ═══════════════════════════════════════════════════════════════════════════
// CTD — Convergence Test  (BASE Run 4)
//
// Corre: cargo run --release --bin convergence
//
// Para iteraciones rápidas: cambiar N_EPISODES a 1000.
// Para run completo:        N_EPISODES = 5000.
//
// PARÁMETROS BASE (Run 4 — el mejor run hasta ahora)
// ────────────────────────────────────────────────────
// Estos parámetros produjeron: ticks_media=1261, comidas=35.9, surv=1.5%
// Cualquier experimento nuevo parte de aquí y cambia UNA sola variable.
// ═══════════════════════════════════════════════════════════════════════════

use ctd::{TensionField, ActionModule, ActionMode};
use ctd::field::FieldConfig;
use std::fs::File;
use std::io::Write;

use rand::Rng;

// ─────────────────────────────────────────────────────────────────────────
// CONSTANTES — BASE Run 4
// ─────────────────────────────────────────────────────────────────────────

const GRID_SIZE:     usize = 8;
const N_FOOD:        usize = 6;
const HUNGER_RATE:   f32   = 0.003;
const HUNGER_RELIEF: f32   = 0.55;
const FOOD_STOCK:    u32   = 5;
const RESPAWN_TICKS: u32   = 60;
const MAX_TICKS:     u32   = 5_000;

/// Cambiar a 1000 para iteraciones rápidas, 5000 para runs completos.
const N_EPISODES:    usize = 10000;
const OUTPUT_FILE:   &str  = "convergence_data.csv";

// ─────────────────────────────────────────────────────────────────────────
// MUNDO
// ─────────────────────────────────────────────────────────────────────────

#[derive(Clone)]
struct FoodSource { x: usize, y: usize, stock: u32, respawn_in: u32 }

impl FoodSource {
    fn new(x: usize, y: usize) -> Self {
        Self { x, y, stock: FOOD_STOCK, respawn_in: 0 }
    }
    fn is_available(&self) -> bool { self.respawn_in == 0 && self.stock > 0 }
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

struct World { foods: Vec<FoodSource>, rng: rand::rngs::ThreadRng }

impl World {
    fn new() -> Self {
        let mut rng = rand::thread_rng();
        let foods = (0..N_FOOD).map(|_| FoodSource::new(
            rng.gen_range(0..GRID_SIZE), rng.gen_range(0..GRID_SIZE)
        )).collect();
        Self { foods, rng }
    }
    fn reset(&mut self) {
        for f in self.foods.iter_mut() {
            f.x = self.rng.gen_range(0..GRID_SIZE);
            f.y = self.rng.gen_range(0..GRID_SIZE);
            f.stock = FOOD_STOCK;
            f.respawn_in = 0;
        }
    }
    fn tick(&mut self) {
        let mut rng = rand::thread_rng();
        for f in self.foods.iter_mut() { f.tick_respawn(&mut rng); }
    }
    fn try_eat(&mut self, ax: usize, ay: usize) -> bool {
        for f in self.foods.iter_mut() {
            if f.is_available() && f.x == ax && f.y == ay {
                f.stock -= 1;
                if f.stock == 0 { f.respawn_in = RESPAWN_TICKS; }
                return true;
            }
        }
        false
    }
    fn sensors(&self, ax: usize, ay: usize, hunger: f32) -> Vec<f32> {
        let mut s = Vec::with_capacity(3 + N_FOOD * 3);
        s.push(ax as f32 / (GRID_SIZE - 1) as f32);
        s.push(ay as f32 / (GRID_SIZE - 1) as f32);
        s.push(hunger);
        for f in &self.foods {
            if f.respawn_in == 0 {
                s.push((f.x as f32 - ax as f32) / (GRID_SIZE - 1) as f32);
                s.push((f.y as f32 - ay as f32) / (GRID_SIZE - 1) as f32);
                s.push(f.stock as f32 / FOOD_STOCK as f32);
            } else {
                s.push(0.0); s.push(0.0); s.push(0.0);
            }
        }
        s
    }
}

// ─────────────────────────────────────────────────────────────────────────
// AGENTE
// ─────────────────────────────────────────────────────────────────────────

struct Agent { x: usize, y: usize, hunger: f32 }

impl Agent {
    fn new() -> Self {
        let mut rng = rand::thread_rng();
        Self { x: rng.gen_range(0..GRID_SIZE), y: rng.gen_range(0..GRID_SIZE), hunger: 0.2 }
    }
    fn reset(&mut self) {
        let mut rng = rand::thread_rng();
        self.x = rng.gen_range(0..GRID_SIZE);
        self.y = rng.gen_range(0..GRID_SIZE);
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
    fn tick(&mut self) -> bool {
        self.hunger = (self.hunger + HUNGER_RATE).clamp(0.0, 1.0);
        self.hunger < 1.0
    }
    fn eat(&mut self) { self.hunger = (self.hunger - HUNGER_RELIEF).max(0.0); }
}

// ─────────────────────────────────────────────────────────────────────────
// EPISODIO
// ─────────────────────────────────────────────────────────────────────────

struct EpData {
    ticks:       u32,
    hunger_mean: f32,
    hunger_peak: f32,
    times_ate:   u32,
    survived:    bool,
    first_eat:   i32,
    connections: usize,
    tension:     f32,
    pruned:      u32,
}

fn run_episode(
    field:      &mut TensionField,
    action_mod: &mut ActionModule,
    world:      &mut World,
    agent:      &mut Agent,
) -> EpData {
    world.reset();
    agent.reset();
    action_mod.reset_episode();

    let mut hunger_sum  = 0.0f32;
    let mut hunger_peak = 0.0f32;
    let mut times_ate   = 0u32;
    let mut first_eat   = -1i32;
    let mut tick        = 0u32;
    let mut drives      = ctd::DriveState::default();

    loop {
        tick += 1;
        let sensors = world.sensors(agent.x, agent.y, agent.hunger);
        field.inject_input(&sensors);
        drives = field.tick(drives.lr_modifier());
        let output = field.read_output();
        let action = action_mod.act(&output, &drives);
        let chosen = action.discrete.unwrap_or(0);
        agent.apply_action(chosen);

        if world.try_eat(agent.x, agent.y) {
            agent.eat();
            times_ate += 1;
            if first_eat == -1 { first_eat = tick as i32; }
        }
        world.tick();
        let alive = agent.tick();

        // Reconexión espontánea: si el campo pierde demasiada estructura,
        // añadir conexiones nuevas para que pueda seguir aprendiendo.
        if tick % 500 == 0 && field.connection_count() < 1_400 {
            field.spontaneous_reconnect(0.04);
        }

        hunger_sum  += agent.hunger;
        hunger_peak  = hunger_peak.max(agent.hunger);

        if !alive || tick >= MAX_TICKS {
            return EpData {
                ticks: tick,
                hunger_mean: hunger_sum / tick as f32,
                hunger_peak,
                times_ate,
                survived: alive && tick >= MAX_TICKS,
                first_eat,
                connections: field.connection_count(),
                tension:     field.global_tension(),
                pruned:      field.total_pruned,
            };
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────
// CONFIG BASE Run 4 — no tocar, experimentar en ramas separadas
// ─────────────────────────────────────────────────────────────────────────

fn make_config() -> FieldConfig {
    FieldConfig {
        n_units:            150,
        input_size:         12,   // 3 posición/hambre + 4 fuentes × 3 valores
        output_size:        4,    // N S W E
        default_inertia:    0.18,
            base_lr:            0.05,
            conn_lr:            0.008,
            prune_every:        150,
            init_phase_range:   std::f32::consts::PI,
            intrinsic_noise:    0.005,
            output_drift_every: 800,
    }
}

// ─────────────────────────────────────────────────────────────────────────
// MAIN
// ─────────────────────────────────────────────────────────────────────────

fn main() {
    println!("CTD — Convergence Base (Run 4)");
    println!("N_EPISODES={N_EPISODES}  MAX_TICKS={MAX_TICKS}");
    println!("Corre con: cargo run --release --bin convergence\n");

    let mut field      = TensionField::new_dense(make_config(), 0.4);
    let mut action_mod = ActionModule::new(4, 4, ActionMode::Discrete { n_actions: 4 });
    let mut world      = World::new();
    let mut agent      = Agent::new();

    let mut file = File::create(OUTPUT_FILE).unwrap();
    writeln!(file, "ep,ticks,hunger_mean,hunger_peak,times_ate,survived,first_eat,connections,tension,pruned").unwrap();

    // Acumuladores para resumen por ventana
    let window = if N_EPISODES >= 1000 { 200 } else { 100 };
    let mut win_ticks  = 0f32;
    let mut win_ate    = 0f32;
    let mut win_surv   = 0u32;

    for ep in 0..N_EPISODES {
        let d = run_episode(&mut field, &mut action_mod, &mut world, &mut agent);

        writeln!(file, "{},{},{:.4},{:.4},{},{},{},{},{:.4},{}",
                 ep + 1, d.ticks, d.hunger_mean, d.hunger_peak,
                 d.times_ate, if d.survived { 1 } else { 0 },
                 d.first_eat, d.connections, d.tension, d.pruned).unwrap();

                 win_ticks += d.ticks as f32;
                 win_ate   += d.times_ate as f32;
                 win_surv  += d.survived as u32;

                 if (ep + 1) % window == 0 {
                     println!("ep {:>5}  ticks={:>6.0}  comidas={:>5.1}  surv={:.1}%  conns={}  pruned={}  tension={:.3}",
                              ep + 1,
                              win_ticks / window as f32,
                              win_ate   / window as f32,
                              100.0 * win_surv as f32 / window as f32,
                              d.connections,
                              d.pruned,
                              d.tension,
                     );
                     win_ticks = 0.0; win_ate = 0.0; win_surv = 0;
                 }
    }
    println!("\nGuardado en {OUTPUT_FILE}");
}
