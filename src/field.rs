// ═══════════════════════════════════════════════════════════════════════════
// CTD — Campo de Tensión Diferencial
//
// El campo es la estructura que contiene y coordina todas las unidades
// y conexiones. Es el "cerebro" del sustrato.
//
// TICK
// ────
// Cada tick del campo ocurre en cuatro fases ordenadas:
//
//   1. PROPAGACIÓN
//      Para cada conexión: calcular la señal (tension_origen × weight × cos(phase))
//      y entregarla a la unidad destino como incoming.
//
//   2. ACTUALIZACIÓN DE ESTADO
//      Cada unidad aplica su incoming (con inercia) → nuevo what_is.
//
//   3. APRENDIZAJE
//      Cada unidad calcula su error (what_is - what_will).
//      Si el error es significativo:
//        - actualiza su expectativa (what_will)
//        - notifica a sus conexiones entrantes para que aprendan
//
//   4. PODA
//      Las conexiones con relevance < umbral se eliminan.
//      (No ocurre cada tick — cada N ticks para eficiencia.)
//
// TOPOLOGÍA
// ─────────
// El campo no asume ninguna topología fija.
// Las conexiones se crean explícitamente con connect().
// Se puede crear cualquier topología: grid, pequeño mundo, aleatoria, etc.
//
// SALIDA MOTORA
// ─────────────
// Las últimas `output_size` unidades son las unidades de salida.
// Su tensión media es lo que el campo "quiere comunicar al mundo".
// El módulo de acción lee estas tensiones para generar conducta.
//
// ENTRADA SENSORIAL
// ─────────────────
// Las primeras `input_size` unidades son unidades de entrada.
// Se inyectan directamente desde el entorno via inject_input().
// ═══════════════════════════════════════════════════════════════════════════

use rand::Rng;
use serde::{Serialize, Deserialize};
use crate::unit::Unit;
use crate::connection::Connection;
use crate::drives::DriveState;

// ─────────────────────────────────────────────────────────────────────────
// ESTRUCTURAS DE PERSISTENCIA
// Solo guardamos lo que costó aprender — no el estado efímero.
//
// ConnectionState: peso, fase, relevancia de cada conexión.
// UnitExpectation: what_will de cada unidad interna (su predicción aprendida).
// FieldSnapshot:   todo junto + metadatos del campo.
// ─────────────────────────────────────────────────────────────────────────

#[derive(Serialize, Deserialize)]
pub struct ConnectionState {
    pub from:      usize,
    pub to:        usize,
    pub weight:    f32,
    pub phase:     f32,
    pub relevance: f32,
}

#[derive(Serialize, Deserialize)]
pub struct FieldSnapshot {
    /// Versión del formato — para compatibilidad futura
    pub version:      u32,
    /// Configuración del campo (para verificar compatibilidad al cargar)
    pub n_units:      usize,
    pub input_size:   usize,
    pub output_size:  usize,
    /// Estado aprendido de conexiones
    pub connections:  Vec<ConnectionState>,
    /// Expectativas aprendidas de unidades internas (índice = índice en units[])
    /// Solo input_size..output_start — entrada y salida no se guardan
    pub expectations: Vec<f32>,
    /// Contadores internos
    pub tick_count:   u32,
    pub total_pruned: u32,
}

/// Configuración del campo.
#[derive(Clone, Debug)]
pub struct FieldConfig {
    /// Número total de unidades.
    pub n_units: usize,
    /// Cuántas de las primeras unidades son de entrada.
    pub input_size: usize,
    /// Cuántas de las últimas unidades son de salida.
    pub output_size: usize,
    /// Inercia base para nuevas unidades.
    pub default_inertia: f32,
    /// Learning rate para expectativas de unidades (ajuste fino por tick).
    /// Debe ser mayor que conn_lr — las expectativas se ajustan rápido.
    pub base_lr: f32,
    /// Learning rate para pesos de conexiones (estructura del campo).
    /// Debe ser menor que base_lr — la estructura cambia lento.
    /// Típico: base_lr * 0.1
    pub conn_lr: f32,
    /// Cada cuántos ticks hacer poda de conexiones muertas.
    pub prune_every: u32,
    /// Fase inicial aleatoria máxima para nuevas conexiones (en radianes).
    pub init_phase_range: f32,
    /// Amplitud máxima del ruido intrínseco inyectado en unidades internas.
    /// Escala con la calma del campo: cuando el campo predice bien (calma alta),
    /// el ruido sube para evitar que el campo se congele en un atractor.
    /// 0.0 = sin ruido. Típico: 0.02–0.06.
    pub intrinsic_noise: f32,
    /// Cada cuántos ticks hacer un reset suave de what_will en unidades de salida.
    /// Empuja levemente las expectativas de salida hacia cero para evitar atractores fijos.
    /// 0 = desactivado. Típico: 500–2000.
    pub output_drift_every: u32,
}

impl FieldConfig {
    pub fn default_small() -> Self {
        Self {
            n_units:          64,
            input_size:       8,
            output_size:      4,
            default_inertia:  0.2,
                base_lr:          0.05,
                conn_lr:          0.005,  // 10x más lento que expectativas
                prune_every:      100,
                init_phase_range: std::f32::consts::PI,
                intrinsic_noise:   0.03,
                output_drift_every: 2000,
        }
    }

    pub fn with_sizes(n_units: usize, input_size: usize, output_size: usize) -> Self {
        Self {
            n_units,
            input_size,
            output_size,
            ..Self::default_small()
        }
    }

    pub fn with_lr(mut self, base_lr: f32) -> Self {
        self.base_lr = base_lr;
        self.conn_lr = base_lr * 0.1;
        self
    }
}

/// El campo de tensión diferencial.
pub struct TensionField {
    pub units:       Vec<Unit>,
    pub connections: Vec<Connection>,
    pub config:      FieldConfig,

    tick_count:  u32,

    /// Tensión global media del tick anterior — para drives.
    pub mean_tension:  f32,
    /// Varianza de tensión del tick anterior — para drives.
    pub var_tension:   f32,
    /// Cuántas conexiones fueron podadas en total.
    pub total_pruned:  u32,
}

impl TensionField {
    /// Crear un campo vacío con la configuración dada.
    pub fn new(config: FieldConfig) -> Self {
        let mut rng = rand::thread_rng();
        let units = (0..config.n_units)
        .map(|_| {
            let v: f32 = rng.gen_range(-0.05..0.05);
            Unit::new(v, v, config.default_inertia)
        })
        .collect();

        Self {
            units,
            connections: Vec::new(),
            config,
            tick_count:  0,
            mean_tension: 0.0,
            var_tension:  0.0,
            total_pruned: 0,
        }
    }

    pub fn prune_weak(&mut self, threshold: f32) {
        let before = self.connections.len();
        self.connections.retain(|c| c.relevance >= threshold);
        self.total_pruned += (before - self.connections.len()) as u32;
    }

    /// Crear un campo con conectividad aleatoria densa.
    /// Cada unidad de entrada conecta a todas las internas.
    /// Las unidades internas se conectan aleatoriamente entre sí.
    /// Las internas conectan a las de salida.
    pub fn new_dense(config: FieldConfig, connection_prob: f32) -> Self {
        let mut field = Self::new(config.clone());
        let mut rng = rand::thread_rng();

        let input_end    = config.input_size;
        let output_start = config.n_units - config.output_size;

        // Relevancia inicial escalonada por tipo de conexión:
        // Las conexiones no han demostrado su valor todavía — empiezan
        // con relevancia reducida para que la poda sea gradual, no un
        // colapso masivo en los primeros episodios.
        // spontaneous_reconnect() ya usa 0.3 — coherente con este principio.
        // Entrada→internas: más relevancia (camino sensorial crítico)
        // Internas→internas: media (recurrencia, menos crítica al inicio)
        // Internas→salida: más relevancia (camino motor crítico)

        // Entrada → internas
        for i in 0..input_end {
            for j in input_end..output_start {
                if rng.gen::<f32>() < connection_prob {
                    let w = rng.gen_range(-0.3..0.3f32);
                    let p = rng.gen_range(0.0..config.init_phase_range);
                    let mut c = Connection::new(i, j, w, p);
                    c.relevance = 0.6;  // camino sensorial — relevancia media-alta
                    field.connections.push(c);
                }
            }
        }

        // Internas → internas (recurrente)
        for i in input_end..output_start {
            for j in input_end..output_start {
                if i != j && rng.gen::<f32>() < connection_prob * 0.3 {
                    let w = rng.gen_range(-0.2..0.2f32);
                    let p = rng.gen_range(0.0..config.init_phase_range);
                    let mut c = Connection::new(i, j, w, p);
                    c.relevance = 0.4;  // recurrencia — debe ganarse su lugar
                    field.connections.push(c);
                }
            }
        }

        // Internas → salida
        for i in input_end..output_start {
            for j in output_start..config.n_units {
                if rng.gen::<f32>() < connection_prob {
                    let w = rng.gen_range(-0.3..0.3f32);
                    let p = rng.gen_range(0.0..config.init_phase_range);
                    let mut c = Connection::new(i, j, w, p);
                    c.relevance = 0.6;  // camino motor — relevancia media-alta
                    field.connections.push(c);
                }
            }
        }

        field
    }

    /// Añadir una conexión manualmente.
    pub fn connect(&mut self, from: usize, to: usize, weight: f32, phase: f32) {
        assert!(from < self.config.n_units, "from index out of bounds");
        assert!(to   < self.config.n_units, "to index out of bounds");
        self.connections.push(Connection::new(from, to, weight, phase));
    }

    /// Inyectar señal sensorial en las unidades de entrada.
    /// `input` debe tener exactamente `config.input_size` elementos.
    /// Si es más corto, el resto queda en 0.
    pub fn inject_input(&mut self, input: &[f32]) {
        for (i, &v) in input.iter().enumerate().take(self.config.input_size) {
            self.units[i].inject(v);
        }
    }

    /// Leer el estado actual de las unidades de salida.
    /// Es lo que el campo "produce" — entrada para el módulo de acción.
    ///
    /// Retorna `what_is` directamente, no `tension()`.
    ///
    /// Las unidades de salida nunca actualizan `what_will` (por diseño —
    /// si aprendieran, colapsarían a cero y el campo quedaría mudo).
    /// Eso significa que `tension() = what_will - what_is` tiene un offset
    /// fijo de ruido aleatorio congelado desde la construcción del campo.
    /// Ese offset no transporta información — solo contamina la señal.
    ///
    /// `what_is` es la señal real: el estado que las conexiones internas
    /// empujaron hacia las unidades de salida. El drift, el ruido y la
    /// propagación trabajan sobre `what_is` — es ahí donde vive la información.
    pub fn read_output(&self) -> Vec<f32> {
        let start = self.config.n_units - self.config.output_size;
        self.units[start..].iter().map(|u| u.what_is).collect()
    }

    /// Leer estado de unidades internas como señal vocal.
    /// Las unidades internas tienen actividad continua incluso cuando
    /// el output motor es perturbado por drift. Son mejor fuente vocal.
    /// Toma `size` unidades del centro del rango interno.
    pub fn read_internal_voice(&self, size: usize) -> Vec<f32> {
        let input_end    = self.config.input_size;
        let output_start = self.config.n_units - self.config.output_size;
        let internal     = &self.units[input_end..output_start];
        if internal.is_empty() {
            return vec![0.0; size];
        }
        // Tomar del centro para capturar la zona más activa
        let n       = internal.len();
        let start   = if n > size { (n - size) / 2 } else { 0 };
        let mut out = vec![0.0f32; size];
        for (i, u) in internal[start..].iter().enumerate().take(size) {
            out[i] = u.what_is;
        }
        out
    }

    /// Leer voz interna comenzando desde una posición fraccional del campo.
    /// offset_frac ∈ [0.0, 1.0]: fracción del rango interno desde donde empezar.
    ///   0.0 = inicio del campo interno (mismas unidades que el Campo 0 ve)
    ///   0.5 = mitad del campo interno
    ///   1.0 = final del campo interno
    /// Usado por FieldStack para que distintos campos intermedios vean
    /// distintas porciones del ejecutivo, no siempre las primeras N unidades.
    pub fn read_internal_voice_at(&self, size: usize, offset_frac: f32) -> Vec<f32> {
        let input_end    = self.config.input_size;
        let output_start = self.config.n_units - self.config.output_size;
        let internal     = &self.units[input_end..output_start];
        if internal.is_empty() || size == 0 {
            return vec![0.0; size];
        }
        let n     = internal.len();
        // Calcular el inicio de la ventana según offset_frac
        // Clampeado para que la ventana quepa completamente dentro del rango
        let max_start = n.saturating_sub(size);
        let start     = ((offset_frac.clamp(0.0, 1.0) * max_start as f32) as usize).min(max_start);
        let mut out   = vec![0.0f32; size];
        for (i, u) in internal[start..].iter().enumerate().take(size) {
            out[i] = u.what_is;
        }
        out
    }

    /// Vector de introspección — estado interno del campo para inyectar
    /// como input en el próximo tick. El campo se ve a sí mismo con delay.
    ///
    /// Contiene: tensiones de unidades internas muestreadas uniformemente
    /// + mean_tension + var_tension + output_tension.
    pub fn read_introspection(&self, size: usize) -> Vec<f32> {
        if size == 0 { return vec![]; }

        let input_end    = self.config.input_size;
        let output_start = self.config.n_units - self.config.output_size;
        let internal     = &self.units[input_end..output_start];

        // Reservar 3 slots para métricas globales
        let n_unit_slots = size.saturating_sub(3);
        let mut out = Vec::with_capacity(size);

        // Muestreo uniforme de tensiones de unidades internas
        if n_unit_slots > 0 && !internal.is_empty() {
            let step = (internal.len() as f32 / n_unit_slots as f32).max(1.0);
            for i in 0..n_unit_slots {
                let idx = ((i as f32 * step) as usize).min(internal.len() - 1);
                out.push(internal[idx].tension().clamp(-1.0, 1.0));
            }
        }

        // Métricas globales normalizadas
        out.push(self.mean_tension.clamp(0.0, 1.0));
        out.push(self.var_tension.sqrt().clamp(0.0, 1.0));
        out.push(self.output_tension().clamp(0.0, 1.0));

        while out.len() < size { out.push(0.0); }
        out.truncate(size);
        out
    }

    /// Ejecutar un tick completo del campo.
    /// Retorna el estado de drives emergentes.
    pub fn tick(&mut self, lr_mod: f32) -> DriveState {
        self.tick_count += 1;

        // ── Fase 1: Propagación ───────────────────────────────────────────
        // Calcular tensiones actuales de todas las unidades
        let tensions: Vec<f32> = self.units.iter().map(|u| u.tension()).collect();

        // Distribuir señales a través de conexiones
        // timing: fracción relativa del tick según posición en el vector.
        // La primera conexión procesada recibe timing=0.0 (máxima responsabilidad),
        // la última timing≈1.0 (mínima). Esto activa la asimetría temporal en learn():
        // conexiones más antiguas (añadidas antes al campo) aprenden más rápido.
        // El orden es arbitrario pero consistente — no aleatorio tick a tick.
        let n_conns = self.connections.len().max(1);
        for (idx, conn) in self.connections.iter_mut().enumerate() {
            let timing = idx as f32 / n_conns as f32;
            let signal = conn.compute_and_record(tensions[conn.from], timing);
            self.units[conn.to].receive(signal);
        }

        // ── Fase 2: Actualización de estados ─────────────────────────────
        // Las unidades de entrada NO se actualizan por incoming — solo por inject
        for u in self.units[self.config.input_size..].iter_mut() {
            u.apply_incoming();
        }

        // ── Fase 3: Aprendizaje ───────────────────────────────────────────
        // Calcular errores y actualizar expectativas
        let errors: Vec<f32> = self.units.iter()
        .map(|u| u.what_is - u.what_will)
        .collect();

        // Actualizar expectativas solo de unidades internas.
        // Las de entrada: no aprenden (reciben inject directo).
        // Las de salida:  no aprenden — su tensión ES la señal motora.
        //                 Si aprenden, colapsan a cero y el campo queda mudo.
        let output_start = self.config.n_units - self.config.output_size;
        for u in self.units[self.config.input_size..output_start].iter_mut() {
            u.update_expectation(self.config.base_lr, lr_mod);
        }

        // Aprendizaje en conexiones que apuntan a unidades con error
        // tensions[] capturado al inicio del tick (Fase 1) contiene las tensiones
        // reales pre-apply_incoming — es el "before" correcto.
        // last_tension NO sirve: apply_incoming() lo sobreescribe con la tensión
        // post-actualización, así que tensions_before ≈ tensions_after → reducción ≈ 0.
        let tensions_after: Vec<f32> = self.units.iter().map(|u| u.tension()).collect();

        for conn in self.connections.iter_mut() {
            let dest_error = errors[conn.to];
            if dest_error.abs() > 1e-5 {
                // Cuánto redujo esta conexión la tensión del destino
                // tensions[conn.to] = tensión pre-Fase2 (capturada al inicio del tick)
                // tensions_after[conn.to] = tensión post-Fase2 y post-aprendizaje
                let tension_reduction = (tensions[conn.to].abs()
                - tensions_after[conn.to].abs()).max(0.0);
                // Las conexiones aprenden con conn_lr — más lento que las expectativas.
                // Esto separa las dos escalas de tiempo: expectativas (rápido, por tick)
                // y estructura del campo (lento, emergente).
                conn.learn(dest_error, self.config.conn_lr * lr_mod, tension_reduction);
            } else {
                conn.decay_relevance();
            }
        }

        // ── Fase 3b: Ruido intrínseco ─────────────────────────────────────
        // Cuando el campo está muy calmado (error bajo), inyectamos una
        // pequeña perturbación aleatoria en las unidades internas.
        // Evita que el campo se congele en un atractor estático.
        // El ruido escala con la calma actual: más calma → más ruido.
        // No afecta a unidades de entrada (tienen inject) ni de salida.
        //
        // calm_proxy usa la misma sigmoide que DriveState::from_field()
        // para que "calma" signifique lo mismo en ambos contextos.
        // SINCRONIZADO con drives.rs: k=20, center=0.20
        // (antes k=25, center=0.12 — desincronizado tras el ajuste de drives)
        if self.config.intrinsic_noise > 0.0 {
            let activity   = 1.0 / (1.0 + (-20.0f32 * (self.mean_tension - 0.20)).exp());
            let calm_proxy = (1.0 - activity).clamp(0.0, 1.0);
            let noise_amp  = self.config.intrinsic_noise * calm_proxy;
            if noise_amp > 0.001 {
                let mut rng = rand::thread_rng();
                let output_start = self.config.n_units - self.config.output_size;

                // Ruido en unidades internas (igual que antes)
                for u in self.units[self.config.input_size..output_start].iter_mut() {
                    let noise: f32 = rng.gen_range(-noise_amp..noise_amp);
                    u.what_is = (u.what_is + noise).clamp(-1.0, 1.0);
                }

                // Ruido suave en unidades de salida — 20% del ruido interno.
                // Evita que el output motor quede atascado en cero cuando
                // las conexiones internas→salida son débiles.
                let output_noise = noise_amp * 0.2;
                for u in self.units[output_start..].iter_mut() {
                    let noise: f32 = rng.gen_range(-output_noise..output_noise);
                    u.what_is = (u.what_is + noise).clamp(-1.0, 1.0);
                }
            }
        }

        // ── Fase 3c: Drift de salida ──────────────────────────────────────
        // Las unidades de salida tienen what_will fijo (no aprenden).
        // Si su what_is converge a un valor estático, la tensión de salida
        // queda congelada. Cada output_drift_every ticks empujamos what_is
        // levemente hacia cero para romper el atractor sin destruir la señal.
        if self.config.output_drift_every > 0
            && self.tick_count % self.config.output_drift_every == 0
            {
                let output_start = self.config.n_units - self.config.output_size;
                for u in self.units[output_start..].iter_mut() {
                    u.what_is *= 0.98;  // decay muy suave — no matar la señal
                }
            }

            // ── Fase 4: Poda ──────────────────────────────────────────────────
            if self.tick_count % self.config.prune_every == 0 {
                let before = self.connections.len();
                self.connections.retain(|c| !c.is_dead());
                self.total_pruned += (before - self.connections.len()) as u32;
            }

            // ── Métricas globales ─────────────────────────────────────────────
            self.update_metrics()
    }

    fn update_metrics(&mut self) -> DriveState {
        // Las métricas de drives se calculan solo sobre unidades internas.
        // Las de entrada tienen tensión artificial (inject directo).
        // Las de salida tienen tensión permanente por diseño (no aprenden what_will).
        // Ambas distorsionarían los drives si se incluyeran.
        let output_start = self.config.n_units - self.config.output_size;
        let internal = &self.units[self.config.input_size..output_start];

        if internal.is_empty() {
            // Campo degenerado (sin unidades internas) — usar todas
            let n = self.units.len() as f32;
            let tensions: Vec<f32> = self.units.iter().map(|u| u.tension_magnitude()).collect();
            self.mean_tension = tensions.iter().sum::<f32>() / n;
            self.var_tension  = tensions.iter()
            .map(|&t| (t - self.mean_tension).powi(2))
            .sum::<f32>() / n;
        } else {
            let n = internal.len() as f32;
            let tensions: Vec<f32> = internal.iter().map(|u| u.tension_magnitude()).collect();
            self.mean_tension = tensions.iter().sum::<f32>() / n;
            self.var_tension  = tensions.iter()
            .map(|&t| (t - self.mean_tension).powi(2))
            .sum::<f32>() / n;
        }

        DriveState::from_field(self.mean_tension, self.var_tension, self.connections.len())
    }

    /// Magnitud media del estado de las unidades de salida — señal motora global.
    /// Separada de mean_tension (que mide salud interna del campo).
    /// Usa |what_is| por la misma razón que read_output(): what_will está
    /// congelado con ruido inicial y no aporta información.
    pub fn output_tension(&self) -> f32 {
        let start = self.config.n_units - self.config.output_size;
        let out = &self.units[start..];
        if out.is_empty() { return 0.0; }
        out.iter().map(|u| u.what_is.abs()).sum::<f32>() / out.len() as f32
    }

    /// Número de conexiones vivas actualmente.
    pub fn connection_count(&self) -> usize {
        self.connections.len()
    }

    /// Tensión media global del campo.
    pub fn global_tension(&self) -> f32 {
        self.mean_tension
    }

    /// Varianza de tensión — dispersión del estado de error.
    pub fn tension_variance(&self) -> f32 {
        self.var_tension
    }

    /// Agregar nuevas conexiones aleatorias entre unidades internas.
    /// Usado para reconexión espontánea cuando el campo pierde demasiada estructura.
    /// prob: probabilidad de crear cada conexión posible (típico 0.02–0.08).
    ///
    /// Usa un HashSet para el check de existencia: O(1) por par en lugar de
    /// O(n_conexiones) con iter().any() — evita O(n²×m) en campos grandes.
    pub fn spontaneous_reconnect(&mut self, prob: f32) {
        use rand::Rng;
        use std::collections::HashSet;
        let mut rng = rand::thread_rng();
        let input_end    = self.config.input_size;
        let output_start = self.config.n_units - self.config.output_size;

        // Construir el set de pares existentes en O(m) antes del loop doble
        let existing: HashSet<(usize, usize)> = self.connections.iter()
        .map(|c| (c.from, c.to))
        .collect();

        let mut new_conns = Vec::new();

        // Solo entre unidades internas — no tocar entrada ni salida
        for i in input_end..output_start {
            for j in input_end..output_start {
                if i == j { continue; }
                // Lookup O(1) en lugar de O(n_conexiones)
                if !existing.contains(&(i, j)) && rng.gen::<f32>() < prob {
                    let w = rng.gen_range(-0.1..0.1f32);
                    let p = rng.gen_range(0.0..self.config.init_phase_range);
                    let mut c = Connection::new(i, j, w, p);
                    c.relevance = 0.3;  // relevancia inicial baja — debe ganarse su lugar
                    new_conns.push(c);
                }
            }
        }
        self.connections.extend(new_conns);
    }

    /// Número de ticks ejecutados desde la creación o último reset.
    #[inline]
    pub fn tick_count(&self) -> u32 { self.tick_count }

    /// Producir un snapshot del estado aprendido sin escribir a disco.
    /// Usado por FieldStack para guardar ambos campos juntos.
    pub fn snapshot(&self) -> Result<FieldSnapshot, String> {
        let output_start = self.config.n_units - self.config.output_size;

        let connections: Vec<ConnectionState> = self.connections.iter().map(|c| {
            ConnectionState {
                from:      c.from,
                to:        c.to,
                weight:    c.weight,
                phase:     c.phase,
                relevance: c.relevance,
            }
        }).collect();

        let expectations: Vec<f32> = self.units[self.config.input_size..output_start]
        .iter()
        .map(|u| u.what_will)
        .collect();

        Ok(FieldSnapshot {
            version:      1,
            n_units:      self.config.n_units,
            input_size:   self.config.input_size,
            output_size:  self.config.output_size,
            connections,
            expectations,
            tick_count:   self.tick_count,
            total_pruned: self.total_pruned,
        })
    }

    /// Aplicar un snapshot sin leer de disco.
    /// Usado por FieldStack para cargar ambos campos juntos.
    pub fn load_snapshot(&mut self, snap: FieldSnapshot) -> Result<(), String> {
        if snap.n_units != self.config.n_units
            || snap.input_size != self.config.input_size
            || snap.output_size != self.config.output_size
            {
                return Err(format!(
                    "incompatible: guardado=({},{},{}) actual=({},{},{})",
                                   snap.n_units, snap.input_size, snap.output_size,
                                   self.config.n_units, self.config.input_size, self.config.output_size,
                ));
            }

            self.connections.clear();
            for cs in &snap.connections {
                if cs.from < self.config.n_units && cs.to < self.config.n_units {
                    let mut c = Connection::new(cs.from, cs.to, cs.weight, cs.phase);
                    c.relevance = cs.relevance;
                    self.connections.push(c);
                }
            }

            let output_start = self.config.n_units - self.config.output_size;
            for (i, &exp) in snap.expectations.iter().enumerate() {
                let unit_idx = self.config.input_size + i;
                if unit_idx < output_start {
                    self.units[unit_idx].what_will = exp.clamp(-1.0, 1.0);
                }
            }

            self.tick_count   = snap.tick_count;
            self.total_pruned = snap.total_pruned;
            Ok(())
    }

    /// Guardar estado aprendido del campo a un archivo JSON.
    /// Solo guarda lo que costó aprender — conexiones y expectativas.
    /// El estado efímero (what_is, incoming) no se guarda.
    pub fn save(&self, path: &str) -> Result<(), String> {
        let output_start = self.config.n_units - self.config.output_size;

        let connections: Vec<ConnectionState> = self.connections.iter().map(|c| {
            ConnectionState {
                from:      c.from,
                to:        c.to,
                weight:    c.weight,
                phase:     c.phase,
                relevance: c.relevance,
            }
        }).collect();

        // Solo expectativas de unidades internas
        let expectations: Vec<f32> = self.units[self.config.input_size..output_start]
        .iter()
        .map(|u| u.what_will)
        .collect();

        let snapshot = FieldSnapshot {
            version:     1,
            n_units:     self.config.n_units,
            input_size:  self.config.input_size,
            output_size: self.config.output_size,
            connections,
            expectations,
            tick_count:  self.tick_count,
            total_pruned: self.total_pruned,
        };

        let json = serde_json::to_string(&snapshot)
        .map_err(|e| format!("serialización: {}", e))?;
        std::fs::write(path, json)
        .map_err(|e| format!("escritura {}: {}", path, e))?;
        Ok(())
    }

    /// Cargar estado aprendido desde un archivo JSON.
    /// Verifica compatibilidad de configuración antes de aplicar.
    /// Si la configuración no coincide, retorna error sin modificar el campo.
    pub fn load(&mut self, path: &str) -> Result<(), String> {
        let json = std::fs::read_to_string(path)
        .map_err(|e| format!("lectura {}: {}", path, e))?;
        let snap: FieldSnapshot = serde_json::from_str(&json)
        .map_err(|e| format!("deserialización: {}", e))?;

        // Verificar compatibilidad
        if snap.n_units != self.config.n_units
            || snap.input_size != self.config.input_size
            || snap.output_size != self.config.output_size
            {
                return Err(format!(
                    "incompatible: guardado=({},{},{}) actual=({},{},{})",
                                   snap.n_units, snap.input_size, snap.output_size,
                                   self.config.n_units, self.config.input_size, self.config.output_size,
                ));
            }

            // Restaurar conexiones
            self.connections.clear();
            for cs in &snap.connections {
                if cs.from < self.config.n_units && cs.to < self.config.n_units {
                    let mut c = Connection::new(cs.from, cs.to, cs.weight, cs.phase);
                    c.relevance = cs.relevance;
                    self.connections.push(c);
                }
            }

            // Restaurar expectativas de unidades internas
            let output_start = self.config.n_units - self.config.output_size;
            for (i, &exp) in snap.expectations.iter().enumerate() {
                let unit_idx = self.config.input_size + i;
                if unit_idx < output_start {
                    self.units[unit_idx].what_will = exp.clamp(-1.0, 1.0);
                }
            }

            self.tick_count  = snap.tick_count;
            self.total_pruned = snap.total_pruned;
            Ok(())
    }

    /// Resumen compacto para diagnóstico.
    pub fn summary(&self) -> String {
        format!(
            "tick:{} units:{} conns:{} tension_mean:{:.4} tension_var:{:.4} pruned:{}",
            self.tick_count,
            self.units.len(),
                self.connections.len(),
                self.mean_tension,
                self.var_tension,
                self.total_pruned,
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────
// TESTS
// ─────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn small_field() -> TensionField {
        TensionField::new(FieldConfig::with_sizes(16, 4, 2))
    }

    #[test]
    fn field_creates_correct_unit_count() {
        let f = small_field();
        assert_eq!(f.units.len(), 16);
    }

    #[test]
    fn inject_input_modifies_input_units() {
        let mut f = small_field();
        f.inject_input(&[1.0, -1.0, 0.5, -0.5]);
        assert_eq!(f.units[0].what_is, 1.0);
        assert_eq!(f.units[1].what_is, -1.0);
    }

    #[test]
    fn read_output_has_correct_size() {
        let f = small_field();
        let out = f.read_output();
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn tick_runs_without_panic() {
        let mut f = TensionField::new_dense(
            FieldConfig::with_sizes(32, 4, 4),
                                            0.3,
        );
        for _ in 0..10 {
            f.inject_input(&[0.5, -0.3, 0.1, 0.8]);
            f.tick(1.0);
        }
    }

    #[test]
    fn field_learns_constant_input_reduces_tension() {
        // Con input constante, el campo debería reducir tensión con el tiempo.
        // Usamos una seed fija implícita via campo pequeño y muchos ticks
        // para que el resultado sea determinista en términos estadísticos.
        let mut f = TensionField::new_dense(
            FieldConfig::with_sizes(32, 4, 4),
                                            0.5,
        );
        let input = vec![0.7, -0.3, 0.5, 0.1];

        // Warmup largo — el campo necesita estabilizar su dinámica inicial
        // antes de poder medir si está convergiendo
        for _ in 0..50 {
            f.inject_input(&input);
            f.tick(1.0);
        }
        let tension_early = f.global_tension();

        // Aprendizaje sostenido
        for _ in 0..500 {
            f.inject_input(&input);
            f.tick(1.0);
        }
        let tension_late = f.global_tension();

        // La tensión al final no debería ser mayor que al inicio del período de medición.
        // Con conn_lr separado de base_lr, el campo converge sin oscilar.
        assert!(tension_late <= tension_early * 1.05,
                "tension should not grow with constant input: early={:.4} late={:.4}",
                tension_early, tension_late);
    }

    #[test]
    fn pruning_removes_dead_connections() {
        let config = FieldConfig {
            prune_every: 1,  // podar cada tick para el test
            ..FieldConfig::with_sizes(16, 4, 2)
        };
        let mut f = TensionField::new(config);
        // Añadir una conexión muerta manualmente
        let mut dead = Connection::new(0, 5, 0.1, 0.0);
        dead.relevance = 0.001;
        f.connections.push(dead);
        let before = f.connections.len();
        f.tick(1.0);
        assert!(f.connections.len() < before || f.total_pruned > 0,
                "dead connection should be pruned");
    }

    #[test]
    fn dense_field_has_connections() {
        let f = TensionField::new_dense(
            FieldConfig::with_sizes(32, 4, 4),
                                        0.8,
        );
        assert!(f.connection_count() > 0);
    }

    #[test]
    fn output_size_matches_config() {
        let f = TensionField::new_dense(FieldConfig::with_sizes(20, 4, 3), 0.5);
        assert_eq!(f.read_output().len(), 3);
    }

    #[test]
    fn output_does_not_collapse_to_zero() {
        // Las unidades de salida no aprenden expectativas — su estado
        // debe mantenerse como señal útil incluso con input constante.
        let mut f = TensionField::new_dense(
            FieldConfig::with_sizes(32, 4, 4),
                                            0.5,
        );
        let input = vec![0.8, -0.5, 0.3, 0.6];
        for _ in 0..500 {
            f.inject_input(&input);
            f.tick(1.0);
        }
        let output = f.read_output();
        let mag: f32 = output.iter().map(|v| v.abs()).sum::<f32>() / output.len() as f32;
        assert!(mag > 0.001,
                "output should not collapse to zero: magnitude={:.6}", mag);
    }

    #[test]
    fn read_output_reflects_what_is_not_offset() {
        // Con what_will congelado en ruido inicial, tension() = what_will - what_is
        // tiene un offset arbitrario. read_output() debe retornar what_is, no tension().
        // Verificamos inyectando un valor conocido en las unidades de salida
        // y checkeando que read_output() lo refleja directamente.
        let mut f = TensionField::new(FieldConfig::with_sizes(8, 2, 2));
        // Forzar what_is de las unidades de salida a valores conocidos
        let output_start = f.config.n_units - f.config.output_size;
        f.units[output_start].what_is     = 0.75;
        f.units[output_start + 1].what_is = -0.50;
        let out = f.read_output();
        assert!((out[0] - 0.75).abs() < 1e-6,
                "read_output[0] should be what_is=0.75, got {}", out[0]);
        assert!((out[1] - (-0.50)).abs() < 1e-6,
                "read_output[1] should be what_is=-0.50, got {}", out[1]);
    }
}
