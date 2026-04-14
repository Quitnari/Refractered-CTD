// ═══════════════════════════════════════════════════════════════════════════
// CTD — Módulo de Acción
//
// Convierte tensiones de salida del campo en acciones para el entorno.
// Es deliberadamente simple — el campo ya hizo el trabajo cognitivo.
//
// PIPELINE
// ────────
// 1. Recibir tensiones de salida del campo (Vec<f32>)
// 2. Proyectar al espacio de acciones (si output_size ≠ action_size)
// 3. Modular por drives internos (exploración, intensidad)
// 4. Aplicar momentum suave (evitar cambios erráticos tick a tick)
// 5. Producir acción final
//
// MODOS
// ─────
// Continuo:  la acción es un vector de f32 — directo al entorno.
//            Útil para motores, brazos, velocidades.
//
// Discreto:  argmax sobre el vector proyectado — índice de acción.
//            Útil para Grid2D (N/S/E/W), CA (survive/born/reseed).
//            Con exploración: a veces elige aleatorio en lugar de argmax.
//
// MODULACIÓN POR DRIVES
// ──────────────────────
// curiosity  → aumenta probabilidad de exploración
// discomfort → amplifica la magnitud de la acción (actuar con más fuerza)
// calm       → reduce exploración, aplica más momentum (conservador)
// vitality   → si es baja, el módulo es más errático (campo frágil)
// ═══════════════════════════════════════════════════════════════════════════

use rand::Rng;
use crate::drives::DriveState;

/// Modo de salida del módulo de acción.
#[derive(Clone, Debug)]
pub enum ActionMode {
    /// Vector continuo de f32 en [-1, 1].
    Continuous,
    /// Índice discreto elegido por argmax (con exploración).
    Discrete { n_actions: usize },
}

/// Acción producida por el módulo.
#[derive(Clone, Debug)]
pub struct CtdAction {
    /// Vector de valores continuos — siempre presente.
    /// Para modo discreto, es el vector proyectado antes del argmax.
    pub values: Vec<f32>,
    /// Índice discreto — Some si modo Discrete, None si Continuous.
    pub discrete: Option<usize>,
}

/// Módulo de acción.
pub struct ActionModule {
    mode:        ActionMode,
    output_size: usize,  // tamaño del output del campo
    action_size: usize,  // tamaño esperado por el entorno

    /// Matriz de proyección output → acción (output_size × action_size).
    /// Si output_size == action_size, es la identidad.
    /// Se inicializa aleatoriamente y no aprende — es solo un mapeo fijo.
    /// El campo aprende a usar esta proyección implícitamente.
    projection: Vec<f32>,

    /// Momentum: promedio ponderado de la acción anterior.
    /// Suaviza cambios erráticos sin programar persistencia explícita.
    momentum:       Vec<f32>,
    momentum_decay: f32,

    /// Exploración base — modulada por drives en cada tick.
    base_exploration: f32,

    /// Huella del input anterior — usada para detectar cambios de contexto.
    /// Cuando el input cambia bruscamente, el momentum se amortigua
    /// para no contaminar la respuesta a la nueva situación.
    /// Tiene tamaño `action_size` — cubre el vector proyectado completo.
    last_fingerprint: Vec<f32>,
}

impl ActionModule {
    pub fn new(output_size: usize, action_size: usize, mode: ActionMode) -> Self {
        let mut rng = rand::thread_rng();

        // Proyección aleatoria normalizada por columna
        let proj_size = output_size * action_size;
        let mut projection: Vec<f32> = (0..proj_size)
        .map(|_| rng.gen_range(-1.0..1.0f32))
        .collect();

        // Normalizar cada columna (acción) para que la suma de cuadrados = 1
        for col in 0..action_size {
            let norm: f32 = (0..output_size)
            .map(|row| projection[row * action_size + col].powi(2))
            .sum::<f32>()
            .sqrt()
            .max(1e-6);
            for row in 0..output_size {
                projection[row * action_size + col] /= norm;
            }
        }

        Self {
            mode,
            output_size,
            action_size,
            projection,
            momentum:         vec![0.0; action_size],
            momentum_decay:   0.84, // 0.88 hacía que el agente tardara más en encontrar
            // la primera comida (first_eat 24→39 ticks). Con 0.84
            // mantiene más consistencia que 0.82 pero sin perder reflejos.
            base_exploration: 0.1,
            last_fingerprint: vec![0.0; action_size],
        }
    }

    /// Producir una acción dado el output del campo y los drives actuales.
    pub fn act(&mut self, field_output: &[f32], drives: &DriveState) -> CtdAction {
        // 1. Proyectar output del campo al espacio de acciones
        let mut projected = self.project(field_output);

        // 1b. Detección de cambio de contexto
        // Si el input proyectado difiere bruscamente del anterior, el momentum
        // de la situación previa contaminaría la respuesta actual.
        // Calculamos la distancia L2 entre el proyectado actual y la huella anterior.
        // Si supera el umbral, amortiguamos el momentum proporcionalmente al cambio.
        let context_shift: f32 = projected.iter().zip(self.last_fingerprint.iter())
        .map(|(&a, &b)| (a - b).powi(2))
        .sum::<f32>()
        .sqrt();
        // Umbral empírico: cambio > 0.5 en L2 sobre el vector normalizado
        // indica una situación distinta, no ruido tick a tick.
        let shift_threshold = 0.5f32;
        if context_shift > shift_threshold {
            // Amortiguar momentum proporcionalmente al tamaño del cambio.
            // Cambio = 0.5 → factor ~1.0 (sin efecto)
            // Cambio = 1.5 → factor ~0.33 (momentum casi eliminado)
            let decay = (shift_threshold / context_shift).clamp(0.1, 1.0);
            for v in self.momentum.iter_mut() {
                *v *= decay;
            }
        }
        // Actualizar huella con la proyección actual
        self.last_fingerprint.clone_from(&projected);

        // 2. Modular intensidad por malestar
        // Más malestar → acción más intensa (amplificar señal)
        // let intensity = 1.0 + drives.discomfort * 0.8;
        let intensity = 1.0 + drives.discomfort * 0.15; // 0.3 → 0.15: con discomfort≈0.6 permanente,
        // amplificar 30% distorsionaba la señal del campo.
        for v in projected.iter_mut() {
            *v = (*v * intensity).clamp(-1.0, 1.0);
        }

        // 3. Aplicar momentum (suavizar cambios tick a tick)
        // momentum_strength: más calma → más momentum (más conservador)
        // let momentum_strength = (self.momentum_decay + drives.calm * 0.15).clamp(0.0, 0.92);
        let momentum_strength = (self.momentum_decay + drives.calm * 0.15).clamp(0.0, 0.93); // techo 0.95→0.93
        for i in 0..self.action_size {
            self.momentum[i] = self.momentum[i] * momentum_strength
            + projected[i] * (1.0 - momentum_strength);
        }
        let smoothed = self.momentum.clone();

        // 4. Exploración modulada por curiosidad y calma
        let exploration = drives.exploration_modifier(self.base_exploration);

        // 5. Producir acción según modo
        match &self.mode {
            ActionMode::Continuous => CtdAction {
                values:   smoothed,
                discrete: None,
            },
            ActionMode::Discrete { n_actions } => {
                let n = *n_actions;
                let mut rng = rand::thread_rng();

                let chosen = if rng.gen::<f32>() < exploration {
                    // Exploración: muestrear ponderado por magnitud (no uniforme)
                    // Las acciones con mayor tensión tienen más probabilidad
                    // incluso en exploración — no es exploración ciega
                    let weights: Vec<f32> = smoothed.iter()
                    .take(n)
                    .map(|&v| (v.abs() + 0.1))
                    .collect();
                    let total: f32 = weights.iter().sum();
                    let mut r = rng.gen::<f32>() * total;
                    let mut picked = 0;
                    for (i, &w) in weights.iter().enumerate() {
                        r -= w;
                        if r <= 0.0 { picked = i; break; }
                        picked = i;
                    }
                    picked
                } else {
                    // Explotación: argmax sobre los primeros n valores
                    smoothed.iter().take(n).enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(i, _)| i)
                    .unwrap_or(0)
                };

                CtdAction {
                    values:   smoothed,
                    discrete: Some(chosen),
                }
            }
        }
    }

    /// Proyectar output_size → action_size via matriz de proyección.
    fn project(&self, field_output: &[f32]) -> Vec<f32> {
        if self.output_size == self.action_size {
            // Caso identidad — sin proyección
            return field_output.iter()
            .take(self.action_size)
            .cloned()
            .collect();
        }

        // Multiplicación: projected[col] = sum_row(field_output[row] * proj[row, col])
        let mut result = vec![0.0f32; self.action_size];
        for col in 0..self.action_size {
            for row in 0..self.output_size.min(field_output.len()) {
                result[col] += field_output[row] * self.projection[row * self.action_size + col];
            }
            result[col] = result[col].clamp(-1.0, 1.0);
        }
        result
    }

    /// Resetear momentum al inicio de episodio.
    pub fn reset_episode(&mut self) {
        self.momentum.iter_mut().for_each(|v| *v = 0.0);
        self.last_fingerprint.iter_mut().for_each(|v| *v = 0.0);
    }

    /// Ajustar exploración base.
    pub fn set_exploration(&mut self, e: f32) {
        self.base_exploration = e.clamp(0.0, 1.0);
    }
}

// ─────────────────────────────────────────────────────────────────────────
// TESTS
// ─────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn neutral_drives() -> DriveState {
        DriveState::from_field(0.1, 0.05, 200)
    }

    fn calm_drives() -> DriveState {
        DriveState::from_field(0.02, 0.001, 200)
    }

    fn curious_drives() -> DriveState {
        DriveState::from_field(0.8, 0.001, 200)
    }

    #[test]
    fn continuous_output_has_correct_size() {
        let mut m = ActionModule::new(4, 4, ActionMode::Continuous);
        let output = vec![0.5, -0.3, 0.8, -0.1];
        let action = m.act(&output, &neutral_drives());
        assert_eq!(action.values.len(), 4);
        assert!(action.discrete.is_none());
    }

    #[test]
    fn discrete_output_has_valid_index() {
        let mut m = ActionModule::new(4, 4, ActionMode::Discrete { n_actions: 4 });
        let output = vec![0.5, -0.3, 0.8, -0.1];
        let action = m.act(&output, &neutral_drives());
        assert!(action.discrete.is_some());
        assert!(action.discrete.unwrap() < 4);
    }

    #[test]
    fn values_stay_in_range() {
        let mut m = ActionModule::new(4, 4, ActionMode::Continuous);
        for _ in 0..100 {
            let output = vec![1.0, -1.0, 1.0, -1.0];
            let action = m.act(&output, &neutral_drives());
            for &v in &action.values {
                assert!(v >= -1.0 && v <= 1.0,
                        "value out of range: {}", v);
            }
        }
    }

    #[test]
    fn projection_works_with_different_sizes() {
        // output_size=8, action_size=4
        let mut m = ActionModule::new(8, 4, ActionMode::Continuous);
        let output = vec![0.5, -0.3, 0.8, -0.1, 0.2, -0.7, 0.4, 0.1];
        let action = m.act(&output, &neutral_drives());
        assert_eq!(action.values.len(), 4);
    }

    #[test]
    fn momentum_smooths_output() {
        let mut m = ActionModule::new(4, 4, ActionMode::Continuous);
        let output1 = vec![1.0,  1.0,  1.0,  1.0];
        let output2 = vec![-1.0, -1.0, -1.0, -1.0];

        let a1 = m.act(&output1, &neutral_drives());
        let a2 = m.act(&output2, &neutral_drives());

        // Después de un cambio brusco, el momentum debe suavizar
        // a2 no debería ser exactamente -1.0 por el momentum de a1
        let any_smoothed = a2.values.iter().any(|&v| v > -0.99);
        assert!(any_smoothed, "momentum should smooth abrupt changes");
    }

    #[test]
    fn reset_episode_clears_momentum() {
        let mut m = ActionModule::new(4, 4, ActionMode::Continuous);
        // Acumular momentum
        for _ in 0..20 {
            m.act(&vec![1.0, 1.0, 1.0, 1.0], &neutral_drives());
        }
        m.reset_episode();
        assert!(m.momentum.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn discomfort_amplifies_action() {
        let mut m1 = ActionModule::new(4, 4, ActionMode::Continuous);
        let mut m2 = ActionModule::new(4, 4, ActionMode::Continuous);
        let output = vec![0.3, 0.3, 0.3, 0.3];

        // m1 con drives neutros, m2 con alto malestar
        let a1 = m1.act(&output, &calm_drives());
        let mut disc_drives = DriveState::from_field(0.8, 0.6, 200);
        let a2 = m2.act(&output, &disc_drives);

        let mag1: f32 = a1.values.iter().map(|v| v.abs()).sum();
        let mag2: f32 = a2.values.iter().map(|v| v.abs()).sum();
        assert!(mag2 >= mag1,
                "discomfort should amplify action: neutral={:.3} discomfort={:.3}", mag1, mag2);
    }

    #[test]
    fn discrete_argmax_without_exploration() {
        // Con exploración = 0, debe elegir siempre el argmax
        let mut m = ActionModule::new(4, 4, ActionMode::Discrete { n_actions: 4 });
        m.set_exploration(0.0);
        // Output con tercer valor claramente dominante
        let output = vec![-0.5, -0.3, 0.9, -0.1];
        // Varias veces para confirmar — sin exploración es determinista
        let mut chosen_2 = 0;
        for _ in 0..20 {
            let a = m.act(&output, &calm_drives());
            if a.discrete == Some(2) { chosen_2 += 1; }
        }
        // Con momentum y proyección identidad, el índice 2 debería dominar
        assert!(chosen_2 > 10,
                "argmax should prefer index 2 most of the time: got {} / 20", chosen_2);
    }
}
