// ═══════════════════════════════════════════════════════════════════════════
// CTD — FieldStack v3 — N campos en cadena
//
// v3: Generalización de 2 campos fijos a N campos configurables.
//
// ARQUITECTURA
// ─────────────
// Los campos forman una cadena de procesamiento:
//
//   sensores → [Campo 0] → bridge01 → [Campo 1] → bridge12 → ... → [Campo N-1] → acción
//                 ↑ feedback(N-1→0)        ↑ feedback(N-1→1)
//
// Campo 0:   perceptivo — recibe sensores del mundo + feedback del último campo
// Campo 1..N-2: intermedios — abstracción creciente
// Campo N-1: ejecutivo — produce la acción motora
//
// Cada par de campos adyacentes tiene su propio bridge (tamaño configurable).
// El feedback siempre viene del campo ejecutivo (N-1) hacia todos los anteriores,
// con tamaño decreciente: el Campo 0 recibe más contexto que el Campo 1.
//
// COMPATIBILIDAD CON v2
// ──────────────────────
// StackConfig::new(sensor_size, action_size) construye un stack de 2 campos
// con la misma topología que v2 — el código existente no cambia.
// Para más campos: StackConfig::with_depth(sensor_size, action_size, depth).
//
// ESCALABILIDAD
// ─────────────
// 2 campos: perceptivo + ejecutivo (comportamiento actual)
// 3 campos: + campo de planificación intermedio
// 4 campos: + campo de memoria de trabajo
// N campos: abstracción arbitraria — cada campo opera a su propia
//           escala temporal (inercia creciente de 0 a N-1)
// ═══════════════════════════════════════════════════════════════════════════

use serde::{Serialize, Deserialize};
use crate::field::{TensionField, FieldConfig, FieldSnapshot};
use crate::drives::DriveState;
use crate::action::{ActionModule, ActionMode, CtdAction};

// ─────────────────────────────────────────────────────────────────────────
// CONFIGURACIÓN DE UN CAMPO INDIVIDUAL
// ─────────────────────────────────────────────────────────────────────────

/// Rol de un campo en la cadena — determina sus parámetros por defecto.
#[derive(Clone, Debug)]
pub enum FieldRole {
    /// Campo 0: recibe sensores directamente. Más inercia, menos ruido.
    Perceptive,
    /// Campos intermedios: abstracción. Parámetros balanceados.
    Abstract,
    /// Campo N-1: produce acción. Menos inercia, más ruido, más drift.
    Executive,
}

/// Configuración de un campo dentro del stack.
#[derive(Clone, Debug)]
pub struct LayerConfig {
    /// Unidades internas (sin contar input/output).
    pub internal_units: usize,
    /// Bridge hacia el siguiente campo (output_size de este campo).
    /// El último campo usa action_size como output.
    pub bridge_size: usize,
    /// Feedback recibido del campo ejecutivo (input extra).
    pub feedback_size: usize,
    /// Parámetros de aprendizaje — None usa defaults según el rol.
    pub base_lr:            Option<f32>,
    pub conn_lr:            Option<f32>,
    pub inertia:            Option<f32>,
    pub intrinsic_noise:    Option<f32>,
    pub prune_every:        Option<u32>,
    pub output_drift_every: Option<u32>,
}

impl LayerConfig {
    /// Configuración por defecto según el rol del campo.
    pub fn default_for(role: &FieldRole, bridge: usize, feedback: usize) -> Self {
        let (internal, base_lr, conn_lr, inertia, noise, prune, drift) = match role {
            FieldRole::Perceptive => (128, 0.04, 0.004, 0.20, 0.03, 200, 2000),
            FieldRole::Abstract   => (96,  0.035, 0.0035, 0.18, 0.04, 300, 1500),
            FieldRole::Executive  => (96,  0.03, 0.003, 0.15, 0.05, 500, 1000),
        };
        Self {
            internal_units:     internal,
            bridge_size:        bridge,
            feedback_size:      feedback,
            base_lr:            Some(base_lr),
            conn_lr:            Some(conn_lr),
            inertia:            Some(inertia),
            intrinsic_noise:    Some(noise),
            prune_every:        Some(prune),
            output_drift_every: Some(drift),
        }
    }

    /// Convertir a FieldConfig dado el tamaño de input del entorno.
    fn to_field_config(&self, input_from_prev: usize) -> FieldConfig {
        let input_size  = input_from_prev + self.feedback_size;
        let output_size = self.bridge_size;
        let n_units     = input_size + self.internal_units + output_size;

        FieldConfig {
            n_units,
            input_size,
            output_size,
            default_inertia:    self.inertia.unwrap_or(0.2),
                base_lr:            self.base_lr.unwrap_or(0.04),
                conn_lr:            self.conn_lr.unwrap_or(0.004),
                prune_every:        self.prune_every.unwrap_or(200),
                init_phase_range:   std::f32::consts::PI,
                intrinsic_noise:    self.intrinsic_noise.unwrap_or(0.03),
                output_drift_every: self.output_drift_every.unwrap_or(2000),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────
// STACK CONFIG
// ─────────────────────────────────────────────────────────────────────────

/// Configuración completa del stack de N campos.
#[derive(Clone, Debug)]
pub struct StackConfig {
    pub layers:      Vec<LayerConfig>,
    pub sensor_size: usize,
    pub action_size: usize,
}

impl StackConfig {
    /// Stack de 2 campos — compatible con v2.
    /// Misma topología que el FieldStack anterior.
    pub fn new(sensor_size: usize, action_size: usize) -> Self {
        Self::with_depth(sensor_size, action_size, 2)
    }

    /// Stack de N campos con profundidad configurable.
    ///
    /// depth=2: perceptivo + ejecutivo (igual que v2)
    /// depth=3: perceptivo + abstracto + ejecutivo
    /// depth=4: perceptivo + abstracto + abstracto + ejecutivo
    pub fn with_depth(sensor_size: usize, action_size: usize, depth: usize) -> Self {
        assert!(depth >= 2, "El stack necesita al menos 2 campos");

        let bridge = 32usize;
        // El feedback del campo ejecutivo decrece con la distancia:
        // Campo 0 recibe más contexto que campos intermedios
        let feedback_executive = 16usize;

        let mut layers = Vec::with_capacity(depth);

        for i in 0..depth {
            let role = if i == 0 { FieldRole::Perceptive }
            else if i == depth - 1 { FieldRole::Executive }
            else { FieldRole::Abstract };

            // El último campo usa action_size como bridge (output = acción)
            let bridge_out = if i == depth - 1 { action_size } else { bridge };

            // Feedback: solo el Campo 0 recibe feedback directo del ejecutivo.
            // Campos intermedios reciben feedback proporcional a su distancia.
            let feedback = if i == 0 {
                feedback_executive
            } else if i < depth - 1 {
                // Intermedios: feedback reducido, proporcional a distancia del ejecutivo
                (feedback_executive / 2).max(8)
            } else {
                0  // El ejecutivo no recibe feedback de sí mismo
            };

            layers.push(LayerConfig::default_for(&role, bridge_out, feedback));
        }

        Self { layers, sensor_size, action_size }
    }

    /// Número de campos en el stack.
    pub fn depth(&self) -> usize { self.layers.len() }
}

// ─────────────────────────────────────────────────────────────────────────
// STACK STATE — diagnóstico por campo
// ─────────────────────────────────────────────────────────────────────────

/// Estado diagnóstico de un campo individual.
#[derive(Clone, Debug)]
pub struct LayerState {
    pub drives:          DriveState,
    pub t_internal:      f32,
    pub t_output:        f32,
    pub connections:     usize,
    pub feedback_energy: f32,  // energía del feedback recibido en este tick
}

/// Estado diagnóstico completo del stack.
#[derive(Clone, Debug)]
pub struct StackState {
    pub layers: Vec<LayerState>,
}

impl StackState {
    /// Drives del campo perceptivo (Campo 0).
    pub fn drives1(&self) -> &DriveState {
        &self.layers[0].drives
    }

    /// Drives del campo ejecutivo (último campo).
    pub fn drives2(&self) -> &DriveState {
        &self.layers[self.layers.len() - 1].drives
    }

    /// Tensión interna del campo perceptivo.
    pub fn t_internal1(&self) -> f32 { self.layers[0].t_internal }

    /// Tensión interna del campo ejecutivo.
    pub fn t_internal2(&self) -> f32 { self.layers[self.layers.len() - 1].t_internal }

    /// Tensión de output del campo perceptivo.
    pub fn t_output1(&self) -> f32 { self.layers[0].t_output }

    /// Tensión de output del campo ejecutivo.
    pub fn t_output2(&self) -> f32 { self.layers[self.layers.len() - 1].t_output }

    /// Conexiones del campo perceptivo.
    pub fn conns1(&self) -> usize { self.layers[0].connections }

    /// Conexiones del campo ejecutivo.
    pub fn conns2(&self) -> usize { self.layers[self.layers.len() - 1].connections }

    /// Energía del canal de feedback del campo ejecutivo al perceptivo.
    pub fn feedback_energy(&self) -> f32 { self.layers[0].feedback_energy }
}

// ─────────────────────────────────────────────────────────────────────────
// SNAPSHOT — persistencia
// ─────────────────────────────────────────────────────────────────────────

#[derive(Serialize, Deserialize)]
pub struct StackSnapshot {
    pub version: u32,
    pub depth:   usize,
    pub fields:  Vec<FieldSnapshot>,
}

// ─────────────────────────────────────────────────────────────────────────
// FIELD STACK — N campos en cadena
// ─────────────────────────────────────────────────────────────────────────

pub struct FieldStack {
    pub fields:     Vec<TensionField>,
    pub action_mod: ActionModule,
    config:         StackConfig,

    /// Feedback del tick anterior de cada campo hacia los anteriores.
    /// last_feedbacks[i] = feedback que el ejecutivo envió al campo i.
    /// None en el primer tick.
    last_feedbacks: Vec<Option<Vec<f32>>>,
}

impl FieldStack {
    pub fn new(config: StackConfig, connection_prob: f32) -> Self {
        let depth       = config.depth();
        let action_size = config.action_size;

        // Construir cada campo con su FieldConfig correcto
        // El input del campo i = bridge del campo i-1 (o sensor_size para i=0)
        // más el feedback que recibe
        let mut fields = Vec::with_capacity(depth);
        let mut prev_bridge = config.sensor_size;

        for layer in &config.layers {
            let fc = layer.to_field_config(prev_bridge);
            fields.push(TensionField::new_dense(fc, connection_prob));
            prev_bridge = layer.bridge_size;
        }

        let action_mod = ActionModule::new(
            action_size,
            action_size,
            ActionMode::Discrete { n_actions: action_size },
        );

        let last_feedbacks = vec![None; depth];

        Self { fields, action_mod, config, last_feedbacks }
    }

    /// Tick completo del stack de N campos.
    pub fn tick(&mut self, sensors: &[f32], lr_mod: f32) -> (CtdAction, StackState) {
        let depth = self.config.depth();
        let mut layer_states = Vec::with_capacity(depth);

        // ── Procesar cada campo en orden ──────────────────────────────────
        // El output de cada campo se convierte en input del siguiente.
        let mut prev_output: Vec<f32> = sensors.to_vec();

        for i in 0..depth {
            let layer    = &self.config.layers[i];
            let feedback = &self.last_feedbacks[i];

            // Construir input: output del campo anterior + feedback del ejecutivo
            let mut field_input = Vec::with_capacity(prev_output.len() + layer.feedback_size);
            field_input.extend_from_slice(&prev_output);

            match feedback {
                Some(fb) => {
                    for &v in fb.iter().take(layer.feedback_size) {
                        field_input.push(v.clamp(-1.0, 1.0));
                    }
                    // Rellenar si el feedback es más corto
                    while field_input.len() < prev_output.len() + layer.feedback_size {
                        field_input.push(0.0);
                    }
                }
                None => {
                    field_input.extend(vec![0.0f32; layer.feedback_size]);
                }
            }

            self.fields[i].inject_input(&field_input);

            // lr decae ligeramente con la profundidad:
            // los campos más profundos aprenden más despacio
            let field_lr = lr_mod * (1.0 - i as f32 * 0.05).max(0.6);
            let drives   = self.fields[i].tick(field_lr);

            // Reconexión espontánea si el campo pierde estructura
            if drives.vitality < 0.15 {
                self.fields[i].spontaneous_reconnect(0.05);
            }

            let feedback_energy = match feedback {
                Some(fb) => {
                    let sq: f32 = fb.iter().map(|x| x * x).sum();
                    (sq / fb.len().max(1) as f32).sqrt()
                }
                None => 0.0,
            };

            layer_states.push(LayerState {
                drives,
                t_internal:      self.fields[i].global_tension(),
                              t_output:        self.fields[i].output_tension(),
                              connections:     self.fields[i].connection_count(),
                              feedback_energy,
            });

            // El output de este campo es el input del siguiente
            // Para campos intermedios: bridge (output_state)
            // Para el ejecutivo: tensiones de salida (para el ActionModule)
            prev_output = self.fields[i].read_output_state();
        }

        // ── Enriquecer bridge entre campos con drives del campo anterior ──
        // El campo ejecutivo recibe el bridge + drives del campo anterior
        // (igual que en v2 donde el bridge incluía los 7 valores de drives)
        // Esto ya está implícito en el output_state — los drives
        // se transmiten vía la dinámica de tensión, no de forma explícita.
        // El bridge explícito de drives solo se necesita para el último salto.

        // ── Acción desde el campo ejecutivo ──────────────────────────────
        let exec_idx    = depth - 1;
        let output_exec = self.fields[exec_idx].read_output();
        let drives_exec = layer_states[exec_idx].drives.clone();
        let action      = self.action_mod.act(&output_exec, &drives_exec);

        // ── Feedback del ejecutivo hacia todos los campos anteriores ──────
        // El campo ejecutivo transmite su estado interno a los anteriores
        // con tamaño decreciente según la distancia.
        let exec_voice = self.fields[exec_idx].read_internal_voice(
            self.config.layers[0].feedback_size  // tamaño máximo
        );

        for i in 0..exec_idx {
            let fb_size = self.config.layers[i].feedback_size;
            if fb_size == 0 { continue; }

            // Feedback proporcional: campos más cercanos al ejecutivo
            // reciben una porción más pequeña (menos contexto)
            let take = fb_size.min(exec_voice.len());
            let fb: Vec<f32> = exec_voice[..take].to_vec();
            self.last_feedbacks[i] = Some(fb);
        }

        let state = StackState { layers: layer_states };
        (action, state)
    }

    /// Voz del stack — desde el campo perceptivo (Campo 0).
    pub fn voice(&self, size: usize) -> Vec<f32> {
        self.fields[0].read_internal_voice(size)
    }

    /// Introspección — estado interno del campo ejecutivo.
    pub fn introspection(&self, size: usize) -> Vec<f32> {
        self.fields[self.config.depth() - 1].read_introspection(size)
    }

    /// Energía del feedback del ejecutivo al perceptivo.
    pub fn feedback_energy(&self) -> f32 {
        self.last_feedbacks[0].as_ref().map(|fb| {
            let sq: f32 = fb.iter().map(|x| x * x).sum();
            (sq / fb.len().max(1) as f32).sqrt()
        }).unwrap_or(0.0)
    }

    /// Resetear al inicio de episodio.
    pub fn reset_episode(&mut self) {
        self.action_mod.reset_episode();
        for fb in self.last_feedbacks.iter_mut() {
            *fb = None;
        }
    }

    /// Número total de conexiones en todos los campos.
    pub fn total_connections(&self) -> usize {
        self.fields.iter().map(|f| f.connection_count()).sum()
    }

    /// Referencia al campo perceptivo (Campo 0) — para diagnóstico.
    pub fn field1(&self) -> &TensionField { &self.fields[0] }

    /// Referencia al campo ejecutivo (último) — para diagnóstico.
    pub fn field2(&self) -> &TensionField { &self.fields[self.config.depth() - 1] }

    /// Guardar todos los campos.
    pub fn save(&self, path: &str) -> Result<(), String> {
        let fields: Result<Vec<FieldSnapshot>, String> =
        self.fields.iter().map(|f| f.snapshot()).collect();
        let snap = StackSnapshot {
            version: 2,
            depth:   self.config.depth(),
            fields:  fields?,
        };
        let json = serde_json::to_string(&snap)
        .map_err(|e| format!("serialización: {}", e))?;
        std::fs::write(path, json)
        .map_err(|e| format!("escritura {}: {}", path, e))?;
        Ok(())
    }

    /// Cargar todos los campos.
    /// Si el archivo es de v1 (2 campos), lo carga en modo compatibilidad.
    pub fn load(&mut self, path: &str) -> Result<(), String> {
        let json = std::fs::read_to_string(path)
        .map_err(|e| format!("lectura {}: {}", path, e))?;

        // Intentar formato v2 (N campos)
        if let Ok(snap) = serde_json::from_str::<StackSnapshot>(&json) {
            if snap.fields.len() != self.fields.len() {
                return Err(format!(
                    "profundidad incompatible: guardado={} actual={}",
                    snap.fields.len(), self.fields.len()
                ));
            }
            for (field, snap_field) in self.fields.iter_mut().zip(snap.fields) {
                field.load_snapshot(snap_field)?;
            }
            return Ok(());
        }

        // Compatibilidad v1: formato con field1/field2
        #[derive(serde::Deserialize)]
        struct StackSnapshotV1 {
            field1: FieldSnapshot,
            field2: FieldSnapshot,
        }
        let snap_v1: StackSnapshotV1 = serde_json::from_str(&json)
        .map_err(|e| format!("deserialización: {}", e))?;

        if self.fields.len() < 2 {
            return Err("stack actual tiene menos de 2 campos".to_string());
        }
        self.fields[0].load_snapshot(snap_v1.field1)?;
        let last_idx = self.fields.len() - 1;
        self.fields[last_idx].load_snapshot(snap_v1.field2)?;
        Ok(())
    }
}
