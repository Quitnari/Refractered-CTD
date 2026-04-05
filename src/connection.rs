// ═══════════════════════════════════════════════════════════════════════════
// CTD — Conexión
//
// Una conexión une dos unidades: origen (from) y destino (to).
// Transporta tensión del origen al destino, modulada por peso y fase.
//
// COMPONENTES
// ────────────
// weight    ∈ [-1, 1]   magnitud y signo de la influencia
//                        positivo → refuerza la tensión del origen en el destino
//                        negativo → invierte la tensión (inhibición)
//
// phase     ∈ [0, 2π]   alineación temporal entre las dos unidades
//                        fase ≈ 0   → alineadas (la tensión pasa limpia)
//                        fase ≈ π   → opuestas (la tensión se atenúa o invierte)
//
// relevance ∈ [0, 1]    potencial de existencia de la conexión
//                        sube cuando la conexión reduce tensión en el destino
//                        baja cuando no contribuye
//                        si cae a 0 → la conexión desaparece sola
//
// SEÑAL TRANSMITIDA
// ─────────────────
// signal = tension_origen × weight × cos(phase)
//
// El coseno de la fase modula cuánto de la tensión pasa:
//   cos(0)   =  1.0  → pasa todo (alineadas)
//   cos(π/2) =  0.0  → no pasa nada (ortogonales)
//   cos(π)   = -1.0  → invierte (opuestas)
//
// APRENDIZAJE ASIMÉTRICO
// ───────────────────────
// Cuando el destino falla su predicción (error ≠ 0), el peso y la fase
// se ajustan. El ajuste depende de CUÁNDO llegó la señal de esta conexión
// relativo al error: señales que llegaron antes del error absorben más
// del ajuste que las que llegaron después.
//
// Esto se implementa via un timestamp relativo: cada conexión registra
// en qué fracción del tick anterior contribuyó. La fracción modula el lr.
//
// OLVIDO INTRÍNSECO
// ─────────────────
// Cada tick, la relevancia decae pasivamente.
// Si la conexión contribuyó a reducir tensión, la relevancia sube.
// Si no contribuyó, sigue decayendo.
// Cuando relevance < min_relevance → la conexión se elimina del campo.
// ═══════════════════════════════════════════════════════════════════════════

use std::f32::consts::PI;

/// Una conexión dirigida entre dos unidades del campo.
#[derive(Clone, Debug)]
pub struct Connection {
    /// Índice de la unidad origen.
    pub from: usize,
    /// Índice de la unidad destino.
    pub to: usize,

    /// Peso de la conexión ∈ [-1, 1].
    pub weight: f32,

    /// Fase de la conexión ∈ [0, 2π].
    /// Evoluciona más lentamente que el peso.
    pub phase: f32,

    /// Potencial de relevancia ∈ [0, 1].
    /// Cuando cae a 0, la conexión desaparece.
    pub relevance: f32,

    /// Fracción del tick en que esta conexión contribuyó por última vez.
    /// 0.0 = al principio del tick, 1.0 = al final.
    /// Usado para asimetría temporal en el aprendizaje.
    pub(crate) contrib_timing: f32,

    /// Contribución absoluta al último tick (para calcular relevance).
    pub(crate) last_contribution: f32,
}

impl Connection {
    /// Crear una nueva conexión con peso y fase iniciales.
    pub fn new(from: usize, to: usize, weight: f32, phase: f32) -> Self {
        Self {
            from,
            to,
            weight: weight.clamp(-1.0, 1.0),
            phase: phase.rem_euclid(2.0 * PI),
            relevance: 1.0,
            contrib_timing: 0.5,
            last_contribution: 0.0,
        }
    }

    /// Crear una conexión con peso dado y fase aleatoria.
    pub fn with_weight(from: usize, to: usize, weight: f32, rng_phase: f32) -> Self {
        Self::new(from, to, weight, rng_phase)
    }

    /// Señal que esta conexión transmite dado la tensión del origen.
    /// signal = tension × weight × cos(phase)
    #[inline]
    pub fn signal(&self, origin_tension: f32) -> f32 {
        origin_tension * self.weight * self.phase.cos()
    }

    /// Calcular y registrar la contribución de esta conexión.
    /// Retorna la señal para que el campo la entregue al destino.
    pub(crate) fn compute_and_record(&mut self, origin_tension: f32, timing: f32) -> f32 {
        let s = self.signal(origin_tension);
        self.last_contribution = s.abs();
        self.contrib_timing = timing.clamp(0.0, 1.0);
        s
    }

    /// Aprender del error de predicción del destino.
    ///
    /// error: (what_is_destino - what_will_destino) — cuánto falló la predicción
    /// base_lr: learning rate base del campo
    /// tension_reduction: cuánto redujo esta conexión la tensión del destino
    ///
    /// El ajuste del peso es mayor si la conexión llegó antes en el tick
    /// (contrib_timing cercano a 0.0 = llegó antes = más responsable).
    ///
    /// La fase evoluciona más lentamente — solo si hay error sostenido.
    pub fn learn(&mut self, error: f32, base_lr: f32, tension_reduction: f32) {
        // Asimetría temporal: conexiones que llegaron antes son más responsables
        // timing=0 → factor=1.0 (máximo ajuste)
        // timing=1 → factor=0.2 (mínimo ajuste)
        let temporal_factor = 1.0 - self.contrib_timing * 0.8;

        let lr = (base_lr * temporal_factor).clamp(0.0001, 0.3);

        // Ajuste de peso: moverse en dirección que habría reducido el error
        // Si el error es positivo (was_is > what_will), la conexión debería
        // haber empujado más hacia what_is. El ajuste depende del signo del error
        // y de la contribución de esta conexión.
        let weight_delta = lr * error * self.phase.cos();
        self.weight = (self.weight + weight_delta).clamp(-1.0, 1.0);

        // Ajuste de fase: más lento (lr_phase = lr * 0.05)
        // La fase se ajusta para maximizar alineación cuando hay contribución real
        if self.last_contribution > 0.01 {
            let phase_lr = lr * 0.05;
            // Si la conexión contribuyó a reducir tensión, la fase se acerca a 0
            // Si no, la fase deriva hacia π/2 (desacoplamiento)
            let phase_target = if tension_reduction > 0.0 { 0.0 } else { PI / 2.0 };
            let phase_error = phase_target - self.phase;
            // Manejar circularidad: tomar el camino más corto
            let phase_error = if phase_error > PI {
                phase_error - 2.0 * PI
            } else if phase_error < -PI {
                phase_error + 2.0 * PI
            } else {
                phase_error
            };
            self.phase = (self.phase + phase_lr * phase_error).rem_euclid(2.0 * PI);
        }

        // Actualizar relevancia
        // Sube si la conexión contribuyó a reducir tensión
        // Baja pasivamente cada tick
        let relevance_boost = if tension_reduction > 0.01 {
            tension_reduction.min(0.05)
        } else {
            0.0
        };
        self.relevance = (self.relevance * 0.9995 + relevance_boost).clamp(0.0, 1.0);
    }

    /// Decay pasivo de relevancia (llamado cada tick aunque no haya aprendizaje).
    pub(crate) fn decay_relevance(&mut self) {
        self.relevance *= 0.9998;
    }

    /// ¿Debe eliminarse esta conexión?
    #[inline]
    pub fn is_dead(&self) -> bool {
        self.relevance < 0.005
    }
}

// ─────────────────────────────────────────────────────────────────────────
// TESTS
// ─────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signal_zero_when_no_tension() {
        let c = Connection::new(0, 1, 0.8, 0.0);
        assert_eq!(c.signal(0.0), 0.0);
    }

    #[test]
    fn signal_full_when_phase_zero() {
        let c = Connection::new(0, 1, 1.0, 0.0);
        // cos(0) = 1 → signal = tension × 1.0 × 1.0
        let s = c.signal(0.5);
        assert!((s - 0.5).abs() < 1e-6);
    }

    #[test]
    fn signal_zero_when_phase_half_pi() {
        let c = Connection::new(0, 1, 1.0, PI / 2.0);
        // cos(π/2) ≈ 0 → signal ≈ 0
        let s = c.signal(1.0);
        assert!(s.abs() < 1e-5);
    }

    #[test]
    fn signal_inverted_when_phase_pi() {
        let c = Connection::new(0, 1, 1.0, PI);
        // cos(π) = -1 → signal = -tension
        let s = c.signal(0.6);
        assert!((s - (-0.6)).abs() < 1e-5);
    }

    #[test]
    fn inhibitory_weight_inverts_signal() {
        let c = Connection::new(0, 1, -0.5, 0.0);
        let s = c.signal(1.0);
        assert!((s - (-0.5)).abs() < 1e-6);
    }

    #[test]
    fn relevance_decays_over_time() {
        let mut c = Connection::new(0, 1, 0.5, 0.0);
        let initial = c.relevance;
        for _ in 0..1000 {
            c.decay_relevance();
        }
        assert!(c.relevance < initial);
    }

    #[test]
    fn connection_dies_without_use() {
        let mut c = Connection::new(0, 1, 0.5, 0.0);
        // Forzar relevancia muy baja
        c.relevance = 0.004;
        assert!(c.is_dead());
    }

    #[test]
    fn learning_moves_weight_toward_correction() {
        let mut c = Connection::new(0, 1, 0.0, 0.0);
        let initial = c.weight;
        // Error positivo → el peso debería moverse en dirección del error
        c.learn(0.5, 0.1, 0.1);
        assert!(c.weight > initial, "weight should increase with positive error");
    }

    #[test]
    fn weight_stays_in_range_after_learning() {
        let mut c = Connection::new(0, 1, 0.95, 0.0);
        for _ in 0..1000 {
            c.learn(1.0, 0.5, 1.0);
        }
        assert!(c.weight <= 1.0 && c.weight >= -1.0);
    }

    #[test]
    fn early_timing_has_stronger_learning() {
        let mut c_early = Connection::new(0, 1, 0.0, 0.0);
        let mut c_late  = Connection::new(0, 1, 0.0, 0.0);
        c_early.contrib_timing = 0.0;
        c_late.contrib_timing  = 1.0;
        c_early.learn(0.5, 0.1, 0.1);
        c_late.learn(0.5, 0.1, 0.1);
        // La conexión temprana debe haber aprendido más (mayor delta de peso)
        assert!(c_early.weight.abs() > c_late.weight.abs(),
                "early connection should learn more: early={} late={}", c_early.weight, c_late.weight);
    }
}
