// ═══════════════════════════════════════════════════════════════════════════
// CTD — Drives Internos
//
// Los drives no se programan. Emergen de la geometría global de tensión.
//
// CURIOSIDAD
// ──────────
// Actividad significativa + error distribuido uniformemente por el campo.
// Significa: el campo tiene errores de predicción en todas partes.
// No entiende lo que está percibiendo.
// Drive resultante: explorar — buscar inputs que reduzcan la incertidumbre.
//
// Fórmula: curiosity = activity × (1 - norm_var)
// activity = sigmoide(mean_tension) — activa desde ~0.02 de tensión.
// Alta actividad + baja varianza relativa = error uniformemente distribuido.
//
// MALESTAR
// ────────
// Actividad significativa + error concentrado en pocas unidades.
// Significa: hay una inconsistencia local fuerte — el campo predice bien
// en general pero falla gravemente en alguna región.
// Drive resultante: actuar para resolver esa inconsistencia específica.
//
// Fórmula: discomfort = activity × norm_var
// Alta varianza relativa = inconsistencia focal.
//
// CALMA
// ─────
// Actividad baja en general.
// Significa: el campo está en un estado coherente con sus expectativas.
// Drive resultante: consolidar — mantener lo que funciona.
//
// Fórmula: calm = 1 - activity
//
// POR QUÉ SIGMOIDE Y NO mean_tension DIRECTA
// ───────────────────────────────────────────
// Con la fórmula original (curiosity = mean_tension × ...):
//   Un campo convergido tiene mean_tension ≈ 0.001–0.005.
//   Curiosidad y malestar quedan acotados a esos valores diminutos.
//   La calma (1 - mean_tension ≈ 0.999) siempre aplasta a los otros dos.
//   Los drives de "alerta" nunca pueden emerger con fuerza.
//
// La sigmoide con punto de inflexión en 0.12 y pendiente k=25:
//   mean_tension=0.00 → activity≈0.05  (reposo — calm=0.95, correcto)
//   mean_tension=0.05 → activity≈0.18  (activación leve)
//   mean_tension=0.12 → activity≈0.50  (umbral central)
//   mean_tension=0.20 → activity≈0.87  (actividad alta)
// Esto permite que curiosidad y malestar alcancen valores significativos
// con tensiones moderadas (0.05–0.3), mientras la calma es alta
// cuando el campo predice bien (tensión < 0.05).
//
// VITALIDAD
// ─────────
// Número de conexiones vivas normalizado.
// Si el campo tiene pocas conexiones, es estructuralmente frágil.
// Drive resultante: si vitality < umbral → impulso a mantener integridad.
//
// ─────────────────────────────────────────────────────────────────────────
// IMPORTANTE
// ──────────
// Estos drives son observables del estado interno — no variables que
// se setean externamente. El módulo de acción los lee para modular
// cómo convierte la salida del campo en conducta.
//
// No son "emociones" en sentido humano. Son señales de estado del sustrato
// que el resto del sistema interpreta. Si de esa interpretación emerge
// algo que parece una emoción, es un fenómeno emergente, no diseñado.
// ═══════════════════════════════════════════════════════════════════════════

/// Estado de drives emergentes del campo en un tick dado.
#[derive(Clone, Debug, Default)]
pub struct DriveState {
    /// Impulso a explorar — error de predicción alto y distribuido.
    /// Rango [0, 1].
    pub curiosity: f32,

    /// Impulso a resolver inconsistencia — error focal y fuerte.
    /// Rango [0, 1].
    pub discomfort: f32,

    /// Estado de coherencia — tensión baja, el campo predice bien.
    /// Rango [0, 1].
    pub calm: f32,

    /// Integridad estructural — qué tan poblado está el campo de conexiones.
    /// Rango [0, 1]. Bajo → el sistema está perdiendo estructura.
    pub vitality: f32,

    /// Tensión global media (métrica cruda).
    pub mean_tension: f32,

    /// Varianza de tensión (métrica cruda).
    pub var_tension: f32,
}

impl DriveState {
    /// Calcular drives desde métricas crudas del campo.
    pub fn from_field(mean_tension: f32, var_tension: f32, connection_count: usize) -> Self {
        // ── Geometría de la tensión ────────────────────────────────────────
        // norm_var: coeficiente de variación (CV) = std / mean.
        // Mide si la tensión está distribuida (CV bajo) o concentrada (CV alto).
        // Con mean_tension ≈ 0 no tiene sentido — todo es calma.
        let norm_var = if mean_tension > 0.001 {
            (var_tension.sqrt() / mean_tension).clamp(0.0, 1.0)
        } else {
            0.0
        };

        // ── Señal de actividad relativa ───────────────────────────────────
        // El problema de curiosity = mean_tension × (1-norm_var) es que
        // queda acotada por mean_tension (~0.001 en campo convergido).
        // Un campo que predice bien DEBE tener tensión baja — eso es correcto.
        // Pero entonces curiosidad y malestar nunca pueden emerger con fuerza.
        //
        // Solución: medir la geometría de tensión de forma independiente a
        // su escala absoluta, usando una señal de actividad normalizada.
        //
        // activity: sigmoide centrada en 0.12, pendiente k=25.
        //   mean_tension=0.00 → activity≈0.05  (reposo — calm≈0.95)
        //   mean_tension=0.05 → activity≈0.18  (activación leve)
        //   mean_tension=0.12 → activity≈0.50  (umbral central)
        //   mean_tension=0.20 → activity≈0.87  (actividad alta)
        //   mean_tension=0.50 → activity≈1.00  (saturación)
        let k = 25.0f32;
        let center = 0.12f32;
        let activity = (1.0 / (1.0 + (-k * (mean_tension - center)).exp())).clamp(0.0, 1.0);

        // Curiosidad: actividad presente + error bien distribuido (norm_var bajo)
        // El campo "no entiende" uniformemente — impulso a explorar.
        let curiosity = (activity * (1.0 - norm_var)).clamp(0.0, 1.0);

        // Malestar: actividad presente + error concentrado (norm_var alto)
        // El campo falla focalmente — impulso a resolver esa inconsistencia.
        let discomfort = (activity * norm_var).clamp(0.0, 1.0);

        // Calma: inversamente proporcional a la actividad (no a mean_tension crudo)
        // Esto hace que la calma no sea simplemente "1 - tensión mínima"
        // sino "ausencia de actividad significativa".
        let calm = (1.0 - activity).clamp(0.0, 1.0);

        // Vitalidad: función del número de conexiones
        // Umbrales arbitrarios calibrables — 500 conns = 100% vitalidad
        let vitality = ((connection_count as f32) / 500.0).clamp(0.0, 1.0);

        Self {
            curiosity,
            discomfort,
            calm,
            vitality,
            mean_tension,
            var_tension,
        }
    }

    /// Modificador de learning rate sugerido por los drives.
    /// Más malestar → aprender más rápido.
    /// Más calma    → aprender más despacio (consolidar).
    pub fn lr_modifier(&self) -> f32 {
        let urgency = self.discomfort * 2.0 + self.curiosity * 0.5;
        let calm_damp = self.calm * 0.3;
        (1.0 + urgency - calm_damp).clamp(0.5, 4.0)
    }

    /// Modificador de exploración sugerido por los drives.
    /// Curiosidad alta → más exploración.
    /// Calma alta     → menos exploración (explotar lo conocido).
    pub fn exploration_modifier(&self, base: f32) -> f32 {
        let boost = self.curiosity * 0.4;
        let damp  = self.calm * 0.2;
        (base + boost - damp).clamp(0.02, 0.95)
    }

    /// ¿El ser está en un estado de "crisis"?
    /// Alta tensión + alta varianza + baja vitalidad.
    pub fn is_in_crisis(&self) -> bool {
        self.mean_tension > 0.7 && self.discomfort > 0.5 && self.vitality < 0.2
    }

    /// Resumen compacto para diagnóstico.
    pub fn summary(&self) -> String {
        format!(
            "curiosity:{:.2} discomfort:{:.2} calm:{:.2} vitality:{:.2}",
            self.curiosity, self.discomfort, self.calm, self.vitality
        )
    }

    /// Nombre del drive dominante — para diagnóstico cualitativo.
    pub fn dominant(&self) -> &'static str {
        let values = [
            (self.curiosity,  "curiosity"),
            (self.discomfort, "discomfort"),
            (self.calm,       "calm"),
            (self.vitality,   "vitality"),
        ];
        values.iter()
        .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
        .map(|(_, name)| *name)
        .unwrap_or("undefined")
    }
}

// ─────────────────────────────────────────────────────────────────────────
// TESTS
// ─────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calm_when_no_tension() {
        let d = DriveState::from_field(0.0, 0.0, 100);
        assert!(d.calm > 0.9, "should be calm with zero tension");
        assert!(d.curiosity < 0.1);
        assert!(d.discomfort < 0.1);
    }

    #[test]
    fn curiosity_when_high_uniform_tension() {
        // Alta tensión, baja varianza → curiosidad alta
        let d = DriveState::from_field(0.8, 0.001, 100);
        assert!(d.curiosity > d.discomfort,
                "uniform high tension should produce curiosity > discomfort: c={:.2} d={:.2}",
                d.curiosity, d.discomfort);
    }

    #[test]
    fn discomfort_when_high_focal_tension() {
        // Alta tensión, alta varianza → malestar alto
        let d = DriveState::from_field(0.7, 0.6, 100);
        assert!(d.discomfort > d.curiosity,
                "focal high tension should produce discomfort > curiosity: c={:.2} d={:.2}",
                d.curiosity, d.discomfort);
    }

    #[test]
    fn vitality_zero_with_no_connections() {
        let d = DriveState::from_field(0.3, 0.1, 0);
        assert_eq!(d.vitality, 0.0);
    }

    #[test]
    fn vitality_caps_at_one() {
        let d = DriveState::from_field(0.3, 0.1, 10_000);
        assert_eq!(d.vitality, 1.0);
    }

    #[test]
    fn crisis_requires_all_three_conditions() {
        // No crisis si solo hay alta tensión
        let d1 = DriveState::from_field(0.8, 0.5, 500);
        assert!(!d1.is_in_crisis(), "high vitality should prevent crisis");

        // Crisis: alta tensión + alta varianza + baja vitalidad
        let mut d2 = DriveState::from_field(0.8, 0.7, 0);
        d2.discomfort = 0.6;
        assert!(d2.is_in_crisis());
    }

    #[test]
    fn lr_modifier_higher_with_discomfort() {
        let d_calm = DriveState::from_field(0.05, 0.01, 200);
        let d_disc = DriveState::from_field(0.8, 0.6, 200);
        assert!(d_disc.lr_modifier() > d_calm.lr_modifier(),
                "discomfort should raise lr_modifier");
    }

    #[test]
    fn exploration_increases_with_curiosity() {
        let d_calm = DriveState::from_field(0.05, 0.01, 200);
        let d_curi = DriveState::from_field(0.8, 0.005, 200);
        let base = 0.1;
        assert!(d_curi.exploration_modifier(base) > d_calm.exploration_modifier(base),
                "curiosity should increase exploration");
    }

    #[test]
    fn dominant_returns_valid_string() {
        let d = DriveState::from_field(0.5, 0.2, 100);
        let dom = d.dominant();
        assert!(["curiosity", "discomfort", "calm", "vitality"].contains(&dom));
    }

    #[test]
    fn all_drives_in_range() {
        for (mt, vt, cc) in [
            (0.0, 0.0, 0),
            (0.5, 0.1, 100),
            (1.0, 1.0, 1000),
            (0.3, 0.8, 50),
        ] {
            let d = DriveState::from_field(mt, vt, cc);
            assert!(d.curiosity  >= 0.0 && d.curiosity  <= 1.0);
            assert!(d.discomfort >= 0.0 && d.discomfort <= 1.0);
            assert!(d.calm       >= 0.0 && d.calm       <= 1.0);
            assert!(d.vitality   >= 0.0 && d.vitality   <= 1.0);
        }
    }
}
