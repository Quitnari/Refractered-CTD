// ═══════════════════════════════════════════════════════════════════════════
// CTD — Unidad
//
// La unidad es la pieza mínima del campo.
// No es una neurona — no "dispara".
// Es un nodo que vive en tensión entre lo que es y lo que espera ser.
//
// ESTADOS
// ───────
// what_is   ∈ [-1, 1]  estado actual — lo que el campo externo empujó a esta unidad
// what_will ∈ [-1, 1]  expectativa   — lo que esta unidad predice que será en t+1
//
// TENSIÓN
// ───────
// tension = what_will - what_is
//
// Si tension ≈ 0: la unidad predijo correctamente. No aprende mucho.
// Si tension ≠ 0: la unidad se equivocó. Aprende proporcionalmente.
//
// La tensión no es negativa ni positiva en sentido de "bueno/malo".
// Es simplemente la distancia entre expectativa y realidad.
// El campo en su conjunto interpreta la geometría de esas distancias.
//
// INERCIA
// ───────
// El estado actual no cambia abruptamente — tiene inercia.
// Esto evita que el campo sea reactivo puro (sin memoria de instante).
// La inercia es configurable: alta = sistema lento y estable,
// baja = sistema reactivo y sensible.
// ═══════════════════════════════════════════════════════════════════════════

/// Una unidad del campo de tensión diferencial.
#[derive(Clone, Debug)]
pub struct Unit {
    /// Estado actual: lo que la unidad ES en este instante.
    /// Rango [-1, 1]. Se actualiza cada tick con la influencia recibida.
    pub what_is: f32,

    /// Expectativa: lo que la unidad predice que será en t+1.
    /// Rango [-1, 1]. Se actualiza mediante aprendizaje.
    pub what_will: f32,

    /// Inercia del estado actual: cuánto resiste cambiar.
    /// 0.0 = sin inercia (reactiva pura)
    /// 1.0 = sin cambio (congelada)
    /// Valor típico: 0.15–0.35
    pub inertia: f32,

    /// Tensión acumulada en el tick anterior.
    /// Se usa para diagnóstico y para calcular drives globales.
    pub last_tension: f32,

    /// Influencia recibida en este tick (suma de tensiones entrantes).
    /// Se resetea al inicio de cada tick.
    pub(crate) incoming: f32,
}

impl Unit {
    /// Crear una unidad con estado inicial aleatorio dentro de [-init_scale, init_scale].
    pub fn new(init_what_is: f32, init_what_will: f32, inertia: f32) -> Self {
        Self {
            what_is:      init_what_is.clamp(-1.0, 1.0),
            what_will:    init_what_will.clamp(-1.0, 1.0),
            inertia:      inertia.clamp(0.0, 0.99),
            last_tension: 0.0,
            incoming:     0.0,
        }
    }

    /// Crear una unidad en reposo (sin tensión, sin expectativa).
    pub fn at_rest(inertia: f32) -> Self {
        Self::new(0.0, 0.0, inertia)
    }

    /// Tensión actual: distancia entre expectativa y realidad.
    /// Positiva si esperaba más de lo que es.
    /// Negativa si esperaba menos de lo que es.
    #[inline]
    pub fn tension(&self) -> f32 {
        self.what_will - self.what_is
    }

    /// Magnitud absoluta de la tensión — para métricas globales.
    #[inline]
    pub fn tension_magnitude(&self) -> f32 {
        self.tension().abs()
    }

    /// Recibir influencia de otra unidad (llamado por el campo al propagar).
    /// No aplica el cambio todavía — se acumula hasta apply_incoming().
    #[inline]
    pub(crate) fn receive(&mut self, delta: f32) {
        self.incoming += delta;
    }

    /// Aplicar la influencia acumulada al estado actual.
    /// Usa inercia para suavizar el cambio.
    /// Llamado por el campo una vez por tick, después de propagar todas las tensiones.
    pub(crate) fn apply_incoming(&mut self) {
        let raw_new = self.what_is + self.incoming;
        // Interpolación con inercia: el estado se mueve hacia el nuevo valor,
        // pero frenado por la inercia
        self.what_is = (self.what_is * self.inertia
        + raw_new * (1.0 - self.inertia))
        .clamp(-1.0, 1.0);
        self.last_tension = self.tension();
        self.incoming = 0.0;
    }

    /// Actualizar expectativa dado el error de predicción.
    /// lr_mod: modificador de learning rate (puede venir de drives globales).
    /// El error es (what_is - what_will): cuánto nos equivocamos.
    /// La expectativa se mueve hacia la realidad, proporcionalmente al error.
    pub fn update_expectation(&mut self, base_lr: f32, lr_mod: f32) {
        let error = self.what_is - self.what_will;
        let lr = (base_lr * lr_mod).clamp(0.0001, 0.5);
        self.what_will = (self.what_will + lr * error).clamp(-1.0, 1.0);
    }

    /// Inyectar señal externa (input del entorno).
    /// No usa inercia — la señal externa sobreescribe directamente.
    /// Para entradas sensoriales, no para propagación interna.
    pub fn inject(&mut self, value: f32) {
        self.what_is = value.clamp(-1.0, 1.0);
        self.last_tension = self.tension();
    }
}

// ─────────────────────────────────────────────────────────────────────────
// TESTS
// ─────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tension_is_zero_at_rest() {
        let u = Unit::at_rest(0.2);
        assert_eq!(u.tension(), 0.0);
    }

    #[test]
    fn tension_reflects_expectation_vs_reality() {
        let u = Unit::new(0.3, 0.7, 0.2);
        // espera 0.7, es 0.3 → tensión = 0.4
        assert!((u.tension() - 0.4).abs() < 1e-6);
    }

    #[test]
    fn apply_incoming_moves_state_with_inertia() {
        let mut u = Unit::new(0.0, 0.5, 0.5);
        u.receive(0.4);
        u.apply_incoming();
        // Con inercia 0.5: new = 0.0 * 0.5 + (0.0+0.4) * 0.5 = 0.2
        assert!((u.what_is - 0.2).abs() < 1e-5);
    }

    #[test]
    fn incoming_resets_after_apply() {
        let mut u = Unit::at_rest(0.2);
        u.receive(0.5);
        u.apply_incoming();
        assert_eq!(u.incoming, 0.0);
    }

    #[test]
    fn expectation_converges_toward_reality() {
        let mut u = Unit::new(0.8, 0.0, 0.1);
        for _ in 0..100 {
            u.update_expectation(0.1, 1.0);
        }
        // Después de 100 pasos, what_will debe estar cerca de what_is (0.8)
        assert!((u.what_will - u.what_is).abs() < 0.05);
    }

    #[test]
    fn inject_overrides_state_directly() {
        let mut u = Unit::new(0.0, 0.5, 0.9);
        u.inject(1.0);
        assert_eq!(u.what_is, 1.0);
    }

    #[test]
    fn state_clamped_to_range() {
        let u = Unit::new(2.0, -3.0, 0.2);
        assert!(u.what_is <= 1.0 && u.what_is >= -1.0);
        assert!(u.what_will <= 1.0 && u.what_will >= -1.0);
    }
}
