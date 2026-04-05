# CTD: Campo de Tensión Diferencial / Differential Tension Field

> ⚠️ **Estado del Proyecto / Project Status: Experimental (Alpha)**
>
> **ES:** Este sistema es una prueba de concepto en desarrollo activo. Las mecánicas internas, la API y la estructura pueden cambiar drásticamente. Se agradece enormemente el feedback y las contribuciones de quienes quieran explorar esta arquitectura.
>
> **EN:** This system is a proof of concept under active development. Internal mechanics, API, and structure are subject to significant changes. Feedback and contributions from those interested in exploring this architecture are highly welcome.

---

## 🇪🇸 Español

CTD (Campo de Tensión Diferencial) no es una red neuronal tradicional. Es una arquitectura cognitiva emergente escrita en Rust. En lugar de propagar activaciones estáticas o usar backpropagation convencional, CTD modela unidades que "viven en tensión" entre lo que **son** (realidad sensorial) y lo que **esperan ser** (predicción temporal).

De esta topología de errores y tensiones no surgen simples cálculos, sino **drives conductuales emergentes** (Curiosidad, Malestar, Calma, Vitalidad) que guían cómo el sistema interactúa con su entorno.

### ✨ Conceptos Clave
- **Tensión Diferencial:** `tension = what_will - what_is`. El aprendizaje ocurre solo cuando hay sorpresa (error de predicción).
- **Asimetría Temporal:** Las conexiones aprenden en función de *cuándo* aportaron su señal relativa al error, usando fase matemática (`cos(phase)`).
- **Drives Emergentes:** El sistema no tiene emociones programadas. "Curiosidad" es una lectura del estado del campo: alta tensión + baja varianza.
- **Stack Abstracto:** Los campos se pueden apilar (`FieldStack`), creando gradientes desde lo perceptivo (rápido) a lo ejecutivo (lento/abstracto).

---

## 🇬🇧 English

CTD (Differential Tension Field) is not a traditional neural network. It is an emergent cognitive architecture written in Rust. Instead of propagating static activations or using conventional backpropagation, CTD models units that "live in tension" between what they **are** (sensorial reality) and what they **expect to be** (temporal prediction).

From this topology of errors and tensions, the system generates **emergent behavioral drives** (Curiosity, Discomfort, Calm, Vitality) that guide how the agent interacts with its environment.

### ✨ Key Concepts
- **Differential Tension:** `tension = what_will - what_is`. Learning only occurs when there is surprise (prediction error).
- **Temporal Asymmetry:** Connections learn based on *when* they contributed their signal relative to the error, using mathematical phase (`cos(phase)`).
- **Emergent Drives:** The system has no programmed emotions. "Curiosity" is a reading of the field's state: high tension + low variance.
- **Abstract Stack:** Fields can be stacked (`FieldStack`), creating gradients from perceptive (fast) to executive (slow/abstract) layers.

---

## 🚀 Cómo usarlo / How to use

**1. Ver simulación / Run simulation:**
```bash
cargo run --bin ctd_main
```

**2. Batería de pruebas / Evaluation harness:**
```bash
cargo run --bin eval
```

**3. Pruebas técnicas / Unit tests:**
```bash
cargo test
```

## 📜 Licencia / License
Licensed under **GPL v3.0**.
