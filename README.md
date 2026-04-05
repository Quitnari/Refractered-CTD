# CTD: Campo de Tensión Diferencial 🧠⚡

*(English TL;DR below)*

CTD (Campo de Tensión Diferencial) no es una red neuronal tradicional. Es una arquitectura cognitiva emergente escrita en Rust. En lugar de propagar activaciones estáticas o usar backpropagation convencional, CTD modela unidades que "viven en tensión" entre lo que **son** (realidad sensorial) y lo que **esperan ser** (predicción temporal).

De esta topología de errores y tensiones no surgen simples cálculos, sino **drives conductuales emergentes** (Curiosidad, Malestar, Calma, Vitalidad) que guían cómo el sistema interactúa con su entorno.

## ✨ Conceptos Clave
- **Tensión Diferencial:** `tension = what_will - what_is`. El aprendizaje ocurre solo cuando hay sorpresa (error de predicción).
- **Asimetría Temporal:** Las conexiones aprenden en función de *cuándo* aportaron su señal relativa al error, usando fase matemática (`cos(phase)`).
- **Drives Emergentes:** El sistema no tiene emociones programadas. "Curiosidad" es simplemente una lectura del estado del campo: alta tensión + baja varianza (error no comprendido pero generalizado).
- **Stack Abstracto:** Los campos se pueden apilar (`FieldStack`), creando gradientes donde los primeros campos son perceptivos y rápidos, y los últimos son ejecutivos, lentos y abstractos.

## 🚀 Cómo usarlo

Asegúrate de tener [Rust](https://www.rust-lang.org/) instalado.

**1. Ver una simulación en la consola:**
```bash
cargo run --bin ctd_main
```

**2. Ejecutar la batería de pruebas cognitivas (Harness de Evaluación):**
```bash
cargo run --bin eval
```

**3. Ejecutar las pruebas unitarias técnicas:**
```bash
cargo test
```

## 📜 Licencia
Este proyecto es software libre y de código abierto. Está licenciado bajo la **GPL v3.0**. Eres libre de usarlo, modificarlo y distribuirlo asegurando que las obras derivadas mantengan la misma libertad.

---

### 🇬🇧 English TL;DR
**CTD (Differential Tension Field)** is a novel, emergent cognitive architecture written in Rust. It distances itself from traditional artificial neural networks (ANNs). Instead of activations and static weights, nodes possess state (`what_is`) and expectation (`what_will`). The system minimizes "tension" (prediction error) over time.

Fascinatingly, instead of hardcoded reinforcement learning algorithms, CTD produces **emergent psychological drives** (Curiosity, Discomfort, Calm) derived purely from the mathematical topology of the field's tension variance. These drives modulate exploration and learning rates dynamically.

Run `cargo run --bin eval` to watch the cognitive evaluation harness in action! Licensed under GPL v3.0.
