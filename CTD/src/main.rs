// cargo run   → pipeline completo: campo + acción
// cargo test  → todos los tests

use ctd::{TensionField, ActionModule, ActionMode};
use ctd::field::FieldConfig;

fn main() {
    println!("═══════════════════════════════════════");
    println!("  CTD — Campo de Tensión Diferencial   ");
    println!("  Refractered Sinlanix v0.1             ");
    println!("═══════════════════════════════════════\n");

    // Campo: 4 sensores de entrada, 4 unidades de salida
    let config = FieldConfig::with_sizes(32, 4, 4);
    let mut field = TensionField::new_dense(config, 0.4);

    // Módulo de acción: 4 output del campo → 4 acciones discretas (N/S/W/E)
    let mut action_mod = ActionModule::new(4, 4, ActionMode::Discrete { n_actions: 4 });

    let labels = ["↑N", "↓S", "←W", "→E"];
    let input   = vec![0.8, -0.5, 0.3, -0.2];

    println!("{:<6} {:>14} {:>13} {:>10} {:>10}  {}",
             "tick", "t_interna", "t_salida", "malestar", "calma", "acción");
    println!("{}", "─".repeat(72));

    for tick in 1..=300 {
        field.inject_input(&input);
        let drives = field.tick(1.0);
        let output = field.read_output();
        let action = action_mod.act(&output, &drives);

        if tick % 30 == 0 || tick == 1 {
            let label = action.discrete
            .and_then(|i| labels.get(i))
            .copied()
            .unwrap_or("?");
            println!("{:<6} {:>14.4} {:>13.4} {:>10.4} {:>10.4}  {}",
                     tick,
                     field.global_tension(),
                     field.output_tension(),
                     drives.discomfort,
                     drives.calm,
                     label,
            );
        }
    }

    println!("\n── Cambio de input (el ser debería detectarlo) ──");
    let new_input = vec![-0.9, 0.7, -0.6, 0.4];
    field.inject_input(&new_input);
    let drives = field.tick(1.0);
    let output = field.read_output();
    let action = action_mod.act(&output, &drives);
    println!("t_interna: {:.4}  drive: {}  acción: {}",
             field.global_tension(),
             drives.dominant(),
             action.discrete.and_then(|i| labels.get(i)).copied().unwrap_or("?"),
    );

    println!("\n── Reset de episodio ──");
    action_mod.reset_episode();
    println!("Momentum reseteado. Listo para nuevo episodio.");
}
