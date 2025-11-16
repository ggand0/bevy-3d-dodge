use bevy::prelude::*;
use crate::game::player::Player;
use crate::game::projectile::Projectile;

#[derive(Event)]
pub struct CollisionEvent;

#[derive(Resource, Default)]
pub struct GameState {
    pub is_game_over: bool,
}

pub struct CollisionPlugin;

impl Plugin for CollisionPlugin {
    fn build(&self, app: &mut App) {
        app.add_event::<CollisionEvent>()
            .init_resource::<GameState>()
            .add_systems(Update, (detect_collisions, check_out_of_bounds, handle_collisions));
    }
}

fn detect_collisions(
    player_query: Query<&Transform, With<Player>>,
    projectile_query: Query<&Transform, With<Projectile>>,
    mut collision_events: EventWriter<CollisionEvent>,
    game_state: Res<GameState>,
) {
    if game_state.is_game_over {
        return;
    }

    if let Ok(player_transform) = player_query.get_single() {
        for projectile_transform in projectile_query.iter() {
            let distance = player_transform
                .translation
                .distance(projectile_transform.translation);

            // Collision threshold: player radius (0.5) + projectile radius (0.3)
            if distance < 0.8 {
                collision_events.send(CollisionEvent);
            }
        }
    }
}

fn check_out_of_bounds(
    player_query: Query<&Transform, With<Player>>,
    mut collision_events: EventWriter<CollisionEvent>,
    game_state: Res<GameState>,
) {
    if game_state.is_game_over {
        return;
    }

    // Play zone boundaries (match the zone marker in main.rs)
    let zone_width = 10.0;
    let zone_depth = 8.0;
    let x_bound = zone_width / 2.0;
    let y_bound = zone_depth / 2.0;

    if let Ok(player_transform) = player_query.get_single() {
        let pos = player_transform.translation;

        // Check if player is outside the zone (with small tolerance for clamping)
        if pos.x.abs() >= x_bound - 0.01 || pos.y.abs() >= y_bound - 0.01 {
            // Player touched the boundary - this is allowed, no game over
            // Only trigger game over if they somehow get significantly outside
            if pos.x.abs() > x_bound + 0.1 || pos.y.abs() > y_bound + 0.1 {
                collision_events.send(CollisionEvent);
                info!("Out of bounds!");
            }
        }
    }
}

fn handle_collisions(
    mut collision_events: EventReader<CollisionEvent>,
    mut game_state: ResMut<GameState>,
    mut player_query: Query<&MeshMaterial3d<StandardMaterial>, With<Player>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    for _ in collision_events.read() {
        if !game_state.is_game_over {
            game_state.is_game_over = true;

            // Change player color to indicate game over
            if let Ok(material_handle) = player_query.get_single_mut() {
                if let Some(material) = materials.get_mut(&material_handle.0) {
                    material.base_color = Color::srgb(0.8, 0.1, 0.1);
                }
            }

            info!("Game Over! Press R to restart.");
        }
    }
}
