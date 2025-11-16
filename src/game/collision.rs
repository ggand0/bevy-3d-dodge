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
            .add_systems(Update, (detect_collisions, handle_collisions));
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

fn handle_collisions(
    mut collision_events: EventReader<CollisionEvent>,
    mut game_state: ResMut<GameState>,
    mut player_query: Query<&mut Handle<StandardMaterial>, With<Player>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    for _ in collision_events.read() {
        if !game_state.is_game_over {
            game_state.is_game_over = true;

            // Change player color to indicate game over
            if let Ok(material_handle) = player_query.get_single_mut() {
                if let Some(material) = materials.get_mut(&*material_handle) {
                    material.base_color = Color::srgb(0.8, 0.1, 0.1);
                }
            }

            info!("Game Over! Press R to restart.");
        }
    }
}
