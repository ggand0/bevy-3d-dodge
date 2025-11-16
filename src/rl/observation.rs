use bevy::prelude::*;
use crate::game::player::{Player, Velocity as PlayerVelocity};
use crate::game::projectile::{Projectile, ProjectileVelocity};

/// Maximum number of projectiles to include in observation
/// If fewer projectiles exist, the remaining slots are zero-padded
pub const MAX_PROJECTILES: usize = 10;

/// Size of observation vector:
/// - Player: position (3) + velocity (2) = 5
/// - Projectiles: 10 * (position (3) + velocity (3)) = 60
/// - Total: 65 floats
pub const OBSERVATION_SIZE: usize = 5 + (MAX_PROJECTILES * 6);

/// Extract observation from game state
/// Returns a 65-dimensional vector of f32 values
pub fn extract_observation(
    player_query: &Query<(&Transform, &PlayerVelocity), With<Player>>,
    projectile_query: &Query<(&Transform, &ProjectileVelocity), With<Projectile>>,
) -> Vec<f32> {
    let mut observation = vec![0.0; OBSERVATION_SIZE];

    // Extract player state (5 values)
    if let Ok((player_transform, player_velocity)) = player_query.get_single() {
        let pos = player_transform.translation;
        observation[0] = pos.x;
        observation[1] = pos.y;
        observation[2] = pos.z;
        observation[3] = player_velocity.0.x;
        observation[4] = player_velocity.0.y;
    }

    // Extract projectile states (up to 10 projectiles, 6 values each)
    let mut proj_index = 0;
    for (projectile_transform, projectile_velocity) in projectile_query.iter() {
        if proj_index >= MAX_PROJECTILES {
            break;
        }

        let base_idx = 5 + (proj_index * 6);
        let pos = projectile_transform.translation;
        let vel = projectile_velocity.0;

        observation[base_idx] = pos.x;
        observation[base_idx + 1] = pos.y;
        observation[base_idx + 2] = pos.z;
        observation[base_idx + 3] = vel.x;
        observation[base_idx + 4] = vel.y;
        observation[base_idx + 5] = vel.z;

        proj_index += 1;
    }

    // Remaining projectile slots are already zero-padded from initialization

    observation
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_observation_size() {
        assert_eq!(OBSERVATION_SIZE, 65);
    }
}
