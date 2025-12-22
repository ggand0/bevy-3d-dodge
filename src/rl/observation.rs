use bevy::prelude::*;
use crate::config::ObservationMode;
use crate::game::player::{Player, Velocity as PlayerVelocity};
use crate::game::projectile::{Projectile, ProjectileVelocity, ThrowerIndicator};

/// Maximum number of projectiles to include in observation
/// If fewer projectiles exist, the remaining slots are zero-padded
pub const MAX_PROJECTILES: usize = 10;

/// Size of standard observation vector (backward compatible):
/// - Player: position (3) + velocity (2) = 5
/// - Projectiles: 10 * (position (3) + velocity (3)) = 60
/// - Total: 65 floats
pub const OBSERVATION_SIZE: usize = 5 + (MAX_PROJECTILES * 6);

/// Size of extended observation vector with thrower indicator:
/// - Standard: 65 floats
/// - Thrower position: 3 floats (x, y, z)
/// - Time until throw: 1 float (0.0-1.0 normalized)
/// - Total: 69 floats
pub const OBSERVATION_SIZE_EXTENDED: usize = OBSERVATION_SIZE + 4;

/// Extract observation from game state (standard 65-dim, backward compatible)
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

/// Extract observation from game state with configurable mode
/// Returns 65-dim (Standard) or 69-dim (WithThrowerIndicator) vector
pub fn extract_observation_with_mode(
    observation_mode: ObservationMode,
    thrower_delay_seconds: f32,
    player_query: &Query<(&Transform, &PlayerVelocity), With<Player>>,
    projectile_query: &Query<(&Transform, &ProjectileVelocity), With<Projectile>>,
    thrower_query: &Query<&ThrowerIndicator>,
) -> Vec<f32> {
    match observation_mode {
        ObservationMode::Standard => {
            extract_observation(player_query, projectile_query)
        }
        ObservationMode::TopDownImage => {
            // For image mode, return empty vector - images are handled separately
            Vec::new()
        }
        ObservationMode::WithThrowerIndicator => {
            let mut observation = vec![0.0; OBSERVATION_SIZE_EXTENDED];

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

            // Extract thrower indicator info (4 values at indices 65-68)
            // If no indicator exists, these remain zero
            if let Some(indicator) = thrower_query.iter().next() {
                let pos = indicator.spawn_position;
                observation[65] = pos.x;
                observation[66] = pos.y;
                observation[67] = pos.z;
                // Normalize time remaining to 0.0-1.0
                let time_remaining = indicator.spawn_timer.remaining_secs();
                observation[68] = (time_remaining / thrower_delay_seconds).clamp(0.0, 1.0);
            }

            observation
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_observation_size() {
        assert_eq!(OBSERVATION_SIZE, 65);
    }

    #[test]
    fn test_observation_size_extended() {
        assert_eq!(OBSERVATION_SIZE_EXTENDED, 69);
    }
}
