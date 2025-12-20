use bevy::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use crate::game::collision::GameState;
use crate::game::player::Player;
use crate::game::projectile::Projectile;

/// Control mode for player input
#[derive(Resource, Default, Clone, Copy, PartialEq, Eq)]
pub enum ControlMode {
    #[default]
    Human,  // Keyboard/mouse control
    RLAgent, // RL agent control via API
}

/// Training mode to prevent accidental interruptions during RL training
#[derive(Resource, Default, Clone, Copy, PartialEq, Eq)]
pub struct TrainingMode {
    pub enabled: bool,
}

/// RL environment state management
#[derive(Resource, Default)]
pub struct RLEnvironmentState {
    pub episode_steps: u32,
    pub total_reward: f32,
    pub last_reward: f32,
}

/// Information returned with each step
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepInfo {
    pub episode_steps: u32,
    pub projectile_count: usize,
}

/// Calculate reward for current timestep
pub fn calculate_reward(
    game_state: &GameState,
    player_query: &Query<&Transform, With<Player>>,
    projectile_query: &Query<&Transform, With<Projectile>>,
) -> f32 {
    // Collision penalty (terminal state)
    if game_state.is_game_over {
        return -100.0;
    }

    // Base survival reward
    let mut reward = 1.0;

    // Optional: Dodge bonus for close calls
    if let Ok(player_transform) = player_query.get_single() {
        let player_pos = player_transform.translation;

        // Find closest projectile
        let min_distance = projectile_query
            .iter()
            .map(|proj_transform| {
                player_pos.distance(proj_transform.translation)
            })
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(f32::MAX);

        // Bonus for close dodges (distance < 2.0 units)
        if min_distance < 2.0 {
            reward += (2.0 - min_distance) * 0.5;
        }
    }

    reward
}

/// Check if episode should terminate
pub fn is_episode_done(game_state: &GameState, _env_state: &RLEnvironmentState) -> bool {
    // Done if game over (collision or out of bounds)
    if game_state.is_game_over {
        return true;
    }

    false
}

/// Check if episode should be truncated (max steps reached)
pub fn is_episode_truncated(env_state: &RLEnvironmentState, max_steps: u32) -> bool {
    env_state.episode_steps >= max_steps
}

/// Create step info for response
pub fn create_step_info(
    env_state: &RLEnvironmentState,
    projectile_count: usize,
) -> HashMap<String, serde_json::Value> {
    let mut info = HashMap::new();
    info.insert("episode_steps".to_string(), serde_json::json!(env_state.episode_steps));
    info.insert("projectile_count".to_string(), serde_json::json!(projectile_count));
    info
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reward_collision() {
        let game_state = GameState { is_game_over: true };
        let mut world = World::new();
        let player_query = world.query::<&Transform, With<Player>>();
        let projectile_query = world.query::<&Transform, With<Projectile>>();

        let reward = calculate_reward(&game_state, &player_query, &projectile_query);
        assert_eq!(reward, -100.0);
    }
}
