use bevy::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Resource)]
pub struct GameConfig {
    pub player_speed: f32,
    pub player_start_height: f32,
    pub projectile_speed: f32,
    pub projectile_spawn_interval: f32,
    pub projectile_spawn_distance: f32,
    pub max_projectiles: usize,
}

impl Default for GameConfig {
    fn default() -> Self {
        Self {
            player_speed: 5.0,
            player_start_height: 1.0,
            projectile_speed: 3.0,
            projectile_spawn_interval: 2.0,
            projectile_spawn_distance: 20.0,
            max_projectiles: 10,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Resource)]
pub struct ApiConfig {
    pub host: String,
    pub port: u16,
}

impl Default for ApiConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 8000,
        }
    }
}
