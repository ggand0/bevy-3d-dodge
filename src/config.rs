use bevy::prelude::*;
use serde::{Deserialize, Serialize};

/// Game difficulty levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Resource)]
pub enum Level {
    Level1,  // Original difficulty - for model evaluation and baseline
    Level2,  // Harder difficulty - faster projectiles, shorter spawn interval
}

impl Level {
    /// Get the next level (cycles back to Level1 after Level2)
    pub fn next(self) -> Self {
        match self {
            Level::Level1 => Level::Level2,
            Level::Level2 => Level::Level1,
        }
    }

    /// Get level number for display
    pub fn number(self) -> u8 {
        match self {
            Level::Level1 => 1,
            Level::Level2 => 2,
        }
    }

    /// Get level name for display
    pub fn name(self) -> &'static str {
        match self {
            Level::Level1 => "Level 1 (Baseline)",
            Level::Level2 => "Level 2 (Hard)",
        }
    }
}

impl Default for Level {
    fn default() -> Self {
        Level::Level1
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Resource)]
pub struct GameConfig {
    pub player_speed: f32,
    pub player_start_height: f32,
    pub projectile_speed: f32,
    pub projectile_spawn_interval: f32,
    pub projectile_spawn_distance: f32,
    pub max_projectiles: usize,
}

impl GameConfig {
    /// Create config for a specific level
    pub fn for_level(level: Level) -> Self {
        match level {
            Level::Level1 => Self::level1(),
            Level::Level2 => Self::level2(),
        }
    }

    /// Level 1: Original baseline difficulty
    /// This is the original configuration used for training previous models.
    /// Keep this unchanged for model evaluation and comparison.
    fn level1() -> Self {
        Self {
            player_speed: 5.0,
            player_start_height: 1.0,
            projectile_speed: 3.0,
            projectile_spawn_interval: 2.0,
            projectile_spawn_distance: 20.0,
            max_projectiles: 10,
        }
    }

    /// Level 2: Hard difficulty
    /// Faster projectiles, more frequent spawning, and more projectiles
    fn level2() -> Self {
        Self {
            player_speed: 5.0,               // Keep player speed same
            player_start_height: 1.0,
            projectile_speed: 4.5,            // 50% faster projectiles
            projectile_spawn_interval: 1.2,   // 40% faster spawning
            projectile_spawn_distance: 20.0,
            max_projectiles: 15,              // 50% more projectiles
        }
    }
}

impl Default for GameConfig {
    fn default() -> Self {
        Self::level1()
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
