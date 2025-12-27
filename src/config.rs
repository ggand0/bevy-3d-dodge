use bevy::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;

/// Resource wrapper for shared game config with API server
#[derive(Clone, Resource)]
pub struct SharedGameConfig(pub Arc<Mutex<GameConfig>>);

/// Continuous action space configurations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContinuousActionConfig {
    /// 3D: [vx, vy, sprint] - Basic movement with sprint
    Basic3D,
    /// 4D: [vx, vy, sprint, jump] - Basic + jump control
    BasicWithJump4D,
    /// 5D: [vx, vy, pitch, roll, sprint] - Movement with tilt and sprint (current default)
    Tilt5D,
    /// 6D: [vx, vy, jump, pitch, roll, sprint] - Full control (conditional jump)
    Full6D,
}

impl ContinuousActionConfig {
    /// Get dimension of action space
    pub fn dimension(&self) -> usize {
        match self {
            Self::Basic3D => 3,
            Self::BasicWithJump4D => 4,
            Self::Tilt5D => 5,
            Self::Full6D => 6,
        }
    }

    /// Get human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            Self::Basic3D => "3D Basic (vx, vy, sprint)",
            Self::BasicWithJump4D => "4D with Jump (vx, vy, sprint, jump)",
            Self::Tilt5D => "5D with Tilt (vx, vy, pitch, roll, sprint)",
            Self::Full6D => "6D Full (vx, vy, jump, pitch, roll, sprint)",
        }
    }

    /// Get component names for debugging
    pub fn component_names(&self) -> Vec<&'static str> {
        match self {
            Self::Basic3D => vec!["vx", "vy", "sprint"],
            Self::BasicWithJump4D => vec!["vx", "vy", "sprint", "jump"],
            Self::Tilt5D => vec!["vx", "vy", "pitch", "roll", "sprint"],
            Self::Full6D => vec!["vx", "vy", "jump", "pitch", "roll", "sprint"],
        }
    }

    /// Parse from string (for API)
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "basic3d" | "basic_3d" | "3d" => Some(Self::Basic3D),
            "basic4d" | "basic_4d" | "jump4d" | "4d_jump" => Some(Self::BasicWithJump4D),
            "tilt5d" | "tilt_5d" | "5d" => Some(Self::Tilt5D),
            "full6d" | "full_6d" | "6d" => Some(Self::Full6D),
            _ => None,
        }
    }

    /// Convert to string for API
    #[allow(dead_code)]
    pub fn to_string(&self) -> &'static str {
        match self {
            Self::Basic3D => "basic_3d",
            Self::BasicWithJump4D => "basic_4d_jump",
            Self::Tilt5D => "tilt_5d",
            Self::Full6D => "full_6d",
        }
    }
}

impl Default for ContinuousActionConfig {
    fn default() -> Self {
        Self::Tilt5D  // Current default (matches existing 5D implementation)
    }
}

/// Action space types for RL training
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActionSpaceType {
    /// Discrete action space: 5 actions (NOOP, Up, Down, Left, Right)
    Discrete,
    /// Continuous action space with configurable dimensions
    Continuous(ContinuousActionConfig),
}

impl Default for ActionSpaceType {
    fn default() -> Self {
        ActionSpaceType::Discrete
    }
}

/// Observation space modes for RL training
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ObservationMode {
    #[default]
    /// Standard 65-dim observation (backward compatible with existing models)
    Standard,
    /// Extended 69-dim observation with thrower indicator info
    WithThrowerIndicator,
    /// Top-down rendered image (256x256 RGB)
    TopDownImage,
}

/// Image observation configuration - default values (256x256 for accurate projection)
pub const IMAGE_OBS_WIDTH: u32 = 256;
pub const IMAGE_OBS_HEIGHT: u32 = 256;
pub const IMAGE_OBS_CHANNELS: u32 = 3;  // RGB

impl ObservationMode {
    /// Parse from string (for API)
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "standard" | "default" => Some(Self::Standard),
            "with_thrower" | "thrower" | "with_thrower_indicator" => Some(Self::WithThrowerIndicator),
            "topdown" | "topdown_image" | "image" => Some(Self::TopDownImage),
            _ => None,
        }
    }

    /// Check if this mode uses image observations
    pub fn is_image_mode(&self) -> bool {
        matches!(self, Self::TopDownImage)
    }

    /// Get observation size for vector modes (returns 0 for image mode)
    pub fn observation_size(&self) -> usize {
        match self {
            Self::Standard => 65,
            Self::WithThrowerIndicator => 69,
            Self::TopDownImage => 0,  // Image mode doesn't use vector size
        }
    }

    /// Get image dimensions for image mode (width, height, channels)
    pub fn image_shape(&self) -> Option<(u32, u32, u32)> {
        match self {
            Self::TopDownImage => Some((IMAGE_OBS_WIDTH, IMAGE_OBS_HEIGHT, IMAGE_OBS_CHANNELS)),
            _ => None,
        }
    }
}

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
    #[allow(dead_code)]
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
    pub random_spawn_position: bool,  // If true, spawn from random positions on a circle
    pub action_space_type: ActionSpaceType,  // Discrete or Continuous action space
    pub sprint_multiplier: f32,  // Speed multiplier when sprinting (e.g., 1.0 = 2x speed, 2.0 = 3x speed)
    pub spawn_angle_degrees: f32,  // Half-angle for spawn fan in degrees (e.g., 60 = ±60° = 120° total)
    pub observation_mode: ObservationMode,  // Standard (65-dim) or WithThrowerIndicator (69-dim)
    pub thrower_delay_seconds: f32,  // Delay before thrower indicator spawns projectile
    pub image_obs_width: u32,   // Image observation width (default 84, Atari-standard)
    pub image_obs_height: u32,  // Image observation height (default 84, Atari-standard)
    pub image_grayscale: bool,  // If true, use grayscale (1 channel) instead of RGB (3 channels)
}

impl GameConfig {
    /// Get the number of image channels based on grayscale setting
    pub fn image_channels(&self) -> u32 {
        if self.image_grayscale { 1 } else { 3 }
    }

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
            random_spawn_position: false,     // Spawn from fixed +Y side
            action_space_type: ActionSpaceType::Continuous(ContinuousActionConfig::default()),
            sprint_multiplier: 2.0,  // Sprint gives 3.0x speed (5.0 -> 15.0)
            spawn_angle_degrees: 60.0,  // ±60° = 120° total fan (not used when random_spawn_position=false)
            observation_mode: ObservationMode::default(),  // Standard 65-dim
            thrower_delay_seconds: 0.5,  // 0.5 second warning before throw
            image_obs_width: IMAGE_OBS_WIDTH,
            image_obs_height: IMAGE_OBS_HEIGHT,
            image_grayscale: false,  // RGB by default
        }
    }

    /// Level 2: Hard difficulty
    /// Faster projectiles, more frequent spawning, more projectiles, and random spawn positions
    fn level2() -> Self {
        Self {
            player_speed: 5.0,               // Keep player speed same
            player_start_height: 1.0,
            projectile_speed: 4.5,            // 50% faster projectiles
            projectile_spawn_interval: 0.5,   // 4x faster spawning (was 2.0s, now 0.5s)
            projectile_spawn_distance: 20.0,
            max_projectiles: 25,              // 2.5x more projectiles (was 10, now 25)
            random_spawn_position: true,      // Spawn from random positions in spawn_angle fan
            action_space_type: ActionSpaceType::Continuous(ContinuousActionConfig::default()),
            sprint_multiplier: 2.0,  // Sprint gives 3.0x speed (5.0 -> 15.0)
            spawn_angle_degrees: 60.0,  // ±60° = 120° total fan
            observation_mode: ObservationMode::default(),  // Standard 65-dim
            thrower_delay_seconds: 0.5,  // Must equal spawn_interval for same arrival rate
            image_obs_width: IMAGE_OBS_WIDTH,
            image_obs_height: IMAGE_OBS_HEIGHT,
            image_grayscale: false,  // RGB by default
        }
    }
}

impl Default for GameConfig {
    fn default() -> Self {
        Self::level1()
    }
}

#[allow(dead_code)]
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
