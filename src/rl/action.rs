use bevy::prelude::*;
use serde::{Deserialize, Serialize};
use crate::game::player::{Velocity as PlayerVelocity, PlayerTilt, VerticalVelocity, OnGround};
use crate::config::{GameConfig, ContinuousActionConfig};

/// Discrete action space for RL agent
/// 5 actions total: NOOP, UP, DOWN, LEFT, RIGHT
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RLAction {
    Noop = 0,
    Up = 1,      // +Y direction
    Down = 2,    // -Y direction
    Left = 3,    // -X direction
    Right = 4,   // +X direction
}

impl RLAction {
    /// Parse action from integer index
    pub fn from_index(index: usize) -> Result<Self, String> {
        match index {
            0 => Ok(RLAction::Noop),
            1 => Ok(RLAction::Up),
            2 => Ok(RLAction::Down),
            3 => Ok(RLAction::Left),
            4 => Ok(RLAction::Right),
            _ => Err(format!("Invalid action index: {}. Must be in range [0, 5)", index)),
        }
    }

    /// Convert action to movement direction vector
    pub fn to_direction(&self) -> Vec2 {
        match self {
            RLAction::Noop => Vec2::ZERO,
            RLAction::Up => Vec2::new(0.0, 1.0),
            RLAction::Down => Vec2::new(0.0, -1.0),
            RLAction::Left => Vec2::new(-1.0, 0.0),
            RLAction::Right => Vec2::new(1.0, 0.0),
        }
    }
}

/// Continuous action space for RL agent
/// Variable dimensions based on ContinuousActionConfig
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ContinuousAction {
    pub velocity: Vec2,  // (vx, vy) in range [-1, 1]
    pub pitch: f32,      // Forward/backward tilt in range [-1, 1]
    pub roll: f32,       // Left/right tilt in range [-1, 1]
    pub sprint: f32,     // Sprint intensity in range [0, 1] (normalized from [-1, 1])
    pub jump: f32,       // Jump trigger in range [0, 1] (normalized from [-1, 1])
}

impl ContinuousAction {
    /// Create from variable-length array based on config
    pub fn from_array(arr: &[f32], config: ContinuousActionConfig) -> Result<Self, String> {
        // Validate array length matches config
        if arr.len() != config.dimension() {
            return Err(format!(
                "Expected {} components for {}, got {}",
                config.dimension(),
                config.name(),
                arr.len()
            ));
        }

        // Validate all components are in [-1, 1]
        for (i, &val) in arr.iter().enumerate() {
            if val < -1.0 || val > 1.0 {
                return Err(format!(
                    "Component {} ({}) has value {} outside valid range [-1, 1]",
                    i, config.component_names()[i], val
                ));
            }
        }

        // Extract components based on configuration
        match config {
            ContinuousActionConfig::Basic3D => {
                // [vx, vy, sprint]
                Ok(Self {
                    velocity: Vec2::new(arr[0], arr[1]),
                    pitch: 0.0,
                    roll: 0.0,
                    sprint: (arr[2] + 1.0) / 2.0,  // Normalize to [0, 1]
                    jump: 0.0,
                })
            }
            ContinuousActionConfig::BasicWithJump4D => {
                // [vx, vy, sprint, jump]
                Ok(Self {
                    velocity: Vec2::new(arr[0], arr[1]),
                    pitch: 0.0,
                    roll: 0.0,
                    sprint: (arr[2] + 1.0) / 2.0,
                    jump: (arr[3] + 1.0) / 2.0,
                })
            }
            ContinuousActionConfig::Tilt5D => {
                // [vx, vy, pitch, roll, sprint] - current implementation
                Ok(Self {
                    velocity: Vec2::new(arr[0], arr[1]),
                    pitch: arr[2],
                    roll: arr[3],
                    sprint: (arr[4] + 1.0) / 2.0,
                    jump: 0.0,
                })
            }
            ContinuousActionConfig::Full6D => {
                // [vx, vy, jump, pitch, roll, sprint]
                Ok(Self {
                    velocity: Vec2::new(arr[0], arr[1]),
                    pitch: arr[3],
                    roll: arr[4],
                    sprint: (arr[5] + 1.0) / 2.0,
                    jump: (arr[2] + 1.0) / 2.0,
                })
            }
        }
    }
}

/// Unified action enum that can hold either discrete or continuous actions
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Action {
    Discrete(RLAction),
    Continuous(ContinuousAction),
}

/// Apply discrete RL action to player velocity
pub fn apply_action(
    action: RLAction,
    velocity: &mut PlayerVelocity,
    config: &GameConfig,
) {
    let direction = action.to_direction();
    velocity.0 = direction * config.player_speed;
}

/// Apply continuous action to player velocity, tilt, and jump
pub fn apply_continuous_action(
    action: ContinuousAction,
    velocity: &mut PlayerVelocity,
    tilt: &mut PlayerTilt,
    vertical_velocity: &mut VerticalVelocity,
    on_ground: &OnGround,
    config: &GameConfig,
) {
    // Calculate effective speed with sprint multiplier
    // sprint is in [0, 1], so: speed = base_speed * (1 + sprint * multiplier)
    let speed_multiplier = 1.0 + action.sprint * config.sprint_multiplier;
    let effective_speed = config.player_speed * speed_multiplier;

    // Apply velocity (normalized action * effective_speed)
    velocity.0 = action.velocity * effective_speed;

    // Apply tilt with constraints (max ±30° = ±0.523 radians)
    const MAX_TILT: f32 = 0.523; // 30 degrees in radians
    tilt.pitch = action.pitch * MAX_TILT;
    tilt.roll = action.roll * MAX_TILT;

    // Apply jump if jump > 0.5 threshold and on ground
    const JUMP_THRESHOLD: f32 = 0.5;
    const JUMP_VELOCITY: f32 = 7.0;
    if action.jump > JUMP_THRESHOLD && on_ground.0 {
        vertical_velocity.0 = JUMP_VELOCITY;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_action_from_index() {
        assert_eq!(RLAction::from_index(0).unwrap(), RLAction::Noop);
        assert_eq!(RLAction::from_index(1).unwrap(), RLAction::Up);
        assert_eq!(RLAction::from_index(2).unwrap(), RLAction::Down);
        assert_eq!(RLAction::from_index(3).unwrap(), RLAction::Left);
        assert_eq!(RLAction::from_index(4).unwrap(), RLAction::Right);
        assert!(RLAction::from_index(5).is_err());
    }

    #[test]
    fn test_action_to_direction() {
        assert_eq!(RLAction::Noop.to_direction(), Vec2::ZERO);
        assert_eq!(RLAction::Up.to_direction(), Vec2::new(0.0, 1.0));
        assert_eq!(RLAction::Down.to_direction(), Vec2::new(0.0, -1.0));
        assert_eq!(RLAction::Left.to_direction(), Vec2::new(-1.0, 0.0));
        assert_eq!(RLAction::Right.to_direction(), Vec2::new(1.0, 0.0));
    }

    #[test]
    fn test_continuous_action_tilt5d() {
        // Test Tilt5D: [vx, vy, pitch, roll, sprint]
        let action = ContinuousAction::from_array(&[0.5, -0.3, 0.1, -0.7, 0.5], ContinuousActionConfig::Tilt5D).unwrap();
        assert_eq!(action.velocity, Vec2::new(0.5, -0.3));
        assert_eq!(action.pitch, 0.1);
        assert_eq!(action.roll, -0.7);
        // sprint 0.5 normalized from [-1, 1] to [0, 1] = (0.5 + 1.0) / 2.0 = 0.75
        assert!((action.sprint - 0.75).abs() < 0.001);
        assert_eq!(action.jump, 0.0);
    }

    #[test]
    fn test_continuous_action_basic3d() {
        // Test Basic3D: [vx, vy, sprint]
        let action = ContinuousAction::from_array(&[0.5, -0.3, 1.0], ContinuousActionConfig::Basic3D).unwrap();
        assert_eq!(action.velocity, Vec2::new(0.5, -0.3));
        assert_eq!(action.pitch, 0.0);
        assert_eq!(action.roll, 0.0);
        assert!((action.sprint - 1.0).abs() < 0.001); // (1.0 + 1.0) / 2.0 = 1.0
        assert_eq!(action.jump, 0.0);
    }

    #[test]
    fn test_continuous_action_basic4d_jump() {
        // Test BasicWithJump4D: [vx, vy, sprint, jump]
        let action = ContinuousAction::from_array(&[0.5, -0.3, 0.0, 0.6], ContinuousActionConfig::BasicWithJump4D).unwrap();
        assert_eq!(action.velocity, Vec2::new(0.5, -0.3));
        assert_eq!(action.pitch, 0.0);
        assert_eq!(action.roll, 0.0);
        assert!((action.sprint - 0.5).abs() < 0.001); // (0.0 + 1.0) / 2.0 = 0.5
        assert!((action.jump - 0.8).abs() < 0.001); // (0.6 + 1.0) / 2.0 = 0.8
    }

    #[test]
    fn test_continuous_action_full6d() {
        // Test Full6D: [vx, vy, jump, pitch, roll, sprint]
        let action = ContinuousAction::from_array(&[0.5, -0.3, 0.2, 0.1, -0.7, 0.5], ContinuousActionConfig::Full6D).unwrap();
        assert_eq!(action.velocity, Vec2::new(0.5, -0.3));
        assert_eq!(action.pitch, 0.1);
        assert_eq!(action.roll, -0.7);
        assert!((action.sprint - 0.75).abs() < 0.001); // (0.5 + 1.0) / 2.0 = 0.75
        assert!((action.jump - 0.6).abs() < 0.001); // (0.2 + 1.0) / 2.0 = 0.6
    }

    #[test]
    fn test_continuous_action_validation() {
        // Valid actions for each config
        assert!(ContinuousAction::from_array(&[1.0, 1.0, 1.0], ContinuousActionConfig::Basic3D).is_ok());
        assert!(ContinuousAction::from_array(&[1.0, 1.0, 1.0, 1.0], ContinuousActionConfig::BasicWithJump4D).is_ok());
        assert!(ContinuousAction::from_array(&[1.0, 1.0, 1.0, 1.0, 1.0], ContinuousActionConfig::Tilt5D).is_ok());
        assert!(ContinuousAction::from_array(&[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], ContinuousActionConfig::Full6D).is_ok());

        // Invalid: wrong array length
        assert!(ContinuousAction::from_array(&[1.0, 1.0], ContinuousActionConfig::Basic3D).is_err());
        assert!(ContinuousAction::from_array(&[1.0, 1.0, 1.0], ContinuousActionConfig::Tilt5D).is_err());

        // Invalid: out of range
        assert!(ContinuousAction::from_array(&[1.1, 0.0, 0.0], ContinuousActionConfig::Basic3D).is_err());
        assert!(ContinuousAction::from_array(&[0.0, -1.1, 0.0, 0.0, 0.0], ContinuousActionConfig::Tilt5D).is_err());
    }
}
