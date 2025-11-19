use bevy::prelude::*;
use serde::{Deserialize, Serialize};
use crate::game::player::{Velocity as PlayerVelocity, PlayerTilt};
use crate::config::GameConfig;

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
/// 5D Box: [vx, vy, pitch, roll, sprint] all in range [-1, 1]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ContinuousAction {
    pub velocity: Vec2,  // (vx, vy) in range [-1, 1]
    pub pitch: f32,      // Forward/backward tilt in range [-1, 1]
    pub roll: f32,       // Left/right tilt in range [-1, 1]
    pub sprint: f32,     // Sprint intensity in range [0, 1] (normalized from [-1, 1])
}

impl ContinuousAction {
    /// Create from array of 5 floats: [vx, vy, pitch, roll, sprint]
    pub fn from_array(arr: [f32; 5]) -> Result<Self, String> {
        // Validate ranges
        for (i, &val) in arr.iter().enumerate() {
            if val < -1.0 || val > 1.0 {
                return Err(format!(
                    "Action component {} has value {} outside valid range [-1, 1]",
                    i, val
                ));
            }
        }

        // Normalize sprint from [-1, 1] to [0, 1] for clearer semantics
        // -1 and 0 both mean no sprint, +1 means full sprint
        let sprint_normalized = (arr[4] + 1.0) / 2.0;

        Ok(Self {
            velocity: Vec2::new(arr[0], arr[1]),
            pitch: arr[2],
            roll: arr[3],
            sprint: sprint_normalized.clamp(0.0, 1.0),
        })
    }

    /// Convert to array format
    pub fn to_array(&self) -> [f32; 5] {
        // Convert sprint back from [0, 1] to [-1, 1]
        let sprint_denormalized = self.sprint * 2.0 - 1.0;
        [self.velocity.x, self.velocity.y, self.pitch, self.roll, sprint_denormalized]
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

/// Apply continuous action to player velocity and tilt
pub fn apply_continuous_action(
    action: ContinuousAction,
    velocity: &mut PlayerVelocity,
    tilt: &mut PlayerTilt,
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
    fn test_continuous_action_from_array() {
        let action = ContinuousAction::from_array([0.5, -0.3, 0.1, -0.7, 0.5]).unwrap();
        assert_eq!(action.velocity, Vec2::new(0.5, -0.3));
        assert_eq!(action.pitch, 0.1);
        assert_eq!(action.roll, -0.7);
        // sprint 0.5 normalized from [-1, 1] to [0, 1] = (0.5 + 1.0) / 2.0 = 0.75
        assert!((action.sprint - 0.75).abs() < 0.001);
    }

    #[test]
    fn test_continuous_action_validation() {
        // Valid actions
        assert!(ContinuousAction::from_array([1.0, 1.0, 1.0, 1.0, 1.0]).is_ok());
        assert!(ContinuousAction::from_array([-1.0, -1.0, -1.0, -1.0, -1.0]).is_ok());
        assert!(ContinuousAction::from_array([0.0, 0.0, 0.0, 0.0, 0.0]).is_ok());

        // Invalid actions (out of range)
        assert!(ContinuousAction::from_array([1.1, 0.0, 0.0, 0.0, 0.0]).is_err());
        assert!(ContinuousAction::from_array([0.0, -1.1, 0.0, 0.0, 0.0]).is_err());
        assert!(ContinuousAction::from_array([0.0, 0.0, 2.0, 0.0, 0.0]).is_err());
        assert!(ContinuousAction::from_array([0.0, 0.0, 0.0, -1.5, 0.0]).is_err());
        assert!(ContinuousAction::from_array([0.0, 0.0, 0.0, 0.0, 1.5]).is_err());
    }

    #[test]
    fn test_continuous_action_round_trip() {
        let original = [0.5, -0.3, 0.1, -0.7, 0.8];
        let action = ContinuousAction::from_array(original).unwrap();
        let converted = action.to_array();
        // Check each component within floating point tolerance
        for i in 0..5 {
            assert!((original[i] - converted[i]).abs() < 0.001);
        }
    }
}
