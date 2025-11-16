use bevy::prelude::*;
use serde::{Deserialize, Serialize};
use crate::game::player::Velocity as PlayerVelocity;
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

/// Apply RL action to player velocity
pub fn apply_action(
    action: RLAction,
    velocity: &mut PlayerVelocity,
    config: &GameConfig,
) {
    let direction = action.to_direction();
    velocity.0 = direction * config.player_speed;
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
}
