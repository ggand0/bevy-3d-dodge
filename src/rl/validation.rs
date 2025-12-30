//! Shared validation logic for HTTP and gRPC APIs
#![allow(clippy::manual_range_contains)]

use crate::config::{ContinuousActionConfig, ObservationMode};

/// Result type for validation errors
pub type ValidationResult<T> = Result<T, ValidationError>;

#[derive(Debug)]
pub struct ValidationError {
    pub message: String,
}

impl ValidationError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

/// Validate level (1 or 2)
pub fn validate_level(level: i32) -> ValidationResult<()> {
    if level < 1 || level > 2 {
        return Err(ValidationError::new(format!(
            "Invalid level: {}. Must be 1 or 2",
            level
        )));
    }
    Ok(())
}

/// Validate action space type string
pub fn validate_action_space_type(action_space_type: &str) -> ValidationResult<()> {
    let lower = action_space_type.to_lowercase();
    if lower != "discrete" && ContinuousActionConfig::from_str(&lower).is_none() {
        return Err(ValidationError::new(format!(
            "Invalid action_space_type: '{}'. Must be 'discrete' or a continuous variant",
            action_space_type
        )));
    }
    Ok(())
}

/// Validate spawn angle degrees (0 < angle <= 180)
pub fn validate_spawn_angle(angle: f32) -> ValidationResult<()> {
    if angle <= 0.0 || angle > 180.0 {
        return Err(ValidationError::new(format!(
            "Invalid spawn_angle_degrees: {}. Must be between 0 and 180",
            angle
        )));
    }
    Ok(())
}

/// Validate sprint multiplier (0 <= mult <= 10)
pub fn validate_sprint_multiplier(mult: f32) -> ValidationResult<()> {
    if mult < 0.0 || mult > 10.0 {
        return Err(ValidationError::new(format!(
            "Invalid sprint_multiplier: {}. Must be between 0 and 10",
            mult
        )));
    }
    Ok(())
}

/// Validate observation mode string
pub fn validate_observation_mode(obs_mode: &str) -> ValidationResult<()> {
    if ObservationMode::from_str(obs_mode).is_none() {
        return Err(ValidationError::new(format!(
            "Invalid observation_mode: '{}'. Must be 'standard', 'with_thrower', or 'topdown'",
            obs_mode
        )));
    }
    Ok(())
}

/// Validate thrower delay seconds (0 < delay <= 10)
pub fn validate_thrower_delay(delay: f32) -> ValidationResult<()> {
    if delay <= 0.0 || delay > 10.0 {
        return Err(ValidationError::new(format!(
            "Invalid thrower_delay_seconds: {}. Must be between 0 and 10",
            delay
        )));
    }
    Ok(())
}

/// Validate image dimension (32 <= dim <= 512)
pub fn validate_image_dimension(dim: u32, name: &str) -> ValidationResult<()> {
    if dim < 32 || dim > 512 {
        return Err(ValidationError::new(format!(
            "Invalid {}: {}. Must be between 32 and 512",
            name, dim
        )));
    }
    Ok(())
}

// ============================================================================
// Reward Parameter Validation
// ============================================================================

/// Validate collision penalty (must be negative or zero)
pub fn validate_collision_penalty(penalty: f32) -> ValidationResult<()> {
    if penalty > 0.0 {
        return Err(ValidationError::new(format!(
            "Invalid collision_penalty: {}. Must be <= 0 (negative penalty)",
            penalty
        )));
    }
    Ok(())
}

/// Validate survival reward (can be any value, but typically positive)
pub fn validate_survival_reward(reward: f32) -> ValidationResult<()> {
    // No strict validation - allow any value for experimentation
    // Just warn if it's negative (unusual)
    if reward < -100.0 || reward > 100.0 {
        return Err(ValidationError::new(format!(
            "Invalid survival_reward: {}. Must be between -100 and 100",
            reward
        )));
    }
    Ok(())
}

/// Validate dodge bonus threshold (must be positive)
pub fn validate_dodge_bonus_threshold(threshold: f32) -> ValidationResult<()> {
    if threshold <= 0.0 || threshold > 50.0 {
        return Err(ValidationError::new(format!(
            "Invalid dodge_bonus_threshold: {}. Must be between 0 and 50",
            threshold
        )));
    }
    Ok(())
}

/// Validate dodge bonus multiplier (can be any non-negative value)
pub fn validate_dodge_bonus_multiplier(multiplier: f32) -> ValidationResult<()> {
    if multiplier < 0.0 || multiplier > 100.0 {
        return Err(ValidationError::new(format!(
            "Invalid dodge_bonus_multiplier: {}. Must be between 0 and 100",
            multiplier
        )));
    }
    Ok(())
}

// ============================================================================
// Level Parameter Validation
// ============================================================================

/// Validate projectile speed (must be positive)
pub fn validate_projectile_speed(speed: f32) -> ValidationResult<()> {
    if speed <= 0.0 || speed > 50.0 {
        return Err(ValidationError::new(format!(
            "Invalid projectile_speed: {}. Must be between 0 and 50",
            speed
        )));
    }
    Ok(())
}

/// Validate projectile spawn interval (must be positive)
pub fn validate_projectile_spawn_interval(interval: f32) -> ValidationResult<()> {
    if interval <= 0.0 || interval > 60.0 {
        return Err(ValidationError::new(format!(
            "Invalid projectile_spawn_interval: {}. Must be between 0 and 60 seconds",
            interval
        )));
    }
    Ok(())
}

/// Validate max projectiles (must be positive, reasonable limit)
pub fn validate_max_projectiles(max: u32) -> ValidationResult<()> {
    if max == 0 || max > 1000 {
        return Err(ValidationError::new(format!(
            "Invalid max_projectiles: {}. Must be between 1 and 1000",
            max
        )));
    }
    Ok(())
}

/// Validate player speed (must be positive)
pub fn validate_player_speed(speed: f32) -> ValidationResult<()> {
    if speed <= 0.0 || speed > 50.0 {
        return Err(ValidationError::new(format!(
            "Invalid player_speed: {}. Must be between 0 and 50",
            speed
        )));
    }
    Ok(())
}
