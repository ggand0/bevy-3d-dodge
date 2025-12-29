/// Shared validation logic for HTTP and gRPC APIs

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
