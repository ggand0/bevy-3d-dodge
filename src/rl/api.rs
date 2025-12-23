use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex, RwLock};
use tower_http::cors::{Any, CorsLayer};

use crate::config::{IMAGE_OBS_WIDTH, IMAGE_OBS_HEIGHT, IMAGE_OBS_CHANNELS, ObservationMode};
use crate::rl::observation::OBSERVATION_SIZE;

/// Shared state between Axum server and Bevy game loop
#[derive(Clone, bevy::prelude::Resource)]
pub struct SharedEnvState {
    pub observation: Arc<RwLock<Vec<f32>>>,
    pub image_observation: Arc<RwLock<Vec<u8>>>,  // RGB image bytes for TopDownImage mode
    pub reward: Arc<RwLock<f32>>,
    pub done: Arc<RwLock<bool>>,
    pub truncated: Arc<RwLock<bool>>,
    pub info: Arc<RwLock<std::collections::HashMap<String, serde_json::Value>>>,
}

// Note: These are tokio::sync::RwLock, not std::sync::RwLock

impl Default for SharedEnvState {
    fn default() -> Self {
        Self {
            observation: Arc::new(RwLock::new(vec![0.0; OBSERVATION_SIZE])),
            image_observation: Arc::new(RwLock::new(vec![0u8; (IMAGE_OBS_WIDTH * IMAGE_OBS_HEIGHT * IMAGE_OBS_CHANNELS) as usize])),
            reward: Arc::new(RwLock::new(0.0)),
            done: Arc::new(RwLock::new(false)),
            truncated: Arc::new(RwLock::new(false)),
            info: Arc::new(RwLock::new(std::collections::HashMap::new())),
        }
    }
}

/// Commands sent from API to game loop
#[derive(Debug, Clone)]
pub enum EnvCommand {
    Reset,
    StepDiscrete { action: usize },
    StepContinuous { action: Vec<f32>, config: crate::config::ContinuousActionConfig },
    StartTraining,
    EndTraining,
    SetLevel { level: u8 },
    Configure {
        level: Option<u8>,
        action_space_type: Option<String>,
        sprint_multiplier: Option<f32>,
        spawn_angle_degrees: Option<f32>,
        observation_mode: Option<String>,
        thrower_delay_seconds: Option<f32>,
    },
}

/// API server state
#[derive(Clone)]
struct ApiState {
    shared_state: SharedEnvState,
    command_tx: mpsc::UnboundedSender<EnvCommand>,
    game_config: Arc<Mutex<crate::config::GameConfig>>,
}

// ============================================================================
// Request/Response Types
// ============================================================================

#[derive(Deserialize)]
struct StepRequest {
    action: serde_json::Value,
}

#[derive(Deserialize)]
struct SetLevelRequest {
    level: u8,
}

#[derive(Deserialize)]
struct ConfigureRequest {
    level: Option<u8>,
    action_space_type: Option<String>,
    sprint_multiplier: Option<f32>,  // Speed multiplier (e.g., 2.0 = 3x speed at full sprint)
    spawn_angle_degrees: Option<f32>,  // Half-angle for spawn fan (e.g., 30 = ±30° = 60° total)
    observation_mode: Option<String>,  // "standard" or "with_thrower"
    thrower_delay_seconds: Option<f32>,  // Delay before thrower indicator spawns projectile
}

#[derive(Serialize)]
struct ResetResponse {
    observation: Vec<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    image_observation: Option<String>,  // Base64-encoded RGB image for TopDownImage mode
    info: std::collections::HashMap<String, serde_json::Value>,
}

#[derive(Serialize)]
struct StepResponse {
    observation: Vec<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    image_observation: Option<String>,  // Base64-encoded RGB image for TopDownImage mode
    reward: f32,
    done: bool,
    truncated: bool,
    info: std::collections::HashMap<String, serde_json::Value>,
}

#[derive(Serialize)]
struct ObservationSpaceResponse {
    shape: Vec<usize>,
    dtype: String,
    low: f32,
    high: f32,
}

#[derive(Serialize)]
#[serde(untagged)]
enum ActionSpaceResponse {
    Discrete {
        r#type: String,
        n: usize,
    },
    Box {
        r#type: String,
        shape: Vec<usize>,
        low: f32,
        high: f32,
    },
}

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

// ============================================================================
// API Handlers
// ============================================================================

async fn reset_handler(
    State(state): State<ApiState>,
) -> Result<Json<ResetResponse>, AppError> {
    // Send reset command to game loop
    state
        .command_tx
        .send(EnvCommand::Reset)
        .map_err(|_| AppError::InternalError("Failed to send reset command".to_string()))?;

    // Wait a bit for game loop to process reset (simple synchronization)
    // Increased from 50ms to 100ms to give more time for headless mode
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    // Check observation mode
    let game_config = state.game_config.lock().await;
    let obs_mode = game_config.observation_mode;
    drop(game_config);

    // Read observation and info from shared state (using read lock for concurrent access)
    let observation = state
        .shared_state
        .observation
        .read()
        .await
        .clone();

    // Get image observation if in TopDownImage mode
    let image_observation = if obs_mode == ObservationMode::TopDownImage {
        let image_bytes = state
            .shared_state
            .image_observation
            .read()
            .await
            .clone();

        // Pre-allocate base64 buffer to avoid allocation during encoding
        // Base64 expands data by ~4/3, so allocate accordingly
        let mut base64_string = String::with_capacity((image_bytes.len() * 4 / 3) + 4);
        BASE64.encode_string(&image_bytes, &mut base64_string);
        Some(base64_string)
    } else {
        None
    };

    let info = state
        .shared_state
        .info
        .read()
        .await
        .clone();

    Ok(Json(ResetResponse { observation, image_observation, info }))
}

async fn step_handler(
    State(state): State<ApiState>,
    Json(request): Json<StepRequest>,
) -> Result<Json<StepResponse>, AppError> {
    // Get current action space type and observation mode from config
    let game_config = state.game_config.lock().await;
    let action_space_type = game_config.action_space_type;
    let obs_mode = game_config.observation_mode;
    drop(game_config); // Release lock

    // Parse and validate action based on action space type
    let command = match action_space_type {
        crate::config::ActionSpaceType::Discrete => {
            // Expect integer action
            let action = request.action.as_u64()
                .ok_or_else(|| AppError::InvalidAction("Action must be an integer for discrete action space".to_string()))?
                as usize;

            if action >= 5 {
                return Err(AppError::InvalidAction(format!(
                    "Invalid action: {}. Must be in range [0, 5)",
                    action
                )));
            }

            EnvCommand::StepDiscrete { action }
        }
        crate::config::ActionSpaceType::Continuous(cont_config) => {
            // Expect array of floats with length matching config dimension
            let action_array = request.action.as_array()
                .ok_or_else(|| AppError::InvalidAction("Action must be an array for continuous action space".to_string()))?;

            let expected_dim = cont_config.dimension();
            if action_array.len() != expected_dim {
                return Err(AppError::InvalidAction(format!(
                    "Continuous action must have {} components for {}, got {}",
                    expected_dim,
                    cont_config.name(),
                    action_array.len()
                )));
            }

            let mut action = Vec::with_capacity(expected_dim);
            for (i, val) in action_array.iter().enumerate() {
                let f = val.as_f64()
                    .ok_or_else(|| AppError::InvalidAction(format!("Action component {} must be a number", i)))?
                    as f32;

                if f < -1.0 || f > 1.0 {
                    return Err(AppError::InvalidAction(format!(
                        "Component {} ({}) has value {} outside valid range [-1, 1]",
                        i, cont_config.component_names()[i], f
                    )));
                }

                action.push(f);
            }

            EnvCommand::StepContinuous { action, config: cont_config }
        }
    };

    // Send step command to game loop
    state
        .command_tx
        .send(command)
        .map_err(|_| AppError::InternalError("Failed to send step command".to_string()))?;

    // Wait a bit for game loop to process step (simple synchronization)
    tokio::time::sleep(tokio::time::Duration::from_millis(16)).await; // ~60 FPS

    // Read state from shared state (using read locks for concurrent access)
    let observation = state
        .shared_state
        .observation
        .read()
        .await
        .clone();

    let reward = *state
        .shared_state
        .reward
        .read()
        .await;

    let done = *state
        .shared_state
        .done
        .read()
        .await;

    let truncated = *state
        .shared_state
        .truncated
        .read()
        .await;

    let info = state
        .shared_state
        .info
        .read()
        .await
        .clone();

    // Get image observation if in TopDownImage mode
    let image_observation = if obs_mode == ObservationMode::TopDownImage {
        let image_bytes = state
            .shared_state
            .image_observation
            .read()
            .await
            .clone();

        // Pre-allocate base64 buffer to avoid allocation during encoding
        let mut base64_string = String::with_capacity((image_bytes.len() * 4 / 3) + 4);
        BASE64.encode_string(&image_bytes, &mut base64_string);
        Some(base64_string)
    } else {
        None
    };

    Ok(Json(StepResponse {
        observation,
        image_observation,
        reward,
        done,
        truncated,
        info,
    }))
}

async fn observation_space_handler(
    State(state): State<ApiState>,
) -> Json<ObservationSpaceResponse> {
    let game_config = state.game_config.lock().await;
    let obs_mode = game_config.observation_mode;

    // Return appropriate shape based on observation mode
    if let Some((width, height, channels)) = obs_mode.image_shape() {
        // Image observation: (height, width, channels) in HWC format
        Json(ObservationSpaceResponse {
            shape: vec![height as usize, width as usize, channels as usize],
            dtype: "uint8".to_string(),
            low: 0.0,
            high: 255.0,
        })
    } else {
        // Vector observation
        let obs_size = obs_mode.observation_size();
        Json(ObservationSpaceResponse {
            shape: vec![obs_size],
            dtype: "float32".to_string(),
            low: -100.0,
            high: 100.0,
        })
    }
}

async fn action_space_handler(
    State(state): State<ApiState>,
) -> Result<Json<ActionSpaceResponse>, AppError> {
    let game_config = state.game_config.lock().await;

    let response = match game_config.action_space_type {
        crate::config::ActionSpaceType::Discrete => ActionSpaceResponse::Discrete {
            r#type: "Discrete".to_string(),
            n: 5,
        },
        crate::config::ActionSpaceType::Continuous(cont_config) => ActionSpaceResponse::Box {
            r#type: "Box".to_string(),
            shape: vec![cont_config.dimension()],  // Dynamic dimension based on config
            low: -1.0,
            high: 1.0,
        },
    };

    Ok(Json(response))
}

async fn start_training_handler(
    State(state): State<ApiState>,
) -> Result<StatusCode, AppError> {
    state
        .command_tx
        .send(EnvCommand::StartTraining)
        .map_err(|_| AppError::InternalError("Failed to send start training command".to_string()))?;

    Ok(StatusCode::OK)
}

async fn end_training_handler(
    State(state): State<ApiState>,
) -> Result<StatusCode, AppError> {
    state
        .command_tx
        .send(EnvCommand::EndTraining)
        .map_err(|_| AppError::InternalError("Failed to send end training command".to_string()))?;

    Ok(StatusCode::OK)
}

async fn set_level_handler(
    State(state): State<ApiState>,
    Json(payload): Json<SetLevelRequest>,
) -> Result<StatusCode, AppError> {
    // Validate level number (1 or 2)
    if payload.level < 1 || payload.level > 2 {
        return Err(AppError::InvalidAction(format!("Invalid level: {}. Must be 1 or 2", payload.level)));
    }

    state
        .command_tx
        .send(EnvCommand::SetLevel { level: payload.level })
        .map_err(|_| AppError::InternalError("Failed to send set level command".to_string()))?;

    Ok(StatusCode::OK)
}

async fn configure_handler(
    State(state): State<ApiState>,
    Json(payload): Json<ConfigureRequest>,
) -> Result<StatusCode, AppError> {
    // Validate level if provided
    if let Some(level) = payload.level {
        if level < 1 || level > 2 {
            return Err(AppError::InvalidAction(format!("Invalid level: {}. Must be 1 or 2", level)));
        }
    }

    // Validate action_space_type if provided
    if let Some(ref action_space_type) = payload.action_space_type {
        let action_space_lower = action_space_type.to_lowercase();

        // Check if it's discrete or a valid continuous config
        if action_space_lower != "discrete" &&
           crate::config::ContinuousActionConfig::from_str(&action_space_lower).is_none() {
            return Err(AppError::InvalidAction(format!(
                "Invalid action_space_type: '{}'. Must be 'discrete' or a continuous variant (basic_3d, basic_4d_jump, tilt_5d, full_6d)",
                action_space_type
            )));
        }
    }

    // Validate spawn_angle_degrees if provided
    if let Some(angle) = payload.spawn_angle_degrees {
        if angle <= 0.0 || angle > 180.0 {
            return Err(AppError::InvalidAction(format!(
                "Invalid spawn_angle_degrees: {}. Must be between 0 and 180",
                angle
            )));
        }
    }

    // Validate sprint_multiplier if provided
    if let Some(mult) = payload.sprint_multiplier {
        if mult < 0.0 || mult > 10.0 {
            return Err(AppError::InvalidAction(format!(
                "Invalid sprint_multiplier: {}. Must be between 0 and 10",
                mult
            )));
        }
    }

    // Validate observation_mode if provided
    if let Some(ref obs_mode) = payload.observation_mode {
        if crate::config::ObservationMode::from_str(obs_mode).is_none() {
            return Err(AppError::InvalidAction(format!(
                "Invalid observation_mode: '{}'. Must be 'standard', 'with_thrower', or 'topdown'",
                obs_mode
            )));
        }
    }

    // Validate thrower_delay_seconds if provided
    if let Some(delay) = payload.thrower_delay_seconds {
        if delay <= 0.0 || delay > 10.0 {
            return Err(AppError::InvalidAction(format!(
                "Invalid thrower_delay_seconds: {}. Must be between 0 and 10",
                delay
            )));
        }
    }

    state
        .command_tx
        .send(EnvCommand::Configure {
            level: payload.level,
            action_space_type: payload.action_space_type,
            sprint_multiplier: payload.sprint_multiplier,
            spawn_angle_degrees: payload.spawn_angle_degrees,
            observation_mode: payload.observation_mode,
            thrower_delay_seconds: payload.thrower_delay_seconds,
        })
        .map_err(|_| AppError::InternalError("Failed to send configure command".to_string()))?;

    Ok(StatusCode::OK)
}

// ============================================================================
// Error Handling
// ============================================================================

enum AppError {
    InvalidAction(String),
    InternalError(String),
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, error_message) = match self {
            AppError::InvalidAction(msg) => (StatusCode::BAD_REQUEST, msg),
            AppError::InternalError(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg),
        };

        let body = Json(ErrorResponse {
            error: error_message,
        });

        (status, body).into_response()
    }
}

// ============================================================================
// Server Setup
// ============================================================================

pub fn create_router(
    shared_state: SharedEnvState,
    command_tx: mpsc::UnboundedSender<EnvCommand>,
    game_config: Arc<Mutex<crate::config::GameConfig>>,
) -> Router {
    let api_state = ApiState {
        shared_state,
        command_tx,
        game_config,
    };

    // Configure CORS to allow requests from Python clients
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    Router::new()
        .route("/reset", post(reset_handler))
        .route("/step", post(step_handler))
        .route("/observation_space", get(observation_space_handler))
        .route("/action_space", get(action_space_handler))
        .route("/start_training", post(start_training_handler))
        .route("/end_training", post(end_training_handler))
        .route("/set_level", post(set_level_handler))
        .route("/configure", post(configure_handler))
        .layer(cors)
        .with_state(api_state)
}

/// Start the HTTP API server on a separate tokio runtime
pub fn start_api_server(
    port: u16,
    shared_state: SharedEnvState,
    command_tx: mpsc::UnboundedSender<EnvCommand>,
    game_config: Arc<Mutex<crate::config::GameConfig>>,
) {
    std::thread::spawn(move || {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let app = create_router(shared_state, command_tx, game_config);
            let addr = format!("127.0.0.1:{}", port);
            let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();

            println!("RL API server listening on http://{}", addr);

            axum::serve(listener, app).await.unwrap();
        });
    });
}
