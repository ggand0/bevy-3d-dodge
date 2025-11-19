use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};
use tower_http::cors::{Any, CorsLayer};

use crate::rl::observation::OBSERVATION_SIZE;

/// Shared state between Axum server and Bevy game loop
#[derive(Clone, bevy::prelude::Resource)]
pub struct SharedEnvState {
    pub observation: Arc<Mutex<Vec<f32>>>,
    pub reward: Arc<Mutex<f32>>,
    pub done: Arc<Mutex<bool>>,
    pub truncated: Arc<Mutex<bool>>,
    pub info: Arc<Mutex<std::collections::HashMap<String, serde_json::Value>>>,
}

// Note: These are tokio::sync::Mutex, not std::sync::Mutex

impl Default for SharedEnvState {
    fn default() -> Self {
        Self {
            observation: Arc::new(Mutex::new(vec![0.0; OBSERVATION_SIZE])),
            reward: Arc::new(Mutex::new(0.0)),
            done: Arc::new(Mutex::new(false)),
            truncated: Arc::new(Mutex::new(false)),
            info: Arc::new(Mutex::new(std::collections::HashMap::new())),
        }
    }
}

/// Commands sent from API to game loop
#[derive(Debug, Clone)]
pub enum EnvCommand {
    Reset,
    StepDiscrete { action: usize },
    StepContinuous { action: [f32; 5] },
    StartTraining,
    EndTraining,
    SetLevel { level: u8 },
    Configure { level: Option<u8>, action_space_type: Option<String> },
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
}

#[derive(Serialize)]
struct ResetResponse {
    observation: Vec<f32>,
    info: std::collections::HashMap<String, serde_json::Value>,
}

#[derive(Serialize)]
struct StepResponse {
    observation: Vec<f32>,
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
    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

    // Read observation and info from shared state
    let observation = state
        .shared_state
        .observation
        .lock()
        .await
        .clone();

    let info = state
        .shared_state
        .info
        .lock()
        .await
        .clone();

    Ok(Json(ResetResponse { observation, info }))
}

async fn step_handler(
    State(state): State<ApiState>,
    Json(request): Json<StepRequest>,
) -> Result<Json<StepResponse>, AppError> {
    // Get current action space type from config
    let game_config = state.game_config.lock().await;
    let action_space_type = game_config.action_space_type;
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
        crate::config::ActionSpaceType::Continuous => {
            // Expect array of 5 floats: [vx, vy, pitch, roll, sprint]
            let action_array = request.action.as_array()
                .ok_or_else(|| AppError::InvalidAction("Action must be an array for continuous action space".to_string()))?;

            if action_array.len() != 5 {
                return Err(AppError::InvalidAction(format!(
                    "Continuous action must have 5 components [vx, vy, pitch, roll, sprint], got {}",
                    action_array.len()
                )));
            }

            let mut action = [0.0f32; 5];
            for (i, val) in action_array.iter().enumerate() {
                let f = val.as_f64()
                    .ok_or_else(|| AppError::InvalidAction(format!("Action component {} must be a number", i)))?
                    as f32;

                if f < -1.0 || f > 1.0 {
                    return Err(AppError::InvalidAction(format!(
                        "Action component {} has value {} outside valid range [-1, 1]",
                        i, f
                    )));
                }

                action[i] = f;
            }

            EnvCommand::StepContinuous { action }
        }
    };

    // Send step command to game loop
    state
        .command_tx
        .send(command)
        .map_err(|_| AppError::InternalError("Failed to send step command".to_string()))?;

    // Wait a bit for game loop to process step (simple synchronization)
    tokio::time::sleep(tokio::time::Duration::from_millis(16)).await; // ~60 FPS

    // Read state from shared state
    let observation = state
        .shared_state
        .observation
        .lock()
        .await
        .clone();

    let reward = *state
        .shared_state
        .reward
        .lock()
        .await;

    let done = *state
        .shared_state
        .done
        .lock()
        .await;

    let truncated = *state
        .shared_state
        .truncated
        .lock()
        .await;

    let info = state
        .shared_state
        .info
        .lock()
        .await
        .clone();

    Ok(Json(StepResponse {
        observation,
        reward,
        done,
        truncated,
        info,
    }))
}

async fn observation_space_handler() -> Json<ObservationSpaceResponse> {
    Json(ObservationSpaceResponse {
        shape: vec![OBSERVATION_SIZE],
        dtype: "float32".to_string(),
        low: -100.0,
        high: 100.0,
    })
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
        crate::config::ActionSpaceType::Continuous => ActionSpaceResponse::Box {
            r#type: "Box".to_string(),
            shape: vec![5],  // [vx, vy, pitch, roll, sprint]
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
        if action_space_lower != "discrete" && action_space_lower != "continuous" {
            return Err(AppError::InvalidAction(format!(
                "Invalid action_space_type: '{}'. Must be 'discrete' or 'continuous'",
                action_space_type
            )));
        }
    }

    state
        .command_tx
        .send(EnvCommand::Configure {
            level: payload.level,
            action_space_type: payload.action_space_type,
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
