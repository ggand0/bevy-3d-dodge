use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;
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
    Step { action: usize },
    StartTraining,
    EndTraining,
    SetLevel { level: u8 },
}

/// API server state
#[derive(Clone)]
struct ApiState {
    shared_state: SharedEnvState,
    command_tx: mpsc::UnboundedSender<EnvCommand>,
}

// ============================================================================
// Request/Response Types
// ============================================================================

#[derive(Deserialize)]
struct StepRequest {
    action: usize,
}

#[derive(Deserialize)]
struct SetLevelRequest {
    level: u8,
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
struct ActionSpaceResponse {
    r#type: String,
    n: usize,
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
        .map_err(|_| AppError::InternalError("Failed to lock observation".to_string()))?
        .clone();

    let info = state
        .shared_state
        .info
        .lock()
        .map_err(|_| AppError::InternalError("Failed to lock info".to_string()))?
        .clone();

    Ok(Json(ResetResponse { observation, info }))
}

async fn step_handler(
    State(state): State<ApiState>,
    Json(request): Json<StepRequest>,
) -> Result<Json<StepResponse>, AppError> {
    // Validate action
    if request.action >= 5 {
        return Err(AppError::InvalidAction(format!(
            "Invalid action: {}. Must be in range [0, 5)",
            request.action
        )));
    }

    // Send step command to game loop
    state
        .command_tx
        .send(EnvCommand::Step {
            action: request.action,
        })
        .map_err(|_| AppError::InternalError("Failed to send step command".to_string()))?;

    // Wait a bit for game loop to process step (simple synchronization)
    tokio::time::sleep(tokio::time::Duration::from_millis(16)).await; // ~60 FPS

    // Read state from shared state
    let observation = state
        .shared_state
        .observation
        .lock()
        .map_err(|_| AppError::InternalError("Failed to lock observation".to_string()))?
        .clone();

    let reward = *state
        .shared_state
        .reward
        .lock()
        .map_err(|_| AppError::InternalError("Failed to lock reward".to_string()))?;

    let done = *state
        .shared_state
        .done
        .lock()
        .map_err(|_| AppError::InternalError("Failed to lock done".to_string()))?;

    let truncated = *state
        .shared_state
        .truncated
        .lock()
        .map_err(|_| AppError::InternalError("Failed to lock truncated".to_string()))?;

    let info = state
        .shared_state
        .info
        .lock()
        .map_err(|_| AppError::InternalError("Failed to lock info".to_string()))?
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

async fn action_space_handler() -> Json<ActionSpaceResponse> {
    Json(ActionSpaceResponse {
        r#type: "Discrete".to_string(),
        n: 5,
    })
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
) -> Router {
    let api_state = ApiState {
        shared_state,
        command_tx,
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
        .layer(cors)
        .with_state(api_state)
}

/// Start the HTTP API server on a separate tokio runtime
pub fn start_api_server(
    port: u16,
    shared_state: SharedEnvState,
    command_tx: mpsc::UnboundedSender<EnvCommand>,
) {
    std::thread::spawn(move || {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let app = create_router(shared_state, command_tx);
            let addr = format!("127.0.0.1:{}", port);
            let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();

            println!("RL API server listening on http://{}", addr);

            axum::serve(listener, app).await.unwrap();
        });
    });
}
