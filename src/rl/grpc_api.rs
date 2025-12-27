use std::sync::Arc;
use tokio::net::UnixListener;
use tokio::sync::{mpsc, Mutex};
use tokio_stream::wrappers::UnixListenerStream;
use tonic::{Request, Response, Status, transport::Server};

use crate::config::{GameConfig, ObservationMode};
use crate::rl::api::{EnvCommand, SharedEnvState};

// Include generated protobuf code
pub mod rl_env {
    tonic::include_proto!("rl_env");
}

use rl_env::rl_environment_server::{RlEnvironment, RlEnvironmentServer};
use rl_env::{
    ActionSpace, ActionSpaceRequest, BoxSpace, ConfigureRequest, ConfigureResponse,
    ContinuousAction, DiscreteSpace, EndTrainingRequest, EndTrainingResponse,
    ObservationSpace, ObservationSpaceRequest, ResetRequest, ResetResponse,
    SetLevelRequest, SetLevelResponse, StartTrainingRequest, StartTrainingResponse,
    StepRequest, StepResponse,
};

/// gRPC service implementation
pub struct GrpcEnvService {
    shared_state: SharedEnvState,
    command_tx: mpsc::UnboundedSender<EnvCommand>,
    game_config: Arc<Mutex<GameConfig>>,
}

impl GrpcEnvService {
    pub fn new(
        shared_state: SharedEnvState,
        command_tx: mpsc::UnboundedSender<EnvCommand>,
        game_config: Arc<Mutex<GameConfig>>,
    ) -> Self {
        Self {
            shared_state,
            command_tx,
            game_config,
        }
    }
}

#[tonic::async_trait]
impl RlEnvironment for GrpcEnvService {
    async fn get_observation_space(
        &self,
        _request: Request<ObservationSpaceRequest>,
    ) -> Result<Response<ObservationSpace>, Status> {
        let game_config = self.game_config.lock().await;
        let obs_mode = game_config.observation_mode;

        let response = if obs_mode.is_image_mode() {
            ObservationSpace {
                shape: vec![
                    game_config.image_obs_height as i32,
                    game_config.image_obs_width as i32,
                    game_config.image_channels() as i32,
                ],
                dtype: "uint8".to_string(),
                low: 0.0,
                high: 255.0,
            }
        } else {
            let obs_size = obs_mode.observation_size();
            ObservationSpace {
                shape: vec![obs_size as i32],
                dtype: "float32".to_string(),
                low: -100.0,
                high: 100.0,
            }
        };

        Ok(Response::new(response))
    }

    async fn get_action_space(
        &self,
        _request: Request<ActionSpaceRequest>,
    ) -> Result<Response<ActionSpace>, Status> {
        let game_config = self.game_config.lock().await;

        let response = match game_config.action_space_type {
            crate::config::ActionSpaceType::Discrete => ActionSpace {
                space_type: Some(rl_env::action_space::SpaceType::Discrete(DiscreteSpace {
                    n: 5,
                })),
            },
            crate::config::ActionSpaceType::Continuous(cont_config) => ActionSpace {
                space_type: Some(rl_env::action_space::SpaceType::Box(BoxSpace {
                    shape: vec![cont_config.dimension() as i32],
                    low: -1.0,
                    high: 1.0,
                })),
            },
        };

        Ok(Response::new(response))
    }

    async fn reset(
        &self,
        _request: Request<ResetRequest>,
    ) -> Result<Response<ResetResponse>, Status> {
        // Send reset command to game loop
        self.command_tx
            .send(EnvCommand::Reset)
            .map_err(|_| Status::internal("Failed to send reset command"))?;

        // Wait for game loop to process reset (reduced for higher throughput)
        tokio::time::sleep(tokio::time::Duration::from_millis(20)).await;

        // Get observation mode and image size
        let game_config = self.game_config.lock().await;
        let obs_mode = game_config.observation_mode;
        let image_size = (game_config.image_obs_width
            * game_config.image_obs_height
            * game_config.image_channels()) as usize;
        drop(game_config);

        // Read observation from shared state
        let observation = self.shared_state.observation.read().await.clone();

        // Get image observation if in TopDownImage mode (raw bytes, no Base64!)
        let image_observation = if obs_mode == ObservationMode::TopDownImage {
            let image_buffer = self.shared_state.image_observation.read().await;
            image_buffer[..image_size].to_vec()
        } else {
            Vec::new()
        };

        // Read info and convert to string map
        let info_map = self.shared_state.info.read().await;
        let info: std::collections::HashMap<String, String> = info_map
            .iter()
            .map(|(k, v)| (k.clone(), v.to_string()))
            .collect();

        Ok(Response::new(ResetResponse {
            observation,
            image_observation,
            info,
        }))
    }

    async fn step(
        &self,
        request: Request<StepRequest>,
    ) -> Result<Response<StepResponse>, Status> {
        let req = request.into_inner();

        // Get current action space type and observation mode
        let game_config = self.game_config.lock().await;
        let action_space_type = game_config.action_space_type;
        let obs_mode = game_config.observation_mode;
        let image_size = (game_config.image_obs_width
            * game_config.image_obs_height
            * game_config.image_channels()) as usize;
        drop(game_config);

        // Parse action and create command
        let command = match req.action {
            Some(rl_env::step_request::Action::DiscreteAction(action)) => {
                if action < 0 || action >= 5 {
                    return Err(Status::invalid_argument(format!(
                        "Invalid action: {}. Must be in range [0, 5)",
                        action
                    )));
                }
                EnvCommand::StepDiscrete {
                    action: action as usize,
                }
            }
            Some(rl_env::step_request::Action::ContinuousAction(ContinuousAction { values })) => {
                match action_space_type {
                    crate::config::ActionSpaceType::Discrete => {
                        return Err(Status::invalid_argument(
                            "Received continuous action but action space is discrete",
                        ));
                    }
                    crate::config::ActionSpaceType::Continuous(cont_config) => {
                        let expected_dim = cont_config.dimension();
                        if values.len() != expected_dim {
                            return Err(Status::invalid_argument(format!(
                                "Continuous action must have {} components, got {}",
                                expected_dim,
                                values.len()
                            )));
                        }

                        // Validate value ranges
                        for (i, &v) in values.iter().enumerate() {
                            if v < -1.0 || v > 1.0 {
                                return Err(Status::invalid_argument(format!(
                                    "Component {} has value {} outside valid range [-1, 1]",
                                    i, v
                                )));
                            }
                        }

                        EnvCommand::StepContinuous {
                            action: values,
                            config: cont_config,
                        }
                    }
                }
            }
            None => {
                return Err(Status::invalid_argument("No action provided"));
            }
        };

        // Get current step counter
        let current_step = *self.shared_state.step_counter.read().await;

        // Send step command to game loop
        self.command_tx
            .send(command)
            .map_err(|_| Status::internal("Failed to send step command"))?;

        // Wait for step counter to increment (max 100ms timeout)
        let deadline = tokio::time::Instant::now() + tokio::time::Duration::from_millis(100);
        loop {
            let new_step = *self.shared_state.step_counter.read().await;
            if new_step > current_step {
                break;
            }
            if tokio::time::Instant::now() >= deadline {
                break; // Timeout - return current state
            }
            tokio::task::yield_now().await;
        }

        // Read state from shared state
        let observation = self.shared_state.observation.read().await.clone();
        let reward = *self.shared_state.reward.read().await;
        let done = *self.shared_state.done.read().await;
        let truncated = *self.shared_state.truncated.read().await;

        // Get image observation if in TopDownImage mode (raw bytes!)
        let image_observation = if obs_mode == ObservationMode::TopDownImage {
            let image_buffer = self.shared_state.image_observation.read().await;
            image_buffer[..image_size].to_vec()
        } else {
            Vec::new()
        };

        // Read info and convert to string map
        let info_map = self.shared_state.info.read().await;
        let info: std::collections::HashMap<String, String> = info_map
            .iter()
            .map(|(k, v)| (k.clone(), v.to_string()))
            .collect();

        Ok(Response::new(StepResponse {
            observation,
            image_observation,
            reward,
            done,
            truncated,
            info,
        }))
    }

    async fn configure(
        &self,
        request: Request<ConfigureRequest>,
    ) -> Result<Response<ConfigureResponse>, Status> {
        let req = request.into_inner();

        // Validate level if provided
        if let Some(level) = req.level {
            if level < 1 || level > 2 {
                return Err(Status::invalid_argument(format!(
                    "Invalid level: {}. Must be 1 or 2",
                    level
                )));
            }
        }

        // Validate action_space_type if provided
        if let Some(ref action_space_type) = req.action_space_type {
            let action_space_lower = action_space_type.to_lowercase();
            if action_space_lower != "discrete"
                && crate::config::ContinuousActionConfig::from_str(&action_space_lower).is_none()
            {
                return Err(Status::invalid_argument(format!(
                    "Invalid action_space_type: '{}'. Must be 'discrete' or a continuous variant",
                    action_space_type
                )));
            }
        }

        // Validate spawn_angle_degrees if provided
        if let Some(angle) = req.spawn_angle_degrees {
            if angle <= 0.0 || angle > 180.0 {
                return Err(Status::invalid_argument(format!(
                    "Invalid spawn_angle_degrees: {}. Must be between 0 and 180",
                    angle
                )));
            }
        }

        // Validate sprint_multiplier if provided
        if let Some(mult) = req.sprint_multiplier {
            if mult < 0.0 || mult > 10.0 {
                return Err(Status::invalid_argument(format!(
                    "Invalid sprint_multiplier: {}. Must be between 0 and 10",
                    mult
                )));
            }
        }

        // Validate observation_mode if provided
        if let Some(ref obs_mode) = req.observation_mode {
            if ObservationMode::from_str(obs_mode).is_none() {
                return Err(Status::invalid_argument(format!(
                    "Invalid observation_mode: '{}'. Must be 'standard', 'with_thrower', or 'topdown'",
                    obs_mode
                )));
            }
        }

        // Validate thrower_delay_seconds if provided
        if let Some(delay) = req.thrower_delay_seconds {
            if delay <= 0.0 || delay > 10.0 {
                return Err(Status::invalid_argument(format!(
                    "Invalid thrower_delay_seconds: {}. Must be between 0 and 10",
                    delay
                )));
            }
        }

        // Validate image dimensions if provided
        if let Some(width) = req.image_obs_width {
            if width < 32 || width > 512 {
                return Err(Status::invalid_argument(format!(
                    "Invalid image_obs_width: {}. Must be between 32 and 512",
                    width
                )));
            }
        }

        if let Some(height) = req.image_obs_height {
            if height < 32 || height > 512 {
                return Err(Status::invalid_argument(format!(
                    "Invalid image_obs_height: {}. Must be between 32 and 512",
                    height
                )));
            }
        }

        self.command_tx
            .send(EnvCommand::Configure {
                level: req.level.map(|l| l as u8),
                action_space_type: req.action_space_type,
                sprint_multiplier: req.sprint_multiplier,
                spawn_angle_degrees: req.spawn_angle_degrees,
                observation_mode: req.observation_mode,
                thrower_delay_seconds: req.thrower_delay_seconds,
                image_obs_width: req.image_obs_width,
                image_obs_height: req.image_obs_height,
                image_grayscale: req.image_grayscale,
            })
            .map_err(|_| Status::internal("Failed to send configure command"))?;

        Ok(Response::new(ConfigureResponse {}))
    }

    async fn set_level(
        &self,
        request: Request<SetLevelRequest>,
    ) -> Result<Response<SetLevelResponse>, Status> {
        let req = request.into_inner();

        if req.level < 1 || req.level > 2 {
            return Err(Status::invalid_argument(format!(
                "Invalid level: {}. Must be 1 or 2",
                req.level
            )));
        }

        self.command_tx
            .send(EnvCommand::SetLevel {
                level: req.level as u8,
            })
            .map_err(|_| Status::internal("Failed to send set level command"))?;

        Ok(Response::new(SetLevelResponse {}))
    }

    async fn start_training(
        &self,
        _request: Request<StartTrainingRequest>,
    ) -> Result<Response<StartTrainingResponse>, Status> {
        self.command_tx
            .send(EnvCommand::StartTraining)
            .map_err(|_| Status::internal("Failed to send start training command"))?;

        Ok(Response::new(StartTrainingResponse {}))
    }

    async fn end_training(
        &self,
        _request: Request<EndTrainingRequest>,
    ) -> Result<Response<EndTrainingResponse>, Status> {
        self.command_tx
            .send(EnvCommand::EndTraining)
            .map_err(|_| Status::internal("Failed to send end training command"))?;

        Ok(Response::new(EndTrainingResponse {}))
    }
}

/// Start the gRPC server with Unix domain socket
pub fn start_grpc_server_uds(
    socket_path: String,
    shared_state: SharedEnvState,
    command_tx: mpsc::UnboundedSender<EnvCommand>,
    game_config: Arc<Mutex<GameConfig>>,
) {
    std::thread::spawn(move || {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            // Remove existing socket file if present
            let _ = std::fs::remove_file(&socket_path);

            let uds = UnixListener::bind(&socket_path).expect("Failed to bind Unix socket");
            let uds_stream = UnixListenerStream::new(uds);

            let service = GrpcEnvService::new(shared_state, command_tx, game_config);

            println!("gRPC server listening on unix://{}", socket_path);

            Server::builder()
                .add_service(RlEnvironmentServer::new(service))
                .serve_with_incoming(uds_stream)
                .await
                .expect("gRPC server error");
        });
    });
}

/// Start the gRPC server with TCP socket
pub fn start_grpc_server_tcp(
    port: u16,
    shared_state: SharedEnvState,
    command_tx: mpsc::UnboundedSender<EnvCommand>,
    game_config: Arc<Mutex<GameConfig>>,
) {
    std::thread::spawn(move || {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let addr = format!("0.0.0.0:{}", port).parse().unwrap();
            let service = GrpcEnvService::new(shared_state, command_tx, game_config);

            println!("gRPC server listening on {}", addr);

            Server::builder()
                .add_service(RlEnvironmentServer::new(service))
                .serve(addr)
                .await
                .expect("gRPC server error");
        });
    });
}
