mod config;
mod game;
mod rl;

use bevy::prelude::*;
use bevy::app::ScheduleRunnerPlugin;
use clap::Parser;
use config::{GameConfig, Level, SharedGameConfig};
use tokio::sync::mpsc;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;
use rl::api::{EnvCommand, SharedEnvState, start_api_server};
use rl::grpc_api::{start_grpc_server_uds, start_grpc_server_tcp};
use rl::environment::{RLEnvironmentState, ControlMode, TrainingMode};

/// Bevy 3D Dodge - RL Training Game
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run in headless mode (no window, for training)
    #[arg(long, default_value_t = false)]
    headless: bool,

    /// API server port (only used with --http)
    #[arg(long, default_value_t = 8000)]
    port: u16,

    /// Target FPS for headless mode (ignored in windowed mode)
    /// Note: 60 FPS is recommended for CNN training to reduce lock contention
    #[arg(long, default_value_t = 60)]
    fps: u32,

    /// Use HTTP transport instead of gRPC (default: gRPC with Unix socket)
    #[arg(long, default_value_t = false)]
    http: bool,

    /// Unix domain socket path for gRPC (only used without --http and without --grpc-port)
    #[arg(long, default_value = "/tmp/bevy_rl.sock")]
    socket_path: String,

    /// Use TCP instead of Unix socket for gRPC (specify port number)
    #[arg(long)]
    grpc_port: Option<u16>,
}

fn main() {
    let args = Args::parse();
    // Create channel for RL API commands
    let (command_tx, command_rx) = mpsc::unbounded_channel::<EnvCommand>();

    // Create shared state for RL API
    let shared_state = SharedEnvState::default();

    // Create shared game config for API server
    let game_config = Arc::new(Mutex::new(GameConfig::default()));
    let shared_config = SharedGameConfig(game_config.clone());

    // Start API server (HTTP or gRPC based on flag)
    if args.http {
        start_api_server(args.port, shared_state.clone(), command_tx, game_config);
    } else if let Some(grpc_port) = args.grpc_port {
        start_grpc_server_tcp(grpc_port, shared_state.clone(), command_tx, game_config);
    } else {
        start_grpc_server_uds(
            args.socket_path.clone(),
            shared_state.clone(),
            command_tx,
            game_config,
        );
    }

    if args.headless {
        if args.http {
            info!("Starting in HEADLESS mode with HTTP (port: {}, fps: {})", args.port, args.fps);
        } else if let Some(grpc_port) = args.grpc_port {
            info!("Starting in HEADLESS mode with gRPC/TCP (port: {}, fps: {})", grpc_port, args.fps);
        } else {
            info!("Starting in HEADLESS mode with gRPC/UDS (socket: {}, fps: {})", args.socket_path, args.fps);
        }
        run_headless(args, command_rx, shared_state, shared_config);
    } else {
        if args.http {
            info!("Starting in WINDOWED mode with HTTP (port: {})", args.port);
        } else if let Some(grpc_port) = args.grpc_port {
            info!("Starting in WINDOWED mode with gRPC/TCP (port: {})", grpc_port);
        } else {
            info!("Starting in WINDOWED mode with gRPC/UDS (socket: {})", args.socket_path);
        }
        run_windowed(command_rx, shared_state, shared_config);
    }
}

/// Run the game in headless mode (no window, for training)
fn run_headless(
    args: Args,
    command_rx: mpsc::UnboundedReceiver<EnvCommand>,
    shared_state: SharedEnvState,
    shared_config: SharedGameConfig,
) {
    let frame_duration = Duration::from_secs_f64(1.0 / args.fps as f64);

    App::new()
        .add_plugins(MinimalPlugins.set(ScheduleRunnerPlugin::run_loop(frame_duration)))
        .add_plugins(bevy::log::LogPlugin::default())
        .insert_resource(Level::default())
        .insert_resource(GameConfig::default())
        .insert_resource(RLEnvironmentState::default())
        .insert_resource(ControlMode::default())
        .insert_resource(TrainingMode::default())
        .insert_non_send_resource(command_rx)
        .insert_resource(shared_state)
        .insert_resource(shared_config)
        .insert_resource(HeadlessMode(true))
        .add_plugins(game::HeadlessGamePlugin)
        .add_systems(Startup, setup_headless)
        .add_systems(Update, (
            handle_rl_commands_headless,
            update_rl_state,
        ))
        .run();
}

/// Run the game in windowed mode (normal GUI)
fn run_windowed(
    command_rx: mpsc::UnboundedReceiver<EnvCommand>,
    shared_state: SharedEnvState,
    shared_config: SharedGameConfig,
) {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Bevy 3D Dodge - RL Training Game".to_string(),
                resolution: (1280., 720.).into(),
                ..default()
            }),
            ..default()
        }))
        .insert_resource(Level::default())
        .insert_resource(GameConfig::default())
        .insert_resource(RLEnvironmentState::default())
        .insert_resource(ControlMode::default())
        .insert_resource(TrainingMode::default())
        .insert_non_send_resource(command_rx)
        .insert_resource(shared_state)
        .insert_resource(shared_config)
        .insert_resource(HeadlessMode(false))
        .add_plugins(game::GamePlugin)
        .add_systems(Startup, setup_scene)
        .add_systems(Update, (
            handle_reset,
            handle_level_change,
            update_ui,
            update_action_debug,
            update_spawn_arc,
            handle_rl_commands,
            update_rl_state
        ))
        .run();
}

/// Resource to track if we're in headless mode
#[allow(dead_code)]
#[derive(Resource)]
struct HeadlessMode(bool);

/// Minimal setup for headless mode (no rendering)
fn setup_headless() {
    info!("Headless mode initialized - API ready for RL training");
}

fn setup_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Ground plane (Isaac Sim style)
    commands.spawn((
        Mesh3d(meshes.add(Mesh::from(Rectangle::new(50.0, 50.0)))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.40, 0.60, 0.90), // More blue
            perceptual_roughness: 0.85, // Slightly rough concrete/gym floor
            metallic: 0.0,
            reflectance: 0.35, // Moderate reflectance for indoor floor
            cull_mode: None,
            ..default()
        })),
        Transform::from_xyz(0.0, 0.0, 0.0),
    ));

    // Create grid lines using thin boxes
    let grid_size = 50;
    let grid_spacing = 1.0;
    let line_thickness = 0.02;
    let line_height = 0.001;

    let grid_material = materials.add(StandardMaterial {
        base_color: Color::srgba(1.0, 1.0, 1.0, 0.6), // White grid lines, slightly more visible
        alpha_mode: AlphaMode::Blend,
        perceptual_roughness: 0.95, // Very rough for painted lines
        metallic: 0.0,
        reflectance: 0.1, // Minimal reflectance for matte paint
        ..default()
    });

    // Create grid lines parallel to X axis
    for i in -(grid_size / 2)..=(grid_size / 2) {
        let y = i as f32 * grid_spacing;
        commands.spawn((
            Mesh3d(meshes.add(Cuboid::new(grid_size as f32, line_thickness, line_height))),
            MeshMaterial3d(grid_material.clone()),
            Transform::from_xyz(0.0, y, line_height / 2.0),
        ));
    }

    // Create grid lines parallel to Y axis
    for i in -(grid_size / 2)..=(grid_size / 2) {
        let x = i as f32 * grid_spacing;
        commands.spawn((
            Mesh3d(meshes.add(Cuboid::new(line_thickness, grid_size as f32, line_height))),
            MeshMaterial3d(grid_material.clone()),
            Transform::from_xyz(x, 0.0, line_height / 2.0),
        ));
    }

    // Softer directional light for subtle shadows (mimics skylights/windows)
    commands.spawn((
        DirectionalLight {
            illuminance: 10000.0, // Increased for brighter scene
            shadows_enabled: true,
            shadow_depth_bias: 0.02, // Prevents shadow acne
            shadow_normal_bias: 0.6, // Prevents peter-panning artifacts
            ..default()
        },
        Transform::from_xyz(15.0, -10.0, 25.0).looking_at(Vec3::ZERO, Vec3::Z),
    ));

    // Multiple point lights to simulate ceiling-mounted gym lights
    let light_height = 8.0;
    let light_positions = [
        Vec3::new(-6.0, -6.0, light_height),
        Vec3::new(6.0, -6.0, light_height),
        Vec3::new(-6.0, 6.0, light_height),
        Vec3::new(6.0, 6.0, light_height),
        Vec3::new(0.0, 0.0, light_height),
    ];

    for pos in light_positions.iter() {
        commands.spawn((
            PointLight {
                intensity: 1200000.0, // Increased for brighter gym lighting
                color: Color::srgb(1.0, 0.98, 0.95), // Warm white
                radius: 20.0,
                range: 30.0, // Increased range for better coverage
                shadows_enabled: false, // Disable for performance, directional light handles shadows
                ..default()
            },
            Transform::from_translation(*pos),
        ));
    }

    // Ambient light (now lower since we have point lights)
    commands.insert_resource(AmbientLight {
        color: Color::srgb(0.95, 0.95, 1.0), // Brighter cool ambient
        brightness: 250.0, // Increased for overall brighter scene
    });

    // Play zone marker (dodgeball court)
    let zone_width = 10.0;  // X dimension
    let zone_depth = 8.0;   // Y dimension
    let zone_line_thickness = 0.08;
    let zone_line_height = 0.002;

    let zone_material = materials.add(StandardMaterial {
        base_color: Color::srgba(1.0, 1.0, 0.0, 0.8), // Yellow with transparency
        alpha_mode: AlphaMode::Blend,
        emissive: Color::srgb(0.5, 0.5, 0.0).into(),
        unlit: true,
        ..default()
    });

    // Front line (toward where projectiles come from, +Y side)
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(zone_width, zone_line_thickness, zone_line_height))),
        MeshMaterial3d(zone_material.clone()),
        Transform::from_xyz(0.0, zone_depth / 2.0, zone_line_height / 2.0),
    ));

    // Back line (-Y side)
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(zone_width, zone_line_thickness, zone_line_height))),
        MeshMaterial3d(zone_material.clone()),
        Transform::from_xyz(0.0, -zone_depth / 2.0, zone_line_height / 2.0),
    ));

    // Left line (-X side)
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(zone_line_thickness, zone_depth, zone_line_height))),
        MeshMaterial3d(zone_material.clone()),
        Transform::from_xyz(-zone_width / 2.0, 0.0, zone_line_height / 2.0),
    ));

    // Right line (+X side)
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(zone_line_thickness, zone_depth, zone_line_height))),
        MeshMaterial3d(zone_material.clone()),
        Transform::from_xyz(zone_width / 2.0, 0.0, zone_line_height / 2.0),
    ));

    // Spawn arc visualization (shows where projectiles spawn from in Level 2)
    // Arc at spawn_distance=20, angle is dynamic based on config
    let spawn_radius = 20.0;
    let arc_height = 0.1;  // Slightly above ground
    let arc_segments = 30;  // Number of points along the arc
    // Use default angle for initial spawn - will be updated dynamically by update_spawn_arc system
    let half_angle = std::f32::consts::PI / 3.0;  // 60 degrees default

    let arc_material = materials.add(StandardMaterial {
        base_color: Color::srgba(1.0, 0.3, 0.3, 0.8),  // Red-ish for danger zone
        alpha_mode: AlphaMode::Blend,
        emissive: Color::srgb(0.5, 0.1, 0.1).into(),
        unlit: true,
        ..default()
    });

    // Create arc segments
    for i in 0..arc_segments {
        let t = i as f32 / (arc_segments - 1) as f32;  // 0 to 1
        let angle = -half_angle + t * 2.0 * half_angle;

        let x = angle.sin() * spawn_radius;
        let y = angle.cos() * spawn_radius;

        commands.spawn((
            Mesh3d(meshes.add(Sphere::new(0.15))),
            MeshMaterial3d(arc_material.clone()),
            Transform::from_xyz(x, y, arc_height),
            SpawnArcMarker { index: i, is_edge: false },
        ));
    }

    // Add edge markers (larger spheres at arc boundaries)
    for (edge_idx, sign) in [(0_usize, -1.0_f32), (1_usize, 1.0_f32)] {
        let angle = sign * half_angle;
        let x = angle.sin() * spawn_radius;
        let y = angle.cos() * spawn_radius;

        commands.spawn((
            Mesh3d(meshes.add(Sphere::new(0.3))),
            MeshMaterial3d(arc_material.clone()),
            Transform::from_xyz(x, y, arc_height),
            SpawnArcMarker { index: arc_segments + edge_idx, is_edge: true },
        ));
    }

    // Coordinate axes visualization (hidden by default, shown in debug mode)
    let axis_length = 5.0;
    let axis_thickness = 0.1;
    let arrow_head_size = 0.3;

    // X axis (Red) - pointing in +X direction
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(axis_length, axis_thickness, axis_thickness))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(1.0, 0.0, 0.0),
            emissive: Color::srgb(0.5, 0.0, 0.0).into(),
            unlit: true,
            ..default()
        })),
        Transform::from_xyz(axis_length / 2.0, 0.0, 0.1),
        Visibility::Hidden,
        CoordinateAxis,
    ));
    // X axis arrow head
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(arrow_head_size, arrow_head_size * 2.0, arrow_head_size))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(1.0, 0.0, 0.0),
            emissive: Color::srgb(0.5, 0.0, 0.0).into(),
            unlit: true,
            ..default()
        })),
        Transform::from_xyz(axis_length + arrow_head_size / 2.0, 0.0, 0.1)
            .with_rotation(Quat::from_rotation_z(std::f32::consts::FRAC_PI_4)),
        Visibility::Hidden,
        CoordinateAxis,
    ));

    // Y axis (Green) - pointing in +Y direction
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(axis_thickness, axis_length, axis_thickness))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.0, 1.0, 0.0),
            emissive: Color::srgb(0.0, 0.5, 0.0).into(),
            unlit: true,
            ..default()
        })),
        Transform::from_xyz(0.0, axis_length / 2.0, 0.1),
        Visibility::Hidden,
        CoordinateAxis,
    ));
    // Y axis arrow head
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(arrow_head_size * 2.0, arrow_head_size, arrow_head_size))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.0, 1.0, 0.0),
            emissive: Color::srgb(0.0, 0.5, 0.0).into(),
            unlit: true,
            ..default()
        })),
        Transform::from_xyz(0.0, axis_length + arrow_head_size / 2.0, 0.1)
            .with_rotation(Quat::from_rotation_z(std::f32::consts::FRAC_PI_4)),
        Visibility::Hidden,
        CoordinateAxis,
    ));

    // Z axis (Blue) - pointing in +Z direction
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(axis_thickness, axis_thickness, axis_length))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.0, 0.0, 1.0),
            emissive: Color::srgb(0.0, 0.0, 0.5).into(),
            unlit: true,
            ..default()
        })),
        Transform::from_xyz(0.0, 0.0, axis_length / 2.0),
        Visibility::Hidden,
        CoordinateAxis,
    ));
    // Z axis arrow head (cone-like)
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(arrow_head_size, arrow_head_size, arrow_head_size * 2.0))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.0, 0.0, 1.0),
            emissive: Color::srgb(0.0, 0.0, 0.5).into(),
            unlit: true,
            ..default()
        })),
        Transform::from_xyz(0.0, 0.0, axis_length + arrow_head_size),
        Visibility::Hidden,
        CoordinateAxis,
    ));

    // UI Text - Controls
    commands.spawn((
        Text::new("WASD: Move | Shift: Sprint | Space: Jump | R: Reset | L: Level | F1: Free Cam | ESC: Quit"),
        TextFont {
            font_size: 20.0,
            ..default()
        },
        TextColor(Color::WHITE),
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(10.0),
            left: Val::Px(10.0),
            ..default()
        },
        ControlsText,
    ));

    // Level indicator (top right, below control legend)
    commands.spawn((
        Text::new("Level 1 (Baseline)"),
        TextFont {
            font_size: 22.0,
            ..default()
        },
        TextColor(Color::srgb(0.3, 1.0, 0.3)), // Green color
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(40.0),
            right: Val::Px(10.0),
            ..default()
        },
        LevelText,
    ));

    // Config info (below level indicator)
    commands.spawn((
        Text::new("Action Space: Discrete"),
        TextFont {
            font_size: 16.0,
            ..default()
        },
        TextColor(Color::srgb(0.7, 0.7, 0.7)), // Gray color
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(70.0),
            right: Val::Px(10.0),
            ..default()
        },
        ConfigInfoText,
    ));

    // Training mode indicator (shown when training mode is enabled)
    commands.spawn((
        Text::new("TRAINING MODE - Agent Control Active"),
        TextFont {
            font_size: 18.0,
            ..default()
        },
        TextColor(Color::srgb(1.0, 0.3, 0.3)), // Red color
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(35.0),
            left: Val::Px(10.0),
            display: Display::None,
            ..default()
        },
        TrainingModeText,
    ));

    // Action debug text (shown during training mode, left side below training indicator)
    commands.spawn((
        Text::new("vx: 0.00 | vy: 0.00 | sprint: 0.00 | speed: 5.00"),
        TextFont {
            font_size: 16.0,
            ..default()
        },
        TextColor(Color::srgb(0.8, 0.8, 1.0)), // Light blue
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(60.0),
            left: Val::Px(10.0),
            display: Display::None,
            ..default()
        },
        ActionDebugText,
    ));

    // Camera help text for free camera mode
    commands.spawn((
        Text::new("Free Cam: LMB+Drag: Look | MMB+Drag: Pan | Scroll: Zoom | UO: Up/Down"),
        TextFont {
            font_size: 16.0,
            ..default()
        },
        TextColor(Color::srgb(1.0, 1.0, 0.0)),
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(60.0),
            left: Val::Px(10.0),
            ..default()
        },
        CameraDebugText,
    ));

    // Game over text (initially hidden)
    // Centered horizontally using left: 50% and transform translateX(-50%)
    commands.spawn((
        Text::new("GAME OVER!"),
        TextFont {
            font_size: 40.0,
            ..default()
        },
        TextColor(Color::srgb(1.0, 0.5, 0.0)), // Orange color
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(300.0),
            left: Val::Percent(50.0),
            display: Display::None,
            margin: UiRect {
                left: Val::Px(-100.0), // Approximate half-width for centering (adjust based on text width)
                ..default()
            },
            ..default()
        },
        GameOverText,
    ));
}

#[derive(Component)]
struct GameOverText;

#[derive(Component)]
struct TrainingModeText;

#[derive(Component)]
struct ControlsText;

#[derive(Component)]
struct LevelText;

#[derive(Component)]
struct ConfigInfoText;

#[derive(Component)]
struct CameraDebugText;

#[derive(Component)]
struct CoordinateAxis;

#[derive(Component)]
struct SpawnArcMarker {
    /// Index of this marker in the arc (0..arc_segments for segment markers, arc_segments+ for edge markers)
    index: usize,
    /// Whether this is an edge marker (larger sphere at arc boundaries)
    is_edge: bool,
}

#[derive(Component)]
struct ActionDebugText;

fn update_ui(
    game_state: Res<game::collision::GameState>,
    debug_mode: Res<game::camera::CameraDebugMode>,
    free_camera_mode: Res<game::camera::FreeCameraMode>,
    training_mode: Res<TrainingMode>,
    level: Res<Level>,
    config: Res<config::GameConfig>,
    mut game_over_query: Query<&mut Node, (With<GameOverText>, Without<CameraDebugText>, Without<TrainingModeText>, Without<ControlsText>, Without<LevelText>, Without<ConfigInfoText>, Without<ActionDebugText>)>,
    mut training_text_query: Query<&mut Node, (With<TrainingModeText>, Without<CameraDebugText>, Without<GameOverText>, Without<ControlsText>, Without<LevelText>, Without<ConfigInfoText>, Without<ActionDebugText>)>,
    mut controls_text_query: Query<&mut Node, (With<ControlsText>, Without<CameraDebugText>, Without<GameOverText>, Without<TrainingModeText>, Without<LevelText>, Without<ConfigInfoText>, Without<ActionDebugText>)>,
    mut level_text_query: Query<&mut Text, (With<LevelText>, Without<ConfigInfoText>)>,
    mut config_info_query: Query<&mut Text, (With<ConfigInfoText>, Without<LevelText>)>,
    mut debug_text_query: Query<&mut Node, (With<CameraDebugText>, Without<GameOverText>, Without<TrainingModeText>, Without<ControlsText>, Without<LevelText>, Without<ConfigInfoText>, Without<ActionDebugText>)>,
    mut action_debug_query: Query<&mut Node, With<ActionDebugText>>,
    mut axis_query: Query<&mut Visibility, With<CoordinateAxis>>,
) {
    if let Ok(mut node) = game_over_query.get_single_mut() {
        node.display = if game_state.is_game_over {
            Display::Flex
        } else {
            Display::None
        };
    }

    // Training mode indicator is visible when training mode is enabled
    if let Ok(mut node) = training_text_query.get_single_mut() {
        node.display = if training_mode.enabled {
            Display::Flex
        } else {
            Display::None
        };
    }

    // Controls text is hidden during training mode
    if let Ok(mut node) = controls_text_query.get_single_mut() {
        node.display = if training_mode.enabled {
            Display::None
        } else {
            Display::Flex
        };
    }

    // Action debug text is visible when training mode is enabled
    if let Ok(mut node) = action_debug_query.get_single_mut() {
        node.display = if training_mode.enabled {
            Display::Flex
        } else {
            Display::None
        };
    }

    // Update level indicator text
    if let Ok(mut text) = level_text_query.get_single_mut() {
        **text = level.name().to_string();
    }

    // Update config info text
    if let Ok(mut text) = config_info_query.get_single_mut() {
        let action_space_str = match config.action_space_type {
            config::ActionSpaceType::Discrete => "Discrete",
            config::ActionSpaceType::Continuous(cont_config) => cont_config.name(),
        };
        **text = format!("Action Space: {}", action_space_str);
    }

    // Camera help text is only visible in free camera mode
    if let Ok(mut node) = debug_text_query.get_single_mut() {
        node.display = if free_camera_mode.enabled {
            Display::Flex
        } else {
            Display::None
        };
    }

    // Toggle coordinate axes visibility based on debug mode
    for mut visibility in axis_query.iter_mut() {
        *visibility = if debug_mode.enabled {
            Visibility::Visible
        } else {
            Visibility::Hidden
        };
    }
}

fn update_action_debug(
    player_query: Query<&game::player::Velocity, With<game::player::Player>>,
    config: Res<GameConfig>,
    mut debug_text_query: Query<&mut Text, With<ActionDebugText>>,
) {
    if let Ok(velocity) = player_query.get_single() {
        if let Ok(mut text) = debug_text_query.get_single_mut() {
            let vx = velocity.0.x;
            let vy = velocity.0.y;
            let speed = velocity.0.length();

            // Calculate sprint value from speed
            // speed = base_speed * (1.0 + sprint * multiplier)
            // sprint = (speed / base_speed - 1.0) / multiplier
            let sprint = if config.sprint_multiplier > 0.0 {
                ((speed / config.player_speed - 1.0) / config.sprint_multiplier).max(0.0).min(1.0)
            } else {
                0.0
            };

            **text = format!(
                "vx: {:.2} | vy: {:.2} | sprint: {:.2} | speed: {:.2}",
                vx, vy, sprint, speed
            );
        }
    }
}

fn handle_reset(
    keyboard_input: Res<ButtonInput<KeyCode>>,
    training_mode: Res<TrainingMode>,
    mut game_state: ResMut<game::collision::GameState>,
    mut player_query: Query<
        (
            &mut Transform,
            &mut game::player::Velocity,
            &mut game::player::VerticalVelocity,
            &mut game::player::OnGround,
            &MeshMaterial3d<StandardMaterial>,
        ),
        With<game::player::Player>,
    >,
    projectile_query: Query<Entity, With<game::projectile::Projectile>>,
    mut commands: Commands,
    config: Res<GameConfig>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Disable R key reset during training mode to prevent accidental interruptions
    if keyboard_input.just_pressed(KeyCode::KeyR) && !training_mode.enabled {
        // Reset game state
        game_state.is_game_over = false;

        // Reset player
        if let Ok((mut transform, mut velocity, mut v_vel, mut on_ground, material_handle)) = player_query.get_single_mut() {
            transform.translation = Vec3::new(0.0, 0.0, config.player_start_height);
            velocity.0 = Vec2::ZERO;
            v_vel.0 = 0.0;
            on_ground.0 = true;

            // Reset player color
            if let Some(material) = materials.get_mut(&material_handle.0) {
                material.base_color = Color::srgb(0.3, 0.8, 0.4);
            }
        }

        // Despawn all projectiles
        for entity in projectile_query.iter() {
            commands.entity(entity).despawn();
        }

        info!("Game reset!");
    }

    // Handle quit
    if keyboard_input.just_pressed(KeyCode::Escape) {
        std::process::exit(0);
    }
}

/// Handle level changes with L key (only when not in training mode)
fn handle_level_change(
    keyboard_input: Res<ButtonInput<KeyCode>>,
    training_mode: Res<TrainingMode>,
    mut level: ResMut<Level>,
    mut config: ResMut<GameConfig>,
    mut projectile_timer: ResMut<game::projectile::ProjectileSpawnTimer>,
    mut game_state: ResMut<game::collision::GameState>,
    mut player_query: Query<
        (
            &mut Transform,
            &mut game::player::Velocity,
            &mut game::player::VerticalVelocity,
            &mut game::player::OnGround,
            &MeshMaterial3d<StandardMaterial>,
        ),
        With<game::player::Player>,
    >,
    projectile_query: Query<Entity, With<game::projectile::Projectile>>,
    mut commands: Commands,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Only allow level changes when not in training mode
    if keyboard_input.just_pressed(KeyCode::KeyL) && !training_mode.enabled {
        // Change to next level
        *level = level.next();

        // Update game config for new level
        *config = GameConfig::for_level(*level);

        // Update projectile spawn timer with new interval
        projectile_timer.timer.set_duration(std::time::Duration::from_secs_f32(config.projectile_spawn_interval));
        projectile_timer.timer.reset();

        // Reset game state
        game_state.is_game_over = false;

        // Reset player
        if let Ok((mut transform, mut velocity, mut v_vel, mut on_ground, material_handle)) = player_query.get_single_mut() {
            transform.translation = Vec3::new(0.0, 0.0, config.player_start_height);
            velocity.0 = Vec2::ZERO;
            v_vel.0 = 0.0;
            on_ground.0 = true;

            // Reset player color
            if let Some(material) = materials.get_mut(&material_handle.0) {
                material.base_color = Color::srgb(0.3, 0.8, 0.4);
            }
        }

        // Despawn all projectiles
        for entity in projectile_query.iter() {
            commands.entity(entity).despawn();
        }

        info!("Level changed to: {}", level.name());
    }
}

/// Handle RL API commands (reset, step, start/end training, set level)
fn handle_rl_commands(
    mut command_rx: NonSendMut<mpsc::UnboundedReceiver<EnvCommand>>,
    mut player_query: Query<(
        &mut Transform,
        &mut game::player::Velocity,
        &mut game::player::PlayerTilt,
        &mut game::player::VerticalVelocity,
        &game::player::OnGround,
        &MeshMaterial3d<StandardMaterial>
    ), With<game::player::Player>>,
    mut game_state: ResMut<game::collision::GameState>,
    mut env_state: ResMut<RLEnvironmentState>,
    mut control_mode: ResMut<ControlMode>,
    mut training_mode: ResMut<TrainingMode>,
    mut level: ResMut<Level>,
    mut config: ResMut<GameConfig>,
    shared_config: Res<config::SharedGameConfig>,
    mut projectile_timer: ResMut<game::projectile::ProjectileSpawnTimer>,
    projectile_query: Query<Entity, With<game::projectile::Projectile>>,
    mut commands: Commands,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Process all pending commands
    while let Ok(command) = command_rx.try_recv() {
        match command {
            EnvCommand::StartTraining => {
                training_mode.enabled = true;
                info!("Training mode ENABLED - keyboard reset (R) disabled");
            }
            EnvCommand::EndTraining => {
                training_mode.enabled = false;
                *control_mode = ControlMode::Human;
                info!("Training mode DISABLED - returning to human control");
            }
            EnvCommand::SetLevel { level: level_num } => {
                // Convert level number to Level enum
                let new_level = match level_num {
                    1 => Level::Level1,
                    2 => Level::Level2,
                    _ => {
                        warn!("Invalid level number: {}. Ignoring.", level_num);
                        continue;
                    }
                };

                // Update level and config
                *level = new_level;
                *config = GameConfig::for_level(new_level);

                // Sync shared config for API server
                let mut shared = shared_config.0.blocking_lock();
                *shared = config.clone();

                // Update projectile spawn timer with new interval
                projectile_timer.timer.set_duration(std::time::Duration::from_secs_f32(config.projectile_spawn_interval));
                projectile_timer.timer.reset();

                // Reset game state
                game_state.is_game_over = false;
                env_state.episode_steps = 0;
                env_state.total_reward = 0.0;
                env_state.last_reward = 0.0;

                // Reset player
                if let Ok((mut transform, mut velocity, mut tilt, mut v_vel, _on_ground, material_handle)) = player_query.get_single_mut() {
                    transform.translation = Vec3::new(0.0, 0.0, config.player_start_height);
                    velocity.0 = Vec2::ZERO;
                    tilt.pitch = 0.0;
                    tilt.roll = 0.0;
                    v_vel.0 = 0.0;

                    // Reset player color to green
                    if let Some(material) = materials.get_mut(&material_handle.0) {
                        material.base_color = Color::srgb(0.3, 0.8, 0.4);
                    }
                }

                // Despawn all projectiles
                for entity in projectile_query.iter() {
                    commands.entity(entity).despawn();
                }

                info!("Level set to: {}", new_level.name());
            }
            EnvCommand::Reset => {
                // Switch to RL agent control
                *control_mode = ControlMode::RLAgent;
                // Reset game state
                game_state.is_game_over = false;
                env_state.episode_steps = 0;
                env_state.total_reward = 0.0;
                env_state.last_reward = 0.0;

                // Reset player
                if let Ok((mut transform, mut velocity, mut tilt, mut v_vel, _on_ground, material_handle)) = player_query.get_single_mut() {
                    transform.translation = Vec3::new(0.0, 0.0, config.player_start_height);
                    velocity.0 = Vec2::ZERO;
                    tilt.pitch = 0.0;
                    tilt.roll = 0.0;
                    v_vel.0 = 0.0;

                    // Reset player color to green
                    if let Some(material) = materials.get_mut(&material_handle.0) {
                        material.base_color = Color::srgb(0.3, 0.8, 0.4);
                    }
                }

                // Despawn all projectiles
                for entity in projectile_query.iter() {
                    commands.entity(entity).despawn();
                }

                info!("RL Environment reset");
            }
            EnvCommand::StepDiscrete { action } => {
                // Parse and apply discrete action
                if let Ok(rl_action) = rl::action::RLAction::from_index(action) {
                    if let Ok((_, mut velocity, _, _, _, _)) = player_query.get_single_mut() {
                        rl::action::apply_action(rl_action, &mut velocity, &config);
                    }
                }

                // Increment step counter
                env_state.episode_steps += 1;
            }
            EnvCommand::StepContinuous { action, config: cont_config } => {
                // Parse and apply continuous action
                if let Ok(continuous_action) = rl::action::ContinuousAction::from_array(&action, cont_config) {
                    if let Ok((_, mut velocity, mut tilt, mut v_vel, on_ground, _)) = player_query.get_single_mut() {
                        rl::action::apply_continuous_action(continuous_action, &mut velocity, &mut tilt, &mut v_vel, &on_ground, &config);
                    }
                }

                // Increment step counter
                env_state.episode_steps += 1;
            }
            EnvCommand::Configure { level: level_num, action_space_type, sprint_multiplier, spawn_angle_degrees, observation_mode, thrower_delay_seconds, image_obs_width, image_obs_height, image_grayscale, collision_penalty, survival_reward, dodge_bonus_threshold, dodge_bonus_multiplier, projectile_speed, projectile_spawn_interval, max_projectiles, player_speed } => {
                // Update level if provided
                if let Some(level_num) = level_num {
                    let new_level = match level_num {
                        1 => Level::Level1,
                        2 => Level::Level2,
                        _ => {
                            warn!("Invalid level number: {}. Ignoring.", level_num);
                            continue;
                        }
                    };

                    // Update level
                    *level = new_level;
                    *config = GameConfig::for_level(new_level);

                    // Update projectile spawn timer with new interval
                    projectile_timer.timer.set_duration(std::time::Duration::from_secs_f32(config.projectile_spawn_interval));
                    projectile_timer.timer.reset();

                    info!("Level set to: {}", new_level.name());
                }

                // Update action space type if provided
                if let Some(action_space_str) = action_space_type {
                    let action_space_lower = action_space_str.to_lowercase();
                    let action_space = if action_space_lower == "discrete" {
                        config::ActionSpaceType::Discrete
                    } else if let Some(cont_config) = config::ContinuousActionConfig::from_str(&action_space_lower) {
                        config::ActionSpaceType::Continuous(cont_config)
                    } else {
                        warn!("Invalid action space type: {}. Ignoring.", action_space_str);
                        continue;
                    };

                    config.action_space_type = action_space;
                    info!("Action space type set to: {:?}", action_space);
                }

                // Update sprint_multiplier if provided
                if let Some(mult) = sprint_multiplier {
                    config.sprint_multiplier = mult;
                    info!("Sprint multiplier set to: {} ({}x speed at full sprint)", mult, 1.0 + mult);
                }

                // Update spawn_angle_degrees if provided
                if let Some(angle) = spawn_angle_degrees {
                    config.spawn_angle_degrees = angle;
                    info!("Spawn angle set to: ±{}° ({}° total fan)", angle, angle * 2.0);
                }

                // Update observation_mode if provided
                if let Some(obs_mode_str) = observation_mode {
                    if let Some(obs_mode) = config::ObservationMode::from_str(&obs_mode_str) {
                        config.observation_mode = obs_mode;
                        info!("Observation mode set to: {:?} ({}-dim)", obs_mode, obs_mode.observation_size());
                    } else {
                        warn!("Invalid observation mode: {}. Ignoring.", obs_mode_str);
                    }
                }

                // Update thrower_delay_seconds if provided
                if let Some(delay) = thrower_delay_seconds {
                    config.thrower_delay_seconds = delay;
                    info!("Thrower delay set to: {}s", delay);
                }

                // Update image dimensions if provided
                if let Some(width) = image_obs_width {
                    config.image_obs_width = width;
                    info!("Image observation width set to: {}", width);
                }
                if let Some(height) = image_obs_height {
                    config.image_obs_height = height;
                    info!("Image observation height set to: {}", height);
                }

                // Update image_grayscale if provided
                if let Some(grayscale) = image_grayscale {
                    config.image_grayscale = grayscale;
                    info!("Image grayscale mode set to: {}", grayscale);
                }

                // Update reward parameters if provided
                if let Some(penalty) = collision_penalty {
                    config.collision_penalty = penalty;
                    info!("Collision penalty set to: {}", penalty);
                }
                if let Some(reward) = survival_reward {
                    config.survival_reward = reward;
                    info!("Survival reward set to: {}", reward);
                }
                if let Some(threshold) = dodge_bonus_threshold {
                    config.dodge_bonus_threshold = threshold;
                    info!("Dodge bonus threshold set to: {}", threshold);
                }
                if let Some(multiplier) = dodge_bonus_multiplier {
                    config.dodge_bonus_multiplier = multiplier;
                    info!("Dodge bonus multiplier set to: {}", multiplier);
                }

                // Update level parameters if provided (override level defaults)
                if let Some(speed) = projectile_speed {
                    config.projectile_speed = speed;
                    info!("Projectile speed set to: {}", speed);
                }
                if let Some(interval) = projectile_spawn_interval {
                    config.projectile_spawn_interval = interval;
                    projectile_timer.timer.set_duration(std::time::Duration::from_secs_f32(interval));
                    projectile_timer.timer.reset();
                    info!("Projectile spawn interval set to: {}s", interval);
                }
                if let Some(max) = max_projectiles {
                    config.max_projectiles = max as usize;
                    info!("Max projectiles set to: {}", max);
                }
                if let Some(speed) = player_speed {
                    config.player_speed = speed;
                    info!("Player speed set to: {}", speed);
                }

                // Sync shared config for API server
                let mut shared = shared_config.0.blocking_lock();
                *shared = config.clone();

                // Reset game state when configuration changes
                game_state.is_game_over = false;
                env_state.episode_steps = 0;
                env_state.total_reward = 0.0;
                env_state.last_reward = 0.0;

                // Reset player
                if let Ok((mut transform, mut velocity, mut tilt, mut v_vel, _on_ground, material_handle)) = player_query.get_single_mut() {
                    transform.translation = Vec3::new(0.0, 0.0, config.player_start_height);
                    velocity.0 = Vec2::ZERO;
                    tilt.pitch = 0.0;
                    tilt.roll = 0.0;
                    v_vel.0 = 0.0;

                    // Reset player color to green
                    if let Some(material) = materials.get_mut(&material_handle.0) {
                        material.base_color = Color::srgb(0.3, 0.8, 0.4);
                    }
                }

                // Despawn all projectiles
                for entity in projectile_query.iter() {
                    commands.entity(entity).despawn();
                }

                info!("Game configuration updated via /configure endpoint");
            }
        }
    }
}

/// Update spawn arc visualization to match current config
fn update_spawn_arc(
    config: Res<GameConfig>,
    mut arc_query: Query<(&SpawnArcMarker, &mut Transform)>,
) {
    // Only update when config changes
    if !config.is_changed() {
        return;
    }

    let spawn_radius = 20.0;
    let arc_height = 0.1;
    let arc_segments = 30;
    let half_angle = config.spawn_angle_degrees.to_radians();

    for (marker, mut transform) in arc_query.iter_mut() {
        let angle = if marker.is_edge {
            // Edge markers: index 0 = left edge (-), index 1 = right edge (+)
            let sign = if marker.index == arc_segments { -1.0 } else { 1.0 };
            sign * half_angle
        } else {
            // Segment markers: evenly distributed along the arc
            let t = marker.index as f32 / (arc_segments - 1) as f32;
            -half_angle + t * 2.0 * half_angle
        };

        let x = angle.sin() * spawn_radius;
        let y = angle.cos() * spawn_radius;
        transform.translation = Vec3::new(x, y, arc_height);
    }
}

/// Update RL shared state after each frame
fn update_rl_state(
    player_query: Query<(&Transform, &game::player::Velocity), With<game::player::Player>>,
    projectile_query: Query<(&Transform, &game::projectile::ProjectileVelocity), With<game::projectile::Projectile>>,
    thrower_query: Query<&game::projectile::ThrowerIndicator>,
    player_transform_query: Query<&Transform, With<game::player::Player>>,
    projectile_transform_query: Query<&Transform, With<game::projectile::Projectile>>,
    game_state: Res<game::collision::GameState>,
    mut env_state: ResMut<RLEnvironmentState>,
    shared_state: Res<SharedEnvState>,
    game_config: Res<GameConfig>,
) {
    // Extract observation (with mode support)
    let observation = rl::observation::extract_observation_with_mode(
        game_config.observation_mode,
        game_config.thrower_delay_seconds,
        &player_query,
        &projectile_query,
        &thrower_query,
    );

    // Generate image observation if in TopDownImage mode
    if game_config.observation_mode == config::ObservationMode::TopDownImage {
        // Collect player position
        let player_pos = player_query
            .get_single()
            .map(|(t, _)| t.translation)
            .unwrap_or(Vec3::ZERO);

        // Collect projectile positions and velocities
        let projectiles: Vec<(Vec3, Vec3)> = projectile_query
            .iter()
            .map(|(t, v)| (t.translation, v.0))
            .collect();

        // Collect thrower position if present
        let thrower_pos = thrower_query
            .iter()
            .next()
            .map(|t| t.spawn_position);

        // Generate synthetic top-down image (in-place to avoid allocation)
        let mut image_buffer = shared_state.image_observation.blocking_write();
        rl::image_observation::generate_synthetic_topdown_image_into(
            &mut image_buffer,
            player_pos,
            &projectiles,
            thrower_pos,
            24.0, // Arena size
            game_config.image_obs_width,
            game_config.image_obs_height,
            game_config.image_grayscale,
        );
        drop(image_buffer); // Release lock explicitly
    }

    // Calculate reward using configurable parameters
    let reward = rl::environment::calculate_reward(&game_state, &player_transform_query, &projectile_transform_query, &game_config);

    env_state.last_reward = reward;
    env_state.total_reward += reward;

    // Check episode termination
    let done = rl::environment::is_episode_done(&game_state, &env_state);
    let truncated = rl::environment::is_episode_truncated(&env_state, 1000); // Max 1000 steps

    // Create step info
    let info = rl::environment::create_step_info(&env_state, projectile_query.iter().count());

    // Update shared state (using blocking_write since we're not in async context)
    *shared_state.observation.blocking_write() = observation;
    *shared_state.reward.blocking_write() = reward;
    *shared_state.done.blocking_write() = done;
    *shared_state.truncated.blocking_write() = truncated;
    *shared_state.info.blocking_write() = info;
    // Increment step counter to signal new observation is ready
    *shared_state.step_counter.blocking_write() += 1;
}

/// Handle RL API commands in headless mode (no materials/rendering)
fn handle_rl_commands_headless(
    mut command_rx: NonSendMut<mpsc::UnboundedReceiver<EnvCommand>>,
    mut player_query: Query<(
        &mut Transform,
        &mut game::player::Velocity,
        &mut game::player::PlayerTilt,
        &mut game::player::VerticalVelocity,
        &game::player::OnGround,
    ), With<game::player::Player>>,
    mut game_state: ResMut<game::collision::GameState>,
    mut env_state: ResMut<RLEnvironmentState>,
    mut control_mode: ResMut<ControlMode>,
    mut training_mode: ResMut<TrainingMode>,
    mut level: ResMut<Level>,
    mut config: ResMut<GameConfig>,
    shared_config: Res<config::SharedGameConfig>,
    mut projectile_timer: ResMut<game::projectile::ProjectileSpawnTimer>,
    projectile_query: Query<Entity, With<game::projectile::Projectile>>,
    mut commands: Commands,
) {
    while let Ok(command) = command_rx.try_recv() {
        match command {
            EnvCommand::StartTraining => {
                training_mode.enabled = true;
                info!("Training mode ENABLED (headless)");
            }
            EnvCommand::EndTraining => {
                training_mode.enabled = false;
                *control_mode = ControlMode::Human;
                info!("Training mode DISABLED (headless)");
            }
            EnvCommand::SetLevel { level: level_num } => {
                let new_level = match level_num {
                    1 => Level::Level1,
                    2 => Level::Level2,
                    _ => {
                        warn!("Invalid level number: {}. Ignoring.", level_num);
                        continue;
                    }
                };

                *level = new_level;
                *config = GameConfig::for_level(new_level);

                let mut shared = shared_config.0.blocking_lock();
                *shared = config.clone();

                projectile_timer.timer.set_duration(std::time::Duration::from_secs_f32(config.projectile_spawn_interval));
                projectile_timer.timer.reset();

                game_state.is_game_over = false;
                env_state.episode_steps = 0;
                env_state.total_reward = 0.0;
                env_state.last_reward = 0.0;

                if let Ok((mut transform, mut velocity, mut tilt, mut v_vel, _on_ground)) = player_query.get_single_mut() {
                    transform.translation = Vec3::new(0.0, 0.0, config.player_start_height);
                    velocity.0 = Vec2::ZERO;
                    tilt.pitch = 0.0;
                    tilt.roll = 0.0;
                    v_vel.0 = 0.0;
                }

                for entity in projectile_query.iter() {
                    commands.entity(entity).despawn();
                }

                info!("Level set to: {}", new_level.name());
            }
            EnvCommand::Reset => {
                *control_mode = ControlMode::RLAgent;
                game_state.is_game_over = false;
                env_state.episode_steps = 0;
                env_state.total_reward = 0.0;
                env_state.last_reward = 0.0;

                if let Ok((mut transform, mut velocity, mut tilt, mut v_vel, _on_ground)) = player_query.get_single_mut() {
                    transform.translation = Vec3::new(0.0, 0.0, config.player_start_height);
                    velocity.0 = Vec2::ZERO;
                    tilt.pitch = 0.0;
                    tilt.roll = 0.0;
                    v_vel.0 = 0.0;
                }

                for entity in projectile_query.iter() {
                    commands.entity(entity).despawn();
                }
            }
            EnvCommand::StepDiscrete { action } => {
                if let Ok(rl_action) = rl::action::RLAction::from_index(action) {
                    if let Ok((_, mut velocity, _, _, _)) = player_query.get_single_mut() {
                        rl::action::apply_action(rl_action, &mut velocity, &config);
                    }
                }
                env_state.episode_steps += 1;
            }
            EnvCommand::StepContinuous { action, config: cont_config } => {
                if let Ok(continuous_action) = rl::action::ContinuousAction::from_array(&action, cont_config) {
                    if let Ok((_, mut velocity, mut tilt, mut v_vel, on_ground)) = player_query.get_single_mut() {
                        rl::action::apply_continuous_action(continuous_action, &mut velocity, &mut tilt, &mut v_vel, &on_ground, &config);
                    }
                }
                env_state.episode_steps += 1;
            }
            EnvCommand::Configure { level: level_num, action_space_type, sprint_multiplier, spawn_angle_degrees, observation_mode, thrower_delay_seconds, image_obs_width, image_obs_height, image_grayscale, collision_penalty, survival_reward, dodge_bonus_threshold, dodge_bonus_multiplier, projectile_speed, projectile_spawn_interval, max_projectiles, player_speed } => {
                if let Some(level_num) = level_num {
                    let new_level = match level_num {
                        1 => Level::Level1,
                        2 => Level::Level2,
                        _ => {
                            warn!("Invalid level number: {}. Ignoring.", level_num);
                            continue;
                        }
                    };

                    *level = new_level;
                    *config = GameConfig::for_level(new_level);

                    projectile_timer.timer.set_duration(std::time::Duration::from_secs_f32(config.projectile_spawn_interval));
                    projectile_timer.timer.reset();

                    info!("Level set to: {}", new_level.name());
                }

                if let Some(action_space_str) = action_space_type {
                    let action_space_lower = action_space_str.to_lowercase();
                    let action_space = if action_space_lower == "discrete" {
                        config::ActionSpaceType::Discrete
                    } else if let Some(cont_config) = config::ContinuousActionConfig::from_str(&action_space_lower) {
                        config::ActionSpaceType::Continuous(cont_config)
                    } else {
                        warn!("Invalid action space type: {}. Ignoring.", action_space_str);
                        continue;
                    };

                    config.action_space_type = action_space;
                    info!("Action space type set to: {:?}", action_space);
                }

                if let Some(mult) = sprint_multiplier {
                    config.sprint_multiplier = mult;
                    info!("Sprint multiplier set to: {} ({}x speed at full sprint)", mult, 1.0 + mult);
                }

                if let Some(angle) = spawn_angle_degrees {
                    config.spawn_angle_degrees = angle;
                    info!("Spawn angle set to: ±{}° ({}° total fan)", angle, angle * 2.0);
                }

                // Update observation_mode if provided
                if let Some(obs_mode_str) = observation_mode {
                    if let Some(obs_mode) = config::ObservationMode::from_str(&obs_mode_str) {
                        config.observation_mode = obs_mode;
                        info!("Observation mode set to: {:?} ({}-dim)", obs_mode, obs_mode.observation_size());
                    } else {
                        warn!("Invalid observation mode: {}. Ignoring.", obs_mode_str);
                    }
                }

                // Update thrower_delay_seconds if provided
                if let Some(delay) = thrower_delay_seconds {
                    config.thrower_delay_seconds = delay;
                    info!("Thrower delay set to: {}s", delay);
                }

                // Update image dimensions if provided
                // Note: This updates the config but requires server restart to resize buffer
                if let Some(width) = image_obs_width {
                    config.image_obs_width = width;
                    info!("Image observation width set to: {}", width);
                }
                if let Some(height) = image_obs_height {
                    config.image_obs_height = height;
                    info!("Image observation height set to: {}", height);
                }

                // Update image_grayscale if provided
                if let Some(grayscale) = image_grayscale {
                    config.image_grayscale = grayscale;
                    info!("Image grayscale mode set to: {}", grayscale);
                }

                // Update reward parameters if provided
                if let Some(penalty) = collision_penalty {
                    config.collision_penalty = penalty;
                    info!("Collision penalty set to: {}", penalty);
                }
                if let Some(reward) = survival_reward {
                    config.survival_reward = reward;
                    info!("Survival reward set to: {}", reward);
                }
                if let Some(threshold) = dodge_bonus_threshold {
                    config.dodge_bonus_threshold = threshold;
                    info!("Dodge bonus threshold set to: {}", threshold);
                }
                if let Some(multiplier) = dodge_bonus_multiplier {
                    config.dodge_bonus_multiplier = multiplier;
                    info!("Dodge bonus multiplier set to: {}", multiplier);
                }

                // Update level parameters if provided (override level defaults)
                if let Some(speed) = projectile_speed {
                    config.projectile_speed = speed;
                    info!("Projectile speed set to: {}", speed);
                }
                if let Some(interval) = projectile_spawn_interval {
                    config.projectile_spawn_interval = interval;
                    projectile_timer.timer.set_duration(std::time::Duration::from_secs_f32(interval));
                    projectile_timer.timer.reset();
                    info!("Projectile spawn interval set to: {}s", interval);
                }
                if let Some(max) = max_projectiles {
                    config.max_projectiles = max as usize;
                    info!("Max projectiles set to: {}", max);
                }
                if let Some(speed) = player_speed {
                    config.player_speed = speed;
                    info!("Player speed set to: {}", speed);
                }

                let mut shared = shared_config.0.blocking_lock();
                *shared = config.clone();

                game_state.is_game_over = false;
                env_state.episode_steps = 0;
                env_state.total_reward = 0.0;
                env_state.last_reward = 0.0;

                if let Ok((mut transform, mut velocity, mut tilt, mut v_vel, _on_ground)) = player_query.get_single_mut() {
                    transform.translation = Vec3::new(0.0, 0.0, config.player_start_height);
                    velocity.0 = Vec2::ZERO;
                    tilt.pitch = 0.0;
                    tilt.roll = 0.0;
                    v_vel.0 = 0.0;
                }

                for entity in projectile_query.iter() {
                    commands.entity(entity).despawn();
                }

                info!("Game configuration updated via /configure endpoint (headless)");
            }
        }
    }
}
