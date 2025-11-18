mod config;
mod game;
mod rl;

use bevy::prelude::*;
use config::GameConfig;
use tokio::sync::mpsc;
use rl::api::{EnvCommand, SharedEnvState, start_api_server};
use rl::environment::{RLEnvironmentState, ControlMode};

fn main() {
    // Create channel for RL API commands
    let (command_tx, command_rx) = mpsc::unbounded_channel::<EnvCommand>();

    // Create shared state for RL API
    let shared_state = SharedEnvState::default();

    // Start HTTP API server on port 8000
    start_api_server(8000, shared_state.clone(), command_tx);

    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Bevy 3D Dodge - RL Training Game".to_string(),
                resolution: (1280., 720.).into(),
                ..default()
            }),
            ..default()
        }))
        .insert_resource(GameConfig::default())
        .insert_resource(RLEnvironmentState::default())
        .insert_resource(ControlMode::default())
        .insert_non_send_resource(command_rx)
        .insert_resource(shared_state)
        .add_plugins(game::GamePlugin)
        .add_systems(Startup, setup_scene)
        .add_systems(Update, (handle_reset, update_ui, handle_rl_commands, update_rl_state))
        .run();
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

    // UI Text
    commands.spawn((
        Text::new("WASD: Move | Space: Jump | R: Reset | F1: Free Cam | F2: Toggle Axes | ESC: Quit"),
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
            top: Val::Px(35.0),
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
struct CameraDebugText;

#[derive(Component)]
struct CoordinateAxis;

fn update_ui(
    game_state: Res<game::collision::GameState>,
    debug_mode: Res<game::camera::CameraDebugMode>,
    free_camera_mode: Res<game::camera::FreeCameraMode>,
    mut game_over_query: Query<&mut Node, (With<GameOverText>, Without<CameraDebugText>)>,
    mut debug_text_query: Query<&mut Node, (With<CameraDebugText>, Without<GameOverText>)>,
    mut axis_query: Query<&mut Visibility, With<CoordinateAxis>>,
) {
    if let Ok(mut node) = game_over_query.get_single_mut() {
        node.display = if game_state.is_game_over {
            Display::Flex
        } else {
            Display::None
        };
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

fn handle_reset(
    keyboard_input: Res<ButtonInput<KeyCode>>,
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
    if keyboard_input.just_pressed(KeyCode::KeyR) {
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

/// Handle RL API commands (reset, step)
fn handle_rl_commands(
    mut command_rx: NonSendMut<mpsc::UnboundedReceiver<EnvCommand>>,
    mut player_query: Query<(&mut Transform, &mut game::player::Velocity, &MeshMaterial3d<StandardMaterial>), With<game::player::Player>>,
    mut game_state: ResMut<game::collision::GameState>,
    mut env_state: ResMut<RLEnvironmentState>,
    mut control_mode: ResMut<ControlMode>,
    projectile_query: Query<Entity, With<game::projectile::Projectile>>,
    mut commands: Commands,
    config: Res<GameConfig>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Process all pending commands
    while let Ok(command) = command_rx.try_recv() {
        match command {
            EnvCommand::Reset => {
                // Switch to RL agent control
                *control_mode = ControlMode::RLAgent;
                // Reset game state
                game_state.is_game_over = false;
                env_state.episode_steps = 0;
                env_state.total_reward = 0.0;
                env_state.last_reward = 0.0;

                // Reset player
                if let Ok((mut transform, mut velocity, material_handle)) = player_query.get_single_mut() {
                    transform.translation = Vec3::new(0.0, 0.0, config.player_start_height);
                    velocity.0 = Vec2::ZERO;

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
            EnvCommand::Step { action } => {
                // Parse and apply action
                if let Ok(rl_action) = rl::action::RLAction::from_index(action) {
                    if let Ok((_, mut velocity, _)) = player_query.get_single_mut() {
                        rl::action::apply_action(rl_action, &mut velocity, &config);
                    }
                }

                // Increment step counter
                env_state.episode_steps += 1;
            }
        }
    }
}

/// Update RL shared state after each frame
fn update_rl_state(
    player_query: Query<(&Transform, &game::player::Velocity), With<game::player::Player>>,
    projectile_query: Query<(&Transform, &game::projectile::ProjectileVelocity), With<game::projectile::Projectile>>,
    player_transform_query: Query<&Transform, With<game::player::Player>>,
    projectile_transform_query: Query<&Transform, With<game::projectile::Projectile>>,
    game_state: Res<game::collision::GameState>,
    mut env_state: ResMut<RLEnvironmentState>,
    shared_state: Res<SharedEnvState>,
) {
    // Extract observation
    let observation = rl::observation::extract_observation(&player_query, &projectile_query);

    // Calculate reward
    let reward = rl::environment::calculate_reward(&game_state, &player_transform_query, &projectile_transform_query);

    env_state.last_reward = reward;
    env_state.total_reward += reward;

    // Check episode termination
    let done = rl::environment::is_episode_done(&game_state, &env_state);
    let truncated = rl::environment::is_episode_truncated(&env_state, 1000); // Max 1000 steps

    // Create step info
    let info = rl::environment::create_step_info(&env_state, projectile_query.iter().count());

    // Update shared state
    if let Ok(mut obs) = shared_state.observation.lock() {
        *obs = observation;
    }
    if let Ok(mut r) = shared_state.reward.lock() {
        *r = reward;
    }
    if let Ok(mut d) = shared_state.done.lock() {
        *d = done;
    }
    if let Ok(mut t) = shared_state.truncated.lock() {
        *t = truncated;
    }
    if let Ok(mut i) = shared_state.info.lock() {
        *i = info;
    }
}
