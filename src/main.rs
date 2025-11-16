mod config;
mod game;

use bevy::prelude::*;
use config::GameConfig;

fn main() {
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
        .add_plugins(game::GamePlugin)
        .add_systems(Startup, setup_scene)
        .add_systems(Update, (handle_reset, update_ui))
        .run();
}

fn setup_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Ground plane (Isaac Sim style)
    commands.spawn(PbrBundle {
        mesh: meshes.add(Mesh::from(Rectangle::new(50.0, 50.0))),
        material: materials.add(StandardMaterial {
            base_color: Color::srgb(0.40, 0.60, 0.90), // More blue
            perceptual_roughness: 0.8,
            metallic: 0.0,
            cull_mode: None,
            ..default()
        }),
        transform: Transform::from_xyz(0.0, 0.0, 0.0),
        ..default()
    });

    // Create grid lines using thin boxes
    let grid_size = 50;
    let grid_spacing = 1.0;
    let line_thickness = 0.02;
    let line_height = 0.001;

    let grid_material = materials.add(StandardMaterial {
        base_color: Color::srgba(1.0, 1.0, 1.0, 0.5), // White grid lines
        alpha_mode: AlphaMode::Blend,
        perceptual_roughness: 0.9,
        metallic: 0.0,
        ..default()
    });

    // Create grid lines parallel to X axis
    for i in -(grid_size / 2)..=(grid_size / 2) {
        let y = i as f32 * grid_spacing;
        commands.spawn(PbrBundle {
            mesh: meshes.add(Cuboid::new(grid_size as f32, line_thickness, line_height)),
            material: grid_material.clone(),
            transform: Transform::from_xyz(0.0, y, line_height / 2.0),
            ..default()
        });
    }

    // Create grid lines parallel to Y axis
    for i in -(grid_size / 2)..=(grid_size / 2) {
        let x = i as f32 * grid_spacing;
        commands.spawn(PbrBundle {
            mesh: meshes.add(Cuboid::new(line_thickness, grid_size as f32, line_height)),
            material: grid_material.clone(),
            transform: Transform::from_xyz(x, 0.0, line_height / 2.0),
            ..default()
        });
    }

    // Softer directional light for subtle shadows (mimics skylights/windows)
    commands.spawn(DirectionalLightBundle {
        directional_light: DirectionalLight {
            illuminance: 5000.0, // Further reduced since we'll add point lights
            shadows_enabled: true,
            ..default()
        },
        transform: Transform::from_xyz(15.0, -10.0, 25.0).looking_at(Vec3::ZERO, Vec3::Z),
        ..default()
    });

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
        commands.spawn(PointLightBundle {
            point_light: PointLight {
                intensity: 800000.0, // Bright ceiling lights
                color: Color::srgb(1.0, 0.98, 0.95), // Warm white
                radius: 20.0,
                range: 25.0,
                shadows_enabled: false, // Disable for performance, directional light handles shadows
                ..default()
            },
            transform: Transform::from_translation(*pos),
            ..default()
        });
    }

    // Ambient light (now lower since we have point lights)
    commands.insert_resource(AmbientLight {
        color: Color::srgb(0.9, 0.9, 0.95), // Slightly cool ambient
        brightness: 150.0, // Reduced since point lights provide main illumination
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
    commands.spawn(PbrBundle {
        mesh: meshes.add(Cuboid::new(zone_width, zone_line_thickness, zone_line_height)),
        material: zone_material.clone(),
        transform: Transform::from_xyz(0.0, zone_depth / 2.0, zone_line_height / 2.0),
        ..default()
    });

    // Back line (-Y side)
    commands.spawn(PbrBundle {
        mesh: meshes.add(Cuboid::new(zone_width, zone_line_thickness, zone_line_height)),
        material: zone_material.clone(),
        transform: Transform::from_xyz(0.0, -zone_depth / 2.0, zone_line_height / 2.0),
        ..default()
    });

    // Left line (-X side)
    commands.spawn(PbrBundle {
        mesh: meshes.add(Cuboid::new(zone_line_thickness, zone_depth, zone_line_height)),
        material: zone_material.clone(),
        transform: Transform::from_xyz(-zone_width / 2.0, 0.0, zone_line_height / 2.0),
        ..default()
    });

    // Right line (+X side)
    commands.spawn(PbrBundle {
        mesh: meshes.add(Cuboid::new(zone_line_thickness, zone_depth, zone_line_height)),
        material: zone_material.clone(),
        transform: Transform::from_xyz(zone_width / 2.0, 0.0, zone_line_height / 2.0),
        ..default()
    });

    // Coordinate axes visualization (hidden by default, shown in debug mode)
    let axis_length = 5.0;
    let axis_thickness = 0.1;
    let arrow_head_size = 0.3;

    // X axis (Red) - pointing in +X direction
    commands.spawn((
        PbrBundle {
        mesh: meshes.add(Cuboid::new(axis_length, axis_thickness, axis_thickness)),
        material: materials.add(StandardMaterial {
            base_color: Color::srgb(1.0, 0.0, 0.0),
            emissive: Color::srgb(0.5, 0.0, 0.0).into(),
            unlit: true,
            ..default()
        }),
        transform: Transform::from_xyz(axis_length / 2.0, 0.0, 0.1),
        visibility: Visibility::Hidden,
        ..default()
    },
        CoordinateAxis,
    ));
    // X axis arrow head
    commands.spawn((
        PbrBundle {
        mesh: meshes.add(Cuboid::new(arrow_head_size, arrow_head_size * 2.0, arrow_head_size)),
        material: materials.add(StandardMaterial {
            base_color: Color::srgb(1.0, 0.0, 0.0),
            emissive: Color::srgb(0.5, 0.0, 0.0).into(),
            unlit: true,
            ..default()
        }),
        transform: Transform::from_xyz(axis_length + arrow_head_size / 2.0, 0.0, 0.1)
            .with_rotation(Quat::from_rotation_z(std::f32::consts::FRAC_PI_4)),
        visibility: Visibility::Hidden,
        ..default()
    },
        CoordinateAxis,
    ));

    // Y axis (Green) - pointing in +Y direction
    commands.spawn((
        PbrBundle {
        mesh: meshes.add(Cuboid::new(axis_thickness, axis_length, axis_thickness)),
        material: materials.add(StandardMaterial {
            base_color: Color::srgb(0.0, 1.0, 0.0),
            emissive: Color::srgb(0.0, 0.5, 0.0).into(),
            unlit: true,
            ..default()
        }),
        transform: Transform::from_xyz(0.0, axis_length / 2.0, 0.1),
        visibility: Visibility::Hidden,
        ..default()
    },
        CoordinateAxis,
    ));
    // Y axis arrow head
    commands.spawn((
        PbrBundle {
        mesh: meshes.add(Cuboid::new(arrow_head_size * 2.0, arrow_head_size, arrow_head_size)),
        material: materials.add(StandardMaterial {
            base_color: Color::srgb(0.0, 1.0, 0.0),
            emissive: Color::srgb(0.0, 0.5, 0.0).into(),
            unlit: true,
            ..default()
        }),
        transform: Transform::from_xyz(0.0, axis_length + arrow_head_size / 2.0, 0.1)
            .with_rotation(Quat::from_rotation_z(std::f32::consts::FRAC_PI_4)),
        visibility: Visibility::Hidden,
        ..default()
    },
        CoordinateAxis,
    ));

    // Z axis (Blue) - pointing in +Z direction
    commands.spawn((
        PbrBundle {
        mesh: meshes.add(Cuboid::new(axis_thickness, axis_thickness, axis_length)),
        material: materials.add(StandardMaterial {
            base_color: Color::srgb(0.0, 0.0, 1.0),
            emissive: Color::srgb(0.0, 0.0, 0.5).into(),
            unlit: true,
            ..default()
        }),
        transform: Transform::from_xyz(0.0, 0.0, axis_length / 2.0),
        visibility: Visibility::Hidden,
        ..default()
    },
        CoordinateAxis,
    ));
    // Z axis arrow head (cone-like)
    commands.spawn((
        PbrBundle {
        mesh: meshes.add(Cuboid::new(arrow_head_size, arrow_head_size, arrow_head_size * 2.0)),
        material: materials.add(StandardMaterial {
            base_color: Color::srgb(0.0, 0.0, 1.0),
            emissive: Color::srgb(0.0, 0.0, 0.5).into(),
            unlit: true,
            ..default()
        }),
        transform: Transform::from_xyz(0.0, 0.0, axis_length + arrow_head_size),
        visibility: Visibility::Hidden,
        ..default()
    },
        CoordinateAxis,
    ));

    // UI Text
    commands.spawn(
        TextBundle::from_section(
            "WASD: Move | Space: Jump | R: Reset | F1: Camera Debug | ESC: Quit",
            TextStyle {
                font_size: 20.0,
                color: Color::WHITE,
                ..default()
            },
        )
        .with_style(Style {
            position_type: PositionType::Absolute,
            top: Val::Px(10.0),
            left: Val::Px(10.0),
            ..default()
        }),
    );

    // Camera debug help text (initially hidden)
    commands.spawn((
        TextBundle::from_section(
            "Camera Debug: MMB+Drag: Pan | Scroll: Zoom | RMB+Drag: Look | UO: Up/Down",
            TextStyle {
                font_size: 16.0,
                color: Color::srgb(1.0, 1.0, 0.0),
                ..default()
            },
        )
        .with_style(Style {
            position_type: PositionType::Absolute,
            top: Val::Px(35.0),
            left: Val::Px(10.0),
            display: Display::None,
            ..default()
        }),
        CameraDebugText,
    ));

    // Game over text (initially hidden)
    commands.spawn((
        TextBundle::from_section(
            "GAME OVER! Press R to restart",
            TextStyle {
                font_size: 40.0,
                color: Color::srgb(1.0, 0.2, 0.2),
                ..default()
            },
        )
        .with_style(Style {
            position_type: PositionType::Absolute,
            top: Val::Px(300.0),
            left: Val::Px(400.0),
            display: Display::None,
            ..default()
        }),
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
    mut game_over_query: Query<&mut Style, (With<GameOverText>, Without<CameraDebugText>)>,
    mut debug_text_query: Query<&mut Style, (With<CameraDebugText>, Without<GameOverText>)>,
    mut axis_query: Query<&mut Visibility, With<CoordinateAxis>>,
) {
    if let Ok(mut style) = game_over_query.get_single_mut() {
        style.display = if game_state.is_game_over {
            Display::Flex
        } else {
            Display::None
        };
    }

    if let Ok(mut style) = debug_text_query.get_single_mut() {
        style.display = if debug_mode.enabled {
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
            &Handle<StandardMaterial>,
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
            if let Some(material) = materials.get_mut(&*material_handle) {
                material.base_color = Color::srgb(0.2, 0.5, 0.9);
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
