use bevy::prelude::*;
use crate::config::{GameConfig, PLAYER_RADIUS};

#[derive(Component)]
pub struct Player;

#[derive(Component)]
pub struct Velocity(pub Vec2);

#[derive(Component)]
pub struct VerticalVelocity(pub f32);

#[derive(Component)]
pub struct OnGround(pub bool);

/// Player body tilt for continuous action space
/// Pitch: forward/backward tilt, Roll: left/right tilt
#[derive(Component)]
pub struct PlayerTilt {
    pub pitch: f32,  // Radians, positive = tilt forward
    pub roll: f32,   // Radians, positive = tilt right
}

pub struct PlayerPlugin;

impl Plugin for PlayerPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn_player)
            .add_systems(Update, (player_movement, player_jump, apply_velocity, apply_tilt, apply_gravity));
    }
}

/// Headless player plugin (no rendering)
pub struct HeadlessPlayerPlugin;

impl Plugin for HeadlessPlayerPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn_player_headless)
            .add_systems(Update, (apply_velocity, apply_tilt, apply_gravity));
        // Note: player_movement and player_jump are skipped in headless (no keyboard input)
    }
}

fn spawn_player(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    config: Res<GameConfig>,
) {
    commands.spawn((
        Mesh3d(meshes.add(Capsule3d::new(PLAYER_RADIUS, 1.0))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.3, 0.8, 0.4),
            perceptual_roughness: 0.4, // Smooth plastic/jersey material
            metallic: 0.0,
            reflectance: 0.4, // Slight reflectance for synthetic fabric
            ..default()
        })),
        Transform::from_xyz(0.0, 0.0, config.player_start_height)
            .with_rotation(Quat::from_rotation_x(std::f32::consts::FRAC_PI_2)),
        Player,
        Velocity(Vec2::ZERO),
        VerticalVelocity(0.0),
        OnGround(true),
        PlayerTilt { pitch: 0.0, roll: 0.0 },
    ));
}

/// Spawn player without rendering components (for headless mode)
fn spawn_player_headless(
    mut commands: Commands,
    config: Res<GameConfig>,
) {
    commands.spawn((
        Transform::from_xyz(0.0, 0.0, config.player_start_height)
            .with_rotation(Quat::from_rotation_x(std::f32::consts::FRAC_PI_2)),
        Player,
        Velocity(Vec2::ZERO),
        VerticalVelocity(0.0),
        OnGround(true),
        PlayerTilt { pitch: 0.0, roll: 0.0 },
    ));
}

fn player_movement(
    keyboard_input: Res<ButtonInput<KeyCode>>,
    control_mode: Res<crate::rl::environment::ControlMode>,
    mut query: Query<&mut Velocity, With<Player>>,
    config: Res<GameConfig>,
) {
    use crate::rl::environment::ControlMode;

    // Only process keyboard input in Human control mode
    if *control_mode != ControlMode::Human {
        return;
    }

    if let Ok(mut velocity) = query.get_single_mut() {
        let mut direction = Vec2::ZERO;

        // Based on camera view at (0, -15, 10) looking at origin:
        // X axis runs left-right (right is positive)
        // Y axis runs forward-backward (away from camera is positive)
        // velocity.0.x controls X translation, velocity.0.y controls Y translation

        // A/Left = left on screen = -X
        if keyboard_input.pressed(KeyCode::KeyA) || keyboard_input.pressed(KeyCode::ArrowLeft) {
            direction.x -= 1.0;
        }
        // D/Right = right on screen = +X
        if keyboard_input.pressed(KeyCode::KeyD) || keyboard_input.pressed(KeyCode::ArrowRight) {
            direction.x += 1.0;
        }
        // W/Up = forward (away from camera) = +Y
        if keyboard_input.pressed(KeyCode::KeyW) || keyboard_input.pressed(KeyCode::ArrowUp) {
            direction.y += 1.0;
        }
        // S/Down = backward (toward camera) = -Y
        if keyboard_input.pressed(KeyCode::KeyS) || keyboard_input.pressed(KeyCode::ArrowDown) {
            direction.y -= 1.0;
        }

        // Normalize direction and apply to velocity
        if direction.length() > 0.0 {
            direction = direction.normalize();

            // Sprint with Shift key
            let sprint = if keyboard_input.pressed(KeyCode::ShiftLeft) || keyboard_input.pressed(KeyCode::ShiftRight) {
                1.0
            } else {
                0.0
            };
            let speed_multiplier = 1.0 + sprint * config.sprint_multiplier;
            let effective_speed = config.player_speed * speed_multiplier;

            velocity.0 = direction * effective_speed;
        } else {
            // No keys pressed - stop the player
            velocity.0 = Vec2::ZERO;
        }
    }
}

fn player_jump(
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mut query: Query<(&mut VerticalVelocity, &OnGround), With<Player>>,
) {
    if let Ok((mut v_vel, on_ground)) = query.get_single_mut() {
        // Jump with Space key, only when on ground
        if keyboard_input.just_pressed(KeyCode::Space) && on_ground.0 {
            v_vel.0 = 7.0; // Jump velocity
        }
    }
}

fn apply_velocity(
    mut query: Query<(&mut Transform, &Velocity)>,
    time: Res<Time>,
) {
    // Play zone boundaries (match the zone marker in main.rs)
    let zone_width = 10.0;
    let zone_depth = 8.0;
    let x_bound = zone_width / 2.0;  // ±5.0
    let y_bound = zone_depth / 2.0;  // ±4.0

    for (mut transform, velocity) in query.iter_mut() {
        transform.translation.x += velocity.0.x * time.delta_secs();
        transform.translation.y += velocity.0.y * time.delta_secs();

        // Clamp player position to play zone bounds
        transform.translation.x = transform.translation.x.clamp(-x_bound, x_bound);
        transform.translation.y = transform.translation.y.clamp(-y_bound, y_bound);
    }
}

fn apply_tilt(
    mut query: Query<(&mut Transform, &PlayerTilt), With<Player>>,
    time: Res<Time>,
) {
    const TILT_SMOOTHING: f32 = 10.0; // How quickly tilt interpolates

    for (mut transform, tilt) in query.iter_mut() {
        // Base rotation: capsule standing upright (90° around X-axis)
        let base_rotation = Quat::from_rotation_x(std::f32::consts::FRAC_PI_2);

        // Target rotation: base + tilt adjustments
        // Pitch (forward/backward): rotation around local X-axis (left-right axis)
        // Roll (left/right): rotation around local Y-axis (forward-backward axis)
        let tilt_rotation = Quat::from_rotation_y(tilt.pitch) * Quat::from_rotation_x(tilt.roll);
        let target_rotation = base_rotation * tilt_rotation;

        // Smoothly interpolate to target rotation
        transform.rotation = transform.rotation.slerp(target_rotation, TILT_SMOOTHING * time.delta_secs());
    }
}

fn apply_gravity(
    mut query: Query<(&mut Transform, &mut VerticalVelocity, &mut OnGround), With<Player>>,
    time: Res<Time>,
    config: Res<GameConfig>,
) {
    let gravity = 20.0; // Gravity acceleration
    let dt = time.delta_secs();

    for (mut transform, mut v_vel, mut on_ground) in query.iter_mut() {
        // Apply gravity
        v_vel.0 -= gravity * dt;

        // Update vertical position
        transform.translation.z += v_vel.0 * dt;

        // Ground collision (no bounce for player)
        if transform.translation.z <= config.player_start_height {
            transform.translation.z = config.player_start_height;
            v_vel.0 = 0.0; // Stop vertical velocity (no bounce)
            on_ground.0 = true;
        } else {
            on_ground.0 = false;
        }
    }
}
