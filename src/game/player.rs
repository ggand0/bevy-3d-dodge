use bevy::prelude::*;
use crate::config::GameConfig;

#[derive(Component)]
pub struct Player;

#[derive(Component)]
pub struct Velocity(pub Vec2);

#[derive(Component)]
pub struct VerticalVelocity(pub f32);

#[derive(Component)]
pub struct OnGround(pub bool);

pub struct PlayerPlugin;

impl Plugin for PlayerPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn_player)
            .add_systems(Update, (player_movement, player_jump, apply_velocity, apply_gravity));
    }
}

fn spawn_player(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    config: Res<GameConfig>,
) {
    commands.spawn((
        PbrBundle {
            mesh: meshes.add(Capsule3d::new(0.5, 1.0)),
            material: materials.add(Color::srgb(0.2, 0.5, 0.9)),
            transform: Transform::from_xyz(0.0, 0.0, config.player_start_height)
                .with_rotation(Quat::from_rotation_x(std::f32::consts::FRAC_PI_2)),
            ..default()
        },
        Player,
        Velocity(Vec2::ZERO),
        VerticalVelocity(0.0),
        OnGround(true),
    ));
}

fn player_movement(
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mut query: Query<&mut Velocity, With<Player>>,
    config: Res<GameConfig>,
) {
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

        if direction.length() > 0.0 {
            direction = direction.normalize();
        }

        velocity.0 = direction * config.player_speed;
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
        transform.translation.x += velocity.0.x * time.delta_seconds();
        transform.translation.y += velocity.0.y * time.delta_seconds();

        // Clamp player position to play zone bounds
        transform.translation.x = transform.translation.x.clamp(-x_bound, x_bound);
        transform.translation.y = transform.translation.y.clamp(-y_bound, y_bound);
    }
}

fn apply_gravity(
    mut query: Query<(&mut Transform, &mut VerticalVelocity, &mut OnGround), With<Player>>,
    time: Res<Time>,
    config: Res<GameConfig>,
) {
    let gravity = 20.0; // Gravity acceleration
    let dt = time.delta_seconds();

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
