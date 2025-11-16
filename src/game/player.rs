use bevy::prelude::*;
use crate::config::GameConfig;

#[derive(Component)]
pub struct Player;

#[derive(Component)]
pub struct Velocity(pub Vec2);

pub struct PlayerPlugin;

impl Plugin for PlayerPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn_player)
            .add_systems(Update, (player_movement, apply_velocity));
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

fn apply_velocity(
    mut query: Query<(&mut Transform, &Velocity)>,
    time: Res<Time>,
) {
    for (mut transform, velocity) in query.iter_mut() {
        transform.translation.x += velocity.0.x * time.delta_seconds();
        transform.translation.y += velocity.0.y * time.delta_seconds();

        // Clamp player position to reasonable bounds
        transform.translation.x = transform.translation.x.clamp(-15.0, 15.0);
        transform.translation.y = transform.translation.y.clamp(-15.0, 15.0);
    }
}
