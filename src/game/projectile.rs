use bevy::prelude::*;
use crate::config::GameConfig;
use crate::game::player::Player;

#[derive(Component)]
pub struct Projectile;

#[derive(Component)]
pub struct ProjectileVelocity(pub Vec3);

#[derive(Resource)]
pub struct ProjectileSpawnTimer {
    pub timer: Timer,
}

pub struct ProjectilePlugin;

impl Plugin for ProjectilePlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup_projectile_timer)
            .add_systems(Update, (spawn_projectiles, move_projectiles, cleanup_projectiles));
    }
}

fn setup_projectile_timer(mut commands: Commands, config: Res<GameConfig>) {
    commands.insert_resource(ProjectileSpawnTimer {
        timer: Timer::from_seconds(config.projectile_spawn_interval, TimerMode::Repeating),
    });
}

fn spawn_projectiles(
    mut commands: Commands,
    mut timer: ResMut<ProjectileSpawnTimer>,
    time: Res<Time>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    config: Res<GameConfig>,
    player_query: Query<&Transform, With<Player>>,
) {
    timer.timer.tick(time.delta());

    if timer.timer.just_finished() {
        // Get player position to aim at
        if let Ok(player_transform) = player_query.get_single() {
            let (spawn_x, spawn_y) = if config.random_spawn_position {
                // Level 2: Spawn from random position in a 120° fan (±60° from +Y forward)
                // Angle range: -π/3 to π/3 (-60° to +60°, centered on +Y axis)
                let angle = (rand::random::<f32>() - 0.5) * (std::f32::consts::PI / 1.5); // Random angle -π/3 to π/3
                let radius = config.projectile_spawn_distance;
                // Rotate to face +Y: use sin for x, cos for y (rotated 90° from standard)
                let x = angle.sin() * radius;
                let y = angle.cos() * radius;
                (x, y)
            } else {
                // Level 1: Spawn from the +Y side (where the thrower would be)
                let x = rand::random::<f32>() * 3.0 - 1.5; // Random X between -1.5 and 1.5
                let y = config.projectile_spawn_distance;
                (x, y)
            };

            // Level-specific trajectory settings
            let (spawn_z, flight_time) = if config.random_spawn_position {
                // Level 2: Lower, faster trajectory (like a real dodgeball throw)
                (1.5, 0.8)
            } else {
                // Level 1: Original high arc trajectory
                (2.5, 2.0)
            };

            // Target the player's current position
            let target_x = player_transform.translation.x;
            let target_y = player_transform.translation.y;
            let target_z = config.player_start_height;

            // Calculate velocity for arc trajectory
            let dx = target_x - spawn_x;
            let dy = target_y - spawn_y;
            let dz = target_z - spawn_z;

            // Initial velocity components
            let vx = dx / flight_time;
            let vy = dy / flight_time;
            // For Z, we need to account for gravity: z = z0 + vz*t - 0.5*g*t^2
            // So vz = (z - z0 + 0.5*g*t^2) / t
            let gravity = 9.8;
            let vz = (dz + 0.5 * gravity * flight_time * flight_time) / flight_time;

            commands.spawn((
                Mesh3d(meshes.add(Sphere::new(0.3))),
                MeshMaterial3d(materials.add(StandardMaterial {
                    base_color: Color::srgb(0.9, 0.2, 0.2),
                    perceptual_roughness: 0.3, // Smooth rubber ball
                    metallic: 0.0,
                    reflectance: 0.45, // Good reflectance for rubber
                    ..default()
                })),
                Transform::from_xyz(spawn_x, spawn_y, spawn_z),
                Projectile,
                ProjectileVelocity(Vec3::new(vx, vy, vz)),
            ));
        }
    }
}

fn move_projectiles(
    mut query: Query<(&mut Transform, &mut ProjectileVelocity), With<Projectile>>,
    time: Res<Time>,
) {
    let gravity = 9.8;
    let dt = time.delta_secs();
    let ground_level = 0.3; // Sphere radius to keep ball on surface
    let restitution = 0.7; // Bounce coefficient (0.7 = loses 30% energy per bounce)

    for (mut transform, mut velocity) in query.iter_mut() {
        // Apply gravity to Z velocity
        velocity.0.z -= gravity * dt;

        // Update position
        transform.translation += velocity.0 * dt;

        // Ground bounce physics
        if transform.translation.z <= ground_level && velocity.0.z < 0.0 {
            // Position correction to prevent sinking
            transform.translation.z = ground_level;

            // Reverse and dampen Z velocity (bounce)
            velocity.0.z = -velocity.0.z * restitution;

            // Apply friction to horizontal velocities when bouncing
            let friction = 0.95;
            velocity.0.x *= friction;
            velocity.0.y *= friction;
        }
    }
}

fn cleanup_projectiles(
    mut commands: Commands,
    query: Query<(Entity, &Transform), With<Projectile>>,
) {
    for (entity, transform) in query.iter() {
        // Despawn projectiles that have fallen below ground or gone too far
        if transform.translation.z < -5.0
            || transform.translation.y < -25.0
            || transform.translation.y > 30.0
            || transform.translation.x.abs() > 30.0 {
            commands.entity(entity).despawn();
        }
    }
}
