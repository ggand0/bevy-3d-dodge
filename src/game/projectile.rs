use bevy::prelude::*;
use crate::config::{GameConfig, ObservationMode};
use crate::game::player::Player;

#[derive(Component)]
pub struct Projectile;

#[derive(Component)]
pub struct ProjectileVelocity(pub Vec3);

#[derive(Resource)]
pub struct ProjectileSpawnTimer {
    pub timer: Timer,
}

/// Thrower indicator - spawns before projectile to give agent anticipation info
#[derive(Component)]
pub struct ThrowerIndicator {
    /// Countdown timer until projectile spawns
    pub spawn_timer: Timer,
    /// Pre-computed spawn position for the projectile
    pub spawn_position: Vec3,
    /// Pre-computed velocity for the projectile
    pub spawn_velocity: Vec3,
}

pub struct ProjectilePlugin;

impl Plugin for ProjectilePlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup_projectile_timer)
            .add_systems(Update, (
                spawn_projectiles,
                spawn_thrower_indicators,
                process_thrower_indicators,
                move_projectiles,
                cleanup_projectiles,
            ));
    }
}

/// Headless projectile plugin (no rendering)
pub struct HeadlessProjectilePlugin;

impl Plugin for HeadlessProjectilePlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup_projectile_timer)
            .add_systems(Update, (
                spawn_projectiles_headless,
                spawn_thrower_indicators_headless,
                process_thrower_indicators_headless,
                move_projectiles,
                cleanup_projectiles,
            ));
    }
}

/// Compute spawn position and velocity for a projectile
fn compute_projectile_spawn(
    config: &GameConfig,
    player_pos: Vec3,
) -> (Vec3, Vec3) {
    let (spawn_x, spawn_y) = if config.random_spawn_position {
        let half_angle_rad = config.spawn_angle_degrees.to_radians();
        let angle = (rand::random::<f32>() - 0.5) * 2.0 * half_angle_rad;
        let radius = config.projectile_spawn_distance;
        let x = angle.sin() * radius;
        let y = angle.cos() * radius;
        (x, y)
    } else {
        let x = rand::random::<f32>() * 3.0 - 1.5;
        let y = config.projectile_spawn_distance;
        (x, y)
    };

    let (spawn_z, flight_time) = if config.random_spawn_position {
        (1.5, 0.8)
    } else {
        (2.5, 2.0)
    };

    let target_x = player_pos.x;
    let target_y = player_pos.y;
    let target_z = config.player_start_height;

    let dx = target_x - spawn_x;
    let dy = target_y - spawn_y;
    let dz = target_z - spawn_z;

    let vx = dx / flight_time;
    let vy = dy / flight_time;
    let gravity = 9.8;
    let vz = (dz + 0.5 * gravity * flight_time * flight_time) / flight_time;

    let spawn_pos = Vec3::new(spawn_x, spawn_y, spawn_z);
    let spawn_vel = Vec3::new(vx, vy, vz);

    (spawn_pos, spawn_vel)
}

fn setup_projectile_timer(mut commands: Commands, config: Res<GameConfig>) {
    commands.insert_resource(ProjectileSpawnTimer {
        timer: Timer::from_seconds(config.projectile_spawn_interval, TimerMode::Repeating),
    });
}

/// Spawn projectiles directly (Standard observation mode only)
fn spawn_projectiles(
    mut commands: Commands,
    mut timer: ResMut<ProjectileSpawnTimer>,
    time: Res<Time>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    config: Res<GameConfig>,
    player_query: Query<&Transform, With<Player>>,
) {
    // Only spawn directly in Standard mode
    if config.observation_mode != ObservationMode::Standard {
        return;
    }

    timer.timer.tick(time.delta());

    if timer.timer.just_finished() {
        if let Ok(player_transform) = player_query.get_single() {
            let (spawn_pos, spawn_vel) = compute_projectile_spawn(&config, player_transform.translation);

            commands.spawn((
                Mesh3d(meshes.add(Sphere::new(0.3))),
                MeshMaterial3d(materials.add(StandardMaterial {
                    base_color: Color::srgb(0.9, 0.2, 0.2),
                    perceptual_roughness: 0.3,
                    metallic: 0.0,
                    reflectance: 0.45,
                    ..default()
                })),
                Transform::from_translation(spawn_pos),
                Projectile,
                ProjectileVelocity(spawn_vel),
            ));
        }
    }
}

/// Spawn thrower indicators (WithThrowerIndicator mode only)
fn spawn_thrower_indicators(
    mut commands: Commands,
    mut timer: ResMut<ProjectileSpawnTimer>,
    time: Res<Time>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    config: Res<GameConfig>,
    player_query: Query<&Transform, With<Player>>,
) {
    // Only spawn indicators in WithThrowerIndicator mode
    if config.observation_mode != ObservationMode::WithThrowerIndicator {
        return;
    }

    timer.timer.tick(time.delta());

    if timer.timer.just_finished() {
        if let Ok(player_transform) = player_query.get_single() {
            let (spawn_pos, spawn_vel) = compute_projectile_spawn(&config, player_transform.translation);

            commands.spawn((
                Mesh3d(meshes.add(Sphere::new(0.2))),
                MeshMaterial3d(materials.add(StandardMaterial {
                    base_color: Color::srgb(1.0, 0.5, 0.0), // Orange indicator
                    emissive: LinearRgba::new(1.0, 0.3, 0.0, 1.0), // Glowing
                    perceptual_roughness: 0.5,
                    metallic: 0.0,
                    ..default()
                })),
                Transform::from_translation(spawn_pos),
                ThrowerIndicator {
                    spawn_timer: Timer::from_seconds(config.thrower_delay_seconds, TimerMode::Once),
                    spawn_position: spawn_pos,
                    spawn_velocity: spawn_vel,
                },
            ));
        }
    }
}

/// Process thrower indicators - spawn projectile when timer expires
fn process_thrower_indicators(
    mut commands: Commands,
    time: Res<Time>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut query: Query<(Entity, &mut ThrowerIndicator)>,
) {
    for (entity, mut indicator) in query.iter_mut() {
        indicator.spawn_timer.tick(time.delta());

        if indicator.spawn_timer.just_finished() {
            // Spawn the projectile
            commands.spawn((
                Mesh3d(meshes.add(Sphere::new(0.3))),
                MeshMaterial3d(materials.add(StandardMaterial {
                    base_color: Color::srgb(0.9, 0.2, 0.2),
                    perceptual_roughness: 0.3,
                    metallic: 0.0,
                    reflectance: 0.45,
                    ..default()
                })),
                Transform::from_translation(indicator.spawn_position),
                Projectile,
                ProjectileVelocity(indicator.spawn_velocity),
            ));

            // Despawn the indicator
            commands.entity(entity).despawn();
        }
    }
}

/// Spawn projectiles without rendering components (for headless mode, Standard mode only)
fn spawn_projectiles_headless(
    mut commands: Commands,
    mut timer: ResMut<ProjectileSpawnTimer>,
    time: Res<Time>,
    config: Res<GameConfig>,
    player_query: Query<&Transform, With<Player>>,
) {
    // Only spawn directly in Standard mode
    if config.observation_mode != ObservationMode::Standard {
        return;
    }

    timer.timer.tick(time.delta());

    if timer.timer.just_finished() {
        if let Ok(player_transform) = player_query.get_single() {
            let (spawn_pos, spawn_vel) = compute_projectile_spawn(&config, player_transform.translation);

            commands.spawn((
                Transform::from_translation(spawn_pos),
                Projectile,
                ProjectileVelocity(spawn_vel),
            ));
        }
    }
}

/// Spawn thrower indicators in headless mode (WithThrowerIndicator mode only)
fn spawn_thrower_indicators_headless(
    mut commands: Commands,
    mut timer: ResMut<ProjectileSpawnTimer>,
    time: Res<Time>,
    config: Res<GameConfig>,
    player_query: Query<&Transform, With<Player>>,
) {
    // Only spawn indicators in WithThrowerIndicator mode
    if config.observation_mode != ObservationMode::WithThrowerIndicator {
        return;
    }

    // Multiple indicators can exist simultaneously
    // (e.g., spawn_interval=0.5s with thrower_delay=1.0s = up to 2 indicators in flight)

    timer.timer.tick(time.delta());

    if timer.timer.just_finished() {
        if let Ok(player_transform) = player_query.get_single() {
            let (spawn_pos, spawn_vel) = compute_projectile_spawn(&config, player_transform.translation);

            commands.spawn((
                Transform::from_translation(spawn_pos),
                ThrowerIndicator {
                    spawn_timer: Timer::from_seconds(config.thrower_delay_seconds, TimerMode::Once),
                    spawn_position: spawn_pos,
                    spawn_velocity: spawn_vel,
                },
            ));
        }
    }
}

/// Process thrower indicators in headless mode - spawn projectile when timer expires
fn process_thrower_indicators_headless(
    mut commands: Commands,
    time: Res<Time>,
    mut query: Query<(Entity, &mut ThrowerIndicator)>,
) {
    for (entity, mut indicator) in query.iter_mut() {
        indicator.spawn_timer.tick(time.delta());

        if indicator.spawn_timer.just_finished() {
            // Spawn the projectile without rendering
            commands.spawn((
                Transform::from_translation(indicator.spawn_position),
                Projectile,
                ProjectileVelocity(indicator.spawn_velocity),
            ));

            // Despawn the indicator
            commands.entity(entity).despawn();
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
