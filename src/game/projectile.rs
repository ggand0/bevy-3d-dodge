use bevy::prelude::*;
use crate::config::GameConfig;

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
) {
    timer.timer.tick(time.delta());

    if timer.timer.just_finished() {
        // Spawn from the +Y side (where the thrower would be)
        let spawn_x = rand::random::<f32>() * 3.0 - 1.5; // Random X between -1.5 and 1.5 (narrow range)
        let spawn_y = config.projectile_spawn_distance;
        let spawn_z = 2.5; // Start higher for arc trajectory

        // Target a random position in the play zone
        let target_x = rand::random::<f32>() * 8.0 - 4.0; // Target X between -4 and 4
        let target_y = rand::random::<f32>() * 4.0 - 2.0; // Target Y between -2 and 2
        let target_z = config.player_start_height;

        // Calculate velocity for arc trajectory
        let dx = target_x - spawn_x;
        let dy = target_y - spawn_y;
        let dz = target_z - spawn_z;

        // Time of flight (adjust for desired arc)
        let flight_time = 2.0;

        // Initial velocity components
        let vx = dx / flight_time;
        let vy = dy / flight_time;
        // For Z, we need to account for gravity: z = z0 + vz*t - 0.5*g*t^2
        // So vz = (z - z0 + 0.5*g*t^2) / t
        let gravity = 9.8;
        let vz = (dz + 0.5 * gravity * flight_time * flight_time) / flight_time;

        commands.spawn((
            PbrBundle {
                mesh: meshes.add(Sphere::new(0.3)),
                material: materials.add(Color::srgb(0.9, 0.2, 0.2)),
                transform: Transform::from_xyz(spawn_x, spawn_y, spawn_z),
                ..default()
            },
            Projectile,
            ProjectileVelocity(Vec3::new(vx, vy, vz)),
        ));
    }
}

fn move_projectiles(
    mut query: Query<(&mut Transform, &mut ProjectileVelocity), With<Projectile>>,
    time: Res<Time>,
) {
    let gravity = 9.8;
    let dt = time.delta_seconds();

    for (mut transform, mut velocity) in query.iter_mut() {
        // Apply gravity to Z velocity
        velocity.0.z -= gravity * dt;

        // Update position
        transform.translation += velocity.0 * dt;
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
