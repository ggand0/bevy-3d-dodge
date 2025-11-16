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
        let spawn_x = rand::random::<f32>() * 10.0 - 5.0; // Random X between -5 and 5
        let spawn_y = config.projectile_spawn_distance;
        let spawn_z = config.player_start_height;

        commands.spawn((
            PbrBundle {
                mesh: meshes.add(Sphere::new(0.3)),
                material: materials.add(Color::srgb(0.9, 0.2, 0.2)),
                transform: Transform::from_xyz(spawn_x, spawn_y, spawn_z),
                ..default()
            },
            Projectile,
            ProjectileVelocity(Vec3::new(0.0, -config.projectile_speed, 0.0)),
        ));
    }
}

fn move_projectiles(
    mut query: Query<(&mut Transform, &ProjectileVelocity), With<Projectile>>,
    time: Res<Time>,
) {
    for (mut transform, velocity) in query.iter_mut() {
        transform.translation += velocity.0 * time.delta_seconds();
    }
}

fn cleanup_projectiles(
    mut commands: Commands,
    query: Query<(Entity, &Transform), With<Projectile>>,
) {
    for (entity, transform) in query.iter() {
        // Despawn projectiles that have gone too far
        if transform.translation.y < -25.0 {
            commands.entity(entity).despawn();
        }
    }
}
