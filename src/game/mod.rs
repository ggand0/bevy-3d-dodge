pub mod camera;
pub mod collision;
pub mod player;
pub mod projectile;

use bevy::prelude::*;

/// Full game plugin with rendering (for windowed mode)
pub struct GamePlugin;

impl Plugin for GamePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            player::PlayerPlugin,
            projectile::ProjectilePlugin,
            camera::CameraPlugin,
            collision::CollisionPlugin,
        ));
    }
}

/// Headless game plugin without rendering (for training mode)
pub struct HeadlessGamePlugin;

impl Plugin for HeadlessGamePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            player::HeadlessPlayerPlugin,
            projectile::HeadlessProjectilePlugin,
            collision::HeadlessCollisionPlugin,
            // No CameraPlugin in headless mode
        ));
    }
}
