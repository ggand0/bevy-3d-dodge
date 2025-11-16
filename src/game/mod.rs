pub mod camera;
pub mod collision;
pub mod player;
pub mod projectile;

use bevy::prelude::*;

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
