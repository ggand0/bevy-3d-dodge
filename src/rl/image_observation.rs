//! Top-down image observation for CNN-based RL agents.
//!
//! Renders a 256x256 RGB top-down view of the arena for use as observation.

use bevy::prelude::*;
use bevy::render::camera::RenderTarget;
use bevy::render::render_resource::{
    Extent3d, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages,
};
use bevy::render::view::RenderLayers;

use crate::config::{IMAGE_OBS_HEIGHT, IMAGE_OBS_WIDTH};

/// Render layer for top-down camera (separate from main camera)
pub const TOPDOWN_RENDER_LAYER: usize = 1;

/// Marker component for the top-down observation camera
#[derive(Component)]
pub struct TopDownCamera;

/// Marker component for entities visible to top-down camera
#[derive(Component)]
pub struct TopDownVisible;

/// Resource to store the rendered image data
#[derive(Resource, Default)]
pub struct TopDownImageBuffer {
    /// Raw RGBA pixel data (256x256x4 = 262144 bytes)
    pub pixels: Vec<u8>,
    /// Whether the buffer has been updated this frame
    pub updated: bool,
}

/// Handle to the render target image
#[derive(Resource)]
pub struct TopDownRenderTarget {
    pub image_handle: Handle<Image>,
}

/// Setup the top-down observation camera and render target
pub fn setup_topdown_camera(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
) {
    // Create render target image
    let size = Extent3d {
        width: IMAGE_OBS_WIDTH,
        height: IMAGE_OBS_HEIGHT,
        depth_or_array_layers: 1,
    };

    let mut image = Image {
        texture_descriptor: TextureDescriptor {
            label: Some("topdown_render_target"),
            size,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8UnormSrgb,
            mip_level_count: 1,
            sample_count: 1,
            usage: TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_SRC
                | TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        },
        ..default()
    };
    image.resize(size);

    let image_handle = images.add(image);

    // Store render target handle
    commands.insert_resource(TopDownRenderTarget {
        image_handle: image_handle.clone(),
    });

    // Initialize image buffer
    commands.insert_resource(TopDownImageBuffer {
        pixels: vec![0u8; (IMAGE_OBS_WIDTH * IMAGE_OBS_HEIGHT * 4) as usize],
        updated: false,
    });

    // Create orthographic camera looking straight down at the arena
    // Arena is roughly -10 to 10 on x/y, so we use ortho scale to fit it
    let arena_size = 24.0; // Slightly larger than play area to show edges

    commands.spawn((
        Camera3d::default(),
        Camera {
            target: RenderTarget::Image(image_handle),
            order: -1, // Render before main camera
            clear_color: ClearColorConfig::Custom(Color::srgb(0.1, 0.1, 0.15)), // Dark background
            ..default()
        },
        Projection::Orthographic(OrthographicProjection {
            scaling_mode: bevy::render::camera::ScalingMode::Fixed {
                width: arena_size,
                height: arena_size,
            },
            ..OrthographicProjection::default_3d()
        }),
        Transform::from_xyz(0.0, 0.0, 30.0) // High above, looking down
            .looking_at(Vec3::ZERO, Vec3::Y), // Y-up in screen space
        RenderLayers::layer(TOPDOWN_RENDER_LAYER),
        TopDownCamera,
    ));

    info!("Top-down observation camera initialized ({}x{})", IMAGE_OBS_WIDTH, IMAGE_OBS_HEIGHT);
}

/// System to copy rendered image to buffer for API access
/// Note: This requires async readback which is complex in Bevy.
/// For now, we'll use a simpler approach: generate a synthetic top-down view.
pub fn update_topdown_buffer(
    _render_target: Option<Res<TopDownRenderTarget>>,
    _images: Res<Assets<Image>>,
    _buffer: Option<ResMut<TopDownImageBuffer>>,
) {
    // TODO: Implement GPU readback
    // This is complex in Bevy and requires render graph modifications.
    // For Phase 1, we'll use a synthetic renderer instead.
}

/// Generate a synthetic top-down image from game state (CPU-based)
/// This is simpler than GPU readback and works in headless mode too.
pub fn generate_synthetic_topdown_image(
    player_pos: Vec3,
    projectile_positions: &[(Vec3, Vec3)], // (position, velocity)
    thrower_pos: Option<Vec3>,
    arena_size: f32,
) -> Vec<u8> {
    let width = IMAGE_OBS_WIDTH as usize;
    let height = IMAGE_OBS_HEIGHT as usize;
    let mut pixels = vec![0u8; width * height * 3]; // RGB

    let half_arena = arena_size / 2.0;

    // Helper to convert world coords to pixel coords
    let world_to_pixel = |x: f32, y: f32| -> Option<(usize, usize)> {
        // Map world coords (-half_arena to +half_arena) to pixel coords (0 to width/height)
        let px = ((x + half_arena) / arena_size * width as f32) as i32;
        let py = ((half_arena - y) / arena_size * height as f32) as i32; // Flip Y
        if px >= 0 && px < width as i32 && py >= 0 && py < height as i32 {
            Some((px as usize, py as usize))
        } else {
            None
        }
    };

    // Draw background (dark gray arena)
    for i in 0..pixels.len() / 3 {
        pixels[i * 3] = 30;     // R
        pixels[i * 3 + 1] = 30; // G
        pixels[i * 3 + 2] = 40; // B
    }

    // Draw play zone boundary (yellow rectangle)
    let zone_half_w = 5.0;
    let zone_half_h = 4.0;
    for edge_y in [-zone_half_h, zone_half_h] {
        for x in -50..=50 {
            let wx = x as f32 * 0.1;
            if wx.abs() <= zone_half_w {
                if let Some((px, py)) = world_to_pixel(wx, edge_y) {
                    let idx = (py * width + px) * 3;
                    if idx + 2 < pixels.len() {
                        pixels[idx] = 200;     // R
                        pixels[idx + 1] = 200; // G
                        pixels[idx + 2] = 50;  // B (yellow)
                    }
                }
            }
        }
    }
    for edge_x in [-zone_half_w, zone_half_w] {
        for y in -40..=40 {
            let wy = y as f32 * 0.1;
            if wy.abs() <= zone_half_h {
                if let Some((px, py)) = world_to_pixel(edge_x, wy) {
                    let idx = (py * width + px) * 3;
                    if idx + 2 < pixels.len() {
                        pixels[idx] = 200;
                        pixels[idx + 1] = 200;
                        pixels[idx + 2] = 50;
                    }
                }
            }
        }
    }

    // Draw thrower indicator (orange, if present)
    if let Some(thrower) = thrower_pos {
        draw_circle(&mut pixels, width, height, thrower.x, thrower.y, 0.8, arena_size, [255, 140, 0]);
    }

    // Draw projectiles (red circles)
    for (pos, _vel) in projectile_positions {
        draw_circle(&mut pixels, width, height, pos.x, pos.y, 0.5, arena_size, [255, 50, 50]);
    }

    // Draw player (blue circle, slightly larger)
    draw_circle(&mut pixels, width, height, player_pos.x, player_pos.y, 0.6, arena_size, [50, 150, 255]);

    pixels
}

/// Helper to draw a filled circle on the image
fn draw_circle(
    pixels: &mut [u8],
    width: usize,
    height: usize,
    world_x: f32,
    world_y: f32,
    radius_world: f32,
    arena_size: f32,
    color: [u8; 3],
) {
    let half_arena = arena_size / 2.0;
    let pixels_per_unit = width as f32 / arena_size;
    let radius_px = (radius_world * pixels_per_unit) as i32;

    // Center in pixel coords
    let cx = ((world_x + half_arena) / arena_size * width as f32) as i32;
    let cy = ((half_arena - world_y) / arena_size * height as f32) as i32;

    // Draw filled circle
    for dy in -radius_px..=radius_px {
        for dx in -radius_px..=radius_px {
            if dx * dx + dy * dy <= radius_px * radius_px {
                let px = cx + dx;
                let py = cy + dy;
                if px >= 0 && px < width as i32 && py >= 0 && py < height as i32 {
                    let idx = (py as usize * width + px as usize) * 3;
                    if idx + 2 < pixels.len() {
                        pixels[idx] = color[0];
                        pixels[idx + 1] = color[1];
                        pixels[idx + 2] = color[2];
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synthetic_image_dimensions() {
        let image = generate_synthetic_topdown_image(
            Vec3::ZERO,
            &[],
            None,
            24.0,
        );
        assert_eq!(image.len(), 256 * 256 * 3);
    }

    #[test]
    fn test_synthetic_image_with_objects() {
        let projectiles = vec![
            (Vec3::new(5.0, 5.0, 1.0), Vec3::new(0.0, -1.0, 0.0)),
            (Vec3::new(-3.0, 2.0, 1.0), Vec3::new(1.0, 0.0, 0.0)),
        ];
        let image = generate_synthetic_topdown_image(
            Vec3::new(0.0, -2.0, 1.0),
            &projectiles,
            Some(Vec3::new(0.0, 10.0, 1.0)),
            24.0,
        );
        assert_eq!(image.len(), 256 * 256 * 3);
        // Image should have non-zero pixels (not all black)
        assert!(image.iter().any(|&p| p > 0));
    }
}
