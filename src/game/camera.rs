use bevy::prelude::*;
use bevy::input::mouse::{MouseMotion, MouseWheel};
use bevy::core_pipeline::Skybox;
use bevy::render::camera::{Exposure, PhysicalCameraParameters};

#[derive(Component)]
pub struct DebugCamera;

#[derive(Resource)]
pub struct CameraDebugMode {
    pub enabled: bool,
}

impl Default for CameraDebugMode {
    fn default() -> Self {
        Self { enabled: false }
    }
}

pub struct CameraPlugin;

impl Plugin for CameraPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<CameraDebugMode>()
            .add_systems(Startup, spawn_camera)
            .add_systems(Update, (toggle_debug_mode, debug_camera_controls));
    }
}

fn spawn_camera(mut commands: Commands, asset_server: Res<AssetServer>) {
    // Load environment map texture (using specular map for skybox since it's sharper)
    let skybox_handle = asset_server.load("textures/pisa_specular_rgb9e5_zstd.ktx2");

    // Rotation to convert from Y-up (typical for graphics) to Z-up (our scene)
    // Rotate +90 degrees around X axis to align sky with +Z direction
    let env_rotation = Quat::from_rotation_x(std::f32::consts::FRAC_PI_2);

    commands.spawn((
        Camera3d::default(),
        Camera {
            hdr: true, // Enable HDR for better lighting range
            ..default()
        },
        Exposure::from_physical_camera(PhysicalCameraParameters {
            aperture_f_stops: 8.0,        // f/8 aperture (common for indoor sports)
            shutter_speed_s: 1.0 / 125.0, // 1/125s shutter (freezes motion)
            sensitivity_iso: 400.0,        // ISO 400 (good for indoor gym lighting)
            sensor_height: 0.01866,        // Standard full-frame sensor
        }),
        Transform::from_xyz(0.0, -15.0, 10.0)
            .looking_at(Vec3::new(0.0, 0.0, 1.0), Vec3::Z),
        // Skybox - visible background environment
        Skybox {
            image: skybox_handle.clone(),
            brightness: 500.0, // Moderate brightness for indoor setting
            rotation: env_rotation,
            ..default()
        },
        // Environment map for realistic ambient lighting and reflections
        EnvironmentMapLight {
            diffuse_map: asset_server.load("textures/pisa_diffuse_rgb9e5_zstd.ktx2"),
            specular_map: skybox_handle,
            intensity: 300.0, // Moderate intensity for indoor gym setting
            rotation: env_rotation,
        },
        DebugCamera,
    ));
}

fn toggle_debug_mode(
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mut debug_mode: ResMut<CameraDebugMode>,
) {
    if keyboard_input.just_pressed(KeyCode::F1) {
        debug_mode.enabled = !debug_mode.enabled;
        info!("Camera debug mode: {}", if debug_mode.enabled { "ON" } else { "OFF" });
    }
}

fn debug_camera_controls(
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mouse_button: Res<ButtonInput<MouseButton>>,
    mut mouse_motion: EventReader<MouseMotion>,
    mut mouse_wheel: EventReader<MouseWheel>,
    debug_mode: Res<CameraDebugMode>,
    mut camera_query: Query<&mut Transform, With<DebugCamera>>,
    time: Res<Time>,
) {
    if !debug_mode.enabled {
        return;
    }

    if let Ok(mut transform) = camera_query.get_single_mut() {
        let rotate_speed = 1.0;
        let mouse_sensitivity = 0.003;
        let pan_sensitivity = 0.05;
        let zoom_sensitivity = 1.0;
        let dt = time.delta_secs();

        // Mouse wheel zoom (scroll)
        for wheel in mouse_wheel.read() {
            let zoom = wheel.y * zoom_sensitivity;
            let forward = transform.forward();
            transform.translation += forward * zoom;
        }

        // Middle mouse button drag for panning
        if mouse_button.pressed(MouseButton::Middle) {
            for motion in mouse_motion.read() {
                let pan_x = -motion.delta.x * pan_sensitivity;
                let pan_y = motion.delta.y * pan_sensitivity;

                // Pan left/right and up/down in camera space
                let right = transform.right();
                transform.translation += right * pan_x;
                transform.translation += Vec3::Z * pan_y;
            }
        }

        // Right mouse button drag for rotation
        if mouse_button.pressed(MouseButton::Right) {
            for motion in mouse_motion.read() {
                let yaw = -motion.delta.x * mouse_sensitivity;
                let pitch = -motion.delta.y * mouse_sensitivity;

                // Rotate around Z axis (yaw)
                transform.rotate_z(yaw);

                // Rotate around local X axis (pitch)
                transform.rotate_local_x(pitch);
            }
        }

        // Camera rotation (Arrow keys when in debug mode)
        if keyboard_input.pressed(KeyCode::ArrowUp) {
            transform.rotate_local_x(rotate_speed * dt);
        }
        if keyboard_input.pressed(KeyCode::ArrowDown) {
            transform.rotate_local_x(-rotate_speed * dt);
        }
        if keyboard_input.pressed(KeyCode::ArrowLeft) {
            transform.rotate_z(rotate_speed * dt);
        }
        if keyboard_input.pressed(KeyCode::ArrowRight) {
            transform.rotate_z(-rotate_speed * dt);
        }

        // U/O keys for vertical movement
        if keyboard_input.pressed(KeyCode::KeyU) {
            transform.translation += Vec3::Z * 10.0 * dt;
        }
        if keyboard_input.pressed(KeyCode::KeyO) {
            transform.translation -= Vec3::Z * 10.0 * dt;
        }
    }
}
