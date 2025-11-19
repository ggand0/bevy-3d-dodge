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
        Self { enabled: false }  // Coordinate axes hidden by default
    }
}

#[derive(Resource)]
pub struct FreeCameraMode {
    pub enabled: bool,
}

impl Default for FreeCameraMode {
    fn default() -> Self {
        Self { enabled: false }  // Free camera disabled by default
    }
}

pub struct CameraPlugin;

impl Plugin for CameraPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<CameraDebugMode>()
            .init_resource::<FreeCameraMode>()
            .add_systems(Startup, spawn_camera)
            .add_systems(Update, (toggle_debug_mode, toggle_free_camera, debug_camera_controls));
    }
}

fn spawn_camera(mut commands: Commands, asset_server: Res<AssetServer>) {
    // Load bright autumn field environment map (converted with bevy_skybox_cli)
    let skybox_handle = asset_server.load("textures/skybox.ktx2");

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
        Skybox {
            image: skybox_handle.clone(),
            brightness: 500.0,
            rotation: env_rotation,
            ..default()
        },
        EnvironmentMapLight {
            diffuse_map: asset_server.load("textures/diffuse_map.ktx2"),
            specular_map: asset_server.load("textures/specular_map.ktx2"),
            intensity: 1500.0,
            rotation: env_rotation,
        },
        DebugCamera,
    ));
}

fn toggle_debug_mode(
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mut debug_mode: ResMut<CameraDebugMode>,
) {
    if keyboard_input.just_pressed(KeyCode::F2) {
        debug_mode.enabled = !debug_mode.enabled;
        info!("Coordinate axes: {}", if debug_mode.enabled { "VISIBLE" } else { "HIDDEN" });
    }
}

fn toggle_free_camera(
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mut free_camera_mode: ResMut<FreeCameraMode>,
    mut camera_query: Query<&mut Transform, With<DebugCamera>>,
) {
    if keyboard_input.just_pressed(KeyCode::F1) {
        free_camera_mode.enabled = !free_camera_mode.enabled;

        // Reset camera to default position when exiting free camera mode
        if !free_camera_mode.enabled {
            if let Ok(mut transform) = camera_query.get_single_mut() {
                // Reset to default camera position and orientation
                transform.translation = Vec3::new(0.0, -15.0, 10.0);
                *transform = transform.looking_at(Vec3::new(0.0, 0.0, 1.0), Vec3::Z);
            }
        }

        info!("Free camera mode: {}", if free_camera_mode.enabled { "ENABLED" } else { "DISABLED" });
    }
}

fn debug_camera_controls(
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mouse_button: Res<ButtonInput<MouseButton>>,
    mut mouse_motion: EventReader<MouseMotion>,
    mut mouse_wheel: EventReader<MouseWheel>,
    mut camera_query: Query<&mut Transform, With<DebugCamera>>,
    time: Res<Time>,
    free_camera_mode: Res<FreeCameraMode>,
    mut double_click_timer: Local<Option<f64>>,
) {
    if let Ok(mut transform) = camera_query.get_single_mut() {
        let rotate_speed = 1.0;
        let pan_sensitivity = 0.05;
        let zoom_sensitivity = 1.0;
        let dt = time.delta_secs();

        // Double-click detection to reset camera
        const DOUBLE_CLICK_THRESHOLD: f64 = 0.3; // 300ms threshold for double-click
        if mouse_button.just_pressed(MouseButton::Left) {
            let current_time = time.elapsed_secs_f64();

            if let Some(last_time) = *double_click_timer {
                let time_since_last_click = current_time - last_time;

                if time_since_last_click < DOUBLE_CLICK_THRESHOLD {
                    // Double-click detected! Reset camera to default position
                    transform.translation = Vec3::new(0.0, -15.0, 10.0);
                    *transform = transform.looking_at(Vec3::new(0.0, 0.0, 1.0), Vec3::Z);
                    info!("Camera reset to default position");
                    *double_click_timer = None; // Clear timer after double-click
                } else {
                    // Single click - update timer
                    *double_click_timer = Some(current_time);
                }
            } else {
                // First click - start timer
                *double_click_timer = Some(current_time);
            }
        }

        // Mouse wheel scroll for zoom (forward/backward movement) - works in both modes
        for wheel in mouse_wheel.read() {
            let zoom_amount = wheel.y * zoom_sensitivity;
            let forward = transform.forward();
            transform.translation += forward * zoom_amount;
        }

        // Middle mouse button drag for XY plane panning - only in free camera mode
        if free_camera_mode.enabled && mouse_button.pressed(MouseButton::Middle) {
            for motion in mouse_motion.read() {
                let pan_x = -motion.delta.x * pan_sensitivity;
                let pan_y = motion.delta.y * pan_sensitivity;

                // Pan in camera's right direction (X) and world up direction (Z for vertical)
                let right = transform.right();
                transform.translation += right * pan_x;
                transform.translation += Vec3::Z * pan_y;
            }
        }

        // Left mouse button drag - orbit in default mode, first-person rotation in free camera mode
        if mouse_button.pressed(MouseButton::Left) {
            for motion in mouse_motion.read() {
                if free_camera_mode.enabled {
                    // Free camera mode: first-person rotation
                    let rotation_sensitivity = 0.003;
                    let yaw = -motion.delta.x * rotation_sensitivity;
                    let pitch = -motion.delta.y * rotation_sensitivity;

                    // Apply yaw rotation around world Z axis
                    let yaw_rotation = Quat::from_rotation_z(yaw);
                    transform.rotation = yaw_rotation * transform.rotation;

                    // Apply pitch rotation around camera's local right axis
                    let pitch_rotation = Quat::from_axis_angle(transform.right().into(), pitch);
                    transform.rotation = pitch_rotation * transform.rotation;

                    // Normalize to prevent drift
                    transform.rotation = transform.rotation.normalize();
                } else {
                    // Default mode: orbit around play zone center
                    let orbit_center = Vec3::new(0.0, 0.0, 1.0); // Center of play zone at ground level
                    let orbit_sensitivity = 0.005;

                    let yaw = -motion.delta.x * orbit_sensitivity;
                    let pitch = -motion.delta.y * orbit_sensitivity;

                    // Calculate vector from orbit center to camera
                    let offset = transform.translation - orbit_center;

                    // Apply yaw rotation (around Z axis) at orbit center
                    let yaw_rotation = Quat::from_rotation_z(yaw);
                    let rotated_offset = yaw_rotation * offset;

                    // Apply pitch rotation (around camera's right axis at orbit center)
                    let right = transform.right();
                    let pitch_rotation = Quat::from_axis_angle(*right, pitch);
                    let final_offset = pitch_rotation * rotated_offset;

                    // Update camera position and make it look at the orbit center
                    transform.translation = orbit_center + final_offset;
                    transform.look_at(orbit_center, Vec3::Z);
                }
            }
        }

        // Camera rotation (Arrow keys when in free camera mode)
        if free_camera_mode.enabled {
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
}
