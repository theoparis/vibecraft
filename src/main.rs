#![feature(portable_simd)]
use bevy::input::mouse::MouseMotion;
use bevy::prelude::*;
use bevy::window::{CursorGrabMode, CursorOptions};
use bevy::{
    ecs::world::CommandQueue,
    tasks::{AsyncComputeTaskPool, Task, futures::check_ready},
};
use core::simd::prelude::*;
use std::collections::{HashMap, HashSet};
use std::io::{Read, Write};
use zstd::stream::{Decoder, Encoder};

mod mesh;
mod noise;
use crate::mesh::generate_chunk_mesh;
use crate::noise::PerlinSimd;

/// A minecraft clone implemented in Rust with Bevy.
/// Uses zstd for compression, and z-order for 16x256x16 chunk storage.

pub const CHUNK_WIDTH: usize = 16;
pub const CHUNK_HEIGHT: usize = 256;
pub const CHUNK_DEPTH: usize = 16;
pub const CHUNK_VOLUME: usize = CHUNK_WIDTH * CHUNK_HEIGHT * CHUNK_DEPTH;

/// Represents a block type in the world.
/// Using #[repr(u8)] for efficient storage and serialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum Block {
    #[default]
    Air = 0,
    Stone = 1,
    Dirt = 2,
    Grass = 3,
}

impl From<u8> for Block {
    fn from(value: u8) -> Self {
        match value {
            1 => Block::Stone,
            2 => Block::Dirt,
            3 => Block::Grass,
            _ => Block::Air,
        }
    }
}

/// A 16x256x16 chunk of blocks.
/// Uses z-order (Morton curve) indexing for spatial locality.
pub struct Chunk {
    pub blocks: Box<[Block; CHUNK_VOLUME]>,
}

impl Default for Chunk {
    fn default() -> Self {
        Self {
            blocks: Box::new([Block::Air; CHUNK_VOLUME]),
        }
    }
}

impl Chunk {
    pub fn new() -> Self {
        Self::default()
    }

    /// Gets a block at the given local coordinates.
    pub fn get_block(&self, x: usize, y: usize, z: usize) -> Block {
        self.blocks[Self::get_index(x, y, z)]
    }

    /// Sets a block at the given local coordinates.
    pub fn set_block(&mut self, x: usize, y: usize, z: usize, block: Block) {
        let index = Self::get_index(x, y, z);
        self.blocks[index] = block;
    }

    /// Converts 3D coordinates (16x256x16) to a Z-order (Morton) index.
    /// Bit pattern: Y7 Y6 Y5 Y4 Z3 X3 Y3 Z2 X2 Y2 Z1 X1 Y1 Z0 X0 Y0
    #[inline]
    pub fn get_index(x: usize, y: usize, z: usize) -> usize {
        debug_assert!(x < CHUNK_WIDTH);
        debug_assert!(y < CHUNK_HEIGHT);
        debug_assert!(z < CHUNK_DEPTH);

        let mut i = 0;

        // Interleave the first 4 bits of X, Y, Z
        for bit in 0..4 {
            i |= (y & (1 << bit)) << (2 * bit);
            i |= (x & (1 << bit)) << (2 * bit + 1);
            i |= (z & (1 << bit)) << (2 * bit + 2);
        }

        // Add the remaining 4 bits of Y (bits 4-7)
        // These are shifted to positions 12-15
        i |= (y & 0xF0) << 8;

        i
    }

    /// Compresses the chunk data using zstd.
    pub fn compress(&self) -> std::io::Result<Vec<u8>> {
        let mut encoder = Encoder::new(Vec::new(), 3)?;
        // Safety: Block is #[repr(u8)] and CHUNK_VOLUME is the exact size.
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(self.blocks.as_ptr() as *const u8, CHUNK_VOLUME) };
        encoder.write_all(bytes)?;
        encoder.finish()
    }

    /// Decompresses chunk data from zstd.
    pub fn decompress(compressed_data: &[u8]) -> std::io::Result<Self> {
        let mut decoder = Decoder::new(compressed_data)?;
        let mut chunk = Self::default();
        // Safety: Block is #[repr(u8)] and CHUNK_VOLUME is the exact size.
        let bytes: &mut [u8] = unsafe {
            std::slice::from_raw_parts_mut(chunk.blocks.as_mut_ptr() as *mut u8, CHUNK_VOLUME)
        };
        decoder.read_exact(bytes)?;
        Ok(chunk)
    }
}

/// A coordinate point for a chunk in the world (X, Z).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ChunkPos {
    pub x: i32,
    pub z: i32,
}

impl ChunkPos {
    pub fn new(x: i32, z: i32) -> Self {
        Self { x, z }
    }
}

#[derive(Component)]
pub struct FpsController {
    pub sensitivity: f32,
    pub speed: f32,
    pub velocity: Vec3,
    pub gravity: f32,
    pub jump_force: f32,
    pub grounded: bool,
}

#[derive(Component)]
pub struct ChunkEntity(pub ChunkPos);

/// A resource that stores all loaded chunks.
#[derive(Resource, Default)]
pub struct ChunkMap {
    pub chunks: HashMap<ChunkPos, Chunk>,
}

impl ChunkMap {
    /// Gets a block at global coordinates.
    pub fn get_block(&self, x: i32, y: i32, z: i32) -> Option<Block> {
        if y < 0 || y >= CHUNK_HEIGHT as i32 {
            return None;
        }

        let chunk_pos = ChunkPos::new(
            x.div_euclid(CHUNK_WIDTH as i32),
            z.div_euclid(CHUNK_DEPTH as i32),
        );
        let local_x = x.rem_euclid(CHUNK_WIDTH as i32) as usize;
        let local_z = z.rem_euclid(CHUNK_DEPTH as i32) as usize;

        self.chunks
            .get(&chunk_pos)
            .map(|chunk| chunk.get_block(local_x, y as usize, local_z))
    }

    /// Sets a block at global coordinates. Returns true if the chunk existed.
    pub fn set_block(&mut self, x: i32, y: i32, z: i32, block: Block) -> bool {
        if y < 0 || y >= CHUNK_HEIGHT as i32 {
            return false;
        }

        let chunk_pos = ChunkPos::new(
            x.div_euclid(CHUNK_WIDTH as i32),
            z.div_euclid(CHUNK_DEPTH as i32),
        );
        let local_x = x.rem_euclid(CHUNK_WIDTH as i32) as usize;
        let local_z = z.rem_euclid(CHUNK_DEPTH as i32) as usize;

        if let Some(chunk) = self.chunks.get_mut(&chunk_pos) {
            chunk.set_block(local_x, y as usize, local_z, block);
            true
        } else {
            false
        }
    }
}

#[derive(Resource, Default)]
struct LoadingChunks(HashSet<ChunkPos>);

// Async chunk generation

#[derive(Component)]
pub struct ComputeChunk(pub Task<CommandQueue>);

fn generate_chunk(pos: ChunkPos, noise: &PerlinSimd) -> Chunk {
    let mut chunk = Chunk::new();

    // Use SIMD to generate heightmap for the chunk
    // We process 8 columns at a time
    for z in 0..CHUNK_DEPTH {
        let world_z = (pos.z * CHUNK_DEPTH as i32 + z as i32) as f32;
        let z_vec = f32x8::splat(world_z);

        for x_base in (0..CHUNK_WIDTH).step_by(8) {
            let x_vec = f32x8::from_array([
                (pos.x * CHUNK_WIDTH as i32 + x_base as i32) as f32,
                (pos.x * CHUNK_WIDTH as i32 + (x_base + 1) as i32) as f32,
                (pos.x * CHUNK_WIDTH as i32 + (x_base + 2) as i32) as f32,
                (pos.x * CHUNK_WIDTH as i32 + (x_base + 3) as i32) as f32,
                (pos.x * CHUNK_WIDTH as i32 + (x_base + 4) as i32) as f32,
                (pos.x * CHUNK_WIDTH as i32 + (x_base + 5) as i32) as f32,
                (pos.x * CHUNK_WIDTH as i32 + (x_base + 6) as i32) as f32,
                (pos.x * CHUNK_WIDTH as i32 + (x_base + 7) as i32) as f32,
            ]);

            // Base terrain noise (large scale)
            let n1 = noise.get_3d(
                x_vec * f32x8::splat(0.01),
                f32x8::splat(0.0),
                z_vec * f32x8::splat(0.01),
            );

            // Detail noise
            let n2 = noise.get_3d(
                x_vec * f32x8::splat(0.05),
                f32x8::splat(100.0), // Offset Y to get different noise
                z_vec * f32x8::splat(0.05),
            );

            // Calculate height: Base height 64 + variations
            let height_vec = f32x8::splat(64.0) + n1 * f32x8::splat(40.0) + n2 * f32x8::splat(10.0);
            let height_arr = height_vec.to_array();

            for i in 0..8 {
                let x = x_base + i;
                let h = height_arr[i] as i32;

                for y in 0..CHUNK_HEIGHT {
                    if y < h as usize {
                        chunk.set_block(x, y, z, Block::Stone);
                    } else if y == h as usize {
                        chunk.set_block(x, y, z, Block::Grass);
                    } else if y < 40 {
                        // Water level (optional, simple stone for now)
                        // chunk.set_block(x, y, z, Block::Water);
                    }
                }

                // Add some dirt layers
                if h > 0 {
                    let dirt_depth = 3;
                    for d in 0..dirt_depth {
                        let dy = h - 1 - d;
                        if dy >= 0 {
                            chunk.set_block(x, dy as usize, z, Block::Dirt);
                        }
                    }
                    if h >= 0 {
                        chunk.set_block(x, h as usize, z, Block::Grass);
                    }
                }
            }
        }
    }
    chunk
}

fn update_chunks(
    mut commands: Commands,
    mut chunk_map: ResMut<ChunkMap>,
    mut loading: ResMut<LoadingChunks>,
    mut chunk_entities: Query<(Entity, &ChunkEntity)>,
    player_query: Query<&Transform, With<FpsController>>,
) {
    let player_transform = if let Some(t) = player_query.iter().next() {
        t
    } else {
        return;
    };

    let player_pos = player_transform.translation;
    let center_chunk_x = (player_pos.x / CHUNK_WIDTH as f32).floor() as i32;
    let center_chunk_z = (player_pos.z / CHUNK_DEPTH as f32).floor() as i32;

    let render_distance = 8;
    let mut required_chunks = HashSet::new();

    // Identify which chunks should be loaded
    for x in -render_distance..=render_distance {
        for z in -render_distance..=render_distance {
            if x * x + z * z <= render_distance * render_distance {
                required_chunks.insert(ChunkPos::new(center_chunk_x + x, center_chunk_z + z));
            }
        }
    }

    // Despawn far chunks
    for (entity, chunk_entity) in &mut chunk_entities {
        if !required_chunks.contains(&chunk_entity.0) {
            commands.entity(entity).despawn();
            chunk_map.chunks.remove(&chunk_entity.0);
        }
    }

    // Spawn tasks for missing chunks
    let thread_pool = AsyncComputeTaskPool::get();
    let perlin = PerlinSimd::new(42);

    for chunk_pos in required_chunks {
        if !chunk_map.chunks.contains_key(&chunk_pos) && !loading.0.contains(&chunk_pos) {
            loading.0.insert(chunk_pos);
            let perlin = perlin.clone();
            let entity = commands.spawn_empty().id();

            let task = thread_pool.spawn(async move {
                let chunk = generate_chunk(chunk_pos, &perlin);

                let mut command_queue = CommandQueue::default();
                command_queue.push(move |world: &mut World| {
                    // Check if we still want this chunk (could be unloaded while generating)
                    // For simplicity, we just insert it.
                    let mut chunk_map = world.resource_mut::<ChunkMap>();
                    chunk_map.chunks.insert(chunk_pos, chunk);

                    // Mesh generation
                    let chunk_ref = chunk_map.chunks.get(&chunk_pos).unwrap();
                    let mesh = generate_chunk_mesh(chunk_ref);
                    let mut meshes = world.resource_mut::<Assets<Mesh>>();
                    let mesh_handle = meshes.add(mesh);

                    let mut materials = world.resource_mut::<Assets<StandardMaterial>>();
                    let material_handle = materials.add(StandardMaterial {
                        base_color: Color::WHITE,
                        perceptual_roughness: 0.95,
                        reflectance: 0.1,
                        ..default()
                    });

                    world.spawn((
                        Mesh3d(mesh_handle),
                        MeshMaterial3d(material_handle),
                        Transform::from_xyz(
                            chunk_pos.x as f32 * CHUNK_WIDTH as f32,
                            0.0,
                            chunk_pos.z as f32 * CHUNK_DEPTH as f32,
                        ),
                        ChunkEntity(chunk_pos),
                    ));

                    // Remove from loading set
                    let mut loading = world.resource_mut::<LoadingChunks>();
                    loading.0.remove(&chunk_pos);

                    world.entity_mut(entity).despawn();
                });

                command_queue
            });

            commands.entity(entity).insert(ComputeChunk(task));
        }
    }
}

fn handle_chunk_tasks(mut commands: Commands, mut tasks: Query<(Entity, &mut ComputeChunk)>) {
    for (_entity, mut task) in &mut tasks {
        if let Some(mut command_queue) = check_ready(&mut task.0) {
            commands.append(&mut command_queue);
        }
    }
}

fn setup(mut commands: Commands) {
    // Ambient light
    commands.insert_resource(GlobalAmbientLight {
        color: Color::srgb(0.7, 0.8, 1.0),
        brightness: 500.0, // Increased brightness
        ..default()
    });

    // Player/Camera
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 80.0, 0.0).looking_at(Vec3::new(32.0, 64.0, 32.0), Vec3::Y),
        FpsController {
            sensitivity: 0.002,
            speed: 10.0, // Walking speed
            velocity: Vec3::ZERO,
            gravity: 25.0,
            jump_force: 8.0,
            grounded: false,
        },
    ));

    // Directional light
    commands.spawn((
        DirectionalLight {
            illuminance: 10000.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(50.0, 200.0, 50.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
}

fn controller_input(
    key: Res<ButtonInput<KeyCode>>,
    mut mouse_events: MessageReader<MouseMotion>,
    mut cursor_options: Single<&mut CursorOptions>,
    mouse: Res<ButtonInput<MouseButton>>,
    mut query: Query<(&mut Transform, &mut FpsController)>,
    time: Res<Time>,
    chunk_map: Res<ChunkMap>,
) {
    if mouse.just_pressed(MouseButton::Left) {
        cursor_options.visible = false;
        cursor_options.grab_mode = CursorGrabMode::Locked;
    }

    if key.just_pressed(KeyCode::Escape) {
        cursor_options.visible = true;
        cursor_options.grab_mode = CursorGrabMode::None;
    }

    if cursor_options.grab_mode == CursorGrabMode::Locked {
        let dt = time.delta_secs();

        for (mut transform, mut controller) in &mut query {
            // --- Rotation ---
            for event in mouse_events.read() {
                let (mut yaw, mut pitch, _): (f32, f32, f32) =
                    transform.rotation.to_euler(EulerRot::YXZ);
                yaw -= event.delta.x * controller.sensitivity;
                pitch -= event.delta.y * controller.sensitivity;
                pitch = pitch.clamp(-1.54, 1.54); // Look limit
                transform.rotation = Quat::from_euler(EulerRot::YXZ, yaw, pitch, 0.0);
            }

            // --- Movement Input ---
            let mut wish_dir = Vec3::ZERO;
            let forward = transform.forward();
            let right = transform.right();
            // Flatten forward/right for ground movement
            let flat_forward = Vec3::new(forward.x, 0.0, forward.z).normalize_or_zero();
            let flat_right = Vec3::new(right.x, 0.0, right.z).normalize_or_zero();

            if key.pressed(KeyCode::KeyW) {
                wish_dir += flat_forward;
            }
            if key.pressed(KeyCode::KeyS) {
                wish_dir -= flat_forward;
            }
            if key.pressed(KeyCode::KeyD) {
                wish_dir += flat_right;
            }
            if key.pressed(KeyCode::KeyA) {
                wish_dir -= flat_right;
            }

            if wish_dir.length_squared() > 0.0 {
                wish_dir = wish_dir.normalize();
            }

            // Sprint
            let current_speed = if key.pressed(KeyCode::ControlLeft) {
                controller.speed * 1.5
            } else {
                controller.speed
            };

            // Accelerate horizontally
            let target_vel = wish_dir * current_speed;
            let accel = 10.0; // Acceleration factor

            // Move generic velocity towards target X/Z
            controller.velocity.x = controller.velocity.x.lerp(target_vel.x, accel * dt);
            controller.velocity.z = controller.velocity.z.lerp(target_vel.z, accel * dt);

            // --- Gravity & Jumping ---
            if controller.grounded && key.just_pressed(KeyCode::Space) {
                controller.velocity.y = controller.jump_force;
                controller.grounded = false;
            } else if !controller.grounded {
                controller.velocity.y -= controller.gravity * dt;
            }

            // --- simple Collision & Integration ---
            let velocity = controller.velocity;

            // Define player AABB dimensions (relative to camera eye at Transform)
            // Eye height ~1.6m. Player height ~1.8m. Width ~0.6m.
            // AABB: min = pos - (0.3, 1.6, 0.3), max = pos + (0.3, 0.2, 0.3)
            let check_collision = |pos: Vec3| -> bool {
                let min = pos - Vec3::new(0.3, 1.6, 0.3);
                let max = pos + Vec3::new(0.3, 0.2, 0.3);

                let min_x = min.x.floor() as i32;
                let max_x = max.x.floor() as i32;
                let min_y = min.y.floor() as i32;
                let max_y = max.y.floor() as i32;
                let min_z = min.z.floor() as i32;
                let max_z = max.z.floor() as i32;

                for y in min_y..=max_y {
                    for z in min_z..=max_z {
                        for x in min_x..=max_x {
                            if let Some(block) = chunk_map.get_block(x, y, z) {
                                if block != Block::Air {
                                    return true; // Collision
                                }
                            } else {
                                // Unloaded chunk - treat as collision to prevent falling?
                                // Or air? Let's treat as air for smoothness, but maybe risky.
                                // Actually solid is safer to prevent falling into void.
                                return true;
                            }
                        }
                    }
                }
                false
            };

            // Axis-independent collision resolution

            // X-axis
            let mut test_pos = transform.translation;
            test_pos.x += velocity.x * dt;
            if check_collision(test_pos) {
                controller.velocity.x = 0.0;
            } else {
                transform.translation.x = test_pos.x;
            }

            // Z-axis
            test_pos = transform.translation;
            test_pos.z += velocity.z * dt;
            if check_collision(test_pos) {
                controller.velocity.z = 0.0;
            } else {
                transform.translation.z = test_pos.z;
            }

            // Y-axis
            test_pos = transform.translation;
            test_pos.y += velocity.y * dt;
            if check_collision(test_pos) {
                if velocity.y < 0.0 {
                    controller.grounded = true;
                }
                controller.velocity.y = 0.0;
            } else {
                transform.translation.y = test_pos.y;
                controller.grounded = false;
            }
        }
    }
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Voxel Engine".into(),
                ..default()
            }),
            ..default()
        }))
        .init_resource::<ChunkMap>()
        .init_resource::<LoadingChunks>()
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (handle_chunk_tasks, update_chunks, controller_input),
        )
        .run();
}
