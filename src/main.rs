#![feature(portable_simd)]
use bevy::input::mouse::MouseMotion;
use bevy::prelude::*;
use bevy::tasks::{AsyncComputeTaskPool, Task, futures::check_ready};
use bevy::window::{CursorGrabMode, CursorOptions};
use core::simd::prelude::*;
use std::collections::{HashMap, HashSet};
use std::sync::RwLock;
use std::io::{Read, Write};
use zstd::stream::{Decoder, Encoder};

mod mesh;
mod noise;
use crate::mesh::{ChunkMeshData, Face, LodLevel, generate_chunk_mesh_data, generate_chunk_mesh_data_lod};
use crate::noise::PerlinSimd;
use bevy::asset::LoadedFolder;
use bevy::image::{ImageFilterMode, ImageSampler, ImageSamplerDescriptor};
use bevy::math::Rect;

pub const CHUNK_WIDTH: usize = 16;
pub const CHUNK_HEIGHT: usize = 256;
pub const CHUNK_DEPTH: usize = 16;
pub const CHUNK_VOLUME: usize = CHUNK_WIDTH * CHUNK_HEIGHT * CHUNK_DEPTH;

/// Represents a block type in the world.
/// Using #[repr(u8)] for efficient storage and serialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
#[repr(u8)]
pub enum Block {
    #[default]
    Air = 0,
    Stone = 1,
    Dirt = 2,
    Grass = 3,
    Water = 4,
    Sand = 5,
}

impl From<u8> for Block {
    fn from(value: u8) -> Self {
        match value {
            1 => Block::Stone,
            2 => Block::Dirt,
            3 => Block::Grass,
            4 => Block::Water,
            5 => Block::Sand,
            _ => Block::Air,
        }
    }
}

impl Block {
    /// Returns true if this block is transparent (air or water)
    pub fn is_transparent(&self) -> bool {
        matches!(self, Block::Air | Block::Water)
    }
    
    /// Returns true if this block is a liquid
    pub fn is_liquid(&self) -> bool {
        matches!(self, Block::Water)
    }
}

/// A 16x256x16 chunk of blocks.
/// Uses z-order (Morton curve) indexing for spatial locality.
#[derive(Clone)]
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
    /// Time since last grounded (for coyote time)
    pub time_since_grounded: f32,
    /// Time since jump was pressed (for jump buffering)
    pub jump_buffer: f32,
}

#[derive(Component)]
pub struct ChunkEntity(pub ChunkPos);

/// A resource that stores all loaded chunks (compressed with zstd).
/// Uses a cache for decompressed chunks to avoid repeated decompression.
#[derive(Resource)]
pub struct ChunkMap {
    /// Compressed chunk storage
    pub chunks: HashMap<ChunkPos, Vec<u8>>,
    /// Cache of decompressed chunks for fast access
    pub cache: RwLock<HashMap<ChunkPos, Chunk>>,
}

impl Default for ChunkMap {
    fn default() -> Self {
        Self {
            chunks: HashMap::new(),
            cache: RwLock::new(HashMap::new()),
        }
    }
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

        // Try cache first (read lock)
        {
            let cache = self.cache.read().unwrap();
            if let Some(chunk) = cache.get(&chunk_pos) {
                return Some(chunk.get_block(local_x, y as usize, local_z));
            }
        }

        // Decompress and cache (write lock)
        if let Some(compressed) = self.chunks.get(&chunk_pos) {
            if let Ok(chunk) = Chunk::decompress(compressed) {
                let block = chunk.get_block(local_x, y as usize, local_z);
                self.cache.write().unwrap().insert(chunk_pos, chunk);
                return Some(block);
            }
        }
        None
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

        let mut cache = self.cache.write().unwrap();
        
        // If not in cache, decompress it
        if !cache.contains_key(&chunk_pos) {
            if let Some(compressed) = self.chunks.get(&chunk_pos) {
                if let Ok(chunk) = Chunk::decompress(compressed) {
                    cache.insert(chunk_pos, chunk);
                }
            }
        }

        // Modify the cached chunk
        if let Some(chunk) = cache.get_mut(&chunk_pos) {
            chunk.set_block(local_x, y as usize, local_z, block);
            // Update compressed storage
            if let Ok(new_compressed) = chunk.compress() {
                self.chunks.insert(chunk_pos, new_compressed);
                return true;
            }
        }
        false
    }

    /// Removes a chunk from both storage and cache.
    pub fn remove(&mut self, chunk_pos: &ChunkPos) {
        self.chunks.remove(chunk_pos);
        self.cache.write().unwrap().remove(chunk_pos);
    }

    /// Inserts a compressed chunk with its already-decompressed data for the cache.
    pub fn insert(&mut self, chunk_pos: ChunkPos, chunk: Chunk, compressed: Vec<u8>) {
        self.cache.write().unwrap().insert(chunk_pos, chunk);
        self.chunks.insert(chunk_pos, compressed);
    }
}

#[derive(Resource, Default)]
struct LoadingChunks(HashSet<ChunkPos>);

#[derive(Debug, Clone, Copy, Default, Eq, PartialEq, Hash, States)]
enum AppState {
    #[default]
    Loading,
    InGame,
}

#[derive(Resource)]
struct TextureLoading(Handle<LoadedFolder>);

#[derive(Resource)]
pub struct BlockTextureMap {
    pub layout: Handle<TextureAtlasLayout>,
    pub image: Handle<Image>,
    pub map: HashMap<(Block, Face), usize>, // Map block+face to atlas index
}

/// Shared material for all chunks to reduce GPU state changes
#[derive(Resource)]
pub struct SharedChunkMaterial(pub Handle<StandardMaterial>);

/// Timer to throttle chunk loading/unloading checks (expensive operations)
#[derive(Resource)]
pub struct ChunkUpdateTimer {
    /// Timer for chunk loading/unloading (runs every 0.5s)
    pub load_timer: Timer,
    /// Timer for LOD updates (runs every 0.1s)
    pub lod_timer: Timer,
    /// Cached player chunk position to detect movement
    pub last_player_chunk: Option<ChunkPos>,
}

// Async chunk generation result - contains everything needed to spawn the chunk on main thread
pub struct ChunkGenResult {
    pub chunk: Chunk,
    pub compressed_chunk: Vec<u8>,
    pub mesh_lod0: ChunkMeshData, // Full detail
    pub mesh_lod1: ChunkMeshData, // Medium detail (2x2x2)
    pub mesh_lod2: ChunkMeshData, // Low detail (4x4x4)
    pub chunk_pos: ChunkPos,
    pub distance_sq: i32,         // Distance from player for prioritization
}

/// Stores mesh handles for all LOD levels of a chunk
#[derive(Component)]
pub struct ChunkLodMeshes {
    pub lod0: Handle<Mesh>,
    pub lod1: Handle<Mesh>,
    pub lod2: Handle<Mesh>,
    pub current_lod: u8,
}

#[derive(Component)]
pub struct ComputeChunk(pub Task<ChunkGenResult>);

/// Event sent when a block is modified
#[derive(Event)]
pub struct BlockModifiedEvent {
    pub chunk_pos: ChunkPos,
}

/// Resource to track chunks that need remeshing
#[derive(Resource, Default)]
pub struct ChunksToRemesh(pub HashSet<ChunkPos>);

/// Sea level for water generation
pub const SEA_LEVEL: usize = 62;

fn generate_chunk(pos: ChunkPos, noise: &PerlinSimd) -> Chunk {
    let mut chunk = Chunk::new();
    
    // Pre-compute cave data for the entire chunk using SIMD
    // We'll sample at every 2 blocks and interpolate conceptually (or just use nearest)
    // Process 8 Y values at a time for each (x, z) column
    
    // First pass: compute heightmap for all columns
    let mut heightmap = [[0usize; CHUNK_DEPTH]; CHUNK_WIDTH];
    
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
                f32x8::splat(100.0),
                z_vec * f32x8::splat(0.05),
            );

            // Calculate height
            let height_vec = f32x8::splat(64.0) + n1 * f32x8::splat(40.0) + n2 * f32x8::splat(10.0);
            let height_arr = height_vec.to_array();

            for (i, h) in height_arr.iter().enumerate() {
                heightmap[x_base + i][z] = (*h as usize).min(CHUNK_HEIGHT - 1);
            }
        }
    }
    
    // Second pass: generate terrain with caves using SIMD for Y batches
    for z in 0..CHUNK_DEPTH {
        let world_z = (pos.z * CHUNK_DEPTH as i32 + z as i32) as f32;
        
        for x in 0..CHUNK_WIDTH {
            let world_x = (pos.x * CHUNK_WIDTH as i32 + x as i32) as f32;
            let surface_height = heightmap[x][z];
            
            // Process Y in batches of 8 for cave noise
            let max_y = surface_height.max(SEA_LEVEL);
            
            for y_base in (0..=max_y).step_by(8) {
                // Create Y vector for this batch
                let y_vec = f32x8::from_array([
                    y_base as f32,
                    (y_base + 1) as f32,
                    (y_base + 2) as f32,
                    (y_base + 3) as f32,
                    (y_base + 4) as f32,
                    (y_base + 5) as f32,
                    (y_base + 6) as f32,
                    (y_base + 7) as f32,
                ]);
                
                // Compute cave noise for 8 Y values at once
                let cave_noise1 = noise.get_3d(
                    f32x8::splat(world_x * 0.05),
                    y_vec * f32x8::splat(0.05),
                    f32x8::splat(world_z * 0.05),
                );
                
                let cave_noise2 = noise.get_3d(
                    f32x8::splat(world_x * 0.08 + 1000.0),
                    y_vec * f32x8::splat(0.08),
                    f32x8::splat(world_z * 0.08 + 1000.0),
                );
                
                let cave1_arr = cave_noise1.to_array();
                let cave2_arr = cave_noise2.to_array();
                
                // Process each Y in this batch
                for i in 0..8 {
                    let y = y_base + i;
                    if y > max_y {
                        break;
                    }
                    
                    let is_cave = cave1_arr[i].abs() < 0.1 
                        && cave2_arr[i].abs() < 0.1 
                        && y > 5 
                        && y < surface_height.saturating_sub(2);
                    
                    if y <= surface_height {
                        if is_cave {
                            // Cave - fill with water if below sea level
                            if y <= SEA_LEVEL {
                                chunk.set_block(x, y, z, Block::Water);
                            }
                            // else leave as air (default)
                        } else {
                            let block = if y == surface_height {
                                if surface_height <= SEA_LEVEL + 2 {
                                    Block::Sand
                                } else {
                                    Block::Grass
                                }
                            } else if y >= surface_height.saturating_sub(3) {
                                if surface_height <= SEA_LEVEL + 2 {
                                    Block::Sand
                                } else {
                                    Block::Dirt
                                }
                            } else {
                                Block::Stone
                            };
                            chunk.set_block(x, y, z, block);
                        }
                    } else if y <= SEA_LEVEL {
                        // Water above terrain
                        chunk.set_block(x, y, z, Block::Water);
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
    texture_map: Res<BlockTextureMap>,
    layouts: Res<Assets<TextureAtlasLayout>>,
    time: Res<Time>,
    mut timer: ResMut<ChunkUpdateTimer>,
) {
    // Only run chunk loading/unloading logic periodically OR when player crosses chunk boundary
    let player_transform = if let Some(t) = player_query.iter().next() {
        t
    } else {
        return;
    };

    let player_pos = player_transform.translation;
    let center_chunk_x = (player_pos.x / CHUNK_WIDTH as f32).floor() as i32;
    let center_chunk_z = (player_pos.z / CHUNK_DEPTH as f32).floor() as i32;
    let current_chunk = ChunkPos::new(center_chunk_x, center_chunk_z);

    // Check if player moved to a new chunk
    let player_moved_chunk = timer.last_player_chunk != Some(current_chunk);
    if player_moved_chunk {
        timer.last_player_chunk = Some(current_chunk);
    }

    // Tick timer and check if we should run
    timer.load_timer.tick(time.delta());
    let should_run = timer.load_timer.just_finished() || player_moved_chunk;
    
    if !should_run {
        return;
    }

    // Reasonable render distance - 64 chunks = 1024 blocks
    // This gives ~12,000 chunks which is manageable
    let render_distance = 64;
    let mut required_chunks = HashSet::new();

    // Identify which chunks should be loaded
    for x in -render_distance..=render_distance {
        for z in -render_distance..=render_distance {
            if x * x + z * z <= render_distance * render_distance {
                required_chunks.insert(ChunkPos::new(center_chunk_x + x, center_chunk_z + z));
            }
        }
    }

    // Despawn far chunks (despawn handles children automatically in newer Bevy)
    for (entity, chunk_entity) in &mut chunk_entities {
        if !required_chunks.contains(&chunk_entity.0) {
            commands.entity(entity).despawn();
            chunk_map.remove(&chunk_entity.0);
        }
    }

    // Spawn tasks for missing chunks - do heavy work (terrain + mesh generation) async
    let thread_pool = AsyncComputeTaskPool::get();
    let perlin = PerlinSimd::new(42);

    // Pre-compute UV data for async tasks (clone what we need)
    let texture_data = texture_map.map.clone();
    let layout = layouts.get(&texture_map.layout).cloned();

    // Limit how many chunks we start generating per frame to avoid spawn overhead
    let max_new_chunks_per_frame = 16;
    let mut started = 0;

    // Sort chunks by distance for priority loading (closer chunks first)
    let mut chunks_to_load: Vec<_> = required_chunks
        .iter()
        .filter(|pos| !chunk_map.chunks.contains_key(pos) && !loading.0.contains(pos))
        .map(|pos| {
            let dx = pos.x - center_chunk_x;
            let dz = pos.z - center_chunk_z;
            (*pos, dx * dx + dz * dz)
        })
        .collect();
    chunks_to_load.sort_by_key(|(_, dist)| *dist);

    for (chunk_pos, distance_sq) in chunks_to_load {
        if started >= max_new_chunks_per_frame {
            break;
        }
        loading.0.insert(chunk_pos);
        let perlin = perlin.clone();
        let texture_data = texture_data.clone();
        let layout = layout.clone();

        let task = thread_pool.spawn(async move {
            // Generate terrain - this is the heavy CPU work
            let chunk = generate_chunk(chunk_pos, &perlin);

            // Generate mesh data for LOD levels (reduced to 3 for performance)
            let (mesh_lod0, mesh_lod1, mesh_lod2) = if let Some(layout) = layout {
                let atlas_size = layout.size.as_vec2();
                let get_uv = |block: &Block, face: &Face| {
                    texture_data.get(&(*block, *face)).and_then(|&index| {
                        layout.textures.get(index).map(|rect| {
                            let min = rect.min.as_vec2() / atlas_size;
                            let max = rect.max.as_vec2() / atlas_size;
                            Rect { min, max }
                        })
                    })
                };
                
                let lod0 = generate_chunk_mesh_data(&chunk, &get_uv);
                let lod1 = generate_chunk_mesh_data_lod(&chunk, LodLevel::Medium, &get_uv);
                let lod2 = generate_chunk_mesh_data_lod(&chunk, LodLevel::Low, &get_uv);
                (lod0, lod1, lod2)
            } else {
                // Fallback empty meshes if layout not ready
                let empty = || ChunkMeshData {
                    positions: Vec::new(),
                    normals: Vec::new(),
                    uvs: Vec::new(),
                    colors: Vec::new(),
                    indices: Vec::new(),
                };
                (empty(), empty(), empty())
            };

            // Compress chunk data using zstd for efficient storage
            let compressed_chunk = chunk
                .compress()
                .expect("Failed to compress chunk");

            ChunkGenResult {
                chunk,
                compressed_chunk,
                mesh_lod0,
                mesh_lod1,
                mesh_lod2,
                chunk_pos,
                distance_sq,
            }
        });

        commands.spawn(ComputeChunk(task));
        started += 1;
    }
}

fn handle_chunk_tasks(
    mut commands: Commands,
    mut tasks: Query<(Entity, &mut ComputeChunk)>,
    mut chunk_map: ResMut<ChunkMap>,
    mut loading: ResMut<LoadingChunks>,
    mut meshes: ResMut<Assets<Mesh>>,
    shared_material: Option<Res<SharedChunkMaterial>>,
) {
    // LOD distance thresholds (in chunk units squared)
    // More generous thresholds for Distant Horizons-like experience
    const LOD0_DIST_SQ: i32 = 12 * 12; // Full detail: 0-12 chunks (192 blocks)
    const LOD1_DIST_SQ: i32 = 32 * 32; // Medium detail: 12-32 chunks (512 blocks)

    let Some(material) = shared_material else {
        return;
    };

    // Process a limited number of completed chunks per frame to avoid stuttering
    let max_chunks_per_frame = 16;
    let mut processed = 0;

    for (entity, mut task) in &mut tasks {
        if processed >= max_chunks_per_frame {
            break;
        }

        if let Some(result) = check_ready(&mut task.0) {
            let chunk_pos = result.chunk_pos;
            let distance_sq = result.distance_sq;

            // Only cache decompressed chunks for nearby chunks (within LOD0 range)
            const CACHE_DISTANCE_SQ: i32 = 14 * 14;
            if distance_sq <= CACHE_DISTANCE_SQ {
                chunk_map.insert(chunk_pos, result.chunk, result.compressed_chunk);
            } else {
                // For distant chunks, only store compressed data
                chunk_map.chunks.insert(chunk_pos, result.compressed_chunk);
            }

            // Create mesh handles for all LOD levels
            let mesh_lod0 = meshes.add(result.mesh_lod0.into_mesh());
            let mesh_lod1 = meshes.add(result.mesh_lod1.into_mesh());
            let mesh_lod2 = meshes.add(result.mesh_lod2.into_mesh());

            // Select initial LOD based on distance
            let (initial_mesh, current_lod) = if distance_sq <= LOD0_DIST_SQ {
                (mesh_lod0.clone(), 0)
            } else if distance_sq <= LOD1_DIST_SQ {
                (mesh_lod1.clone(), 1)
            } else {
                (mesh_lod2.clone(), 2)
            };

            // Spawn SINGLE entity per chunk with the appropriate LOD mesh
            commands.spawn((
                Mesh3d(initial_mesh),
                MeshMaterial3d(material.0.clone()),
                Transform::from_xyz(
                    chunk_pos.x as f32 * CHUNK_WIDTH as f32,
                    0.0,
                    chunk_pos.z as f32 * CHUNK_DEPTH as f32,
                ),
                ChunkEntity(chunk_pos),
                ChunkLodMeshes {
                    lod0: mesh_lod0,
                    lod1: mesh_lod1,
                    lod2: mesh_lod2,
                    current_lod,
                },
            ));

            // Remove from loading set and despawn task entity
            loading.0.remove(&chunk_pos);
            commands.entity(entity).despawn();

            processed += 1;
        }
    }
}

/// Update chunk LOD meshes based on player distance
fn update_chunk_lods(
    player_query: Query<&Transform, With<FpsController>>,
    mut chunks: Query<(&ChunkEntity, &mut ChunkLodMeshes, &mut Mesh3d)>,
    time: Res<Time>,
    mut timer: ResMut<ChunkUpdateTimer>,
) {
    // Throttle LOD updates - only run every 0.1 seconds
    timer.lod_timer.tick(time.delta());
    if !timer.lod_timer.just_finished() {
        return;
    }

    // LOD distance thresholds (in chunk units squared)
    // More generous thresholds for Distant Horizons-like experience
    const LOD0_DIST_SQ: i32 = 12 * 12; // Full detail: 0-12 chunks (192 blocks)
    const LOD1_DIST_SQ: i32 = 32 * 32; // Medium detail: 12-32 chunks (512 blocks)

    let player_transform = if let Some(t) = player_query.iter().next() {
        t
    } else {
        return;
    };

    let player_chunk_x = (player_transform.translation.x / CHUNK_WIDTH as f32).floor() as i32;
    let player_chunk_z = (player_transform.translation.z / CHUNK_DEPTH as f32).floor() as i32;

    for (chunk_entity, mut lod_meshes, mut mesh) in &mut chunks {
        let dx = chunk_entity.0.x - player_chunk_x;
        let dz = chunk_entity.0.z - player_chunk_z;
        let dist_sq = dx * dx + dz * dz;

        let target_lod = if dist_sq <= LOD0_DIST_SQ {
            0
        } else if dist_sq <= LOD1_DIST_SQ {
            1
        } else {
            2
        };

        // Only update if LOD changed
        if target_lod != lod_meshes.current_lod {
            lod_meshes.current_lod = target_lod;
            mesh.0 = match target_lod {
                0 => lod_meshes.lod0.clone(),
                1 => lod_meshes.lod1.clone(),
                _ => lod_meshes.lod2.clone(),
            };
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

    // Chunk update timer (throttles expensive chunk operations)
    commands.insert_resource(ChunkUpdateTimer {
        load_timer: Timer::from_seconds(0.5, TimerMode::Repeating),
        lod_timer: Timer::from_seconds(0.1, TimerMode::Repeating),
        last_player_chunk: None,
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
            time_since_grounded: 0.0,
            jump_buffer: 0.0,
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

fn load_textures(mut commands: Commands, asset_server: Res<AssetServer>) {
    // Load all textures in assets/textures/block
    let handle = asset_server.load_folder("textures/block");
    commands.insert_resource(TextureLoading(handle));
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
    // Only capture cursor if not already captured
    if mouse.just_pressed(MouseButton::Left) && cursor_options.grab_mode != CursorGrabMode::Locked {
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
            // Minecraft-like constants
            const COYOTE_TIME: f32 = 0.1; // Can jump shortly after leaving ground
            const JUMP_BUFFER: f32 = 0.15; // Buffer jump input before landing

            // Update jump buffer
            if key.just_pressed(KeyCode::Space) {
                controller.jump_buffer = JUMP_BUFFER;
            } else {
                controller.jump_buffer = (controller.jump_buffer - dt).max(0.0);
            }

            // Check if we can jump (grounded or within coyote time) and want to jump (buffered)
            let can_jump = controller.grounded || controller.time_since_grounded < COYOTE_TIME;
            if can_jump && controller.jump_buffer > 0.0 {
                controller.velocity.y = controller.jump_force;
                controller.grounded = false;
                controller.time_since_grounded = COYOTE_TIME; // Prevent double jump
                controller.jump_buffer = 0.0; // Consume the buffer
            }

            // Apply gravity when not grounded
            if !controller.grounded {
                controller.velocity.y -= controller.gravity * dt;
                controller.time_since_grounded += dt;
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
                    controller.time_since_grounded = 0.0;
                }
                controller.velocity.y = 0.0;
            } else {
                transform.translation.y = test_pos.y;
                // Only set grounded to false if we're actually moving up or 
                // there's no ground directly below us
            }

            // Dedicated ground check - check slightly below feet
            // This prevents grounded flickering when standing still
            let ground_check_pos = transform.translation - Vec3::new(0.0, 0.01, 0.0);
            if check_collision(ground_check_pos) {
                controller.grounded = true;
                controller.time_since_grounded = 0.0;
            } else if velocity.y >= 0.0 {
                // Only become ungrounded if not falling into ground
                controller.grounded = false;
            }
        }
    }
}

/// Raycast through the world to find which block the player is looking at
fn raycast_block(
    origin: Vec3,
    direction: Vec3,
    max_distance: f32,
    chunk_map: &ChunkMap,
) -> Option<(i32, i32, i32)> {
    // DDA-style raycast through voxel grid
    let step = 0.1;
    let mut t = 0.0;
    
    while t < max_distance {
        let pos = origin + direction * t;
        let bx = pos.x.floor() as i32;
        let by = pos.y.floor() as i32;
        let bz = pos.z.floor() as i32;
        
        if let Some(block) = chunk_map.get_block(bx, by, bz) {
            if block != Block::Air && block != Block::Water {
                return Some((bx, by, bz));
            }
        }
        
        t += step;
    }
    
    None
}

/// System to handle block breaking with left click
fn block_breaking(
    mouse: Res<ButtonInput<MouseButton>>,
    cursor_options: Single<&CursorOptions>,
    player_query: Query<&Transform, With<FpsController>>,
    mut chunk_map: ResMut<ChunkMap>,
    mut chunks_to_remesh: ResMut<ChunksToRemesh>,
) {
    // Only break blocks when cursor is captured
    if cursor_options.grab_mode != CursorGrabMode::Locked {
        return;
    }
    
    if !mouse.just_pressed(MouseButton::Left) {
        return;
    }
    
    let Ok(transform) = player_query.single() else {
        return;
    };
    
    let origin = transform.translation;
    let direction = transform.forward().as_vec3();
    
    if let Some((bx, by, bz)) = raycast_block(origin, direction, 5.0, &chunk_map) {
        // Set the block to air
        if chunk_map.set_block(bx, by, bz, Block::Air) {
            // Mark the chunk for remeshing
            let chunk_pos = ChunkPos::new(
                bx.div_euclid(CHUNK_WIDTH as i32),
                bz.div_euclid(CHUNK_DEPTH as i32),
            );
            chunks_to_remesh.0.insert(chunk_pos);
            
            // Also mark adjacent chunks if the block is on a boundary
            let local_x = bx.rem_euclid(CHUNK_WIDTH as i32) as usize;
            let local_z = bz.rem_euclid(CHUNK_DEPTH as i32) as usize;
            
            if local_x == 0 {
                chunks_to_remesh.0.insert(ChunkPos::new(chunk_pos.x - 1, chunk_pos.z));
            }
            if local_x == CHUNK_WIDTH - 1 {
                chunks_to_remesh.0.insert(ChunkPos::new(chunk_pos.x + 1, chunk_pos.z));
            }
            if local_z == 0 {
                chunks_to_remesh.0.insert(ChunkPos::new(chunk_pos.x, chunk_pos.z - 1));
            }
            if local_z == CHUNK_DEPTH - 1 {
                chunks_to_remesh.0.insert(ChunkPos::new(chunk_pos.x, chunk_pos.z + 1));
            }
        }
    }
}

/// System to handle chunk remeshing when blocks are modified
fn remesh_chunks(
    mut chunks_to_remesh: ResMut<ChunksToRemesh>,
    chunk_map: Res<ChunkMap>,
    mut chunks: Query<(&ChunkEntity, &mut ChunkLodMeshes, &mut Mesh3d)>,
    mut meshes: ResMut<Assets<Mesh>>,
    texture_map: Res<BlockTextureMap>,
    layouts: Res<Assets<TextureAtlasLayout>>,
) {
    if chunks_to_remesh.0.is_empty() {
        return;
    }
    
    let Some(layout) = layouts.get(&texture_map.layout) else {
        return;
    };
    
    let atlas_size = layout.size.as_vec2();
    let texture_data = &texture_map.map;
    
    // Process chunks that need remeshing
    for chunk_pos in chunks_to_remesh.0.drain() {
        // Get the chunk data from cache
        let cache = chunk_map.cache.read().unwrap();
        let Some(chunk) = cache.get(&chunk_pos) else {
            continue;
        };
        
        // Generate new mesh
        let get_uv = |block: &Block, face: &Face| {
            texture_data.get(&(*block, *face)).and_then(|&index| {
                layout.textures.get(index).map(|rect| {
                    let min = rect.min.as_vec2() / atlas_size;
                    let max = rect.max.as_vec2() / atlas_size;
                    Rect { min, max }
                })
            })
        };
        
        let mesh_data = generate_chunk_mesh_data(chunk, &get_uv);
        let new_mesh = meshes.add(mesh_data.into_mesh());
        
        // Update the chunk entity's mesh
        for (chunk_entity, mut lod_meshes, mut mesh) in &mut chunks {
            if chunk_entity.0 == chunk_pos {
                // Update LOD0 mesh (the detailed one)
                lod_meshes.lod0 = new_mesh.clone();
                
                // If currently showing LOD0, update the visible mesh
                if lod_meshes.current_lod == 0 {
                    mesh.0 = new_mesh.clone();
                }
                break;
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
        .init_resource::<ChunksToRemesh>()
        .init_state::<AppState>()
        .add_systems(Startup, load_textures)
        .add_systems(Update, check_textures.run_if(in_state(AppState::Loading)))
        .add_systems(OnEnter(AppState::InGame), setup)
        .add_systems(
            Update,
            (handle_chunk_tasks, update_chunks, update_chunk_lods, controller_input, block_breaking, remesh_chunks)
                .run_if(in_state(AppState::InGame)),
        )
        .run();
}

fn check_textures(
    mut commands: Commands,
    loading: Res<TextureLoading>,
    loaded_folders: Res<Assets<LoadedFolder>>,
    mut texture_assets: ResMut<Assets<Image>>,
    mut texture_layouts: ResMut<Assets<TextureAtlasLayout>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut next_state: ResMut<NextState<AppState>>,
) {
    if let Some(folder) = loaded_folders.get(&loading.0) {
        let mut builder = TextureAtlasBuilder::default();
        builder.padding(UVec2::splat(0));

        // Check if all handles in folder are valid and loaded
        let mut ready = true;
        for handle in &folder.handles {
            let handle = handle.clone().try_typed::<Image>().unwrap();
            if !texture_assets.contains(&handle) {
                ready = false;
                break;
            }
        }
        if !ready {
            return;
        }

        let mut named_handles = HashMap::new();
        for handle in &folder.handles {
            let path = handle.path().unwrap();
            let name = path.path().file_name().unwrap().to_str().unwrap();
            let image_handle = handle.clone().try_typed::<Image>().unwrap();

            // We can safely unwrap here because we checked contains above (mostly, unless race condition but single threaded mostly)
            let texture = texture_assets.get(&image_handle).unwrap();
            builder.add_texture(Some(image_handle.id()), texture);
            named_handles.insert(name.to_string(), image_handle);
        }

        if let Ok((layout, sources, image)) = builder.build() {
            let mut final_image = image;
            final_image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
                mag_filter: ImageFilterMode::Nearest,
                min_filter: ImageFilterMode::Nearest,
                mipmap_filter: ImageFilterMode::Nearest,
                ..default()
            });
            let image = final_image;

            let image_handle = texture_assets.add(image);
            let layout_handle = texture_layouts.add(layout.clone());

            // Create shared material for all chunks (reduces GPU state changes)
            let shared_material = materials.add(StandardMaterial {
                base_color_texture: Some(image_handle.clone()),
                perceptual_roughness: 0.9,
                reflectance: 0.1,
                unlit: false,
                ..default()
            });
            commands.insert_resource(SharedChunkMaterial(shared_material));

            let mut map = HashMap::new();

            let faces_all = [
                Face::Top,
                Face::Bottom,
                Face::Left,
                Face::Right,
                Face::Back,
                Face::Forward,
            ];
            let faces_side = [Face::Left, Face::Right, Face::Back, Face::Forward];

            // Stone
            if let Some(h) = named_handles.get("stone.png")
                && let Some(idx) = sources.texture_index(h.id())
            {
                for f in faces_all {
                    map.insert((Block::Stone, f), idx);
                }
            }

            // Dirt
            if let Some(h) = named_handles.get("dirt.png")
                && let Some(idx) = sources.texture_index(h.id())
            {
                for f in faces_all {
                    map.insert((Block::Dirt, f), idx);
                }
                // Grass Bottom is Dirt
                map.insert((Block::Grass, Face::Bottom), idx);
            }

            // Grass Top
            if let Some(h) = named_handles.get("grass_block_top.png")
                && let Some(idx) = sources.texture_index(h.id())
            {
                map.insert((Block::Grass, Face::Top), idx);
            }

            // Grass Side
            if let Some(h) = named_handles.get("grass_block_side.png")
                && let Some(idx) = sources.texture_index(h.id())
            {
                for f in faces_side {
                    map.insert((Block::Grass, f), idx);
                }
            }

            // Water
            if let Some(h) = named_handles.get("water_still.png")
                && let Some(idx) = sources.texture_index(h.id())
            {
                for f in faces_all {
                    map.insert((Block::Water, f), idx);
                }
            }

            // Sand
            if let Some(h) = named_handles.get("sand.png")
                && let Some(idx) = sources.texture_index(h.id())
            {
                for f in faces_all {
                    map.insert((Block::Sand, f), idx);
                }
            }

            commands.insert_resource(BlockTextureMap {
                layout: layout_handle,
                image: image_handle,
                map,
            });

            next_state.set(AppState::InGame);
        }
    }
}
