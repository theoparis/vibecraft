use crate::{Block, CHUNK_DEPTH, CHUNK_HEIGHT, CHUNK_WIDTH, Chunk};
use bevy::asset::RenderAssetUsages;
use bevy::math::Rect;
use bevy::mesh::Indices;
use bevy::prelude::*;
use bevy::render::render_resource::PrimitiveTopology;

/// LOD level for chunk meshes
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(dead_code)]
pub enum LodLevel {
    /// Full detail - every block face
    Full,
    /// Medium detail - 2x2x2 block groups
    Medium,
    /// Low detail - 4x4x4 block groups
    Low,
    /// Very low detail - 8x8x8 block groups
    VeryLow,
    /// Lowest detail - 16x16x16 block groups (1 per chunk horizontally)
    Lowest,
}

impl LodLevel {
    pub fn block_size(&self) -> usize {
        match self {
            LodLevel::Full => 1,
            LodLevel::Medium => 2,
            LodLevel::Low => 4,
            LodLevel::VeryLow => 8,
            LodLevel::Lowest => 16,
        }
    }
}

/// Raw mesh data that can be computed off the main thread
pub struct ChunkMeshData {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub uvs: Vec<[f32; 2]>,
    pub colors: Vec<[f32; 4]>,
    pub indices: Vec<u32>,
}

impl ChunkMeshData {
    pub fn into_mesh(self) -> Mesh {
        Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        )
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, self.positions)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, self.normals)
        .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, self.uvs)
        .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, self.colors)
        .with_inserted_indices(Indices::U32(self.indices))
    }
}

// Grass tint color (Minecraft plains biome green, sRGB values)
// The grass_block_top.png is grayscale and gets multiplied by this color
const GRASS_TINT: [f32; 4] = [0.569, 0.741, 0.349, 1.0]; // #91BD59 in hex
const NO_TINT: [f32; 4] = [1.0, 1.0, 1.0, 1.0];

/// Water surface height offset (Minecraft water is 14/16 = 0.875 of a block)
const WATER_HEIGHT: f32 = 0.875;

/// Result containing both solid and water meshes
pub struct ChunkMeshes {
    pub solid: ChunkMeshData,
    pub water: ChunkMeshData,
}

pub fn generate_chunk_mesh_data(
    chunk: &Chunk,
    get_uv_rect: impl Fn(&Block, &Face) -> Option<Rect>,
) -> ChunkMeshes {
    // Pre-allocate with reasonable capacity (average chunk has ~10k visible faces)
    let mut positions = Vec::with_capacity(40000);
    let mut normals = Vec::with_capacity(40000);
    let mut uvs = Vec::with_capacity(40000);
    let mut colors = Vec::with_capacity(40000);
    let mut indices = Vec::with_capacity(60000);

    // Separate water mesh data
    let mut water_positions = Vec::with_capacity(8000);
    let mut water_normals = Vec::with_capacity(8000);
    let mut water_uvs = Vec::with_capacity(8000);
    let mut water_colors = Vec::with_capacity(8000);
    let mut water_indices = Vec::with_capacity(12000);

    // First pass: find min/max Y for each column to avoid iterating all 256 levels
    let mut y_ranges = [[0u8; CHUNK_DEPTH]; CHUNK_WIDTH]; // max Y per column (0 = empty)

    for x in 0..CHUNK_WIDTH {
        for z in 0..CHUNK_DEPTH {
            // Scan from top down to find highest non-air block
            for y in (0..CHUNK_HEIGHT).rev() {
                if chunk.get_block(x, y, z) != Block::Air {
                    y_ranges[x][z] = (y + 1).min(255) as u8;
                    break;
                }
            }
        }
    }

    for x in 0..CHUNK_WIDTH {
        for z in 0..CHUNK_DEPTH {
            let max_y = y_ranges[x][z] as usize;
            if max_y == 0 {
                continue; // Empty column
            }

            for y in 0..max_y {
                let block = chunk.get_block(x, y, z);
                if block == Block::Air {
                    continue;
                }

                let fx = x as f32;
                let fy = y as f32;
                let fz = z as f32;

                if block == Block::Water {
                    // Generate water mesh separately
                    // Only render water faces adjacent to air (not other water or solid blocks)
                    // At chunk boundaries, don't render faces - they'll be handled by adjacent chunks
                    let neighbor_right = if x == CHUNK_WIDTH - 1 {
                        None
                    } else {
                        Some(chunk.get_block(x + 1, y, z))
                    };
                    let neighbor_left = if x == 0 {
                        None
                    } else {
                        Some(chunk.get_block(x - 1, y, z))
                    };
                    let neighbor_top = if y == CHUNK_HEIGHT - 1 {
                        Some(Block::Air)
                    } else {
                        Some(chunk.get_block(x, y + 1, z))
                    };
                    let neighbor_bottom = if y == 0 {
                        None
                    } else {
                        Some(chunk.get_block(x, y - 1, z))
                    };
                    let neighbor_back = if z == CHUNK_DEPTH - 1 {
                        None
                    } else {
                        Some(chunk.get_block(x, y, z + 1))
                    };
                    let neighbor_forward = if z == 0 {
                        None
                    } else {
                        Some(chunk.get_block(x, y, z - 1))
                    };

                    let faces = [
                        (Face::Right, neighbor_right == Some(Block::Air)),
                        (Face::Left, neighbor_left == Some(Block::Air)),
                        (Face::Top, neighbor_top == Some(Block::Air)),
                        (Face::Bottom, neighbor_bottom == Some(Block::Air)),
                        (Face::Back, neighbor_back == Some(Block::Air)),
                        (Face::Forward, neighbor_forward == Some(Block::Air)),
                    ];

                    for (face, visible) in faces {
                        if visible && let Some(rect) = get_uv_rect(&block, &face) {
                            add_water_face(
                                &mut water_positions,
                                &mut water_normals,
                                &mut water_uvs,
                                &mut water_colors,
                                &mut water_indices,
                                fx,
                                fy,
                                fz,
                                face,
                                rect,
                            );
                        }
                    }
                } else {
                    // Solid block
                    let faces = [
                        (
                            Face::Right,
                            x == CHUNK_WIDTH - 1 || chunk.get_block(x + 1, y, z).is_transparent(),
                        ),
                        (
                            Face::Left,
                            x == 0 || chunk.get_block(x - 1, y, z).is_transparent(),
                        ),
                        (
                            Face::Top,
                            y == CHUNK_HEIGHT - 1 || chunk.get_block(x, y + 1, z).is_transparent(),
                        ),
                        (
                            Face::Bottom,
                            y == 0 || chunk.get_block(x, y - 1, z).is_transparent(),
                        ),
                        (
                            Face::Back,
                            z == CHUNK_DEPTH - 1 || chunk.get_block(x, y, z + 1).is_transparent(),
                        ),
                        (
                            Face::Forward,
                            z == 0 || chunk.get_block(x, y, z - 1).is_transparent(),
                        ),
                    ];

                    for (face, visible) in faces {
                        if visible && let Some(rect) = get_uv_rect(&block, &face) {
                            let tint = if block == Block::Grass && face == Face::Top {
                                GRASS_TINT
                            } else {
                                NO_TINT
                            };

                            add_face(
                                &mut positions,
                                &mut normals,
                                &mut uvs,
                                &mut colors,
                                &mut indices,
                                fx,
                                fy,
                                fz,
                                face,
                                rect,
                                tint,
                            );
                        }
                    }
                }
            }
        }
    }

    ChunkMeshes {
        solid: ChunkMeshData {
            positions,
            normals,
            uvs,
            colors,
            indices,
        },
        water: ChunkMeshData {
            positions: water_positions,
            normals: water_normals,
            uvs: water_uvs,
            colors: water_colors,
            indices: water_indices,
        },
    }
}

/// Generate a lower-detail mesh by sampling blocks at intervals
/// Note: For LOD, we only generate solid blocks - water is skipped at distance
pub fn generate_chunk_mesh_data_lod(
    chunk: &Chunk,
    lod: LodLevel,
    get_uv_rect: impl Fn(&Block, &Face) -> Option<Rect>,
) -> ChunkMeshData {
    let block_size = lod.block_size();

    // For full detail, use the regular function and extract solid mesh
    if block_size == 1 {
        return generate_chunk_mesh_data(chunk, get_uv_rect).solid;
    }

    // Pre-allocate with smaller capacity for LOD (fewer faces)
    let mut positions = Vec::with_capacity(10000);
    let mut normals = Vec::with_capacity(10000);
    let mut uvs = Vec::with_capacity(10000);
    let mut colors = Vec::with_capacity(10000);
    let mut indices = Vec::with_capacity(15000);

    let step = block_size;
    let fsize = block_size as f32;

    // Find max Y to limit iteration
    let mut max_y = 0usize;
    for x in 0..CHUNK_WIDTH {
        for z in 0..CHUNK_DEPTH {
            for y in (0..CHUNK_HEIGHT).rev() {
                if chunk.get_block(x, y, z) != Block::Air {
                    max_y = max_y.max(y + 1);
                    break;
                }
            }
        }
    }
    // Round up to next step boundary
    let max_y = ((max_y + step - 1) / step) * step;

    // Iterate over block groups
    for x in (0..CHUNK_WIDTH).step_by(step) {
        for y in (0..max_y).step_by(step) {
            for z in (0..CHUNK_DEPTH).step_by(step) {
                // Find the dominant non-air block in this group
                let block = get_dominant_block(chunk, x, y, z, block_size);
                // Skip air and water for LOD (water is only rendered at full detail)
                if block == Block::Air || block == Block::Water {
                    continue;
                }

                let fx = x as f32;
                let fy = y as f32;
                let fz = z as f32;

                // Check neighbors for face visibility (at LOD scale)
                let faces = [
                    (
                        Face::Right,
                        x + step >= CHUNK_WIDTH
                            || is_group_transparent(chunk, x + step, y, z, block_size),
                    ),
                    (
                        Face::Left,
                        x == 0
                            || is_group_transparent(
                                chunk,
                                x.saturating_sub(step),
                                y,
                                z,
                                block_size,
                            ),
                    ),
                    (
                        Face::Top,
                        y + step >= CHUNK_HEIGHT
                            || is_group_transparent(chunk, x, y + step, z, block_size),
                    ),
                    (
                        Face::Bottom,
                        y == 0
                            || is_group_transparent(
                                chunk,
                                x,
                                y.saturating_sub(step),
                                z,
                                block_size,
                            ),
                    ),
                    (
                        Face::Back,
                        z + step >= CHUNK_DEPTH
                            || is_group_transparent(chunk, x, y, z + step, block_size),
                    ),
                    (
                        Face::Forward,
                        z == 0
                            || is_group_transparent(
                                chunk,
                                x,
                                y,
                                z.saturating_sub(step),
                                block_size,
                            ),
                    ),
                ];

                for (face, visible) in faces {
                    if visible && let Some(rect) = get_uv_rect(&block, &face) {
                        let tint = if block == Block::Grass && face == Face::Top {
                            GRASS_TINT
                        } else {
                            NO_TINT
                        };

                        add_face_scaled(
                            &mut positions,
                            &mut normals,
                            &mut uvs,
                            &mut colors,
                            &mut indices,
                            fx,
                            fy,
                            fz,
                            fsize,
                            face,
                            rect,
                            tint,
                        );
                    }
                }
            }
        }
    }

    ChunkMeshData {
        positions,
        normals,
        uvs,
        colors,
        indices,
    }
}

/// Get the dominant (most common non-air) block in a group
fn get_dominant_block(
    chunk: &Chunk,
    start_x: usize,
    start_y: usize,
    start_z: usize,
    size: usize,
) -> Block {
    let mut counts = [0u32; 6]; // Air, Stone, Dirt, Grass, Water, Sand

    let end_x = (start_x + size).min(CHUNK_WIDTH);
    let end_y = (start_y + size).min(CHUNK_HEIGHT);
    let end_z = (start_z + size).min(CHUNK_DEPTH);

    for x in start_x..end_x {
        for y in start_y..end_y {
            for z in start_z..end_z {
                let block = chunk.get_block(x, y, z);
                counts[block as usize] += 1;
            }
        }
    }

    // Find most common non-air block
    let mut best_block = Block::Air;
    let mut best_count = 0;

    for (i, &count) in counts.iter().enumerate().skip(1) {
        if count > best_count {
            best_count = count;
            best_block = Block::from(i as u8);
        }
    }

    // Only return a block if it fills at least 25% of the group
    let total = (end_x - start_x) * (end_y - start_y) * (end_z - start_z);
    if best_count as usize * 4 >= total {
        best_block
    } else {
        Block::Air
    }
}

/// Check if a block group is mostly transparent (air or water)
fn is_group_transparent(
    chunk: &Chunk,
    start_x: usize,
    start_y: usize,
    start_z: usize,
    size: usize,
) -> bool {
    let end_x = (start_x + size).min(CHUNK_WIDTH);
    let end_y = (start_y + size).min(CHUNK_HEIGHT);
    let end_z = (start_z + size).min(CHUNK_DEPTH);

    let mut transparent_count = 0;
    let mut total = 0;

    for x in start_x..end_x {
        for y in start_y..end_y {
            for z in start_z..end_z {
                if chunk.get_block(x, y, z).is_transparent() {
                    transparent_count += 1;
                }
                total += 1;
            }
        }
    }

    // Consider transparent if more than 50% transparent
    transparent_count * 2 > total
}

/// Add a scaled face for LOD meshes
fn add_face_scaled(
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    uvs: &mut Vec<[f32; 2]>,
    colors: &mut Vec<[f32; 4]>,
    indices: &mut Vec<u32>,
    x: f32,
    y: f32,
    z: f32,
    size: f32,
    face: Face,
    rect: Rect,
    tint: [f32; 4],
) {
    let start_idx = positions.len() as u32;

    let (u_min, v_min) = (rect.min.x, rect.min.y);
    let (u_max, v_max) = (rect.max.x, rect.max.y);

    match face {
        Face::Top => {
            positions.extend_from_slice(&[
                [x, y + size, z],
                [x, y + size, z + size],
                [x + size, y + size, z + size],
                [x + size, y + size, z],
            ]);
            normals.extend_from_slice(&[[0.0, 1.0, 0.0]; 4]);
            uvs.extend_from_slice(&[
                [u_min, v_min],
                [u_min, v_max],
                [u_max, v_max],
                [u_max, v_min],
            ]);
        }
        Face::Bottom => {
            positions.extend_from_slice(&[
                [x, y, z],
                [x + size, y, z],
                [x + size, y, z + size],
                [x, y, z + size],
            ]);
            normals.extend_from_slice(&[[0.0, -1.0, 0.0]; 4]);
            uvs.extend_from_slice(&[
                [u_min, v_min],
                [u_max, v_min],
                [u_max, v_max],
                [u_min, v_max],
            ]);
        }
        Face::Right => {
            positions.extend_from_slice(&[
                [x + size, y, z + size],
                [x + size, y, z],
                [x + size, y + size, z],
                [x + size, y + size, z + size],
            ]);
            normals.extend_from_slice(&[[1.0, 0.0, 0.0]; 4]);
            uvs.extend_from_slice(&[
                [u_min, v_max],
                [u_max, v_max],
                [u_max, v_min],
                [u_min, v_min],
            ]);
        }
        Face::Left => {
            positions.extend_from_slice(&[
                [x, y, z],
                [x, y, z + size],
                [x, y + size, z + size],
                [x, y + size, z],
            ]);
            normals.extend_from_slice(&[[-1.0, 0.0, 0.0]; 4]);
            uvs.extend_from_slice(&[
                [u_min, v_max],
                [u_max, v_max],
                [u_max, v_min],
                [u_min, v_min],
            ]);
        }
        Face::Back => {
            positions.extend_from_slice(&[
                [x, y, z + size],
                [x + size, y, z + size],
                [x + size, y + size, z + size],
                [x, y + size, z + size],
            ]);
            normals.extend_from_slice(&[[0.0, 0.0, 1.0]; 4]);
            uvs.extend_from_slice(&[
                [u_min, v_max],
                [u_max, v_max],
                [u_max, v_min],
                [u_min, v_min],
            ]);
        }
        Face::Forward => {
            positions.extend_from_slice(&[
                [x + size, y, z],
                [x, y, z],
                [x, y + size, z],
                [x + size, y + size, z],
            ]);
            normals.extend_from_slice(&[[0.0, 0.0, -1.0]; 4]);
            uvs.extend_from_slice(&[
                [u_min, v_max],
                [u_max, v_max],
                [u_max, v_min],
                [u_min, v_min],
            ]);
        }
    }

    colors.extend_from_slice(&[tint; 4]);

    indices.extend_from_slice(&[
        start_idx,
        start_idx + 1,
        start_idx + 2,
        start_idx,
        start_idx + 2,
        start_idx + 3,
    ]);
}

#[derive(Clone, Copy, Hash, Eq, PartialEq)]
pub enum Face {
    Top,
    Bottom,
    Left,
    Right,
    Back,
    Forward,
}

fn add_face(
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    uvs: &mut Vec<[f32; 2]>,
    colors: &mut Vec<[f32; 4]>,
    indices: &mut Vec<u32>,
    x: f32,
    y: f32,
    z: f32,
    face: Face,
    rect: Rect,
    tint: [f32; 4],
) {
    let start_idx = positions.len() as u32;

    let (u_min, v_min) = (rect.min.x, rect.min.y);
    let (u_max, v_max) = (rect.max.x, rect.max.y);

    match face {
        // Top face: looking down at the XZ plane
        // Vertices ordered CCW when viewed from above (+Y)
        Face::Top => {
            positions.extend_from_slice(&[
                [x, y + 1.0, z],             // 0: corner at (0,0) in XZ
                [x, y + 1.0, z + 1.0],       // 1: corner at (0,1) in XZ
                [x + 1.0, y + 1.0, z + 1.0], // 2: corner at (1,1) in XZ
                [x + 1.0, y + 1.0, z],       // 3: corner at (1,0) in XZ
            ]);
            normals.extend_from_slice(&[[0.0, 1.0, 0.0]; 4]);
            // Map X to U, Z to V
            uvs.extend_from_slice(&[
                [u_min, v_min], // 0: X=0, Z=0
                [u_min, v_max], // 1: X=0, Z=1
                [u_max, v_max], // 2: X=1, Z=1
                [u_max, v_min], // 3: X=1, Z=0
            ]);
        }
        // Bottom face: looking up at the XZ plane
        Face::Bottom => {
            positions.extend_from_slice(&[
                [x, y, z],             // 0
                [x + 1.0, y, z],       // 1
                [x + 1.0, y, z + 1.0], // 2
                [x, y, z + 1.0],       // 3
            ]);
            normals.extend_from_slice(&[[0.0, -1.0, 0.0]; 4]);
            // Map X to U, Z to V (mirrored for bottom)
            uvs.extend_from_slice(&[
                [u_min, v_min], // 0
                [u_max, v_min], // 1
                [u_max, v_max], // 2
                [u_min, v_max], // 3
            ]);
        }
        // Right face (+X): looking at it from +X direction
        // Horizontal axis is Z (left=+Z, right=-Z when looking at face)
        // Vertical axis is Y (bottom=y, top=y+1)
        Face::Right => {
            positions.extend_from_slice(&[
                [x + 1.0, y, z + 1.0],       // 0: bottom-left (high Z, low Y)
                [x + 1.0, y, z],             // 1: bottom-right (low Z, low Y)
                [x + 1.0, y + 1.0, z],       // 2: top-right (low Z, high Y)
                [x + 1.0, y + 1.0, z + 1.0], // 3: top-left (high Z, high Y)
            ]);
            normals.extend_from_slice(&[[1.0, 0.0, 0.0]; 4]);
            // For side faces: U maps horizontally, V maps vertically
            // v_max = bottom of texture, v_min = top of texture
            uvs.extend_from_slice(&[
                [u_min, v_max], // 0: left, bottom
                [u_max, v_max], // 1: right, bottom
                [u_max, v_min], // 2: right, top
                [u_min, v_min], // 3: left, top
            ]);
        }
        // Left face (-X): looking at it from -X direction
        Face::Left => {
            positions.extend_from_slice(&[
                [x, y, z],             // 0: bottom-left (low Z, low Y)
                [x, y, z + 1.0],       // 1: bottom-right (high Z, low Y)
                [x, y + 1.0, z + 1.0], // 2: top-right (high Z, high Y)
                [x, y + 1.0, z],       // 3: top-left (low Z, high Y)
            ]);
            normals.extend_from_slice(&[[-1.0, 0.0, 0.0]; 4]);
            uvs.extend_from_slice(&[
                [u_min, v_max], // 0: left, bottom
                [u_max, v_max], // 1: right, bottom
                [u_max, v_min], // 2: right, top
                [u_min, v_min], // 3: left, top
            ]);
        }
        // Back face (+Z): looking at it from +Z direction
        Face::Back => {
            positions.extend_from_slice(&[
                [x, y, z + 1.0],             // 0: bottom-left (low X, low Y)
                [x + 1.0, y, z + 1.0],       // 1: bottom-right (high X, low Y)
                [x + 1.0, y + 1.0, z + 1.0], // 2: top-right (high X, high Y)
                [x, y + 1.0, z + 1.0],       // 3: top-left (low X, high Y)
            ]);
            normals.extend_from_slice(&[[0.0, 0.0, 1.0]; 4]);
            uvs.extend_from_slice(&[
                [u_min, v_max], // 0: left, bottom
                [u_max, v_max], // 1: right, bottom
                [u_max, v_min], // 2: right, top
                [u_min, v_min], // 3: left, top
            ]);
        }
        // Forward face (-Z): looking at it from -Z direction
        Face::Forward => {
            positions.extend_from_slice(&[
                [x + 1.0, y, z],       // 0: bottom-left (high X, low Y)
                [x, y, z],             // 1: bottom-right (low X, low Y)
                [x, y + 1.0, z],       // 2: top-right (low X, high Y)
                [x + 1.0, y + 1.0, z], // 3: top-left (high X, high Y)
            ]);
            normals.extend_from_slice(&[[0.0, 0.0, -1.0]; 4]);
            uvs.extend_from_slice(&[
                [u_min, v_max], // 0: left, bottom
                [u_max, v_max], // 1: right, bottom
                [u_max, v_min], // 2: right, top
                [u_min, v_min], // 3: left, top
            ]);
        }
    }

    // Add vertex colors (4 vertices per face)
    colors.extend_from_slice(&[tint; 4]);

    indices.extend_from_slice(&[
        start_idx,
        start_idx + 1,
        start_idx + 2,
        start_idx,
        start_idx + 2,
        start_idx + 3,
    ]);
}

/// Add a water face with lowered top surface (like Minecraft)
fn add_water_face(
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    uvs: &mut Vec<[f32; 2]>,
    colors: &mut Vec<[f32; 4]>,
    indices: &mut Vec<u32>,
    x: f32,
    y: f32,
    z: f32,
    face: Face,
    rect: Rect,
) {
    let start_idx = positions.len() as u32;

    let (u_min, v_min) = (rect.min.x, rect.min.y);
    let (u_max, v_max) = (rect.max.x, rect.max.y);

    // Water top is slightly lower than a full block
    let water_top = y + WATER_HEIGHT;

    match face {
        Face::Top => {
            positions.extend_from_slice(&[
                [x, water_top, z],
                [x, water_top, z + 1.0],
                [x + 1.0, water_top, z + 1.0],
                [x + 1.0, water_top, z],
            ]);
            normals.extend_from_slice(&[[0.0, 1.0, 0.0]; 4]);
            uvs.extend_from_slice(&[
                [u_min, v_min],
                [u_min, v_max],
                [u_max, v_max],
                [u_max, v_min],
            ]);
        }
        Face::Bottom => {
            positions.extend_from_slice(&[
                [x, y, z],
                [x + 1.0, y, z],
                [x + 1.0, y, z + 1.0],
                [x, y, z + 1.0],
            ]);
            normals.extend_from_slice(&[[0.0, -1.0, 0.0]; 4]);
            uvs.extend_from_slice(&[
                [u_min, v_min],
                [u_max, v_min],
                [u_max, v_max],
                [u_min, v_max],
            ]);
        }
        Face::Right => {
            positions.extend_from_slice(&[
                [x + 1.0, y, z + 1.0],
                [x + 1.0, y, z],
                [x + 1.0, water_top, z],
                [x + 1.0, water_top, z + 1.0],
            ]);
            normals.extend_from_slice(&[[1.0, 0.0, 0.0]; 4]);
            uvs.extend_from_slice(&[
                [u_min, v_max],
                [u_max, v_max],
                [u_max, v_min],
                [u_min, v_min],
            ]);
        }
        Face::Left => {
            positions.extend_from_slice(&[
                [x, y, z],
                [x, y, z + 1.0],
                [x, water_top, z + 1.0],
                [x, water_top, z],
            ]);
            normals.extend_from_slice(&[[-1.0, 0.0, 0.0]; 4]);
            uvs.extend_from_slice(&[
                [u_min, v_max],
                [u_max, v_max],
                [u_max, v_min],
                [u_min, v_min],
            ]);
        }
        Face::Back => {
            positions.extend_from_slice(&[
                [x, y, z + 1.0],
                [x + 1.0, y, z + 1.0],
                [x + 1.0, water_top, z + 1.0],
                [x, water_top, z + 1.0],
            ]);
            normals.extend_from_slice(&[[0.0, 0.0, 1.0]; 4]);
            uvs.extend_from_slice(&[
                [u_min, v_max],
                [u_max, v_max],
                [u_max, v_min],
                [u_min, v_min],
            ]);
        }
        Face::Forward => {
            positions.extend_from_slice(&[
                [x + 1.0, y, z],
                [x, y, z],
                [x, water_top, z],
                [x + 1.0, water_top, z],
            ]);
            normals.extend_from_slice(&[[0.0, 0.0, -1.0]; 4]);
            uvs.extend_from_slice(&[
                [u_min, v_max],
                [u_max, v_max],
                [u_max, v_min],
                [u_min, v_min],
            ]);
        }
    }

    // White color - tinting handled by material
    colors.extend_from_slice(&[NO_TINT; 4]);

    indices.extend_from_slice(&[
        start_idx,
        start_idx + 1,
        start_idx + 2,
        start_idx,
        start_idx + 2,
        start_idx + 3,
    ]);
}
