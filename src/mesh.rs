use crate::{Block, CHUNK_DEPTH, CHUNK_HEIGHT, CHUNK_WIDTH, Chunk};
use bevy::asset::RenderAssetUsages;
use bevy::math::Rect; // Added Rect import
use bevy::mesh::Indices;
use bevy::prelude::*;
use bevy::render::render_resource::PrimitiveTopology;

pub fn generate_chunk_mesh(
    chunk: &Chunk,
    get_uv_rect: impl Fn(&Block, &Face) -> Option<Rect>,
) -> Mesh {
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new(); // Changed from colors
    let mut indices = Vec::new();

    for x in 0..CHUNK_WIDTH {
        for y in 0..CHUNK_HEIGHT {
            for z in 0..CHUNK_DEPTH {
                let block = chunk.get_block(x, y, z);
                if block == Block::Air {
                    continue;
                }

                let fx = x as f32;
                let fy = y as f32;
                let fz = z as f32;

                let faces = [
                    (
                        Face::Right,
                        x == CHUNK_WIDTH - 1 || chunk.get_block(x + 1, y, z) == Block::Air,
                    ),
                    (
                        Face::Left,
                        x == 0 || chunk.get_block(x - 1, y, z) == Block::Air,
                    ),
                    (
                        Face::Top,
                        y == CHUNK_HEIGHT - 1 || chunk.get_block(x, y + 1, z) == Block::Air,
                    ),
                    (
                        Face::Bottom,
                        y == 0 || chunk.get_block(x, y - 1, z) == Block::Air,
                    ),
                    (
                        Face::Back,
                        z == CHUNK_DEPTH - 1 || chunk.get_block(x, y, z + 1) == Block::Air,
                    ),
                    (
                        Face::Forward,
                        z == 0 || chunk.get_block(x, y, z - 1) == Block::Air,
                    ),
                ];

                for (face, visible) in faces {
                    if visible {
                        if let Some(rect) = get_uv_rect(&block, &face) {
                            add_face(
                                &mut positions,
                                &mut normals,
                                &mut uvs, // Changed from colors
                                &mut indices,
                                fx,
                                fy,
                                fz,
                                face,
                                rect, // Changed from color
                            );
                        }
                    }
                }
            }
        }
    }

    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
    .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs) // Changed from ATTRIBUTE_COLOR
    .with_inserted_indices(Indices::U32(indices))
}

// Removed shade function

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
    uvs: &mut Vec<[f32; 2]>, // Changed from colors
    indices: &mut Vec<u32>,
    x: f32,
    y: f32,
    z: f32,
    face: Face,
    rect: Rect, // Changed from color
) {
    let start_idx = positions.len() as u32;

    match face {
        // All vertices follow CCW order viewed from outside the cube
        Face::Top => {
            positions.extend_from_slice(&[
                [x, y + 1.0, z],
                [x, y + 1.0, z + 1.0],
                [x + 1.0, y + 1.0, z + 1.0],
                [x + 1.0, y + 1.0, z],
            ]);
            normals.extend_from_slice(&[[0.0, 1.0, 0.0]; 4]);
        }
        Face::Bottom => {
            positions.extend_from_slice(&[
                [x, y, z],
                [x + 1.0, y, z],
                [x + 1.0, y, z + 1.0],
                [x, y, z + 1.0],
            ]);
            normals.extend_from_slice(&[[0.0, -1.0, 0.0]; 4]);
        }
        Face::Right => {
            positions.extend_from_slice(&[
                [x + 1.0, y, z + 1.0],
                [x + 1.0, y, z],
                [x + 1.0, y + 1.0, z],
                [x + 1.0, y + 1.0, z + 1.0],
            ]);
            normals.extend_from_slice(&[[1.0, 0.0, 0.0]; 4]);
        }
        Face::Left => {
            positions.extend_from_slice(&[
                [x, y, z],
                [x, y, z + 1.0],
                [x, y + 1.0, z + 1.0],
                [x, y + 1.0, z],
            ]);
            normals.extend_from_slice(&[[-1.0, 0.0, 0.0]; 4]);
        }
        Face::Back => {
            positions.extend_from_slice(&[
                [x, y, z + 1.0],
                [x + 1.0, y, z + 1.0],
                [x + 1.0, y + 1.0, z + 1.0],
                [x, y + 1.0, z + 1.0],
            ]);
            normals.extend_from_slice(&[[0.0, 0.0, 1.0]; 4]);
        }
        Face::Forward => {
            positions.extend_from_slice(&[
                [x + 1.0, y, z],
                [x, y, z],
                [x, y + 1.0, z],
                [x + 1.0, y + 1.0, z],
            ]);
            normals.extend_from_slice(&[[0.0, 0.0, -1.0]; 4]);
        }
    }

    // Map rect to UVs
    // Assume vertices are ordered: 0, 1, 2, 3 corresponding to corners
    // Standard quad order above seems to be:
    // Top: (0,1), (0,1+1), (1,1+1), (1,1) -> Top-Left, Bottom-Left, Bottom-Right, Top-Right (in XZ plane)
    // Wait, let's trace:
    // Top: [x, y+1, z], [x, y+1, z+1], ...
    // p0: (0,0) offset
    // p1: (0,1) offset
    // p2: (1,1) offset
    // p3: (1,0) offset
    // So UVs should follow similar 0,0 -> 0,1 -> 1,1 -> 1,0 pattern to match orientation.
    // Rect.min is top-left in texture space? No, usually UV (0,0) is top-left in Bevy images.

    let (u_min, v_min) = (rect.min.x, rect.min.y);
    let (u_max, v_max) = (rect.max.x, rect.max.y);

    uvs.extend_from_slice(&[
        [u_min, v_min], // 0
        [u_min, v_max], // 1
        [u_max, v_max], // 2
        [u_max, v_min], // 3
    ]);

    indices.extend_from_slice(&[
        start_idx,
        start_idx + 1,
        start_idx + 2,
        start_idx,
        start_idx + 2,
        start_idx + 3,
    ]);
}
