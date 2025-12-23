use crate::{Block, CHUNK_DEPTH, CHUNK_HEIGHT, CHUNK_WIDTH, Chunk};
use bevy::asset::RenderAssetUsages;
use bevy::mesh::Indices;
use bevy::prelude::*;
use bevy::render::render_resource::PrimitiveTopology;

pub fn generate_chunk_mesh(chunk: &Chunk) -> Mesh {
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut colors = Vec::new();
    let mut indices = Vec::new();

    for x in 0..CHUNK_WIDTH {
        for y in 0..CHUNK_HEIGHT {
            for z in 0..CHUNK_DEPTH {
                let block = chunk.get_block(x, y, z);
                if block == Block::Air {
                    continue;
                }

                let base_color = match block {
                    Block::Stone => [0.5, 0.5, 0.5, 1.0],
                    Block::Dirt => [0.4, 0.2, 0.1, 1.0],
                    Block::Grass => [0.1, 0.8, 0.1, 1.0],
                    _ => [1.0, 1.0, 1.0, 1.0],
                };

                let fx = x as f32;
                let fy = y as f32;
                let fz = z as f32;

                // Right (+X)
                if x == CHUNK_WIDTH - 1 || chunk.get_block(x + 1, y, z) == Block::Air {
                    add_face(
                        &mut positions,
                        &mut normals,
                        &mut colors,
                        &mut indices,
                        fx,
                        fy,
                        fz,
                        Face::Right,
                        shade(base_color, 0.8),
                    );
                }
                // Left (-X)
                if x == 0 || chunk.get_block(x - 1, y, z) == Block::Air {
                    add_face(
                        &mut positions,
                        &mut normals,
                        &mut colors,
                        &mut indices,
                        fx,
                        fy,
                        fz,
                        Face::Left,
                        shade(base_color, 0.8),
                    );
                }
                // Top (+Y)
                if y == CHUNK_HEIGHT - 1 || chunk.get_block(x, y + 1, z) == Block::Air {
                    add_face(
                        &mut positions,
                        &mut normals,
                        &mut colors,
                        &mut indices,
                        fx,
                        fy,
                        fz,
                        Face::Top,
                        shade(base_color, 1.0),
                    );
                }
                // Bottom (-Y)
                if y == 0 || chunk.get_block(x, y - 1, z) == Block::Air {
                    add_face(
                        &mut positions,
                        &mut normals,
                        &mut colors,
                        &mut indices,
                        fx,
                        fy,
                        fz,
                        Face::Bottom,
                        shade(base_color, 0.5),
                    );
                }
                // Back (+Z)
                if z == CHUNK_DEPTH - 1 || chunk.get_block(x, y, z + 1) == Block::Air {
                    add_face(
                        &mut positions,
                        &mut normals,
                        &mut colors,
                        &mut indices,
                        fx,
                        fy,
                        fz,
                        Face::Back,
                        shade(base_color, 0.9),
                    );
                }
                // Forward (-Z)
                if z == 0 || chunk.get_block(x, y, z - 1) == Block::Air {
                    add_face(
                        &mut positions,
                        &mut normals,
                        &mut colors,
                        &mut indices,
                        fx,
                        fy,
                        fz,
                        Face::Forward,
                        shade(base_color, 0.9),
                    );
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
    .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, colors)
    .with_inserted_indices(Indices::U32(indices))
}

fn shade(mut color: [f32; 4], factor: f32) -> [f32; 4] {
    color[0] *= factor;
    color[1] *= factor;
    color[2] *= factor;
    color
}

enum Face {
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
    colors: &mut Vec<[f32; 4]>,
    indices: &mut Vec<u32>,
    x: f32,
    y: f32,
    z: f32,
    face: Face,
    color: [f32; 4],
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

    colors.extend_from_slice(&[color; 4]);
    indices.extend_from_slice(&[
        start_idx,
        start_idx + 1,
        start_idx + 2,
        start_idx,
        start_idx + 2,
        start_idx + 3,
    ]);
}
