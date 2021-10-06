#include <metal_stdlib>

using namespace metal;

constant uint2 kBlockSize [[function_constant(0)]];
constant uint kPixelSearchRadius [[function_constant(1)]];
constant uint kMaxBlockSize = 64;

float computeSumOfAbsoluteDifference(thread const half* frame_block_data,
                                     threadgroup const half* tile_data,
                                     const uint TileWidth,
                                     const uint2 offset_in_tile) {
  float total = 0.0;
  for (uint y = 0; y < kBlockSize.y; ++y) {
    for (uint x = 0; x < kBlockSize.x; ++x) {
      uint i = y * kBlockSize.x + x;
      uint t = (offset_in_tile.y + y) * TileWidth + (offset_in_tile.x + x);
      total += abs(frame_block_data[i] - tile_data[t]);
    }
  }
  return total;
}

typedef struct {
    float2 vector;
    float residual;
} MotionVectorWithResidual;

MotionVectorWithResidual findBestMotionVector(
  thread const half* frame_block_data,
  threadgroup const half* tile_data,
  const uint TileWidth,
  const uint2 block_offset_in_tile) {

  int2 lds_offsets[9] = {
    int2(0,0),
    int2(-2,0), int2(2,0),
    int2(0,-2), int2(0,2),
    int2(-1,-1), int2(-1,1),
    int2(1,-1), int2(1,1),
  };

  uint2 cur_offset = block_offset_in_tile;
  uint min_index = 0;
  float residual = computeSumOfAbsoluteDifference(
    frame_block_data, tile_data, TileWidth,
    cur_offset);

  // Large Diamond Search
  while (true) {
    uint2 next_offset;
    for (uint i = 1; i < 9; ++i) {
      int2 offset = int2(cur_offset) + lds_offsets[i];
      if (any(abs(offset - int2(cur_offset)) > int2(kPixelSearchRadius, kPixelSearchRadius))) {
        continue;
      }

      float sad = computeSumOfAbsoluteDifference(
        frame_block_data, tile_data, TileWidth,
        uint2(offset));
      if (sad < residual) {
        residual = sad;
        min_index = i;
        next_offset = uint2(offset);
      }
    }

    if (min_index == 0) {
      break;
    }

    cur_offset = next_offset;
    min_index = 0;
  }

  // Small Diamond Search
  int2 sds_offsets[4] = {
    int2(-1,0), int2(1,0),
    int2(0,-1), int2(0,1)
  };

  uint2 next_offset = cur_offset;
  for (uint i = 0; i < 4; ++i) {
    int2 offset = int2(cur_offset) + sds_offsets[i];
    if (any(abs(offset - int2(cur_offset)) > int2(kPixelSearchRadius, kPixelSearchRadius))) {
      continue;
    }

    float sad = computeSumOfAbsoluteDifference(
      frame_block_data, tile_data, TileWidth,
      uint2(offset));
    if (sad < residual) {
      residual = sad;
      next_offset = uint2(offset);
    }
  }

  MotionVectorWithResidual result;
  result.vector = float2(next_offset) - float2(block_offset_in_tile);
  result.residual = residual;

  return result;
}

kernel void motionVectorSearch(
  uint2 GlobalID [[ thread_position_in_grid ]],
  uint2 GroupID [[ threadgroup_position_in_grid ]],
  uint2 LocalID [[ thread_position_in_threadgroup ]],
  uint2 ThreadsPerGroup [[ threads_per_threadgroup ]],
  texture2d<half, access::read> frame[[texture(0)]],
  texture2d<half, access::read> previous[[texture(1)]],
  texture2d<half, access::write> mv_texture[[texture(2)]],
  threadgroup half* tile_data [[threadgroup(0)]]
) {
  // Alias some names to make it easier to understand code in context.
  const uint2 BlockIDInGroup = LocalID;
  const uint2 BlocksPerGroup = ThreadsPerGroup;
  const uint2 TileSize = BlocksPerGroup * kBlockSize + 2 * kPixelSearchRadius;

  half frame_block_data[kMaxBlockSize * kMaxBlockSize];

  // Load the current frame block into registers.
  for (uint by = 0; by < kBlockSize.y; ++by) {
    for (uint bx = 0; bx < kBlockSize.x; ++bx) {
      uint2 global_idx = GlobalID * kBlockSize + uint2(bx, by);
      if (global_idx.x >= frame.get_width() || global_idx.y >= frame.get_height()) {
        continue;
      }
      frame_block_data[by * kBlockSize.x + bx] = frame.read(global_idx)[0];
    }
  }

  // top left corner, in pixels, of the first block in the group
  uint2 first_block_origin = GroupID * BlocksPerGroup * kBlockSize;

  // Load pixels |kSearchRadius| away from the top left corner of a block,
  // offset by half the block size.
  int2 tile_origin_offset = -int2(kPixelSearchRadius);

  int2 tile_origin = int2(first_block_origin) + tile_origin_offset;

  // lpt: loads per thread
  uint2 lpt = (TileSize + ThreadsPerGroup - 1) / ThreadsPerGroup;
  // Load a (TileSize x TileSize) region from |previous| into |tile_data|
  for (uint ly = 0; ly < lpt.y; ++ly) {
    for (uint lx = 0; lx < lpt.x; ++lx) {
      uint2 offset_in_tile = lpt * LocalID + uint2(lx, ly);
      int2 global_idx = tile_origin + int2(offset_in_tile);
      if (global_idx.x < 0 ||
          global_idx.y < 0 ||
          uint(global_idx.x) >= previous.get_width() ||
          uint(global_idx.y) >= previous.get_height()) {
        continue;
      }
      tile_data[offset_in_tile.y * TileSize.x + offset_in_tile.x] =
        previous.read(uint2(global_idx))[0];
    }
  }

  // Wait for all threads to load from main memory into tile_data
  threadgroup_barrier(mem_flags::mem_threadgroup);

  uint2 block_origin = first_block_origin + BlockIDInGroup * kBlockSize;
  uint2 block_offset_in_tile = uint2(int2(block_origin) - tile_origin);

  MotionVectorWithResidual mv = findBestMotionVector(
    frame_block_data, tile_data, TileSize.x, block_offset_in_tile);

  mv_texture.write(half4(mv.vector.x, mv.vector.y, 0.0, 1.0),
                   GroupID * BlocksPerGroup + BlockIDInGroup);
};
