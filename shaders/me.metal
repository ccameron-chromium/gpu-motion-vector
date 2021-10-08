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

kernel void motionVectorSearch(
  uint2 GlobalID [[ thread_position_in_grid ]],
  uint2 GroupID [[ threadgroup_position_in_grid ]],
  uint2 LocalID [[ thread_position_in_threadgroup ]],
  uint2 ThreadsPerGroup [[ threads_per_threadgroup ]],
  texture2d<half, access::read> frame[[texture(0)]],
  texture2d<half, access::read> previous[[texture(1)]],
  texture2d<half, access::write> mv_texture[[texture(2)]],
  threadgroup half* tile_data[[threadgroup(0)]]
) {
  const uint2 BlockID = GroupID;
  const uint LocalIndex = LocalID.y * ThreadsPerGroup.x + LocalID.x;
  const uint LocalThreadCount = ThreadsPerGroup.x * ThreadsPerGroup.y;

  half frame_block_data[kMaxBlockSize * kMaxBlockSize];

  uint2 block_origin = BlockID * kBlockSize;

  const uint2 TileSize = kBlockSize + 2 * kPixelSearchRadius;
  const int2 tile_origin = int2(block_origin) - kPixelSearchRadius;
  const uint2 block_offset_in_tile = uint2(kPixelSearchRadius, kPixelSearchRadius);

  // Cooperative load into |tile_data|
  for (uint flat_idx = LocalIndex;
            flat_idx < TileSize.x * TileSize.y;
            flat_idx += LocalThreadCount) {
    uint y = flat_idx / TileSize.x;
    uint x = flat_idx - y * TileSize.x;
    int2 global_idx = tile_origin + int2(x, y);
    global_idx.y = previous.get_height() - global_idx.y;
    tile_data[flat_idx] = previous.read(uint2(clamp(
        global_idx,
        int2(0,0),
        int2(previous.get_width(), previous.get_height()))))[0];
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Load the current frame block into registers.
  for (uint by = 0; by < kBlockSize.y; ++by) {
    for (uint bx = 0; bx < kBlockSize.x; ++bx) {
      uint2 global_idx = min(
        block_origin + uint2(bx, by),
        uint2(frame.get_width() - 1, frame.get_height() - 1)
      );
      global_idx.y = previous.get_height() - global_idx.y;
      frame_block_data[by * kBlockSize.x + bx] = frame.read(global_idx)[0];
    }
  }

  constexpr uint kLDSCount = 1 + 8;
  constexpr int2 lds_offsets[kLDSCount] = {
    int2(0,0),

    // abs(x) + abs(y) == 2
    int2(-2,0), int2(2,0),
    int2(0,-2), int2(0,2),
    int2(-1,-1), int2(-1,1),
    int2(1,-1), int2(1,1),
  };

  int2 cur_offset = int2(0, 0);

  float best_residual;
  threadgroup float residuals[kLDSCount];

  while (true && LocalIndex < kLDSCount) {
    int2 offset = cur_offset + lds_offsets[LocalIndex];
    if (any(abs(offset) > int2(kPixelSearchRadius, kPixelSearchRadius))) {
      residuals[LocalIndex] = INFINITY;
    } else {
      residuals[LocalIndex] = computeSumOfAbsoluteDifference(
        frame_block_data, tile_data, TileSize.x,
        uint2(int2(block_offset_in_tile) + offset));
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    best_residual = residuals[0];
    uint best_i = 0;
    int2 next_offset = cur_offset;
    for (uint i = 1; i < kLDSCount; ++i) {
      float r = residuals[i];
      if (r < best_residual) {
        best_residual = r;
        best_i = i;
        next_offset = cur_offset + lds_offsets[i];
      }
    }

    if (best_i == 0) {
      break;
    }

    cur_offset = next_offset;
  }

  // Small Diamond Search
  constexpr uint kSDSCount = 4;
  constexpr int2 sds_offsets[kSDSCount] = {
    // abs(x) + abs(y) == 1
    int2(-1,0), int2(1,0),
    int2(0,-1), int2(0,1)
  };

  if (LocalIndex < kSDSCount) {
    int2 offset = cur_offset + sds_offsets[LocalIndex];
    if (any(abs(offset) > int2(kPixelSearchRadius, kPixelSearchRadius))) {
      residuals[LocalIndex] = INFINITY;
    } else {
      residuals[LocalIndex] = computeSumOfAbsoluteDifference(
        frame_block_data, tile_data, TileSize.x,
        uint2(int2(block_offset_in_tile) + offset));
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (LocalIndex == 0) {
    int2 best_offset = cur_offset;
    for (uint i = 0; i < kSDSCount; ++i) {
      float r = residuals[i];
      if (r < best_residual) {
        best_residual = r;
        best_offset = cur_offset + sds_offsets[i];
      }
    }

    mv_texture.write(half4(float(best_offset.x), float(best_offset.y), 0.0, 1.0),
                  BlockID);
  }
};
