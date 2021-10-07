// clang++ main.mm -framework Metal -framework MetalKit -framework Cocoa -framework QuartzCore -framework IOSurface
#include <Metal/Metal.h>
#include <MetalKit/MetalKit.h>
#include <IOSurface/IOSurface.h>

#include <sstream>
#include <fstream>
#include <string>
#include <vector>

const MTLPixelFormat pixelFormat = MTLPixelFormatBGRA8Unorm;
unsigned int width = 1280;
unsigned int height = 720;
const int kBlockWidth = 16;
const int kBlockHeight = 16;
constexpr unsigned int kPixelSearchRadius = 32;
constexpr unsigned int kTileSize = 128;

id<MTLDevice> device = nil;
id<MTLLibrary> library = nil;
id<MTLCommandQueue> commandQueue;
CAMetalLayer* metalLayer = nil;

id<MTLRenderPipelineState> renderPipelineState = nil;
id<MTLRenderPipelineState> reconstructRenderPipelineState = nil;
id<MTLRenderPipelineState> motionVectorSearchRenderPipelineState = nil;
id<MTLRenderPipelineState> motionVectorDrawRenderPipelineState = nil;
id<MTLComputePipelineState> motionVectorSearchComputePipelineState = nil;

size_t frame_index = 0;
size_t previous_frame_index = 0;

enum DisplayMode {
  DMPreviousFrame,
  DMCurrentFrame,
  DMReconstructedFrame,
};
DisplayMode display_mode = DMCurrentFrame;
bool display_motion_vectors = false;

std::vector<id<MTLTexture>> frames;

id<MTLTexture> motion_vector_texture;

#define CHECK(x) \
  do { \
    if (!(x)) { \
      fprintf(stderr, "Check \"%s\" failed at %s:%d\n", #x, __FILE__, __LINE__); \
      exit(1); \
    } \
  } while (0);

IOSurfaceRef LoadImageToIOSurface(const char* filename, bool use_nv12) {
  printf("Loading %s\n", filename);
  NSString* string = [NSString stringWithUTF8String:filename];
  CHECK(string);
  NSData* data = [NSData dataWithContentsOfFile:string];
  if (!data)
    return nullptr;

  CGDataProviderRef data_provider = CGDataProviderCreateWithCFData(
      (__bridge CFDataRef) data);
  CGImageRef image = CGImageCreateWithPNGDataProvider(
      data_provider, nullptr, true, kCGRenderingIntentDefault);
  CHECK(image);
  int image_width = CGImageGetWidth(image);
  int image_height = CGImageGetHeight(image);

  CGDataProviderRef image_data_provider = CGImageGetDataProvider(image);
  CHECK(image_data_provider);

  CFDataRef image_data = CGDataProviderCopyData(image_data_provider);
  CHECK(image_data);

  const uint8_t* image_data_ptr = CFDataGetBytePtr(image_data);
  CHECK(image_data_ptr);
  const size_t image_data_size = CFDataGetLength(image_data);
  CHECK(image_data_size);
  CHECK(image_data_size == 4 * image_width * image_height);

  IOSurfaceRef io_surface = nullptr;
  if (use_nv12) {
    CHECK(!"This isn't implemented");
  } else {
    uint32_t io_format = 'BGRA';
    const size_t bytes_per_pixel = 4;
    const size_t bytes_per_row = IOSurfaceAlignProperty(
        kIOSurfaceBytesPerRow, image_width * bytes_per_pixel);
    const size_t bytes_total = IOSurfaceAlignProperty(
        kIOSurfaceAllocSize, image_height * bytes_per_row);
    NSDictionary *options = @{
        (id)kIOSurfaceWidth: @(image_width),
        (id)kIOSurfaceHeight: @(image_height),
        (id)kIOSurfacePixelFormat: @(io_format),
        (id)kIOSurfaceBytesPerElement: @(bytes_per_pixel),
        (id)kIOSurfaceBytesPerRow: @(bytes_per_row),
        (id)kIOSurfaceAllocSize: @(bytes_total),
    };
    io_surface = IOSurfaceCreate(
        (__bridge CFDictionaryRef)options);
    CHECK(io_surface);

    IOSurfaceLock(io_surface, kIOSurfaceLockAvoidSync, nullptr);
    uint8_t* io_surface_data_ptr = (uint8_t*)IOSurfaceGetBaseAddressOfPlane(
        io_surface, 0);
    for (int row = 0; row < image_height; ++row) {
      const void* src = image_data_ptr + row * 4 * image_width;
      void* dst = io_surface_data_ptr + row * bytes_per_row;
      memcpy(dst, src, 4 * image_width);
    }
    IOSurfaceUnlock(io_surface, kIOSurfaceLockAvoidSync, nullptr);
  }

  CFRelease(image_data);
  CFRelease(image);
  return io_surface;
}

id<MTLTexture> IOSurfaceToTexture(IOSurfaceRef io_surface) {
  MTLTextureDescriptor* tex_desc = [MTLTextureDescriptor new];
  [tex_desc setTextureType:MTLTextureType2D];
  [tex_desc setUsage:MTLTextureUsageShaderRead];
  [tex_desc setPixelFormat:MTLPixelFormatRGBA8Unorm];
  [tex_desc setWidth:IOSurfaceGetWidthOfPlane(io_surface, 0)];
  [tex_desc setHeight:IOSurfaceGetHeightOfPlane(io_surface, 0)];
  [tex_desc setDepth:1];
  [tex_desc setMipmapLevelCount:1];
  [tex_desc setArrayLength:1];
  [tex_desc setSampleCount:1];
  [tex_desc setStorageMode:MTLStorageModeManaged];
  id<MTLTexture> texture = [device newTextureWithDescriptor:tex_desc
                                                  iosurface:io_surface
                                                      plane:0];
  CHECK(texture);
  return texture;
}

id<MTLTexture> CreateMotionVectorTexture(size_t block_width, size_t block_height) {
  MTLTextureDescriptor* tex_desc = [MTLTextureDescriptor new];
  [tex_desc setTextureType:MTLTextureType2D];
  [tex_desc setUsage:MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite | MTLTextureUsageRenderTarget];
  [tex_desc setPixelFormat:MTLPixelFormatRG16Float];
  [tex_desc setWidth:width / block_width];
  [tex_desc setHeight:height / block_height];
  [tex_desc setDepth:1];
  [tex_desc setMipmapLevelCount:1];
  [tex_desc setArrayLength:1];
  [tex_desc setSampleCount:1];
  [tex_desc setStorageMode:MTLStorageModePrivate];
  id<MTLTexture> texture = [device newTextureWithDescriptor:tex_desc];
  CHECK(texture);
  return texture;
}

void LoadShaders() {
  {
    std::ifstream file("shaders.txt");
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string string = buffer.str();
    const char* cSource = string.c_str();

    NSError *error = NULL;
    NSString *libraryFile = [[NSBundle mainBundle] pathForResource:@"shaders" ofType:@"metallib"];
    library = [device newLibraryWithFile:libraryFile error:&error];
    if (!library || error) {
        NSLog(@"Failed to load library: %@", error);
        exit(1);
    }
  }

  {
    id<MTLFunction> vertexFunction = [library newFunctionWithName:@"vertexShader"];
    id<MTLFunction> fragmentFunction = [library newFunctionWithName:@"fragmentShader"];

    NSError* error = nil;
    MTLRenderPipelineDescriptor* desc = [[MTLRenderPipelineDescriptor alloc] init];
    desc.label = @"Simple Pipeline";
    desc.vertexFunction = vertexFunction;
    desc.fragmentFunction = fragmentFunction;
    desc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
    renderPipelineState = [device newRenderPipelineStateWithDescriptor:desc
                                                                 error:&error];
    if (error)
      NSLog(@"Failed to create render pipeline state: %@", error);
  }

  {
    id<MTLFunction> vertexFunction = [library newFunctionWithName:@"reconstructVertexShader"];
    id<MTLFunction> fragmentFunction = [library newFunctionWithName:@"reconstructFragmentShader"];

    NSError* error = nil;
    MTLRenderPipelineDescriptor* desc = [[MTLRenderPipelineDescriptor alloc] init];
    desc.label = @"Simple Pipeline";
    desc.vertexFunction = vertexFunction;
    desc.fragmentFunction = fragmentFunction;
    desc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
    reconstructRenderPipelineState = [device newRenderPipelineStateWithDescriptor:desc
                                                                 error:&error];
    if (error)
      NSLog(@"Failed to create render pipeline state: %@", error);
  }

  {
    id<MTLFunction> vertexFunction = [library newFunctionWithName:@"motionVectorSearchVertexShader"];
    id<MTLFunction> fragmentFunction = [library newFunctionWithName:@"motionVectorSearchFragmentShader"];

    NSError* error = nil;
    MTLRenderPipelineDescriptor* desc = [[MTLRenderPipelineDescriptor alloc] init];
    desc.label = @"Simple Pipeline";
    desc.vertexFunction = vertexFunction;
    desc.fragmentFunction = fragmentFunction;
    desc.colorAttachments[0].pixelFormat = MTLPixelFormatRG16Float;
    motionVectorSearchRenderPipelineState = [device newRenderPipelineStateWithDescriptor:desc
                                                                 error:&error];
    if (error)
      NSLog(@"Failed to create render pipeline state: %@", error);
    CHECK(motionVectorSearchRenderPipelineState);
  }

  {
    id<MTLFunction> vertexFunction = [library newFunctionWithName:@"motionVectorDrawVertexShader"];
    id<MTLFunction> fragmentFunction = [library newFunctionWithName:@"motionVectorDrawFragmentShader"];

    NSError* error = nil;
    MTLRenderPipelineDescriptor* desc = [[MTLRenderPipelineDescriptor alloc] init];
    desc.label = @"Simple Pipeline";
    desc.vertexFunction = vertexFunction;
    desc.fragmentFunction = fragmentFunction;
    desc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
    desc.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
    desc.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
    motionVectorDrawRenderPipelineState = [device newRenderPipelineStateWithDescriptor:desc
                                                                 error:&error];
    if (error)
      NSLog(@"Failed to create render pipeline state: %@", error);
    CHECK(motionVectorDrawRenderPipelineState);
  }

  {
    MTLFunctionConstantValues* constantValues = [MTLFunctionConstantValues new];
    uint32_t blockSize[2] = {kBlockWidth, kBlockHeight};
    [constantValues setConstantValue:blockSize type:MTLDataTypeUInt2 atIndex:0];
    [constantValues setConstantValue:&kPixelSearchRadius type:MTLDataTypeUInt atIndex:1];

    NSError* error = nil;
    id<MTLFunction> computeFunction = [library newFunctionWithName:@"motionVectorSearch"
                                                    constantValues:constantValues
                                                             error:&error];
    if (error)
      NSLog(@"Failed to create compute pipeline state: %@", error);
    CHECK(computeFunction);

    motionVectorSearchComputePipelineState =
      [device newComputePipelineStateWithFunction:computeFunction error:&error];
    if (error)
      NSLog(@"Failed to create compute pipeline state: %@", error);
    CHECK(motionVectorSearchComputePipelineState);
  }
}

void ComputeMotionVectors(unsigned int block_width, unsigned int block_height) {
  unsigned int width_in_blocks = (width + block_width - 1) / block_width;
  unsigned int height_in_blocks = (height + block_height - 1) / block_height;

  unsigned int threadgroup_memory_for_tile =
    2 * (block_width + 2 * kPixelSearchRadius) * (block_height + 2 * kPixelSearchRadius);

  id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
  {
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [encoder setComputePipelineState:motionVectorSearchComputePipelineState];
    [encoder setTexture:frames[frame_index] atIndex:0];
    [encoder setTexture:frames[previous_frame_index] atIndex:1];
    [encoder setTexture:motion_vector_texture atIndex:2];
    [encoder setThreadgroupMemoryLength:threadgroup_memory_for_tile
                                atIndex:0];
    [encoder dispatchThreadgroups:MTLSizeMake(width_in_blocks, height_in_blocks, 1)
             threadsPerThreadgroup:MTLSizeMake(
               8, 8, 1)
    ];
    [encoder endEncoding];
  }
  [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> cb) {
    CFTimeInterval executionDuration = cb.GPUEndTime - cb.GPUStartTime;
    NSLog(@"Execution time: %f", executionDuration);
  }];
  [commandBuffer commit];
}

void ComputeMotionVectorUsingDraw() {
  id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];

  id<MTLRenderCommandEncoder> encoder = nil;
  {
    MTLRenderPassDescriptor* desc = [MTLRenderPassDescriptor renderPassDescriptor];
    desc.colorAttachments[0].texture = motion_vector_texture;
    desc.colorAttachments[0].loadAction = MTLLoadActionClear;
    desc.colorAttachments[0].storeAction = MTLStoreActionStore;
    desc.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 1);
    encoder = [commandBuffer renderCommandEncoderWithDescriptor:desc];
  }

  {
    MTLViewport viewport;
    viewport.originX = 0;
    viewport.originY = 0;
    viewport.width = width / kBlockWidth;
    viewport.height = height / kBlockHeight;
    viewport.znear = -1.0;
    viewport.zfar = 1.0;
    [encoder setViewport:viewport];
    [encoder setRenderPipelineState:motionVectorSearchRenderPipelineState];
    vector_float2 positions[6] = {
      {  1,  -1 }, { -1,  -1 }, {  1,   1 },
      {  1,   1 }, { -1,   1 }, { -1,  -1 },
    };
    [encoder setVertexBytes:positions
                     length:sizeof(positions)
                    atIndex:0];
    [encoder setFragmentTexture:frames[frame_index] atIndex:0];
    [encoder setFragmentTexture:frames[previous_frame_index] atIndex:1];
    [encoder drawPrimitives:MTLPrimitiveTypeTriangle
                vertexStart:0
                vertexCount:6];
  }
  [encoder endEncoding];

  [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> cb) {
    CFTimeInterval executionDuration = cb.GPUEndTime - cb.GPUStartTime;
    NSLog(@"Execution time: %f", executionDuration);
  }];
  [commandBuffer commit];
}

void Draw() {
  id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
  id<CAMetalDrawable> drawable = [metalLayer nextDrawable];

  id<MTLRenderCommandEncoder> encoder = nil;
  {
    MTLRenderPassDescriptor* desc = [MTLRenderPassDescriptor renderPassDescriptor];
    desc.colorAttachments[0].texture = drawable.texture;
    desc.colorAttachments[0].loadAction = MTLLoadActionClear;
    desc.colorAttachments[0].storeAction = MTLStoreActionStore;
    desc.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 1);
    encoder = [commandBuffer renderCommandEncoderWithDescriptor:desc];

    MTLViewport viewport;
    viewport.originX = 0;
    viewport.originY = 0;
    viewport.width = width;
    viewport.height = height;
    viewport.znear = -1.0;
    viewport.zfar = 1.0;
    [encoder setViewport:viewport];
  }

  {
    switch (display_mode) {
      case DMCurrentFrame:
        [encoder setRenderPipelineState:renderPipelineState];
        [encoder setFragmentTexture:frames[frame_index] atIndex:0];
        break;
      case DMPreviousFrame:
        [encoder setRenderPipelineState:renderPipelineState];
        [encoder setFragmentTexture:frames[previous_frame_index] atIndex:0];
        break;
      case DMReconstructedFrame:
        [encoder setRenderPipelineState:reconstructRenderPipelineState];
        [encoder setFragmentTexture:frames[previous_frame_index] atIndex:0];
        [encoder setFragmentTexture:motion_vector_texture atIndex:1];
        break;
    }
    vector_float2 positions[6] = {
      {  1,  -1 }, { -1,  -1 }, {  1,   1 },
      {  1,   1 }, { -1,   1 }, { -1,  -1 },
    };
    [encoder setVertexBytes:positions
                     length:sizeof(positions)
                    atIndex:0];
    [encoder drawPrimitives:MTLPrimitiveTypeTriangle
                vertexStart:0
                vertexCount:6];
  }

  if (display_motion_vectors) {
    [encoder setRenderPipelineState:motionVectorDrawRenderPipelineState];
    [encoder setVertexTexture:motion_vector_texture atIndex:0];
    [encoder drawPrimitives:MTLPrimitiveTypeLine
                vertexStart:0
                vertexCount:2 * (width / kBlockWidth) * (height / kBlockHeight)];
  }
  [encoder endEncoding];

  [commandBuffer presentDrawable:drawable];
  [commandBuffer commit];
}


@interface MainWindow : NSWindow
@end

@implementation MainWindow
- (void)keyDown:(NSEvent *)event {
  if ([event isARepeat])
    return;

  NSString *characters = [event charactersIgnoringModifiers];
  if ([characters length] != 1)
    return;

  switch ([characters characterAtIndex:0]) {
    case ' ':
      previous_frame_index = frame_index;
      frame_index = (frame_index + 1) % frames.size();
      printf("Reconstructing frame %lu using frame %lu\n",
          frame_index,
          (frame_index + frames.size() - 1) % frames.size());
      ComputeMotionVectors(kBlockWidth, kBlockHeight);
      // ComputeMotionVectorUsingDraw();
      Draw();
      break;
    case 'p':
      display_mode = DMPreviousFrame;
      Draw();
      break;
    case 'v':
      display_motion_vectors = !display_motion_vectors;
      Draw();
      break;
    case 'c':
      display_mode = DMCurrentFrame;
      Draw();
      break;
    case 'r':
      display_mode = DMReconstructedFrame;
      Draw();
      break;
    case 'q':
      [NSApp terminate:nil];
      break;
  }
}
@end

bool StrEndsWith(const char* str, const char* suffix) {
  size_t lenstr = strlen(str);
  size_t lensuffix = strlen(suffix);
  if (lensuffix >  lenstr)
    return false;
  return strncmp(str + lenstr - lensuffix, suffix, lensuffix) == 0;
}

void LoadYUVTextures(const char* filepath, unsigned int count) {
  const unsigned int chroma_shift = 1;

  unsigned int align = (1 << chroma_shift) - 1;
  unsigned int w = (width + align) & ~align;
  unsigned int h = (height + align) & ~align;

  unsigned int bytes_per_row_align = 32;
  unsigned int bytes_per_row = w;
  bytes_per_row = (bytes_per_row + bytes_per_row_align - 1) & ~(bytes_per_row_align - 1);

  MTLTextureDescriptor* tex_desc = [MTLTextureDescriptor new];
  [tex_desc setTextureType:MTLTextureType2D];
  [tex_desc setUsage:MTLTextureUsageShaderRead];
  [tex_desc setPixelFormat:MTLPixelFormatR8Unorm];
  [tex_desc setWidth:width];
  [tex_desc setHeight:height];
  [tex_desc setDepth:1];
  [tex_desc setMipmapLevelCount:1];
  [tex_desc setArrayLength:1];
  [tex_desc setSampleCount:1];
  [tex_desc setStorageMode:MTLStorageModePrivate];

  FILE* file = fopen(filepath, "rb");

  id <MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
  id <MTLBlitCommandEncoder> blitCommandEncoder = [commandBuffer blitCommandEncoder];

  // Load just the Y data for each image.
  for (unsigned int i = 0; i < count; ++i) {
    id<MTLTexture> frame = [device newTextureWithDescriptor:tex_desc];

    const unsigned int bytespp = 1; // bytes per pixel

    id<MTLBuffer> staging_buffer =
        [device newBufferWithLength:h * bytes_per_row
                            options:MTLResourceStorageModeShared];
    char* ptr = static_cast<char*>([staging_buffer contents]);

    auto LoadPlanes = [&]() {
      for (int plane : {0, 1, 2}) {
        unsigned int plane_width = plane > 0 ? (w + 1) >> chroma_shift : w;
        unsigned int plane_height = plane > 0 ? (h + 1) >> chroma_shift : h;
        unsigned int plane_bytes_per_row = (plane > 0 ? bytes_per_row >> chroma_shift : bytes_per_row);

        for (int r = 0; r < plane_height; ++r) {
          size_t bytes_to_read = plane_width * bytespp;
          if (plane == 0) {
            if (fread(ptr, 1, bytes_to_read, file) < bytes_to_read) {
              return false;
            }
            ptr += plane_bytes_per_row;
          } else {
            if (fseek(file, bytes_to_read, SEEK_CUR) != 0) {
              return false;
            }
          }
        }
      }
      return true;
    };

    if (!LoadPlanes()) {
      break;
    }

    [blitCommandEncoder copyFromBuffer:staging_buffer
                          sourceOffset:0
                      sourceBytesPerRow:bytes_per_row
                    sourceBytesPerImage:bytes_per_row * h
                            sourceSize:MTLSizeMake(width, height, 1)
                              toTexture:frame
                      destinationSlice:0
                      destinationLevel:0
                      destinationOrigin:MTLOriginMake(0, 0, 0)];
    frames.push_back(frame);
  }

  [blitCommandEncoder endEncoding];
  [commandBuffer commit];

  fclose(file);
}

// path/to/app path/to/file.yuv width height
int main(int argc, char* argv[]) {
  [NSApplication sharedApplication];
  [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];

  device = MTLCreateSystemDefaultDevice();
  commandQueue = [device newCommandQueue];

  unsigned int max_frame_count = 16;

  if (argc >= 4 && StrEndsWith(argv[1], ".yuv")) {
    width = strtoul(argv[2], NULL, 0);
    height = strtoul(argv[3], NULL, 0);
    LoadYUVTextures(argv[1], max_frame_count);
  } else {
    for (size_t i = 0; i < max_frame_count; ++i) {
      std::stringstream buffer;
      buffer << "./scroll/" << std::setfill('0') << std::setw(2) << i << ".png";
      IOSurfaceRef io_surface = LoadImageToIOSurface(buffer.str().c_str(), false);
      if (!io_surface) {
        break;
      }
      frames.push_back(IOSurfaceToTexture(io_surface));
    }
  }
  if (frames.size() < 2) {
    CHECK(!"not enough frames :(");
  }

  motion_vector_texture = CreateMotionVectorTexture(kBlockWidth, kBlockHeight);

  NSMenu* menubar = [NSMenu alloc];
  [NSApp setMainMenu:menubar];

  NSWindow* window = [[MainWindow alloc]
    initWithContentRect:NSMakeRect(0, 0, width, height)
    styleMask:NSWindowStyleMaskResizable | NSWindowStyleMaskTitled
    backing:NSBackingStoreBuffered
    defer:NO];
  [window setOpaque:YES];

  metalLayer = [[CAMetalLayer alloc] init];
  metalLayer.device = device;
  metalLayer.pixelFormat = pixelFormat;
  metalLayer.bounds = CGRectMake(0, 0, width, height);
  metalLayer.colorspace = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);

  LoadShaders();

  [[window contentView] setLayer:metalLayer];
  [[window contentView] setWantsLayer:YES];

  [window setTitle:@"Tiny Metal App"];
  [window cascadeTopLeftFromPoint:NSMakePoint(100,100)];
  [window makeKeyAndOrderFront:nil];

  [NSApp activateIgnoringOtherApps:YES];
  [NSApp run];
  return 0;
}

