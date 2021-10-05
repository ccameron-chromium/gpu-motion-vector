// clang++ main.mm -framework Metal -framework MetalKit -framework Cocoa -framework QuartzCore -framework IOSurface
#include <Metal/Metal.h>
#include <MetalKit/MetalKit.h>
#include <IOSurface/IOSurface.h>

#include <sstream>
#include <fstream>
#include <string>
#include <vector>

const MTLPixelFormat pixelFormat = MTLPixelFormatBGRA8Unorm;
const int width = 1280;
const int height = 720;
const int kBlockWidth = 16;
const int kBlockHeight = 16;

id<MTLDevice> device = nil;
id<MTLLibrary> library = nil;
id<MTLCommandQueue> commandQueue;
CAMetalLayer* metalLayer = nil;

id<MTLRenderPipelineState> renderPipelineState = nil;
id<MTLRenderPipelineState> reconstructRenderPipelineState = nil;
id<MTLRenderPipelineState> motionVectorSearchRenderPipelineState = nil;
id<MTLRenderPipelineState> motionVectorDrawRenderPipelineState = nil;

size_t texture_index = 0;
size_t previous_texture_index = 0;

enum DisplayMode {
  DMPreviousFrame,
  DMCurrentFrame,
  DMReconstructedFrame,
};
DisplayMode display_mode = DMCurrentFrame;
bool display_motion_vectors = false;

size_t num_frames = 16;
IOSurfaceRef io_surfaces[16];
id<MTLTexture> textures[16];

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
  [tex_desc setStorageMode:MTLStorageModePrivate];
  id<MTLTexture> texture = [device newTextureWithDescriptor:tex_desc
                                                  iosurface:io_surface
                                                      plane:0];
  CHECK(texture);
  return texture;
}

id<MTLTexture> CreateMotionVectorTexture(size_t block_width, size_t block_height) {
  MTLTextureDescriptor* tex_desc = [MTLTextureDescriptor new];
  [tex_desc setTextureType:MTLTextureType2D];
  [tex_desc setUsage:MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget];
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
    [encoder setFragmentTexture:textures[texture_index] atIndex:0];
    [encoder setFragmentTexture:textures[previous_texture_index] atIndex:1];
    [encoder drawPrimitives:MTLPrimitiveTypeTriangle
                vertexStart:0
                vertexCount:6];
  }
  [encoder endEncoding];

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
        [encoder setFragmentTexture:textures[texture_index] atIndex:0];
        break;
      case DMPreviousFrame:
        [encoder setRenderPipelineState:renderPipelineState];
        [encoder setFragmentTexture:textures[previous_texture_index] atIndex:0];
        break;
      case DMReconstructedFrame:
        [encoder setRenderPipelineState:reconstructRenderPipelineState];
        [encoder setFragmentTexture:textures[previous_texture_index] atIndex:0];
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
      previous_texture_index = texture_index;
      texture_index = (texture_index + 1) % num_frames;
      printf("Reconstructing frame %lu using frame %lu\n",
          texture_index,
          (texture_index + num_frames - 1) % num_frames);
      ComputeMotionVectorUsingDraw();
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

int main(int argc, char* argv[]) {
  [NSApplication sharedApplication];
  [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];

  device = MTLCreateSystemDefaultDevice();
  commandQueue = [device newCommandQueue];
  for (size_t i = 0; i < 16; ++i) {
    std::stringstream buffer;
    buffer << "./scroll/" << std::setfill('0') << std::setw(2) << i << ".png";
    io_surfaces[i] = LoadImageToIOSurface(buffer.str().c_str(), false);
    if (!io_surfaces[i]) {
      if (i > 2) {
        num_frames = i;
        break;
      } else {
        CHECK(!"not enough frames :(");
      }
    }
    textures[i] = IOSurfaceToTexture(io_surfaces[i]);
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
  [window makeKeyAndOrderFront:nil];

  [NSApp activateIgnoringOtherApps:YES];
  [NSApp run];
  return 0;
}

