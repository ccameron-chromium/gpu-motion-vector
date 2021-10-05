CXX=clang++
CC=clang
CFLAGS=-I.
CPPFLAGS=-std=c++14
OBJCFLAGS=-framework IOSurface -framework Cocoa -framework Metal -framework MetalKit -framework AppKit -framework QuartzCore -fobjc-arc

all: app shaders.metallib

%.o: %.c
	$(CC) -c $< -o $@ $(CFLAGS)

%.o: %.cpp
	$(CXX) -c $< -o $@ $(CFLAGS) $(CPPFLAGS)

%.o: %.mm %.h
	$(CXX) -c $< -o $@ $(CFLAGS) $(CPPFLAGS) $(OBJCFLAGS)

app: main.mm
	mkdir -p bin && $(CXX) $(OBJCFLAGS) $(CFLAGS) $(CPPFLAGS) $^ -o bin/$@

%.air: %.metal
	xcrun -sdk macosx metal -O2 -std=osx-metal1.2 -c $< -o $@

shaders.metallib: shaders/shaders.air
	mkdir -p bin && xcrun -sdk macosx metallib $< -o bin/$@
