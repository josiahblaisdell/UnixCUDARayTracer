CUDA_INSTALL_PATH :=  /usr/local/apps/cuda/cuda-9.2

CXX := g++
CC := gcc
LINK := g++ -fPIC
NVCC  := nvcc -ccbin g++
GENCODE_FLAGS  = -gencode arch=compute_35,code=sm_35

# Includes
INCLUDES = -I. -I$(CUDA_INSTALL_PATH)/include

# Common flags
COMMONFLAGS += $(INCLUDES)
NVCCFLAGS += $(COMMONFLAGS)
NVCCFLAGS += $(GENCODE_FLAGS)
CXXFLAGS += $(COMMONFLAGS)
CFLAGS += $(COMMONFLAGS)


LIB_CUDA := -L$(CUDA_INSTALL_PATH)/lib64 -lcudart
OBJS = vec3.cu.o main.cu.o RayTracer.cu.o
TARGET = raytracer
LINKLINE = $(LINK) -o $(TARGET) $(OBJS) $(LIB_CUDA)

.SUFFIXES: .c .cpp .cu .o

%.c.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.cu.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) --ptxas-options="-v" -c $< -o $@ --verbose 

%.cpp.o: %.cpp$(CXX) $(CXXFLAGS) -c $< -o $@
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(TARGET): $(OBJS) Makefile
	$(LINKLINE)


clean:
	rm -f raytracer $(OBJS) out.ppm out.jpg image.ppm
