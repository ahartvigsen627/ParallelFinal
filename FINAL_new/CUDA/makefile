CC=/usr/local/apps/cuda/3.2/cuda/bin/nvcc
INCLUDE=-I/usr/local/apps/cuda/3.2/cuda/include \
        -I/usr/local/apps/cuda/SDK2/C/common/inc \

LIBDIR=-L/usr/local/apps/cuda/SDK2/C/lib
LIBS=-lcutil

DOUBLE = --gpu-architecture sm_13

SOURCE=cuda_SVD.cu
EXECUTABLE=cuda_SVD

$(EXECUTABLE): $(SOURCE)
	$(CC) $(INCLUDE) $(LIBDIR) $(DOUBLE) $< -o $@ $(LIBS)

clean:
