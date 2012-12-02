CC=/usr/local/apps/cuda/3.2/cuda/bin/nvcc
INCLUDE=-I/usr/local/apps/cuda/3.2/cuda/include \
        -I/usr/local/apps/cuda/SDK2/C/common/inc \

LIBDIR=-L/usr/local/apps/cuda/SDK2/C/lib
LIBS=-lcutil

SOURCE=sparse_matvec.cu
EXECUTABLE=sparse

$(EXECUTABLE): $(SOURCE)
	$(CC) $(INCLUDE) $(LIBDIR) $< -o $@ $(LIBS)

clean:
