NVCC=nvcc
OPTIONS=-gencode arch=compute_61,code=sm_61 \
		-gencode arch=compute_62,code=sm_62 \
		-gencode arch=compute_70,code=sm_70 \
		-gencode arch=compute_72,code=sm_72 \
		-gencode arch=compute_75,code=sm_75 \
		-gencode arch=compute_80,code=sm_80 \
		-gencode arch=compute_86,code=sm_86

TARGETS=$(patsubst %.cu, %.fatbin, $(wildcard *.cu))

all: $(TARGETS)

%.fatbin: %.cu
	$(NVCC) -fatbin $^ $(OPTIONS) -o $@

.PHONY : clean, copy
clean:
	rm $(TARGETS)

copy:
	cp $(TARGETS) ../kernels/
