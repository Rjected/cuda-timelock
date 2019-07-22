ifdef GMP_HOME
  INC := -I$(GMP_HOME)/include
  LIB := -L$(GMP_HOME)/lib
endif
ifndef GMP_HOME
  INC :=
  LIB :=
endif

# TESTS = test/test_powmo.cu

pick:
	@echo
	@echo Please run one of the following:
	@echo "   make kepler"
	@echo "   make maxwell"
	@echo "   make pascal"
	@echo "   make volta"
	@echo "   make check"
	@echo

clean:
	rm -rf bin

check:
	mkdir -p bin
	nvcc $(INC) $(LIB) -Iinclude test/test_powmo.cu -o bin/test_powmo -lgmp
	./bin/test_powmo

kepler:
	mkdir -p bin
	nvcc $(INC) $(LIB) -Iinclude -arch=sm_35 src/cudasquare.cu -o bin/powmo -lgmp

maxwell:
	mkdir -p bin
	nvcc $(INC) $(LIB) -Iinclude -arch=sm_50 src/cudasquare.cu -o bin/powmo -lgmp

pascal:
	mkdir -p bin
	nvcc $(INC) $(LIB) -Iinclude -arch=sm_60 src/cudasquare.cu -o bin/powmo -lgmp

volta:
	mkdir -p bin
	nvcc $(INC) $(LIB) -Iinclude -arch=sm_70 src/cudasquare.cu -o bin/powmo -lgmp
