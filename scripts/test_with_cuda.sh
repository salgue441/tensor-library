# Remove old build
rm -rf build

# Configure with CUDA
cmake -B build -DCMAKE_TOOLCHAIN_FILE=~/vcpkg/scripts/buildsystems/vcpkg.cmake -DTENSOR_USE_CUDA=ON

# Build and test
cd build
make
./tests/tensor_tests
