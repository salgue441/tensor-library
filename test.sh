rm -rf build/
mkdir build && cd build

cmake .. -DCMAKE_TOOLCHAIN_FILE=~/vcpkg/scripts/buildsystems/vcpkg.cmake
make

./tests/tensor_tests
