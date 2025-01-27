@echo off
setlocal EnableDelayedExpansion

:: Default values
set BUILD_TYPE=Release
set BUILD_TESTS=ON
set BUILD_BENCHMARKS=ON
set BUILD_EXAMPLES=ON
set USE_CUDA=OFF
set CLEAN=false
set JOBS=%NUMBER_OF_PROCESSORS%

:: Parse arguments
:parse_args
if "%~1"=="" goto :done_parsing
if "%~1"=="-h" goto :show_help
if "%~1"=="--help" goto :show_help
if "%~1"=="-t" set BUILD_TYPE=%~2 & shift & goto :next_arg
if "%~1"=="--type" set BUILD_TYPE=%~2 & shift & goto :next_arg
if "%~1"=="--no-tests" set BUILD_TESTS=OFF & goto :next_arg
if "%~1"=="--no-benchmarks" set BUILD_BENCHMARKS=OFF & goto :next_arg
if "%~1"=="--no-examples" set BUILD_EXAMPLES=OFF & goto :next_arg
if "%~1"=="--cuda" set USE_CUDA=ON & goto :next_arg
if "%~1"=="--clean" set CLEAN=true & goto :next_arg
if "%~1"=="-j" set JOBS=%~2 & shift & goto :next_arg
echo Unknown option: %~1
goto :show_help

:next_arg
shift
goto :parse_args

:show_help
echo Usage: %0 [options]
echo Options:
echo   -h, --help              Show this help message
echo   -t, --type TYPE         Build type (Debug^|Release^|RelWithDebInfo) [default: Release]
echo   --no-tests              Disable building tests
echo   --no-benchmarks         Disable building benchmarks
echo   --no-examples           Disable building examples
echo   --cuda                  Enable CUDA support
echo   --clean                 Clean build directory before building
echo   -j N                    Number of parallel jobs [default: number of CPU cores]
exit /b 1

:done_parsing

:: Create build directory
set BUILD_DIR=build\msvc-%BUILD_TYPE%
if not exist %BUILD_DIR% mkdir %BUILD_DIR%

if "%CLEAN%"=="true" (
    echo Cleaning build directory...
    rd /s /q "%BUILD_DIR%"
    mkdir "%BUILD_DIR%"
)

cd %BUILD_DIR%

:: Configure
echo Configuring with CMake...
cmake ..\.. ^
    -DCMAKE_BUILD_TYPE=%BUILD_TYPE% ^
    -DTF_BUILD_TESTS=%BUILD_TESTS% ^
    -DTF_BUILD_BENCHMARKS=%BUILD_BENCHMARKS% ^
    -DTF_BUILD_EXAMPLES=%BUILD_EXAMPLES% ^
    -DTF_USE_CUDA=%USE_CUDA%

:: Build
echo Building...
cmake --build . --config %BUILD_TYPE% -j %JOBS%

cd ..\..

endlocal