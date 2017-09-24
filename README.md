# ssd_fft_gpu_combine<br>

Currently Altera OpenCL SDK only support Visual Studio toolkit from VS2010 to VS2014. To correctly set the environment for the OpenCL compiler, run the vcvars64.bat provided by Visual Studio. If you want to use VS2015 or newer version, make sure to include the legacy_stdio_definitions.lib and define __iob_func().
Set board-spec file (_mmd.dll) directory to environment variable path   <br>

Compile host program in Visual Studio 2017, and compile device program (.aocx) using command: <br>
for A10GX FPGA board, use <code>aoc ssd_fft_gpu_kernel.cl -o ssd_fft_gpu.aocx --board a10gx</code><br>
for emulator, use <code>aoc -march=emulator ssd_fft_gpu_kernel.cl -o ssd_fft_gpu.aocx </code>
which generate aocx files<br>

Execute emulator in command prompt: first run the vcvars64.bat, and <code>set CL_CONTEXT_EMULATOR_DEVICE_ALTERA=1</code>, then cd to x64/Debug and execute <code>ssd_fft_GUI </code>

