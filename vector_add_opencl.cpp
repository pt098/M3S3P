#define CL_TARGET_OPENCL_VERSION 200 // Define OpenCL version 2.0
#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <chrono>

#define PRINT 1 // Controls whether to print vectors
int SZ = 100000000; // Default vector size (100 million elements)

int *v1, *v2, *v_out; // Host arrays for input vectors (v1, v2) and output (v_out)

// OpenCL objects
cl_mem bufV1, bufV2, bufV_out; // Device memory buffers
cl_device_id device_id;        // Device identifier (GPU or CPU)
cl_context context;            // OpenCL context
cl_program program;            // OpenCL program
cl_kernel kernel;              // OpenCL kernel
cl_command_queue queue;        // Command queue for device operations
cl_event event = NULL;         // Event for timing kernel execution
int err;                       // Error code for OpenCL calls

// Function declarations
cl_device_id create_device();
void setup_openCL_device_context_queue_kernel(char *filename, char *kernelname);
cl_program build_program(cl_context ctx, cl_device_id dev, const char *filename);
void setup_kernel_memory();
void copy_kernel_args();
void free_memory();
void init(int *&A, int size);
void print(int *A, int size);

int main(int argc, char **argv) {
    // Allow vector size to be set via command-line argument
    if (argc > 1) {
        SZ = atoi(argv[1]);
    }
    
    // Allocate and initialize vectors with random integers
    init(v1, SZ);
    init(v2, SZ);
    init(v_out, SZ); // v_out initialized but will be overwritten
    
    // Define global work size for OpenCL kernel (one work item per element)
    size_t global[1] = {(size_t)SZ};
    
    // Print input vectors for verification
    printf("Vector v1:\n");
    print(v1, SZ);
    printf("Vector v2:\n");
    print(v2, SZ);
    
    // Set up OpenCL environment and kernel
    setup_openCL_device_context_queue_kernel((char *)"./vector_ops_ocl.cl", (char *)"vector_add_ocl");
    
    // Allocate device memory and copy input data to device
    setup_kernel_memory();
    
    // Set kernel arguments
    copy_kernel_args();
    
    // Measure OpenCL kernel execution time
    auto start_ocl = std::chrono::high_resolution_clock::now();
    // Launch kernel with global work size (one thread per vector element)
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global, NULL, 0, NULL, &event);
    
    clWaitForEvents(1, &event); // Wait for kernel to finish
    auto stop_ocl = std::chrono::high_resolution_clock::now();
    
    // Copy result back from device to host
    clEnqueueReadBuffer(queue, bufV_out, CL_TRUE, 0, SZ * sizeof(int), &v_out[0], 0, NULL, NULL);
    
    // Print OpenCL result
    printf("Vector v_out (OpenCL):\n");
    print(v_out, SZ);
    
    // Calculate and display OpenCL execution time
    std::chrono::duration<double, std::milli> elapsed_ocl = stop_ocl - start_ocl;
    printf("OpenCL Kernel Execution Time: %f ms\n", elapsed_ocl.count());
    
    // Clean up resources
    free_memory();
    
    return 0;
}

// Initialize an array with random integers between 0 and 99
void init(int *&A, int size) {
    A = (int *)malloc(sizeof(int) * size);
    for (long i = 0; i < size; i++) {
        A[i] = rand() % 100;
    }
}

// Print array elements (all if small, first 5 and last 5 if large)
void print(int *A, int size) {
    if (PRINT == 0) {
        return;
    }
    
    if (PRINT == 1 && size > 15) {
        for (long i = 0; i < 5; i++) {
            printf("%d ", A[i]);
        }
        printf(" ..... ");
        for (long i = size - 5; i < size; i++) {
            printf("%d ", A[i]);
        }
    } else {
        for (long i = 0; i < size; i++) {
            printf("%d ", A[i]);
        }
    }
    printf("\n----------------------------\n");
}

// Free OpenCL resources and host memory
void free_memory() {
    // Release OpenCL objects in reverse order of creation
    clReleaseMemObject(bufV1);
    clReleaseMemObject(bufV2);
    clReleaseMemObject(bufV_out);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseProgram(program);
    clReleaseContext(context);
    
    // Free host memory
    free(v1);
    free(v2);
    free(v_out);
}

// Set kernel arguments (size and memory buffers)
void copy_kernel_args() {
    clSetKernelArg(kernel, 0, sizeof(int), (void *)&SZ);       // Argument 0: vector size
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&bufV1); // Argument 1: input vector 1
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&bufV2); // Argument 2: input vector 2
    clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&bufV_out); // Argument 3: output vector
    
    if (err < 0) {
        perror("Couldn't set kernel arguments");
        exit(1);
    }
}

// Allocate device memory buffers and copy input data to device
void setup_kernel_memory() {
    // Create OpenCL buffer objects for device memory allocation
    bufV1 = clCreateBuffer(context, CL_MEM_READ_WRITE, SZ * sizeof(int), NULL, NULL);
    bufV2 = clCreateBuffer(context, CL_MEM_READ_WRITE, SZ * sizeof(int), NULL, NULL);
    bufV_out = clCreateBuffer(context, CL_MEM_READ_WRITE, SZ * sizeof(int), NULL, NULL);
    
    // Transfer input data from host to device
    clEnqueueWriteBuffer(queue, bufV1, CL_TRUE, 0, SZ * sizeof(int), &v1[0], 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, bufV2, CL_TRUE, 0, SZ * sizeof(int), &v2[0], 0, NULL, NULL);
}

// Set up OpenCL device, context, command queue, and kernel
void setup_openCL_device_context_queue_kernel(char *filename, char *kernelname) {
    device_id = create_device(); // Select GPU or CPU
    
    // Create OpenCL context for the selected device
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
    if (err < 0) {
        perror("Couldn't create a context");
        exit(1);
    }
    
    // Build program from source file
    program = build_program(context, device_id, filename);
    
    // Create command queue for the device
    queue = clCreateCommandQueueWithProperties(context, device_id, 0, &err);
    if (err < 0) {
        perror("Couldn't create a command queue");
        exit(1);
    }
    
    // Create kernel from the compiled program
    kernel = clCreateKernel(program, kernelname, &err);
    if (err < 0) {
        perror("Couldn't create a kernel");
        exit(1);
    }
}

// Build OpenCL program from source file
cl_program build_program(cl_context ctx, cl_device_id dev, const char *filename) {
    // Open program file
    FILE *program_handle = fopen(filename, "r");
    if (program_handle == NULL) {
        perror("Couldn't find the program file");
        exit(1);
    }
    
    // Get file size
    fseek(program_handle, 0, SEEK_END);
    size_t program_size = ftell(program_handle);
    rewind(program_handle);
    
    // Read the program source into a buffer
    char *program_buffer = (char *)malloc(program_size + 1);
    program_buffer[program_size] = '\0';
    fread(program_buffer, sizeof(char), program_size, program_handle);
    fclose(program_handle);
    
    // Create program from source
    cl_program program = clCreateProgramWithSource(ctx, 1, (const char **)&program_buffer, &program_size, &err);
    if (err < 0) {
        perror("Couldn't create the program");
        exit(1);
    }
    free(program_buffer);
    
    // Build program (compile and link)
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err < 0) {
        // If build fails, get and print build log
        size_t log_size;
        clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *program_log = (char *)malloc(log_size + 1);
        program_log[log_size] = '\0';
        clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, log_size + 1, program_log, NULL);
        printf("%s\n", program_log);
        free(program_log);
        exit(1);
    }
    
    return program;
}

// Select a device (prefer GPU, fall back to CPU)
cl_device_id create_device() {
    cl_platform_id platform;
    cl_device_id dev;
    
    // Get the first available platform
    err = clGetPlatformIDs(1, &platform, NULL);
    if (err < 0) {
        perror("Couldn't identify a platform");
        exit(1);
    }
    
    // Try to get a GPU device first
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
    if (err == CL_DEVICE_NOT_FOUND) {
        // If no GPU is available, fall back to CPU
        printf("GPU not found, falling back to CPU\n");
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &dev, NULL);
    }
    
    if (err < 0) {
        perror("Couldn't access any devices");
        exit(1);
    }
    
    return dev;
}
