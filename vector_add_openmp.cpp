#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <omp.h> // For OpenMP multi-threading

#define PRINT 1 // Controls whether to print vectors
int SZ = 100000000; // Default vector size (100 million elements)

// Function declarations
void init(int *&A, int size);
void print(int *A, int size);

// Multi-threaded CPU vector addition using OpenMP
void vector_add_openmp(int *v1, int *v2, int *v_out, int size) {
    #pragma omp parallel for // Parallelize the loop across CPU threads
    for (int i = 0; i < size; i++) {
        v_out[i] = v1[i] + v2[i]; // Compute sum for each element
    }
}

int main(int argc, char **argv) {
    int *v1, *v2, *v_out; // Host arrays for input and output vectors
    
    // Allow vector size to be set via command-line argument
    if (argc > 1) {
        SZ = atoi(argv[1]);
    }
    
    // Display the number of threads being used
    int num_threads = omp_get_max_threads();
    printf("Running OpenMP implementation with %d threads\n", num_threads);
    
    // Allocate and initialize vectors with random integers
    init(v1, SZ);
    init(v2, SZ);
    init(v_out, SZ); // v_out initialized but will be overwritten
    
    // Print input vectors for verification
    printf("Vector v1:\n");
    print(v1, SZ);
    printf("Vector v2:\n");
    print(v2, SZ);
    
    // Measure OpenMP execution time
    auto start_cpu = std::chrono::high_resolution_clock::now();
    vector_add_openmp(v1, v2, v_out, SZ); // Call multi-threaded CPU function
    auto stop_cpu = std::chrono::high_resolution_clock::now();
    
    // Print OpenMP result
    printf("Vector v_out (OpenMP):\n");
    print(v_out, SZ);
    
    // Calculate and display OpenMP execution time
    std::chrono::duration<double, std::milli> elapsed_cpu = stop_cpu - start_cpu;
    printf("CPU (OpenMP) Execution Time: %f ms\n", elapsed_cpu.count());
    
    // Free host memory
    free(v1);
    free(v2);
    free(v_out);
    
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
