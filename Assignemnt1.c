#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

void initialize_matrices(int N, double *A, double *B) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = sin(i) * cos(j) + sqrt(i + j + 1);
            B[i * N + j] = cos(i) * sin(j) + sqrt(i + j + 2);
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s <N> <mode>\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    int mode = atoi(argv[2]);

    double *A = (double *)malloc(N * N * sizeof(double));
    double *B = (double *)malloc(N * N * sizeof(double));
    double *C = (double *)malloc(N * N * sizeof(double));

    initialize_matrices(N, A, B);

    double sumC = 0.0, maxC = -1.0, start_time, end_time;
    long long checksum = 0;
    int i, j, k;
      if (mode == 0) {
        start_time = omp_get_wtime();
        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                double acc = 0.0;
                for (k = 0; k < N; k++) {
                    acc += A[i * N + k] * B[k * N + j];
                }
                C[i * N + j] = acc;
                sumC += acc;
                if (acc > maxC) maxC = acc;
                checksum += (long long)(acc * 1000.0) % 100000;
            }
        }
        end_time = omp_get_wtime();
        printf("%d, Mode 0 (Serial)\nThreads: 1\n", N);
    } 
    else if (mode == 1 || mode == 2) {
        start_time = omp_get_wtime();
        if (mode == 1) {
            #pragma omp parallel for reduction(+:sumC) reduction(max:maxC) private(j, k)
            for (i = 0; i < N; i++) {
                for (j = 0; j < N; j++) {
                    double acc = 0.0;
                    for (k = 0; k < N; k++) {
                        acc += A[i * N + k] * B[k * N + j];
                    }
                    C[i * N + j] = acc;
                    sumC += acc;
                    if (acc > maxC) maxC = acc;
                }
            }
        } else {
            #pragma omp parallel for collapse(2) reduction(+:sumC) reduction(max:maxC) private(k)
            for (i = 0; i < N; i++) {
                for (j = 0; j < N; j++) {
                    double acc = 0.0;
                    for (k = 0; k < N; k++) {
                        acc += A[i * N + k] * B[k * N + j];
                    }
                    C[i * N + j] = acc;
                    sumC += acc;
                    if (acc > maxC) maxC = acc;
                }
            }
        }
        end_time = omp_get_wtime();
        printf("%d, Mode %d\nThreads: %d\n", N, mode, omp_get_max_threads());
    }