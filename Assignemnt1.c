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