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
     else if (mode == 3) {
        long long checksum_atomic = 0;
        double start_atomic = omp_get_wtime();
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
                long long val = (long long)(acc * 1000.0) % 100000;
                #pragma omp atomic
                checksum_atomic += val;
            }
        }
        double end_atomic = omp_get_wtime();

        long long checksum_critical = 0;
        double start_critical = omp_get_wtime();
        #pragma omp parallel for private(j, k)
        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                long long val = (long long)(C[i * N + j] * 1000.0) % 100000;
                #pragma omp critical
                {
                    checksum_critical += val;
                }
            }
        }
        double end_critical = omp_get_wtime();
        printf("%d, Mode 3\nThreads: %d\nAtomic Time: %f s, Checksum: %lld\nCritical Time: %f s, Checksum: %lld\n", 
               N, omp_get_max_threads(), end_atomic - start_atomic, checksum_atomic, end_critical - start_critical, checksum_critical);
        free(A); free(B); free(C);
        return 0;
    }
    else if (mode == 4) {
        start_time = omp_get_wtime();
        #pragma omp parallel
        {
            #pragma omp single
            {
                for (i = 0; i < N; i += 16) {
                    #pragma omp task firstprivate(i) shared(A, B, C, N) private(j, k)
                    {
                        int i_end = (i + 16 > N) ? N : i + 16;
                        for (int bi = i; bi < i_end; bi++) {
                            for (int bj = 0; bj < N; bj++) {
                                double acc = 0.0;
                                for (int bk = 0; bk < N; bk++) {
                                    acc += A[bi * N + bk] * B[bk * N + bj];
                                }
                                C[bi * N + bj] = acc;
                            }
                        }
                    }
                }
            }
     for(int idx=0; idx<N*N; idx++) {
            sumC += C[idx];
            if(C[idx] > maxC) maxC = C[idx];
            checksum += (long long)(C[idx] * 1000.0) % 100000;
        }
        end_time = omp_get_wtime();
        printf("%d, Mode 4\nThreads: %d\n", N, omp_get_max_threads());
    }
    else if (mode == 5 || mode == 6) {
        if (mode == 5) omp_set_num_threads(1);
        start_time = omp_get_wtime();
        #pragma omp parallel for reduction(+:sumC) reduction(max:maxC) private(j, k)
        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                double acc = 0.0;
                #pragma omp simd reduction(+:acc)
                for (k = 0; k < N; k++) {
                    acc += A[i * N + k] * B[k * N + j];
                }
                C[i * N + j] = acc;
                sumC += acc;
                if (acc > maxC) maxC = acc;
                #pragma omp atomic
                checksum += (long long)(acc * 1000.0) % 100000;
            }
        }
        end_time = omp_get_wtime();
        printf("%d, Mode %d\nThreads: %d\n", N, mode, (mode == 5 ? 1 : omp_get_max_threads()));
    }

    printf("Kernel Time: %f s\nSum: %f, Max: %f, Checksum: %lld\n", end_time - start_time, sumC, maxC, checksum);

    free(A); free(B); free(C);
    return 0;
}
