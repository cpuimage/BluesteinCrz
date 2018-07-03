#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdbool.h>

#ifndef nullptr
#define nullptr 0
#endif

bool is_pow_2(size_t n) {
    return (n & (n - 1)) == 0;
}

size_t next_pow_2(size_t x) {
    size_t h = 1;
    while (h < x)
        h *= 2;
    return h;
}

void swap_complex(float *a_re, float *a_im, float *b_re, float *b_im) {
    const float p_re = *a_re;
    const float p_im = *a_im;
    *a_re = *b_re;
    *a_im = *b_im;
    *b_re = p_re;
    *b_im = p_im;
}

static void bitrev(float *in_re, float *in_im, const size_t n) {
    for (size_t i = 0, j = 0; i < n - 1; i++) {
        if (i < j)
            swap_complex(&in_re[i], &in_im[i], &in_re[j], &in_im[j]);
        size_t m = n;
        do {
            m >>= 1;
            j ^= m;
        } while (!(j & m));
    }
}

static void fft_pow_2(float *in_re, float *in_im, const size_t n, int sign) {
    bitrev(in_re, in_im, n);
    float PI = (float) (sign * M_PI);
    for (size_t i = 1; i < n; i = 2 * i) {
        const float delta = PI / i;
        for (int j = 0; j < i; j++) {
            const float m = delta * j;
            const float w_re = cosf(m);
            const float w_im = sinf(m);
            for (int k = j; k < n; k += 2 * i) {
                const float v_re = in_re[k];
                const float v_im = in_im[k];
                const float x_re = in_re[k + i] * w_re - in_im[k + i] * w_im;
                const float x_im = in_re[k + i] * w_im + in_im[k + i] * w_re;
                in_re[k] = v_re + x_re;
                in_im[k] = v_im + x_im;
                in_re[k + i] = v_re - x_re;
                in_im[k + i] = v_im - x_im;
            }
        }
    }
}

void bluestein(float *in_re, float *in_im, const size_t n) {
    size_t n2 = 2 * n;
    const size_t nb = next_pow_2(n2);
    float *w_re = (float *) malloc(sizeof(float) * n);
    float *w_im = (float *) malloc(sizeof(float) * n);
    float *y_re = (float *) calloc(sizeof(float), nb);
    float *y_im = (float *) calloc(sizeof(float), nb);
    float *b_re = (float *) calloc(sizeof(float), nb);
    float *b_im = (float *) calloc(sizeof(float), nb);
    if (w_re == nullptr || w_im == nullptr ||
        y_re == nullptr || y_im == nullptr ||
        b_re == nullptr || b_im == nullptr) {
        if (w_re)free(w_re);
        if (w_im)free(w_im);
        if (y_re)free(y_re);
        if (y_im)free(y_im);
        if (b_re)free(b_re);
        if (b_im)free(b_im);
        return;
    }
    const float delta = (float) M_PI / n;
    w_re[0] = 1;
    w_im[0] = 0;
    y_re[0] = 1;
    y_im[0] = 0;
    for (int k = 1; k < n; k++) {
        const float m = delta * (int) ((k * k) % n2);
        w_re[k] = cosf(m);
        w_im[k] = sinf(m);
        y_re[k] = w_re[k];
        y_im[k] = w_im[k];
        y_re[nb - k] = w_re[k];
        y_im[nb - k] = w_im[k];
    }
    fft_pow_2(y_re, y_im, nb, -1);
    for (int i = 0; i < n; i++) {
        b_re[i] = w_re[i] * in_re[i] + w_im[i] * in_im[i];
        b_im[i] = w_re[i] * in_im[i] - w_im[i] * in_re[i];
    }
    fft_pow_2(b_re, b_im, nb, -1);
    for (int i = 0; i < nb; i++) {
        const float t = b_re[i];
        b_re[i] = t * y_re[i] - b_im[i] * y_im[i];
        b_im[i] = t * y_im[i] + b_im[i] * y_re[i];
    }
    fft_pow_2(b_re, b_im, nb, 1);
    for (int i = 0; i < n; i++) {
        in_re[i] = (w_re[i] * b_re[i] + w_im[i] * b_im[i]) / nb;
        in_im[i] = (w_re[i] * b_im[i] - w_im[i] * b_re[i]) / nb;
    }
    free(w_re);
    free(w_im);
    free(y_re);
    free(y_im);
    free(b_re);
    free(b_im);
}


static void fft(float *in_re, float *in_im, size_t n, int sign) {
    if (is_pow_2(n)) {
        fft_pow_2(in_re, in_im, n, sign);
    } else {
        if (sign == 1) {
            bluestein(in_im, in_re, n);
        } else {
            bluestein(in_re, in_im, n);
        }
    }
}

int main(int argc, char **argv) {
    unsigned int size = 1235;
    float *test_re = (float *) malloc(sizeof(float) * size);
    float *test_im = (float *) calloc(sizeof(float), size);
    if (test_re == nullptr || test_im == nullptr)
        return -1;
    for (int i = 0; i < size; ++i) {
        test_re[i] = (float) i;
    }
    printf("\n\n");

    fft(test_re, test_im, size, 1);

    for (int i = 0; i < size; ++i) {
        printf("%f , %f \t", test_re[i], test_im[i]);
    }
    printf("\n\n");

    fft(test_re, test_im, size, -1);
    for (int i = 0; i < size; ++i) {
        printf("%f \t", test_re[i] / size);
    }
    printf("\n\n");
    free(test_re);
    free(test_im);

    getchar();
    return 0;
}


