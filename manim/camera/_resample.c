/* Window of a Pillow BICUBIC resize, bit-identical to
 *   Image.frombytes("RGBa", src).resize((out_w, out_h), BICUBIC).crop((x0, y0, x1, y1))
 * for Pillow 12.3 (libImaging/Resample.c, 8 bpc, 4 bands), but only the
 * window is computed. Coefficients use the full-image formulas (in0 = 0,
 * in1 = in size), so each output pixel sees the same integer arithmetic.
 * Pillow's own box= argument shifts the float sample centres and is not exact.
 * Build with -ffp-contract=off: an FMA would change the coefficients. */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define PRECISION_BITS (32 - 8 - 2)

static inline double bicubic_filter(double x)
{
#define a -0.5
    if (x < 0.0)
        x = -x;
    if (x < 1.0)
        return ((a + 2.0) * x - (a + 3.0)) * x * x + 1;
    if (x < 2.0)
        return (((x - 5) * x + 8) * x - 4) * a;
    return 0.0;
#undef a
}

static inline uint8_t clip8(int in)
{
    int v = in >> PRECISION_BITS;
    return v < 0 ? 0 : v > 255 ? 255 : (uint8_t)v;
}

/* precompute_coeffs + normalize_coeffs_8bpc for output index xx only */
static int coeffs(int in_size, int out_size, int xx, int ksize, int32_t *k, int *xmin_out, double *tmp)
{
    double scale, filterscale, support, ww = 0.0;
    int x, xmin, xmax;

    filterscale = scale = (double)((float)in_size - 0.0f) / out_size;
    if (filterscale < 1.0)
        filterscale = 1.0;
    support = 2.0 * filterscale;
    double inv_filterscale = 1.0 / filterscale;
    double center = 0.0f + (xx + 0.5) * scale;
    xmin = (int)(center - support + 0.5);
    if (xmin < 0)
        xmin = 0;
    xmax = (int)(center + support + 0.5);
    if (xmax > in_size)
        xmax = in_size;
    xmax -= xmin;
    for (x = 0; x < xmax; x++) {
        double w = bicubic_filter((x + xmin - center + 0.5) * inv_filterscale);
        tmp[x] = w;
        ww += w;
    }
    if (ww != 0.0)
        for (x = 0; x < xmax; x++)
            tmp[x] /= ww;
    for (x = 0; x < ksize; x++) {
        double v = x < xmax ? tmp[x] : 0.0;
        k[x] = v < 0 ? (int)(-0.5 + v * (1 << PRECISION_BITS)) : (int)(0.5 + v * (1 << PRECISION_BITS));
    }
    *xmin_out = xmin;
    return xmax;
}

static int ksize_for(int in_size, int out_size)
{
    double filterscale = (double)((float)in_size - 0.0f) / out_size;
    if (filterscale < 1.0)
        filterscale = 1.0;
    return (int)ceil(2.0 * filterscale) * 2 + 1;
}

static PyObject *bicubic_window(PyObject *self, PyObject *args)
{
    Py_buffer src;
    int in_w, in_h, out_w, out_h, x0, x1, y0, y1;

    (void)self;
    if (!PyArg_ParseTuple(args, "y*iiiiiiii", &src, &in_w, &in_h, &out_w, &out_h, &x0, &x1, &y0, &y1))
        return NULL;
    int ksh = ksize_for(in_w, out_w), ksv = ksize_for(in_h, out_h);
    if (src.len != (Py_ssize_t)in_w * in_h * 4 || in_w <= 0 || in_h <= 0 || out_w <= 0 || out_h <= 0 ||
        x0 < 0 || y0 < 0 || x1 > out_w || y1 > out_h || x0 >= x1 || y0 >= y1) {
        PyBuffer_Release(&src);
        PyErr_SetString(PyExc_ValueError, "bad bicubic_window arguments");
        return NULL;
    }
    int need_h = out_w != in_w, need_v = out_h != in_h;
    int ww = x1 - x0, wh = y1 - y0;
    const uint8_t *in = src.buf;

    PyObject *res = PyBytes_FromStringAndSize(NULL, (Py_ssize_t)ww * wh * 4);
    if (!res) {
        PyBuffer_Release(&src);
        return NULL;
    }
    uint8_t *out = (uint8_t *)PyBytes_AS_STRING(res);

    int32_t *kv = malloc(sizeof(int32_t) * ksv * wh), *kh = malloc(sizeof(int32_t) * ksh * ww);
    int *bv = malloc(sizeof(int) * 2 * wh), *bh = malloc(sizeof(int) * 2 * ww);
    double *dk = malloc(sizeof(double) * (ksh > ksv ? ksh : ksv));
    uint8_t *tmp = NULL;
    int ok = kv && kh && bv && bh && dk;

    Py_BEGIN_ALLOW_THREADS
    if (ok) {
        /* source rows the window needs: the vertical taps, or the rows themselves */
        int r0 = y0, r1 = y1;
        if (need_v) {
            r0 = in_h, r1 = 0;
            for (int y = 0; y < wh; y++) {
                bv[2 * y + 1] = coeffs(in_h, out_h, y0 + y, ksv, kv + y * ksv, &bv[2 * y], dk);
                if (bv[2 * y] < r0)
                    r0 = bv[2 * y];
                if (bv[2 * y] + bv[2 * y + 1] > r1)
                    r1 = bv[2 * y] + bv[2 * y + 1];
            }
        }
        /* horizontal pass on rows [r0, r1), window columns only */
        const uint8_t *hsrc = in + (size_t)r0 * in_w * 4;   /* row r of the pass input */
        int hstride = in_w * 4, hx0 = x0;
        if (need_h) {
            tmp = malloc((size_t)(r1 - r0) * ww * 4);
            if (!tmp) {
                ok = 0;
            } else {
                for (int x = 0; x < ww; x++)
                    bh[2 * x + 1] = coeffs(in_w, out_w, x0 + x, ksh, kh + x * ksh, &bh[2 * x], dk);
                for (int r = r0; r < r1; r++) {
                    const uint8_t *line = in + (size_t)r * in_w * 4;
                    uint8_t *o = tmp + (size_t)(r - r0) * ww * 4;
                    for (int x = 0; x < ww; x++) {
                        const int32_t *k = kh + x * ksh;
                        int xmin = bh[2 * x], xmax = bh[2 * x + 1];
                        int ss0, ss1, ss2, ss3;
                        ss0 = ss1 = ss2 = ss3 = 1 << (PRECISION_BITS - 1);
                        for (int i = 0; i < xmax; i++) {
                            ss0 += line[(i + xmin) * 4 + 0] * k[i];
                            ss1 += line[(i + xmin) * 4 + 1] * k[i];
                            ss2 += line[(i + xmin) * 4 + 2] * k[i];
                            ss3 += line[(i + xmin) * 4 + 3] * k[i];
                        }
                        o[x * 4 + 0] = clip8(ss0);
                        o[x * 4 + 1] = clip8(ss1);
                        o[x * 4 + 2] = clip8(ss2);
                        o[x * 4 + 3] = clip8(ss3);
                    }
                }
                hsrc = tmp, hstride = ww * 4, hx0 = 0;
            }
        }
        /* vertical pass, or copy of the window rows */
        if (ok) {
            for (int y = 0; y < wh; y++) {
                uint8_t *o = out + (size_t)y * ww * 4;
                if (!need_v) {
                    memcpy(o, hsrc + (size_t)(y0 + y - r0) * hstride + hx0 * 4, (size_t)ww * 4);
                    continue;
                }
                const int32_t *k = kv + y * ksv;
                int ymin = bv[2 * y] - r0, ymax = bv[2 * y + 1];
                for (int x = 0; x < ww; x++) {
                    int ss0, ss1, ss2, ss3;
                    ss0 = ss1 = ss2 = ss3 = 1 << (PRECISION_BITS - 1);
                    for (int i = 0; i < ymax; i++) {
                        const uint8_t *p = hsrc + (size_t)(i + ymin) * hstride + (hx0 + x) * 4;
                        ss0 += p[0] * k[i];
                        ss1 += p[1] * k[i];
                        ss2 += p[2] * k[i];
                        ss3 += p[3] * k[i];
                    }
                    o[x * 4 + 0] = clip8(ss0);
                    o[x * 4 + 1] = clip8(ss1);
                    o[x * 4 + 2] = clip8(ss2);
                    o[x * 4 + 3] = clip8(ss3);
                }
            }
        }
    }
    Py_END_ALLOW_THREADS

    free(kv), free(kh), free(bv), free(bh), free(dk), free(tmp);
    PyBuffer_Release(&src);
    if (!ok) {
        Py_DECREF(res);
        return PyErr_NoMemory();
    }
    return res;
}

static PyMethodDef methods[] = {
    {"bicubic_window", bicubic_window, METH_VARARGS,
     "bicubic_window(rgba_premultiplied, in_w, in_h, out_w, out_h, x0, x1, y0, y1) -> bytes"},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "_resample", NULL, -1, methods, NULL, NULL, NULL, NULL};

PyMODINIT_FUNC PyInit__resample(void)
{
    return PyModule_Create(&module);
}
