/* Window of a Pillow PERSPECTIVE transform, bit-identical to
 *   src.transform(size, Image.Transform.PERSPECTIVE, coeffs, resample).crop(window)
 * for Pillow 12.3 (libImaging/Geometry.c: perspective_transform with
 * nearest_filter32 / bilinear_filter32RGB / bicubic_filter32RGB, 4 bands),
 * but only the window's output pixels are computed. Each output pixel depends
 * only on its own (x, y) in the full output, so this is exact. Pillow's own box
 * argument offsets x and y to the box origin, which moves the sample points.
 * Build with -ffp-contract=off: an FMA would change the sample positions. */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#define COORD(v) ((v) < 0.0 ? -1 : ((int)(v)))
#define FLOOR(v) ((v) < 0.0 ? ((int)floor(v)) : ((int)(v)))
#define XCLIP(x) (((x) < 0) ? 0 : ((x) < w) ? (x) : w - 1)
#define YCLIP(y) (((y) < 0) ? 0 : ((y) < h) ? (y) : h - 1)

/* Pillow's IMAGING_TRANSFORM_* / Image.Resampling values */
enum { NEAREST = 0, BILINEAR = 2, BICUBIC = 3 };

static int nearest(uint8_t *out, const uint8_t *im, int w, int h, double xin, double yin)
{
    int x = COORD(xin), y = COORD(yin);
    if (x < 0 || x >= w || y < 0 || y >= h)
        return 0;
    memcpy(out, im + ((size_t)y * w + x) * 4, 4);
    return 1;
}

#define BILINEAR_(v, a, b, d) (v = (a) + ((b) - (a)) * (d))

static int bilinear(uint8_t *out, const uint8_t *im, int w, int h, double xin, double yin)
{
    int x, y, x0, x1;
    double v1, v2, dx, dy;
    if (xin < 0.0 || xin >= w || yin < 0.0 || yin >= h)
        return 0;
    xin -= 0.5;
    yin -= 0.5;
    x = FLOOR(xin);
    y = FLOOR(yin);
    dx = xin - x;
    dy = yin - y;
    for (int b = 0; b < 4; b++) {
        const uint8_t *in = im + (size_t)YCLIP(y) * w * 4 + b;
        x0 = XCLIP(x + 0) * 4;
        x1 = XCLIP(x + 1) * 4;
        BILINEAR_(v1, in[x0], in[x1], dx);
        if (y + 1 >= 0 && y + 1 < h) {
            in = im + (size_t)(y + 1) * w * 4 + b;
            BILINEAR_(v2, in[x0], in[x1], dx);
        } else {
            v2 = v1;
        }
        BILINEAR_(v1, v1, v2, dy);
        out[b] = (uint8_t)v1;
    }
    return 1;
}

#define BICUBIC_(v, v1, v2, v3, v4, d)               \
    {                                                \
        double p1 = v2;                              \
        double p2 = -v1 + v3;                        \
        double p3 = 2 * (v1 - v2) + v3 - v4;         \
        double p4 = -v1 + v2 - v3 + v4;              \
        v = p1 + (d) * (p2 + (d) * (p3 + (d) * p4)); \
    }

static int bicubic(uint8_t *out, const uint8_t *im, int w, int h, double xin, double yin)
{
    int x, y, x0, x1, x2, x3;
    double v1, v2, v3, v4, dx, dy;
    if (xin < 0.0 || xin >= w || yin < 0.0 || yin >= h)
        return 0;
    xin -= 0.5;
    yin -= 0.5;
    x = FLOOR(xin);
    y = FLOOR(yin);
    dx = xin - x;
    dy = yin - y;
    x--;
    y--;
    for (int b = 0; b < 4; b++) {
        const uint8_t *in = im + (size_t)YCLIP(y) * w * 4 + b;
        x0 = XCLIP(x + 0) * 4;
        x1 = XCLIP(x + 1) * 4;
        x2 = XCLIP(x + 2) * 4;
        x3 = XCLIP(x + 3) * 4;
        BICUBIC_(v1, in[x0], in[x1], in[x2], in[x3], dx);
        if (y + 1 >= 0 && y + 1 < h) {
            in = im + (size_t)(y + 1) * w * 4 + b;
            BICUBIC_(v2, in[x0], in[x1], in[x2], in[x3], dx);
        } else {
            v2 = v1;
        }
        if (y + 2 >= 0 && y + 2 < h) {
            in = im + (size_t)(y + 2) * w * 4 + b;
            BICUBIC_(v3, in[x0], in[x1], in[x2], in[x3], dx);
        } else {
            v3 = v2;
        }
        if (y + 3 >= 0 && y + 3 < h) {
            in = im + (size_t)(y + 3) * w * 4 + b;
            BICUBIC_(v4, in[x0], in[x1], in[x2], in[x3], dx);
        } else {
            v4 = v3;
        }
        BICUBIC_(v1, v1, v2, v3, v4, dy);
        out[b] = v1 <= 0.0 ? 0 : v1 >= 255.0 ? 255 : (uint8_t)v1;
    }
    return 1;
}

typedef int (*filter_fn)(uint8_t *, const uint8_t *, int, int, double, double);

static PyObject *perspective_window(PyObject *self, PyObject *args)
{
    Py_buffer src;
    int w, h, resample, x0, x1, y0, y1;
    double a[8];

    (void)self;
    if (!PyArg_ParseTuple(args, "y*ii(dddddddd)iiiii", &src, &w, &h, &a[0], &a[1], &a[2], &a[3],
                          &a[4], &a[5], &a[6], &a[7], &resample, &x0, &x1, &y0, &y1))
        return NULL;
    filter_fn filter = resample == NEAREST ? nearest : resample == BILINEAR ? bilinear
                     : resample == BICUBIC ? bicubic : NULL;
    if (!filter || src.len != (Py_ssize_t)w * h * 4 || w <= 0 || h <= 0 || x0 < 0 || y0 < 0 ||
        x0 >= x1 || y0 >= y1) {
        PyBuffer_Release(&src);
        PyErr_SetString(PyExc_ValueError, "bad perspective_window arguments");
        return NULL;
    }
    int ww = x1 - x0, wh = y1 - y0;
    PyObject *res = PyBytes_FromStringAndSize(NULL, (Py_ssize_t)ww * wh * 4);
    if (!res) {
        PyBuffer_Release(&src);
        return NULL;
    }
    uint8_t *out = (uint8_t *)PyBytes_AS_STRING(res);
    const uint8_t *im = src.buf;

    Py_BEGIN_ALLOW_THREADS
    for (int y = y0; y < y1; y++) {
        uint8_t *o = out + (size_t)(y - y0) * ww * 4;
        for (int x = x0; x < x1; x++, o += 4) {
            /* perspective_transform, with the full output's (x, y) */
            double xin = x + 0.5;
            double yin = y + 0.5;
            double xx = (a[0] * xin + a[1] * yin + a[2]) / (a[6] * xin + a[7] * yin + 1);
            double yy = (a[3] * xin + a[4] * yin + a[5]) / (a[6] * xin + a[7] * yin + 1);
            if (!filter(o, im, w, h, xx, yy))
                memset(o, 0, 4); /* fill, as Image.transform with fillcolor=None */
        }
    }
    Py_END_ALLOW_THREADS

    PyBuffer_Release(&src);
    return res;
}

static PyMethodDef methods[] = {
    {"perspective_window", perspective_window, METH_VARARGS,
     "perspective_window(rgba, w, h, coeffs8, resample, x0, x1, y0, y1) -> bytes"},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "_transform", NULL, -1, methods, NULL, NULL, NULL, NULL};

PyMODINIT_FUNC PyInit__transform(void)
{
    return PyModule_Create(&module);
}
