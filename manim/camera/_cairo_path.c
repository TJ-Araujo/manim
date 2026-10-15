/* C port of Camera.set_cairo_context_path for nppcc == 4.
 * Issues the same cairo calls with the same doubles as the Python loop
 * (subpath split by VMobject.consider_points_equals_2d, one curve_to per
 * cubic), so frames are bit-identical. Build with -ffp-contract=off: an FMA
 * would change the rounding of the point-equality test. */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include <string.h>
#include <py3cairo.h> /* defines Pycairo_CAPI, filled by import_cairo() */

/* VMobject.consider_points_equals_2d, rtol = 1e-5 as in np.isclose */
static int eq2d(const double *p0, const double *p1, double atol)
{
    if (fabs(p0[0] - p1[0]) > atol + 1.0e-5 * fabs(p1[0]))
        return 0;
    return fabs(p0[1] - p1[1]) <= atol + 1.0e-5 * fabs(p1[1]);
}

static void emit_subpath(cairo_t *cr, const double *s, Py_ssize_t len, double atol)
{
    Py_ssize_t full = len - len % 4;
    cairo_new_sub_path(cr);
    cairo_move_to(cr, s[0], s[1]);
    for (Py_ssize_t i = 0; i < full; i += 4) {
        const double *q = s + 3 * i;
        cairo_curve_to(cr, q[3], q[4], q[6], q[7], q[9], q[10]);
    }
    if (eq2d(s, s + 3 * (len - 1), atol))
        cairo_close_path(cr);
}

static PyObject *set_path(PyObject *self, PyObject *args)
{
    PyObject *ctx_obj, *arr;
    double atol;
    Py_buffer b;

    (void)self;
    if (!PyArg_ParseTuple(args, "O!Od", &PycairoContext_Type, &ctx_obj, &arr, &atol))
        return NULL;
    if (PyObject_GetBuffer(arr, &b, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) < 0)
        return NULL;
    if (b.ndim != 2 || b.shape[1] != 3 || b.itemsize != 8 || strcmp(b.format, "d") != 0) {
        PyBuffer_Release(&b);
        PyErr_SetString(PyExc_ValueError, "points must be a C-contiguous float64 (N, 3) array");
        return NULL;
    }

    cairo_t *cr = ((PycairoContext *)ctx_obj)->ctx;
    const double *p = b.buf;
    Py_ssize_t n = b.shape[0];

    if (n > 0) {
        Py_BEGIN_ALLOW_THREADS
        cairo_new_path(cr);
        /* split points as _gen_subpaths_from_points: at every anchor k
         * (k = 4, 8, ...) whose point differs from the previous one */
        Py_ssize_t start = 0;
        for (Py_ssize_t k = 4; k < n; k += 4) {
            if (eq2d(p + 3 * (k - 1), p + 3 * k, atol))
                continue;
            if (k - start >= 4)
                emit_subpath(cr, p + 3 * start, k - start, atol);
            start = k;
        }
        if (n - start >= 4)
            emit_subpath(cr, p + 3 * start, n - start, atol);
        Py_END_ALLOW_THREADS
    }
    PyBuffer_Release(&b);

    cairo_status_t st = cairo_status(cr);
    if (st != CAIRO_STATUS_SUCCESS) {
        PyErr_SetString(PyExc_RuntimeError, cairo_status_to_string(st));
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyMethodDef methods[] = {
    {"set_path", set_path, METH_VARARGS, "set_path(ctx, points, atol): cairo path of a VMobject"},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "_cairo_path", NULL, -1, methods};

PyMODINIT_FUNC PyInit__cairo_path(void)
{
    if (import_cairo() < 0)
        return NULL;
    return PyModule_Create(&module);
}
