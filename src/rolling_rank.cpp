#include <Python.h>
#include <numpy/arrayobject.h>
#include <cmath>
#include <vector>
#include <cstdint>   // ✅ for int32_t
#include <omp.h>

// 把相同 symbol 的行稳定分组到连续区间，返回 starts 与 idx_sorted（分组后的顺序对应的原行号）
static void stable_group_by_codes(
    const int32_t* codes, npy_intp n, int n_symbols,
    std::vector<npy_intp>& starts,          // size n_symbols+1
    std::vector<npy_intp>& idx_sorted       // size n
){
    starts.assign(n_symbols + 1, 0);
    for (npy_intp i = 0; i < n; ++i) starts[codes[i] + 1]++;

    for (int s = 0; s < n_symbols; ++s) starts[s + 1] += starts[s];

    idx_sorted.resize(n);
    std::vector<npy_intp> write_ptr = starts;
    for (npy_intp i = 0; i < n; ++i) {
        const int s = codes[i];
        const npy_intp p = write_ptr[s]++;
        idx_sorted[p] = i;              // 分组后位置 p 对应原行 i
    }
}

static PyObject* rolling_rank(PyObject* self, PyObject* args) {
    PyArrayObject *codes_obj, *values_obj;
    int window, n_symbols;

    if (!PyArg_ParseTuple(args, "OOii", &codes_obj, &values_obj, &window, &n_symbols))
        return NULL;

    // ✅ 类型假设：codes=int32，values=float32（与 Python 侧保持一致）
    int32_t* codes  = (int32_t*)PyArray_DATA(codes_obj);
    float*   values = (float*)PyArray_DATA(values_obj);
    const npy_intp n = PyArray_SIZE(values_obj);

    // 1) 稳定分组，得到每个 symbol 的连续区间
    std::vector<npy_intp> starts, idx_sorted;
    stable_group_by_codes(codes, n, n_symbols, starts, idx_sorted);

    // 2) 构造分组后的 values_sorted（连续内存，利于缓存）
    std::vector<float> values_sorted(n);
    for (npy_intp i = 0; i < n; ++i) values_sorted[i] = values[idx_sorted[i]];

    // 3) 输出缓冲（分组序）
    std::vector<float> out_sorted(n);

    // 4) 每个 symbol 段并行滚动计算：百分位 = (<=v 的非 NaN 个数)/(非 NaN 个数)
    #pragma omp parallel for schedule(static)
    for (int s = 0; s < n_symbols; ++s) {
        const npy_intp start = starts[s];
        const npy_intp end   = starts[s + 1];

        float buf[256];     // ✅ 窗口上限 256（够用）；若 window=20 可改成 buf[20] 更快
        int   head = 0, count = 0;

        for (npy_intp i = start; i < end; ++i) {
            const float v = values_sorted[i];

            if (count < window) {
                buf[count++] = v;
            } else {
                buf[head] = v;
                head += 1; if (head == window) head = 0;
            }

            if (std::isnan(v)) {    // 题面：当前值是 NaN -> 结果 0.0
                out_sorted[i] = 0.0f;
                continue;
            }

            int valid = 0, leq = 0;
            int idx = head;         // 当前有效窗口从 head 开始，连续 count 个
            for (int t = 0; t < count; ++t) {
                const float u = buf[idx];
                if (!std::isnan(u)) {
                    valid++;
                    if (u <= v) leq++;
                }
                idx += 1; if (idx == window) idx = 0;
            }

            if (valid == 0) {
                out_sorted[i] = 0.0f;    // 窗口内都是 NaN
            } else {
                out_sorted[i] = static_cast<float>(leq) / static_cast<float>(valid);
            }
        }
    }

    // 5) 把分组序结果 scatter 回原行序
    npy_intp dims[1] = { n };
    PyArrayObject* result_obj = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_FLOAT32);
    float* result = (float*)PyArray_DATA(result_obj);
    for (npy_intp i = 0; i < n; ++i) {
        result[idx_sorted[i]] = out_sorted[i];
    }

    return PyArray_Return(result_obj);
}

static PyMethodDef methods[] = {
    {"rolling_rank", (PyCFunction)rolling_rank, METH_VARARGS, "Rolling percentile rank per symbol (<= / valid) with OpenMP"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "rolling_rank_cpp",
    "Fast rolling rank",
    -1,
    methods
};

PyMODINIT_FUNC PyInit_rolling_rank_cpp(void) {
    import_array();
    return PyModule_Create(&moduledef);
}
