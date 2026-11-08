// CUDA image math mirrors the software renderer. Planes are transposed so that
// adjacent row threads in the recursive filters access adjacent memory.
#include <cuda_fp16.h>
#include <math.h>

struct Params {
    int w, h, op, plane, mode, delay, first, frame, aux, vertical;
    float shift, frequency, intensity;
    int detail;
    float coefficients[7], initial[4];
    float transition[9];
};

__device__ float read_component(const char* row, int component, int half) {
    return half ? __half2float(((const __half*)row)[component]) : ((const float*)row)[component];
}
__device__ void write_component(char* row, int component, float value, int half) {
    if (half) ((__half*)row)[component] = __float2half_rn(value);
    else ((float*)row)[component] = value;
}

extern "C" __global__ void ntsc_frame(const char* src, char* dst, float* planes, Params p,
    int src_pitch, int dst_pitch, int height, int half_in, int half_out) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= p.w * height) return;
    int x = t / height, y = t % height;
    int n = p.w * p.h;
    if (p.op == 0) {
        if (y >= p.h) return;
        int row_index = min(p.first + y * p.mode, height - 1);
        const char* row = src + (long long)row_index * src_pitch;
        float r = read_component(row, x * 4 + 2, half_in);
        float g = read_component(row, x * 4 + 1, half_in);
        float b = read_component(row, x * 4, half_in);
        int j = x * p.h + y;
        planes[j] = fmaf(r, .299f, fmaf(g, .587f, b * .114f));
        planes[n + j] = fmaf(r, .5959f, fmaf(g, -.2746f, b * -.3213f));
        planes[n * 2 + j] = fmaf(r, .2115f, fmaf(g, -.5227f, b * .3112f));
    } else {
        if (p.mode == 2 && !p.detail && (y & 1) != p.first) return;
        int row = p.mode == 1 ? y : min(y / 2, p.h - 1);
        int j = x * p.h + row;
        float yy = planes[j], ii = planes[n + j], qq = planes[n * 2 + j];
        if (p.detail && ((y & 1) != p.first) && y > 0 && y < height - 1) {
            int lo = x * p.h + (y - 1) / 2, hi = x * p.h + (y + 1) / 2;
            yy = (planes[lo] + planes[hi]) * .5f;
            ii = (planes[n + lo] + planes[n + hi]) * .5f;
            qq = (planes[2 * n + lo] + planes[2 * n + hi]) * .5f;
        }
        char* out = dst + (long long)y * dst_pitch;
        write_component(out, x * 4 + 2, fmaf(qq, .619f, fmaf(ii, .956f, yy)), half_out);
        write_component(out, x * 4 + 1, fmaf(qq, -.647f, fmaf(ii, -.272f, yy)), half_out);
        write_component(out, x * 4, fmaf(qq, 1.703f, fmaf(ii, -1.106f, yy)), half_out);
        // Preserve transparency through the effect.
        write_component(out, x * 4 + 3, read_component(src + (long long)y * src_pitch, x * 4 + 3, half_in), half_out);
    }
}

__device__ unsigned hash1(unsigned v) {
    v = (v ^ 2747636419u) * 2654435769u;
    v = (v ^ (v >> 16)) * 2654435769u;
    return (v ^ (v >> 16)) * 2654435769u;
}
__device__ float gradient(unsigned hash) {
    float v = float(((hash >> 28) & 7) + 1);
    return __uint_as_float(__float_as_uint(v) | (hash & 0x80000000u));
}
__device__ float simplex(float x, int seed) {
    float f = floorf(x), a = x - f, b = a - 1.f;
    int i = (int)f;
    float t0 = fmaf(-a, a, 1.f), t1 = fmaf(-b, b, 1.f);
    t0 *= t0; t1 *= t1;
    float g0 = gradient(hash1((unsigned)(i ^ seed)));
    float g1 = gradient(hash1((unsigned)((i + 1) ^ seed)));
    return fmaf((t0 * t0) * g0, a, (t1 * t1) * g1 * b);
}
__device__ float imult(int phase) { return (phase & 1) ? 0.f : ((phase & 2) ? 1.f : -1.f); }
__device__ float qmult(int phase) { return (phase & 1) ? ((phase & 2) ? 1.f : -1.f) : 0.f; }

__device__ float sample_shift(const float* src, int x, int y, int w, int h, float shift, int extend) {
    int integer = (int)shift - (shift < 0.f ? 1 : 0);
    float fraction = shift - (float)integer;
    int left = x - integer - 1, right = left + 1;
    float boundary = extend ? src[(integer >= 0 ? 0 : w - 1) * h + y] : 0.f;
    float a = left >= 0 && left < w ? src[left * h + y] : boundary;
    float b = right >= 0 && right < w ? src[right * h + y] : boundary;
    // Scalar boundaries use the CPU scalar formula. Interior blocks use FMA.
    return left >= 0 && right < w ? fmaf(a - b, fraction, b) : a * fraction + b * (1.f - fraction);
}

extern "C" __global__ void ntsc_operation(float* planes, float* scratch,
    const float* modulated, const float* aux, Params p) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int n = p.w * p.h;
    if (p.op == 13 || p.op == 14 || p.op == 15) {
        int segments = (p.w + 127) / 128;
        int states = segments * p.h * 3;
        if (p.op == 14) {
            if (t >= p.h) return;
            float* row = planes + p.plane * n + t;
            float z[4] = {0.f,0.f,0.f,0.f};
            for (int i = 0; i < 4; ++i) z[i] = p.initial[i] * (p.first ? row[0] : 0.f);
            // Warm up before the delayed first output.
            for (int x = 0; x < p.delay; ++x) {
                float sample = row[min(x, p.w - 1) * p.h];
                float result = fmaf(p.coefficients[0], sample, z[0]);
                for (int i = 0; i < p.mode - 1; ++i)
                    z[i] = fmaf(-p.coefficients[4+i], result, fmaf(p.coefficients[i+1], sample, z[i+1]));
            }
            for (int segment = 0; segment < segments; ++segment) {
                int k = (segment * p.h + t) * 3;
                float next[3] = {0.f,0.f,0.f};
                for (int i = 0; i < 3; ++i) {
                    scratch[n + states + k + i] = z[i];
                    next[i] = scratch[n + k + i];
                    for (int j = 0; j < 3; ++j) next[i] += p.transition[i*3+j] * z[j];
                }
                for (int i = 0; i < 3; ++i) z[i] = next[i];
            }
            return;
        }
        if (t >= segments * p.h) return;
        int y = t % p.h, segment = t / p.h;
        int start = segment * 128, end = min(start + 128, p.w);
        float z[4] = {0.f,0.f,0.f,0.f};
        int k = t * 3;
        if (p.op == 15) for (int i = 0; i < 3; ++i) z[i] = scratch[n + states + k + i];
        for (int x = start; x < end; ++x) {
            int index = x * p.h + y;
            if (p.op == 13) {
                float sample = planes[p.plane * n + min(x + p.delay, p.w - 1) * p.h + y];
                float result = fmaf(p.coefficients[0], sample, z[0]);
                for (int i = 0; i < p.mode - 1; ++i)
                    z[i] = fmaf(-p.coefficients[4+i], result, fmaf(p.coefficients[i+1], sample, z[i+1]));
                scratch[index] = result;
            } else {
                float result = z[0];
                planes[p.plane * n + index] = scratch[index] + result;
                for (int i = 0; i < p.mode - 1; ++i) z[i] = fmaf(-p.coefficients[4+i], result, z[i+1]);
            }
        }
        if (p.op == 13) for (int i = 0; i < 3; ++i) scratch[n + k + i] = z[i];
        return;
    }
    if (p.op == 1) {
        if (t >= p.h) return;
        float* row = planes + p.plane * n + t;
        float value = p.first ? row[0] : 0.f;
        float z[4];
        for (int i = 0; i < 4; ++i) z[i] = p.initial[i] * value;
        for (int x = 0; x < p.w + p.delay; ++x) {
            float sample = row[min(x, p.w - 1) * p.h];
            float result = fmaf(p.coefficients[0], sample, z[0]);
            for (int i = 0; i < p.mode - 1; ++i)
                z[i] = fmaf(-p.coefficients[4 + i], result, fmaf(p.coefficients[i + 1], sample, z[i + 1]));
            if (x >= p.delay) row[(x - p.delay) * p.h] = result;
        }
        return;
    }
    if (p.op == 10) {
        if (t >= p.h) return;
        float* row = planes + t;
        float queue[4] = {16.f / 255.f, 16.f / 255.f, row[0], row[p.h]};
        float sum = ((queue[0] + queue[1]) + queue[2]) + queue[3];
        for (int x = 0; x < p.w - 2; ++x) {
            float c = row[(x + 2) * p.h];
            sum -= queue[x & 3]; queue[x & 3] = c; sum += c;
            row[x * p.h] = sum * .25f;
        }
        float last = row[(p.w - 1) * p.h];
        for (int x = p.w - 2; x < p.w; ++x) {
            sum -= queue[x & 3]; queue[x & 3] = last; sum += last;
            row[x * p.h] = sum * .25f;
        }
        return;
    }
    if (p.op == 9) {
        if (t >= p.detail) return;
        int index = __float_as_uint(aux[p.aux + t * 2]);
        int x = index % p.w, y = index / p.w;
        planes[x * p.h + y] += aux[p.aux + t * 2 + 1];
        return;
    }
    if (t >= n) return;
    int x = t / p.h, y = t % p.h;
    if (p.op == 2) {
        int phase = (x + ((p.frame + (y << 1)) & 2)) & 3;
        float im = (phase & 1) ? 0.f : ((phase & 2) ? -1.f : 1.f);
        int qp = (phase + 3) & 3;
        float qm = (qp & 1) ? 0.f : ((qp & 2) ? -1.f : 1.f);
        planes[t] += planes[n + t] * im + planes[2 * n + t] * qm;
    } else if (p.op == 4) {
        if (p.mode == 0) {
            float a = x > 0 ? modulated[(x - 1) * p.h + y] : 16.f / 255.f;
            planes[t] = ((a + modulated[t]) + modulated[min(x + 1, p.w - 1) * p.h + y]
                + modulated[min(x + 2, p.w - 1) * p.h + y]) * .25f;
        } else {
            int prev = y == 0 ? 1 : y - 1, next = y == p.h - 1 ? p.h - 2 : y + 1;
            float c = modulated[t], top = modulated[x * p.h + prev], bottom = modulated[x * p.h + next];
            if (p.mode == 1) { planes[t] = (top + c) * .5f; return; }
            if (p.mode == 2) { planes[t] = c * .5f + top * .25f + bottom * .25f; return; }
            int lx = x < 2 ? x + 2 : x - 2, rx = x > p.w - 3 ? x - 2 : x + 2;
            float tl = modulated[lx * p.h + prev], tr = modulated[rx * p.h + prev];
            float l = modulated[lx * p.h + y], r = modulated[rx * p.h + y];
            float bl = modulated[lx * p.h + next], br = modulated[rx * p.h + next];
            if (p.mode == 3) {
                planes[t] = c * .75f + ((l + r) + (top + bottom)) * .125f - ((tl + tr) + (bl + br)) * .0625f;
            } else {
                float gx = (tr - tl) + (r - l) + (br - bl), gy = (bl - tl) + (bottom - top) + (br - tr);
                float gxt = gx * 1000.f, gyt = gy * 1000.f;
                float gd1 = (gx + gy) * (1.f / (.001f * 1.41421356237f));
                float gd2 = (gx - gy) * (1.f / (.001f * 1.41421356237f));
                float ax = 1.f / (1.f + gxt * gxt), ay = 1.f / (1.f + gyt * gyt);
                float norm = .25f / (ax + ay), wx = ax * norm, wy = ay * norm;
                float wd1 = norm / fmaxf(1.f + gd1 * gd1, 2.f / (.001f * .001f));
                float wd2 = norm / fmaxf(1.f + gd2 * gd2, 2.f / (.001f * .001f));
                planes[t] = c * (.5f + 2.f * (wd1 + wd2)) + ((l + r) * wx + (top + bottom) * wy)
                    - ((tl + br) * wd1 + (tr + bl) * wd2);
            }
        }
    } else if (p.op == 5) {
        int phase = (x + ((p.frame + (y << 1)) & 2)) & 3;
        float center = planes[t] - modulated[t];
        float i = center * imult(phase), q = center * qmult(phase);
        if (x > 0) {
            float a = planes[t - p.h] - modulated[t - p.h];
            i = fmaf(a, imult((phase + 3) & 3) * .5f, i);
            q = fmaf(a, qmult((phase + 3) & 3) * .5f, q);
        }
        if (x + 1 < p.w) {
            float b = planes[t + p.h] - modulated[t + p.h];
            i = fmaf(b, imult((phase + 1) & 3) * .5f, i);
            q = fmaf(b, qmult((phase + 1) & 3) * .5f, q);
        }
        planes[n + t] = i; planes[2 * n + t] = q;
    } else if (p.op == 6) {
        int base = p.aux + y * 3;
        int seed = __float_as_int(aux[base]);
        float coordinate = (aux[base + 1] + (float)x) * p.frequency;
        float noise = simplex(coordinate, seed);
        for (int octave = 1; octave < p.detail; ++octave) {
            coordinate *= 2.f;
            noise = fmaf(simplex(coordinate, seed), 1.f, noise);
        }
        planes[p.plane * n + t] += noise * aux[base + 2];
    } else if (p.op == 7) {
        int source_y = y - p.vertical;
        float shift = aux[p.aux + y * 2];
        int start = (int)aux[p.aux + y * 2 + 1];
        for (int plane = 0; plane < 3; ++plane) if (p.plane & (1 << plane)) {
            float result = scratch[plane * n + t];
            if (x >= start) result = source_y < 0 || source_y >= p.h ? 0.f : sample_shift(scratch + plane * n, x, source_y, p.w, p.h, shift, p.mode);
            planes[plane * n + t] = result;
        }
    } else if (p.op == 8) {
        for (int plane = 1; plane <= 2; ++plane)
            planes[plane * n + t] = ((y > 0 ? scratch[plane * n + t - 1] : 0.f) + scratch[plane * n + t]) * .5f;
    } else if (p.op == 11) {
        float s = aux[p.aux + y * 2], c = aux[p.aux + y * 2 + 1];
        float i = planes[n + t], q = planes[2 * n + t];
        planes[n + t] = i * c - q * s;
        planes[2 * n + t] = i * s + q * c;
    } else if (p.op == 12) {
        if (aux[p.aux + y] != 0.f) { planes[n + t] = 0.f; planes[2 * n + t] = 0.f; }
    }
}
