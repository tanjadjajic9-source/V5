#include "ImageUtil.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace XboxOTFM {

namespace {

// CRC-32 (the one PNG and zip use), computed eight bytes at a time. The picture file check reads gigabytes, so the
// plain one-byte-at-a-time loop was too slow for it.
struct CrcTables {
    uint32_t t[8][256];
    CrcTables() {
        for (uint32_t n = 0; n < 256; ++n) {
            uint32_t c = n;
            for (int k = 0; k < 8; ++k) c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
            t[0][n] = c;
        }
        for (uint32_t n = 0; n < 256; ++n) {
            uint32_t c = t[0][n];
            for (int k = 1; k < 8; ++k) {
                c = t[0][c & 0xFFu] ^ (c >> 8);
                t[k][n] = c;
            }
        }
    }
};

const CrcTables& GetCrcTables() {
    static const CrcTables tables;
    return tables;
}

// For every output position: which source pixels contribute and with which weights (the weights of one output sum to 1).
struct AxisFilter {
    std::vector<uint32_t> first;   // first source index per output
    std::vector<uint32_t> count;   // taps per output
    std::vector<uint32_t> offset;  // where this output's weights start in `weights`
    std::vector<float> weights;
};

void BuildAxisFilter(AxisFilter& f, uint32_t srcSize, double start, double length, uint32_t outSize, bool flip, bool nearest) {
    f.first.assign(outSize, 0);
    f.count.assign(outSize, 0);
    f.offset.assign(outSize, 0);
    f.weights.clear();
    const double scale = length / static_cast<double>(outSize);
    for (uint32_t o = 0; o < outSize; ++o) {
        const uint32_t oo = flip ? (outSize - 1 - o) : o;
        f.offset[o] = static_cast<uint32_t>(f.weights.size());
        if (nearest) {
            // the one source pixel under the centre of this output pixel (enlarging copies pixels, shrinking picks one)
            const double centre = start + (static_cast<double>(oo) + 0.5) * scale;
            const int64_t p = std::clamp<int64_t>(static_cast<int64_t>(std::floor(centre)), 0, static_cast<int64_t>(srcSize) - 1);
            f.weights.push_back(1.0f);
            f.first[o] = static_cast<uint32_t>(p);
            f.count[o] = 1;
            continue;
        }
        if (scale > 1.0) {
            const double lo = std::max(0.0, start + static_cast<double>(oo) * scale);
            const double hi = std::min(static_cast<double>(srcSize), start + static_cast<double>(oo + 1) * scale);
            int64_t p0 = static_cast<int64_t>(std::floor(lo));
            int64_t p1 = static_cast<int64_t>(std::ceil(hi));
            p0 = std::clamp<int64_t>(p0, 0, static_cast<int64_t>(srcSize) - 1);
            p1 = std::clamp<int64_t>(p1, p0 + 1, static_cast<int64_t>(srcSize));
            double total = 0.0;
            const size_t base = f.weights.size();
            for (int64_t p = p0; p < p1; ++p) {
                const double a = std::max(lo, static_cast<double>(p));
                const double b = std::min(hi, static_cast<double>(p + 1));
                const double w = std::max(0.0, b - a);
                f.weights.push_back(static_cast<float>(w));
                total += w;
            }
            if (total <= 1e-12) {
                f.weights[base] = 1.0f;
                for (size_t i = base + 1; i < f.weights.size(); ++i) f.weights[i] = 0.0f;
                total = 1.0;
            }
            for (size_t i = base; i < f.weights.size(); ++i) f.weights[i] = static_cast<float>(f.weights[i] / total);
            f.first[o] = static_cast<uint32_t>(p0);
            f.count[o] = static_cast<uint32_t>(p1 - p0);
        } else {
            const double c = start + (static_cast<double>(oo) + 0.5) * scale - 0.5;
            int64_t p0 = static_cast<int64_t>(std::floor(c));
            const double frac = c - static_cast<double>(p0);
            const int64_t last = static_cast<int64_t>(srcSize) - 1;
            const int64_t a = std::clamp<int64_t>(p0, 0, last);
            const int64_t b = std::clamp<int64_t>(p0 + 1, 0, last);
            if (a == b) {
                f.weights.push_back(1.0f);
                f.first[o] = static_cast<uint32_t>(a);
                f.count[o] = 1;
            } else {
                f.weights.push_back(static_cast<float>(1.0 - frac));
                f.weights.push_back(static_cast<float>(frac));
                f.first[o] = static_cast<uint32_t>(a);
                f.count[o] = 2;
            }
        }
    }
}

uint8_t ToByte(float v) {
    const float c = std::min(255.0f, std::max(0.0f, v));
    return static_cast<uint8_t>(c + 0.5f);
}

void PutU32(std::vector<uint8_t>& v, uint32_t x) {
    v.push_back(static_cast<uint8_t>(x >> 24));
    v.push_back(static_cast<uint8_t>(x >> 16));
    v.push_back(static_cast<uint8_t>(x >> 8));
    v.push_back(static_cast<uint8_t>(x));
}

void PutChunk(std::vector<uint8_t>& out, const char type[4], const std::vector<uint8_t>& data) {
    PutU32(out, static_cast<uint32_t>(data.size()));
    const size_t crcStart = out.size();
    out.insert(out.end(), type, type + 4);
    out.insert(out.end(), data.begin(), data.end());
    const uint32_t crc = Crc32(out.data() + crcStart, out.size() - crcStart);
    PutU32(out, crc);
}

} // namespace

uint32_t Crc32(const uint8_t* data, size_t size, uint32_t previous) {
    const CrcTables& T = GetCrcTables();
    uint32_t c = ~previous;
    while (size >= 8) {
        const uint32_t lo = c ^ (static_cast<uint32_t>(data[0]) | (static_cast<uint32_t>(data[1]) << 8) |
                                 (static_cast<uint32_t>(data[2]) << 16) | (static_cast<uint32_t>(data[3]) << 24));
        const uint32_t hi = static_cast<uint32_t>(data[4]) | (static_cast<uint32_t>(data[5]) << 8) |
                            (static_cast<uint32_t>(data[6]) << 16) | (static_cast<uint32_t>(data[7]) << 24);
        c = T.t[7][lo & 0xFFu] ^ T.t[6][(lo >> 8) & 0xFFu] ^ T.t[5][(lo >> 16) & 0xFFu] ^ T.t[4][lo >> 24] ^
            T.t[3][hi & 0xFFu] ^ T.t[2][(hi >> 8) & 0xFFu] ^ T.t[1][(hi >> 16) & 0xFFu] ^ T.t[0][hi >> 24];
        data += 8;
        size -= 8;
    }
    while (size > 0) {
        c = T.t[0][(c ^ *data) & 0xFFu] ^ (c >> 8);
        ++data;
        --size;
    }
    return ~c;
}

uint64_t Fnv1a64(const uint8_t* data, size_t size) {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < size; ++i) {
        h ^= data[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}

bool ResampleRegion(const uint8_t* rgb, uint32_t w, uint32_t h,
                    double cx, double cy, double cw, double ch,
                    uint32_t dstW, uint32_t dstH, bool flipHorizontal, std::vector<float>& out, bool nearest) {
    out.clear();
    if (!rgb || w == 0 || h == 0 || dstW == 0 || dstH == 0 || dstW > 8192 || dstH > 8192) return false;
    if (!(cw > 0.0) || !(ch > 0.0) || !std::isfinite(cx) || !std::isfinite(cy) || !std::isfinite(cw) || !std::isfinite(ch)) return false;
    // The region must lie inside the picture (tiny float slack is tolerated).
    cx = std::max(0.0, cx);
    cy = std::max(0.0, cy);
    cw = std::min(cw, static_cast<double>(w) - cx);
    ch = std::min(ch, static_cast<double>(h) - cy);
    if (!(cw > 0.0) || !(ch > 0.0)) return false;

    AxisFilter fx, fy;
    BuildAxisFilter(fx, w, cx, cw, dstW, flipHorizontal, nearest);
    BuildAxisFilter(fy, h, cy, ch, dstH, false, nearest);

    // Only the source rows that are used are filtered horizontally.
    uint32_t rowLo = h, rowHi = 0;
    for (uint32_t o = 0; o < dstH; ++o) {
        rowLo = std::min(rowLo, fy.first[o]);
        rowHi = std::max(rowHi, fy.first[o] + fy.count[o]);
    }
    if (rowHi <= rowLo) return false;
    const uint32_t rows = rowHi - rowLo;

    std::vector<float> tmp(static_cast<size_t>(rows) * dstW * 3);
    for (uint32_t r = 0; r < rows; ++r) {
        const uint8_t* src = rgb + static_cast<size_t>(rowLo + r) * w * 3;
        float* dst = tmp.data() + static_cast<size_t>(r) * dstW * 3;
        for (uint32_t o = 0; o < dstW; ++o) {
            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f;
            const float* wt = fx.weights.data() + fx.offset[o];
            const uint8_t* p = src + static_cast<size_t>(fx.first[o]) * 3;
            for (uint32_t k = 0; k < fx.count[o]; ++k, p += 3) {
                a0 += wt[k] * p[0];
                a1 += wt[k] * p[1];
                a2 += wt[k] * p[2];
            }
            dst[o * 3 + 0] = a0;
            dst[o * 3 + 1] = a1;
            dst[o * 3 + 2] = a2;
        }
    }

    out.resize(static_cast<size_t>(dstW) * dstH * 3);
    for (uint32_t o = 0; o < dstH; ++o) {
        const float* wt = fy.weights.data() + fy.offset[o];
        float* dst = out.data() + static_cast<size_t>(o) * dstW * 3;
        const uint32_t firstRow = fy.first[o] - rowLo;
        for (uint32_t x = 0; x < dstW * 3; ++x) {
            float acc = 0.0f;
            for (uint32_t k = 0; k < fy.count[o]; ++k) {
                acc += wt[k] * tmp[static_cast<size_t>(firstRow + k) * dstW * 3 + x];
            }
            dst[x] = acc;
        }
    }
    return true;
}

Rgb8Image ResizeRgb(const uint8_t* rgb, uint32_t w, uint32_t h, uint32_t dstW, uint32_t dstH) {
    Rgb8Image img;
    std::vector<float> f;
    if (!ResampleRegion(rgb, w, h, 0.0, 0.0, w, h, dstW, dstH, false, f)) return img;
    img.w = dstW;
    img.h = dstH;
    img.px.resize(f.size());
    for (size_t i = 0; i < f.size(); ++i) img.px[i] = ToByte(f[i]);
    return img;
}

bool LooksLikePixelArt(const uint8_t* rgb, uint32_t w, uint32_t h, uint32_t x0, uint32_t y0, uint32_t rw, uint32_t rh) {
    if (!rgb || w == 0 || h == 0 || x0 >= w || y0 >= h || rw == 0 || rh == 0) return false;
    rw = std::min(rw, w - x0);
    rh = std::min(rh, h - y0);
    const uint64_t pixels = static_cast<uint64_t>(rw) * rh;
    constexpr uint32_t kMaxColors = 256;
    constexpr uint32_t kSlots = 1024;            // a power of two, at least four times the colours we accept
    constexpr uint32_t kEmpty = 0xFFFFFFFFu;     // no 24-bit colour can be this
    uint32_t table[kSlots];
    std::fill(table, table + kSlots, kEmpty);
    uint32_t unique = 0;
    for (uint32_t y = 0; y < rh; ++y) {
        const uint8_t* p = rgb + (static_cast<size_t>(y0 + y) * w + x0) * 3;
        for (uint32_t x = 0; x < rw; ++x, p += 3) {
            const uint32_t key = (static_cast<uint32_t>(p[0]) << 16) | (static_cast<uint32_t>(p[1]) << 8) | p[2];
            uint32_t slot = (key * 2654435761u) >> 22; // top 10 bits
            while (table[slot] != kEmpty && table[slot] != key) slot = (slot + 1) & (kSlots - 1);
            if (table[slot] == kEmpty) {
                table[slot] = key;
                if (++unique > kMaxColors) return false; // a photo: give up at once
            }
        }
    }
    return static_cast<uint64_t>(unique) * 4 <= pixels;
}

Rgb8Image FitToSquare(const uint8_t* rgb, uint32_t w, uint32_t h, uint32_t size, FitMode mode, ResizeStyle style) {
    Rgb8Image img;
    if (!rgb || w == 0 || h == 0 || size == 0) return img;
    double cx = 0.0, cy = 0.0, cw = w, ch = h;
    if (mode == FitMode::CropCenter) {
        const double side = static_cast<double>(std::min(w, h));
        cx = (static_cast<double>(w) - side) * 0.5;
        cy = (static_cast<double>(h) - side) * 0.5;
        cw = side;
        ch = side;
    }
    bool nearest = style == ResizeStyle::Sharp;
    if (style == ResizeStyle::Auto && (cw < static_cast<double>(size) || ch < static_cast<double>(size))) {
        // Only a picture that has to be enlarged is looked at: shrinking is left alone, because averaging many pixels is right
        // for photos, and a big picture that was blown up from pixel art shrinks back cleanly anyway.
        nearest = LooksLikePixelArt(rgb, w, h, static_cast<uint32_t>(cx), static_cast<uint32_t>(cy),
                                    static_cast<uint32_t>(std::ceil(cw)), static_cast<uint32_t>(std::ceil(ch)));
    }
    std::vector<float> f;
    if (!ResampleRegion(rgb, w, h, cx, cy, cw, ch, size, size, false, f, nearest)) return img;
    img.w = size;
    img.h = size;
    img.px.resize(f.size());
    for (size_t i = 0; i < f.size(); ++i) img.px[i] = ToByte(f[i]);
    return img;
}

Rgb8Image UpscaleNearest(const Rgb8Image& src, uint32_t factor) {
    Rgb8Image out;
    if (!src.Valid() || factor == 0 || factor > 64 || static_cast<uint64_t>(src.w) * factor > 8192 || static_cast<uint64_t>(src.h) * factor > 8192) return out;
    out.w = src.w * factor;
    out.h = src.h * factor;
    out.px.resize(static_cast<size_t>(out.w) * out.h * 3);
    for (uint32_t y = 0; y < out.h; ++y) {
        const uint8_t* srow = src.px.data() + static_cast<size_t>(y / factor) * src.w * 3;
        uint8_t* drow = out.px.data() + static_cast<size_t>(y) * out.w * 3;
        for (uint32_t x = 0; x < out.w; ++x) {
            std::memcpy(drow + static_cast<size_t>(x) * 3, srow + static_cast<size_t>(x / factor) * 3, 3);
        }
    }
    return out;
}

Rgb8Image MakeGrid(const std::vector<Rgb8Image>& tiles, uint32_t columns, uint32_t pad, uint8_t bgR, uint8_t bgG, uint8_t bgB) {
    Rgb8Image out;
    if (tiles.empty() || columns == 0) return out;
    uint32_t tw = 0, th = 0;
    for (const auto& t : tiles) {
        if (!t.Valid()) continue;
        tw = std::max(tw, t.w);
        th = std::max(th, t.h);
    }
    if (tw == 0 || th == 0) return out;
    const uint32_t cols = std::min<uint32_t>(columns, static_cast<uint32_t>(tiles.size()));
    const uint32_t rows = (static_cast<uint32_t>(tiles.size()) + cols - 1) / cols;
    const uint64_t W = static_cast<uint64_t>(cols) * tw + static_cast<uint64_t>(cols + 1) * pad;
    const uint64_t H = static_cast<uint64_t>(rows) * th + static_cast<uint64_t>(rows + 1) * pad;
    if (W > 8192 || H > 8192) return out;
    out.w = static_cast<uint32_t>(W);
    out.h = static_cast<uint32_t>(H);
    out.px.resize(static_cast<size_t>(out.w) * out.h * 3);
    for (size_t i = 0; i < out.px.size(); i += 3) {
        out.px[i] = bgR;
        out.px[i + 1] = bgG;
        out.px[i + 2] = bgB;
    }
    for (size_t i = 0; i < tiles.size(); ++i) {
        const Rgb8Image& t = tiles[i];
        if (!t.Valid()) continue;
        const uint32_t col = static_cast<uint32_t>(i) % cols;
        const uint32_t row = static_cast<uint32_t>(i) / cols;
        const uint32_t x0 = pad + col * (tw + pad) + (tw - t.w) / 2;
        const uint32_t y0 = pad + row * (th + pad) + (th - t.h) / 2;
        for (uint32_t y = 0; y < t.h; ++y) {
            std::memcpy(out.px.data() + (static_cast<size_t>(y0 + y) * out.w + x0) * 3,
                        t.px.data() + static_cast<size_t>(y) * t.w * 3, static_cast<size_t>(t.w) * 3);
        }
    }
    return out;
}

Rgb8Image FloatChwToRgb8(const float* chw, uint32_t w, uint32_t h) {
    Rgb8Image img;
    if (!chw || w == 0 || h == 0) return img;
    img.w = w;
    img.h = h;
    const size_t hw = static_cast<size_t>(w) * h;
    img.px.resize(hw * 3);
    for (size_t i = 0; i < hw; ++i) {
        for (size_t c = 0; c < 3; ++c) {
            const float v = chw[c * hw + i];
            const float u = std::isfinite(v) ? std::min(1.0f, std::max(-1.0f, v)) : 0.0f;
            img.px[i * 3 + c] = static_cast<uint8_t>((u + 1.0f) * 127.5f + 0.5f);
        }
    }
    return img;
}

// ---------------------------------------------------------------------------------------------
// Colour palettes (the optional "crisp colours" finish)
// ---------------------------------------------------------------------------------------------
namespace {

// One colour of the picture set and how many pixels have it (after merging: the average of a few close colours).
struct PaletteEntry {
    float c[3] = { 0.0f, 0.0f, 0.0f };
    double weight = 0.0;
    uint32_t key = 0; // only used to break ties, so the result never depends on how the sort is implemented
};

struct PaletteBox {
    size_t lo = 0, hi = 0; // entries [lo, hi)
    double sse = 0.0;      // weighted squared distance of the box's colours from their average
};

struct BoxStats {
    double weight = 0.0;
    double mean[3] = { 0.0, 0.0, 0.0 };
    double variance[3] = { 0.0, 0.0, 0.0 };
};

BoxStats StatsOf(const std::vector<PaletteEntry>& e, size_t lo, size_t hi) {
    BoxStats s;
    double sum[3] = { 0.0, 0.0, 0.0 };
    double sq[3] = { 0.0, 0.0, 0.0 };
    for (size_t i = lo; i < hi; ++i) {
        s.weight += e[i].weight;
        for (int k = 0; k < 3; ++k) {
            const double v = static_cast<double>(e[i].c[k]);
            sum[k] += e[i].weight * v;
            sq[k] += e[i].weight * v * v;
        }
    }
    if (s.weight <= 0.0) return s;
    for (int k = 0; k < 3; ++k) {
        s.mean[k] = sum[k] / s.weight;
        s.variance[k] = std::max(0.0, sq[k] / s.weight - s.mean[k] * s.mean[k]);
    }
    return s;
}

double BoxSse(const std::vector<PaletteEntry>& e, size_t lo, size_t hi) {
    const BoxStats s = StatsOf(e, lo, hi);
    return s.weight * (s.variance[0] + s.variance[1] + s.variance[2]);
}

} // namespace

std::vector<uint8_t> MakePalette(const uint8_t* rgb, size_t pixelCount, uint32_t maxColors) {
    std::vector<uint8_t> palette;
    if (!rgb || pixelCount == 0) return palette;
    const size_t limit = std::clamp<size_t>(maxColors, 1, 256);

    // 1. Every different colour and how often it occurs.
    std::vector<PaletteEntry> entries;
    {
        std::vector<uint32_t> keys(pixelCount);
        for (size_t i = 0; i < pixelCount; ++i) {
            keys[i] = (static_cast<uint32_t>(rgb[i * 3]) << 16) | (static_cast<uint32_t>(rgb[i * 3 + 1]) << 8) | rgb[i * 3 + 2];
        }
        std::sort(keys.begin(), keys.end());
        for (size_t i = 0; i < keys.size();) {
            size_t j = i;
            while (j < keys.size() && keys[j] == keys[i]) ++j;
            PaletteEntry e;
            e.c[0] = static_cast<float>((keys[i] >> 16) & 0xFFu);
            e.c[1] = static_cast<float>((keys[i] >> 8) & 0xFFu);
            e.c[2] = static_cast<float>(keys[i] & 0xFFu);
            e.weight = static_cast<double>(j - i);
            e.key = keys[i];
            entries.push_back(e);
            i = j;
        }
    }

    // 2. Few enough colours: they are the palette, exactly as they are.
    if (entries.size() <= limit) {
        for (const PaletteEntry& e : entries) {
            palette.push_back(static_cast<uint8_t>(e.c[0]));
            palette.push_back(static_cast<uint8_t>(e.c[1]));
            palette.push_back(static_cast<uint8_t>(e.c[2]));
        }
        return palette;
    }

    // 3. A photo-like set has far too many colours to search through: merge colours that share a 5-bit cell first.
    constexpr size_t kMaxEntries = 32768;
    if (entries.size() > kMaxEntries) {
        struct Cell { double w = 0.0; double s[3] = { 0.0, 0.0, 0.0 }; };
        std::vector<Cell> cells(32768);
        for (const PaletteEntry& e : entries) {
            const uint32_t r = (e.key >> 16) & 0xFFu, g = (e.key >> 8) & 0xFFu, b = e.key & 0xFFu;
            Cell& cell = cells[((r >> 3) << 10) | ((g >> 3) << 5) | (b >> 3)];
            cell.w += e.weight;
            cell.s[0] += e.weight * r;
            cell.s[1] += e.weight * g;
            cell.s[2] += e.weight * b;
        }
        entries.clear();
        for (size_t i = 0; i < cells.size(); ++i) {
            if (cells[i].w <= 0.0) continue;
            PaletteEntry e;
            for (int k = 0; k < 3; ++k) e.c[k] = static_cast<float>(cells[i].s[k] / cells[i].w);
            e.weight = cells[i].w;
            e.key = static_cast<uint32_t>(i);
            entries.push_back(e);
        }
    }

    // 4. Median cut: keep splitting the box of colours with the most spread, along its widest colour channel, at the middle pixel.
    std::vector<PaletteBox> boxes;
    {
        PaletteBox all;
        all.lo = 0;
        all.hi = entries.size();
        all.sse = BoxSse(entries, all.lo, all.hi);
        boxes.push_back(all);
    }
    while (boxes.size() < limit) {
        size_t best = boxes.size();
        for (size_t i = 0; i < boxes.size(); ++i) {
            if (boxes[i].hi - boxes[i].lo < 2 || boxes[i].sse <= 0.0) continue;
            if (best == boxes.size() || boxes[i].sse > boxes[best].sse) best = i;
        }
        if (best == boxes.size()) break;
        const PaletteBox box = boxes[best];
        const BoxStats st = StatsOf(entries, box.lo, box.hi);
        int channel = 0;
        if (st.variance[1] > st.variance[channel]) channel = 1;
        if (st.variance[2] > st.variance[channel]) channel = 2;
        std::sort(entries.begin() + static_cast<std::ptrdiff_t>(box.lo), entries.begin() + static_cast<std::ptrdiff_t>(box.hi),
                  [channel](const PaletteEntry& a, const PaletteEntry& b) {
                      if (a.c[channel] != b.c[channel]) return a.c[channel] < b.c[channel];
                      return a.key < b.key;
                  });
        double run = 0.0;
        size_t mid = box.lo + 1;
        for (size_t i = box.lo; i < box.hi; ++i) {
            run += entries[i].weight;
            if (run * 2.0 >= st.weight) { mid = i + 1; break; }
        }
        mid = std::clamp(mid, box.lo + 1, box.hi - 1);
        PaletteBox a, b;
        a.lo = box.lo; a.hi = mid; a.sse = BoxSse(entries, a.lo, a.hi);
        b.lo = mid;    b.hi = box.hi; b.sse = BoxSse(entries, b.lo, b.hi);
        boxes[best] = a;
        boxes.push_back(b);
    }

    // 5. The colour of each box is its weighted average.
    const size_t count = boxes.size();
    std::vector<double> centre(count * 3, 0.0);
    for (size_t i = 0; i < count; ++i) {
        const BoxStats st = StatsOf(entries, boxes[i].lo, boxes[i].hi);
        for (int k = 0; k < 3; ++k) centre[i * 3 + static_cast<size_t>(k)] = st.mean[k];
    }

    // 6. A few rounds of k-means: every colour goes to its nearest centre, and the centres move to the average of their colours.
    std::vector<uint16_t> owner(entries.size(), 0xFFFFu);
    for (int round = 0; round < 6; ++round) {
        std::vector<double> sum(count * 3, 0.0);
        std::vector<double> weight(count, 0.0);
        bool moved = false;
        for (size_t i = 0; i < entries.size(); ++i) {
            size_t bestJ = 0;
            double bestD = 1e300;
            for (size_t j = 0; j < count; ++j) {
                const double dr = static_cast<double>(entries[i].c[0]) - centre[j * 3];
                const double dg = static_cast<double>(entries[i].c[1]) - centre[j * 3 + 1];
                const double db = static_cast<double>(entries[i].c[2]) - centre[j * 3 + 2];
                const double d = dr * dr + dg * dg + db * db;
                if (d < bestD) { bestD = d; bestJ = j; }
            }
            if (owner[i] != static_cast<uint16_t>(bestJ)) { owner[i] = static_cast<uint16_t>(bestJ); moved = true; }
            weight[bestJ] += entries[i].weight;
            for (int k = 0; k < 3; ++k) sum[bestJ * 3 + static_cast<size_t>(k)] += entries[i].weight * static_cast<double>(entries[i].c[k]);
        }
        for (size_t j = 0; j < count; ++j) {
            if (weight[j] <= 0.0) continue; // an empty centre stays where it is
            for (size_t k = 0; k < 3; ++k) centre[j * 3 + k] = sum[j * 3 + k] / weight[j];
        }
        if (!moved) break;
    }

    // 7. Whole numbers, and no colour twice.
    for (size_t j = 0; j < count; ++j) {
        uint8_t c[3];
        for (size_t k = 0; k < 3; ++k) c[k] = static_cast<uint8_t>(std::clamp(std::floor(centre[j * 3 + k] + 0.5), 0.0, 255.0));
        bool seen = false;
        for (size_t q = 0; q + 2 < palette.size(); q += 3) {
            if (palette[q] == c[0] && palette[q + 1] == c[1] && palette[q + 2] == c[2]) { seen = true; break; }
        }
        if (!seen) palette.insert(palette.end(), c, c + 3);
    }
    return palette;
}

void SnapToPalette(Rgb8Image& img, const std::vector<uint8_t>& palette) {
    if (!img.Valid() || palette.size() < 3) return;
    const size_t count = palette.size() / 3;
    uint32_t lastKey = 0xFFFFFFFFu;
    uint8_t lastOut[3] = { 0, 0, 0 };
    const size_t pixels = static_cast<size_t>(img.w) * img.h;
    for (size_t i = 0; i < pixels; ++i) {
        uint8_t* p = img.px.data() + i * 3;
        const uint32_t key = (static_cast<uint32_t>(p[0]) << 16) | (static_cast<uint32_t>(p[1]) << 8) | p[2];
        if (key != lastKey) { // pixel art has long runs of one colour: the last answer is often right
            int bestD = 0x7FFFFFFF;
            size_t best = 0;
            for (size_t j = 0; j < count; ++j) {
                const int dr = static_cast<int>(p[0]) - static_cast<int>(palette[j * 3]);
                const int dg = static_cast<int>(p[1]) - static_cast<int>(palette[j * 3 + 1]);
                const int db = static_cast<int>(p[2]) - static_cast<int>(palette[j * 3 + 2]);
                const int d = dr * dr + dg * dg + db * db;
                if (d < bestD) { bestD = d; best = j; }
            }
            lastKey = key;
            lastOut[0] = palette[best * 3];
            lastOut[1] = palette[best * 3 + 1];
            lastOut[2] = palette[best * 3 + 2];
        }
        p[0] = lastOut[0];
        p[1] = lastOut[1];
        p[2] = lastOut[2];
    }
}

std::vector<uint8_t> EncodeBmp24(const uint8_t* rgb, uint32_t w, uint32_t h) {
    std::vector<uint8_t> bmp;
    if (!rgb || w == 0 || h == 0 || w > 16384 || h > 16384) return bmp;
    const uint32_t rowBytes = ((w * 3 + 3) / 4) * 4;
    const uint32_t imgSize = rowBytes * h;
    const uint32_t fileSize = 54 + imgSize;
    bmp.assign(fileSize, 0);
    auto put32 = [&](size_t at, uint32_t v) { for (int i = 0; i < 4; ++i) bmp[at + static_cast<size_t>(i)] = static_cast<uint8_t>(v >> (8 * i)); };
    auto put16 = [&](size_t at, uint16_t v) { bmp[at] = static_cast<uint8_t>(v); bmp[at + 1] = static_cast<uint8_t>(v >> 8); };
    bmp[0] = 'B';
    bmp[1] = 'M';
    put32(2, fileSize);
    put32(10, 54);
    put32(14, 40);
    put32(18, w);
    put32(22, h); // positive height: rows are stored bottom-up (the most widely supported form)
    put16(26, 1);
    put16(28, 24);
    put32(34, imgSize);
    put32(38, 2835);
    put32(42, 2835);
    for (uint32_t y = 0; y < h; ++y) {
        const uint8_t* src = rgb + static_cast<size_t>(h - 1 - y) * w * 3;
        uint8_t* dst = bmp.data() + 54 + static_cast<size_t>(y) * rowBytes;
        for (uint32_t x = 0; x < w; ++x) {
            dst[x * 3 + 0] = src[x * 3 + 2];
            dst[x * 3 + 1] = src[x * 3 + 1];
            dst[x * 3 + 2] = src[x * 3 + 0];
        }
    }
    return bmp;
}

std::vector<uint8_t> EncodePngRgb(const uint8_t* rgb, uint32_t w, uint32_t h) {
    std::vector<uint8_t> out;
    if (!rgb || w == 0 || h == 0 || w > 16384 || h > 16384) return out;
    // Raw scanlines, each prefixed by filter type 0 (None).
    const size_t rowLen = static_cast<size_t>(w) * 3 + 1;
    std::vector<uint8_t> raw(rowLen * h);
    for (uint32_t y = 0; y < h; ++y) {
        raw[static_cast<size_t>(y) * rowLen] = 0;
        std::memcpy(raw.data() + static_cast<size_t>(y) * rowLen + 1, rgb + static_cast<size_t>(y) * w * 3, static_cast<size_t>(w) * 3);
    }
    // zlib stream made of "stored" (uncompressed) deflate blocks: valid PNG, no compressor needed.
    std::vector<uint8_t> z;
    z.reserve(raw.size() + raw.size() / 65535 * 5 + 16);
    z.push_back(0x78);
    z.push_back(0x01);
    size_t pos = 0;
    do {
        const size_t n = std::min<size_t>(65535, raw.size() - pos);
        const bool last = (pos + n >= raw.size());
        z.push_back(last ? 1 : 0);
        z.push_back(static_cast<uint8_t>(n & 0xFF));
        z.push_back(static_cast<uint8_t>((n >> 8) & 0xFF));
        z.push_back(static_cast<uint8_t>(~n & 0xFF));
        z.push_back(static_cast<uint8_t>((~n >> 8) & 0xFF));
        z.insert(z.end(), raw.begin() + static_cast<std::ptrdiff_t>(pos), raw.begin() + static_cast<std::ptrdiff_t>(pos + n));
        pos += n;
    } while (pos < raw.size());
    uint32_t a = 1, b = 0;
    for (size_t i = 0; i < raw.size(); ++i) {
        a = (a + raw[i]) % 65521u;
        b = (b + a) % 65521u;
    }
    PutU32(z, (b << 16) | a);

    static const uint8_t sig[8] = { 0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A };
    out.insert(out.end(), sig, sig + 8);
    std::vector<uint8_t> ihdr;
    PutU32(ihdr, w);
    PutU32(ihdr, h);
    ihdr.push_back(8); // bit depth
    ihdr.push_back(2); // colour type: truecolour RGB
    ihdr.push_back(0);
    ihdr.push_back(0);
    ihdr.push_back(0);
    PutChunk(out, "IHDR", ihdr);
    PutChunk(out, "IDAT", z);
    PutChunk(out, "IEND", std::vector<uint8_t>());
    return out;
}

// ---------------------------------------------------------------------------------------------
// Wave form (Haar)
// ---------------------------------------------------------------------------------------------
bool HaarForward(const float* picture, uint32_t channels, uint32_t height, uint32_t width, float* wave) {
    if (!picture || !wave || channels == 0 || height < 2 || width < 2 || (height & 1u) != 0 || (width & 1u) != 0) return false;
    const size_t halfH = height / 2, halfW = width / 2;
    const size_t plane = halfH * halfW;
    for (uint32_t c = 0; c < channels; ++c) {
        const float* src = picture + static_cast<size_t>(c) * height * width;
        float* avg = wave + (0 * static_cast<size_t>(channels) + c) * plane;
        float* horiz = wave + (1 * static_cast<size_t>(channels) + c) * plane;
        float* vert = wave + (2 * static_cast<size_t>(channels) + c) * plane;
        float* diag = wave + (3 * static_cast<size_t>(channels) + c) * plane;
        for (size_t y = 0; y < halfH; ++y) {
            const float* top = src + (2 * y) * width;
            const float* bottom = top + width;
            for (size_t x = 0; x < halfW; ++x) {
                const float a = top[2 * x], b = top[2 * x + 1], cc = bottom[2 * x], d = bottom[2 * x + 1];
                const size_t i = y * halfW + x;
                avg[i] = 0.5f * (a + b + cc + d);
                horiz[i] = 0.5f * (a - b + cc - d);
                vert[i] = 0.5f * (a + b - cc - d);
                diag[i] = 0.5f * (a - b - cc + d);
            }
        }
    }
    return true;
}

bool HaarInverse(const float* wave, uint32_t channels, uint32_t height, uint32_t width, float* picture) {
    if (!picture || !wave || channels == 0 || height < 2 || width < 2 || (height & 1u) != 0 || (width & 1u) != 0) return false;
    const size_t halfH = height / 2, halfW = width / 2;
    const size_t plane = halfH * halfW;
    for (uint32_t c = 0; c < channels; ++c) {
        float* dst = picture + static_cast<size_t>(c) * height * width;
        const float* avg = wave + (0 * static_cast<size_t>(channels) + c) * plane;
        const float* horiz = wave + (1 * static_cast<size_t>(channels) + c) * plane;
        const float* vert = wave + (2 * static_cast<size_t>(channels) + c) * plane;
        const float* diag = wave + (3 * static_cast<size_t>(channels) + c) * plane;
        for (size_t y = 0; y < halfH; ++y) {
            float* top = dst + (2 * y) * width;
            float* bottom = top + width;
            for (size_t x = 0; x < halfW; ++x) {
                const size_t i = y * halfW + x;
                const float s = avg[i], h = horiz[i], v = vert[i], g = diag[i];
                top[2 * x] = 0.5f * (s + h + v + g);
                top[2 * x + 1] = 0.5f * (s - h + v - g);
                bottom[2 * x] = 0.5f * (s + h - v - g);
                bottom[2 * x + 1] = 0.5f * (s - h - v + g);
            }
        }
    }
    return true;
}

} // namespace XboxOTFM
