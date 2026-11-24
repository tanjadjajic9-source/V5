#include "QrCode.h"

#include <algorithm>
#include <array>
#include <cstdlib>

namespace XboxOTFM {

namespace {

struct VersionInfo {
    int size;
    int totalCodewords;
    int eccPerBlock;
    int blocks;
    int alignCount;
    int align[2]; // alignment pattern centre coordinates (first is always 6)
    int remainderBits;
};

// Error correction level M. Every block of a version has the same length for versions 1 to 6.
constexpr VersionInfo kVersions[6] = {
    { 21, 26, 10, 1, 0, { 0, 0 }, 0 },
    { 25, 44, 16, 1, 2, { 6, 18 }, 7 },
    { 29, 70, 26, 1, 2, { 6, 22 }, 7 },
    { 33, 100, 18, 2, 2, { 6, 26 }, 7 },
    { 37, 134, 24, 2, 2, { 6, 30 }, 7 },
    { 41, 172, 16, 4, 2, { 6, 34 }, 7 }
};

uint8_t GfMul(uint8_t x, uint8_t y) {
    unsigned z = 0;
    for (int i = 7; i >= 0; --i) {
        z = (z << 1) ^ ((z >> 7) * 0x11Du);
        z ^= ((static_cast<unsigned>(y) >> i) & 1u) * x;
    }
    return static_cast<uint8_t>(z);
}

std::vector<uint8_t> RsDivisor(int degree) {
    std::vector<uint8_t> result(static_cast<size_t>(degree), 0);
    result[static_cast<size_t>(degree) - 1] = 1;
    uint8_t root = 1;
    for (int i = 0; i < degree; ++i) {
        for (size_t j = 0; j < result.size(); ++j) {
            result[j] = GfMul(result[j], root);
            if (j + 1 < result.size()) result[j] = static_cast<uint8_t>(result[j] ^ result[j + 1]);
        }
        root = GfMul(root, 0x02);
    }
    return result;
}

std::vector<uint8_t> RsRemainder(const std::vector<uint8_t>& data, const std::vector<uint8_t>& divisor) {
    std::vector<uint8_t> result(divisor.size(), 0);
    for (uint8_t b : data) {
        const uint8_t factor = static_cast<uint8_t>(b ^ result.front());
        result.erase(result.begin());
        result.push_back(0);
        for (size_t i = 0; i < divisor.size(); ++i) result[i] = static_cast<uint8_t>(result[i] ^ GfMul(divisor[i], factor));
    }
    return result;
}

class Grid {
public:
    explicit Grid(int size) : m_size(size), m_dark(static_cast<size_t>(size) * size, 0), m_fn(static_cast<size_t>(size) * size, 0) {}
    int Size() const { return m_size; }
    bool Dark(int x, int y) const { return m_dark[Idx(x, y)] != 0; }
    bool IsFunction(int x, int y) const { return m_fn[Idx(x, y)] != 0; }
    void SetFunction(int x, int y, bool dark) { m_dark[Idx(x, y)] = dark ? 1 : 0; m_fn[Idx(x, y)] = 1; }
    void SetData(int x, int y, bool dark) { m_dark[Idx(x, y)] = dark ? 1 : 0; }
    void Flip(int x, int y) { m_dark[Idx(x, y)] ^= 1; }
    const std::vector<uint8_t>& Modules() const { return m_dark; }
private:
    size_t Idx(int x, int y) const { return static_cast<size_t>(y) * static_cast<size_t>(m_size) + static_cast<size_t>(x); }
    int m_size;
    std::vector<uint8_t> m_dark;
    std::vector<uint8_t> m_fn;
};

void DrawFinder(Grid& g, int cx, int cy) {
    for (int dy = -4; dy <= 4; ++dy) {
        for (int dx = -4; dx <= 4; ++dx) {
            const int x = cx + dx, y = cy + dy;
            if (x < 0 || y < 0 || x >= g.Size() || y >= g.Size()) continue;
            const int dist = std::max(std::abs(dx), std::abs(dy));
            g.SetFunction(x, y, dist != 2 && dist != 4);
        }
    }
}

void DrawAlignment(Grid& g, int cx, int cy) {
    for (int dy = -2; dy <= 2; ++dy)
        for (int dx = -2; dx <= 2; ++dx)
            g.SetFunction(cx + dx, cy + dy, std::max(std::abs(dx), std::abs(dy)) != 1);
}

bool Bit(unsigned value, int index) { return ((value >> index) & 1u) != 0; }

void DrawFormat(Grid& g, int mask) {
    const unsigned data = (0u << 3) | static_cast<unsigned>(mask); // error correction level M has the indicator 00
    unsigned rem = data;
    for (int i = 0; i < 10; ++i) rem = (rem << 1) ^ ((rem >> 9) * 0x537u);
    const unsigned bits = ((data << 10) | rem) ^ 0x5412u;
    const int size = g.Size();
    for (int i = 0; i <= 5; ++i) g.SetFunction(8, i, Bit(bits, i));
    g.SetFunction(8, 7, Bit(bits, 6));
    g.SetFunction(8, 8, Bit(bits, 7));
    g.SetFunction(7, 8, Bit(bits, 8));
    for (int i = 9; i < 15; ++i) g.SetFunction(14 - i, 8, Bit(bits, i));
    for (int i = 0; i < 8; ++i) g.SetFunction(size - 1 - i, 8, Bit(bits, i));
    for (int i = 8; i < 15; ++i) g.SetFunction(8, size - 15 + i, Bit(bits, i));
    g.SetFunction(8, size - 8, true);
}

bool MaskHit(int mask, int x, int y) {
    switch (mask) {
        case 0: return (x + y) % 2 == 0;
        case 1: return y % 2 == 0;
        case 2: return x % 3 == 0;
        case 3: return (x + y) % 3 == 0;
        case 4: return (x / 3 + y / 2) % 2 == 0;
        case 5: return (x * y) % 2 + (x * y) % 3 == 0;
        case 6: return ((x * y) % 2 + (x * y) % 3) % 2 == 0;
        default: return ((x + y) % 2 + (x * y) % 3) % 2 == 0;
    }
}

int Penalty(const Grid& g) {
    const int n = g.Size();
    int score = 0;
    // runs of five or more of the same colour (rows, then columns)
    for (int pass = 0; pass < 2; ++pass) {
        for (int a = 0; a < n; ++a) {
            int run = 1;
            for (int b = 1; b < n; ++b) {
                const bool same = pass == 0 ? g.Dark(b, a) == g.Dark(b - 1, a) : g.Dark(a, b) == g.Dark(a, b - 1);
                if (same) {
                    ++run;
                } else {
                    if (run >= 5) score += 3 + (run - 5);
                    run = 1;
                }
            }
            if (run >= 5) score += 3 + (run - 5);
        }
    }
    // 2 x 2 blocks of one colour
    for (int y = 0; y + 1 < n; ++y)
        for (int x = 0; x + 1 < n; ++x) {
            const bool c = g.Dark(x, y);
            if (c == g.Dark(x + 1, y) && c == g.Dark(x, y + 1) && c == g.Dark(x + 1, y + 1)) score += 3;
        }
    // patterns that look like a finder pattern (dark light dark dark dark light dark) with four light modules on one side
    static const int kPatA[11] = { 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0 };
    static const int kPatB[11] = { 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1 };
    for (int pass = 0; pass < 2; ++pass) {
        for (int a = 0; a < n; ++a) {
            for (int b = 0; b + 11 <= n; ++b) {
                bool matchA = true, matchB = true;
                for (int k = 0; k < 11 && (matchA || matchB); ++k) {
                    const int v = pass == 0 ? (g.Dark(b + k, a) ? 1 : 0) : (g.Dark(a, b + k) ? 1 : 0);
                    if (v != kPatA[k]) matchA = false;
                    if (v != kPatB[k]) matchB = false;
                }
                if (matchA) score += 40;
                if (matchB) score += 40;
            }
        }
    }
    // balance of dark and light
    int dark = 0;
    for (uint8_t m : g.Modules()) dark += m;
    const int total = n * n;
    const int percentTimes2 = dark * 200 / total; // 2 x percent
    const int deviation = std::abs(percentTimes2 - 100); // 2 x |percent - 50|
    score += (deviation / 10) * 10;
    return score;
}

} // namespace

bool MakeQrCode(const std::string& text, QrMatrix& out, int forceMask) {
    out = QrMatrix();
    // pick the smallest version that holds the text
    int version = 0;
    for (int v = 0; v < 6; ++v) {
        const int dataCodewords = kVersions[v].totalCodewords - kVersions[v].eccPerBlock * kVersions[v].blocks;
        // 4 bits mode + 8 bits length + 8 bits per byte
        if ((4 + 8 + 8 * static_cast<int>(text.size()) + 7) / 8 <= dataCodewords) { version = v + 1; break; }
    }
    if (version == 0 || text.size() > 255) return false;
    const VersionInfo& vi = kVersions[version - 1];
    const int dataCodewords = vi.totalCodewords - vi.eccPerBlock * vi.blocks;

    // ---- data bits ----
    std::vector<uint8_t> bits; // one entry per bit
    auto put = [&](unsigned value, int count) { for (int i = count - 1; i >= 0; --i) bits.push_back(static_cast<uint8_t>((value >> i) & 1u)); };
    put(0x4, 4);                                  // byte mode
    put(static_cast<unsigned>(text.size()), 8);   // character count (8 bits for versions 1 to 9)
    for (unsigned char c : text) put(c, 8);
    const size_t capacityBits = static_cast<size_t>(dataCodewords) * 8;
    const size_t terminator = std::min<size_t>(4, capacityBits - bits.size());
    for (size_t i = 0; i < terminator; ++i) bits.push_back(0);
    while (bits.size() % 8 != 0) bits.push_back(0);
    std::vector<uint8_t> data;
    for (size_t i = 0; i < bits.size(); i += 8) {
        unsigned b = 0;
        for (size_t k = 0; k < 8; ++k) b = (b << 1) | bits[i + k];
        data.push_back(static_cast<uint8_t>(b));
    }
    for (uint8_t pad = 0xEC; data.size() < static_cast<size_t>(dataCodewords); pad = static_cast<uint8_t>(pad == 0xEC ? 0x11 : 0xEC)) data.push_back(pad);

    // ---- error correction, blocks and interleaving ----
    const int blockData = dataCodewords / vi.blocks;
    const std::vector<uint8_t> divisor = RsDivisor(vi.eccPerBlock);
    std::vector<std::vector<uint8_t>> dataBlocks, eccBlocks;
    for (int b = 0; b < vi.blocks; ++b) {
        std::vector<uint8_t> block(data.begin() + static_cast<long>(b) * blockData, data.begin() + static_cast<long>(b + 1) * blockData);
        eccBlocks.push_back(RsRemainder(block, divisor));
        dataBlocks.push_back(std::move(block));
    }
    std::vector<uint8_t> all;
    for (int i = 0; i < blockData; ++i)
        for (int b = 0; b < vi.blocks; ++b) all.push_back(dataBlocks[static_cast<size_t>(b)][static_cast<size_t>(i)]);
    for (int i = 0; i < vi.eccPerBlock; ++i)
        for (int b = 0; b < vi.blocks; ++b) all.push_back(eccBlocks[static_cast<size_t>(b)][static_cast<size_t>(i)]);

    // ---- fixed patterns ----
    const int size = vi.size;
    Grid g(size);
    for (int i = 0; i < size; ++i) {
        g.SetFunction(6, i, i % 2 == 0);
        g.SetFunction(i, 6, i % 2 == 0);
    }
    DrawFinder(g, 3, 3);
    DrawFinder(g, size - 4, 3);
    DrawFinder(g, 3, size - 4);
    if (vi.alignCount == 2) {
        const int pos[2] = { vi.align[0], vi.align[1] };
        for (int a = 0; a < 2; ++a)
            for (int b = 0; b < 2; ++b) {
                if (a == 0 && b == 0) continue; // overlaps the top-left finder
                if (a == 0 && b == 1) continue; // overlaps the top-right finder
                if (a == 1 && b == 0) continue; // overlaps the bottom-left finder
                DrawAlignment(g, pos[a], pos[b]);
            }
    }
    DrawFormat(g, 0); // reserves the format areas; rewritten below with the chosen mask

    // ---- place the data (zig-zag from the bottom right) ----
    size_t bitIndex = 0;
    for (int right = size - 1; right >= 1; right -= 2) {
        if (right == 6) right = 5;
        for (int vert = 0; vert < size; ++vert) {
            for (int j = 0; j < 2; ++j) {
                const int x = right - j;
                const bool upward = ((right + 1) & 2) == 0;
                const int y = upward ? size - 1 - vert : vert;
                if (!g.IsFunction(x, y) && bitIndex < all.size() * 8) {
                    g.SetData(x, y, ((all[bitIndex >> 3] >> (7 - (bitIndex & 7))) & 1) != 0);
                    ++bitIndex;
                }
            }
        }
    }

    // ---- masks ----
    auto applyMask = [&](Grid& target, int mask) {
        for (int y = 0; y < size; ++y)
            for (int x = 0; x < size; ++x)
                if (!target.IsFunction(x, y) && MaskHit(mask, x, y)) target.Flip(x, y);
    };
    int bestMask = 0, bestScore = 1 << 30;
    if (forceMask >= 0 && forceMask < 8) {
        bestMask = forceMask;
    } else {
        for (int m = 0; m < 8; ++m) {
            Grid trial = g;
            applyMask(trial, m);
            DrawFormat(trial, m);
            const int score = Penalty(trial);
            if (score < bestScore) { bestScore = score; bestMask = m; }
        }
    }
    applyMask(g, bestMask);
    DrawFormat(g, bestMask);

    out.size = size;
    out.version = version;
    out.mask = bestMask;
    out.dark = g.Modules();
    return true;
}

Rgb8Image RenderQrImage(const QrMatrix& qr, uint32_t pixelsPerModule) {
    Rgb8Image img;
    if (!qr.Valid() || pixelsPerModule == 0) return img;
    const uint32_t quiet = 4;
    const uint32_t modules = static_cast<uint32_t>(qr.size) + 2 * quiet;
    img.w = modules * pixelsPerModule;
    img.h = img.w;
    img.px.assign(static_cast<size_t>(img.w) * img.h * 3, 255);
    for (int my = 0; my < qr.size; ++my) {
        for (int mx = 0; mx < qr.size; ++mx) {
            if (!qr.At(mx, my)) continue;
            for (uint32_t dy = 0; dy < pixelsPerModule; ++dy) {
                const size_t row = static_cast<size_t>((static_cast<uint32_t>(my) + quiet) * pixelsPerModule + dy) * img.w;
                for (uint32_t dx = 0; dx < pixelsPerModule; ++dx) {
                    const size_t at = (row + (static_cast<uint32_t>(mx) + quiet) * pixelsPerModule + dx) * 3;
                    img.px[at] = 0;
                    img.px[at + 1] = 0;
                    img.px[at + 2] = 0;
                }
            }
        }
    }
    return img;
}

} // namespace XboxOTFM
