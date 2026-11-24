#pragma once
// Makes a QR code for a short text (the phone address shown on the TV). Plain C++, no Windows code.
// Supports up to 106 bytes of text (QR versions 1 to 6, error correction level M), which is far more than a web address needs.

#include "ImageUtil.h"

#include <cstdint>
#include <string>
#include <vector>

namespace XboxOTFM {

struct QrMatrix {
    int size = 0;                 // modules per side (21 for version 1, 4 more for each further version)
    std::vector<uint8_t> dark;    // size * size, row after row, 1 = dark module
    int version = 0;
    int mask = 0;

    bool Valid() const { return size > 0 && dark.size() == static_cast<size_t>(size) * static_cast<size_t>(size); }
    bool At(int x, int y) const { return dark[static_cast<size_t>(y) * static_cast<size_t>(size) + static_cast<size_t>(x)] != 0; }
};

// forceMask: -1 picks the best of the 8 mask patterns (normal use); 0..7 forces one (tests).
bool MakeQrCode(const std::string& text, QrMatrix& out, int forceMask = -1);

// Black modules on white with a 4-module quiet zone, `pixelsPerModule` pixels per module.
Rgb8Image RenderQrImage(const QrMatrix& qr, uint32_t pixelsPerModule);

} // namespace XboxOTFM
