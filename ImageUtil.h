#pragma once
// Plain-C++ image helpers shared by the dataset, the phone web page and the TV screen.
// Everything works on 8-bit RGB (3 bytes per pixel, row after row, no padding).

#include <cstddef>
#include <cstdint>
#include <vector>

namespace XboxOTFM {

struct Rgb8Image {
    uint32_t w = 0;
    uint32_t h = 0;
    std::vector<uint8_t> px; // w * h * 3

    bool Valid() const { return w > 0 && h > 0 && px.size() == static_cast<size_t>(w) * h * 3; }
};

enum class FitMode : uint32_t {
    CropCenter = 0, // cut the largest centred square out of the picture (recommended)
    Squeeze = 1     // squash the whole picture into a square
};

// How a picture is resized when it is fitted into the square card.
enum class ResizeStyle : uint32_t {
    Smooth = 0, // neighbouring pixels are blended: right for photos, but it blurs small pixel art when that is enlarged
    Sharp = 1,  // every new pixel is an exact copy of one pixel of the original: no blur and no colour that was not there before
    Auto = 2    // Sharp for small pictures that use only a few colours (pixel art), Smooth for everything else
};

uint32_t Crc32(const uint8_t* data, size_t size, uint32_t previous = 0);
uint64_t Fnv1a64(const uint8_t* data, size_t size);

// Resamples the rectangle [cx, cx+cw) x [cy, cy+ch) (pixel units, may be fractional) of an RGB image to dstW x dstH.
// Shrinking averages every covered source pixel (no aliasing); enlarging interpolates. Output is float, 0..255, interleaved RGB.
// With `nearest` every output pixel is instead a copy of the one source pixel under its centre (nothing is blended).
// Returns false (and leaves `out` empty) for empty / absurd sizes.
bool ResampleRegion(const uint8_t* rgb, uint32_t w, uint32_t h,
                    double cx, double cy, double cw, double ch,
                    uint32_t dstW, uint32_t dstH, bool flipHorizontal, std::vector<float>& out, bool nearest = false);

// True when the pixels of the rectangle (x0, y0, rw, rh) use few colours for their number, like pixel art, icons and flat sprites,
// and not the thousands of slightly different colours of a photo: at most 256 different colours and at most one colour per
// four pixels. (The phone page uses the same rule.)
bool LooksLikePixelArt(const uint8_t* rgb, uint32_t w, uint32_t h, uint32_t x0, uint32_t y0, uint32_t rw, uint32_t rh);

Rgb8Image ResizeRgb(const uint8_t* rgb, uint32_t w, uint32_t h, uint32_t dstW, uint32_t dstH);
// Cuts or squashes the picture into a square. `style` says how pixels are made when it is enlarged or shrunk (see ResizeStyle).
Rgb8Image FitToSquare(const uint8_t* rgb, uint32_t w, uint32_t h, uint32_t size, FitMode mode,
                      ResizeStyle style = ResizeStyle::Smooth);

// Chooses at most maxColors colours (1..256) that stand for the given pixels (3 bytes each). When the pixels use no more than
// maxColors different colours, exactly those colours are returned, so nothing is invented; otherwise they are found with
// median cut and polished with a few rounds of k-means. Returns red, green, blue triples (empty when there are no pixels).
std::vector<uint8_t> MakePalette(const uint8_t* rgb, size_t pixelCount, uint32_t maxColors);
// Replaces every pixel with the nearest colour of the palette (red, green, blue triples).
void SnapToPalette(Rgb8Image& img, const std::vector<uint8_t>& palette);
Rgb8Image UpscaleNearest(const Rgb8Image& src, uint32_t factor);
Rgb8Image MakeGrid(const std::vector<Rgb8Image>& tiles, uint32_t columns, uint32_t pad, uint8_t bgR, uint8_t bgG, uint8_t bgB);
// [3, H, W] floats in [-1, 1] (what the network works with) -> 8-bit RGB
Rgb8Image FloatChwToRgb8(const float* chw, uint32_t w, uint32_t h);

// ---- wave form: one level of the orthonormal 2-D Haar transform ----
// Turns a picture [C, H, W] (H and W even) into its wave form [4C, H/2, W/2]: every 2x2 block of pixels becomes four numbers per
// colour - the average (so the first C channels are a half-size copy of the picture), the left-right difference, the up-down
// difference and the diagonal difference. Channels are ordered band by band: the C averages, then the C left-right differences,
// then the C up-down ones, then the C diagonal ones. Nothing is lost: HaarInverse gives the picture back (apart from rounding of
// about 1e-7). Because the transform is orthonormal, noise stays noise of the same strength and the distance between two pictures
// stays the same. Both return false (and write nothing) when a size is zero or odd. (`height` and `width` are those of the picture.)
bool HaarForward(const float* picture, uint32_t channels, uint32_t height, uint32_t width, float* wave);
bool HaarInverse(const float* wave, uint32_t channels, uint32_t height, uint32_t width, float* picture);

std::vector<uint8_t> EncodeBmp24(const uint8_t* rgb, uint32_t w, uint32_t h);
std::vector<uint8_t> EncodePngRgb(const uint8_t* rgb, uint32_t w, uint32_t h);

} // namespace XboxOTFM
