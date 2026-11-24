#pragma once
// The Xbox side of the TV screen: reads the controller and draws the list of shapes that TvModel (TvUi.h) produces.
// All decisions about what to show live in TvUi.cpp; this file only knows how to put shapes on the screen.

#include "TvUi.h"

#include <d3d12.h>
#include <d3d11on12.h>
#include <d2d1_3.h>
#include <dwrite_3.h>
#include <wrl/client.h>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace XboxOTFM {

class GamepadUI {
public:
    HRESULT Initialize(
        ID3D12Device* d3d12Device,
        ID3D12CommandQueue* cmdQueue,
        const std::vector<Microsoft::WRL::ComPtr<ID3D12Resource>>& backBuffers,
        uint32_t width,
        uint32_t height);

    void Release();

    // Reads every connected controller (Windows.Gaming.Input first, XInput as a back-up) and merges them into one PadState.
    PadState PollInput();

    // Draws one finished screen into the given back buffer.
    HRESULT Render(uint32_t backBufferIndex, const TvScene& scene);

private:
    struct CachedBitmap {
        Microsoft::WRL::ComPtr<ID2D1Bitmap> bitmap;
        uint32_t w = 0;
        uint32_t h = 0;
        uint64_t lastUsedFrame = 0;
    };

    ID2D1Bitmap* BitmapFor(const TvItem& item);
    void DrawItem(const TvItem& item);
    void DrawTextItem(const TvItem& item);
    void TrimBitmapCache();

    uint32_t m_width = 1920;
    uint32_t m_height = 1080;
    uint64_t m_frame = 0;

    Microsoft::WRL::ComPtr<ID3D11Device> m_d3d11Device;
    Microsoft::WRL::ComPtr<ID3D11DeviceContext> m_d3d11Context;
    Microsoft::WRL::ComPtr<ID3D11On12Device> m_d3d11On12Device;
    Microsoft::WRL::ComPtr<ID2D1Factory3> m_d2dFactory;
    Microsoft::WRL::ComPtr<ID2D1Device2> m_d2dDevice;
    Microsoft::WRL::ComPtr<ID2D1DeviceContext2> m_d2dContext;
    Microsoft::WRL::ComPtr<IDWriteFactory> m_dwriteFactory;

    Microsoft::WRL::ComPtr<IDWriteTextFormat> m_fonts[static_cast<size_t>(TvFont::Count)];
    Microsoft::WRL::ComPtr<IDWriteTextFormat> m_badgeFont;
    Microsoft::WRL::ComPtr<ID2D1SolidColorBrush> m_brushes[static_cast<size_t>(TvColor::Count)];
    Microsoft::WRL::ComPtr<ID2D1StrokeStyle> m_roundStroke;

    std::vector<Microsoft::WRL::ComPtr<ID3D11Resource>> m_wrappedBackBuffers;
    std::vector<Microsoft::WRL::ComPtr<ID2D1Bitmap1>> m_d2dRenderTargets;
    std::unordered_map<uint64_t, CachedBitmap> m_bitmaps;
};

} // namespace XboxOTFM
