#include "GamepadUI.h"

#include <winrt/Windows.Foundation.Collections.h>
#include <winrt/Windows.Gaming.Input.h>
#include <Xinput.h>
#include <dxgi1_4.h>

#include <algorithm>
#include <cmath>

namespace XboxOTFM {

namespace {

std::wstring ToWide(const std::string& utf8) {
    if (utf8.empty()) return std::wstring();
    const int len = MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), static_cast<int>(utf8.size()), nullptr, 0);
    if (len <= 0) return std::wstring();
    std::wstring out(static_cast<size_t>(len), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), static_cast<int>(utf8.size()), out.data(), len);
    return out;
}

D2D1_COLOR_F ColorFromArgb(uint32_t argb) {
    D2D1_COLOR_F c;
    c.a = static_cast<float>((argb >> 24) & 255u) / 255.0f;
    c.r = static_cast<float>((argb >> 16) & 255u) / 255.0f;
    c.g = static_cast<float>((argb >> 8) & 255u) / 255.0f;
    c.b = static_cast<float>(argb & 255u) / 255.0f;
    return c;
}

// the stick has to move well away from the middle before it counts
float StickValue(double raw) {
    const double dead = 0.2;
    if (std::abs(raw) < dead) return 0.0f;
    return static_cast<float>(std::clamp(raw, -1.0, 1.0));
}

struct FontSpec {
    const wchar_t* family;
    DWRITE_FONT_WEIGHT weight;
};

// same order as the TvFont list in TvUi.h: Display, Title, Heading, Body, BodyBold, Small
const FontSpec kFontSpecs[] = {
    { L"Segoe UI", DWRITE_FONT_WEIGHT_LIGHT },
    { L"Segoe UI", DWRITE_FONT_WEIGHT_SEMI_BOLD },
    { L"Segoe UI", DWRITE_FONT_WEIGHT_SEMI_BOLD },
    { L"Segoe UI", DWRITE_FONT_WEIGHT_NORMAL },
    { L"Segoe UI", DWRITE_FONT_WEIGHT_SEMI_BOLD },
    { L"Segoe UI", DWRITE_FONT_WEIGHT_NORMAL }
};
static_assert(sizeof(kFontSpecs) / sizeof(kFontSpecs[0]) == static_cast<size_t>(TvFont::Count), "one font description per TvFont");

} // namespace

// ---------------------------------------------------------------------------------------------
// Set-up
// ---------------------------------------------------------------------------------------------
HRESULT GamepadUI::Initialize(
    ID3D12Device* d3d12Device,
    ID3D12CommandQueue* cmdQueue,
    const std::vector<Microsoft::WRL::ComPtr<ID3D12Resource>>& backBuffers,
    uint32_t width,
    uint32_t height) {
    Release();
    m_width = width;
    m_height = height;

    UINT d3d11Flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;
#ifdef _DEBUG
    d3d11Flags |= D3D11_CREATE_DEVICE_DEBUG;
#endif

    IUnknown* queues[] = { cmdQueue };
    HRESULT hr = D3D11On12CreateDevice(
        d3d12Device,
        d3d11Flags,
        nullptr,
        0,
        queues,
        1,
        0,
        &m_d3d11Device,
        &m_d3d11Context,
        nullptr);
    if (FAILED(hr)) return hr;

    hr = m_d3d11Device.As(&m_d3d11On12Device);
    if (FAILED(hr)) return hr;

    D2D1_FACTORY_OPTIONS d2dOpts = {};
    hr = D2D1CreateFactory(D2D1_FACTORY_TYPE_SINGLE_THREADED, __uuidof(ID2D1Factory3), &d2dOpts, &m_d2dFactory);
    if (FAILED(hr)) return hr;

    // Query IDXGIDevice directly from the underlying D3D11 device (NOT ID3D11On12Device).
    Microsoft::WRL::ComPtr<IDXGIDevice> dxgiDevice;
    hr = m_d3d11Device.As(&dxgiDevice);
    if (FAILED(hr)) return hr;

    hr = m_d2dFactory->CreateDevice(dxgiDevice.Get(), &m_d2dDevice);
    if (FAILED(hr)) return hr;

    hr = m_d2dDevice->CreateDeviceContext(D2D1_DEVICE_CONTEXT_OPTIONS_NONE, &m_d2dContext);
    if (FAILED(hr)) return hr;

    hr = DWriteCreateFactory(DWRITE_FACTORY_TYPE_SHARED, __uuidof(IDWriteFactory), reinterpret_cast<IUnknown**>(m_dwriteFactory.GetAddressOf()));
    if (FAILED(hr)) return hr;

    // Fonts are made at their size on the 1920 x 1080 page; the whole page is scaled to the real screen when drawing.
    for (size_t i = 0; i < static_cast<size_t>(TvFont::Count); ++i) {
        hr = m_dwriteFactory->CreateTextFormat(kFontSpecs[i].family, nullptr, kFontSpecs[i].weight, DWRITE_FONT_STYLE_NORMAL,
                                               DWRITE_FONT_STRETCH_NORMAL, TvFontSize(static_cast<TvFont>(i)), L"en-US", &m_fonts[i]);
        if (FAILED(hr)) return hr;
    }
    hr = m_dwriteFactory->CreateTextFormat(L"Segoe UI", nullptr, DWRITE_FONT_WEIGHT_SEMI_BOLD, DWRITE_FONT_STYLE_NORMAL,
                                           DWRITE_FONT_STRETCH_NORMAL, 26.0f, L"en-US", &m_badgeFont);
    if (FAILED(hr)) return hr;
    m_badgeFont->SetTextAlignment(DWRITE_TEXT_ALIGNMENT_CENTER);
    m_badgeFont->SetParagraphAlignment(DWRITE_PARAGRAPH_ALIGNMENT_CENTER);
    m_badgeFont->SetWordWrapping(DWRITE_WORD_WRAPPING_NO_WRAP);

    for (size_t i = 0; i < static_cast<size_t>(TvColor::Count); ++i) {
        const D2D1_COLOR_F color = ColorFromArgb(TvPaletteArgb(static_cast<TvColor>(i)));
        hr = m_d2dContext->CreateSolidColorBrush(color, m_brushes[i].GetAddressOf());
        if (FAILED(hr)) return hr;
    }

    // round ends and corners for the chart line (if this cannot be made, plain lines are used)
    {
        D2D1_STROKE_STYLE_PROPERTIES strokeProps = {};
        strokeProps.startCap = D2D1_CAP_STYLE_ROUND;
        strokeProps.endCap = D2D1_CAP_STYLE_ROUND;
        strokeProps.dashCap = D2D1_CAP_STYLE_ROUND;
        strokeProps.lineJoin = D2D1_LINE_JOIN_ROUND;
        strokeProps.miterLimit = 10.0f;
        strokeProps.dashStyle = D2D1_DASH_STYLE_SOLID;
        strokeProps.dashOffset = 0.0f;
        ID2D1Factory* baseFactory = m_d2dFactory.Get(); // the plain factory interface has exactly one CreateStrokeStyle
        baseFactory->CreateStrokeStyle(&strokeProps, nullptr, 0, m_roundStroke.GetAddressOf());
    }

    const float dpi = 96.0f;
    D2D1_BITMAP_PROPERTIES1 bitmapProps = D2D1::BitmapProperties1(
        D2D1_BITMAP_OPTIONS_TARGET | D2D1_BITMAP_OPTIONS_CANNOT_DRAW,
        D2D1::PixelFormat(DXGI_FORMAT_B8G8R8A8_UNORM, D2D1_ALPHA_MODE_PREMULTIPLIED),
        dpi,
        dpi);

    m_wrappedBackBuffers.resize(backBuffers.size());
    m_d2dRenderTargets.resize(backBuffers.size());
    for (size_t i = 0; i < backBuffers.size(); ++i) {
        D3D11_RESOURCE_FLAGS resFlags = {};
        resFlags.BindFlags = D3D11_BIND_RENDER_TARGET;
        hr = m_d3d11On12Device->CreateWrappedResource(
            backBuffers[i].Get(),
            &resFlags,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
            D3D12_RESOURCE_STATE_PRESENT,
            IID_PPV_ARGS(&m_wrappedBackBuffers[i]));
        if (FAILED(hr)) return hr;

        Microsoft::WRL::ComPtr<IDXGISurface> surface;
        hr = m_wrappedBackBuffers[i].As(&surface);
        if (FAILED(hr)) return hr;

        hr = m_d2dContext->CreateBitmapFromDxgiSurface(surface.Get(), &bitmapProps, &m_d2dRenderTargets[i]);
        if (FAILED(hr)) return hr;
    }

    return S_OK;
}

void GamepadUI::Release() {
    m_bitmaps.clear();
    m_d2dRenderTargets.clear();
    m_wrappedBackBuffers.clear();
    m_roundStroke.Reset();
    for (auto& brush : m_brushes) brush.Reset();
    m_badgeFont.Reset();
    for (auto& font : m_fonts) font.Reset();
    m_dwriteFactory.Reset();
    m_d2dContext.Reset();
    m_d2dDevice.Reset();
    m_d2dFactory.Reset();
    m_d3d11On12Device.Reset();
    m_d3d11Context.Reset();
    m_d3d11Device.Reset();
}

// ---------------------------------------------------------------------------------------------
// Controller
// ---------------------------------------------------------------------------------------------
PadState GamepadUI::PollInput() {
    PadState pad;
    uint32_t buttons = 0;
    double stickX = 0.0;
    double stickY = 0.0;
    bool found = false;

    try {
        auto gamepads = winrt::Windows::Gaming::Input::Gamepad::Gamepads();
        using WGI = winrt::Windows::Gaming::Input::GamepadButtons;
        const uint32_t count = gamepads.Size();
        for (uint32_t i = 0; i < count && i < 8; ++i) {
            const auto reading = gamepads.GetAt(i).GetCurrentReading();
            found = true;
            const auto btns = reading.Buttons;
            if ((btns & WGI::A) == WGI::A) buttons |= kPadA;
            if ((btns & WGI::B) == WGI::B) buttons |= kPadB;
            if ((btns & WGI::X) == WGI::X) buttons |= kPadX;
            if ((btns & WGI::Y) == WGI::Y) buttons |= kPadY;
            if ((btns & WGI::DPadUp) == WGI::DPadUp) buttons |= kPadUp;
            if ((btns & WGI::DPadDown) == WGI::DPadDown) buttons |= kPadDown;
            if ((btns & WGI::DPadLeft) == WGI::DPadLeft) buttons |= kPadLeft;
            if ((btns & WGI::DPadRight) == WGI::DPadRight) buttons |= kPadRight;
            if ((btns & WGI::LeftShoulder) == WGI::LeftShoulder) buttons |= kPadLB;
            if ((btns & WGI::RightShoulder) == WGI::RightShoulder) buttons |= kPadRB;
            if ((btns & WGI::View) == WGI::View) buttons |= kPadView;
            if ((btns & WGI::Menu) == WGI::Menu) buttons |= kPadMenu;
            if (std::abs(reading.LeftThumbstickX) > std::abs(stickX)) stickX = reading.LeftThumbstickX;
            if (std::abs(reading.LeftThumbstickY) > std::abs(stickY)) stickY = reading.LeftThumbstickY;
        }
    } catch (...) {
        // XInput below is the back-up
    }

    if (!found) {
        for (DWORD index = 0; index < 4; ++index) {
            XINPUT_STATE state = {};
            if (XInputGetState(index, &state) != ERROR_SUCCESS) continue;
            found = true;
            const WORD b = state.Gamepad.wButtons;
            if (b & XINPUT_GAMEPAD_A) buttons |= kPadA;
            if (b & XINPUT_GAMEPAD_B) buttons |= kPadB;
            if (b & XINPUT_GAMEPAD_X) buttons |= kPadX;
            if (b & XINPUT_GAMEPAD_Y) buttons |= kPadY;
            if (b & XINPUT_GAMEPAD_DPAD_UP) buttons |= kPadUp;
            if (b & XINPUT_GAMEPAD_DPAD_DOWN) buttons |= kPadDown;
            if (b & XINPUT_GAMEPAD_DPAD_LEFT) buttons |= kPadLeft;
            if (b & XINPUT_GAMEPAD_DPAD_RIGHT) buttons |= kPadRight;
            if (b & XINPUT_GAMEPAD_LEFT_SHOULDER) buttons |= kPadLB;
            if (b & XINPUT_GAMEPAD_RIGHT_SHOULDER) buttons |= kPadRB;
            if (b & XINPUT_GAMEPAD_BACK) buttons |= kPadView;
            if (b & XINPUT_GAMEPAD_START) buttons |= kPadMenu;
            const double x = static_cast<double>(state.Gamepad.sThumbLX) / 32767.0;
            const double y = static_cast<double>(state.Gamepad.sThumbLY) / 32767.0;
            if (std::abs(x) > std::abs(stickX)) stickX = x;
            if (std::abs(y) > std::abs(stickY)) stickY = y;
        }
    }

    pad.buttons = buttons;
    pad.stickX = StickValue(stickX);
    pad.stickY = StickValue(stickY);
    pad.connected = found;
    return pad;
}

// ---------------------------------------------------------------------------------------------
// Drawing
// ---------------------------------------------------------------------------------------------
ID2D1Bitmap* GamepadUI::BitmapFor(const TvItem& item) {
    if (!item.image || !item.image->Valid()) return nullptr;
    const Rgb8Image& img = *item.image;

    auto found = m_bitmaps.find(item.imageKey);
    if (found != m_bitmaps.end() && found->second.bitmap && found->second.w == img.w && found->second.h == img.h) {
        found->second.lastUsedFrame = m_frame;
        return found->second.bitmap.Get();
    }

    const size_t count = static_cast<size_t>(img.w) * img.h;
    std::vector<uint8_t> bgra(count * 4);
    for (size_t i = 0; i < count; ++i) {
        bgra[i * 4 + 0] = img.px[i * 3 + 2];
        bgra[i * 4 + 1] = img.px[i * 3 + 1];
        bgra[i * 4 + 2] = img.px[i * 3 + 0];
        bgra[i * 4 + 3] = 255;
    }

    D2D1_BITMAP_PROPERTIES props = D2D1::BitmapProperties(
        D2D1::PixelFormat(DXGI_FORMAT_B8G8R8A8_UNORM, D2D1_ALPHA_MODE_IGNORE));
    CachedBitmap entry;
    const HRESULT hr = m_d2dContext->CreateBitmap(D2D1::SizeU(img.w, img.h), bgra.data(), img.w * 4, &props, &entry.bitmap);
    if (FAILED(hr) || !entry.bitmap) return nullptr;
    entry.w = img.w;
    entry.h = img.h;
    entry.lastUsedFrame = m_frame;
    ID2D1Bitmap* raw = entry.bitmap.Get();
    m_bitmaps[item.imageKey] = std::move(entry);
    return raw;
}

void GamepadUI::TrimBitmapCache() {
    for (auto it = m_bitmaps.begin(); it != m_bitmaps.end();) {
        if (m_frame - it->second.lastUsedFrame > 240) it = m_bitmaps.erase(it);
        else ++it;
    }
}

void GamepadUI::DrawTextItem(const TvItem& item) {
    if (item.text.empty() || item.w <= 0.0f || item.h <= 0.0f) return;
    const size_t fontIndex = static_cast<size_t>(item.font);
    const size_t colorIndex = static_cast<size_t>(item.color);
    if (fontIndex >= static_cast<size_t>(TvFont::Count) || colorIndex >= static_cast<size_t>(TvColor::Count)) return;
    IDWriteTextFormat* format = m_fonts[fontIndex].Get();
    ID2D1SolidColorBrush* brush = m_brushes[colorIndex].Get();
    if (!format || !brush) return;

    DWRITE_TEXT_ALIGNMENT alignment = DWRITE_TEXT_ALIGNMENT_LEADING;
    if (item.align == TvAlign::Center) alignment = DWRITE_TEXT_ALIGNMENT_CENTER;
    else if (item.align == TvAlign::Right) alignment = DWRITE_TEXT_ALIGNMENT_TRAILING;
    format->SetTextAlignment(alignment);
    format->SetWordWrapping(item.wrap ? DWRITE_WORD_WRAPPING_WRAP : DWRITE_WORD_WRAPPING_NO_WRAP);

    const std::wstring wide = ToWide(item.text);
    if (wide.empty()) return;
    const D2D1_RECT_F rect = D2D1::RectF(item.x, item.y, item.x + item.w, item.y + item.h);
    m_d2dContext->DrawText(wide.c_str(), static_cast<UINT32>(wide.size()), format, rect, brush, D2D1_DRAW_TEXT_OPTIONS_CLIP);
}

void GamepadUI::DrawItem(const TvItem& item) {
    const size_t colorIndex = static_cast<size_t>(item.color);
    if (colorIndex >= static_cast<size_t>(TvColor::Count)) return;
    ID2D1SolidColorBrush* brush = m_brushes[colorIndex].Get();
    if (!brush) return;
    const D2D1_RECT_F rect = D2D1::RectF(item.x, item.y, item.x + item.w, item.y + item.h);

    switch (item.kind) {
        case TvItem::Kind::Fill:
            if (item.radius > 0.0f) m_d2dContext->FillRoundedRectangle(D2D1::RoundedRect(rect, item.radius, item.radius), brush);
            else m_d2dContext->FillRectangle(rect, brush);
            break;
        case TvItem::Kind::Outline:
            if (item.radius > 0.0f) m_d2dContext->DrawRoundedRectangle(D2D1::RoundedRect(rect, item.radius, item.radius), brush, item.thickness);
            else m_d2dContext->DrawRectangle(rect, brush, item.thickness);
            break;
        case TvItem::Kind::Text:
            DrawTextItem(item);
            break;
        case TvItem::Kind::Lines:
            for (size_t i = 0; i + 3 < item.points.size(); i += 2) {
                m_d2dContext->DrawLine(D2D1::Point2F(item.points[i], item.points[i + 1]),
                                       D2D1::Point2F(item.points[i + 2], item.points[i + 3]),
                                       brush, item.thickness, m_roundStroke.Get());
            }
            break;
        case TvItem::Kind::Image: {
            ID2D1Bitmap* bitmap = BitmapFor(item);
            if (bitmap) {
                m_d2dContext->DrawBitmap(bitmap, rect, 1.0f,
                                         item.smooth ? D2D1_BITMAP_INTERPOLATION_MODE_LINEAR : D2D1_BITMAP_INTERPOLATION_MODE_NEAREST_NEIGHBOR);
            }
            break;
        }
        case TvItem::Kind::Badge: {
            const float radius = item.w * 0.5f;
            m_d2dContext->FillEllipse(D2D1::Ellipse(D2D1::Point2F(item.x + radius, item.y + radius), radius, radius), brush);
            ID2D1SolidColorBrush* ink = m_brushes[static_cast<size_t>(TvColor::OnAccent)].Get();
            const std::wstring letter = ToWide(item.text);
            if (ink && m_badgeFont && !letter.empty()) {
                m_d2dContext->DrawText(letter.c_str(), static_cast<UINT32>(letter.size()), m_badgeFont.Get(), rect, ink);
            }
            break;
        }
    }
}

HRESULT GamepadUI::Render(uint32_t backBufferIndex, const TvScene& scene) {
    if (backBufferIndex >= m_wrappedBackBuffers.size() || backBufferIndex >= m_d2dRenderTargets.size() || !m_d2dContext) {
        return E_INVALIDARG;
    }
    ++m_frame;

    ID3D11Resource* wrapped = m_wrappedBackBuffers[backBufferIndex].Get();
    m_d3d11On12Device->AcquireWrappedResources(&wrapped, 1);

    m_d2dContext->SetTarget(m_d2dRenderTargets[backBufferIndex].Get());
    m_d2dContext->BeginDraw();

    const D2D1_COLOR_F background = ColorFromArgb(TvPaletteArgb(TvColor::Background));
    m_d2dContext->Clear(background); // also paints the bars if the screen is not exactly 16:9

    // the page is 1920 x 1080; fit it into the real screen and keep it centred
    const float screenW = static_cast<float>(m_width);
    const float screenH = static_cast<float>(m_height);
    const float scale = std::min(screenW / kTvWidth, screenH / kTvHeight);
    const float offsetX = (screenW - kTvWidth * scale) * 0.5f;
    const float offsetY = (screenH - kTvHeight * scale) * 0.5f;
    m_d2dContext->SetTransform(D2D1::Matrix3x2F::Scale(scale, scale) * D2D1::Matrix3x2F::Translation(offsetX, offsetY));

    for (const TvItem& item : scene.items) DrawItem(item);

    m_d2dContext->SetTransform(D2D1::Matrix3x2F::Identity());
    const HRESULT hr = m_d2dContext->EndDraw();

    m_d3d11On12Device->ReleaseWrappedResources(&wrapped, 1);
    m_d3d11Context->Flush();
    TrimBitmapCache();
    return hr;
}

} // namespace XboxOTFM
