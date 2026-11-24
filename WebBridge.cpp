#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include "WebBridge.h"

#include <winsock2.h>
#include <ws2tcpip.h>
#include <iphlpapi.h>
#include <wincodec.h>
#include <wrl/client.h>
#include <winrt/Windows.Storage.h>

#include <algorithm>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <system_error>
#include <fstream>
#include <iterator>
#include <vector>

#pragma comment(lib, "ws2_32.lib")
#pragma comment(lib, "iphlpapi.lib")
#pragma comment(lib, "windowscodecs.lib")

namespace XboxOTFM {

namespace {

constexpr uintmax_t kCrashLogMaxBytes = 256u * 1024u;
constexpr uint32_t kDecodeMaxSide = 512;           // pictures are shrunk to this while they are decoded
constexpr uint64_t kDecodeMaxPixels = 120ull * 1000ull * 1000ull; // refuse absurd pictures instead of running out of memory

// Starts COM on the calling thread for the life of the object (the phone page's requests arrive on worker threads).
class ComScope {
public:
    ComScope() {
        const HRESULT hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
        m_needsUninit = (hr == S_OK || hr == S_FALSE);
    }
    ~ComScope() {
        if (m_needsUninit) CoUninitialize();
    }
    ComScope(const ComScope&) = delete;
    ComScope& operator=(const ComScope&) = delete;
private:
    bool m_needsUninit = false;
};

std::string WideToUtf8(const std::wstring& wide) {
    if (wide.empty()) return std::string();
    const int len = WideCharToMultiByte(CP_UTF8, 0, wide.c_str(), static_cast<int>(wide.size()), nullptr, 0, nullptr, nullptr);
    if (len <= 0) return std::string();
    std::string out(static_cast<size_t>(len), '\0');
    WideCharToMultiByte(CP_UTF8, 0, wide.c_str(), static_cast<int>(wide.size()), out.data(), len, nullptr, nullptr);
    return out;
}

std::wstring Utf8ToWide(const std::string& text) {
    if (text.empty()) return std::wstring();
    const int len = MultiByteToWideChar(CP_UTF8, 0, text.c_str(), static_cast<int>(text.size()), nullptr, 0);
    if (len <= 0) return std::wstring();
    std::wstring out(static_cast<size_t>(len), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, text.c_str(), static_cast<int>(text.size()), out.data(), len);
    return out;
}

void SetError(std::string* error, const char* text) {
    if (error) *error = text;
}

} // namespace

bool WebBridge::FreeBytes(const std::string& folder, uint64_t& freeBytes) {
    try {
        const std::wstring path = Utf8ToWide(folder.empty() ? std::string(".") : folder);
        if (path.empty()) return false;
        ULARGE_INTEGER availableToUs = {};
        ULARGE_INTEGER total = {};
        ULARGE_INTEGER totalFree = {};
        // "available to the caller" respects any limit the system puts on this app
        if (!GetDiskFreeSpaceExW(path.c_str(), &availableToUs, &total, &totalFree)) return false;
        freeBytes = availableToUs.QuadPart;
        return true;
    } catch (...) {
        return false;
    }
}

std::string WebBridge::LocalFolderPath() {
    try {
        const winrt::hstring folder = winrt::Windows::Storage::ApplicationData::Current().LocalFolder().Path();
        std::string path = WideToUtf8(std::wstring(folder.c_str(), folder.size()));
        while (!path.empty() && (path.back() == '\\' || path.back() == '/')) path.pop_back();
        if (!path.empty()) return path;
    } catch (...) {
    }
    return ".";
}

std::string WebBridge::CrashLogFilePath() {
    return LocalFolderPath() + "\\crash_log.txt";
}

void WebBridge::WriteCrashLogToDisk(const std::string& entry) {
    try {
        const std::string path = CrashLogFilePath();
        std::error_code ec;
        const uintmax_t size = std::filesystem::file_size(path, ec);
        if (!ec && size > kCrashLogMaxBytes) {
            std::error_code ec2;
            std::filesystem::remove(path, ec2);
        }
        std::ofstream out(path, std::ios::app);
        if (!out.is_open()) return;
        char stamp[64] = "";
        const std::time_t now = std::time(nullptr);
        std::tm utc = {};
        if (gmtime_s(&utc, &now) == 0) {
            std::snprintf(stamp, sizeof(stamp), "%04d-%02d-%02d %02d:%02d:%02d UTC  ",
                          utc.tm_year + 1900, utc.tm_mon + 1, utc.tm_mday, utc.tm_hour, utc.tm_min, utc.tm_sec);
        }
        out << stamp << entry << "\n";
        out.flush();
    } catch (...) {
    }
}

std::string WebBridge::ReadCrashLog(size_t maxBytes) {
    try {
        std::ifstream in(CrashLogFilePath(), std::ios::binary);
        if (!in.is_open()) return std::string("(nothing has been written to the crash log)\n");
        in.seekg(0, std::ios::end);
        const std::streamoff total = in.tellg();
        std::streamoff start = 0;
        if (total > static_cast<std::streamoff>(maxBytes)) start = total - static_cast<std::streamoff>(maxBytes);
        in.seekg(start, std::ios::beg);
        std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        if (text.empty()) return std::string("(the crash log is empty)\n");
        return text;
    } catch (...) {
        return std::string("(the crash log could not be read)\n");
    }
}

std::string WebBridge::DetectLanIPv4() {
    try {
        ULONG size = 16384;
        std::vector<uint8_t> buffer(size);
        ULONG rc = GetAdaptersAddresses(AF_INET, GAA_FLAG_SKIP_ANYCAST | GAA_FLAG_SKIP_MULTICAST, nullptr,
                                        reinterpret_cast<IP_ADAPTER_ADDRESSES*>(buffer.data()), &size);
        if (rc == ERROR_BUFFER_OVERFLOW) {
            buffer.resize(size);
            rc = GetAdaptersAddresses(AF_INET, GAA_FLAG_SKIP_ANYCAST | GAA_FLAG_SKIP_MULTICAST, nullptr,
                                      reinterpret_cast<IP_ADAPTER_ADDRESSES*>(buffer.data()), &size);
        }
        if (rc != ERROR_SUCCESS) return std::string();

        std::string best;
        int bestScore = 0;
        for (const IP_ADAPTER_ADDRESSES* adapter = reinterpret_cast<const IP_ADAPTER_ADDRESSES*>(buffer.data());
             adapter != nullptr; adapter = adapter->Next) {
            if (adapter->OperStatus != IfOperStatusUp) continue;
            if (adapter->IfType == IF_TYPE_SOFTWARE_LOOPBACK) continue;
            for (const IP_ADAPTER_UNICAST_ADDRESS* ua = adapter->FirstUnicastAddress; ua != nullptr; ua = ua->Next) {
                if (!ua->Address.lpSockaddr || ua->Address.lpSockaddr->sa_family != AF_INET) continue;
                const sockaddr_in* sin = reinterpret_cast<const sockaddr_in*>(ua->Address.lpSockaddr);
                const uint32_t ip = ntohl(sin->sin_addr.s_addr);
                const uint32_t a = (ip >> 24) & 255u;
                const uint32_t b = (ip >> 16) & 255u;
                if (a == 0 || a == 127) continue;            // not an address anyone else can reach
                if (a == 169 && b == 254) continue;          // "link-local": the network never gave this device an address
                if (a >= 224) continue;                      // multicast / reserved
                // an ordinary home network address is the best guess
                const bool home = a == 10 || (a == 172 && b >= 16 && b <= 31) || (a == 192 && b == 168);
                const int score = home ? 2 : 1;
                if (score > bestScore) {
                    char text[INET_ADDRSTRLEN] = {};
                    if (inet_ntop(AF_INET, const_cast<in_addr*>(&sin->sin_addr), text, sizeof(text))) {
                        best = text;
                        bestScore = score;
                    }
                }
            }
        }
        return best;
    } catch (...) {
        return std::string();
    }
}

bool WebBridge::DecodeImageFile(const uint8_t* bytes, size_t size, Rgb8Image& out, std::string* error) {
    out = Rgb8Image();
    if (!bytes || size == 0 || size > 0x7FFFFFFFu) {
        SetError(error, "The picture file is empty.");
        return false;
    }
    try {
        ComScope com;

        Microsoft::WRL::ComPtr<IWICImagingFactory> factory;
        HRESULT hr = CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&factory));
        if (FAILED(hr)) {
            SetError(error, "The Xbox's picture decoder could not be started.");
            return false;
        }

        Microsoft::WRL::ComPtr<IWICStream> stream;
        hr = factory->CreateStream(&stream);
        if (FAILED(hr)) {
            SetError(error, "The Xbox's picture decoder could not be started.");
            return false;
        }
        hr = stream->InitializeFromMemory(const_cast<BYTE*>(bytes), static_cast<DWORD>(size));
        if (FAILED(hr)) {
            SetError(error, "The picture file could not be read.");
            return false;
        }

        Microsoft::WRL::ComPtr<IWICBitmapDecoder> decoder;
        hr = factory->CreateDecoderFromStream(stream.Get(), nullptr, WICDecodeMetadataCacheOnDemand, &decoder);
        if (FAILED(hr)) {
            SetError(error, "This kind of picture file is not supported. PNG, JPEG, BMP and GIF work.");
            return false;
        }

        Microsoft::WRL::ComPtr<IWICBitmapFrameDecode> frame;
        hr = decoder->GetFrame(0, &frame);
        if (FAILED(hr)) {
            SetError(error, "The picture file could not be read.");
            return false;
        }

        UINT width = 0, height = 0;
        hr = frame->GetSize(&width, &height);
        if (FAILED(hr) || width == 0 || height == 0) {
            SetError(error, "The picture file could not be read.");
            return false;
        }
        if (static_cast<uint64_t>(width) * height > kDecodeMaxPixels) {
            SetError(error, "The picture is far too large. Make it smaller first.");
            return false;
        }

        // shrink while reading so that a 40 megapixel photo never has to exist in full
        UINT outW = width, outH = height;
        const UINT longest = std::max(width, height);
        if (longest > kDecodeMaxSide) {
            const double scale = static_cast<double>(kDecodeMaxSide) / static_cast<double>(longest);
            outW = std::max<UINT>(1u, static_cast<UINT>(static_cast<double>(width) * scale + 0.5));
            outH = std::max<UINT>(1u, static_cast<UINT>(static_cast<double>(height) * scale + 0.5));
        }

        Microsoft::WRL::ComPtr<IWICBitmapSource> source = frame;
        if (outW != width || outH != height) {
            Microsoft::WRL::ComPtr<IWICBitmapScaler> scaler;
            hr = factory->CreateBitmapScaler(&scaler);
            if (FAILED(hr)) {
                SetError(error, "The picture could not be shrunk.");
                return false;
            }
            hr = scaler->Initialize(frame.Get(), outW, outH, WICBitmapInterpolationModeFant);
            if (FAILED(hr)) {
                SetError(error, "The picture could not be shrunk.");
                return false;
            }
            source = scaler;
        }

        Microsoft::WRL::ComPtr<IWICFormatConverter> converter;
        hr = factory->CreateFormatConverter(&converter);
        if (FAILED(hr)) {
            SetError(error, "The picture colours could not be read.");
            return false;
        }
        hr = converter->Initialize(source.Get(), GUID_WICPixelFormat24bppRGB, WICBitmapDitherTypeNone, nullptr, 0.0,
                                   WICBitmapPaletteTypeCustom);
        if (FAILED(hr)) {
            SetError(error, "The picture colours could not be read.");
            return false;
        }

        const UINT stride = outW * 3u;
        std::vector<uint8_t> rgb(static_cast<size_t>(stride) * outH);
        hr = converter->CopyPixels(nullptr, stride, static_cast<UINT>(rgb.size()), rgb.data());
        if (FAILED(hr)) {
            SetError(error, "The picture could not be read.");
            return false;
        }

        out.w = outW;
        out.h = outH;
        out.px = std::move(rgb);
        return true;
    } catch (...) {
        out = Rgb8Image();
        SetError(error, "The picture could not be read (the Xbox ran out of memory or the file is damaged).");
        return false;
    }
}

} // namespace XboxOTFM
