#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <winrt/Windows.ApplicationModel.h>
#include <winrt/Windows.ApplicationModel.Core.h>
#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.System.h>
#include <winrt/Windows.UI.Core.h>

#include "AppCore.h"
#include "GamepadUI.h"
#include "TvUi.h"
#include "WebBridge.h"
#include "WebHost.h"
#include "WebUiAssets.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <exception>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")

using namespace XboxOTFM;
using namespace winrt::Windows::ApplicationModel::Core;
using namespace winrt::Windows::UI::Core;

namespace {

constexpr uint32_t kFrameCount = 2;
constexpr uint16_t kWebPort = 8080;

const char* FeatureLevelName(D3D_FEATURE_LEVEL fl) {
    switch (fl) {
        case D3D_FEATURE_LEVEL_12_0: return "D3D 12.0";
        case D3D_FEATURE_LEVEL_11_1: return "D3D 11.1";
        case D3D_FEATURE_LEVEL_11_0: return "D3D 11.0";
        default: return "D3D";
    }
}

std::string WideToAscii(const wchar_t* text) {
    std::string out;
    for (; text && *text; ++text) out.push_back(*text < 128 && *text >= 32 ? static_cast<char>(*text) : '?');
    return out;
}

std::string HexText(unsigned long value) {
    std::ostringstream oss;
    oss << "0x" << std::hex << value;
    return oss.str();
}

// "1.0.12.1": the version the build gave the app (each build counts up)
std::string PackageVersionText() {
    try {
        const auto v = winrt::Windows::ApplicationModel::Package::Current().Id().Version();
        return std::to_string(v.Major) + "." + std::to_string(v.Minor) + "." + std::to_string(v.Build) + "." + std::to_string(v.Revision);
    } catch (...) {
        return "1.0";
    }
}

// How much memory the console lets the whole app use (about 1 GB normally, much more with "Treat UWP apps as games"); 0 = unknown.
uint64_t AppMemoryLimitBytes() {
    try {
        return winrt::Windows::System::MemoryManager::AppMemoryUsageLimit();
    } catch (...) {
        return 0;
    }
}

} // namespace

class XboxOTFMApp : public winrt::implements<XboxOTFMApp, IFrameworkViewSource, IFrameworkView> {
public:
    IFrameworkView CreateView() {
        return *this;
    }

    void Initialize(CoreApplicationView const&) {
        std::set_terminate([]() {
            WebBridge::WriteCrashLogToDisk("[std::terminate] Unhandled C++ exception terminated process.");
            std::abort();
        });

        // The Xbox can put the app to sleep at any moment (for example when someone opens another app). Save first.
        m_suspendToken = CoreApplication::Suspending([this](auto&&, auto&& args) {
            auto deferral = args.SuspendingOperation().GetDeferral();
            try {
                if (m_core) {
                    m_wasTrainingAtSuspend = m_core->IsTrainingWanted();
                    m_core->PrepareForSuspend(3000);
                }
            } catch (...) {
            }
            deferral.Complete();
        });
        m_resumeToken = CoreApplication::Resuming([this](auto&&, auto&&) {
            m_resumed.store(true);
        });
    }

    void SetWindow(CoreWindow const& window) {
        m_window = window;
        const auto bounds = window.Bounds();
        if (bounds.Width >= 640.0f && bounds.Height >= 360.0f) {
            m_width = static_cast<uint32_t>(bounds.Width);
            m_height = static_cast<uint32_t>(bounds.Height);
        }
        window.Closed([this](auto&&, auto&&) { m_running = false; });
        window.SizeChanged([this](auto&&, WindowSizeChangedEventArgs const& args) {
            m_width = std::max(640u, static_cast<uint32_t>(args.Size().Width));
            m_height = std::max(360u, static_cast<uint32_t>(args.Size().Height));
            m_resizePending = true;
        });
    }

    void Load(winrt::hstring const&) {}

    void Run() {
        m_window.Activate();
        try {
            RunApp();
        } catch (const std::exception& ex) {
            WebBridge::WriteCrashLogToDisk(std::string("[Fatal] ") + ex.what());
        } catch (...) {
            WebBridge::WriteCrashLogToDisk("[Fatal] Unknown exception ended the app.");
        }
        ShutDown();
    }

    void Uninitialize() {}

private:
    // ------------------------------------------------------------------------------------------
    // Start-up and the main loop
    // ------------------------------------------------------------------------------------------
    void RunApp() {
        m_core = std::make_unique<AppCore>(WebBridge::LocalFolderPath() + "\\data", PackageVersionText());
        m_core->SetExtraLogReader([]() { return WebBridge::ReadCrashLog(); });
        // the picture set refuses new pictures before the storage is full (it asks the system how much room is left)
        m_core->SetFreeSpaceReader([](const std::string& folder, uint64_t& freeBytes) { return WebBridge::FreeBytes(folder, freeBytes); });

        // The graphics memory for the pictures of one lesson follows the allowance the console gives the app (a bigger allowance
        // lets bigger lessons run); with the usual small allowance, or when it cannot be read, the careful default stays.
        {
            const uint64_t limit = AppMemoryLimitBytes();
            FMBrain::SetActivationBudgetBytes(FMBrain::ActivationBudgetForAppLimit(limit));
            m_core->AppendLog("[Memory] The console allows the app about " + std::to_string(limit >> 20) + " MB; lessons may use up to " +
                              std::to_string(FMBrain::ActivationBudgetBytes() >> 20) + " MB of graphics memory for their pictures.");
        }

        const HRESULT gfx = InitializeD3D12();
        m_gfxOk = SUCCEEDED(gfx);
        if (!m_gfxOk) {
            const std::string text = "The screen could not be started (error " + HexText(static_cast<unsigned long>(gfx)) + "). The phone page still works.";
            WebBridge::WriteCrashLogToDisk("[Graphics] " + text);
            m_core->AppendLog("[Graphics] " + text);
        }

        // Every stored picture is checked when the app starts, so a very big set takes a little while: say so.
        std::string loadingLine = "Loading your pictures and saves...";
        {
            std::ifstream pictureFile(WebBridge::LocalFolderPath() + "\\data\\pictures\\cards.otds", std::ios::binary | std::ios::ate);
            if (pictureFile && static_cast<long long>(pictureFile.tellg()) > 1024ll * 1024ll * 1024ll) {
                loadingLine = "Checking your pictures. A big set can take a minute...";
            }
        }
        ShowSplash(loadingLine.c_str());
        m_core->Start();

        StartWebPage();

        ShowSplash("Starting the graphics chip...");
        AttachGpuNow();

        m_tv = std::make_unique<TvModel>(*m_core);
        RefreshLanAddress(true);

        auto last = std::chrono::steady_clock::now();
        double sinceRender = 1.0e9;
        double sinceLan = 0.0;
        while (m_running) {
            m_window.Dispatcher().ProcessEvents(CoreProcessEventsOption::ProcessAllIfPresent);
            if (!m_running) break;

            const auto now = std::chrono::steady_clock::now();
            double dt = std::chrono::duration<double>(now - last).count();
            last = now;
            if (dt > 0.5) dt = 0.5;

            if (m_resumed.exchange(false)) HandleResume();
            if (m_resizePending) {
                m_resizePending = false;
                RebuildSwapChain();
            }
            sinceLan += dt;
            if (sinceLan >= 5.0) {
                sinceLan = 0.0;
                RefreshLanAddress(false);
            }

            RunGuardedFrame(dt, sinceRender);
        }
    }

    void ShutDown() {
        try {
            CoreApplication::Suspending(m_suspendToken);
            CoreApplication::Resuming(m_resumeToken);
        } catch (...) {
        }
        try {
            // The core goes first: it saves the progress and releases anything on the phone page that is still waiting for it,
            // so stopping the page afterwards cannot get stuck.
            if (m_core) m_core->Stop();
            m_tv.reset();
            if (m_web) m_web->Stop();
            WaitForGpu();
            if (m_core) m_core->DetachGpu(false);
            ReleaseGraphics();
            m_web.reset();
            m_core.reset();
        } catch (const std::exception& ex) {
            WebBridge::WriteCrashLogToDisk(std::string("[Shutdown] ") + ex.what());
        } catch (...) {
            WebBridge::WriteCrashLogToDisk("[Shutdown] Unknown exception.");
        }
    }

    void StartWebPage() {
        WebApiHooks hooks;
        hooks.decodeImageFile = [](const uint8_t* bytes, size_t size, Rgb8Image& out, std::string* error) {
            return WebBridge::DecodeImageFile(bytes, size, out, error);
        };
        m_web = std::make_unique<WebHost>(*m_core, hooks, kWebIndexHtml, kWebIndexHtmlSize);
        std::string error;
        if (!m_web->Start(kWebPort, &error)) {
            WebBridge::WriteCrashLogToDisk("[Web] The phone page could not start: " + error);
            m_core->AppendLog("[Web] The phone page could not start: " + error);
            m_core->PushToast("error", "The phone page could not start: " + error);
        }
    }

    void RefreshLanAddress(bool force) {
        if (!m_core) return;
        std::string url;
        if (m_web && m_web->Running()) {
            const std::string ip = WebBridge::DetectLanIPv4();
            if (!ip.empty()) url = "http://" + ip + ":" + std::to_string(m_web->Port());
        }
        if (force || url != m_lanUrl) {
            m_lanUrl = url;
            m_core->SetLanUrl(url);
        }
    }

    void HandleResume() {
        if (!m_core) return;
        m_core->AppendLog("[App] The app woke up again.");
        std::string error;
        if (m_web && !m_web->Restart(&error)) {
            m_core->PushToast("warn", "The phone page could not restart after waking up: " + error);
        }
        RefreshLanAddress(true);
        if (m_gfxOk && m_d3dDevice && FAILED(m_d3dDevice->GetDeviceRemovedReason())) {
            RecoverGraphics(DXGI_ERROR_DEVICE_REMOVED);
        }
        if (m_wasTrainingAtSuspend) {
            m_wasTrainingAtSuspend = false;
            m_core->StartTraining();
            m_core->PushToast("info", "Welcome back. Training carries on where it stopped.");
        }
    }

    // ------------------------------------------------------------------------------------------
    // One pass of the loop: read the controller, update the screen's state, draw when something changed
    // ------------------------------------------------------------------------------------------
    void RunGuardedFrame(double dt, double& sinceRender) {
        try {
            StepFrame(dt, sinceRender);
        } catch (const std::exception& ex) {
            NoteFrameException(ex.what());
        } catch (...) {
            NoteFrameException("unknown error");
        }
    }

    void NoteFrameException(const char* what) {
        if (m_frameExceptions < 10) {
            WebBridge::WriteCrashLogToDisk(std::string("[Screen] ") + what);
            if (m_core) m_core->AppendLog(std::string("[Screen] ") + what);
        }
        ++m_frameExceptions;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    void StepFrame(double dt, double& sinceRender) {
        if (!m_tv) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            return;
        }
        const PadState pad = m_ui.PollInput();
        m_tv->Update(pad, dt);

        sinceRender += dt;
        const bool dirty = m_tv->TakeDirty();
        const double interval = m_tv->NeedsFastRedraw() ? (1.0 / 30.0) : 0.25;
        if (!dirty && sinceRender < interval) {
            std::this_thread::sleep_for(std::chrono::milliseconds(6));
            return;
        }
        sinceRender = 0.0;
        if (!m_gfxOk || !m_swapChain) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            return;
        }

        m_tv->BuildScene(m_scene);
        const HRESULT hr = DrawFrame(m_scene);
        if (SUCCEEDED(hr)) {
            m_failedFrames = 0;
            return;
        }
        if (hr == DXGI_ERROR_DEVICE_REMOVED || hr == DXGI_ERROR_DEVICE_RESET || hr == D2DERR_RECREATE_TARGET) {
            RecoverGraphics(hr);
            return;
        }
        // some other problem with one frame: skip it, and only start over if it keeps happening
        if (++m_failedFrames == 1) {
            WebBridge::WriteCrashLogToDisk("[Screen] A frame could not be drawn (error " + HexText(static_cast<unsigned long>(hr)) + ").");
        }
        if (m_failedFrames >= 60) {
            m_failedFrames = 0;
            RecoverGraphics(hr);
        }
    }

    void ShowSplash(const char* line) {
        if (!m_gfxOk || !m_swapChain) return;
        try {
            TvScene scene;
            TvItem bg;
            bg.kind = TvItem::Kind::Fill;
            bg.x = 0.0f; bg.y = 0.0f; bg.w = kTvWidth; bg.h = kTvHeight;
            bg.color = TvColor::Background;
            scene.items.push_back(bg);
            TvItem title;
            title.kind = TvItem::Kind::Text;
            title.x = 0.0f; title.y = 430.0f; title.w = kTvWidth; title.h = 80.0f;
            title.font = TvFont::Title;
            title.color = TvColor::Text;
            title.align = TvAlign::Center;
            title.text = "Image Trainer";
            scene.items.push_back(title);
            TvItem sub;
            sub.kind = TvItem::Kind::Text;
            sub.x = 0.0f; sub.y = 510.0f; sub.w = kTvWidth; sub.h = 50.0f;
            sub.font = TvFont::Body;
            sub.color = TvColor::Muted;
            sub.align = TvAlign::Center;
            sub.text = line;
            scene.items.push_back(sub);
            DrawFrame(scene);
        } catch (...) {
        }
    }

    // ------------------------------------------------------------------------------------------
    // Graphics
    // ------------------------------------------------------------------------------------------
    void AttachGpuNow() {
        m_gpuAttached = false;
        if (!m_core || !m_d3dDevice || !m_cmdQueue) return;
        try {
            m_gpuAttached = m_core->AttachGpu(m_d3dDevice.Get(), m_cmdQueue.Get());
        } catch (const std::exception& ex) {
            WebBridge::WriteCrashLogToDisk(std::string("[GPU] ") + ex.what());
        } catch (...) {
            WebBridge::WriteCrashLogToDisk("[GPU] Unknown exception while starting the graphics chip.");
        }
    }

    void RecoverGraphics(HRESULT cause) {
        HRESULT reason = cause;
        if (m_d3dDevice) {
            const HRESULT r = m_d3dDevice->GetDeviceRemovedReason();
            if (FAILED(r)) reason = r;
        }
        const std::string text = "Graphics problem (error " + HexText(static_cast<unsigned long>(cause)) + ", reason " +
                                 HexText(static_cast<unsigned long>(reason)) + "). Starting the screen again.";
        WebBridge::WriteCrashLogToDisk("[D3D12] " + text);
        if (m_core) m_core->AppendLog("[D3D12] " + text);

        // three restarts in a row without 15 quiet seconds in between: stop trying (the phone page and training keep working)
        const auto now = std::chrono::steady_clock::now();
        if (now - m_lastRecovery < std::chrono::seconds(15)) ++m_quickRecoveries;
        else m_quickRecoveries = 0;
        m_lastRecovery = now;

        if (m_core) m_core->DetachGpu(true);
        m_gpuAttached = false;
        ReleaseGraphics();
        if (m_quickRecoveries >= 3) {
            m_gfxOk = false;
            if (m_core) m_core->PushToast("error", "The TV screen keeps failing, so it was switched off. The phone page still works. Restart the app to try again.");
            return;
        }
        const HRESULT hr = InitializeD3D12();
        if (FAILED(hr)) {
            m_gfxOk = false;
            WebBridge::WriteCrashLogToDisk("[D3D12] The screen could not be started again (error " + HexText(static_cast<unsigned long>(hr)) + ").");
            if (m_core) m_core->PushToast("error", "The TV screen could not be started again. The phone page still works. Restart the app to try again.");
            return;
        }
        m_gfxOk = true;
        AttachGpuNow();
        if (m_core) m_core->PushToast("warn", "The graphics chip was reset. Training can carry on.");
    }

    HRESULT InitializeD3D12() {
        UINT dxgiFlags = 0;
#ifdef _DEBUG
        Microsoft::WRL::ComPtr<ID3D12Debug> debugCtrl;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debugCtrl)))) {
            debugCtrl->EnableDebugLayer();
            dxgiFlags |= DXGI_CREATE_FACTORY_DEBUG;
        }
#endif
        HRESULT hr = CreateDXGIFactory2(dxgiFlags, IID_PPV_ARGS(&m_dxgiFactory));
        if (FAILED(hr)) return hr;

        constexpr D3D_FEATURE_LEVEL kLevels[] = {
            D3D_FEATURE_LEVEL_12_0,
            D3D_FEATURE_LEVEL_11_1,
            D3D_FEATURE_LEVEL_11_0
        };

        Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter;
        for (UINT idx = 0; m_dxgiFactory->EnumAdapterByGpuPreference(idx, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&adapter)) != DXGI_ERROR_NOT_FOUND; ++idx) {
            DXGI_ADAPTER_DESC1 desc = {};
            adapter->GetDesc1(&desc);
            if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;

            for (D3D_FEATURE_LEVEL fl : kLevels) {
                if (SUCCEEDED(D3D12CreateDevice(adapter.Get(), fl, IID_PPV_ARGS(&m_d3dDevice)))) {
                    if (m_core) {
                        m_core->AppendLog("[Graphics] " + WideToAscii(desc.Description) + ", " + FeatureLevelName(fl) + ", " +
                                          std::to_string(static_cast<unsigned long long>(desc.DedicatedVideoMemory / (1024ULL * 1024ULL))) + " MB own memory");
                    }
                    break;
                }
            }
            if (m_d3dDevice) break;
        }
        if (!m_d3dDevice) return E_FAIL;

        D3D12_COMMAND_QUEUE_DESC qDesc = {};
        qDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        hr = m_d3dDevice->CreateCommandQueue(&qDesc, IID_PPV_ARGS(&m_cmdQueue));
        if (FAILED(hr)) return hr;

        DXGI_SWAP_CHAIN_DESC1 scDesc = {};
        scDesc.Width = m_width;
        scDesc.Height = m_height;
        scDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        scDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        scDesc.BufferCount = kFrameCount;
        scDesc.SampleDesc.Count = 1;
        scDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;

        Microsoft::WRL::ComPtr<IDXGISwapChain1> sc1;
        hr = m_dxgiFactory->CreateSwapChainForCoreWindow(
            m_cmdQueue.Get(),
            reinterpret_cast<IUnknown*>(winrt::get_abi(m_window)),
            &scDesc,
            nullptr,
            &sc1);
        if (FAILED(hr)) return hr;
        hr = sc1.As(&m_swapChain);
        if (FAILED(hr)) return hr;

        m_frameIndex = m_swapChain->GetCurrentBackBufferIndex();

        for (uint32_t i = 0; i < kFrameCount; ++i) {
            hr = m_d3dDevice->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&m_cmdAllocators[i]));
            if (FAILED(hr)) return hr;
        }
        hr = m_d3dDevice->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, m_cmdAllocators[0].Get(), nullptr, IID_PPV_ARGS(&m_cmdList));
        if (FAILED(hr)) return hr;
        m_cmdList->Close();

        hr = m_d3dDevice->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_fence));
        if (FAILED(hr)) return hr;
        m_fenceValue = 0;
        m_fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);

        m_backBuffers.resize(kFrameCount);
        for (uint32_t i = 0; i < kFrameCount; ++i) {
            hr = m_swapChain->GetBuffer(i, IID_PPV_ARGS(&m_backBuffers[i]));
            if (FAILED(hr)) return hr;
        }

        hr = m_ui.Initialize(m_d3dDevice.Get(), m_cmdQueue.Get(), m_backBuffers, m_width, m_height);
        if (FAILED(hr)) return hr;
        return S_OK;
    }

    HRESULT DrawFrame(const TvScene& scene) {
        HRESULT hr = m_cmdAllocators[m_frameIndex]->Reset();
        if (FAILED(hr)) return hr;
        hr = m_cmdList->Reset(m_cmdAllocators[m_frameIndex].Get(), nullptr);
        if (FAILED(hr)) return hr;

        D3D12_RESOURCE_BARRIER barrier = {};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition.pResource = m_backBuffers[m_frameIndex].Get();
        barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
        barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET;
        barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        m_cmdList->ResourceBarrier(1, &barrier);

        hr = m_cmdList->Close();
        if (FAILED(hr)) return hr;

        ID3D12CommandList* lists[] = { m_cmdList.Get() };
        m_cmdQueue->ExecuteCommandLists(1, lists);

        hr = m_d3dDevice->GetDeviceRemovedReason();
        if (FAILED(hr)) return hr;

        hr = m_ui.Render(m_frameIndex, scene);
        if (FAILED(hr)) return hr;

        hr = m_swapChain->Present(1, 0);
        if (FAILED(hr)) return hr;

        WaitForGpu();
        m_frameIndex = m_swapChain->GetCurrentBackBufferIndex();
        return S_OK;
    }

    void ReleaseGraphics() {
        m_ui.Release();
        m_backBuffers.clear();
        m_cmdList.Reset();
        for (auto& alloc : m_cmdAllocators) alloc.Reset();
        m_fence.Reset();
        if (m_fenceEvent) {
            CloseHandle(m_fenceEvent);
            m_fenceEvent = nullptr;
        }
        m_swapChain.Reset();
        m_cmdQueue.Reset();
        m_d3dDevice.Reset();
        m_dxgiFactory.Reset();
    }

    void RebuildSwapChain() {
        if (!m_swapChain || !m_gfxOk) return;
        WaitForGpu();
        m_ui.Release();
        m_backBuffers.clear();
        HRESULT hr = m_swapChain->ResizeBuffers(kFrameCount, m_width, m_height, DXGI_FORMAT_B8G8R8A8_UNORM, 0);
        if (SUCCEEDED(hr)) {
            m_backBuffers.resize(kFrameCount);
            for (uint32_t i = 0; i < kFrameCount && SUCCEEDED(hr); ++i) {
                hr = m_swapChain->GetBuffer(i, IID_PPV_ARGS(&m_backBuffers[i]));
            }
        }
        if (SUCCEEDED(hr)) {
            m_frameIndex = m_swapChain->GetCurrentBackBufferIndex();
            hr = m_ui.Initialize(m_d3dDevice.Get(), m_cmdQueue.Get(), m_backBuffers, m_width, m_height);
        }
        if (FAILED(hr)) RecoverGraphics(hr);
    }

    void WaitForGpu() {
        if (!m_cmdQueue || !m_fence || !m_fenceEvent) return;
        const uint64_t val = ++m_fenceValue;
        if (SUCCEEDED(m_cmdQueue->Signal(m_fence.Get(), val))) {
            if (m_fence->GetCompletedValue() < val) {
                m_fence->SetEventOnCompletion(val, m_fenceEvent);
                WaitForSingleObject(m_fenceEvent, 2000);
            }
        }
    }

    // ------------------------------------------------------------------------------------------
    // State
    // ------------------------------------------------------------------------------------------
    CoreWindow m_window{ nullptr };
    winrt::event_token m_suspendToken{};
    winrt::event_token m_resumeToken{};
    bool m_running = true;
    bool m_resizePending = false;
    uint32_t m_width = 1920;
    uint32_t m_height = 1080;
    uint32_t m_frameIndex = 0;

    std::atomic<bool> m_resumed{ false };
    bool m_wasTrainingAtSuspend = false;
    bool m_gfxOk = false;
    bool m_gpuAttached = false;
    uint32_t m_failedFrames = 0;
    uint32_t m_frameExceptions = 0;
    uint32_t m_quickRecoveries = 0;
    std::chrono::steady_clock::time_point m_lastRecovery{};
    std::string m_lanUrl;

    std::unique_ptr<AppCore> m_core;
    std::unique_ptr<WebHost> m_web;
    std::unique_ptr<TvModel> m_tv;
    GamepadUI m_ui;
    TvScene m_scene;

    Microsoft::WRL::ComPtr<IDXGIFactory6> m_dxgiFactory;
    Microsoft::WRL::ComPtr<ID3D12Device> m_d3dDevice;
    Microsoft::WRL::ComPtr<ID3D12CommandQueue> m_cmdQueue;
    Microsoft::WRL::ComPtr<IDXGISwapChain3> m_swapChain;
    Microsoft::WRL::ComPtr<ID3D12CommandAllocator> m_cmdAllocators[kFrameCount];
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> m_cmdList;
    std::vector<Microsoft::WRL::ComPtr<ID3D12Resource>> m_backBuffers;
    Microsoft::WRL::ComPtr<ID3D12Fence> m_fence;
    HANDLE m_fenceEvent = nullptr;
    uint64_t m_fenceValue = 0;
};

int WINAPI wWinMain(HINSTANCE, HINSTANCE, PWSTR, int) {
    winrt::init_apartment();
    CoreApplication::Run(winrt::make<XboxOTFMApp>());
    return 0;
}
