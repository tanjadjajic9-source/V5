#pragma once
// The TV screen: what to show and how the controller moves around it. There is no Windows code in here. The Windows part only
// reads the controller into a PadState and draws the list of shapes this produces, so everything else is tested natively.

#include "AppCore.h"
#include "ImageUtil.h"
#include "QrCode.h"

#include <atomic>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace XboxOTFM {

// ---------------------------------------------------------------------------------------------
// Controller
// ---------------------------------------------------------------------------------------------
enum PadBits : uint32_t {
    kPadA = 1u << 0,
    kPadB = 1u << 1,
    kPadX = 1u << 2,
    kPadY = 1u << 3,
    kPadUp = 1u << 4,
    kPadDown = 1u << 5,
    kPadLeft = 1u << 6,
    kPadRight = 1u << 7,
    kPadLB = 1u << 8,
    kPadRB = 1u << 9,
    kPadView = 1u << 10,
    kPadMenu = 1u << 11
};

struct PadState {
    uint32_t buttons = 0;   // PadBits held right now
    float stickX = 0.0f;    // left stick, -1 (left) .. 1 (right)
    float stickY = 0.0f;    // left stick, -1 (down) .. 1 (up)
    bool connected = false;
};

// ---------------------------------------------------------------------------------------------
// What to draw. Everything is placed on a 1920 x 1080 page; the Windows code scales that to the real screen.
// ---------------------------------------------------------------------------------------------
constexpr float kTvWidth = 1920.0f;
constexpr float kTvHeight = 1080.0f;

enum class TvColor : uint8_t {
    Background, Panel, PanelRaised, Line, Text, Muted, Accent, OnAccent, Good, Warn, Bad, Shade, ChartLine, ChartGrid, Count
};
enum class TvFont : uint8_t { Display, Title, Heading, Body, BodyBold, Small, Count };
enum class TvAlign : uint8_t { Left, Center, Right };

uint32_t TvPaletteArgb(TvColor color);   // 0xAARRGGBB
float TvFontSize(TvFont font);           // pixels on the 1920 x 1080 page

struct TvItem {
    enum class Kind : uint8_t {
        Fill,     // filled rectangle (rounded when radius > 0)
        Outline,  // rectangle outline
        Text,     // text inside the rectangle (clipped to it)
        Lines,    // connected line segments through `points` (x0, y0, x1, y1, ...)
        Image,    // a picture stretched into the rectangle
        Badge     // a filled circle with a letter (a controller button)
    };
    Kind kind = Kind::Fill;
    float x = 0.0f, y = 0.0f, w = 0.0f, h = 0.0f;
    float radius = 0.0f;
    float thickness = 1.0f;
    TvColor color = TvColor::Text;
    TvFont font = TvFont::Body;
    TvAlign align = TvAlign::Left;
    bool wrap = false;                         // Text: wrap onto several lines instead of staying on one
    bool smooth = false;                       // Image: smooth enlargement instead of crisp squares
    std::string text;
    std::vector<float> points;
    std::shared_ptr<const Rgb8Image> image;
    uint64_t imageKey = 0;                     // same key = same picture (lets the drawing code keep the converted copy)
};

struct TvScene {
    std::vector<TvItem> items;
};

class TvBuilder;

enum class TvTab : int { Train = 0, Settings = 1, Saves = 2, Pictures = 3, Phone = 4 };
constexpr int kTvTabCount = 5;

// ---------------------------------------------------------------------------------------------
// The model
// ---------------------------------------------------------------------------------------------
class TvModel {
public:
    explicit TvModel(AppCore& core);
    ~TvModel();
    TvModel(const TvModel&) = delete;
    TvModel& operator=(const TvModel&) = delete;

    // Call once per frame with the controller and the seconds since the previous call.
    void Update(const PadState& pad, double dtSeconds);
    // The shapes for the current moment.
    void BuildScene(TvScene& scene);

    // True while something on screen changes quickly enough to deserve many redraws per second.
    bool NeedsFastRedraw() const;
    // True once after the person pressed something (so an idle screen can still react at once).
    bool TakeDirty();

    // ---- for tests ----
    TvTab Tab() const { return static_cast<TvTab>(m_tab); }
    bool DialogOpen() const { return m_dialog.open; }
    bool ViewerOpen() const { return m_viewerOpen; }
    bool JobRunning() const { return m_jobBusy.load(); }
    int Focus() const { return m_focus[m_tab]; }
    void WaitForJob(int maxMs);

private:
    struct SettingRow {
        enum class Kind { Choice, Toggle, Action };
        std::string label;
        std::string help;
        Kind kind = Kind::Choice;
        std::function<std::string()> value;
        std::function<void(int)> change;   // -1 / +1 = left / right, 0 = the A button
        bool danger = false;
    };
    struct Dialog {
        bool open = false;
        std::string title;
        std::vector<std::string> lines;
        std::string yesLabel;
        bool danger = false;
        double openFor = 0.0;
        std::function<void()> onYes;
    };
    struct DreamTile {
        uint64_t id = 0;
        std::shared_ptr<Rgb8Image> image;
        uint64_t step = 0;
        uint32_t dreamSteps = 0;
        bool usedAverage = false;
    };
    struct ShownToast {
        std::string level;
        std::string text;
        double age = 0.0;
    };

    void Refresh();
    void HandlePress(uint32_t pressed);
    void HandleTrain(uint32_t pressed);
    void HandleSettings(uint32_t pressed);
    void HandleSaves(uint32_t pressed);
    void HandlePictures(uint32_t pressed);
    void MoveFocus(int tab, int delta, int count, int visibleRows);
    void SwitchTab(int tab);
    void BuildRows();
    bool StartJob(const std::string& label, std::function<void()> work);
    void OpenDialog(const std::string& title, std::vector<std::string> lines, const std::string& yesLabel, bool danger, std::function<void()> onYes);
    void MakePicturesJob();
    void SaveNowJob();
    bool BusyForActions() const;

    // drawing
    void DrawTopBar(TvBuilder& b) const;
    void DrawTrain(TvBuilder& b) const;
    void DrawChart(TvBuilder& b, float x, float y, float w, float h) const;
    void DrawSamplePanel(TvBuilder& b, float x, float y, float w, float h) const;
    void DrawNotice(TvBuilder& b, float x, float y, float w, float h) const;
    void DrawSettings(TvBuilder& b) const;
    void DrawSaves(TvBuilder& b) const;
    void DrawPictures(TvBuilder& b) const;
    void DrawPhone(TvBuilder& b) const;
    void DrawFooter(TvBuilder& b) const;
    void DrawToasts(TvBuilder& b) const;
    void DrawViewer(TvBuilder& b) const;
    void DrawDialog(TvBuilder& b) const;
    std::string TrendText() const;

    AppCore& m_core;

    // what is on screen
    int m_tab = 0;
    int m_focus[kTvTabCount] = { 0, 0, 0, 0, 0 };
    int m_scroll[kTvTabCount] = { 0, 0, 0, 0, 0 };
    bool m_viewerOpen = false;
    Dialog m_dialog;
    double m_clock = 0.0;
    bool m_dirty = true;

    // controller bookkeeping
    uint32_t m_prevButtons = 0;
    double m_holdTime = 0.0;
    double m_nextRepeat = 0.4;

    // copies of core state, refreshed ten times a second
    double m_sinceRefresh = 99.0;
    AppSnapshot m_snap;
    std::shared_ptr<Rgb8Image> m_preview;
    uint64_t m_previewShown = ~0ull;
    uint64_t m_previewStep = 0;
    uint32_t m_previewDreamSteps = 0;
    bool m_previewAverage = false;
    uint32_t m_previewTiles = 0;
    std::vector<std::pair<uint64_t, float>> m_curve;
    uint64_t m_lossShown = ~0ull;
    std::vector<CheckpointMeta> m_saves;
    uint64_t m_savesShown = ~0ull;
    std::vector<DreamTile> m_dreams;   // newest first
    std::deque<ShownToast> m_toasts;
    uint64_t m_lastToastId = 0;

    // settings screen
    std::vector<SettingRow> m_rows;
    int m_quickChoice = 1;
    int m_brainChoice = 1;
    bool m_brainChoiceTouched = false;
    int m_typeChoice = 0;          // 0 = a normal brain, 1 = a wave brain, 2 = a one-pass brain (for "Start over")
    bool m_typeChoiceTouched = false;

    // phone screen
    std::string m_qrUrl;
    std::shared_ptr<Rgb8Image> m_qrImage;
    uint64_t m_qrKey = 0;

    // background jobs (saving, loading, drawing pictures: anything that may wait for the worker)
    std::atomic<bool> m_jobBusy{ false };
    std::atomic<int> m_pendingTab{ -1 };
    mutable std::mutex m_jobMutex;
    std::string m_jobLabel;
    std::thread m_jobThread;
};

} // namespace XboxOTFM
