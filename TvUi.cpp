#include "TvUi.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <ctime>

namespace XboxOTFM {

// ---------------------------------------------------------------------------------------------
// Look
// ---------------------------------------------------------------------------------------------
uint32_t TvPaletteArgb(TvColor color) {
    switch (color) {
        case TvColor::Background: return 0xFF0B1626u;
        case TvColor::Panel:      return 0xFF132338u;
        case TvColor::PanelRaised:return 0xFF1D3452u;
        case TvColor::Line:       return 0xFF2E4766u;
        case TvColor::Text:       return 0xFFF2F5FAu;
        case TvColor::Muted:      return 0xFFA9BAD0u;
        case TvColor::Accent:     return 0xFFFF7A4Du;
        case TvColor::OnAccent:   return 0xFF0B1626u;
        case TvColor::Good:       return 0xFF4CC38Au;
        case TvColor::Warn:       return 0xFFF2B544u;
        case TvColor::Bad:        return 0xFFFF6B6Bu;
        case TvColor::Shade:      return 0xB8000000u;
        case TvColor::ChartLine:  return 0xFF7FB8FFu;
        case TvColor::ChartGrid:  return 0xFF24384F;
        default:                  return 0xFFFFFFFFu;
    }
}

float TvFontSize(TvFont font) {
    switch (font) {
        case TvFont::Display:  return 80.0f;
        case TvFont::Title:    return 46.0f;
        case TvFont::Heading:  return 34.0f;
        case TvFont::Body:     return 28.0f;
        case TvFont::BodyBold: return 28.0f;
        case TvFont::Small:    return 23.0f;
        default:               return 28.0f;
    }
}

namespace {

constexpr float kMarginX = 88.0f;
constexpr float kRightEdge = kTvWidth - kMarginX;      // 1832
constexpr float kContentTop = 156.0f;
constexpr float kContentBottom = 936.0f;
constexpr int kSettingsVisible = 10;
constexpr int kSavesVisible = 8;
constexpr int kPictureColumns = 6;
constexpr size_t kMaxDreamTiles = 12;

// ---- text helpers (ASCII only) ----
std::string Int(uint64_t v) {
    std::string digits = std::to_string(v);
    std::string out;
    int count = 0;
    for (size_t i = digits.size(); i > 0; --i) {
        out.insert(out.begin(), digits[i - 1]);
        if (++count % 3 == 0 && i > 1) out.insert(out.begin(), ',');
    }
    return out;
}

std::string Fixed(double v, int decimals) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.*f", decimals, v);
    return buf;
}

std::string Duration(double sec) {
    if (!std::isfinite(sec) || sec < 0) return "-";
    const long s = static_cast<long>(sec + 0.5);
    if (s < 60) return std::to_string(s) + " s";
    const long m = (s + 30) / 60;
    if (m < 60) return std::to_string(m) + " min";
    const long hrs = s / 3600;
    const long mm = (s % 3600 + 30) / 60;
    if (hrs < 48) return std::to_string(hrs) + " h" + (mm > 0 ? " " + std::to_string(mm) + " min" : std::string());
    return std::to_string(hrs / 24) + " days";
}

std::string AgoFromUnix(uint64_t unixSeconds) {
    if (unixSeconds == 0) return std::string();
    const double now = static_cast<double>(std::time(nullptr));
    const double s = std::max(0.0, now - static_cast<double>(unixSeconds));
    if (s < 60) return "just now";
    if (s < 3600) return std::to_string(static_cast<long>(s / 60 + 0.5)) + " min ago";
    if (s < 86400) return std::to_string(static_cast<long>(s / 3600 + 0.5)) + " h ago";
    return std::to_string(static_cast<long>(s / 86400 + 0.5)) + " days ago";
}

std::string Bytes(uint64_t n) {
    if (n < 1024) return std::to_string(n) + " B";
    if (n < 1024 * 1024) return std::to_string((n + 512) / 1024) + " KB";
    return Fixed(static_cast<double>(n) / (1024.0 * 1024.0), n < 10ull * 1024 * 1024 ? 1 : 0) + " MB";
}

// Cuts text that is too long for `widthPx` (the drawing code also clips, this keeps an ellipsis visible).
std::string Fit(const std::string& s, float widthPx, TvFont font) {
    const float size = TvFontSize(font);
    const size_t maxChars = static_cast<size_t>(std::max(4.0f, widthPx / (size * 0.55f)));
    if (s.size() <= maxChars) return s;
    return s.substr(0, maxChars - 3) + "...";
}

float TextWidth(const std::string& s, TvFont font) {
    return static_cast<float>(s.size()) * TvFontSize(font) * 0.54f;
}

const char* BrainName(uint32_t preset) {
    switch (preset) {
        case 0: return "Small";
        case 1: return "Medium";
        default: return "Large";
    }
}

// "Medium", "Medium wave" or "Medium one-pass" (type: 0 normal, 1 wave, 2 one-pass)
std::string BrainFullName(uint32_t preset, uint32_t type) {
    return std::string(BrainName(preset)) + (type == 1 ? " wave" : (type == 2 ? " one-pass" : ""));
}

const char* BrainDials(uint32_t preset, uint32_t type) {
    if (type == 2) {
        switch (preset) {
            case 0: return "about 1 million dials";
            case 1: return "about 3 million dials";
            default: return "about 8 million dials";
        }
    }
    switch (preset) {
        case 0: return "about 25,000 dials";
        case 1: return "about 80,000 dials";
        default: return "about 250,000 dials";
    }
}

const char* BrainTypeName(int type) {
    return type == 1 ? "Wave (faster)" : (type == 2 ? "One-pass (experiment)" : "Normal");
}

std::string KindLabel(const std::string& kind) {
    if (kind == "auto") return "Autosave";
    if (kind == "safety") return "Safety copy";
    if (kind == "imported") return "Imported";
    return "Saved by you";
}

// The value in `values` closest to `current`, then one step in `dir` (0 = forward, wrapping round).
double NextInList(const double* values, size_t count, double current, int dir, bool wrap) {
    size_t best = 0;
    double bestDiff = 1e300;
    for (size_t i = 0; i < count; ++i) {
        const double scale = std::max(1e-9, std::fabs(values[i]) + std::fabs(current));
        const double diff = std::fabs(values[i] - current) / scale;
        if (diff < bestDiff) { bestDiff = diff; best = i; }
    }
    long idx = static_cast<long>(best) + (dir == 0 ? 1 : dir);
    const long n = static_cast<long>(count);
    if (idx < 0) idx = wrap ? n - 1 : 0;
    if (idx >= n) idx = wrap ? 0 : n - 1;
    return values[idx];
}

const double kGoalValues[] = { 0, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000 };
const double kRateValues[] = { 0.0002, 0.0003, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.01 };
const double kSizeValues[] = { 16, 24, 32, 48, 64, 96, 128 };
const double kBatchValues[] = { 1, 2, 4, 8, 12, 16, 24, 32 };
const double kSaveValues[] = { 0, 2, 5, 10, 30, 60 };
const double kCrispValues[] = { 0, 8, 16, 32, 64 };

template <size_t N>
size_t Count(const double (&)[N]) { return N; }

} // namespace

// ---------------------------------------------------------------------------------------------
// Scene builder
// ---------------------------------------------------------------------------------------------
class TvBuilder {
public:
    explicit TvBuilder(TvScene& scene) : m_scene(scene) {}

    void Fill(float x, float y, float w, float h, TvColor c, float radius = 0.0f) {
        TvItem it;
        it.kind = TvItem::Kind::Fill;
        it.x = x; it.y = y; it.w = w; it.h = h; it.color = c; it.radius = radius;
        m_scene.items.push_back(std::move(it));
    }
    void Outline(float x, float y, float w, float h, TvColor c, float thickness = 2.0f, float radius = 0.0f) {
        TvItem it;
        it.kind = TvItem::Kind::Outline;
        it.x = x; it.y = y; it.w = w; it.h = h; it.color = c; it.thickness = thickness; it.radius = radius;
        m_scene.items.push_back(std::move(it));
    }
    void Text(float x, float y, float w, float h, const std::string& text, TvFont font, TvColor c,
              TvAlign align = TvAlign::Left, bool wrap = false) {
        if (text.empty()) return;
        TvItem it;
        it.kind = TvItem::Kind::Text;
        it.x = x; it.y = y; it.w = w; it.h = h; it.text = text; it.font = font; it.color = c; it.align = align; it.wrap = wrap;
        m_scene.items.push_back(std::move(it));
    }
    void Lines(const std::vector<float>& points, TvColor c, float thickness) {
        if (points.size() < 4) return;
        TvItem it;
        it.kind = TvItem::Kind::Lines;
        it.points = points; it.color = c; it.thickness = thickness;
        m_scene.items.push_back(std::move(it));
    }
    void Image(float x, float y, float w, float h, std::shared_ptr<const Rgb8Image> image, uint64_t key, bool smooth = false) {
        if (!image || !image->Valid()) return;
        TvItem it;
        it.kind = TvItem::Kind::Image;
        it.x = x; it.y = y; it.w = w; it.h = h; it.image = std::move(image); it.imageKey = key; it.smooth = smooth;
        m_scene.items.push_back(std::move(it));
    }
    void Badge(float cx, float cy, float r, const std::string& letter, TvColor c) {
        TvItem it;
        it.kind = TvItem::Kind::Badge;
        it.x = cx - r; it.y = cy - r; it.w = 2 * r; it.h = 2 * r; it.text = letter; it.color = c;
        m_scene.items.push_back(std::move(it));
    }
    // "(A) Start training": returns where the next hint can start.
    float Hint(float x, float y, const std::string& letter, TvColor badge, const std::string& label) {
        Badge(x + 22, y + 22, 22, letter, badge);
        const float w = TextWidth(label, TvFont::Body) + 8;
        Text(x + 58, y + 4, w, 40, label, TvFont::Body, TvColor::Text);
        return x + 58 + w + 36;
    }
    // "[LB RB] Switch screen" for buttons that have no coloured badge.
    float Chip(float x, float y, const std::string& chip, const std::string& label) {
        const float cw = TextWidth(chip, TvFont::Small) + 28;
        Outline(x, y + 4, cw, 40, TvColor::Muted, 2.0f, 10.0f);
        Text(x, y + 8, cw, 34, chip, TvFont::Small, TvColor::Muted, TvAlign::Center);
        const float w = TextWidth(label, TvFont::Body) + 8;
        Text(x + cw + 12, y + 4, w, 40, label, TvFont::Body, TvColor::Text);
        return x + cw + 12 + w + 36;
    }

private:
    TvScene& m_scene;
};

// ---------------------------------------------------------------------------------------------
// Model: set-up and data
// ---------------------------------------------------------------------------------------------
TvModel::TvModel(AppCore& core) : m_core(core) {
    BuildRows();
    Refresh();
}

TvModel::~TvModel() {
    if (m_jobThread.joinable()) m_jobThread.join();
}

void TvModel::WaitForJob(int maxMs) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(maxMs);
    while (m_jobBusy.load() && std::chrono::steady_clock::now() < deadline) std::this_thread::sleep_for(std::chrono::milliseconds(5));
}

bool TvModel::TakeDirty() {
    const bool d = m_dirty;
    m_dirty = false;
    return d;
}

bool TvModel::NeedsFastRedraw() const {
    if (m_dialog.open || m_viewerOpen || !m_toasts.empty() || m_jobBusy.load() || m_snap.busy) return true;
    if (m_snap.state == RunState::Training) return !m_snap.config.blindSpeedMode;
    return false;
}

void TvModel::Refresh() {
    m_sinceRefresh = 0.0;
    m_snap = m_core.Snapshot();

    if (m_snap.previewVersion != m_previewShown) {
        PreviewImage p = m_core.GetTrainer().LatestPreview();
        m_previewShown = m_snap.previewVersion;
        if (p.version != 0 && p.grid.Valid()) {
            m_previewStep = p.step;
            m_previewDreamSteps = p.dreamSteps;
            m_previewAverage = p.usedAverage;
            m_previewTiles = p.tiles;
            m_preview = std::make_shared<Rgb8Image>(std::move(p.grid));
        } else {
            m_preview.reset();
        }
    }
    if (m_snap.lossVersion != m_lossShown) {
        m_curve = m_core.GetTrainer().LossCurve(320);
        m_lossShown = m_snap.lossVersion;
    }
    if (m_snap.snapshotListVersion != m_savesShown) {
        m_saves = m_core.ListSnapshots();
        std::stable_sort(m_saves.begin(), m_saves.end(), [](const CheckpointMeta& a, const CheckpointMeta& b) {
            if (a.createdUnix != b.createdUnix) return a.createdUnix > b.createdUnix;
            return a.step > b.step;
        });
        m_savesShown = m_snap.snapshotListVersion;
        const int n = static_cast<int>(m_saves.size());
        m_focus[static_cast<int>(TvTab::Saves)] = std::clamp(m_focus[static_cast<int>(TvTab::Saves)], 0, std::max(0, n - 1));
        m_scroll[static_cast<int>(TvTab::Saves)] = std::clamp(m_scroll[static_cast<int>(TvTab::Saves)], 0, std::max(0, n - kSavesVisible));
    }

    // pictures made on request: keep the newest few, fetch pixels only for ones not seen before
    {
        const std::vector<DreamEntry> list = m_core.ListDreams(); // newest last
        std::vector<DreamTile> next;
        for (auto it = list.rbegin(); it != list.rend() && next.size() < kMaxDreamTiles; ++it) {
            DreamTile tile;
            bool reused = false;
            for (const DreamTile& old : m_dreams) {
                if (old.id == it->id) { tile = old; reused = true; break; }
            }
            if (!reused) {
                DreamEntry full;
                if (!m_core.GetDream(it->id, full) || full.rgb.size() != static_cast<size_t>(full.w) * full.h * 3 || full.w == 0) continue;
                auto img = std::make_shared<Rgb8Image>();
                img->w = full.w;
                img->h = full.h;
                img->px = std::move(full.rgb);
                tile.id = it->id;
                tile.image = std::move(img);
                tile.step = full.step;
                tile.dreamSteps = full.dreamSteps;
                tile.usedAverage = full.usedAverage;
            }
            next.push_back(std::move(tile));
        }
        m_dreams = std::move(next);
        const int pictures = static_cast<int>(TvTab::Pictures);
        const int n = static_cast<int>(m_dreams.size());
        m_focus[pictures] = std::clamp(m_focus[pictures], 0, std::max(0, n - 1));
        if (m_viewerOpen && n == 0) m_viewerOpen = false;
    }

    // messages from the core
    {
        const std::vector<Toast> fresh = m_core.ToastsAfter(m_lastToastId);
        for (const Toast& t : fresh) {
            m_lastToastId = std::max(m_lastToastId, t.id);
            ShownToast st;
            st.level = t.level;
            st.text = t.text;
            m_toasts.push_back(std::move(st));
            while (m_toasts.size() > 3) m_toasts.pop_front();
            m_dirty = true;
        }
    }

    if (!m_brainChoiceTouched) m_brainChoice = static_cast<int>(std::min<uint32_t>(2, m_snap.presetIndex));
    if (!m_typeChoiceTouched) m_typeChoice = m_snap.family == 1 ? 2 : static_cast<int>(std::min<uint32_t>(1, m_snap.waveLevel));

    // the phone address as a code
    if (m_snap.lanUrl != m_qrUrl) {
        m_qrUrl = m_snap.lanUrl;
        m_qrImage.reset();
        QrMatrix qr;
        if (!m_qrUrl.empty() && MakeQrCode(m_qrUrl, qr)) {
            auto img = std::make_shared<Rgb8Image>(RenderQrImage(qr, 6));
            m_qrImage = img;
            m_qrKey = Fnv1a64(reinterpret_cast<const uint8_t*>(m_qrUrl.data()), m_qrUrl.size()) | 1ull;
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Model: controller
// ---------------------------------------------------------------------------------------------
void TvModel::Update(const PadState& pad, double dt) {
    if (dt < 0.0) dt = 0.0;
    if (dt > 0.25) dt = 0.25;
    m_clock += dt;
    m_sinceRefresh += dt;
    if (m_dialog.open) m_dialog.openFor += dt;
    for (ShownToast& t : m_toasts) t.age += dt;
    while (!m_toasts.empty() && m_toasts.front().age > (m_toasts.front().level == "error" ? 9.0 : 6.0)) m_toasts.pop_front();

    if (!m_dialog.open) {
        const int pending = m_pendingTab.exchange(-1);
        if (pending >= 0 && pending < kTvTabCount) SwitchTab(pending);
    }

    uint32_t buttons = pad.buttons;
    if (pad.stickY > 0.6f) buttons |= kPadUp;
    else if (pad.stickY < -0.6f) buttons |= kPadDown;
    if (pad.stickX > 0.6f) buttons |= kPadRight;
    else if (pad.stickX < -0.6f) buttons |= kPadLeft;

    uint32_t pressed = buttons & ~m_prevButtons;
    const uint32_t dirMask = kPadUp | kPadDown | kPadLeft | kPadRight;
    const uint32_t heldDirs = buttons & dirMask;
    if (heldDirs != 0 && heldDirs == (m_prevButtons & dirMask)) {
        m_holdTime += dt;
        if (m_holdTime >= m_nextRepeat) {
            pressed |= heldDirs;
            m_nextRepeat += 0.09;
        }
    } else {
        m_holdTime = 0.0;
        m_nextRepeat = 0.4;
    }
    m_prevButtons = buttons;

    if (pressed != 0) HandlePress(pressed);

    if (m_sinceRefresh >= 0.1 || pressed != 0) Refresh();
}

void TvModel::SwitchTab(int tab) {
    m_tab = std::clamp(tab, 0, kTvTabCount - 1);
    m_viewerOpen = false;
    m_dirty = true;
}

void TvModel::MoveFocus(int tab, int delta, int count, int visibleRows) {
    if (count <= 0) { m_focus[tab] = 0; m_scroll[tab] = 0; return; }
    m_focus[tab] = std::clamp(m_focus[tab] + delta, 0, count - 1);
    if (m_focus[tab] < m_scroll[tab]) m_scroll[tab] = m_focus[tab];
    if (m_focus[tab] >= m_scroll[tab] + visibleRows) m_scroll[tab] = m_focus[tab] - visibleRows + 1;
    m_scroll[tab] = std::clamp(m_scroll[tab], 0, std::max(0, count - visibleRows));
}

bool TvModel::BusyForActions() const {
    return m_jobBusy.load() || m_snap.busy;
}

bool TvModel::StartJob(const std::string& label, std::function<void()> work) {
    bool expected = false;
    if (!m_jobBusy.compare_exchange_strong(expected, true)) return false;
    if (m_jobThread.joinable()) m_jobThread.join();
    {
        std::lock_guard<std::mutex> lock(m_jobMutex);
        m_jobLabel = label;
    }
    try {
        m_jobThread = std::thread([this, work]() {
            try { work(); } catch (...) {}
            m_jobBusy.store(false);
        });
    } catch (...) {
        m_jobBusy.store(false);
        return false;
    }
    return true;
}

void TvModel::OpenDialog(const std::string& title, std::vector<std::string> lines, const std::string& yesLabel, bool danger, std::function<void()> onYes) {
    m_dialog = Dialog();
    m_dialog.open = true;
    m_dialog.title = title;
    m_dialog.lines = std::move(lines);
    m_dialog.yesLabel = yesLabel;
    m_dialog.danger = danger;
    m_dialog.onYes = std::move(onYes);
    m_dirty = true;
}

void TvModel::HandlePress(uint32_t pressed) {
    m_dirty = true;
    if (m_dialog.open) {
        if (m_dialog.openFor < 0.35) return; // a double press must not answer a question the person has not read yet
        if (pressed & kPadA) {
            std::function<void()> yes = std::move(m_dialog.onYes);
            m_dialog = Dialog();
            if (yes) yes();
        } else if (pressed & kPadB) {
            m_dialog = Dialog();
        }
        return;
    }
    if (m_viewerOpen) {
        const int n = static_cast<int>(m_dreams.size());
        int& focus = m_focus[static_cast<int>(TvTab::Pictures)];
        if (pressed & (kPadA | kPadB)) m_viewerOpen = false;
        else if ((pressed & kPadLeft) && n > 0) focus = (focus + n - 1) % n;
        else if ((pressed & kPadRight) && n > 0) focus = (focus + 1) % n;
        return;
    }
    if (pressed & kPadLB) { SwitchTab((m_tab + kTvTabCount - 1) % kTvTabCount); return; }
    if (pressed & kPadRB) { SwitchTab((m_tab + 1) % kTvTabCount); return; }
    if (pressed & kPadMenu) { SwitchTab(static_cast<int>(TvTab::Settings)); return; }
    if (pressed & kPadView) { SwitchTab(m_tab == static_cast<int>(TvTab::Phone) ? static_cast<int>(TvTab::Train) : static_cast<int>(TvTab::Phone)); return; }
    switch (static_cast<TvTab>(m_tab)) {
        case TvTab::Train: HandleTrain(pressed); break;
        case TvTab::Settings: HandleSettings(pressed); break;
        case TvTab::Saves: HandleSaves(pressed); break;
        case TvTab::Pictures: HandlePictures(pressed); break;
        case TvTab::Phone: break;
    }
}

void TvModel::MakePicturesJob() {
    if (BusyForActions()) { m_core.PushToast("info", "One moment, the Xbox is busy."); return; }
    const bool jump = m_tab != static_cast<int>(TvTab::Pictures);
    StartJob("Making pictures...", [this, jump]() {
        DreamResult r;
        if (!m_core.DreamNow(6, 0, r)) {
            m_core.PushToast("warn", r.error.empty() ? std::string("The brain could not make pictures right now.") : r.error);
            return;
        }
        m_core.PushToast("good", "Made " + std::to_string(r.ids.size()) + (r.ids.size() == 1 ? " picture." : " pictures.") + (jump ? " They are on the Pictures screen." : ""));
        if (jump) m_pendingTab.store(static_cast<int>(TvTab::Pictures));
    });
}

void TvModel::SaveNowJob() {
    if (BusyForActions()) { m_core.PushToast("info", "One moment, the Xbox is busy."); return; }
    StartJob("Saving...", [this]() {
        std::string id, err;
        if (!m_core.SaveSnapshot(std::string(), &id, &err)) m_core.PushToast("warn", "Could not save: " + err);
    });
}

void TvModel::HandleTrain(uint32_t pressed) {
    if (pressed & kPadA) m_core.ToggleTraining();
    if (pressed & kPadX) MakePicturesJob();
    if (pressed & kPadY) SaveNowJob();
}

void TvModel::HandleSettings(uint32_t pressed) {
    const int tab = static_cast<int>(TvTab::Settings);
    const int count = static_cast<int>(m_rows.size());
    if (pressed & kPadUp) MoveFocus(tab, -1, count, kSettingsVisible);
    if (pressed & kPadDown) MoveFocus(tab, 1, count, kSettingsVisible);
    int dir = 0;
    bool act = false;
    if (pressed & kPadLeft) { dir = -1; act = true; }
    else if (pressed & kPadRight) { dir = 1; act = true; }
    else if (pressed & kPadA) { dir = 0; act = true; }
    const int f = m_focus[tab];
    if (act && f >= 0 && f < count) m_rows[static_cast<size_t>(f)].change(dir);
}

void TvModel::HandleSaves(uint32_t pressed) {
    const int tab = static_cast<int>(TvTab::Saves);
    const int count = static_cast<int>(m_saves.size());
    if (pressed & kPadUp) MoveFocus(tab, -1, count, kSavesVisible);
    if (pressed & kPadDown) MoveFocus(tab, 1, count, kSavesVisible);
    if (pressed & kPadY) { SaveNowJob(); return; }
    const int f = m_focus[tab];
    if (f < 0 || f >= count) return;
    const CheckpointMeta meta = m_saves[static_cast<size_t>(f)];
    const std::string label = meta.name.empty() ? KindLabel(meta.kind) : meta.name;
    if (pressed & kPadA) {
        OpenDialog("Load this save?",
                   { "\"" + label + "\" (lesson " + Int(meta.step) + ")",
                     "Training pauses and the brain is replaced by this save. If the brain learned something since your last save, it is kept first as a safety copy." },
                   "Load", false,
                   [this, meta]() {
                       if (BusyForActions()) { m_core.PushToast("info", "One moment, the Xbox is busy."); return; }
                       const std::string id = meta.id;
                       StartJob("Loading...", [this, id]() {
                           std::string err;
                           if (!m_core.LoadSnapshot(id, &err)) m_core.PushToast("warn", "Could not load: " + err);
                       });
                   });
    } else if (pressed & kPadX) {
        OpenDialog("Delete this save?",
                   { "\"" + label + "\" (lesson " + Int(meta.step) + ")", "This cannot be undone." },
                   "Delete", true,
                   [this, meta]() {
                       if (BusyForActions()) { m_core.PushToast("info", "One moment, the Xbox is busy."); return; }
                       const std::string id = meta.id;
                       StartJob("Deleting...", [this, id]() {
                           std::string err;
                           if (!m_core.DeleteSnapshot(id, &err)) m_core.PushToast("warn", "Could not delete: " + err);
                       });
                   });
    }
}

void TvModel::HandlePictures(uint32_t pressed) {
    const int tab = static_cast<int>(TvTab::Pictures);
    const int n = static_cast<int>(m_dreams.size());
    int& f = m_focus[tab];
    if (n > 0) {
        if (pressed & kPadLeft) f = std::max(0, f - 1);
        if (pressed & kPadRight) f = std::min(n - 1, f + 1);
        if (pressed & kPadUp) f = (f >= kPictureColumns) ? f - kPictureColumns : f;
        if (pressed & kPadDown) f = (f + kPictureColumns < n) ? f + kPictureColumns : f;
        if (pressed & kPadA) m_viewerOpen = true;
    }
    if (pressed & kPadX) MakePicturesJob();
}

void TvModel::BuildRows() {
    m_rows.clear();
    auto add = [this](const std::string& label, SettingRow::Kind kind, const std::string& help,
                      std::function<std::string()> value, std::function<void(int)> change, bool danger = false) {
        SettingRow r;
        r.label = label;
        r.kind = kind;
        r.help = help;
        r.value = std::move(value);
        r.change = std::move(change);
        r.danger = danger;
        m_rows.push_back(std::move(r));
    };
    static const char* const kQuickNames[4] = { "Quick test", "Balanced", "Detailed", "Pixel art" };

    add("Quick setup", SettingRow::Kind::Action,
        "Not sure what to pick? Use Left and Right to choose a ready-made set of settings, then press A to apply it. "
        "Quick test: small 16 pixel pictures that stop after 2,000 lessons. Balanced: the recommended starting point. "
        "Detailed: 64 pixel pictures that stop after 50,000 lessons and take much longer. "
        "Pixel art: 64 pixel pictures with sharp pixels and 32 clean colours, no limit. Use size 32 for 32 by 32 sprites.",
        [this]() { return std::string("<  ") + kQuickNames[std::clamp(m_quickChoice, 0, 3)] + "  >"; },
        [this](int dir) {
            if (dir != 0) { m_quickChoice = std::clamp(m_quickChoice + dir, 0, 3); return; }
            if (m_quickChoice == 1) {
                m_core.ResetSettingsToRecommended();
                m_core.PushToast("good", "Balanced applied.");
                return;
            }
            const char* json = "{}";
            switch (m_quickChoice) {
                case 0: json = "{\"res\":16,\"batch\":16,\"goal\":2000,\"schedule\":\"cosine\",\"warmup\":50,\"lr\":0.002}"; break;
                case 2: json = "{\"res\":64,\"batch\":4,\"goal\":50000,\"schedule\":\"cosine\",\"warmup\":200,\"lr\":0.0007}"; break;
                default:
                    json = "{\"res\":64,\"batch\":8,\"goal\":0,\"schedule\":\"cosine\",\"warmup\":100,\"lr\":0.001,"
                           "\"resize\":\"sharp\",\"crisp\":32,\"zoom\":false,\"color\":false}";
                    break;
            }
            JsonValue patch;
            if (JsonParser::Parse(json, patch) && m_core.PatchSettings(patch, nullptr)) {
                m_core.PushToast("good", std::string(kQuickNames[std::clamp(m_quickChoice, 0, 3)]) + " applied.");
            }
        });

    add("Stop after", SettingRow::Kind::Choice,
        "Training stops by itself after this many lessons. 'None' keeps going until you pause it. With a number set, the learning speed fades out smoothly, which finishes with cleaner pictures.",
        [this]() {
            const uint64_t g = m_core.GetConfig().targetSteps;
            return g == 0 ? std::string("None") : Int(g) + " lessons";
        },
        [this](int dir) {
            const double now = static_cast<double>(m_core.GetConfig().targetSteps);
            m_core.SetNumberSetting("goal", NextInList(kGoalValues, Count(kGoalValues), now, dir, dir == 0));
        });

    add("Learning speed", SettingRow::Kind::Choice,
        "How big a nudge the brain gets after each lesson. Too high and the error jumps around or the brain breaks. Too low and it learns very slowly. If unsure, leave it.",
        [this]() { return Fixed(m_core.GetConfig().learningRate, 4); },
        [this](int dir) {
            const double now = static_cast<double>(m_core.GetConfig().learningRate);
            m_core.SetNumberSetting("lr", NextInList(kRateValues, Count(kRateValues), now, dir, dir == 0));
        });

    add("Picture size", SettingRow::Kind::Choice,
        "The brain learns and draws pictures this many pixels wide and tall. Bigger shows more detail but is much slower: doubling the size is about four times the work. If you change it during training, the error may jump for a while.",
        [this]() { return std::to_string(m_core.GetConfig().resolution) + " px"; },
        [this](int dir) {
            const double now = static_cast<double>(m_core.GetConfig().resolution);
            m_core.SetNumberSetting("res", NextInList(kSizeValues, Count(kSizeValues), now, dir, dir == 0));
        });

    add("Pictures per lesson", SettingRow::Kind::Choice,
        "How many pictures the brain studies in each lesson. More is steadier but slower. At 128 pixels the Xbox allows only a few per lesson.",
        [this]() { return std::to_string(m_core.GetConfig().batchSize); },
        [this](int dir) {
            const double now = static_cast<double>(m_core.GetConfig().batchSize);
            m_core.SetNumberSetting("batch", NextInList(kBatchValues, Count(kBatchValues), now, dir, dir == 0));
        });

    add("Smoothed copy of the brain", SettingRow::Kind::Toggle,
        "While it trains, the brain wobbles a little after every lesson. A smoothed copy averages the wobble out, which usually draws cleaner pictures. Recommended.",
        [this]() { return std::string(m_core.GetConfig().emaEnabled ? "On" : "Off"); },
        [this](int) { m_core.SetBoolSetting("ema", !m_core.GetConfig().emaEnabled); });

    add("Mirror pictures at random", SettingRow::Kind::Toggle,
        "Randomly flips pictures left to right while training, so the brain sees more variations. Turn it off for pictures with writing.",
        [this]() { return std::string(m_core.GetConfig().augment.flip ? "On" : "Off"); },
        [this](int) { m_core.SetBoolSetting("flip", !m_core.GetConfig().augment.flip); });

    add("Enlarge small pictures", SettingRow::Kind::Choice,
        "Pictures are stored at 128 by 128 pixels. Automatic copies pixels when a small picture with few colours (pixel art) is enlarged, so the edges stay sharp, "
        "and blends the pixels of everything else. Sharp copies pixels for every picture. Smooth always blends them, which suits photos. "
        "This only affects pictures added from now on, and the phone page has the same setting.",
        [this]() {
            const ResizeStyle r = m_core.GetConfig().resize;
            return std::string(r == ResizeStyle::Auto ? "Automatic" : (r == ResizeStyle::Sharp ? "Sharp" : "Smooth"));
        },
        [this](int dir) {
            static const char* const kStyles[3] = { "auto", "sharp", "smooth" };
            const ResizeStyle r = m_core.GetConfig().resize;
            int idx = r == ResizeStyle::Auto ? 0 : (r == ResizeStyle::Sharp ? 1 : 2);
            idx = dir == 0 ? (idx + 1) % 3 : std::clamp(idx + dir, 0, 2);
            m_core.SetTextSetting("resize", kStyles[idx]);
        });

    add("Crisp colours", SettingRow::Kind::Choice,
        "Moves every pixel the brain draws to the nearest of this many colours, taken from your pictures, so pixel art gets clean flat colours. "
        "If your pictures use fewer colours than this, exactly their colours are used. It only changes how pictures are shown, not how the brain learns.",
        [this]() {
            const uint32_t n = m_core.GetConfig().crispColors;
            return n == 0 ? std::string("Off") : std::to_string(n) + " colours";
        },
        [this](int dir) {
            const double now = static_cast<double>(m_core.GetConfig().crispColors);
            m_core.SetNumberSetting("crisp", NextInList(kCrispValues, Count(kCrispValues), now, dir, dir == 0));
        });

    add("Save by itself", SettingRow::Kind::Choice,
        "While training, the Xbox saves a snapshot this often and keeps the newest four. When the app opens again it carries on from the newest snapshot.",
        [this]() {
            const uint32_t m = m_core.GetConfig().autosaveMinutes;
            return m == 0 ? std::string("Never") : std::to_string(m) + " min";
        },
        [this](int dir) {
            const double now = static_cast<double>(m_core.GetConfig().autosaveMinutes);
            m_core.SetNumberSetting("autosaveMin", NextInList(kSaveValues, Count(kSaveValues), now, dir, dir == 0));
        });

    add("Leave room for the TV screen", SettingRow::Kind::Toggle,
        "Adds tiny pauses so the picture on the TV stays smooth while training. Training runs a little slower.",
        [this]() { return std::string(m_core.GetConfig().gentle ? "On" : "Off"); },
        [this](int) { m_core.SetBoolSetting("gentle", !m_core.GetConfig().gentle); });

    add("Speed mode", SettingRow::Kind::Toggle,
        "Skips measuring the error after each lesson. A little faster, but the chart stays empty and the Xbox can't notice if the brain breaks. Most people should leave it off.",
        [this]() { return std::string(m_core.GetConfig().blindSpeedMode ? "On" : "Off"); },
        [this](int) { m_core.SetBoolSetting("blind", !m_core.GetConfig().blindSpeedMode); });

    add("Calculate on", SettingRow::Kind::Choice,
        "The graphics chip is much faster. The processor always works but is slow, so it is mostly a fallback.",
        [this]() { return std::string(m_core.GetConfig().engine == ComputeEngine::GPU ? "Graphics chip" : "Processor"); },
        [this](int) { m_core.SetTextSetting("engine", m_core.GetConfig().engine == ComputeEngine::GPU ? "cpu" : "gpu"); });

    add("Type of new brain", SettingRow::Kind::Choice,
        "A normal brain works on the pixels of your pictures. A wave brain works on a compressed form of every picture (a half-size copy "
        "plus its small details). Nothing is lost, and each lesson takes about 4 times less work, so it trains faster. A one-pass brain "
        "squeezes each picture into a small code and learns to draw it back, then makes new pictures from random codes in a single go. "
        "It has 1 to 8 million dials, and what it draws is usually softer. Whether any of them draws better is for you to compare. "
        "This only matters when you start over with a new brain, below.",
        [this]() { return std::string(BrainTypeName(m_typeChoice)); },
        [this](int dir) {
            m_typeChoiceTouched = true;
            m_typeChoice = dir == 0 ? (m_typeChoice + 1) % 3 : std::clamp(m_typeChoice + dir, 0, 2); // A goes round, Left and Right stop at the ends
        });

    add("Start over with a new brain", SettingRow::Kind::Action,
        "A bigger brain can hold more detail but trains slower. A normal brain has about 25,000 (Small), 80,000 (Medium) or 250,000 (Large) dials. "
        "A one-pass brain has about 1, 3 or 8 million. The type above says which kind it is. "
        "This makes a brand-new brain that knows nothing and restarts the chart. Your pictures and settings stay. "
        "Anything the current brain learned is first saved as a safety copy.",
        [this]() {
            const uint32_t p = static_cast<uint32_t>(std::clamp(m_brainChoice, 0, 2));
            return std::string("<  ") + BrainFullName(p, static_cast<uint32_t>(std::clamp(m_typeChoice, 0, 2))) + "  >";
        },
        [this](int dir) {
            if (dir != 0) { m_brainChoiceTouched = true; m_brainChoice = std::clamp(m_brainChoice + dir, 0, 2); return; }
            const uint32_t preset = static_cast<uint32_t>(std::clamp(m_brainChoice, 0, 2));
            const uint32_t type = static_cast<uint32_t>(std::clamp(m_typeChoice, 0, 2));
            const uint32_t wave = type == 1 ? 1u : 0u;
            const uint32_t family = type == 2 ? 1u : 0u;
            OpenDialog(std::string("Start over with a new ") + BrainFullName(preset, type) + " brain?",
                       { std::string("This makes a brand-new brain with ") + BrainDials(preset, type) + ". It knows nothing, and the chart starts again. Your pictures and settings stay.",
                         "If the current brain has learned anything, it is saved first as a safety copy that you can load back on the Saves screen." },
                       "Start over", true,
                       [this, preset, wave, family]() {
                           if (BusyForActions()) { m_core.PushToast("info", "One moment, the Xbox is busy."); return; }
                           StartJob("Starting over...", [this, preset, wave, family]() {
                               std::string err;
                               if (!m_core.StartOver(preset, &err, wave, family)) m_core.PushToast("warn", "Could not start over: " + err);
                               else m_pendingTab.store(static_cast<int>(TvTab::Train));
                           });
                       });
        }, true);

    add("Reset all settings", SettingRow::Kind::Action,
        "Every setting goes back to the recommended value. The brain, your pictures and your saves are not touched.",
        []() { return std::string("press A"); },
        [this](int dir) {
            if (dir != 0) return;
            OpenDialog("Reset all settings?",
                       { "Every setting goes back to the recommended value. The brain, your pictures and your saves are not touched." },
                       "Reset settings", false,
                       [this]() { m_core.ResetSettingsToRecommended(); });
        });
}

// ---------------------------------------------------------------------------------------------
// Drawing
// ---------------------------------------------------------------------------------------------
void TvModel::BuildScene(TvScene& scene) {
    scene.items.clear();
    scene.items.reserve(600);
    TvBuilder b(scene);
    b.Fill(0, 0, kTvWidth, kTvHeight, TvColor::Background);
    DrawTopBar(b);
    switch (static_cast<TvTab>(m_tab)) {
        case TvTab::Train: DrawTrain(b); break;
        case TvTab::Settings: DrawSettings(b); break;
        case TvTab::Saves: DrawSaves(b); break;
        case TvTab::Pictures: DrawPictures(b); break;
        case TvTab::Phone: DrawPhone(b); break;
    }
    DrawFooter(b);
    DrawToasts(b);
    if (m_viewerOpen) DrawViewer(b);
    if (m_dialog.open) DrawDialog(b);
}

void TvModel::DrawTopBar(TvBuilder& b) const {
    // logo: four small squares
    b.Fill(kMarginX, 58, 20, 20, TvColor::Text, 4);
    b.Fill(kMarginX + 24, 58, 20, 20, TvColor::Muted, 4);
    b.Fill(kMarginX, 82, 20, 20, TvColor::Line, 4);
    b.Fill(kMarginX + 24, 82, 20, 20, TvColor::Accent, 4);
    b.Text(kMarginX + 64, 56, 280, 52, "Image Trainer", TvFont::Heading, TvColor::Text);

    static const char* const kNames[kTvTabCount] = { "Train", "Settings", "Saves", "Pictures", "Phone" };
    const float tabsX = 496.0f, tabW = 176.0f;
    b.Outline(430, 64, 56, 40, TvColor::Muted, 2.0f, 10.0f);
    b.Text(430, 68, 56, 34, "LB", TvFont::Small, TvColor::Muted, TvAlign::Center);
    for (int i = 0; i < kTvTabCount; ++i) {
        const float x = tabsX + static_cast<float>(i) * tabW;
        const bool on = i == m_tab;
        if (on) {
            b.Fill(x + 4, 56, tabW - 8, 56, TvColor::PanelRaised, 12);
            b.Fill(x + 24, 106, tabW - 48, 5, TvColor::Accent, 2);
        }
        b.Text(x, 66, tabW, 40, kNames[i], on ? TvFont::BodyBold : TvFont::Body, on ? TvColor::Text : TvColor::Muted, TvAlign::Center);
    }
    const float rbX = tabsX + static_cast<float>(kTvTabCount) * tabW + 12;
    b.Outline(rbX, 64, 56, 40, TvColor::Muted, 2.0f, 10.0f);
    b.Text(rbX, 68, 56, 34, "RB", TvFont::Small, TvColor::Muted, TvAlign::Center);

    // status pill
    std::string label = m_snap.stateText;
    TvColor color = TvColor::Muted;
    switch (m_snap.state) {
        case RunState::Training: color = TvColor::Good; break;
        case RunState::Finished: color = TvColor::Accent; break;
        case RunState::Problem: color = TvColor::Bad; break;
        default: break;
    }
    std::string job;
    {
        std::lock_guard<std::mutex> lock(m_jobMutex);
        job = m_jobLabel;
    }
    if (m_jobBusy.load() && !job.empty()) { label = job; color = TvColor::Warn; }
    else if (m_snap.busy && !m_snap.busyText.empty()) { label = m_snap.busyText; color = TvColor::Warn; }
    const float pillX = 1480.0f, pillW = kRightEdge - pillX;
    b.Fill(pillX, 60, pillW, 48, TvColor::Panel, 24);
    b.Outline(pillX, 60, pillW, 48, color, 2.0f, 24);
    b.Fill(pillX + 20, 77, 14, 14, color, 7);
    b.Text(pillX + 44, 66, pillW - 56, 38, Fit(label, pillW - 64, TvFont::Body), TvFont::Body, TvColor::Text);
}

std::string TvModel::TrendText() const {
    if (m_snap.config.blindSpeedMode) return "Speed mode: error not measured";
    const size_t n = m_curve.size();
    if (n < 8) return "Too early to tell";
    const size_t q = std::max<size_t>(2, n / 4);
    double recent = 0.0, before = 0.0;
    for (size_t i = n - q; i < n; ++i) recent += m_curve[i].second;
    for (size_t i = n - 2 * q; i < n - q; ++i) before += m_curve[i].second;
    recent /= static_cast<double>(q);
    before /= static_cast<double>(q);
    if (before <= 0.0) return "Too early to tell";
    if (recent < before * 0.97) return "Mistakes are falling";
    if (recent > before * 1.10) return "Mistakes are rising";
    return "Mistakes have levelled off";
}

void TvModel::DrawTrain(TvBuilder& b) const {
    const AppSnapshot& s = m_snap;
    const float lx = kMarginX, lw = 960.0f;

    // headline
    b.Fill(lx, kContentTop, lw, 170, TvColor::Panel, 16);
    b.Text(lx + 32, kContentTop + 6, 460, 100, Int(s.metrics.stepCount), TvFont::Display, TvColor::Text);
    b.Text(lx + 32, kContentTop + 108, 460, 36, "lessons learned so far", TvFont::Small, TvColor::Muted);
    TvColor stateColor = TvColor::Text;
    if (s.state == RunState::Training) stateColor = TvColor::Good;
    else if (s.state == RunState::Problem) stateColor = TvColor::Bad;
    else if (s.state == RunState::Finished) stateColor = TvColor::Accent;
    b.Text(lx + 500, kContentTop + 18, 430, 56, Fit(s.stateText, 430, TvFont::Title), TvFont::Title, stateColor);
    b.Text(lx + 500, kContentTop + 78, 430, 36, Fit(TrendText(), 430, TvFont::Body), TvFont::Body, TvColor::Muted);
    std::string errorLine;
    if (!s.config.blindSpeedMode && s.metrics.smoothLoss > 0.0f) {
        errorLine = "Error " + Fixed(s.metrics.smoothLoss, 3);
        if (s.metrics.bestSmoothLoss > 0.0f) errorLine += "   (best " + Fixed(s.metrics.bestSmoothLoss, 3) + ")";
    }
    b.Text(lx + 500, kContentTop + 114, 430, 36, Fit(errorLine, 430, TvFont::Small), TvFont::Small, TvColor::Muted);

    DrawChart(b, lx, 346, lw, 300);

    // four numbers
    const float tileW = (lw - 3 * 16.0f) / 4.0f;
    struct Tile { std::string label, value; };
    std::string eta;
    if (s.state == RunState::Training && s.etaSeconds >= 0.0) {
        const std::string d = Duration(s.etaSeconds);
        eta = ("about " + d).size() <= 14 ? "about " + d : d;
    }
    else if (s.config.targetSteps == 0) eta = "no goal set";
    else if (s.state == RunState::Finished) eta = "done";
    else eta = "paused";
    std::string saved = s.lastSavedText;
    if (saved.rfind("Saved ", 0) == 0) saved = saved.substr(6);
    const Tile tiles[4] = {
        { "Speed", s.metrics.stepsPerSec > 0.0f ? Fixed(s.metrics.stepsPerSec, s.metrics.stepsPerSec < 10.0f ? 1 : 0) + " lessons/s" : "-" },
        { "Time left", eta },
        { "Pictures", s.practice ? std::string("none yet") : Int(s.pictureCount) + " of " + Int(s.pictureCapacity) },
        { "Last save", saved }
    };
    for (int i = 0; i < 4; ++i) {
        const float x = lx + static_cast<float>(i) * (tileW + 16.0f);
        b.Fill(x, 666, tileW, 100, TvColor::Panel, 14);
        b.Text(x + 20, 676, tileW - 40, 32, tiles[i].label, TvFont::Small, TvColor::Muted);
        const TvFont f = tiles[i].value.size() > 12 ? TvFont::Small : TvFont::Body;
        b.Text(x + 20, 712, tileW - 40, 44, Fit(tiles[i].value, tileW - 40, f), f, TvColor::Text);
    }

    DrawNotice(b, lx, 786, lw, kContentBottom - 786);
    DrawSamplePanel(b, 1088, kContentTop, 744, kContentBottom - kContentTop);
}

void TvModel::DrawChart(TvBuilder& b, float x, float y, float w, float h) const {
    b.Fill(x, y, w, h, TvColor::Panel, 16);
    b.Text(x + 32, y + 14, w - 64, 34, "Mistakes while learning (lower is better)", TvFont::Small, TvColor::Muted);
    const float px0 = x + 132, px1 = x + w - 40, py0 = y + 66, py1 = y + h - 58;

    std::string message;
    if (m_snap.config.blindSpeedMode) message = "Speed mode is on, so the error is not measured and there is no chart.";
    else if (m_curve.size() < 2) message = "The chart appears after the first few lessons.";
    if (!message.empty()) {
        b.Text(x + 60, y + 110, w - 120, 120, message, TvFont::Body, TvColor::Muted, TvAlign::Center, true);
        return;
    }

    double lo = 1e300, hi = -1e300;
    for (const auto& p : m_curve) {
        const double v = p.second;
        if (!std::isfinite(v) || v <= 0.0) continue;
        lo = std::min(lo, v);
        hi = std::max(hi, v);
    }
    if (!(hi >= lo)) {
        b.Text(x + 60, y + 110, w - 120, 120, "No readable numbers yet.", TvFont::Body, TvColor::Muted, TvAlign::Center, true);
        return;
    }
    if (hi <= lo * 1.0001) hi = lo * 1.1;
    const bool logScale = hi / lo >= 6.0;
    auto norm = [&](double v) {
        const double t = logScale ? (std::log(v) - std::log(lo)) / (std::log(hi) - std::log(lo)) : (v - lo) / (hi - lo);
        return std::clamp(t, 0.0, 1.0);
    };
    const uint64_t s0 = m_curve.front().first, s1 = std::max(m_curve.back().first, s0 + 1);

    for (int g = 0; g < 3; ++g) {
        const float gy = py1 - (py1 - py0) * static_cast<float>(g) / 2.0f;
        b.Lines({ px0, gy, px1, gy }, TvColor::ChartGrid, 2.0f);
        const double value = logScale ? std::exp(std::log(lo) + (std::log(hi) - std::log(lo)) * g / 2.0) : lo + (hi - lo) * g / 2.0;
        b.Text(x + 16, gy - 16, 108, 32, Fixed(value, value >= 10 ? 1 : value >= 1 ? 2 : 3), TvFont::Small, TvColor::Muted, TvAlign::Right);
    }
    std::vector<float> pts;
    pts.reserve(m_curve.size() * 2);
    for (const auto& p : m_curve) {
        if (!std::isfinite(p.second) || p.second <= 0.0f) continue;
        const float fx = px0 + (px1 - px0) * static_cast<float>(static_cast<double>(p.first - s0) / static_cast<double>(s1 - s0));
        const float fy = py1 - (py1 - py0) * static_cast<float>(norm(p.second));
        pts.push_back(fx);
        pts.push_back(fy);
    }
    b.Lines(pts, TvColor::ChartLine, 4.0f);
    if (pts.size() >= 2) b.Fill(pts[pts.size() - 2] - 7, pts[pts.size() - 1] - 7, 14, 14, TvColor::Accent, 7);
    b.Text(px0, py1 + 10, 300, 32, "lesson " + Int(s0), TvFont::Small, TvColor::Muted);
    b.Text(px1 - 300, py1 + 10, 300, 32, "lesson " + Int(s1), TvFont::Small, TvColor::Muted, TvAlign::Right);
    if (logScale) b.Text(x + 32, y + h - 40, 300, 30, "log scale", TvFont::Small, TvColor::Muted);
}

void TvModel::DrawSamplePanel(TvBuilder& b, float x, float y, float w, float h) const {
    b.Fill(x, y, w, h, TvColor::Panel, 16);
    b.Text(x + 28, y + 14, w - 56, 34, "What the brain draws from random static", TvFont::Small, TvColor::Muted);
    const float ax = x + 24, ay = y + 60, aw = w - 48, ah = h - 60 - 96;
    if (!m_preview || !m_preview->Valid()) {
        b.Outline(ax, ay + ah / 2 - 140, aw, 280, TvColor::Line, 2.0f, 12.0f);
        b.Text(ax + 40, ay + ah / 2 - 60, aw - 80, 120, "The first pictures appear after the first lessons.", TvFont::Body, TvColor::Muted, TvAlign::Center, true);
        return;
    }
    const float gw = static_cast<float>(m_preview->w), gh = static_cast<float>(m_preview->h);
    const float scale = std::min(aw / gw, ah / gh);
    const float dw = gw * scale, dh = gh * scale;
    const float dx = ax + (aw - dw) / 2, dy = ay + (ah - dh) / 2;
    b.Image(dx, dy, dw, dh, m_preview, 0x5052455600000000ull ^ m_previewShown);
    b.Outline(dx - 2, dy - 2, dw + 4, dh + 4, TvColor::Line, 2.0f);
    std::string caption = "Drawn at lesson " + Int(m_previewStep);
    caption += m_previewDreamSteps > 0 ? " in " + std::to_string(m_previewDreamSteps) + " steps" : std::string(" in one pass");
    if (m_previewAverage) caption += ", with the smoothed copy";
    b.Text(x + 28, y + h - 84, w - 56, 70, caption, TvFont::Small, TvColor::Muted, TvAlign::Center, true);
}

void TvModel::DrawNotice(TvBuilder& b, float x, float y, float w, float h) const {
    const AppSnapshot& s = m_snap;
    TvColor color = TvColor::Muted;
    std::string title, text;
    if (s.state == RunState::Problem) {
        color = TvColor::Bad;
        title = "Training stopped and needs attention";
        text = (s.problem.empty() ? std::string("Something went wrong.") : s.problem) + " Open Saves to load an earlier save, or Settings to start over.";
    } else if (s.state == RunState::Finished) {
        color = TvColor::Accent;
        title = "Goal reached";
        text = "The brain finished " + Int(s.metrics.stepCount) + " lessons. Press A to train more, or open Pictures to see what it can draw.";
    } else if (s.practice) {
        color = TvColor::Warn;
        title = "Practising on built-in shapes";
        text = "You haven't added pictures yet. Press View to see how to add your own from your phone.";
    } else if (s.pictureCount < 12) {
        color = TvColor::Warn;
        title = "Only " + Int(s.pictureCount) + (s.pictureCount == 1 ? " picture" : " pictures");
        text = "With so few pictures the brain mostly memorises them. Twenty or more pictures of one kind of thing work much better.";
    } else if (!s.picturesPersistent) {
        color = TvColor::Warn;
        title = "Pictures aren't being saved";
        text = "The Xbox couldn't write the pictures to its storage. They work for now but will be gone when the app closes.";
    } else if (s.config.engine == ComputeEngine::GPU && !s.gpuAvailable) {
        color = TvColor::Warn;
        title = "Using the processor, which is slower";
        text = s.gpuNote.empty() ? std::string("The graphics chip is not available.") : s.gpuNote;
    } else if (s.config.blindSpeedMode) {
        color = TvColor::Muted;
        title = "Speed mode is on";
        text = "The error isn't measured, so the chart stays empty. You can turn this off in Settings.";
    } else if (s.metrics.stepCount == 0) {
        color = TvColor::Good;
        title = "Ready";
        text = "Press A to start training. The picture on the right shows what the brain would draw.";
    } else {
        return;
    }
    b.Fill(x, y, w, h, TvColor::Panel, 14);
    b.Fill(x, y + 10, 8, h - 20, color, 4);
    b.Text(x + 32, y + 10, w - 60, 36, Fit(title, w - 60, TvFont::BodyBold), TvFont::BodyBold, TvColor::Text);
    b.Text(x + 32, y + 48, w - 60, h - 54, text, TvFont::Small, TvColor::Muted, TvAlign::Left, true);
}

void TvModel::DrawSettings(TvBuilder& b) const {
    const int tab = static_cast<int>(TvTab::Settings);
    const int count = static_cast<int>(m_rows.size());
    const float lx = kMarginX, lw = 1040.0f, rowH = 72.0f;
    b.Fill(lx, kContentTop, lw, kContentBottom - kContentTop, TvColor::Panel, 16);
    const int first = m_scroll[tab];
    for (int i = 0; i < kSettingsVisible; ++i) {
        const int idx = first + i;
        if (idx >= count) break;
        const SettingRow& row = m_rows[static_cast<size_t>(idx)];
        const float y = kContentTop + 18 + static_cast<float>(i) * rowH;
        const bool on = idx == m_focus[tab];
        if (on) {
            b.Fill(lx + 14, y, lw - 52, rowH - 6, TvColor::PanelRaised, 12);
            b.Outline(lx + 14, y, lw - 52, rowH - 6, TvColor::Accent, 3.0f, 12.0f);
        }
        b.Text(lx + 40, y + 16, 520, 40, Fit(row.label, 520, TvFont::Body), on ? TvFont::BodyBold : TvFont::Body, row.danger ? TvColor::Warn : TvColor::Text);
        const std::string value = row.value();
        if (row.kind == SettingRow::Kind::Toggle) {
            const bool isOn = value == "On";
            b.Fill(lx + lw - 52 - 150, y + 14, 120, 40, isOn ? TvColor::Good : TvColor::Line, 20);
            b.Text(lx + lw - 52 - 150, y + 18, 120, 34, value, TvFont::Small, isOn ? TvColor::OnAccent : TvColor::Text, TvAlign::Center);
        } else {
            b.Text(lx + 560, y + 16, lw - 52 - 560 - 30, 40, Fit(value, lw - 52 - 590, TvFont::Body), TvFont::Body, on ? TvColor::Text : TvColor::Muted, TvAlign::Right);
        }
    }
    if (count > kSettingsVisible) {
        const float trackY = kContentTop + 18, trackH = kSettingsVisible * rowH - 6;
        b.Fill(lx + lw - 24, trackY, 6, trackH, TvColor::Line, 3);
        const float thumbH = std::max(40.0f, trackH * static_cast<float>(kSettingsVisible) / static_cast<float>(count));
        const float room = trackH - thumbH;
        const float thumbY = trackY + room * static_cast<float>(first) / static_cast<float>(std::max(1, count - kSettingsVisible));
        b.Fill(lx + lw - 24, thumbY, 6, thumbH, TvColor::Muted, 3);
    }

    // explanation of the highlighted setting
    const float rx = lx + lw + 32, rw = kRightEdge - rx;
    b.Fill(rx, kContentTop, rw, kContentBottom - kContentTop, TvColor::Panel, 16);
    const int f = m_focus[tab];
    if (f >= 0 && f < count) {
        const SettingRow& row = m_rows[static_cast<size_t>(f)];
        b.Text(rx + 32, kContentTop + 20, rw - 64, 32, "About this setting", TvFont::Small, TvColor::Muted);
        b.Text(rx + 32, kContentTop + 58, rw - 64, 90, row.label, TvFont::Heading, TvColor::Text, TvAlign::Left, true);
        b.Text(rx + 32, kContentTop + 160, rw - 64, 520, row.help, TvFont::Body, TvColor::Muted, TvAlign::Left, true);
        const char* how = row.kind == SettingRow::Kind::Toggle ? "Press A, Left or Right to switch it."
                        : row.kind == SettingRow::Kind::Choice ? "Press Left or Right to change it. A moves to the next value."
                        : "Left and Right choose. A applies.";
        b.Text(rx + 32, kContentBottom - 90, rw - 64, 70, how, TvFont::Small, TvColor::Accent, TvAlign::Left, true);
    }
}

void TvModel::DrawSaves(TvBuilder& b) const {
    const int tab = static_cast<int>(TvTab::Saves);
    const int count = static_cast<int>(m_saves.size());
    const float lx = kMarginX, lw = 1100.0f, rowH = 94.0f;
    b.Fill(lx, kContentTop, lw, kContentBottom - kContentTop, TvColor::Panel, 16);
    if (count == 0) {
        b.Text(lx + 60, kContentTop + 200, lw - 120, 260,
               "No saves yet. Press Y to save right now. The Xbox also saves by itself while training, so you can pick up where you left off.",
               TvFont::Body, TvColor::Muted, TvAlign::Center, true);
    }
    const int first = m_scroll[tab];
    for (int i = 0; i < kSavesVisible; ++i) {
        const int idx = first + i;
        if (idx >= count) break;
        const CheckpointMeta& m = m_saves[static_cast<size_t>(idx)];
        const float y = kContentTop + 16 + static_cast<float>(i) * rowH;
        const bool on = idx == m_focus[tab];
        if (on) {
            b.Fill(lx + 14, y, lw - 52, rowH - 8, TvColor::PanelRaised, 12);
            b.Outline(lx + 14, y, lw - 52, rowH - 8, TvColor::Accent, 3.0f, 12.0f);
        }
        const std::string name = m.name.empty() ? KindLabel(m.kind) : m.name;
        b.Text(lx + 40, y + 8, lw - 120, 40, Fit(name, lw - 120, TvFont::BodyBold), TvFont::BodyBold, TvColor::Text);
        std::string sub = "Lesson " + Int(m.step) + "   -   " + KindLabel(m.kind);
        const std::string ago = AgoFromUnix(m.createdUnix);
        if (!ago.empty()) sub += "   -   " + ago;
        sub += "   -   " + Bytes(m.fileBytes);
        b.Text(lx + 40, y + 48, lw - 120, 34, Fit(sub, lw - 120, TvFont::Small), TvFont::Small, TvColor::Muted);
    }
    if (count > kSavesVisible) {
        const float trackY = kContentTop + 16, trackH = kSavesVisible * rowH - 8;
        b.Fill(lx + lw - 24, trackY, 6, trackH, TvColor::Line, 3);
        const float thumbH = std::max(40.0f, trackH * static_cast<float>(kSavesVisible) / static_cast<float>(count));
        const float thumbY = trackY + (trackH - thumbH) * static_cast<float>(first) / static_cast<float>(std::max(1, count - kSavesVisible));
        b.Fill(lx + lw - 24, thumbY, 6, thumbH, TvColor::Muted, 3);
    }

    const float rx = lx + lw + 32, rw = kRightEdge - rx;
    b.Fill(rx, kContentTop, rw, kContentBottom - kContentTop, TvColor::Panel, 16);
    b.Text(rx + 28, kContentTop + 20, rw - 56, 32, "Saves", TvFont::Small, TvColor::Muted);
    b.Text(rx + 28, kContentTop + 56, rw - 56, 260,
           "The Xbox saves by itself while training and keeps the newest few. A save holds the brain and its progress, so you can go back to it later.",
           TvFont::Body, TvColor::Muted, TvAlign::Left, true);
    const int f = m_focus[tab];
    if (f >= 0 && f < count) {
        const CheckpointMeta& m = m_saves[static_cast<size_t>(f)];
        b.Text(rx + 28, kContentTop + 330, rw - 56, 32, "Highlighted save", TvFont::Small, TvColor::Muted);
        std::string d = BrainFullName(m.preset, m.family == 1 ? 2u : (m.wave != 0 ? 1u : 0u)) + " brain, " + std::to_string(m.resolution) + " px pictures\n";
        if (m.loss > 0.0) d += "Error then: " + Fixed(m.loss, 3) + "\n";
        d += Int(m.pictures) + " pictures in the set";
        b.Text(rx + 28, kContentTop + 366, rw - 56, 200, d, TvFont::Body, TvColor::Text, TvAlign::Left, true);
    }
}

void TvModel::DrawPictures(TvBuilder& b) const {
    const int tab = static_cast<int>(TvTab::Pictures);
    b.Text(kMarginX, kContentTop, 1100, 50, "Pictures the brain made", TvFont::Heading, TvColor::Text);
    std::string yours = m_snap.practice ? "You haven't added pictures yet. Add them from your phone."
                                        : "Your set: " + Int(m_snap.pictureCount) + " of " + Int(m_snap.pictureCapacity) + " pictures. Add or remove them from your phone.";
    b.Text(kMarginX, kContentTop + 48, 1500, 36, yours, TvFont::Small, TvColor::Muted);
    const int n = static_cast<int>(m_dreams.size());
    if (n == 0) {
        b.Fill(kMarginX, 260, kRightEdge - kMarginX, 560, TvColor::Panel, 16);
        b.Text(kMarginX + 120, 400, kRightEdge - kMarginX - 240, 300,
               "No pictures yet. Press X and the brain draws six new ones from random static. The longer it has trained, the better they look.",
               TvFont::Body, TvColor::Muted, TvAlign::Center, true);
        return;
    }
    const float tile = 240.0f, gap = 24.0f;
    const float startX = (kTvWidth - (kPictureColumns * tile + (kPictureColumns - 1) * gap)) / 2.0f;
    for (int i = 0; i < n; ++i) {
        const int col = i % kPictureColumns, row = i / kPictureColumns;
        const float x = startX + static_cast<float>(col) * (tile + gap);
        const float y = 262.0f + static_cast<float>(row) * (tile + 74.0f);
        const DreamTile& t = m_dreams[static_cast<size_t>(i)];
        b.Fill(x - 6, y - 6, tile + 12, tile + 12, TvColor::Panel, 8);
        b.Image(x, y, tile, tile, t.image, 0x4452450000000000ull ^ t.id);
        if (i == m_focus[tab]) b.Outline(x - 8, y - 8, tile + 16, tile + 16, TvColor::Accent, 4.0f, 8.0f);
        b.Text(x, y + tile + 10, tile, 32, "Lesson " + Int(t.step), TvFont::Small, TvColor::Muted, TvAlign::Center);
    }
}

void TvModel::DrawPhone(TvBuilder& b) const {
    b.Text(kMarginX, kContentTop, 1100, 60, "Control it from your phone", TvFont::Title, TvColor::Text);
    const bool haveNet = !m_snap.lanUrl.empty() && m_snap.lanUrl.rfind("http://127.0.0.1", 0) != 0;
    const float lw = 1000.0f;
    b.Text(kMarginX, kContentTop + 90, lw, 360,
           "1.  Connect your phone to the same Wi-Fi as this Xbox.\n\n"
           "2.  Point your phone's camera at the code, or type the address below into your phone's browser.\n\n"
           "3.  On the page you can add pictures, start training and change every setting.",
           TvFont::Body, TvColor::Text, TvAlign::Left, true);
    b.Fill(kMarginX, 640, lw, 130, TvColor::Panel, 16);
    if (haveNet) {
        b.Text(kMarginX + 30, 650, lw - 60, 34, "Address", TvFont::Small, TvColor::Muted);
        b.Text(kMarginX + 30, 686, lw - 60, 70, m_snap.lanUrl, TvFont::Title, TvColor::Accent);
    } else {
        b.Text(kMarginX + 30, 654, lw - 60, 100, "The Xbox is not connected to a network yet. Connect it to Wi-Fi or a network cable, then this screen shows the address.",
               TvFont::Body, TvColor::Warn, TvAlign::Left, true);
    }
    b.Text(kMarginX, 800, lw, 100, "Anyone on your Wi-Fi who knows this address can open the page, so use it on a network you trust. Keep this app open on the TV while it trains.",
           TvFont::Small, TvColor::Muted, TvAlign::Left, true);

    const float qx = 1210.0f, qy = kContentTop + 30, qs = 560.0f;
    b.Fill(qx - 20, qy - 20, qs + 40, qs + 40, TvColor::Text, 20);
    if (haveNet && m_qrImage && m_qrImage->Valid()) {
        b.Image(qx, qy, qs, qs, m_qrImage, m_qrKey);
    } else {
        b.Text(qx, qy + 200, qs, 160, "No code yet", TvFont::Heading, TvColor::Background, TvAlign::Center);
    }
}

void TvModel::DrawFooter(TvBuilder& b) const {
    b.Fill(kMarginX, 950, kRightEdge - kMarginX, 2, TvColor::Line);
    const float y = 972.0f;
    float x = kMarginX;
    switch (static_cast<TvTab>(m_tab)) {
        case TvTab::Train: {
            const char* a = m_snap.state == RunState::Training ? "Pause" : (m_snap.state == RunState::Finished ? "Train more" : "Start training");
            x = b.Hint(x, y, "A", TvColor::Good, a);
            x = b.Hint(x, y, "X", TvColor::ChartLine, "Make pictures");
            x = b.Hint(x, y, "Y", TvColor::Warn, "Save now");
            x = b.Chip(x, y, "LB  RB", "Switch screen");
            break;
        }
        case TvTab::Settings:
            x = b.Chip(x, y, "D-pad", "Choose and change");
            x = b.Hint(x, y, "A", TvColor::Good, "Apply");
            x = b.Chip(x, y, "LB  RB", "Switch screen");
            break;
        case TvTab::Saves:
            x = b.Hint(x, y, "A", TvColor::Good, "Load");
            x = b.Hint(x, y, "X", TvColor::ChartLine, "Delete");
            x = b.Hint(x, y, "Y", TvColor::Warn, "Save now");
            x = b.Chip(x, y, "LB  RB", "Switch screen");
            break;
        case TvTab::Pictures:
            x = b.Hint(x, y, "A", TvColor::Good, "Enlarge");
            x = b.Hint(x, y, "X", TvColor::ChartLine, "Make pictures");
            x = b.Chip(x, y, "LB  RB", "Switch screen");
            break;
        case TvTab::Phone:
            x = b.Chip(x, y, "LB  RB", "Switch screen");
            x = b.Chip(x, y, "View", "Back to training");
            break;
    }
    if (m_tab != static_cast<int>(TvTab::Phone) && !m_snap.lanUrl.empty()) {
        const std::string url = "Phone: " + m_snap.lanUrl;
        b.Text(std::max(x, 1100.0f), y + 6, kRightEdge - std::max(x, 1100.0f), 40, Fit(url, kRightEdge - std::max(x, 1100.0f), TvFont::Small), TvFont::Small, TvColor::Muted, TvAlign::Right);
    }
}

void TvModel::DrawToasts(TvBuilder& b) const {
    // Small notes in the bottom right corner (newest at the bottom). They sit over the least important part of every screen.
    const float w = 700.0f;
    const float x = kRightEdge - w;
    const float textW = w - 60.0f;
    float bottom = kContentBottom - 8.0f;
    int shown = 0;
    for (auto it = m_toasts.rbegin(); it != m_toasts.rend() && shown < 3; ++it, ++shown) {
        TvColor c = TvColor::Accent;
        if (it->level == "good") c = TvColor::Good;
        else if (it->level == "warn") c = TvColor::Warn;
        else if (it->level == "error") c = TvColor::Bad;
        else if (it->level == "info") c = TvColor::ChartLine;
        // one line holds about this many letters; longer notes use a second line and anything beyond that is cut
        const size_t oneLine = static_cast<size_t>(textW / (TvFontSize(TvFont::Small) * 0.5f));
        const bool two = it->text.size() > oneLine;
        const float h = two ? 96.0f : 62.0f;
        const float y = bottom - h;
        b.Fill(x, y, w, h, TvColor::PanelRaised, 14);
        b.Outline(x, y, w, h, c, 2.0f, 14.0f);
        b.Fill(x, y + 10, 8, h - 20, c, 4);
        b.Text(x + 30, y + 14, textW, h - 20, Fit(it->text, textW * 2.0f - 40.0f, TvFont::Small), TvFont::Small, TvColor::Text, TvAlign::Left, two);
        bottom = y - 10.0f;
    }
}

void TvModel::DrawViewer(TvBuilder& b) const {
    b.Fill(0, 0, kTvWidth, kTvHeight, TvColor::Shade);
    const int tab = static_cast<int>(TvTab::Pictures);
    const int f = m_focus[tab];
    if (f < 0 || f >= static_cast<int>(m_dreams.size())) return;
    const DreamTile& t = m_dreams[static_cast<size_t>(f)];
    const float px = 300, py = 90, pw = 1320, ph = 900;
    b.Fill(px, py, pw, ph, TvColor::Panel, 20);
    b.Outline(px, py, pw, ph, TvColor::Line, 2.0f, 20.0f);
    b.Image(px + 40, py + 40, 760, 760, t.image, 0x4452450000000000ull ^ t.id);
    b.Outline(px + 38, py + 38, 764, 764, TvColor::Line, 2.0f);
    const float tx = px + 840, tw = pw - 880;
    b.Text(tx, py + 50, tw, 50, "Picture " + std::to_string(f + 1) + " of " + std::to_string(m_dreams.size()), TvFont::Heading, TvColor::Text);
    std::string d = "Made at lesson " + Int(t.step) + "\n\n" +
                    (t.dreamSteps == 0 ? std::string("Drawn in one pass.") : "Drawing steps: " + std::to_string(t.dreamSteps)) + "\n\n";
    d += t.usedAverage ? "Drawn with the smoothed copy of the brain." : "Drawn with the live brain.";
    d += "\n\nSize: " + std::to_string(t.image ? t.image->w : 0) + " by " + std::to_string(t.image ? t.image->h : 0) + " pixels";
    b.Text(tx, py + 120, tw, 420, d, TvFont::Body, TvColor::Muted, TvAlign::Left, true);
    float x = px + 40;
    const float y = py + ph - 78;
    x = b.Hint(x, y, "B", TvColor::Bad, "Back");
    b.Chip(x, y, "Left  Right", "Other pictures");
}

void TvModel::DrawDialog(TvBuilder& b) const {
    b.Fill(0, 0, kTvWidth, kTvHeight, TvColor::Shade);
    const float w = 1100.0f;
    std::string body;
    // how tall the text will be: about 60 letters per line, plus one empty line between paragraphs (a little generous on purpose)
    size_t bodyLines = 0;
    for (size_t i = 0; i < m_dialog.lines.size(); ++i) {
        if (i > 0) { body += "\n\n"; bodyLines += 1; }
        body += m_dialog.lines[i];
        bodyLines += std::max<size_t>(1, (m_dialog.lines[i].size() + 59) / 60);
    }
    const float lineH = TvFontSize(TvFont::Body) * 1.4f;
    const float bodyH = static_cast<float>(bodyLines) * lineH;
    const float h = std::clamp(260.0f + bodyH, 340.0f, 800.0f);
    const float x = (kTvWidth - w) / 2.0f, y = (kTvHeight - h) / 2.0f;
    b.Fill(x, y, w, h, TvColor::Panel, 20);
    b.Outline(x, y, w, h, m_dialog.danger ? TvColor::Bad : TvColor::Accent, 3.0f, 20.0f);
    b.Text(x + 48, y + 36, w - 96, 100, m_dialog.title, TvFont::Heading, TvColor::Text, TvAlign::Left, true);
    b.Text(x + 48, y + 110, w - 96, h - 110 - 110, body, TvFont::Body, TvColor::Muted, TvAlign::Left, true);
    float hx = x + 48;
    const float hy = y + h - 82;
    hx = b.Hint(hx, hy, "A", m_dialog.danger ? TvColor::Bad : TvColor::Good, m_dialog.yesLabel);
    b.Hint(hx, hy, "B", TvColor::Muted, "Cancel");
}

} // namespace XboxOTFM
