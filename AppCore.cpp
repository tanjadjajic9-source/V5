// The C file functions used below (fopen and friends) are fine to use here: every result is checked. Microsoft's compiler flags
// them as "unsafe" and, in an Xbox app, treats that as an error, so the warning is switched off before any header is read.
#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS 1
#endif

#include "AppCore.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <system_error>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>

namespace XboxOTFM {

namespace {

constexpr size_t kMaxToasts = 80;
constexpr size_t kMaxLogLines = 300;
constexpr size_t kMaxDreams = 24;
constexpr int kJobTimeoutMs = 120000;

uint64_t NowUnix() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count());
}

uint64_t NewSeed() {
    std::random_device rd;
    return (static_cast<uint64_t>(rd()) << 32) ^ static_cast<uint64_t>(rd()) ^ (NowUnix() * 0x9E3779B97F4A7C15ULL);
}

bool ReadWholeFile(const std::string& path, std::string& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::ostringstream ss;
    ss << f.rdbuf();
    out = ss.str();
    return true;
}

bool WriteFileAtomically(const std::string& path, const std::string& data) {
    const std::string tmp = path + ".tmp";
    FILE* f = std::fopen(tmp.c_str(), "wb");
    if (!f) return false;
    bool ok = std::fwrite(data.data(), 1, data.size(), f) == data.size();
    ok = (std::fflush(f) == 0) && ok;
    ok = (std::fclose(f) == 0) && ok;
    std::error_code ec;
    if (ok) {
        std::filesystem::rename(tmp, path, ec);
        ok = !ec;
    }
    if (!ok) {
        std::error_code ec2;
        std::filesystem::remove(tmp, ec2);
    }
    return ok;
}

const char* StateName(RunState s) {
    switch (s) {
        case RunState::Training: return "training";
        case RunState::Finished: return "finished";
        case RunState::Problem: return "problem";
        default: return "idle";
    }
}

std::string AgoText(double seconds) {
    char buf[64];
    if (seconds < 20.0) return "just now";
    if (seconds < 90.0) return "1 min ago";
    if (seconds < 3600.0) {
        std::snprintf(buf, sizeof(buf), "%d min ago", static_cast<int>(seconds / 60.0 + 0.5));
        return buf;
    }
    std::snprintf(buf, sizeof(buf), "%.1f h ago", seconds / 3600.0);
    return buf;
}

} // namespace

// ---------------------------------------------------------------------------------------------
// Construction / lifecycle
// ---------------------------------------------------------------------------------------------
AppCore::AppCore(std::string dataRoot, std::string appVersion, const DatasetOptions& pictureOptions)
    : m_root(std::move(dataRoot))
    , m_version(std::move(appVersion))
    , m_brain(BrainPreset::Medium80k)
    , m_dataset(m_root + "/pictures", pictureOptions)
    , m_store(m_root + "/checkpoints") {
}

AppCore::~AppCore() {
    Stop();
}

void AppCore::Start() {
    if (m_started.exchange(true)) return;
    std::error_code ec;
    std::filesystem::create_directories(m_root, ec);

    std::string err;
    if (!m_store.Init(&err)) {
        AppendLog("Snapshot folder problem: " + err);
        PushToast("warn", "Snapshots can't be saved right now: " + err);
    }
    std::vector<std::string> warnings;
    m_dataset.Open(&warnings);
    for (const std::string& w : warnings) {
        AppendLog("Pictures: " + w);
        PushToast("warn", w);
    }
    AppendLog("App " + m_version + " started. " + std::to_string(m_dataset.Size()) + " saved picture(s) found.");

    LoadSettingsFile();

    // Pick up where the last session ended (the newest snapshot that is intact). Training never starts by itself.
    const std::vector<CheckpointMeta> list = m_store.List();
    for (const CheckpointMeta& meta : list) {
        CheckpointData data;
        std::string loadErr;
        if (!m_store.Load(meta.id, data, &loadErr)) {
            AppendLog("Skipped snapshot " + meta.id + ": " + loadErr);
            continue;
        }
        std::string applyErr;
        if (DoLoadSnapshot(data, false, &applyErr)) {
            NoteSaved(data.meta.step);
            const std::string label = meta.name.empty() ? (meta.kind == "auto" ? std::string("autosave") : meta.id) : meta.name;
            PushToast("info", "Picked up where you left off (" + label + ", lesson " + std::to_string(data.meta.step) + "). Press Start to continue.");
            AppendLog("Resumed from snapshot " + meta.id + " at lesson " + std::to_string(data.meta.step));
            break;
        }
        AppendLog("Could not use snapshot " + meta.id + ": " + applyErr);
    }
    {
        std::lock_guard<std::mutex> lock(m_stateMutex);
        m_lastSaveTime = std::chrono::steady_clock::now();
    }

    m_worker = std::thread(&AppCore::WorkerMain, this);
}

void AppCore::Stop() {
    if (!m_started.load() || m_stop.load()) {
        if (m_worker.joinable()) m_worker.join();
        return;
    }
    m_wantTraining.store(false);
    // final save of unsaved progress (on the worker, so it never runs in the middle of a lesson)
    RunOnWorker([this]() {
        MaybeAutosave(true);
        SaveSettingsFileIfDirty(true);
    }, 8000);
    m_stop.store(true);
    {
        std::lock_guard<std::mutex> lock(m_jobMutex);
    }
    m_jobCv.notify_all();
    if (m_worker.joinable()) m_worker.join();
}

void AppCore::PrepareForSuspend(int maxMs) {
    if (!m_started.load() || m_stop.load()) return;
    m_wantTraining.store(false);
    RunOnWorker([this]() {
        const RunState st = [this]() { std::lock_guard<std::mutex> l(m_stateMutex); return m_state; }();
        if (st == RunState::Training) {
            std::lock_guard<std::mutex> l(m_stateMutex);
            m_state = RunState::Idle;
        }
        MaybeAutosave(true);
        SaveSettingsFileIfDirty(true);
    }, maxMs);
}

void AppCore::WaitIdleForTests(int maxMs) {
    RunOnWorker([]() {}, maxMs);
}

std::string AppCore::SwapBusy(const std::string& text) {
    std::lock_guard<std::mutex> lock(m_stateMutex);
    std::string previous = std::move(m_busyText);
    m_busyText = text;
    return previous;
}

void AppCore::NoteSaved(uint64_t step) {
    m_lastSavedStep = step;
    std::lock_guard<std::mutex> lock(m_stateMutex);
    m_lastSaveTime = std::chrono::steady_clock::now();
    m_everSaved = true;
}

uint32_t AppCore::SnapshotCount() const {
    const uint64_t version = m_snapshotListVersion.load();
    std::lock_guard<std::mutex> lock(m_countMutex);
    if (m_countVersion != version) {
        m_countValue = static_cast<uint32_t>(m_store.List().size());
        m_countVersion = version;
    }
    return m_countValue;
}

// ---------------------------------------------------------------------------------------------
// Worker
// ---------------------------------------------------------------------------------------------
void AppCore::Post(std::function<void()> job) {
    {
        std::lock_guard<std::mutex> lock(m_jobMutex);
        m_jobs.push_back(std::move(job));
    }
    m_jobCv.notify_all();
}

bool AppCore::RunOnWorker(const std::function<void()>& job, int timeoutMs) {
    if (!m_started.load() || m_stop.load()) return false;
    struct Sync {
        std::mutex m;
        std::condition_variable cv;
        bool done = false;
    };
    auto sync = std::make_shared<Sync>();
    Post([sync, job]() {
        try { job(); } catch (...) {}
        {
            std::lock_guard<std::mutex> lock(sync->m);
            sync->done = true;
        }
        sync->cv.notify_all();
    });
    // Wait in short slices so a caller (for example a phone request) never stays stuck while the app
    // is shutting down: once stopping has begun, a job that is already running gets a moment to
    // finish and a job that never started is given up on.
    std::unique_lock<std::mutex> lock(sync->m);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
    for (;;) {
        if (sync->done) return true;
        const auto now = std::chrono::steady_clock::now();
        if (m_stop.load()) {
            return sync->cv.wait_for(lock, std::chrono::milliseconds(300), [&]() { return sync->done; });
        }
        if (now >= deadline) return false;
        const auto slice = std::min<std::chrono::steady_clock::duration>(std::chrono::milliseconds(100), deadline - now);
        sync->cv.wait_for(lock, slice, [&]() { return sync->done; });
    }
}

bool AppCore::RunQueuedJobs() {
    bool any = false;
    for (;;) {
        std::function<void()> job;
        {
            std::lock_guard<std::mutex> lock(m_jobMutex);
            if (m_jobs.empty()) break;
            job = std::move(m_jobs.front());
            m_jobs.pop_front();
        }
        any = true;
        try {
            job();
        } catch (const std::exception& ex) {
            AppendLog(std::string("A background task failed: ") + ex.what());
        } catch (...) {
            AppendLog("A background task failed.");
        }
    }
    return any;
}

void AppCore::WorkerMain() {
    while (!m_stop.load()) {
        bool didWork = false;
        bool gentle = false;
        try {
            std::lock_guard<std::mutex> engine(m_engineMutex);
            didWork = RunQueuedJobs();
            if (m_stop.load()) break;
            if (m_wantTraining.load()) {
                const bool trained = TrainingIteration();
                didWork = didWork || trained;
                gentle = trained && m_gentleNow;
            } else {
                RunState st;
                {
                    std::lock_guard<std::mutex> lock(m_stateMutex);
                    st = m_state;
                    if (st == RunState::Training) m_state = RunState::Idle;
                }
                if (st == RunState::Training) {
                    AppendLog("Training paused at lesson " + std::to_string(m_trainer.SnapshotMetrics().stepCount) + ".");
                    MaybeAutosave(true);
                    m_previewWanted.store(true);
                    didWork = true;
                }
                if (m_previewWanted.exchange(false) && !m_computeSuspended.load()) {
                    TrainerConfig cfg = GetConfig();
                    RefreshPreview(cfg);
                    didWork = true;
                }
            }
        } catch (const std::exception& ex) {
            MarkProblem(std::string("An unexpected error stopped the training: ") + ex.what());
        } catch (...) {
            MarkProblem("An unexpected error stopped the training.");
        }

        ReportLostPictures();
        SaveSettingsFileIfDirty(false);

        if (gentle) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        } else if (!didWork) {
            std::unique_lock<std::mutex> lock(m_jobMutex);
            m_jobCv.wait_for(lock, std::chrono::milliseconds(200), [this]() {
                return m_stop.load() || !m_jobs.empty() || (m_wantTraining.load() && !m_computeSuspended.load());
            });
        }
    }
}

void AppCore::ReportLostPictures() {
    const size_t lost = m_dataset.TakeUnreadableCount();
    if (lost == 0) return;
    const std::string text = std::to_string(lost) + (lost == 1 ? " picture" : " pictures") +
                             " could not be read back from the Xbox's storage and " + (lost == 1 ? "was" : "were") +
                             " taken out of the set.";
    AppendLog("Pictures: " + text);
    PushToast("warn", text);
}

bool AppCore::TrainingIteration() {
    if (m_computeSuspended.load()) return false;
    RunState st;
    {
        std::lock_guard<std::mutex> lock(m_stateMutex);
        st = m_state;
    }
    if (st == RunState::Problem) {
        m_wantTraining.store(false);
        return false;
    }

    TrainerConfig cfg = GetConfig();
    auto now = std::chrono::steady_clock::now();
    if (st != RunState::Training) {
        {
            std::lock_guard<std::mutex> lock(m_stateMutex);
            m_state = RunState::Training;
            m_problem.clear();
        }
        m_trainer.NoteTrainingStarted();
        m_lastLoopTime = now;
        // the first sample grid appears about 3 seconds after the start
        m_lastPreview = now - std::chrono::seconds(std::max(0, static_cast<int>(cfg.previewSeconds) - 3));
        AppendLog("Training started at lesson " + std::to_string(m_trainer.SnapshotMetrics().stepCount) + ".");
        if (cfg.engine == ComputeEngine::GPU && !m_brain.IsGPUReady()) {
            {
                std::lock_guard<std::mutex> lock(m_cfgMutex);
                m_config.engine = ComputeEngine::CPU;
            }
            cfg.engine = ComputeEngine::CPU;
            PushToast("warn", "The graphics chip is not available, so training runs on the processor (much slower).");
        }
    }

    // goal reached before even starting this lesson?
    {
        const TrainingMetrics before = m_trainer.SnapshotMetrics();
        if (cfg.targetSteps > 0 && before.stepCount >= cfg.targetSteps) {
            FinishTraining(RunState::Finished, "Training goal reached (" + std::to_string(before.stepCount) + " lessons).");
            return true;
        }
    }

    m_trainer.Step(m_brain, m_dataset, cfg);
    const TrainingMetrics m = m_trainer.SnapshotMetrics();
    m_gentleNow = cfg.gentle;

    if (m.gpuFallbackTriggered) {
        {
            std::lock_guard<std::mutex> lock(m_cfgMutex);
            m_config.engine = ComputeEngine::CPU;
        }
        {
            std::lock_guard<std::mutex> lock(m_stateMutex);
            m_gpuNote = "The graphics chip stopped answering, so the processor is doing the training (slower). See the log for details.";
        }
        AppendLog("The graphics chip failed during a lesson: switched to the processor.");
        PushToast("warn", "The graphics chip stopped answering, so training continues on the processor (slower).");
    }

    now = std::chrono::steady_clock::now();
    {
        const double dt = std::chrono::duration<double>(now - m_lastLoopTime).count();
        m_lastLoopTime = now;
        std::lock_guard<std::mutex> lock(m_stateMutex);
        m_trainSeconds += std::clamp(dt, 0.0, 2.0);
    }

    if (m.nonFinite) {
        MarkProblem("The numbers inside the brain became invalid (this usually means the learning rate was too high). "
                    "Load an earlier snapshot or start over, then try a lower learning rate.");
        return true;
    }

    if (cfg.targetSteps > 0 && m.stepCount >= cfg.targetSteps) {
        FinishTraining(RunState::Finished, "Training goal reached (" + std::to_string(m.stepCount) + " lessons).");
        return true;
    }

    if (!cfg.blindSpeedMode) {
        uint32_t interval = cfg.previewSeconds;
        if (cfg.engine == ComputeEngine::CPU) interval = std::max(interval, 30u); // the processor is slow at making pictures
        if (now - m_lastPreview >= std::chrono::seconds(interval)) {
            RefreshPreview(cfg);
            m_lastPreview = std::chrono::steady_clock::now();
            m_lastLoopTime = m_lastPreview; // picture-making time does not count as training time
        }
    }

    MaybeAutosave(false);
    return true;
}

void AppCore::FinishTraining(RunState newState, const std::string& text) {
    m_wantTraining.store(false);
    {
        std::lock_guard<std::mutex> lock(m_stateMutex);
        m_state = newState;
    }
    AppendLog(text);
    PushToast("good", text);
    MaybeAutosave(true);
    RefreshPreview(GetConfig());
}

void AppCore::MarkProblem(const std::string& text) {
    m_wantTraining.store(false);
    {
        std::lock_guard<std::mutex> lock(m_stateMutex);
        m_state = RunState::Problem;
        m_problem = text;
    }
    AppendLog("PROBLEM: " + text);
    PushToast("error", text);
}

// ---------------------------------------------------------------------------------------------
// Start / pause
// ---------------------------------------------------------------------------------------------
void AppCore::StartTraining() {
    {
        std::lock_guard<std::mutex> lock(m_stateMutex);
        if (m_state == RunState::Problem) {
            PushToast("warn", "Training can't start yet: " + m_problem);
            return;
        }
    }
    const TrainerConfig cfg = GetConfig();
    const TrainingMetrics m = m_trainer.SnapshotMetrics();
    if (cfg.targetSteps > 0 && m.stepCount >= cfg.targetSteps) {
        // the goal was already reached: raise it so "Train more" does what the person expects
        const uint64_t extra = std::max<uint64_t>(1000, m.stepCount / 2);
        const uint64_t newGoal = ((m.stepCount + extra + 249) / 250) * 250;
        JsonValue patch;
        patch.type = JsonValue::Type::Object;
        JsonValue n;
        n.type = JsonValue::Type::Number;
        n.number = static_cast<double>(newGoal);
        patch.members["goal"] = n;
        PatchSettings(patch, nullptr, nullptr);
        PushToast("info", "The goal had already been reached, so it was raised to " + std::to_string(newGoal) + " lessons.");
    }
    m_wantTraining.store(true);
    {
        std::lock_guard<std::mutex> lock(m_jobMutex);
    }
    m_jobCv.notify_all();
}

void AppCore::PauseTraining() {
    m_wantTraining.store(false);
    {
        std::lock_guard<std::mutex> lock(m_jobMutex);
    }
    m_jobCv.notify_all();
}

void AppCore::ToggleTraining() {
    if (m_wantTraining.load()) PauseTraining();
    else StartTraining();
}

// ---------------------------------------------------------------------------------------------
// Graphics chip
// ---------------------------------------------------------------------------------------------
bool AppCore::GpuUsable() const {
    return m_brain.IsGPUReady();
}

bool AppCore::AttachGpu(ID3D12Device* device, ID3D12CommandQueue* queue) {
    bool ok = false;
    {
        std::lock_guard<std::mutex> engine(m_engineMutex);
        const HRESULT hr = m_brain.InitializeGPU(device, queue);
        ok = SUCCEEDED(hr) && m_brain.IsGPUReady();
        {
            std::lock_guard<std::mutex> lock(m_stateMutex);
            m_gpuNote = ok ? "The graphics chip is ready (it passed its self-test)."
                           : "The graphics chip could not be used, so the processor does the training (much slower). See the log.";
        }
        std::lock_guard<std::mutex> lock(m_cfgMutex);
        if (!ok) m_config.engine = ComputeEngine::CPU;
        else if (m_gpuPreferred) m_config.engine = ComputeEngine::GPU;
    }
    m_computeSuspended.store(false);
    AppendLog(ok ? "Graphics chip ready." : "Graphics chip not usable; using the processor.");
    if (!ok) PushToast("warn", "The graphics chip could not be started. Training will use the processor (much slower).");
    {
        std::lock_guard<std::mutex> lock(m_jobMutex);
    }
    m_jobCv.notify_all();
    return ok;
}

void AppCore::DetachGpu(bool deviceLost) {
    m_computeSuspended.store(true);
    std::lock_guard<std::mutex> engine(m_engineMutex);
    if (!deviceLost) m_brain.ReadbackWeightsFromGPU(); // keep what the graphics chip learned
    m_brain.ReleaseGPU();
    {
        std::lock_guard<std::mutex> lock(m_stateMutex);
        m_gpuNote = deviceLost ? "The graphics chip was reset by the system; restarting it..." : "The graphics chip is switched off.";
    }
    if (!deviceLost) {
        std::lock_guard<std::mutex> lock(m_cfgMutex);
        m_config.engine = ComputeEngine::CPU;
    }
    AppendLog(deviceLost ? "Graphics device lost: recovering." : "Graphics chip released.");
}

// ---------------------------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------------------------
TrainerConfig AppCore::GetConfig() const {
    std::lock_guard<std::mutex> lock(m_cfgMutex);
    return m_config;
}

std::string AppCore::BuildSettingsJsonLocked() const {
    JsonWriter w;
    WriteConfigJson(w, m_config);
    return w.Take();
}

std::string AppCore::SettingsJson() const {
    std::lock_guard<std::mutex> lock(m_cfgMutex);
    return BuildSettingsJsonLocked();
}

bool AppCore::PatchSettings(const JsonValue& patch, std::string* error, std::vector<std::string>* changedOut) {
    std::vector<std::string> changed;
    bool engineRefused = false;
    {
        std::lock_guard<std::mutex> lock(m_cfgMutex);
        if (!ApplyConfigPatch(m_config, patch, &changed)) {
            if (error) *error = "The settings must be sent as a JSON object.";
            return false;
        }
        if (std::find(changed.begin(), changed.end(), std::string("engine")) != changed.end()) {
            m_gpuPreferred = (m_config.engine == ComputeEngine::GPU);
        }
        if (m_config.engine == ComputeEngine::GPU && !m_brain.IsGPUReady()) {
            m_config.engine = ComputeEngine::CPU;
            engineRefused = true;
        }
        if (!changed.empty()) {
            m_settingsDirty = true;
            m_settingsDirtySince = std::chrono::steady_clock::now();
        }
    }
    if (engineRefused) PushToast("warn", "The graphics chip is not available, so the processor is used.");
    for (const std::string& key : changed) {
        if ((key == "res" || key == "dreamSteps" || key == "ema" || key == "emaDecay" || key == "crisp") && m_trainer.SnapshotMetrics().stepCount > 0) {
            m_previewWanted.store(true);
        }
    }
    if (!changed.empty()) {
        std::string list;
        for (const std::string& k : changed) list += (list.empty() ? "" : ", ") + k;
        AppendLog("Settings changed: " + list);
    }
    if (changedOut) *changedOut = std::move(changed);
    {
        std::lock_guard<std::mutex> lock(m_jobMutex);
    }
    m_jobCv.notify_all();
    return true;
}

void AppCore::SetNumberSetting(const char* key, double value) {
    JsonValue patch;
    patch.type = JsonValue::Type::Object;
    JsonValue n;
    n.type = JsonValue::Type::Number;
    n.number = value;
    patch.members[key] = n;
    PatchSettings(patch, nullptr, nullptr);
}

void AppCore::SetBoolSetting(const char* key, bool value) {
    JsonValue patch;
    patch.type = JsonValue::Type::Object;
    JsonValue b;
    b.type = JsonValue::Type::Bool;
    b.boolean = value;
    patch.members[key] = b;
    PatchSettings(patch, nullptr, nullptr);
}

void AppCore::SetTextSetting(const char* key, const char* value) {
    JsonValue patch;
    patch.type = JsonValue::Type::Object;
    JsonValue t;
    t.type = JsonValue::Type::String;
    t.text = value ? value : "";
    patch.members[key] = t;
    PatchSettings(patch, nullptr, nullptr);
}

void AppCore::ResetSettingsToRecommended() {
    {
        std::lock_guard<std::mutex> lock(m_cfgMutex);
        const ComputeEngine engine = m_config.engine;
        m_config = TrainerConfig();
        if (m_brain.IsOnePass()) m_config.learningRate = kOnePassLearningRate; // the recommended rate depends on the brain type
        m_config.engine = engine;
        m_gpuPreferred = true;
        if (!m_brain.IsGPUReady()) m_config.engine = ComputeEngine::CPU;
        else m_config.engine = ComputeEngine::GPU;
        m_settingsDirty = true;
        m_settingsDirtySince = std::chrono::steady_clock::now();
    }
    m_previewWanted.store(m_trainer.SnapshotMetrics().stepCount > 0);
    PushToast("info", "Settings were set back to the recommended values.");
    AppendLog("Settings reset to the recommended values.");
}

void AppCore::LoadSettingsFile() {
    std::string text;
    if (!ReadWholeFile(m_root + "/settings.json", text)) return;
    JsonValue v;
    std::string err;
    if (!JsonParser::Parse(text, v, &err) || !v.IsObject()) {
        AppendLog("settings.json could not be read (" + err + "); using the recommended settings.");
        return;
    }
    const JsonValue* cfg = v.Find("config");
    if (!cfg) return;
    std::lock_guard<std::mutex> lock(m_cfgMutex);
    ApplyConfigPatch(m_config, *cfg, nullptr);
    m_config.engine = ComputeEngine::GPU; // the engine is decided at start-up by what the graphics chip can do
}

void AppCore::SaveSettingsFileIfDirty(bool force) {
    std::string body;
    {
        std::lock_guard<std::mutex> lock(m_cfgMutex);
        if (!m_settingsDirty) return;
        if (!force && std::chrono::steady_clock::now() - m_settingsDirtySince < std::chrono::seconds(1)) return;
        JsonWriter w;
        w.BeginObject();
        w.Field("v", 1);
        w.Key("config");
        WriteConfigJson(w, m_config);
        w.EndObject();
        body = w.Take();
        m_settingsDirty = false;
    }
    std::error_code ec;
    std::filesystem::create_directories(m_root, ec);
    if (!WriteFileAtomically(m_root + "/settings.json", body)) {
        AppendLog("Could not save settings.json.");
        std::lock_guard<std::mutex> lock(m_cfgMutex);
        m_settingsDirty = true;
        m_settingsDirtySince = std::chrono::steady_clock::now();
    }
}

// ---------------------------------------------------------------------------------------------
// Start over / dreams / snapshots (worker side)
// ---------------------------------------------------------------------------------------------
std::string AppCore::DoStartOver(uint32_t presetIndex, uint32_t waveLevel, uint32_t family) {
    BusyGuard busy(*this, "Starting a fresh brain...");
    // Nothing is lost by accident: unsaved progress gets an automatic snapshot first.
    {
        const TrainingMetrics m = m_trainer.SnapshotMetrics();
        if (m.stepCount > 0 && m.stepCount > m_lastSavedStep) {
            std::string id, err;
            bool ok = false;
            DoSaveSnapshot("Before starting over", "safety", &id, &err, &ok);
        }
    }
    m_wantTraining.store(false);
    {
        std::lock_guard<std::mutex> lock(m_stateMutex);
        m_state = RunState::Idle;
        m_problem.clear();
        m_trainSeconds = 0.0;
    }
    const bool wasOnePass = m_brain.IsOnePass();
    const bool onePass = family == 1;
    {
        std::lock_guard<std::recursive_mutex> lock(m_brain.Mutex());
        m_brain.InitializeFromScratch(static_cast<BrainPreset>(std::min(presetIndex, 2u)), NewSeed(),
                                      onePass ? 0u : std::min(waveLevel, FMBrain::kMaxWaveLevel),
                                      onePass ? BrainFamily::OnePass : BrainFamily::Flow);
    }
    // The one-pass brain learns best with a gentler learning rate. It is lowered for the person (never raised behind their back,
    // except to undo exactly that change when they go back to a flow brain).
    std::string note;
    {
        std::lock_guard<std::mutex> lock(m_cfgMutex);
        if (onePass && m_config.learningRate > kOnePassLearningRate + 1e-9f) {
            m_config.learningRate = kOnePassLearningRate;
            m_settingsDirty = true;
            m_settingsDirtySince = std::chrono::steady_clock::now();
            note = "The learning rate was set to 0.0003, which suits this type.";
        } else if (!onePass && wasOnePass && std::fabs(m_config.learningRate - kOnePassLearningRate) < 1e-9f) {
            m_config.learningRate = TrainerConfig().learningRate;
            m_settingsDirty = true;
            m_settingsDirtySince = std::chrono::steady_clock::now();
            note = "The learning rate was set back to 0.001, which suits this type.";
        }
    }
    m_trainer.ResetProgress();
    m_dataset.ResetEpochCounter();
    m_lastSavedStep = 0;
    ScanDialsNow();
    AppendLog(std::string("Started a fresh brain (size ") + std::to_string(presetIndex) +
              (onePass ? ", one-pass type" : (waveLevel ? ", wave type" : ", normal type")) + ")." + (note.empty() ? "" : " " + note));
    return note;
}

bool AppCore::StartOver(uint32_t presetIndex, std::string* error, uint32_t waveLevel, uint32_t family) {
    if (presetIndex > 2) {
        if (error) *error = "Unknown brain size.";
        return false;
    }
    if (waveLevel > FMBrain::kMaxWaveLevel || family > 1 || (family == 1 && waveLevel != 0)) {
        if (error) *error = "Unknown brain type.";
        return false;
    }
    auto note = std::make_shared<std::string>();
    const bool done = RunOnWorker([this, presetIndex, waveLevel, family, note]() { *note = DoStartOver(presetIndex, waveLevel, family); }, kJobTimeoutMs);
    if (!done) {
        if (error) *error = "The Xbox is busy right now. Try again in a moment.";
        return false;
    }
    PushToast("info", "Started a fresh brain. Your pictures are still there." + (note->empty() ? std::string() : " " + *note));
    return true;
}

void AppCore::ScanDialsNow() {
    if (std::this_thread::get_id() == m_worker.get_id()) {
        m_trainer.SetDialScanReport(m_brain.ScanDials(0.01f));
        return;
    }
    RunOnWorker([this]() { m_trainer.SetDialScanReport(m_brain.ScanDials(0.01f)); }, kJobTimeoutMs);
}

std::vector<uint8_t> AppCore::CrispPalette(const TrainerConfig& cfg) {
    if (cfg.crispColors < 2) return std::vector<uint8_t>();
    const uint64_t version = m_dataset.Version();
    {
        std::lock_guard<std::mutex> lock(m_crispMutex);
        if (m_crispValid && m_crispVersion == version && m_crispColors == cfg.crispColors) return m_crispPalette;
    }
    // The colours of up to 64 pictures spread over the whole set are enough to find the palette (and quick to collect).
    constexpr size_t kSampleCards = 64;
    std::vector<uint8_t> sample;
    std::vector<uint8_t> palette;
    if (m_dataset.SamplePixels(kSampleCards, sample) > 0) palette = MakePalette(sample.data(), sample.size() / 3, cfg.crispColors);
    std::lock_guard<std::mutex> lock(m_crispMutex);
    m_crispPalette = palette;
    m_crispVersion = version;
    m_crispColors = cfg.crispColors;
    m_crispValid = true;
    return palette;
}

void AppCore::RefreshPreview(const TrainerConfig& cfg) {
    const std::vector<uint8_t> palette = CrispPalette(cfg);
    m_trainer.UpdatePreview(m_brain, cfg, palette.empty() ? nullptr : &palette);
}

bool AppCore::DreamNow(uint32_t count, uint32_t dreamSteps, DreamResult& result, float variety) {
    result = DreamResult();
    auto out = std::make_shared<DreamResult>();
    const bool done = RunOnWorker([this, count, dreamSteps, variety, out]() {
        BusyGuard busy(*this, "Making pictures...");
        const TrainerConfig cfg = GetConfig();
        const bool onePass = m_brain.IsOnePass();
        // (a wave brain draws even sizes, a one-pass brain sizes that are a multiple of 8)
        const uint32_t res = Trainer::PictureSizeFor(cfg.resolution, m_brain.WaveLevel(), m_brain.Family());
        uint32_t n = std::clamp(count, 1u, 16u);
        const bool onGpu = cfg.engine == ComputeEngine::GPU && m_brain.IsGPUReady();
        if (!onGpu && res >= 64) n = std::min(n, 2u);
        const uint32_t steps = onePass ? 1u : std::clamp(dreamSteps == 0 ? cfg.dreamSteps : dreamSteps, 1u, onGpu ? 50u : 12u);
        std::vector<std::vector<float>> images;
        bool avg = false;
        if (!m_trainer.MakeSamples(m_brain, cfg, n, res, steps, false, true, images, &avg, variety)) {
            out->error = "The brain could not make pictures right now.";
            return;
        }
        const uint64_t step = m_trainer.SnapshotMetrics().stepCount;
        const std::vector<uint8_t> palette = CrispPalette(cfg);
        std::lock_guard<std::mutex> lock(m_dreamMutex);
        for (const auto& chw : images) {
            Rgb8Image img = FloatChwToRgb8(chw.data(), res, res);
            if (!palette.empty()) SnapToPalette(img, palette);
            DreamEntry e;
            e.id = m_nextDreamId++;
            e.w = res;
            e.h = res;
            e.step = step;
            e.dreamSteps = onePass ? 0u : steps;
            e.usedAverage = avg;
            e.rgb = std::move(img.px);
            out->ids.push_back(e.id);
            m_dreams.push_back(std::move(e));
            while (m_dreams.size() > kMaxDreams) m_dreams.pop_front();
        }
        out->w = res;
        out->h = res;
        out->usedAverage = avg;
        out->ok = true;
    }, kJobTimeoutMs);
    if (!done) {
        result.error = "The Xbox is busy right now. Try again in a moment.";
        return false;
    }
    result = *out;
    return result.ok;
}

bool AppCore::GetDream(uint64_t id, DreamEntry& out) const {
    std::lock_guard<std::mutex> lock(m_dreamMutex);
    for (const DreamEntry& e : m_dreams) {
        if (e.id == id) {
            out = e;
            return true;
        }
    }
    return false;
}

std::vector<DreamEntry> AppCore::ListDreams() const {
    std::lock_guard<std::mutex> lock(m_dreamMutex);
    std::vector<DreamEntry> list;
    for (const DreamEntry& e : m_dreams) {
        DreamEntry c = e;
        c.rgb.clear();
        c.rgb.shrink_to_fit();
        list.push_back(std::move(c));
    }
    return list;
}

void AppCore::DoSaveSnapshot(const std::string& name, const std::string& kind, std::string* idOut, std::string* error, bool* ok) {
    *ok = false;
    BusyGuard busy(*this, "Saving a snapshot...");
    std::lock_guard<std::recursive_mutex> lock(m_brain.Mutex());
    if (FAILED(m_brain.ReadbackWeightsFromGPU())) {
        if (error) *error = "The brain could not be read back from the graphics chip.";
        return;
    }
    CheckpointData d;
    BrainIO::CollectParams(m_brain, d.params);
    BrainIO::CollectAdam(m_brain, d.adamM, d.adamV);
    d.ema = m_trainer.AveragedDials();
    if (d.ema.size() != d.params.size()) d.ema.clear();
    const TrainingMetrics m = m_trainer.SnapshotMetrics();
    const TrainerConfig cfg = GetConfig();
    d.meta.kind = kind;
    d.meta.name = name;
    d.meta.step = m.stepCount;
    d.meta.preset = static_cast<uint32_t>(m_brain.Architecture().preset);
    d.meta.wave = m_brain.WaveLevel();
    d.meta.family = m_brain.IsOnePass() ? 1u : 0u;
    d.meta.resolution = cfg.resolution;
    d.meta.loss = static_cast<double>(m.smoothLoss);
    d.meta.epochs = m.epochs;
    d.meta.pictures = static_cast<uint32_t>(m_dataset.Size());
    {
        JsonWriter w;
        WriteConfigJson(w, cfg);
        d.settingsJson = w.Take();
    }
    d.curve = m_trainer.LossCurve(240);
    std::string err;
    if (!m_store.Save(d, &err)) {
        if (error) *error = err;
        return;
    }
    if (idOut) *idOut = d.meta.id;
    NoteSaved(m.stepCount);
    ++m_snapshotListVersion;
    *ok = true;
    AppendLog("Saved snapshot \"" + name + "\" (lesson " + std::to_string(m.stepCount) + ").");
}

void AppCore::MaybeAutosave(bool force) {
    const TrainerConfig cfg = GetConfig();
    if (cfg.autosaveMinutes == 0) return;
    const TrainingMetrics m = m_trainer.SnapshotMetrics();
    if (m.stepCount == 0 || m.stepCount <= m_lastSavedStep) return;
    const auto now = std::chrono::steady_clock::now();
    if (!force) {
        std::chrono::steady_clock::time_point last;
        {
            std::lock_guard<std::mutex> lock(m_stateMutex);
            last = m_lastSaveTime;
        }
        if (now - last < std::chrono::minutes(cfg.autosaveMinutes)) return;
    }

    std::string id, err;
    bool ok = false;
    DoSaveSnapshot("Autosave", "auto", &id, &err, &ok);
    if (ok) {
        m_autosaveWarned = false;
        return;
    }
    {
        std::lock_guard<std::mutex> lock(m_stateMutex);
        m_lastSaveTime = now; // do not retry every lesson
    }
    AppendLog("Autosave failed: " + err);
    if (err.find("broken numbers") != std::string::npos) {
        MarkProblem("The numbers inside the brain became invalid. Load an earlier snapshot or start over, then try a lower learning rate.");
    } else if (!m_autosaveWarned) {
        m_autosaveWarned = true;
        PushToast("warn", "Autosave didn't work: " + err);
    }
}

bool AppCore::DoLoadSnapshot(const CheckpointData& data, bool applySettings, std::string* error) {
    const BrainFamily family = data.meta.family == 1 ? BrainFamily::OnePass : BrainFamily::Flow;
    if (data.meta.preset > 2 || data.meta.wave > FMBrain::kMaxWaveLevel || data.meta.family > 1 ||
        (family == BrainFamily::OnePass && data.meta.wave != 0) ||
        data.params.size() != FMBrain::ExpectedParameterCount(static_cast<BrainPreset>(data.meta.preset), data.meta.wave, family)) {
        if (error) *error = "That snapshot does not fit this app.";
        return false;
    }
    {
        std::lock_guard<std::recursive_mutex> lock(m_brain.Mutex());
        m_brain.InitializeFromScratch(static_cast<BrainPreset>(data.meta.preset), NewSeed(), data.meta.wave, family);
        if (!BrainIO::ApplyParams(m_brain, data.params)) {
            if (error) *error = "The snapshot's numbers do not fit the brain.";
            return false;
        }
        if (!data.adamM.empty() && !BrainIO::ApplyAdam(m_brain, data.adamM, data.adamV)) {
            if (error) *error = "The snapshot's learning memory does not fit the brain.";
            return false;
        }
    }
    if (applySettings && !data.settingsJson.empty()) {
        JsonValue v;
        if (JsonParser::Parse(data.settingsJson, v) && v.IsObject()) {
            std::lock_guard<std::mutex> lock(m_cfgMutex);
            const ComputeEngine engine = m_config.engine;
            const bool blind = m_config.blindSpeedMode;
            const bool gentle = m_config.gentle;
            ApplyConfigPatch(m_config, v, nullptr);
            m_config.engine = engine; // these three belong to this Xbox, not to the snapshot
            m_config.blindSpeedMode = blind;
            m_config.gentle = gentle;
            m_settingsDirty = true;
            m_settingsDirtySince = std::chrono::steady_clock::now();
        }
    }
    m_trainer.RestoreProgress(data.meta.step, static_cast<float>(data.meta.loss), data.meta.epochs, data.curve, data.ema);
    m_dataset.ResetEpochCounter();
    {
        std::lock_guard<std::mutex> lock(m_stateMutex);
        m_state = RunState::Idle;
        m_problem.clear();
        m_trainSeconds = 0.0;
    }
    m_wantTraining.store(false);
    m_previewWanted.store(true);
    m_trainer.SetDialScanReport(m_brain.ScanDials(0.01f));
    return true;
}

// ---------------------------------------------------------------------------------------------
// Snapshots (public)
// ---------------------------------------------------------------------------------------------
bool AppCore::SaveSnapshot(const std::string& name, std::string* idOut, std::string* error) {
    struct Out { std::string id, err; bool ok = false; };
    auto out = std::make_shared<Out>();
    const std::string label = name.empty() ? std::string("My snapshot") : name;
    const bool done = RunOnWorker([this, label, out]() { DoSaveSnapshot(label, "manual", &out->id, &out->err, &out->ok); }, kJobTimeoutMs);
    if (!done) {
        if (error) *error = "The Xbox is busy right now. Try again in a moment.";
        return false;
    }
    if (!out->ok) {
        if (error) *error = out->err.empty() ? std::string("The snapshot could not be saved.") : out->err;
        return false;
    }
    if (idOut) *idOut = out->id;
    PushToast("good", "Snapshot saved: " + label);
    return true;
}

bool AppCore::LoadSnapshot(const std::string& id, std::string* error) {
    struct Out { std::string err; bool ok = false; std::string label; uint64_t step = 0; };
    auto out = std::make_shared<Out>();
    const bool done = RunOnWorker([this, id, out]() {
        BusyGuard busy(*this, "Loading a snapshot...");
        CheckpointData data;
        if (!m_store.Load(id, data, &out->err)) return;
        // keep unsaved progress before it is replaced
        {
            const TrainingMetrics m = m_trainer.SnapshotMetrics();
            if (m.stepCount > 0 && m.stepCount > m_lastSavedStep) {
                std::string sid, serr;
                bool sok = false;
                DoSaveSnapshot("Before loading another snapshot", "safety", &sid, &serr, &sok);
            }
        }
        if (!DoLoadSnapshot(data, true, &out->err)) return;
        NoteSaved(data.meta.step);
        out->label = data.meta.name.empty() ? id : data.meta.name;
        out->step = data.meta.step;
        out->ok = true;
    }, kJobTimeoutMs);
    if (!done) {
        if (error) *error = "The Xbox is busy right now. Try again in a moment.";
        return false;
    }
    if (!out->ok) {
        if (error) *error = out->err.empty() ? std::string("The snapshot could not be loaded.") : out->err;
        return false;
    }
    PushToast("good", "Loaded \"" + out->label + "\" (lesson " + std::to_string(out->step) + "). Training is paused: press Start to continue.");
    AppendLog("Loaded snapshot " + id + ".");
    ++m_snapshotListVersion;
    return true;
}

bool AppCore::DeleteSnapshot(const std::string& id, std::string* error) {
    if (!m_store.Delete(id, error)) return false;
    ++m_snapshotListVersion;
    AppendLog("Deleted snapshot " + id + ".");
    return true;
}

bool AppCore::RenameSnapshot(const std::string& id, const std::string& name, std::string* error) {
    if (!m_store.Rename(id, name, error)) return false;
    ++m_snapshotListVersion;
    return true;
}

bool AppCore::ImportSnapshot(const uint8_t* bytes, size_t size, const std::string& name, std::string* idOut, std::string* error) {
    CheckpointData data;
    std::string err;
    if (bytes && size >= 16 && std::memcmp(bytes, "OTFM", 4) == 0) {
        // the older, weights-only backup file of version 3
        uint32_t preset = 0;
        uint64_t count = 0;
        std::memcpy(&preset, bytes + 4, 4);
        std::memcpy(&count, bytes + 8, 8);
        const uint32_t wave = preset >> 8; // wave brains keep their type in the second byte of this word (older files have zero there)
        preset &= 0xFFu;
        if (preset > 2 || wave > FMBrain::kMaxWaveLevel || count != FMBrain::ExpectedParameterCount(static_cast<BrainPreset>(preset), wave) ||
            size != 16 + static_cast<size_t>(count) * sizeof(float)) {
            if (error) *error = "This backup file is damaged or from an unknown brain size.";
            return false;
        }
        data.params.resize(static_cast<size_t>(count));
        std::memcpy(data.params.data(), bytes + 16, static_cast<size_t>(count) * sizeof(float));
        if (!BrainIO::AllFinite(data.params)) {
            if (error) *error = "This backup file contains broken numbers.";
            return false;
        }
        data.meta.preset = preset;
        data.meta.wave = wave;
        data.meta.step = 0;
    } else if (!CheckpointStore::Parse(bytes, size, data, &err)) {
        if (error) *error = err;
        return false;
    }
    data.meta.kind = "imported";
    data.meta.createdUnix = 0;
    if (!name.empty()) data.meta.name = name;
    else if (data.meta.name.empty()) data.meta.name = "Imported file";
    if (!m_store.Save(data, &err)) {
        if (error) *error = err;
        return false;
    }
    ++m_snapshotListVersion;
    if (idOut) *idOut = data.meta.id;
    PushToast("good", "Imported \"" + data.meta.name + "\". Open it in the list and press Load to use it.");
    return true;
}

bool AppCore::ExportSnapshotNow(std::vector<uint8_t>& out, std::string* error) {
    auto result = std::make_shared<std::pair<std::vector<uint8_t>, std::string>>();
    const bool done = RunOnWorker([this, result]() {
        std::lock_guard<std::recursive_mutex> lock(m_brain.Mutex());
        if (FAILED(m_brain.ReadbackWeightsFromGPU())) {
            result->second = "The brain could not be read back from the graphics chip.";
            return;
        }
        CheckpointData d;
        BrainIO::CollectParams(m_brain, d.params);
        BrainIO::CollectAdam(m_brain, d.adamM, d.adamV);
        d.ema = m_trainer.AveragedDials();
        if (d.ema.size() != d.params.size()) d.ema.clear();
        if (!BrainIO::AllFinite(d.params)) {
            result->second = "The brain contains broken numbers, so it can't be exported.";
            return;
        }
        const TrainingMetrics m = m_trainer.SnapshotMetrics();
        const TrainerConfig cfg = GetConfig();
        d.meta.kind = "manual";
        d.meta.name = "Backup";
        d.meta.createdUnix = NowUnix();
        d.meta.step = m.stepCount;
        d.meta.preset = static_cast<uint32_t>(m_brain.Architecture().preset);
        d.meta.wave = m_brain.WaveLevel();
        d.meta.family = m_brain.IsOnePass() ? 1u : 0u;
        d.meta.resolution = cfg.resolution;
        d.meta.loss = static_cast<double>(m.smoothLoss);
        d.meta.epochs = m.epochs;
        d.meta.pictures = static_cast<uint32_t>(m_dataset.Size());
        JsonWriter w;
        WriteConfigJson(w, cfg);
        d.settingsJson = w.Take();
        d.curve = m_trainer.LossCurve(240);
        result->first = CheckpointStore::Serialize(d);
    }, kJobTimeoutMs);
    if (!done) {
        if (error) *error = "The Xbox is busy right now. Try again in a moment.";
        return false;
    }
    if (result->first.empty()) {
        if (error) *error = result->second.empty() ? std::string("Export failed.") : result->second;
        return false;
    }
    out = std::move(result->first);
    return true;
}

// ---------------------------------------------------------------------------------------------
// Reading state
// ---------------------------------------------------------------------------------------------
void AppCore::SetLanUrl(const std::string& url) {
    std::lock_guard<std::mutex> lock(m_stateMutex);
    m_lanUrl = url;
}

AppSnapshot AppCore::Snapshot() const {
    AppSnapshot s;
    s.version = m_version;
    s.config = GetConfig();
    s.metrics = m_trainer.SnapshotMetrics();
    s.previewVersion = m_trainer.PreviewVersion();
    s.previewReady = m_trainer.PreviewReady();
    s.lossVersion = m_trainer.LossVersion();
    s.gpuAvailable = m_brain.IsGPUReady();
    s.presetIndex = static_cast<uint32_t>(m_brain.Architecture().preset);
    s.waveLevel = m_brain.WaveLevel();
    s.family = m_brain.IsOnePass() ? 1u : 0u;
    s.paramCount = m_brain.ParameterCount();
    s.pictureCount = m_dataset.Size();
    s.pictureCapacity = m_dataset.Capacity();
    s.pictureBytes = m_dataset.DiskBytes();
    s.pictureVersion = m_dataset.Version();
    s.picturesPersistent = m_dataset.Persistent();
    s.snapshotListVersion = m_snapshotListVersion.load();
    s.snapshotCount = SnapshotCount();
    s.practice = s.pictureCount == 0;
    s.uptimeSeconds = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - m_startTime).count());
    {
        std::lock_guard<std::mutex> lock(m_stateMutex);
        s.state = m_state;
        s.problem = m_problem;
        s.gpuNote = m_gpuNote;
        s.lanUrl = m_lanUrl;
        s.trainSeconds = m_trainSeconds;
        s.busyText = m_busyText;
        s.busy = !m_busyText.empty();
        if (m_everSaved) {
            s.lastSavedText = "Saved " + AgoText(std::chrono::duration<double>(std::chrono::steady_clock::now() - m_lastSaveTime).count());
        } else {
            s.lastSavedText = "Not saved yet";
        }
    }
    if (m_wantTraining.load() && s.state == RunState::Idle) s.state = RunState::Training; // the worker is about to start
    switch (s.state) {
        case RunState::Training: s.stateText = m_wantTraining.load() ? "Training" : "Pausing..."; break;
        case RunState::Finished: s.stateText = "Goal reached"; break;
        case RunState::Problem: s.stateText = "Needs attention"; break;
        default: s.stateText = s.metrics.stepCount == 0 ? "Ready" : "Paused"; break;
    }
    if (s.state == RunState::Training && s.config.targetSteps > s.metrics.stepCount && s.metrics.stepsPerSec > 0.0f) {
        s.etaSeconds = static_cast<double>(s.config.targetSteps - s.metrics.stepCount) / static_cast<double>(s.metrics.stepsPerSec);
    }
    {
        std::lock_guard<std::mutex> lock(m_noticeMutex);
        s.latestToastId = m_nextToastId - 1;
    }
    return s;
}

std::string AppCore::SnapshotJson() const {
    const AppSnapshot s = Snapshot();
    JsonWriter w;
    w.BeginObject();
    w.Field("version", s.version);
    w.Field("state", StateName(s.state));
    w.Field("stateText", s.stateText);
    w.Field("problem", s.problem);
    w.Field("practice", s.practice);
    w.Key("gpu").BeginObject().Field("available", s.gpuAvailable).Field("note", s.gpuNote).EndObject();
    w.Field("engine", s.metrics.usingGpuEngine ? "gpu" : (s.config.engine == ComputeEngine::GPU ? "gpu" : "cpu"));
    w.Field("step", s.metrics.stepCount);
    w.Field("goal", s.config.targetSteps);
    w.Field("epochs", s.metrics.epochs);
    w.Field("loss", s.metrics.smoothLoss);
    w.Field("lossLast", s.metrics.lastMseLoss);
    w.Field("lossBest", s.metrics.bestSmoothLoss);
    w.Field("lr", s.metrics.currentLr);
    w.Field("stepsPerSec", s.metrics.stepsPerSec);
    w.Field("imagesPerSec", s.metrics.lessonsPerSec);
    w.Field("usedBatch", s.metrics.usedBatch);
    w.Field("usedRes", s.metrics.usedResolution);
    w.Field("trainSeconds", s.trainSeconds);
    w.Key("eta");
    if (s.etaSeconds >= 0.0) w.Value(s.etaSeconds); else w.Null();
    w.Field("nonFinite", s.metrics.nonFinite);
    w.Key("brain").BeginObject();
    w.Field("preset", s.presetIndex);
    w.Field("wave", s.waveLevel);
    w.Field("family", s.family);
    w.Field("params", s.paramCount);
    w.Field("dialsTotal", s.metrics.lastDialScan.totalDials);
    w.Field("dialsBusy", s.metrics.lastDialScan.busyDials);
    w.Field("dialsSleeping", s.metrics.lastDialScan.sleepingDials);
    w.EndObject();
    w.Key("pictures").BeginObject();
    w.Field("count", s.pictureCount);
    w.Field("capacity", s.pictureCapacity);
    w.Field("bytes", s.pictureBytes);
    w.Field("version", s.pictureVersion);
    w.Field("persistent", s.picturesPersistent);
    w.EndObject();
    w.Field("previewVersion", s.previewVersion);
    w.Field("previewReady", s.previewReady);
    w.Field("lossVersion", s.lossVersion);
    w.Key("snapshots").BeginObject().Field("version", s.snapshotListVersion).Field("count", s.snapshotCount).EndObject();
    w.Field("lastSaved", s.lastSavedText);
    w.Field("busy", s.busy);
    w.Field("busyText", s.busyText);
    w.Key("settings");
    WriteConfigJson(w, s.config);
    w.Field("lanUrl", s.lanUrl);
    w.Field("latestToast", s.latestToastId);
    w.Field("uptime", s.uptimeSeconds);
    w.EndObject();
    return w.Take();
}

// ---------------------------------------------------------------------------------------------
// Toasts and logs
// ---------------------------------------------------------------------------------------------
void AppCore::PushToast(const std::string& level, const std::string& text) {
    std::lock_guard<std::mutex> lock(m_noticeMutex);
    Toast t;
    t.id = m_nextToastId++;
    t.level = level;
    t.text = text;
    m_toasts.push_back(std::move(t));
    while (m_toasts.size() > kMaxToasts) m_toasts.pop_front();
}

std::vector<Toast> AppCore::ToastsAfter(uint64_t afterId) const {
    std::lock_guard<std::mutex> lock(m_noticeMutex);
    std::vector<Toast> out;
    for (const Toast& t : m_toasts) {
        if (t.id > afterId) out.push_back(t);
    }
    return out;
}

void AppCore::AppendLog(const std::string& line) {
    char stamp[64];
    const uint64_t up = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - m_startTime).count());
    std::snprintf(stamp, sizeof(stamp), "[%02llu:%02llu:%02llu] ", static_cast<unsigned long long>(up / 3600),
                  static_cast<unsigned long long>((up / 60) % 60), static_cast<unsigned long long>(up % 60));
    const std::string full = std::string(stamp) + line;
    {
        std::lock_guard<std::mutex> lock(m_noticeMutex);
        m_logs.push_back(full);
        while (m_logs.size() > kMaxLogLines) m_logs.pop_front();
    }
    if (m_logSink) m_logSink(full);
}

std::string AppCore::LogText() const {
    std::string out;
    if (m_extraLog) {
        out += m_extraLog();
        if (!out.empty() && out.back() != '\n') out += '\n';
        out += "---- app log ----\n";
    }
    std::lock_guard<std::mutex> lock(m_noticeMutex);
    for (const std::string& l : m_logs) {
        out += l;
        out += '\n';
    }
    return out;
}

} // namespace XboxOTFM
