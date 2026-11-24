#pragma once
// The brain of the app: one background worker that trains, makes pictures, saves and loads snapshots, plus the state that the
// phone web page and the TV screen both show. Nothing in here is Windows-specific (so it can be tested anywhere).

#include "Checkpoint.h"
#include "Dataset.h"
#include "FMBrain.h"
#include "Json.h"
#include "Trainer.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace XboxOTFM {

enum class RunState : uint32_t {
    Idle = 0,      // ready / paused
    Training = 1,
    Finished = 2,  // the training goal was reached
    Problem = 3    // something went wrong (the brain broke, or an unexpected error); needs a snapshot load or a fresh start
};

struct Toast {
    uint64_t id = 0;
    std::string level; // "info", "good", "warn", "error"
    std::string text;
};

// Everything the screens need, in one copy.
struct AppSnapshot {
    std::string version;
    RunState state = RunState::Idle;
    std::string stateText;        // short line for the headline ("Training", "Paused", ...)
    std::string problem;          // what went wrong (state == Problem)
    bool practice = true;         // training on the built-in patterns
    bool gpuAvailable = false;
    std::string gpuNote;
    TrainingMetrics metrics;
    TrainerConfig config;
    uint32_t presetIndex = 1;
    uint32_t waveLevel = 0;       // 0 = the brain works on pixels, 1 = on the Haar wave form (about 4 times less work per lesson)
    uint32_t family = 0;          // 0 = flow brain (pixels or waves), 1 = one-pass brain (draws a picture from random numbers in one go)
    uint64_t paramCount = 0;
    uint64_t pictureCount = 0;
    uint64_t pictureCapacity = 0;
    uint64_t pictureBytes = 0;
    uint64_t pictureVersion = 0;
    bool picturesPersistent = false;
    uint64_t previewVersion = 0;
    bool previewReady = false;
    uint64_t lossVersion = 0;
    uint64_t snapshotListVersion = 0;
    uint32_t snapshotCount = 0;
    double trainSeconds = 0.0;
    double etaSeconds = -1.0;
    std::string lanUrl;
    uint64_t latestToastId = 0;
    uint64_t uptimeSeconds = 0;
    bool busy = false;
    std::string busyText;
    std::string lastSavedText;    // "Saved 3 min ago" style info for autosave
};

struct DreamEntry {
    uint64_t id = 0;
    uint32_t w = 0;
    uint32_t h = 0;
    uint64_t step = 0;
    uint32_t dreamSteps = 0;  // 0 = drawn in one pass (a one-pass brain)
    bool usedAverage = false;
    std::vector<uint8_t> rgb; // w * h * 3 (empty in list results)
};

struct DreamResult {
    bool ok = false;
    std::string error;
    std::vector<uint64_t> ids;
    uint32_t w = 0;
    uint32_t h = 0;
    bool usedAverage = false;
};

class AppCore {
public:
    // dataRoot: a writable folder (created if needed). Pictures, snapshots and settings live inside it.
    // pictureOptions: the picture limit and memory use (the defaults are what the app ships with; tests use smaller ones).
    AppCore(std::string dataRoot, std::string appVersion, const DatasetOptions& pictureOptions = DatasetOptions());
    ~AppCore();
    AppCore(const AppCore&) = delete;
    AppCore& operator=(const AppCore&) = delete;

    // Loads settings, pictures and the newest snapshot (never starts training by itself), then starts the worker.
    void Start();
    // Saves an automatic snapshot if there is unsaved progress, stops the worker.
    void Stop();
    // Called when the system is about to suspend the app: pause + save, quickly (waits at most maxMs).
    void PrepareForSuspend(int maxMs = 3000);

    // ---- graphics chip (called by the screen code) ----
    // Both wait until the worker is between two lessons. AttachGpu returns true if the GPU path passed its self-test.
    bool AttachGpu(ID3D12Device* device, ID3D12CommandQueue* queue);
    void DetachGpu(bool deviceLost);

    // ---- commands (thread-safe; most return at once and report through toasts) ----
    void StartTraining();
    void PauseTraining();
    void ToggleTraining();
    bool IsTrainingWanted() const { return m_wantTraining.load(); }

    // Applies a partial settings object. Returns false (and error) if it is not an object. `changed` lists what changed.
    bool PatchSettings(const JsonValue& patch, std::string* error, std::vector<std::string>* changed = nullptr);
    // Changes one setting by name using the same rules as PatchSettings (used by the TV screen).
    void SetNumberSetting(const char* key, double value);
    void SetBoolSetting(const char* key, bool value);
    void SetTextSetting(const char* key, const char* value);
    TrainerConfig GetConfig() const;
    std::string SettingsJson() const;
    void ResetSettingsToRecommended();

    // New random brain of the given size (0 small, 1 medium, 2 large) and type (waveLevel 0 = works on pixels, 1 = works on the wave
    // form of the pictures; family 1 = the one-pass brain, which needs waveLevel 0). Progress and chart restart; pictures stay.
    bool StartOver(uint32_t presetIndex, std::string* error = nullptr, uint32_t waveLevel = 0, uint32_t family = 0);

    // Pictures from fresh noise (appear in the dream list). Waits for the worker (up to ~2 minutes).
    // variety only matters for a one-pass brain: how wild the random numbers are (1 = normal, 0 = the "average" picture).
    bool DreamNow(uint32_t count, uint32_t dreamSteps, DreamResult& result, float variety = 1.0f);
    bool GetDream(uint64_t id, DreamEntry& out) const;
    std::vector<DreamEntry> ListDreams() const; // newest last, without the pixels

    // Snapshots
    bool SaveSnapshot(const std::string& name, std::string* idOut, std::string* error);
    bool LoadSnapshot(const std::string& id, std::string* error);
    bool DeleteSnapshot(const std::string& id, std::string* error);
    bool RenameSnapshot(const std::string& id, const std::string& name, std::string* error);
    bool ImportSnapshot(const uint8_t* bytes, size_t size, const std::string& name, std::string* idOut, std::string* error);
    bool ExportSnapshotNow(std::vector<uint8_t>& out, std::string* error); // the live brain as a file (for a backup download)
    std::vector<CheckpointMeta> ListSnapshots() const { return m_store.List(); }
    bool ReadSnapshotFile(const std::string& id, std::vector<uint8_t>& out, std::string* error) const { return m_store.ReadRaw(id, out, error); }

    // Quick look at how many dials are asleep (also shown on the TV).
    void ScanDialsNow();

    // ---- reading state ----
    AppSnapshot Snapshot() const;
    std::string SnapshotJson() const;
    Dataset& Pictures() { return m_dataset; }
    Trainer& GetTrainer() { return m_trainer; }
    FMBrain& Brain() { return m_brain; }
    void SetLanUrl(const std::string& url);
    // Tells the picture set how to ask the system how much storage is free (the Xbox build provides this). Call before Start().
    void SetFreeSpaceReader(Dataset::FreeSpaceReader reader) { m_dataset.SetFreeSpaceReader(std::move(reader)); }
    void SetExtraLogReader(std::function<std::string()> reader) { m_extraLog = std::move(reader); }
    void SetLogSink(std::function<void(const std::string&)> sink) { m_logSink = std::move(sink); }

    std::vector<Toast> ToastsAfter(uint64_t afterId) const;
    void PushToast(const std::string& level, const std::string& text);
    void AppendLog(const std::string& line);
    std::string LogText() const;

    // Waits until the worker has finished everything that is queued (tests / shutdown).
    void WaitIdleForTests(int maxMs = 10000);

private:
    friend struct AppCoreTestAccess;

    void WorkerMain();
    bool RunQueuedJobs();
    bool TrainingIteration();
    void Post(std::function<void()> job);
    bool RunOnWorker(const std::function<void()>& job, int timeoutMs);

    // worker-thread operations
    void DoSaveSnapshot(const std::string& name, const std::string& kind, std::string* idOut, std::string* error, bool* ok);
    bool DoLoadSnapshot(const CheckpointData& data, bool applySettings, std::string* error);
    // Returns a short note for the person when something was set up for them (e.g. the learning rate), else empty.
    std::string DoStartOver(uint32_t presetIndex, uint32_t waveLevel, uint32_t family);
    void FinishTraining(RunState newState, const std::string& text);
    void MaybeAutosave(bool force);
    void ReportLostPictures();      // pictures that could not be read back from storage were dropped: say so once
    void SaveSettingsFileIfDirty(bool force);
    void LoadSettingsFile();
    // The colours of the picture set for the "crisp colours" finish (red, green, blue triples); empty when the finish is off or
    // there are no pictures. Remembered until the pictures change. Worker thread (the cache has its own lock anyway).
    std::vector<uint8_t> CrispPalette(const TrainerConfig& cfg);
    // Refreshes the sample grid, with the crisp colours finish when it is on. Worker thread only.
    void RefreshPreview(const TrainerConfig& cfg);
    void MarkProblem(const std::string& text);
    std::string BuildSettingsJsonLocked() const;
    bool GpuUsable() const;
    void NoteSaved(uint64_t step);
    uint32_t SnapshotCount() const;

    std::string SwapBusy(const std::string& text);

    // Shows "Saving..." style text on the screens while a long job runs; restores the previous text afterwards.
    class BusyGuard {
    public:
        BusyGuard(AppCore& core, const std::string& text) : m_core(core), m_previous(core.SwapBusy(text)) {}
        ~BusyGuard() { m_core.SwapBusy(m_previous); }
        BusyGuard(const BusyGuard&) = delete;
        BusyGuard& operator=(const BusyGuard&) = delete;
    private:
        AppCore& m_core;
        std::string m_previous;
    };

    const std::string m_root;
    const std::string m_version;
    FMBrain m_brain;
    Trainer m_trainer;
    Dataset m_dataset;
    CheckpointStore m_store;

    // settings
    mutable std::mutex m_cfgMutex;
    TrainerConfig m_config;
    bool m_settingsDirty = false;
    std::chrono::steady_clock::time_point m_settingsDirtySince{};

    // worker
    std::thread m_worker;
    std::atomic<bool> m_stop{ false };
    std::atomic<bool> m_started{ false };
    std::atomic<bool> m_wantTraining{ false };
    bool m_gpuPreferred = true;                     // the person's choice (guarded by m_cfgMutex); the real engine can be CPU for a while
    std::atomic<bool> m_previewWanted{ false };     // refresh the sample grid once the worker is free
    std::atomic<bool> m_computeSuspended{ false }; // the graphics chip is being swapped: no lessons for a moment
    std::mutex m_jobMutex;
    std::condition_variable m_jobCv;
    std::deque<std::function<void()>> m_jobs;
    std::mutex m_engineMutex; // held by the worker while it computes; the screen code takes it to swap the GPU in or out

    // state shown to the screens
    mutable std::mutex m_stateMutex;
    RunState m_state = RunState::Idle;
    std::string m_problem;
    std::string m_busyText;
    std::string m_gpuNote = "The graphics chip has not been started yet.";
    std::string m_lanUrl;
    double m_trainSeconds = 0.0;
    std::atomic<uint64_t> m_lastSavedStep{ 0 };
    std::chrono::steady_clock::time_point m_lastSaveTime{};
    bool m_everSaved = false;
    std::chrono::steady_clock::time_point m_startTime = std::chrono::steady_clock::now();
    std::atomic<uint64_t> m_snapshotListVersion{ 1 };
    mutable std::mutex m_countMutex;
    mutable uint64_t m_countVersion = 0;
    mutable uint32_t m_countValue = 0;

    // notices and logs
    mutable std::mutex m_noticeMutex;
    std::deque<Toast> m_toasts;
    uint64_t m_nextToastId = 1;
    std::deque<std::string> m_logs;
    std::function<std::string()> m_extraLog;
    std::function<void(const std::string&)> m_logSink;

    // colours of the picture set (crisp colours finish)
    std::mutex m_crispMutex;
    bool m_crispValid = false;
    uint64_t m_crispVersion = 0;
    uint32_t m_crispColors = 0;
    std::vector<uint8_t> m_crispPalette;

    // pictures made on request
    mutable std::mutex m_dreamMutex;
    std::deque<DreamEntry> m_dreams;
    uint64_t m_nextDreamId = 1;

    // worker-only scheduling state
    std::chrono::steady_clock::time_point m_lastPreview{};
    std::chrono::steady_clock::time_point m_lastLoopTime{};
    bool m_autosaveWarned = false;
    bool m_gentleNow = false;
};

} // namespace XboxOTFM
