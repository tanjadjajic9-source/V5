#pragma once

#include "Dataset.h"
#include "FMBrain.h"
#include "ImageUtil.h"
#include "Json.h"
#include "OptimalTransport.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace XboxOTFM {

enum class LrSchedule : uint32_t {
    Constant = 0, // the learning rate stays at the chosen value (after the warm-up)
    Cosine = 1    // smoothly falls to a small fraction of it by the training goal (needs a goal)
};

// The learning rate that suits a one-pass brain (the flow brains' default is TrainerConfig().learningRate = 0.001).
constexpr float kOnePassLearningRate = 0.0003f;

struct TrainerConfig {
    // --- learning ---
    float learningRate = 0.001f;
    LrSchedule schedule = LrSchedule::Cosine;
    uint32_t warmupSteps = 100;   // the learning rate rises from ~0 to its full value over this many lessons
    uint64_t targetSteps = 0;     // 0 = no goal (keep training until paused); otherwise training stops by itself here
    float lrFloor = 0.05f;        // cosine schedule ends at this fraction of the learning rate
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float adamEpsilon = 1e-8f;
    float gradClipNorm = 1.0f;
    // --- what is trained on ---
    uint32_t resolution = 32;     // 8..128
    uint32_t batchSize = 8;       // 1..32 (limited further at 128x128)
    AugmentOptions augment = AugmentOptions{ true, false, false }; // flip, zoom, colour
    // --- averaging (EMA): a slowly-moving average of the brain's dials that makes smoother, better pictures ---
    bool emaEnabled = true;
    float emaDecay = 0.999f;
    // --- pictures the brain makes ---
    uint32_t dreamSteps = 6;      // 1..50 Euler steps from noise to picture (a one-pass brain always draws in one pass)
    // One-pass brain only: how strongly the codes are pulled towards plain random numbers (the "KL weight"). It rises from 0 to this
    // value over the first kKlWarmupLessons lessons. Too low: sharp copies but drawings from random codes look like noise. Too high: blurry.
    float klWeight = 0.004f;
    // "Crisp colours" finish for pixel art: every pixel of the live monitor and of Create pictures is moved to the nearest of
    // this many colours, taken from your pictures. 0 = off. It is only a finish: training is not affected.
    uint32_t crispColors = 0;     // 0 (off) or 2..256
    // --- preparing new pictures ---
    // How a picture is resized when it is added: Auto keeps small pictures with few colours (pixel art) sharp and smooths the rest.
    ResizeStyle resize = ResizeStyle::Auto;
    // --- engine / pacing / housekeeping ---
    bool blindSpeedMode = false;  // skip loss read-back (a little faster, but no chart and no warning when training goes wrong)
    ComputeEngine engine = ComputeEngine::GPU;
    bool gentle = false;          // leave room for the TV screen: short pauses between lessons
    uint32_t autosaveMinutes = 5; // 0 = never
    uint32_t previewSeconds = 10; // how often the sample grid is refreshed while training
};

// Writes every setting as one JSON object.
void WriteConfigJson(JsonWriter& w, const TrainerConfig& c);
// Applies the fields present in `patch` (unknown fields are ignored, numbers are clamped into a safe range).
// `changed` (optional) receives the names of the settings that were really changed. Returns false if `patch` is not an object.
bool ApplyConfigPatch(TrainerConfig& c, const JsonValue& patch, std::vector<std::string>* changed = nullptr);

struct TrainingMetrics {
    uint64_t stepCount = 0;
    float lastMseLoss = 0.0f;
    float smoothLoss = 0.0f;       // smoothed over about the last 30 lessons
    float bestSmoothLoss = 0.0f;
    float currentLr = 0.0f;
    float stepsPerSec = 0.0f;
    float lessonsPerSec = 0.0f;    // pictures per second
    double epochs = 0.0;           // full passes over the pictures
    float lastRandomOtCost = 0.0f;
    float lastOptimalOtCost = 0.0f;
    uint32_t pictureCount = 0;     // 0 = practising on the built-in patterns
    bool practiceMode = false;
    uint32_t usedBatch = 0;
    uint32_t usedResolution = 0;
    bool usingGpuEngine = false;
    bool gpuFallbackTriggered = false;
    bool nonFinite = false;        // the loss became NaN/Infinity: the brain is broken
    DialScanReport lastDialScan;
};

struct PreviewImage {
    Rgb8Image grid;                // all samples side by side
    uint64_t version = 0;          // 0 = nothing generated yet
    uint64_t step = 0;             // training step the pictures were made at
    uint32_t tiles = 0;
    uint32_t columns = 0;
    uint32_t resolution = 0;       // size of one sample before it was enlarged
    uint32_t dreamSteps = 0;       // 0 = drawn in one pass (a one-pass brain)
    bool usedAverage = false;
};

class Trainer {
public:
    Trainer();

    // Limits the batch so a big lesson cannot run out of memory or trip the GPU watchdog. `resolution` is the width and height the
    // network really works at: the picture size for a normal brain, half of it for a wave brain.
    static uint32_t ClampBatchForResolution(uint32_t batchSize, uint32_t resolution, ComputeEngine engine = ComputeEngine::CPU) noexcept;
    // The picture size a brain really trains on and draws at: the setting limited to 8..128 and, for a wave brain (whose first step
    // halves the width and height), rounded down to an even number; for a one-pass brain rounded down to a multiple of 8.
    static uint32_t PictureSizeFor(uint32_t resolution, uint32_t waveLevel, BrainFamily family = BrainFamily::Flow) noexcept;
    // The one-pass brain's KL weight rises linearly from 0 over this many lessons.
    static constexpr uint64_t kKlWarmupLessons = 2000;
    static float KlWeightForStep(const TrainerConfig& config, uint64_t step) noexcept;
    // Learning rate for the (1-based) lesson number, including warm-up and schedule.
    static float ScheduledLearningRate(const TrainerConfig& config, uint64_t step) noexcept;

    // One lesson (GPU, or CPU when the GPU is off). With no pictures it practises on four built-in patterns.
    // Worker thread only. May switch config.engine to CPU if the GPU just failed. Returns the loss (0 in blind speed mode).
    float Step(FMBrain& brain, Dataset& dataset, TrainerConfig& config);

    // Makes `count` pictures from noise. fixedSeeds = the same noise every time (so progress can be compared); otherwise fresh noise.
    // useAverage = use the averaged dials when available. Worker thread only.
    // variety (one-pass brain only): the random codes are scaled by this (1 = ordinary random numbers, lower = tamer pictures).
    bool MakeSamples(FMBrain& brain, const TrainerConfig& config, uint32_t count, uint32_t resolution, uint32_t dreamSteps,
                     bool fixedSeeds, bool useAverage, std::vector<std::vector<float>>& outImagesCHW, bool* usedAverage = nullptr,
                     float variety = 1.0f);
    // Refreshes the fixed-noise sample grid shown on the phone and TV. Worker thread only.
    // finishPalette (optional): red, green, blue triples; when given, every pixel is moved to its nearest colour (crisp colours).
    bool UpdatePreview(FMBrain& brain, const TrainerConfig& config, const std::vector<uint8_t>* finishPalette = nullptr);

    // ---- progress bookkeeping (worker thread) ----
    void ResetProgress();                                  // new brain: step 0, empty chart, no averaged dials
    void NoteTrainingStarted();                            // forget the time of the last lesson (for the speed readout)
    void RestoreProgress(uint64_t step, float smoothLoss, double epochs,
                         const std::vector<std::pair<uint64_t, float>>& curve, std::vector<float> averagedDials);
    void SetEpochBase(double epochs) { m_epochBase = epochs; }
    const std::vector<float>& AveragedDials() const { return m_ema; }
    bool HasAveragedDials() const { return !m_ema.empty(); }
    void SetDialScanReport(const DialScanReport& report);
    void SetPreviewSeed(uint64_t seed) { m_previewSeed = seed; }

    // ---- read from any thread ----
    TrainingMetrics SnapshotMetrics() const;
    PreviewImage LatestPreview() const;
    uint64_t PreviewVersion() const;
    bool PreviewReady() const;     // false right after a reset, until the first sample grid is made
    // Loss chart points (step, loss) reduced to at most maxPoints.
    std::vector<std::pair<uint64_t, float>> LossCurve(size_t maxPoints) const;
    uint64_t LossVersion() const;

    static void PracticePattern(uint32_t index, uint32_t resolution, std::vector<float>& outCHW);
    static constexpr uint32_t kPracticePatterns = 4;

private:
    static void Conv2dBackward(const float* input, const float* gradOutput, const ConvLayerWeights& layer, uint32_t batch,
                               uint32_t height, uint32_t width, std::vector<float>& gradInput, std::vector<float>& gradWeight,
                               std::vector<float>& gradBias);
    static void AdamUpdate(std::vector<float>& param, const std::vector<float>& grad, std::vector<float>& m, std::vector<float>& v,
                           uint64_t step, float lr, float beta1, float beta2, float eps, float clipScale);

    void RecordLoss(uint64_t step, float loss);
    void UpdateAverage(FMBrain& brain, const TrainerConfig& config, uint64_t step);

    // What one lesson did (the bookkeeping that follows every lesson is the same for both brain families).
    struct LessonOutcome {
        uint64_t step = 0;
        float lr = 0.0f;
        uint32_t batch = 0;
        uint32_t resolution = 0;
        bool practice = false;
        size_t pictureCount = 0;
        bool usedGpu = false;
        bool fallbackTriggered = false;
        float loss = 0.0f;
        float randomPairingCost = 0.0f;   // flow brain only
        float optimalPairingCost = 0.0f;  // flow brain only
    };
    // Picks `batch` pictures of size `res` (or the built-in patterns) into m_targets. Returns true when it used the built-in patterns.
    bool DealTargets(Dataset& dataset, const TrainerConfig& config, uint32_t batch, uint32_t res);
    // Metrics, loss chart and averaged dials after a lesson; returns the loss.
    float FinishLesson(FMBrain& brain, Dataset& dataset, const TrainerConfig& config, const LessonOutcome& o);
    // The lesson of a one-pass brain (m_targets already holds the pictures).
    float StepOnePass(FMBrain& brain, Dataset& dataset, TrainerConfig& config, uint32_t batch, uint32_t res, bool practice,
                      size_t pictureCount, bool fallbackTriggered);

    // Worker-thread-only state
    OptimalTransportMatcher m_otMatcher;
    ForwardCache m_cache;
    std::mt19937_64 m_sampleRng;
    std::mt19937_64 m_dreamRng;
    std::vector<float> m_targets;
    std::vector<float> m_noise;        // one-pass brain: the random numbers of the codes of one lesson
    std::vector<float> m_waveTargets;  // the same pictures in wave form (wave brains only)
    std::vector<float> m_ema;          // averaged copy of the dials
    uint64_t m_emaStep = 0;            // lesson at which m_ema was last updated
    std::vector<float> m_scratch;
    std::vector<std::vector<float>> m_practice; // the built-in patterns at m_practiceRes
    uint32_t m_practiceRes = 0;
    uint64_t m_previewSeed = 0x5EEDF00DULL;
    double m_epochBase = 0.0;
    bool m_haveLastTime = false;
    std::chrono::steady_clock::time_point m_lastStepTime{};

    // Shared state
    mutable std::mutex m_stateMutex;
    TrainingMetrics m_metrics;
    PreviewImage m_preview;
    std::vector<std::pair<uint64_t, float>> m_curve; // decimated (step, loss) points
    uint64_t m_curveStride = 1;
    double m_curveAcc = 0.0;
    uint64_t m_curveCount = 0;
    uint64_t m_lossVersion = 0;
    uint64_t m_smoothCount = 0;
};

} // namespace XboxOTFM
