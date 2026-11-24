#include "Trainer.h"

#include "Checkpoint.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>

namespace XboxOTFM {

namespace {

constexpr uint32_t kAverageEvery = 8;        // the averaged dials are refreshed every 8th lesson
constexpr size_t kCurveCapacity = 1024;      // loss chart points kept before neighbours are merged
constexpr float kPi = 3.14159265358979f;

inline float SiLUDerivative(float x) noexcept {
    const float clamped = std::clamp(x, -20.0f, 20.0f);
    const float sig = 1.0f / (1.0f + std::exp(-clamped));
    return sig * (1.0f + clamped * (1.0f - sig));
}

// Puts the dials back the way they were, even if something throws on the way.
struct DialRestore {
    FMBrain& brain;
    std::vector<float> raw;
    bool active = false;
    explicit DialRestore(FMBrain& b) : brain(b) {}
    ~DialRestore() {
        if (active) {
            try { BrainIO::ApplyParams(brain, raw); } catch (...) {}
        }
    }
    DialRestore(const DialRestore&) = delete;
    DialRestore& operator=(const DialRestore&) = delete;
};

template <class T>
bool Number(const JsonValue& v, const char* key, double lo, double hi, T& out, bool integer) {
    double d = 0.0;
    if (!v.GetNumber(key, d)) return false;
    d = std::clamp(d, lo, hi);
    if (integer) d = std::floor(d + 0.5);
    out = static_cast<T>(d);
    return true;
}

} // namespace

// ---------------------------------------------------------------------------------------------
// Settings <-> JSON
// ---------------------------------------------------------------------------------------------
void WriteConfigJson(JsonWriter& w, const TrainerConfig& c) {
    w.BeginObject();
    w.Field("lr", c.learningRate);
    w.Field("schedule", c.schedule == LrSchedule::Constant ? "constant" : "cosine");
    w.Field("warmup", c.warmupSteps);
    w.Field("goal", c.targetSteps);
    w.Field("lrFloor", c.lrFloor);
    w.Field("beta1", c.beta1);
    w.Field("beta2", c.beta2);
    w.Field("epsilon", static_cast<double>(c.adamEpsilon));
    w.Field("clip", c.gradClipNorm);
    w.Field("res", c.resolution);
    w.Field("batch", c.batchSize);
    w.Field("flip", c.augment.flip);
    w.Field("zoom", c.augment.zoom);
    w.Field("color", c.augment.color);
    w.Field("ema", c.emaEnabled);
    w.Field("emaDecay", c.emaDecay);
    w.Field("dreamSteps", c.dreamSteps);
    w.Field("kl", c.klWeight);
    w.Field("crisp", c.crispColors);
    w.Field("resize", c.resize == ResizeStyle::Sharp ? "sharp" : (c.resize == ResizeStyle::Smooth ? "smooth" : "auto"));
    w.Field("blind", c.blindSpeedMode);
    w.Field("engine", c.engine == ComputeEngine::GPU ? "gpu" : "cpu");
    w.Field("gentle", c.gentle);
    w.Field("autosaveMin", c.autosaveMinutes);
    w.Field("previewSec", c.previewSeconds);
    w.EndObject();
}

bool ApplyConfigPatch(TrainerConfig& c, const JsonValue& p, std::vector<std::string>* changed) {
    if (!p.IsObject()) return false;
    const TrainerConfig before = c;
    auto note = [&](const char* name, bool differs) { if (differs && changed) changed->push_back(name); };

    Number(p, "lr", 1e-5, 0.05, c.learningRate, false);
    Number(p, "warmup", 0, 5000, c.warmupSteps, true);
    {
        double goal = 0.0;
        if (p.GetNumber("goal", goal)) {
            goal = std::floor(goal + 0.5);
            c.targetSteps = goal < 1.0 ? 0u : static_cast<uint64_t>(std::clamp(goal, 100.0, 10000000.0));
        }
    }
    Number(p, "lrFloor", 0.0, 1.0, c.lrFloor, false);
    Number(p, "beta1", 0.5, 0.999, c.beta1, false);
    Number(p, "beta2", 0.9, 0.9999, c.beta2, false);
    Number(p, "epsilon", 1e-10, 1e-4, c.adamEpsilon, false);
    Number(p, "clip", 0.05, 100.0, c.gradClipNorm, false);
    Number(p, "res", 8, 128, c.resolution, true);
    Number(p, "batch", 1, 32, c.batchSize, true);
    Number(p, "emaDecay", 0.9, 0.99999, c.emaDecay, false);
    Number(p, "dreamSteps", 1, 50, c.dreamSteps, true);
    Number(p, "kl", 0.0, 1.0, c.klWeight, false);
    if (Number(p, "crisp", 0, 256, c.crispColors, true) && c.crispColors == 1) c.crispColors = 2; // one colour makes no picture
    Number(p, "autosaveMin", 0, 120, c.autosaveMinutes, true);
    Number(p, "previewSec", 2, 120, c.previewSeconds, true);
    p.GetBool("flip", c.augment.flip);
    p.GetBool("zoom", c.augment.zoom);
    p.GetBool("color", c.augment.color);
    p.GetBool("ema", c.emaEnabled);
    p.GetBool("blind", c.blindSpeedMode);
    p.GetBool("gentle", c.gentle);
    std::string s;
    if (p.GetString("schedule", s)) {
        if (s == "constant") c.schedule = LrSchedule::Constant;
        else if (s == "cosine") c.schedule = LrSchedule::Cosine;
    }
    if (p.GetString("engine", s)) {
        if (s == "gpu") c.engine = ComputeEngine::GPU;
        else if (s == "cpu") c.engine = ComputeEngine::CPU;
    }
    if (p.GetString("resize", s)) {
        if (s == "auto") c.resize = ResizeStyle::Auto;
        else if (s == "sharp") c.resize = ResizeStyle::Sharp;
        else if (s == "smooth") c.resize = ResizeStyle::Smooth;
    }

    note("lr", c.learningRate != before.learningRate);
    note("schedule", c.schedule != before.schedule);
    note("warmup", c.warmupSteps != before.warmupSteps);
    note("goal", c.targetSteps != before.targetSteps);
    note("lrFloor", c.lrFloor != before.lrFloor);
    note("beta1", c.beta1 != before.beta1);
    note("beta2", c.beta2 != before.beta2);
    note("epsilon", c.adamEpsilon != before.adamEpsilon);
    note("clip", c.gradClipNorm != before.gradClipNorm);
    note("res", c.resolution != before.resolution);
    note("batch", c.batchSize != before.batchSize);
    note("flip", c.augment.flip != before.augment.flip);
    note("zoom", c.augment.zoom != before.augment.zoom);
    note("color", c.augment.color != before.augment.color);
    note("ema", c.emaEnabled != before.emaEnabled);
    note("emaDecay", c.emaDecay != before.emaDecay);
    note("dreamSteps", c.dreamSteps != before.dreamSteps);
    note("kl", c.klWeight != before.klWeight);
    note("crisp", c.crispColors != before.crispColors);
    note("resize", c.resize != before.resize);
    note("blind", c.blindSpeedMode != before.blindSpeedMode);
    note("engine", c.engine != before.engine);
    note("gentle", c.gentle != before.gentle);
    note("autosaveMin", c.autosaveMinutes != before.autosaveMinutes);
    note("previewSec", c.previewSeconds != before.previewSeconds);
    return true;
}

// ---------------------------------------------------------------------------------------------
// Basics
// ---------------------------------------------------------------------------------------------
Trainer::Trainer()
    : m_otMatcher(0x9E3779B97F4A7C15ULL)
    , m_sampleRng(std::random_device{}() ^ 0xA5A55A5A12345678ULL)
    , m_dreamRng(std::random_device{}() ^ 0xCAFEBABE12345678ULL) {
}

uint32_t Trainer::ClampBatchForResolution(uint32_t batchSize, uint32_t resolution, ComputeEngine engine) noexcept {
    if (resolution >= 128u) {
        const uint32_t cap = (engine == ComputeEngine::GPU) ? 2u : 4u;
        return std::clamp(batchSize, 1u, cap);
    }
    return std::clamp(batchSize, 1u, 32u);
}

uint32_t Trainer::PictureSizeFor(uint32_t resolution, uint32_t waveLevel, BrainFamily family) noexcept {
    const uint32_t res = std::clamp(resolution, 8u, 128u);
    const uint32_t unit = (family == BrainFamily::OnePass) ? FMBrain::kOnePassUnit : (1u << std::min(waveLevel, FMBrain::kMaxWaveLevel));
    return res - (res % unit);
}

float Trainer::KlWeightForStep(const TrainerConfig& c, uint64_t step) noexcept {
    const double ramp = std::min(1.0, static_cast<double>(step) / static_cast<double>(kKlWarmupLessons));
    return static_cast<float>(static_cast<double>(c.klWeight) * ramp);
}

float Trainer::ScheduledLearningRate(const TrainerConfig& c, uint64_t step) noexcept {
    double scale = 1.0;
    if (c.warmupSteps > 0 && step <= c.warmupSteps) {
        scale = static_cast<double>(std::max<uint64_t>(step, 1)) / static_cast<double>(c.warmupSteps);
    }
    if (c.schedule == LrSchedule::Cosine && c.targetSteps > c.warmupSteps) {
        const double span = static_cast<double>(c.targetSteps - c.warmupSteps);
        const double done = step > c.warmupSteps ? static_cast<double>(step - c.warmupSteps) : 0.0;
        const double progress = std::clamp(done / span, 0.0, 1.0);
        const double floorFrac = std::clamp(static_cast<double>(c.lrFloor), 0.0, 1.0);
        scale *= floorFrac + (1.0 - floorFrac) * 0.5 * (1.0 + std::cos(static_cast<double>(kPi) * progress));
    }
    return static_cast<float>(static_cast<double>(c.learningRate) * scale);
}

void Trainer::PracticePattern(uint32_t index, uint32_t resolution, std::vector<float>& card) {
    const uint32_t s = std::clamp(resolution, 8u, 128u);
    const size_t hw = static_cast<size_t>(s) * s;
    card.assign(3 * hw, 0.0f);
    for (uint32_t y = 0; y < s; ++y) {
        const float ny = 2.0f * static_cast<float>(y) / static_cast<float>(s - 1) - 1.0f;
        for (uint32_t x = 0; x < s; ++x) {
            const float nx = 2.0f * static_cast<float>(x) / static_cast<float>(s - 1) - 1.0f;
            const size_t p = static_cast<size_t>(y) * s + x;
            switch (index % kPracticePatterns) {
                case 0: { // emerald ring
                    const float r = std::sqrt(nx * nx + ny * ny);
                    const float ring = std::exp(-14.0f * (r - 0.55f) * (r - 0.55f));
                    card[0 * hw + p] = -0.8f + 0.6f * ring;
                    card[1 * hw + p] = -0.7f + 1.6f * ring;
                    card[2 * hw + p] = -0.7f + 0.9f * ring;
                    break;
                }
                case 1: { // warm diagonal wave
                    const float wave = 0.5f + 0.5f * std::sin(kPi * (nx + ny));
                    card[0 * hw + p] = -0.5f + 1.4f * wave;
                    card[1 * hw + p] = -0.6f + 0.9f * wave;
                    card[2 * hw + p] = -0.8f + 0.3f * (1.0f - wave);
                    break;
                }
                case 2: { // diamond
                    const float d = std::abs(nx) + std::abs(ny);
                    const float mask = (d < 0.72f) ? 1.0f : 0.0f;
                    card[0 * hw + p] = -0.7f + 1.3f * mask;
                    card[1 * hw + p] = -0.4f + 1.2f * mask;
                    card[2 * hw + p] = 0.4f + 0.5f * mask;
                    break;
                }
                default: { // quadrant checker
                    const float sgn = (nx * ny >= 0.0f) ? 0.75f : -0.75f;
                    card[0 * hw + p] = 0.2f * sgn;
                    card[1 * hw + p] = 0.8f * sgn;
                    card[2 * hw + p] = -0.5f * sgn;
                    break;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// CPU maths (unchanged from the proven CPU path)
// ---------------------------------------------------------------------------------------------
void Trainer::Conv2dBackward(
    const float* input,
    const float* gradOutput,
    const ConvLayerWeights& layer,
    uint32_t batch,
    uint32_t height,
    uint32_t width,
    std::vector<float>& gradInput,
    std::vector<float>& gradWeight,
    std::vector<float>& gradBias) {
    const uint32_t inC = layer.inChannels;
    const uint32_t outC = layer.outChannels;
    const uint32_t k = layer.kernelSize;
    const int pad = static_cast<int>(k / 2);
    const size_t hw = static_cast<size_t>(height) * width;

    gradInput.assign(static_cast<size_t>(batch) * inC * hw, 0.0f);
    gradWeight.assign(layer.weight.size(), 0.0f);
    gradBias.assign(outC, 0.0f);

    for (uint32_t n = 0; n < batch; ++n) {
        const float* inBatch = input + static_cast<size_t>(n) * inC * hw;
        const float* gOutBatch = gradOutput + static_cast<size_t>(n) * outC * hw;
        float* gInBatch = gradInput.data() + static_cast<size_t>(n) * inC * hw;

        for (uint32_t oc = 0; oc < outC; ++oc) {
            const float* gOutPlane = gOutBatch + static_cast<size_t>(oc) * hw;

            double bSum = 0.0;
            for (size_t idx = 0; idx < hw; ++idx) {
                bSum += gOutPlane[idx];
            }
            gradBias[oc] += static_cast<float>(bSum);

            for (uint32_t ic = 0; ic < inC; ++ic) {
                const float* inPlane = inBatch + static_cast<size_t>(ic) * hw;
                float* gInPlane = gInBatch + static_cast<size_t>(ic) * hw;
                const float* wKernel = layer.weight.data() + ((static_cast<size_t>(oc) * inC + ic) * k * k);
                float* gWKernel = gradWeight.data() + ((static_cast<size_t>(oc) * inC + ic) * k * k);

                if (k == 1) {
                    const float w = wKernel[0];
                    double gwAcc = 0.0;
                    for (size_t idx = 0; idx < hw; ++idx) {
                        const float go = gOutPlane[idx];
                        gwAcc += static_cast<double>(go) * static_cast<double>(inPlane[idx]);
                        gInPlane[idx] += go * w;
                    }
                    gWKernel[0] += static_cast<float>(gwAcc);
                } else {
                    for (uint32_t y = 0; y < height; ++y) {
                        for (uint32_t x = 0; x < width; ++x) {
                            const float go = gOutPlane[static_cast<size_t>(y) * width + x];
                            for (int ky = 0; ky < static_cast<int>(k); ++ky) {
                                const int iy = static_cast<int>(y) + ky - pad;
                                if (iy < 0 || iy >= static_cast<int>(height)) continue;
                                for (int kx = 0; kx < static_cast<int>(k); ++kx) {
                                    const int ix = static_cast<int>(x) + kx - pad;
                                    if (ix < 0 || ix >= static_cast<int>(width)) continue;
                                    const size_t inIdx = static_cast<size_t>(iy) * width + ix;
                                    const size_t kIdx = static_cast<size_t>(ky) * k + kx;
                                    gWKernel[kIdx] += go * inPlane[inIdx];
                                    gInPlane[inIdx] += go * wKernel[kIdx];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void Trainer::AdamUpdate(
    std::vector<float>& param,
    const std::vector<float>& grad,
    std::vector<float>& m,
    std::vector<float>& v,
    uint64_t step,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float clipScale) {
    const float bc1 = 1.0f - std::pow(beta1, static_cast<float>(step));
    const float bc2 = 1.0f - std::pow(beta2, static_cast<float>(step));

    for (size_t i = 0; i < param.size(); ++i) {
        const float g = grad[i] * clipScale;
        m[i] = beta1 * m[i] + (1.0f - beta1) * g;
        v[i] = beta2 * v[i] + (1.0f - beta2) * g * g;
        const float mHat = m[i] / std::max(1e-8f, bc1);
        const float vHat = v[i] / std::max(1e-8f, bc2);
        param[i] -= lr * mHat / (std::sqrt(vHat) + eps);
    }
}

// ---------------------------------------------------------------------------------------------
// One lesson
// ---------------------------------------------------------------------------------------------
bool Trainer::DealTargets(Dataset& dataset, const TrainerConfig& config, uint32_t batch, uint32_t res) {
    const size_t stride = static_cast<size_t>(3) * res * res;
    m_targets.assign(static_cast<size_t>(batch) * stride, 0.0f);
    const size_t pictureCount = dataset.Size();
    bool practice = true;
    if (pictureCount > 0 && dataset.SampleBatch(batch, res, config.augment, m_sampleRng, m_targets) &&
        m_targets.size() == static_cast<size_t>(batch) * stride) {
        practice = false;
    }
    if (practice) {
        m_targets.assign(static_cast<size_t>(batch) * stride, 0.0f);
        if (m_practiceRes != res || m_practice.size() != kPracticePatterns) {
            m_practice.assign(kPracticePatterns, std::vector<float>());
            for (uint32_t i = 0; i < kPracticePatterns; ++i) PracticePattern(i, res, m_practice[i]);
            m_practiceRes = res;
        }
        std::uniform_int_distribution<uint32_t> pick(0, kPracticePatterns - 1);
        std::uniform_real_distribution<float> coin(0.0f, 1.0f);
        for (uint32_t i = 0; i < batch; ++i) {
            float* dst = m_targets.data() + static_cast<size_t>(i) * stride;
            const std::vector<float>& src = m_practice[pick(m_sampleRng)];
            const bool flip = config.augment.flip && coin(m_sampleRng) < 0.5f;
            for (uint32_t c = 0; c < 3; ++c) {
                for (uint32_t y = 0; y < res; ++y) {
                    const float* srow = src.data() + (static_cast<size_t>(c) * res + y) * res;
                    float* drow = dst + (static_cast<size_t>(c) * res + y) * res;
                    if (flip) {
                        for (uint32_t x = 0; x < res; ++x) drow[x] = srow[res - 1 - x];
                    } else {
                        std::memcpy(drow, srow, static_cast<size_t>(res) * sizeof(float));
                    }
                }
            }
        }
    }
    return practice;
}

float Trainer::FinishLesson(FMBrain& brain, Dataset& dataset, const TrainerConfig& config, const LessonOutcome& o) {
    const auto t1 = std::chrono::steady_clock::now();
    float instRate = 0.0f;
    if (m_haveLastTime) {
        const float dt = std::chrono::duration<float>(t1 - m_lastStepTime).count();
        if (dt > 1e-5f && dt < 60.0f) instRate = 1.0f / dt;
    }
    m_lastStepTime = t1;
    m_haveLastTime = true;

    const bool lossKnown = !config.blindSpeedMode;
    const bool lossFinite = lossKnown && std::isfinite(o.loss);
    {
        std::lock_guard<std::mutex> lock(m_stateMutex);
        m_metrics.usingGpuEngine = o.usedGpu;
        // True only in the snapshot taken right after a lesson that had to leave the GPU.
        m_metrics.gpuFallbackTriggered = o.fallbackTriggered;
        m_metrics.pictureCount = static_cast<uint32_t>(o.practice ? 0 : o.pictureCount);
        m_metrics.practiceMode = o.practice;
        m_metrics.usedBatch = o.batch;
        m_metrics.usedResolution = o.resolution;
        m_metrics.currentLr = o.lr;
        if (instRate > 0.0f) {
            m_metrics.stepsPerSec = (m_metrics.stepsPerSec <= 0.0f) ? instRate : (0.9f * m_metrics.stepsPerSec + 0.1f * instRate);
            m_metrics.lessonsPerSec = m_metrics.stepsPerSec * static_cast<float>(o.batch);
        }
        m_metrics.epochs = o.practice ? 0.0 : (m_epochBase + dataset.EpochProgress());
        if (lossKnown) {
            if (lossFinite) {
                m_metrics.lastMseLoss = o.loss;
                ++m_smoothCount;
                const float alpha = std::max(0.03f, 1.0f / static_cast<float>(std::min<uint64_t>(m_smoothCount, 1000)));
                m_metrics.smoothLoss = (m_smoothCount == 1) ? o.loss : (1.0f - alpha) * m_metrics.smoothLoss + alpha * o.loss;
                if (m_smoothCount >= 20 && (m_metrics.bestSmoothLoss <= 0.0f || m_metrics.smoothLoss < m_metrics.bestSmoothLoss)) {
                    m_metrics.bestSmoothLoss = m_metrics.smoothLoss;
                }
                m_metrics.lastRandomOtCost = o.randomPairingCost;
                m_metrics.lastOptimalOtCost = o.optimalPairingCost;
            } else {
                m_metrics.nonFinite = true;
            }
        }
    }
    if (lossFinite) RecordLoss(o.step, o.loss);

    // The averaged dials are only kept from healthy lessons.
    if (!(lossKnown && !lossFinite)) UpdateAverage(brain, config, o.step);
    return o.loss;
}

float Trainer::StepOnePass(FMBrain& brain, Dataset& dataset, TrainerConfig& config, uint32_t batch, uint32_t res, bool practice,
                           size_t pictureCount, bool fallbackTriggered) {
    // The random numbers behind the codes: plain standard normal numbers, new ones every lesson.
    const uint64_t codeSize = brain.OnePassCodeSize(res);
    m_noise.resize(static_cast<size_t>(batch) * static_cast<size_t>(codeSize));
    {
        std::normal_distribution<float> nd(0.0f, 1.0f);
        for (float& v : m_noise) v = nd(m_sampleRng);
    }

    uint64_t nextStep = 0;
    {
        std::lock_guard<std::mutex> stateLock(m_stateMutex);
        nextStep = ++m_metrics.stepCount;
    }
    const float lr = ScheduledLearningRate(config, nextStep);
    const float klNow = KlWeightForStep(config, nextStep);

    float loss = 0.0f;
    bool usedGpu = false;
    {
        std::lock_guard<std::recursive_mutex> brainLock(brain.Mutex());
        if (config.engine == ComputeEngine::GPU && brain.IsGPUReady()) {
            const HRESULT hr = brain.TrainStepOnePassGpu(m_targets.data(), m_noise.data(), batch, res, nextStep, lr, config.beta1, config.beta2,
                                                         config.adamEpsilon, klNow, config.gradClipNorm, !config.blindSpeedMode, loss);
            if (SUCCEEDED(hr)) {
                usedGpu = true;
            } else {
                // Same graceful fallback as the flow brain: salvage what the graphics chip had learned, then carry on on the processor.
                config.engine = ComputeEngine::CPU;
                fallbackTriggered = true;
                loss = 0.0f;
                brain.ReadbackWeightsFromGPU();
            }
        }
        if (!usedGpu) {
            OnePassLossInfo info;
            const bool ok = brain.TrainStepOnePassCpu(m_targets.data(), m_noise.data(), batch, res, nextStep, lr, config.beta1, config.beta2,
                                                      config.adamEpsilon, klNow, config.gradClipNorm, &info);
            if (config.blindSpeedMode) loss = 0.0f;
            else loss = ok ? info.recon : std::numeric_limits<float>::quiet_NaN();
        }
    }

    LessonOutcome o;
    o.step = nextStep;
    o.lr = lr;
    o.batch = batch;
    o.resolution = res;
    o.practice = practice;
    o.pictureCount = pictureCount;
    o.usedGpu = usedGpu;
    o.fallbackTriggered = fallbackTriggered;
    o.loss = loss;
    return FinishLesson(brain, dataset, config, o);
}

float Trainer::Step(FMBrain& brain, Dataset& dataset, TrainerConfig& config) {
    // A wave brain works on the wave form of every picture: half the width and height, four times the channels. A one-pass brain
    // works on the pictures themselves, whose sides are a multiple of 8.
    const bool onePass = brain.IsOnePass();
    const uint32_t wave = onePass ? 0u : brain.WaveLevel();
    const uint32_t res = PictureSizeFor(config.resolution, wave, brain.Family()); // size of the pictures
    const uint32_t netRes = res >> wave;                                          // width and height the network works at
    const uint32_t netC = brain.Architecture().inputImageChannels;

    // If [Engine: GPU] is selected but the brain no longer has a working GPU path (e.g. the GPU hung), say so once
    // instead of silently training on the CPU while the screen claims "GPU".
    bool fallbackTriggered = false;
    if (config.engine == ComputeEngine::GPU && !brain.IsGPUReady()) {
        config.engine = ComputeEngine::CPU;
        fallbackTriggered = true;
    }

    // The chosen batch size in the settings is never changed here; only this lesson is limited.
    uint32_t batch = ClampBatchForResolution(config.batchSize, netRes, config.engine);
    if (config.engine == ComputeEngine::GPU) {
        batch = std::min(batch, brain.MaxGpuBatchForResolution(netRes));
    }
    batch = std::max(1u, batch);

    // ---- deal the target pictures (in shuffled passes over the whole set) ----
    const size_t stride = static_cast<size_t>(3) * res * res;
    const size_t pictureCount = dataset.Size();
    const bool practice = DealTargets(dataset, config, batch, res);
    if (onePass) return StepOnePass(brain, dataset, config, batch, res, practice, pictureCount, fallbackTriggered);

    // The lesson is made in the form the network works in. For a wave brain every picture is first turned into its wave form (nothing
    // is lost), so the noise, the pairing of noise with pictures and the loss all live there; the numbers per picture stay the same
    // (12 x half x half = 3 x full x full), so the loss is on the same scale as for a normal brain.
    const float* lessonTargets = m_targets.data();
    if (wave != 0) {
        m_waveTargets.resize(m_targets.size());
        for (uint32_t i = 0; i < batch; ++i) {
            const size_t at = static_cast<size_t>(i) * stride;
            if (!HaarForward(m_targets.data() + at, 3, res, res, m_waveTargets.data() + at)) {
                std::fill(m_waveTargets.begin() + static_cast<std::ptrdiff_t>(at), m_waveTargets.begin() + static_cast<std::ptrdiff_t>(at + stride), 0.0f);
            }
        }
        lessonTargets = m_waveTargets.data();
    }
    TensorShape shape;
    shape.batch = batch;
    shape.channels = netC;
    shape.height = netRes;
    shape.width = netRes;
    OTBatchResult otBatch = m_otMatcher.BuildTrainingBatchFromTargets(lessonTargets, shape);

    uint64_t nextStep = 0;
    {
        std::lock_guard<std::mutex> stateLock(m_stateMutex);
        nextStep = ++m_metrics.stepCount;
    }
    const float lr = ScheduledLearningRate(config, nextStep);

    float mseLoss = 0.0f;
    bool usedGpu = false;

    {
        std::lock_guard<std::recursive_mutex> brainLock(brain.Mutex());

        // Attempt real GPU training when [Engine: GPU] is selected
        if (config.engine == ComputeEngine::GPU && brain.IsGPUReady()) {
            const HRESULT hr = brain.TrainStepGPU(
                otBatch.xt.data(),
                otBatch.vTarget.data(),
                otBatch.tValues.data(),
                batch,
                netRes,
                netRes,
                nextStep,
                lr,
                config.beta1,
                config.beta2,
                config.adamEpsilon,
                !config.blindSpeedMode, // completely skip loss read-back in blind speed mode
                mseLoss,
                config.gradClipNorm);   // same gradient clipping as the CPU trainer

            if (SUCCEEDED(hr)) {
                usedGpu = true;
            } else {
                // Graceful fallback to the CPU trainer without crashing. Salvage whatever the GPU had learned first
                // (a no-op if the GPU is gone; the CPU copy is then the last 30-second checkpoint).
                config.engine = ComputeEngine::CPU;
                fallbackTriggered = true;
                brain.ReadbackWeightsFromGPU();
            }
        }

        if (!usedGpu) {
            // Sync any GPU-trained weights before the CPU step if switching from GPU -> CPU
            brain.ReadbackWeightsFromGPU();
            brain.Forward(otBatch.xt.data(), otBatch.tValues.data(), batch, netRes, netRes, m_cache);

            const size_t outCount = otBatch.vTarget.size();
            const float invCount = 1.0f / static_cast<float>(std::max<size_t>(1, outCount));
            std::vector<float> gradPred(outCount);

            if (config.blindSpeedMode) {
                for (size_t i = 0; i < outCount; ++i) {
                    gradPred[i] = 2.0f * invCount * (m_cache.velocityPred[i] - otBatch.vTarget[i]);
                }
            } else {
                double mseSum = 0.0;
                for (size_t i = 0; i < outCount; ++i) {
                    const float diff = m_cache.velocityPred[i] - otBatch.vTarget[i];
                    mseSum += static_cast<double>(diff) * static_cast<double>(diff);
                    gradPred[i] = 2.0f * invCount * diff;
                }
                mseLoss = static_cast<float>(mseSum * invCount);
            }

            std::vector<float> gradHidden;
            std::vector<float> gradHeadW, gradHeadB;
            Conv2dBackward(m_cache.headInput.data(), gradPred.data(), brain.Head(), batch, netRes, netRes, gradHidden, gradHeadW, gradHeadB);

            auto& blocks = brain.Blocks();
            const size_t numBlocks = blocks.size();
            std::vector<std::vector<float>> gradBlkW1(numBlocks), gradBlkB1(numBlocks);
            std::vector<std::vector<float>> gradBlkW2(numBlocks), gradBlkB2(numBlocks);
            std::vector<float> gradActMid;
            std::vector<float> gradBlockIn;

            for (size_t b = numBlocks; b-- > 0;) {
                Conv2dBackward(m_cache.blockMidAct[b].data(), gradHidden.data(), blocks[b].conv2, batch, netRes, netRes,
                               gradActMid, gradBlkW2[b], gradBlkB2[b]);
                for (size_t i = 0; i < gradActMid.size(); ++i) {
                    gradActMid[i] *= SiLUDerivative(m_cache.blockMidPre[b][i]);
                }
                Conv2dBackward(m_cache.blockIn[b].data(), gradActMid.data(), blocks[b].conv1, batch, netRes, netRes,
                               gradBlockIn, gradBlkW1[b], gradBlkB1[b]);
                for (size_t i = 0; i < gradHidden.size(); ++i) {
                    gradHidden[i] = gradHidden[i] + gradBlockIn[i];
                }
            }

            for (size_t i = 0; i < gradHidden.size(); ++i) {
                gradHidden[i] *= SiLUDerivative(m_cache.stemPreAct[i]);
            }

            std::vector<float> gradStemIn, gradStemW, gradStemB;
            Conv2dBackward(m_cache.stemInput.data(), gradHidden.data(), brain.Stem(), batch, netRes, netRes, gradStemIn, gradStemW, gradStemB);

            double gradNormSq = 0.0;
            auto accumNorm = [&](const std::vector<float>& g) {
                for (float val : g) gradNormSq += static_cast<double>(val) * static_cast<double>(val);
            };
            accumNorm(gradStemW);
            accumNorm(gradStemB);
            for (size_t b = 0; b < numBlocks; ++b) {
                accumNorm(gradBlkW1[b]);
                accumNorm(gradBlkB1[b]);
                accumNorm(gradBlkW2[b]);
                accumNorm(gradBlkB2[b]);
            }
            accumNorm(gradHeadW);
            accumNorm(gradHeadB);

            const float gradNorm = static_cast<float>(std::sqrt(gradNormSq));
            const float clipScale = (gradNorm > config.gradClipNorm && gradNorm > 1e-6f) ? (config.gradClipNorm / gradNorm) : 1.0f;

            const float b1 = config.beta1, b2 = config.beta2, eps = config.adamEpsilon;
            AdamUpdate(brain.Stem().weight, gradStemW, brain.StemMW(), brain.StemVW(), nextStep, lr, b1, b2, eps, clipScale);
            AdamUpdate(brain.Stem().bias, gradStemB, brain.StemMB(), brain.StemVB(), nextStep, lr, b1, b2, eps, clipScale);
            for (size_t b = 0; b < numBlocks; ++b) {
                AdamUpdate(blocks[b].conv1.weight, gradBlkW1[b], blocks[b].mW1, blocks[b].vW1, nextStep, lr, b1, b2, eps, clipScale);
                AdamUpdate(blocks[b].conv1.bias, gradBlkB1[b], blocks[b].mB1, blocks[b].vB1, nextStep, lr, b1, b2, eps, clipScale);
                AdamUpdate(blocks[b].conv2.weight, gradBlkW2[b], blocks[b].mW2, blocks[b].vW2, nextStep, lr, b1, b2, eps, clipScale);
                AdamUpdate(blocks[b].conv2.bias, gradBlkB2[b], blocks[b].mB2, blocks[b].vB2, nextStep, lr, b1, b2, eps, clipScale);
            }
            AdamUpdate(brain.Head().weight, gradHeadW, brain.HeadMW(), brain.HeadVW(), nextStep, lr, b1, b2, eps, clipScale);
            AdamUpdate(brain.Head().bias, gradHeadB, brain.HeadMB(), brain.HeadVB(), nextStep, lr, b1, b2, eps, clipScale);

            // The CPU vectors are now newer than any copy in video memory: refresh it before the GPU is used again.
            brain.NotifyCpuWeightsChanged();
        }
    }

    LessonOutcome o;
    o.step = nextStep;
    o.lr = lr;
    o.batch = batch;
    o.resolution = res;
    o.practice = practice;
    o.pictureCount = pictureCount;
    o.usedGpu = usedGpu;
    o.fallbackTriggered = fallbackTriggered;
    o.loss = mseLoss;
    o.randomPairingCost = otBatch.randomPairingCost;
    o.optimalPairingCost = otBatch.optimalPairingCost;
    return FinishLesson(brain, dataset, config, o);
}

// ---------------------------------------------------------------------------------------------
// Loss chart
// ---------------------------------------------------------------------------------------------
void Trainer::RecordLoss(uint64_t step, float loss) {
    std::lock_guard<std::mutex> lock(m_stateMutex);
    m_curveAcc += static_cast<double>(loss);
    ++m_curveCount;
    if (m_curveCount >= m_curveStride) {
        m_curve.emplace_back(step, static_cast<float>(m_curveAcc / static_cast<double>(m_curveCount)));
        m_curveAcc = 0.0;
        m_curveCount = 0;
        if (m_curve.size() >= kCurveCapacity) {
            // Merge neighbours: half as many points, each covering twice as many lessons.
            std::vector<std::pair<uint64_t, float>> merged;
            merged.reserve(m_curve.size() / 2 + 1);
            for (size_t i = 0; i + 1 < m_curve.size(); i += 2) {
                merged.emplace_back(m_curve[i + 1].first, 0.5f * (m_curve[i].second + m_curve[i + 1].second));
            }
            if (m_curve.size() % 2 == 1) merged.push_back(m_curve.back());
            m_curve.swap(merged);
            m_curveStride *= 2;
        }
    }
    ++m_lossVersion;
}

std::vector<std::pair<uint64_t, float>> Trainer::LossCurve(size_t maxPoints) const {
    std::lock_guard<std::mutex> lock(m_stateMutex);
    maxPoints = std::max<size_t>(2, maxPoints);
    if (m_curve.size() <= maxPoints) return m_curve;
    std::vector<std::pair<uint64_t, float>> out;
    const size_t bucket = (m_curve.size() + maxPoints - 1) / maxPoints;
    for (size_t i = 0; i < m_curve.size(); i += bucket) {
        const size_t end = std::min(m_curve.size(), i + bucket);
        double sum = 0.0;
        for (size_t j = i; j < end; ++j) sum += static_cast<double>(m_curve[j].second);
        out.emplace_back(m_curve[end - 1].first, static_cast<float>(sum / static_cast<double>(end - i)));
    }
    return out;
}

uint64_t Trainer::LossVersion() const {
    std::lock_guard<std::mutex> lock(m_stateMutex);
    return m_lossVersion;
}

// ---------------------------------------------------------------------------------------------
// Averaged dials (EMA)
// ---------------------------------------------------------------------------------------------
void Trainer::UpdateAverage(FMBrain& brain, const TrainerConfig& config, uint64_t step) {
    if (!config.emaEnabled || step % kAverageEvery != 0) return;
    std::lock_guard<std::recursive_mutex> brainLock(brain.Mutex());
    if (FAILED(brain.ReadbackWeightsFromGPU())) return;
    BrainIO::CollectParams(brain, m_scratch);
    if (!BrainIO::AllFinite(m_scratch)) return;

    const bool stale = m_ema.size() != m_scratch.size() || step < m_emaStep || step - m_emaStep > 4ull * kAverageEvery;
    if (stale) {
        m_ema = m_scratch; // (re)start the average from the dials as they are now
        m_emaStep = step;
        return;
    }
    // Per-lesson decay with the usual warm-up (the average follows quickly at first), applied for the lessons since the last update.
    const double warm = (1.0 + static_cast<double>(step)) / (10.0 + static_cast<double>(step));
    const double perStep = std::min(static_cast<double>(config.emaDecay), warm);
    const double keep = std::pow(perStep, static_cast<double>(step - m_emaStep));
    const float k = static_cast<float>(keep);
    const float n = static_cast<float>(1.0 - keep);
    for (size_t i = 0; i < m_ema.size(); ++i) m_ema[i] = k * m_ema[i] + n * m_scratch[i];
    m_emaStep = step;
}

// ---------------------------------------------------------------------------------------------
// Pictures made by the brain
// ---------------------------------------------------------------------------------------------
bool Trainer::MakeSamples(FMBrain& brain, const TrainerConfig& config, uint32_t count, uint32_t resolution, uint32_t dreamSteps,
                          bool fixedSeeds, bool useAverage, std::vector<std::vector<float>>& out, bool* usedAverage, float variety) {
    out.clear();
    if (usedAverage) *usedAverage = false;
    const bool onePass = brain.IsOnePass();
    const uint32_t wave = onePass ? 0u : brain.WaveLevel();
    const uint32_t res = PictureSizeFor(resolution, wave, brain.Family()); // size of the pictures made
    const uint32_t netRes = res >> wave;                   // width and height the network works at
    const uint32_t steps = std::clamp(dreamSteps, 1u, 50u);
    count = std::clamp(count, 1u, 64u);
    // numbers per picture: 3 x res x res as pixels, or 12 x half x half in wave form (the same amount)
    const size_t sampleSize = static_cast<size_t>(3) * res * res;

    std::lock_guard<std::recursive_mutex> brainLock(brain.Mutex());

    // Show the averaged dials if there are any (they are swapped in for the duration of the dreams and swapped back after).
    DialRestore restore(brain);
    bool averaged = false;
    if (useAverage && config.emaEnabled && !m_ema.empty() && m_ema.size() == brain.ParameterCount() &&
        SUCCEEDED(brain.ReadbackWeightsFromGPU())) { // (if the read-back failed the CPU copy is stale: do not swap)
        BrainIO::CollectParams(brain, restore.raw);
        if (restore.raw.size() == m_ema.size() && BrainIO::ApplyParams(brain, m_ema)) {
            restore.active = true;
            averaged = true;
        }
    }

    const uint64_t freshSeed = m_dreamRng();
    bool gpuOk = config.engine == ComputeEngine::GPU && brain.IsGPUReady();

    if (onePass) {
        // One pass: random numbers -> the decoder half -> pictures. No steps: `dreamSteps` does not apply.
        const uint64_t codeSize = brain.OnePassCodeSize(res);
        if (codeSize == 0) return false;
        const float scale = std::isfinite(variety) ? std::clamp(variety, 0.0f, 2.0f) : 1.0f;
        std::vector<float> codes(static_cast<size_t>(count) * static_cast<size_t>(codeSize));
        for (uint32_t i = 0; i < count; ++i) {
            // (the same noise recipe as the flow brain: picture i of a fixed-seed grid always gets the same numbers)
            std::mt19937_64 rng((fixedSeeds ? m_previewSeed : freshSeed) * 0x9E3779B97F4A7C15ULL + 0x1234567ULL * (i + 1));
            std::normal_distribution<float> norm(0.0f, 1.0f);
            float* dst = codes.data() + static_cast<size_t>(i) * static_cast<size_t>(codeSize);
            for (uint64_t k = 0; k < codeSize; ++k) dst[k] = norm(rng) * scale;
        }
        const uint32_t chunk = std::max(1u, std::min(count, brain.OnePassMaxBatch(res)));
        for (uint32_t first = 0; first < count; first += chunk) {
            const uint32_t n = std::min(chunk, count - first);
            const float* chunkCodes = codes.data() + static_cast<size_t>(first) * static_cast<size_t>(codeSize);
            std::vector<float> pics;
            bool done = false;
            if (gpuOk) {
                done = SUCCEEDED(brain.DrawOnePassGpu(chunkCodes, n, res, pics)) && pics.size() == static_cast<size_t>(n) * sampleSize;
                if (!done) gpuOk = false; // the GPU just failed: the rest is made on the CPU
            }
            if (!done) {
                brain.ReadbackWeightsFromGPU();
                done = brain.DrawOnePassCpu(chunkCodes, n, res, pics) && pics.size() == static_cast<size_t>(n) * sampleSize;
            }
            if (!done) { out.clear(); return false; }
            for (uint32_t j = 0; j < n; ++j) {
                std::vector<float> img(pics.begin() + static_cast<std::ptrdiff_t>(j * sampleSize),
                                       pics.begin() + static_cast<std::ptrdiff_t>((j + 1) * sampleSize));
                for (float& v : img) v = std::isfinite(v) ? std::clamp(v, -1.0f, 1.0f) : 0.0f;
                out.push_back(std::move(img));
            }
        }
        if (usedAverage) *usedAverage = averaged;
        return true;
    }

    for (uint32_t i = 0; i < count; ++i) {
        std::mt19937_64 rng((fixedSeeds ? m_previewSeed : freshSeed) * 0x9E3779B97F4A7C15ULL + 0x1234567ULL * (i + 1));
        std::normal_distribution<float> norm(0.0f, 1.0f);
        std::vector<float> z0(sampleSize);
        for (float& v : z0) v = norm(rng);

        std::vector<float> img;
        bool done = false;
        if (gpuOk) {
            done = SUCCEEDED(brain.DreamEulerODEGPU(z0.data(), netRes, netRes, steps, img)) && img.size() == sampleSize;
            if (!done) gpuOk = false; // the GPU just failed: the rest is made on the CPU
        }
        if (!done) {
            brain.ReadbackWeightsFromGPU();
            img = brain.DreamEulerODE(z0.data(), netRes, netRes, steps);
        }
        if (img.size() != sampleSize) { out.clear(); return false; }
        if (wave != 0) { // a wave brain draws in wave form: turn it back into a picture
            std::vector<float> picture(sampleSize);
            if (!HaarInverse(img.data(), 3, res, res, picture.data())) { out.clear(); return false; }
            img.swap(picture);
        }
        for (float& v : img) v = std::isfinite(v) ? std::clamp(v, -1.0f, 1.0f) : 0.0f;
        out.push_back(std::move(img));
    }
    if (usedAverage) *usedAverage = averaged;
    return true;
}

bool Trainer::UpdatePreview(FMBrain& brain, const TrainerConfig& config, const std::vector<uint8_t>* finishPalette) {
    const uint32_t res = PictureSizeFor(config.resolution, brain.IsOnePass() ? 0u : brain.WaveLevel(), brain.Family());
    const bool onGpu = config.engine == ComputeEngine::GPU && brain.IsGPUReady();
    uint32_t tiles = 16;
    if (onGpu) {
        tiles = res <= 32 ? 16 : (res <= 64 ? 9 : 4);
    } else {
        tiles = res <= 16 ? 9 : (res <= 32 ? 4 : 1); // the CPU is slow: fewer samples
    }
    const uint32_t columns = tiles >= 16 ? 4 : (tiles >= 9 ? 3 : (tiles >= 4 ? 2 : 1));
    const uint32_t steps = brain.IsOnePass() ? 1u : std::clamp(config.dreamSteps, 1u, onGpu ? 50u : 12u); // one pass: always 1

    std::vector<std::vector<float>> images;
    bool averaged = false;
    if (!MakeSamples(brain, config, tiles, res, steps, true, true, images, &averaged)) return false;

    const uint32_t zoom = std::max(1u, 64u / res);
    std::vector<Rgb8Image> tileImages;
    tileImages.reserve(images.size());
    for (const auto& chw : images) {
        Rgb8Image t = FloatChwToRgb8(chw.data(), res, res);
        if (finishPalette && !finishPalette->empty()) SnapToPalette(t, *finishPalette); // before enlarging, so it is cheap
        tileImages.push_back(zoom > 1 ? UpscaleNearest(t, zoom) : std::move(t));
    }
    Rgb8Image grid = MakeGrid(tileImages, columns, 2, 18, 20, 28);
    if (!grid.Valid()) return false;

    std::lock_guard<std::mutex> lock(m_stateMutex);
    m_preview.grid = std::move(grid);
    m_preview.version += 1;
    m_preview.step = m_metrics.stepCount;
    m_preview.tiles = tiles;
    m_preview.columns = columns;
    m_preview.resolution = res;
    m_preview.dreamSteps = brain.IsOnePass() ? 0u : steps; // 0 = drawn in one pass
    m_preview.usedAverage = averaged;
    return true;
}

PreviewImage Trainer::LatestPreview() const {
    std::lock_guard<std::mutex> lock(m_stateMutex);
    return m_preview;
}

uint64_t Trainer::PreviewVersion() const {
    std::lock_guard<std::mutex> lock(m_stateMutex);
    return m_preview.version;
}

bool Trainer::PreviewReady() const {
    std::lock_guard<std::mutex> lock(m_stateMutex);
    return m_preview.grid.Valid();
}

// ---------------------------------------------------------------------------------------------
// Bookkeeping
// ---------------------------------------------------------------------------------------------
void Trainer::ResetProgress() {
    m_ema.clear();
    m_emaStep = 0;
    m_epochBase = 0.0;
    m_haveLastTime = false;
    std::lock_guard<std::mutex> lock(m_stateMutex);
    const DialScanReport scan = m_metrics.lastDialScan;
    m_metrics = TrainingMetrics();
    m_metrics.lastDialScan = scan;
    m_curve.clear();
    m_curveStride = 1;
    m_curveAcc = 0.0;
    m_curveCount = 0;
    m_smoothCount = 0;
    ++m_lossVersion;
    const uint64_t v = m_preview.version + 1;
    m_preview = PreviewImage();
    m_preview.version = v;
}

void Trainer::NoteTrainingStarted() {
    m_haveLastTime = false;
}

void Trainer::RestoreProgress(uint64_t step, float smoothLoss, double epochs,
                              const std::vector<std::pair<uint64_t, float>>& curve, std::vector<float> averagedDials) {
    m_ema = std::move(averagedDials);
    m_emaStep = step;
    m_epochBase = epochs;
    m_haveLastTime = false;
    std::lock_guard<std::mutex> lock(m_stateMutex);
    const DialScanReport scan = m_metrics.lastDialScan;
    m_metrics = TrainingMetrics();
    m_metrics.lastDialScan = scan;
    m_metrics.stepCount = step;
    m_metrics.smoothLoss = smoothLoss;
    m_metrics.bestSmoothLoss = smoothLoss;
    m_metrics.epochs = epochs;
    m_curve = curve;
    m_curveStride = 1;
    if (m_curve.size() >= 2) {
        const uint64_t gap = m_curve.back().first - m_curve[m_curve.size() - 2].first;
        m_curveStride = std::max<uint64_t>(1, gap);
    }
    m_curveAcc = 0.0;
    m_curveCount = 0;
    m_smoothCount = smoothLoss > 0.0f ? 100 : 0;
    ++m_lossVersion;
}

void Trainer::SetDialScanReport(const DialScanReport& report) {
    std::lock_guard<std::mutex> lock(m_stateMutex);
    m_metrics.lastDialScan = report;
}

TrainingMetrics Trainer::SnapshotMetrics() const {
    std::lock_guard<std::mutex> lock(m_stateMutex);
    return m_metrics;
}

} // namespace XboxOTFM
