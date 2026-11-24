#pragma once
// Saved training snapshots ("checkpoints"): the brain's dials, the averaged (EMA) dials, the optimizer memory and the settings,
// in one file with a checksum. Written atomically (temp file + rename) so a power cut can never leave a half-written snapshot.

#include "FMBrain.h"

#include <cstdint>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace XboxOTFM {

struct CheckpointMeta {
    std::string id;           // file name without extension; only letters, digits, '_' and '-'
    std::string name;         // label shown to the person
    std::string kind;         // "auto", "safety" (saved automatically before a risky action), "manual" or "imported"
    uint64_t createdUnix = 0; // seconds since 1970
    uint64_t step = 0;
    uint32_t preset = 1;      // BrainPreset
    uint32_t wave = 0;        // 0 = the brain works on pixels, 1 = on the Haar wave form of the pictures (see FMBrain)
    uint32_t family = 0;      // 0 = flow brain (pixels or wave form), 1 = one-pass brain (BrainFamily); `wave` is 0 for family 1
    uint32_t resolution = 32;
    double loss = 0.0;        // smoothed training loss when it was saved (0 = unknown)
    double epochs = 0.0;
    uint32_t pictures = 0;    // pictures in the set when it was saved
    bool hasEma = false;
    bool hasAdam = false;
    uint64_t fileBytes = 0;
};

struct CheckpointData {
    CheckpointMeta meta;
    std::string settingsJson;        // the trainer settings (opaque to this module)
    std::vector<std::pair<uint64_t, float>> curve; // the loss chart so far (step, loss), at most ~240 points
    std::vector<float> params;       // every weight and bias, in the order described in Checkpoint.cpp
    std::vector<float> ema;          // averaged copy of params (empty = none)
    std::vector<float> adamM;        // optimizer memory (empty = none; adamM and adamV come together)
    std::vector<float> adamV;
};

// Moves the brain's dials in and out of flat float arrays. The caller must hold brain.Mutex() and must have called
// brain.ReadbackWeightsFromGPU() before collecting. Apply* check sizes first and call NotifyCpuWeightsChanged().
namespace BrainIO {
void CollectParams(FMBrain& brain, std::vector<float>& out);
bool ApplyParams(FMBrain& brain, const std::vector<float>& in);
void CollectAdam(FMBrain& brain, std::vector<float>& m, std::vector<float>& v);
bool ApplyAdam(FMBrain& brain, const std::vector<float>& m, const std::vector<float>& v);
bool AllFinite(const std::vector<float>& values);
} // namespace BrainIO

bool IsSafeCheckpointId(const std::string& id);

class CheckpointStore {
public:
    static constexpr size_t kMaxManual = 40; // saving more than this is refused (nothing is ever deleted behind the person's back)
    static constexpr size_t kKeepAuto = 4;   // older automatic snapshots are removed
    static constexpr size_t kKeepSafety = 8; // older safety copies are removed (they are only a few lines of work each)

    explicit CheckpointStore(std::string folder);

    bool Init(std::string* error = nullptr);

    // Newest first. Reads only the small header of each file (fast); problems with a file are skipped.
    std::vector<CheckpointMeta> List() const;

    // Assigns meta.id (and createdUnix when 0), writes the file atomically. `kind` "auto" / "safety" prunes the oldest ones of that kind.
    bool Save(CheckpointData& data, std::string* error = nullptr);
    bool Load(const std::string& id, CheckpointData& out, std::string* error = nullptr) const;
    bool Delete(const std::string& id, std::string* error = nullptr);
    bool Rename(const std::string& id, const std::string& newName, std::string* error = nullptr);
    bool ReadRaw(const std::string& id, std::vector<uint8_t>& out, std::string* error = nullptr) const;

    // Whole-file encode / decode (also used for downloads and uploads). Parse verifies the checksum, the sizes against the
    // preset, and that every number is finite.
    static std::vector<uint8_t> Serialize(const CheckpointData& data);
    static bool Parse(const uint8_t* bytes, size_t size, CheckpointData& out, std::string* error = nullptr);

    const std::string& Folder() const { return m_folder; }

private:
    std::string PathFor(const std::string& id) const;
    std::string NewId(const std::string& kind, uint64_t step, uint64_t unix) const;
    void PruneKindLocked(char prefix, const char* kind, size_t keep);
    static bool ReadHeader(const std::string& path, CheckpointMeta& meta);

    std::string m_folder;
    mutable std::mutex m_mutex;
};

} // namespace XboxOTFM
