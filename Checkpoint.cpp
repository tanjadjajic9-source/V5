// The C file functions used below (fopen and friends) are fine to use here: every result is checked. Microsoft's compiler flags
// them as "unsafe" and, in an Xbox app, treats that as an error, so the warning is switched off before any header is read.
#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS 1
#endif

#include "Checkpoint.h"

#include "ImageUtil.h"
#include "Json.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <system_error>

// File layout (all numbers little-endian):
//   "OTCK" | u32 version (1) | u32 headerBytes | header JSON (UTF-8)
//   u64 paramCount | u32 flags (bit0 = EMA copy present, bit1 = optimizer memory present) | u32 reserved
//   float params[paramCount]  (flow brain: stem weight, stem bias, then per block conv1 weight/bias, conv2 weight/bias, then head
//                              weight/bias; one-pass brain: weight then bias of every layer in the order of its plan, see FMBrain.h)
//   [float ema[paramCount]] [float adamM[paramCount]] [float adamV[paramCount]]
//   u32 CRC-32 of every byte before it

namespace XboxOTFM {

namespace {

constexpr uint32_t kVersion = 1;
constexpr uint32_t kMaxHeaderBytes = 16 * 1024;
constexpr size_t kMaxNameBytes = 80;
constexpr size_t kMaxSettingsBytes = 8 * 1024;
constexpr size_t kMaxCurvePoints = 240;
constexpr char kExt[] = ".otck";

void Put32(std::vector<uint8_t>& v, uint32_t x) { for (int i = 0; i < 4; ++i) v.push_back(static_cast<uint8_t>(x >> (8 * i))); }
void Put64(std::vector<uint8_t>& v, uint64_t x) { for (int i = 0; i < 8; ++i) v.push_back(static_cast<uint8_t>(x >> (8 * i))); }
uint32_t Get32(const uint8_t* p) { return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) | (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24); }
uint64_t Get64(const uint8_t* p) { return static_cast<uint64_t>(Get32(p)) | (static_cast<uint64_t>(Get32(p + 4)) << 32); }

void PutFloats(std::vector<uint8_t>& v, const std::vector<float>& f) {
    const size_t at = v.size();
    v.resize(at + f.size() * sizeof(float));
    if (!f.empty()) std::memcpy(v.data() + at, f.data(), f.size() * sizeof(float));
}

std::string CleanText(const std::string& in, size_t maxBytes) {
    std::string out;
    for (unsigned char c : in) {
        if (c < 0x20 || c == 0x7F) continue;
        out += static_cast<char>(c);
        if (out.size() >= maxBytes) break;
    }
    // never leave half of a multi-byte character at the end
    while (!out.empty() && (static_cast<unsigned char>(out.back()) & 0xC0) == 0x80) out.pop_back();
    if (!out.empty() && static_cast<unsigned char>(out.back()) >= 0xC0) out.pop_back();
    // no spaces at either end
    size_t first = 0;
    while (first < out.size() && out[first] == ' ') ++first;
    out.erase(0, first);
    while (!out.empty() && out.back() == ' ') out.pop_back();
    return out;
}

bool PresetOk(uint32_t p) { return p <= static_cast<uint32_t>(BrainPreset::Max250k); }

uint64_t NowUnix() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count());
}

void Fail(std::string* error, const std::string& why) {
    if (error) *error = why;
}

// Reads the JSON header fields shared by List() and Parse().
bool ParseHeaderJson(const std::string& text, CheckpointMeta& meta, std::string* settings,
                     std::vector<std::pair<uint64_t, float>>* curve = nullptr) {
    JsonValue v;
    if (!JsonParser::Parse(text, v) || !v.IsObject()) return false;
    double d = 0.0;
    std::string s;
    if (!v.GetNumber("preset", d) || d < 0 || d > 2 || d != std::floor(d)) return false;
    meta.preset = static_cast<uint32_t>(d);
    meta.wave = 0; // snapshots from before wave brains existed have no "wave" field and are pixel brains
    if (v.GetNumber("wave", d)) {
        if (d < 0 || d > static_cast<double>(FMBrain::kMaxWaveLevel) || d != std::floor(d)) return false;
        meta.wave = static_cast<uint32_t>(d);
    }
    meta.family = 0; // snapshots from before one-pass brains existed have no "family" field and are flow brains
    if (v.GetNumber("family", d)) {
        if (d < 0 || d > 1 || d != std::floor(d)) return false;
        meta.family = static_cast<uint32_t>(d);
        if (meta.family == 1 && meta.wave != 0) return false; // a one-pass brain has no wave form
    }
    if (v.GetNumber("step", d) && d >= 0 && d < 1e18) meta.step = static_cast<uint64_t>(d);
    if (v.GetNumber("created", d) && d >= 0 && d < 1e12) meta.createdUnix = static_cast<uint64_t>(d);
    if (v.GetNumber("res", d) && d >= 8 && d <= 128) meta.resolution = static_cast<uint32_t>(d);
    if (v.GetNumber("loss", d) && d >= 0 && d < 1e9) meta.loss = d;
    if (v.GetNumber("epochs", d) && d >= 0 && d < 1e12) meta.epochs = d;
    if (v.GetNumber("pictures", d) && d >= 0 && d < 1e9) meta.pictures = static_cast<uint32_t>(d);
    if (v.GetString("name", s)) meta.name = CleanText(s, kMaxNameBytes);
    if (v.GetString("kind", s) && (s == "auto" || s == "safety" || s == "manual" || s == "imported")) meta.kind = s; else meta.kind = "manual";
    if (settings && v.GetString("settings", s) && s.size() <= kMaxSettingsBytes) *settings = s;
    if (curve) {
        const JsonValue* c = v.Find("curve");
        if (c && c->type == JsonValue::Type::Array) {
            uint64_t lastStep = 0;
            for (const JsonValue& pt : c->items) {
                if (curve->size() >= kMaxCurvePoints) break;
                if (pt.type != JsonValue::Type::Array || pt.items.size() != 2) continue;
                const JsonValue& a = pt.items[0];
                const JsonValue& b = pt.items[1];
                if (a.type != JsonValue::Type::Number || b.type != JsonValue::Type::Number) continue;
                if (!(a.number >= 0 && a.number < 1e18) || !std::isfinite(b.number) || b.number < 0 || b.number > 1e9) continue;
                const uint64_t st = static_cast<uint64_t>(a.number);
                if (!curve->empty() && st <= lastStep) continue; // steps must increase
                lastStep = st;
                curve->emplace_back(st, static_cast<float>(b.number));
            }
        }
    }
    return true;
}

} // namespace

// ---------------------------------------------------------------------------------------------
// BrainIO
// ---------------------------------------------------------------------------------------------
namespace BrainIO {

void CollectParams(FMBrain& brain, std::vector<float>& out) {
    out.clear();
    if (brain.IsOnePass()) {
        brain.OnePassCollect(&out, nullptr, nullptr);
        return;
    }
    out.reserve(static_cast<size_t>(brain.ParameterCount()));
    auto add = [&](const std::vector<float>& v) { out.insert(out.end(), v.begin(), v.end()); };
    add(brain.Stem().weight);
    add(brain.Stem().bias);
    for (auto& b : brain.Blocks()) {
        add(b.conv1.weight);
        add(b.conv1.bias);
        add(b.conv2.weight);
        add(b.conv2.bias);
    }
    add(brain.Head().weight);
    add(brain.Head().bias);
}

bool ApplyParams(FMBrain& brain, const std::vector<float>& in) {
    if (in.size() != brain.ParameterCount()) return false;
    if (brain.IsOnePass()) return brain.OnePassApply(&in, nullptr, nullptr);
    size_t pos = 0;
    auto take = [&](std::vector<float>& dst) {
        if (!dst.empty()) std::memcpy(dst.data(), in.data() + pos, dst.size() * sizeof(float));
        pos += dst.size();
    };
    take(brain.Stem().weight);
    take(brain.Stem().bias);
    for (auto& b : brain.Blocks()) {
        take(b.conv1.weight);
        take(b.conv1.bias);
        take(b.conv2.weight);
        take(b.conv2.bias);
    }
    take(brain.Head().weight);
    take(brain.Head().bias);
    brain.NotifyCpuWeightsChanged();
    return pos == in.size();
}

void CollectAdam(FMBrain& brain, std::vector<float>& m, std::vector<float>& v) {
    m.clear();
    v.clear();
    if (brain.IsOnePass()) {
        brain.OnePassCollect(nullptr, &m, &v);
        return;
    }
    m.reserve(static_cast<size_t>(brain.ParameterCount()));
    v.reserve(static_cast<size_t>(brain.ParameterCount()));
    auto addM = [&](const std::vector<float>& x) { m.insert(m.end(), x.begin(), x.end()); };
    auto addV = [&](const std::vector<float>& x) { v.insert(v.end(), x.begin(), x.end()); };
    addM(brain.StemMW()); addM(brain.StemMB());
    addV(brain.StemVW()); addV(brain.StemVB());
    for (auto& b : brain.Blocks()) {
        addM(b.mW1); addM(b.mB1); addM(b.mW2); addM(b.mB2);
        addV(b.vW1); addV(b.vB1); addV(b.vW2); addV(b.vB2);
    }
    addM(brain.HeadMW()); addM(brain.HeadMB());
    addV(brain.HeadVW()); addV(brain.HeadVB());
}

bool ApplyAdam(FMBrain& brain, const std::vector<float>& m, const std::vector<float>& v) {
    if (m.size() != brain.ParameterCount() || v.size() != brain.ParameterCount()) return false;
    if (brain.IsOnePass()) return brain.OnePassApply(nullptr, &m, &v);
    size_t posM = 0, posV = 0;
    auto takeM = [&](std::vector<float>& dst) {
        if (!dst.empty()) std::memcpy(dst.data(), m.data() + posM, dst.size() * sizeof(float));
        posM += dst.size();
    };
    auto takeV = [&](std::vector<float>& dst) {
        if (!dst.empty()) std::memcpy(dst.data(), v.data() + posV, dst.size() * sizeof(float));
        posV += dst.size();
    };
    takeM(brain.StemMW()); takeM(brain.StemMB());
    takeV(brain.StemVW()); takeV(brain.StemVB());
    for (auto& b : brain.Blocks()) {
        takeM(b.mW1); takeM(b.mB1); takeM(b.mW2); takeM(b.mB2);
        takeV(b.vW1); takeV(b.vB1); takeV(b.vW2); takeV(b.vB2);
    }
    takeM(brain.HeadMW()); takeM(brain.HeadMB());
    takeV(brain.HeadVW()); takeV(brain.HeadVB());
    brain.NotifyCpuWeightsChanged();
    return posM == m.size() && posV == v.size();
}

bool AllFinite(const std::vector<float>& values) {
    for (float f : values) {
        if (!std::isfinite(f)) return false;
    }
    return true;
}

} // namespace BrainIO

bool IsSafeCheckpointId(const std::string& id) {
    if (id.empty() || id.size() > 64) return false;
    for (unsigned char c : id) {
        const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '-';
        if (!ok) return false;
    }
    return true;
}

// ---------------------------------------------------------------------------------------------
// Whole-file encode / decode
// ---------------------------------------------------------------------------------------------
std::vector<uint8_t> CheckpointStore::Serialize(const CheckpointData& d) {
    JsonWriter w;
    w.BeginObject();
    w.Field("name", CleanText(d.meta.name, kMaxNameBytes));
    w.Field("kind", d.meta.kind.empty() ? std::string("manual") : d.meta.kind);
    w.Field("created", d.meta.createdUnix);
    w.Field("step", d.meta.step);
    w.Field("preset", d.meta.preset);
    w.Field("wave", d.meta.wave);
    if (d.meta.family != 0) w.Field("family", d.meta.family); // absent = flow brain (keeps the files of flow brains as they always were)
    w.Field("res", d.meta.resolution);
    w.Field("loss", d.meta.loss);
    w.Field("epochs", d.meta.epochs);
    w.Field("pictures", d.meta.pictures);
    w.Field("settings", d.settingsJson.size() <= kMaxSettingsBytes ? d.settingsJson : std::string());
    w.Key("curve").BeginArray();
    {
        // at most kMaxCurvePoints, evenly spread, always including the last point
        const size_t n = d.curve.size();
        const size_t take = std::min(n, kMaxCurvePoints);
        for (size_t i = 0; i < take; ++i) {
            const size_t idx = (take <= 1 || n <= kMaxCurvePoints) ? i : (i * (n - 1)) / (take - 1);
            w.BeginArray().Value(d.curve[idx].first).Value(static_cast<double>(d.curve[idx].second)).EndArray();
        }
    }
    w.EndArray();
    w.EndObject();
    const std::string header = w.Take();

    const bool hasEma = !d.ema.empty();
    const bool hasAdam = !d.adamM.empty() && d.adamM.size() == d.adamV.size();

    std::vector<uint8_t> out;
    out.reserve(32 + header.size() + sizeof(float) * d.params.size() * (1 + (hasEma ? 1 : 0) + (hasAdam ? 2 : 0)));
    out.insert(out.end(), { 'O', 'T', 'C', 'K' });
    Put32(out, kVersion);
    Put32(out, static_cast<uint32_t>(header.size()));
    out.insert(out.end(), header.begin(), header.end());
    Put64(out, d.params.size());
    Put32(out, (hasEma ? 1u : 0u) | (hasAdam ? 2u : 0u));
    Put32(out, 0);
    PutFloats(out, d.params);
    if (hasEma) PutFloats(out, d.ema);
    if (hasAdam) {
        PutFloats(out, d.adamM);
        PutFloats(out, d.adamV);
    }
    Put32(out, Crc32(out.data(), out.size()));
    return out;
}

bool CheckpointStore::Parse(const uint8_t* bytes, size_t size, CheckpointData& out, std::string* error) {
    out = CheckpointData();
    if (!bytes || size < 12 + 16 + 4) { Fail(error, "This file is too small to be a saved snapshot."); return false; }
    if (std::memcmp(bytes, "OTCK", 4) != 0) { Fail(error, "This is not a saved snapshot from this app."); return false; }
    if (Get32(bytes + 4) != kVersion) { Fail(error, "This snapshot was made by a different version of the app."); return false; }
    const uint32_t headerBytes = Get32(bytes + 8);
    if (headerBytes > kMaxHeaderBytes || 12 + static_cast<uint64_t>(headerBytes) + 16 + 4 > size) {
        Fail(error, "The snapshot file is damaged (bad header).");
        return false;
    }
    if (Crc32(bytes, size - 4) != Get32(bytes + size - 4)) {
        Fail(error, "The snapshot file is damaged (checksum does not match).");
        return false;
    }
    const std::string headerText(reinterpret_cast<const char*>(bytes + 12), headerBytes);
    CheckpointData d;
    if (!ParseHeaderJson(headerText, d.meta, &d.settingsJson, &d.curve)) { Fail(error, "The snapshot file is damaged (unreadable header)."); return false; }

    const uint8_t* p = bytes + 12 + headerBytes;
    const uint64_t paramCount = Get64(p);
    const uint32_t flags = Get32(p + 8);
    p += 16;
    if (!PresetOk(d.meta.preset) ||
        paramCount != FMBrain::ExpectedParameterCount(static_cast<BrainPreset>(d.meta.preset), d.meta.wave,
                                                      d.meta.family == 1 ? BrainFamily::OnePass : BrainFamily::Flow)) {
        Fail(error, "The snapshot does not match any brain size of this app.");
        return false;
    }
    d.meta.hasEma = (flags & 1u) != 0;
    d.meta.hasAdam = (flags & 2u) != 0;
    const uint64_t arrays = 1 + (d.meta.hasEma ? 1u : 0u) + (d.meta.hasAdam ? 2u : 0u);
    const uint64_t expected = 12 + static_cast<uint64_t>(headerBytes) + 16 + arrays * paramCount * sizeof(float) + 4;
    if (expected != size) { Fail(error, "The snapshot file has the wrong size."); return false; }

    auto readArray = [&](std::vector<float>& dst) {
        dst.resize(static_cast<size_t>(paramCount));
        std::memcpy(dst.data(), p, static_cast<size_t>(paramCount) * sizeof(float));
        p += paramCount * sizeof(float);
    };
    readArray(d.params);
    if (d.meta.hasEma) readArray(d.ema);
    if (d.meta.hasAdam) {
        readArray(d.adamM);
        readArray(d.adamV);
    }
    if (!BrainIO::AllFinite(d.params) || !BrainIO::AllFinite(d.ema) || !BrainIO::AllFinite(d.adamM) || !BrainIO::AllFinite(d.adamV)) {
        Fail(error, "The snapshot contains broken numbers (the training had gone wrong when it was saved).");
        return false;
    }
    d.meta.fileBytes = size;
    out = std::move(d);
    return true;
}

// ---------------------------------------------------------------------------------------------
// Store
// ---------------------------------------------------------------------------------------------
CheckpointStore::CheckpointStore(std::string folder) : m_folder(std::move(folder)) {}

bool CheckpointStore::Init(std::string* error) {
    std::lock_guard<std::mutex> lock(m_mutex);
    std::error_code ec;
    std::filesystem::create_directories(m_folder, ec);
    if (ec) {
        Fail(error, "The snapshot folder could not be created: " + ec.message());
        return false;
    }
    // Remove temp files left behind by a power cut in the middle of a save.
    for (std::filesystem::directory_iterator it(m_folder, ec), end; !ec && it != end; it.increment(ec)) {
        const std::string name = it->path().filename().string();
        if (name.size() > 4 && name.compare(name.size() - 4, 4, ".tmp") == 0) {
            std::error_code ec2;
            std::filesystem::remove(it->path(), ec2);
        }
    }
    return true;
}

std::string CheckpointStore::PathFor(const std::string& id) const {
    return m_folder + "/" + id + kExt;
}

bool CheckpointStore::ReadHeader(const std::string& path, CheckpointMeta& meta) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    bool ok = false;
    do {
        uint8_t head[12];
        if (std::fread(head, 1, sizeof(head), f) != sizeof(head)) break;
        if (std::memcmp(head, "OTCK", 4) != 0 || Get32(head + 4) != kVersion) break;
        const uint32_t headerBytes = Get32(head + 8);
        if (headerBytes > kMaxHeaderBytes) break;
        std::string text(headerBytes, '\0');
        if (headerBytes > 0 && std::fread(&text[0], 1, headerBytes, f) != headerBytes) break;
        uint8_t tail[16];
        if (std::fread(tail, 1, sizeof(tail), f) != sizeof(tail)) break;
        CheckpointMeta m;
        if (!ParseHeaderJson(text, m, nullptr)) break;
        const uint32_t flags = Get32(tail + 8);
        m.hasEma = (flags & 1u) != 0;
        m.hasAdam = (flags & 2u) != 0;
        std::error_code ec;
        const uint64_t sz = std::filesystem::file_size(path, ec);
        m.fileBytes = ec ? 0 : sz;
        meta = m;
        ok = true;
    } while (false);
    std::fclose(f);
    return ok;
}

std::vector<CheckpointMeta> CheckpointStore::List() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    std::vector<CheckpointMeta> out;
    std::error_code ec;
    for (std::filesystem::directory_iterator it(m_folder, ec), end; !ec && it != end; it.increment(ec)) {
        const std::filesystem::path& p = it->path();
        const std::string fileName = p.filename().string();
        const size_t extLen = sizeof(kExt) - 1;
        if (fileName.size() <= extLen || fileName.compare(fileName.size() - extLen, extLen, kExt) != 0) continue;
        const std::string id = fileName.substr(0, fileName.size() - extLen);
        if (!IsSafeCheckpointId(id)) continue;
        CheckpointMeta meta;
        if (!ReadHeader(m_folder + "/" + fileName, meta)) continue;
        meta.id = id;
        out.push_back(std::move(meta));
    }
    std::sort(out.begin(), out.end(), [](const CheckpointMeta& a, const CheckpointMeta& b) {
        if (a.createdUnix != b.createdUnix) return a.createdUnix > b.createdUnix;
        if (a.step != b.step) return a.step > b.step;
        return a.id > b.id;
    });
    return out;
}

std::string CheckpointStore::NewId(const std::string& kind, uint64_t step, uint64_t unix) const {
    const char prefix = kind == "auto" ? 'a' : (kind == "safety" ? 's' : (kind == "imported" ? 'i' : 'm'));
    char base[64];
    std::snprintf(base, sizeof(base), "%c%llu_s%llu", prefix, static_cast<unsigned long long>(unix), static_cast<unsigned long long>(step));
    std::string id = base;
    std::error_code ec;
    for (int n = 2; n < 1000 && std::filesystem::exists(PathFor(id), ec); ++n) {
        id = std::string(base) + "_" + std::to_string(n);
    }
    return id;
}

bool CheckpointStore::Save(CheckpointData& data, std::string* error) {
    const uint32_t preset = data.meta.preset;
    if (!PresetOk(preset) || data.meta.wave > FMBrain::kMaxWaveLevel ||
        data.meta.family > 1 || (data.meta.family == 1 && data.meta.wave != 0) ||
        data.params.size() != FMBrain::ExpectedParameterCount(static_cast<BrainPreset>(preset), data.meta.wave,
                                                              data.meta.family == 1 ? BrainFamily::OnePass : BrainFamily::Flow)) {
        Fail(error, "The brain does not have the size this snapshot expects.");
        return false;
    }
    if ((!data.ema.empty() && data.ema.size() != data.params.size()) ||
        (data.adamM.size() != data.adamV.size()) || (!data.adamM.empty() && data.adamM.size() != data.params.size())) {
        Fail(error, "The snapshot parts do not match each other.");
        return false;
    }
    if (!BrainIO::AllFinite(data.params) || !BrainIO::AllFinite(data.ema) || !BrainIO::AllFinite(data.adamM) || !BrainIO::AllFinite(data.adamV)) {
        Fail(error, "The brain contains broken numbers, so it was not saved. Start a fresh brain or load an earlier snapshot.");
        return false;
    }

    std::lock_guard<std::mutex> lock(m_mutex);
    if (data.meta.kind != "auto" && data.meta.kind != "safety" && data.meta.kind != "imported") data.meta.kind = "manual";
    if (data.meta.createdUnix == 0) data.meta.createdUnix = NowUnix();
    data.meta.name = CleanText(data.meta.name, kMaxNameBytes);
    data.meta.hasEma = !data.ema.empty();
    data.meta.hasAdam = !data.adamM.empty();

    std::error_code ec;
    std::filesystem::create_directories(m_folder, ec);

    if (data.meta.kind == "manual") {
        size_t manual = 0;
        for (std::filesystem::directory_iterator it(m_folder, ec), end; !ec && it != end; it.increment(ec)) {
            const std::string n = it->path().filename().string();
            if (!n.empty() && n[0] == 'm' && n.size() > 5 && n.compare(n.size() - 5, 5, kExt) == 0) ++manual;
        }
        if (manual >= kMaxManual) {
            Fail(error, "You already have " + std::to_string(kMaxManual) + " saved snapshots. Delete some first.");
            return false;
        }
    }

    data.meta.id = NewId(data.meta.kind, data.meta.step, data.meta.createdUnix);
    const std::vector<uint8_t> bytes = Serialize(data);
    const std::string finalPath = PathFor(data.meta.id);
    const std::string tmpPath = finalPath + ".tmp";

    FILE* f = std::fopen(tmpPath.c_str(), "wb");
    if (!f) { Fail(error, "The snapshot file could not be created (is the storage full?)."); return false; }
    bool ok = std::fwrite(bytes.data(), 1, bytes.size(), f) == bytes.size();
    ok = (std::fflush(f) == 0) && ok;
    ok = (std::fclose(f) == 0) && ok;
    if (ok) {
        std::filesystem::rename(tmpPath, finalPath, ec);
        ok = !ec;
    }
    if (!ok) {
        std::error_code ec2;
        std::filesystem::remove(tmpPath, ec2);
        Fail(error, "The snapshot could not be written (is the storage full?).");
        return false;
    }
    data.meta.fileBytes = bytes.size();
    if (data.meta.kind == "auto") PruneKindLocked('a', "auto", kKeepAuto);
    if (data.meta.kind == "safety") PruneKindLocked('s', "safety", kKeepSafety);
    return true;
}

void CheckpointStore::PruneKindLocked(char prefix, const char* kind, size_t keep) {
    std::error_code ec;
    std::vector<std::pair<uint64_t, std::string>> autos; // (created, id)
    for (std::filesystem::directory_iterator it(m_folder, ec), end; !ec && it != end; it.increment(ec)) {
        const std::string n = it->path().filename().string();
        if (n.empty() || n[0] != prefix || n.size() <= 5 || n.compare(n.size() - 5, 5, kExt) != 0) continue;
        const std::string id = n.substr(0, n.size() - 5);
        CheckpointMeta meta;
        if (!IsSafeCheckpointId(id) || !ReadHeader(m_folder + "/" + n, meta) || meta.kind != kind) continue;
        autos.emplace_back(meta.createdUnix * 1000000ull + std::min<uint64_t>(meta.step, 999999ull), id);
    }
    if (autos.size() <= keep) return;
    std::sort(autos.begin(), autos.end());
    for (size_t i = 0; i + keep < autos.size(); ++i) {
        std::error_code ec2;
        std::filesystem::remove(PathFor(autos[i].second), ec2);
    }
}

bool CheckpointStore::ReadRaw(const std::string& id, std::vector<uint8_t>& out, std::string* error) const {
    out.clear();
    if (!IsSafeCheckpointId(id)) { Fail(error, "Unknown snapshot."); return false; }
    std::lock_guard<std::mutex> lock(m_mutex);
    FILE* f = std::fopen(PathFor(id).c_str(), "rb");
    if (!f) { Fail(error, "That snapshot no longer exists."); return false; }
    std::error_code ec;
    const uint64_t size = std::filesystem::file_size(PathFor(id), ec);
    bool ok = !ec && size < (1ull << 30);
    if (ok) {
        out.resize(static_cast<size_t>(size));
        ok = size == 0 || std::fread(out.data(), 1, out.size(), f) == out.size();
    }
    std::fclose(f);
    if (!ok) { out.clear(); Fail(error, "The snapshot file could not be read."); }
    return ok;
}

bool CheckpointStore::Load(const std::string& id, CheckpointData& out, std::string* error) const {
    std::vector<uint8_t> bytes;
    if (!ReadRaw(id, bytes, error)) return false;
    if (!Parse(bytes.data(), bytes.size(), out, error)) return false;
    out.meta.id = id;
    return true;
}

bool CheckpointStore::Delete(const std::string& id, std::string* error) {
    if (!IsSafeCheckpointId(id)) { Fail(error, "Unknown snapshot."); return false; }
    std::lock_guard<std::mutex> lock(m_mutex);
    std::error_code ec;
    const bool removed = std::filesystem::remove(PathFor(id), ec);
    if (ec || !removed) { Fail(error, "That snapshot no longer exists."); return false; }
    return true;
}

bool CheckpointStore::Rename(const std::string& id, const std::string& newName, std::string* error) {
    CheckpointData d;
    if (!Load(id, d, error)) return false;
    d.meta.name = CleanText(newName, kMaxNameBytes);
    const std::vector<uint8_t> bytes = Serialize(d);
    std::lock_guard<std::mutex> lock(m_mutex);
    const std::string finalPath = PathFor(id);
    const std::string tmpPath = finalPath + ".tmp";
    FILE* f = std::fopen(tmpPath.c_str(), "wb");
    if (!f) { Fail(error, "The snapshot file could not be written."); return false; }
    bool ok = std::fwrite(bytes.data(), 1, bytes.size(), f) == bytes.size();
    ok = (std::fflush(f) == 0) && ok;
    ok = (std::fclose(f) == 0) && ok;
    std::error_code ec;
    if (ok) {
        std::filesystem::rename(tmpPath, finalPath, ec);
        ok = !ec;
    }
    if (!ok) {
        std::error_code ec2;
        std::filesystem::remove(tmpPath, ec2);
        Fail(error, "The snapshot could not be renamed.");
        return false;
    }
    return true;
}

} // namespace XboxOTFM
