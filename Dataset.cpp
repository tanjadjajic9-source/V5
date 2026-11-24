// The C file functions used below (fopen and friends) are fine to use here: every result is checked. Microsoft's compiler flags
// them as "unsafe" and, in an Xbox app, treats that as an error, so the warning is switched off before any header is read.
#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS 1
#endif

#include "Dataset.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <numeric>
#include <system_error>
#include <utility>

namespace XboxOTFM {

namespace {

// ---- the file: "OTDS", a version, then records. A record is [type 1 byte][payload size 4][CRC 4][payload]. ----
// type 1 = a picture was added: id 8, fingerprint 8, width 4, height 4, name length 2, name, then the pixels (the last 49152 bytes)
// type 2 = a picture was removed: id 8
constexpr uint8_t kRecAdd = 1;
constexpr uint8_t kRecRemove = 2;
constexpr uint32_t kMaxRecordPayload = 1u << 20;
constexpr size_t kMaxNameBytes = 120;
constexpr uint32_t kFileVersion = 1;

constexpr uint64_t kFileHeaderBytes = 8;
constexpr uint32_t kRecordHeaderBytes = 9;
constexpr uint32_t kAddFixedBytes = 26;
constexpr uint32_t kCardPixelBytes = Dataset::kCardSize * Dataset::kCardSize * 3;
constexpr uint32_t kRemovePayloadBytes = 8;
constexpr uint64_t kTypicalRecordBytes = kRecordHeaderBytes + kAddFixedBytes + 32 + kCardPixelBytes; // for estimates only

constexpr size_t kScanChunkBytes = 4u << 20;        // the file is read in pieces this big when it is checked
constexpr uint64_t kMinDeadBytesToTidy = 1u << 20;  // never rewrite the file for less than this much wasted space
constexpr uint64_t kWarnSkippedBytes = 1u << 20;    // damage smaller than this (a cut-off last picture) is not worth a warning

void Put16(std::vector<uint8_t>& v, uint16_t x) { v.push_back(static_cast<uint8_t>(x)); v.push_back(static_cast<uint8_t>(x >> 8)); }
void Put32(std::vector<uint8_t>& v, uint32_t x) { for (int i = 0; i < 4; ++i) v.push_back(static_cast<uint8_t>(x >> (8 * i))); }
void Put64(std::vector<uint8_t>& v, uint64_t x) { for (int i = 0; i < 8; ++i) v.push_back(static_cast<uint8_t>(x >> (8 * i))); }
uint16_t Get16(const uint8_t* p) { return static_cast<uint16_t>(p[0] | (p[1] << 8)); }
uint32_t Get32(const uint8_t* p) { return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) | (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24); }
uint64_t Get64(const uint8_t* p) { return static_cast<uint64_t>(Get32(p)) | (static_cast<uint64_t>(Get32(p + 4)) << 32); }

uint32_t RecordCrc(uint8_t type, const uint8_t* payload, size_t len) {
    return Crc32(payload, len, Crc32(&type, 1));
}

std::string CleanName(const std::string& in) {
    std::string out;
    for (unsigned char c : in) {
        if (c < 0x20 || c == 0x7F) continue;
        out += static_cast<char>(c);
        if (out.size() >= kMaxNameBytes) break;
    }
    // do not cut a UTF-8 sequence in half
    while (!out.empty() && (static_cast<unsigned char>(out.back()) & 0xC0) == 0x80) out.pop_back();
    if (!out.empty() && static_cast<unsigned char>(out.back()) >= 0xC0) out.pop_back();
    return out;
}

bool WriteRecord(FILE* f, uint8_t type, const std::vector<uint8_t>& payload) {
    uint8_t head[kRecordHeaderBytes];
    head[0] = type;
    const uint32_t len = static_cast<uint32_t>(payload.size());
    const uint32_t crc = RecordCrc(type, payload.data(), payload.size());
    for (int i = 0; i < 4; ++i) head[1 + i] = static_cast<uint8_t>(len >> (8 * i));
    for (int i = 0; i < 4; ++i) head[5 + i] = static_cast<uint8_t>(crc >> (8 * i));
    if (std::fwrite(head, 1, sizeof(head), f) != sizeof(head)) return false;
    if (!payload.empty() && std::fwrite(payload.data(), 1, payload.size(), f) != payload.size()) return false;
    return true;
}

bool WriteFileHeader(FILE* f) {
    uint8_t head[kFileHeaderBytes] = { 'O', 'T', 'D', 'S', 0, 0, 0, 0 };
    for (int i = 0; i < 4; ++i) head[4 + i] = static_cast<uint8_t>(kFileVersion >> (8 * i));
    return std::fwrite(head, 1, sizeof(head), f) == sizeof(head);
}

// fseek takes a 32-bit offset on Windows; the picture file can be several gigabytes.
bool SeekTo(FILE* f, uint64_t pos) {
#ifdef _WIN32
    return _fseeki64(f, static_cast<__int64>(pos), SEEK_SET) == 0;
#else
    return fseeko(f, static_cast<off_t>(pos), SEEK_SET) == 0;
#endif
}

DatasetOptions OptionsForCount(size_t maxCards) {
    DatasetOptions o;
    o.maxCards = maxCards;
    return o;
}

DatasetOptions Sanitized(DatasetOptions o) {
    o.maxCards = std::max<size_t>(1, o.maxCards);
    o.cacheCards = std::max<size_t>(1, o.cacheCards);
    return o;
}

#ifndef _WIN32
bool StandardFreeBytes(const std::string& folder, uint64_t& freeBytes) {
    std::error_code ec;
    const std::filesystem::space_info info = std::filesystem::space(folder, ec);
    if (ec) return false;
    freeBytes = static_cast<uint64_t>(info.available);
    return true;
}
#endif

// Reads a big file in large pieces and hands out small parts of it. Checking gigabytes of pictures one tiny read at a time
// would be slow; reading them in 4 MB pieces runs at the speed of the storage.
class ChunkReader {
public:
    ChunkReader(FILE* f, uint64_t size) : m_f(f), m_size(size), m_buf(kScanChunkBytes) {}

    // The n bytes at `off`, valid until the next call. Null if the file has no such bytes or reading failed.
    const uint8_t* At(uint64_t off, size_t n) {
        if (n > m_buf.size() || off > m_size || n > m_size - off) return nullptr;
        if (off < m_base || off - m_base + n > m_have) {
            m_have = 0;
            if (!SeekTo(m_f, off)) { m_ioError = true; return nullptr; }
            const size_t want = static_cast<size_t>(std::min<uint64_t>(m_buf.size(), m_size - off));
            m_have = std::fread(m_buf.data(), 1, want, m_f);
            m_base = off;
            if (m_have < want && std::ferror(m_f) != 0) { m_ioError = true; m_have = 0; return nullptr; }
            if (m_have < n) return nullptr;
        }
        return m_buf.data() + (off - m_base);
    }
    bool ioError() const { return m_ioError; }

private:
    FILE* m_f;
    uint64_t m_size;
    std::vector<uint8_t> m_buf;
    uint64_t m_base = 0;
    size_t m_have = 0;
    bool m_ioError = false;
};

struct ParsedRecord {
    uint8_t type = 0;
    uint32_t len = 0;
    uint64_t id = 0;
    uint64_t hash = 0;
    std::string name;
};

// True if a whole, intact record starts at `off`.
bool ParseRecord(ChunkReader& rd, uint64_t off, uint64_t fileSize, ParsedRecord& out) {
    if (off > fileSize || fileSize - off < kRecordHeaderBytes) return false;
    const uint8_t* head = rd.At(off, kRecordHeaderBytes);
    if (!head) return false;
    const uint8_t type = head[0];
    const uint32_t len = Get32(head + 1);
    const uint32_t crc = Get32(head + 5);
    if (type == kRecAdd) {
        if (len < kAddFixedBytes + kCardPixelBytes || len > kAddFixedBytes + 65535u + kCardPixelBytes) return false;
    } else if (type == kRecRemove) {
        if (len != kRemovePayloadBytes) return false;
    } else {
        return false;
    }
    if (len > kMaxRecordPayload || len > fileSize - off - kRecordHeaderBytes) return false;
    const uint8_t* rec = rd.At(off, kRecordHeaderBytes + len);
    if (!rec) return false;
    const uint8_t* payload = rec + kRecordHeaderBytes;
    if (RecordCrc(type, payload, len) != crc) return false;
    out.type = type;
    out.len = len;
    out.id = Get64(payload);
    if (type == kRecAdd) {
        out.hash = Get64(payload + 8);
        const uint32_t w = Get32(payload + 16);
        const uint32_t h = Get32(payload + 20);
        const uint32_t nameLen = Get16(payload + 24);
        if (w != Dataset::kCardSize || h != Dataset::kCardSize || kAddFixedBytes + nameLen + kCardPixelBytes != len) return false;
        out.name = CleanName(std::string(reinterpret_cast<const char*>(payload + kAddFixedBytes), nameLen));
    }
    return true;
}

// After damage: the first place from `from` on where a whole record starts (the end of the file if there is none).
uint64_t FindNextRecord(ChunkReader& rd, uint64_t from, uint64_t fileSize) {
    for (uint64_t p = from; p + kRecordHeaderBytes + kRemovePayloadBytes <= fileSize; ++p) {
        const uint8_t* head = rd.At(p, kRecordHeaderBytes);
        if (!head) return fileSize;
        if (head[0] != kRecAdd && head[0] != kRecRemove) continue;
        ParsedRecord ignored;
        if (ParseRecord(rd, p, fileSize, ignored)) return p;
        if (rd.ioError()) return fileSize;
    }
    return fileSize;
}

} // namespace

struct Dataset::ScanInfo {
    uint64_t skippedBytes = 0;  // damaged stretches that were stepped over
    uint64_t overLimit = 0;     // pictures that were not loaded because the set was already full
    bool ioError = false;       // the storage failed while reading: what was found so far is all there is
};

// ---------------------------------------------------------------------------------------------
// Construction, files
// ---------------------------------------------------------------------------------------------
Dataset::Dataset(std::string folder, size_t maxCards)
    : Dataset(std::move(folder), OptionsForCount(maxCards)) {
}

Dataset::Dataset(std::string folder, const DatasetOptions& options)
    : m_folder(std::move(folder))
    , m_opt(Sanitized(options)) {
    m_memoryCap = m_opt.maxCards;
}

Dataset::~Dataset() {
    std::lock_guard<std::mutex> lock(m_mutex);
    CloseHandlesLocked();
}

void Dataset::SetFreeSpaceReader(FreeSpaceReader reader) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_freeReader = std::move(reader);
    m_freeCheckedAt = std::chrono::steady_clock::time_point{};
}

std::string Dataset::FilePath() const {
    return m_folder + "/cards.otds";
}

std::string Dataset::TempFilePath() const {
    return m_folder + "/cards.otds.new";
}

void Dataset::CloseHandlesLocked() {
    if (m_append) {
        std::fclose(m_append);
        m_append = nullptr;
    }
    if (m_read) {
        std::fclose(m_read);
        m_read = nullptr;
    }
}

bool Dataset::OpenHandlesLocked() {
    CloseHandlesLocked();
    m_append = std::fopen(FilePath().c_str(), "ab");
    if (m_append) std::setvbuf(m_append, nullptr, _IOFBF, 1u << 16); // one record = one write
    m_read = std::fopen(FilePath().c_str(), "rb");
    if (m_read) std::setvbuf(m_read, nullptr, _IONBF, 0);            // pictures are read whole, so buffering would only copy
    std::error_code ec;
    const uint64_t size = static_cast<uint64_t>(std::filesystem::file_size(FilePath(), ec));
    if (!ec) m_fileBytes = size;
    return m_append != nullptr && m_read != nullptr;
}

bool Dataset::ResetFileLocked() {
    CloseHandlesLocked();
    FILE* f = std::fopen(FilePath().c_str(), "wb");
    bool ok = f != nullptr && WriteFileHeader(f) && std::fflush(f) == 0;
    if (f) std::fclose(f);
    m_fileBytes = kFileHeaderBytes;
    m_liveBytes = 0;
    if (ok) ok = OpenHandlesLocked();
    return ok;
}

// ---------------------------------------------------------------------------------------------
// Opening: check every record, build the index
// ---------------------------------------------------------------------------------------------
bool Dataset::Open(std::vector<std::string>* warnings) {
    std::lock_guard<std::mutex> lock(m_mutex);
    return OpenLocked(warnings);
}

bool Dataset::ScanLocked(FILE* f, uint64_t fileSize, ScanInfo& info) {
    ChunkReader rd(f, fileSize);
    const size_t guess = static_cast<size_t>(std::min<uint64_t>(fileSize / kTypicalRecordBytes + 16, m_opt.maxCards));
    m_cards.reserve(guess);
    m_hashToId.reserve(guess);
    size_t live = 0;
    uint64_t off = kFileHeaderBytes;
    while (off < fileSize) {
        ParsedRecord rec;
        if (ParseRecord(rd, off, fileSize, rec)) {
            if (rec.type == kRecAdd) {
                m_nextId = std::max(m_nextId, rec.id + 1);
                const bool inOrder = m_cards.empty() || rec.id > m_cards.back().id;
                if (!inOrder || m_hashToId.find(rec.hash) != m_hashToId.end()) {
                    // a second copy of a picture that is already there, or a record out of order: not used
                } else if (live >= m_opt.maxCards) {
                    ++info.overLimit;
                } else {
                    Entry e;
                    e.id = rec.id;
                    e.hash = rec.hash;
                    e.offset = off;
                    e.recordBytes = kRecordHeaderBytes + rec.len;
                    e.name = std::move(rec.name);
                    m_hashToId[e.hash] = e.id;
                    m_cards.push_back(std::move(e));
                    ++live;
                }
            } else {
                const size_t idx = FindIndexLocked(rec.id);
                if (idx != kNpos && !m_cards[idx].gone) {
                    Entry& e = m_cards[idx];
                    e.gone = true;
                    const auto hit = m_hashToId.find(e.hash);
                    if (hit != m_hashToId.end() && hit->second == e.id) m_hashToId.erase(hit);
                    --live;
                }
            }
            off += static_cast<uint64_t>(kRecordHeaderBytes) + rec.len;
            continue;
        }
        if (rd.ioError()) { info.ioError = true; break; }
        // Something here is damaged (usually a picture that was only half written when the app was closed). Step over it to
        // the next record that is whole.
        const uint64_t next = FindNextRecord(rd, off + 1, fileSize);
        if (rd.ioError()) { info.ioError = true; break; }
        info.skippedBytes += next - off;
        off = next;
    }

    // keep only the pictures that were not removed afterwards
    size_t keep = 0;
    uint64_t liveBytes = 0;
    for (size_t i = 0; i < m_cards.size(); ++i) {
        if (m_cards[i].gone) continue;
        liveBytes += m_cards[i].recordBytes;
        if (keep != i) m_cards[keep] = std::move(m_cards[i]);
        ++keep;
    }
    m_cards.erase(m_cards.begin() + static_cast<std::ptrdiff_t>(keep), m_cards.end());
    m_liveBytes = liveBytes;
    return true;
}

bool Dataset::OpenLocked(std::vector<std::string>* warnings) {
    auto warn = [&](const std::string& w) { if (warnings) warnings->push_back(w); };
    CloseHandlesLocked();
    std::vector<Entry>().swap(m_cards);
    m_hashToId.clear();
    CacheClearLocked();
    m_fileBytes = 0;
    m_liveBytes = 0;
    m_noCompact = false;
    m_unreadablePending = 0;
    m_orderDirty = true;
    m_cursor = 0;
    m_mode = Mode::Memory;
    m_memoryCap = m_opt.maxCards;
    if (m_folder.empty()) return true;

    // The pictures cannot be kept on storage: work from memory for this session (with a smaller limit).
    auto memoryOnly = [&](const std::string& why) {
        warn(why);
        CloseHandlesLocked();
        std::vector<Entry>().swap(m_cards);
        m_hashToId.clear();
        m_fileBytes = 0;
        m_liveBytes = 0;
        m_mode = Mode::Memory;
        m_memoryCap = std::min(m_opt.maxCards, m_opt.cacheCards);
        ++m_version;
        return true;
    };

    std::error_code ec;
    std::filesystem::create_directories(m_folder, ec);
    if (ec) {
        return memoryOnly("The picture folder could not be created (" + ec.message() + "): pictures will not be kept after the app closes.");
    }

    // A crash in the middle of tidying up can leave only the new file behind.
    if (!std::filesystem::exists(FilePath(), ec) && std::filesystem::exists(TempFilePath(), ec)) {
        std::filesystem::rename(TempFilePath(), FilePath(), ec);
    } else if (std::filesystem::exists(TempFilePath(), ec)) {
        std::filesystem::remove(TempFilePath(), ec); // left over from a tidy-up that did not finish; the real file is intact
    }

    bool needFreshFile = true;
    uint64_t skippedBytes = 0;
    bool ioError = false;
    if (std::filesystem::exists(FilePath(), ec)) {
        FILE* f = std::fopen(FilePath().c_str(), "rb");
        if (!f) {
            return memoryOnly("The saved pictures could not be opened, so this session starts with an empty set that will not be kept.");
        }
        uint8_t head[kFileHeaderBytes] = {};
        const bool headOk = std::fread(head, 1, sizeof(head), f) == sizeof(head) && std::memcmp(head, "OTDS", 4) == 0 && Get32(head + 4) == kFileVersion;
        if (!headOk) {
            std::fclose(f);
            warn("The saved picture file was not recognised; it was set aside and a new one started.");
            std::filesystem::rename(FilePath(), FilePath() + ".bad", ec);
        } else {
            needFreshFile = false;
            const uint64_t size = static_cast<uint64_t>(std::filesystem::file_size(FilePath(), ec));
            if (ec) {
                std::fclose(f);
                return memoryOnly("The size of the saved picture file could not be read, so this session starts with an empty set that will not be kept.");
            }
            ScanInfo info;
            ScanLocked(f, size, info);
            std::fclose(f);
            m_fileBytes = size;
            skippedBytes = info.skippedBytes;
            ioError = info.ioError;
            if (info.overLimit > 0) {
                warn("The saved set holds more pictures than this version can use, so " + std::to_string(info.overLimit) + " of them were left out.");
            }
        }
    }

    if (needFreshFile) {
        FILE* f = std::fopen(FilePath().c_str(), "wb");
        if (!f || !WriteFileHeader(f) || std::fflush(f) != 0) {
            if (f) std::fclose(f);
            return memoryOnly("The picture file could not be created: pictures will not be kept after the app closes.");
        }
        std::fclose(f);
        m_fileBytes = kFileHeaderBytes;
    }

    m_mode = Mode::File;
    if (!OpenHandlesLocked()) {
        warn("The picture file could not be opened for writing: new pictures cannot be saved right now.");
    }

    if (ioError) {
        m_noCompact = true; // never rewrite a file that was not read completely
        warn("The saved pictures could not all be read (a storage problem). Nothing was changed. Restart the app to try again.");
    } else if (skippedBytes >= kWarnSkippedBytes) {
        warn("Part of the saved picture file was damaged and was skipped. The other pictures were kept.");
        if (m_append) RewriteLocked(); // also removes the damaged part for good
    }
    MaybeCompactLocked(true);
    ++m_version;
    return true;
}

// ---------------------------------------------------------------------------------------------
// Writing the file
// ---------------------------------------------------------------------------------------------
void Dataset::AfterWriteFailureLocked() {
    // A failed write can leave part of a record behind. That is harmless (the next start steps over it) as long as the next
    // record starts after it, so start again from the real end of the file.
    if (m_append) {
        std::fclose(m_append);
        m_append = nullptr;
    }
    std::error_code ec;
    const uint64_t size = static_cast<uint64_t>(std::filesystem::file_size(FilePath(), ec));
    if (!ec) m_fileBytes = size;
    m_append = std::fopen(FilePath().c_str(), "ab");
    if (m_append) std::setvbuf(m_append, nullptr, _IOFBF, 1u << 16);
    m_freeCheckedAt = std::chrono::steady_clock::time_point{}; // look at the free space again next time
}

bool Dataset::AppendRecordLocked(uint8_t type, const std::vector<uint8_t>& payload, uint64_t* startOut) {
    if (!m_append) return false;
    const uint64_t start = m_fileBytes;
    errno = 0;
    if (!WriteRecord(m_append, type, payload) || std::fflush(m_append) != 0) {
        m_lastWriteErrno = errno;
        AfterWriteFailureLocked();
        return false;
    }
    m_fileBytes += kRecordHeaderBytes + payload.size();
    m_writtenSinceCheck += kRecordHeaderBytes + payload.size();
    if (startOut) *startOut = start;
    return true;
}

bool Dataset::AppendRemovesLocked(const std::vector<uint64_t>& ids) {
    if (!m_append) return false;
    if (ids.empty()) return true;
    std::vector<uint8_t> buf;
    buf.reserve(ids.size() * (kRecordHeaderBytes + kRemovePayloadBytes));
    for (uint64_t id : ids) {
        std::vector<uint8_t> p;
        Put64(p, id);
        buf.push_back(kRecRemove);
        Put32(buf, kRemovePayloadBytes);
        Put32(buf, RecordCrc(kRecRemove, p.data(), p.size()));
        buf.insert(buf.end(), p.begin(), p.end());
    }
    errno = 0;
    if (std::fwrite(buf.data(), 1, buf.size(), m_append) != buf.size() || std::fflush(m_append) != 0) {
        m_lastWriteErrno = errno;
        AfterWriteFailureLocked();
        return false;
    }
    m_fileBytes += buf.size();
    m_writtenSinceCheck += buf.size();
    return true;
}

uint64_t Dataset::DeadBytesLocked() const {
    const uint64_t used = kFileHeaderBytes + m_liveBytes;
    return m_fileBytes > used ? m_fileBytes - used : 0;
}

void Dataset::MaybeCompactLocked(bool atStartup) {
    if (m_mode != Mode::File || !m_append || m_noCompact) return;
    const uint64_t dead = DeadBytesLocked();
    if (dead < kMinDeadBytesToTidy || dead * 3 < m_fileBytes) return;
    if (!atStartup && m_liveBytes > m_opt.quickCompactBytes) return; // big sets are tidied up at the next start instead
    RewriteLocked();
}

// Copies the pictures that are still in the set into a fresh file and swaps it in. The records are copied unchanged (after
// being checked), so a picture that cannot be read any more is left behind instead of being carried over.
bool Dataset::RewriteLocked() {
    if (m_mode != Mode::File || m_noCompact) return false;
    CloseHandlesLocked();
    const std::string tmpPath = TempFilePath();
    FILE* src = std::fopen(FilePath().c_str(), "rb");
    FILE* dst = std::fopen(tmpPath.c_str(), "wb");
    bool ok = src != nullptr && dst != nullptr;
    if (ok) {
        std::setvbuf(src, nullptr, _IONBF, 0);
        std::setvbuf(dst, nullptr, _IOFBF, 1u << 20);
        ok = WriteFileHeader(dst);
    }
    std::vector<uint64_t> newOffsets(m_cards.size(), 0);
    uint64_t bytes = kFileHeaderBytes;
    size_t unreadable = 0;
    std::vector<uint8_t> rec;
    for (size_t i = 0; ok && i < m_cards.size(); ++i) {
        Entry& e = m_cards[i];
        const ReadResult r = ReadRecord(src, e, rec);
        if (r == ReadResult::Corrupt) {
            e.gone = true;
            ++unreadable;
            continue;
        }
        if (r == ReadResult::IoError || std::fwrite(rec.data(), 1, rec.size(), dst) != rec.size()) {
            ok = false;
            break;
        }
        newOffsets[i] = bytes;
        bytes += rec.size();
    }
    if (src) std::fclose(src);
    if (dst) {
        ok = (std::fflush(dst) == 0) && ok;
        ok = (std::fclose(dst) == 0) && ok;
    }
    std::error_code ec;
    if (ok) {
        std::filesystem::rename(tmpPath, FilePath(), ec);
        if (ec) {
            ec.clear();
            std::filesystem::remove(FilePath(), ec);
            ec.clear();
            std::filesystem::rename(tmpPath, FilePath(), ec);
            ok = !ec;
        }
    }
    if (!ok) {
        std::filesystem::remove(tmpPath, ec);
        for (Entry& e : m_cards) e.gone = false; // nothing was changed
        if (std::filesystem::exists(FilePath(), ec)) {
            OpenHandlesLocked();
        } else {
            m_noCompact = true; // the old file is gone and the new one could not be put in place: leave everything alone
        }
        return false;
    }
    for (size_t i = 0; i < m_cards.size(); ++i) {
        if (!m_cards[i].gone) m_cards[i].offset = newOffsets[i];
    }
    m_unreadablePending += unreadable;
    DropGoneLocked();
    m_fileBytes = bytes;
    m_liveBytes = bytes - kFileHeaderBytes;
    OpenHandlesLocked();
    return true;
}

// ---------------------------------------------------------------------------------------------
// Free space on the storage
// ---------------------------------------------------------------------------------------------
void Dataset::RefreshFreeSpaceLocked(bool force) const {
    const auto now = std::chrono::steady_clock::now();
    const std::chrono::steady_clock::time_point never{};
    if (!force && m_freeCheckedAt != never && now - m_freeCheckedAt < std::chrono::seconds(2)) return;
    m_freeCheckedAt = now;
    uint64_t freeBytes = 0;
    bool ok = false;
    if (m_freeReader) {
        ok = m_freeReader(m_folder, freeBytes);
    } else {
#ifndef _WIN32
        ok = StandardFreeBytes(m_folder, freeBytes);
#endif
    }
    m_freeKnown = ok;
    if (ok) {
        m_freeBytes = freeBytes;
        m_writtenSinceCheck = 0;
    }
}

bool Dataset::RoomForLocked(uint64_t bytes, uint64_t* freeOut) const {
    RefreshFreeSpaceLocked(false);
    if (!m_freeKnown) return true;
    const uint64_t avail = m_freeBytes > m_writtenSinceCheck ? m_freeBytes - m_writtenSinceCheck : 0;
    if (freeOut) *freeOut = avail;
    return avail >= bytes && avail - bytes >= m_opt.keepFreeBytes;
}

// ---------------------------------------------------------------------------------------------
// The cache of pictures kept in memory
// ---------------------------------------------------------------------------------------------
void Dataset::CacheInsertLocked(uint64_t id, const uint8_t* px) {
    uint32_t slot = 0;
    const auto it = m_cacheIndex.find(id);
    if (it != m_cacheIndex.end()) {
        slot = it->second;
    } else if (!m_freeSlots.empty()) {
        slot = m_freeSlots.back();
        m_freeSlots.pop_back();
    } else if (m_cacheSlots.size() < m_opt.cacheCards) {
        slot = static_cast<uint32_t>(m_cacheSlots.size());
        m_cacheSlots.emplace_back();
    } else {
        // full: the picture that has been in memory the longest makes room
        slot = static_cast<uint32_t>(m_cacheHand % m_cacheSlots.size());
        m_cacheHand = (m_cacheHand + 1) % m_cacheSlots.size();
        m_cacheIndex.erase(m_cacheSlots[slot].id);
    }
    CacheSlot& s = m_cacheSlots[slot];
    s.id = id;
    s.px.assign(px, px + kCardPixelBytes);
    m_cacheIndex[id] = slot;
}

void Dataset::CacheEraseLocked(uint64_t id) {
    const auto it = m_cacheIndex.find(id);
    if (it == m_cacheIndex.end()) return;
    const uint32_t slot = it->second;
    m_cacheIndex.erase(it);
    m_cacheSlots[slot].id = 0;
    m_freeSlots.push_back(slot);
}

void Dataset::CacheClearLocked() {
    std::vector<CacheSlot>().swap(m_cacheSlots);
    m_cacheIndex.clear();
    std::vector<uint32_t>().swap(m_freeSlots);
    m_cacheHand = 0;
}

// ---------------------------------------------------------------------------------------------
// Pictures
// ---------------------------------------------------------------------------------------------
size_t Dataset::FindIndexLocked(uint64_t id) const {
    const auto it = std::lower_bound(m_cards.begin(), m_cards.end(), id, [](const Entry& e, uint64_t v) { return e.id < v; });
    if (it == m_cards.end() || it->id != id) return kNpos;
    return static_cast<size_t>(it - m_cards.begin());
}

size_t Dataset::EffectiveMaxLocked() const {
    return m_mode == Mode::File ? m_opt.maxCards : m_memoryCap;
}

// Reads one picture's whole record and checks it. `record` is only valid when the result is Ok.
Dataset::ReadResult Dataset::ReadRecord(FILE* f, const Entry& e, std::vector<uint8_t>& record) {
    if (!f || e.recordBytes < kRecordHeaderBytes + kAddFixedBytes + kCardPixelBytes) return ReadResult::IoError;
    record.resize(e.recordBytes);
    if (!SeekTo(f, e.offset)) return ReadResult::IoError;
    const size_t got = std::fread(record.data(), 1, record.size(), f);
    if (got != record.size()) {
        const bool failed = std::ferror(f) != 0;
        std::clearerr(f);
        return failed ? ReadResult::IoError : ReadResult::Corrupt; // a short read without an error = the file is shorter than it should be
    }
    const uint32_t len = Get32(record.data() + 1);
    if (record[0] != kRecAdd || len != record.size() - kRecordHeaderBytes) return ReadResult::Corrupt;
    const uint8_t* payload = record.data() + kRecordHeaderBytes;
    if (RecordCrc(kRecAdd, payload, len) != Get32(record.data() + 5)) return ReadResult::Corrupt;
    if (Get64(payload) != e.id || Get64(payload + 8) != e.hash) return ReadResult::Corrupt;
    return ReadResult::Ok;
}

// The pixels of a picture: from memory if they are there, otherwise from the file (and then kept in the cache). The pointer
// stays valid until the next call. Null if they could not be read; a picture whose data is damaged is marked `gone`.
const uint8_t* Dataset::PixelsLocked(Entry& e, ReadResult* why) {
    if (!e.px.empty()) return e.px.data();
    const auto it = m_cacheIndex.find(e.id);
    if (it != m_cacheIndex.end()) return m_cacheSlots[it->second].px.data();
    const ReadResult r = ReadRecord(m_read, e, m_scratch);
    if (r != ReadResult::Ok) {
        if (why) *why = r;
        if (r == ReadResult::Corrupt) {
            e.gone = true;
            ++m_unreadablePending;
        }
        return nullptr;
    }
    CacheInsertLocked(e.id, m_scratch.data() + (e.recordBytes - kCardPixelBytes));
    return m_cacheSlots[m_cacheIndex[e.id]].px.data();
}

// Takes the pictures marked `gone` out of the set.
void Dataset::DropGoneLocked() {
    size_t keep = 0;
    for (size_t i = 0; i < m_cards.size(); ++i) {
        Entry& e = m_cards[i];
        if (e.gone) {
            const auto hit = m_hashToId.find(e.hash);
            if (hit != m_hashToId.end() && hit->second == e.id) m_hashToId.erase(hit);
            CacheEraseLocked(e.id);
            m_liveBytes -= std::min<uint64_t>(m_liveBytes, e.recordBytes);
            continue;
        }
        if (keep != i) m_cards[keep] = std::move(m_cards[i]);
        ++keep;
    }
    m_cards.erase(m_cards.begin() + static_cast<std::ptrdiff_t>(keep), m_cards.end());
    m_orderDirty = true;
    ++m_version;
}

Dataset::AddResult Dataset::Add(const uint8_t* rgb, uint32_t w, uint32_t h, const std::string& name, ResizeStyle style) {
    AddResult res;
    if (!rgb || w < 8 || h < 8 || w > 8192 || h > 8192) {
        res.status = AddStatus::Invalid;
        res.message = "The picture size is not usable (it must be between 8 and 8192 pixels).";
        return res;
    }
    std::vector<uint8_t> px;
    if (w == kCardSize && h == kCardSize) {
        px.assign(rgb, rgb + static_cast<size_t>(w) * h * 3);
    } else {
        Rgb8Image fitted = FitToSquare(rgb, w, h, kCardSize, FitMode::CropCenter, style);
        if (!fitted.Valid()) {
            res.status = AddStatus::Invalid;
            res.message = "The picture could not be resized.";
            return res;
        }
        px = std::move(fitted.px);
    }
    const uint64_t hash = Fnv1a64(px.data(), px.size());
    const std::string cleanName = CleanName(name);

    std::lock_guard<std::mutex> lock(m_mutex);
    const auto dup = m_hashToId.find(hash);
    if (dup != m_hashToId.end()) {
        const size_t idx = FindIndexLocked(dup->second);
        if (idx != kNpos) {
            Entry& existing = m_cards[idx];
            ReadResult why = ReadResult::Ok;
            const uint8_t* have = PixelsLocked(existing, &why);
            if (have && std::memcmp(have, px.data(), px.size()) == 0) {
                res.status = AddStatus::Duplicate;
                res.id = existing.id;
                res.message = "This picture is already in the set.";
                return res;
            }
            if (!have && why == ReadResult::Corrupt) DropGoneLocked(); // the stored copy is damaged: this one replaces it
        }
    }
    if (m_cards.size() >= EffectiveMaxLocked()) {
        res.status = AddStatus::Full;
        res.message = "The set is full (" + std::to_string(EffectiveMaxLocked()) + " pictures). Remove some pictures first.";
        return res;
    }

    const uint64_t id = m_nextId++; // an id is never used twice, even if saving fails
    uint64_t start = 0;
    uint32_t recordBytes = 0;
    if (m_mode == Mode::File) {
        if (!m_append) {
            res.status = AddStatus::DiskError;
            res.message = "The picture could not be saved: the Xbox's storage cannot be written right now.";
            return res;
        }
        const uint64_t need = static_cast<uint64_t>(kRecordHeaderBytes) + kAddFixedBytes + cleanName.size() + kCardPixelBytes;
        uint64_t freeBytes = 0;
        if (!RoomForLocked(need, &freeBytes)) {
            res.status = AddStatus::Full;
            res.message = "The Xbox's storage is almost full (about " + std::to_string(freeBytes / (1024 * 1024)) +
                          " MB left), so no more pictures can be saved. Free up space on the console, or remove pictures here; space from "
                          "removed pictures comes back the next time the app starts.";
            return res;
        }
        std::vector<uint8_t> payload;
        payload.reserve(kAddFixedBytes + cleanName.size() + px.size());
        Put64(payload, id);
        Put64(payload, hash);
        Put32(payload, kCardSize);
        Put32(payload, kCardSize);
        Put16(payload, static_cast<uint16_t>(cleanName.size()));
        payload.insert(payload.end(), cleanName.begin(), cleanName.end());
        payload.insert(payload.end(), px.begin(), px.end());
        if (!AppendRecordLocked(kRecAdd, payload, &start)) {
            const bool full = m_lastWriteErrno == ENOSPC
#ifdef EDQUOT
                              || m_lastWriteErrno == EDQUOT
#endif
                              || m_lastWriteErrno == EFBIG;
            res.status = full ? AddStatus::Full : AddStatus::DiskError;
            res.message = full ? "The Xbox's storage is full, so no more pictures can be saved. Free up space on the console, or remove pictures here."
                               : "The picture could not be saved to the Xbox's storage.";
            return res;
        }
        recordBytes = static_cast<uint32_t>(payload.size() + kRecordHeaderBytes);
    }

    Entry e;
    e.id = id;
    e.hash = hash;
    e.offset = start;
    e.recordBytes = recordBytes;
    e.name = cleanName;
    if (m_mode == Mode::File) {
        CacheInsertLocked(id, px.data());
        m_liveBytes += recordBytes;
    } else {
        e.px = std::move(px);
    }
    m_hashToId[hash] = id;
    m_cards.push_back(std::move(e));

    // A picture that arrives in the middle of a pass is dealt later in the same pass.
    if (!m_orderDirty && m_order.size() + 1 == m_cards.size()) {
        const size_t from = std::min(m_cursor, m_order.size());
        const size_t span = m_order.size() - from + 1;
        const size_t at = from + static_cast<size_t>(((id * 0x9E3779B97F4A7C15ull) >> 33) % span);
        m_order.insert(m_order.begin() + static_cast<std::ptrdiff_t>(at), static_cast<uint32_t>(m_cards.size() - 1));
    } else {
        m_orderDirty = true;
    }
    ++m_version;
    res.status = AddStatus::Added;
    res.id = id;
    return res;
}

bool Dataset::Remove(uint64_t id) {
    std::vector<uint64_t> ids(1, id);
    return RemoveMany(ids) == 1;
}

size_t Dataset::RemoveMany(const std::vector<uint64_t>& ids) {
    std::lock_guard<std::mutex> lock(m_mutex);
    std::vector<uint64_t> removedIds;
    removedIds.reserve(ids.size());
    for (uint64_t id : ids) {
        const size_t idx = FindIndexLocked(id);
        if (idx == kNpos || m_cards[idx].gone) continue;
        m_cards[idx].gone = true;
        removedIds.push_back(id);
    }
    if (removedIds.empty()) return 0;
    if (m_mode == Mode::File) AppendRemovesLocked(removedIds);
    DropGoneLocked();
    if (m_mode == Mode::File) {
        if (m_cards.empty()) {
            ResetFileLocked(); // nothing left: the whole file is given back at once
        } else {
            MaybeCompactLocked(false);
        }
    }
    return removedIds.size();
}

void Dataset::Clear() {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_cards.empty()) return;
    std::vector<Entry>().swap(m_cards);
    m_hashToId.clear();
    CacheClearLocked();
    m_orderDirty = true;
    m_cursor = 0;
    ++m_version;
    if (m_mode == Mode::File) ResetFileLocked();
}

size_t Dataset::Size() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_cards.size();
}

size_t Dataset::Capacity() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    size_t cap = EffectiveMaxLocked();
    if (m_mode == Mode::File) {
        RefreshFreeSpaceLocked(false);
        if (m_freeKnown) {
            const uint64_t avail = m_freeBytes > m_writtenSinceCheck ? m_freeBytes - m_writtenSinceCheck : 0;
            const uint64_t usable = avail > m_opt.keepFreeBytes ? avail - m_opt.keepFreeBytes : 0;
            const uint64_t room = std::min<uint64_t>(usable / kTypicalRecordBytes, cap);
            cap = std::min(cap, m_cards.size() + static_cast<size_t>(room));
        }
    }
    return std::max(cap, m_cards.size());
}

uint64_t Dataset::Version() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_version;
}

uint64_t Dataset::DiskBytes() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_fileBytes;
}

bool Dataset::Persistent() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_mode == Mode::File && m_append != nullptr;
}

size_t Dataset::CachedCards() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_cacheIndex.size();
}

size_t Dataset::TakeUnreadableCount() {
    std::lock_guard<std::mutex> lock(m_mutex);
    const size_t n = m_unreadablePending;
    m_unreadablePending = 0;
    return n;
}

std::vector<DatasetCardInfo> Dataset::List(size_t offset, size_t limit) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    std::vector<DatasetCardInfo> out;
    for (size_t i = offset; i < m_cards.size() && out.size() < limit; ++i) {
        DatasetCardInfo info;
        info.id = m_cards[i].id;
        info.hash = m_cards[i].hash;
        info.w = kCardSize;
        info.h = kCardSize;
        info.name = m_cards[i].name;
        out.push_back(std::move(info));
    }
    return out;
}

bool Dataset::GetImage(uint64_t id, Rgb8Image& out, std::string* name) {
    std::lock_guard<std::mutex> lock(m_mutex);
    const size_t idx = FindIndexLocked(id);
    if (idx == kNpos) return false;
    Entry& e = m_cards[idx];
    ReadResult why = ReadResult::Ok;
    const uint8_t* px = PixelsLocked(e, &why);
    if (!px) {
        if (why == ReadResult::Corrupt) DropGoneLocked();
        return false;
    }
    out.w = kCardSize;
    out.h = kCardSize;
    out.px.assign(px, px + kCardPixelBytes);
    if (name) *name = e.name;
    return true;
}

size_t Dataset::SamplePixels(size_t maxCards, std::vector<uint8_t>& out) {
    std::lock_guard<std::mutex> lock(m_mutex);
    out.clear();
    const size_t total = m_cards.size();
    if (total == 0 || maxCards == 0) return 0;
    const size_t take = std::min(total, maxCards);
    out.reserve(take * static_cast<size_t>(kCardPixelBytes));
    size_t copied = 0;
    bool anyUnreadable = false;
    for (size_t k = 0; k < take; ++k) {
        Entry& card = m_cards[static_cast<size_t>((static_cast<uint64_t>(k) * total) / take)]; // evenly spread over the set
        if (card.gone) continue;
        ReadResult why = ReadResult::Ok;
        const uint8_t* px = PixelsLocked(card, &why);
        if (!px) {
            if (why == ReadResult::Corrupt) anyUnreadable = true;
            continue;
        }
        out.insert(out.end(), px, px + kCardPixelBytes);
        ++copied;
    }
    if (anyUnreadable) DropGoneLocked();
    return copied;
}

double Dataset::EpochProgress() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    const double frac = m_order.empty() ? 0.0 : static_cast<double>(m_cursor) / static_cast<double>(m_order.size());
    return static_cast<double>(m_epochs) + std::min(1.0, frac);
}

void Dataset::ResetEpochCounter() {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_epochs = 0;
    m_cursor = 0;
    m_orderDirty = true;
}

bool Dataset::SampleBatch(uint32_t count, uint32_t res, const AugmentOptions& aug, std::mt19937_64& rng, std::vector<float>& outCHW) {
    std::lock_guard<std::mutex> lock(m_mutex);
    outCHW.clear();
    if (m_cards.empty() || count == 0) return false;
    res = std::clamp<uint32_t>(res, 8u, 128u);
    const size_t hw = static_cast<size_t>(res) * res;
    const size_t stride = 3 * hw;
    outCHW.assign(static_cast<size_t>(count) * stride, 0.0f);

    std::uniform_real_distribution<double> unit(0.0, 1.0);
    std::vector<float> tmp;
    uint32_t produced = 0;
    uint32_t attempts = 0;
    bool anyUnreadable = false;
    const uint32_t maxAttempts = count * 4 + 64; // pictures that cannot be read are skipped, but never forever
    while (produced < count && attempts < maxAttempts) {
        ++attempts;
        if (m_orderDirty || m_order.size() != m_cards.size()) {
            m_order.resize(m_cards.size());
            std::iota(m_order.begin(), m_order.end(), 0u);
            std::shuffle(m_order.begin(), m_order.end(), rng);
            m_cursor = 0;
            m_orderDirty = false;
        }
        if (m_cursor >= m_order.size()) {
            std::shuffle(m_order.begin(), m_order.end(), rng);
            m_cursor = 0;
            ++m_epochs;
        }
        Entry& card = m_cards[m_order[m_cursor++]];
        if (card.gone) continue;
        ReadResult why = ReadResult::Ok;
        const uint8_t* px = PixelsLocked(card, &why);
        if (!px) {
            if (why == ReadResult::Corrupt) anyUnreadable = true;
            continue;
        }

        double cx = 0.0, cy = 0.0, side = kCardSize;
        if (aug.zoom) {
            const double scale = 0.85 + 0.15 * unit(rng);
            side = kCardSize * scale;
            cx = (kCardSize - side) * unit(rng);
            cy = (kCardSize - side) * unit(rng);
        }
        const bool flip = aug.flip && unit(rng) < 0.5;
        float contrast = 1.0f, bright = 0.0f;
        if (aug.color) {
            contrast = static_cast<float>(0.92 + 0.16 * unit(rng));
            bright = static_cast<float>(-0.06 + 0.12 * unit(rng));
        }
        if (!ResampleRegion(px, kCardSize, kCardSize, cx, cy, side, side, res, res, flip, tmp)) {
            continue; // cannot happen with valid sizes
        }
        float* dst = outCHW.data() + static_cast<size_t>(produced) * stride;
        for (size_t i = 0; i < hw; ++i) {
            for (size_t c = 0; c < 3; ++c) {
                float v = tmp[i * 3 + c] / 127.5f - 1.0f;
                v = std::clamp(v * contrast + bright, -1.0f, 1.0f);
                dst[c * hw + i] = v;
            }
        }
        ++produced;
    }
    if (anyUnreadable) DropGoneLocked();
    if (produced == 0) {
        outCHW.clear();
        return false;
    }
    // Some pictures could not be read: repeat the ones that could, so the batch still has the size the trainer asked for.
    for (uint32_t n = produced; n < count; ++n) {
        const float* src = outCHW.data() + static_cast<size_t>(n % produced) * stride;
        std::copy(src, src + stride, outCHW.data() + static_cast<size_t>(n) * stride);
    }
    return true;
}

} // namespace XboxOTFM
