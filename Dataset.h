#pragma once
// The training pictures (128 x 128, 8-bit RGB).
//
// They are kept in one file on the Xbox's storage (cards.otds) and written there as they arrive, so they are still there the
// next time the app starts. Only a small index (name, fingerprint, place in the file) and a limited number of recently used
// pictures are held in memory, so the set can grow far beyond what fits in the app's memory.
//
// Thread-safe: the phone page, the training thread and the TV screen all use it.

#include "ImageUtil.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <mutex>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace XboxOTFM {

struct DatasetCardInfo {
    uint64_t id = 0;
    uint64_t hash = 0; // fingerprint of the pixels (the web page uses it so a cached thumbnail can never show the wrong picture)
    uint32_t w = 0;
    uint32_t h = 0;
    std::string name;
};

struct AugmentOptions {
    bool flip = false;  // mirror left/right half of the time
    bool zoom = false;  // random zoom 85-100% with a random shift
    bool color = false; // small random brightness / contrast changes
};

struct DatasetOptions {
    // The most pictures the set will hold. Each one costs about 49 KB of storage and about 0.2 KB of memory for the index.
    size_t maxCards = 100000;
    // How many pictures are kept in memory (49 KB each). Pictures outside the cache are read from the file when needed.
    size_t cacheCards = 3000;
    // New pictures are refused when the storage would drop below this much free space.
    uint64_t keepFreeBytes = 512ull * 1024 * 1024;
    // After removing pictures the file is tidied up at once only while the remaining pictures take less space than this;
    // bigger sets are tidied up the next time the app starts, so a removal never freezes the app for a long time.
    uint64_t quickCompactBytes = 256ull * 1024 * 1024;
};

class Dataset {
public:
    static constexpr uint32_t kCardSize = 128;
    static constexpr size_t kDefaultMaxCards = 100000;
    static constexpr size_t kDefaultCacheCards = 3000;

    // folder: where cards.otds is kept (created if needed). An empty folder = memory only (used by tests).
    explicit Dataset(std::string folder = std::string(), size_t maxCards = kDefaultMaxCards);
    Dataset(std::string folder, const DatasetOptions& options);
    ~Dataset();
    Dataset(const Dataset&) = delete;
    Dataset& operator=(const Dataset&) = delete;

    // Says how much free space the storage that holds `folder` has (the Xbox build provides this). Without it the free space
    // is not checked on Windows; on other systems the standard library is asked.
    using FreeSpaceReader = std::function<bool(const std::string& folder, uint64_t& freeBytes)>;
    void SetFreeSpaceReader(FreeSpaceReader reader);

    // Loads the saved pictures. Problems that can be survived (a damaged part, an unwritable folder) are reported in warnings.
    bool Open(std::vector<std::string>* warnings = nullptr);

    enum class AddStatus { Added, Duplicate, Full, Invalid, DiskError };
    struct AddResult {
        AddStatus status = AddStatus::Invalid;
        uint64_t id = 0;
        std::string message;
    };
    // rgb: w*h*3 bytes. Anything that is not 128x128 is centre-cropped and resized (`style` says how the pixels are made).
    AddResult Add(const uint8_t* rgb, uint32_t w, uint32_t h, const std::string& name, ResizeStyle style = ResizeStyle::Smooth);

    bool Remove(uint64_t id);
    size_t RemoveMany(const std::vector<uint64_t>& ids);
    void Clear();

    size_t Size() const;
    // How many pictures the set can hold right now: the limit, or less when the storage is nearly full.
    size_t Capacity() const;
    uint64_t Version() const;
    uint64_t DiskBytes() const;
    // True when new pictures are being written to storage (false in memory-only mode or when the file cannot be written).
    bool Persistent() const;
    // How many pictures are held in memory right now (for tests and diagnostics).
    size_t CachedCards() const;
    // Pictures that could not be read back from storage are dropped from the set. Returns how many were dropped since the
    // last call (and starts counting again from zero).
    size_t TakeUnreadableCount();

    std::vector<DatasetCardInfo> List(size_t offset, size_t limit) const;
    // Not const: a picture that is not in memory is read from the file and then kept in the cache.
    bool GetImage(uint64_t id, Rgb8Image& out, std::string* name = nullptr);
    // Copies the pixels of up to maxCards pictures spread evenly over the whole set, one after the other (128 x 128 x 3 bytes each).
    // Returns how many pictures were copied (used to find the colours of the set).
    size_t SamplePixels(size_t maxCards, std::vector<uint8_t>& out);

    // Fills outCHW with `count` pictures as [count, 3, res, res] floats in [-1, 1]. Pictures are dealt in shuffled passes over
    // the whole set (every picture once per pass). Returns false if there are no pictures.
    bool SampleBatch(uint32_t count, uint32_t res, const AugmentOptions& aug, std::mt19937_64& rng, std::vector<float>& outCHW);
    // How many full passes (epochs) have been dealt so far, and how far into the current pass we are (0..1).
    double EpochProgress() const;
    void ResetEpochCounter();

private:
    struct Entry {
        uint64_t id = 0;
        uint64_t hash = 0;
        uint64_t offset = 0;        // file mode: where this picture's record starts in cards.otds
        uint32_t recordBytes = 0;   // file mode: size of the record (header + payload)
        bool gone = false;          // removed or unreadable: dropped when the current operation finishes
        std::string name;
        std::vector<uint8_t> px;    // memory-only mode: the pixels (file mode keeps them in the cache instead)
    };
    struct CacheSlot {
        uint64_t id = 0;            // 0 = empty
        std::vector<uint8_t> px;
    };
    enum class Mode { Memory, File };
    enum class ReadResult { Ok, Corrupt, IoError };
    struct ScanInfo;

    // ---- opening and the file ----
    bool OpenLocked(std::vector<std::string>* warnings);
    bool ScanLocked(FILE* f, uint64_t fileSize, ScanInfo& info);
    bool OpenHandlesLocked();
    void CloseHandlesLocked();
    bool ResetFileLocked();                       // an empty file (just the header)
    bool RewriteLocked();                         // copies the live pictures into a fresh file (drops dead space)
    void MaybeCompactLocked(bool atStartup);
    uint64_t DeadBytesLocked() const;             // bytes of the file that no picture in the set needs
    bool AppendRecordLocked(uint8_t type, const std::vector<uint8_t>& payload, uint64_t* startOut);
    bool AppendRemovesLocked(const std::vector<uint64_t>& ids);
    void AfterWriteFailureLocked();
    std::string FilePath() const;
    std::string TempFilePath() const;

    // ---- pictures ----
    static constexpr size_t kNpos = static_cast<size_t>(-1);
    size_t FindIndexLocked(uint64_t id) const;
    size_t EffectiveMaxLocked() const;
    const uint8_t* PixelsLocked(Entry& e, ReadResult* why);
    static ReadResult ReadRecord(FILE* f, const Entry& e, std::vector<uint8_t>& record);
    void DropGoneLocked();
    void CacheInsertLocked(uint64_t id, const uint8_t* px);
    void CacheEraseLocked(uint64_t id);
    void CacheClearLocked();

    // ---- free space ----
    void RefreshFreeSpaceLocked(bool force) const;
    bool RoomForLocked(uint64_t bytes, uint64_t* freeOut) const;

    const std::string m_folder;
    const DatasetOptions m_opt;

    mutable std::mutex m_mutex;
    Mode m_mode = Mode::Memory;
    size_t m_memoryCap = 0;                       // memory-only mode: how many pictures fit
    std::vector<Entry> m_cards;                   // ascending by id
    std::unordered_map<uint64_t, uint64_t> m_hashToId;
    uint64_t m_nextId = 1;
    uint64_t m_version = 1;

    FILE* m_append = nullptr;                     // file mode: where new records are written
    FILE* m_read = nullptr;                       // file mode: where pictures are read back (no buffering)
    uint64_t m_fileBytes = 0;                     // size of the file = where the next record goes
    uint64_t m_liveBytes = 0;                     // bytes of the file that belong to pictures that are still in the set
    bool m_noCompact = false;                     // set when the file could not be read completely: never rewrite it then
    int m_lastWriteErrno = 0;                     // why the last write failed
    std::vector<uint8_t> m_scratch;

    std::vector<CacheSlot> m_cacheSlots;
    std::unordered_map<uint64_t, uint32_t> m_cacheIndex;
    std::vector<uint32_t> m_freeSlots;
    size_t m_cacheHand = 0;
    size_t m_unreadablePending = 0;

    FreeSpaceReader m_freeReader;
    mutable bool m_freeKnown = false;
    mutable uint64_t m_freeBytes = 0;
    mutable uint64_t m_writtenSinceCheck = 0;
    mutable std::chrono::steady_clock::time_point m_freeCheckedAt{};

    std::vector<uint32_t> m_order;                // shuffled positions for the current pass
    size_t m_cursor = 0;
    bool m_orderDirty = true;
    uint64_t m_epochs = 0;
};

} // namespace XboxOTFM
