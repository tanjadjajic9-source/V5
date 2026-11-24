#pragma once
// The few things that only exist on Windows / the Xbox: where the app may keep files, the crash log, the phone address,
// and turning a picture file (PNG, JPEG, ...) into pixels. Everything else in the app is platform independent.

#include "ImageUtil.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace XboxOTFM {

class WebBridge {
public:
    // The app's private writable folder, without a closing slash. Falls back to "." if the system does not say.
    static std::string LocalFolderPath();

    // crash_log.txt inside that folder.
    static std::string CrashLogFilePath();
    // Adds one dated line to the crash log (never throws; the file is emptied when it gets large).
    static void WriteCrashLogToDisk(const std::string& entry);
    // The end of the crash log as text (for the log view on the phone page).
    static std::string ReadCrashLog(size_t maxBytes = 32768);

    // The address other devices on the network can use to reach this Xbox, like "192.168.1.20". Empty if there is none.
    static std::string DetectLanIPv4();

    // How many bytes this app may still write on the storage that holds `folder` (the picture set asks before it saves).
    // Returns false if the system does not say.
    static bool FreeBytes(const std::string& folder, uint64_t& freeBytes);

    // Turns a picture file into pixels with the system's picture decoder. Pictures bigger than 512 pixels are shrunk while they
    // are read, so a huge photo never needs a huge amount of memory.
    static bool DecodeImageFile(const uint8_t* bytes, size_t size, Rgb8Image& out, std::string* error);
};

} // namespace XboxOTFM
