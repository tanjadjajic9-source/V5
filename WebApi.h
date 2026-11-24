#pragma once
// The "API" behind the phone web page: turns web requests into calls on AppCore. No Windows code in here, so it is tested natively.

#include "AppCore.h"
#include "HttpServer.h"
#include "ImageUtil.h"

#include <functional>
#include <string>

namespace XboxOTFM {

struct WebApiHooks {
    // Turns a picture file (PNG, JPEG, BMP, GIF, ...) into pixels. Leave empty when the platform can't do it (then the page's own
    // in-browser conversion is the only way to add pictures).
    std::function<bool(const uint8_t* bytes, size_t size, Rgb8Image& out, std::string* error)> decodeImageFile;
};

class WebApi {
public:
    // pageBytes must stay valid for the life of the WebApi (it is the single-file web page).
    WebApi(AppCore& core, WebApiHooks hooks, const uint8_t* pageBytes, size_t pageSize);

    void Handle(const HttpRequest& req, HttpResponse& resp);
    size_t BodyLimit(const std::string& method, const std::string& path) const;

    // Largest upload for one picture sent as raw pixels, and for a snapshot file. A snapshot of the biggest one-pass brain (7.8 million
    // dials, with the smoothed copy and the optimizer memory) is about 125 MB; the limit stays under the server's 160 MB upload budget.
    static constexpr size_t kMaxRawPicturePixels = 1024u * 1024u;
    static constexpr size_t kMaxPictureFileBytes = 48u * 1024u * 1024u;
    static constexpr size_t kMaxSnapshotFileBytes = 140u * 1024u * 1024u;

private:
    void ServePage(HttpResponse& resp) const;
    void ServeStatus(HttpResponse& resp);
    void ServeToasts(const HttpRequest& req, HttpResponse& resp);
    void ServeLoss(const HttpRequest& req, HttpResponse& resp);
    void ServeLogs(const HttpRequest& req, HttpResponse& resp);
    void ServeSettings(const HttpRequest& req, HttpResponse& resp);
    void ServeTrain(const HttpRequest& req, const std::string& action, HttpResponse& resp);
    void ServePictures(const HttpRequest& req, HttpResponse& resp);
    void ServePictureImage(const HttpRequest& req, uint64_t id, HttpResponse& resp);
    void ServePictureUpload(const HttpRequest& req, HttpResponse& resp);
    void ServePictureFileUpload(const HttpRequest& req, HttpResponse& resp);
    void ServePictureDelete(const HttpRequest& req, HttpResponse& resp);
    void ServePreview(HttpResponse& resp, bool image);
    void ServeDream(const HttpRequest& req, HttpResponse& resp);
    void ServeDreamList(HttpResponse& resp);
    void ServeDreamImage(const HttpRequest& req, uint64_t id, HttpResponse& resp);
    void ServeSnapshotList(HttpResponse& resp);
    void ServeSnapshotCreate(const HttpRequest& req, HttpResponse& resp);
    void ServeSnapshotAction(const HttpRequest& req, const std::string& id, const std::string& action, HttpResponse& resp);
    void ServeSnapshotImport(const HttpRequest& req, HttpResponse& resp);
    void ServeExportBrain(HttpResponse& resp);

    static void Ok(HttpResponse& resp, JsonWriter& w);
    static void Fail(HttpResponse& resp, int status, const std::string& message);
    static bool ParseBodyObject(const HttpRequest& req, JsonValue& out, std::string* error);

    AppCore& m_core;
    WebApiHooks m_hooks;
    const uint8_t* m_page;
    size_t m_pageSize;
};

} // namespace XboxOTFM
