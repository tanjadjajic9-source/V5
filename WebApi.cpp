#include "WebApi.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace XboxOTFM {

namespace {

const char* const kJson = "application/json; charset=utf-8";
const char* const kPng = "image/png";

bool ParseUnsignedText(const std::string& s, uint64_t& out) {
    if (s.empty() || s.size() > 18) return false;
    uint64_t v = 0;
    for (char c : s) {
        if (c < '0' || c > '9') return false;
        v = v * 10 + static_cast<uint64_t>(c - '0');
    }
    out = v;
    return true;
}

uint64_t QueryUInt(const HttpRequest& req, const char* name, uint64_t fallback, uint64_t lo, uint64_t hi) {
    uint64_t v = 0;
    if (!req.HasQuery(name) || !ParseUnsignedText(req.Query(name), v)) return std::clamp(fallback, lo, hi);
    return std::clamp(v, lo, hi);
}

bool StartsWith(const std::string& s, const char* prefix) {
    const size_t n = std::strlen(prefix);
    return s.size() >= n && s.compare(0, n, prefix) == 0;
}

bool EndsWith(const std::string& s, const char* suffix) {
    const size_t n = std::strlen(suffix);
    return s.size() >= n && s.compare(s.size() - n, n, suffix) == 0;
}

// "/api/pictures/123.png" with prefix "/api/pictures/" and suffix ".png" -> 123
bool IdBetween(const std::string& path, const char* prefix, const char* suffix, uint64_t& id) {
    if (!StartsWith(path, prefix) || !EndsWith(path, suffix)) return false;
    const size_t a = std::strlen(prefix);
    const size_t b = std::strlen(suffix);
    if (path.size() <= a + b) return false;
    return ParseUnsignedText(path.substr(a, path.size() - a - b), id);
}

std::string HashText(uint64_t hash) {
    char buf[16];
    std::snprintf(buf, sizeof(buf), "%08x", static_cast<unsigned>((hash ^ (hash >> 32)) & 0xFFFFFFFFu));
    return buf;
}

std::string FileStem(const std::string& name, const std::string& fallback) {
    std::string out;
    for (unsigned char c : name) {
        const bool ok = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '-' || c == '_';
        if (ok) out += static_cast<char>(c);
        else if (c == ' ' && !out.empty() && out.back() != '_') out += '_';
        if (out.size() >= 40) break;
    }
    while (!out.empty() && out.back() == '_') out.pop_back();
    return out.empty() ? fallback : out;
}

const char* AddStatusName(Dataset::AddStatus s) {
    switch (s) {
        case Dataset::AddStatus::Added: return "added";
        case Dataset::AddStatus::Duplicate: return "duplicate";
        case Dataset::AddStatus::Full: return "full";
        case Dataset::AddStatus::Invalid: return "invalid";
        default: return "diskError";
    }
}

std::vector<uint8_t> PngOf(const Rgb8Image& img) {
    return EncodePngRgb(img.px.data(), img.w, img.h);
}

} // namespace

WebApi::WebApi(AppCore& core, WebApiHooks hooks, const uint8_t* pageBytes, size_t pageSize)
    : m_core(core), m_hooks(std::move(hooks)), m_page(pageBytes), m_pageSize(pageSize) {
}

void WebApi::Ok(HttpResponse& resp, JsonWriter& w) {
    resp.SetText(200, kJson, w.Take());
}

void WebApi::Fail(HttpResponse& resp, int status, const std::string& message) {
    JsonWriter w;
    w.BeginObject().Field("ok", false).Field("error", message).EndObject();
    resp.SetText(status, kJson, w.Take());
}

bool WebApi::ParseBodyObject(const HttpRequest& req, JsonValue& out, std::string* error) {
    if (req.body.empty()) {
        out = JsonValue();
        out.type = JsonValue::Type::Object;
        return true;
    }
    const std::string text(req.body.begin(), req.body.end());
    std::string err;
    if (!JsonParser::Parse(text, out, &err) || !out.IsObject()) {
        if (error) *error = "The request was not understood (it should be a JSON object).";
        return false;
    }
    return true;
}

size_t WebApi::BodyLimit(const std::string& method, const std::string& path) const {
    if (method == "POST") {
        if (path == "/api/pictures") return kMaxRawPicturePixels * 3 + 1024;
        if (path == "/api/pictures/file") return kMaxPictureFileBytes;
        if (path == "/api/snapshots/import") return kMaxSnapshotFileBytes;
        // up to 100,000 picture numbers (about 0.9 MB). The JSON reader takes at most 1 MB of text, so a longer list is refused
        // here with a clear "too large" instead of a confusing "not understood" later.
        if (path == "/api/pictures/delete") return 1024 * 1024;
    }
    return 64 * 1024;
}

void WebApi::ServePage(HttpResponse& resp) const {
    resp.status = 200;
    resp.contentType = "text/html; charset=utf-8";
    resp.body.assign(m_page, m_page + m_pageSize);
    resp.cacheControl = "no-store";
    resp.csp = "default-src 'self'; img-src 'self' data: blob:; font-src data:; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; "
               "connect-src 'self'; object-src 'none'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'";
}

void WebApi::Handle(const HttpRequest& req, HttpResponse& resp) {
    const bool isRead = req.method == "GET" || req.method == "HEAD";
    const std::string& p = req.path;

    if (isRead && (p == "/" || p == "/index.html")) {
        ServePage(resp);
        return;
    }
    if (isRead && p == "/favicon.ico") {
        resp.status = 204;
        return;
    }
    if (!StartsWith(p, "/api/")) {
        Fail(resp, 404, "There is nothing at this address. Open the main page instead.");
        return;
    }
    // Everything that changes something needs this header; a web page on another site can't add it (browsers block that), so
    // other sites can't press the buttons on your Xbox.
    if (!isRead && req.Header("x-otfm") != "1") {
        Fail(resp, 403, "The safety header is missing. Reload the page and try again.");
        return;
    }

    uint64_t id = 0;

    // ---- read-only state ----
    if (isRead) {
        if (p == "/api/health") {
            JsonWriter w;
            w.BeginObject().Field("ok", true).EndObject();
            Ok(resp, w);
        } else if (p == "/api/status") {
            ServeStatus(resp);
        } else if (p == "/api/toasts") {
            ServeToasts(req, resp);
        } else if (p == "/api/loss") {
            ServeLoss(req, resp);
        } else if (p == "/api/logs") {
            ServeLogs(req, resp);
        } else if (p == "/api/settings") {
            ServeSettings(req, resp);
        } else if (p == "/api/pictures") {
            ServePictures(req, resp);
        } else if (IdBetween(p, "/api/pictures/", ".png", id)) {
            ServePictureImage(req, id, resp);
        } else if (p == "/api/preview.png") {
            ServePreview(resp, true);
        } else if (p == "/api/preview") {
            ServePreview(resp, false);
        } else if (p == "/api/dreams") {
            ServeDreamList(resp);
        } else if (IdBetween(p, "/api/dream/", ".png", id)) {
            ServeDreamImage(req, id, resp);
        } else if (p == "/api/snapshots") {
            ServeSnapshotList(resp);
        } else if (StartsWith(p, "/api/snapshots/") && EndsWith(p, "/download")) {
            const std::string sid = p.substr(15, p.size() - 15 - 9);
            ServeSnapshotAction(req, sid, "download", resp);
        } else if (p == "/api/export-brain") {
            ServeExportBrain(resp);
        } else {
            Fail(resp, 404, "Unknown request.");
        }
        return;
    }

    // ---- changes ----
    if (req.method == "POST") {
        if (p == "/api/settings" || p == "/api/settings/reset") {
            ServeSettings(req, resp);
        } else if (StartsWith(p, "/api/train/")) {
            ServeTrain(req, p.substr(11), resp);
        } else if (p == "/api/pictures") {
            ServePictureUpload(req, resp);
        } else if (p == "/api/pictures/file") {
            ServePictureFileUpload(req, resp);
        } else if (p == "/api/pictures/delete" || p == "/api/pictures/clear") {
            ServePictureDelete(req, resp);
        } else if (p == "/api/dream") {
            ServeDream(req, resp);
        } else if (p == "/api/snapshots") {
            ServeSnapshotCreate(req, resp);
        } else if (p == "/api/snapshots/import") {
            ServeSnapshotImport(req, resp);
        } else if (StartsWith(p, "/api/snapshots/")) {
            const std::string rest = p.substr(15);
            const size_t slash = rest.find('/');
            if (slash == std::string::npos || slash == 0) {
                Fail(resp, 404, "Unknown request.");
            } else {
                ServeSnapshotAction(req, rest.substr(0, slash), rest.substr(slash + 1), resp);
            }
        } else {
            Fail(resp, 404, "Unknown request.");
        }
        return;
    }
    if (req.method == "DELETE") {
        if (StartsWith(p, "/api/pictures/") && ParseUnsignedText(p.substr(14), id)) {
            const bool removed = m_core.Pictures().Remove(id);
            JsonWriter w;
            w.BeginObject().Field("ok", true).Field("removed", removed).Field("total", static_cast<uint64_t>(m_core.Pictures().Size())).EndObject();
            Ok(resp, w);
        } else if (StartsWith(p, "/api/snapshots/") && p.find('/', 15) == std::string::npos) {
            ServeSnapshotAction(req, p.substr(15), "delete", resp);
        } else {
            Fail(resp, 404, "Unknown request.");
        }
        return;
    }
    Fail(resp, 405, "That kind of request is not supported here.");
}

// ---------------------------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------------------------
void WebApi::ServeStatus(HttpResponse& resp) {
    const std::string j = m_core.SnapshotJson();
    resp.SetText(200, kJson, std::string("{\"ok\":true,") + j.substr(1));
}

void WebApi::ServeToasts(const HttpRequest& req, HttpResponse& resp) {
    const uint64_t after = QueryUInt(req, "after", 0, 0, ~0ull >> 2);
    const std::vector<Toast> toasts = m_core.ToastsAfter(after);
    JsonWriter w;
    w.BeginObject().Field("ok", true);
    w.Key("toasts").BeginArray();
    uint64_t latest = after;
    for (const Toast& t : toasts) {
        w.BeginObject().Field("id", t.id).Field("level", t.level).Field("text", t.text).EndObject();
        latest = std::max(latest, t.id);
    }
    w.EndArray();
    w.Field("latest", latest);
    w.EndObject();
    Ok(resp, w);
}

void WebApi::ServeLoss(const HttpRequest& req, HttpResponse& resp) {
    const size_t maxPoints = static_cast<size_t>(QueryUInt(req, "max", 300, 10, 1000));
    const auto pts = m_core.GetTrainer().LossCurve(maxPoints);
    JsonWriter w;
    w.BeginObject().Field("ok", true).Field("version", m_core.GetTrainer().LossVersion());
    w.Key("points").BeginArray();
    for (const auto& pt : pts) {
        w.BeginArray().Value(pt.first).Value(pt.second).EndArray();
    }
    w.EndArray().EndObject();
    Ok(resp, w);
}

void WebApi::ServeLogs(const HttpRequest& req, HttpResponse& resp) {
    resp.SetText(200, "text/plain; charset=utf-8", m_core.LogText());
    if (req.HasQuery("download")) resp.downloadName = "xbox-trainer-log.txt";
}

// ---------------------------------------------------------------------------------------------
// Settings and training control
// ---------------------------------------------------------------------------------------------
void WebApi::ServeSettings(const HttpRequest& req, HttpResponse& resp) {
    std::vector<std::string> changed;
    if (req.method == "POST") {
        if (req.path == "/api/settings/reset") {
            m_core.ResetSettingsToRecommended();
        } else {
            JsonValue patch;
            std::string err;
            if (!ParseBodyObject(req, patch, &err)) {
                Fail(resp, 400, err);
                return;
            }
            if (!m_core.PatchSettings(patch, &err, &changed)) {
                Fail(resp, 400, err);
                return;
            }
        }
    }
    JsonWriter w;
    w.BeginObject().Field("ok", true);
    w.Key("changed").BeginArray();
    for (const std::string& c : changed) w.Value(c);
    w.EndArray();
    w.Key("settings");
    WriteConfigJson(w, m_core.GetConfig());
    w.Key("defaults");
    {
        TrainerConfig recommended;
        if (m_core.Brain().IsOnePass()) recommended.learningRate = kOnePassLearningRate; // the recommended rate depends on the brain type
        WriteConfigJson(w, recommended);
    }
    w.EndObject();
    Ok(resp, w);
}

void WebApi::ServeTrain(const HttpRequest& req, const std::string& action, HttpResponse& resp) {
    std::string err;
    if (action == "start") {
        m_core.StartTraining();
    } else if (action == "pause") {
        m_core.PauseTraining();
    } else if (action == "toggle") {
        m_core.ToggleTraining();
    } else if (action == "reset") {
        JsonValue body;
        if (!ParseBodyObject(req, body, &err)) {
            Fail(resp, 400, err);
            return;
        }
        double preset = 1;
        if (!body.GetNumber("preset", preset) || preset < 0 || preset > 2) {
            Fail(resp, 400, "Pick a brain size first (0 small, 1 medium, 2 large).");
            return;
        }
        double wave = 0; // optional: 0 = a normal brain (works on pixels), 1 = a wave brain
        if (const JsonValue* wv = body.Find("wave")) {
            if (wv->type != JsonValue::Type::Number) {
                Fail(resp, 400, "Pick a brain type first (0 normal, 1 wave).");
                return;
            }
            wave = wv->number;
        }
        if (wave < 0 || wave > 1 || wave != std::floor(wave)) {
            Fail(resp, 400, "Pick a brain type first (0 normal, 1 wave).");
            return;
        }
        double family = 0; // optional: 0 = the normal and wave types, 1 = the one-pass type (needs wave 0)
        if (const JsonValue* fv = body.Find("family")) {
            if (fv->type != JsonValue::Type::Number || fv->number < 0 || fv->number > 1 || fv->number != std::floor(fv->number)) {
                Fail(resp, 400, "Pick a brain family first (0 normal or wave, 1 one-pass).");
                return;
            }
            family = fv->number;
        }
        if (family == 1 && wave != 0) {
            Fail(resp, 400, "The one-pass type has no wave form.");
            return;
        }
        if (!m_core.StartOver(static_cast<uint32_t>(preset), &err, static_cast<uint32_t>(wave), static_cast<uint32_t>(family))) {
            Fail(resp, 503, err);
            return;
        }
    } else {
        Fail(resp, 404, "Unknown request.");
        return;
    }
    const AppSnapshot s = m_core.Snapshot();
    if (s.state == RunState::Problem && (action == "start" || action == "toggle")) {
        Fail(resp, 409, s.problem.empty() ? std::string("Training can't start right now.") : s.problem);
        return;
    }
    JsonWriter w;
    w.BeginObject().Field("ok", true).Field("state", s.state == RunState::Training ? "training" : "idle").EndObject();
    Ok(resp, w);
}

// ---------------------------------------------------------------------------------------------
// Pictures
// ---------------------------------------------------------------------------------------------
void WebApi::ServePictures(const HttpRequest& req, HttpResponse& resp) {
    Dataset& ds = m_core.Pictures();
    const size_t offset = static_cast<size_t>(QueryUInt(req, "offset", 0, 0, 100000000));
    const size_t limit = static_cast<size_t>(QueryUInt(req, "limit", 60, 1, 300));
    const std::vector<DatasetCardInfo> items = ds.List(offset, limit);
    JsonWriter w;
    w.BeginObject().Field("ok", true);
    w.Field("total", static_cast<uint64_t>(ds.Size()));
    w.Field("capacity", static_cast<uint64_t>(ds.Capacity()));
    w.Field("version", ds.Version());
    w.Field("offset", static_cast<uint64_t>(offset));
    w.Key("items").BeginArray();
    for (const DatasetCardInfo& it : items) {
        w.BeginObject().Field("id", it.id).Field("k", HashText(it.hash)).Field("name", it.name).EndObject();
    }
    w.EndArray().EndObject();
    Ok(resp, w);
}

void WebApi::ServePictureImage(const HttpRequest& req, uint64_t id, HttpResponse& resp) {
    Rgb8Image img;
    if (!m_core.Pictures().GetImage(id, img) || !img.Valid()) {
        Fail(resp, 404, "That picture is not in the set (it may have been removed).");
        return;
    }
    const uint32_t s = static_cast<uint32_t>(QueryUInt(req, "s", img.w, 16, img.w));
    if (s != img.w) img = ResizeRgb(img.px.data(), img.w, img.h, s, s);
    resp.SetBytes(200, kPng, PngOf(img));
    resp.cacheControl = "private, max-age=86400"; // the page adds ?k=<fingerprint>, so a changed picture never hits an old copy
}

void WebApi::ServePictureUpload(const HttpRequest& req, HttpResponse& resp) {
    const uint64_t w = QueryUInt(req, "w", 0, 0, 100000);
    const uint64_t h = QueryUInt(req, "h", 0, 0, 100000);
    if (w < 1 || h < 1 || w > 1024 || h > 1024) {
        Fail(resp, 400, "The picture size is missing or out of range (1 to 1024 pixels).");
        return;
    }
    if (req.body.size() != static_cast<size_t>(w) * static_cast<size_t>(h) * 3) {
        Fail(resp, 400, "The picture data does not have the size it announced.");
        return;
    }
    const Dataset::AddResult r = m_core.Pictures().Add(req.body.data(), static_cast<uint32_t>(w), static_cast<uint32_t>(h), req.Query("name"),
                                                       m_core.GetConfig().resize);
    JsonWriter jw;
    jw.BeginObject().Field("ok", r.status != Dataset::AddStatus::Invalid && r.status != Dataset::AddStatus::DiskError);
    jw.Field("status", AddStatusName(r.status)).Field("id", r.id).Field("message", r.message);
    jw.Field("total", static_cast<uint64_t>(m_core.Pictures().Size())).EndObject();
    Ok(resp, jw);
}

void WebApi::ServePictureFileUpload(const HttpRequest& req, HttpResponse& resp) {
    if (!m_hooks.decodeImageFile) {
        Fail(resp, 501, "This build can't open picture files on the Xbox. Use the page's own picture picker instead.");
        return;
    }
    Rgb8Image img;
    std::string err;
    if (req.body.empty() || !m_hooks.decodeImageFile(req.body.data(), req.body.size(), img, &err) || !img.Valid()) {
        JsonWriter jw;
        jw.BeginObject().Field("ok", false).Field("status", "invalid")
          .Field("error", err.empty() ? std::string("That file is not a picture the Xbox can read.") : err).EndObject();
        resp.SetText(200, kJson, jw.Take());
        return;
    }
    const Dataset::AddResult r = m_core.Pictures().Add(img.px.data(), img.w, img.h, req.Query("name"), m_core.GetConfig().resize);
    JsonWriter jw;
    jw.BeginObject().Field("ok", r.status != Dataset::AddStatus::Invalid && r.status != Dataset::AddStatus::DiskError);
    jw.Field("status", AddStatusName(r.status)).Field("id", r.id).Field("message", r.message);
    jw.Field("total", static_cast<uint64_t>(m_core.Pictures().Size())).EndObject();
    Ok(resp, jw);
}

void WebApi::ServePictureDelete(const HttpRequest& req, HttpResponse& resp) {
    Dataset& ds = m_core.Pictures();
    size_t removed = 0;
    if (req.path == "/api/pictures/clear") {
        removed = ds.Size();
        ds.Clear();
        m_core.AppendLog("All pictures were removed from the set.");
    } else {
        JsonValue body;
        std::string err;
        if (!ParseBodyObject(req, body, &err)) {
            Fail(resp, 400, err);
            return;
        }
        const JsonValue* ids = body.Find("ids");
        if (!ids || ids->type != JsonValue::Type::Array) {
            Fail(resp, 400, "Say which pictures to remove.");
            return;
        }
        std::vector<uint64_t> list;
        for (const JsonValue& v : ids->items) {
            if (v.type == JsonValue::Type::Number && v.number >= 0 && v.number < 1e15) list.push_back(static_cast<uint64_t>(v.number));
        }
        removed = ds.RemoveMany(list);
        m_core.AppendLog(std::to_string(removed) + " picture(s) removed from the set.");
    }
    JsonWriter w;
    w.BeginObject().Field("ok", true).Field("removed", static_cast<uint64_t>(removed)).Field("total", static_cast<uint64_t>(ds.Size())).EndObject();
    Ok(resp, w);
}

// ---------------------------------------------------------------------------------------------
// Sample grid and dreams
// ---------------------------------------------------------------------------------------------
void WebApi::ServePreview(HttpResponse& resp, bool image) {
    const PreviewImage pv = m_core.GetTrainer().LatestPreview();
    if (pv.version == 0 || !pv.grid.Valid()) {
        Fail(resp, 404, "No sample pictures yet. They appear a few seconds after training starts.");
        return;
    }
    if (image) {
        resp.SetBytes(200, kPng, PngOf(pv.grid));
        return;
    }
    JsonWriter w;
    w.BeginObject().Field("ok", true).Field("version", pv.version).Field("step", pv.step).Field("tiles", pv.tiles)
        .Field("columns", pv.columns).Field("res", pv.resolution).Field("dreamSteps", pv.dreamSteps).Field("average", pv.usedAverage)
        .Field("w", pv.grid.w).Field("h", pv.grid.h).EndObject();
    Ok(resp, w);
}

void WebApi::ServeDream(const HttpRequest& req, HttpResponse& resp) {
    JsonValue body;
    std::string err;
    if (!ParseBodyObject(req, body, &err)) {
        Fail(resp, 400, err);
        return;
    }
    double count = 4;
    double steps = 0;
    double variety = 1.0; // one-pass brains only: how wild the random numbers are
    body.GetNumber("count", count);
    body.GetNumber("steps", steps);
    body.GetNumber("variety", variety);
    if (!std::isfinite(variety)) variety = 1.0;
    DreamResult r;
    if (!m_core.DreamNow(static_cast<uint32_t>(std::clamp(count, 1.0, 16.0)), static_cast<uint32_t>(std::clamp(steps, 0.0, 50.0)), r,
                         static_cast<float>(std::clamp(variety, 0.0, 2.0)))) {
        Fail(resp, 503, r.error.empty() ? std::string("The brain could not make pictures right now.") : r.error);
        return;
    }
    JsonWriter w;
    w.BeginObject().Field("ok", true).Field("w", r.w).Field("h", r.h).Field("average", r.usedAverage);
    w.Key("ids").BeginArray();
    for (uint64_t id : r.ids) w.Value(id);
    w.EndArray().EndObject();
    Ok(resp, w);
}

void WebApi::ServeDreamList(HttpResponse& resp) {
    const std::vector<DreamEntry> list = m_core.ListDreams();
    JsonWriter w;
    w.BeginObject().Field("ok", true);
    w.Key("items").BeginArray();
    for (const DreamEntry& d : list) {
        w.BeginObject().Field("id", d.id).Field("w", d.w).Field("h", d.h).Field("step", d.step).Field("steps", d.dreamSteps)
            .Field("average", d.usedAverage).EndObject();
    }
    w.EndArray().EndObject();
    Ok(resp, w);
}

void WebApi::ServeDreamImage(const HttpRequest& req, uint64_t id, HttpResponse& resp) {
    DreamEntry d;
    if (!m_core.GetDream(id, d) || d.rgb.size() != static_cast<size_t>(d.w) * d.h * 3 || d.w == 0) {
        Fail(resp, 404, "That picture is no longer available (only the newest ones are kept).");
        return;
    }
    Rgb8Image img;
    img.w = d.w;
    img.h = d.h;
    img.px = std::move(d.rgb);
    uint32_t factor = static_cast<uint32_t>(std::clamp<uint64_t>(256 / std::max(1u, img.w), 1, 32));
    if (req.HasQuery("scale")) factor = static_cast<uint32_t>(QueryUInt(req, "scale", factor, 1, 32));
    while (factor > 1 && static_cast<uint64_t>(img.w) * factor > 1024) --factor;
    if (factor > 1) img = UpscaleNearest(img, factor);
    resp.SetBytes(200, kPng, PngOf(img));
    resp.cacheControl = "private, max-age=3600"; // dream ids are never reused while the app runs
}

// ---------------------------------------------------------------------------------------------
// Snapshots
// ---------------------------------------------------------------------------------------------
void WebApi::ServeSnapshotList(HttpResponse& resp) {
    const std::vector<CheckpointMeta> list = m_core.ListSnapshots();
    JsonWriter w;
    w.BeginObject().Field("ok", true).Field("version", m_core.Snapshot().snapshotListVersion);
    w.Key("items").BeginArray();
    for (const CheckpointMeta& m : list) {
        w.BeginObject().Field("id", m.id).Field("name", m.name).Field("kind", m.kind).Field("created", m.createdUnix)
            .Field("step", m.step).Field("preset", m.preset).Field("wave", m.wave).Field("family", m.family).Field("res", m.resolution).Field("loss", m.loss)
            .Field("epochs", m.epochs).Field("pictures", m.pictures).Field("ema", m.hasEma).Field("adam", m.hasAdam)
            .Field("bytes", m.fileBytes).EndObject();
    }
    w.EndArray().EndObject();
    Ok(resp, w);
}

void WebApi::ServeSnapshotCreate(const HttpRequest& req, HttpResponse& resp) {
    JsonValue body;
    std::string err;
    if (!ParseBodyObject(req, body, &err)) {
        Fail(resp, 400, err);
        return;
    }
    std::string name;
    body.GetString("name", name);
    std::string id;
    if (!m_core.SaveSnapshot(name, &id, &err)) {
        Fail(resp, 409, err);
        return;
    }
    JsonWriter w;
    w.BeginObject().Field("ok", true).Field("id", id).EndObject();
    Ok(resp, w);
}

void WebApi::ServeSnapshotAction(const HttpRequest& req, const std::string& id, const std::string& action, HttpResponse& resp) {
    if (!IsSafeCheckpointId(id)) {
        Fail(resp, 400, "That is not a valid snapshot name.");
        return;
    }
    std::string err;
    if (action == "load") {
        if (!m_core.LoadSnapshot(id, &err)) {
            Fail(resp, 409, err);
            return;
        }
    } else if (action == "rename") {
        JsonValue body;
        if (!ParseBodyObject(req, body, &err)) {
            Fail(resp, 400, err);
            return;
        }
        std::string name;
        body.GetString("name", name);
        if (!m_core.RenameSnapshot(id, name, &err)) {
            Fail(resp, 409, err);
            return;
        }
    } else if (action == "delete") {
        if (!m_core.DeleteSnapshot(id, &err)) {
            Fail(resp, 404, err.empty() ? std::string("That snapshot does not exist.") : err);
            return;
        }
    } else if (action == "download") {
        std::vector<uint8_t> bytes;
        if (!m_core.ReadSnapshotFile(id, bytes, &err)) {
            Fail(resp, 404, err.empty() ? std::string("That snapshot does not exist.") : err);
            return;
        }
        std::string stem = "snapshot";
        uint64_t step = 0;
        for (const CheckpointMeta& m : m_core.ListSnapshots()) {
            if (m.id == id) {
                stem = FileStem(m.name, "snapshot");
                step = m.step;
                break;
            }
        }
        resp.SetBytes(200, "application/octet-stream", std::move(bytes));
        resp.downloadName = stem + "_lesson" + std::to_string(step) + ".otck";
        return;
    } else {
        Fail(resp, 404, "Unknown request.");
        return;
    }
    JsonWriter w;
    w.BeginObject().Field("ok", true).EndObject();
    Ok(resp, w);
}

void WebApi::ServeSnapshotImport(const HttpRequest& req, HttpResponse& resp) {
    std::string id;
    std::string err;
    if (req.body.empty()) {
        Fail(resp, 400, "The file was empty.");
        return;
    }
    if (!m_core.ImportSnapshot(req.body.data(), req.body.size(), req.Query("name"), &id, &err)) {
        Fail(resp, 422, err.empty() ? std::string("That file is not a snapshot this app can use.") : err);
        return;
    }
    JsonWriter w;
    w.BeginObject().Field("ok", true).Field("id", id).EndObject();
    Ok(resp, w);
}

void WebApi::ServeExportBrain(HttpResponse& resp) {
    std::vector<uint8_t> bytes;
    std::string err;
    if (!m_core.ExportSnapshotNow(bytes, &err)) {
        Fail(resp, 503, err);
        return;
    }
    resp.SetBytes(200, "application/octet-stream", std::move(bytes));
    resp.downloadName = "xbox-trainer-backup_lesson" + std::to_string(m_core.Snapshot().metrics.stepCount) + ".otck";
}

} // namespace XboxOTFM
