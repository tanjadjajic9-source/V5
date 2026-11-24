#include "WebHost.h"

namespace XboxOTFM {

WebHost::WebHost(AppCore& core, WebApiHooks hooks, const uint8_t* pageBytes, size_t pageSize)
    : m_core(core), m_api(core, std::move(hooks), pageBytes, pageSize) {}

WebHost::~WebHost() {
    Stop();
}

bool WebHost::TryStart(uint16_t port, std::string* error) {
    HttpServer::Options options;
    options.port = port;
    options.restrictHostNames = m_restrictHostNames;
    return m_server.Start(
        options,
        [this](const HttpRequest& req, HttpResponse& resp) { m_api.Handle(req, resp); },
        [this](const std::string& method, const std::string& path) { return m_api.BodyLimit(method, path); },
        error);
}

bool WebHost::Start(uint16_t preferredPort, std::string* error) {
    if (m_server.Running()) return true;
    m_preferredPort = preferredPort;
    std::string lastError;
    if (preferredPort == 0) {
        if (TryStart(0, &lastError)) return true;
    } else {
        // the preferred port, then the next nine
        for (unsigned i = 0; i < 10; ++i) {
            const unsigned port = static_cast<unsigned>(preferredPort) + i;
            if (port > 65535u) break;
            if (TryStart(static_cast<uint16_t>(port), &lastError)) {
                if (i > 0) m_core.AppendLog("Port " + std::to_string(preferredPort) + " was busy, so the web page uses port " + std::to_string(port) + ".");
                return true;
            }
        }
        // any free port at all
        if (TryStart(0, &lastError)) {
            m_core.AppendLog("The usual ports were busy, so the web page uses port " + std::to_string(m_server.Port()) + ".");
            return true;
        }
    }
    if (error) *error = lastError.empty() ? std::string("The web server could not start.") : lastError;
    return false;
}

void WebHost::Stop() {
    m_server.Stop();
}

bool WebHost::Restart(std::string* error) {
    const uint16_t port = m_server.Running() ? m_server.Port() : m_preferredPort;
    m_server.Stop();
    // Try the very same port first so the address on the phone stays valid.
    std::string firstError;
    if (port != 0 && TryStart(port, &firstError)) return true;
    return Start(m_preferredPort, error);
}

} // namespace XboxOTFM
