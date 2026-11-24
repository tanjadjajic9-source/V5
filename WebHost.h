#pragma once
// Runs the phone web page: one HttpServer plus the WebApi that answers its requests. No Windows code in here, so the exact same
// start-up path is exercised by the native tests.

#include "AppCore.h"
#include "HttpServer.h"
#include "WebApi.h"

#include <cstdint>
#include <string>

namespace XboxOTFM {

class WebHost {
public:
    // pageBytes must stay valid for the life of the WebHost (it is the single-file web page).
    WebHost(AppCore& core, WebApiHooks hooks, const uint8_t* pageBytes, size_t pageSize);
    ~WebHost();
    WebHost(const WebHost&) = delete;
    WebHost& operator=(const WebHost&) = delete;

    // Starts listening on preferredPort. If that port is taken, the next few ports are tried, then any free port.
    // preferredPort 0 means "any free port" (tests). Returns false (and an explanation) only if nothing worked.
    bool Start(uint16_t preferredPort, std::string* error);
    void Stop();
    // Stop and start again on the same port. Used after the app was suspended and resumed: open sockets don't survive that.
    bool Restart(std::string* error);

    bool Running() const { return m_server.Running(); }
    uint16_t Port() const { return m_server.Port(); }
    uint64_t RequestsServed() const { return m_server.RequestsServed(); }

    // Tests only: allow requests addressed to any host name.
    void SetRestrictHostNames(bool restrictNames) { m_restrictHostNames = restrictNames; }

private:
    bool TryStart(uint16_t port, std::string* error);

    AppCore& m_core;
    WebApi m_api;
    HttpServer m_server;
    uint16_t m_preferredPort = 8080;
    bool m_restrictHostNames = true;
};

} // namespace XboxOTFM
