#pragma once
// A small, careful web server for the phone page. It runs on plain sockets (Winsock on the Xbox, the usual sockets on Linux so it can be
// tested anywhere), one short thread per request, with hard limits on everything a client could abuse.
//
// What it does NOT do: encryption (HTTPS), keep-alive, chunked uploads, compression. Every answer closes the connection, which is
// plenty fast on a home network and keeps the code simple enough to trust.

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace XboxOTFM {

struct HttpRequest {
    std::string method;                                        // upper case: GET, HEAD, POST, DELETE, OPTIONS
    std::string target;                                        // exactly as sent (path and query)
    std::string path;                                          // percent-decoded path
    std::vector<std::pair<std::string, std::string>> query;    // percent-decoded
    std::vector<std::pair<std::string, std::string>> headers;  // names in lower case
    std::vector<uint8_t> body;
    std::string remoteIp;

    // lowerName must be lower case. Returns an empty string when the header is missing.
    std::string Header(const std::string& lowerName) const;
    bool HasHeader(const std::string& lowerName) const;
    std::string Query(const std::string& name, const std::string& fallback = std::string()) const;
    bool HasQuery(const std::string& name) const;
};

struct HttpResponse {
    int status = 200;
    std::string contentType = "application/json; charset=utf-8";
    std::vector<uint8_t> body;
    std::string cacheControl = "no-store";
    std::string downloadName; // when set, the browser saves the answer as a file with this name
    std::string csp;          // Content-Security-Policy header value (only set for the web page itself)
    std::vector<std::pair<std::string, std::string>> extraHeaders;

    void SetText(int statusCode, const std::string& type, const std::string& text);
    void SetBytes(int statusCode, const std::string& type, std::vector<uint8_t> bytes);
};

class HttpServer {
public:
    struct Options {
        uint16_t port = 8080;                    // 0 = let the system pick (used by tests)
        size_t maxHeaderBytes = 16 * 1024;
        size_t defaultMaxBody = 256 * 1024;      // used when the body-limit callback is not set
        size_t totalBodyBudget = 160u * 1024u * 1024u; // bodies being received at the same moment, all connections together
        size_t maxConnections = 24;              // more than this at once get "busy, try again"
        int idleTimeoutMs = 20000;               // a client that sends nothing for this long is dropped
        int requestTimeoutMs = 900000;           // upper bound for receiving one whole request (big uploads over slow Wi-Fi)
        int sendTimeoutMs = 20000;               // a client that stops reading is dropped after this long
        bool restrictHostNames = true;           // refuse requests addressed to internet names (protects against DNS-rebinding tricks)
    };

    using Handler = std::function<void(const HttpRequest&, HttpResponse&)>;
    // Largest body the server accepts for this request (decided before the body is read).
    using BodyLimit = std::function<size_t(const std::string& method, const std::string& path)>;

    HttpServer();
    ~HttpServer();
    HttpServer(const HttpServer&) = delete;
    HttpServer& operator=(const HttpServer&) = delete;

    bool Start(const Options& options, Handler handler, BodyLimit bodyLimit, std::string* error);
    void Stop();
    bool Running() const { return m_running.load(); }
    uint16_t Port() const { return m_port.load(); }

    // Numbers for the log / status page.
    uint64_t RequestsServed() const { return m_served.load(); }
    uint32_t ActiveConnections() const { return m_active.load(); }

    // Exposed for tests.
    static bool ParsePercentEncoded(const std::string& in, bool plusIsSpace, std::string& out);
    static bool HostNameAllowed(const std::string& hostHeader);
    static const char* StatusText(int status);

private:
    void AcceptLoop();
    void ServeConnection(uintptr_t sock, std::string remoteIp);
    void ConnectionThread(uintptr_t sock, std::string remoteIp);

    Options m_options;
    Handler m_handler;
    BodyLimit m_bodyLimit;

    std::atomic<bool> m_running{ false };
    std::atomic<bool> m_stopping{ false };
    std::atomic<uint16_t> m_port{ 0 };
    uintptr_t m_listen = static_cast<uintptr_t>(-1);
    bool m_winsockStarted = false;
    std::thread m_acceptThread;

    std::atomic<uint32_t> m_active{ 0 };
    std::atomic<uint64_t> m_served{ 0 };
    std::atomic<size_t> m_bodyBytes{ 0 };
    std::mutex m_doneMutex;
    std::condition_variable m_doneCv;
};

} // namespace XboxOTFM
