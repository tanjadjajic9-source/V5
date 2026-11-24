#include "HttpServer.h"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>
#include <cerrno>
#endif

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <exception>

namespace XboxOTFM {

namespace {

using Clock = std::chrono::steady_clock;

constexpr uintptr_t kNoSocket = static_cast<uintptr_t>(-1);
constexpr int kHeaderPhaseTimeoutMs = 15000;   // the request line and headers must arrive within this time
constexpr size_t kMaxTargetBytes = 2048;
constexpr size_t kMaxHeaderCount = 64;
constexpr size_t kMaxQueryParams = 32;

// ---------------------------------------------------------------------------------------------
// Socket helpers (the only platform-specific part)
// ---------------------------------------------------------------------------------------------
#ifdef _WIN32
void CloseSock(uintptr_t s) { closesocket(static_cast<SOCKET>(s)); }
bool WouldBlockError() {
    const int e = WSAGetLastError();
    return e == WSAEWOULDBLOCK || e == WSAEINTR;
}
void SetBlockingMode(uintptr_t s, bool blocking) {
    u_long mode = blocking ? 0 : 1;
    ioctlsocket(static_cast<SOCKET>(s), FIONBIO, &mode);
}
void ShutdownSend(uintptr_t s) { shutdown(static_cast<SOCKET>(s), SD_SEND); }
void SetSendTimeout(uintptr_t s, int ms) {
    DWORD t = static_cast<DWORD>(ms);
    setsockopt(static_cast<SOCKET>(s), SOL_SOCKET, SO_SNDTIMEO, reinterpret_cast<const char*>(&t), sizeof(t));
}
int SendBytes(uintptr_t s, const uint8_t* data, size_t len) {
    return send(static_cast<SOCKET>(s), reinterpret_cast<const char*>(data), static_cast<int>(len), 0);
}
int RecvBytes(uintptr_t s, uint8_t* data, size_t len) {
    return recv(static_cast<SOCKET>(s), reinterpret_cast<char*>(data), static_cast<int>(len), 0);
}
bool SocketUsableForSelect(uintptr_t) { return true; }
#else
void CloseSock(uintptr_t s) { ::close(static_cast<int>(s)); }
bool WouldBlockError() { return errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR; }
void SetBlockingMode(uintptr_t s, bool blocking) {
    const int fd = static_cast<int>(s);
    const int flags = fcntl(fd, F_GETFL, 0);
    if (flags < 0) return;
    fcntl(fd, F_SETFL, blocking ? (flags & ~O_NONBLOCK) : (flags | O_NONBLOCK));
}
void ShutdownSend(uintptr_t s) { ::shutdown(static_cast<int>(s), SHUT_WR); }
void SetSendTimeout(uintptr_t s, int ms) {
    timeval tv;
    tv.tv_sec = ms / 1000;
    tv.tv_usec = (ms % 1000) * 1000;
    setsockopt(static_cast<int>(s), SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
}
int SendBytes(uintptr_t s, const uint8_t* data, size_t len) {
    return static_cast<int>(::send(static_cast<int>(s), data, len, MSG_NOSIGNAL));
}
int RecvBytes(uintptr_t s, uint8_t* data, size_t len) {
    return static_cast<int>(::recv(static_cast<int>(s), data, len, 0));
}
bool SocketUsableForSelect(uintptr_t s) { return static_cast<int>(s) < FD_SETSIZE; }
#endif

// 1 = readable, 0 = timed out, -1 = error
int WaitReadable(uintptr_t s, int ms) {
    if (!SocketUsableForSelect(s)) return -1;
    fd_set rd;
    FD_ZERO(&rd);
#ifdef _WIN32
    FD_SET(static_cast<SOCKET>(s), &rd);
#else
    FD_SET(static_cast<int>(s), &rd);
#endif
    timeval tv;
    tv.tv_sec = ms / 1000;
    tv.tv_usec = (ms % 1000) * 1000;
#ifdef _WIN32
    const int r = select(0, &rd, nullptr, nullptr, &tv);
#else
    const int r = select(static_cast<int>(s) + 1, &rd, nullptr, nullptr, &tv);
#endif
    if (r < 0) return WouldBlockError() ? 0 : -1;
    return r > 0 ? 1 : 0;
}

// ---------------------------------------------------------------------------------------------
// Text helpers
// ---------------------------------------------------------------------------------------------
std::string ToLower(std::string s) {
    for (char& c : s) {
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
    }
    return s;
}

int HexValue(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

bool IsTokenChar(char c) {
    if ((c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) return true;
    return std::strchr("!#$%&'*+-.^_`|~", c) != nullptr && c != '\0';
}

std::string Trim(const std::string& s) {
    size_t a = 0;
    size_t b = s.size();
    while (a < b && (s[a] == ' ' || s[a] == '\t')) ++a;
    while (b > a && (s[b - 1] == ' ' || s[b - 1] == '\t')) --b;
    return s.substr(a, b - a);
}

// Removes anything that could break out of a header line.
std::string HeaderSafe(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (unsigned char c : s) {
        if (c == '\r' || c == '\n' || c == 0) continue;
        out += static_cast<char>(c);
    }
    return out;
}

std::string FileNameSafe(const std::string& s) {
    std::string out;
    for (unsigned char c : s) {
        const bool ok = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '.' || c == '_' || c == '-';
        out += ok ? static_cast<char>(c) : '_';
        if (out.size() >= 100) break;
    }
    if (out.empty()) out = "download";
    return out;
}

bool ParseUnsigned(const std::string& s, uint64_t maxValue, uint64_t& out) {
    if (s.empty() || s.size() > 18) return false;
    uint64_t v = 0;
    for (char c : s) {
        if (c < '0' || c > '9') return false;
        v = v * 10 + static_cast<uint64_t>(c - '0');
    }
    if (v > maxValue) return false;
    out = v;
    return true;
}

// ---------------------------------------------------------------------------------------------
// Sending
// ---------------------------------------------------------------------------------------------
bool SendAll(uintptr_t s, const uint8_t* data, size_t len, const std::atomic<bool>& stopping) {
    size_t sent = 0;
    while (sent < len) {
        if (stopping.load()) return false;
        const size_t chunk = std::min<size_t>(64 * 1024, len - sent);
        const int n = SendBytes(s, data + sent, chunk);
        if (n > 0) {
            sent += static_cast<size_t>(n);
            continue;
        }
        if (n < 0 && WouldBlockError()) continue; // interrupted; the send timeout bounds a stuck client
        return false;
    }
    return true;
}

bool SendAll(uintptr_t s, const std::string& text, const std::atomic<bool>& stopping) {
    return SendAll(s, reinterpret_cast<const uint8_t*>(text.data()), text.size(), stopping);
}

// Gives a client that is still uploading a moment to read our answer before the connection closes (closing at once could reset
// the connection and make the browser lose the answer).
void LingerClose(uintptr_t s) {
    ShutdownSend(s);
    uint8_t scratch[4096];
    const auto start = Clock::now();
    size_t drained = 0;
    while (drained < 256 * 1024 && Clock::now() - start < std::chrono::milliseconds(400)) {
        const int w = WaitReadable(s, 100);
        if (w < 0) break;
        if (w == 0) {
            if (Clock::now() - start > std::chrono::milliseconds(150)) break;
            continue;
        }
        const int n = RecvBytes(s, scratch, sizeof(scratch));
        if (n <= 0) break;
        drained += static_cast<size_t>(n);
    }
    CloseSock(s);
}

std::string BuildHead(const HttpResponse& r, size_t contentLength) {
    std::string h;
    h.reserve(400);
    h += "HTTP/1.1 " + std::to_string(r.status) + " " + HttpServer::StatusText(r.status) + "\r\n";
    h += "Content-Type: " + HeaderSafe(r.contentType) + "\r\n";
    h += "Content-Length: " + std::to_string(contentLength) + "\r\n";
    h += "Cache-Control: " + HeaderSafe(r.cacheControl) + "\r\n";
    h += "X-Content-Type-Options: nosniff\r\n";
    h += "Referrer-Policy: no-referrer\r\n";
    h += "X-Frame-Options: DENY\r\n";
    if (!r.csp.empty()) h += "Content-Security-Policy: " + HeaderSafe(r.csp) + "\r\n";
    if (!r.downloadName.empty()) h += "Content-Disposition: attachment; filename=\"" + FileNameSafe(r.downloadName) + "\"\r\n";
    for (const auto& kv : r.extraHeaders) {
        const std::string name = HeaderSafe(kv.first);
        if (name.empty()) continue;
        h += name + ": " + HeaderSafe(kv.second) + "\r\n";
    }
    h += "Connection: close\r\n\r\n";
    return h;
}

bool SendResponse(uintptr_t s, const HttpResponse& r, bool headOnly, const std::atomic<bool>& stopping) {
    const bool noBody = r.status == 204 || r.status == 304 || (r.status >= 100 && r.status < 200);
    const std::string head = BuildHead(r, noBody ? 0 : r.body.size());
    if (!SendAll(s, head, stopping)) return false;
    if (headOnly || noBody || r.body.empty()) return true;
    return SendAll(s, r.body.data(), r.body.size(), stopping);
}

void SendSimpleError(uintptr_t s, int status, const std::string& message, const std::atomic<bool>& stopping) {
    HttpResponse r;
    r.status = status;
    std::string text = "{\"ok\":false,\"error\":\"";
    for (char c : message) {
        if (c == '"' || c == '\\') text += '\\';
        if (static_cast<unsigned char>(c) < 0x20) continue;
        text += c;
    }
    text += "\"}";
    r.body.assign(text.begin(), text.end());
    SendResponse(s, r, false, stopping);
}

// ---------------------------------------------------------------------------------------------
// Reading
// ---------------------------------------------------------------------------------------------
struct Reader {
    uintptr_t sock;
    const std::atomic<bool>& stopping;
    int idleTimeoutMs;
    Clock::time_point lastData = Clock::now();
    Clock::time_point deadline = Clock::time_point::max(); // overall limit for the current phase

    // > 0 bytes read, 0 peer closed, -1 error / timeout / shutting down
    int ReadSome(uint8_t* buf, size_t cap) {
        for (;;) {
            if (stopping.load()) return -1;
            const auto now = Clock::now();
            if (now >= deadline) return -1;
            if (now - lastData >= std::chrono::milliseconds(idleTimeoutMs)) return -1;
            const int w = WaitReadable(sock, 250);
            if (w < 0) return -1;
            if (w == 0) continue;
            const int n = RecvBytes(sock, buf, cap);
            if (n > 0) {
                lastData = Clock::now();
                return n;
            }
            if (n == 0) return 0;
            if (WouldBlockError()) continue;
            return -1;
        }
    }
};

size_t FindHeaderEnd(const std::vector<uint8_t>& buf, size_t from) {
    if (buf.size() < 4) return std::string::npos;
    for (size_t i = from; i + 3 < buf.size(); ++i) {
        if (buf[i] == '\r' && buf[i + 1] == '\n' && buf[i + 2] == '\r' && buf[i + 3] == '\n') return i;
    }
    return std::string::npos;
}

struct BodyBudgetGuard {
    std::atomic<size_t>& counter;
    size_t held = 0;
    explicit BodyBudgetGuard(std::atomic<size_t>& c) : counter(c) {}
    ~BodyBudgetGuard() { Release(); }
    BodyBudgetGuard(const BodyBudgetGuard&) = delete;
    BodyBudgetGuard& operator=(const BodyBudgetGuard&) = delete;
    bool Reserve(size_t n, size_t budget) {
        const size_t before = counter.fetch_add(n);
        if (before + n > budget) {
            counter.fetch_sub(n);
            return false;
        }
        held = n;
        return true;
    }
    void Release() {
        if (held) counter.fetch_sub(held);
        held = 0;
    }
};

} // namespace

// ---------------------------------------------------------------------------------------------
// Public helpers
// ---------------------------------------------------------------------------------------------
std::string HttpRequest::Header(const std::string& lowerName) const {
    for (const auto& kv : headers) {
        if (kv.first == lowerName) return kv.second;
    }
    return std::string();
}

bool HttpRequest::HasHeader(const std::string& lowerName) const {
    for (const auto& kv : headers) {
        if (kv.first == lowerName) return true;
    }
    return false;
}

std::string HttpRequest::Query(const std::string& name, const std::string& fallback) const {
    for (const auto& kv : query) {
        if (kv.first == name) return kv.second;
    }
    return fallback;
}

bool HttpRequest::HasQuery(const std::string& name) const {
    for (const auto& kv : query) {
        if (kv.first == name) return true;
    }
    return false;
}

void HttpResponse::SetText(int statusCode, const std::string& type, const std::string& text) {
    status = statusCode;
    contentType = type;
    body.assign(text.begin(), text.end());
}

void HttpResponse::SetBytes(int statusCode, const std::string& type, std::vector<uint8_t> bytes) {
    status = statusCode;
    contentType = type;
    body = std::move(bytes);
}

const char* HttpServer::StatusText(int status) {
    switch (status) {
        case 100: return "Continue";
        case 200: return "OK";
        case 201: return "Created";
        case 204: return "No Content";
        case 304: return "Not Modified";
        case 400: return "Bad Request";
        case 403: return "Forbidden";
        case 404: return "Not Found";
        case 405: return "Method Not Allowed";
        case 408: return "Request Timeout";
        case 409: return "Conflict";
        case 411: return "Length Required";
        case 413: return "Payload Too Large";
        case 414: return "URI Too Long";
        case 415: return "Unsupported Media Type";
        case 421: return "Misdirected Request";
        case 422: return "Unprocessable Entity";
        case 431: return "Request Header Fields Too Large";
        case 500: return "Internal Server Error";
        case 501: return "Not Implemented";
        case 503: return "Service Unavailable";
        default: return "OK";
    }
}

bool HttpServer::ParsePercentEncoded(const std::string& in, bool plusIsSpace, std::string& out) {
    out.clear();
    out.reserve(in.size());
    for (size_t i = 0; i < in.size(); ++i) {
        const char c = in[i];
        if (c == '%') {
            if (i + 2 >= in.size()) return false; // needs two more characters after the percent sign
            const int hi = HexValue(in[i + 1]);
            const int lo = HexValue(in[i + 2]);
            if (hi < 0 || lo < 0) return false;
            const char decoded = static_cast<char>(hi * 16 + lo);
            if (decoded == '\0') return false;
            out += decoded;
            i += 2;
        } else if (c == '+' && plusIsSpace) {
            out += ' ';
        } else if (static_cast<unsigned char>(c) < 0x20 || c == 0x7F) {
            return false;
        } else {
            out += c;
        }
    }
    return true;
}

bool HttpServer::HostNameAllowed(const std::string& hostHeader) {
    std::string h = ToLower(Trim(hostHeader));
    if (h.empty() || h.size() > 255) return false;
    std::string name = h;
    if (h[0] == '[') {
        // IPv6 literal: [::1]:8080
        const size_t close = h.find(']');
        if (close == std::string::npos) return false;
        for (size_t i = 1; i < close; ++i) {
            const char c = h[i];
            if (!(HexValue(c) >= 0 || c == ':' || c == '.' || c == '%')) return false;
        }
        const std::string rest = h.substr(close + 1);
        if (rest.empty()) return true;
        uint64_t port = 0;
        return rest[0] == ':' && ParseUnsigned(rest.substr(1), 65535, port) && port > 0;
    }
    const size_t colon = h.rfind(':');
    if (colon != std::string::npos) {
        uint64_t port = 0;
        if (!ParseUnsigned(h.substr(colon + 1), 65535, port) || port == 0) return false;
        name = h.substr(0, colon);
    }
    if (name.empty() || name.size() > 253) return false;
    if (name.back() == '.') name.pop_back(); // "xbox." is the same name
    if (name.empty()) return false;
    for (char c : name) {
        const bool ok = (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '-' || c == '.' || c == '_';
        if (!ok) return false;
    }
    if (name.front() == '.' || name.find("..") != std::string::npos) return false;

    // IPv4 literal
    {
        int groups = 0;
        bool onlyDigitsAndDots = true;
        for (char c : name) {
            if (!((c >= '0' && c <= '9') || c == '.')) onlyDigitsAndDots = false;
        }
        if (onlyDigitsAndDots) {
            size_t pos = 0;
            while (pos <= name.size()) {
                const size_t dot = name.find('.', pos);
                const std::string part = name.substr(pos, dot == std::string::npos ? std::string::npos : dot - pos);
                uint64_t v = 0;
                if (part.empty() || part.size() > 3 || !ParseUnsigned(part, 255, v)) return false;
                ++groups;
                if (dot == std::string::npos) break;
                pos = dot + 1;
            }
            return groups == 4;
        }
    }
    if (name == "localhost") return true;
    if (name.find('.') == std::string::npos) return true; // a single word such as "xbox" can only be a name on the home network
    // names that only exist on home networks (they can never be registered on the internet)
    static const char* const kLocalSuffixes[] = { ".local", ".lan", ".home", ".internal", ".home.arpa", ".localdomain", ".localhost" };
    for (const char* suffix : kLocalSuffixes) {
        const size_t n = std::strlen(suffix);
        if (name.size() > n && name.compare(name.size() - n, n, suffix) == 0) return true;
    }
    return false;
}

// ---------------------------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------------------------
HttpServer::HttpServer() = default;

HttpServer::~HttpServer() {
    Stop();
}

bool HttpServer::Start(const Options& options, Handler handler, BodyLimit bodyLimit, std::string* error) {
    if (m_running.load()) return true;
    m_options = options;
    m_handler = std::move(handler);
    m_bodyLimit = std::move(bodyLimit);
    m_stopping.store(false);
    m_bodyBytes.store(0);

#ifdef _WIN32
    WSADATA wsa = {};
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
        if (error) *error = "The network could not be started (WSAStartup failed).";
        return false;
    }
    m_winsockStarted = true;
#endif

    auto fail = [&](const std::string& why) {
        if (m_listen != kNoSocket) {
            CloseSock(m_listen);
            m_listen = kNoSocket;
        }
#ifdef _WIN32
        if (m_winsockStarted) {
            WSACleanup();
            m_winsockStarted = false;
        }
#endif
        if (error) *error = why;
        return false;
    };

#ifdef _WIN32
    SOCKET sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (sock == INVALID_SOCKET) return fail("The web server could not create its network socket.");
    m_listen = static_cast<uintptr_t>(sock);
#else
    const int sock = ::socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) return fail("The web server could not create its network socket.");
    m_listen = static_cast<uintptr_t>(sock);
#endif

#ifndef _WIN32
    // On Linux this lets the server restart at once. On Windows the same option would let a second program share the port
    // (so "port busy" would go unnoticed), and Windows does not need it to restart.
    int reuse = 1;
    setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&reuse), sizeof(reuse));
#endif

    sockaddr_in addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(options.port);
    if (bind(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        return fail("The web server could not use port " + std::to_string(options.port) + " (is another app using it?).");
    }
    if (listen(sock, 64) != 0) return fail("The web server could not start listening.");
    SetBlockingMode(m_listen, false);

    sockaddr_in bound;
    std::memset(&bound, 0, sizeof(bound));
#ifdef _WIN32
    int boundLen = static_cast<int>(sizeof(bound));
#else
    socklen_t boundLen = sizeof(bound);
#endif
    if (getsockname(sock, reinterpret_cast<sockaddr*>(&bound), &boundLen) == 0) {
        m_port.store(ntohs(bound.sin_port));
    } else {
        m_port.store(options.port);
    }

    m_running.store(true);
    try {
        m_acceptThread = std::thread(&HttpServer::AcceptLoop, this);
    } catch (...) {
        m_running.store(false);
        return fail("The web server could not start its thread.");
    }
    return true;
}

void HttpServer::Stop() {
    if (!m_running.exchange(false)) {
        if (m_acceptThread.joinable()) m_acceptThread.join();
        return;
    }
    m_stopping.store(true);
    if (m_acceptThread.joinable()) m_acceptThread.join();
    if (m_listen != kNoSocket) {
        CloseSock(m_listen);
        m_listen = kNoSocket;
    }
    {
        // every connection thread notices m_stopping within a quarter of a second
        std::unique_lock<std::mutex> lock(m_doneMutex);
        m_doneCv.wait_for(lock, std::chrono::seconds(60), [this]() { return m_active.load() == 0; });
    }
#ifdef _WIN32
    if (m_winsockStarted) {
        WSACleanup();
        m_winsockStarted = false;
    }
#endif
    m_port.store(0);
}

void HttpServer::AcceptLoop() {
    while (!m_stopping.load()) {
        const int w = WaitReadable(m_listen, 200);
        if (w < 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            continue;
        }
        if (w == 0) continue;

        sockaddr_in peer;
        std::memset(&peer, 0, sizeof(peer));
#ifdef _WIN32
        int peerLen = static_cast<int>(sizeof(peer));
        SOCKET c = accept(static_cast<SOCKET>(m_listen), reinterpret_cast<sockaddr*>(&peer), &peerLen);
        if (c == INVALID_SOCKET) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }
        const uintptr_t client = static_cast<uintptr_t>(c);
#else
        socklen_t peerLen = sizeof(peer);
        const int c = ::accept(static_cast<int>(m_listen), reinterpret_cast<sockaddr*>(&peer), &peerLen);
        if (c < 0) {
            if (!WouldBlockError()) std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }
        const uintptr_t client = static_cast<uintptr_t>(c);
#endif
        SetBlockingMode(client, true);
        SetSendTimeout(client, m_options.sendTimeoutMs);

        char ip[64] = {};
        inet_ntop(AF_INET, &peer.sin_addr, ip, sizeof(ip));

        if (m_active.load() >= m_options.maxConnections) {
            SendSimpleError(client, 503, "The Xbox is busy with other requests. Try again in a moment.", m_stopping);
            LingerClose(client);
            continue;
        }
        m_active.fetch_add(1);
        try {
            std::thread(&HttpServer::ConnectionThread, this, client, std::string(ip)).detach();
        } catch (...) {
            m_active.fetch_sub(1);
            CloseSock(client);
        }
    }
}

void HttpServer::ConnectionThread(uintptr_t sock, std::string remoteIp) {
    try {
        ServeConnection(sock, std::move(remoteIp));
    } catch (...) {
        CloseSock(sock);
    }
    m_served.fetch_add(1);
    {
        std::lock_guard<std::mutex> lock(m_doneMutex);
        m_active.fetch_sub(1);
    }
    m_doneCv.notify_all();
}

// ---------------------------------------------------------------------------------------------
// One request
// ---------------------------------------------------------------------------------------------
void HttpServer::ServeConnection(uintptr_t sock, std::string remoteIp) {
    auto finish = [&](int status, const std::string& message) {
        SendSimpleError(sock, status, message, m_stopping);
        LingerClose(sock);
    };

    Reader reader{ sock, m_stopping, m_options.idleTimeoutMs };
    const auto started = Clock::now();

    // ---- request line and headers ----
    std::vector<uint8_t> buf;
    buf.reserve(2048);
    reader.deadline = started + std::chrono::milliseconds(kHeaderPhaseTimeoutMs);
    size_t headerEnd = std::string::npos;
    size_t scanFrom = 0;
    for (;;) {
        uint8_t chunk[4096];
        const int n = reader.ReadSome(chunk, sizeof(chunk));
        if (n <= 0) {
            if (n == 0 && buf.empty()) {
                CloseSock(sock); // the browser opened a connection and closed it (it does this to warm up)
            } else if (m_stopping.load()) {
                CloseSock(sock);
            } else {
                finish(408, "The request did not arrive in time.");
            }
            return;
        }
        buf.insert(buf.end(), chunk, chunk + n);
        headerEnd = FindHeaderEnd(buf, scanFrom > 3 ? scanFrom - 3 : 0);
        if (headerEnd != std::string::npos) break;
        scanFrom = buf.size();
        if (buf.size() > m_options.maxHeaderBytes) {
            finish(431, "The request headers are too large.");
            return;
        }
    }
    if (headerEnd > m_options.maxHeaderBytes) {
        finish(431, "The request headers are too large.");
        return;
    }

    const std::string head(reinterpret_cast<const char*>(buf.data()), headerEnd);
    HttpRequest req;
    req.remoteIp = remoteIp;

    // request line
    const size_t lineEnd = head.find("\r\n");
    const std::string requestLine = head.substr(0, lineEnd);
    {
        const size_t sp1 = requestLine.find(' ');
        const size_t sp2 = sp1 == std::string::npos ? std::string::npos : requestLine.find(' ', sp1 + 1);
        if (sp1 == std::string::npos || sp2 == std::string::npos || requestLine.find(' ', sp2 + 1) != std::string::npos) {
            finish(400, "That is not a valid web request.");
            return;
        }
        req.method = requestLine.substr(0, sp1);
        req.target = requestLine.substr(sp1 + 1, sp2 - sp1 - 1);
        const std::string version = requestLine.substr(sp2 + 1);
        if (version != "HTTP/1.1" && version != "HTTP/1.0") {
            finish(400, "Only HTTP/1.0 and HTTP/1.1 are supported.");
            return;
        }
        if (req.method.empty() || req.method.size() > 8) {
            finish(400, "That is not a valid web request.");
            return;
        }
        for (char c : req.method) {
            if (c < 'A' || c > 'Z') {
                finish(400, "That is not a valid web request.");
                return;
            }
        }
        if (req.method != "GET" && req.method != "HEAD" && req.method != "POST" && req.method != "DELETE" && req.method != "OPTIONS") {
            finish(405, "That request method is not supported.");
            return;
        }
        if (req.target.empty() || req.target[0] != '/') {
            finish(400, "That is not a valid web address.");
            return;
        }
        if (req.target.size() > kMaxTargetBytes) {
            finish(414, "The web address is too long.");
            return;
        }
        for (unsigned char c : req.target) {
            if (c <= 0x20 || c == 0x7F) {
                finish(400, "That is not a valid web address.");
                return;
            }
        }
    }

    // headers
    {
        size_t pos = lineEnd == std::string::npos ? head.size() : lineEnd + 2;
        while (pos < head.size()) {
            size_t eol = head.find("\r\n", pos);
            if (eol == std::string::npos) eol = head.size();
            const std::string line = head.substr(pos, eol - pos);
            pos = eol + 2;
            if (line.empty()) continue;
            if (line[0] == ' ' || line[0] == '\t') {
                finish(400, "Folded headers are not supported.");
                return;
            }
            const size_t colon = line.find(':');
            if (colon == std::string::npos || colon == 0) {
                finish(400, "A header line is not valid.");
                return;
            }
            const std::string rawName = line.substr(0, colon);
            for (char c : rawName) {
                if (!IsTokenChar(c)) {
                    finish(400, "A header name is not valid.");
                    return;
                }
            }
            if (req.headers.size() >= kMaxHeaderCount) {
                finish(431, "Too many headers.");
                return;
            }
            req.headers.emplace_back(ToLower(rawName), Trim(line.substr(colon + 1)));
        }
    }

    // path and query
    {
        const size_t qm = req.target.find('?');
        const std::string rawPath = req.target.substr(0, qm);
        if (!ParsePercentEncoded(rawPath, false, req.path)) {
            finish(400, "The web address contains invalid characters.");
            return;
        }
        if (qm != std::string::npos) {
            const std::string rawQuery = req.target.substr(qm + 1);
            size_t p = 0;
            while (p <= rawQuery.size() && !rawQuery.empty()) {
                size_t amp = rawQuery.find('&', p);
                if (amp == std::string::npos) amp = rawQuery.size();
                const std::string pair = rawQuery.substr(p, amp - p);
                p = amp + 1;
                if (pair.empty()) continue;
                if (req.query.size() >= kMaxQueryParams) break;
                const size_t eq = pair.find('=');
                std::string k;
                std::string v;
                if (!ParsePercentEncoded(pair.substr(0, eq), true, k) ||
                    (eq != std::string::npos && !ParsePercentEncoded(pair.substr(eq + 1), true, v))) {
                    finish(400, "The web address contains invalid characters.");
                    return;
                }
                req.query.emplace_back(std::move(k), std::move(v));
            }
        }
    }

    // who is this addressed to?
    if (m_options.restrictHostNames) {
        const std::string host = req.Header("host");
        if (!HostNameAllowed(host)) {
            finish(421, "Open this page with the address shown on your Xbox screen (it looks like http://192.168.x.x:8080).");
            return;
        }
    }

    // ---- body ----
    {
        int lengthHeaders = 0;
        uint64_t declared = 0;
        for (const auto& kv : req.headers) {
            if (kv.first == "transfer-encoding") {
                finish(501, "Chunked uploads are not supported.");
                return;
            }
            if (kv.first == "content-length") {
                uint64_t v = 0;
                if (!ParseUnsigned(kv.second, static_cast<uint64_t>(1) << 40, v)) {
                    finish(400, "The Content-Length header is not valid.");
                    return;
                }
                if (lengthHeaders > 0 && v != declared) {
                    finish(400, "The Content-Length headers disagree.");
                    return;
                }
                declared = v;
                ++lengthHeaders;
            }
        }
        // (a POST with neither Content-Length nor Transfer-Encoding simply has no body, as HTTP/1.1 says)

        const size_t limit = m_bodyLimit ? m_bodyLimit(req.method, req.path) : m_options.defaultMaxBody;
        if (declared > limit) {
            finish(413, "That upload is too large for this request.");
            return;
        }

        BodyBudgetGuard budget(m_bodyBytes);
        if (declared > 0) {
            if (!budget.Reserve(static_cast<size_t>(declared), m_options.totalBodyBudget)) {
                finish(503, "The Xbox is receiving other large uploads. Try again in a moment.");
                return;
            }
            if (ToLower(req.Header("expect")) == "100-continue") {
                SendAll(sock, std::string("HTTP/1.1 100 Continue\r\n\r\n"), m_stopping);
            }
            req.body.reserve(static_cast<size_t>(declared));
            const size_t already = std::min<size_t>(buf.size() - (headerEnd + 4), static_cast<size_t>(declared));
            req.body.insert(req.body.end(), buf.begin() + static_cast<std::ptrdiff_t>(headerEnd + 4),
                            buf.begin() + static_cast<std::ptrdiff_t>(headerEnd + 4 + already));
            buf.clear();
            buf.shrink_to_fit();

            reader.deadline = started + std::chrono::milliseconds(m_options.requestTimeoutMs);
            const auto bodyStart = Clock::now();
            std::vector<uint8_t> chunk(64 * 1024);
            while (req.body.size() < declared) {
                const size_t want = std::min<size_t>(chunk.size(), static_cast<size_t>(declared) - req.body.size());
                const int n = reader.ReadSome(chunk.data(), want);
                if (n <= 0) {
                    if (m_stopping.load()) CloseSock(sock);
                    else finish(n == 0 ? 400 : 408, "The upload was interrupted.");
                    return;
                }
                req.body.insert(req.body.end(), chunk.begin(), chunk.begin() + n);
                // a client that crawls (less than ~2 KB/s after the first 10 s) would hold memory for no reason
                const double secs = std::chrono::duration<double>(Clock::now() - bodyStart).count();
                if (secs > 10.0 && static_cast<double>(req.body.size()) / secs < 2048.0) {
                    finish(408, "The upload is too slow.");
                    return;
                }
            }
        }

        // ---- handle ----
        HttpResponse resp;
        if (req.method == "OPTIONS") {
            resp.status = 204;
            resp.extraHeaders.emplace_back("Allow", "GET, HEAD, POST, DELETE, OPTIONS");
        } else {
            try {
                m_handler(req, resp);
            } catch (const std::exception& ex) {
                resp = HttpResponse();
                resp.status = 500;
                const std::string msg = std::string("{\"ok\":false,\"error\":\"Something went wrong inside the app: ") + HeaderSafe(ex.what()) + "\"}";
                resp.body.assign(msg.begin(), msg.end());
            } catch (...) {
                resp = HttpResponse();
                resp.status = 500;
                const std::string msg = "{\"ok\":false,\"error\":\"Something went wrong inside the app.\"}";
                resp.body.assign(msg.begin(), msg.end());
            }
        }
        req.body.clear();
        req.body.shrink_to_fit();
        budget.Release(); // give the memory budget back before the (possibly slow) answer is sent

        SendResponse(sock, resp, req.method == "HEAD", m_stopping);
        LingerClose(sock);
    }
}

} // namespace XboxOTFM
