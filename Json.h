#pragma once
// Tiny JSON writer + reader used by the phone web API. No dependencies, ASCII-only, no exceptions escape.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace XboxOTFM {

// ---------------------------------------------------------------------------------------------
// Writer: builds a JSON text; commas are inserted automatically.
// ---------------------------------------------------------------------------------------------
class JsonWriter {
public:
    JsonWriter& BeginObject() { Prefix(); m_out += '{'; m_first.push_back(true); return *this; }
    JsonWriter& EndObject() { m_out += '}'; Pop(); return *this; }
    JsonWriter& BeginArray() { Prefix(); m_out += '['; m_first.push_back(true); return *this; }
    JsonWriter& EndArray() { m_out += ']'; Pop(); return *this; }

    JsonWriter& Key(const char* key) {
        Comma();
        AppendString(key);
        m_out += ':';
        m_afterKey = true;
        return *this;
    }

    JsonWriter& Value(bool v) { Prefix(); m_out += v ? "true" : "false"; return *this; }
    JsonWriter& Value(const char* v) { Prefix(); AppendString(v ? v : ""); return *this; }
    JsonWriter& Value(const std::string& v) { Prefix(); AppendString(v.c_str()); return *this; }
    // Every built-in integer type has its own overload (int64_t / uint64_t / size_t are one of these on every compiler).
    JsonWriter& Value(int v) { return Value(static_cast<long long>(v)); }
    JsonWriter& Value(unsigned v) { return Value(static_cast<unsigned long long>(v)); }
    JsonWriter& Value(long v) { return Value(static_cast<long long>(v)); }
    JsonWriter& Value(unsigned long v) { return Value(static_cast<unsigned long long>(v)); }
    JsonWriter& Value(long long v) {
        Prefix();
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%lld", v);
        m_out += buf;
        return *this;
    }
    JsonWriter& Value(unsigned long long v) {
        Prefix();
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%llu", v);
        m_out += buf;
        return *this;
    }
    JsonWriter& Value(double v) {
        Prefix();
        if (!std::isfinite(v)) {
            m_out += "null";
        } else {
            char buf[40];
            std::snprintf(buf, sizeof(buf), "%.7g", v);
            m_out += buf;
        }
        return *this;
    }
    JsonWriter& Value(float v) { return Value(static_cast<double>(v)); }
    JsonWriter& Null() { Prefix(); m_out += "null"; return *this; }

    template <class T> JsonWriter& Field(const char* key, const T& v) { Key(key); return Value(v); }

    const std::string& str() const { return m_out; }
    std::string Take() { return std::move(m_out); }

private:
    void Comma() {
        if (!m_first.empty()) {
            if (!m_first.back()) m_out += ',';
            m_first.back() = false;
        }
    }
    void Prefix() {
        if (m_afterKey) { m_afterKey = false; return; }
        Comma();
    }
    void Pop() { if (!m_first.empty()) m_first.pop_back(); }

    void AppendString(const char* s) {
        m_out += '"';
        for (const unsigned char* p = reinterpret_cast<const unsigned char*>(s); *p; ++p) {
            const unsigned char c = *p;
            switch (c) {
                case '"': m_out += "\\\""; break;
                case '\\': m_out += "\\\\"; break;
                case '\n': m_out += "\\n"; break;
                case '\r': m_out += "\\r"; break;
                case '\t': m_out += "\\t"; break;
                default:
                    if (c < 0x20) {
                        char buf[8];
                        std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned>(c));
                        m_out += buf;
                    } else if (c >= 0x80) {
                        // Keep the output pure ASCII: bytes >= 0x80 (from file names) become '?' unless they are valid UTF-8.
                        AppendUtf8(p);
                    } else {
                        m_out += static_cast<char>(c);
                    }
            }
        }
        m_out += '"';
    }

    // Copies a UTF-8 sequence as a \uXXXX escape; invalid bytes become '?'. Advances p to the last byte consumed.
    void AppendUtf8(const unsigned char*& p) {
        const unsigned char c = *p;
        uint32_t cp = 0;
        int extra = 0;
        if ((c & 0xE0) == 0xC0) { cp = c & 0x1F; extra = 1; }
        else if ((c & 0xF0) == 0xE0) { cp = c & 0x0F; extra = 2; }
        else if ((c & 0xF8) == 0xF0) { cp = c & 0x07; extra = 3; }
        else { m_out += '?'; return; }
        const unsigned char* q = p + 1;
        for (int i = 0; i < extra; ++i, ++q) {
            if ((*q & 0xC0) != 0x80) { m_out += '?'; return; }
            cp = (cp << 6) | (*q & 0x3F);
        }
        p += extra;
        char buf[16];
        if (cp >= 0x10000) {
            cp -= 0x10000;
            std::snprintf(buf, sizeof(buf), "\\u%04x\\u%04x", 0xD800u + (cp >> 10), 0xDC00u + (cp & 0x3FF));
        } else {
            std::snprintf(buf, sizeof(buf), "\\u%04x", cp);
        }
        m_out += buf;
    }

    std::string m_out;
    std::vector<bool> m_first;
    bool m_afterKey = false;
};

// ---------------------------------------------------------------------------------------------
// Reader
// ---------------------------------------------------------------------------------------------
struct JsonValue {
    enum class Type { Null, Bool, Number, String, Array, Object };
    Type type = Type::Null;
    bool boolean = false;
    double number = 0.0;
    std::string text;
    std::vector<JsonValue> items;
    std::map<std::string, JsonValue> members;

    bool IsObject() const { return type == Type::Object; }
    const JsonValue* Find(const std::string& key) const {
        if (type != Type::Object) return nullptr;
        const auto it = members.find(key);
        return it == members.end() ? nullptr : &it->second;
    }
    bool GetNumber(const std::string& key, double& out) const {
        const JsonValue* v = Find(key);
        if (!v || v->type != Type::Number || !std::isfinite(v->number)) return false;
        out = v->number;
        return true;
    }
    bool GetBool(const std::string& key, bool& out) const {
        const JsonValue* v = Find(key);
        if (!v || v->type != Type::Bool) return false;
        out = v->boolean;
        return true;
    }
    bool GetString(const std::string& key, std::string& out) const {
        const JsonValue* v = Find(key);
        if (!v || v->type != Type::String) return false;
        out = v->text;
        return true;
    }
};

class JsonParser {
public:
    // Returns true and fills `out` on success; on failure `error` says why. Input is limited to 1 MB and 24 levels of nesting.
    static bool Parse(const std::string& text, JsonValue& out, std::string* error = nullptr) {
        JsonParser p(text);
        if (text.size() > (1u << 20)) { p.Fail("JSON text is too large"); }
        else {
            p.SkipWs();
            p.ParseValue(out, 0);
            p.SkipWs();
            if (p.m_ok && p.m_pos != text.size()) p.Fail("extra characters after the JSON value");
        }
        if (!p.m_ok && error) *error = p.m_error;
        return p.m_ok;
    }

private:
    explicit JsonParser(const std::string& t) : m_s(t) {}

    void Fail(const char* why) {
        if (m_ok) { m_ok = false; m_error = why; }
    }
    void SkipWs() {
        while (m_pos < m_s.size() && (m_s[m_pos] == ' ' || m_s[m_pos] == '\n' || m_s[m_pos] == '\r' || m_s[m_pos] == '\t')) ++m_pos;
    }
    bool Eat(char c) {
        if (m_pos < m_s.size() && m_s[m_pos] == c) { ++m_pos; return true; }
        return false;
    }
    bool EatWord(const char* w) {
        size_t n = 0;
        while (w[n]) ++n;
        if (m_s.compare(m_pos, n, w) == 0) { m_pos += n; return true; }
        return false;
    }

    void ParseValue(JsonValue& v, int depth) {
        if (!m_ok) return;
        if (depth > 24) { Fail("JSON is nested too deeply"); return; }
        if (m_pos >= m_s.size()) { Fail("unexpected end of JSON"); return; }
        const char c = m_s[m_pos];
        if (c == '{') {
            ++m_pos;
            v.type = JsonValue::Type::Object;
            SkipWs();
            if (Eat('}')) return;
            for (;;) {
                SkipWs();
                std::string key;
                if (m_pos >= m_s.size() || m_s[m_pos] != '"') { Fail("expected a quoted name"); return; }
                ParseString(key);
                SkipWs();
                if (!Eat(':')) { Fail("expected ':'"); return; }
                SkipWs();
                JsonValue child;
                ParseValue(child, depth + 1);
                if (!m_ok) return;
                v.members[key] = std::move(child);
                SkipWs();
                if (Eat(',')) continue;
                if (Eat('}')) return;
                Fail("expected ',' or '}'");
                return;
            }
        } else if (c == '[') {
            ++m_pos;
            v.type = JsonValue::Type::Array;
            SkipWs();
            if (Eat(']')) return;
            for (;;) {
                SkipWs();
                JsonValue child;
                ParseValue(child, depth + 1);
                if (!m_ok) return;
                v.items.push_back(std::move(child));
                if (v.items.size() > 100000) { Fail("JSON array is too long"); return; }
                SkipWs();
                if (Eat(',')) continue;
                if (Eat(']')) return;
                Fail("expected ',' or ']'");
                return;
            }
        } else if (c == '"') {
            v.type = JsonValue::Type::String;
            ParseString(v.text);
        } else if (EatWord("true")) {
            v.type = JsonValue::Type::Bool; v.boolean = true;
        } else if (EatWord("false")) {
            v.type = JsonValue::Type::Bool; v.boolean = false;
        } else if (EatWord("null")) {
            v.type = JsonValue::Type::Null;
        } else {
            const char* begin = m_s.c_str() + m_pos;
            char* end = nullptr;
            const double d = std::strtod(begin, &end);
            if (end == begin || !std::isfinite(d)) { Fail("unexpected character"); return; }
            v.type = JsonValue::Type::Number;
            v.number = d;
            m_pos += static_cast<size_t>(end - begin);
        }
    }

    static void AppendCodePoint(std::string& s, uint32_t cp) {
        if (cp < 0x80) s += static_cast<char>(cp);
        else if (cp < 0x800) { s += static_cast<char>(0xC0 | (cp >> 6)); s += static_cast<char>(0x80 | (cp & 0x3F)); }
        else if (cp < 0x10000) { s += static_cast<char>(0xE0 | (cp >> 12)); s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F)); s += static_cast<char>(0x80 | (cp & 0x3F)); }
        else { s += static_cast<char>(0xF0 | (cp >> 18)); s += static_cast<char>(0x80 | ((cp >> 12) & 0x3F)); s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F)); s += static_cast<char>(0x80 | (cp & 0x3F)); }
    }

    bool Hex4(uint32_t& out) {
        if (m_pos + 4 > m_s.size()) return false;
        uint32_t v = 0;
        for (int i = 0; i < 4; ++i) {
            const char h = m_s[m_pos + static_cast<size_t>(i)];
            v <<= 4;
            if (h >= '0' && h <= '9') v |= static_cast<uint32_t>(h - '0');
            else if (h >= 'a' && h <= 'f') v |= static_cast<uint32_t>(h - 'a' + 10);
            else if (h >= 'A' && h <= 'F') v |= static_cast<uint32_t>(h - 'A' + 10);
            else return false;
        }
        m_pos += 4;
        out = v;
        return true;
    }

    void ParseString(std::string& out) {
        ++m_pos; // opening quote
        out.clear();
        while (m_pos < m_s.size()) {
            const char c = m_s[m_pos++];
            if (c == '"') return;
            if (static_cast<unsigned char>(c) < 0x20) { Fail("control character in a string"); return; }
            if (c != '\\') { out += c; if (out.size() > 65536) { Fail("string is too long"); return; } continue; }
            if (m_pos >= m_s.size()) break;
            const char e = m_s[m_pos++];
            switch (e) {
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case 'u': {
                    uint32_t cp = 0;
                    if (!Hex4(cp)) { Fail("bad \\u escape"); return; }
                    if (cp >= 0xD800 && cp <= 0xDBFF) {
                        uint32_t lo = 0;
                        if (m_pos + 1 < m_s.size() && m_s[m_pos] == '\\' && m_s[m_pos + 1] == 'u') {
                            m_pos += 2;
                            if (!Hex4(lo) || lo < 0xDC00 || lo > 0xDFFF) { Fail("bad surrogate pair"); return; }
                            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                        } else { Fail("lone surrogate"); return; }
                    } else if (cp >= 0xDC00 && cp <= 0xDFFF) { Fail("lone surrogate"); return; }
                    AppendCodePoint(out, cp);
                    break;
                }
                default: Fail("bad escape in a string"); return;
            }
        }
        Fail("unterminated string");
    }

    const std::string& m_s;
    size_t m_pos = 0;
    bool m_ok = true;
    std::string m_error;
};

} // namespace XboxOTFM
