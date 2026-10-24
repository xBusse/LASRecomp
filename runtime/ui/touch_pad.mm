// On-screen controller for iPhone / iPad (LAS_TOUCH): a left stick, A / B / X / Y, L / R, Plus / Minus and a menu
// button, drawn over the game by the overlay menu's ImGui pass (runtime/ui/menu.mm) and driven by SDL finger events.
//
// The screen is a Switch controller: its buttons are the Switch's own (hle/input.c input_touch_state), not the
// keyboard / controller bindings. Several fingers at once; a finger that slides from one button to another presses
// the new one. The layout is in fractions of the screen height, anchored to the left / right / middle, so it follows
// the shape of the device.
//
// Threads: events arrive on the main (UI) thread, the drawing happens on the render thread; what the drawing reads
// (what is held, the stick) is atomic.
#include <SDL3/SDL.h>
#include <atomic>
#include <math.h>
#include <stdint.h>
#include <string.h>

#include "imgui.h"
#include "../platform.h"

#if LAS_TOUCH

extern "C" {
void input_touch_state(uint64_t buttons, int32_t lx, int32_t ly);  // runtime/hle/input.c
void menu_touch_toggle(void);                                      // runtime/ui/menu.mm: open / close the overlay menu
}

namespace {

// hle/input.c's button bits
enum { B_A = 0, B_B, B_X, B_Y, B_L = 6, B_R, B_PLUS = 10, B_MINUS = 11, B_MENU = -1 };
enum Anchor { LEFT, RIGHT, CENTER };

struct Ctl {
    Anchor anchor;
    float dx, y, r;  // x from the anchor (left edge, right edge, middle), y from the top, radius: all in screen heights
    int bit;         // Switch button bit, or B_MENU
    const char *label;
};

// A east, B south, X north, Y west: the Switch's layout, 0.17 apart; shoulder buttons in the top corners; Minus, menu
// and Plus in a row at the top middle.
const Ctl CTLS[] = {
    {RIGHT, -0.19f, 0.66f, 0.075f, B_A, "A"},    {RIGHT, -0.36f, 0.83f, 0.075f, B_B, "B"},
    {RIGHT, -0.36f, 0.49f, 0.075f, B_X, "X"},    {RIGHT, -0.53f, 0.66f, 0.075f, B_Y, "Y"},
    {LEFT, 0.12f, 0.12f, 0.085f, B_L, "L"},      {RIGHT, -0.12f, 0.12f, 0.085f, B_R, "R"},
    {CENTER, -0.17f, 0.07f, 0.05f, B_MINUS, "-"}, {CENTER, 0.17f, 0.07f, 0.05f, B_PLUS, "+"},
    {CENTER, 0.0f, 0.07f, 0.05f, B_MENU, "="},
};
const float STICK_X = 0.38f, STICK_Y = 0.62f, STICK_R = 0.20f;  // centre from the left edge / the top, radius (heights)
// The stick is at full tilt once the finger is this far out (a share of the radius): the thumb doesn't have to reach the
// rim, nor go anywhere near the screen edge, to run. (A controller's stick reaches full tilt at its own, small, rim.)
const float STICK_FULL = 0.65f, STICK_DEAD = 0.08f;
const float HIT = 1.1f;                                         // a button's touch area against its drawn size

struct Finger {
    SDL_FingerID id;
    bool active, on_stick;
    float x, y;  // pixels
};
Finger g_f[10];

std::atomic<uint64_t> g_held{0};
std::atomic<int> g_stick_x{0}, g_stick_y{0};  // -1000..1000, screen axes (y down)
std::atomic<bool> g_menu_open{false};
float g_w = 1, g_h = 1;  // the window in pixels (main thread: set from each event)

// A game controller is connected: the on-screen controller hides (and stops reading fingers) until it is gone. Asked
// every time rather than kept from the ADDED / REMOVED events: a pad already there at launch had its ADDED event before
// this code ran. (SDL's iOS joystick list also has the accelerometer, which is not a gamepad: SDL_HasGamepad skips it.)
bool pad_connected() { return SDL_HasGamepad(); }

void ctl_xy(const Ctl &c, float w, float h, float &x, float &y) {
    x = (c.anchor == LEFT ? 0.0f : c.anchor == RIGHT ? w : w * 0.5f) + c.dx * h;
    y = c.y * h;
}

bool inside(float x, float y, float cx, float cy, float r) {
    float dx = x - cx, dy = y - cy;
    return dx * dx + dy * dy <= r * r;
}

void publish() {
    float u = g_h;
    uint64_t held = 0;
    int sx = 0, sy = 0;
    for (const Finger &f : g_f) {
        if (!f.active) continue;
        if (f.on_stick) {
            float dx = (f.x - STICK_X * u) / (STICK_R * u * STICK_FULL), dy = (f.y - STICK_Y * u) / (STICK_R * u * STICK_FULL);
            float m = sqrtf(dx * dx + dy * dy);
            if (m > 1.0f) dx /= m, dy /= m, m = 1.0f;
            if (m > STICK_DEAD) sx += (int)(dx * 1000), sy += (int)(dy * 1000);  // (a small dead zone)
            continue;
        }
        for (const Ctl &c : CTLS) {
            if (c.bit == B_MENU) continue;
            float cx, cy;
            ctl_xy(c, g_w, g_h, cx, cy);
            if (inside(f.x, f.y, cx, cy, c.r * u * HIT)) held |= 1ull << c.bit;
        }
    }
    sx = sx > 1000 ? 1000 : sx < -1000 ? -1000 : sx;
    sy = sy > 1000 ? 1000 : sy < -1000 ? -1000 : sy;
    g_held = held;
    g_stick_x = sx;
    g_stick_y = sy;
    bool menu = g_menu_open.load();
    // (the stick: up on the screen is up for the game, whose axis points the other way)
    input_touch_state(menu ? 0 : held, menu ? 0 : sx * 32767 / 1000, menu ? 0 : -sy * 32767 / 1000);
}

void release_all() {
    memset(g_f, 0, sizeof g_f);
    publish();
}

Finger *find(SDL_FingerID id) {
    for (Finger &f : g_f)
        if (f.active && f.id == id) return &f;
    return NULL;
}

bool on_menu_button(float x, float y) {
    for (const Ctl &c : CTLS) {
        if (c.bit != B_MENU) continue;
        float cx, cy;
        ctl_xy(c, g_w, g_h, cx, cy);
        return inside(x, y, cx, cy, c.r * g_h * 1.3f);  // (small and in a corner of the screen: a bigger area)
    }
    return false;
}

}  // namespace

// Main thread. true: the event was a touch on the controller (the menu and the game don't see it).
extern "C" bool touch_pad_event(const SDL_Event *e) {
    static bool had_pad;
    bool pad = pad_connected();
    if (pad != had_pad) {  // a controller came or went: let go of every finger and button the screen was holding
        had_pad = pad;
        release_all();
    }
    switch (e->type) {
    case SDL_EVENT_WINDOW_FOCUS_LOST: case SDL_EVENT_WILL_ENTER_BACKGROUND:
        release_all();  // (a finger lifted while the app wasn't looking would stay pressed)
        return false;
    case SDL_EVENT_FINGER_DOWN: case SDL_EVENT_FINGER_MOTION: case SDL_EVENT_FINGER_UP: case SDL_EVENT_FINGER_CANCELED:
        break;
    default:
        return false;
    }
    if (pad) return false;  // (the screen controller is hidden: touches are only the menu's own, as mouse events)
    SDL_Window *w = SDL_GetWindowFromEvent(e);
    int pw = 0, ph = 0;
    if (w) SDL_GetWindowSizeInPixels(w, &pw, &ph);
    if (pw > 0 && ph > 0) g_w = (float)pw, g_h = (float)ph;
    const SDL_TouchFingerEvent &t = e->tfinger;
    float x = t.x * g_w, y = t.y * g_h;  // (finger coordinates are 0..1 across the window)
    if (e->type == SDL_EVENT_FINGER_DOWN) {
        if (on_menu_button(x, y)) {  // a tap on it opens / closes the overlay menu (also with the menu up)
            menu_touch_toggle();
            release_all();
            return true;
        }
        if (g_menu_open.load()) return false;  // (the menu's own widgets: they get SDL's mouse events from this touch)
        Finger *slot = NULL;
        for (Finger &f : g_f)
            if (!f.active) { slot = &f; break; }
        if (!slot) return false;
        bool stick = inside(x, y, STICK_X * g_h, STICK_Y * g_h, STICK_R * g_h * 1.5f);  // (its base and around it)
        *slot = {t.fingerID, true, stick, x, y};
        publish();
        return true;
    }
    Finger *f = find(t.fingerID);
    if (!f) return false;
    if (e->type == SDL_EVENT_FINGER_MOTION) f->x = x, f->y = y;
    else f->active = false;
    publish();
    return true;
}

// Called by the menu when it opens or closes: the controller keeps only its menu button while it is up
extern "C" void touch_pad_set_menu_open(bool open) {
    if (g_menu_open.exchange(open) != open) input_touch_state(0, 0, 0);  // (the next touch publishes again)
}

// Render thread, inside the ImGui frame: draw it. display: the target's size in pixels.
extern "C" void touch_pad_draw(ImVec2 display, ImFont *font) {
    if (pad_connected()) return;  // (a controller is connected: the Home / Guide button or Back + Start open the menu)
    ImDrawList *dl = ImGui::GetForegroundDrawList();
    float u = display.y;
    bool menu = g_menu_open.load();
    uint64_t held = g_held.load();
    if (!menu) {
        ImVec2 c(STICK_X * u, STICK_Y * u);
        float R = STICK_R * u, kx = g_stick_x.load() / 1000.0f, ky = g_stick_y.load() / 1000.0f;
        dl->AddCircleFilled(c, R, IM_COL32(255, 255, 255, 28), 48);
        dl->AddCircle(c, R, IM_COL32(255, 255, 255, 90), 48, u * 0.004f);
        dl->AddCircleFilled(ImVec2(c.x + kx * R * STICK_FULL, c.y + ky * R * STICK_FULL), R * 0.42f, IM_COL32(255, 255, 255, kx != 0 || ky != 0 ? 120 : 70), 32);
    }
    for (const Ctl &k : CTLS) {
        if (menu && k.bit != B_MENU) continue;  // (with the menu up only its button stays, to close it)
        float x, y;
        ctl_xy(k, display.x, display.y, x, y);
        bool down = k.bit >= 0 && (held >> k.bit & 1);
        float r = k.r * u;
        dl->AddCircleFilled(ImVec2(x, y), r, down ? IM_COL32(255, 255, 255, 120) : IM_COL32(255, 255, 255, 40), 32);
        dl->AddCircle(ImVec2(x, y), r, IM_COL32(255, 255, 255, 110), 32, u * 0.004f);
        const char *label = k.bit == B_MENU && menu ? "X" : k.label;
        float fs = r * 0.95f;
        ImVec2 ts = font->CalcTextSizeA(fs, 1e9f, 0.0f, label);
        dl->AddText(font, fs, ImVec2(x - ts.x * 0.5f, y - ts.y * 0.5f), IM_COL32(255, 255, 255, down ? 255 : 190), label);
    }
}

#else  // !LAS_TOUCH: nothing (the desktop builds)

extern "C" bool touch_pad_event(const SDL_Event *) { return false; }
extern "C" void touch_pad_set_menu_open(bool) {}
extern "C" void touch_pad_draw(ImVec2, ImFont *) {}

#endif
