// On-screen controller for iPhone / iPad (LAS_TOUCH): a left stick, A / B / X / Y, L / R, Plus / Minus and a menu
// button, drawn over the game by the overlay menu's ImGui pass (runtime/ui/menu.mm) and driven by SDL finger events.
//
// The screen is a Switch controller: its buttons are the Switch's own (hle/input.c input_touch_state), not the
// keyboard / controller bindings. Several fingers at once; a finger that slides from one button to another presses
// the new one. The layout is in fractions of the screen height, anchored to the left / right / middle, so it follows
// the shape of the device.
//
// The player's layout (Touch tab of the menu, iOS only): every control can sit somewhere else and be bigger or smaller
// (g_settings.touch_ctl, offsets in screen heights from where the layout above puts it), all of them together bigger or
// smaller (touch_size) and more or less solid (touch_opacity). "Edit layout" lets the player drag the controls
// (touch_pad_edit); the menu draws the small panel with Size / Reset / Done (runtime/ui/menu.mm touch_edit_panel).
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
extern "C" {
#include "../settings.h"
}

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
const int N_BTN = (int)(sizeof CTLS / sizeof *CTLS), STICK = N_BTN, N_ALL = N_BTN + 1;  // (the stick is control 9)
const char *const NAMES[] = {"A", "B", "X", "Y", "L", "R", "Minus", "Plus", "Menu", "Stick"};
const float STICK_X = 0.38f, STICK_Y = 0.62f, STICK_R = 0.20f;  // centre from the left edge / the top, radius (heights)
// The stick is at full tilt once the finger is this far out (a share of the radius): the thumb doesn't have to reach the
// rim, nor go anywhere near the screen edge, to run. (A controller's stick reaches full tilt at its own, small, rim.)
const float STICK_FULL = 0.65f, STICK_DEAD = 0.08f;
const float HIT = 1.1f;                                         // a button's touch area against its drawn size
const float BASE_ALPHA = 0.4f;  // touch_opacity at which the alphas below are as drawn (the default look)

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

// layout editing
std::atomic<bool> g_edit{false};
std::atomic<int> g_sel{-1};                        // the selected control (its size is the panel's slider)
std::atomic<float> g_panel[4] = {{0}, {0}, {0}, {0}};  // the panel's rectangle (pixels): touches in it are the panel's
SDL_FingerID g_drag_id;                            // main thread: the finger dragging a control
int g_drag = -1;
float g_grab_x, g_grab_y;

// A game controller is connected: the on-screen controller hides (and stops reading fingers) until it is gone. Asked
// every time rather than kept from the ADDED / REMOVED events: a pad already there at launch had its ADDED event before
// this code ran. (SDL's iOS joystick list also has the accelerometer, which is not a gamepad: SDL_HasGamepad skips it.)
bool pad_connected() { return SDL_HasGamepad(); }

float clampf(float v, float lo, float hi) { return v < lo ? lo : v > hi ? hi : v; }
float opacity_k() { return clampf(g_settings.touch_opacity, 0.1f, 1.0f) / BASE_ALPHA; }
int A(int a) { float v = a * opacity_k(); return v > 255 ? 255 : (int)v; }  // an alpha, with the player's opacity

struct Geo { float x, y, r; };  // centre and radius (pixels)

// where the layout above puts control i (before the player's offset), pixels
void base_xy(int i, float w, float h, float &x, float &y) {
    if (i == STICK) { x = STICK_X * h; y = STICK_Y * h; return; }
    const Ctl &c = CTLS[i];
    x = (c.anchor == LEFT ? 0.0f : c.anchor == RIGHT ? w : w * 0.5f) + c.dx * h;
    y = c.y * h;
}
Geo geo(int i, float w, float h) {
    const float *o = g_settings.touch_ctl[i];  // dx, dy (heights), size change
    float k = clampf(1.0f + o[2], 0.4f, 2.5f) * clampf(g_settings.touch_size, 0.4f, 2.0f);
    Geo g;
    base_xy(i, w, h, g.x, g.y);
    g.x += o[0] * h, g.y += o[1] * h;
    g.r = (i == STICK ? STICK_R : CTLS[i].r) * h * k;
    return g;
}

bool inside(float x, float y, float cx, float cy, float r) {
    float dx = x - cx, dy = y - cy;
    return dx * dx + dy * dy <= r * r;
}

void publish() {
    uint64_t held = 0;
    int sx = 0, sy = 0;
    Geo sg = geo(STICK, g_w, g_h);
    for (const Finger &f : g_f) {
        if (!f.active) continue;
        if (f.on_stick) {
            float dx = (f.x - sg.x) / (sg.r * STICK_FULL), dy = (f.y - sg.y) / (sg.r * STICK_FULL);
            float m = sqrtf(dx * dx + dy * dy);
            if (m > 1.0f) dx /= m, dy /= m, m = 1.0f;
            if (m > STICK_DEAD) sx += (int)(dx * 1000), sy += (int)(dy * 1000);  // (a small dead zone)
            continue;
        }
        for (int i = 0; i < N_BTN; i++) {
            if (CTLS[i].bit == B_MENU) continue;
            Geo g = geo(i, g_w, g_h);
            if (inside(f.x, f.y, g.x, g.y, g.r * HIT)) held |= 1ull << CTLS[i].bit;
        }
    }
    sx = sx > 1000 ? 1000 : sx < -1000 ? -1000 : sx;
    sy = sy > 1000 ? 1000 : sy < -1000 ? -1000 : sy;
    g_held = held;
    g_stick_x = sx;
    g_stick_y = sy;
    bool menu = g_menu_open.load() || g_edit.load();
    // (the stick: up on the screen is up for the game, whose axis points the other way)
    input_touch_state(menu ? 0 : held, menu ? 0 : sx * 32767 / 1000, menu ? 0 : -sy * 32767 / 1000);
}

void release_all() {
    memset(g_f, 0, sizeof g_f);
    g_drag = -1;
    publish();
}

Finger *find(SDL_FingerID id) {
    for (Finger &f : g_f)
        if (f.active && f.id == id) return &f;
    return NULL;
}

bool on_menu_button(float x, float y) {
    for (int i = 0; i < N_BTN; i++) {
        if (CTLS[i].bit != B_MENU) continue;
        Geo g = geo(i, g_w, g_h);
        return inside(x, y, g.x, g.y, g.r * 1.3f);  // (small and in a corner of the screen: a bigger area)
    }
    return false;
}

// ---- layout editing (main thread)

// the control under (x, y): the nearest centre among those whose (slightly bigger) area holds the point
int pick(float x, float y) {
    int best = -1;
    float bd = 1e30f;
    for (int i = 0; i < N_ALL; i++) {
        Geo g = geo(i, g_w, g_h);
        float dx = x - g.x, dy = y - g.y, d = dx * dx + dy * dy;
        if (d <= g.r * g.r * 1.3f * 1.3f && d / (g.r * g.r) < bd) best = i, bd = d / (g.r * g.r);
    }
    return best;
}

void move_to(int i, float cx, float cy) {  // centre (pixels) -> the control's offset, kept on the screen
    Geo g = geo(i, g_w, g_h);
    float m = g.r * 0.6f;
    cx = clampf(cx, m, g_w - m), cy = clampf(cy, m, g_h - m);
    float bx, by;
    base_xy(i, g_w, g_h, bx, by);
    g_settings.touch_ctl[i][0] = (cx - bx) / g_h;
    g_settings.touch_ctl[i][1] = (cy - by) / g_h;
}

bool in_panel(float x, float y) {
    return x >= g_panel[0].load() && x <= g_panel[2].load() && y >= g_panel[1].load() && y <= g_panel[3].load();
}

bool edit_event(const SDL_Event *e, float x, float y) {
    const SDL_TouchFingerEvent &t = e->tfinger;
    if (e->type == SDL_EVENT_FINGER_DOWN) {
        if (in_panel(x, y)) return false;  // (the panel: SDL's mouse events from this touch go to its widgets)
        int i = pick(x, y);
        if (i < 0) return true;
        g_sel = i;
        g_drag = i, g_drag_id = t.fingerID;
        Geo g = geo(i, g_w, g_h);
        g_grab_x = x - g.x, g_grab_y = y - g.y;
        return true;
    }
    bool mine = g_drag >= 0 && t.fingerID == g_drag_id;
    if (!mine) return !in_panel(x, y);
    if (e->type == SDL_EVENT_FINGER_MOTION) move_to(g_drag, x - g_grab_x, y - g_grab_y);
    else g_drag = -1;  // up / cancelled
    return true;
}

}  // namespace

// Main thread. true: the event was a touch on the controller (the menu and the game don't see it).
extern "C" bool touch_pad_event(const SDL_Event *e) {
    static bool had_pad, had_edit;
    bool edit = g_edit.load();
    bool pad = pad_connected() && !edit;
    if (pad != had_pad || edit != had_edit) {  // a controller came or went, or editing began / ended: let go of everything
        had_pad = pad, had_edit = edit;
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
    if (edit) return edit_event(e, x, y);
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
        Geo sg = geo(STICK, g_w, g_h);
        bool stick = inside(x, y, sg.x, sg.y, sg.r * 1.5f);  // (its base and around it)
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

// Whether the on-screen controller is drawn at all (the menu then renders it every frame)
extern "C" bool touch_pad_wanted(void) { return !pad_connected() || g_edit.load(); }

// ---- the menu's layout editor (runtime/ui/menu.mm)
extern "C" bool touch_pad_editing(void) { return g_edit.load(); }
extern "C" void touch_pad_edit(bool on) {
    if (on == g_edit.load()) return;
    g_sel = -1;
    g_edit.store(on);
    if (!on) settings_save();  // (the layout, opacity and size: larecomp.ini)
}
extern "C" void touch_pad_edit_panel(float x0, float y0, float x1, float y1) {
    g_panel[0] = x0, g_panel[1] = y0, g_panel[2] = x1, g_panel[3] = y1;
}
extern "C" const char *touch_pad_selected_name(void) {
    int i = g_sel.load();
    return i >= 0 && i < N_ALL ? NAMES[i] : NULL;
}
// the selected control's own size, percent of its normal size (the "all together" size is a separate setting)
extern "C" int touch_pad_selected_size(void) {
    int i = g_sel.load();
    return i >= 0 ? (int)lroundf((1.0f + g_settings.touch_ctl[i][2]) * 100) : 100;
}
extern "C" void touch_pad_set_selected_size(int pct) {
    int i = g_sel.load();
    if (i < 0) return;
    g_settings.touch_ctl[i][2] = clampf(pct / 100.0f, 0.4f, 2.5f) - 1.0f;
    move_to(i, geo(i, g_w, g_h).x, geo(i, g_w, g_h).y);  // (a bigger control near the edge stays inside it)
}
extern "C" void touch_pad_reset_selected(void) {
    int i = g_sel.load();
    if (i >= 0) memset(g_settings.touch_ctl[i], 0, sizeof g_settings.touch_ctl[i]);
}
extern "C" void touch_pad_reset_layout(void) {  // every control back (opacity and the overall size stay)
    memset(g_settings.touch_ctl, 0, sizeof g_settings.touch_ctl);
}

// Render thread, inside the ImGui frame: draw it. display: the target's size in pixels.
extern "C" void touch_pad_draw(ImVec2 display, ImFont *font) {
    bool edit = g_edit.load();
    if (pad_connected() && !edit) return;  // (a controller is connected: the Home / Guide button or Back + Start open the menu)
    ImDrawList *dl = ImGui::GetForegroundDrawList();
    float u = display.y;
    bool menu = g_menu_open.load();
    uint64_t held = g_held.load();
    int sel = g_sel.load();
    const ImU32 gold = IM_COL32(236, 194, 84, 255);
    if (!menu || edit) {
        Geo g = geo(STICK, display.x, display.y);
        ImVec2 c(g.x, g.y);
        float R = g.r, kx = g_stick_x.load() / 1000.0f, ky = g_stick_y.load() / 1000.0f;
        dl->AddCircleFilled(c, R, IM_COL32(255, 255, 255, A(28)), 48);
        dl->AddCircle(c, R, IM_COL32(255, 255, 255, A(90)), 48, u * 0.004f);
        dl->AddCircleFilled(ImVec2(c.x + kx * R * STICK_FULL, c.y + ky * R * STICK_FULL), R * 0.42f,
                            IM_COL32(255, 255, 255, A(kx != 0 || ky != 0 ? 120 : 70)), 32);
        if (edit) {  // (the stick has no label: its outline in the editor's colours)
            dl->AddCircle(c, R, sel == STICK ? gold : IM_COL32(255, 255, 255, 200), 48, u * (sel == STICK ? 0.008f : 0.003f));
            ImVec2 ts = font->CalcTextSizeA(R * 0.3f, 1e9f, 0.0f, "Stick");
            dl->AddText(font, R * 0.3f, ImVec2(c.x - ts.x * 0.5f, c.y - R - ts.y * 1.2f), IM_COL32(255, 255, 255, 230), "Stick");
        }
    }
    for (int i = 0; i < N_BTN; i++) {
        const Ctl &k = CTLS[i];
        if (menu && !edit && k.bit != B_MENU) continue;  // (with the menu up only its button stays, to close it)
        Geo g = geo(i, display.x, display.y);
        bool down = k.bit >= 0 && (held >> k.bit & 1);
        float r = g.r;
        dl->AddCircleFilled(ImVec2(g.x, g.y), r, down ? IM_COL32(255, 255, 255, A(120)) : IM_COL32(255, 255, 255, A(40)), 32);
        dl->AddCircle(ImVec2(g.x, g.y), r, IM_COL32(255, 255, 255, A(110)), 32, u * 0.004f);
        if (edit) dl->AddCircle(ImVec2(g.x, g.y), r, sel == i ? gold : IM_COL32(255, 255, 255, 200), 32, u * (sel == i ? 0.008f : 0.003f));
        const char *label = k.bit == B_MENU && menu && !edit ? "X" : k.label;
        float fs = r * 0.95f;
        ImVec2 ts = font->CalcTextSizeA(fs, 1e9f, 0.0f, label);
        dl->AddText(font, fs, ImVec2(g.x - ts.x * 0.5f, g.y - ts.y * 0.5f), IM_COL32(255, 255, 255, A(down ? 255 : 190)), label);
    }
}

#else  // !LAS_TOUCH: nothing (the desktop builds)

extern "C" bool touch_pad_event(const SDL_Event *) { return false; }
extern "C" void touch_pad_set_menu_open(bool) {}
extern "C" void touch_pad_draw(ImVec2, ImFont *) {}
extern "C" bool touch_pad_wanted(void) { return false; }
extern "C" bool touch_pad_editing(void) { return false; }
extern "C" void touch_pad_edit(bool) {}
extern "C" void touch_pad_edit_panel(float, float, float, float) {}
extern "C" const char *touch_pad_selected_name(void) { return NULL; }
extern "C" int touch_pad_selected_size(void) { return 100; }
extern "C" void touch_pad_set_selected_size(int) {}
extern "C" void touch_pad_reset_selected(void) {}
extern "C" void touch_pad_reset_layout(void) {}

#endif
