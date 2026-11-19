/*
 * Copyright 2026 Avantis Systems Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "unity/android/reprojection.h"

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>
#include <android/choreographer.h>
#include <android/looper.h>
#include <android/native_window.h>
#include <android/native_window_jni.h>
#include <sys/resource.h>
#include <sys/system_properties.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "jni_utils/android/jni_utils.h"
#include "util/logging.h"

// Threading. Four kinds of thread come through here:
//   - Unity's main thread (SetRequested, InitializeAndroid, OnHeadPose and the
//     head tracker setters);
//   - Unity's render thread, in Unity's EGL context (SetLens, AcquireSet,
//     SubmitFrame, ForgetFrames, OnRenderingStopped);
//   - the Android UI thread (the native* JNI calls, as the overlay's Surface
//     comes and goes);
//   - the warp thread started here, which owns the overlay's EGL surface and
//     a context of its own.
// Fences are made, waited on and deleted on Unity's render thread; the warp
// thread only polls them, and hands the ones it has finished with back through
// retired_fences_. So a fence that Unity's render thread is waiting on, which
// it does without the mutex, cannot be deleted under it.
// mutex_ covers the state they share. The warp thread holds it while it issues
// a refresh's draw calls, but not across the swap, so that once ForgetFrames()
// returns no more draws will name the eye buffers Unity is about to delete.
// control_mutex_ is held, on its own, for the whole of starting or stopping
// the warp thread, so that a stop has always finished, and the Surface been
// let go, before the call that asked for it returns.

namespace cardboard::unity::reprojection {
namespace {

constexpr char kJavaClass[] = "avn/cardboard/reprojection/ReprojectionSurface";

// Unity is seen to call, once a frame and in this order: the input provider
// for the pose of the update, the display provider for the next frame's
// textures (AcquireSet), the input provider again for the pose just before
// rendering, which the cameras use, and then, when the frame's draw calls
// have been issued, the display provider's submit. So the pose a frame was
// rendered with is the latest before-render pose at the time it is submitted.
//
// Self-test, chosen with `adb shell setprop debug.avn.reprojection.test <n>`
// before the app starts. The head tracker is replaced by a head that moves on
// all three axes:
//   1 - a smooth sway, and the warp reprojects every frame to one fixed
//       orientation, so with a correct reprojection the picture on the panel
//       stands perfectly still (bar the black that swings in at the edges);
//   2 - the same sway with no reprojection of it, to show what 1 is removing;
//   3 - as 1, but the head jumps between two orientations once a second, so
//       that a screenshot nearly always catches it at rest: any difference
//       between screenshots is then an error of geometry, not of timing;
//   4 - the jumps with no reprojection.
constexpr char kTestProperty[] = "debug.avn.reprojection.test";

// How far ahead of a refresh's start the head is predicted. The frame drawn in
// one refresh is latched by the compositor at the next and starts to scan out
// at the one after, two refreshes on (see Warp::DrawRefresh for keeping it
// so). The panel then takes a while longer to show it: on a CVR355128 a wearer
// found 45 ms right, so the light arrives about 19 ms after scan-out begins.
constexpr double kPredictionRefreshes = 3.25;

constexpr int64_t kNanosPerSecond = 1000000000;

constexpr char kVertexShader[] = R"glsl(#version 300 es
layout (location = 0) in vec2 a_Position;
layout (location = 1) in vec2 a_TexCoords;
// Turns a view ray of the head as it is now into one of the head as it was
// when the eye buffer was rendered.
uniform mat3 u_Rotation;
// Tangents of the eye buffer's half angles: left, right, bottom, top.
uniform vec4 u_TanFov;
out highp vec2 v_TexCoords;
out highp float v_Fade;
void main() {
  gl_Position = vec4(a_Position, 0.0, 1.0);
  // The lens mesh's texture coordinates run evenly across the eye buffer's
  // field of view in tan-angle, so they give the view ray directly.
  vec2 low = -u_TanFov.xz;
  vec2 high = u_TanFov.yw;
  vec3 ray = u_Rotation * vec3(mix(low, high, a_TexCoords), -1.0);
  vec2 tan_angle = ray.xy / max(-ray.z, 0.0001);
  v_TexCoords = (tan_angle - low) / (high - low);
  // Beyond the edge of the eye buffer there is nothing rendered to show. As
  // the ClassVR client's own reprojection does, carry the edge's colour on
  // outwards and fade it to black over half the eye buffer's width again,
  // which reads as a soft vignette where a hard black edge would catch the
  // eye.
  vec2 from_centre = abs(v_TexCoords * 2.0 - 1.0);
  v_Fade = clamp(2.0 - max(from_centre.x, from_centre.y), 0.0, 1.0);
})glsl";

// The fragment shaders: a header that defines SAMPLE(uv) to read the eye
// buffer, then this body. It is the distortion pass's scale and sharpen, with
// the picture's edge carried outwards and faded where the turned ray has left
// the eye buffer.
constexpr char kFragmentHeaderArray[] = R"glsl(#version 300 es
precision mediump float;
precision mediump sampler2DArray;
uniform sampler2DArray u_Texture;
uniform int u_Layer;
#define SAMPLE(uv) texture(u_Texture, vec3(uv, float(u_Layer))).rgb
)glsl";

constexpr char kFragmentHeader2D[] = R"glsl(#version 300 es
precision mediump float;
uniform sampler2D u_Texture;
#define SAMPLE(uv) texture(u_Texture, uv).rgb
)glsl";

constexpr char kFragmentBody[] = R"glsl(
uniform vec2 u_Scale;      // part of the texture that holds the eye
uniform vec2 u_TexelSize;  // 1 / size of that part in pixels
uniform float u_Sharpen;
uniform float u_Gamma;
in highp vec2 v_TexCoords;
in highp float v_Fade;
out vec4 o_Color;
void main() {
  // Held half a texel inside the eye, so that outside it the edge repeats
  // and nothing from the unused part of a larger texture is sampled.
  highp vec2 coords =
      clamp(v_TexCoords, 0.5 * u_TexelSize, 1.0 - 0.5 * u_TexelSize) * u_Scale;
  vec3 c = SAMPLE(coords);
  if (u_Sharpen > 0.0) {
    highp vec2 offset = vec2(0.75 * u_TexelSize.x, 0.0);
    vec3 mean = 0.5 * (SAMPLE(coords + offset) + SAMPLE(coords - offset));
    c = c + u_Sharpen * (c - mean);
  }
  c = clamp(c, 0.0, 1.0);
  if (u_Gamma != 1.0) {
    c = pow(c, vec3(u_Gamma));
  }
  o_Color = vec4(c * v_Fade, 1.0);
})glsl";

// The head-locked layer: the lens mesh again, with the view ray as it is, not
// turned, carried to where it meets the layer's rectangle.
constexpr char kLayerVertexShader[] = R"glsl(#version 300 es
layout (location = 0) in vec2 a_Position;
layout (location = 1) in vec2 a_TexCoords;
uniform vec4 u_TanFov;
// Takes the tangents of a view ray to coordinates in the layer's texture:
// scale in xy, offset in zw.
uniform vec4 u_Quad;
out highp vec2 v_LayerCoords;
void main() {
  gl_Position = vec4(a_Position, 0.0, 1.0);
  vec2 tan_angle = mix(-u_TanFov.xz, u_TanFov.yw, a_TexCoords);
  v_LayerCoords = tan_angle * u_Quad.xy + u_Quad.zw;
})glsl";

constexpr char kLayerFragmentShader[] = R"glsl(#version 300 es
precision mediump float;
uniform sampler2D u_Texture;
uniform vec4 u_Tint;
uniform bool u_Premultiplied;
uniform bool u_Linear;
in highp vec2 v_LayerCoords;
out vec4 o_Color;
void main() {
  if (any(lessThan(v_LayerCoords, vec2(0.0))) ||
      any(greaterThan(v_LayerCoords, vec2(1.0)))) {
    discard;
  }
  vec4 c = texture(u_Texture, v_LayerCoords);
  vec3 rgb = c.rgb;
  if (u_Premultiplied) {
    rgb = c.a > 0.0 ? rgb / c.a : vec3(0.0);
  }
  rgb = clamp(rgb, 0.0, 1.0);
  if (u_Linear) {
    rgb = pow(rgb, vec3(1.0 / 2.2));
  }
  float alpha = clamp(c.a * u_Tint.a, 0.0, 1.0);
  o_Color = vec4(rgb * u_Tint.rgb * alpha, alpha);
})glsl";

struct LayerProgram {
  GLuint id = 0;
  GLint tan_fov = -1;
  GLint quad = -1;
  GLint texture = -1;
  GLint tint = -1;
  GLint premultiplied = -1;
  GLint linear = -1;
};

struct Program {
  GLuint id = 0;
  GLint rotation = -1;
  GLint tan_fov = -1;
  GLint texture = -1;
  GLint layer = -1;
  GLint scale = -1;
  GLint texel_size = -1;
  GLint sharpen = -1;
  GLint gamma = -1;
};

struct Lens {
  std::vector<float> vertices;
  std::vector<float> uvs;
  std::vector<int> indices;
  float tan_fov[4] = {1.0f, 1.0f, 1.0f, 1.0f};
  // Where the middle of the head is from the eye, across and up, in metres.
  float head_offset[2] = {0.0f, 0.0f};
  // Set when the copy above is newer than the warp thread's buffers.
  bool changed = false;
};

// An eye-buffer set with a finished (or finishing) frame in it.
struct Slot {
  bool valid = false;
  Frame frame = {};
  // The head tracker's orientation the frame was rendered with.
  float orientation[4] = {0.0f, 0.0f, 0.0f, 1.0f};
  // Signalled when the GPU has finished rendering the frame. Pending only.
  GLsync fence = nullptr;
  // Counts the frames submitted, to tell one from the next.
  uint64_t serial = 0;
  // The head-locked layer to draw over the frame, if any.
  bool has_layer = false;
  HeadLockedLayer layer = {};
};

std::mutex control_mutex_;
std::mutex mutex_;

// Settings
std::atomic<bool> requested_{false};
std::atomic<int> device_setting_{-1};  // -1 none, 0 off, 1 on
std::atomic<int> test_mode_{0};

// Java side
JavaVM* vm_ = nullptr;
jclass java_class_ = nullptr;
jmethodID set_active_method_ = nullptr;
float refresh_rate_ = 72.0f;
bool active_reported_ = false;

// Lens and screen
Lens lenses_[2];
int viewport_[4] = {0, 0, 0, 0};

// Unity's EGL objects, learnt on its render thread from the first frame
EGLDisplay display_ = EGL_NO_DISPLAY;
EGLContext unity_context_ = EGL_NO_CONTEXT;
EGLConfig unity_config_ = nullptr;
// Whether Unity's own surface converts what is written to it to sRGB.
bool unity_surface_is_srgb_ = false;

// The overlay
ANativeWindow* window_ = nullptr;

// The head-locked layer Unity last asked for, and how many frames ago.
bool layer_set_ = false;
HeadLockedLayer layer_ = {};
int layer_age_ = 0;

// Frames
Slot displayed_;
Slot pending_;
// Fences the warp thread has finished with, for Unity's render thread to
// delete.
std::vector<GLsync> retired_fences_;
uint64_t frame_serial_ = 0;

// How long Unity's render thread will wait for the GPU to finish one frame
// before it submits the next. A frame that takes longer is passed over.
constexpr GLuint64 kFrameTimeoutNanos = 250000000;

// The warp thread
std::thread thread_;  // control_mutex_
std::atomic<bool> thread_started_{false};
std::atomic<bool> stop_requested_{false};
std::atomic<bool> presenting_{false};
// Set when the overlay could not be set up, so that it is tried only once.
bool failed_ = false;

// Head pose
std::mutex pose_mutex_;
float render_orientation_[4] = {0.0f, 0.0f, 0.0f, 1.0f};
bool seen_before_render_ = false;
std::mutex tracker_mutex_;
CardboardHeadTracker* head_tracker_ = nullptr;
std::atomic<int> viewport_orientation_{kLandscapeLeft};

int64_t ClockNanos(clockid_t clock) {
  timespec time;
  clock_gettime(clock, &time);
  return static_cast<int64_t>(time.tv_sec) * kNanosPerSecond + time.tv_nsec;
}

// a * b, quaternions as (x, y, z, w)
void MultiplyQuaternions(const float* a, const float* b, float* out) {
  out[0] = a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1];
  out[1] = a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0];
  out[2] = a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3];
  out[3] = a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2];
}

// The rotation matrix of a unit quaternion, in OpenGL's column order.
void QuaternionToMatrix(const float* q, float* m) {
  const float x = q[0], y = q[1], z = q[2], w = q[3];
  m[0] = 1.0f - 2.0f * (y * y + z * z);
  m[1] = 2.0f * (x * y + z * w);
  m[2] = 2.0f * (x * z - y * w);
  m[3] = 2.0f * (x * y - z * w);
  m[4] = 1.0f - 2.0f * (x * x + z * z);
  m[5] = 2.0f * (y * z + x * w);
  m[6] = 2.0f * (x * z + y * w);
  m[7] = 2.0f * (y * z - x * w);
  m[8] = 1.0f - 2.0f * (x * x + y * y);
}

// The swaying head of the self-test.
void TestOrientation(float* orientation) {
  const double t = ClockNanos(CLOCK_MONOTONIC) * 1e-9;
  const double degree = M_PI / 180.0;
  double yaw_wave = std::sin(2.0 * M_PI * 0.7 * t);
  double pitch_wave = std::sin(2.0 * M_PI * 0.9 * t);
  double roll_wave = std::sin(2.0 * M_PI * 1.1 * t);
  if (test_mode_ >= 3) {
    yaw_wave = pitch_wave = roll_wave =
        (static_cast<int64_t>(t) % 2 == 0) ? 1.0 : -1.0;
  }
  const double yaw = 6.0 * degree * yaw_wave;
  const double pitch = 4.0 * degree * pitch_wave;
  const double roll = 3.0 * degree * roll_wave;
  const float about_y[4] = {0.0f, static_cast<float>(std::sin(yaw / 2)), 0.0f,
                            static_cast<float>(std::cos(yaw / 2))};
  const float about_x[4] = {static_cast<float>(std::sin(pitch / 2)), 0.0f, 0.0f,
                            static_cast<float>(std::cos(pitch / 2))};
  const float about_z[4] = {0.0f, 0.0f, static_cast<float>(std::sin(roll / 2)),
                            static_cast<float>(std::cos(roll / 2))};
  float yaw_pitch[4];
  MultiplyQuaternions(about_y, about_x, yaw_pitch);
  MultiplyQuaternions(yaw_pitch, about_z, orientation);
}

// The head's orientation predicted for when the refresh being drawn reaches
// the eye. False when there is no head tracker to ask.
bool ReadDisplayOrientation(float* orientation) {
  std::lock_guard<std::mutex> lock(tracker_mutex_);
  if (head_tracker_ == nullptr) {
    return false;
  }
  const int64_t ahead = static_cast<int64_t>(
      kPredictionRefreshes * kNanosPerSecond / refresh_rate_);
  float position[3];
  CardboardHeadTracker_getPose(
      head_tracker_, ClockNanos(CLOCK_BOOTTIME) + ahead,
      static_cast<CardboardViewportOrientation>(viewport_orientation_.load()),
      position, orientation);
  return true;
}

// Tilt-to-spin. spin_mutex_ covers all of it.
//
// The turn is kept here, not in Unity, so that it can advance on every refresh
// of the panel and be reprojected like a turn of the head: a turn that Unity
// applied to a rig of its own would only move on with each of Unity's frames.
// It is a rotation about the vertical of the head tracker's space, applied to
// every orientation that leaves here: the one Unity renders with, which is
// stored with the frame, and the one each refresh is drawn for. The warp's
// difference between the two then takes in the turn with no more said.
std::mutex spin_mutex_;
bool spin_enabled_ = false;
// Sine of the tilt at which the turn starts, and its top speed.
double spin_threshold_ = 0.24740395925452294;  // sin(0.25)
double spin_speed_ = 4.0;  // radians a second
// How far the viewer has been turned to their right, in radians.
double spin_angle_ = 0.0;
// How fast it was turning when last advanced, radians a second.
double spin_rate_ = 0.0;
// When it was last advanced (CLOCK_MONOTONIC), or 0.
int64_t spin_time_ = 0;
// When Unity last took a pose to render with, and its frame time since.
int64_t spin_render_time_ = 0;
double spin_frame_seconds_ = 0.0;

// Longest gap the turn is carried across: no turning through a pause.
constexpr double kSpinLongestStepSeconds = 0.25;
// Furthest ahead the turn is predicted for a frame Unity is about to render.
constexpr double kSpinLongestAheadSeconds = 0.1;

// How fast a head at @p orientation (as the head tracker gives it: the
// rotation from its space to the head's) turns the viewer, in radians a
// second, to the right when positive. As avn.input.TiltToSwivel in the ClassVR
// client: nothing inside the dead zone, faster the further the head is tilted
// towards a shoulder, and slower when looking up or down, which is dizzying.
double SpinRateLocked(const float* orientation) {
  float m[9];
  QuaternionToMatrix(orientation, m);
  // The second column is the vertical as the head sees it: along its right,
  // its up and its back.
  const double sin_roll = std::sin(std::atan2(-m[3], m[4]));
  const double level = 1.0 - std::min(1.0, std::fabs(static_cast<double>(m[5])));
  if (sin_roll > spin_threshold_) {
    return spin_speed_ * level * (sin_roll - spin_threshold_);
  }
  if (sin_roll < -spin_threshold_) {
    return spin_speed_ * level * (sin_roll + spin_threshold_);
  }
  return 0.0;
}

// Carries the turn on to @p now with the head at @p orientation.
void AdvanceSpinLocked(const float* orientation, int64_t now) {
  spin_rate_ = spin_enabled_ ? SpinRateLocked(orientation) : 0.0;
  if (spin_time_ != 0 && now > spin_time_) {
    const double seconds = (now - spin_time_) * 1e-9;
    if (seconds <= kSpinLongestStepSeconds) {
      spin_angle_ = std::remainder(spin_angle_ + spin_rate_ * seconds,
                                   2.0 * M_PI);
    }
  }
  spin_time_ = now;
}

// Turns @p orientation by @p angle to the viewer's right about the vertical.
void TurnOrientation(double angle, float* orientation) {
  if (angle == 0.0) {
    return;
  }
  const float about_y[4] = {0.0f, static_cast<float>(std::sin(angle / 2)),
                            0.0f, static_cast<float>(std::cos(angle / 2))};
  float turned[4];
  MultiplyQuaternions(orientation, about_y, turned);
  std::memcpy(orientation, turned, sizeof(turned));
}

// Adds the turn to an orientation Unity is about to use. The frame will not be
// seen for a while, and is seen with the turn as it is by then, so it is
// rendered with the turn as it is expected to be: otherwise the warp would
// have to make up the difference, and pull black in at the leading edge for
// as long as the turn lasts. A wrong guess costs only that edge.
void SpinRenderOrientation(float* orientation, bool before_render) {
  std::lock_guard<std::mutex> lock(spin_mutex_);
  const int64_t now = ClockNanos(CLOCK_MONOTONIC);
  const bool presenting = presenting_;
  if (!presenting) {
    // No warp thread to carry the turn on: it moves at Unity's rate.
    AdvanceSpinLocked(orientation, now);
  }
  if (before_render) {
    if (spin_render_time_ != 0) {
      spin_frame_seconds_ = (now - spin_render_time_) * 1e-9;
    }
    spin_render_time_ = now;
  }
  double ahead = 0.0;
  if (presenting) {
    // A frame is on the panel from about one of Unity's frame times after
    // its pose is taken until two.
    ahead = std::min(kSpinLongestAheadSeconds,
                     std::max(1.0 / refresh_rate_, 1.5 * spin_frame_seconds_));
  }
  TurnOrientation(spin_angle_ + spin_rate_ * ahead, orientation);
}

// Carries the turn on to the refresh being drawn and adds it to the
// orientation the refresh is drawn for.
void SpinDisplayOrientation(float* orientation, int64_t now) {
  std::lock_guard<std::mutex> lock(spin_mutex_);
  AdvanceSpinLocked(orientation, now);
  TurnOrientation(spin_angle_, orientation);
}

// Forgets the frame waiting to be shown and deletes every fence that is done
// with. Unity's render thread only, in its context, with the mutex held.
void DropPendingLocked() {
  if (pending_.valid) {
    retired_fences_.push_back(pending_.fence);
  }
  pending_ = Slot();
  for (const GLsync fence : retired_fences_) {
    glDeleteSync(fence);
  }
  retired_fences_.clear();
}

void ReportActive(bool active) {
  if (vm_ == nullptr || java_class_ == nullptr ||
      set_active_method_ == nullptr) {
    return;
  }
  JNIEnv* env = nullptr;
  jni::LoadJNIEnv(vm_, &env);
  if (env == nullptr) {
    return;
  }
  env->CallStaticVoidMethod(java_class_, set_active_method_,
                            active ? JNI_TRUE : JNI_FALSE);
  jni::CheckExceptionInJava(env);
}

// ---------------------------------------------------------------------------
// The warp thread
// ---------------------------------------------------------------------------

class Warp {
 public:
  // Sets up EGL and GL on the calling thread. False if the overlay cannot be
  // drawn, with everything released again.
  bool Start(EGLDisplay display, EGLContext unity_context,
             EGLConfig unity_config, ANativeWindow* window) {
    display_ = display;

    // A config of our own, with no depth or stencil to pay for; Unity's if the
    // driver will not share objects between contexts of different configs.
    const EGLint config_attributes[] = {EGL_RENDERABLE_TYPE,
                                        EGL_OPENGL_ES3_BIT,
                                        EGL_SURFACE_TYPE,
                                        EGL_WINDOW_BIT,
                                        EGL_RED_SIZE,
                                        8,
                                        EGL_GREEN_SIZE,
                                        8,
                                        EGL_BLUE_SIZE,
                                        8,
                                        EGL_ALPHA_SIZE,
                                        8,
                                        EGL_DEPTH_SIZE,
                                        0,
                                        EGL_STENCIL_SIZE,
                                        0,
                                        EGL_NONE};
    EGLConfig config = nullptr;
    EGLint count = 0;
    if (!eglChooseConfig(display_, config_attributes, &config, 1, &count) ||
        count < 1) {
      config = unity_config;
    }

    // High priority, so that the GPU takes this context's small draw ahead of
    // whatever Unity has queued. A driver without the extension ignores it.
    const EGLint context_attributes[] = {EGL_CONTEXT_CLIENT_VERSION, 3,
                                         EGL_CONTEXT_PRIORITY_LEVEL_IMG,
                                         EGL_CONTEXT_PRIORITY_HIGH_IMG,
                                         EGL_NONE};
    context_ =
        eglCreateContext(display_, config, unity_context, context_attributes);
    if (context_ == EGL_NO_CONTEXT && config != unity_config) {
      config = unity_config;
      context_ =
          eglCreateContext(display_, config, unity_context, context_attributes);
    }
    if (context_ == EGL_NO_CONTEXT) {
      CARDBOARD_LOGE("Reprojection: eglCreateContext failed: 0x%x",
                     eglGetError());
      return false;
    }

    EGLint format = 0;
    if (eglGetConfigAttrib(display_, config, EGL_NATIVE_VISUAL_ID, &format)) {
      ANativeWindow_setBuffersGeometry(window, 0, 0, format);
    }
    surface_ = eglCreateWindowSurface(display_, config, window, nullptr);
    if (surface_ == EGL_NO_SURFACE) {
      CARDBOARD_LOGE("Reprojection: eglCreateWindowSurface failed: 0x%x",
                     eglGetError());
      Stop();
      return false;
    }
    if (!eglMakeCurrent(display_, surface_, surface_, context_)) {
      CARDBOARD_LOGE("Reprojection: eglMakeCurrent failed: 0x%x",
                     eglGetError());
      Stop();
      return false;
    }
    eglQuerySurface(display_, surface_, EGL_WIDTH, &width_);
    eglQuerySurface(display_, surface_, EGL_HEIGHT, &height_);

    if (!CreateProgram(&program_array_, kFragmentHeaderArray) ||
        !CreateProgram(&program_2d_, kFragmentHeader2D) ||
        !CreateLayerProgram()) {
      Stop();
      return false;
    }
    glGenVertexArrays(2, vertex_arrays_);
    glGenBuffers(2, vertex_buffers_);
    glGenBuffers(2, uv_buffers_);
    glGenBuffers(2, index_buffers_);
    {
      // The lens meshes were copied before this context existed.
      std::lock_guard<std::mutex> lock(mutex_);
      lenses_[0].changed = true;
      lenses_[1].changed = true;
    }

    EGLint priority = 0;
    eglQueryContext(display_, context_, EGL_CONTEXT_PRIORITY_LEVEL_IMG,
                    &priority);
    CARDBOARD_LOGI(
        "Reprojection: overlay %dx%d, context priority %s, %.0f Hz, test %d",
        width_, height_,
        priority == EGL_CONTEXT_PRIORITY_HIGH_IMG ? "high" : "not high",
        refresh_rate_, test_mode_.load());
    return true;
  }

  // Releases everything, leaving the Surface free for someone else.
  void Stop() {
    if (display_ == EGL_NO_DISPLAY) {
      return;
    }
    if (context_ != EGL_NO_CONTEXT && eglGetCurrentContext() == context_) {
      if (last_draw_fence_ != nullptr) {
        glDeleteSync(last_draw_fence_);
        last_draw_fence_ = nullptr;
      }
      glDeleteProgram(program_array_.id);
      glDeleteProgram(program_2d_.id);
      glDeleteProgram(program_layer_.id);
      glDeleteVertexArrays(2, vertex_arrays_);
      glDeleteBuffers(2, vertex_buffers_);
      glDeleteBuffers(2, uv_buffers_);
      glDeleteBuffers(2, index_buffers_);
    }
    eglMakeCurrent(display_, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    if (surface_ != EGL_NO_SURFACE) {
      eglDestroySurface(display_, surface_);
      surface_ = EGL_NO_SURFACE;
    }
    if (context_ != EGL_NO_CONTEXT) {
      eglDestroyContext(display_, context_);
      context_ = EGL_NO_CONTEXT;
    }
    display_ = EGL_NO_DISPLAY;
  }

  // Draws one refresh. False when the overlay has been lost.
  bool DrawRefresh() {
    const int64_t start = ClockNanos(CLOCK_MONOTONIC);
    NoteInterval(start);

    // If the last refresh's draw is still on the GPU at this vsync, it has
    // missed the compositor's latch and is still queued. The only buffer free
    // to draw into is then the one leaving the panel, which the GPU must wait
    // for, so this draw would miss the next latch too, and every refresh from
    // here on would reach the panel a refresh later than it needs to: three
    // refreshes from queue to panel instead of two. Skipping one refresh lets
    // the queue empty and the pipeline settle back. Seen on a CVR355128 in most
    // starts and after the screen comes back on; one skip puts it right, and it
    // stays right however slowly Unity renders. Never twice running, so that a
    // draw that is simply slow cannot halve the refresh rate.
    if (last_draw_fence_ != nullptr) {
      if (!skipped_last_ &&
          glClientWaitSync(last_draw_fence_, 0, 0) == GL_TIMEOUT_EXPIRED) {
        skipped_last_ = true;
        ++skipped_refreshes_;
        return true;
      }
      glDeleteSync(last_draw_fence_);
      last_draw_fence_ = nullptr;
    }
    skipped_last_ = false;

    float display_orientation[4] = {0.0f, 0.0f, 0.0f, 1.0f};
    const int test_mode = test_mode_;
    const bool have_orientation =
        test_mode == 0 && ReadDisplayOrientation(display_orientation);
    if (have_orientation) {
      SpinDisplayOrientation(display_orientation, start);
    }

    bool drew_frame = false;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      LatchPendingLocked();
      UploadLensesLocked();

      glBindFramebuffer(GL_FRAMEBUFFER, 0);
      if (viewport_[2] > 0 && viewport_[3] > 0) {
        glViewport(viewport_[0], viewport_[1], viewport_[2], viewport_[3]);
      } else {
        glViewport(0, 0, width_, height_);
      }
      glDisable(GL_SCISSOR_TEST);
      glDisable(GL_DEPTH_TEST);
      glDisable(GL_BLEND);
      glDisable(GL_CULL_FACE);
      glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
      glClear(GL_COLOR_BUFFER_BIT);

      if (displayed_.valid && index_counts_[0] > 0 && index_counts_[1] > 0) {
        // Turn a ray of the head as it will be seen into one of the head as
        // it was rendered: view(rendered) * inverse(view(displayed)).
        float rotation[9] = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};
        if (have_orientation || test_mode == 1 || test_mode == 3) {
          const float inverse_displayed[4] = {
              -display_orientation[0], -display_orientation[1],
              -display_orientation[2], display_orientation[3]};
          float delta[4];
          MultiplyQuaternions(displayed_.orientation, inverse_displayed, delta);
          QuaternionToMatrix(delta, rotation);
        }
        DrawEyesLocked(displayed_.frame, rotation);
        if (displayed_.has_layer) {
          DrawLayerLocked(displayed_.layer);
        }
        drew_frame = true;
        if (displayed_.serial != last_serial_) {
          last_serial_ = displayed_.serial;
          ++new_frames_;
        }
      }
    }

    last_draw_fence_ = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
    if (!eglSwapBuffers(display_, surface_)) {
      CARDBOARD_LOGE("Reprojection: overlay lost (0x%x)", eglGetError());
      return false;
    }
    if (drew_frame) {
      presenting_ = true;
    }
    max_draw_nanos_ =
        std::max(max_draw_nanos_, ClockNanos(CLOCK_MONOTONIC) - start);
    return true;
  }

 private:
  static GLuint CompileShader(GLenum type, const char* source) {
    const GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);
    GLint compiled = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
    if (compiled != GL_TRUE) {
      char log[512] = {0};
      glGetShaderInfoLog(shader, sizeof(log) - 1, nullptr, log);
      CARDBOARD_LOGE("Reprojection: shader failed to compile: %s", log);
      glDeleteShader(shader);
      return 0;
    }
    return shader;
  }

  // Compiles and links a program. Zero if it cannot be made.
  static GLuint LinkProgram(const char* vertex_source,
                            const char* fragment_source) {
    const GLuint vertex = CompileShader(GL_VERTEX_SHADER, vertex_source);
    const GLuint fragment = CompileShader(GL_FRAGMENT_SHADER, fragment_source);
    if (vertex == 0 || fragment == 0) {
      glDeleteShader(vertex);
      glDeleteShader(fragment);
      return 0;
    }
    const GLuint id = glCreateProgram();
    glAttachShader(id, vertex);
    glAttachShader(id, fragment);
    glLinkProgram(id);
    glDeleteShader(vertex);
    glDeleteShader(fragment);
    GLint linked = GL_FALSE;
    glGetProgramiv(id, GL_LINK_STATUS, &linked);
    if (linked != GL_TRUE) {
      CARDBOARD_LOGE("Reprojection: program failed to link");
      glDeleteProgram(id);
      return 0;
    }
    return id;
  }

  bool CreateLayerProgram() {
    const GLuint id = LinkProgram(kLayerVertexShader, kLayerFragmentShader);
    if (id == 0) {
      return false;
    }
    program_layer_.id = id;
    program_layer_.tan_fov = glGetUniformLocation(id, "u_TanFov");
    program_layer_.quad = glGetUniformLocation(id, "u_Quad");
    program_layer_.texture = glGetUniformLocation(id, "u_Texture");
    program_layer_.tint = glGetUniformLocation(id, "u_Tint");
    program_layer_.premultiplied = glGetUniformLocation(id, "u_Premultiplied");
    program_layer_.linear = glGetUniformLocation(id, "u_Linear");
    return true;
  }

  static bool CreateProgram(Program* program, const char* fragment_header) {
    const std::string fragment_source =
        std::string(fragment_header) + kFragmentBody;
    const GLuint id = LinkProgram(kVertexShader, fragment_source.c_str());
    if (id == 0) {
      return false;
    }
    program->id = id;
    program->rotation = glGetUniformLocation(id, "u_Rotation");
    program->tan_fov = glGetUniformLocation(id, "u_TanFov");
    program->texture = glGetUniformLocation(id, "u_Texture");
    program->layer = glGetUniformLocation(id, "u_Layer");
    program->scale = glGetUniformLocation(id, "u_Scale");
    program->texel_size = glGetUniformLocation(id, "u_TexelSize");
    program->sharpen = glGetUniformLocation(id, "u_Sharpen");
    program->gamma = glGetUniformLocation(id, "u_Gamma");
    return true;
  }

  // Puts the pending frame on the panel once the GPU has finished it.
  void LatchPendingLocked() {
    ++refreshes_;
    if (!pending_.valid) {
      return;
    }
    const GLenum state = glClientWaitSync(pending_.fence, 0, 0);
    if (state == GL_TIMEOUT_EXPIRED) {
      ++refreshes_waiting_;
      return;
    }
    retired_fences_.push_back(pending_.fence);
    displayed_ = pending_;
    displayed_.fence = nullptr;
    pending_ = Slot();
  }

  void UploadLensesLocked() {
    for (int eye = 0; eye < 2; ++eye) {
      Lens& lens = lenses_[eye];
      if (!lens.changed) {
        continue;
      }
      lens.changed = false;
      glBindVertexArray(vertex_arrays_[eye]);
      glBindBuffer(GL_ARRAY_BUFFER, vertex_buffers_[eye]);
      glBufferData(GL_ARRAY_BUFFER, lens.vertices.size() * sizeof(float),
                   lens.vertices.data(), GL_STATIC_DRAW);
      glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
      glEnableVertexAttribArray(0);
      glBindBuffer(GL_ARRAY_BUFFER, uv_buffers_[eye]);
      glBufferData(GL_ARRAY_BUFFER, lens.uvs.size() * sizeof(float),
                   lens.uvs.data(), GL_STATIC_DRAW);
      glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
      glEnableVertexAttribArray(1);
      glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffers_[eye]);
      glBufferData(GL_ELEMENT_ARRAY_BUFFER, lens.indices.size() * sizeof(int),
                   lens.indices.data(), GL_STATIC_DRAW);
      glBindVertexArray(0);
      index_counts_[eye] = static_cast<GLsizei>(lens.indices.size());
      std::memcpy(tan_fov_[eye], lens.tan_fov, sizeof(tan_fov_[eye]));
    }
  }

  // The rectangle of the window, in pixels, that the layer covers in an eye:
  // every triangle of the lens mesh that the layer touches. False if none.
  bool LayerScissorLocked(int eye, const float* quad, GLint* box) const {
    const Lens& lens = lenses_[eye];
    const float low[2] = {-lens.tan_fov[0], -lens.tan_fov[2]};
    const float high[2] = {lens.tan_fov[1], lens.tan_fov[3]};
    const size_t count = lens.indices.size();
    float min_x = 2.0f, min_y = 2.0f, max_x = -2.0f, max_y = -2.0f;
    for (size_t i = 0; i + 2 < count; ++i) {
      // The mesh is a triangle strip: each index with the two after it.
      float u_min = 2.0f, v_min = 2.0f, u_max = -1.0f, v_max = -1.0f;
      for (size_t corner = 0; corner < 3; ++corner) {
        const size_t vertex = static_cast<size_t>(lens.indices[i + corner]);
        const float tan_x =
            low[0] + (high[0] - low[0]) * lens.uvs[vertex * 2];
        const float tan_y =
            low[1] + (high[1] - low[1]) * lens.uvs[vertex * 2 + 1];
        const float u = tan_x * quad[0] + quad[2];
        const float v = tan_y * quad[1] + quad[3];
        u_min = std::min(u_min, u);
        u_max = std::max(u_max, u);
        v_min = std::min(v_min, v);
        v_max = std::max(v_max, v);
      }
      if (u_max < 0.0f || u_min > 1.0f || v_max < 0.0f || v_min > 1.0f) {
        continue;
      }
      for (size_t corner = 0; corner < 3; ++corner) {
        const size_t vertex = static_cast<size_t>(lens.indices[i + corner]);
        min_x = std::min(min_x, lens.vertices[vertex * 2]);
        max_x = std::max(max_x, lens.vertices[vertex * 2]);
        min_y = std::min(min_y, lens.vertices[vertex * 2 + 1]);
        max_y = std::max(max_y, lens.vertices[vertex * 2 + 1]);
      }
    }
    if (min_x > max_x || min_y > max_y) {
      return false;
    }
    GLint view[4] = {viewport_[0], viewport_[1], viewport_[2], viewport_[3]};
    if (view[2] <= 0 || view[3] <= 0) {
      view[0] = 0;
      view[1] = 0;
      view[2] = width_;
      view[3] = height_;
    }
    const float left = view[0] + (min_x * 0.5f + 0.5f) * view[2];
    const float right = view[0] + (max_x * 0.5f + 0.5f) * view[2];
    const float bottom = view[1] + (min_y * 0.5f + 0.5f) * view[3];
    const float top = view[1] + (max_y * 0.5f + 0.5f) * view[3];
    box[0] = static_cast<GLint>(std::floor(left)) - 1;
    box[1] = static_cast<GLint>(std::floor(bottom)) - 1;
    box[2] = static_cast<GLint>(std::ceil(right)) + 1 - box[0];
    box[3] = static_cast<GLint>(std::ceil(top)) + 1 - box[1];
    return true;
  }

  // Draws the head-locked layer over both eyes. It is not turned with the
  // head: wherever the head is by now, the layer is straight ahead of it.
  void DrawLayerLocked(const HeadLockedLayer& layer) {
    if (layer.distance <= 0.0f || layer.width <= 0.0f ||
        layer.height <= 0.0f || layer.tint[3] <= 0.0f ||
        glIsTexture(layer.texture) != GL_TRUE) {
      return;
    }
    glActiveTexture(GL_TEXTURE0);
    for (int i = 0; i < 8 && glGetError() != GL_NO_ERROR; ++i) {
    }
    glBindTexture(GL_TEXTURE_2D, layer.texture);
    if (glGetError() != GL_NO_ERROR) {
      // Not a 2D texture: the name has gone to something else.
      glBindTexture(GL_TEXTURE_2D, 0);
      return;
    }

    glUseProgram(program_layer_.id);
    glUniform1i(program_layer_.texture, 0);
    glUniform4fv(program_layer_.tint, 1, layer.tint);
    glUniform1i(program_layer_.premultiplied, layer.premultiplied_alpha ? 1 : 0);
    glUniform1i(program_layer_.linear, layer.linear ? 1 : 0);
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_SCISSOR_TEST);
    for (int eye = 0; eye < 2; ++eye) {
      // A view ray with tangents t meets the layer's plane at t * distance
      // in the eye's space. The middle of the head, which the rectangle is
      // centred on, is at head_offset there, and the texture runs from one
      // edge of the rectangle to the other.
      const Lens& lens = lenses_[eye];
      const float quad[4] = {
          layer.distance / layer.width, layer.distance / layer.height,
          0.5f - lens.head_offset[0] / layer.width,
          0.5f - lens.head_offset[1] / layer.height};
      GLint box[4];
      if (!LayerScissorLocked(eye, quad, box)) {
        continue;
      }
      glScissor(box[0], box[1], box[2], box[3]);
      glUniform4fv(program_layer_.quad, 1, quad);
      glUniform4fv(program_layer_.tan_fov, 1, tan_fov_[eye]);
      glBindVertexArray(vertex_arrays_[eye]);
      glDrawElements(GL_TRIANGLE_STRIP, index_counts_[eye], GL_UNSIGNED_INT,
                     nullptr);
    }
    glBindVertexArray(0);
    glBindTexture(GL_TEXTURE_2D, 0);
    glDisable(GL_SCISSOR_TEST);
    glDisable(GL_BLEND);
  }

  void DrawEyesLocked(const Frame& frame, const float* rotation) {
    const Program& program = frame.is_texture_array ? program_array_ : program_2d_;
    const GLenum target =
        frame.is_texture_array ? GL_TEXTURE_2D_ARRAY : GL_TEXTURE_2D;
    // Unity's distortion pass raises the single-pass picture to the power 2.2
    // and leaves the multi-pass one alone, and Unity's surface then encodes
    // either as sRGB if it is an sRGB surface. The overlay is not one, so do
    // here whatever the two together would have done.
    float gamma;
    if (frame.is_texture_array) {
      gamma = unity_surface_is_srgb_ ? 1.0f : 2.2f;
    } else {
      gamma = unity_surface_is_srgb_ ? 1.0f / 2.2f : 1.0f;
    }

    glUseProgram(program.id);
    glUniformMatrix3fv(program.rotation, 1, GL_FALSE, rotation);
    glUniform1i(program.texture, 0);
    glUniform2f(program.scale, frame.used_u, frame.used_v);
    glUniform2f(program.texel_size, 1.0f / static_cast<float>(frame.width),
                1.0f / static_cast<float>(frame.height));
    glUniform1f(program.sharpen, frame.sharpen);
    glUniform1f(program.gamma, gamma);
    glActiveTexture(GL_TEXTURE0);
    for (int eye = 0; eye < 2; ++eye) {
      glBindTexture(target, frame.textures[eye]);
      if (program.layer >= 0) {
        glUniform1i(program.layer, eye);
      }
      glUniform4fv(program.tan_fov, 1, tan_fov_[eye]);
      glBindVertexArray(vertex_arrays_[eye]);
      glDrawElements(GL_TRIANGLE_STRIP, index_counts_[eye], GL_UNSIGNED_INT,
                     nullptr);
    }
    glBindVertexArray(0);
    glBindTexture(target, 0);
  }

  // Counts the refreshes that started late and reports every few seconds.
  void NoteInterval(int64_t now) {
    const int64_t period =
        static_cast<int64_t>(kNanosPerSecond / refresh_rate_);
    if (last_refresh_nanos_ != 0 && now - last_refresh_nanos_ > period * 3 / 2) {
      ++late_refreshes_;
    }
    last_refresh_nanos_ = now;
    if (report_start_nanos_ == 0) {
      report_start_nanos_ = now;
    } else if (now - report_start_nanos_ >= 5 * kNanosPerSecond) {
      const double seconds = (now - report_start_nanos_) * 1e-9;
      CARDBOARD_LOGI(
          "Reprojection: %.1f refreshes/s (%d late, %d skipped to catch up), "
          "%.1f new frames/s, %d refreshes waited on the GPU, slowest draw "
          "%.1f ms",
          refreshes_ / seconds, late_refreshes_, skipped_refreshes_,
          new_frames_ / seconds, refreshes_waiting_, max_draw_nanos_ * 1e-6);
      report_start_nanos_ = now;
      skipped_refreshes_ = 0;
      refreshes_ = 0;
      late_refreshes_ = 0;
      new_frames_ = 0;
      refreshes_waiting_ = 0;
      max_draw_nanos_ = 0;
    }
  }

  EGLDisplay display_ = EGL_NO_DISPLAY;
  EGLContext context_ = EGL_NO_CONTEXT;
  EGLSurface surface_ = EGL_NO_SURFACE;
  EGLint width_ = 0;
  EGLint height_ = 0;

  Program program_array_;
  Program program_2d_;
  LayerProgram program_layer_;
  GLuint vertex_arrays_[2] = {0, 0};
  GLuint vertex_buffers_[2] = {0, 0};
  GLuint uv_buffers_[2] = {0, 0};
  GLuint index_buffers_[2] = {0, 0};
  GLsizei index_counts_[2] = {0, 0};
  float tan_fov_[2][4] = {{1.0f, 1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f, 1.0f}};

  // The last refresh's draw, to tell whether it made the compositor's latch
  GLsync last_draw_fence_ = nullptr;
  bool skipped_last_ = false;

  // Statistics
  int64_t last_refresh_nanos_ = 0;
  int64_t report_start_nanos_ = 0;
  int64_t max_draw_nanos_ = 0;
  int refreshes_ = 0;
  int late_refreshes_ = 0;
  int new_frames_ = 0;
  int refreshes_waiting_ = 0;
  int skipped_refreshes_ = 0;
  uint64_t last_serial_ = 0;
};

Warp* warp_ = nullptr;  // warp thread only
bool warp_lost_ = false;  // warp thread only

void OnRefresh(long /*frame_time_nanos*/, void* /*data*/) {  // NOLINT
  if (stop_requested_ || warp_lost_) {
    return;
  }
  AChoreographer_postFrameCallback(AChoreographer_getInstance(), OnRefresh,
                                   nullptr);
  if (!warp_->DrawRefresh()) {
    warp_lost_ = true;
  }
}

void WarpThreadMain(EGLDisplay display, EGLContext unity_context,
                    EGLConfig unity_config, ANativeWindow* window) {
  // As urgent as the system's own display threads, where the app may ask.
  setpriority(PRIO_PROCESS, static_cast<id_t>(gettid()), -8);

  Warp warp;
  if (!warp.Start(display, unity_context, unity_config, window)) {
    std::lock_guard<std::mutex> lock(mutex_);
    failed_ = true;
    return;
  }
  warp_ = &warp;
  warp_lost_ = false;

  ALooper_prepare(0);
  AChoreographer_postFrameCallback(AChoreographer_getInstance(), OnRefresh,
                                   nullptr);
  while (!stop_requested_ && !warp_lost_) {
    ALooper_pollOnce(50, nullptr, nullptr, nullptr);
  }

  presenting_ = false;
  warp_ = nullptr;
  warp.Stop();
  if (warp_lost_) {
    std::lock_guard<std::mutex> lock(mutex_);
    failed_ = true;
  }
}

// Stops the warp thread and waits for it to let go of the Surface. Requires
// control_mutex_, and mutex_ not to be held.
void StopThreadControlled() {
  if (thread_.joinable()) {
    stop_requested_ = true;
    thread_.join();
    stop_requested_ = false;
  }
  thread_started_ = false;
  presenting_ = false;
}

// Starts the warp thread if everything it needs is there. Requires
// control_mutex_, and mutex_ not to be held.
void StartThreadControlled() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (thread_started_ || failed_ || window_ == nullptr ||
      unity_context_ == EGL_NO_CONTEXT) {
    return;
  }
  thread_started_ = true;
  thread_ = std::thread(WarpThreadMain, display_, unity_context_, unity_config_,
                        window_);
}

}  // namespace

void SetRequested(bool requested) { requested_ = requested; }

void InitializeAndroid(JavaVM* vm, jobject activity) {
  JNIEnv* env = nullptr;
  jni::LoadJNIEnv(vm, &env);
  if (env == nullptr) {
    return;
  }

  char value[PROP_VALUE_MAX] = {0};
  if (__system_property_get(kTestProperty, value) > 0) {
    test_mode_ = std::atoi(value);
  }

  jclass java_class = nullptr;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    vm_ = vm;
    if (java_class_ == nullptr) {
      java_class_ = jni::LoadJClass(env, kJavaClass);
      if (java_class_ == nullptr) {
        CARDBOARD_LOGE("Reprojection: %s is missing", kJavaClass);
        return;
      }
      set_active_method_ =
          env->GetStaticMethodID(java_class_, "setActive", "(Z)V");
      jni::CheckExceptionInJava(env);
    }
    java_class = java_class_;
  }

  const jmethodID attach = env->GetStaticMethodID(
      java_class, "attach", "(Landroid/content/Context;)V");
  if (!jni::CheckExceptionInJava(env) && attach != nullptr) {
    env->CallStaticVoidMethod(java_class, attach, activity);
    jni::CheckExceptionInJava(env);
  }
  const jmethodID device_setting = env->GetStaticMethodID(
      java_class, "deviceSetting", "(Landroid/content/Context;)I");
  if (!jni::CheckExceptionInJava(env) && device_setting != nullptr) {
    device_setting_ =
        env->CallStaticIntMethod(java_class, device_setting, activity);
    jni::CheckExceptionInJava(env);
  }
  const jmethodID refresh_rate = env->GetStaticMethodID(
      java_class, "refreshRate", "(Landroid/content/Context;)F");
  if (!jni::CheckExceptionInJava(env) && refresh_rate != nullptr) {
    const float rate =
        env->CallStaticFloatMethod(java_class, refresh_rate, activity);
    jni::CheckExceptionInJava(env);
    if (rate > 1.0f) {
      std::lock_guard<std::mutex> lock(mutex_);
      refresh_rate_ = rate;
    }
  }
  CARDBOARD_LOGI("Reprojection: project asks for %s, device says %s: %s",
                 requested_ ? "on" : "off",
                 device_setting_ < 0 ? "nothing"
                                     : (device_setting_ > 0 ? "on" : "off"),
                 IsEnabled() ? "on" : "off");
}

bool IsEnabled() {
  const int device_setting = device_setting_;
  return device_setting < 0 ? requested_.load() : device_setting > 0;
}

bool IsPresenting() { return presenting_; }

void SetHeadTracker(CardboardHeadTracker* head_tracker) {
  std::lock_guard<std::mutex> lock(tracker_mutex_);
  head_tracker_ = head_tracker;
}

void SetViewportOrientation(CardboardViewportOrientation orientation) {
  viewport_orientation_ = orientation;
}

bool OnHeadPose(float* orientation, bool before_render) {
  const bool testing = test_mode_ != 0;
  if (testing) {
    TestOrientation(orientation);
  } else {
    SpinRenderOrientation(orientation, before_render);
  }
  std::lock_guard<std::mutex> lock(pose_mutex_);
  // The cameras are placed with the pose of the update just before rendering.
  // Should Unity never make that update, any pose is better than none.
  if (before_render || !seen_before_render_) {
    std::memcpy(render_orientation_, orientation, sizeof(render_orientation_));
  }
  seen_before_render_ = seen_before_render_ || before_render;
  return testing;
}

void SetTiltToSpin(bool enabled, float threshold_radians,
                   float radians_per_second) {
  std::lock_guard<std::mutex> lock(spin_mutex_);
  spin_enabled_ = enabled;
  spin_threshold_ = std::sin(
      std::min(1.5, std::max(0.0, static_cast<double>(threshold_radians))));
  spin_speed_ = std::max(0.0, static_cast<double>(radians_per_second));
  if (!enabled) {
    spin_rate_ = 0.0;
  }
}

void ResetTiltToSpin() {
  std::lock_guard<std::mutex> lock(spin_mutex_);
  spin_angle_ = 0.0;
}

float GetTiltToSpinYaw() {
  std::lock_guard<std::mutex> lock(spin_mutex_);
  return static_cast<float>(spin_angle_);
}

void SetLens(CardboardEye eye, const CardboardMesh& mesh, const float* fov,
             const float* eye_from_head) {
  std::lock_guard<std::mutex> lock(mutex_);
  Lens& lens = lenses_[eye];
  // The matrix's translation is where the head's middle is in eye space.
  lens.head_offset[0] = eye_from_head[12];
  lens.head_offset[1] = eye_from_head[13];
  lens.vertices.assign(mesh.vertices, mesh.vertices + mesh.n_vertices * 2);
  lens.uvs.assign(mesh.uvs, mesh.uvs + mesh.n_vertices * 2);
  lens.indices.assign(mesh.indices, mesh.indices + mesh.n_indices);
  for (int i = 0; i < 4; ++i) {
    lens.tan_fov[i] = std::abs(std::tan(fov[i]));
  }
  lens.changed = true;
}

void SetHeadLockedLayer(const HeadLockedLayer& layer) {
  std::lock_guard<std::mutex> lock(mutex_);
  layer_ = layer;
  layer_set_ = true;
  layer_age_ = 0;
}

void ClearHeadLockedLayer() {
  std::lock_guard<std::mutex> lock(mutex_);
  layer_set_ = false;
}

void SetViewport(int x, int y, int width, int height) {
  std::lock_guard<std::mutex> lock(mutex_);
  viewport_[0] = x;
  viewport_[1] = y;
  viewport_[2] = width;
  viewport_[3] = height;
}

int AcquireSet() {
  std::lock_guard<std::mutex> lock(mutex_);
  for (int set = 0; set < kSwapChainDepth; ++set) {
    if ((displayed_.valid && displayed_.frame.set == set) ||
        (pending_.valid && pending_.frame.set == set)) {
      continue;
    }
    return set;
  }
  return 0;  // Not reached: two slots cannot hold three sets.
}

void SubmitFrame(const Frame& frame) {
  // The frame before this one must be finished by the GPU, and so able to be
  // shown, before this one takes its place in the queue. Passing over a frame
  // that is still rendering would mean that an app the GPU cannot keep up
  // with never had a finished frame to show. Unity is held back to the GPU's
  // pace here just as it would be by swapping its own surface.
  GLsync previous = nullptr;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (pending_.valid) {
      previous = pending_.fence;
    }
  }
  bool previous_finished = false;
  if (previous != nullptr) {
    const GLenum state = glClientWaitSync(previous, 0, kFrameTimeoutNanos);
    previous_finished =
        state == GL_ALREADY_SIGNALED || state == GL_CONDITION_SATISFIED;
  }

  bool start = false;
  bool report_active = false;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    // Unless the warp thread took it meanwhile, the frame before is still
    // queued: show it from the next refresh, now that it is finished.
    if (pending_.valid && pending_.fence == previous && previous_finished) {
      retired_fences_.push_back(pending_.fence);
      displayed_ = pending_;
      displayed_.fence = nullptr;
      pending_ = Slot();
    }
    DropPendingLocked();

    if (unity_context_ == EGL_NO_CONTEXT) {
      display_ = eglGetCurrentDisplay();
      const EGLContext context = eglGetCurrentContext();
      const EGLSurface draw = eglGetCurrentSurface(EGL_DRAW);
      EGLint config_id = 0;
      EGLint count = 0;
      if (display_ == EGL_NO_DISPLAY || context == EGL_NO_CONTEXT ||
          draw == EGL_NO_SURFACE ||
          !eglQuerySurface(display_, draw, EGL_CONFIG_ID, &config_id)) {
        return;
      }
      const EGLint attributes[] = {EGL_CONFIG_ID, config_id, EGL_NONE};
      if (!eglChooseConfig(display_, attributes, &unity_config_, 1, &count) ||
          count != 1) {
        return;
      }
      GLint framebuffer = 0;
      GLint encoding = GL_LINEAR;
      glGetIntegerv(GL_FRAMEBUFFER_BINDING, &framebuffer);
      glGetFramebufferAttachmentParameteriv(
          GL_FRAMEBUFFER, framebuffer == 0 ? GL_BACK : GL_COLOR_ATTACHMENT0,
          GL_FRAMEBUFFER_ATTACHMENT_COLOR_ENCODING, &encoding);
      unity_surface_is_srgb_ = encoding == GL_SRGB;
      unity_context_ = context;
      CARDBOARD_LOGI("Reprojection: Unity's surface is %s",
                     unity_surface_is_srgb_ ? "sRGB" : "linear");
    }

    pending_.valid = true;
    pending_.serial = ++frame_serial_;
    pending_.frame = frame;
    // The layer goes with the frame, so that a texture drawn for it this
    // frame is finished, as the frame is, before either is shown. Unity's
    // main thread sets it up to a frame ahead of this thread, so it is kept
    // for a few frames, not taken once.
    if (layer_set_ && ++layer_age_ > kHeadLockedLayerFrames) {
      layer_set_ = false;
    }
    pending_.has_layer = layer_set_;
    pending_.layer = layer_;
    pending_.fence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
    {
      std::lock_guard<std::mutex> pose_lock(pose_mutex_);
      std::memcpy(pending_.orientation, render_orientation_,
                  sizeof(pending_.orientation));
    }
    // The fence must reach the GPU for the warp thread ever to see it signal.
    glFlush();

    start = !thread_started_ && !failed_ && window_ != nullptr;
    if (!active_reported_) {
      active_reported_ = true;
      report_active = true;
    }
  }
  if (report_active) {
    ReportActive(true);
  }
  if (start) {
    std::lock_guard<std::mutex> control(control_mutex_);
    StartThreadControlled();
  }
}

void ForgetFrames() {
  std::lock_guard<std::mutex> lock(mutex_);
  DropPendingLocked();
  displayed_ = Slot();
}

void OnRenderingStopped() {
  bool report = false;
  {
    std::lock_guard<std::mutex> control(control_mutex_);
    StopThreadControlled();
    std::lock_guard<std::mutex> lock(mutex_);
    DropPendingLocked();
    displayed_ = Slot();
    unity_context_ = EGL_NO_CONTEXT;
    failed_ = false;
    report = active_reported_;
    active_reported_ = false;
  }
  if (report) {
    ReportActive(false);
  }
}

}  // namespace cardboard::unity::reprojection

extern "C" {

using namespace cardboard::unity::reprojection;  // NOLINT

// The overlay's Surface is there, or has changed size.
JNIEXPORT void JNICALL
Java_avn_cardboard_reprojection_ReprojectionSurface_nativeSurfaceChanged(
    JNIEnv* env, jclass /*clazz*/, jobject surface) {
  std::lock_guard<std::mutex> control(control_mutex_);
  StopThreadControlled();
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (window_ != nullptr) {
      ANativeWindow_release(window_);
    }
    window_ = ANativeWindow_fromSurface(env, surface);
    failed_ = false;
  }
  StartThreadControlled();
}

// The overlay's Surface is going. Returns once nothing here is using it.
JNIEXPORT void JNICALL
Java_avn_cardboard_reprojection_ReprojectionSurface_nativeSurfaceDestroyed(
    JNIEnv* /*env*/, jclass /*clazz*/) {
  std::lock_guard<std::mutex> control(control_mutex_);
  StopThreadControlled();
  std::lock_guard<std::mutex> lock(mutex_);
  if (window_ != nullptr) {
    ANativeWindow_release(window_);
    window_ = nullptr;
  }
}

}  // extern "C"
