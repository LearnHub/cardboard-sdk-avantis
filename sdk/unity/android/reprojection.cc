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
// one refresh is latched by the compositor at the next and scanned out over the
// one after, so the middle of the picture reaches the eye about two and a half
// refreshes on.
constexpr double kPredictionRefreshes = 2.5;

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
void main() {
  gl_Position = vec4(a_Position, 0.0, 1.0);
  // The lens mesh's texture coordinates run evenly across the eye buffer's
  // field of view in tan-angle, so they give the view ray directly.
  vec2 low = -u_TanFov.xz;
  vec2 high = u_TanFov.yw;
  vec3 ray = u_Rotation * vec3(mix(low, high, a_TexCoords), -1.0);
  vec2 tan_angle = ray.xy / max(-ray.z, 0.0001);
  v_TexCoords = (tan_angle - low) / (high - low);
})glsl";

// The fragment shaders: a header that defines SAMPLE(uv) to read the eye
// buffer, then this body. It is the distortion pass's scale and sharpen, then
// black wherever the turned ray has left the eye buffer.
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
out vec4 o_Color;
void main() {
  highp vec2 inside = step(vec2(0.0), v_TexCoords) * step(v_TexCoords, vec2(1.0));
  highp vec2 coords = v_TexCoords * u_Scale;
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
  o_Color = vec4(c * inside.x * inside.y, 1.0);
})glsl";

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
        !CreateProgram(&program_2d_, kFragmentHeader2D)) {
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
      glDeleteProgram(program_array_.id);
      glDeleteProgram(program_2d_.id);
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

    float display_orientation[4] = {0.0f, 0.0f, 0.0f, 1.0f};
    const int test_mode = test_mode_;
    const bool have_orientation =
        test_mode == 0 && ReadDisplayOrientation(display_orientation);

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
        drew_frame = true;
        if (displayed_.serial != last_serial_) {
          last_serial_ = displayed_.serial;
          ++new_frames_;
        }
      }
    }

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

  static bool CreateProgram(Program* program, const char* fragment_header) {
    const std::string fragment_source =
        std::string(fragment_header) + kFragmentBody;
    const GLuint vertex = CompileShader(GL_VERTEX_SHADER, kVertexShader);
    const GLuint fragment =
        CompileShader(GL_FRAGMENT_SHADER, fragment_source.c_str());
    if (vertex == 0 || fragment == 0) {
      glDeleteShader(vertex);
      glDeleteShader(fragment);
      return false;
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
          "Reprojection: %.1f refreshes/s (%d late), %.1f new frames/s, %d "
          "refreshes waited on the GPU, slowest draw %.1f ms",
          refreshes_ / seconds, late_refreshes_, new_frames_ / seconds,
          refreshes_waiting_, max_draw_nanos_ * 1e-6);
      report_start_nanos_ = now;
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
  GLuint vertex_arrays_[2] = {0, 0};
  GLuint vertex_buffers_[2] = {0, 0};
  GLuint uv_buffers_[2] = {0, 0};
  GLuint index_buffers_[2] = {0, 0};
  GLsizei index_counts_[2] = {0, 0};
  float tan_fov_[2][4] = {{1.0f, 1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f, 1.0f}};

  // Statistics
  int64_t last_refresh_nanos_ = 0;
  int64_t report_start_nanos_ = 0;
  int64_t max_draw_nanos_ = 0;
  int refreshes_ = 0;
  int late_refreshes_ = 0;
  int new_frames_ = 0;
  int refreshes_waiting_ = 0;
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

void SetLens(CardboardEye eye, const CardboardMesh& mesh, const float* fov) {
  std::lock_guard<std::mutex> lock(mutex_);
  Lens& lens = lenses_[eye];
  lens.vertices.assign(mesh.vertices, mesh.vertices + mesh.n_vertices * 2);
  lens.uvs.assign(mesh.uvs, mesh.uvs + mesh.n_vertices * 2);
  lens.indices.assign(mesh.indices, mesh.indices + mesh.n_indices);
  for (int i = 0; i < 4; ++i) {
    lens.tan_fov[i] = std::abs(std::tan(fov[i]));
  }
  lens.changed = true;
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
