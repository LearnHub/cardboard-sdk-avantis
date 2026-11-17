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
#include "unity/android/classview_capture.h"

#include <EGL/egl.h>
#include <GLES3/gl3.h>
#include <android/native_window.h>
#include <android/native_window_jni.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <mutex>

#include "jni_utils/android/jni_utils.h"
#include "util/logging.h"

// Threading. Three kinds of thread come through here:
//   - Unity's render thread (OnFrameSubmitted, OnRenderingStopped), which owns
//     the GL context and is the only one to make the capture surface current;
//   - binder threads from the Config app (the native* JNI calls), which create
//     and destroy the EGL surface;
//   - Unity's main thread (InitializeAndroid).
// One mutex covers all the state. The render thread holds it across the whole
// capture draw and always puts Unity's own surface back before releasing it,
// so a binder thread that holds the mutex knows the capture surface is not
// current anywhere and eglDestroySurface disconnects from the Surface at once.
// That matters: Config connects its mirror to the Surface the moment stop
// returns. Creating and destroying an EGL surface needs no current context,
// which is why those calls need not be marshalled to the render thread.

namespace cardboard::unity::classview {
namespace {

using Clock = std::chrono::steady_clock;

constexpr char kJavaClass[] = "avn/cardboard/classview/ClassViewFrameSource";

// A capture is refused unless a frame was submitted this recently: an app that
// is not drawing has nothing to give, and Config's mirror should have it.
constexpr std::chrono::milliseconds kRenderingTimeout{1000};

// The fragment shaders write the eye texture's samples as they are. That is
// what the panel shows in single-pass rendering, where the distortion pass
// raises them to the power 2.2 and Unity's sRGB back buffer undoes it
// (compared pixel for pixel on a CVR355128, linear colour space).
constexpr char kVertexShader[] = R"glsl(#version 300 es
uniform vec4 u_UvRect;  // origin, size
out vec2 v_Uv;
void main() {
  vec2 corner = vec2(float(gl_VertexID & 1), float((gl_VertexID >> 1) & 1));
  v_Uv = u_UvRect.xy + corner * u_UvRect.zw;
  gl_Position = vec4(corner * 2.0 - 1.0, 0.0, 1.0);
})glsl";

constexpr char kFragmentShaderArray[] = R"glsl(#version 300 es
precision mediump float;
precision mediump sampler2DArray;
uniform sampler2DArray u_Texture;
in vec2 v_Uv;
out vec4 o_Color;
void main() {
  o_Color = vec4(texture(u_Texture, vec3(v_Uv, 0.0)).rgb, 1.0);
})glsl";

constexpr char kFragmentShader2D[] = R"glsl(#version 300 es
precision mediump float;
uniform sampler2D u_Texture;
in vec2 v_Uv;
out vec4 o_Color;
void main() {
  o_Color = vec4(texture(u_Texture, v_Uv).rgb, 1.0);
})glsl";

struct Program {
  GLuint id = 0;
  GLint uv_rect = -1;
  GLint texture = -1;
};

std::mutex mutex_;

// Java side
JavaVM* vm_ = nullptr;
jclass java_class_ = nullptr;
jmethodID set_rendering_method_ = nullptr;

// Unity's EGL display and the config of its window, learnt on the render
// thread from the first frame.
EGLDisplay display_ = EGL_NO_DISPLAY;
EGLConfig config_ = nullptr;
bool have_egl_ = false;
Clock::time_point last_frame_time_;
bool rendering_reported_ = false;

// The running capture
ANativeWindow* window_ = nullptr;
EGLSurface surface_ = EGL_NO_SURFACE;
int width_ = 0;
int height_ = 0;
int divisor_ = 1;
bool swap_interval_set_ = false;
uint64_t frame_index_ = 0;
uint64_t frames_drawn_ = 0;

// GL objects, render thread only
Program program_array_;
Program program_2d_;
GLuint vertex_array_ = 0;

void ReportRendering(bool rendering) {
  if (vm_ == nullptr || java_class_ == nullptr ||
      set_rendering_method_ == nullptr) {
    return;
  }
  JNIEnv* env = nullptr;
  jni::LoadJNIEnv(vm_, &env);
  if (env == nullptr) {
    return;
  }
  env->CallStaticVoidMethod(java_class_, set_rendering_method_,
                            rendering ? JNI_TRUE : JNI_FALSE);
  jni::CheckExceptionInJava(env);
}

// Ends the capture and disconnects from the Surface. Requires the mutex.
void DestroyCaptureLocked() {
  if (surface_ != EGL_NO_SURFACE) {
    eglDestroySurface(display_, surface_);
    surface_ = EGL_NO_SURFACE;
    CARDBOARD_LOGI("ClassView capture stopped after %llu frames",
                   static_cast<unsigned long long>(frames_drawn_));
  }
  if (window_ != nullptr) {
    ANativeWindow_release(window_);
    window_ = nullptr;
  }
}

GLuint CompileShader(GLenum type, const char* source) {
  const GLuint shader = glCreateShader(type);
  glShaderSource(shader, 1, &source, nullptr);
  glCompileShader(shader);
  GLint compiled = GL_FALSE;
  glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
  if (compiled != GL_TRUE) {
    char log[512] = {0};
    glGetShaderInfoLog(shader, sizeof(log) - 1, nullptr, log);
    CARDBOARD_LOGE("ClassView capture shader failed to compile: %s", log);
    glDeleteShader(shader);
    return 0;
  }
  return shader;
}

bool EnsureProgram(Program* program, const char* fragment_source) {
  if (program->id != 0) {
    return true;
  }
  const GLuint vertex = CompileShader(GL_VERTEX_SHADER, kVertexShader);
  const GLuint fragment = CompileShader(GL_FRAGMENT_SHADER, fragment_source);
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
    CARDBOARD_LOGE("ClassView capture program failed to link");
    glDeleteProgram(id);
    return false;
  }
  program->id = id;
  program->uv_rect = glGetUniformLocation(id, "u_UvRect");
  program->texture = glGetUniformLocation(id, "u_Texture");
  return true;
}

// Draws the eye into the capture surface. Render thread, mutex held.
void DrawCaptureLocked(const EyeFrame& frame) {
  const EGLContext context = eglGetCurrentContext();
  const EGLSurface unity_draw = eglGetCurrentSurface(EGL_DRAW);
  const EGLSurface unity_read = eglGetCurrentSurface(EGL_READ);
  GLint unity_framebuffer = 0;
  GLint unity_viewport[4] = {0, 0, 0, 0};
  glGetIntegerv(GL_FRAMEBUFFER_BINDING, &unity_framebuffer);
  glGetIntegerv(GL_VIEWPORT, unity_viewport);

  if (!eglMakeCurrent(display_, surface_, surface_, context)) {
    CARDBOARD_LOGE("ClassView capture surface lost (0x%x); dropping it",
                   eglGetError());
    eglMakeCurrent(display_, unity_draw, unity_read, context);
    DestroyCaptureLocked();
    return;
  }

  if (!swap_interval_set_) {
    // Never wait in the capture swap: the encoder holds frames, and a full
    // queue must drop one, not stall the panel.
    eglSwapInterval(display_, 0);
    swap_interval_set_ = true;
  }

  Program* program = frame.is_texture_array ? &program_array_ : &program_2d_;
  if (EnsureProgram(program, frame.is_texture_array ? kFragmentShaderArray
                                                    : kFragmentShader2D)) {
    if (vertex_array_ == 0) {
      glGenVertexArrays(1, &vertex_array_);
    }

    // The largest centred part of the eye that has the capture's shape.
    const float eye_aspect =
        static_cast<float>(frame.width) / static_cast<float>(frame.height);
    const float capture_aspect =
        static_cast<float>(width_) / static_cast<float>(height_);
    const float part_u = std::min(1.0f, capture_aspect / eye_aspect);
    const float part_v = std::min(1.0f, eye_aspect / capture_aspect);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, width_, height_);
    glDisable(GL_SCISSOR_TEST);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_STENCIL_TEST);
    glDisable(GL_BLEND);
    glDisable(GL_CULL_FACE);
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);

    glUseProgram(program->id);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(frame.is_texture_array ? GL_TEXTURE_2D_ARRAY : GL_TEXTURE_2D,
                  frame.texture);
    glUniform1i(program->texture, 0);
    glUniform4f(program->uv_rect, 0.5f * (1.0f - part_u) * frame.used_u,
                0.5f * (1.0f - part_v) * frame.used_v, part_u * frame.used_u,
                part_v * frame.used_v);

    glBindVertexArray(vertex_array_);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glBindVertexArray(0);

    eglSwapBuffers(display_, surface_);
    if (++frames_drawn_ % 600 == 0) {
      CARDBOARD_LOGI("ClassView capture drawn %llu frames",
                     static_cast<unsigned long long>(frames_drawn_));
    }
  }

  eglMakeCurrent(display_, unity_draw, unity_read, context);
  glBindFramebuffer(GL_FRAMEBUFFER, static_cast<GLuint>(unity_framebuffer));
  glViewport(unity_viewport[0], unity_viewport[1], unity_viewport[2],
             unity_viewport[3]);
}

}  // namespace

void InitializeAndroid(JavaVM* vm, jobject activity) {
  JNIEnv* env = nullptr;
  jni::LoadJNIEnv(vm, &env);
  if (env == nullptr) {
    return;
  }

  jclass java_class = nullptr;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    vm_ = vm;
    if (java_class_ == nullptr) {
      java_class_ = jni::LoadJClass(env, kJavaClass);
      if (java_class_ == nullptr) {
        CARDBOARD_LOGE("ClassView capture: %s is missing", kJavaClass);
        return;
      }
      set_rendering_method_ =
          env->GetStaticMethodID(java_class_, "setRendering", "(Z)V");
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
}

void OnFrameSubmitted(const EyeFrame& frame) {
  bool report_rendering = false;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!have_egl_) {
      display_ = eglGetCurrentDisplay();
      const EGLSurface draw = eglGetCurrentSurface(EGL_DRAW);
      EGLint config_id = 0;
      EGLint count = 0;
      if (display_ == EGL_NO_DISPLAY || draw == EGL_NO_SURFACE ||
          !eglQuerySurface(display_, draw, EGL_CONFIG_ID, &config_id)) {
        return;
      }
      const EGLint attributes[] = {EGL_CONFIG_ID, config_id, EGL_NONE};
      if (!eglChooseConfig(display_, attributes, &config_, 1, &count) ||
          count != 1) {
        return;
      }
      have_egl_ = true;
    }
    last_frame_time_ = Clock::now();
    if (!rendering_reported_) {
      rendering_reported_ = true;
      report_rendering = true;
    }

    if (surface_ != EGL_NO_SURFACE && frame.texture != 0 && frame.width > 0 &&
        frame.height > 0 && frame_index_++ % divisor_ == 0) {
      DrawCaptureLocked(frame);
    }
  }
  if (report_rendering) {
    ReportRendering(true);
  }
}

void OnRenderingStopped() {
  bool report = false;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    DestroyCaptureLocked();
    glDeleteProgram(program_array_.id);
    glDeleteProgram(program_2d_.id);
    glDeleteVertexArrays(1, &vertex_array_);
    program_array_ = Program();
    program_2d_ = Program();
    vertex_array_ = 0;
    have_egl_ = false;
    report = rendering_reported_;
    rendering_reported_ = false;
  }
  if (report) {
    ReportRendering(false);
  }
}

}  // namespace cardboard::unity::classview

extern "C" {

using namespace cardboard::unity::classview;  // NOLINT

JNIEXPORT jstring JNICALL
Java_avn_cardboard_classview_ClassViewFrameSource_nativeStart(
    JNIEnv* env, jclass /*clazz*/, jobject surface, jint width, jint height,
    jint divisor) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!have_egl_ || Clock::now() - last_frame_time_ > kRenderingTimeout) {
    return env->NewStringUTF("not rendering");
  }
  if (width <= 0 || height <= 0) {
    return env->NewStringUTF("bad size");
  }
  DestroyCaptureLocked();

  window_ = ANativeWindow_fromSurface(env, surface);
  if (window_ == nullptr) {
    return env->NewStringUTF("no native window for the surface");
  }
  surface_ = eglCreateWindowSurface(display_, config_, window_, nullptr);
  if (surface_ == EGL_NO_SURFACE) {
    char error[64];
    snprintf(error, sizeof(error), "eglCreateWindowSurface failed: 0x%x",
             eglGetError());
    ANativeWindow_release(window_);
    window_ = nullptr;
    return env->NewStringUTF(error);
  }
  width_ = width;
  height_ = height;
  divisor_ = std::max(1, static_cast<int>(divisor));
  swap_interval_set_ = false;
  frame_index_ = 0;
  frames_drawn_ = 0;
  CARDBOARD_LOGI("ClassView capture started: %dx%d every %d frame(s)", width_,
                 height_, divisor_);
  return nullptr;
}

JNIEXPORT jstring JNICALL
Java_avn_cardboard_classview_ClassViewFrameSource_nativeResize(
    JNIEnv* env, jclass /*clazz*/, jint width, jint height, jint divisor) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (surface_ == EGL_NO_SURFACE) {
    return env->NewStringUTF("not capturing");
  }
  if (width <= 0 || height <= 0) {
    return env->NewStringUTF("bad size");
  }
  width_ = width;
  height_ = height;
  divisor_ = std::max(1, static_cast<int>(divisor));
  CARDBOARD_LOGI("ClassView capture resized: %dx%d every %d frame(s)", width_,
                 height_, divisor_);
  return nullptr;
}

JNIEXPORT void JNICALL
Java_avn_cardboard_classview_ClassViewFrameSource_nativeStop(
    JNIEnv* /*env*/, jclass /*clazz*/) {
  std::lock_guard<std::mutex> lock(mutex_);
  DestroyCaptureLocked();
}

}  // extern "C"
