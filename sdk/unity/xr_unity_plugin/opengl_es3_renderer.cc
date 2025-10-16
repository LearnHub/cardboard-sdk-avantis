/*
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifdef __ANDROID__
#include <GLES3/gl3.h>
#endif
#ifdef __APPLE__
#include <OpenGLES/ES3/gl.h>
#endif
#include "util/logging.h"
#include "unity/xr_unity_plugin/renderer.h"
#include <EGL/egl.h>

// @def Forwards the call to CheckGlError().
#define CHECKGLERROR(label) CheckGlError(__FILE__, __LINE__, label)

namespace XRRendering {
    extern bool g_isSRGB;
}

namespace cardboard::unity {
namespace {

/**
 * Checks for OpenGL errors, and crashes if one has occurred.  Note that this
 * can be an expensive call, so real applications should call this rarely.
 *
 * @param file File name
 * @param line Line number
 * @param label Error label
 */
void CheckGlError(const char* file, int line, const char* label) {
  const int gl_error = glGetError();
  if (gl_error != GL_NO_ERROR) {
    CARDBOARD_LOGF("[%s : %d] GL error: %d. %s", file, line, gl_error, label);
    // Crash immediately to make OpenGL errors obvious.
    abort();
  }
}

// TODO(b/155457703): De-dupe GL utility function here and in
// distortion_renderer.cc
GLuint LoadShader(GLenum shader_type, const char* source) {
  GLuint shader = glCreateShader(shader_type);
  glShaderSource(shader, 1, &source, nullptr);
  glCompileShader(shader);
  CHECKGLERROR("glCompileShader");
  GLint result = GL_FALSE;
  glGetShaderiv(shader, GL_COMPILE_STATUS, &result);
  if (result == GL_FALSE) {
    int log_length;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &log_length);
    if (log_length == 0) {
      return 0;
    }

    std::vector<char> log_string(log_length);
    glGetShaderInfoLog(shader, log_length, nullptr, log_string.data());
    CARDBOARD_LOGE("Could not compile shader of type %d: %s", shader_type,
                   log_string.data());

    shader = 0;
  }

  return shader;
}

// TODO(b/155457703): De-dupe GL utility function here and in
// distortion_renderer.cc
GLuint CreateProgram(const char* vertex, const char* fragment) {
  GLuint vertex_shader = LoadShader(GL_VERTEX_SHADER, vertex);
  if (vertex_shader == 0) {
    return 0;
  }

  GLuint fragment_shader = LoadShader(GL_FRAGMENT_SHADER, fragment);
  if (fragment_shader == 0) {
    return 0;
  }

  GLuint program = glCreateProgram();

  glAttachShader(program, vertex_shader);
  glAttachShader(program, fragment_shader);
  glLinkProgram(program);
  CHECKGLERROR("glLinkProgram");

  GLint result = GL_FALSE;
  glGetProgramiv(program, GL_LINK_STATUS, &result);
  if (result == GL_FALSE) {
    int log_length;
    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &log_length);
    if (log_length == 0) {
      return 0;
    }

    std::vector<char> log_string(log_length);
    glGetShaderInfoLog(program, log_length, nullptr, log_string.data());
    CARDBOARD_LOGE("Could not compile program: %s", log_string.data());

    return 0;
  }

  glDetachShader(program, vertex_shader);
  glDetachShader(program, fragment_shader);
  glDeleteShader(vertex_shader);
  glDeleteShader(fragment_shader);
  CHECKGLERROR("GlCreateProgram");

  return program;
}

/// @brief Vertex shader for rendering the widgets when using OpenGL ES3.0.
const char kWidgetVertexShaderOpenGlEs3[] =
    R"glsl(#version 300 es
  layout (location = 0) in vec2 a_Position;
  layout (location = 1) in vec2 a_TexCoords;
  out vec2 v_TexCoords;
  void main() {
    gl_Position = vec4(a_Position, 0, 1);
    v_TexCoords = a_TexCoords;
  }
  )glsl";

/// @brief Fragment shader for rendering the widgets when using OpenGL ES3.0.
const char kWidgetFragmentShaderOpenGlEs3[] =
    R"glsl(#version 300 es
  precision mediump float;
  uniform sampler2D u_Texture;
  in vec2 v_TexCoords;
  out vec4 o_FragColor;
  void main() {
    o_FragColor = texture(u_Texture, v_TexCoords);
  }
  )glsl";


#include <time.h>
#include <stdint.h>
#include <math.h>


namespace hz72Limiter {

    // Monotonic clock
    static inline int64_t NowNs() {
        timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        return (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec;
    }

    // State
    static double  s_refresh_hz    = 72.0;
    static int64_t s_vsync_ns      = (int64_t)(1000000000.0 / 72.0 + 0.5); // ~13,888,889
    static int64_t s_last_target   = 0;   // last target deadline in ns
    static double  s_accum_frac    = 0.0; // fractional accumulator

    inline void SetRefreshHz(double hz) {
        if (hz > 1.0 && hz < 1000.0) {
            s_refresh_hz = hz;
            s_vsync_ns   = (int64_t)(1000000000.0 / hz + 0.5);
            // Don’t reset s_last_target/s_accum_frac so cadence continues smoothly.
        }
    }

    inline void FPSLimit(double target_fps) {
        if (target_fps <= 0.0) return;

        // Convert FPS to vsyncs per frame, then alternate between floor/ceil.
        double vsyncs_per_frame = s_refresh_hz / target_fps;
        int    base_vs     = (int)floor(vsyncs_per_frame);
        if (base_vs < 1) base_vs = 1;
        double frac_vs     = vsyncs_per_frame - (double)base_vs;

        int vs_this_frame = base_vs;
        s_accum_frac += frac_vs;
        if (s_accum_frac >= 1.0) {
            vs_this_frame += 1;
            s_accum_frac -= 1.0;
        }

        // First call: start from "now"
        const int64_t now = NowNs();
        if (s_last_target == 0) {
            s_last_target = now;
        }

        // Target time is last target + N vsync intervals
        const int64_t target = s_last_target + (int64_t)vs_this_frame * s_vsync_ns;

        // Coarse sleep, then fine spin to hit the deadline
        int64_t dt = target - now;
        if (dt > 1000000) { // > 1 ms
            timespec ts{
                    static_cast<time_t>(dt / 1000000000LL),
                    static_cast<long>(dt % 1000000000LL)
            };
            nanosleep(&ts, nullptr);
        }
        while ((target - NowNs()) > 0) { /* busy-wait tail for precision */ }

        // Advance schedule
        s_last_target = target;
    }
}

static EGLDisplay g_display = EGL_NO_DISPLAY;
static EGLSurface g_surface = EGL_NO_SURFACE;
static EGLContext g_context = EGL_NO_CONTEXT;


class OpenGlEs3Renderer : public Renderer {
 public:
  OpenGlEs3Renderer() = default;
  ~OpenGlEs3Renderer() { TeardownWidgets(); }

  void SetupWidgets() override
  {
    if (widget_program_ != 0) {
      return;
    }
    widget_program_ = CreateProgram(kWidgetVertexShaderOpenGlEs3,
                                    kWidgetFragmentShaderOpenGlEs3);
    widget_attrib_position_ =
        glGetAttribLocation(widget_program_, "a_Position");
    widget_attrib_tex_coords_ =
        glGetAttribLocation(widget_program_, "a_TexCoords");
    widget_uniform_texture_ =
        glGetUniformLocation(widget_program_, "u_Texture");
  }

  void RenderWidgets(const ScreenParams& screen_params,
                     const std::vector<WidgetParams>& widget_params) override {
    if (widget_program_ == 0) {
      CARDBOARD_LOGF(
          "Trying to RenderWidgets without setting up the renderer.");
      return;
    }

    glViewport(screen_params.viewport_x, screen_params.viewport_y,
               screen_params.viewport_width, screen_params.viewport_height);

    for (const WidgetParams& widget_param : widget_params) {
      RenderWidget(screen_params.viewport_width, screen_params.viewport_height,
                   widget_param);
    }
  }

  void TeardownWidgets() override {
    if (widget_program_ == 0) {
      return;
    }
    glDeleteProgram(widget_program_);
    CHECKGLERROR("GlDeleteProgram");
    widget_program_ = 0;
  }

  void CreateRenderTexture(RenderTexture* render_texture, int screen_width, int screen_height) override
  {
    // Create texture color buffer.
    GLuint tmp = 0;
    glGenTextures(1, &tmp);
    glBindTexture(GL_TEXTURE_2D, tmp);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    GLint internalFormat = XRRendering::g_isSRGB ?  GL_SRGB8_ALPHA8 :GL_RGB;
    GLint format = XRRendering::g_isSRGB ?  GL_RGBA :GL_RGB;
    glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, screen_width / 2, screen_height, 0,
                 format, GL_UNSIGNED_BYTE, 0);
    CHECKGLERROR("Create texture color buffer.");
    render_texture->color_buffer = tmp;

    // Create texture depth buffer.
    tmp = 0;
    glGenRenderbuffers(1, &tmp);
    glBindRenderbuffer(GL_RENDERBUFFER, tmp);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT16,
                          screen_width / 2, screen_height);
    CHECKGLERROR("Create texture depth buffer.");
    render_texture->depth_buffer = tmp;
  }
  //---------------------------------------------------------------------------

  void CreateRenderTextureArray(RenderTexture* render_texture, int width, int height, int layer_count)  override
    {
        // --- Create color buffer as a 2D array texture ---
        GLuint tex = 0;
        glGenTextures(1, &tex);
        glBindTexture(GL_TEXTURE_2D_ARRAY, tex);

        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        GLint internalFormat = XRRendering::g_isSRGB ?  GL_SRGB8_ALPHA8 :GL_RGB;
        GLint format = XRRendering::g_isSRGB ?  GL_RGBA :GL_RGB;

        // Allocate storage for all layers (no data upload yet)
        glTexImage3D(GL_TEXTURE_2D_ARRAY,
                     0,                      // mip level
                     internalFormat,               // internal format
                     width / 2,
                     height,
                     layer_count,            // depth = number of slices
                     0,                     // border
                     format,                       // format
                     GL_UNSIGNED_BYTE,       // type
                     nullptr);              // no initial data

        CHECKGLERROR("Create texture array color buffer.");
        render_texture->color_buffer = tex;

        // --- Create depth buffer as a layered renderbuffer ---
        GLuint depth = 0;
        glGenRenderbuffers(1, &depth);
        glBindRenderbuffer(GL_RENDERBUFFER, depth);

        // --- Depth: GL_TEXTURE_2D_ARRAY with 'layer_count' slices ---
        // We can't make a "layered renderbuffer" in ES3, so we use a depth texture array.
        GLuint depth_tex = 0;
        glGenTextures(1, &depth_tex);
        glBindTexture(GL_TEXTURE_2D_ARRAY, depth_tex);

        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        glTexImage3D(GL_TEXTURE_2D_ARRAY,
                     0,
                     GL_DEPTH_COMPONENT16, // match original depth precision
                     width,
                     height,
                     layer_count,
                     0,
                     GL_DEPTH_COMPONENT,
                     GL_UNSIGNED_SHORT,
                     nullptr);

        CHECKGLERROR("Create texture array depth buffer.");
        render_texture->depth_buffer = depth_tex;
    }
  //---------------------------------------------------------------------------

  void DestroyRenderTexture(RenderTexture* render_texture) override {
    GLuint tmp = static_cast<GLuint>(render_texture->depth_buffer);
    glDeleteRenderbuffers(1, &tmp);
    render_texture->depth_buffer = 0;

    tmp = static_cast<GLuint>(render_texture->color_buffer);
    glDeleteTextures(1, &tmp);
    render_texture->color_buffer = 0;
  }

  void RenderEyesToDisplay(
      CardboardDistortionRenderer* renderer, const ScreenParams& screen_params,
      const CardboardEyeTextureDescription* left_eye,
      const CardboardEyeTextureDescription* right_eye) override
      {
      hz72Limiter::FPSLimit(144.0);
      glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
      int bound_framebuffer = 0;
      glGetIntegerv(GL_FRAMEBUFFER_BINDING, &bound_framebuffer);

      CardboardDistortionRenderer_renderEyeToDisplay(
        renderer, bound_framebuffer, screen_params.viewport_x,
        screen_params.viewport_y, screen_params.viewport_width,
        screen_params.viewport_height, left_eye, right_eye);
      }

  void RunRenderingPreProcessing(
      const ScreenParams& /* screen_params */) override {
    // Nothing to do.
  }

  void RunRenderingPostProcessing() override {
    // Nothing to do.
  }

 private:
  static constexpr float Lerp(float start, float end, float val) {
    return start + (end - start) * val;
  }

  void RenderWidget(int screen_width, int screen_height,
                    const WidgetParams& params) {
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glDisable(GL_CULL_FACE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glBlendEquation(GL_FUNC_ADD);

    // Convert coordinates to normalized space (-1,-1 - +1,+1)
    float x = Lerp(-1, +1, static_cast<float>(params.x) / screen_width);
    float y = Lerp(-1, +1, static_cast<float>(params.y) / screen_height);
    float width = params.width * 2.0f / screen_width;
    float height = params.height * 2.0f / screen_height;
    const float position[] = {x, y,          x + width, y,
                              x, y + height, x + width, y + height};

    // This extra call is required in #gles3 with respect to #gles2. That API is
    // not available in the latter.
    // {
    glBindVertexArray(0);
    // }
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glEnableVertexAttribArray(widget_attrib_position_);
    glVertexAttribPointer(
        widget_attrib_position_, /*size=*/2, /*type=*/GL_FLOAT,
        /*normalized=*/GL_FALSE, /*stride=*/0, /*pointer=*/position);
    CHECKGLERROR("RenderWidget");

    const float uv[] = {0, 0, 1, 0, 0, 1, 1, 1};
    glEnableVertexAttribArray(widget_attrib_tex_coords_);
    glVertexAttribPointer(
        widget_attrib_tex_coords_, /*size=*/2, /*type=*/GL_FLOAT,
        /*normalized=*/GL_FALSE, /*stride=*/0, /*pointer=*/uv);

    glUseProgram(widget_program_);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, static_cast<int>(params.texture));
    glUniform1i(widget_uniform_texture_, 0);

    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

    CHECKGLERROR("RenderWidget");
  }

  // @brief Widgets GL program.
  GLuint widget_program_{0};

  // @brief Widgets "a_Position" attrib location.
  GLint widget_attrib_position_;

  // @brief Widgets "a_TexCoords" attrib location.
  GLint widget_attrib_tex_coords_;

  // @brief Widgets "u_Texture" uniform location.
  GLint widget_uniform_texture_;
};

}  // namespace

std::unique_ptr<Renderer> MakeOpenGlEs3Renderer() {
  return std::make_unique<OpenGlEs3Renderer>();
}

}  // namespace cardboard::unity
