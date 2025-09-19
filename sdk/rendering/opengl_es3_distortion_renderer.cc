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
 *
 * #gles3 - This file is only needed if OpenGL ES 3.0 support is desired. Delete
 * the contents of this file if OpenGL ES 3.0 support is not needed.
 */
#include <array>
#include <vector>

#ifdef CARDBOARD_USE_CUSTOM_GL_BINDINGS
// If required, add a configuration header file with the OpenGL ES 3.0 binding
// customization.
#include "opengl_es3_custom_bindings.h"
#else
#ifdef __ANDROID__
#include <GLES3/gl3.h>
#endif
#ifdef __APPLE__
#include <OpenGLES/ES3/gl.h>
#endif
#ifdef __ANDROID__
#include <GLES2/gl2ext.h>
#endif
#endif  // CARDBOARD_USE_CUSTOM_GL_BINDINGS
#include "distortion_renderer.h"
#include "include/cardboard.h"
#include "util/is_arg_null.h"
#include "util/is_initialized.h"
#include "util/logging.h"

namespace {

    static std::array<GLuint, 2> upscale_tex = {0, 0};

    constexpr const char* kDistortionVertexShader =
    R"glsl(#version 300 es
    precision mediump float;
    layout (location = 0) in vec2 a_Position;
    layout (location = 1) in vec2 a_TexCoords;
    out vec2 v_TexCoords;

    void main() {
      gl_Position = vec4(a_Position, 0, 1);
      v_TexCoords = a_TexCoords;
    })glsl";

    // Multi Pass version
    constexpr const char* kDistortionFragmentShaderTexture2D =
    R"glsl(#version 300 es
    precision mediump float;

    uniform sampler2D u_Texture;   // <-- single 2D texture
    uniform vec2  u_Start;
    uniform vec2  u_End;
    uniform vec2  u_Scale;       // low_res_size / full_res_size
    uniform vec2  u_TexelSize;   // 1.0 / low_res_size
    uniform float u_Sharpen;     // 0.0..~0.35

    in vec2 v_TexCoords;
    out vec4 o_FragColor;

    // Luma for edge measure (mediump is fine on Adreno 610)
    mediump float luma(mediump vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }

    void main() {
        // Distortion UVs
        mediump vec2 coords = u_Start + v_TexCoords * (u_End - u_Start);

        // Skip scale if 1:1
        if (u_Scale.x != 1.0 || u_Scale.y != 1.0) {
            coords *= u_Scale;
        }

    // Derivatives for anisotropy and explicit LOD control
    mediump vec2 dxc = dFdx(coords);
    mediump vec2 dyc = dFdy(coords);
    mediump float sx = length(dxc);
    mediump float sy = length(dyc);

    // Small dynamic LOD bias to reduce shimmer in high distortion
    mediump float aniso = max(sx, sy);
    mediump float biasScale = 0.25;
    mediump float bias = clamp((aniso - 1.0) * biasScale, 0.0, 0.5);
    mediump vec2 dxb = dxc * (1.0 + bias);
    mediump vec2 dyb = dyc * (1.0 + bias);

    // Base sample from texture
    mediump vec3 c = textureGrad(u_Texture, coords, dxb, dyb).rgb;

    // Early out if no sharpening requested
    if (u_Sharpen <= 0.0) {
        o_FragColor = vec4(c, 1.0);
        return;
    }

    // Direction choice for neighbor taps
    bool horiz = (sx >= sy);
    mediump vec2 off = horiz ? vec2(0.75 * u_TexelSize.x, 0.0) : vec2(0.0, 0.75 * u_TexelSize.y);

    // Neighbor samples
    mediump vec3 a = textureGrad(u_Texture, coords + off, dxb, dyb).rgb;
    mediump vec3 b = textureGrad(u_Texture, coords - off, dxb, dyb).rgb;

    // Edge measure
    mediump float g = abs(luma(a) - luma(b));
    mediump float edgeW = clamp(g * 2.0, 0.0, 1.0);

    // Local variance suppression
    mediump float variance = abs(luma(c) - 0.5 * (luma(a) + luma(b)));
    mediump float suppress = clamp(variance * 2.0, 0.0, 1.0);

    // Unsharp mask
    mediump vec3 meanAB = 0.5 * (a + b);
    mediump vec3 hf     = c - meanAB;

    // Adaptive sharpening strength
    mediump float k = u_Sharpen;
    k *= (1.0 - 0.6 * edgeW);
    k *= (1.0 - 0.3 * clamp(aniso - 1.0, 0.0, 1.0));
    k *= (1.0 - 0.4 * suppress); // suppress in noisy regions

    // Local limiter
    mediump vec3 limit = max(vec3(0.02), 0.75 * abs(c - meanAB));
    mediump vec3 boosted = c + k * clamp(hf, -limit, limit);

    // Softening for ultra-low scale
    if (u_Scale.x < 0.5 || u_Scale.y < 0.5) {
        mediump vec3 blur = 0.25 * (a + b) + 0.5 * c;
        boosted = mix(boosted, blur, 0.2);
    }

    // Clamp to displayable range
    o_FragColor = vec4(clamp(boosted, 0.0, 1.0), 1.0);
    })glsl";

    // Single Pass version
    constexpr const char* kDistortionFragmentShaderTexture2DArray =
    R"glsl(#version 300 es
    precision mediump float;

    uniform sampler2DArray u_TextureArray;
    uniform int   u_SliceIndex;

    uniform vec2  u_Start;
    uniform vec2  u_End;
    uniform vec2  u_Scale;       // low_res_size / full_res_size
    uniform vec2  u_TexelSize;   // 1.0 / low_res_size
    uniform float u_Sharpen;     // 0.0..~0.35

    in vec2 v_TexCoords;
    out vec4 o_FragColor;

    // Luma for edge measure (mediump is fine on Adreno 610)
    mediump float luma(mediump vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }

    void main() {
        // Distortion UVs
        mediump vec2 coords = u_Start + v_TexCoords * (u_End - u_Start);

        // Skip scale if 1:1
        if (u_Scale.x != 1.0 || u_Scale.y != 1.0) {
            coords *= u_Scale;
        }

    // Derivatives for anisotropy and explicit LOD control
    mediump vec2 dxc = dFdx(coords);
    mediump vec2 dyc = dFdy(coords);
    mediump float sx = length(dxc);
    mediump float sy = length(dyc);

    // Small dynamic LOD bias to reduce shimmer in high distortion
    mediump float aniso = max(sx, sy);
    mediump float biasScale = 0.25;
    mediump float bias = clamp((aniso - 1.0) * biasScale, 0.0, 0.5);
    mediump vec2 dxb = dxc * (1.0 + bias);
    mediump vec2 dyb = dyc * (1.0 + bias);

    // Base sample from texture array
    mediump vec3 c = textureGrad(u_TextureArray, vec3(coords, float(u_SliceIndex)), dxb, dyb).rgb;

    // Early out if no sharpening requested
    if (u_Sharpen <= 0.0) {
        o_FragColor = vec4(c, 1.0);
        return;
    }

    // Direction choice for neighbor taps
    bool horiz = (sx >= sy);
    mediump vec2 off = horiz ? vec2(0.75 * u_TexelSize.x, 0.0) : vec2(0.0, 0.75 * u_TexelSize.y);

    // Neighbor samples
    mediump vec3 a = textureGrad(u_TextureArray, vec3(coords + off, float(u_SliceIndex)), dxb, dyb).rgb;
    mediump vec3 b = textureGrad(u_TextureArray, vec3(coords - off, float(u_SliceIndex)), dxb, dyb).rgb;

    // Edge measure
    mediump float g = abs(luma(a) - luma(b));
    mediump float edgeW = clamp(g * 2.0, 0.0, 1.0);

    // Local variance suppression
    mediump float variance = abs(luma(c) - 0.5 * (luma(a) + luma(b)));
    mediump float suppress = clamp(variance * 2.0, 0.0, 1.0);

    // Unsharp mask
    mediump vec3 meanAB = 0.5 * (a + b);
    mediump vec3 hf     = c - meanAB;

    // Adaptive sharpening strength
    mediump float k = u_Sharpen;
    k *= (1.0 - 0.6 * edgeW);
    k *= (1.0 - 0.3 * clamp(aniso - 1.0, 0.0, 1.0));
    k *= (1.0 - 0.4 * suppress); // suppress in noisy regions

    // Local limiter
    mediump vec3 limit = max(vec3(0.02), 0.75 * abs(c - meanAB));
    mediump vec3 boosted = c + k * clamp(hf, -limit, limit);

    // Optional softening for ultra-low scale
    if (u_Scale.x < 0.5 || u_Scale.y < 0.5) {
        mediump vec3 blur = 0.25 * (a + b) + 0.5 * c;
        boosted = mix(boosted, blur, 0.2);
    }

    // Clamp to displayable range
    o_FragColor = vec4(clamp(boosted, 0.0, 1.0), 1.0);
    })glsl";

#ifdef __ANDROID__
constexpr const char* kDistortionFragmentShaderTextureExternalOes =
    R"glsl(
    #extension GL_OES_EGL_image_external : require
    precision mediump float;

    uniform samplerExternalOES u_Texture;
    uniform vec2 u_Start;
    uniform vec2 u_End;
    varying vec2 v_TexCoords;

    void main() {
      vec2 coords = u_Start + v_TexCoords * (u_End - u_Start);
      gl_FragColor = texture2D(u_Texture, coords);
    })glsl";
#endif

void CheckGlError(const char* label) {
  int gl_error = glGetError();
  if (gl_error != GL_NO_ERROR) {
    CARDBOARD_LOGE("GL error %s: %d", label, gl_error);
  }
}

GLuint LoadShader(GLenum shader_type, const char* source) {
  GLuint shader = glCreateShader(shader_type);
  glShaderSource(shader, 1, &source, nullptr);
  glCompileShader(shader);
  CheckGlError("glCompileShader");
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
  CheckGlError("glLinkProgram");

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
  CheckGlError("GlCreateProgram");

  return program;
}

}  // namespace

namespace XRRendering {
    extern int g_full_width;
    extern int g_full_height;
    extern float g_eye_internal_scale;
    extern float g_sharpening_value;
}

namespace cardboard::rendering {

// @brief OpenGL ES 3.0 concrete implementation of DistortionRenderer.
class OpenGlEs3DistortionRenderer : public DistortionRenderer {
 public:
  OpenGlEs3DistortionRenderer(
      const CardboardOpenGlEsDistortionRendererConfig* config)
      : vertices_vbo_{0, 0},
        uvs_vbo_{0, 0},
        elements_vbo_{0, 0},
        elements_count_{0, 0},
        eye_texture_type_{GL_TEXTURE_2D} {
    const char* fragment_shader;

    is_single_pass_ = config->single_pass;
    switch (config->texture_type)
    {
      case kGlTexture2D:
        fragment_shader = config->single_pass
        ? kDistortionFragmentShaderTexture2DArray
        : kDistortionFragmentShaderTexture2D;
        eye_texture_type_ = GL_TEXTURE_2D;
        break;
#ifdef __ANDROID__
      case kGlTextureExternalOes:
        fragment_shader = kDistortionFragmentShaderTextureExternalOes;
        eye_texture_type_ = GL_TEXTURE_EXTERNAL_OES;
        break;
#endif
      default:
        CARDBOARD_LOGE(
            "The Cardboard SDK does not support the selected texture type on "
            "this platform. Setting GL_TEXTURE_2D as default.");

        fragment_shader = kDistortionFragmentShaderTexture2D;
        eye_texture_type_ = GL_TEXTURE_2D;
        break;
    }

    program_ = CreateProgram(kDistortionVertexShader, fragment_shader);
    attrib_pos_ = glGetAttribLocation(program_, "a_Position");
    attrib_tex_ = glGetAttribLocation(program_, "a_TexCoords");
    uniform_start_ = glGetUniformLocation(program_, "u_Start");
    uniform_end_ = glGetUniformLocation(program_, "u_End");

    // New uniforms for upscale/sharpen
    uniform_scale_     = glGetUniformLocation(program_, "u_Scale");      // vec2
    uniform_texel_size_= glGetUniformLocation(program_, "u_TexelSize");  // vec2
    uniform_sharpen_   = glGetUniformLocation(program_, "u_Sharpen");    // float

    if (config->single_pass) {
        uniform_texture_array_ = glGetUniformLocation(program_, "u_TextureArray"); // sampler2DArray
        uniform_slice_index_ = glGetUniformLocation(program_, "u_SliceIndex");   // int
    }
    else
    {
        uniform_texture_ = glGetUniformLocation(program_, "u_Texture");
    }

      // Gen buffers, one per eye.
    glGenBuffers(2, &vertices_vbo_[0]);
    glGenBuffers(2, &uvs_vbo_[0]);
    glGenBuffers(2, &elements_vbo_[0]);
    CheckGlError("OpenGlEs3DistortionRendererSetUp");
  }
  //---------------------------------------------------------------------------

  ~OpenGlEs3DistortionRenderer() {
    glDeleteBuffers(2, &vertices_vbo_[0]);
    glDeleteBuffers(2, &uvs_vbo_[0]);
    glDeleteBuffers(2, &elements_vbo_[0]);
    CheckGlError("~OpenGlEs3DistortionRenderer");
  }
    //---------------------------------------------------------------------------

  /*
   * Modifies the OpenGL global state. In particular:
   *   - glGet(GL_ARRAY_BUFFER_BINDING)
   *   - glGet(GL_ELEMENT_ARRAY_BUFFER_BINDING)
   */
  void SetMesh(const CardboardMesh* mesh, CardboardEye eye) override {
    glBindBuffer(GL_ARRAY_BUFFER, vertices_vbo_[eye]);
    glBufferData(
        GL_ARRAY_BUFFER,
        mesh->n_vertices * sizeof(float) * 2,  // Two components per vertex
        mesh->vertices, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, uvs_vbo_[eye]);
    glBufferData(GL_ARRAY_BUFFER,
                 mesh->n_vertices * sizeof(float) * 2,  // Two components per uv
                 mesh->uvs, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elements_vbo_[eye]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, mesh->n_indices * sizeof(int),
                 mesh->indices, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    CheckGlError("OpenGlEs3DistortionRenderer::SetMesh");
    elements_count_[eye] = mesh->n_indices;
  }
    //---------------------------------------------------------------------------

  /*
   * Modifies the OpenGL global state. In particular:
   *   - glGet(GL_VIEWPORT)
   *   - glGet(GL_FRAMEBUFFER_BINDING)
   *   - glIsEnabled(GL_SCISSOR_TEST)
   *   - glIsEnabled(GL_CULL_FACE)
   *   - glGet(GL_CLEAR_COLOR_VALUE)
   *   - glGet(GL_CURRENT_PROGRAM)
   *   - glGet(GL_SCISSOR_BOX)
   *   - glGet(GL_ACTIVE_TEXTURE+i)
   *   - glGet(GL_ARRAY_BUFFER_BINDING)
   *   - glGet(GL_ELEMENT_ARRAY_BUFFER_BINDING)
   */
    void RenderEyeToDisplay(
            uint64_t target,
            int x, int y,
            int width, int height,
            const CardboardEyeTextureDescription* left_eye,
            const CardboardEyeTextureDescription* right_eye) override
    {
        if (elements_count_[0] == 0 || elements_count_[1] == 0) {
            CARDBOARD_LOGE("Distortion mesh is empty. SetMesh was not called.");
            return;
        }

        //CARDBOARD_LOGE("SIZE X %d y %d w %d, h %d", x, y, width, height);
        glViewport(x, y, width, height);
        glBindFramebuffer(GL_FRAMEBUFFER, static_cast<GLuint>(target));
        glDisable(GL_SCISSOR_TEST);
        glDisable(GL_CULL_FACE);
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

        glUseProgram(program_);

        if (is_single_pass_)
        {
            // Bind the shared texture array once
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D_ARRAY, static_cast<GLuint>(left_eye->texture));
            glUniform1i(uniform_texture_array_, 0);

            // Render left eye (slice 0)
            glDisable(GL_SCISSOR_TEST);
            //glScissor(x, y, width / 2, height);
            glUniform1i(uniform_slice_index_, 0);
            RenderDistortionMesh(left_eye, kLeft);

            // Render right eye (slice 1)
            //glScissor(x + width / 2, y, width / 2, height);
            glUniform1i(uniform_slice_index_, 1);
            RenderDistortionMesh(right_eye, kRight);
        }
        else
        {
            // Left eye
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(left_eye->texture));
            glUniform1i(uniform_texture_, 0);
            RenderDistortionMesh(left_eye, kLeft);

            // Right eye
            glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(right_eye->texture));
            glUniform1i(uniform_texture_, 0);
            RenderDistortionMesh(right_eye, kRight);
        }

        // Cleanup
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
        glDisable(GL_SCISSOR_TEST);
        glFlush(); // Upload & Execute the command buffer!!!!!
        CheckGlError("RenderEyeToDisplay");
    }
    //---------------------------------------------------------------------------

 private:
  /*
   * Modifies the OpenGL global state. In particular:
   *   - glGet(GL_ARRAY_BUFFER_BINDING)
   *   - glGet(GL_ELEMENT_ARRAY_BUFFER_BINDING)
   *   - glGetVertexAttrib(i, GL_VERTEX_ATTRIB_*)
   *   - glGetVertextAttrib(i, GL_VERTEX_ATTRIB_ARRAY_ENABLED)
   *   - glGet(GL_ACTIVE_TEXTURE+i)
   *   - glGet(GL_TEXTURE_BINDING_2D)
   *   - glGetUniform(program, location)
   *   - glGet(GL_ARRAY_BUFFER_BINDING)
   *   - glGet(GL_ELEMENT_ARRAY_BUFFER_BINDING)
   */
  void RenderDistortionMesh(
          const CardboardEyeTextureDescription* eye_description,
          CardboardEye eye) const
  {
      // Bind vertex positions
      glBindBuffer(GL_ARRAY_BUFFER, vertices_vbo_[eye]);
      glVertexAttribPointer(attrib_pos_, 2, GL_FLOAT, false, 0, 0);
      glEnableVertexAttribArray(attrib_pos_);

      // Bind UVs
      glBindBuffer(GL_ARRAY_BUFFER, uvs_vbo_[eye]);
      glVertexAttribPointer(attrib_tex_, 2, GL_FLOAT, false, 0, 0);
      glEnableVertexAttribArray(attrib_tex_);

      glActiveTexture(GL_TEXTURE0);
      if (is_single_pass_)
      {
          // Bind shared texture array and set slice index
          glBindTexture(GL_TEXTURE_2D_ARRAY, static_cast<GLuint>(eye_description->texture));
          glUniform1i(uniform_texture_array_, 0);
          glUniform1i(uniform_slice_index_, (eye == kLeft) ? 0 : 1);
      }
      else
      {
          // Bind 2D texture for this eye
          glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(eye_description->texture));
          glUniform1i(uniform_texture_, 0);
      }

      // Distortion UV remap
      CardboardEyeTextureDescription local = *eye_description;
      local.left_u    = 0.0f;
      local.right_u   = 1.0f;
      local.top_v     = 1.0f;
      local.bottom_v  = 0.0f;

      glUniform2f(uniform_start_, local.left_u, local.bottom_v);
      glUniform2f(uniform_end_,   local.right_u, local.top_v);

      const int small_width  = static_cast<int>(XRRendering::g_full_width * XRRendering::g_eye_internal_scale);
      const int small_height = static_cast<int>(XRRendering::g_full_height * XRRendering::g_eye_internal_scale);

      glUniform2f(uniform_scale_,
                  static_cast<float>(small_width) / static_cast<float>(XRRendering::g_full_width),
                  static_cast<float>(small_height) / static_cast<float>(XRRendering::g_full_height));

      glUniform2f(uniform_texel_size_,
                  1.0f / static_cast<float>(small_width),
                  1.0f / static_cast<float>(small_height));

      glUniform1f(uniform_sharpen_, XRRendering::g_sharpening_value);

      // Draw mesh
      glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elements_vbo_[eye]);
      glDrawElements(GL_TRIANGLE_STRIP, elements_count_[eye], GL_UNSIGNED_INT, 0);
      CheckGlError("OpenGlEs3DistortionRenderer::RenderDistortionMesh");
  }
  //---------------------------------------------------------------------------

  std::array<GLuint, 2> vertices_vbo_;  // One per eye.
  std::array<GLuint, 2> uvs_vbo_;
  std::array<GLuint, 2> elements_vbo_;
  std::array<int, 2> elements_count_;

  GLuint program_;
  GLuint attrib_pos_;
  GLuint attrib_tex_;
  GLuint uniform_start_;
  GLuint uniform_end_;

  GLint uniform_scale_;
  GLint uniform_texel_size_;
  GLint uniform_sharpen_;

  // Single Pass
  GLint uniform_texture_array_;
  GLint uniform_slice_index_;

  // Multi Pass
  GLint uniform_texture_;

  GLenum eye_texture_type_;
  bool is_single_pass_ = false;
};

}  // namespace cardboard::rendering

extern "C" {

CardboardDistortionRenderer* CardboardOpenGlEs3DistortionRenderer_create(
    const CardboardOpenGlEsDistortionRendererConfig* config) {
  if (CARDBOARD_IS_NOT_INITIALIZED() || CARDBOARD_IS_ARG_NULL(config)) {
    return nullptr;
  }
  return reinterpret_cast<CardboardDistortionRenderer*>(
      new cardboard::rendering::OpenGlEs3DistortionRenderer(config));
}

}  // extern "C"
