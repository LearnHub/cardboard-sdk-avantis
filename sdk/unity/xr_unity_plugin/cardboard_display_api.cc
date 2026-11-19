/*
 * Copyright 2021 Google LLC
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
#include "unity/xr_unity_plugin/cardboard_display_api.h"

#include <time.h>
#include <unistd.h>

#include <array>
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>  // NOLINT(build/c++11)
#include <string>
#include <vector>

// The following block makes log macros available for Android and iOS.
#if defined(__ANDROID__)
#include <android/log.h>
#define LOG_TAG "CardboardXRUnity"
#define LOGW(fmt, ...)                                                       \
  __android_log_print(ANDROID_LOG_WARN, LOG_TAG, "[%s : %d] " fmt, __FILE__, \
                      __LINE__, ##__VA_ARGS__)
#define LOGD(fmt, ...)                                                        \
  __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "[%s : %d] " fmt, __FILE__, \
                      __LINE__, ##__VA_ARGS__)
#define LOGE(fmt, ...)                                                        \
  __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, "[%s : %d] " fmt, __FILE__, \
                      __LINE__, ##__VA_ARGS__)
#define LOGF(fmt, ...)                                                        \
  __android_log_print(ANDROID_LOG_FATAL, LOG_TAG, "[%s : %d] " fmt, __FILE__, \
                      __LINE__, ##__VA_ARGS__)
#elif defined(__APPLE__)
#include <CoreFoundation/CFString.h>
#include <CoreFoundation/CoreFoundation.h>
extern "C" {
void NSLog(CFStringRef format, ...);
}
#define LOGW(fmt, ...) \
  NSLog(CFSTR("[%s : %d] " fmt), __FILE__, __LINE__, ##__VA_ARGS__)
#define LOGD(fmt, ...) \
  NSLog(CFSTR("[%s : %d] " fmt), __FILE__, __LINE__, ##__VA_ARGS__)
#define LOGE(fmt, ...) \
  NSLog(CFSTR("[%s : %d] " fmt), __FILE__, __LINE__, ##__VA_ARGS__)
#define LOGF(fmt, ...) \
  NSLog(CFSTR("[%s : %d] " fmt), __FILE__, __LINE__, ##__VA_ARGS__)
#elif
#define LOGW(...)
#define LOGD(...)
#define LOGE(...)
#define LOGF(...)
#endif

#ifdef __ANDROID__
#include "unity/android/reprojection.h"
#endif

namespace XRRendering {
    extern bool is_single_pass;
    extern float g_eye_internal_scale ;
    extern float g_sharpening_value;
    extern int g_texture_width;
    extern int g_texture_height;
}

namespace cardboard::unity {

std::atomic<CardboardDisplayApi::ScreenParams>
    CardboardDisplayApi::unity_screen_params_({0, 0, 0, 0, 0, 0});

std::vector<Renderer::WidgetParams> CardboardDisplayApi::widget_params_;

std::mutex CardboardDisplayApi::widget_mutex_;

std::atomic<bool> CardboardDisplayApi::device_params_changed_(true);

std::atomic<CardboardGraphicsApi> CardboardDisplayApi::selected_graphics_api_(
    kNone);

IUnityInterfaces* CardboardDisplayApi::xr_interfaces_{nullptr};

CardboardDisplayApi::CardboardDisplayApi() {
  switch (selected_graphics_api_) {
    case CardboardGraphicsApi::kOpenGlEs2:
      renderer_ = MakeOpenGlEs2Renderer();
      break;
    case CardboardGraphicsApi::kOpenGlEs3:
      renderer_ = MakeOpenGlEs3Renderer();
      break;
#if defined(__APPLE__)
    case CardboardGraphicsApi::kMetal:
      renderer_ = MakeMetalRenderer(xr_interfaces_);
      break;
#endif
#if defined(__ANDROID__)
    case CardboardGraphicsApi::kVulkan:
      renderer_ = MakeVulkanRenderer(xr_interfaces_);
      break;
#endif
    default:
      LOGF(
          "The Cardboard SDK cannot be initialized given that the selected "
          "Graphics API (%d) is not supported. Please use OpenGL ES 2.0, "
          "OpenGL ES 3.0, Metal or Vulkan.",
          static_cast<int>(selected_graphics_api_));
      break;
  }
}

CardboardDisplayApi::~CardboardDisplayApi() { RenderingResourcesTeardown(); }

void CardboardDisplayApi::ScanDeviceParams() {
  CardboardQrCode_scanQrCodeAndSaveDeviceParams();
}

void CardboardDisplayApi::UpdateDeviceParams(bool isSinglePass)
{
  if (selected_graphics_api_ == kNone) {
    LOGE(
        "Misconfigured Graphics API. Neither OpenGL ES 2.0 nor OpenGL ES 3.0 "
        "nor Metal nor Vulkan was selected.");
    return;
  }

  // Updates the screen size.
  screen_params_ = unity_screen_params_;

  // Get saved device parameters
  uint8_t* data;
  int size;
  CardboardQrCode_getSavedDeviceParams(&data, &size);
  CardboardLensDistortion* lens_distortion;
  if (size == 0) {
    // Loads Cardboard V1 device parameters when no device parameters are
    // available.
    CardboardQrCode_getCardboardV1DeviceParams(&data, &size);
    lens_distortion = CardboardLensDistortion_create(
        data, size, screen_params_.viewport_width,
        screen_params_.viewport_height);
  } else {
    lens_distortion = CardboardLensDistortion_create(
        data, size, screen_params_.viewport_width,
        screen_params_.viewport_height);
    CardboardQrCode_destroy(data);
  }
  device_params_changed_ = false;

  RenderingResourcesSetup(isSinglePass);

  const CardboardOpenGlEsDistortionRendererConfig opengl_distortion_renderer_config{kGlTexture2D, isSinglePass};
  switch (selected_graphics_api_) {
    case CardboardGraphicsApi::kOpenGlEs2:
      distortion_renderer_.reset(CardboardOpenGlEs2DistortionRenderer_create(&opengl_distortion_renderer_config));
      break;
    case CardboardGraphicsApi::kOpenGlEs3:
      // #gles3 - This call is only needed if OpenGL ES 3.0 support is
      // desired. Remove the following line if OpenGL ES 3.0 is not needed.
      distortion_renderer_.reset(CardboardOpenGlEs3DistortionRenderer_create(&opengl_distortion_renderer_config));
      break;
#if defined(__APPLE__)
    case CardboardGraphicsApi::kMetal:
      distortion_renderer_.reset(
          MakeCardboardMetalDistortionRenderer(xr_interfaces_));
      break;
#endif
#if defined(__ANDROID__)
    case CardboardGraphicsApi::kVulkan:
      distortion_renderer_.reset(
          MakeCardboardVulkanDistortionRenderer(xr_interfaces_));
      break;
#endif
    default:
      LOGF(
          "The distortion renderer cannot be initialized given that the "
          "selected Graphics API (%d) is not supported. Please use OpenGL ES "
          "2.0, OpenGL ES 3.0, Metal or Vulkan.",
          static_cast<int>(selected_graphics_api_));
      break;
  }

  CardboardLensDistortion_getDistortionMesh(
      lens_distortion, CardboardEye::kLeft,
      &eye_data_[CardboardEye::kLeft].distortion_mesh);
  CardboardLensDistortion_getDistortionMesh(
      lens_distortion, CardboardEye::kRight,
      &eye_data_[CardboardEye::kRight].distortion_mesh);

  CardboardDistortionRenderer_setMesh(
      distortion_renderer_.get(),
      &eye_data_[CardboardEye::kLeft].distortion_mesh, CardboardEye::kLeft);
  CardboardDistortionRenderer_setMesh(
      distortion_renderer_.get(),
      &eye_data_[CardboardEye::kRight].distortion_mesh, CardboardEye::kRight);

  // Get eye matrices
  CardboardLensDistortion_getEyeFromHeadMatrix(
      lens_distortion, CardboardEye::kLeft,
      eye_data_[CardboardEye::kLeft].eye_from_head_matrix);
  CardboardLensDistortion_getEyeFromHeadMatrix(
      lens_distortion, CardboardEye::kRight,
      eye_data_[CardboardEye::kRight].eye_from_head_matrix);
  CardboardLensDistortion_getFieldOfView(lens_distortion, CardboardEye::kLeft,
                                         eye_data_[CardboardEye::kLeft].fov);
  CardboardLensDistortion_getFieldOfView(lens_distortion, CardboardEye::kRight,
                                         eye_data_[CardboardEye::kRight].fov);

#ifdef __ANDROID__
  // AVN: The reprojection draws the lens meshes itself, on its own thread.
  if (IsReprojecting()) {
    for (const CardboardEye eye : {CardboardEye::kLeft, CardboardEye::kRight}) {
      reprojection::SetLens(eye, eye_data_[eye].distortion_mesh,
                            eye_data_[eye].fov,
                            eye_data_[eye].eye_from_head_matrix);
    }
    reprojection::SetViewport(
        screen_params_.viewport_x, screen_params_.viewport_y,
        screen_params_.viewport_width, screen_params_.viewport_height);
  }
#endif

  CardboardLensDistortion_destroy(lens_distortion);
}

void CardboardDisplayApi::AdvanceTextureSet() {
#ifdef __ANDROID__
  if (IsReprojecting()) {
    SelectTextureSet(reprojection::AcquireSet());
  }
#endif
}

void CardboardDisplayApi::SelectTextureSet(int set) {
  texture_set_ = set;
  render_textures_ = texture_sets_[set];
  eye_data_[CardboardEye::kLeft].texture.texture =
      render_textures_[CardboardEye::kLeft].color_buffer;
  eye_data_[CardboardEye::kRight].texture.texture =
      render_textures_[CardboardEye::kRight].color_buffer;
}

void CardboardDisplayApi::GetEyeMatrices(int eye, float* eye_from_head,
                                         float* fov) {
  std::memcpy(eye_from_head, eye_data_[eye].eye_from_head_matrix,
              sizeof(float) * 16);
  std::memcpy(fov, eye_data_[eye].fov, sizeof(float) * 4);
}

void CardboardDisplayApi::RenderEyesToDisplay() {
  const Renderer::ScreenParams screen_params =
      ScreenParamsToRendererScreenParams(screen_params_);
  renderer_->RenderEyesToDisplay(distortion_renderer_.get(), screen_params,
                                 &eye_data_[CardboardEye::kLeft].texture,
                                 &eye_data_[CardboardEye::kRight].texture);
}

void CardboardDisplayApi::RenderWidgets() {
  std::lock_guard<std::mutex> l(widget_mutex_);
  const Renderer::ScreenParams screen_params =
      ScreenParamsToRendererScreenParams(screen_params_);
  renderer_->RenderWidgets(screen_params, widget_params_);
}

void CardboardDisplayApi::RunRenderingPreProcessing() {
  const Renderer::ScreenParams screen_params =
      ScreenParamsToRendererScreenParams(screen_params_);
  renderer_->RunRenderingPreProcessing(screen_params);
}

void CardboardDisplayApi::RunRenderingPostProcessing() {
  renderer_->RunRenderingPostProcessing();
}

uint64_t CardboardDisplayApi::GetLeftTextureColorBufferId() {
  return render_textures_[CardboardEye::kLeft].color_buffer;
}

uint64_t CardboardDisplayApi::GetRightTextureColorBufferId() {
    return render_textures_[CardboardEye::kRight].color_buffer;
}

uint64_t CardboardDisplayApi::GetLeftTextureDepthBufferId() {
    return render_textures_[CardboardEye::kLeft].depth_buffer;
}

uint64_t CardboardDisplayApi::GetRightTextureDepthBufferId() {
    return render_textures_[CardboardEye::kRight].depth_buffer;
}

void CardboardDisplayApi::GetScreenParams(int* width, int* height) {
  const ScreenParams screen_params = unity_screen_params_;
  *width = screen_params.viewport_width;
  *height = screen_params.viewport_height;
}

void CardboardDisplayApi::SetUnityScreenParams(int screen_width,
                                               int screen_height,
                                               int viewport_x, int viewport_y,
                                               int viewport_width,
                                               int viewport_height) {
  unity_screen_params_ =
      ScreenParams{screen_width, screen_height,  viewport_x,
                   viewport_y,   viewport_width, viewport_height};
}

void CardboardDisplayApi::SetDeviceParametersChanged() {
  device_params_changed_ = true;
}

bool CardboardDisplayApi::GetDeviceParametersChanged() {
  return device_params_changed_;
}

void CardboardDisplayApi::SetWidgetCount(int count) {
  std::lock_guard<std::mutex> l(widget_mutex_);
  widget_params_.resize(count);
}

void CardboardDisplayApi::SetWidgetParams(
    int i, const Renderer::WidgetParams& params) {
  std::lock_guard<std::mutex> l(widget_mutex_);
  if (i < 0 || i >= static_cast<int>(widget_params_.size())) {
    LOGE("SetWidgetParams parameter i=%d, out of bounds (size=%d)", i,
         static_cast<int>(widget_params_.size()));
    return;
  }

  widget_params_[i] = params;
}

void CardboardDisplayApi::SetGraphicsApi(CardboardGraphicsApi graphics_api) {
  selected_graphics_api_ = graphics_api;
}

CardboardGraphicsApi CardboardDisplayApi::GetGraphicsApi() {
  return selected_graphics_api_;
}

void CardboardDisplayApi::SetUnityInterfaces(IUnityInterfaces* xr_interfaces) {
  xr_interfaces_ = xr_interfaces;
}

// @brief Configures rendering resources.
void CardboardDisplayApi::RenderingResourcesSetup(bool isSinglePass)
{
    if (selected_graphics_api_ == kNone) {
        LOGE("Misconfigured Graphics API. Neither OpenGL ES 2.0 nor OpenGL ES 3.0 "
             "nor Metal nor Vulkan was selected.");
        return;
    }

    if (texture_set_count_ != 0) {
        RenderingResourcesTeardown();
    }

    // AVN: One set of eye textures, or several for the reprojection to hold
    // one on the display while Unity draws into another.
    int set_count = 1;
#ifdef __ANDROID__
    if (selected_graphics_api_ == kOpenGlEs3 && reprojection::IsEnabled()) {
        set_count = reprojection::kSwapChainDepth;
    }
#endif
    // Size of the textures, as the renderer takes it: both eyes side by side.
    // AVN: Several sets at the full size of the screen would take several
    // times the memory, so when reprojecting they are only as large as Unity
    // will draw into, which is the same pixels in less.
    int textures_width = screen_params_.viewport_width;
    int textures_height = screen_params_.viewport_height;
    if (set_count > 1) {
        textures_width = 2 * std::max(1, static_cast<int>(
            (textures_width / 2) * XRRendering::g_eye_internal_scale));
        textures_height = std::max(1, static_cast<int>(
            textures_height * XRRendering::g_eye_internal_scale));
    }
    XRRendering::g_texture_width = textures_width / 2;
    XRRendering::g_texture_height = textures_height;

    const bool is_texture_array = selected_graphics_api_ == kOpenGlEs3 && isSinglePass;
    LOGD("%s Rendering - Initialized (%d set(s) of eye textures)",
         is_texture_array ? "Single Pass" : "Multi Pass", set_count);

    for (int set = 0; set < set_count; ++set)
    {
        auto& textures = texture_sets_[set];
        if (is_texture_array)
        {
            // SPI path (GLES3+): single Texture2DArray for color + depth, 2 layers (L/R)
            // IMPORTANT: CreateRenderTextureArray should allocate with per-eye size:
            // layer_count = 2
            renderer_->CreateRenderTextureArray(&textures[CardboardEye::kLeft],
                                                textures_width,
                                                textures_height,
                                                2);

            // Alias right eye to the same array handles (color + depth)
            textures[CardboardEye::kRight] = textures[CardboardEye::kLeft];
        }
        else
        {
            // Original multi-pass fallback (e.g., ES2)
            renderer_->CreateRenderTexture(&textures[CardboardEye::kLeft],
                                           textures_width, textures_height);
            renderer_->CreateRenderTexture(&textures[CardboardEye::kRight],
                                           textures_width, textures_height);
        }
    }
    texture_set_count_ = set_count;
    textures_are_arrays_ = is_texture_array;

    // We don't care about those 2D widgets for phones when single pass
    if (!is_texture_array) {
        renderer_->SetupWidgets();
    }

    // Both eyes reference the same texture array when single pass; the layer
    // is selected at FBO bind (or in instanced draw)
    for (int eye = 0; eye < CardboardEye::kMax; ++eye)
    {
        eye_data_[eye].texture.left_u   = 0;
        eye_data_[eye].texture.right_u  = 1;
        eye_data_[eye].texture.top_v    = 1;
        eye_data_[eye].texture.bottom_v = 0;
    }
    SelectTextureSet(0);
}
//---------------------------------------------------------------------------

void CardboardDisplayApi::RenderingResourcesTeardown() {
    if (texture_set_count_ == 0) return;

#ifdef __ANDROID__
    // AVN: The reprojection must not show anything from textures that are going.
    if (IsReprojecting()) {
        reprojection::ForgetFrames();
    }
#endif

    for (int set = 0; set < texture_set_count_; ++set)
    {
        auto& textures = texture_sets_[set];
        renderer_->DestroyRenderTexture(&textures[CardboardEye::kLeft]);
        // Right eye aliases the array in single-pass — don’t destroy twice
        if (!textures_are_arrays_) {
            renderer_->DestroyRenderTexture(&textures[CardboardEye::kRight]);
        }
        textures[CardboardEye::kLeft] = {};
        textures[CardboardEye::kRight] = {};
    }
    texture_set_count_ = 0;
    texture_set_ = 0;

    renderer_->TeardownWidgets();
    render_textures_[CardboardEye::kLeft] = {};
    render_textures_[CardboardEye::kRight] = {};
}

Renderer::ScreenParams CardboardDisplayApi::ScreenParamsToRendererScreenParams(
    const ScreenParams& screen_params) const {
  return Renderer::ScreenParams{
      screen_params.width,          screen_params.height,
      screen_params.viewport_x,     screen_params.viewport_y,
      screen_params.viewport_width, screen_params.viewport_height};
}

}  // namespace cardboard::unity

#ifdef __cplusplus
extern "C" {
#endif

void CardboardUnity_setScreenParams(int screen_width, int screen_height,
                                    int viewport_x, int viewport_y,
                                    int viewport_width, int viewport_height) {
  cardboard::unity::CardboardDisplayApi::SetUnityScreenParams(
      screen_width, screen_height, viewport_x, viewport_y, viewport_width,
      viewport_height);
}

void CardboardUnity_setDeviceParametersChanged() {
  cardboard::unity::CardboardDisplayApi::SetDeviceParametersChanged();
}

void CardboardUnity_setWidgetCount(int count) {
  cardboard::unity::CardboardDisplayApi::SetWidgetCount(count);
}

void CardboardUnity_setWidgetParams(int i, void* texture, int x, int y,
                                    int width, int height) {
  cardboard::unity::Renderer::WidgetParams params;

  params.texture = reinterpret_cast<intptr_t>(texture);
  params.x = x;
  params.y = y;
  params.width = width;
  params.height = height;
  cardboard::unity::CardboardDisplayApi::SetWidgetParams(i, params);
}

void CardboardUnity_setGraphicsApi(CardboardGraphicsApi graphics_api) {
  switch (graphics_api) {
    case CardboardGraphicsApi::kOpenGlEs2:
      LOGD("Configured OpenGL ES2.0 as Graphics API.");
      cardboard::unity::CardboardDisplayApi::SetGraphicsApi(graphics_api);
      break;
    case CardboardGraphicsApi::kOpenGlEs3:
      LOGD("Configured OpenGL ES3.0 as Graphics API.");
      cardboard::unity::CardboardDisplayApi::SetGraphicsApi(graphics_api);
      break;
#if defined(__APPLE__)
    case CardboardGraphicsApi::kMetal:
      LOGD("Configured Metal as Graphics API.");
      cardboard::unity::CardboardDisplayApi::SetGraphicsApi(graphics_api);
      break;
#endif
#if defined(__ANDROID__)
    case CardboardGraphicsApi::kVulkan:
      LOGD("Configured Vulkan as Graphics API.");
      cardboard::unity::CardboardDisplayApi::SetGraphicsApi(graphics_api);
      break;
#endif
    default:
      LOGE(
          "Misconfigured Graphics API. Neither OpenGL ES 2.0 nor OpenGL ES 3.0 "
          "nor Metal nor Vulkan was selected.");
  }
}
void CardboardUnity_setReprojection(bool enabled) {
#ifdef __ANDROID__
  cardboard::unity::reprojection::SetRequested(enabled);
#endif
}

bool CardboardUnity_isReprojectionPresenting() {
#ifdef __ANDROID__
  return cardboard::unity::reprojection::IsPresenting();
#else
  return false;
#endif
}

void CardboardUnity_setHeadLockedLayer(void* texture, float distance,
                                       float width, float height, float red,
                                       float green, float blue, float alpha,
                                       bool premultiplied_alpha, bool linear) {
#ifdef __ANDROID__
  cardboard::unity::reprojection::HeadLockedLayer layer = {};
  // Under OpenGL ES, Unity's native texture pointer is the texture's name.
  layer.texture = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(texture));
  layer.distance = distance;
  layer.width = width;
  layer.height = height;
  layer.tint[0] = red;
  layer.tint[1] = green;
  layer.tint[2] = blue;
  layer.tint[3] = alpha;
  layer.premultiplied_alpha = premultiplied_alpha;
  layer.linear = linear;
  cardboard::unity::reprojection::SetHeadLockedLayer(layer);
#endif
}

void CardboardUnity_clearHeadLockedLayer() {
#ifdef __ANDROID__
  cardboard::unity::reprojection::ClearHeadLockedLayer();
#endif
}

void CardboardUnity_setRenderParams(bool isSpi, float scale, float sharpening)
{
    XRRendering::is_single_pass = isSpi;
    XRRendering::g_eye_internal_scale = scale;
    XRRendering::g_sharpening_value = sharpening;
}

#ifdef __cplusplus
}
#endif
