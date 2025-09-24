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
#include <array>
#include <cassert>
#include <map>
#include <memory>

#include "util/is_arg_null.h"
#include "unity/xr_provider/load.h"
#include "unity/xr_provider/math_tools.h"
#include "unity/xr_unity_plugin/cardboard_display_api.h"
#include "IUnityInterface.h"
#include "IUnityXRDisplay.h"
#include "IUnityXRTrace.h"
#include "UnitySubsystemTypes.h"

// @def Logs to Unity XR Trace interface @p message.
#define CARDBOARD_DISPLAY_XR_TRACE_LOG(trace, message, ...)          \
  XR_TRACE_LOG(trace, "[CardboardXrDisplayProvider]: " message "\n", \
               ##__VA_ARGS__)

namespace XRRendering {
    // Internal eye texture scale (0.1–1.0). 0.5 means Unity renders each eye at 50%
    // width/height of the normal per-eye size. Adjust via code or expose a JNI setter.
    bool is_single_pass = false;
    float g_eye_internal_scale = 1.0f;
    float g_sharpening_value = 0.0f;
    int g_full_width = 0;
    int g_full_height = 0;

    static inline int ScaleDim(int v, float s)
    {
        return std::max(1, static_cast<int>(v * s));
    }
}

namespace {

// @brief Holds the implementation methods of UnityLifecycleProvider and
//        UnityXRDisplayGraphicsThreadProvider
class CardboardDisplayProvider {
 public:
  CardboardDisplayProvider() {
    if (CARDBOARD_IS_ARG_NULL(xr_interfaces_)) {
      return;
    }
    display_ = xr_interfaces_->Get<IUnityXRDisplayInterface>();
    trace_ = xr_interfaces_->Get<IUnityXRTrace>();
  }

  IUnityXRDisplayInterface* GetDisplay() { return display_; }

  IUnityXRTrace* GetTrace() { return trace_; }

  /// @return false When one of the provided IUnityInterface pointers is
  /// nullptr, otherwise true.
  bool IsValid() const {
    return !CARDBOARD_IS_ARG_NULL(xr_interfaces_) &&
           !CARDBOARD_IS_ARG_NULL(display_) && !CARDBOARD_IS_ARG_NULL(trace_);
  }

  void SetHandle(UnitySubsystemHandle handle) { handle_ = handle; }

  /// @return A reference to the static instance of this class, which is thought
  ///         to be the only one.
  static std::unique_ptr<CardboardDisplayProvider>& GetInstance();

  /// @brief Keeps a copy of @p xr_interfaces and sets it to
  ///        cardboard::unity::CardboardDisplayApi.
  /// @param xr_interfaces A pointer to obtain Unity XR interfaces.
  static void SetUnityInterfaces(IUnityInterfaces* xr_interfaces);

  /// @brief Initializes the display subsystem.
  ///
  /// @details Loads and configures a UnityXRDisplayGraphicsThreadProvider and
  ///          UnityXRDisplayProvider with pointers to `display_provider_`'s
  ///          methods.
  /// @param handle Opaque Unity pointer type passed between plugins.
  /// @return kUnitySubsystemErrorCodeSuccess when the registration is
  ///         successful. Otherwise, a value in UnitySubsystemErrorCode flagging
  ///         the error.
  UnitySubsystemErrorCode Initialize(UnitySubsystemHandle handle) {
    SetHandle(handle);

    // Register for callbacks on the graphics thread.
    UnityXRDisplayGraphicsThreadProvider gfx_thread_provider{};
    gfx_thread_provider.userData = NULL;
    gfx_thread_provider.Start = [](UnitySubsystemHandle, void*, UnityXRRenderingCapabilities* rendering_caps)
            -> UnitySubsystemErrorCode
            {
                // Advertise SPI capability as soon as the Interface is being initialized!
                // Otherwise, Unity will default to Multi-Pass
                rendering_caps->noSinglePassRenderingSupport = !XRRendering::is_single_pass;
                rendering_caps->invalidateRenderStateAfterEachCallback = true;
                rendering_caps->skipPresentToMainScreen = false;
                return GetInstance()->GfxThread_Start(rendering_caps);
    };
    gfx_thread_provider.SubmitCurrentFrame =
        [](UnitySubsystemHandle, void*) -> UnitySubsystemErrorCode {
      return GetInstance()->GfxThread_SubmitCurrentFrame();
    };
    gfx_thread_provider.PopulateNextFrameDesc =
        [](UnitySubsystemHandle, void*,
           const UnityXRFrameSetupHints* frame_hints,
           UnityXRNextFrameDesc* next_frame) -> UnitySubsystemErrorCode {
      return GetInstance()->GfxThread_PopulateNextFrameDesc(frame_hints,
                                                            next_frame);
    };
    gfx_thread_provider.Stop = [](UnitySubsystemHandle,
                                  void*) -> UnitySubsystemErrorCode {
      return GetInstance()->GfxThread_Stop();
    };
    GetInstance()->GetDisplay()->RegisterProviderForGraphicsThread(
        handle, &gfx_thread_provider);

    UnityXRDisplayProvider provider{NULL, NULL, NULL};
    // Note: Setting focusLost to true is a workaround for
    // <a
    // href=https://fogbugz.unity3d.com/default.asp?1427493_c43j6si13dh08epg>Issue
    // #1427493</a> in Unity.
    provider.UpdateDisplayState =
        [](UnitySubsystemHandle, void*,
           UnityXRDisplayState* state) -> UnitySubsystemErrorCode {
      state->focusLost = true;
      return kUnitySubsystemErrorCodeSuccess;
    };
    GetInstance()->GetDisplay()->RegisterProvider(handle, &provider);

    return kUnitySubsystemErrorCodeSuccess;
  }

  UnitySubsystemErrorCode Start() const {
    return kUnitySubsystemErrorCodeSuccess;
  }

  void Stop() const {}

  void Shutdown() const {}
  UnitySubsystemErrorCode GfxThread_Start(UnityXRRenderingCapabilities* rendering_caps) const
  {
    // Report Single Stereo Pass support
    if (cardboard_display_api_->GetGraphicsApi() == CardboardGraphicsApi::kOpenGlEs3)
      rendering_caps->noSinglePassRenderingSupport = !XRRendering::is_single_pass;
    else
      rendering_caps->noSinglePassRenderingSupport = true;
    rendering_caps->invalidateRenderStateAfterEachCallback = true;
    // Unity will swap buffers for us after GfxThread_SubmitCurrentFrame() is
    // executed.
    rendering_caps->skipPresentToMainScreen = false;
    return kUnitySubsystemErrorCodeSuccess;
  }

  UnitySubsystemErrorCode GfxThread_SubmitCurrentFrame() {
    if (!is_initialized_) {
      CARDBOARD_DISPLAY_XR_TRACE_LOG(trace_, "Skip the rendering because Cardboard SDK is uninitialized.");
      return kUnitySubsystemErrorCodeFailure;
    }
    cardboard_display_api_->RunRenderingPreProcessing();
    cardboard_display_api_->RenderEyesToDisplay();
    //cardboard_display_api_->RenderWidgets(); // NOT required
    cardboard_display_api_->RunRenderingPostProcessing(); // Vulkan uses this
    return kUnitySubsystemErrorCodeSuccess;
  }

    UnitySubsystemErrorCode GfxThread_PopulateNextFrameDesc(
            const UnityXRFrameSetupHints* frame_hints,
            UnityXRNextFrameDesc* next_frame)
    {
        static bool once_sp_ = false;
        static bool once_mp_ = false;
        if (frame_hints->appSetup.singlePassRendering)
        {
            if (!once_sp_) {
                CARDBOARD_DISPLAY_XR_TRACE_LOG(trace_, "----------------------------------");
                CARDBOARD_DISPLAY_XR_TRACE_LOG(trace_, "Unity: Single Pass Rendering Mode.");
                CARDBOARD_DISPLAY_XR_TRACE_LOG(trace_, "----------------------------------");
                once_sp_ = true;
                once_mp_ = false;
            }
            return GfxThread_PopulateNextFrameDesc_SinglePass(frame_hints, next_frame);
        }
        else
        {
            if (!once_mp_) {
                CARDBOARD_DISPLAY_XR_TRACE_LOG(trace_, "-----------------------------------------------------------");
                CARDBOARD_DISPLAY_XR_TRACE_LOG(trace_, "Unity: singlePassRendering = false — fallback to Multi-Pass");
                CARDBOARD_DISPLAY_XR_TRACE_LOG(trace_, "-----------------------------------------------------------");
                once_mp_ = true;
                once_sp_ = false;
            }
            return GfxThread_PopulateNextFrameDesc_MultiPass(frame_hints, next_frame);
        }
    }
//---------------------------------------------------------------------------

    UnitySubsystemErrorCode GfxThread_PopulateNextFrameDesc_SinglePass(
            const UnityXRFrameSetupHints* frame_hints,
            UnityXRNextFrameDesc* next_frame)
    {
        if ((frame_hints->changedFlags & kUnityXRFrameSetupHintsChangedTextureResolutionScale) != 0 || !is_initialized_ || cardboard::unity::CardboardDisplayApi::GetDeviceParametersChanged())
        {
            CARDBOARD_DISPLAY_XR_TRACE_LOG(trace_, "Initializes Cardboard API.");
            cardboard_display_api_.reset(new cardboard::unity::CardboardDisplayApi());

            for (auto&& tex : tex_map_) {
                display_->DestroyTexture(handle_, tex.second);
            }
            tex_map_.clear();

            cardboard::unity::CardboardDisplayApi::GetScreenParams(&width_, &height_);
            cardboard_display_api_->UpdateDeviceParams(true);
            is_initialized_ = true;

            // --- Single stereo descriptor ---
            texture_descriptors_.fill({}); // clear both slots
            const int base_eye_w = width_ / 2;
            const int base_eye_h = height_;
            const int scaled_eye_w = XRRendering::ScaleDim(base_eye_w, XRRendering::g_eye_internal_scale);
            const int scaled_eye_h = XRRendering::ScaleDim(base_eye_h, XRRendering::g_eye_internal_scale);
            XRRendering::g_full_width = base_eye_w;
            XRRendering::g_full_height = base_eye_h;

            //CARDBOARD_DISPLAY_XR_TRACE_LOG(trace_, "SIZE w %d, h %d", base_eye_w, base_eye_h);

            texture_descriptors_[0].width              = scaled_eye_w;
            texture_descriptors_[0].height             = scaled_eye_h;
            texture_descriptors_[0].textureArrayLength = 2; // stereo array
            texture_descriptors_[0].depthFormat        = kUnityXRDepthTextureFormat24bitOrGreater;
            texture_descriptors_[0].flags              = 0; // or SRGB/UV origin flags if needed
        }

        // --- Create/bind the stereo array texture ---
        const uint64_t tex_color_id = cardboard_display_api_->GetLeftTextureColorBufferId();
        const uint64_t tex_depth_id = cardboard_display_api_->GetLeftTextureDepthBufferId();

        UnityXRRenderTextureId unity_tex_id = 0;
        const auto found = tex_map_.find(tex_color_id);
        if (found == tex_map_.end()) {
            UnityXRRenderTextureDesc desc = texture_descriptors_[0];
            desc.color.nativePtr = ToVoidPointer(tex_color_id);
            desc.depth.nativePtr = ToVoidPointer(tex_depth_id);
            display_->CreateTexture(handle_, &desc, &unity_tex_id);
            tex_map_[tex_color_id] = unity_tex_id;
        } else {
            unity_tex_id = found->second;
        }

        // --- Collapse to one render pass with two eyes ---
        next_frame->renderPassesCount = 1;
        next_frame->renderPasses[0].textureId = unity_tex_id;
        next_frame->renderPasses[0].renderParamsCount = 2;

        // Fill both eyes' params into the same pass
        for (int eye = 0; eye < 2; ++eye) {
            std::array<float, 4> fov;
            std::array<float, 16> eye_from_head;
            cardboard_display_api_->GetEyeMatrices(eye, eye_from_head.data(), fov.data());

            auto* eye_params = &next_frame->renderPasses[0].renderParams[eye];
            eye_params->deviceAnchorToEyePose = cardboard::unity::CardboardTransformToUnityPose(eye_from_head);
            eye_params->viewportRect = frame_hints->appSetup.renderViewport;
            eye_params->textureArraySlice = (eye == 0) ? 0 : 1;
            ConfigureFieldOfView(fov, &eye_params->projection);
        }

        // One culling pass for both eyes
        next_frame->cullingPasses[0].deviceAnchorToCullingPose = next_frame->renderPasses[0].renderParams[0].deviceAnchorToEyePose;
        next_frame->cullingPasses[0].projection = next_frame->renderPasses[0].renderParams[0].projection;
        next_frame->cullingPasses[0].separation = kCullingSphereDiameter;

        return kUnitySubsystemErrorCodeSuccess;
    }
//---------------------------------------------------------------------------

    UnitySubsystemErrorCode GfxThread_PopulateNextFrameDesc_MultiPass(
      const UnityXRFrameSetupHints* frame_hints,
      UnityXRNextFrameDesc* next_frame)
      {
    // Allocate new color texture descriptors if needed and update device
    // parameters in Cardboard SDK.
    if ((frame_hints->changedFlags & kUnityXRFrameSetupHintsChangedTextureResolutionScale) != 0 || !is_initialized_
    || cardboard::unity::CardboardDisplayApi::GetDeviceParametersChanged())
    {
      // Create a new Cardboard SDK to clear previous truncated initializations
      // or just do it for the first time.
      CARDBOARD_DISPLAY_XR_TRACE_LOG(trace_, "Initializes Cardboard API.");
      cardboard_display_api_.reset(new cardboard::unity::CardboardDisplayApi());
      // Deallocate old textures since we're completely reallocating new
      // textures for Cardboard SDK.
      for (auto&& tex : tex_map_)
      {
        display_->DestroyTexture(handle_, tex.second);
      }
      tex_map_.clear();

      cardboard::unity::CardboardDisplayApi::GetScreenParams(&width_, &height_);
      cardboard_display_api_->UpdateDeviceParams(false);
      is_initialized_ = true;

      // Initialize texture descriptors.
      for (size_t i = 0; i < texture_descriptors_.size(); ++i)
      {
          // Per-eye base dimensions:
          const int base_eye_w = width_ / 2;
          const int base_eye_h = height_;
          const int scaled_eye_w = XRRendering::ScaleDim(base_eye_w, XRRendering::g_eye_internal_scale);
          const int scaled_eye_h = XRRendering::ScaleDim(base_eye_h, XRRendering::g_eye_internal_scale);
          XRRendering::g_full_width = base_eye_w;
          XRRendering::g_full_height = base_eye_h;

          texture_descriptors_[i] = {};
          //texture_descriptors_[i].width = width_ / 2;
          //texture_descriptors_[i].height = height_;
          texture_descriptors_[i].width = scaled_eye_w;
          texture_descriptors_[i].height = scaled_eye_h;
          texture_descriptors_[i].flags = 0;
          texture_descriptors_[i].depthFormat = kUnityXRDepthTextureFormat24bitOrGreater;
      }
    }

    // Setup render passes + texture ids for eye textures and layers.
    for (size_t i = 0; i < texture_descriptors_.size(); ++i)
    {
      // Sets the color texture ID to Unity texture descriptors.
      const uint64_t texture_color_buffer_id =
          i == 0 ? cardboard_display_api_->GetLeftTextureColorBufferId()
                 : cardboard_display_api_->GetRightTextureColorBufferId();
      const uint64_t texture_depth_buffer_id =
          i == 0 ? cardboard_display_api_->GetLeftTextureDepthBufferId()
                 : cardboard_display_api_->GetRightTextureDepthBufferId();

      UnityXRRenderTextureId unity_texture_id = 0;
      const auto found = tex_map_.find(texture_color_buffer_id);
      if (found == tex_map_.end()) {
        UnityXRRenderTextureDesc texture_descriptor = texture_descriptors_[i];
        texture_descriptor.color.nativePtr =
            ToVoidPointer(texture_color_buffer_id);
        texture_descriptor.depth.nativePtr =
            ToVoidPointer(texture_depth_buffer_id);
        display_->CreateTexture(handle_, &texture_descriptor,
                                &unity_texture_id);
        tex_map_[texture_color_buffer_id] = unity_texture_id;
      } else {
        unity_texture_id = found->second;
      }
      next_frame->renderPasses[i].textureId = unity_texture_id;
    }

    {
      auto* left_eye_params = &next_frame->renderPasses[0].renderParams[0];
      auto* right_eye_params = &next_frame->renderPasses[1].renderParams[0];

      for (int i = 0; i < 2; ++i)
      {
        std::array<float, 4> fov;
        std::array<float, 16> eye_from_head;
        cardboard_display_api_->GetEyeMatrices(i, eye_from_head.data(),
                                               fov.data());

        auto* eye_params = i == 0 ? left_eye_params : right_eye_params;
        // Update pose for rendering.
        eye_params->deviceAnchorToEyePose =
            cardboard::unity::CardboardTransformToUnityPose(eye_from_head);
        // Field of view and viewport.
        eye_params->viewportRect = frame_hints->appSetup.renderViewport;
        ConfigureFieldOfView(fov, &eye_params->projection);
      }

      // Configure the culling passes for both eyes.
      // - Left eye: index == 0.
      // - Right eye: index == 1.
      next_frame->renderPasses[0].cullingPassIndex = 0;
      next_frame->cullingPasses[0].deviceAnchorToCullingPose =
          left_eye_params->deviceAnchorToEyePose;
      next_frame->cullingPasses[0].projection = left_eye_params->projection;
      next_frame->cullingPasses[0].separation = kCullingSphereDiameter;

      next_frame->renderPasses[1].cullingPassIndex = 1;
      next_frame->cullingPasses[1].deviceAnchorToCullingPose =
          right_eye_params->deviceAnchorToEyePose;
      next_frame->cullingPasses[1].projection = right_eye_params->projection;
      next_frame->cullingPasses[1].separation = kCullingSphereDiameter;
    }

    // Configure multipass rendering with one pass for each eye.
    next_frame->renderPassesCount = 2;
    next_frame->renderPasses[0].renderParamsCount = 1;
    next_frame->renderPasses[1].renderParamsCount = 1;

    return kUnitySubsystemErrorCodeSuccess;
  }
//---------------------------------------------------------------------------



  UnitySubsystemErrorCode GfxThread_Stop() {
    cardboard_display_api_.reset();
    is_initialized_ = false;
    return kUnitySubsystemErrorCodeSuccess;
  }

 private:
  /// @brief Diameter of the bounding sphere used for culling.
  /// TODO(b/155084408): Properly document this constant value.
  static constexpr float kCullingSphereDiameter = 0.064f;

  /// @brief Converts @p i to a void*
  /// @param i A uint64_t integer to convert to void*.
  /// @return A void* whose value is @p i.
  static void* ToVoidPointer(uint64_t i) { return reinterpret_cast<void*>(i); }

  /// @brief Loads Unity @p projection eye params from Cardboard field of view.
  /// @details Sets Unity @p projection to use half angles as Cardboard reported
  ///          field of view angles.
  /// @param[in] cardboard_fov A float vector containing
  ///            [left, right, bottom, top] angles in radians.
  /// @param[out] projection A Unity projection structure pointer to load.
  static void ConfigureFieldOfView(const std::array<float, 4>& cardboard_fov,
                                   UnityXRProjection* projection) {
    projection->type = kUnityXRProjectionTypeHalfAngles;
    projection->data.halfAngles.bottom = -std::abs(tan(cardboard_fov[2]));
    projection->data.halfAngles.top = std::abs(tan(cardboard_fov[3]));
    projection->data.halfAngles.left = -std::abs(tan(cardboard_fov[0]));
    projection->data.halfAngles.right = std::abs(tan(cardboard_fov[1]));
  }

  /// @brief Points to Unity XR Trace interface.
  IUnityXRTrace* trace_ = nullptr;

  /// @brief Points to Unity XR Display interface.
  IUnityXRDisplayInterface* display_ = nullptr;

  /// @brief Opaque Unity pointer type passed between plugins.
  UnitySubsystemHandle handle_;

  /// @brief Tracks Cardboard API initialization status. It is set to true once
  /// the CardboardDisplayApi::UpdateDeviceParams() is called and returns true.
  bool is_initialized_ = false;

  /// @brief Screen width in pixels.
  int width_;

  /// @brief Screen height in pixels.
  int height_;

  /// @brief Cardboard SDK API wrapper.
  std::unique_ptr<cardboard::unity::CardboardDisplayApi> cardboard_display_api_;

  /// @brief Unity XR texture descriptors.
  std::array<UnityXRRenderTextureDesc, 2> texture_descriptors_{};

  /// @brief Map to link Cardboard API and Unity XR texture IDs.
  std::map<uint64_t, UnityXRRenderTextureId> tex_map_{};

  /// @brief Holds the unique instance of this class. It is accessible via
  ///        GetInstance().
  static std::unique_ptr<CardboardDisplayProvider> display_provider_;

  /// @brief Unity XR interface provider to get the display and trace
  /// interfaces. When using Metal rendering API, its context will be
  /// retrieved as well.
  static IUnityInterfaces* xr_interfaces_;
};

std::unique_ptr<CardboardDisplayProvider>
    CardboardDisplayProvider::display_provider_;

IUnityInterfaces* CardboardDisplayProvider::xr_interfaces_ = nullptr;

std::unique_ptr<CardboardDisplayProvider>&
CardboardDisplayProvider::GetInstance() {
  return display_provider_;
}

void CardboardDisplayProvider::SetUnityInterfaces(
    IUnityInterfaces* xr_interfaces) {
  xr_interfaces_ = xr_interfaces;
  cardboard::unity::CardboardDisplayApi::SetUnityInterfaces(xr_interfaces_);
}

}  // namespace

UnitySubsystemErrorCode LoadDisplay(IUnityInterfaces* xr_interfaces) {
  CardboardDisplayProvider::SetUnityInterfaces(xr_interfaces);
  CardboardDisplayProvider::GetInstance().reset(new CardboardDisplayProvider());
  if (!CardboardDisplayProvider::GetInstance()->IsValid()) {
    return kUnitySubsystemErrorCodeFailure;
  }

  UnityLifecycleProvider display_lifecycle_handler;
  display_lifecycle_handler.userData = NULL;
  display_lifecycle_handler.Initialize = [](UnitySubsystemHandle handle,
                                            void*) -> UnitySubsystemErrorCode {
    return CardboardDisplayProvider::GetInstance()->Initialize(handle);
  };
  display_lifecycle_handler.Start = [](UnitySubsystemHandle,
                                       void*) -> UnitySubsystemErrorCode {
    return CardboardDisplayProvider::GetInstance()->Start();
  };
  display_lifecycle_handler.Stop = [](UnitySubsystemHandle, void*) -> void {
    CardboardDisplayProvider::GetInstance()->Stop();
  };
  display_lifecycle_handler.Shutdown = [](UnitySubsystemHandle, void*) -> void {
    CardboardDisplayProvider::GetInstance()->Shutdown();
  };

  return CardboardDisplayProvider::GetInstance()
      ->GetDisplay()
      ->RegisterLifecycleProvider("Cardboard", "CardboardDisplay",
                                  &display_lifecycle_handler);
}

void UnloadDisplay() { CardboardDisplayProvider::GetInstance().reset(); }
