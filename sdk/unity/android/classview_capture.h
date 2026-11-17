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
#ifndef CARDBOARD_SDK_UNITY_ANDROID_CLASSVIEW_CAPTURE_H_
#define CARDBOARD_SDK_UNITY_ANDROID_CLASSVIEW_CAPTURE_H_

#include <jni.h>

#include <cstdint>

/// ClassView capture drawn by the plug-in (OpenGL ES 3 only).
///
/// The ClassVR Config app can hand the app in front the Surface that feeds the
/// ClassView video stream, so that the app draws one undistorted eye into it
/// instead of Config mirroring the display, which costs a GPU-bound app a vsync
/// per frame. The Java class avn.cardboard.classview.ClassViewFrameSource does
/// the registration with Config and forwards its start/resize/stop calls here;
/// this side owns the EGL window surface on that Surface and draws into it from
/// Unity's render thread.
namespace cardboard::unity::classview {

/// Hands the Unity activity to the Java side. Call from a thread whose class
/// loader can see the app's classes (Unity's main thread).
void InitializeAndroid(JavaVM* vm, jobject activity);

/// The left eye as Unity has just rendered it.
struct EyeFrame {
  /// GL texture name: a two-layer GL_TEXTURE_2D_ARRAY (left eye in layer 0)
  /// when @c is_texture_array, else a GL_TEXTURE_2D.
  uint32_t texture;
  bool is_texture_array;
  /// The part of the texture, from its origin, that holds the eye.
  float used_u;
  float used_v;
  /// Size of that part in pixels.
  int width;
  int height;
};

/// Call on Unity's render thread, in its EGL context, once the frame has been
/// drawn to the display. Draws the capture when one is running and this frame
/// is due. Changes the current EGL surface for the duration and restores it,
/// along with the framebuffer binding and viewport; other GL state is left
/// changed, so the caller must have asked Unity to invalidate its state cache.
void OnFrameSubmitted(const EyeFrame& frame);

/// Call on Unity's render thread when it stops submitting frames. Ends any
/// capture and withdraws the app as a capture source.
void OnRenderingStopped();

}  // namespace cardboard::unity::classview

#endif  // CARDBOARD_SDK_UNITY_ANDROID_CLASSVIEW_CAPTURE_H_
