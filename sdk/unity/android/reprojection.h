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
#ifndef CARDBOARD_SDK_UNITY_ANDROID_REPROJECTION_H_
#define CARDBOARD_SDK_UNITY_ANDROID_REPROJECTION_H_

#include <jni.h>

#include <cstdint>

#include "include/cardboard.h"

/// Reprojection for the Unity plug-in (OpenGL ES 3 on Android only).
///
/// Without it the panel shows a new picture only when Unity finishes a frame,
/// so an app that misses the panel's refresh rate judders on every head turn.
/// With it the panel is drawn by a thread of the plug-in's own on every
/// refresh: it takes the newest eye buffers Unity has finished, reads the head
/// tracker again and turns the picture by however far the head has moved since
/// Unity drew it, in the same pass that applies the lens distortion. The head
/// stays smooth at the panel rate whatever rate the app manages.
///
/// How it is put together:
///   - The Java class avn.cardboard.reprojection.ReprojectionSurface lays a
///     SurfaceView over Unity's own while the plug-in is rendering. Unity
///     keeps its window surface, which cannot be shared, and goes on swapping
///     it underneath.
///   - A thread here owns that overlay, with an EGL context that shares
///     objects with Unity's and asks for high priority, so that its small draw
///     gets in ahead of whatever Unity has queued on the GPU.
///   - Unity renders into a rotating set of eye buffers (kSwapChainDepth) so
///     that one can be on the panel and one finishing on the GPU while a third
///     is being drawn. A GL fence says when a submitted frame may be shown.
///     The sets are allocated at the scaled size Unity draws at, not the full
///     size of the screen, so that three of them take no more than one did.
///
/// Only the head's rotation is made up for: what moves in the scene, and the
/// eyes' own movement with the neck model, still change at Unity's rate.
///
/// It is a big switch, decided once when the XR display starts: the setting
/// from the Unity project, unless the device overrides it (see
/// ReprojectionSurface.deviceSetting). When it is off, or the overlay cannot be
/// set up, the plug-in presents through Unity as it always has.
namespace cardboard::unity::reprojection {

/// How many sets of eye buffers Unity rotates through when reprojecting.
constexpr int kSwapChainDepth = 3;

/// Sets what the Unity project asks for. Call before the XR display starts.
void SetRequested(bool requested);

/// Hands the Unity activity to the Java side and reads the device's override.
/// Call from a thread whose class loader can see the app's classes (Unity's
/// main thread).
void InitializeAndroid(JavaVM* vm, jobject activity);

/// Whether reprojection is switched on for this run. The caller must also be
/// rendering with OpenGL ES 3.
bool IsEnabled();

/// Whether the overlay is on the panel. While it is, the caller must not draw
/// the eyes to Unity's own surface.
bool IsPresenting();

/// Gives the head tracker to read at display time, or nullptr when it is
/// going away. Blocks while a read is in progress.
void SetHeadTracker(CardboardHeadTracker* head_tracker);

/// Sets the viewport orientation the head tracker's poses are asked for in.
void SetViewportOrientation(CardboardViewportOrientation orientation);

/// Notes the orientation Unity is about to render with: the head tracker's
/// quaternion (x, y, z, w) exactly as CardboardHeadTracker_getPose gave it.
/// Call on Unity's main thread from the input provider.
/// @param before_render Whether this is Unity's update just before rendering,
///        which is the one the cameras use.
/// @return true when the self-test (see reprojection.cc) has replaced
///         @p orientation with its own. The caller should then keep the head
///         at the origin, as a turn that also moves the eyes, which the neck
///         model does, cannot be undone exactly by turning the picture.
bool OnHeadPose(float* orientation, bool before_render);

/// Copies the lens mesh and field of view of an eye. Call on Unity's render
/// thread whenever the device parameters are loaded.
/// @param fov [left, right, bottom, top] half angles in radians.
/// @param eye_from_head The eye's 4x4 column-major matrix from head space,
///        for where the eye sits (see HeadLockedLayer).
void SetLens(CardboardEye eye, const CardboardMesh& mesh, const float* fov,
             const float* eye_from_head);

/// A flat picture fixed to the head: a rectangle straight ahead of the point
/// midway between the eyes, facing it. What Unity draws into the eye buffers
/// is turned with the head between Unity's frames, so anything meant to stay
/// in the middle of the view, a reticle above all, shakes when Unity is slow.
/// This is drawn after that turn, at the panel's rate, and over everything.
/// Each eye sees it from where that eye sits, so it appears at its distance.
/// It is OpenXR's quad layer in view space, with one layer and no pose.
struct HeadLockedLayer {
  /// Name of a GL_TEXTURE_2D in Unity's context. It is read from the frame
  /// after this is set onwards; a picture that changes should be drawn into a
  /// different texture each time, not over the one being shown.
  uint32_t texture;
  /// How far ahead the rectangle is, and its size, in metres.
  float distance;
  float width;
  float height;
  /// Multiplies the picture: red, green, blue and opacity.
  float tint[4];
  /// Whether the colours are already multiplied by the opacity, as they are
  /// in a texture a camera has rendered over transparent black.
  bool premultiplied_alpha;
  /// Whether sampling gives linear light, as in a Unity project in linear
  /// colour space, to be encoded for the panel here.
  bool linear;
};

/// Shows @p layer from Unity's next frame on. It must be set again at least
/// every few frames (kHeadLockedLayerFrames) or it goes, so that it cannot
/// outlive an app that has stopped drawing it. Any thread.
void SetHeadLockedLayer(const HeadLockedLayer& layer);

/// Takes the layer away from Unity's next frame on. Call before the texture
/// is destroyed. Any thread.
void ClearHeadLockedLayer();

/// How many of Unity's frames a layer lasts without being set again.
constexpr int kHeadLockedLayerFrames = 3;

/// Sets the rectangle of the screen that the lens meshes cover, in pixels.
void SetViewport(int x, int y, int width, int height);

/// Picks the set of eye buffers Unity should draw the next frame into: one
/// that is neither on the panel nor waiting to go there.
/// @return An index in [0, kSwapChainDepth).
int AcquireSet();

/// A frame Unity has finished issuing draw calls for.
struct Frame {
  /// Which set of eye buffers it is in, as AcquireSet() gave it.
  int set;
  /// GL texture names, left then right. For @c is_texture_array both are the
  /// same two-layer GL_TEXTURE_2D_ARRAY, else each is a GL_TEXTURE_2D.
  uint32_t textures[2];
  bool is_texture_array;
  /// The part of the texture, from its origin, that holds the eye.
  float used_u;
  float used_v;
  /// Size of that part in pixels.
  int width;
  int height;
  /// Sharpening strength, as the distortion pass takes it.
  float sharpen;
};

/// Call on Unity's render thread, in its EGL context, when a frame's draw
/// calls have all been issued. Starts the overlay on the first call.
void SubmitFrame(const Frame& frame);

/// Call on Unity's render thread before the eye buffers are deleted: nothing
/// is shown from them afterwards.
void ForgetFrames();

/// Call on Unity's render thread when it stops submitting frames. Takes the
/// overlay down.
void OnRenderingStopped();

}  // namespace cardboard::unity::reprojection

#endif  // CARDBOARD_SDK_UNITY_ANDROID_REPROJECTION_H_
