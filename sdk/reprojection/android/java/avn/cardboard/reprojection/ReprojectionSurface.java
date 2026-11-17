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
package avn.cardboard.reprojection;

import android.app.Activity;
import android.content.Context;
import android.os.Handler;
import android.os.Looper;
import android.provider.Settings;
import android.util.Log;
import android.view.Surface;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.ViewGroup;
import android.view.ViewParent;
import com.google.cardboard.sdk.UsedByNative;

/**
 * The surface the Cardboard plug-in's reprojection draws the panel's picture on: a SurfaceView
 * laid over Unity's own for as long as the plug-in is rendering. Unity's window surface cannot be
 * shared with another thread, so the reprojection, which has to draw on every refresh of the panel
 * whether or not Unity has a new frame, is given this one instead. See {@code reprojection.h}.
 *
 * <p>The view takes no input: it is neither focusable nor clickable, so keys and touches go to
 * Unity's view underneath as before.
 */
public final class ReprojectionSurface implements SurfaceHolder.Callback {
  private static final String TAG = "ReprojectionSurface";

  /**
   * System setting that overrides what the Unity project asks for: 1 turns reprojection on, 0
   * turns it off, and anything else, or no setting, leaves the choice to the project. Read once as
   * the app starts. For example: {@code adb shell settings put system
   * CLASSVR_CARDBOARD_REPROJECTION 1}.
   */
  private static final String DEVICE_SETTING = "CLASSVR_CARDBOARD_REPROJECTION";

  private static ReprojectionSurface instance;

  private final Activity activity;
  private final Handler uiHandler = new Handler(Looper.getMainLooper());

  // UI thread only
  private SurfaceView view;

  private ReprojectionSurface(Activity activity) {
    this.activity = activity;
  }

  /** Called once the Cardboard SDK has the Unity activity. Later calls are ignored. */
  @UsedByNative
  public static synchronized void attach(Context activityContext) {
    if (instance != null) {
      return;
    }
    if (!(activityContext instanceof Activity)) {
      Log.w(TAG, "No activity to draw over: no reprojection");
      return;
    }
    instance = new ReprojectionSurface((Activity) activityContext);
  }

  /** The device's override of the project's setting: 1 on, 0 off, -1 none. */
  @UsedByNative
  public static int deviceSetting(Context context) {
    int value = Settings.System.getInt(context.getContentResolver(), DEVICE_SETTING, -1);
    return (value == 0 || value == 1) ? value : -1;
  }

  /** The refresh rate of the display the activity is on, in Hz. */
  @UsedByNative
  public static float refreshRate(Context context) {
    if (!(context instanceof Activity)) {
      return 0;
    }
    return ((Activity) context).getWindowManager().getDefaultDisplay().getRefreshRate();
  }

  /**
   * Puts the overlay up or takes it down. Called from Unity's render thread when the plug-in
   * starts and stops submitting frames.
   */
  @UsedByNative
  public static void setActive(boolean active) {
    ReprojectionSurface surface;
    synchronized (ReprojectionSurface.class) {
      surface = instance;
    }
    if (surface != null) {
      surface.uiHandler.post(active ? surface::show : surface::hide);
    }
  }

  private void show() {
    if (view != null) {
      return;
    }
    view = new SurfaceView(activity);
    // Above Unity's SurfaceView, which is an ordinary one, but still below the window itself
    view.setZOrderMediaOverlay(true);
    view.getHolder().addCallback(this);
    activity.addContentView(
        view,
        new ViewGroup.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT));
    Log.i(TAG, "Overlay added");
  }

  private void hide() {
    if (view == null) {
      return;
    }
    // Removing the view destroys its Surface, which calls surfaceDestroyed before this returns
    ViewParent parent = view.getParent();
    if (parent instanceof ViewGroup) {
      ((ViewGroup) parent).removeView(view);
    }
    view.getHolder().removeCallback(this);
    view = null;
    nativeSurfaceDestroyed();
    Log.i(TAG, "Overlay removed");
  }

  @Override
  public void surfaceCreated(SurfaceHolder holder) {}

  @Override
  public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {
    Log.i(TAG, "Overlay surface " + width + "x" + height);
    nativeSurfaceChanged(holder.getSurface());
  }

  @Override
  public void surfaceDestroyed(SurfaceHolder holder) {
    nativeSurfaceDestroyed();
  }

  /** The Surface is there, or has changed size: (re)start drawing on it. */
  private static native void nativeSurfaceChanged(Surface surface);

  /** Returns once nothing native is using the Surface. */
  private static native void nativeSurfaceDestroyed();
}
