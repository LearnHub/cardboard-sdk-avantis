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
package avn.cardboard.classview;

import android.app.Activity;
import android.app.Application;
import android.content.ComponentName;
import android.content.Context;
import android.content.Intent;
import android.content.ServiceConnection;
import android.os.Binder;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.IBinder;
import android.util.Log;
import android.view.Surface;
import avn.config.capture.ICaptureService;
import avn.config.capture.IFrameSource;
import com.google.cardboard.sdk.UsedByNative;

/**
 * Registers this app with the ClassVR Config app's capture service as an {@link IFrameSource}, so
 * that a ClassView capture is drawn by the Cardboard plug-in itself, one undistorted eye into the
 * Surface Config hands over, instead of through Config's virtual display mirror. The mirror costs
 * the app in front a vsync per frame when the GPU is busy; drawing the capture here does not.
 *
 * <p>The app is offered as a source only while its activity is resumed and the plug-in is
 * submitting frames. Config falls back to its mirror whenever no source is registered, a source
 * refuses a capture, or the source's process dies, so nothing here can leave ClassView without a
 * picture.
 *
 * <p>Binding and registration run on a thread of their own: Config answers a registration by
 * calling {@link IFrameSource#start} straight back on a binder thread. The native side takes those
 * calls directly; see {@code classview_capture.cc}.
 *
 * <p>The AIDL files under {@code avn/config/capture} must match the Config app's copies byte for
 * byte.
 */
public final class ClassViewFrameSource {
  private static final String TAG = "ClassViewFrameSource";
  private static final String CONFIG_PACKAGE = "co.learnpad.config";
  private static final String CAPTURE_SERVICE = "avn.config.capture.CaptureService";

  private static ClassViewFrameSource instance;

  private final Context context;
  private final Handler handler;
  private final Binder token = new Binder();
  private final float displayRefreshRate;

  // Handler thread only
  private ICaptureService service;
  private boolean bound;
  private boolean registered;
  private boolean resumed = true;
  private boolean rendering;

  private final IFrameSource.Stub stub =
      new IFrameSource.Stub() {
        @Override
        public String start(Surface surface, int width, int height, int frameRate) {
          if (surface == null || !surface.isValid()) {
            return "surface is not valid";
          }
          int divisor = fitDivisor(frameRate);
          String error = nativeStart(surface, width, height, divisor);
          if (error == null) {
            Log.i(TAG, "Capture " + width + "x" + height + " every " + divisor + " frame(s) of "
                + Math.round(displayRefreshRate) + " Hz for " + frameRate + " fps");
          } else {
            Log.w(TAG, "Capture refused: " + error);
          }
          return error;
        }

        @Override
        public String resize(int width, int height, int frameRate) {
          return nativeResize(width, height, fitDivisor(frameRate));
        }

        @Override
        public void stop() {
          nativeStop();
        }
      };

  private final ServiceConnection connection =
      new ServiceConnection() {
        @Override
        public void onServiceConnected(ComponentName name, IBinder binder) {
          handler.post(
              () -> {
                service = ICaptureService.Stub.asInterface(binder);
                update();
              });
        }

        @Override
        public void onServiceDisconnected(ComponentName name) {
          // Config died: its mirror and our registration went with it. The system rebinds when it
          // is back and onServiceConnected registers again.
          handler.post(
              () -> {
                service = null;
                registered = false;
                nativeStop();
              });
        }
      };

  private ClassViewFrameSource(Activity activity) {
    context = activity.getApplicationContext();
    displayRefreshRate = activity.getWindowManager().getDefaultDisplay().getRefreshRate();
    HandlerThread thread = new HandlerThread(TAG);
    thread.start();
    handler = new Handler(thread.getLooper());

    activity
        .getApplication()
        .registerActivityLifecycleCallbacks(
            new Application.ActivityLifecycleCallbacks() {
              @Override
              public void onActivityResumed(Activity resumedActivity) {
                if (resumedActivity == activity) {
                  setResumed(true);
                }
              }

              @Override
              public void onActivityPaused(Activity pausedActivity) {
                if (pausedActivity == activity) {
                  setResumed(false);
                }
              }

              @Override
              public void onActivityCreated(Activity a, Bundle savedInstanceState) {}

              @Override
              public void onActivityStarted(Activity a) {}

              @Override
              public void onActivityStopped(Activity a) {}

              @Override
              public void onActivitySaveInstanceState(Activity a, Bundle outState) {}

              @Override
              public void onActivityDestroyed(Activity a) {}
            });
  }

  /**
   * Called once the Cardboard SDK has the Unity activity. Later calls, for example when XR is
   * initialized again, are ignored.
   */
  @UsedByNative
  public static synchronized void attach(Context activityContext) {
    if (instance != null) {
      return;
    }
    if (!(activityContext instanceof Activity)) {
      Log.w(TAG, "No activity to follow: ClassView capture stays with the Config mirror");
      return;
    }
    instance = new ClassViewFrameSource((Activity) activityContext);
  }

  /** Called from the render thread when the plug-in starts and stops submitting frames. */
  @UsedByNative
  public static void setRendering(boolean rendering) {
    ClassViewFrameSource source;
    synchronized (ClassViewFrameSource.class) {
      source = instance;
    }
    if (source != null) {
      source.handler.post(
          () -> {
            source.rendering = rendering;
            source.update();
          });
    }
  }

  private void setResumed(boolean nowResumed) {
    handler.post(
        () -> {
          resumed = nowResumed;
          update();
        });
  }

  /** Brings the binding and the registration in line with the state. Handler thread only. */
  private void update() {
    boolean wanted = resumed && rendering;
    if (wanted) {
      if (!bound) {
        Intent intent = new Intent().setComponent(new ComponentName(CONFIG_PACKAGE, CAPTURE_SERVICE));
        try {
          bound = context.bindService(intent, connection, Context.BIND_AUTO_CREATE);
        } catch (RuntimeException e) {
          Log.w(TAG, "Bind failed: " + e);
        }
        if (!bound) {
          Log.i(TAG, "No Config capture service on this device");
        }
      } else if (service != null && !registered) {
        try {
          service.registerSource(stub, token);
          registered = true;
          Log.i(TAG, "Registered as the ClassView capture source");
        } catch (Exception e) {
          Log.w(TAG, "Register failed: " + e);
        }
      }
    } else {
      if (registered && service != null) {
        try {
          service.unregisterSource(token);
          Log.i(TAG, "Unregistered as the ClassView capture source");
        } catch (Exception e) {
          Log.w(TAG, "Unregister failed: " + e);
        }
      }
      registered = false;
      // Config stops a running source before it forgets it; make sure of it in case it could not
      nativeStop();
    }
  }

  /**
   * The frame rate is a ceiling. Draw every Nth frame for the largest whole rate under it that
   * divides the panel rate evenly, as Config's mirror and the fframe apps do, so that the cadence
   * of the capture is regular: a 36 fps request on a 72 Hz panel is every 2nd frame.
   */
  private int fitDivisor(int maxFrameRate) {
    if (maxFrameRate <= 0) {
      return 1;
    }
    for (int divisor = 1; divisor <= 12; divisor++) {
      float rate = displayRefreshRate / divisor;
      if (rate <= maxFrameRate + 0.05f && Math.abs(rate - Math.round(rate)) < 0.05f) {
        return divisor;
      }
    }
    return Math.max(1, (int) Math.ceil(displayRefreshRate / maxFrameRate));
  }

  /** Returns null on success, else the reason the capture cannot start. */
  private static native String nativeStart(Surface surface, int width, int height, int divisor);

  /** Returns null on success, else the reason. */
  private static native String nativeResize(int width, int height, int divisor);

  /** Returns once the Surface passed to nativeStart has been disconnected. */
  private static native void nativeStop();
}
