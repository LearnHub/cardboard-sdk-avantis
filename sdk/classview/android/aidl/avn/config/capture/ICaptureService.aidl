package avn.config.capture;

import android.view.Surface;
import avn.config.capture.IFrameSource;

/**
 * Screen capture for the ClassVR Client (plans repo: device-comms-realtime.md §4.5 #10).
 * Config holds the capture permission; the Client owns the WebRTC peer and supplies the
 * Surface its video source consumes. One capture at a time: a second start replaces the first.
 * The Client's copy of this file must match byte for byte.
 */
interface ICaptureService {
    /**
     * Render the screen into surface at width x height. clientToken is any Binder owned by the
     * caller: when it dies the capture stops. Returns null on success, else the reason it can't.
     */
    String start(in Surface surface, int width, int height, int frameRate, IBinder clientToken);

    /** Change the output size of the running capture. Returns null on success, else the reason. */
    String resize(int width, int height, int frameRate);

    void stop();

    /** What the capture is doing ("left eye 1024x1024 @36", "eye 2048x1024 @36", "source avn.app.hubs 1024x1024 @36", "stopped"). */
    String describe();

    /**
     * Offer to render the capture from inside the foreground app (see IFrameSource). token is any
     * Binder owned by the caller: when it dies, or on unregisterSource, the service goes back to
     * its own mirror. A capture already running is handed over at once.
     */
    void registerSource(IFrameSource source, IBinder token);

    void unregisterSource(IBinder token);

    /**
     * Width over height of the picture this headset captures. The caller picks a height and sizes
     * its Surface width to match: 1.0 where one eye is a square (the left eye of the side-by-side
     * panels, every frame source), 16:9 on the CVR6, whose Spaces compositor renders a 16:9 view.
     */
    float aspectRatio();
}
