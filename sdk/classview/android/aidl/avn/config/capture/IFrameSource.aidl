package avn.config.capture;

import android.view.Surface;

/**
 * A foreground app that can render the capture itself, registered with ICaptureService
 * (plans repo: device-comms-realtime.md §4.5 #10). While a source is registered the
 * capture service hands it the Client's Surface instead of opening a virtual display mirror,
 * which on some headsets costs the app's own compositor a vsync per frame under GPU load.
 * The app draws one undistorted eye into the Surface at the given size and rate.
 * Implementations must return from stop() only once the Surface has been disconnected, so the
 * service can connect the mirror to it. Every app's copy of this file must match byte for byte.
 */
interface IFrameSource {
    /** Begin rendering into surface at width x height. Returns null on success, else the reason it can't. */
    String start(in Surface surface, int width, int height, int frameRate);

    /** Change the output size or rate of the running capture. Returns null on success, else the reason. */
    String resize(int width, int height, int frameRate);

    /** Stop rendering and disconnect from the Surface before returning. */
    void stop();
}
