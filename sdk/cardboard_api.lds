VERS_1.0 {
  global:
    # Export Cardboard Symbols.
    Cardboard*;
    *nativeIncrementDeviceParamsChangedCount;
    *JNIEnv*;
    *JavaVM*;
    JNI_OnLoad;
    *Unity*;
    RenderAPI_Vulkan_OnPluginLoad;
    # AVN: ClassView capture source (avn.cardboard.classview.ClassViewFrameSource).
    Java_avn_cardboard_classview_*;
    # AVN: Reprojection overlay (avn.cardboard.reprojection.ReprojectionSurface).
    Java_avn_cardboard_reprojection_*;

  # Hide everything else.
  local:
    *;
};
