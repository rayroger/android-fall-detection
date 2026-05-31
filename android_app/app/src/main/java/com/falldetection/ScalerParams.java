package com.falldetection;

import android.content.Context;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.InputStream;
import java.nio.charset.StandardCharsets;

/**
 * ScalerParams
 * ------------
 * Loads the StandardScaler parameters exported by train_model.py
 * (scaler_params.json) and applies z-score normalisation at runtime.
 *
 * JSON format:
 * {
 *   "mean":  [f, f, …],
 *   "scale": [f, f, …],
 *   "n_features": 46
 * }
 */
public class ScalerParams {

    private final float[] mean;
    private final float[] scale;

    private ScalerParams(float[] mean, float[] scale) {
        this.mean  = mean;
        this.scale = scale;
    }

    /**
     * Apply z-score normalisation: (x - mean) / scale
     */
    public float[] transform(float[] features) {
        float[] out = new float[features.length];
        for (int i = 0; i < features.length; i++) {
            out[i] = (features[i] - mean[i]) / (scale[i] + 1e-8f);
        }
        return out;
    }

    /**
     * Load scaler_params.json from Android assets folder.
     */
    public static ScalerParams loadFromAssets(Context context, String fileName)
            throws Exception {
        InputStream is  = context.getAssets().open(fileName);
        byte[] bytes    = is.readAllBytes();
        String json     = new String(bytes, StandardCharsets.UTF_8);
        JSONObject obj  = new JSONObject(json);

        JSONArray meanArr  = obj.getJSONArray("mean");
        JSONArray scaleArr = obj.getJSONArray("scale");
        int n              = meanArr.length();

        float[] mean  = new float[n];
        float[] scale = new float[n];
        for (int i = 0; i < n; i++) {
            mean[i]  = (float) meanArr.getDouble(i);
            scale[i] = (float) scaleArr.getDouble(i);
        }
        return new ScalerParams(mean, scale);
    }
}
