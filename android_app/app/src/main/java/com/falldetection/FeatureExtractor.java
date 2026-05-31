package com.falldetection;

import java.util.Arrays;

/**
 * FeatureExtractor
 * ----------------
 * Mirrors the extract_window_features() function in train_model.py.
 * Must produce EXACTLY the same feature vector the model was trained on.
 *
 * Input:  float[WINDOW_SIZE][6]  → [acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z]
 * Output: float[53]              → statistical + SMV features
 */
public class FeatureExtractor {

    // 6 channels × 7 stats + 4 SMV features = 46 features
    // (matches train_model.py extract_window_features)
    public static final int FEATURE_COUNT = 46;

    /**
     * Extract features from a sensor window.
     *
     * @param window float[WINDOW_SIZE][6] sensor readings
     * @return float[FEATURE_COUNT] feature vector
     */
    public static float[] extract(float[][] window) {
        int rows = window.length;
        int cols = window[0].length; // 6

        float[] features = new float[FEATURE_COUNT];
        int idx = 0;

        // Per-channel statistics (7 stats × 6 channels = 42)
        for (int c = 0; c < cols; c++) {
            float[] col = new float[rows];
            for (int r = 0; r < rows; r++) col[r] = window[r][c];

            features[idx++] = mean(col);
            features[idx++] = std(col);
            features[idx++] = min(col);
            features[idx++] = max(col);
            features[idx++] = max(col) - min(col);          // range
            features[idx++] = percentile(col, 25);
            features[idx++] = percentile(col, 75);
        }

        // SMV features (4)
        float[] smv = new float[rows];
        for (int r = 0; r < rows; r++) {
            float ax = window[r][0], ay = window[r][1], az = window[r][2];
            smv[r] = (float) Math.sqrt(ax*ax + ay*ay + az*az);
        }
        features[idx++] = max(smv);
        features[idx++] = mean(smv);
        features[idx++] = std(smv);
        features[idx]   = (float) argmax(smv) / rows;  // normalised peak position

        return features;
    }

    // ── Math Helpers ──────────────────────────────────────────────────────────

    static float mean(float[] a) {
        float sum = 0;
        for (float v : a) sum += v;
        return sum / a.length;
    }

    static float std(float[] a) {
        float m = mean(a);
        float s = 0;
        for (float v : a) s += (v - m) * (v - m);
        return (float) Math.sqrt(s / a.length);
    }

    static float min(float[] a) {
        float m = a[0];
        for (float v : a) if (v < m) m = v;
        return m;
    }

    static float max(float[] a) {
        float m = a[0];
        for (float v : a) if (v > m) m = v;
        return m;
    }

    static int argmax(float[] a) {
        int idx = 0;
        for (int i = 1; i < a.length; i++) if (a[i] > a[idx]) idx = i;
        return idx;
    }

    /**
     * Nearest-rank percentile (matches numpy default).
     */
    static float percentile(float[] a, int p) {
        float[] sorted = Arrays.copyOf(a, a.length);
        Arrays.sort(sorted);
        int idx = (int) Math.ceil(p / 100.0 * sorted.length) - 1;
        idx = Math.max(0, Math.min(idx, sorted.length - 1));
        return sorted[idx];
    }
}
