package com.falldetection;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Context;
import android.content.Intent;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.IBinder;
import android.telephony.SmsManager;
import android.util.Log;

import androidx.core.app.NotificationCompat;

import org.tensorflow.lite.Interpreter;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.io.FileInputStream;
import java.util.ArrayDeque;
import java.util.Deque;

/**
 * FallDetectionService
 * --------------------
 * Foreground service that:
 *  1. Reads accelerometer + gyroscope via SensorManager
 *  2. Builds sliding windows of WINDOW_SIZE samples
 *  3. Extracts the same 53-feature vector used during training
 *  4. Runs inference with the TFLite model
 *  5. If fall detected → waits CONFIRM_DELAY_MS for no movement
 *     → sends SMS alert and raises notification
 */
public class FallDetectionService extends Service implements SensorEventListener {

    private static final String TAG = "FallDetectionService";

    // Must match values in train_model.py
    private static final int    WINDOW_SIZE       = 100;
    private static final int    STEP_SIZE         = 50;
    private static final float  FALL_THRESHOLD    = 0.75f; // model confidence
    private static final long   CONFIRM_DELAY_MS  = 3000;  // 3 s stillness check
    private static final float  STILLNESS_THRESHOLD = 0.5f; // m/s² variance

    // Notification
    private static final String CHANNEL_ID = "fall_detection_channel";
    private static final int    NOTIF_ID   = 1001;

    // Sensors
    private SensorManager sensorManager;
    private Sensor accelerometer;
    private Sensor gyroscope;

    // Rolling window buffers  [acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z]
    private final Deque<float[]> sensorWindow = new ArrayDeque<>();
    private float[] latestGyro = new float[3];
    private int stepCounter = 0;

    // TFLite
    private Interpreter tfliteInterpreter;
    private ScalerParams scalerParams;

    // State
    private boolean fallPending      = false;
    private long    fallTimestamp    = 0;
    private float[] postFallAccBuffer = new float[0];

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    @Override
    public void onCreate() {
        super.onCreate();
        createNotificationChannel();
        startForeground(NOTIF_ID, buildForegroundNotification("Monitoring active…"));
        initTFLite();
        initSensors();
        Log.i(TAG, "FallDetectionService started");
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        return START_STICKY; // restart automatically if killed
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        sensorManager.unregisterListener(this);
        if (tfliteInterpreter != null) tfliteInterpreter.close();
        Log.i(TAG, "FallDetectionService stopped");
    }

    @Override
    public IBinder onBind(Intent intent) { return null; }

    // ── Sensor Setup ──────────────────────────────────────────────────────────

    private void initSensors() {
        sensorManager = (SensorManager) getSystemService(Context.SENSOR_SERVICE);
        accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        gyroscope     = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE);

        // SENSOR_DELAY_GAME ≈ 50 Hz
        sensorManager.registerListener(this, accelerometer,
                SensorManager.SENSOR_DELAY_GAME);
        if (gyroscope != null) {
            sensorManager.registerListener(this, gyroscope,
                    SensorManager.SENSOR_DELAY_GAME);
        }
    }

    // ── SensorEventListener ───────────────────────────────────────────────────

    @Override
    public void onSensorChanged(SensorEvent event) {
        if (event.sensor.getType() == Sensor.TYPE_GYROSCOPE) {
            latestGyro = event.values.clone();
            return;
        }
        if (event.sensor.getType() != Sensor.TYPE_ACCELEROMETER) return;

        float[] acc = event.values;

        // Post-fall stillness check
        if (fallPending) {
            checkStillness(acc);
            return;
        }

        // Add sample to rolling window
        sensorWindow.addLast(new float[]{
            acc[0], acc[1], acc[2],
            latestGyro[0], latestGyro[1], latestGyro[2]
        });

        if (sensorWindow.size() > WINDOW_SIZE)
            sensorWindow.pollFirst();

        // Run inference every STEP_SIZE samples
        stepCounter++;
        if (stepCounter >= STEP_SIZE && sensorWindow.size() == WINDOW_SIZE) {
            stepCounter = 0;
            float confidence = runInference();
            if (confidence >= FALL_THRESHOLD) {
                Log.w(TAG, "Possible fall detected! Confidence=" + confidence);
                fallPending   = true;
                fallTimestamp = System.currentTimeMillis();
                postFallAccBuffer = new float[0];
                updateNotification("Possible fall — confirming…");
            }
        }
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {}

    // ── Stillness Check ───────────────────────────────────────────────────────

    private void checkStillness(float[] acc) {
        float smv = (float) Math.sqrt(acc[0]*acc[0] + acc[1]*acc[1] + acc[2]*acc[2]);

        // Append to post-fall buffer
        float[] newBuf = new float[postFallAccBuffer.length + 1];
        System.arraycopy(postFallAccBuffer, 0, newBuf, 0, postFallAccBuffer.length);
        newBuf[newBuf.length - 1] = smv;
        postFallAccBuffer = newBuf;

        long elapsed = System.currentTimeMillis() - fallTimestamp;
        if (elapsed >= CONFIRM_DELAY_MS) {
            float variance = computeVariance(postFallAccBuffer);
            if (variance < STILLNESS_THRESHOLD) {
                Log.e(TAG, "FALL CONFIRMED — variance=" + variance);
                onFallConfirmed();
            } else {
                Log.i(TAG, "False alarm — user is moving (variance=" + variance + ")");
                fallPending = false;
                updateNotification("Monitoring active…");
            }
        }
    }

    private float computeVariance(float[] data) {
        if (data.length == 0) return 0f;
        float mean = 0;
        for (float v : data) mean += v;
        mean /= data.length;
        float var = 0;
        for (float v : data) var += (v - mean) * (v - mean);
        return var / data.length;
    }

    // ── Fall Response ─────────────────────────────────────────────────────────

    private void onFallConfirmed() {
        fallPending = false;

        // Update notification with high-priority alert
        NotificationManager nm =
                (NotificationManager) getSystemService(Context.NOTIFICATION_SERVICE);
        Notification alertNotif = new NotificationCompat.Builder(this, CHANNEL_ID)
                .setContentTitle("⚠️ Fall Detected!")
                .setContentText("A fall has been detected. Sending emergency alert.")
                .setSmallIcon(android.R.drawable.ic_dialog_alert)
                .setPriority(NotificationCompat.PRIORITY_HIGH)
                .setCategory(NotificationCompat.CATEGORY_ALARM)
                .setAutoCancel(true)
                .build();
        nm.notify(NOTIF_ID + 1, alertNotif);

        // Send SMS
        sendEmergencySms();

        // Resume monitoring
        updateNotification("Monitoring active…");
    }

    private void sendEmergencySms() {
        // Emergency number is set in SharedPreferences by the user
        String number = getSharedPreferences("fall_prefs", MODE_PRIVATE)
                .getString("emergency_number", "");
        if (number.isEmpty()) {
            Log.w(TAG, "No emergency number configured");
            return;
        }
        try {
            SmsManager sms = SmsManager.getDefault();
            sms.sendTextMessage(number, null,
                    "⚠️ FALL DETECTED! Your contact may have fallen. " +
                    "Please check on them immediately.", null, null);
            Log.i(TAG, "Emergency SMS sent to " + number);
        } catch (Exception e) {
            Log.e(TAG, "Failed to send SMS: " + e.getMessage());
        }
    }

    // ── TFLite Inference ──────────────────────────────────────────────────────

    private void initTFLite() {
        try {
            MappedByteBuffer modelBuffer = loadModelFile("fall_detection.tflite");
            tfliteInterpreter = new Interpreter(modelBuffer);
            scalerParams = ScalerParams.loadFromAssets(this, "scaler_params.json");
            Log.i(TAG, "TFLite model loaded successfully");
        } catch (IOException e) {
            Log.e(TAG, "Failed to load TFLite model: " + e.getMessage());
        }
    }

    private MappedByteBuffer loadModelFile(String fileName) throws IOException {
        android.content.res.AssetFileDescriptor afd =
                getAssets().openFd(fileName);
        FileInputStream fis = new FileInputStream(afd.getFileDescriptor());
        FileChannel fc = fis.getChannel();
        return fc.map(FileChannel.MapMode.READ_ONLY,
                afd.getStartOffset(), afd.getDeclaredLength());
    }

    private float runInference() {
        if (tfliteInterpreter == null || scalerParams == null) return 0f;

        float[][] windowArray = sensorWindow.toArray(new float[0][]);
        float[] features = FeatureExtractor.extract(windowArray);
        float[] scaled   = scalerParams.transform(features);

        ByteBuffer inputBuffer = ByteBuffer
                .allocateDirect(scaled.length * 4)
                .order(ByteOrder.nativeOrder());
        for (float v : scaled) inputBuffer.putFloat(v);

        float[][] output = new float[1][1];
        tfliteInterpreter.run(inputBuffer, output);
        return output[0][0];
    }

    // ── Notification Helpers ──────────────────────────────────────────────────

    private void createNotificationChannel() {
        NotificationChannel channel = new NotificationChannel(
                CHANNEL_ID, "Fall Detection",
                NotificationManager.IMPORTANCE_LOW);
        channel.setDescription("Background fall monitoring service");
        ((NotificationManager) getSystemService(Context.NOTIFICATION_SERVICE))
                .createNotificationChannel(channel);
    }

    private Notification buildForegroundNotification(String text) {
        Intent stopIntent = new Intent(this, FallDetectionService.class);
        stopIntent.setAction("STOP");
        PendingIntent stopPendingIntent = PendingIntent.getService(
                this, 0, stopIntent, PendingIntent.FLAG_IMMUTABLE);

        return new NotificationCompat.Builder(this, CHANNEL_ID)
                .setContentTitle("Fall Detection")
                .setContentText(text)
                .setSmallIcon(android.R.drawable.ic_menu_compass)
                .setOngoing(true)
                .addAction(android.R.drawable.ic_delete, "Stop", stopPendingIntent)
                .build();
    }

    private void updateNotification(String text) {
        Notification notif = buildForegroundNotification(text);
        ((NotificationManager) getSystemService(Context.NOTIFICATION_SERVICE))
                .notify(NOTIF_ID, notif);
    }
}
