package com.falldetection;

import android.Manifest;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.widget.Button;
import android.widget.EditText;
import android.widget.Switch;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

/**
 * MainActivity
 * ------------
 * Simple UI to:
 *  - Enter emergency contact number
 *  - Start / stop FallDetectionService
 *  - Show current monitoring status
 */
public class MainActivity extends AppCompatActivity {

    private static final int PERMISSION_REQUEST_CODE = 100;

    private EditText  emergencyNumberInput;
    private Switch    monitoringSwitch;
    private TextView  statusText;
    private Button    saveButton;

    private SharedPreferences prefs;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        prefs = getSharedPreferences("fall_prefs", MODE_PRIVATE);

        emergencyNumberInput = findViewById(R.id.emergencyNumber);
        monitoringSwitch     = findViewById(R.id.monitoringSwitch);
        statusText           = findViewById(R.id.statusText);
        saveButton           = findViewById(R.id.saveButton);

        // Restore saved number
        String savedNumber = prefs.getString("emergency_number", "");
        emergencyNumberInput.setText(savedNumber);

        saveButton.setOnClickListener(v -> saveSettings());

        monitoringSwitch.setOnCheckedChangeListener((btn, isChecked) -> {
            if (isChecked) {
                requestPermissionsAndStart();
            } else {
                stopMonitoring();
            }
        });
    }

    private void saveSettings() {
        String number = emergencyNumberInput.getText().toString().trim();
        prefs.edit().putString("emergency_number", number).apply();
        Toast.makeText(this, "Settings saved", Toast.LENGTH_SHORT).show();
    }

    private void requestPermissionsAndStart() {
        String[] perms = {
            Manifest.permission.SEND_SMS,
            Manifest.permission.BODY_SENSORS,
            Manifest.permission.FOREGROUND_SERVICE,
        };

        boolean allGranted = true;
        for (String p : perms) {
            if (ContextCompat.checkSelfPermission(this, p)
                    != PackageManager.PERMISSION_GRANTED) {
                allGranted = false;
                break;
            }
        }

        if (allGranted) {
            startMonitoring();
        } else {
            ActivityCompat.requestPermissions(this, perms, PERMISSION_REQUEST_CODE);
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode,
            @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == PERMISSION_REQUEST_CODE) {
            boolean allGranted = true;
            for (int result : grantResults) {
                if (result != PackageManager.PERMISSION_GRANTED) {
                    allGranted = false;
                    break;
                }
            }
            if (allGranted) {
                startMonitoring();
            } else {
                monitoringSwitch.setChecked(false);
                Toast.makeText(this,
                        "Permissions required for fall detection & SMS alerts",
                        Toast.LENGTH_LONG).show();
            }
        }
    }

    private void startMonitoring() {
        String number = emergencyNumberInput.getText().toString().trim();
        if (number.isEmpty()) {
            Toast.makeText(this, "Please enter an emergency contact number first",
                    Toast.LENGTH_LONG).show();
            monitoringSwitch.setChecked(false);
            return;
        }
        saveSettings();
        Intent serviceIntent = new Intent(this, FallDetectionService.class);
        startForegroundService(serviceIntent);
        statusText.setText("Status: Monitoring active ✅");
        Toast.makeText(this, "Fall detection started", Toast.LENGTH_SHORT).show();
    }

    private void stopMonitoring() {
        Intent serviceIntent = new Intent(this, FallDetectionService.class);
        stopService(serviceIntent);
        statusText.setText("Status: Monitoring stopped ⛔");
        Toast.makeText(this, "Fall detection stopped", Toast.LENGTH_SHORT).show();
    }
}
