
#include <Wire.h>
#include <Adafruit_BNO08x.h>
#include <math.h>

// ============================================================
// SETTINGS
// ============================================================

#define SDA_PIN D4
#define SCL_PIN D5

#define BNO08X_RESET -1

Adafruit_BNO08x bno08x(BNO08X_RESET);
sh2_SensorValue_t sensorValue;

// ACC variables
float ax = 0.0;
float ay = 0.0;
float az = 0.0;

// GYRO varlues
float roll = 0.0;
float pitch = 0.0;
float yaw = 0.0;
float pitchRate = 0.0;   // [LOG] deg/s, from the calibrated gyro



void setReports() {
  
  // angles 
  if (!bno08x.enableReport(SH2_ROTATION_VECTOR, 10000)) {
    Serial.println("Failed to enable rotation vector");
  }

  // for angular velocities
  if (!bno08x.enableReport(SH2_GYROSCOPE_CALIBRATED, 20000)) {
    Serial.println("Failed to enable gyroscope");
  }

  // // Accelerations without gravity
  if (!bno08x.enableReport(SH2_LINEAR_ACCELERATION, 10000)) {
    Serial.println("Failed to enable accelerometer");
  }

  // // Accelerations with gravity
  // if (!bno08x.enableReport(SH2_ACCELEROMETER, 10000)) {
  //   Serial.println("Failed to enable accelerometer");
  // }
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(100000);

  Serial.println("Adafruit BNO08x - ESP32-S3 I2C");

  if (!bno08x.begin_I2C(0x4A, &Wire)) {
    Serial.println("Could not find BNO08x!");
    while (1) {
      delay(100);
    }
  }

  Serial.println("BNO08x found!");
  setReports();
}

void loop() {
  if (bno08x.wasReset()) {
    setReports();
  }

  if (!bno08x.getSensorEvent(&sensorValue)) {
    return;
  }

  // if (sensorValue.sensorId == SH2_ACCELEROMETER) {
  //   ax = sensorValue.un.accelerometer.x;
  //   ay = sensorValue.un.accelerometer.y;
  //   az = sensorValue.un.accelerometer.z;
  // }

  if (sensorValue.sensorId == SH2_GYROSCOPE_CALIBRATED){
    
    // pitchRate = sensorValue.un.gyroscope.y;

    float q = sensorValue.un.gyroscope.y;
    float r = sensorValue.un.gyroscope.z;
    pitchRate = (q * cosf(roll) - r * sinf(roll));

  }

  if (sensorValue.sensorId == SH2_LINEAR_ACCELERATION) {
      ax = sensorValue.un.linearAcceleration.x;
      ay = sensorValue.un.linearAcceleration.y;
      az = sensorValue.un.linearAcceleration.z;
  }

  if (sensorValue.sensorId == SH2_ROTATION_VECTOR) {
    float w = sensorValue.un.rotationVector.real;
    float x = sensorValue.un.rotationVector.i;
    float y = sensorValue.un.rotationVector.j;
    float z = sensorValue.un.rotationVector.k;

    roll = atan2(2.0 * (w*x + y*z), 1.0 - 2.0 * (x*x + y*y));
    pitch = asin(constrain(2.0 * (w*y - z*x), -1.0, 1.0));
    yaw = atan2(2.0 * (w*z + x*y), 1.0 - 2.0 * (y*y + z*z));
  }

  // if (sensorValue.sensorId != SH2_LINEAR_ACCELERATION ||  sensorValue.sensorId != SH2_GYROSCOPE_CALIBRATED || sensorValue.sensorId != SH2_ROTATION_VECTOR) {
  //   return;
  // }

    // Serial.println("---------------------------------------");
    // Serial.print("Yaw: ");
    // Serial.print(yaw * RAD_TO_DEG);
    // Serial.print("\tPitch: ");
    // Serial.print(pitch * RAD_TO_DEG);
    // Serial.print("\tRoll: ");
    // Serial.print(roll * RAD_TO_DEG);
    // // Serial.println(roll * RAD_TO_DEG);

    // Serial.print("\tAccel X: ");
    // Serial.print(ax);
    // Serial.print("\tY: ");
    // Serial.print(ay);
    // Serial.print("\tZ: ");
    // Serial.println(az);

    Serial.print("\tYaw:");
    Serial.print(yaw * RAD_TO_DEG);

    Serial.print("\tPitch:");
    Serial.print(pitch * RAD_TO_DEG);

    Serial.print("\tRoll:");
    Serial.print(roll * RAD_TO_DEG);

    Serial.print("\tPitchRate:");
    Serial.println(pitchRate * RAD_TO_DEG);

    // Serial.print("\tAccel_X:");
    // Serial.print(ax);

    // Serial.print("\tAccel_Y:");
    // Serial.print(ay);

    // Serial.print("\tAccel_Z:");
    // Serial.println(az);

}
