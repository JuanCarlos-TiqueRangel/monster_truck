
#include <Wire.h>
#include <Adafruit_BNO08x.h>
#include <math.h>

// ============================================================
// SETTINGS
// ============================================================

#define SDA_PIN D4
#define SCL_PIN D5
#define ESC_PIN D0
#define OUTPUT_PIN D1
#define BNO08X_RESET -1

Adafruit_BNO08x bno08x(BNO08X_RESET);
sh2_SensorValue_t sensorValue;

// ESC Profile & Timing Parameters
const int PWM_MIN = 1000;
const int PWM_NEUTRAL = 1500;
const int PWM_MAX = 2000;
const int PWM_FREQUENCY = 50;
const int PWM_RESOLUTION = 14;
uint32_t neutralDuty = 0;

// ACC variables
float ax = 0.0;
float ay = 0.0;
float az = 0.0;

// quaternions
float qw = 0;
float qx = 0;
float qy = 0;
float qz = 0;

// GYRO varlues
float roll = 0.0;
float pitch = 0.0;
float yaw = 0.0;
float pitchRate = 0.0;   // [LOG] deg/s, from the calibrated gyro
char serial_data;

int current_target_pwm = 1500; 

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

}

void setup() {
  Serial.begin(115200);

  // Calculate strict hardware neutral duty limits
  neutralDuty = PWM_NEUTRAL * (1UL << PWM_RESOLUTION) * (PWM_FREQUENCY / 1000000.0f);

  // Attach Pins and initialize both to safe neutral state
  ledcAttach(ESC_PIN, PWM_FREQUENCY, PWM_RESOLUTION);
  ledcWrite(ESC_PIN, neutralDuty);
  ledcAttach(OUTPUT_PIN, PWM_FREQUENCY, PWM_RESOLUTION);
  ledcWrite(OUTPUT_PIN, neutralDuty);

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
    checkIncomingSerial(); // Constantly read /dev/ttyACM0 input commands
    return;
  }

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
    qw = sensorValue.un.rotationVector.real;
    qx = sensorValue.un.rotationVector.i;
    qy = sensorValue.un.rotationVector.j;
    qz = sensorValue.un.rotationVector.k;

    roll = atan2(2.0 * (qw*qx + qy*qz), 1.0 - 2.0 * (qx*qx + qy*qy));
    pitch = asin(constrain(2.0 * (qw*qy - qz*qx), -1.0, 1.0));
    yaw = atan2(2.0 * (qw*qz + qx*qy), 1.0 - 2.0 * (qy*qy + qz*qz));
  }

  if (Serial.available() > 0){
    serial_data = Serial.read();
  }
  
  // Intercept the PWM command string calculated by the computer
  checkIncomingSerial();

  Serial.print(ax); Serial.print(",");
  Serial.print(ay); Serial.print(",");
  Serial.print(az); Serial.print(",");

  Serial.print(0.0); Serial.print(",");
  Serial.print(pitchRate); Serial.print(",");
  Serial.print(0.0); Serial.print(",");

  Serial.print(qw); Serial.print(",");
  Serial.print(qx); Serial.print(",");
  Serial.print(qy); Serial.print(",");
  Serial.print(qz); Serial.print(",");

  Serial.print(roll); Serial.print(",");
  Serial.print(pitch * RAD_TO_DEG); Serial.print(",");
  Serial.println(yaw); 
  
  // Ends line with \n so Python's readline() works

  // Optional: A small delay to avoid choking the serial buffer 
  delay(1); 


}

void checkIncomingSerial() {
  if (Serial.available() > 0) {
    // Collect the string value sent from your computer script
    int input_val = Serial.parseInt();
    
    // Bounds control verification
    if (input_val >= PWM_MIN && input_val <= PWM_MAX) {
      current_target_pwm = input_val;
      
      // Map directly to 14-bit ESP32 register
      uint32_t active_duty = current_target_pwm * (1UL << PWM_RESOLUTION) * (PWM_FREQUENCY / 1000000.0f);
      ledcWrite(ESC_PIN, active_duty);
    }
  }
}
