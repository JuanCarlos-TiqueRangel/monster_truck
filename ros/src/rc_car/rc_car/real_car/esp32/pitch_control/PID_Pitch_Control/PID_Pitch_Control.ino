
#include <Wire.h>
#include <Adafruit_BNO08x.h>
#include <math.h>
#include <WiFi.h>
#include <esp_now.h>

// ============================================================
// SETTINGS
// ============================================================

#define SDA_PIN D4
#define SCL_PIN D5
#define ESC_PIN D0

#define BNO08X_RESET -1

Adafruit_BNO08x bno08x(BNO08X_RESET);
sh2_SensorValue_t sensorValue;

// Variables for the PID enable
const uint8_t TX_MAC[6] = {0xDC, 0xDA, 0x0C, 0x57, 0xB0, 0x4C};
volatile bool runPID = false;
volatile unsigned long lastStart = 0;
bool pidWasRunning = false;

unsigned long lastPitchTime = 0;
uint32_t neutralDuty = 0;

// output pin servo
const int outputPin = D1;

// ACC variables
float ax = 0.0;
float ay = 0.0;
float az = 0.0;

// GYRO varlues
float roll = 0.0;
float pitch = 0.0;
float yaw = 0.0;


// PID Tuning Parameters
// double Kp = 4.0;
// double Ki = 2.0;
// double Kd = 1.0;

double Kp = 3.8;
double Ki = 2.0;
double Kd = 0.5;

// ESC parameters
const int PWM_MIN = 1000;
const int PWM_NEUTRAL = 1500;
const int PWM_MAX = 2000;
const int PWM_FREQUENCY = 50;
const int PWM_RESOLUTION = 14;

// Working Variables
double Setpoint = -64.0;
double Input, Output;
double error, lastError;
double errorSum, dError;
unsigned long lastTime;



void setReports() {
  if (!bno08x.enableReport(SH2_ROTATION_VECTOR, 10000)) {
    Serial.println("Failed to enable rotation vector");
  }

  // // Accelerations without gravity
  if (!bno08x.enableReport(SH2_LINEAR_ACCELERATION, 10000)) {
    Serial.println("Failed to enable accelerometer");
  }

}



void onDataRecv(const esp_now_recv_info_t *info, const uint8_t *data, int len) {
  if (len != 1 || memcmp(info->src_addr, TX_MAC, 6) != 0) {
    return;
  }

  if (data[0] == 'S') {
    runPID = true;
  }

  if (data[0] == 'X') {
    runPID = false;
  }
}



// void onDataRecv(const esp_now_recv_info_t *info, const uint8_t *data, int len) {
//   if (len != 1 || memcmp(info->src_addr, TX_MAC, 6) != 0) {
//     return;
//   }

//   if (data[0] == 'S') {
//     lastStart = millis();
//     runPID = true;
//   }

//   if (data[0] == 'X') {
//     runPID = false;
//   }
// }


// ============================================================
// SETUP
// ============================================================

void setup() {
  Serial.begin(115200);

  // ESP-NOW SETUP
  WiFi.mode(WIFI_STA);

  while (!WiFi.STA.started()) {
    delay(10);
  }

  WiFi.setChannel(1);

  Serial.print("Car MAC: ");
  Serial.println(WiFi.macAddress());

  // PWM SETUP D0
  ledcAttach(ESC_PIN, PWM_FREQUENCY, PWM_RESOLUTION);
  neutralDuty = PWM_NEUTRAL * (1UL << PWM_RESOLUTION) * (PWM_FREQUENCY / 1000000.0f);
  ledcWrite(ESC_PIN, neutralDuty);

  // PWM SETUP D1
  ledcAttach(outputPin, PWM_FREQUENCY, PWM_RESOLUTION);
  ledcWrite(outputPin, neutralDuty);

  delay(2000);

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

  if (esp_now_init() != ESP_OK) {
    Serial.println("ESP-NOW initialization failed");
    while (1) {
      delay(100);
    }
  }

  esp_now_register_recv_cb(onDataRecv);

  Serial.println("Waiting for START command");

  lastTime = millis();
}





// ============================================================
// LOOP
// ============================================================

void loop() {

  if (!runPID) {
    ledcWrite(ESC_PIN, neutralDuty);
    errorSum = 0.0;
    pidWasRunning = false;
  }

  // Serial.println("ESP32 LOOP RUNNING");

  if (bno08x.wasReset()) {
    setReports();
  }

  if (!bno08x.getSensorEvent(&sensorValue)) {
    return;
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

    lastPitchTime = millis();

  }

  if (sensorValue.sensorId != SH2_ROTATION_VECTOR) {
    return;
  }


  if (!runPID) {
    return;
  }

  // ============================================================
  // PID CONTROLLER
  // ============================================================

  unsigned long now = millis();
  double dt = (double)(now - lastTime) / 1000.0;

  if (dt <= 0.0){
    return;
  }
    
  Input = (pitch * RAD_TO_DEG);

  // Calculate error
  error = Input - Setpoint;

  // prevention of error spikes
  if (!pidWasRunning) {
    lastError = error;
    lastTime = now;
    pidWasRunning = true;
    return;
  }

  // Integral component: Accumulate error over time
  errorSum += error * dt;
  
  // Derivative component: Rate of change of error
  dError = (error - lastError) / dt;

  // Calculate Total PID Output
  // Output = (Kp * error) + (Kd * dError);
  Output = (Kp * error) + (Ki * errorSum) + (Kd * dError);

  // Output Saturation
  Output = constrain(Output, -500.0, 500.0);

  // // Convert PID Output to ESC pulse width
  // int pwm = PWM_NEUTRAL + (int)Output;

  // Send Command to ESC
  int pwm = constrain((int)(PWM_NEUTRAL + Output), PWM_MIN, PWM_MAX);

  uint32_t duty = pwm * (1UL << PWM_RESOLUTION) * (PWM_FREQUENCY / 1000000.0f);

  ledcWrite(ESC_PIN, duty);

  // Save variables for next cycle
  lastError = error;
  lastTime = now;

// ============================================================
// SERIAL MONITOR / PLOTTER
// ============================================================
  // Serial.print("Setpoint:");
  // Serial.print(Setpoint);

  // Serial.print("\tPitch:");
  // Serial.print(Input);

  // Serial.print("\tOutput:");
  // Serial.print(Output);

  // Serial.print("\tPWM:");
  // Serial.println(pwm);

}
