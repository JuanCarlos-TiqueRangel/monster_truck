
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
float pitchRate = 0.0;   // [LOG] deg/s, from the calibrated gyro

// [LOG] ======================================================
// DATA LOGGING
// While the PID runs, samples are ONLY stored in RAM (nothing is sent).
// When 'X' arrives, the motor goes to neutral and the whole buffer is
// sent to the PC ESP32 (TX_MAC). Each packet waits for an ACK from the
// PC and is re-sent if it was lost.
// The two structs must be IDENTICAL in the PC-side sketch.
// ============================================================
#define MAX_SAMPLES 10000        // 140 KB of RAM = 100 s at 100 Hz
#define SAMPLES_PER_PACKET 17    // 7 + 17*14 = 245 bytes (ESP-NOW max 250)

typedef struct __attribute__((packed)) {
  uint32_t t_ms;       // millis() on the car
  float pitch;         // deg
  float pitch_rate;    // deg/s
  int16_t pwm;         // us sent to the ESC
} Sample;

typedef struct __attribute__((packed)) {
  uint16_t run;        // run id (one run = S ... X), +1 every run
  uint16_t seq;        // packet index inside this run
  uint16_t total;      // number of packets in this run
  uint8_t n;           // samples in this packet
  Sample s[SAMPLES_PER_PACKET];
} LogPacket;

Sample logBuf[MAX_SAMPLES];
uint32_t logCount = 0;
uint16_t runId = 0;
uint16_t nextSeq = 0;                // where to resume if the PC stopped answering
volatile bool sendRequested = false;
volatile int32_t ackSeq = -1;

// Send one packet and wait for the PC's ACK; re-send if it does not come
bool sendAcked(const LogPacket &pkt) {
  size_t len = offsetof(LogPacket, s) + pkt.n * sizeof(Sample);
  for (int tries = 0; tries < 30; tries++) {
    ackSeq = -1;
    esp_now_send(TX_MAC, (const uint8_t *)&pkt, len);
    unsigned long t0 = millis();
    while (millis() - t0 < 100) {
      if (ackSeq == pkt.seq) return true;
      delay(1);
    }
  }
  return false;
}

// Send the whole RAM buffer to the PC
void sendLog() {
  if (logCount == 0) return;

  LogPacket pkt;
  pkt.run = runId;
  pkt.total = (logCount + SAMPLES_PER_PACKET - 1) / SAMPLES_PER_PACKET;
  Serial.printf("[LOG] sending run %u: %lu samples\n", runId, (unsigned long)logCount);

  for (; nextSeq < pkt.total; nextSeq++) {
    uint32_t first = (uint32_t)nextSeq * SAMPLES_PER_PACKET;
    uint32_t left = logCount - first;
    pkt.seq = nextSeq;
    pkt.n = left < SAMPLES_PER_PACKET ? left : SAMPLES_PER_PACKET;
    memcpy(pkt.s, &logBuf[first], pkt.n * sizeof(Sample));

    if (!sendAcked(pkt)) {
      Serial.println("[LOG] PC not answering. Data kept: send X again to retry");
      return;
    }
  }

  logCount = 0;
  nextSeq = 0;
  Serial.println("[LOG] done");
}
// [LOG] ======================================================


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

  // [LOG] Calibrated gyro for the pitch rate. It replaces the linear
  // acceleration report (ax, ay, az were not used) so the BNO08x
  // report/I2C load stays the same as before.
  if (!bno08x.enableReport(SH2_GYROSCOPE_CALIBRATED, 10000)) {
    Serial.println("Failed to enable gyroscope");
  }

  // // Accelerations without gravity
  // if (!bno08x.enableReport(SH2_LINEAR_ACCELERATION, 10000)) {
  //   Serial.println("Failed to enable accelerometer");
  // }

}



void onDataRecv(const esp_now_recv_info_t *info, const uint8_t *data, int len) {
  // [LOG] ACK from the PC for one log packet: 'A' + seq (2 bytes)
  if (len == 3 && data[0] == 'A' && memcmp(info->src_addr, TX_MAC, 6) == 0) {
    ackSeq = data[1] | (data[2] << 8);
    return;
  }

  if (len != 1 || memcmp(info->src_addr, TX_MAC, 6) != 0) {
    return;
  }

  if (data[0] == 'S') {
    runPID = true;
  }

  if (data[0] == 'X') {
    runPID = false;
    sendRequested = true;   // [LOG] send the RAM buffer
  }
}



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

  // [LOG] Add the PC ESP32 as a peer so the car can send data to it
  esp_now_peer_info_t peer = {};
  memcpy(peer.peer_addr, TX_MAC, 6);
  peer.channel = 1;
  peer.encrypt = false;
  if (esp_now_add_peer(&peer) != ESP_OK) {
    Serial.println("Failed to add PC peer");
  }
  runId = esp_random();   // [LOG] random start: run ids stay unique after a reboot

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

    // [LOG] After 'X' (motor already at neutral): send everything in RAM
    if (sendRequested) {
      sendRequested = false;
      sendLog();
    }
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

  // [LOG] Euler pitch rate: theta_dot = q*cos(roll) - r*sin(roll)
  // (equals gyro y when roll ~ 0)
  if (sensorValue.sensorId == SH2_GYROSCOPE_CALIBRATED) {
    float q = sensorValue.un.gyroscope.y;
    float r = sensorValue.un.gyroscope.z;
    pitchRate = (q * cosf(roll) - r * sinf(roll)) * RAD_TO_DEG;
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

    // [LOG] New run: start with an empty buffer
    logCount = 0;
    nextSeq = 0;
    runId++;
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

  // [LOG] Store this sample in RAM only (stops when the buffer is full)
  if (logCount < MAX_SAMPLES) {
    logBuf[logCount++] = Sample{(uint32_t)now, (float)Input, pitchRate, (int16_t)pwm};
  }

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
