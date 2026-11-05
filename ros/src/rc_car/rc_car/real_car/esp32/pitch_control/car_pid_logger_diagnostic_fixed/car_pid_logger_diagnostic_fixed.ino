// ESP32-S3 RC car: IMU PID + local CSV logging + reliable ESP-NOW file upload.
// Arduino-ESP32 3.x required. ESP-NOW channel must match the PC bridge (1).
#include <Wire.h>
#include <Adafruit_BNO08x.h>
#include <math.h>
#include <WiFi.h>
#include <esp_now.h>
#include <LittleFS.h>

// ============================================================
// SETTINGS
// ============================================================
#define SDA_PIN D4
#define SCL_PIN D5
#define ESC_PIN D0
#define BNO08X_RESET -1

const uint8_t TX_MAC[6] = {0xDC, 0xDA, 0x0C, 0x57, 0xB0, 0x4C};
const int ESPNOW_CHANNEL = 1;
const int BAUD = 115200;
const int outputPin = D1;

const double Kp = 3.8;
const double Ki = 2.0;
const double Kd = 0.5;
const double Setpoint = -64.0;

const int PWM_MIN = 1000;
const int PWM_NEUTRAL = 1500;
const int PWM_MAX = 2000;
const int PWM_FREQUENCY = 50;
const int PWM_RESOLUTION = 14;

const int MAX_SAMPLES = 3000;     // ~30 seconds at 100 Hz
const int CHUNK_BYTES = 180;      // Fits ESP-NOW v1 250-byte limit
const uint32_t ACK_TIMEOUT_MS = 200;
const int MAX_RETRIES = 20;

// ============================================================
// IMU / PID
// ============================================================
Adafruit_BNO08x bno08x(BNO08X_RESET);
sh2_SensorValue_t sensorValue;

volatile bool runPID = false;
volatile bool resendRequested = false;
bool pidWasRunning = false;
bool wasRunning = false;
uint32_t neutralDuty = 0;

float ax = 0.0;
float ay = 0.0;
float az = 0.0;
float roll = 0.0;
float pitch = 0.0;
float yaw = 0.0;

// Variables for the angular velocities
float pitch_rate = 0.0;

double Input = 0.0;
double Output = 0.0;
double error = 0.0;
double lastError = 0.0;
double errorSum = 0.0;
double dError = 0.0;
unsigned long lastTime = 0;

// Non-blocking diagnostic counters (printed once a second)
uint32_t rotationEvents = 0;
uint32_t gyroEvents = 0;
uint32_t pidWrites = 0;
uint32_t lastDiagnostic = 0;
int lastPwm = PWM_NEUTRAL;

// ============================================================
// LOCAL RECORDING
// ============================================================
struct __attribute__((packed)) LogSample {
  uint32_t time_us;
  float pitch_deg;
  float pitch_rate_rad_s;
  uint16_t pwm_us;
};

LogSample logData[MAX_SAMPLES];
int logCount = 0;
bool logOverflow = false;
bool storageReady = false;
uint16_t trialNumber = 1;
uint16_t lastSavedTrial = 0;

// ============================================================
// ESP-NOW FILE TRANSFER
// ============================================================
struct __attribute__((packed)) LogPacket {
  uint8_t type;            // 'B': begin, 'D': bytes, 'E': end
  uint16_t trial;
  uint16_t sequence;
  uint8_t length;
  uint8_t payload[180];
};

struct __attribute__((packed)) AckPacket {
  uint8_t type;            // 'A'
  uint16_t trial;
  uint16_t sequence;
};

File uploadFile;
LogPacket pendingPacket = {};
bool uploadActive = false;
volatile bool awaitingAck = false;
volatile bool ackReceived = false;
uint16_t uploadTrial = 0;
uint16_t uploadSequence = 0;
uint32_t lastUploadSend = 0;
int uploadRetries = 0;

void setReports() {
  // Pitch is essential for control: request it first at 100 Hz.
  if (!bno08x.enableReport(SH2_ROTATION_VECTOR, 10000)) {
    Serial.println("Failed to enable rotation vector");
  }
  // Pitch rate is logged only; request it at 50 Hz to reduce I2C load.
  if (!bno08x.enableReport(SH2_GYROSCOPE_CALIBRATED, 20000)) {
    Serial.println("Failed to enable calibrated gyroscope");
  }
  // The linear-acceleration report is disabled for this diagnostic test:
  // it is not used by the PID or CSV logger.
}

void onDataRecv(const esp_now_recv_info_t *info, const uint8_t *data, int len) {
  if (memcmp(info->src_addr, TX_MAC, 6) != 0) return;

  if (len == 1) {
    if (data[0] == 'S') runPID = true;
    if (data[0] == 'X') runPID = false;
    if (data[0] == 'R') resendRequested = true;
    return;
  }

  if (len == sizeof(AckPacket)) {
    AckPacket ack;
    memcpy(&ack, data, sizeof(ack));
    if (ack.type == 'A' && awaitingAck && ack.trial == uploadTrial && ack.sequence == uploadSequence) {
      ackReceived = true;
    }
  }
}

void cancelUpload() {
  if (uploadFile) uploadFile.close();
  uploadActive = false;
  awaitingAck = false;
  ackReceived = false;
}

void startUpload(uint16_t trial) {
  if (!storageReady || trial == 0) return;

  cancelUpload();
  char filename[32];
  snprintf(filename, sizeof(filename), "/wheelie_%03u.csv", trial);
  uploadFile = LittleFS.open(filename, "r");
  if (!uploadFile) {
    Serial.printf("Could not upload %s\n", filename);
    return;
  }

  uploadTrial = trial;
  uploadSequence = 0;
  uploadRetries = 0;
  uploadActive = true;
  Serial.printf("Uploading %s over ESP-NOW\n", filename);
}

void saveLog() {
  Serial.printf("LOG: storage=%d, samples=%d, trial=%u\n",
                storageReady, logCount, trialNumber);

  if (!storageReady) {
    Serial.println("ERROR: LittleFS is not mounted!");
    return;
  }

  if (logCount == 0) {
    Serial.println("ERROR: No PID samples recorded!");
    return;
  }

  if (trialNumber > 999) {
    Serial.println("ERROR: Trial limit reached!");
    return;
  }

  char filename[32];
  snprintf(filename, sizeof(filename), "/wheelie_%03u.csv", trialNumber);
  File file = LittleFS.open(filename, "w");
  if (!file) {
    Serial.println("Failed to save CSV");
    return;
  }

  file.println("time_us,pitch_deg,omega_rad_s,pwm_us");
  for (int i = 0; i < logCount; i++) {
    file.print(logData[i].time_us);
    file.print(",");
    file.print(logData[i].pitch_deg, 4);
    file.print(",");
    file.print(logData[i].pitch_rate_rad_s, 5);
    file.print(",");
    file.println(logData[i].pwm_us);
  }
  file.close();

  lastSavedTrial = trialNumber;
  Serial.printf("Saved %s (%d samples)%s\n", filename, logCount, logOverflow ? " [TRUNCATED]" : "");
  trialNumber++;
  startUpload(lastSavedTrial);
}

void sendPendingPacket() {
  esp_err_t result = esp_now_send(TX_MAC, (uint8_t *)&pendingPacket, sizeof(pendingPacket));
  lastUploadSend = millis();
  if (result != ESP_OK) {
    Serial.printf("ESP-NOW send failed: %d\n", (int)result);
  }
}

// Called only while PID is OFF; the fast PID loop never performs file I/O or upload.
void serviceUpload() {
  if (!uploadActive || runPID) return;

  if (awaitingAck) {
    if (ackReceived) {
      ackReceived = false;
      awaitingAck = false;
      if (pendingPacket.type == 'E') {
        Serial.printf("Upload completed: trial %u\n", uploadTrial);
        cancelUpload();
        return;
      }
      uploadSequence++;
    } else {
      if (millis() - lastUploadSend >= ACK_TIMEOUT_MS) {
        if (uploadRetries >= MAX_RETRIES) {
          Serial.println("Upload failed; CSV remains in LittleFS. Send R to retry.");
          cancelUpload();
          return;
        }
        uploadRetries++;
        sendPendingPacket();
      }
      return;
    }
  }

  memset(&pendingPacket, 0, sizeof(pendingPacket));
  pendingPacket.trial = uploadTrial;
  pendingPacket.sequence = uploadSequence;

  if (uploadSequence == 0) {
    pendingPacket.type = 'B';
  } else if (uploadFile.available()) {
    pendingPacket.type = 'D';
    pendingPacket.length = uploadFile.read(pendingPacket.payload, CHUNK_BYTES);
  } else {
    pendingPacket.type = 'E';
  }

  ackReceived = false;
  awaitingAck = true;
  uploadRetries = 0;
  sendPendingPacket();
}

void setup() {
  Serial.begin(BAUD);

  WiFi.mode(WIFI_STA);
  while (!WiFi.STA.started()) delay(10);
  WiFi.setChannel(ESPNOW_CHANNEL);
  Serial.print("Car MAC: ");
  Serial.println(WiFi.macAddress());

  ledcAttach(ESC_PIN, PWM_FREQUENCY, PWM_RESOLUTION);
  neutralDuty = PWM_NEUTRAL * (1UL << PWM_RESOLUTION) * (PWM_FREQUENCY / 1000000.0f);
  ledcWrite(ESC_PIN, neutralDuty);
  ledcAttach(outputPin, PWM_FREQUENCY, PWM_RESOLUTION);
  ledcWrite(outputPin, neutralDuty);
  delay(2000);

  storageReady = LittleFS.begin(true);
  if (!storageReady) {
    Serial.println("LittleFS mount failed: choose a filesystem partition, then format ONCE if needed");
  } else {
    char filename[32];
    while (trialNumber <= 999) {
      snprintf(filename, sizeof(filename), "/wheelie_%03u.csv", trialNumber);
      if (!LittleFS.exists(filename)) break;
      trialNumber++;
    }
    if (trialNumber > 1) lastSavedTrial = trialNumber - 1;
  }

  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(100000);
  if (!bno08x.begin_I2C(0x4A, &Wire)) {
    Serial.println("Could not find BNO08x!");
    while (1) delay(100);
  }
  setReports();

  if (esp_now_init() != ESP_OK) {
    Serial.println("ESP-NOW initialization failed");
    while (1) delay(100);
  }

  esp_now_peer_info_t peer = {};
  memcpy(peer.peer_addr, TX_MAC, 6);
  peer.channel = ESPNOW_CHANNEL;
  peer.ifidx = WIFI_IF_STA;
  peer.encrypt = false;
  if (esp_now_add_peer(&peer) != ESP_OK) {
    Serial.println("Could not add computer ESP32 as peer");
    while (1) delay(100);
  }
  esp_now_register_recv_cb(onDataRecv);

  Serial.println("Waiting for START command");
  lastTime = millis();
}

void loop() {
  // Print once per second, including when no rotation vector arrives.
  if (millis() - lastDiagnostic >= 1000) {
    lastDiagnostic = millis();
    Serial.printf("STATUS run=%d rot=%lu gyro=%lu pid=%lu pitch=%.2f gyroY=%.3f pwm=%d\n",
                  (int)runPID, (unsigned long)rotationEvents, (unsigned long)gyroEvents,
                  (unsigned long)pidWrites, pitch * RAD_TO_DEG, pitch_rate, lastPwm);
  }

  // Command transitions are handled before reading sensor data.
  if (runPID && !wasRunning) {
    cancelUpload();                     // Wheelie always takes priority over upload.
    logCount = 0;
    logOverflow = false;
    wasRunning = true;
    Serial.println("START received on car");
  }

  if (!runPID && wasRunning) {
    ledcWrite(ESC_PIN, neutralDuty);
    wasRunning = false;
    lastPwm = PWM_NEUTRAL;
    Serial.println("STOP received on car");
    saveLog();                          // File I/O only after STOP.
  }

  if (!runPID) {
    ledcWrite(ESC_PIN, neutralDuty);
    errorSum = 0.0;
    pidWasRunning = false;
    if (resendRequested) {
      resendRequested = false;
      startUpload(lastSavedTrial);
    }
    serviceUpload();
  }

  if (bno08x.wasReset()) setReports();
  if (!bno08x.getSensorEvent(&sensorValue)) return;

  if (sensorValue.sensorId == SH2_GYROSCOPE_CALIBRATED) {
    pitch_rate = sensorValue.un.gyroscope.y;
    gyroEvents++;
  }

  if (sensorValue.sensorId == SH2_LINEAR_ACCELERATION) {
    ax = sensorValue.un.linearAcceleration.x;
    ay = sensorValue.un.linearAcceleration.y;
    az = sensorValue.un.linearAcceleration.z;
  }

  if (sensorValue.sensorId != SH2_ROTATION_VECTOR) return;
  rotationEvents++;

  float w = sensorValue.un.rotationVector.real;
  float x = sensorValue.un.rotationVector.i;
  float y = sensorValue.un.rotationVector.j;
  float z = sensorValue.un.rotationVector.k;

  roll = atan2(2.0 * (w*x + y*z), 1.0 - 2.0 * (x*x + y*y));
  pitch = asin(constrain(2.0 * (w*y - z*x), -1.0, 1.0));
  yaw = atan2(2.0 * (w*z + x*y), 1.0 - 2.0 * (y*y + z*z));

  if (!runPID) return;

  // ============================================================
  // ORIGINAL PID CONTROLLER
  // ============================================================
  unsigned long now = millis();
  double dt = (double)(now - lastTime) / 1000.0;
  if (dt <= 0.0) return;

  Input = pitch * RAD_TO_DEG;
  error = Input - Setpoint;

  if (!pidWasRunning) {
    lastError = error;
    lastTime = now;
    pidWasRunning = true;
    return;
  }

  errorSum += error * dt;
  dError = (error - lastError) / dt;
  Output = (Kp * error) + (Ki * errorSum) + (Kd * dError);
  Output = constrain(Output, -500.0, 500.0);

  int pwm = constrain((int)(PWM_NEUTRAL + Output), PWM_MIN, PWM_MAX);
  uint32_t duty = pwm * (1UL << PWM_RESOLUTION) * (PWM_FREQUENCY / 1000000.0f);
  ledcWrite(ESC_PIN, duty);
  lastPwm = pwm;
  pidWrites++;

  if (logCount < MAX_SAMPLES) {
    logData[logCount].time_us = micros();
    logData[logCount].pitch_deg = Input;
    logData[logCount].pitch_rate_rad_s = pitch_rate; 
    logData[logCount].pwm_us = pwm;
    logCount++;
  } else {
    logOverflow = true;
  }

  lastError = error;
  lastTime = now;
}
