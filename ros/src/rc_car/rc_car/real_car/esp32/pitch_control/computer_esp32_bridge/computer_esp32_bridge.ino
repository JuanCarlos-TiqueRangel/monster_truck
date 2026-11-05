#include <WiFi.h>
#include <esp_now.h>

// ============================================================
// SETTINGS
// ============================================================

const uint8_t CAR_MAC[6] = {0x30, 0x30, 0xF9, 0x34, 0x77, 0x30};

bool running = false;
unsigned long lastSend = 0;

// [LOG] ======================================================
// DATA LOGGING
// After 'X' the car sends its whole RAM buffer. Each packet is
// printed once as CSV lines and ACKed back to the car.
// The two structs must be IDENTICAL to the ones in the car sketch.
// ============================================================
#define SAMPLES_PER_PACKET 17

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

QueueHandle_t logQueue;

// Runs in the WiFi task: only copy the packet here, print it in loop()
void onDataRecv(const esp_now_recv_info_t *info, const uint8_t *data, int len) {
  if (memcmp(info->src_addr, CAR_MAC, 6) != 0) return;
  if (len < (int)offsetof(LogPacket, s) || len > (int)sizeof(LogPacket)) return;

  LogPacket pkt = {};
  memcpy(&pkt, data, len);

  int maxN = (len - (int)offsetof(LogPacket, s)) / (int)sizeof(Sample);
  if (pkt.n > maxN) pkt.n = maxN;

  xQueueSend(logQueue, &pkt, 0);
}
// [LOG] ======================================================


void setup() {
  Serial.begin(115200);

  logQueue = xQueueCreate(32, sizeof(LogPacket));   // [LOG]

  WiFi.mode(WIFI_STA);
  WiFi.setChannel(1);

  Serial.print("Transmitter MAC: ");
  Serial.println(WiFi.macAddress());

  if (esp_now_init() != ESP_OK) {
    Serial.println("ESP-NOW initialization failed");
    return;
  }

  esp_now_register_recv_cb(onDataRecv);   // [LOG]

  esp_now_peer_info_t peer = {};
  memcpy(peer.peer_addr, CAR_MAC, 6);
  peer.channel = 1;
  peer.encrypt = false;

  if (esp_now_add_peer(&peer) != ESP_OK) {
    Serial.println("Failed to add peer");
    return;
  }

  Serial.println("S = START, X = STOP");
}

void loop() {
  // [LOG] Print received samples as CSV lines tagged with "D,"
  // Format: D,run,t_ms,pitch_deg,pitch_rate_dps,pwm_us
  LogPacket pkt;
  while (xQueueReceive(logQueue, &pkt, 0) == pdTRUE) {
    static bool gotAny = false;
    static uint16_t lastRun = 0, lastSeq = 0;
    static uint32_t runSamples = 0;

    // Packets come in order; a copy of an old one (lost ACK) is printed only once
    if (!gotAny || pkt.run != lastRun || pkt.seq > lastSeq) {
      if (!gotAny || pkt.run != lastRun) runSamples = 0;
      gotAny = true;

      for (int i = 0; i < pkt.n; i++) {
        Serial.printf("D,%u,%lu,%.3f,%.3f,%d\n",
                      (unsigned)pkt.run,
                      (unsigned long)pkt.s[i].t_ms,
                      pkt.s[i].pitch,
                      pkt.s[i].pitch_rate,
                      (int)pkt.s[i].pwm);
      }
      runSamples += pkt.n;
      lastRun = pkt.run;
      lastSeq = pkt.seq;

      if (pkt.seq + 1 == pkt.total) {
        Serial.printf("Run %u: %lu samples received\n", (unsigned)pkt.run, (unsigned long)runSamples);
      }
    }

    // ACK every copy so the car sends the next packet
    uint8_t ack[3] = {'A', (uint8_t)(pkt.seq & 0xFF), (uint8_t)(pkt.seq >> 8)};
    esp_now_send(CAR_MAC, ack, 3);
  }

  if (Serial.available()) {
    char command = Serial.read();

    if (command == 'S' || command == 'X') {
      uint8_t message = command;

      for (int i = 0; i < 5; i++) {
        esp_err_t result = esp_now_send(CAR_MAC, &message, 1);

        if (result != ESP_OK) {
          Serial.print("Send error: ");
          Serial.println(result);
        }

        delay(40);
      }

      Serial.print("Sent command: ");
      Serial.println(command);
    }
  }
}
