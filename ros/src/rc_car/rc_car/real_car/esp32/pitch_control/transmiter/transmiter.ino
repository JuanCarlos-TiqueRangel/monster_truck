
#include <WiFi.h>
#include <esp_now.h>

// ============================================================
// SETTINGS
// ============================================================

const uint8_t CAR_MAC[6] = {0x30, 0x30, 0xF9, 0x34, 0x77, 0x30};

bool running = false;
unsigned long lastSend = 0;

void setup() {
  Serial.begin(115200);

  WiFi.mode(WIFI_STA);
  WiFi.setChannel(1);

  Serial.print("Transmitter MAC: ");
  Serial.println(WiFi.macAddress());

  if (esp_now_init() != ESP_OK) {
    Serial.println("ESP-NOW initialization failed");
    return;
  }

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
