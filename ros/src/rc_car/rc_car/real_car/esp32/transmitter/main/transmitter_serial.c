#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "driver/usb_serial_jtag.h"

#include "esp_event.h"
#include "esp_now.h"
#include "esp_timer.h"
#include "esp_wifi.h"
#include "nvs_flash.h"


#define ESPNOW_CHANNEL 11

/* Keepalive period when no fresh command arrives. Fresh commands
 * are sent immediately. Requires CONFIG_FREERTOS_HZ=1000. */
#define HEARTBEAT_MS 10

/* Auto-disarm if armed and no valid serial line for this long. */
#define SERIAL_DEADMAN_US 300000

#define CONTROL_PACKET_MAGIC 0x4C4D5432U
#define CONTROL_FLAG_ARMED   (1U << 0)


static const uint8_t receiver_mac[ESP_NOW_ETH_ALEN] = {
    0x34, 0x85, 0x18, 0x91, 0xCF, 0x40
};


typedef struct __attribute__((packed)) {
    uint32_t magic;
    int16_t throttle;   /* -1000..1000 */
    int16_t steering;   /* -1000..1000 */
    uint16_t flags;
} control_packet_t;


static uint32_t send_failures;


static void serial_write(const char *text)
{
    usb_serial_jtag_write_bytes(
        text,
        strlen(text),
        pdMS_TO_TICKS(10)
    );
}


void app_main(void)
{
    esp_err_t result = nvs_flash_init();

    if (result == ESP_ERR_NVS_NO_FREE_PAGES ||
        result == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ESP_ERROR_CHECK(nvs_flash_init());
    } else {
        ESP_ERROR_CHECK(result);
    }

    usb_serial_jtag_driver_config_t serial_config = {
        .rx_buffer_size = 256,
        .tx_buffer_size = 256,
    };

    ESP_ERROR_CHECK(
        usb_serial_jtag_driver_install(&serial_config)
    );

    ESP_ERROR_CHECK(esp_event_loop_create_default());

    wifi_init_config_t wifi_config = WIFI_INIT_CONFIG_DEFAULT();

    ESP_ERROR_CHECK(esp_wifi_init(&wifi_config));
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_start());
    ESP_ERROR_CHECK(esp_wifi_set_ps(WIFI_PS_NONE));
    ESP_ERROR_CHECK(
        esp_wifi_set_channel(
            ESPNOW_CHANNEL,
            WIFI_SECOND_CHAN_NONE
        )
    );

    ESP_ERROR_CHECK(esp_now_init());

    esp_now_peer_info_t peer = {0};

    memcpy(peer.peer_addr, receiver_mac, ESP_NOW_ETH_ALEN);
    peer.channel = ESPNOW_CHANNEL;
    peer.ifidx = WIFI_IF_STA;
    peer.encrypt = false;

    ESP_ERROR_CHECK(esp_now_add_peer(&peer));

    serial_write("READY\n");

    uint8_t mac[6];

    ESP_ERROR_CHECK(esp_wifi_get_mac(WIFI_IF_STA, mac));

    char mac_line[32];

    snprintf(
        mac_line,
        sizeof(mac_line),
        "MAC %02X:%02X:%02X:%02X:%02X:%02X\n",
        mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]
    );

    serial_write(mac_line);

    int16_t throttle = 0;
    int16_t steering = 0;
    bool armed = false;

    char line[64];
    size_t line_length = 0;
    bool discarding = false;

    int64_t last_line_us = esp_timer_get_time();
    TickType_t last_send = xTaskGetTickCount();

    control_packet_t packet = {
        .magic = CONTROL_PACKET_MAGIC,
        .throttle = 0,
        .steering = 0,
        .flags = 0,
    };

    const TickType_t heartbeat_ticks =
        pdMS_TO_TICKS(HEARTBEAT_MS);

    while (true) {
        /* Block on serial until the next heartbeat is due, so
         * fresh commands are handled with minimal latency. */
        TickType_t elapsed =
            xTaskGetTickCount() - last_send;

        TickType_t wait =
            (elapsed >= heartbeat_ticks)
                ? 0
                : heartbeat_ticks - elapsed;

        uint8_t received[64];

        int count = usb_serial_jtag_read_bytes(
            received,
            sizeof(received),
            wait
        );

        bool fresh = false;

        for (int i = 0; i < count; i++) {
            char character = (char)received[i];

            if (character != '\r' && character != '\n') {
                if (discarding) {
                    continue;
                }

                if (line_length < sizeof(line) - 1) {
                    line[line_length++] = character;
                } else {
                    discarding = true;
                }

                continue;
            }

            if (discarding) {
                discarding = false;
                line_length = 0;
                continue;
            }

            if (line_length == 0) {
                continue;
            }

            line[line_length] = '\0';
            line_length = 0;

            if (strcmp(line, "ARM") == 0) {
                armed = true;
                throttle = 0;
                steering = 0;
                fresh = true;
                last_line_us = esp_timer_get_time();
                serial_write("OK ARM\n");
            } else if (strcmp(line, "DISARM") == 0) {
                armed = false;
                throttle = 0;
                steering = 0;
                fresh = true;
                last_line_us = esp_timer_get_time();
                serial_write("OK DISARM\n");
            } else if (strcmp(line, "STATUS") == 0) {
                char response[80];

                snprintf(
                    response,
                    sizeof(response),
                    "STATUS throttle=%d steering=%d "
                    "armed=%s sendfail=%lu\n",
                    throttle,
                    steering,
                    armed ? "yes" : "no",
                    (unsigned long)send_failures
                );

                last_line_us = esp_timer_get_time();
                serial_write(response);
            } else {
                /* "<throttle> <steering>", both -1000..1000.
                 * No ack: this arrives at 100 Hz. */
                char *middle = NULL;
                char *end = NULL;

                long t = strtol(line, &middle, 10);
                long s = 0;

                bool valid =
                    middle != line && *middle == ' ';

                if (valid) {
                    char *rest = middle + 1;

                    s = strtol(rest, &end, 10);
                    valid = end != rest && *end == '\0';
                }

                if (valid &&
                    t >= -1000 && t <= 1000 &&
                    s >= -1000 && s <= 1000) {
                    throttle = (int16_t)t;
                    steering = (int16_t)s;
                    fresh = true;
                    last_line_us = esp_timer_get_time();
                } else {
                    serial_write("ERR\n");
                }
            }
        }

        if (armed &&
            esp_timer_get_time() - last_line_us >
                SERIAL_DEADMAN_US) {
            armed = false;
            throttle = 0;
            steering = 0;
            fresh = true;
            serial_write("DEADMAN\n");
        }

        if (fresh ||
            xTaskGetTickCount() - last_send >=
                heartbeat_ticks) {
            packet.throttle = throttle;
            packet.steering = steering;
            packet.flags =
                armed ? CONTROL_FLAG_ARMED : 0;

            if (esp_now_send(
                    receiver_mac,
                    (const uint8_t *)&packet,
                    sizeof(packet)
                ) != ESP_OK) {
                send_failures++;
            }

            last_send = xTaskGetTickCount();
        }
    }
}