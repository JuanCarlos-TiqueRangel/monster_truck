#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "driver/ledc.h"

#include "esp_event.h"
#include "esp_now.h"
#include "esp_timer.h"
#include "esp_wifi.h"
#include "nvs_flash.h"


#define ESPNOW_CHANNEL 11

#define THROTTLE_GPIO    1
#define STEERING_GPIO    2
#define THROTTLE_CHANNEL LEDC_CHANNEL_0
#define STEERING_CHANNEL LEDC_CHANNEL_1

/* Servo frame rate. 100 Hz matches the control rate. */
#define PWM_FRAME_HZ    100
#define PWM_RESOLUTION  LEDC_TIMER_14_BIT
#define PWM_DUTY_STEPS  (1U << 14)
#define PWM_PERIOD_US   (1000000U / PWM_FRAME_HZ)

#define PULSE_NEUTRAL_US 1500U

/* Both outputs to neutral if no valid packet for this long. */
#define FAILSAFE_TIMEOUT_US      100000
#define FAILSAFE_CHECK_PERIOD_US 25000

#define CONTROL_PACKET_MAGIC 0x4C4D5432U
#define CONTROL_FLAG_ARMED   (1U << 0)


static const uint8_t transmitter_mac[ESP_NOW_ETH_ALEN] = {
    0xDC, 0xDA, 0x0C, 0x57, 0xB0, 0x4C
};


typedef struct __attribute__((packed)) {
    uint32_t magic;
    int16_t throttle;   /* -1000..1000 -> 1000..2000 us */
    int16_t steering;   /* -1000..1000 -> 1000..2000 us */
    uint16_t flags;
} control_packet_t;


static volatile int64_t last_packet_us;


static void set_pulse(ledc_channel_t channel, uint32_t pulse_us)
{
    uint32_t duty =
        (pulse_us * PWM_DUTY_STEPS) / PWM_PERIOD_US;

    ledc_set_duty(LEDC_LOW_SPEED_MODE, channel, duty);
    ledc_update_duty(LEDC_LOW_SPEED_MODE, channel);
}


static void receive_callback(
    const esp_now_recv_info_t *info,
    const uint8_t *data,
    int length)
{
    if (data == NULL ||
        length != sizeof(control_packet_t)) {
        return;
    }

    if (memcmp(info->src_addr,
               transmitter_mac,
               ESP_NOW_ETH_ALEN) != 0) {
        return;
    }

    control_packet_t packet;
    memcpy(&packet, data, sizeof(packet));

    if (packet.magic != CONTROL_PACKET_MAGIC) {
        return;
    }

    uint32_t throttle_pulse = PULSE_NEUTRAL_US;
    uint32_t steering_pulse = PULSE_NEUTRAL_US;

    if ((packet.flags & CONTROL_FLAG_ARMED) != 0) {
        int32_t throttle = packet.throttle;
        int32_t steering = packet.steering;

        if (throttle > 1000) {
            throttle = 1000;
        } else if (throttle < -1000) {
            throttle = -1000;
        }

        if (steering > 1000) {
            steering = 1000;
        } else if (steering < -1000) {
            steering = -1000;
        }

        /* -1000..1000 -> 1000..2000 us */
        throttle_pulse = (uint32_t)(1500 + throttle / 2);
        steering_pulse = (uint32_t)(1500 + steering / 2);
    }

    last_packet_us = esp_timer_get_time();

    set_pulse(THROTTLE_CHANNEL, throttle_pulse);
    set_pulse(STEERING_CHANNEL, steering_pulse);
}


static void failsafe_check(void *arg)
{
    (void)arg;

    int64_t age = esp_timer_get_time() - last_packet_us;

    if (age > FAILSAFE_TIMEOUT_US) {
        set_pulse(THROTTLE_CHANNEL, PULSE_NEUTRAL_US);
        set_pulse(STEERING_CHANNEL, PULSE_NEUTRAL_US);
    }
}


void app_main(void)
{
    ledc_timer_config_t timer = {
        .speed_mode = LEDC_LOW_SPEED_MODE,
        .timer_num = LEDC_TIMER_0,
        .duty_resolution = PWM_RESOLUTION,
        .freq_hz = PWM_FRAME_HZ,
        .clk_cfg = LEDC_AUTO_CLK,
    };

    ledc_channel_config_t channel = {
        .gpio_num = THROTTLE_GPIO,
        .speed_mode = LEDC_LOW_SPEED_MODE,
        .channel = THROTTLE_CHANNEL,
        .timer_sel = LEDC_TIMER_0,
        .duty =
            (PULSE_NEUTRAL_US * PWM_DUTY_STEPS) /
            PWM_PERIOD_US,
        .hpoint = 0,
    };

    ESP_ERROR_CHECK(ledc_timer_config(&timer));
    ESP_ERROR_CHECK(ledc_channel_config(&channel));

    channel.gpio_num = STEERING_GPIO;
    channel.channel = STEERING_CHANNEL;

    ESP_ERROR_CHECK(ledc_channel_config(&channel));

    esp_err_t result = nvs_flash_init();

    if (result == ESP_ERR_NVS_NO_FREE_PAGES ||
        result == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ESP_ERROR_CHECK(nvs_flash_init());
    } else {
        ESP_ERROR_CHECK(result);
    }

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
    ESP_ERROR_CHECK(
        esp_now_register_recv_cb(receive_callback)
    );

    last_packet_us = esp_timer_get_time();

    const esp_timer_create_args_t failsafe_args = {
        .callback = failsafe_check,
        .name = "failsafe",
    };

    esp_timer_handle_t failsafe_timer;

    ESP_ERROR_CHECK(
        esp_timer_create(&failsafe_args, &failsafe_timer)
    );
    ESP_ERROR_CHECK(
        esp_timer_start_periodic(
            failsafe_timer,
            FAILSAFE_CHECK_PERIOD_US
        )
    );
}