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

#define THROTTLE_GPIO    2
#define STEERING_GPIO    1
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





























// /* ESP-NOW -> ESC/steering PWM, with an ACK AFTER successful software updates.
//  * Target: ESP32-S3, ESP-IDF >= 5.0 (including 5.0.7).
//  * ACK is NOT proof of the changed physical PWM pulse, ESC response, or motion.
//  * One control task owns both LEDC outputs AND the receiver failsafe.
//  */
// #include <stdbool.h>
// #include <stdint.h>
// #include <string.h>
// #include "freertos/FreeRTOS.h"
// #include "freertos/task.h"
// #include "freertos/queue.h"
// #include "freertos/semphr.h"
// #include "driver/gpio.h"
// #include "driver/ledc.h"
// #include "esp_event.h"
// #include "esp_idf_version.h"
// #include "esp_now.h"
// #include "esp_timer.h"
// #include "esp_wifi.h"
// #include "nvs_flash.h"

// #define ESPNOW_CHANNEL 11
// #define THROTTLE_GPIO 2
// #define STEERING_GPIO 1
// #define THROTTLE_CHANNEL LEDC_CHANNEL_0
// #define STEERING_CHANNEL LEDC_CHANNEL_1
// #define PWM_FRAME_HZ 100
// #define PWM_RESOLUTION LEDC_TIMER_14_BIT
// #define PWM_DUTY_STEPS (1U << 14)
// #define PWM_PERIOD_US (1000000U / PWM_FRAME_HZ)
// #define PULSE_NEUTRAL_US 1500U
// #define FAILSAFE_TIMEOUT_US 100000
// #define FAILSAFE_CHECK_MS 5
// #define TIMING_GPIO (-1)  /* Optional FREE output GPIO; NOT throttle GPIO2 or steering GPIO1. */
// #define MS_TICKS(ms) ((pdMS_TO_TICKS(ms) > 0) ? pdMS_TO_TICKS(ms) : 1)

// static const uint8_t transmitter_mac[ESP_NOW_ETH_ALEN] = {0xDC, 0xDA, 0x0C, 0x57, 0xB0, 0x4C};
// /* Protocol v3: this block MUST remain identical in TX and RX. No shared header needed. */
// #define PROTOCOL_VERSION 3
// #define CONTROL_PACKET_MAGIC 0x4C4D5433U
// #define ACK_PACKET_MAGIC 0x41434B33U
// #define FLAG_ARMED (1U << 0)
// #define FLAG_ACK_REQUEST (1U << 1)

// typedef struct __attribute__((packed)) {
//     uint32_t magic;
//     uint32_t session;
//     uint32_t wire_seq;
//     uint32_t command_id;
//     int64_t tx_send_us;
//     uint32_t tx_prepare_us;
//     int16_t throttle;
//     int16_t steering;
//     uint16_t flags;
// } control_packet_t;

// typedef struct __attribute__((packed)) {
//     uint32_t magic;
//     uint32_t session;
//     uint32_t wire_seq;
//     uint32_t command_id;
//     int64_t tx_send_us;
//     uint32_t tx_prepare_us;
//     uint32_t rx_queue_us;
//     uint32_t rx_apply_us;
//     uint32_t rx_turnaround_us;
//     int32_t status;             /* ESP_OK means BOTH LEDC update calls succeeded. */
//     uint16_t throttle_us;
//     uint16_t steering_us;
//     uint16_t flags;
// } ack_packet_t;

// _Static_assert(sizeof(control_packet_t) == 34, "Control protocol layout changed");
// _Static_assert(sizeof(ack_packet_t) == 50, "ACK protocol layout changed");

// typedef struct {
//     control_packet_t packet;
//     int64_t rx_us;
// } rx_job_t;

// typedef struct {
//     ack_packet_t ack;
//     int64_t rx_us;
// } ack_job_t;

// static QueueHandle_t control_queue, ack_queue;
// static SemaphoreHandle_t ack_send_done;

// static esp_err_t set_pulse(ledc_channel_t channel, uint32_t pulse_us)
// {
//     uint32_t duty = (pulse_us * PWM_DUTY_STEPS) / PWM_PERIOD_US;
//     esp_err_t result = ledc_set_duty(LEDC_LOW_SPEED_MODE, channel, duty);
//     if (result != ESP_OK) return result;
//     return ledc_update_duty(LEDC_LOW_SPEED_MODE, channel);
// }

// static void neutral_outputs(void)
// {
//     ESP_ERROR_CHECK(set_pulse(THROTTLE_CHANNEL, PULSE_NEUTRAL_US));
//     ESP_ERROR_CHECK(set_pulse(STEERING_CHANNEL, PULSE_NEUTRAL_US));
// }

// static void receive_callback(const esp_now_recv_info_t *info, const uint8_t *data, int length)
// {
//     int64_t now = esp_timer_get_time();
//     if (!info || !info->src_addr || !data || length != (int)sizeof(control_packet_t)) return;
//     if (memcmp(info->src_addr, transmitter_mac, ESP_NOW_ETH_ALEN) != 0) return;
//     rx_job_t job = {.rx_us = now};
//     memcpy(&job.packet, data, sizeof(job.packet));
//     if (job.packet.magic != CONTROL_PACKET_MAGIC || job.packet.session == 0) return;
//     if (job.packet.throttle < -1000 || job.packet.throttle > 1000) return;
//     if (job.packet.steering < -1000 || job.packet.steering > 1000) return;
//     if (job.packet.flags & ~(FLAG_ARMED | FLAG_ACK_REQUEST)) return;
//     xQueueOverwrite(control_queue, &job);  /* Latest wins; no unbounded command backlog. */
// }

// #if ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(5, 5, 0)
// static void send_callback(const esp_now_send_info_t *info, esp_now_send_status_t status)
// #else
// static void send_callback(const uint8_t *info, esp_now_send_status_t status)
// #endif
// {
//     (void)info;
//     (void)status;
//     xSemaphoreGive(ack_send_done);
// }

// static void ack_task(void *arg)
// {
//     (void)arg;
//     ack_job_t job;
//     while (true) {
//         if (xQueueReceive(ack_queue, &job, portMAX_DELAY) != pdTRUE) continue;
//         job.ack.rx_turnaround_us = (uint32_t)(esp_timer_get_time() - job.rx_us);
//         esp_err_t result = esp_now_send(transmitter_mac, (const uint8_t *)&job.ack, sizeof(job.ack));
//         if (result == ESP_OK) xSemaphoreTake(ack_send_done, portMAX_DELAY);
//         /* A failed/lost ACK becomes a host ACK timeout, NOT a fabricated zero latency.
//          * This task never delays PWM handling or the receiver failsafe. */
//     }
// }

// static void control_task(void *arg)
// {
//     (void)arg;
//     rx_job_t job;
//     int64_t last_packet_us = esp_timer_get_time();
//     uint32_t current_session = 0, last_wire_seq = 0;
//     bool have_sequence = false, neutral = true;
//     while (true) {
//         bool received = xQueueReceive(control_queue, &job, MS_TICKS(FAILSAFE_CHECK_MS)) == pdTRUE;
//         int64_t now = esp_timer_get_time();
//         if (now - last_packet_us > FAILSAFE_TIMEOUT_US && !neutral) {
//             neutral_outputs();
//             neutral = true;
//         }
//         if (!received || now - job.rx_us > FAILSAFE_TIMEOUT_US) continue;
//         control_packet_t *p = &job.packet;
//         /* Session IDs prevent host restarts from reusing command IDs in telemetry.
//          * Within a session, discard duplicate / older wire packets. */
//         if (have_sequence && p->session == current_session && (int32_t)(p->wire_seq - last_wire_seq) <= 0) continue;
//         current_session = p->session;
//         last_wire_seq = p->wire_seq;
//         have_sequence = true;
//         uint16_t throttle_us = PULSE_NEUTRAL_US, steering_us = PULSE_NEUTRAL_US;
//         if (p->flags & FLAG_ARMED) {
//             throttle_us = (uint16_t)(1500 + p->throttle / 2);
//             steering_us = (uint16_t)(1500 + p->steering / 2);
//         }
//         int64_t apply_start = esp_timer_get_time();
//         esp_err_t e1 = set_pulse(THROTTLE_CHANNEL, throttle_us);
//         esp_err_t e2 = set_pulse(STEERING_CHANNEL, steering_us);
//         int64_t applied = esp_timer_get_time();
//         esp_err_t status = (e1 != ESP_OK) ? e1 : e2;
//         if (status != ESP_OK) {
//             neutral_outputs();
//             throttle_us = steering_us = PULSE_NEUTRAL_US;
//         }
//         last_packet_us = job.rx_us;
//         neutral = throttle_us == PULSE_NEUTRAL_US && steering_us == PULSE_NEUTRAL_US;
//         if (TIMING_GPIO >= 0 && (p->flags & FLAG_ACK_REQUEST) && status == ESP_OK) {
//             gpio_set_level((gpio_num_t)TIMING_GPIO, (int)(p->command_id & 1U));
//         }
//         if (p->flags & FLAG_ACK_REQUEST) {
//             ack_job_t response = {
//                 .ack = {
//                     .magic = ACK_PACKET_MAGIC, .session = p->session, .wire_seq = p->wire_seq,
//                     .command_id = p->command_id, .tx_send_us = p->tx_send_us, .tx_prepare_us = p->tx_prepare_us,
//                     .rx_queue_us = (uint32_t)(apply_start - job.rx_us),
//                     .rx_apply_us = (uint32_t)(applied - apply_start), .status = status,
//                     .throttle_us = throttle_us, .steering_us = steering_us, .flags = p->flags,
//                 },
//                 .rx_us = job.rx_us,
//             };
//             /* No blocking if telemetry is congested. A missing ACK is recorded on the host. */
//             xQueueSend(ack_queue, &response, 0);
//         }
//     }
// }

// void app_main(void)
// {
//     ledc_timer_config_t timer = {
//         .speed_mode = LEDC_LOW_SPEED_MODE, .timer_num = LEDC_TIMER_0,
//         .duty_resolution = PWM_RESOLUTION, .freq_hz = PWM_FRAME_HZ, .clk_cfg = LEDC_AUTO_CLK,
//     };
//     ledc_channel_config_t channel = {
//         .gpio_num = THROTTLE_GPIO, .speed_mode = LEDC_LOW_SPEED_MODE,
//         .channel = THROTTLE_CHANNEL, .timer_sel = LEDC_TIMER_0,
//         .duty = (PULSE_NEUTRAL_US * PWM_DUTY_STEPS) / PWM_PERIOD_US, .hpoint = 0,
//     };
//     ESP_ERROR_CHECK(ledc_timer_config(&timer));
//     ESP_ERROR_CHECK(ledc_channel_config(&channel));
//     channel.gpio_num = STEERING_GPIO;
//     channel.channel = STEERING_CHANNEL;
//     ESP_ERROR_CHECK(ledc_channel_config(&channel));
//     if (TIMING_GPIO >= 0) {
//         configASSERT(TIMING_GPIO != THROTTLE_GPIO && TIMING_GPIO != STEERING_GPIO);
//         gpio_config_t pin = {.pin_bit_mask = 1ULL << (TIMING_GPIO & 63), .mode = GPIO_MODE_OUTPUT};
//         ESP_ERROR_CHECK(gpio_config(&pin));
//         ESP_ERROR_CHECK(gpio_set_level((gpio_num_t)TIMING_GPIO, 0));
//     }
//     control_queue = xQueueCreate(1, sizeof(rx_job_t));
//     ack_queue = xQueueCreate(16, sizeof(ack_job_t));
//     ack_send_done = xSemaphoreCreateBinary();
//     configASSERT(control_queue && ack_queue && ack_send_done);
//     if (xTaskCreate(control_task, "car_control", 4096, NULL, 10, NULL) != pdPASS) ESP_ERROR_CHECK(ESP_ERR_NO_MEM);
//     esp_err_t result = nvs_flash_init();
//     if (result == ESP_ERR_NVS_NO_FREE_PAGES || result == ESP_ERR_NVS_NEW_VERSION_FOUND) {
//         ESP_ERROR_CHECK(nvs_flash_erase());
//         ESP_ERROR_CHECK(nvs_flash_init());
//     } else { ESP_ERROR_CHECK(result); }
//     ESP_ERROR_CHECK(esp_event_loop_create_default());
//     wifi_init_config_t wifi_config = WIFI_INIT_CONFIG_DEFAULT();
//     ESP_ERROR_CHECK(esp_wifi_init(&wifi_config));
//     ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
//     ESP_ERROR_CHECK(esp_wifi_start());
//     ESP_ERROR_CHECK(esp_wifi_set_ps(WIFI_PS_NONE));
//     ESP_ERROR_CHECK(esp_wifi_set_channel(ESPNOW_CHANNEL, WIFI_SECOND_CHAN_NONE));
//     ESP_ERROR_CHECK(esp_now_init());
//     esp_now_peer_info_t peer = {.channel = ESPNOW_CHANNEL, .ifidx = WIFI_IF_STA, .encrypt = false};
//     memcpy(peer.peer_addr, transmitter_mac, ESP_NOW_ETH_ALEN);
//     ESP_ERROR_CHECK(esp_now_add_peer(&peer));
//     ESP_ERROR_CHECK(esp_now_register_send_cb(send_callback));
//     if (xTaskCreate(ack_task, "ack_tx", 4096, NULL, 7, NULL) != pdPASS) ESP_ERROR_CHECK(ESP_ERR_NO_MEM);
//     ESP_ERROR_CHECK(esp_now_register_recv_cb(receive_callback));
// }

