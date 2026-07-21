idf_component_register(
    SRCS "receiver.c"
    INCLUDE_DIRS "."
    PRIV_REQUIRES
        nvs_flash
        esp_event
        esp_netif
        esp_wifi
        esp_timer
        esp_driver_ledc
)