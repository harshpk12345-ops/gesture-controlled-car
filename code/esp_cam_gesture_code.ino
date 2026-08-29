// ==========================================================
// GESTURE ESP32-CAM — streams video + controls its OWN flash LED
// Board: AI-Thinker ESP32-CAM
// Stream:    http://<IP>:81/stream
// Flash on:  http://<IP>/flashON
// Flash off: http://<IP>/flashOFF
//
// Purpose: dedicated camera for hand-gesture recognition. Its flash
// is turned on by Python (via /flashON) on a thumbs-up gesture, purely
// to illuminate the hand in low light -- it has NOTHING to do with the
// car and never sends drive commands.
// ==========================================================

#include "esp_camera.h"
#include <WiFi.h>
#include <WebServer.h>
#include "esp_http_server.h"

// ---- Wi-Fi credentials ----
const char* ssid     = "Airtel_srin_4723";
const char* password = "Air@06268";

// ---- Flash LED pin (AI-Thinker onboard flash) ----
#define FLASH_LED_PIN 4
#define FLASH_BRIGHTNESS 102   // 40% of 255 (255 * 0.40 ≈ 102)

// ---- AI-Thinker camera pin map ----
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

#define PART_BOUNDARY "123456789000000000000987654321"
static const char* STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char* STREAM_BOUNDARY = "\r\n--" PART_BOUNDARY "\r\n";
static const char* STREAM_PART = "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

httpd_handle_t stream_httpd = NULL;
WebServer controlServer(80);

// ---------------- Streaming (port 81) ----------------

static esp_err_t stream_handler(httpd_req_t *req) {
  camera_fb_t *fb = NULL;
  esp_err_t res = ESP_OK;
  char part_buf[64];

  res = httpd_resp_set_type(req, STREAM_CONTENT_TYPE);
  if (res != ESP_OK) return res;

  while (true) {
    fb = esp_camera_fb_get();
    if (!fb) {
      res = ESP_FAIL;
    } else {
      if (fb->format != PIXFORMAT_JPEG) {
        esp_camera_fb_return(fb);
        continue;
      }
      size_t hlen = snprintf(part_buf, 64, STREAM_PART, fb->len);
      res = httpd_resp_send_chunk(req, STREAM_BOUNDARY, strlen(STREAM_BOUNDARY));
      if (res == ESP_OK) res = httpd_resp_send_chunk(req, part_buf, hlen);
      if (res == ESP_OK) res = httpd_resp_send_chunk(req, (const char *)fb->buf, fb->len);
      esp_camera_fb_return(fb);
    }
    if (res != ESP_OK) break;
  }
  return res;
}

void startCameraServer() {
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.server_port = 81;
  config.ctrl_port = 81;

  httpd_uri_t stream_uri = {
    .uri = "/stream",
    .method = HTTP_GET,
    .handler = stream_handler,
    .user_ctx = NULL
  };

  if (httpd_start(&stream_httpd, &config) == ESP_OK) {
    httpd_register_uri_handler(stream_httpd, &stream_uri);
  }
}

// ---------------- Flash control (port 80) ----------------

void handleFlashOn() {
  ledcWrite(FLASH_LED_PIN, FLASH_BRIGHTNESS);   // or ledcWrite(0, FLASH_BRIGHTNESS) on old core
  controlServer.send(200, "text/plain", "FLASH ON (40%)");
}

void handleFlashOff() {
  ledcWrite(FLASH_LED_PIN, 0);                  // or ledcWrite(0, 0) on old core
  controlServer.send(200, "text/plain", "FLASH OFF");
}

void handleRoot() {
  controlServer.send(200, "text/plain", "Gesture ESP32-CAM. Stream: :81/stream  Flash: /flashON /flashOFF");
}

void handleNotFound() {
  controlServer.send(404, "text/plain", "Not found");
}

// ---------------- Setup ----------------

void setup() {
  Serial.begin(115200);
  Serial.setDebugOutput(false);

  ledcAttach(FLASH_LED_PIN, 5000, 8);   // pin, freq 5kHz, 8-bit resolution (0-255)
  ledcWrite(FLASH_LED_PIN, 0);          // start off

  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;

  // Force ESP32-CAM to use internal DRAM (No PSRAM)
  config.frame_size   = FRAMESIZE_QVGA;      // 320x240
  config.jpeg_quality = 15;
  config.fb_count     = 1;
  config.fb_location  = CAMERA_FB_IN_DRAM;
  config.grab_mode    = CAMERA_GRAB_WHEN_EMPTY;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x\n", err);
    return;
  }

  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();
  Serial.print("Connected! Stream: http://");
  Serial.print(WiFi.localIP());
  Serial.println(":81/stream");
  Serial.print("Flash control: http://");
  Serial.println(WiFi.localIP());

  startCameraServer();

  controlServer.on("/", handleRoot);
  controlServer.on("/flashON", handleFlashOn);
  controlServer.on("/flashOFF", handleFlashOff);
  controlServer.onNotFound(handleNotFound);
  controlServer.begin();
}

void loop() {
  controlServer.handleClient();
}
