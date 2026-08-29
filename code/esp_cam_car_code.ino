// ==========================================================
// ESP32-CAM RC CAR (AI-Thinker, PSRAM)
// Port 80 : Motor + Servo Control (WebServer)
// Port 81 : Camera Stream (esp_http_server, MJPEG)
// ==========================================================

#include <WiFi.h>
#include <WebServer.h>
#include <ESP32Servo.h>
#include <ESP32PWM.h>
#include "esp_camera.h"
#include "esp_http_server.h"

// ---------- WiFi ----------
const char* ssid     = "Airtel_srin_4723";
const char* password = "Air@06268";

// ---------- Motor (L298N) ----------
#define IN1 14
#define IN2 15
// ENA assumed tied HIGH via the L298N board's jumper cap (always enabled, no speed control)

// ---------- Servo (steering) ----------
#define SERVO_PIN 13
#define SERVO_CENTER 97
#define SERVO_LEFT   52
#define SERVO_RIGHT 142

// ---------- AI-Thinker Camera Pins ----------
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

WebServer server(80);
Servo steeringServo;

String currentCommand = "STOP";
unsigned long lastCommandMillis = 0;
const unsigned long COMMAND_TIMEOUT_MS = 1000;

// ---------- Stream server (port 81) ----------
httpd_handle_t stream_httpd = NULL;

#define PART_BOUNDARY "123456789000000000000987654321"

static const char* STREAM_CONTENT_TYPE =
"multipart/x-mixed-replace;boundary=" PART_BOUNDARY;

static const char* STREAM_BOUNDARY =
"\r\n--" PART_BOUNDARY "\r\n";

static const char* STREAM_PART =
"Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

// ==========================================================
// CAMERA STREAM (runs on its own httpd task -- never blocks
// the port-80 control server, and vice versa)
// ==========================================================

static esp_err_t stream_handler(httpd_req_t *req) {
  camera_fb_t *fb = NULL;
  esp_err_t res = ESP_OK;
  char part_buf[64];

  res = httpd_resp_set_type(req, STREAM_CONTENT_TYPE);
  if (res != ESP_OK) return res;

  while (true) {
    fb = esp_camera_fb_get();

    if (!fb) {
      // No frame ready -- brief backoff instead of a tight busy-loop,
      // which could otherwise trip the watchdog timer.
      delay(5);
      continue;
    }

    if (fb->format != PIXFORMAT_JPEG) {
      esp_camera_fb_return(fb);
      continue;
    }

    size_t hlen = snprintf(part_buf, 64, STREAM_PART, fb->len);

    res = httpd_resp_send_chunk(req, STREAM_BOUNDARY, strlen(STREAM_BOUNDARY));
    if (res == ESP_OK) res = httpd_resp_send_chunk(req, part_buf, hlen);
    if (res == ESP_OK) res = httpd_resp_send_chunk(req, (const char *)fb->buf, fb->len);

    // Always return the frame buffer, even on a failed send -- otherwise
    // the fixed-size frame buffer pool empties out and the camera stalls.
    esp_camera_fb_return(fb);

    if (res != ESP_OK) break;
  }

  return res;
}

void startCameraServer() {
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.server_port = 81;
  config.ctrl_port   = 81;

  httpd_uri_t stream_uri = {
    .uri     = "/stream",
    .method  = HTTP_GET,
    .handler = stream_handler,
    .user_ctx = NULL
  };

  if (httpd_start(&stream_httpd, &config) == ESP_OK) {
    httpd_register_uri_handler(stream_httpd, &stream_uri);
  }
}

// ==========================================================
// MOTOR + STEERING
// ==========================================================

void motorsStop() {
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
}

void motorsForward() {
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
}

void motorsBackward() {
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH);
}

void setSteering(int angle) {
  steeringServo.write(angle);
}

// Single source of truth for what each command means.
void applyCommand(const String &dir) {
  if (dir == "FORWARD") {
    motorsForward();
    setSteering(SERVO_CENTER);
  } else if (dir == "BACKWARD") {
    motorsBackward();
    setSteering(SERVO_CENTER);
  } else if (dir == "LEFT") {
    motorsForward();
    setSteering(SERVO_LEFT);
  } else if (dir == "RIGHT") {
    motorsForward();
    setSteering(SERVO_RIGHT);
  } else { // STOP or unrecognized -> always fail safe
    motorsStop();
    setSteering(SERVO_CENTER);
  }
  currentCommand = dir;
}

// ==========================================================
// HTTP CONTROL (port 80)
// ==========================================================

void handleCommand() {
  if (!server.hasArg("dir")) {
    server.send(400, "text/plain", "Missing dir");
    return;
  }

  String dir = server.arg("dir");
  applyCommand(dir);
  lastCommandMillis = millis();

  server.send(200, "text/plain", "OK " + dir);
}

void handleRoot() {
  String msg =
    "ESP32-CAM RC CAR\n\n"
    "Current command : " + currentCommand + "\n"
    "Control : /command?dir=FORWARD\n"
    "Stream  : :81/stream";

  server.send(200, "text/plain", msg);
}

void handleNotFound() {
  server.send(404, "text/plain", "Not found");
}

// ==========================================================
// SETUP
// ==========================================================

void setup() {
  Serial.begin(115200);

  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  motorsStop();

  // Reserve a LEDC timer the camera does NOT use for its XCLK generation
  // (the camera driver claims LEDC_TIMER_0 / LEDC_CHANNEL_0). Without this,
  // ESP32Servo can silently grab the same timer, causing the servo to
  // freeze or jitter as soon as the camera starts streaming.
  ESP32PWM::allocateTimer(1);
  steeringServo.setPeriodHertz(50);
  steeringServo.attach(SERVO_PIN, 500, 2400);
  steeringServo.write(SERVO_CENTER);

  // -------- Camera --------
  camera_config_t cam;

  cam.ledc_channel = LEDC_CHANNEL_0;
  cam.ledc_timer   = LEDC_TIMER_0;

  cam.pin_d0 = Y2_GPIO_NUM;
  cam.pin_d1 = Y3_GPIO_NUM;
  cam.pin_d2 = Y4_GPIO_NUM;
  cam.pin_d3 = Y5_GPIO_NUM;
  cam.pin_d4 = Y6_GPIO_NUM;
  cam.pin_d5 = Y7_GPIO_NUM;
  cam.pin_d6 = Y8_GPIO_NUM;
  cam.pin_d7 = Y9_GPIO_NUM;

  cam.pin_xclk  = XCLK_GPIO_NUM;
  cam.pin_pclk  = PCLK_GPIO_NUM;
  cam.pin_vsync = VSYNC_GPIO_NUM;
  cam.pin_href  = HREF_GPIO_NUM;

  cam.pin_sscb_sda = SIOD_GPIO_NUM;
  cam.pin_sscb_scl = SIOC_GPIO_NUM;

  cam.pin_pwdn  = PWDN_GPIO_NUM;
  cam.pin_reset = RESET_GPIO_NUM;

  cam.xclk_freq_hz = 20000000;
  cam.pixel_format = PIXFORMAT_JPEG;

  if (psramFound()) {
    cam.frame_size   = FRAMESIZE_VGA;
    cam.jpeg_quality = 12;
    cam.fb_count     = 2;
    cam.fb_location  = CAMERA_FB_IN_PSRAM;
  } else {
    cam.frame_size   = FRAMESIZE_QVGA;
    cam.jpeg_quality = 15;
    cam.fb_count     = 1;
    cam.fb_location  = CAMERA_FB_IN_DRAM;
  }

  if (esp_camera_init(&cam) != ESP_OK) {
    Serial.println("Camera Init Failed -- restarting");
    delay(1000);
    ESP.restart();
  }

  Serial.println("Camera OK");

  // -------- WiFi --------
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi...");

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println();
  Serial.println("WiFi Connected");
  Serial.print("IP Address: ");
  Serial.println(WiFi.localIP());

  // Port 80 -- control
  server.on("/", handleRoot);
  server.on("/command", handleCommand);
  server.onNotFound(handleNotFound);
  server.begin();

  // Port 81 -- stream
  startCameraServer();

  Serial.print("Control : http://");
  Serial.print(WiFi.localIP());
  Serial.println("/");
  Serial.print("Stream  : http://");
  Serial.print(WiFi.localIP());
  Serial.println(":81/stream");

  lastCommandMillis = millis(); // don't trip watchdog immediately at boot
}

// ==========================================================
// LOOP
// ==========================================================

void loop() {
  server.handleClient();

  // Safety watchdog: no command received recently -> force stop
  if (millis() - lastCommandMillis > COMMAND_TIMEOUT_MS) {
    if (currentCommand != "STOP") {
      motorsStop();
      setSteering(SERVO_CENTER);
      currentCommand = "STOP";
      Serial.println("Watchdog: no command received, stopping.");
    }
  }
}
