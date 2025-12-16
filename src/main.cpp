#include "esp_camera.h"
#include "Arduino.h"
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"

#include "NeuralNetwork.h"

// ================= CONFIG =================
#define GRID_SIZE 4

// 🔒 FIXED BOARD CROP (DO NOT CHANGE)
#define X_LEFT   68
#define X_RIGHT  232   // width = 180

#define Y_TOP    28
#define Y_BOTTOM 192   // height = 180

#define CROP_W (X_RIGHT - X_LEFT)
#define CROP_H (Y_BOTTOM - Y_TOP)

#define DOWNSAMPLE 3   // 2 = half res, 3 = third res

#define OUT_W (CROP_W / DOWNSAMPLE)
#define OUT_H (CROP_H / DOWNSAMPLE)

// ================= CAMERA =================
#define CAMERA_MODEL_XIAO_ESP32S3
#include "camera_pins.h"

// GRID_SIZE = 4
int preds[4][4];


// ================= GLOBAL BUFFERS =================
static uint8_t cropped_gray[CROP_W * CROP_H];
static uint8_t cell_raw[64 * 64];   // raw cell
static uint8_t cell28[28 * 28];     // preprocessed
static int8_t  cell28_q[28 * 28];   // quantized (optional scratch)

NeuralNetwork nn;
bool ran_once = false;

void printCroppedDownsampled(camera_fb_t *fb) {
  const int bytes_per_pixel = 1; // GRAYSCALE
  const int stride = fb->width;

  Serial.println("\n===== CROPPED_IMAGE_START =====");
  Serial.printf("SIZE %dx%d\n", OUT_W, OUT_H);

  for (int y = Y_TOP; y < Y_BOTTOM; y += DOWNSAMPLE) {
    for (int x = X_LEFT; x < X_RIGHT; x += DOWNSAMPLE) {

      int idx = y * stride + x;
      uint8_t px = fb->buf[idx];

      Serial.printf("%02X", px);
    }
    Serial.println();
    delay(3);  // USB CDC safety
  }

  Serial.println("===== CROPPED_IMAGE_END =====");
}


void print_cell28_ascii(
    const uint8_t *img,
    int idx_r,
    int idx_c
) {
    Serial.printf("\n=== CELL (%d,%d) 28x28 ===\n", idx_r, idx_c);

    for (int y = 0; y < 28; y++) {
        for (int x = 0; x < 28; x++) {
            uint8_t v = img[y * 28 + x];

            // Visual ramp
            char c;
            if (v > 200)      c = '#';
            else if (v > 150) c = 'O';
            else if (v > 100) c = 'o';
            else if (v > 50)  c = '.';
            else              c = ' ';

            Serial.print(c);
        }
        Serial.println();
    }

    Serial.println("========================");
}

void print_input_tensor_ascii(
    const int8_t *img,
    int row,
    int col
) {
    Serial.printf("\n=== MODEL INPUT (%d,%d) INT8 ===\n", row, col);

    for (int y = 0; y < 28; y++) {
        for (int x = 0; x < 28; x++) {
            int8_t v = img[y * 28 + x];

            char c;
            if (v > 40)       c = '#';
            else if (v > 20)  c = 'O';
            else if (v > 5)   c = 'o';
            else if (v > -5)  c = '.';
            else              c = ' ';

            Serial.print(c);
        }
        Serial.println();
    }

    Serial.println("===============================");
}

void dump_cropped_hex(
    const uint8_t *img,
    int w,
    int h
) {
    Serial.println("\n=== CROPPED_HEX_START ===");
    Serial.printf("W=%d,H=%d\n", w, h);

    for (int i = 0; i < w * h; i++) {
        Serial.printf("0x%02X", img[i]);
        if (i != w * h - 1) Serial.print(",");
    }

    Serial.println("\n=== CROPPED_HEX_END ===");
}



// ------------------------------------------------------------
// PREPROCESS ONE CELL: contrast + resize to 28x28
// ------------------------------------------------------------
void preprocess_cell_to_28x28(
    const uint8_t *cell_in,
    int cell_w,
    int cell_h,
    uint8_t *cell_out
) {
    uint8_t minv = 255;
    uint8_t maxv = 0;

    for (int i = 0; i < cell_w * cell_h; i++) {
        uint8_t v = cell_in[i];
        if (v < minv) minv = v;
        if (v > maxv) maxv = v;
    }

    int range = maxv - minv;
    if (range < 30) {
        memset(cell_out, 0, 28 * 28);
        return;
    }

    for (int oy = 0; oy < 28; oy++) {
        for (int ox = 0; ox < 28; ox++) {

            int x0 = (ox * cell_w) / 28;
            int x1 = ((ox + 1) * cell_w) / 28;
            int y0 = (oy * cell_h) / 28;
            int y1 = ((oy + 1) * cell_h) / 28;

            if (x1 <= x0) x1 = x0 + 1;
            if (y1 <= y0) y1 = y0 + 1;

            int sum = 0;
            int count = 0;

            for (int y = y0; y < y1; y++) {
                for (int x = x0; x < x1; x++) {
                    int v = cell_in[y * cell_w + x];
                    int c = (v - minv) * 255 / range;
                    if (c < 0) c = 0;
                    if (c > 255) c = 255;
                    sum += c;
                    count++;
                }
            }

            uint8_t out = sum / count;

            cell_out[oy * 28 + ox] = out;
        }
    }
}

// ------------------------------------------------------------
// ROTATE 180°
// ------------------------------------------------------------
// Flip image vertically (across X-axis)
void flip_vertical(
    uint8_t *img,
    int w,
    int h
) {
    for (int y = 0; y < h / 2; y++) {
        for (int x = 0; x < w; x++) {
            int top = y * w + x;
            int bottom = (h - 1 - y) * w + x;

            uint8_t tmp = img[top];
            img[top] = img[bottom];
            img[bottom] = tmp;
        }
    }
}


// ------------------------------------------------------------
// UINT8 → INT8 normalization
// ------------------------------------------------------------
void normalize_uint8_to_int8(
    const uint8_t *in,
    int8_t *out,
    int n,
    int zero_point,
    float scale
) {
    for (int i = 0; i < n; i++) {
        // Match training preprocessing EXACTLY
        float f = in[i] / 255.0f;

        int q = (int)round(f / scale) + zero_point;

        if (q < -128) q = -128;
        if (q > 127)  q = 127;

        out[i] = (int8_t)q;
    }
}

// ------------------------------------------------------------
// EXTRACT ONE CELL FROM CROPPED BOARD (NO PADDING)
// ------------------------------------------------------------
void extract_cell(
    const uint8_t *board,
    int board_w,
    int board_h,
    int row,
    int col,
    uint8_t *cell_out,
    int &cell_w,
    int &cell_h
) {
    cell_w = board_w / GRID_SIZE;
    cell_h = board_h / GRID_SIZE;

    int x0 = col * cell_w;
    int y0 = row * cell_h;

    int idx = 0;
    for (int y = 0; y < cell_h; y++) {
        for (int x = 0; x < cell_w; x++) {
            cell_out[idx++] = board[(y0 + y) * board_w + (x0 + x)];
        }
    }
}

void dump_cell28_hex(
    const uint8_t *img,
    const char *label
) {
    Serial.printf("\n=== CELL28_%s_START ===\n", label);

    for (int i = 0; i < 28 * 28; i++) {
        Serial.printf("0x%02X", img[i]);
        if (i != 28 * 28 - 1) Serial.print(",");
    }

    Serial.printf("\n=== CELL28_%s_END ===\n", label);
}


void print_cropped_ascii(
    const uint8_t *img,
    int w,
    int h,
    int step    // e.g. 4 → 180/4 = 45 chars wide
) {
    Serial.println("\n=== CROPPED BOARD (ASCII) ===");

    for (int y = 0; y < h; y += step) {
        for (int x = 0; x < w; x += step) {
            uint8_t v = img[y * w + x];

            char c;
            if (v > 200)      c = '#';
            else if (v > 150) c = 'O';
            else if (v > 100) c = 'o';
            else if (v > 50)  c = '.';
            else              c = ' ';

            Serial.print(c);
        }
        Serial.println();
    }

    Serial.println("============================");
}

void print_cell28(
    const uint8_t *img,
    int row,
    int col
) {
    Serial.printf("\n=== CELL (%d,%d) 28x28 HEX ===\n", row, col);

    for (int y = 0; y < 28; y++) {
        for (int x = 0; x < 28; x++) {
            Serial.printf("0x%02X", img[y * 28 + x]);
            if (!(y == 27 && x == 27)) Serial.print(",");
        }
        Serial.println();  // row separator (for readability)
    }

    Serial.println("================================");
}

void print_cropped_lowres_ascii_raw(
    const uint8_t *img,
    int w,
    int h,
    int step   // e.g. 4 → ~45x45 output
) {
    Serial.println("\n=== CROPPED BOARD (RAW, LOW-RES) ===");

    for (int y = 0; y < h; y += step) {
        for (int x = 0; x < w; x += step) {

            // sample center pixel of the block
            int yy = y + step / 2;
            int xx = x + step / 2;
            if (yy >= h) yy = h - 1;
            if (xx >= w) xx = w - 1;

            uint8_t v = img[yy * w + xx];

            char c;
            if (v > 220)      c = '#';
            else if (v > 180) c = 'O';
            else if (v > 140) c = 'o';
            else if (v > 100) c = '.';
            else              c = ' ';

            Serial.print(c);
        }
        Serial.println();
    }

    Serial.println("=================================");
}


// ================= SETUP =================
void setup()
{
    WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);
    Serial.begin(115200);

    camera_config_t c;
    c.ledc_channel = LEDC_CHANNEL_0;
    c.ledc_timer   = LEDC_TIMER_0;

    c.pin_d0 = Y2_GPIO_NUM;
    c.pin_d1 = Y3_GPIO_NUM;
    c.pin_d2 = Y4_GPIO_NUM;
    c.pin_d3 = Y5_GPIO_NUM;
    c.pin_d4 = Y6_GPIO_NUM;
    c.pin_d5 = Y7_GPIO_NUM;
    c.pin_d6 = Y8_GPIO_NUM;
    c.pin_d7 = Y9_GPIO_NUM;
    c.pin_xclk = XCLK_GPIO_NUM;
    c.pin_pclk = PCLK_GPIO_NUM;
    c.pin_vsync = VSYNC_GPIO_NUM;
    c.pin_href  = HREF_GPIO_NUM;
    c.pin_sccb_sda = SIOD_GPIO_NUM;
    c.pin_sccb_scl = SIOC_GPIO_NUM;
    c.pin_pwdn  = PWDN_GPIO_NUM;
    c.pin_reset = RESET_GPIO_NUM;

    c.xclk_freq_hz = 20000000;
    c.pixel_format = PIXFORMAT_GRAYSCALE;
    c.frame_size   = FRAMESIZE_QVGA;
    c.fb_count     = 1;

    if (esp_camera_init(&c) != ESP_OK) {
        Serial.println("Camera init failed");
        while (1);
    }

    Serial.println("Camera + model initialized");
}

// ================= LOOP =================
void loop()
{
    if (ran_once) return;
    ran_once = true;

    delay(2000);

    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
        Serial.println("Frame capture failed");
        return;
    }

    uint8_t *img = fb->buf;

    // -------- Crop board (DO NOT CHANGE LOGIC) --------
    for (int y = 0; y < CROP_H; y++) {
        for (int x = 0; x < CROP_W; x++) {
            int sx = X_LEFT + x;
            int sy = Y_TOP  + y;
            cropped_gray[y * CROP_W + x] = img[sy * fb->width + sx];
        }
    }

    // print_cropped_lowres_ascii_raw(cropped_gray, CROP_W, CROP_H, 4);

    Serial.println("\n=== SUDOKU PREDICTION ===");

    for (int r = 0; r < GRID_SIZE; r++) {
        for (int c = 0; c < GRID_SIZE; c++) {

            int cell_w, cell_h;

            extract_cell(
                cropped_gray,
                CROP_W,
                CROP_H,
                r,
                c,
                cell_raw,
                cell_w,
                cell_h
            );

            preprocess_cell_to_28x28(cell_raw, cell_w, cell_h, cell28);
            flip_vertical(cell28, 28, 28);

            // Dump ONLY corner cells
            if ((r == 0 && c == 0) ||               // top-left
                (r == 0 && c == GRID_SIZE - 1) ||   // top-right
                (r == GRID_SIZE - 1 && c == 0) ||   // bottom-left
                (r == GRID_SIZE - 1 && c == GRID_SIZE - 1)) {

                const char *label = nullptr;

                if (r == 0 && c == 0) label = "TOP_LEFT";
                else if (r == 0 && c == GRID_SIZE - 1) label = "TOP_RIGHT";
                else if (r == GRID_SIZE - 1 && c == 0) label = "BOTTOM_LEFT";
                else if (r == GRID_SIZE - 1 && c == GRID_SIZE - 1) label = "BOTTOM_RIGHT";
            }


            // print_cell28(cell28, r, c);

            TfLiteTensor* input = nn.getInput();

            normalize_uint8_to_int8(
                cell28,
                input->data.int8,
                28 * 28,
                input->params.zero_point,
                input->params.scale
            );

            // print_cell28_ascii(cell28, r, c);
            // print_input_tensor_ascii(input->data.int8, r, c);

            if (nn.predict() != kTfLiteOk) {
                Serial.println("Inference failed");
                esp_camera_fb_return(fb);
                return;
            }

            int pred = nn.getPredictedClass();
            preds[GRID_SIZE - 1 - r][c] = pred;
        }
        Serial.println();
    }

    Serial.println("\n=== FINAL PREDICTION GRID ===");
    for (int r = 0; r < GRID_SIZE; r++) {
        for (int c = 0; c < GRID_SIZE; c++) {
            int v = preds[r][c];

            // Optional: map 0 -> blank
            if (v == 0) Serial.print("_ ");
            else        Serial.print(v), Serial.print(" ");
        }
        Serial.println();
    }

    printCroppedDownsampled(fb);

    esp_camera_fb_return(fb);
}