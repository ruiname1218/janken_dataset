/*
 * Rock-Paper-Scissors inference on Spresense
 *
 * Requires:
 *   - Spresense main board + camera board
 *   - SD card with "model.nnb" at root (exported from NNC: Export -> NNB)
 *   - Model input: 3 x 64 x 64, RGB, pixel range 0.0 - 1.0
 *   - Class order: 0=paper, 1=rock, 2=scissors  (must match train.csv)
 *
 * Usage:
 *   1. Train in NNC at 64x64 RGB with Image Normalization (1/255)
 *   2. Export as .nnb, copy to SD card root as "model.nnb"
 *   3. Open this sketch in Arduino IDE (with Spresense board package)
 *   4. Open Serial Monitor at 115200 baud, point camera at a hand sign
 */

#include <Camera.h>
#include <DNNRT.h>
#include <SDHCI.h>

#define DNN_IMG_W   64
#define DNN_IMG_H   64
#define CAM_IMG_W   320
#define CAM_IMG_H   240
#define CLIP_X      40            // center crop: (320-240)/2
#define CLIP_Y      0
#define CLIP_W      240
#define CLIP_H      240

static const char* kLabels[] = {"paper", "rock", "scissors"};

SDClass     theSD;
DNNRT       dnnrt;
DNNVariable input(3 * DNN_IMG_W * DNN_IMG_H);

static bool g_ready = false;

static void runInference(CamImage& img) {
  CamImage small;
  CamErr err = img.clipAndResizeImageByHW(
      small,
      CLIP_X, CLIP_Y,
      CLIP_X + CLIP_W - 1,
      CLIP_Y + CLIP_H - 1,
      DNN_IMG_W, DNN_IMG_H);
  if (err != CAM_ERR_SUCCESS) {
    Serial.print("clipAndResize err="); Serial.println(err);
    return;
  }

  err = small.convertPixFormat(CAM_IMAGE_PIX_FMT_RGB565);
  if (err != CAM_ERR_SUCCESS) {
    Serial.print("convertPixFormat err="); Serial.println(err);
    return;
  }

  uint16_t* px = (uint16_t*)small.getImgBuff();
  float*    in = input.data();
  const int plane = DNN_IMG_W * DNN_IMG_H;

  // CHW layout: [R-plane][G-plane][B-plane], normalized to 0-1
  for (int i = 0; i < plane; i++) {
    uint16_t v = px[i];
    uint8_t r = (v >> 11) & 0x1F;   // 5 bits
    uint8_t g = (v >> 5)  & 0x3F;   // 6 bits
    uint8_t b =  v        & 0x1F;   // 5 bits
    in[i]             = r / 31.0f;
    in[i +     plane] = g / 63.0f;
    in[i + 2 * plane] = b / 31.0f;
  }

  dnnrt.inputVariable(input, 0);
  dnnrt.forward();
  DNNVariable out = dnnrt.outputVariable(0);

  int top = out.maxIndex();
  Serial.print("pred=");
  Serial.print(kLabels[top]);
  Serial.print("  scores: paper=");
  Serial.print(out[0], 3);
  Serial.print(" rock=");
  Serial.print(out[1], 3);
  Serial.print(" scissors=");
  Serial.println(out[2], 3);
}

static void CamCB(CamImage img) {
  if (!g_ready || !img.isAvailable()) return;
  runInference(img);
}

void setup() {
  Serial.begin(115200);
  while (!Serial) {}

  Serial.println("Mount SD...");
  while (!theSD.begin()) {
    Serial.println("  insert SD card");
    delay(1000);
  }

  Serial.println("Load model.nnb...");
  File nnb = theSD.open("model.nnb");
  if (!nnb) {
    Serial.println("ERROR: model.nnb not found on SD root");
    while (1) delay(1000);
  }
  int ret = dnnrt.begin(nnb);
  if (ret < 0) {
    Serial.print("ERROR: dnnrt.begin failed: ");
    Serial.println(ret);
    while (1) delay(1000);
  }
  nnb.close();
  Serial.println("Model loaded.");

  Serial.println("Start camera...");
  CamErr cerr = theCamera.begin();
  if (cerr != CAM_ERR_SUCCESS) {
    Serial.print("ERROR: camera begin: "); Serial.println(cerr);
    while (1) delay(1000);
  }
  cerr = theCamera.startStreaming(true, CamCB);
  if (cerr != CAM_ERR_SUCCESS) {
    Serial.print("ERROR: startStreaming: "); Serial.println(cerr);
    while (1) delay(1000);
  }

  g_ready = true;
  Serial.println("Ready. Show a hand sign to the camera.");
}

void loop() {
  // Inference runs from the camera callback; nothing to do here.
  delay(1000);
}
