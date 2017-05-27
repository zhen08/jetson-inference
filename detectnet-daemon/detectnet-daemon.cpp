/*
 * http://github.com/dusty-nv/jetson-inference
 */

#include "detectNet.h"

#include "cudaMappedMemory.h"

#include <QImage>
#include <signal.h>
#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>

#define IMG_FILE_NAME "/dev/shm/detectped.jpg"
#define START_FILE_NAME "/dev/shm/detectped.start"
#define OUTPUT_FILE_NAME "/dev/shm/detectped.out"

uint64_t current_timestamp() {
  struct timeval te;
  gettimeofday(&te, NULL);                       // get current time
  return te.tv_sec * 1000LL + te.tv_usec / 1000; // caculate milliseconds
}

bool signal_recieved = false;

void sig_handler(int signo) {
  if (signo == SIGINT) {
    printf("received SIGINT\n");
    signal_recieved = true;
  }
}

bool loadImage(float4 **cpu, float4 **gpu, int *width, int *height) {
  if (0 != access(START_FILE_NAME, 0))
    return false;
  remove(START_FILE_NAME);

  // load original image
  QImage qImg;

  if (!qImg.load(IMG_FILE_NAME)) {
    printf("failed to load image %s\n", IMG_FILE_NAME);
    return false;
  }

  remove(IMG_FILE_NAME);

  const uint32_t imgWidth = qImg.width();
  const uint32_t imgHeight = qImg.height();
  const uint32_t imgPixels = imgWidth * imgHeight;
  const size_t imgSize = imgWidth * imgHeight * sizeof(float) * 4;

  // allocate buffer for the image
  if (!cudaAllocMapped((void **)cpu, (void **)gpu, imgSize)) {
    printf(LOG_CUDA "failed to allocated %zu bytes for image %s\n", imgSize,
           IMG_FILE_NAME);
    return false;
  }

  float4 *cpuPtr = *cpu;

  for (uint32_t y = 0; y < imgHeight; y++) {
    for (uint32_t x = 0; x < imgWidth; x++) {
      const QRgb rgb = qImg.pixel(x, y);
      const float4 px = make_float4(float(qRed(rgb)), float(qGreen(rgb)),
                                    float(qBlue(rgb)), float(qAlpha(rgb)));

      cpuPtr[y * imgWidth + x] = px;
    }
  }

  *width = imgWidth;
  *height = imgHeight;
  return true;
}

// main entry point
int main(int argc, char **argv) {
  if (signal(SIGINT, sig_handler) == SIG_ERR)
    printf("\ncan't catch SIGINT\n");

  // create detectNet
  detectNet *net = detectNet::Create(argc, argv);

  if (!net) {
    printf("detectnet-console:   failed to initialize detectNet\n");
    return 0;
  }

  net->EnableProfiler();

  // alloc memory for bounding box & confidence value output arrays
  const uint32_t maxBoxes = net->GetMaxBoundingBoxes();
  printf("maximum bounding boxes:  %u\n", maxBoxes);
  const uint32_t classes = net->GetNumClasses();

  float *bbCPU = NULL;
  float *bbCUDA = NULL;
  float *confCPU = NULL;
  float *confCUDA = NULL;

  if (!cudaAllocMapped((void **)&bbCPU, (void **)&bbCUDA,
                       maxBoxes * sizeof(float4)) ||
      !cudaAllocMapped((void **)&confCPU, (void **)&confCUDA,
                       maxBoxes * classes * sizeof(float))) {
    printf("detectnet-console:  failed to alloc output memory\n");
    return 0;
  }

  // load image from file on disk
  float *imgCPU = NULL;
  float *imgCUDA = NULL;
  int imgWidth = 0;
  int imgHeight = 0;

  while (!signal_recieved) {
    if (loadImage((float4 **)&imgCPU, (float4 **)&imgCUDA, &imgWidth,
                  &imgHeight)) {
      int numBoundingBoxes = maxBoxes;

      printf("detectnet-console:  beginning processing network (%zu)\n",
             current_timestamp());

      const bool result = net->Detect(imgCUDA, imgWidth, imgHeight, bbCPU,
                                      &numBoundingBoxes, confCPU);

      printf("detectnet-console:  finished processing network  (%zu)\n",
             current_timestamp());

      if (!result) {
        printf("detectnet-console:  failed to classify '%s'\n", IMG_FILE_NAME);
        numBoundingBoxes = 0;
      }
      FILE *f = fopen("OUTPUT_FILE_NAME", "w");
      if (f == NULL) {
        printf("Error opening file!\n");
        return 0;
      }
      fprintf(f, "%d", numBoundingBoxes);
      fclose(f);
    }
    sleep(1);
  }

  printf("\nshutting down...\n");
  CUDA(cudaFreeHost(imgCPU));
  delete net;
  return 0;
}
