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

#define IMG_FILE_NAME "/dev/shm/detect.jpg"
#define START_FILE_NAME "/dev/shm/detect.start"
#define OUTPUT_FILE_NAME "/dev/shm/detect.out"

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
  detectNet *pednet = detectNet::Create(detectNet::PEDNET,0.5f,2);

  detectNet *facenet = detectNet::Create(detectNet::FACENET,0.5f,2);

  if ((!pednet)||(!facenet)) {
    printf("detectnet-console:   failed to initialize detectNet\n");
    return 0;
  }

  pednet->EnableProfiler();
  facenet->EnableProfiler();

  // alloc memory for bounding box & confidence value output arrays
  const uint32_t maxPedBoxes = pednet->GetMaxBoundingBoxes();
  const uint32_t maxFaceBoxes = facenet->GetMaxBoundingBoxes();

  const uint32_t classes = pednet->GetNumClasses();
  const uint32_t FaceClasses = facenet->GetNumClasses();

  float *bbCPU = NULL;
  float *bbCUDA = NULL;
  float *confCPU = NULL;
  float *confCUDA = NULL;

  if (!cudaAllocMapped((void **)&bbCPU, (void **)&bbCUDA,
                       maxPedBoxes * sizeof(float4)) ||
      !cudaAllocMapped((void **)&confCPU, (void **)&confCUDA,
                       maxPedBoxes * classes * sizeof(float))) {
    printf("detectnet-console:  failed to alloc output memory\n");
    return 0;
  }

  float *bbFaceCPU = NULL;
  float *bbFaceCUDA = NULL;
  float *confFaceCPU = NULL;
  float *confFaceCUDA = NULL;

  if (!cudaAllocMapped((void **)&bbFaceCPU, (void **)&bbFaceCUDA,
                       maxFaceBoxes * sizeof(float4)) ||
      !cudaAllocMapped((void **)&confFaceCPU, (void **)&confFaceCUDA,
                       maxFaceBoxes * FaceClasses * sizeof(float))) {
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
      int numPedBoundingBoxes = maxPedBoxes;
      int numFaceBoundingBoxes = maxFaceBoxes;

      const bool result = pednet->Detect(imgCUDA, imgWidth, imgHeight, bbCPU,
                                      &numPedBoundingBoxes, confCPU);
      if (!result) {
        printf("detectnet-console:  failed to classify '%s'\n", IMG_FILE_NAME);
        numPedBoundingBoxes = 0;
      }

      const bool result = facenet->Detect(imgCUDA, imgWidth, imgHeight, bbFaceCPU,
                                      &numFaceBoundingBoxes, confFaceCPU);
      if (!result) {
        printf("detectnet-console:  failed to classify '%s'\n", IMG_FILE_NAME);
        numFaceBoundingBoxes = 0;
      }


	  printf("writing output file");
      FILE *fd = fopen(OUTPUT_FILE_NAME, "w");
      if (fd != NULL) {
        fprintf(fd, "ped,%d\n", numPedBoundingBoxes);
        fprintf(fd, "face,%d\n", numFaceBoundingBoxes);
        fclose(fd);
      }
	  printf(" done.\n");
    }
    sleep(1);
  }

  printf("\nshutting down...\n");
  CUDA(cudaFreeHost(imgCPU));
  delete net;
  return 0;
}
