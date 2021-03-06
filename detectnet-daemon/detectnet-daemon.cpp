/*
 * http://github.com/dusty-nv/jetson-inference
 */

#include "detectNet.h"

#include "cudaMappedMemory.h"

#include <signal.h>
#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

using namespace cv;

#define VIDEO_FILE_NAME "/dev/shm/detect.mp4"
#define START_FILE_NAME "/dev/shm/detect.start"
#define TEMP_FILE_NAME "/dev/shm/detect.tmp"
#define OUTPUT_FILE_NAME "/dev/shm/detect.out"
#define THUMBNAIL_FILE_NAME "/dev/shm/detect.jpg"

#define FRAME_COLS 1024
#define FRAME_ROWS 768

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

// main entry point
int main(int argc, char **argv) {
  if (signal(SIGINT, sig_handler) == SIG_ERR)
    printf("\ncan't catch SIGINT\n");

  // create detectNet
  detectNet *pednet = detectNet::Create(detectNet::PEDNET, 0.8f, 2);

  detectNet *facenet = detectNet::Create(detectNet::FACENET, 0.5f, 2);

  if ((!pednet) || (!facenet)) {
    printf("detectnet-console:   failed to initialize detectNet\n");
    return 0;
  }

  //   pednet->EnableProfiler();
  //   facenet->EnableProfiler();

  // alloc memory for bounding box & confidence value output arrays
  const uint32_t maxPedBoxes = pednet->GetMaxBoundingBoxes();
  const uint32_t maxFaceBoxes = facenet->GetMaxBoundingBoxes();

  const uint32_t classes = pednet->GetNumClasses();
  const uint32_t FaceClasses = facenet->GetNumClasses();

  float *imgCPU = NULL;
  float *imgCUDA = NULL;

  float *bbCPU = NULL;
  float *bbCUDA = NULL;
  float *confCPU = NULL;
  float *confCUDA = NULL;

  float *bbFaceCPU = NULL;
  float *bbFaceCUDA = NULL;
  float *confFaceCPU = NULL;
  float *confFaceCUDA = NULL;

  if (!cudaAllocMapped((void **)&bbCPU, (void **)&bbCUDA,
                       maxPedBoxes * sizeof(float4)) ||
      !cudaAllocMapped((void **)&confCPU, (void **)&confCUDA,
                       maxPedBoxes * classes * sizeof(float)) ||
      !cudaAllocMapped((void **)&bbFaceCPU, (void **)&bbFaceCUDA,
                       maxFaceBoxes * sizeof(float4)) ||
      !cudaAllocMapped((void **)&imgCPU, (void **)&imgCUDA,
                       FRAME_COLS * FRAME_ROWS * sizeof(float) * 4) ||
      !cudaAllocMapped((void **)&confFaceCPU, (void **)&confFaceCUDA,
                       maxFaceBoxes * FaceClasses * sizeof(float))) {
    printf("detectnet-console:  failed to alloc output memory\n");
    return 0;
  }

  bool result;

  Mat frame, rgbaFrame, rgbaFrameF;

  FILE *fd = NULL;
  while (!signal_recieved) {
    if (0 == access(START_FILE_NAME, 0)) {
      bool firstDetection = true;
      remove(START_FILE_NAME);
      fd = fopen(TEMP_FILE_NAME, "w");
      VideoCapture cap(VIDEO_FILE_NAME);
      int frameCounter = 0;
      while (cap.read(frame)) {
        frameCounter++;
        if ((frameCounter % 50) != 1) {
          if (frameCounter == 2) {
            imwrite(THUMBNAIL_FILE_NAME, frame);
          }
          continue;
        }
        if ((frame.cols != FRAME_COLS) || (frame.rows != FRAME_ROWS)) {
          printf("Wrong frame size (%d,%d) \n", frame.cols, frame.rows);
          return false;
        }

        cvtColor(frame, rgbaFrame, CV_BGR2RGBA, 4);
        rgbaFrame.convertTo(rgbaFrameF, CV_32F);
        float *imgRGBA = rgbaFrameF.ptr<float>();
        for (int j = 0; j < FRAME_COLS * FRAME_ROWS * 4; j++) {
          imgCPU[j] = imgRGBA[j];
        }

        int numPedBoundingBoxes = maxPedBoxes;
        int numFaceBoundingBoxes = maxFaceBoxes;

        result = pednet->Detect(imgCUDA, FRAME_COLS, FRAME_ROWS, bbCPU,
                                &numPedBoundingBoxes, confCPU);
        if (!result) {
          printf("detectnet-console:  failed to classify '%s'\n",
                 VIDEO_FILE_NAME);
          numPedBoundingBoxes = 0;
        }

        if (numPedBoundingBoxes != 0) {
          if (firstDetection) {
            firstDetection = false;
            imwrite(THUMBNAIL_FILE_NAME, frame);
            result = facenet->Detect(imgCUDA, FRAME_COLS, FRAME_ROWS, bbFaceCPU,
                                     &numFaceBoundingBoxes, confFaceCPU);
            if (!result) {
              printf("detectnet-console:  failed to classify '%s'\n",
                     VIDEO_FILE_NAME);
              numFaceBoundingBoxes = 0;
            }
          } else {
            numFaceBoundingBoxes = 0;
          }

          fprintf(fd, "%d,ped,%d,face,%d", frameCounter, numPedBoundingBoxes,
                  numFaceBoundingBoxes);
          int n;
          int nc;
          float *bb;

          for (n = 0; n < numPedBoundingBoxes; n++) {
            nc = confCPU[n * 2 + 1];
            bb = bbCPU + (n * 4);
            fprintf(fd, ",%d,%d,%d,%d", (int)bb[0], (int)bb[1], (int)bb[2],
                    (int)bb[3]);
          }
          for (n = 0; n < numFaceBoundingBoxes; n++) {
            nc = confFaceCPU[n * 2 + 1];
            bb = bbFaceCPU + (n * 4);
            fprintf(fd, ",%d,%d,%d,%d", (int)bb[0], (int)bb[1], (int)bb[2],
                    (int)bb[3]);
          }
          fprintf(fd, "\n");
        }
      }
      remove(VIDEO_FILE_NAME);
      if (fd != NULL) {
        fclose(fd);
      }
      rename(TEMP_FILE_NAME, OUTPUT_FILE_NAME);
    } else {
      sleep(1);
    }
  }

  printf("\nshutting down...\n");
  CUDA(cudaFreeHost(imgCPU));
  delete pednet;
  delete facenet;
  return 0;
}
