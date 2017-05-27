/*
 * http://github.com/dusty-nv/jetson-inference
 */

#include "glDisplay.h"
#include "glTexture.h"

#include <stdio.h>
#include <signal.h>
#include <unistd.h>

#include "cudaMappedMemory.h"
#include "cudaNormalize.h"
#include "cudaFont.h"

#include "detectNet.h"

#include "opencv2/imgproc.hpp"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"

using namespace cv;

bool signal_recieved = false;

void sig_handler(int signo)
{
	if( signo == SIGINT )
	{
		printf("received SIGINT\n");
		signal_recieved = true;
	}
}


int main( int argc, char** argv )
{
	Mat frame;
	Mat rgbaFrame,rgbaFrameF;

	printf("detectnet-camera\n  args (%i):  ", argc);

	for( int i=0; i < argc; i++ )
		printf("%i [%s]  ", i, argv[i]);
		
	printf("\n\n");
	
	if( signal(SIGINT, sig_handler) == SIG_ERR )
		printf("\ncan't catch SIGINT\n");

	VideoCapture capture(argv[argc-1]);

	if (!capture.isOpened()) {
		printf("Error opening the stream");
		return 0;
	}

	if (!capture.read(frame)) {
		printf("Error capturing the first fame");
		return 0;
    }
	printf("Frame width %d  height %d",frame.cols,frame.rows);

	// for (int ii=0;ii<10;ii++) {
	// 	if (!capture.read(frame)) 
	// 		printf("\ndetectnet-camera:  failed to capture frame\n");
	// 	cv::cvtColor(frame, rgbaFrame, CV_BGR2RGBA, 4);
	// 	imshow("captured",rgbaFrame);
	// 	waitKey(30);
	// }

	/*
	 * create detectNet
	 */
	detectNet* net = detectNet::Create(1, argv);
	
	if( !net )
	{
		printf("detectnet-camera:   failed to initialize imageNet\n");
		return 0;
	}


	/*
	 * allocate memory for output bounding boxes and class confidence
	 */
	const uint32_t maxBoxes = net->GetMaxBoundingBoxes();		printf("maximum bounding boxes:  %u\n", maxBoxes);
	const uint32_t classes  = net->GetNumClasses();
	
	float* imgCPU  = NULL;
	float* imgCUDA = NULL;
	float* bbCPU    = NULL;
	float* bbCUDA   = NULL;
	float* confCPU  = NULL;
	float* confCUDA = NULL;
	
	if( !cudaAllocMapped((void**)&bbCPU, (void**)&bbCUDA, maxBoxes * sizeof(float4)) ||
		!cudaAllocMapped((void**)&imgCPU, (void**)&imgCUDA, frame.cols * frame.rows * sizeof(float4)) ||
	    !cudaAllocMapped((void**)&confCPU, (void**)&confCUDA, maxBoxes * classes * sizeof(float)) )
	{
		printf("detectnet-console:  failed to alloc output memory\n");
		return 0;
	}
	

	/*
	 * create openGL window
	 */
	glDisplay* display = glDisplay::Create();
	glTexture* texture = NULL;
	
	if( !display ) {
		printf("\ndetectnet-camera:  failed to create openGL display\n");
	}
	else
	{
		texture = glTexture::Create(frame.cols, frame.rows, GL_RGBA32F_ARB/*GL_RGBA8*/);

		if( !texture )
			printf("detectnet-camera:  failed to create openGL texture\n");
	}
	
	
	/*
	 * create font
	 */
	cudaFont* font = cudaFont::Create();
	
	
	/*
	 * processing loop
	 */
	float confidence = 0.0f;
	
	while( !signal_recieved )
	{
		
		// get the latest frame
		if (!capture.read(frame)) 
			printf("\ndetectnet-camera:  failed to capture frame\n");
		cv::cvtColor(frame, rgbaFrame, CV_RGB2RGBA, 4);
		rgbaFrame.convertTo(rgbaFrameF,CV_32F);
		imwrite("original.jpg",frame);
		imwrite("rgba.jpg",rgbaFrame);

		// convert to RGBA
		float* imgRGBA = rgbaFrameF.ptr<float>();
		
		// classify image with detectNet
		int numBoundingBoxes = maxBoxes;
		
		for (int j=0;j<rgbaFrameF.rows*rgbaFrameF.cols*4;j++){
			imgCPU[j] = imgRGBA[j];
		} 
		waitKey(30);

		printf("\n\n imgRGBA %d, imgCPU %d, width %d, height %d, bbCPU %d \n\n",imgRGBA,imgCPU,rgbaFrameF.cols,rgbaFrameF.rows,bbCPU);
		if(0 && net->Detect((float*)imgCPU, frame.cols, frame.rows, bbCPU, &numBoundingBoxes, confCPU))
		{
			printf("%i bounding boxes detected\n", numBoundingBoxes);
		
			int lastClass = 0;
			int lastStart = 0;
			
			for( int n=0; n < numBoundingBoxes; n++ )
			{
				const int nc = confCPU[n*2+1];
				float* bb = bbCPU + (n * 4);
				
				printf("bounding box %i   (%f, %f)  (%f, %f)  w=%f  h=%f\n", n, bb[0], bb[1], bb[2], bb[3], bb[2] - bb[0], bb[3] - bb[1]); 
				
				if( nc != lastClass || n == (numBoundingBoxes - 1) )
				{
					if( !net->DrawBoxes((float*)imgCPU, (float*)imgRGBA, frame.cols, frame.rows, 
						                        bbCUDA + (lastStart * 4), (n - lastStart) + 1, lastClass) )
						printf("detectnet-console:  failed to draw boxes\n");
						
					lastClass = nc;
					lastStart = n;

					CUDA(cudaDeviceSynchronize());
				}
			}
		
			/*if( font != NULL )
			{
				char str[256];
				sprintf(str, "%05.2f%% %s", confidence * 100.0f, net->GetClassDesc(img_class));
				
				font->RenderOverlay((float4*)imgRGBA, (float4*)imgRGBA, camera->GetWidth(), camera->GetHeight(),
								    str, 10, 10, make_float4(255.0f, 255.0f, 255.0f, 255.0f));
			}*/
			
			if( display != NULL )
			{
				char str[256];
				sprintf(str, "TensorRT build %x | %s | %04.1f FPS", NV_GIE_VERSION, net->HasFP16() ? "FP16" : "FP32", display->GetFPS());
				//sprintf(str, "GIE build %x | %s | %04.1f FPS | %05.2f%% %s", NV_GIE_VERSION, net->GetNetworkName(), display->GetFPS(), confidence * 100.0f, net->GetClassDesc(img_class));
				display->SetTitle(str);	
			}	
		}	


		// update display
		if( display != NULL )
		{
			display->UserEvents();
			display->BeginRender();

			if( texture != NULL )
			{
				// rescale image pixel intensities for display
				CUDA(cudaNormalizeRGBA((float4*)imgCPU, make_float2(0.0f, 255.0f), 
								   (float4*)imgCPU, make_float2(0.0f, 1.0f), 
		 						   frame.cols, frame.rows));

				// map from CUDA to openGL using GL interop
				void* tex_map = texture->MapCUDA();

				if( tex_map != NULL )
				{
					cudaMemcpy(tex_map, imgCPU, texture->GetSize(), cudaMemcpyDeviceToDevice);
					texture->Unmap();
				}

				// draw the texture
				texture->Render(100,100);		
			}

			display->EndRender();
		}
	}
	
	printf("\ndetectnet-camera:  un-initializing video device\n");
	
	capture.release();

	if( display != NULL )
	{
		delete display;
		display = NULL;
	}
	
	printf("detectnet-camera:  video device has been un-initialized.\n");
	printf("detectnet-camera:  this concludes the test of the video device.\n");
	return 0;
}

