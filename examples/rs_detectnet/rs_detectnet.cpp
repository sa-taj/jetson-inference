// running real time inference with a realsense sensor

#include <signal.h>

#include <iostream>              // for cout
#include <librealsense2/rs.hpp>  // Include RealSense Cross Platform API

#include "cudaColorspace.h"
#include "detectNet.h"
#include "videoOutput.h"

#ifdef HEADLESS
#define IS_HEADLESS() "headless"  // run without display
#else
#define IS_HEADLESS() (const char *)NULL
#endif

bool signal_recieved = false;
const size_t inWidth = 300;
const size_t inHeight = 300;
const float WHRatio = inWidth / (float)inHeight;

void sig_handler(int signo) {
  if (signo == SIGINT) {
    LogVerbose("received SIGINT\n");
    signal_recieved = true;
  }
}

int usage() {
  printf(
      "usage: rs_detectnet [--help] [--network=NETWORK] "
      "[--threshold=THRESHOLD] "
      "...\n");
  printf("                 input_URI [output_URI]\n\n");
  printf(
      "Locate objects in a video/image stream using an object detection "
      "DNN.\n");
  printf("See below for additional arguments that may not be shown above.\n\n");
  printf("positional arguments:\n");
  printf(
      "    input_URI       resource URI of input stream  (see videoSource "
      "below)\n");
  printf(
      "    output_URI      resource URI of output stream (see videoOutput "
      "below)\n\n");

  printf("%s", detectNet::Usage());
  printf("%s", Log::Usage());

  return 0;
}

int main(int argc, char **argv) try {
  using namespace rs2;
  /*
   * parse command line
   */
  commandLine cmdLine(argc, argv, IS_HEADLESS());

  if (cmdLine.GetFlag("help")) return usage();

  /*
   * create output stream
   */
  videoOutput *output = videoOutput::Create(cmdLine, ARG_POSITION(1));

  if (!output) LogError("rs_detectnet:  failed to create output stream\n");

  /*
   * attach signal handler
   */
  if (signal(SIGINT, sig_handler) == SIG_ERR) LogError("can't catch SIGINT\n");

  /*
   * create detection network
   */
  detectNet *net = detectNet::Create(cmdLine);

  if (!net) {
    LogError("detectnet:  failed to load detectNet model\n");
    return 0;
  }

  // parse overlay flags
  const uint32_t overlayFlags = detectNet::OverlayFlagsFromStr(
      cmdLine.GetString("overlay", "box,labels,conf"));

  // Start streaming from Intel RealSense Camera
  pipeline pipe;
  // Create a configuration for configuring the pipeline with a non default
  // profile
  config cfg;
  // Add desired streams to configuration
  cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_RGB8, 30);

  // Instruct pipeline to start streaming with the requested configuration
  pipe.start(cfg);

  uchar3 *image = NULL;
  auto image_size = 640 * 480;
  cudaMalloc(&image, image_size * sizeof(uchar3));

  /*
   * processing loop
   */
  while (!signal_recieved) {
    // Wait for the next set of frames
    auto data = pipe.wait_for_frames();

    auto color_frame = data.get_color_frame();
    auto depth_frame = data.get_depth_frame();

    // If we only received new depth frame,
    // but the color did not update, continue
    static int last_frame_number = 0;
    if (color_frame.get_frame_number() == last_frame_number) continue;
    last_frame_number = color_frame.get_frame_number();

    // copy the image over to gpu
    cudaMemcpy(image, (uchar3 *)color_frame.get_data(),
               image_size * sizeof(uchar3), cudaMemcpyHostToDevice);

    // detect objects in the frame
    detectNet::Detection *detections = NULL;

    const int numDetections =
        net->Detect(image, color_frame.get_width(), color_frame.get_height(),
                    &detections, overlayFlags);

    if (numDetections > 0) {
      LogVerbose("%i objects detected\n", numDetections);

      for (int n = 0; n < numDetections; n++) {
        LogVerbose("detected obj %i  class #%u (%s)  confidence=%f\n", n,
                   detections[n].ClassID,
                   net->GetClassDesc(detections[n].ClassID),
                   detections[n].Confidence);
        LogVerbose("bounding box %i  (%f, %f)  (%f, %f)  w=%f  h=%f\n", n,
                   detections[n].Left, detections[n].Top, detections[n].Right,
                   detections[n].Bottom, detections[n].Width(),
                   detections[n].Height());
      }
    }

    // render outputs
    if (output != NULL) {
      output->Render(image, color_frame.get_width(), color_frame.get_height());

      // update the status bar
      char str[256];
      sprintf(str, "TensorRT %i.%i.%i | %s | Network %.0f FPS",
              NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH,
              precisionTypeToStr(net->GetPrecision()), net->GetNetworkFPS());
      output->SetStatus(str);

      // check if the user quit
      if (!output->IsStreaming()) signal_recieved = true;
    }

    // print out timing info
    net->PrintProfilerTimes();
  }
  /*
   * destroy resources
   */
  LogVerbose("detectnet:  shutting down...\n");

  cudaFree(image);
  SAFE_DELETE(output);
  SAFE_DELETE(net);

  LogVerbose("detectnet:  shutdown complete.\n");
  return EXIT_SUCCESS;
} catch (const rs2::error &e) {
  std::cerr << "RealSense error calling " << e.get_failed_function() << "("
            << e.get_failed_args() << "):\n    " << e.what() << std::endl;
  return EXIT_FAILURE;
} catch (const std::exception &e) {
  std::cerr << e.what() << std::endl;
  return EXIT_FAILURE;
}
