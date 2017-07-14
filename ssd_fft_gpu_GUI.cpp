/*
Developed by Pinar Muyan-Ozcelik, March 2010
This project displays the raw video (generated by convertPGMtoRawVideo phyton code which takes sequence of PGM images captured
by EyeQ client and converts it to uchar pixel array) and feeds it to the template-based pipeline by linking to ssd_fft_gpu.dll
This way we do not need Realis front-end to visualize the videos.
Notes:
This project is adapted from ssd_fft_gpu_TestDLL and SimpleGL
SimpleGL creates a VBO on the GPU and both CUDA and OpenGL uses this object for computation.
This way data is not copied back and forth between host and device.
For now I will not create PBO/VBO and use it in CUDA (it will be my second step 
if the display/computation is so slow)
Currently I am only copying the frame to the texture and drawing it to the screen.
glutBitmapCharacter vs glutStrokeCharacter:
-------------------------------------------
# Bitmapped Font
    * relatively simple to define
    * don't scale or rotate easily
# Outline Font - a.k.a Stroke, Vector, TrueType, Scalable, etc.

    * composed of lines, polygons, curves in canonical space
    * more difficult to define
    * scale and rotate easily
    * may be scan-converted into bitmapped font caches for faster display

*/
////////////////////////////////////////////////////////////////////////////////////////////////////////////////// AOCL
/*
#include <assert.h>
#include <stdlib.h>
#include <math.h>
#include <cstring>
#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"

using namespace aocl_utils;

#define STRING_BUFFER_LEN 1024

// Runtime constants
// Used to define the work set over which this kernel will execute.
static const size_t work_group_size = 8;  // 8 threads in the demo workgroup
										  // Defines kernel argument value, which is the workitem ID that will
										  // execute a printf call
static const int thread_id_to_output = 2;

// OpenCL runtime configuration
static cl_platform_id platform = NULL;
static cl_device_id device = NULL;
static cl_context context = NULL;
static cl_command_queue queue = NULL;
static cl_kernel kernel = NULL;
static cl_program program = NULL;

// Function prototypes
bool init();
void AOCLcleanup();
static void device_info_ulong(cl_device_id device, cl_device_info param, const char* name);
static void device_info_uint(cl_device_id device, cl_device_info param, const char* name);
static void device_info_bool(cl_device_id device, cl_device_info param, const char* name);
static void device_info_string(cl_device_id device, cl_device_info param, const char* name);
static void display_device_info(cl_device_id device);*/
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include <stdio.h>
//#include <ssd_fft_gpu_dll.h>
#include <ssd_fft_gpu_common.h>

// includes, GL
#include <GL/glew.h>
#include <GL/glut.h>

// includes
//#include <cutil_inline.h>
#include <helper_cuda.h>
#include <helper_cuda_gl.h>
#include <helper_functions.h>
#include <helper_timer.h>
//#include <cutil_gl_error.h>
//#include <cuda_gl_interop.h>


#include <time.h>
#define CUTFalse false
#define CUTTrue true
#define MAX_VIDEO_NUM 1000 //lets says we can have at most 1000 videos listed in videos.txt
#define PRINT //print details like frame num and video name
/*#ifdef US_SIGNS
		const int giScnBegY = 0;
		const int giScnW = 640;
		const int giScnH = 390;
		const int giShowScnW = 640;
		const int giShowScnH = 384;
		char acMeasure[6] = " mph";
#else
		const int giScnBegY = 48;
		const int giScnW = 640;
		const int giScnH = 480;
		const int giShowScnW = 640;
		const int giShowScnH = 300;//230;
		char acMeasure[6] = " km/h";
#endif*/

const int giScnSz = giScnW * giScnH;
const int giScnOffset = giScnBegY * giScnW;

const int giScnMemSzUChar = giScnSz  * sizeof(unsigned char);
#ifdef US_SIGNS
char g_sPathBegin_GUI[50] = "../convert_pgm_to_RawVideo/raw/";
#else
char g_sPathBegin_GUI[60] = "../../../copied15May17/EU_raw(savedRealisFilesAsBin)/";
#endif
char g_sPath_GUI[100];
FILE *gfVideo;
unsigned char *gacScn;
static GLuint texid = 0;       // Texture for display
bool gbPause = true;
bool gbRewind = false;
bool gbOpenNewVideo = false;
bool gbSlow = false;
bool gbRecog = true;
bool gbClahe = false;
#ifdef US_SIGNS
char gcVideoListFile[] = "videos.txt";
#else
char gcVideoListFile[] = "videos-EU.txt";
#endif
FILE *gfVideoList;
char gcCurrentVideo[100];
//ssd_fft_gpu_findBestTpl params (we do not use gfile_info and gulTimeStamp in GUI mode)
int giMaxPeakIndx, giPartW, giPartH;
int giSLCurFrm_GUI = -1;
int giSLResult_GUI = -1;
int giLastSL = -1;
int gfile_info[3];
unsigned long gulTimeStamp = 0;
int giRow, giCol;
unsigned int gauiVideoListLinePos[MAX_VIDEO_NUM]; 
int iVideoListLine = 0;
bool gbGoEndInNewVideo = false;
int giFrameNum = 0;
int gaiVideoFrameNum[MAX_VIDEO_NUM];
bool gbUpdateFrameNo = false;
char gacClipName_GUI[11];
bool gbShowBox = true; //show red box around the classified sign

//Get the full path name
char* getFullPathOfFile_GUI(char* pcFileName)
{
	strcpy(g_sPath_GUI, g_sPathBegin_GUI);
	strcat(g_sPath_GUI, pcFileName);
	return g_sPath_GUI;
}


void cleanup()
{
#ifdef PRINT
	printf("Cleaning up!\n");
#endif
	ssd_fft_gpu_exit();
	free(gacScn);
	glDeleteTextures(1, &texid);
}

void RecognizeSigns()
{
	gfile_info[0] = giFrameNum-1;
	if (gbClahe)
		gfile_info[2] = -1; //use this flag to turn on CLAHE showing in ssd_fft_gpu (originally it is used to pass last frame num)

	else
		gfile_info[2] = 0;
	ssd_fft_gpu_returnBestTpl(gacScn, &giMaxPeakIndx, &giPartW, &giPartH, gfile_info, gulTimeStamp, &giSLCurFrm_GUI, &giSLResult_GUI, gacClipName_GUI);

	//cuda is row major and zero-based
	giCol = (giMaxPeakIndx-giScnOffset)%giScnW;
	//regular division of integer returns floor
	giRow = (giMaxPeakIndx-giScnOffset)/giScnW;
	//printf("Max peak index row: %d, col: %d\n", giRow, giCol); 
}

void SleepNow()
{
	//I wanted to use sleep() function in C but this program is having trouble finding the unistd.h in cygwin/usr/include/sys
/*
	time_t tEnd;
	for (int i= 0; i< 100000; i++)
	{
	time (&tEnd);
	}
*/
	/*time_t tStart,tEnd;
	time (&tStart);
	time (&tEnd);

	while ((difftime ( tEnd, tStart )) < 1)
	{
	time (&tEnd);
	
	}*/
	int clo = clock();
	while ((clock() - clo) < 50)
	{	}

}
void WriteTitleAndNumber(int iXPos, char* acTitle, int iNumChar, int iSL)
	{
	char acBuffer[20];
	glPushMatrix();

	glLineWidth( 4 );
	glTranslatef(iXPos, 290, 0);
	//don't know why but the text appears upsite down and flipped around vertical axis, fix this with rotations
	glRotatef(180, 0, 0, 1);
	glRotatef(180, 0, 1, 0);
	//for antialiasing
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glEnable(GL_BLEND);
	glEnable(GL_LINE_SMOOTH);

	//Write current vote
	glScalef(0.25, 0.25, 0.25);
	strcpy(acBuffer, acTitle);
	for (int i = 0; i < iNumChar; i++)
		glutStrokeCharacter(GLUT_STROKE_ROMAN, acBuffer[i]);
	glScalef(2, 2, 2);
	itoa(iSL, acBuffer, 10);
	for (int i = 0; i < 3; i++)
		glutStrokeCharacter(GLUT_STROKE_ROMAN, acBuffer[i]);
	glScalef(0.5, 0.5, 0.5);
	strcpy(acBuffer, acMeasure);
	for (int i = 0; i < 5; i++)
		glutStrokeCharacter(GLUT_STROKE_ROMAN, acBuffer[i]);
	
	glPopMatrix();
}

////////////////////////////////////////////////////////////////////////////////
//! Display callback
////////////////////////////////////////////////////////////////////////////////
void display()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // set view matrix
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    // get the new frame
	if(gbOpenNewVideo)
		{
		gbOpenNewVideo = false;
		gauiVideoListLinePos[iVideoListLine] = ftell(gfVideoList);
		if (fscanf(gfVideoList, "%s\n", gcCurrentVideo) == -1)
			{
			//end of video list! 
			if (gbUpdateFrameNo) 
			{
				gaiVideoFrameNum[iVideoListLine-1] = giFrameNum; //number of frames in the video (last 0-based index + 1)
				gbUpdateFrameNo = false;
			}
#ifdef PRINT
			printf("End of Video List\n"); 
#endif
			gbPause = true;
			}
		else 
			{
			if (gbUpdateFrameNo && iVideoListLine > 0) 
			{
				gaiVideoFrameNum[iVideoListLine-1] = giFrameNum;
				gbUpdateFrameNo = false;
			}
			giFrameNum = 0;
			giSLCurFrm_GUI = -1;
			giSLResult_GUI = -1;
			giLastSL = -1;
			iVideoListLine ++;
#ifdef PRINT
			printf("Reading videofile: %s\n", gcCurrentVideo);
#endif
			strncpy(gacClipName_GUI, gcCurrentVideo, 11);

			gfVideo = fopen(getFullPathOfFile_GUI(gcCurrentVideo), "rb");
			if(gfVideo!=NULL)
			printf("Path: %s", getFullPathOfFile_GUI(gcCurrentVideo));
			else
			printf("Video doesn't exist!");

			
			if (gbGoEndInNewVideo)
				{
				gbGoEndInNewVideo = false;
				fseek(gfVideo, giScnSz, SEEK_END); //we go 2 scenes back in rewind, this is why we add one scn to the end
				giFrameNum = gaiVideoFrameNum[iVideoListLine-1]+1;
				}
			}
		}

	if (!gbPause)
	{
		bool bReadScn = true;
		if (gbRewind)
		{
			if (fseek(gfVideo,-giScnSz*2,SEEK_CUR) != 0) 
			{
				//we reached the beginning of current file, 
				iVideoListLine = iVideoListLine - 2;
				if (iVideoListLine == -1)
				{
					//we are processing the very first video, go to the beginning of the video
					fseek(gfVideo,0,SEEK_SET);
					giFrameNum = 0;
					iVideoListLine = 1; //we do not need to reopen the first (index 0) video, just increment it as if we reopened it.
#ifdef PRINT
					printf("Beginning of Video List!\n");
#endif
					gbPause = true;
				}
				else //open prev file in list and go to end of it
				{
					gbOpenNewVideo = true;	
					gbGoEndInNewVideo = true;
					giFrameNum = gaiVideoFrameNum[iVideoListLine];
				}
				fseek(gfVideoList, gauiVideoListLinePos[iVideoListLine], SEEK_SET);
				bReadScn = false; //do not read scn yet.. first open new file, etc in next display screen...
			}
			else
				giFrameNum = giFrameNum - 2;
		}
		if (bReadScn)//good
		{
			//printf("num=%d", giScnSz);
			if (fread(gacScn, sizeof(unsigned char), giScnSz, gfVideo) != giScnMemSzUChar) // actual reading in frame
			{
				//end of video, read the next video
				gbOpenNewVideo = true;
				gbUpdateFrameNo = true;
			}
			else //process the new frame
			{
#ifdef PRINT
				printf("Processing frame number: %d\n", giFrameNum);
#endif
				giFrameNum++;
				if (gbSlow)
					SleepNow();
				if (gbRecog)
					RecognizeSigns(); // GONE!
			}
		}
	}
	
	glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, giShowScnW, giShowScnH, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, gacScn+giScnOffset);
	//render the new frame
	glEnable(GL_TEXTURE_2D);
	glBegin(GL_QUADS);
	glTexCoord2f(0, 0); glVertex2f(0, 0); 
	glTexCoord2f(1, 0); glVertex2f(giShowScnW, 0); 
	glTexCoord2f(1, 1); glVertex2f(giShowScnW, giShowScnH); 
	glTexCoord2f(0, 1); glVertex2f(0, giShowScnH);
	glEnd(); 
	glDisable(GL_TEXTURE_2D);

if (gbShowBox)
{
	//draw the rectangle that shows the candidate sign location
	glLineWidth( 4.0f );
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE); //mode can be GL_POINT, GL_LINE, or GL_FILL to indicate whether the polygon should be drawn as points, outlined, or filled.
	glColor4f(1.0f, 0.0f, 0.0f, 1.0f);
	glBegin(GL_QUADS);
	//calculate the coordinates of the rectangle
	int iRectX1 = giCol;
	int iRectY1 = giRow;
	int iRectX2 = giCol + giPartW;
	int iRectY2 = giRow + giPartH;
	glVertex2f(iRectX1, iRectY1);
	glVertex2f(iRectX2, iRectY1);
	glVertex2f(iRectX2, iRectY2);
	glVertex2f(iRectX1, iRectY2);
	glEnd();
}

	glLineWidth( 1.0f );
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	//render the current vote and the speed limit by text

	if (giSLCurFrm_GUI != -1)
	{
		//WriteTitleAndNumber(145, "Vote for: ", 10, giSLCurFrm);
		giLastSL = -1;
	}	
	if (giSLResult_GUI != -1)
	{
		//WriteTitleAndNumber(300, "New Speed Limit: ", 17, giSLResult);
		giLastSL = giSLResult_GUI;
	}
	glColor4f(1.0f, 0.0f, 0.0f, 1.0f);
	if (giLastSL != -1 && giSLCurFrm_GUI == -1)
		WriteTitleAndNumber(110, "Speed Limit: ", 18, giLastSL );

	glColor4f(1.0f, 1.0f, 1.0f, 1.0f);

	glutSwapBuffers();
    glutPostRedisplay();

}

////////////////////////////////////////////////////////////////////////////////
//! Keyboard events handler
////////////////////////////////////////////////////////////////////////////////
void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
    switch(key) {
    case(27) :
        exit(1);
        break;
	case 'p' :
		gbPause = !gbPause;
		break;
	case 'r' : //reverse: NOTE: recognition might not work properly, use it just for visualization purposes...
		gbPause = false;
		gbRewind = true;
		break;
	case 'f' : //fwd
		gbPause = false;
		gbRewind = false;
		break;
	case 's' : //slow - not slow
		gbSlow = !gbSlow;
		break;
	case 'c' : //clear the prints (big speed limit and the command prompt)
		giLastSL = -1;
		printf("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n");
		break;
	case 't': //do template-based recognition or don't (go faster)
		gbRecog = !gbRecog;
		break;
	case 'l':
		gbClahe = !gbClahe; //turn on/off CLAHE showing
		break;
	case 'b':
		gbShowBox = !gbShowBox;  
		break;
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Initialize GL
////////////////////////////////////////////////////////////////////////////////
int initGL(int argc, char **argv)
{
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(giShowScnW, giShowScnH);
    glutCreateWindow("Template-based Speed-Limit-Sign Recognition on an Embedded System using GPU Computing");
    // register callbacks
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);

	// initialize necessary OpenGL extensions
    glewInit();
    if (! glewIsSupported("GL_VERSION_2_0 ")) {
        fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
        fflush(stderr);
        return CUTFalse;
    }

    // default initialization
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glDisable(GL_DEPTH_TEST);

    // viewport
    glViewport(0, 0, giShowScnW, giShowScnH);

    // projection
	//sets up the OpenGL window so that (0,0) corresponds to the top left corner
	//(640,480) corresponds to the bottom right hand corner.
	glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
	glOrtho(0, giShowScnW, giShowScnH, 0, 0, 1);

    //checkCudaErrors();
	return CUTTrue;
}


int main(int argc, char** argv)
{

	///////////////////////////////////////////////////////////////
	/*
	cl_int status;

	if (!init()) {
		return -1;
	}

	// Set the kernel argument (argument 0)
	status = clSetKernelArg(kernel, 0, sizeof(cl_int), (void*)&thread_id_to_output);
	checkError(status, "Failed to set kernel arg 0");

	printf("\nKernel initialization is complete.\n");
	printf("Launching the kernel...\n\n");

	// Configure work set over which the kernel will execute
	size_t wgSize[3] = { work_group_size, 1, 1 };
	size_t gSize[3] = { work_group_size, 1, 1 };

	// Launch the kernel
	status = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, gSize, wgSize, 0, NULL, NULL);
	checkError(status, "Failed to launch kernel");

	// Wait for command queue to complete pending events
	status = clFinish(queue);
	checkError(status, "Failed to finish");

	printf("\nKernel execution is complete.\n");

	// Free the resources allocated
	AOCLcleanup();*/
	////////////////////////////////////////////////////////////////
	// First initialize OpenGL context, 
	// so we can properly set the GL for CUDA.
    // This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
    if (initGL(argc, argv) == 0 ) {
        return CUTFalse;
    }
	

	gfile_info[1] = 0; //used to indicate the frame number of first frame in the clip (in realis framenum start from nonzero, but in GUI I always start them from 0)
	gfVideoList = fopen(gcVideoListFile, "rt");
	gbOpenNewVideo = true;
	
	//gfVideo = fopen("cpuResults/scn.bin", "rb");
	//gfVideo = fopen(getFullPathOfFile_GUI("09-07-07_clips_0035.bin"), "rb");
	gacScn = (unsigned char *)malloc(giScnMemSzUChar);

	// create a texture
	glGenTextures(1, &texid);
	glBindTexture(GL_TEXTURE_2D, texid);

	// set basic parameters
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	atexit(cleanup);
	
	ssd_fft_gpu_init();

	// start rendering mainloop
    glutMainLoop();

	/*
	unsigned char* acScn = NULL;
	int iMaxPeakIndx, iPartW, iPartH;
    int file_info[3];
	file_info[0] = 0;
	file_info[1] = 0;
	file_info[2] = 0;
	unsigned long ulTimeStamp = 0;

	ssd_fft_gpu_init();
	for (int i = 0; i<10 ; i++)
		ssd_fft_gpu_findBestTpl(acScn, &iMaxPeakIndx, &iPartW, &iPartH, file_info, ulTimeStamp);
	ssd_fft_gpu_exit();

	int iRow, iCol;
	int iImgW = 640;
	//cuda is row major and zero-based
	iCol = iMaxPeakIndx%iImgW;
	//regular division of integer returns floor
	iRow = iMaxPeakIndx/iImgW; 
	printf("Max peak index row: %d, col: %d\n", iRow, iCol); 
	//printf("Press enter to exit\n"); 
	getchar(); 
	return 0; 
	*/
}

/////// AOCL HELPER FUNCTIONS ///////
/*
bool init() {
	cl_int status;

	if (!setCwdToExeDir()) {
		return false;
	}

	// Get the OpenCL platform.
	platform = findPlatform("Intel");
	if (platform == NULL) {
		printf("ERROR: Unable to find Intel FPGA OpenCL platform.\n");
		return false;
	}

	// User-visible output - Platform information
	{
		char char_buffer[STRING_BUFFER_LEN];
		printf("Querying platform for info:\n");
		printf("==========================\n");
		clGetPlatformInfo(platform, CL_PLATFORM_NAME, STRING_BUFFER_LEN, char_buffer, NULL);
		printf("%-40s = %s\n", "CL_PLATFORM_NAME", char_buffer);
		clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, STRING_BUFFER_LEN, char_buffer, NULL);
		printf("%-40s = %s\n", "CL_PLATFORM_VENDOR ", char_buffer);
		clGetPlatformInfo(platform, CL_PLATFORM_VERSION, STRING_BUFFER_LEN, char_buffer, NULL);
		printf("%-40s = %s\n\n", "CL_PLATFORM_VERSION ", char_buffer);
	}

	// Query the available OpenCL devices.
	scoped_array<cl_device_id> devices;
	cl_uint num_devices;

	devices.reset(getDevices(platform, CL_DEVICE_TYPE_ALL, &num_devices));

	// We'll just use the first device.
	device = devices[0];

	// Display some device information.
	display_device_info(device);

	// Create the context.
	context = clCreateContext(NULL, 1, &device, &oclContextCallback, NULL, &status);
	checkError(status, "Failed to create context");

	// Create the command queue.
	queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
	checkError(status, "Failed to create command queue");

	// Create the program.
	std::string binary_file = getBoardBinaryFile("hello_world", device);
	printf("Using AOCX: %s\n", binary_file.c_str());
	program = createProgramFromBinary(context, binary_file.c_str(), &device, 1);

	// Build the program that was just created.
	status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
	checkError(status, "Failed to build program");

	// Create the kernel - name passed in here must match kernel name in the
	// original CL file, that was compiled into an AOCX file using the AOC tool
	const char *kernel_name = "hello_world";  // Kernel name, as defined in the CL file
	kernel = clCreateKernel(program, kernel_name, &status);
	checkError(status, "Failed to create kernel");

	return true;
}

// Free the resources allocated during initialization
void AOCLcleanup() { //cleanup CL
	if (kernel) {
		clReleaseKernel(kernel);
	}
	if (program) {
		clReleaseProgram(program);
	}
	if (queue) {
		clReleaseCommandQueue(queue);
	}
	if (context) {
		clReleaseContext(context);
	}
}

// Helper functions to display parameters returned by OpenCL queries
static void device_info_ulong(cl_device_id device, cl_device_info param, const char* name) {
	cl_ulong a;
	clGetDeviceInfo(device, param, sizeof(cl_ulong), &a, NULL);
	printf("%-40s = %lu\n", name, a);
}
static void device_info_uint(cl_device_id device, cl_device_info param, const char* name) {
	cl_uint a;
	clGetDeviceInfo(device, param, sizeof(cl_uint), &a, NULL);
	printf("%-40s = %u\n", name, a);
}
static void device_info_bool(cl_device_id device, cl_device_info param, const char* name) {
	cl_bool a;
	clGetDeviceInfo(device, param, sizeof(cl_bool), &a, NULL);
	printf("%-40s = %s\n", name, (a ? "true" : "false"));
}
static void device_info_string(cl_device_id device, cl_device_info param, const char* name) {
	char a[STRING_BUFFER_LEN];
	clGetDeviceInfo(device, param, STRING_BUFFER_LEN, &a, NULL);
	printf("%-40s = %s\n", name, a);
}

// Query and display OpenCL information on device and runtime environment
static void display_device_info(cl_device_id device) {

	printf("Querying device for info:\n");
	printf("========================\n");
	device_info_string(device, CL_DEVICE_NAME, "CL_DEVICE_NAME");
	device_info_string(device, CL_DEVICE_VENDOR, "CL_DEVICE_VENDOR");
	device_info_uint(device, CL_DEVICE_VENDOR_ID, "CL_DEVICE_VENDOR_ID");
	device_info_string(device, CL_DEVICE_VERSION, "CL_DEVICE_VERSION");
	device_info_string(device, CL_DRIVER_VERSION, "CL_DRIVER_VERSION");
	device_info_uint(device, CL_DEVICE_ADDRESS_BITS, "CL_DEVICE_ADDRESS_BITS");
	device_info_bool(device, CL_DEVICE_AVAILABLE, "CL_DEVICE_AVAILABLE");
	device_info_bool(device, CL_DEVICE_ENDIAN_LITTLE, "CL_DEVICE_ENDIAN_LITTLE");
	device_info_ulong(device, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, "CL_DEVICE_GLOBAL_MEM_CACHE_SIZE");
	device_info_ulong(device, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, "CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE");
	device_info_ulong(device, CL_DEVICE_GLOBAL_MEM_SIZE, "CL_DEVICE_GLOBAL_MEM_SIZE");
	device_info_bool(device, CL_DEVICE_IMAGE_SUPPORT, "CL_DEVICE_IMAGE_SUPPORT");
	device_info_ulong(device, CL_DEVICE_LOCAL_MEM_SIZE, "CL_DEVICE_LOCAL_MEM_SIZE");
	device_info_ulong(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, "CL_DEVICE_MAX_CLOCK_FREQUENCY");
	device_info_ulong(device, CL_DEVICE_MAX_COMPUTE_UNITS, "CL_DEVICE_MAX_COMPUTE_UNITS");
	device_info_ulong(device, CL_DEVICE_MAX_CONSTANT_ARGS, "CL_DEVICE_MAX_CONSTANT_ARGS");
	device_info_ulong(device, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, "CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE");
	device_info_uint(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS");
	device_info_uint(device, CL_DEVICE_MEM_BASE_ADDR_ALIGN, "CL_DEVICE_MEM_BASE_ADDR_ALIGN");
	device_info_uint(device, CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE, "CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE");
	device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR");
	device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT");
	device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT");
	device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG");
	device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT");
	device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE");

	{
		cl_command_queue_properties ccp;
		clGetDeviceInfo(device, CL_DEVICE_QUEUE_PROPERTIES, sizeof(cl_command_queue_properties), &ccp, NULL);
		printf("%-40s = %s\n", "Command queue out of order? ", ((ccp & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) ? "true" : "false"));
		printf("%-40s = %s\n", "Command queue profiling enabled? ", ((ccp & CL_QUEUE_PROFILING_ENABLE) ? "true" : "false"));
	}
}*/
