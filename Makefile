

#INC = -I./opencv/modules/highgui/include -I./opencv/modules/core/include -I./opencv/modules/imgcodecs/include -I./opencv/modules/videoio/include -I./opencv/modules/imgproc/include
#INC = -I./opencv/modules/ -I./opencv/modules/highgui/include

GPU:
	 nvcc Application.cu -o GPU `pkg-config opencv --cflags --libs`

CPU:
	 g++ Application.cpp -o CPU `pkg-config opencv --cflags --libs`

clean:
	rm -rf GPU CPU
