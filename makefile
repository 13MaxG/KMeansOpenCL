INCLUDE_DIR="D:\\dev\\default\\deps\\include\\"
LIB_DIR="D:\\dev\\default\\deps\\lib\\"
LIBS=-lOpenCL -llibpng
CFLAGS=--std=c++1z

all: main

main: main.cpp
	g++ main.cpp $(LIBS) -I$(INCLUDE_DIR) -L$(LIB_DIR)  $(CFLAGS) 