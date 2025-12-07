CMAKE_MINIMUM_REQUIRED(VERSION 2.4.8)
SET(CMAKE_COLOR_MAKEFILE ON)
SET(CMAKE_VERBOSE_MAKEFILE ON)
SET(CMAKE_INCLUDE_CURRENT_DIR TRUE)

IF(UNIX AND NOT LINUX_SET)
 ADD_DEFINITIONS(-D LINUX)
ENDIF(UNIX AND NOT LINUX_SET)

INCLUDE_DIRECTORIES(
../tmolib/
.
)

LINK_LIBRARIES (
tmo
)

SET(TMO_SOURCES
 TMOKhudair23.cpp
 TMOPlugin.cpp
)


SET(TMOWARD_HEADERS
 TMOKhudair23.h
 TMOPlugin.h
)


add_library( khudair23 SHARED ${TMO_SOURCES} )
add_custom_command( TARGET khudair23 POST_BUILD 
	COMMAND cp -f libkhudair23.so ../TMOCmd/khudair23.tml
	COMMAND cp -f libkhudair23.so ../TMOgui/khudair23.tml 
	COMMAND cp -f libkhudair23.so ../khudair23.tml 
#	COMMAND rm -f libkhudair23.so 
	)


