# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.28

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/aminelmk/Bureau/HTML/cpp_code

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/aminelmk/Bureau/HTML/cpp_code/build

# Include any dependencies generated for this target.
include CMakeFiles/process_data.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/process_data.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/process_data.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/process_data.dir/flags.make

CMakeFiles/process_data.dir/process_data.cpp.o: CMakeFiles/process_data.dir/flags.make
CMakeFiles/process_data.dir/process_data.cpp.o: /home/aminelmk/Bureau/HTML/cpp_code/process_data.cpp
CMakeFiles/process_data.dir/process_data.cpp.o: CMakeFiles/process_data.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/aminelmk/Bureau/HTML/cpp_code/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/process_data.dir/process_data.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/process_data.dir/process_data.cpp.o -MF CMakeFiles/process_data.dir/process_data.cpp.o.d -o CMakeFiles/process_data.dir/process_data.cpp.o -c /home/aminelmk/Bureau/HTML/cpp_code/process_data.cpp

CMakeFiles/process_data.dir/process_data.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/process_data.dir/process_data.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/aminelmk/Bureau/HTML/cpp_code/process_data.cpp > CMakeFiles/process_data.dir/process_data.cpp.i

CMakeFiles/process_data.dir/process_data.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/process_data.dir/process_data.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/aminelmk/Bureau/HTML/cpp_code/process_data.cpp -o CMakeFiles/process_data.dir/process_data.cpp.s

# Object files for target process_data
process_data_OBJECTS = \
"CMakeFiles/process_data.dir/process_data.cpp.o"

# External object files for target process_data
process_data_EXTERNAL_OBJECTS =

libprocess_data.so: CMakeFiles/process_data.dir/process_data.cpp.o
libprocess_data.so: CMakeFiles/process_data.dir/build.make
libprocess_data.so: CMakeFiles/process_data.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/aminelmk/Bureau/HTML/cpp_code/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared module libprocess_data.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/process_data.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/process_data.dir/build: libprocess_data.so
.PHONY : CMakeFiles/process_data.dir/build

CMakeFiles/process_data.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/process_data.dir/cmake_clean.cmake
.PHONY : CMakeFiles/process_data.dir/clean

CMakeFiles/process_data.dir/depend:
	cd /home/aminelmk/Bureau/HTML/cpp_code/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/aminelmk/Bureau/HTML/cpp_code /home/aminelmk/Bureau/HTML/cpp_code /home/aminelmk/Bureau/HTML/cpp_code/build /home/aminelmk/Bureau/HTML/cpp_code/build /home/aminelmk/Bureau/HTML/cpp_code/build/CMakeFiles/process_data.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/process_data.dir/depend

