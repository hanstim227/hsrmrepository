# ============================================================
# 传统视觉装甲板识别（静态图片）—— CMake 配置文件
#
#   配置（OpenCV_DIR 指向本机 OpenCV 解压目录下的 build 文件夹）：
#     cmake -S . -B build -G "Visual Studio 18 2026" -A x64 -DOpenCV_DIR="E:/新建文件夹/opencv/build"
#   编译：
#     cmake --build build --config Release
#   之后用 Visual Studio 打开 build/ArmorDetect.slnx，按 F5 运行
# ============================================================
cmake_minimum_required(VERSION 3.16)

project(ArmorDetect LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED)

# main.cpp    命令行入口、读图、保存结果、终端输出
# detect.cpp  颜色分割、灯条候选、灯条配对、四角点
add_executable(ArmorDetect
    main.cpp
    detect.cpp
)

target_include_directories(ArmorDetect PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(ArmorDetect PRIVATE ${OpenCV_LIBS})

# 源文件是 UTF-8，控制台中文要靠 /utf-8 + SetConsoleOutputCP(CP_UTF8)
if (MSVC)
    target_compile_options(ArmorDetect PRIVATE /utf-8)
endif()

# F5 运行时的工作目录 = 项目根目录，images/ output/ 用相对路径即可
set_target_properties(ArmorDetect PROPERTIES
    VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

# 编译后把 OpenCV 运行库 DLL 复制到可执行文件旁边，双击 exe 也不会缺 DLL
file(GLOB OPENCV_RUNTIME_DLLS "${OpenCV_DIR}/x64/vc16/bin/opencv_world*.dll")
if (OPENCV_RUNTIME_DLLS)
    add_custom_command(TARGET ArmorDetect POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${OPENCV_RUNTIME_DLLS} $<TARGET_FILE_DIR:ArmorDetect>
        COMMENT "复制 OpenCV 运行库 DLL 到可执行文件目录"
    )
endif()
