cmake_minimum_required(VERSION 3.21)
project(volsmoke LANGUAGES CXX)

# MitiruEngine のヘッダ (Dx12ComputeContext) と vendored 依存を参照する。
# 別の場所に置いた場合は -DMITIRU_ROOT=... で上書き。
set(MITIRU_ROOT "E:/user/MitiruEngine" CACHE PATH "MitiruEngine repo root")

set(IMGUI_DIR "${MITIRU_ROOT}/external/vendored/imgui")

add_executable(volsmoke
    src/main.cpp
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/imgui_impl_win32.cpp
    ${IMGUI_DIR}/imgui_impl_dx12.cpp)

target_compile_features(volsmoke PRIVATE cxx_std_20)

target_include_directories(volsmoke PRIVATE
    "${MITIRU_ROOT}/include"
    "${MITIRU_ROOT}/external/stb"
    "${IMGUI_DIR}")

if(MSVC)
    target_compile_options(volsmoke PRIVATE /utf-8 /EHsc)
endif()

target_link_libraries(volsmoke PRIVATE d3d12 dxgi d3dcompiler dwmapi)
