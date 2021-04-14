add_subdirectory(keras)

if(PML_OPENVINO_BRIDGE)
  add_subdirectory(openvino)
endif()
