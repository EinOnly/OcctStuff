# OcctStuff
Build models with occt toolkit
g-z7VyveQ9teDHFJGdTFo5oFSntRvyJV0pyJHdjeCmi41Tov6yVvzJAjo7u2AoHFw1
fastgpt-cHBzk41DsBsIaYZ0PfV1xQQDfwobdBW635KWwXwfr7sUEh2YwwcQxhD

build occt_7.8.1:

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/Users/ein/DevTools/occt_7_8_1 \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DBUILD_RELEASE_DISABLE_EXCEPTIONS=OFF \
  -DBUILD_MODULE_Draw=OFF \
  -DBUILD_MODULE_DETools=OFF \
  -DBUILD_MODULE_FoundationClasses=ON \
  -DBUILD_MODULE_ModelingData=ON \
  -DBUILD_MODULE_ModelingAlgorithms=ON \
  -DBUILD_MODULE_DataExchange=ON \
  -DBUILD_MODULE_Visualization=ON \
  -DBUILD_MODULE_ApplicationFramework=ON

make -j4

build pythonocc-core:
cmake .. \
-DCMAKE_BUILD_TYPE=Release \
-DCMAKE_INSTALL_PREFIX=/Users/ein/EinDev/OcctStuff/.venv/lib/python3.10/site-packages/OCC \
-DOCCT_INCLUDE_DIR=/Users/ein/DevTools/occt_7_8_1/include/opencascade \
-DOCCT_LIBRARY_DIR=/Users/ein/DevTools/occt_7_8_1/lib \
-DCMAKE_PREFIX_PATH=/Users/ein/DevTools/occt_7_8_1 \
-DPYTHONOCC_INSTALL_DIRECTORY=/Users/ein/EinDev/OcctStuff/.venv/lib/python3.10/site-packages

export CPLUS_INCLUDE_PATH=/opt/homebrew/include:$CPLUS_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=/Users/ein/EinDev/OcctStuff/.venv/lib/python3.10/site-packages/numpy/_core/include:$CPLUS_INCLUDE_PATH