#!/bin/bash

# 安装 eigen库
sudo apt update
sudo apt install libeigen3-dev

# 步骤1：进入orocos_kdl目录
cd /mnt/workspace/users/zhangtianle/others/orocos_kinematics_dynamics/orocos_kdl || exit  # 目录不存在则退出

# 步骤2：处理build目录（已存在则清空，不存在则创建）
if [ -d "build" ]; then
    rm -rf build/*  # 清空已有build目录（避免缓存）
else
    mkdir build     # 不存在则创建
fi
cd build || exit    # 进入build目录（失败则退出）

# 步骤3：获取Conda Python的头文件/库文件路径
PYTHON_INCLUDE=$(python -c "import sysconfig; print(sysconfig.get_path('include'))")
# PYTHON_LIB=$(python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
PYTHON_PREFIX=$(python -c "import sys; print(sys.prefix)")
PYTHON_LIB="$PYTHON_PREFIX/lib"


# 步骤4：验证库文件是否存在（不存在则报错退出）
if [ ! -f "${PYTHON_LIB}/libpython3.10.so" ]; then
    echo "错误：未找到libpython3.10.so，路径：${PYTHON_LIB}/libpython3.10.so"
    exit 1
else
    echo "✅ 验证通过：找到Python库文件 ${PYTHON_LIB}/libpython3.10.so"
fi

# 步骤5：执行CMake（关键：删除参数后多余空格，确保路径正确）
cmake \
-DCMAKE_INSTALL_PREFIX=/usr \
-DPYTHON_EXECUTABLE=${PYTHON_PREFIX}/bin/python \
-DPYTHON_INCLUDE_DIR=${PYTHON_INCLUDE} \
-DPYTHON_LIBRARY=${PYTHON_LIB}/libpython3.10.so \
-DBUILD_PYTHON=ON \
..  # 注意：..前无空格，且和上一行连在一起（CMake指定源码目录为上级）

# 步骤6：编译+安装（-j$(nproc)用所有CPU核心加速）
make -j$(nproc) || { echo "编译失败！"; exit 1; }
make install || { echo "安装失败！"; exit 1; }

# 步骤7：编译python_orocos_kdl（PyKDL绑定）
cd ../../python_orocos_kdl || exit  # 回到python_orocos_kdl目录
if [ -d "build" ]; then
    rm -rf build/*
else
    mkdir build
fi
cd build || exit

# 步骤8：CMake配置python_orocos_kdl
cmake \
-DCMAKE_INSTALL_PREFIX=/usr \
-DPYTHON_EXECUTABLE=${PYTHON_PREFIX}/bin/python \
-DPYTHON_INCLUDE_DIR=${PYTHON_INCLUDE} \
-DPYTHON_LIBRARY=${PYTHON_LIB}/libpython3.10.so \
..

# 步骤9：编译+安装Python绑定
make -j$(nproc) || { echo "Python绑定编译失败！"; exit 1; }
make install || { echo "Python绑定安装失败！"; exit 1; }

# 步骤10：链接PyKDL到Conda环境（关键：让Python能找到）
PYKDL_SO=$(find /usr/lib -name "PyKDL*.so" | grep python3 | head -1)
if [ -z "${PYKDL_SO}" ]; then
    echo "错误：未找到编译后的PyKDL.so文件！"
    exit 1
fi
ln -sf ${PYKDL_SO} ${PYTHON_PREFIX}/lib/python3.10/site-packages/
echo "✅ PyKDL安装完成！已链接到Conda环境：${PYKDL_SO} -> ${PYTHON_PREFIX}/lib/python3.10/site-packages/"

# 验证安装
${PYTHON_PREFIX}/bin/python -c "import PyKDL; print('✅ PyKDL导入成功！版本：', PyKDL.__version__)"