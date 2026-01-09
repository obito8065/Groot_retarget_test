cd /mnt/workspace/users/lijiayi/GR00T_QwenVLA



source /mnt/workspace/envs/conda3/bin/activate robocasa

python3 << 'EOF'
import os
import sys

print("=" * 60)
print("测试 EGL 设备可用性")
print("=" * 60)

available_devices = []

for device_id in range(8):
    print(f"\n测试 EGL 设备 {device_id}...")
    os.environ['MUJOCO_EGL_DEVICE_ID'] = str(device_id)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)
    
    try:
        import mujoco
        
        # 创建一个简单的MuJoCo模型来测试EGL渲染
        xml_string = '''
        <mujoco>
            <worldbody>
                <light pos="0 0 3"/>
                <geom name="floor" type="plane" size="1 1 0.1" rgba="0.8 0.8 0.8 1"/>
                <geom name="sphere" type="sphere" size="0.1" pos="0 0 0.5" rgba="1 0 0 1"/>
            </worldbody>
        </mujoco>
        '''
        
        model = mujoco.MjModel.from_xml_string(xml_string)
        renderer = mujoco.Renderer(model, height=240, width=320)
        
        # 尝试渲染一帧
        mujoco.mj_forward(model, mujoco.MjData(model))
        renderer.update_scene(mujoco.MjData(model))
        pixels = renderer.render()
        
        if pixels is not None and pixels.size > 0:
            print(f"  ✓ EGL设备 {device_id} 可用 - 成功渲染 {pixels.shape}")
            available_devices.append(device_id)
        else:
            print(f"  ✗ EGL设备 {device_id} 渲染失败 - 返回空图像")
        
        del renderer
        del model
        
    except RuntimeError as e:
        error_msg = str(e)
        if "MUJOCO_EGL_DEVICE_ID" in error_msg and "between" in error_msg:
            print(f"  ✗ EGL设备 {device_id} 不可用 - {error_msg}")
        else:
            print(f"  ✗ EGL设备 {device_id} 错误 - {error_msg}")
    except Exception as e:
        print(f"  ✗ EGL设备 {device_id} 异常 - {type(e).__name__}: {e}")

print("\n" + "=" * 60)
print("测试结果汇总")
print("=" * 60)
if available_devices:
    print(f"可用的EGL设备: {available_devices}")
    print(f"可用设备数量: {len(available_devices)}")
    if len(available_devices) == 1:
        print("\n⚠️  警告: 只有1个EGL设备可用，所有进程将共享该设备")
        print("   建议: 在shell脚本中设置 MUJOCO_EGL_DEVICE_ID_OVERRIDE=0")
    else:
        print(f"\n✓ 有 {len(available_devices)} 个EGL设备可用，可以分散负载")
        print("   建议: 不设置 MUJOCO_EGL_DEVICE_ID_OVERRIDE，让代码自动分配")
else:
    print("❌ 没有可用的EGL设备！")
    print("   请检查:")
    print("   1. NVIDIA驱动是否正确安装")
    print("   2. EGL库是否正确配置")
    print("   3. 环境变量 PYOPENGL_PLATFORM=egl 和 MUJOCO_GL=egl 是否设置")
EOF

deactivate