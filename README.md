# TRT-VideoPipeline

## 多路视频分析处理案例

功能：

- 完整的视频拉流解码、trt推理、编码推理处理pipeline
- 支持yolo系列模型的tensorrt推理
- 支持单模型多显卡多实例负载调度，数据前后处理均为GPU操作
- 支持nvidia硬件编解码，多路并发节约cpu资源
- 支持多种视频格式输入（RTSP、RTMP、MP4），多种格式推理输出（RTSP、RTMP），本地MP4录制

## 1. 环境配置

项目开发在linux上，依赖的第三方环境和版本

| 依赖       | 版本    |
|----------|-------|
| ffmpeg   | 5.2   |
| opencv   | 4.8.0 |
| tensorrt | 8.6   |
| c++      | 17    |
| gcc      | 7.5以上 |

如果在Linux开发，ffmpeg和opencv环境的搭建请参考addons/BuildBaseEnv.sh脚本，帮你一键构建ffmpeg和opencv环境

ffmpeg的硬件编解码环境需要额外重新编译开启支持，使用默认脚本是不支持nvidia硬件编解码的，请参照脚本中的注释部分进行修改

nvidia显卡硬件编解码能力参[考表](https://developer.nvidia.com/video-encode-decode-gpu-support-matrix)

tensorrt和cuda环境如果不在默认环境上，请到cmake/cuda.cmake和cmake/tensorrt.cmake文件下修改自己环境下的CUDA_TOOLKIT_ROOT_DIR和TENSORRT_ROOT_DIR路径配置

如果是在wendous下开发的，请自行配置第三方环境，依赖的三方库均在win下有实现，估计win下也是可以正常运行的，不过本人实际没有在win下测试过，因此不保证能否正常运行。

## 2、tensorrt模型转换

建议使用onnx模型格式和trtexe进行trt-engine模型转换

在tools/onnx2trtengine.sh下提供了模型转换脚本，请参考根据自身需求修改动态shape、量化精度、最大batch数等参数

# 3、使用demo

~~~cpp
#include "infer/MultipleInferenceInstances.h"
#include "trt/yolo/YoloDetectPipeline.h"
#include "trt/yolo/YoloDetectionInfer.h"

int main() {
    std::string input_stream_url  = "输入流路径";
    std::string output_stream_url = "输出流路径";
    std::string model_path        = "TRTengine模型文件路径";
    std::string label_path        = "检测分类类别文件路径";
    int         max_batch_size    = 16;    // 最大batch数
    float       config_threshold  = 0.25;  // 检测阈值
    float       nms_threshold     = 0.5;   // nms阈值

    // 模型实例数量列表，列表为模型实例数，每个元素代表该模型实例在哪张显卡上的下标
    std::vector<int> device_list{0, 0, 1, 1};
    auto             type = infer::YoloType::V8;  // 模型类型

    // 创建多卡多实例推理对象
    auto trt_instance =
        std::make_shared<infer::MultipleInferenceInstances<infer::YoloDetectionInfer>>(
            "trt_instance", device_list, model_path, label_path, type, config_threshold,
            nms_threshold, max_batch_size);
    if (!trt_instance->init()) {
        std::cout << "init failed" << std::endl;
        return -1;
    }

    // 创建处理pipeline
    auto pipeline = std::make_shared<pipeline::YoloDetectPipeline>(
        "test_pipeline", input_stream_url, output_stream_url, trt_instance);

    // 启动流水线
    pipeline->Start();

    getchar();
}
~~~

不到30行代码即可实现一个yolo目标检测分析demo，拉流-》模型推理-》渲染-》编码-》推流的完整工作流。且模型前处理、推流、后处理均在gpu上，简单且高效。

## 功能节点复用

项目中的功能节点在整个大部分为独立的模块，可以根据自己的需求进行组合，实现自己的功能pipeline。

也非常欢迎各位大佬贡献自己的功能节点，共同完善项目。


| 序号 | Node                     | 说明                                                                    | 开发状态 |
| ---- |--------------------------|-----------------------------------------------------------------------|------|
| 1    | FFmpegReadNode           | 基于ffmpeg的数据输入节点，支持本地视频，RTSP、RTMP、FLV等协议；<br/>直接n卡硬解；<br/>增加了网络波动的容错机制 | 完成   |
| 2    | ImageDrawNode            | 图像画框、文本渲染节点，需要输入目标框数据符合要检测算法封装格式                                      | 完成   |
| 3    | FFmpegRecordNode         | 图片截图、视频录制节点                                                           | 完成   |
| 4    | FFmpegPusherNode         | 推流节点，支持h264/h265编码，硬件编码，rtmp推流                                        | 完成   |
| 5    | FFmpegPusherRecorderNode | 推流和视频录制封装在一起，录制mp4的数据复用推流的数据，节约性能；<br/>缺点是录制视频的尺寸和推流的尺寸必须一致           | 完成   |
| 8    | InferNode                | 对trt infer的封装接收节点，使得多个流水线共用同一个推理实例                                    | 完成   |
| 9    | HttpImageInputNode       | 接收来自http协议的请求图像数据                                                     | 开发中  |
| 10   | KeyframePostNode       | 用于将检测帧目标数据结果通过http协议上传到目标服务器                                          | 完成   |


## 参考

作为cv相关（ctrl+c/ctrl+v）的开发人员，同样在开发过程中参考了很多优秀项目，特别是以下两个项目：

该项目是基于原来对于手写AI的[infer](https://github.com/shouxieai/infer)
项目的简单修改而来，核心实现思路可以查看Old_README.md文档，新版本对其进行了更完善的修改（抄）和封装。

同时项目中的有向无环图的流水线处理结构参考了[video_pipe_c](https://github.com/sherlockchou86/video_pipe_c)
项目的设计思路，自己在开发过程中进行了调整。

## 存在的问题

测试是yolov8-seg的分割后处理还存在问题，没跑通

## 最后

如果对你有帮助，欢迎star，如果有问题，欢迎提issue，pr。
项目中有很多不足的地方，欢迎大家指正，共同进步。
