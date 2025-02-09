//
// Created by lijin on 2023/12/20.
//

#include "FFmpegReadNode.h"
#include "graph/core/common/StatusCode.h"
#include <thread>
#include <utility>

namespace Node {
FFmpegReadNode::FFmpegReadNode(const std::string &name,
                               std::string        open_source,
                               bool               use_hw,
                               bool               cycle)
    : GraphCore::Node(name, GraphCore::NODE_TYPE::SRC_NODE),
      m_open_source(std::move(open_source)),
      m_use_hw(use_hw),
      m_cycle(cycle) {}

void FFmpegReadNode::worker() {
    int frame_index = 0;
    while (m_run) {
        auto pkt = alloc_av_packet();
        int  re  = m_demux->read_packet(pkt);
        if (re == EXIT_SUCCESS) {
            if (pkt->stream_index != m_demux->get_video_stream_index()) {
                continue;  // 忽略非视频帧
            }
            m_decoder->send(pkt);
            auto frame = alloc_av_frame();
            if (!m_decoder->receive(frame)) {
                continue;  // 编码器前几帧的缓存可能接收不到
            }
            cv::Mat image(frame->height, frame->width, CV_8UC3);
            if (!m_scaler->scale<av_frame, cv::Mat>(frame, image)) {
                ErrorL << "scale failed";
                continue;
            }
            auto data          = std::make_shared<Data::BaseData>(Data::DataType::FRAME);
            data->data_name    = getName();
            data->MAT_IMAGE    = image;
            data->FRAME_INDEX  = frame_index++;
            data->FRAME_WIDTH  = frame->width;
            data->FRAME_HEIGHT = frame->height;
//            ErrorL << "输出生成：" << data->FRAME_INDEX;
            send_output_data(data);
        } else if (re == AVERROR_EOF) {
            InfoL << "read end of file";
            if (m_cycle) {
                m_demux->seek(0);
                std::this_thread::sleep_for(std::chrono::milliseconds(1000));
                continue;
            }
            break;
        } else {
            ErrorL << "read error";
            if (re == -5 || re == -1) {
                ErrorL << "读取节点错误，重连中。。。";
                for (int i = 0; i < m_max_reconnect; i++) {
                    if (Reconnect()) {
                        ErrorL << "重连成功";
                        timeout_cb(getName(), GraphCore::StatusCode::FFMpegReadError,
                                   "节点重连成功");
                        break;
                    } else {
                        ErrorL << "读取节点重连中...[第" << i << "次]";
                        std::this_thread::sleep_for(std::chrono::milliseconds(30));
                    }
                }
                // break;
            }
            continue;
            // break;
        }
    }
}

FFmpegReadNode::ptr FFmpegReadNode::CreateShared(const std::string &name,
                                                 std::string        open_source,
                                                 bool               use_hw,
                                                 bool               cycle) {
    return std::make_shared<FFmpegReadNode>(name, std::move(open_source), use_hw, cycle);
}

std::tuple<int, int, int, int64_t> FFmpegReadNode::get_video_info() const {
    return std::make_tuple(m_width, m_height, m_fps, m_bitrate);
}

FFmpegReadNode::~FFmpegReadNode() {
    Stop();
}
bool FFmpegReadNode::Init() {
    if (!m_demux) {
        m_demux = FFmpeg::Demuxer::createShare();
    }
    if (!(m_demux->open(m_open_source))) {
        ErrorL << "open url " << m_open_source << "failed";
        return false;
    }
    if (!m_scaler) {
        if (m_use_hw) {
            m_scaler = FFmpeg::Scaler::createShare(m_demux->get_video_codec_parameters()->width,
                                                   m_demux->get_video_codec_parameters()->height,
                                                   AV_PIX_FMT_NV12,
                                                   m_demux->get_video_codec_parameters()->width,
                                                   m_demux->get_video_codec_parameters()->height,
                                                   AV_PIX_FMT_BGR24);
        } else {
            m_scaler = FFmpeg::Scaler::createShare(
                m_demux->get_video_codec_parameters()->width,
                m_demux->get_video_codec_parameters()->height,
                (AVPixelFormat)m_demux->get_video_codec_parameters()->format,
                m_demux->get_video_codec_parameters()->width,
                m_demux->get_video_codec_parameters()->height, AV_PIX_FMT_BGR24);
        }
    }
    if (!m_decoder) {
        m_decoder = FFmpeg::Decoder::createShare(m_demux);
    }
    if (!(m_decoder->open(m_use_hw))) {
        return false;
    }
    m_width  = m_demux->get_video_codec_parameters()->width;
    m_height = m_demux->get_video_codec_parameters()->height;
    m_fps    = m_demux->get_video_stream()->avg_frame_rate.num /
            m_demux->get_video_stream()->avg_frame_rate.den;
    m_bitrate = m_demux->get_video_codec_parameters()->bit_rate;
    return true;
}

// read error add reconnect
bool FFmpegReadNode::Reconnect() {
    m_demux.reset();
    m_decoder.reset();
    m_scaler.reset();
    return Init();
}

}  // namespace Node