//
// Created by lijin on 2023/12/20.
//

#include "FFmpegReadNode.h"
#include "graph/core/common/StatusCode.h"
#include <utility>
#include <thread>


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
    auto logger = GetLogger();
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
                //std::cout << "scale failed" << std::endl;
                logger.error("[{0}:{1}] Scale failed", __FILENAME__, __LINE__);
                continue;
            }
            auto data = std::make_shared<Data::BaseData>(Data::DataType::FRAME);
            data->Insert<MAT_IMAGE_TYPE>(MAT_IMAGE, image);
            data->Insert<FRAME_INDEX_TYPE>(FRAME_INDEX, frame_index++);
            data->Insert<FRAME_WIDTH_TYPE>(FRAME_WIDTH, frame->width);
            data->Insert<FRAME_HEIGHT_TYPE>(FRAME_HEIGHT, frame->height);
            send_output_data(data);
        } else if (re == AVERROR_EOF) {
            //std::cout << "read eof" << std::endl;
            logger.info("[{0}:{1}] Read eof", __FILENAME__, __LINE__);
            if (m_cycle) {
                m_demux->seek(0);
                std::this_thread::sleep_for(std::chrono::milliseconds(1000));
                continue;
            }
            break;
        } else {
            //std::cout << "read error" << std::endl;
            logger.error("[{0}:{1}] Read error", __FILENAME__, __LINE__);

            if (re == -5 || re == -1){
                //error_cb(getName(), GraphCore::StatusCode::NodeError, "读取节点错误，重连中。。。");
                logger.error("[{0}:{1}] 读取节点错误，重连中", __FILENAME__, __LINE__);
                for(int i=0; i<m_max_reconnect; i++)
                {
                    if(!Reconnect())
                    {
                        //printf("重连成功");
                        logger.info("[{0}:{1}] 重连成功", __FILENAME__, __LINE__);
                        break;
                    }
                    else
                    {
                        logger.info("[{0}:{1}], 读取节点重连中...[第{2}次]", __FILENAME__, __LINE__, i);
                        //printf("读取节点重连中。。。，第%d次", i);
                        std::this_thread::sleep_for(std::chrono::milliseconds(30));
                    }
                }
                //break;
                
            }
            continue;
            //break;
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
    m_demux.reset();
    m_decoder.reset();
    m_scaler.reset();
}
bool FFmpegReadNode::Init() {
    if (!m_demux) {
        m_demux = FFmpeg::Demuxer::createShare();
    }
    if (!(m_demux->open(m_open_source))) {
        std::cout << "open url " << m_open_source << "failed" << std::endl;
        return false;
    }
    if (!m_scaler) {
        m_scaler = FFmpeg::Scaler::createShare(
            m_demux->get_video_codec_parameters()->width,
            m_demux->get_video_codec_parameters()->height,
            (AVPixelFormat)m_demux->get_video_codec_parameters()->format,
            m_demux->get_video_codec_parameters()->width,
            m_demux->get_video_codec_parameters()->height, AV_PIX_FMT_BGR24);
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
bool FFmpegReadNode::Reconnect(){
    m_demux.reset();
    m_decoder.reset();
    m_scaler.reset();
    Init();
    return true;
}

}  // namespace Node