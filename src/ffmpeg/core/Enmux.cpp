#include "Enmuxer.h"
#include "utils/File.h"
namespace FFmpeg {
Enmuxer::Enmuxer(std::shared_ptr<Encoder> encoder, std::string out_url) {
    m_encoder = encoder;
    m_out_url = out_url;
    if (out_url.find("mp4") != std::string::npos) {
        m_format_name = "mp4";
    } else if (out_url.find("flv") != std::string::npos) {
        m_format_name = "flv";
    } else if (out_url.find("rtmp") != std::string::npos) {
        m_format_name = "flv";
    } else if (out_url.find("rtsp") != std::string::npos) {
        m_format_name = "rtsp";
    } else if (out_url.find("jpg") != std::string::npos) {
        m_format_name = "image2";
    } else {
        m_format_name = "mp4";
    }
}

Enmuxer::~Enmuxer() {
    close();
}

bool Enmuxer::open() {
    ASSERT_FFMPEG(avformat_alloc_output_context2(&m_format_ctx, nullptr, m_format_name.c_str(),
                                                 m_out_url.c_str()));
    avformat_new_stream(m_format_ctx, m_encoder->get_codec_ctx()->codec);
    ASSERT_FFMPEG(avcodec_parameters_from_context(get_video_stream()->codecpar,
                                                  m_encoder->get_codec_ctx().get()));
    if (m_format_name == "rtsp") {
        // 为RTSP设置一些特定的选项
        AVDictionary *options = nullptr;
        av_dict_set(&options, "rtsp_transport", "tcp", 0);  // 使用TCP传输方式
        // 使用avformat_write_header来打开RTSP连接
        ASSERT_FFMPEG(avformat_write_header(m_format_ctx, &options));
        av_dict_free(&options);  // 释放选项字典
    } else {
        // 非RTSP流的处理，依然使用avio_open
        ASSERT_FFMPEG(avio_open(&m_format_ctx->pb, m_out_url.c_str(), AVIO_FLAG_WRITE));
        ASSERT_FFMPEG(avformat_write_header(m_format_ctx, nullptr));
    }
    return true;
}

bool Enmuxer::write_packet(av_packet &packet) {
    std::unique_lock<std::mutex> lock(m_mutex);
    packet->dts      = packet->pts;
    packet->duration = 1;

    packet->pts = av_rescale_q(packet->pts, m_encoder->get_time_base(), get_time_base());
    packet->dts = av_rescale_q(packet->dts, m_encoder->get_time_base(), get_time_base());
    packet->duration =
        av_rescale_q(packet->duration, m_encoder->get_time_base(), get_time_base()) / 3;
    packet->pos = -1;

    ASSERT_FFMPEG(av_interleaved_write_frame(m_format_ctx, packet.get()));
    return true;
}

void Enmuxer::close() {
    std::unique_lock<std::mutex> lock(m_mutex);
    if (m_format_ctx) {
        write_trailer();
        avio_close(m_format_ctx->pb);
        avformat_free_context(m_format_ctx);
        m_format_ctx = nullptr;
    }
}

void Enmuxer::write_trailer() {
    static bool is_write_trailer = false;
    if (!is_write_trailer && (m_format_name == "mp4" || m_format_name == "jpg")){
        av_write_trailer(m_format_ctx);
        is_write_trailer = true;
    }

}

}  // namespace FFmpeg