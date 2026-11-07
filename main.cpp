// ============================================================
// main.cpp —— 装甲板静态图片检测（命令行入口）
//
// 用法：
//   ArmorDetect [图片路径] [--out 输出目录] [--min-area 最小面积] [--show]
// 例：
//   ArmorDetect images/outpost_frame.png
//   ArmorDetect images/outpost_frame.png --min-area 100000      (异常测试：0 个候选)
//   ArmorDetect no_such_file.png                                (异常测试：读取失败)
// ============================================================
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN

#include "detect.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <windows.h>

namespace fs = std::filesystem;

namespace {

struct Options {
    std::string imagePath = "images/outpost_frame.png";
    std::string outDir    = "output";
    DetectParams params;
    bool show = false;
};

void printUsage() {
    std::cout <<
        "用法: ArmorDetect [图片路径] [选项]\n"
        "  图片路径            默认 images/outpost_frame.png（相对项目根目录）\n"
        "  --out <目录>        结果输出目录，默认 output\n"
        "  --min-area <像素>   灯条候选最小轮廓面积，默认 25\n"
        "  --show              弹出结果窗口（按任意键关闭）\n"
        "  --help              显示本说明\n";
}

bool parseArgs(int argc, char** argv, Options& opt) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") { printUsage(); return false; }
        else if (arg == "--show") { opt.show = true; }
        else if (arg == "--out" && i + 1 < argc) { opt.outDir = argv[++i]; }
        else if (arg == "--min-area" && i + 1 < argc) {
            opt.params.minArea = std::atof(argv[++i]);
            if (opt.params.minArea < 0) opt.params.minArea = 0;
        }
        else if (!arg.empty() && arg[0] == '-') {
            std::cout << "未知选项: " << arg << "\n";
            printUsage();
            return false;
        }
        else { opt.imagePath = arg; }
    }
    return true;
}

bool saveImage(const std::string& path, const cv::Mat& image) {
    if (image.empty()) return false;
    return cv::imwrite(path, image);
}

}  // namespace

int main(int argc, char** argv) {
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);

    Options opt;
    if (!parseArgs(argc, argv, opt)) return 0;

    // ---------- 1. 读图 ----------
    cv::Mat image = cv::imread(opt.imagePath, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cout << "[错误] 图片读取失败: " << opt.imagePath << "\n"
                  << "       请检查路径是否正确、文件是否存在，程序结束。\n";
        return 2;
    }
    std::cout << "[图片] " << opt.imagePath
              << "   宽 " << image.cols << " x 高 " << image.rows
              << "   通道 " << image.channels() << "\n";

    // ---------- 2. 单帧检测 ----------
    const FrameResult result = detectFrame(image, opt.params);

    // ---------- 3. 准备输出目录 ----------
    std::error_code ec;
    fs::create_directories(opt.outDir, ec);
    if (ec) {
        std::cout << "[错误] 无法创建输出目录 " << opt.outDir
                  << " : " << ec.message() << "，程序结束。\n";
        return 3;
    }

    std::ofstream logFile(fs::path(opt.outDir) / "result.txt", std::ios::binary);
    auto say = [&](const std::string& line) {
        std::cout << line << "\n";
        if (logFile) logFile << line << "\n";
    };

    const double totalPixels = static_cast<double>(image.total());
    {
        std::ostringstream os;
        os << std::fixed << std::setprecision(2);
        os << "[分割] 红色 mask 保留像素 " << result.redMaskPixels
           << " (" << 100.0 * result.redMaskPixels / totalPixels << "%)"
           << "   蓝色 mask 保留像素 " << result.blueMaskPixels
           << " (" << 100.0 * result.blueMaskPixels / totalPixels << "%)";
        say(os.str());
    }

    int redBars = 0, blueBars = 0;
    for (const auto& b : result.bars) (b.color == ArmorColor::RED ? redBars : blueBars)++;
    {
        std::ostringstream os;
        os << "[灯条] 红色候选 " << redBars << " 根，蓝色候选 " << blueBars
           << " 根（面积/长宽比/尺寸/倾角筛选后）";
        say(os.str());
    }
    {
        std::ostringstream os;
        os << "[配对] 装甲板候选 " << result.armors.size() << " 个";
        say(os.str());
    }

    // ---------- 4. 保存中间结果 ----------
    const fs::path out(opt.outDir);
    struct Item { std::string file; cv::Mat image; };
    const std::vector<Item> items = {
        { "mask_red.png",         result.maskRed },
        { "mask_blue.png",        result.maskBlue },
        { "bars_candidates.png",  drawBarCandidates(image, result.bars) },
        { "armor_result.png",     drawArmorResult(image, result) },
    };
    for (const auto& item : items) {
        const fs::path p = out / item.file;
        if (!saveImage(p.string(), item.image)) {
            std::cout << "[错误] 保存失败: " << p.string() << "\n";
            return 3;
        }
        say("[保存] " + p.string());
    }

    // 放大图：方便在报告里看清灯条与四角点
    for (size_t i = 0; i < result.armors.size(); ++i) {
        const ArmorCandidate& a = result.armors[i];
        const int pad = 60;
        const int x = std::max(0, static_cast<int>(a.bbox.x) - pad);
        const int y = std::max(0, static_cast<int>(a.bbox.y) - pad);
        const int w = std::min(image.cols - x, static_cast<int>(a.bbox.width) + 2 * pad);
        const int h = std::min(image.rows - y, static_cast<int>(a.bbox.height) + 2 * pad);
        if (w <= 0 || h <= 0) continue;

        cv::Mat annotated = drawArmorResult(image, result);
        cv::Mat zoom;
        cv::resize(annotated(cv::Rect(x, y, w, h)), zoom, cv::Size(), 4.0, 4.0,
                   cv::INTER_NEAREST);
        const fs::path zp = out / ("armor_zoom_" + std::to_string(i + 1) + ".png");
        if (!saveImage(zp.string(), zoom)) {
            std::cout << "[错误] 保存失败: " << zp.string() << "\n";
            return 3;
        }
        say("[保存] " + zp.string());
    }

    // ---------- 5. 输出每个装甲板候选的信息 ----------
    if (result.armors.empty()) {
        say("[结果] 未检测到装甲板候选（mask 中可能没有目标，或阈值过严）。");
    }
    for (size_t i = 0; i < result.armors.size(); ++i) {
        const ArmorCandidate& a = result.armors[i];
        std::ostringstream os;
        os << std::fixed << std::setprecision(1);
        os << "\n装甲板 #" << (i + 1) << "（" << colorName(a.color) << "色）\n";
        os << "  外接框   : x=" << a.bbox.x << " y=" << a.bbox.y
           << " w=" << a.bbox.width << " h=" << a.bbox.height << "\n";
        os << "  中心点   : (" << a.center.x << ", " << a.center.y << ")\n";
        const char* cornerName[4] = { "左上", "右上", "右下", "左下" };
        for (int k = 0; k < 4; ++k) {
            os << "  角点 " << k << "(" << cornerName[k] << "): ("
               << a.corners[k].x << ", " << a.corners[k].y << ")\n";
        }
        os << "  配对灯条 : #" << a.leftBarId << " / #" << a.rightBarId
           << "   间距 " << a.barGap << " px   灯条长 " << a.barLength << " px"
           << "   宽高比 " << std::setprecision(2) << a.plateRatio << "\n";
        os << "  是否有效 : " << (a.valid ? "是" : "否");
        say(os.str());
    }

    // ---------- 6. 可选：显示窗口 ----------
    if (opt.show) {
        cv::Mat shown = drawArmorResult(image, result);
        const double scale = std::min(1.0, 900.0 / std::max(image.cols, image.rows));
        if (scale < 1.0) cv::resize(shown, shown, cv::Size(), scale, scale);
        cv::imshow("Armor Detection", shown);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }

    {
        std::ostringstream os;
        os << "\n[完成] 共输出 " << result.armors.size()
           << " 个装甲板候选，结果已写入 " << (out / "result.txt").string();
        say(os.str());
    }
    return 0;
}
