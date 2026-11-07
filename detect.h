#pragma once
// ============================================================
// detect.h —— 装甲板检测的数据结构与单帧接口
//
// 流程：颜色分割 → 灯条候选 → 灯条配对 → 装甲板候选(框/中心/四角点)
// 本阶段只处理单张静态图片，后续接视频时直接复用 detectFrame()
// ============================================================
#include <opencv2/opencv.hpp>
// OpenCV 5 把 contourArea / minAreaRect 等几何函数拆到了 geometry 模块，
// opencv.hpp 不一定带进来，这里显式包含（OpenCV 4 下同样有效）
#include <opencv2/geometry.hpp>
#include <opencv2/imgproc.hpp>
#include <array>
#include <string>
#include <vector>

// ---------------- 颜色类别 ----------------
enum class ArmorColor { RED = 0, BLUE = 1 };

const char* colorName(ArmorColor c);      // "红" / "蓝"（终端输出用）
const char* colorTag(ArmorColor c);       // "RED" / "BLUE"（写在图片上，OpenCV 不支持中文）

// ---------------- 灯条候选 ----------------
struct LightBar {
    ArmorColor color = ArmorColor::RED;
    int        id = -1;              // 同色候选中的编号，配对时用
    cv::RotatedRect rect;            // 最小外接旋转矩形
    cv::Point2f center{0.f, 0.f};    // 中心点
    float      length = 0.f;         // 长边（灯条长度，像素）
    float      width = 0.f;          // 短边（灯条宽度，像素）
    float      tiltDeg = 0.f;        // 长轴相对"竖直方向"的倾角，(-90, 90]
    double     area = 0.0;           // 轮廓面积
};

// ---------------- 装甲板候选 ----------------
struct ArmorCandidate {
    ArmorColor color = ArmorColor::RED;
    cv::Rect2f bbox;                 // 轴对齐外接框
    cv::Point2f center{0.f, 0.f};    // 四角点平均位置
    // 四角点固定顺序：0 左上 → 1 右上 → 2 右下 → 3 左下
    // （图像坐标：x 向右、y 向下；倾斜时按 x+y / x-y 排序保持同一顺序）
    std::array<cv::Point2f, 4> corners{};
    int   leftBarId = -1;            // 左灯条编号
    int   rightBarId = -1;           // 右灯条编号
    float barGap = 0.f;              // 两灯条中心水平间距
    float barLength = 0.f;           // 两灯条平均长度
    float plateRatio = 0.f;          // 外接框宽/高
    bool  valid = false;             // 当前结果是否有效
};

// ---------------- 可调参数（集中放置，方便调试） ----------------
struct DetectParams {
    // 颜色分割（HSV）
    int   sMin = 110;                // 饱和度下限
    int   vMin = 140;                // 亮度下限（灯条是过曝的亮条）
    int   hueRedMax = 35;            // 红/橙 色相上限（0~35，覆盖暖白灯芯）
    int   hueRedWrapMin = 160;       // 红色在 HSV 另一端的起点（160~179）
    int   hueBlueMin = 100;          // 蓝色范围
    int   hueBlueMax = 130;
    int   blueVMin = 80;             // 蓝色亮度下限

    // 形态学：灯条像"虚线"，用竖向闭运算把断续的灯珠连成一根
    int   closeW = 3;
    int   closeH = 9;

    // 灯条轮廓筛选
    double minArea = 25.0;           // 最小轮廓面积
    float  minLen = 18.f;            // 长边下限
    float  maxLen = 120.f;           // 长边上限
    float  minWid = 2.f;             // 短边下限
    float  maxWid = 30.f;            // 短边上限
    float  minAspect = 1.8f;         // 长宽比下限（细长）
    float  maxTilt = 30.f;           // 与竖直方向夹角上限（度）

    // 灯条配对
    float  lenRatioMax = 1.8f;       // 两根灯条长度比上限
    float  dyRatioMax = 0.25f;       // 中心上下偏差 / 平均灯条长度
    float  gapRatioMin = 1.2f;       // 间距 / 平均灯条长度 下限
    float  gapRatioMax = 3.2f;       // 间距 / 平均灯条长度 上限
    float  tiltDiffMax = 15.f;       // 两灯条倾角差上限（度）
    float  overlapMin = 0.5f;        // 竖直方向重叠比例下限

    // 整体比例检查
    float  plateRatioMin = 1.2f;     // 外接框宽/高 下限
    float  plateRatioMax = 4.0f;     // 外接框宽/高 上限
};

// ---------------- 单帧检测结果 ----------------
struct FrameResult {
    std::vector<LightBar>       bars;      // 全部灯条候选（红+蓝）
    std::vector<ArmorCandidate> armors;    // 装甲板候选
    cv::Mat maskRed;                       // 红色 mask（白=保留）
    cv::Mat maskBlue;                      // 蓝色 mask（白=保留）
    int redMaskPixels = 0;
    int blueMaskPixels = 0;
};

// ============================================================
// 单帧检测接口：输入一张彩色图，输出装甲板候选列表 + 调试 mask
// 函数内部不依赖任何全局图片，也不含任何写死的目标坐标。
// ============================================================
FrameResult detectFrame(const cv::Mat& bgrImage, const DetectParams& params = DetectParams());

// 绘制灯条候选图（旋转矩形 + 中心点 + 编号）
cv::Mat drawBarCandidates(const cv::Mat& bgrImage, const std::vector<LightBar>& bars);

// 绘制最终结果图（外接框 + 四角点 0/1/2/3 + 中心点 + 配对编号）
cv::Mat drawArmorResult(const cv::Mat& bgrImage, const FrameResult& result);
