// ============================================================
// detect.cpp —— 颜色分割、灯条候选、灯条配对、装甲板四角点
// ============================================================
#include "detect.h"

#include <algorithm>
#include <cmath>
#include <limits>

const char* colorName(ArmorColor c) { return c == ArmorColor::RED ? "红" : "蓝"; }
const char* colorTag(ArmorColor c)  { return c == ArmorColor::RED ? "RED" : "BLUE"; }

namespace {

// ------------------------------------------------------------
// 从旋转矩形的 4 个顶点算出：长边、短边、中心点、长轴相对竖直方向的倾角
// 不用 minAreaRect 自带的 angle（不同 OpenCV 版本语义不同），
// 直接量 4 条边里最长的一条，方向最可靠。
// ------------------------------------------------------------
void measureBar(const cv::Point2f pts[4], float& length, float& width,
                cv::Point2f& center, float& tiltDeg) {
    double maxLen = 0.0;
    double minLen = std::numeric_limits<double>::max();
    cv::Point2f bestP, bestQ;
    for (int k = 0; k < 4; ++k) {
        const cv::Point2f& p = pts[k];
        const cv::Point2f& q = pts[(k + 1) % 4];
        const double len = cv::norm(q - p);
        // 矩形的 4 条边是"长、短、长、短"，最长边给方向，最短边就是灯条宽度
        if (len > maxLen) { maxLen = len; bestP = p; bestQ = q; }
        if (len < minLen) { minLen = len; }
    }
    length = static_cast<float>(maxLen);
    width  = static_cast<float>(minLen == std::numeric_limits<double>::max() ? 0.0 : minLen);

    center = cv::Point2f(0.f, 0.f);
    for (int k = 0; k < 4; ++k) center += pts[k];
    center *= 0.25f;

    // 长轴方向：让向量朝下（dy>0），再算它偏离竖直方向多少度
    float dx = bestQ.x - bestP.x;
    float dy = bestQ.y - bestP.y;
    if (dy < 0.f) { dx = -dx; dy = -dy; }
    tiltDeg = static_cast<float>(std::atan2(dx, dy) * 180.0 / CV_PI);
    if (tiltDeg > 90.f)  tiltDeg -= 180.f;
    if (tiltDeg < -90.f) tiltDeg += 180.f;
}

// ------------------------------------------------------------
// 四角点排序：0 左上 → 1 右上 → 2 右下 → 3 左下
// 用 x+y 的最小/最大值定左上/右下，用 x-y 的最大/最小值定右上/左下，
// 这样目标倾斜时顺序也不会乱。
// ------------------------------------------------------------
std::array<cv::Point2f, 4> orderCorners(std::array<cv::Point2f, 4> pts) {
    std::array<cv::Point2f, 4> out{};
    int iMinSum = 0, iMaxSum = 0, iMaxDiff = 0, iMinDiff = 0;
    for (int k = 1; k < 4; ++k) {
        const float s = pts[k].x + pts[k].y;
        const float d = pts[k].x - pts[k].y;
        if (s < pts[iMinSum].x + pts[iMinSum].y)  iMinSum = k;
        if (s > pts[iMaxSum].x + pts[iMaxSum].y)  iMaxSum = k;
        if (d > pts[iMaxDiff].x - pts[iMaxDiff].y) iMaxDiff = k;
        if (d < pts[iMinDiff].x - pts[iMinDiff].y) iMinDiff = k;
    }
    out[0] = pts[iMinSum];   // 左上
    out[1] = pts[iMaxDiff];  // 右上
    out[2] = pts[iMaxSum];   // 右下
    out[3] = pts[iMinDiff];  // 左下
    return out;
}

// ------------------------------------------------------------
// 生成某个颜色的 mask：白色 = 保留的目标像素
// ------------------------------------------------------------
cv::Mat buildMask(const cv::Mat& hsv, ArmorColor color, const DetectParams& p) {
    cv::Mat mask;
    if (color == ArmorColor::RED) {
        // 红色在 HSV 中跨色相边界：暖红灯芯的 H 落在 0~35，
        // 偏暗的红色光晕落在 160~179，两段取或
        cv::Mat low, high;
        cv::inRange(hsv, cv::Scalar(0, p.sMin, p.vMin),
                    cv::Scalar(p.hueRedMax, 255, 255), low);
        cv::inRange(hsv, cv::Scalar(p.hueRedWrapMin, p.sMin, p.vMin),
                    cv::Scalar(179, 255, 255), high);
        mask = low | high;
    }
    else {
        cv::inRange(hsv, cv::Scalar(p.hueBlueMin, 100, p.blueVMin),
                    cv::Scalar(p.hueBlueMax, 255, 255), mask);
    }

    // 灯条是"一颗颗灯珠"，直接取轮廓会断成几段：先做竖向闭运算连起来，
    // 再轻微开运算去掉孤立噪点。注意横向不开大核，否则相邻灯条会粘连。
    cv::Mat kernel = cv::getStructuringElement(
        cv::MORPH_RECT, cv::Size(p.closeW, p.closeH));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

    return mask;
}

// ------------------------------------------------------------
// 从 mask 提取灯条候选：面积 → 长宽比 → 尺寸 → 倾角
// ------------------------------------------------------------
std::vector<LightBar> extractBars(const cv::Mat& mask, ArmorColor color,
                                  const DetectParams& p) {
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<LightBar> bars;
    for (const auto& contour : contours) {
        const double area = cv::contourArea(contour);
        if (area < p.minArea) continue;                 // 太小：噪声

        const cv::RotatedRect rr = cv::minAreaRect(contour);
        cv::Point2f pts[4];
        rr.points(pts);

        LightBar bar;
        bar.color = color;
        bar.rect  = rr;
        bar.area  = area;
        measureBar(pts, bar.length, bar.width, bar.center, bar.tiltDeg);

        if (bar.length < p.minLen || bar.length > p.maxLen) continue;  // 尺寸不对
        if (bar.width  < p.minWid || bar.width  > p.maxWid)  continue;
        if (bar.width <= 0.f || bar.length / bar.width < p.minAspect) continue;  // 不够细长
        if (std::fabs(bar.tiltDeg) > p.maxTilt) continue;              // 不是近似竖直
        bars.push_back(bar);
    }

    std::sort(bars.begin(), bars.end(),
              [](const LightBar& a, const LightBar& b) { return a.center.x < b.center.x; });
    for (int i = 0; i < static_cast<int>(bars.size()); ++i) bars[i].id = i;
    return bars;
}

// ------------------------------------------------------------
// 一根灯条上取"靠左"的两个点、另一根上取"靠右"的两个点，
// 合成装甲板的 4 个角点
// ------------------------------------------------------------
std::array<cv::Point2f, 4> makePlateCorners(const LightBar& left, const LightBar& right) {
    cv::Point2f pl[4], pr[4];
    left.rect.points(pl);
    right.rect.points(pr);

    std::vector<cv::Point2f> lp(pl, pl + 4), rp(pr, pr + 4);
    std::sort(lp.begin(), lp.end(),
              [](const cv::Point2f& a, const cv::Point2f& b) { return a.x < b.x; });
    std::sort(rp.begin(), rp.end(),
              [](const cv::Point2f& a, const cv::Point2f& b) { return a.x < b.x; });

    std::array<cv::Point2f, 4> raw{ lp[0], lp[1], rp[2], rp[3] };
    return orderCorners(raw);
}

}  // namespace

// ============================================================
// 单帧检测主流程
// ============================================================
FrameResult detectFrame(const cv::Mat& bgrImage, const DetectParams& params) {
    FrameResult result;
    if (bgrImage.empty() || bgrImage.type() != CV_8UC3) return result;

    cv::Mat blurred;
    cv::GaussianBlur(bgrImage, blurred, cv::Size(3, 3), 0);

    cv::Mat hsv;
    cv::cvtColor(blurred, hsv, cv::COLOR_BGR2HSV);

    const ArmorColor colors[2] = { ArmorColor::RED, ArmorColor::BLUE };
    for (ArmorColor color : colors) {
        cv::Mat mask = buildMask(hsv, color, params);
        if (color == ArmorColor::RED) {
            result.maskRed = mask;
            result.redMaskPixels = cv::countNonZero(mask);
        }
        else {
            result.maskBlue = mask;
            result.blueMaskPixels = cv::countNonZero(mask);
        }

        std::vector<LightBar> bars = extractBars(mask, color, params);
        result.bars.insert(result.bars.end(), bars.begin(), bars.end());

        // ---------- 灯条配对 ----------
        struct Pair {
            int left = -1, right = -1;
            float gap = 0.f, len = 0.f, dy = 0.f, score = 0.f;
            ArmorCandidate armor;
        };
        std::vector<Pair> pairs;

        for (size_t i = 0; i < bars.size(); ++i) {
            for (size_t j = i + 1; j < bars.size(); ++j) {
                const LightBar& a = bars[i];
                const LightBar& b = bars[j];
                // 保证 a 在左、b 在右
                const LightBar* L = &a;
                const LightBar* R = &b;
                if (L->center.x > R->center.x) std::swap(L, R);

                const float la = L->length, lb = R->length;
                if (std::min(la, lb) <= 0.f) continue;

                // 1) 两灯条长度相近
                if (std::max(la, lb) / std::min(la, lb) > params.lenRatioMax) continue;

                const float avgLen = (la + lb) * 0.5f;
                const float dy  = std::fabs(L->center.y - R->center.y);
                const float gap = R->center.x - L->center.x;

                // 2) 中心上下偏差合理
                if (dy > params.dyRatioMax * avgLen) continue;
                // 3) 水平间距与灯条长度的比例像一块装甲板
                if (gap < params.gapRatioMin * avgLen) continue;
                if (gap > params.gapRatioMax * avgLen) continue;
                // 4) 方向协调
                if (std::fabs(L->tiltDeg - R->tiltDeg) > params.tiltDiffMax) continue;
                // 5) 竖直方向要有足够重叠
                const float aTop = L->center.y - la * 0.5f, aBot = L->center.y + la * 0.5f;
                const float bTop = R->center.y - lb * 0.5f, bBot = R->center.y + lb * 0.5f;
                const float overlap = std::max(0.f, std::min(aBot, bBot) - std::max(aTop, bTop));
                if (overlap / std::min(la, lb) < params.overlapMin) continue;

                // 6) 合成四角点 + 整体比例检查
                ArmorCandidate armor;
                armor.color = color;
                armor.corners = makePlateCorners(*L, *R);
                armor.leftBarId = L->id;
                armor.rightBarId = R->id;
                armor.barGap = gap;
                armor.barLength = avgLen;

                float minX = armor.corners[0].x, maxX = armor.corners[0].x;
                float minY = armor.corners[0].y, maxY = armor.corners[0].y;
                cv::Point2f sum(0.f, 0.f);
                for (const auto& c : armor.corners) {
                    minX = std::min(minX, c.x); maxX = std::max(maxX, c.x);
                    minY = std::min(minY, c.y); maxY = std::max(maxY, c.y);
                    sum += c;
                }
                armor.center = sum * 0.25f;
                armor.bbox = cv::Rect2f(minX, minY, maxX - minX, maxY - minY);
                armor.plateRatio = (maxY > minY) ? (maxX - minX) / (maxY - minY) : 0.f;
                if (armor.plateRatio < params.plateRatioMin ||
                    armor.plateRatio > params.plateRatioMax) continue;
                armor.valid = true;

                Pair p;
                p.left = L->id;
                p.right = R->id;
                p.gap = gap;
                p.len = avgLen;
                p.dy = dy;
                p.armor = armor;
                // 打分：上下越齐、间距比例越接近典型值(≈2.2)、长度越接近越好
                p.score = dy / avgLen
                        + 0.3f * std::fabs(gap / avgLen - 2.2f)
                        + std::fabs(la - lb) / avgLen;
                pairs.push_back(p);
            }
        }

        // ---------- 去重：同一根灯条只参与一块装甲板 ----------
        std::sort(pairs.begin(), pairs.end(),
                  [](const Pair& x, const Pair& y) { return x.score < y.score; });
        std::vector<int> usedBar;
        std::vector<ArmorCandidate> kept;
        for (const Pair& p : pairs) {
            if (std::find(usedBar.begin(), usedBar.end(), p.left) != usedBar.end()) continue;
            if (std::find(usedBar.begin(), usedBar.end(), p.right) != usedBar.end()) continue;

            bool duplicate = false;
            for (const auto& k : kept) {
                const float dx = std::fabs(k.center.x - p.armor.center.x);
                const float dy2 = std::fabs(k.center.y - p.armor.center.y);
                if (dx < k.bbox.width * 0.6f && dy2 < k.bbox.height) { duplicate = true; break; }
            }
            if (duplicate) continue;

            usedBar.push_back(p.left);
            usedBar.push_back(p.right);
            kept.push_back(p.armor);
        }
        result.armors.insert(result.armors.end(), kept.begin(), kept.end());
    }

    // 蓝色目标即使没有，也保留空 mask（全黑），方便提交时对照
    if (result.maskBlue.empty()) result.maskBlue = cv::Mat::zeros(bgrImage.size(), CV_8UC1);
    if (result.maskRed.empty())  result.maskRed  = cv::Mat::zeros(bgrImage.size(), CV_8UC1);
    return result;
}

// ============================================================
// 灯条候选图
// ============================================================
cv::Mat drawBarCandidates(const cv::Mat& bgrImage, const std::vector<LightBar>& bars) {
    cv::Mat canvas = bgrImage.clone();
    for (const auto& bar : bars) {
        const cv::Scalar col = (bar.color == ArmorColor::RED) ? cv::Scalar(0, 255, 0)
                                                              : cv::Scalar(0, 200, 255);
        cv::Point2f pts[4];
        bar.rect.points(pts);
        for (int k = 0; k < 4; ++k)
            cv::line(canvas, pts[k], pts[(k + 1) % 4], col, 1, cv::LINE_AA);
        cv::circle(canvas, bar.center, 2, cv::Scalar(0, 255, 255), -1, cv::LINE_AA);

        char label[16];
        std::snprintf(label, sizeof(label), "%c%d", colorTag(bar.color)[0], bar.id);
        cv::putText(canvas, label, cv::Point(static_cast<int>(bar.center.x) - 26,
                                             static_cast<int>(bar.center.y) + 4),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
    }
    return canvas;
}

// ============================================================
// 最终结果图：外接框、装甲板四角点(0~3)、中心点
// ============================================================
cv::Mat drawArmorResult(const cv::Mat& bgrImage, const FrameResult& result) {
    cv::Mat canvas = bgrImage.clone();

    // 先把所有灯条候选淡淡画出来，便于核对是"哪两根配成了一块"
    for (const auto& bar : result.bars) {
        cv::Point2f pts[4];
        bar.rect.points(pts);
        const cv::Scalar col = (bar.color == ArmorColor::RED) ? cv::Scalar(0, 160, 0)
                                                              : cv::Scalar(160, 80, 0);
        for (int k = 0; k < 4; ++k)
            cv::line(canvas, pts[k], pts[(k + 1) % 4], col, 1, cv::LINE_AA);
    }

    for (size_t i = 0; i < result.armors.size(); ++i) {
        const ArmorCandidate& a = result.armors[i];
        const cv::Scalar col = (a.color == ArmorColor::RED) ? cv::Scalar(0, 0, 255)
                                                            : cv::Scalar(255, 0, 0);

        std::vector<cv::Point> poly;
        for (const auto& c : a.corners)
            poly.emplace_back(cvRound(c.x), cvRound(c.y));
        cv::polylines(canvas, poly, true, col, 2, cv::LINE_AA);

        cv::Rect box(cvRound(a.bbox.x), cvRound(a.bbox.y),
                     cvRound(a.bbox.width), cvRound(a.bbox.height));
        cv::rectangle(canvas, box, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);

        // 中心点
        cv::drawMarker(canvas, cv::Point(cvRound(a.center.x), cvRound(a.center.y)),
                       cv::Scalar(0, 255, 255), cv::MARKER_CROSS, 10, 1, cv::LINE_AA);

        // 四角点编号 0 左上 / 1 右上 / 2 右下 / 3 左下
        for (int k = 0; k < 4; ++k) {
            const cv::Point pt(cvRound(a.corners[k].x), cvRound(a.corners[k].y));
            cv::circle(canvas, pt, 3, cv::Scalar(255, 0, 255), -1, cv::LINE_AA);
            cv::putText(canvas, std::to_string(k), pt + cv::Point(4, -4),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 255), 1, cv::LINE_AA);
        }

        // 左上角标注：颜色 + 候选项号
        char tag[32];
        std::snprintf(tag, sizeof(tag), "%s #%d", colorTag(a.color), static_cast<int>(i) + 1);
        cv::putText(canvas, tag, cv::Point(box.x, std::max(14, box.y - 20)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
    }
    return canvas;
}
