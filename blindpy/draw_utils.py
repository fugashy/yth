import cv2


def draw_rect(img, result, pargs):
    for r in result.itertuples():
        p1 = (round(r.x1), round(r.y1))
        p2 = (round(r.x2), round(r.y2))
        color = (255, 0, 0)
        cv2.rectangle(img, p1, p2, color, 2)
    return img


def draw_image(img, result, pargs):
    # アルファ値込みで使いたい
    overlay_img = cv2.imread(pargs["draw_image"], cv2.IMREAD_UNCHANGED)

    widths = result["x2"] - result["x1"]
    heights = result["y2"] - result["y1"]
    rs = result.copy()
    rs["largeness"] = widths * heights
    rs = rs.sort_values(by="largeness", ascending=True)

    for r in rs.itertuples():
        # 描画領域の左上と右下座標
        rect_width = round(r.x2) - round(r.x1)
        rect_height = round(r.y2) - round(r.y1)

        resized_overlay = cv2.resize(
                overlay_img,
                (rect_width, rect_height))

        overlay_bgr = resized_overlay[:, :, :3]  # BGRチャンネル
        overlay_alpha = resized_overlay[:, :, 3]  # アルファチャンネル
        # アルファ値を0-1に正規化
        overlay_alpha = overlay_alpha / 255.0

        # 矩形範囲内の背景画像部分を取り出す
        background_region = img[round(r.y1):round(r.y2), round(r.x1):round(r.x2)]

        # アルファチャンネルを使って背景とオーバーレイ画像をブレンド
        for c in range(0, 3):  # 各カラー（BGR）チャンネルで合成
            background_region[:, :, c] = overlay_alpha * overlay_bgr[:, :, c] + (1 - overlay_alpha) * background_region[:, :, c]

            # 合成結果を元の背景画像に適用
            img[round(r.y1):round(r.y2), round(r.x1):round(r.x2)] = background_region



                # img[round(r.y1):round(r.y2), round(r.x1):round(r.x2)] = resized_overlay
    return img


def call(style, img, result, pargs):
    function_by_style = {
            "rect": draw_rect,
            "image": draw_image,
            }
    return function_by_style[style](img, result, pargs)
