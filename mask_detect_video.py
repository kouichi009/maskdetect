# ライブラリのインポート
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# 学習済みモデルのインポート
model = load_model('maskmodel.h5')

#　イメージサイズ
img_width, img_height = 150, 150

# HAARカスケード判定器のロード
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# ビデオの読込み
capture = cv2.VideoCapture('maskvideo.mp4')

# 動画の何フレーム目か？
img_count_total = 0

# 描画のパラメーター
font = cv2.FONT_HERSHEY_SIMPLEX
org = (1, 1)
class_label = ''
fontScale = 1
color = (255, 0, 0)  # Blue
thickness = 2

# 無限ループの開始
while True:
    img_count_total += 1  # フレーム数カウンターをインクリメント(+1)
    response, color_img = capture.read()  # captureの結果と静止画を取得

    if response == False:  # レスポンスがFalseなら終了
        break

    scale = 50  # 計算量を減らすためサイズを縮小する割合（50%）
    width = int(color_img.shape[1]*50/100)  # ウィズ（幅）
    height = int(color_img.shape[0]*50/100)  # ハイト（高さ）
    dim = (width, height)  # 画像サイズ

    color_img = cv2.resize(
        color_img, dim, interpolation=cv2.INTER_AREA)  # カラー画像をリサイズ
    gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)  # グレースケールに変換

    faces = face_cascade.detectMultiScale(gray_img, 1.1, 6)  # 全ての顔領域を取得

    img_count = 0  # 同一画像内の顔データを順に処理
    for (x, y, w, h) in faces:  # 　顔を1枚づつ取り出す
        org = (x-10, y-10)  # ラベルの表示座標（左に10px,　上に10px）
        img_count += 1  # 同一画像内での顔画像のインデックス
        color_face = color_img[y:y+h, x:x+w]  # 顔画像を切り出す
        cv2.imwrite('input/%d%dface.jpg' %
                    (img_count_total, img_count), color_face)  # ファイルに書き出す
        img = load_img('input/%d%dface.jpg' % (img_count_total, img_count),
                       target_size=(img_width, img_height))  # ロードしてリサイズ
        img = img_to_array(img)  # 画像を配列に
        img = np.expand_dims(img, axis=0)  # 先頭に配列要素を追加
        prediction = model.predict(img)  # kerasモデルで推定

        if prediction == 0:
            class_label = "With Mask"
            color = (255, 0, 0)  # Blue (BGR)

        else:
            class_label = 'No Mask'
            color = (0, 255, 0)  # Green(BGR)

        cv2.rectangle(color_img, (x, y), (x+w, y+h),
                      (0, 0, 255), 3)  # 顔を囲む矩形を描画. Red(BGR)
        cv2.putText(color_img, class_label, org, font, fontScale,
                    color, thickness, cv2.LINE_AA)  # テキストラベル

    # タイトルを表示
    cv2.imshow('Face Mask Detection', color_img)
    # ループの停止（qのキー）
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()  # 動画キャプチャを解放
cv2.destroyAllWindows()  # すべてのWindowを閉じる
print("img_count_total@@: ", img_count_total)
