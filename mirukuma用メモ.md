残念ながら、mirukumaの記憶力は絶望的なので、備忘録的に色々書いてく。
ローカルサーバーの立て方:
    python -m http.server 8888を実行
学習させたモデルをkeras.jsで動くようにする方法:
    1.cdで/PerfectMakeGirls/python/ まで移動。
    2.python /path/to/encoder.py generator.h5を実行。
    3."generator_metadata.json","generator_weights.buf","generator.json"の三つのファイルを/PerfectMakeGirls/data/に移動させる
    4.多分動く。ちなみに出力形式は、width*height*3(色数)の長さを持った一次元配列になるはず。（どういう構造かは要検証）
