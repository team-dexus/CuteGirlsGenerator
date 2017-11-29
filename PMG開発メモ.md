mirukumaが受験でPMGをさわれなくなるので、メンバー向けに。  
# JS関連

## kerasのモデルを使う
WebDNNのセットアップは済んでるものとします。済んでいなかったら https://mil-tokyo.github.io/webdnn/docs/ を参考に各自やってください。
例えば、学習済みのモデルがあって、generator.h5というファイルに保存してあるとする。  
まず、下記のコードを実行する。（学習を行うpythonコード(GANTest3.py)とかを実行してからじゃないと動かない）
    
    g = generator_model(round(24/4),round(40/4))
    g.load_weights('generator.h5')

    from webdnn.frontend.keras import KerasConverter
    from webdnn.backend import generate_descriptor
    
    graph = KerasConverter(batch_size=1).convert(g)
    exec_info = generate_descriptor("webgpu", graph)  # also "webassembly", "webgl", "fallback" are available.
    exec_info.save("./output")
    
あとは、https://mil-tokyo.github.io/webdnn/docs/tutorial/keras.html の"3. Run on web browser"を参考ににやれば動くと思う。
## タグに関して
/python/i2v/tag_list.jsonに使用可能なタグがJSONになって保存してある。
また、generatorへのタグの入力は
"0.01 , 0.03 , 0.94 , .........."(1539個)みたいな配列になっている。
単語 to 上記の配列 のやり方はjsに詳しくないのでわからない。
あと、WebDNNで複数入力可能か未検証。できなかったら申し訳ない（多分できると思うが）

# python関連
## モデルに関して
mirukumaははっきり言ってディープラーニングのことをほとんど知らない。
論文とかも一応読んで見たが全然わからなかった（）
だからモデルはもう、大胆に作り直してしまって構わない。

それとディープラーニングは"学問"のレベルに達していないような気がしている（全くの見当違いかもしれないが）
だから、色々モデルを変えて見て、実験するのが最適なのかもしれない

## タグに関して
通常はpythonフォルダのtags.pickleを読み込む。
また、画像を追加・変更、また、学習する解像度を上げる場合は。
https://drive.google.com/open?id=1H4kRNrfNypMWCPcTodp9RR5W0EzctXsS  
ここのillust2vec_tag_ver200.caffemodelを/python/i2vに置くか
illust2vec.pickleを/pythonに置くか（こちらの方が早い......?)すれば、タグ推論を新たに行うことができる。

## 学習元画像に関して
今までは、透化pngの一人立ち絵に限定して学習させていたが、タグ付けの実装で、もはやその必要は無くなったかもしれないし、無くなってないかもしれない。  
ひょっとすると、R18/U18関係なく学習させることが可能になっているかも......?
2500ぐらいの画像が、上記のgoogle driveに入っているので、それを使うのも良いかもしれない
