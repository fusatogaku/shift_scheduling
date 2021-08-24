# shift_scheduling
エクセルで仮作成したシフト表(仮シフト.xlsx)を遺伝的アルゴリズムを用いて最適化したシフト表を作成します。
最適化されたシフトはエクセルファイルで出力されます。(シフト表.xlsx)
 youtube Youtube見ながら作成しました。ほぼ動画通りです。

# 使用環境
仮想環境: venv(python3.8)
(numpy, openpyxl, pandasが入っていれば動作します。)

# 使用方法
python3 shift_scheduling.py

## 細かい項目・調整
シフト日数：
    31日想定で作っています。調整は下記。
    shift_read_excel() の変数 base_table = df.iloc[:, 0:31]... の31を月に合わせて減らしてください。

交叉確率：
    デフォルトで0.5(50%)としています。実行部分の変数 cross_rate で調整できます。

変異確率：
    交叉確率同様、デフォルトは0.5(50%)です。実行部分の変数 mutate_rate で調整できます。

最大世代数：
    デフォルト30世代までで、最も良い個体を出力します。より良い個体(評価関数が０に近い個体)を作るためにもっと増やすことができます。

ファイル名：
    読み込むファイル名・シート名は「仮シフト.xlsx」「仮シフト」となっています。shift_read_excel()の df = pd.read_excel('ファイル名', sheet_name='シート名')で変更できます。
    


