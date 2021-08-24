"""
出勤 : 0
休み : 1
希望休 : 2

元ネタ：遺伝的アルゴリズムでシフト表を作ってみよう！！　#1 ~ #7
https://www.youtube.com/watch?v=0KRlHHud_dQ
"""
import numpy as np, pandas as pd
from random import random
# エクセル読み込み
def shift_read_excel():
    """
    エクセルを読み込み、基礎テーブルと休日テーブルに分ける。

    Returns
    -------
    base_table : pd.DataFrame
        シフト表のシフト部分のみをデータフレームで抽出したもの。
        基礎テーブルとなる。
    holiday : pd.DataFrame
        シフト表の休日(右端１列)のみを抽出したもの。
        休日計算や評価に使用する。
    """
    # エクセル読み込み
    df = pd.read_excel('仮シフト.xlsx', sheet_name='仮シフト')

    # 欲しい部分のみ取り出す
    df = df.iloc[2:7, 1:33]

    # エラー値を変換
    df = df.fillna(0)

    # 希望休を2に変換
    df = df.replace('◎', 2)

    # 休日数を切り離して取得。
    holiday = df.iloc[:, 31:32].reset_index(drop=True)
    holiday.columns = ['休日数']


    # 基礎テーブルを取得
    base_table = df.iloc[:, 0:31].reset_index(drop=True)
    base_table.columns = [i + 1 for i in range(len(base_table.columns))]
    base_table

    return base_table, holiday

# 世代作成
def first_gene(base_table, holiday):
    """
    第一世代作成用の関数。
    勤務日数の抽出と基礎テーブルにランダムな休日を埋め込むための関数。

    Parameters
    ----------
    base_table : pd.DataFrame
        休日を埋め込むための基礎テーブル。
    holiday : pd.DataFrame
        必要休日日数のみのテーブル。
    
    Returns
    -------
    base_table_copy : pd.DataFrame
        基礎テーブルをコピーし、希望休以外の休みを埋め込んだテーブル。
    """
    # 勤務日数を取得
    days = len(base_table.columns)

    # 基礎テーブルコピー
    base_table_copy = base_table.copy()

    # 従業員の数だけループ
    for k in range(len(base_table_copy)):
        # 休日を入れる変数
        h = []

        # 休日数の数だけループ
        while len(h) < holiday.loc[k][0]:
            n = np.random.randint(1, days + 1)
            if not n in h:
                h.append(n)
        
        # 基礎テーブルに休日を埋め込む
        for i in h:
            if base_table_copy.loc[k][i] == 0:
                base_table_copy.loc[k][i] = 1
    
    return base_table_copy

# 休日修正
def holiday_fix(base_table_copy, holiday):
    """
    休日数に関する修正用関数。

    Parameters
    ----------
    base_table_copy : pd.DataFrame
        修正前の世代の個体。
    holiday : pd.DataFrame
        規程休日数のデータフレーム。この休日数をもとに規定数になる様調整する。
    
    Returns
    -------
    base_table_copy : pd.DataFrame
        規程休日数になるように調整された世代。
    """
    
    # 休日数の修正(全員分)
    for k in range(len(base_table_copy)):
        # 日数を取得
        days = len(base_table.columns)
        
        if np.count_nonzero(base_table_copy.iloc[k:k + 1]) != holiday.loc[k][0]:
            sub = np.count_nonzero(base_table_copy.iloc[k:k + 1]) - holiday.loc[k][0]
            buf = 0
            c1 = 1 if sub > 0 else 0
            c2 = 1 if c1 == 0 else 0
            # 増減したい休日数に達するまでループ
            while buf < abs(sub):
                n = np.random.randint(1, days + 1)
                if base_table_copy.loc[k][n] == c1:
                    buf += 1
                    # 休日を変更
                    base_table_copy.loc[k][n] = c2        

        return base_table_copy

# 評価関数
def evalution_function(base_table_copy):
    """
    個体を指定の基準で評価する関数。

    Parameter
    ---------
    base_table_copy : pd.DataFrame
        評価される個体。
    
    Returns
    -------
    score : int
        評価基準に基づいたスコア。
    """
    
    # 評価するため、希望休2を1に変換。->希望は固定のため、評価しないため。
    eva = base_table_copy.replace(2, 1)
    score = 0

    for k in range(len(eva)):
        # 文字列として結合
        x = ''.join([str(i) for i in np.array(eva.iloc[k:k + 1]).flatten()])

        # 評価（制約）
        # ５連勤以上の評価
        score += np.sum([(2 - len(i))** 2 *- 1 for i in x.split('1') if len(i) > 5])
        # ３連休以上の評価
        score += np.sum([(1 - len(i))** 2 *- 1 for i in x.split('1') if len(i) > 3])
        # 飛び石連休の評価
        score += -10 * (len(x.split('101')) -1)

        # 出勤数の評価
        np.sum([abs(len(eva) * 0.7 - (len(eva) - np.sum(eva[k]))) *- 4 for k in eva.columns])

    return score

# 一様交叉実行関数
def exec_crossover(cross_rate, mutate_rate, parent_1, parent_2):
    """
    交叉確率と変異確率に基づいて、交叉・突然変異を行う関数。

    Parameters
    ----------
    cross_rate : float
        交叉を行う確率。(0.1 ~ 1.0)
    mutate_rate : float
        突然変異の確率。(0.1 ~ 1.0)
    parent_1, parent_2 : pd.Dataframe
        交叉する個体。データフレーム。

    Returns
    -------
    child_1, child_2 : pd.Dataframe
        交叉実行後の個体。
    """

    # １ヶ月の日数
    days = len(parent_1.columns)

    # 一次元化
    parent_1 = np.array(parent_1).flatten()
    parent_2 = np.array(parent_2).flatten()

    # 子の変数
    child_1 = []
    child_2 = []

    # 交叉確率によって親と個体を振り分ける。
    for parent_1_, parent_2_ in zip(parent_1, parent_2):
        x = True if cross_rate > random() else False

        if x == True:
            child_1.append(parent_1_)
            child_2.append(parent_2_)
        else:
            child_1.append(parent_2_)
            child_2.append(parent_1_)
    
    # 突然変異
    child_1, child_2 = mutation(mutate_rate, np.array(child_1).flatten(), np.array(child_2).flatten())

    # Pandasに変換
    child_1 = pd.DataFrame(child_1.reshape(int(len(child_1) / days), days))
    child_2 = pd.DataFrame(child_2.reshape(int(len(child_2) / days), days))

    # 列名の変更
    child_1.columns = [i + 1 for i in range(len(child_1.columns))]
    child_2.columns = [i + 1 for i in range(len(child_2.columns))]

    return child_1, child_2

# 突然変異関数
def mutation(mutate_rate, child_1, child_2):
    """
    交叉の際、変異確率に基づいて突然変異を実行する関数。

    Parameters
    ----------
    mutate_rate : float
        突然変異の確率。(0.1 ~ 1.0)
    child_1, child_2 : pd.Dataframe
        突然変異させる個体。

    Returns
    -------
    child_1, child_2
        ランダムな位置を変異(0->1, 1->0)させた個体。
    """
    x = True if mutate_rate > random() else False

    if x == True:
        rand = np.random.permutation([i for i in range(len(child_1))])

        # 遺伝子の10%を変異させる
        rand = rand[:int(len(child_1) // 10)]
        for i in rand:
            if child_1[i] == 1:
                child_1[i] == 0

            if child_1[i] == 0:
                child_1[i] == 1
    
    x = True if mutate_rate > random() else False

    if x == True:
        rand = np.random.permutation([i for i in range(len(child_1))])
        rand = rand[:int(len(child_1) // 10)]
        for i in rand:
            if child_2[i] == 1:
                child_2[i] == 0
            
            if child_2[i] == 0:
                child_2[i] == 1
        
    return child_1, child_2

# 実行
if __name__ == '__main__':
    # エクセルの読み込み
    base_table, holiday = shift_read_excel()

    # 親の保存
    parent = []

    # 遺伝子格納数
    elite_length = 20
    # 最大世代数
    gene_length = 30

    # 交叉確率
    cross_rate = 0.5
    # 変異確率
    mutate_rate = 0.05


    for i in range(100):
        # 世代作成
        base_table_copy = first_gene(base_table, holiday)

        # 休日数の修正
        base_table_copy = holiday_fix(base_table_copy, holiday)
        # 評価
        score = evalution_function(base_table_copy)
        # 世代の格納
        parent.append([score, base_table_copy])


    # 上位20個体を選択
    for i in range(gene_length):
        # 点数が高い順に並び替え
        parent = sorted(np.array(parent), key=lambda x: -x[0])

        # 上位個体を選択
        parent = parent[:elite_length]

        # 優生個体と全世代最強個体の点数を比較。
        # 最高得点の更新
        if i == 0 or top[0] < parent[0][0]:
            top = parent[0]
        else:
            parent.append(top)

        # 各世代
        print('第' + str(i + 1) + '世代')
        # 各世代の最高得点の表示
        print(top[0])
        print(np.array(top[1]))
        
        # 上位20個体を一様交叉して次世代の380個体を生成
        # 子世代
        children = []

        # 遺伝子操作
        for k1, v1 in enumerate(parent):
            for k2, v2 in enumerate(parent):
                if k1 < k2:
                    # 交叉実行
                    child_1, child_2 = exec_crossover(cross_rate, mutate_rate, v1[1], v2[1])
                    # 休日数修正
                    child_1 = holiday_fix(child_1, holiday)
                    child_2 = holiday_fix(child_2, holiday)

                    # 評価
                    score1 = evalution_function(child_1)
                    score2 = evalution_function(child_2)

                    # 子孫を格納
                    children.append([score1, child_1])
                    children.append([score2, child_2])
                    if score1 or score2 == 0.0:
                        break

        # 子孫を親にコピー
        parent = children.copy()

    x = top[1].replace(1, '○').replace(2, '◎').replace(0, '')
    x.to_excel('シフト表.xlsx')