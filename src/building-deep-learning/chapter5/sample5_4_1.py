from layers import MulLayer

def main():
    # りんご2個の買い物

    # ------------------------------
    # 定数を初期化する
    # ------------------------------
    apple_price = 100   # リンゴの値段
    apple_num = 2   # リンゴの個数
    tax = 1.1   # 消費税

    # ------------------------------------------
    # 各ノードに対応するレイヤーオブジェクトを生成する
    # ------------------------------------------

    # リンゴの個数を計算するノード
    mul_apple_layer = MulLayer()

    # 消費税を計算するノード
    mul_tax_layer = MulLayer()

    # ------------------------------
    # forward: 順伝播の計算を実行する
    # ------------------------------
    # リンゴの総金額を求める... (1)
    apple_price = mul_apple_layer.forward(apple_price, apple_num)

    # 支払い金額を求める ... (2)
    price = mul_tax_layer.forward(apple_price, tax)

    # ----------------------
    # backward: 各観点の微分を求める
    # ----------------------
    d_total_price = 1  # 逆伝播の最初の入力値

    # 消費税を計算するノードに対して、逆伝播の計算を実行する ... (2')
    dapple_price, d_tax = mul_tax_layer.backward(d_total_price)

    # リンゴの個数を計算するノードに対して、逆伝搬の計算を実行する ... (1')
    d_apple_price, d_apple_num = mul_apple_layer.backward(dapple_price)

    print("支払い金額:", int(price))
    print("支払い金額をリンゴの価格で微分した値:", d_apple_price)
    print("支払い金額をリンゴの個数で微分した値:", int(d_apple_num))
    print("支払い金額を消費税で微分した値:", d_tax)

    pass


if __name__ == '__main__':
    main()


