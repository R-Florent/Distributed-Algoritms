import numpy as np

def Kaczmarz(matrice_A, matrice_b, inconnue, max_iterations=100, tolerance=1e-6):
    for iteration in range(max_iterations):
        x_old = inconnue.copy()  # 古い解のコピーを作成
        for indise, equation in enumerate(matrice_A):
            transposer_a = 0
            norme = 0
            for j, element in enumerate(equation):
                transposer_a += element * inconnue[j]  # 行とxの内積を計算
                norme += element * element  # 行のノルムを計算

            # カチュマルツアルゴリズムに基づく更新
            atixi = (matrice_b[indise] - transposer_a) / norme
            for j, element in enumerate(equation):
                inconnue[j] = inconnue[j] + atixi * element

        # 収束判定（許容誤差に基づく）
        if np.linalg.norm(np.array(inconnue) - np.array(x_old)) < tolerance:
            print(f"{iteration + 1} 回のループで収束しました。")
            break

    return inconnue

# アルゴリズムの実行
A = np.array([[4, 1, 2],
              [3, 5, 1],
              [1, 1, 3]])  # 係数行列
b = np.array([4, 7, 3])  # 定数項
x = np.zeros(3)  # 初期推定

resulte = Kaczmarz(A, b, x)

print("近似解:", resulte)
