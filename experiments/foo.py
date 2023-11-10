C_U, C_X, C_D, C_K = 3, 1, 5, 4
# C_U, C_X, C_D, C_K = 2, 1, 3, 5
x_k_vals, k_vals = list(range(4)), list(range(5))
n_rows, n_cols = len(x_k_vals), len(k_vals)
res = [[None for _ in range(n_cols)] for _ in range(n_rows)]
# d_dist = [[0.25, 0.1, 0.0, 0.0],
#           [0.25, 0.2, 0.5, 0.0],
#           [0.25, 0.3, 0.5, 0.5],
#           [0.25, 0.4, 0.0, 0.5]]
d_dist = [[0.2, 0.25, 0.25, 0.1],
          [0.2, 0.25, 0.25, 0.4],
          [0.2, 0.25, 0.25, 0.1],
          [0.4, 0.25, 0.25, 0.4]]

for j in range(n_cols - 1, -1, -1):
    for i in range(n_rows):
        # Base case
        if j == n_cols - 1:
            v_k, u_k = C_K * i, 0

        # All other cases (use dynamic programming)
        else:
            e_d_k1 = 0

            for k in range(n_rows):
                e_d_k1 += k * d_dist[k][j]

            possible_v_k_vals = []

            for u in x_k_vals:
                g = (C_U * u) + (C_X * i) + (C_D * max(0, e_d_k1 - i - u))
                e_v_k1 = 0

                for d in x_k_vals:
                    x_k1 = max(0, i - u - d)
                    e_v_k1 += d_dist[d][j] * res[x_k1][j + 1][0]

                possible_v_k_vals.append((g + e_v_k1, u))

            v_k, u_k = min(possible_v_k_vals, key=lambda x: x[0])

        res[i][j] = (v_k, u_k)

for row in res:
    print(row)
