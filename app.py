import streamlit as st
import numpy as np
from scipy.optimize import linprog

st.title("🔢 Bài toán vận tải - LCM + MODI")

# Nhập số hàng và cột
rows = st.number_input("Số hàng (nguồn)", min_value=1, max_value=10, step=1)
cols = st.number_input("Số cột (đích)", min_value=1, max_value=10, step=1)

if rows and cols:
    st.subheader("Nhập ma trận chi phí, cung và cầu")

    cost = []
    for i in range(int(rows)):
        row = st.text_input(f"Chi phí hàng {i+1} (cách nhau bởi dấu cách)", key=f"cost_{i}")
        try:
            row_data = list(map(int, row.strip().split()))
            if len(row_data) != cols:
                raise ValueError
            cost.append(row_data)
        except:
            cost.append([0]*cols)

    supply_input = st.text_input("Cung từng nguồn (cách nhau bởi dấu cách)")
    demand_input = st.text_input("Cầu từng đích (cách nhau bởi dấu cách)")

    try:
        supply = list(map(int, supply_input.strip().split()))
        demand = list(map(int, demand_input.strip().split()))

        cost = np.array(cost)
        supply = np.array(supply)
        demand = np.array(demand)

        # Cân bằng cung - cầu
        total_supply = supply.sum()
        total_demand = demand.sum()
        if total_supply != total_demand:
            diff = abs(total_supply - total_demand)
            if total_supply > total_demand:
                demand = np.append(demand, diff)
                cost = np.hstack([cost, np.zeros((cost.shape[0], 1), dtype=int)])
                st.info("Đã thêm 1 điểm cầu giả (cost = 0)")
            else:
                supply = np.append(supply, diff)
                cost = np.vstack([cost, np.zeros((1, cost.shape[1]), dtype=int)])
                st.info("Đã thêm 1 điểm cung giả (cost = 0)")

        rows, cols = cost.shape

        def least_cost_method(cost, supply, demand):
            allocation = np.zeros_like(cost)
            supply = supply.copy()
            demand = demand.copy()
            while np.any(supply) and np.any(demand):
                masked_cost = cost + (supply[:, None] == 0) * 1e6 + (demand == 0) * 1e6
                i, j = divmod(np.argmin(masked_cost), cost.shape[1])
                qty = min(supply[i], demand[j])
                allocation[i][j] = qty
                supply[i] -= qty
                demand[j] -= qty
            return allocation

        def total_cost(cost, allocation):
            return np.sum(cost * allocation)

        if st.button("🔹 Phân bổ ban đầu (LCM)"):
            alloc = least_cost_method(cost, supply, demand)
            st.success("✅ Kết quả phân bổ theo LCM:")
            st.write(alloc)
            st.write(f"💰 Tổng chi phí vận chuyển: {total_cost(cost, alloc)}")

        if st.button("🔸 Tối ưu bằng MODI"):
            c = cost.flatten()

            A_eq_supply = np.zeros((rows, rows * cols))
            for i in range(rows):
                A_eq_supply[i, i * cols:(i + 1) * cols] = 1

            A_eq_demand = np.zeros((cols, rows * cols))
            for j in range(cols):
                for i in range(rows):
                    A_eq_demand[j, i * cols + j] = 1

            A_eq = np.vstack([A_eq_supply, A_eq_demand])
            b_eq = np.concatenate([supply, demand])
            bounds = [(0, None) for _ in range(rows * cols)]

            res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

            if res.success:
                x_opt = res.x.reshape((rows, cols))
                st.success("✅ Phân bổ tối ưu theo MODI:")
                st.write(np.round(x_opt, 2))
