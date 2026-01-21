import pandas as pd
import matplotlib.pyplot as plt

# ----------------------- k , forward --------------------------------
# 1. import CSV
df1 = pd.read_csv("k1_p10_400700.csv")
df5 = pd.read_csv("k5_p10_400700.csv")
df10 = pd.read_csv("k10_p10_400700.csv")
df1_forward = pd.read_csv("k1_p10_400700_forward.csv")
df5_forward = pd.read_csv("k5_p10_400700_forward.csv")
df10_forward = pd.read_csv("k10_p10_400700_forward.csv")
round = range(900)

plt.figure(figsize=(8, 5))
plt.plot(round, df1['loss'], marker='.', markersize=2, label='k=1 (decomfl-central)')
plt.plot(round, df5['loss'], marker='.', markersize=2, label='k=5 (decomfl-central)')
plt.plot(round, df10['loss'], marker='.', markersize=2, label='k=10 (decomfl-central)')
plt.plot(round, df1_forward['loss'], marker='.', markersize=2, label='k=1 (decomfl-forward)')
plt.plot(round, df5_forward['loss'], marker='.', markersize=2, label='k=5 (decomfl-forward)')
plt.plot(round, df10_forward['loss'], marker='.', markersize=2, label='k=10 (decomfl-forward)')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Evaluation Loss over Rounds with Different Local Steps k, and two gradient estimate methods')
plt.grid(True)
plt.legend()
plt.show()

# 3. accuracy-round
plt.figure(figsize=(8, 5))
plt.plot(round, df1['acc'], marker='.', markersize=2, label='k=1 (decomfl-central)')
plt.plot(round, df5['acc'], marker='.', markersize=2, label='k=5 (decomfl-central)')
plt.plot(round, df10['acc'], marker='.', markersize=2, label='k=10 (decomfl-central)')
plt.plot(round, df1_forward['acc'], marker='.', markersize=2, label='k=1 (decomfl-forward)')
plt.plot(round, df5_forward['acc'], marker='.', markersize=2, label='k=5 (decomfl-forward)')
plt.plot(round, df10_forward['acc'], marker='.', markersize=2, label='k=10 (decomfl-forward)')


plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Evaluation Accuracy over Rounds with Different Local Steps k, and two gradient estimate methods')
plt.grid(True)
plt.legend()
plt.show()


# ----------------------------- p -------------------------------
df5_p20 = pd.read_csv("k5_p20_400700.csv")
df5_p40 = pd.read_csv("k5_p40_400700.csv")

plt.figure(figsize=(8, 5))
plt.plot(round, df5['loss'], marker='.', markersize=2, label='p=10 (decomfl-central)')
plt.plot(round, df5_p20['loss'], marker='.', markersize=2, label='p=20 (decomfl-central)')
plt.plot(round, df5_p40['loss'], marker='.', markersize=2, label='p=40 (decomfl-central)')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Evaluation Loss over Rounds with Different Perturbation P')
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(round, df5['acc'], marker='.', markersize=2, label='p=10 (decomfl-central)')
plt.plot(round, df5_p20['acc']/100, marker='.', markersize=2, label='p=20 (decomfl-central)')
plt.plot(round, df5_p40['acc'], marker='.', markersize=2, label='p=20 (decomfl-central)')

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Evaluation Accuracy over Rounds with Different Perturbation P')
plt.grid(True)
plt.legend()
plt.show()

# --------------------------------- fedavg --------------------------------
fed1 = pd.read_csv("k1_fedavg.csv")
fed5 = pd.read_csv("k5_fedavg.csv")
fed10 = pd.read_csv("k10_fedavg.csv")

jump_round = range(0,900,5)

plt.figure(figsize=(8, 5))
plt.plot(jump_round, fed1['loss'], marker='.', markersize=2, label='k=1 (fedavg)')
plt.plot(round, df1['loss'], marker='.', markersize=2, label='k=1 (decomfl-central)')
plt.plot(jump_round, fed5['loss'], marker='.', markersize=2, label='k=5 (fedavg)')
plt.plot(round, df5['loss'], marker='.', markersize=2, label='k=5 (decomfl-central)')
plt.plot(jump_round, fed10['loss'], marker='.', markersize=2, label='k=10 (fedavg)')
plt.plot(round, df10['loss'], marker='.', markersize=2, label='k=10 (decomfl-central)')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Evaluation Loss over Rounds, comparing fedavg and decomfl-central')
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(jump_round, fed1['acc'], marker='.', markersize=2, label='k=1 (fedavg)')
plt.plot(round, df1['acc'], marker='.', markersize=2, label='k=1 (decomfl-central)')
plt.plot(jump_round, fed5['acc'], marker='.', markersize=2, label='k=5 (fedavg)')
plt.plot(round, df5['acc'], marker='.', markersize=2, label='k=5 (decomfl-central)')
plt.plot(jump_round, fed10['acc'], marker='.', markersize=2, label='k=10 (fedavg)')
plt.plot(round, df10['acc'], marker='.', markersize=2, label='k=10 (decomfl-central)')

plt.xlabel('Epoch')
plt.ylabel('accuracy')
plt.title('Evaluation Accuracy over Rounds, comparing fedavg and decomfl-central')
plt.grid(True)
plt.legend()
plt.show()


# -----------------same communication, P10-20-40, P40-20-10-------------------------------
# df10_reverse = pd.read_csv("k10_p40_200500.csv")
#
# print(len(df10_reverse))
#
# plt.figure(figsize=(8, 5))
#
# plt.plot(round, df10['loss'], marker='.', markersize=2, label='k=10 (decomfl-central), p=10,20,40')
# plt.plot(range(len(df10_reverse)), df10_reverse['loss'], marker='.', markersize=2, label='k=10 (decomfl-central), p=10,20,40')
#
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Evaluation Loss over Rounds with Different P change')
# plt.grid(True)
# plt.legend()
# plt.show()
#
# # 3. accuracy-round
# plt.figure(figsize=(8, 5))
# plt.plot(round, df10['acc'], marker='.', markersize=2, label='k=10 (decomfl-central), p=10,20,40')
# plt.plot(range(len(df10_reverse)), df10_reverse['acc'], marker='.', markersize=2, label='k=10 (decomfl-central), p=10,20,40')
#
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.title('Evaluation Accuracy over Rounds with Different P change')
# plt.grid(True)
# plt.legend()
# plt.show()
