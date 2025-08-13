import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
from kArmedBandit import *

class BanditGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🎰 K-Armed Bandit Simulator")
        self.root.configure(bg="#222")
        self.root.geometry("1400x850")

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TLabel", background="#222", foreground="white", font=("Segoe UI", 14))
        style.configure("TButton", font=("Segoe UI", 14, "bold"), padding=8)
        style.configure("TEntry", font=("Segoe UI", 14))
        style.configure("TCombobox", font=("Segoe UI", 14))
        style.configure("TCheckbutton", font=("Segoe UI", 14), background="#222", foreground="white")

        title_label = ttk.Label(root, text="K-Armed Bandit Simulator", font=("Segoe UI", 20, "bold"), foreground="#00FFAA")
        title_label.grid(row=0, column=0, columnspan=2, pady=15)

        ttk.Label(root, text="Algorithm:").grid(row=1, column=0, sticky="w", padx=10, pady=5)
        self.alg_var = tk.StringVar(value="EPS_GREEDY")
        algo_box = ttk.Combobox(root, textvariable=self.alg_var, values=["EPS_GREEDY", "UCB", "GradientBandit"], width=20)
        algo_box.grid(row=1, column=1, padx=10, pady=5)

        self.param_entries = {}
        params = [
            ("n", 10),
            ("iters", 1000),
            ("eps", 0.1),
            ("c", 1),
            ("alpha", 0.1),
            ("optimistic", True)
        ]
        for i, (name, default) in enumerate(params, start=2):
            ttk.Label(root, text=f"{name}:").grid(row=i, column=0, sticky="w", padx=10, pady=5)
            entry = ttk.Entry(root, width=15)
            entry.insert(0, str(default))
            entry.grid(row=i, column=1, padx=10, pady=5)
            self.param_entries[name] = entry

        self.stationary_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(root, text="Stationary Problem", variable=self.stationary_var).grid(
            row=len(params)+2, column=0, columnspan=2, pady=5
        )

        self.superimpose_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(root, text="Superimpose Graphs", variable=self.superimpose_var).grid(
            row=len(params)+3, column=0, columnspan=2, pady=5
        )

        run_btn = ttk.Button(root, text="🚀 Run Simulation", command=self.run_simulation)
        run_btn.grid(row=len(params)+4, column=0, columnspan=2, pady=10)
        clear_btn = ttk.Button(root, text="🧹 Clear Graphs", command=self.clear_graphs)
        clear_btn.grid(row=len(params)+5, column=0, columnspan=2, pady=5)

        self.fig, (self.ax_gain, self.ax_counts) = plt.subplots(2, 1, figsize=(10, 8))
        self.fig.patch.set_facecolor("#333")
        self.ax_gain.set_facecolor("#222")
        self.ax_counts.set_facecolor("#222")
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().grid(row=0, column=2, rowspan=len(params)+6, padx=15, pady=15)

        self.true_values_text = tk.Text(root, width=25, height=25, font=("Segoe UI", 12), bg="#222", fg="#00FFAA")
        self.true_values_text.grid(row=0, column=3, rowspan=len(params)+6, padx=10, pady=15)
        self.true_values_text.insert(tk.END, "True Bandit Rewards:\n")
        self.true_values_text.configure(state='disabled')

    def run_simulation(self):
        n = int(self.param_entries["n"].get())
        iters = int(self.param_entries["iters"].get())
        eps = float(self.param_entries["eps"].get())
        c = float(self.param_entries["c"].get())
        alpha = float(self.param_entries["alpha"].get())
        optimistic = str(self.param_entries["optimistic"].get()).lower() in ["true","1","yes"]
        stationary = self.stationary_var.get()
        superimpose = self.superimpose_var.get()

        alg = self.alg_var.get()
        if alg=="EPS_GREEDY":
            bandit = EPS_GREEDY(n=n, iters=iters, eps=eps, optimistic=optimistic, stationary=stationary)
        elif alg=="UCB":
            bandit = UCB(n=n, iters=iters, c=c, optimistic=optimistic, stationary=stationary)
        elif alg=="GradientBandit":
            bandit = GradientBandit(n=n, iters=iters, alpha=alpha, stationary=stationary)
        else:
            return

        gain, N = bandit.simulate()

        if not superimpose:
            self.ax_gain.clear()
        self.ax_gain.plot(np.arange(iters), gain, linewidth=2, label=f"{alg} (n={n})")
        self.ax_gain.set_title("Cumulative Gain", color="white", fontsize=16)
        self.ax_gain.tick_params(colors="white", labelsize=12)
        self.ax_gain.spines["bottom"].set_color("white")
        self.ax_gain.spines["left"].set_color("white")
        self.ax_gain.legend(facecolor="#222", edgecolor="white", labelcolor="white")

        if not superimpose:
            self.ax_counts.clear()
        self.ax_counts.bar(np.arange(n), N, color="#FFAA00", alpha=0.6, label=f"{alg} (n={n})")
        self.ax_counts.set_title("Action Counts", color="white", fontsize=16)
        self.ax_counts.tick_params(colors="white", labelsize=12)
        self.ax_counts.spines["bottom"].set_color("white")
        self.ax_counts.spines["left"].set_color("white")
        self.ax_counts.legend(facecolor="#222", edgecolor="white", labelcolor="white")

        self.true_values_text.configure(state='normal')
        self.true_values_text.delete(1.0, tk.END)
        self.true_values_text.insert(tk.END, f"True Bandit Rewards ({alg}):\n\n")
        for i, b in enumerate(bandit.bandits):
            value = getattr(b, "reward", "?")
            self.true_values_text.insert(tk.END, f"Bandit {i}: {value:.3f}\n")
        self.true_values_text.configure(state='disabled')

        self.canvas.draw()

    def clear_graphs(self):
        self.ax_gain.clear()
        self.ax_counts.clear()
        self.true_values_text.configure(state='normal')
        self.true_values_text.delete(1.0, tk.END)
        self.true_values_text.insert(tk.END, "True Bandit Rewards:\n")
        self.true_values_text.configure(state='disabled')
        self.canvas.draw()

if __name__=="__main__":
    root = tk.Tk()
    app = BanditGUI(root)
    root.mainloop()
